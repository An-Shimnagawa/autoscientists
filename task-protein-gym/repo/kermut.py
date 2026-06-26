"""
Fast reproduction of Kermut using compact (pos, aa) structure kernel inputs.

Data source: set KERMUT_DATA env var to the kermut/data directory (see download_data.sh).
  - embeddings/substitutions_singles/ESM2/{PROTEIN}.h5
  - zero_shot_fitness_predictions/ESM2/650M/{PROTEIN}.csv  (col: esm2_t33_650M_UR50D)
  - conditional_probs/ProteinMPNN/{PROTEIN}.npy
  - structures/coords/{PROTEIN}.npy
  - cv_folds_singles_substitutions/{PROTEIN}.csv

Supported proteins (pass as first arg):
  SPIKE_SARS2_Starr_2020_binding   (default)
  SBI_STAAM_Tsuboyama_2023_2JVG

Usage:
    python kermut.py [PROTEIN] [CV_COLUMN]

Examples:
    python kermut.py
    python kermut.py SPIKE_SARS2_Starr_2020_binding fold_contiguous_5
    python kermut.py SBI_STAAM_Tsuboyama_2023_2JVG fold_modulo_5
"""

import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import HalfCauchyPrior
from scipy.stats import spearmanr
from tqdm import trange

# ── CLI args ───────────────────────────────────────────────────────────────────
PROTEIN   = sys.argv[1] if len(sys.argv) > 1 else "SPIKE_SARS2_Starr_2020_binding"
CV_COL    = sys.argv[2] if len(sys.argv) > 2 else "fold_contiguous_5"

# ── Official kermut data paths ─────────────────────────────────────────────────
_kermut_env = os.environ.get("KERMUT_DATA")
if not _kermut_env:
    raise RuntimeError("Set KERMUT_DATA env var to the kermut/data directory (see download_data.sh)")
KERMUT_DATA = Path(_kermut_env)
DATA_PATH   = KERMUT_DATA / "cv_folds_singles_substitutions" / f"{PROTEIN}.csv"
EMB_PATH    = KERMUT_DATA / "embeddings" / "substitutions_singles" / "ESM2" / f"{PROTEIN}.h5"
ZS_PATH     = KERMUT_DATA / "zero_shot_fitness_predictions" / "ESM2" / "650M" / f"{PROTEIN}.csv"
CPROBS_PATH = KERMUT_DATA / "conditional_probs" / "ProteinMPNN" / f"{PROTEIN}.npy"
COORDS_PATH = KERMUT_DATA / "structures" / "coords" / f"{PROTEIN}.npy"
ZS_COL      = "esm2_t33_650M_UR50D"

ALPHABET   = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX  = {aa: i for i, aa in enumerate(ALPHABET)}
SEED       = 2024
N_STEPS    = 150
LR         = 0.1
CDIST_MODE = "donot_use_mm_for_euclid_dist"


# ── Structure kernel ───────────────────────────────────────────────────────────
# Speed trick: precompute the per-fold raw distance matrices (hellinger, logprob
# diff, spatial dist) once before the optimization loop and cache them as fixed
# buffers. Each forward pass then only needs three scalar multiplies + one exp,
# instead of re-indexing the full (L, L) lookup tables every step.

def _hellinger(p: torch.Tensor) -> torch.Tensor:
    """(L, 20) → (L, L) symmetric Hellinger distance matrix."""
    L = p.shape[0]
    i, j = torch.tril_indices(L, L, offset=-1, device=p.device)
    h = torch.sqrt(0.5 * ((p[i].sqrt() - p[j].sqrt()) ** 2).sum(dim=1))
    mat = torch.zeros(L, L, device=p.device)
    mat[i, j] = h; mat[j, i] = h
    return mat


def precompute_struct_matrices(
    pos: torch.Tensor,        # (N,) long, 0-indexed positions
    aa:  torch.Tensor,        # (N,) long, aa indices 0-19
    hell_full: torch.Tensor,  # (L, L) hellinger distance matrix
    log_probs: torch.Tensor,  # (L, 20) log ProteinMPNN probs
    coords:    torch.Tensor,  # (L, 3) Cα coordinates
) -> tuple:
    """Return (h_mat, p_mat, d_mat) — the three raw (N, N) distance matrices."""
    h_mat = hell_full[pos.unsqueeze(1), pos.unsqueeze(0)]              # (N, N)
    lp    = log_probs[pos, aa]                                          # (N,)
    p_mat = torch.abs(lp.unsqueeze(1) - lp.unsqueeze(0))               # (N, N)
    d_mat = torch.cdist(coords[pos], coords[pos], p=2.0,
                        compute_mode=CDIST_MODE)                        # (N, N)
    return h_mat, p_mat, d_mat


class StructureKernelFast(Kernel):
    """
    Structure kernel with pre-cached (N, N) distance matrices.

    At fold setup time, call .set_train_cache(h, p, d) once.
    During forward, if x1 is the training set (same object), use the cache;
    otherwise re-index on the fly (test-time, one call only).

    Three sub-kernels multiplied together:
      k_H = exp(-ls_h * hellinger[pos1, pos2])
      k_p = exp(-ls_p * |log P(mut1) - log P(mut2)|)
      k_d = exp(-ls_d * dist3d[pos1, pos2])
    """

    def __init__(
        self,
        hell_full: torch.Tensor,  # (L, L)
        log_probs: torch.Tensor,  # (L, 20)
        coords:    torch.Tensor,  # (L, 3)
    ):
        super().__init__()
        self.register_buffer("hell_full", hell_full)
        self.register_buffer("log_probs", log_probs)
        self.register_buffer("coords",    coords)

        for name in ("h_ls", "p_ls", "d_ls"):
            self.register_parameter(f"raw_{name}", nn.Parameter(torch.tensor(1.0)))
            self.register_constraint(f"raw_{name}", Positive())

        # Cache for training set (set per fold)
        self._cache: tuple | None = None

    def _ls(self, name: str) -> torch.Tensor:
        return getattr(self, f"raw_{name}_constraint").transform(
            getattr(self, f"raw_{name}")
        )

    def set_train_cache(self, pos: torch.Tensor, aa: torch.Tensor) -> None:
        """Precompute and store the three (N_train, N_train) raw distance matrices.
        Called once per fold before the optimization loop."""
        self._cache = precompute_struct_matrices(
            pos, aa, self.hell_full, self.log_probs, self.coords
        )
        self._train_n = pos.shape[0]
        self._train_pos = pos
        self._train_aa  = aa

    def _apply_lengthscales(
        self, h: torch.Tensor, p: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        """Fused: exp(-ls_h·h - ls_p·p - ls_d·d) — one kernel launch instead of three."""
        return torch.exp(
            -self._ls("h_ls") * h
            - self._ls("p_ls") * p
            - self._ls("d_ls") * d
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **_) -> torch.Tensor:
        pos1, aa1 = x1[:, 0], x1[:, 1]
        pos2, aa2 = x2[:, 0], x2[:, 1]

        # Use cached train-vs-train matrices when sizes match the training set
        if (self._cache is not None
                and x1.shape[0] == self._train_n
                and x2.shape[0] == self._train_n):
            return self._apply_lengthscales(*self._cache)

        # Cross eval (train vs test, or test vs test)
        h = self.hell_full[pos1.unsqueeze(1), pos2.unsqueeze(0)]
        lp1 = self.log_probs[pos1, aa1]; lp2 = self.log_probs[pos2, aa2]
        p = torch.abs(lp1.unsqueeze(1) - lp2.unsqueeze(0))
        d = torch.cdist(self.coords[pos1], self.coords[pos2],
                        p=2.0, compute_mode=CDIST_MODE)
        return self._apply_lengthscales(h, p, d)


# ── Composite GP ───────────────────────────────────────────────────────────────
class KermutGP(ExactGP):
    """
    GP with composite kernel:
      k = π · ScaleKernel(StructureKernelFast) + (1-π) · RBFKernel

    Inputs per variant: (x_struct, x_emb, x_zero)
      x_struct  (N, 2)     [mut_position, mut_aa_index]
      x_emb     (N, 1280)  ESM-2 650M mean-pooled embedding
      x_zero    (N, 1)     ESM-2 zero-shot log-prob (LinearMean input)
    """

    def __init__(self, train_x, train_y, likelihood, struct_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module   = LinearMean(input_size=1, bias=True)
        self.struct_kernel = ScaleKernel(struct_kernel)
        self.seq_kernel    = RBFKernel()   # matches official kermut SequenceKernel(RBF)
        self.register_parameter("raw_pi", nn.Parameter(torch.tensor(0.5)))

    @property
    def pi(self):
        return torch.sigmoid(self.raw_pi)

    def forward(self, x_struct, x_emb, x_zero):
        mean   = self.mean_module(x_zero)
        kernel = self.pi * self.struct_kernel(x_struct) + (1 - self.pi) * self.seq_kernel(x_emb)
        return gpytorch.distributions.MultivariateNormal(mean, kernel)


# ── Training / prediction ──────────────────────────────────────────────────────
def train_gp(gp, likelihood, x_struct, x_emb, x_zero, y_train):
    gp.train(); likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, gp)
    opt = torch.optim.AdamW(gp.parameters(), lr=LR)
    for _ in trange(N_STEPS, leave=False, desc="  opt"):
        opt.zero_grad()
        (-mll(gp(x_struct, x_emb, x_zero), y_train)).backward()
        opt.step()


@torch.no_grad()
def predict_gp(gp, likelihood, x_struct, x_emb, x_zero):
    gp.eval(); likelihood.eval()
    return likelihood(gp(x_struct, x_emb, x_zero)).mean.cpu().numpy()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Protein: {PROTEIN}")
    print(f"CV split: {CV_COL}")
    print(f"Device: {device}")

    # ── Load DMS data ──────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    y_all = torch.tensor(df["DMS_score"].values, dtype=torch.float32)

    # Compact mutation representation: (N, 2) with [0-indexed position, aa index]
    mut_pos = df["mutant"].apply(
        lambda m: int(re.match(r"[A-Z](\d+)[A-Z]", m).group(1)) - 1
    ).values
    mut_aa = df["mutant"].apply(lambda m: AA_TO_IDX.get(m[-1], 0)).values
    x_struct_all = torch.tensor(
        np.stack([mut_pos, mut_aa], axis=1), dtype=torch.long
    ).to(device)

    # ── Load embeddings from official h5, realign to CSV mutant order ──────────
    with h5py.File(EMB_PATH, "r", locking=True) as f:
        h5_emb     = torch.tensor(f["embeddings"][:]).float()
        h5_mutants = [x.decode("utf-8") for x in f["mutants"][:]]
    if h5_emb.ndim == 3:
        h5_emb = h5_emb.mean(dim=1)
    h5_idx    = {m: i for i, m in enumerate(h5_mutants)}
    order     = [h5_idx[m] for m in df["mutant"].tolist()]
    x_emb_all = h5_emb[order].to(device)

    # ── Load zero-shot scores, merge by mutant name, align to CSV order ────────
    df_zs     = pd.read_csv(ZS_PATH)[[ZS_COL, "mutant"]]
    df_merged = df[["mutant"]].merge(df_zs, on="mutant", how="left")
    df_merged = df_merged.groupby("mutant", sort=False).mean(numeric_only=True).reindex(
        df_merged["mutant"].drop_duplicates()
    ).reset_index()
    x_zero_all = torch.tensor(
        df_merged[ZS_COL].values, dtype=torch.float32
    ).unsqueeze(1).to(device)

    # ── Load ProteinMPNN cond probs + Cα coords, build hellinger matrix ────────
    cond_probs = torch.tensor(np.load(CPROBS_PATH), dtype=torch.float32).to(device)
    coords     = torch.tensor(np.load(COORDS_PATH), dtype=torch.float32).to(device)
    hell_full  = _hellinger(cond_probs)                             # (L, L), computed once
    log_probs  = cond_probs.clamp(min=1e-8).log()                  # (L, 20), computed once

    print(f"Dataset: {len(df)} variants, {coords.shape[0]} residues")

    # ── 5-fold CV ──────────────────────────────────────────────────────────────
    all_y_true, all_y_pred = [], []         # raw space — for Spearman
    all_y_true_std, all_y_pred_std = [], [] # z-score space — for MSE
    fold_spearmans, fold_mses = {}, {}

    for fold in range(5):
        print(f"\nFold {fold}/4", flush=True)
        train_mask = (df[CV_COL] != fold).values
        test_mask  = (df[CV_COL] == fold).values

        x_struct_tr, x_struct_te = x_struct_all[train_mask], x_struct_all[test_mask]
        x_emb_tr,    x_emb_te   = x_emb_all[train_mask],    x_emb_all[test_mask]
        x_zero_tr,   x_zero_te  = x_zero_all[train_mask],   x_zero_all[test_mask]
        y_tr = y_all[train_mask].to(device)
        y_te = y_all[test_mask]

        # Standardize targets using train fold stats only (official kermut behaviour)
        mu, sigma = y_tr.mean(), y_tr.std()
        y_tr_std  = (y_tr - mu) / sigma

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Build structure kernel and precompute train-set distance matrices
        struct_kernel = StructureKernelFast(hell_full, log_probs, coords)
        struct_kernel.set_train_cache(x_struct_tr[:, 0], x_struct_tr[:, 1])

        likelihood = GaussianLikelihood(noise_prior=HalfCauchyPrior(scale=0.1)).to(device)
        gp = KermutGP(
            train_x=(x_struct_tr, x_emb_tr, x_zero_tr),
            train_y=y_tr_std,
            likelihood=likelihood,
            struct_kernel=struct_kernel,
        ).to(device)

        train_gp(gp, likelihood, x_struct_tr, x_emb_tr, x_zero_tr, y_tr_std)

        y_pred_std = predict_gp(gp, likelihood, x_struct_te, x_emb_te, x_zero_te)
        y_pred     = y_pred_std * sigma.item() + mu.item()

        y_te_np = y_te.cpu().numpy()
        rho = spearmanr(y_te_np, y_pred)[0]
        fold_spearmans[fold] = rho

        # MSE in z-score space (train fold stats), predictions clipped against GP blow-up
        y_te_zscore    = (y_te_np - mu.item()) / sigma.item()
        y_pred_clipped = np.clip(y_pred_std, -10, 10)
        fold_mse = float(np.mean((y_te_zscore - y_pred_clipped) ** 2))
        fold_mses[fold] = fold_mse
        print(f"  Fold {fold} Spearman: {rho:.4f}  MSE: {fold_mse:.4f}", flush=True)

        all_y_true.extend(y_te_np.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_true_std.extend(y_te_zscore.tolist())
        all_y_pred_std.extend(y_pred_clipped.tolist())

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    overall = spearmanr(y_true_arr, y_pred_arr)[0]
    mse = float(np.mean((np.array(all_y_true_std) - np.array(all_y_pred_std)) ** 2))
    print(f"\n{'='*55}")
    print(f"Protein:   {PROTEIN}")
    print(f"CV split:  {CV_COL}")
    print(f"Per-fold:  {' '.join(f'{fold_spearmans[f]:.4f}' for f in range(5))}")
    print(f"Overall Spearman: {overall:.4f}")
    print(f"MSE (z-score):    {mse:.6f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
