"""
Champion model (SPIKE-ACE2 binding affinity) generalized to any protein.

Architecture:
  GP1: ESM2(1280) + scalars(23) with Matern-3/2 + ARD
  GP2: ESM2(1280) with Matern-5/2 + ARD
  GP3: compact(18) with Linear kernel
  Structure kernel: StructureKernelFast (Hellinger + logprob delta + Ca-dist)
  Likelihood: FixedNoiseGaussianLikelihood with region-aware weights
  Priors: LogNormalPrior on struct outputscale, HalfCauchy(0.3) on noise
"""

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
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel, LinearKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import HalfCauchyPrior, LogNormalPrior
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.stats import spearmanr, rankdata, norm
from tqdm import trange

# ── CLI args ───────────────────────────────────────────────────────────────────
PROTEIN = sys.argv[1] if len(sys.argv) > 1 else "SPIKE_SARS2_Starr_2020_binding"
CV_COL  = sys.argv[2] if len(sys.argv) > 2 else "fold_contiguous_5"

# ── Data paths ─────────────────────────────────────────────────────────────────
KERMUT_DATA = Path("/n/netscratch/mzitnik_lab/Lab/afang/kermut/data")
DATA_PATH   = Path("/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute"
                   "/biomlbench/.cache/data/proteingym-dms/cv_folds_singles_substitutions"
                   ) / f"{PROTEIN}.csv"
EMB_PATH    = KERMUT_DATA / "embeddings" / "substitutions_singles" / "ESM2" / f"{PROTEIN}.h5"
ZS_PATH     = KERMUT_DATA / "zero_shot_fitness_predictions" / "ESM2" / "650M" / f"{PROTEIN}.csv"
CPROBS_PATH = KERMUT_DATA / "conditional_probs" / "ProteinMPNN" / f"{PROTEIN}.npy"
COORDS_PATH = KERMUT_DATA / "structures" / "coords" / f"{PROTEIN}.npy"
ZS_FULL_PATH = KERMUT_DATA / "zero_shot_fitness_predictions" / f"{PROTEIN}.csv"
ZS_COL      = "esm2_t33_650M_UR50D"

EXTRA_ZS_COLS = [
    "MIF", "VenusREM", "ProSST-2048", "RSALOR", "ESCOTT",
    "xTrimoPGLM-1B-CLM", "ProSST-128", "SaProt_650M_AF2", "ESM-IF1", "Unirep_evotune",
    "ProSST-4096", "S3F_MSA", "MSA_Transformer_ensemble", "VespaG", "SiteRM",
]

ALPHABET  = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}
SEED      = 2024
N_STEPS   = 1000
LR        = 0.1
LR_MIN    = 0.001
CDIST_MODE = "donot_use_mm_for_euclid_dist"

REGION_ALPHA       = 0.5
REGION_BETA        = 1.0
REGION_NOISE_SCALE = 0.05


# ── Structure kernel ───────────────────────────────────────────────────────────
def _hellinger(p: torch.Tensor) -> torch.Tensor:
    L = p.shape[0]
    i, j = torch.tril_indices(L, L, offset=-1, device=p.device)
    h = torch.sqrt(0.5 * ((p[i].sqrt() - p[j].sqrt()) ** 2).sum(dim=1))
    mat = torch.zeros(L, L, device=p.device)
    mat[i, j] = h
    mat[j, i] = h
    return mat


def precompute_struct_matrices(pos, aa, hell_full, log_probs, coords):
    h_mat = hell_full[pos.unsqueeze(1), pos.unsqueeze(0)]
    lp    = log_probs[pos, aa]
    p_mat = torch.abs(lp.unsqueeze(1) - lp.unsqueeze(0))
    d_mat = torch.cdist(coords[pos], coords[pos], p=2.0, compute_mode=CDIST_MODE)
    return h_mat, p_mat, d_mat


class StructureKernelFast(Kernel):
    def __init__(self, hell_full, log_probs, coords):
        super().__init__()
        self.register_buffer("hell_full", hell_full)
        self.register_buffer("log_probs", log_probs)
        self.register_buffer("coords",    coords)
        for name in ("h_ls", "p_ls", "d_ls"):
            self.register_parameter(f"raw_{name}", nn.Parameter(torch.tensor(1.0)))
            self.register_constraint(f"raw_{name}", Positive())
        self._cache: tuple | None = None

    def _ls(self, name):
        return getattr(self, f"raw_{name}_constraint").transform(getattr(self, f"raw_{name}"))

    def set_train_cache(self, pos, aa):
        self._cache = precompute_struct_matrices(
            pos, aa, self.hell_full, self.log_probs, self.coords
        )
        self._train_n = pos.shape[0]

    def _apply_kernel(self, h, p, d):
        return torch.exp(-self._ls("h_ls") * h - self._ls("p_ls") * p - self._ls("d_ls") * d)

    def forward(self, x1, x2, **_):
        pos1, aa1 = x1[:, 0], x1[:, 1]
        pos2, aa2 = x2[:, 0], x2[:, 1]
        if self._cache is not None and x1.shape[0] == self._train_n and x2.shape[0] == self._train_n:
            return self._apply_kernel(*self._cache)
        h   = self.hell_full[pos1.unsqueeze(1), pos2.unsqueeze(0)]
        lp1 = self.log_probs[pos1, aa1]; lp2 = self.log_probs[pos2, aa2]
        p   = torch.abs(lp1.unsqueeze(1) - lp2.unsqueeze(0))
        d   = torch.cdist(self.coords[pos1], self.coords[pos2], p=2.0, compute_mode=CDIST_MODE)
        return self._apply_kernel(h, p, d)


# ── GP model ───────────────────────────────────────────────────────────────────
class KermutGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, struct_kernel,
                 seq_kernel_type="matern52", ard_num_dims=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(input_size=1, bias=True)
        self.struct_kernel = ScaleKernel(
            struct_kernel,
            outputscale_prior=LogNormalPrior(loc=0.0, scale=0.5),
        )
        if seq_kernel_type == "linear":
            self.seq_kernel = ScaleKernel(LinearKernel())
        elif seq_kernel_type == "matern32":
            self.seq_kernel = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        else:
            self.seq_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        self.register_parameter("raw_pi", nn.Parameter(torch.tensor(0.5)))

    @property
    def pi(self):
        return torch.sigmoid(self.raw_pi)

    def forward(self, x_struct, x_emb, x_zero):
        mean   = self.mean_module(x_zero)
        kernel = self.pi * self.struct_kernel(x_struct) + (1 - self.pi) * self.seq_kernel(x_emb)
        return gpytorch.distributions.MultivariateNormal(mean, kernel)


def train_gp(gp, likelihood, x_struct, x_emb, x_zero, y_train):
    gp.train(); likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, gp)
    opt = torch.optim.AdamW(gp.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=LR_MIN)
    for _ in trange(N_STEPS, leave=False, desc="  opt"):
        opt.zero_grad()
        (-mll(gp(x_struct, x_emb, x_zero), y_train)).backward()
        opt.step()
        scheduler.step()


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

    df    = pd.read_csv(DATA_PATH)
    y_all = torch.tensor(df["DMS_score"].values, dtype=torch.float32)

    mut_pos = df["mutant"].apply(
        lambda m: int(re.match(r"[A-Z](\d+)[A-Z]", m).group(1)) - 1
    ).values
    mut_aa  = df["mutant"].apply(lambda m: AA_TO_IDX.get(m[-1], 0)).values
    wt_aa   = df["mutant"].apply(lambda m: AA_TO_IDX.get(m[0], 0)).values
    x_struct_all = torch.tensor(np.stack([mut_pos, mut_aa], axis=1), dtype=torch.long).to(device)

    # ESM2 embeddings
    with h5py.File(EMB_PATH, "r", locking=True) as f:
        h5_emb     = torch.tensor(f["embeddings"][:]).float()
        h5_mutants = [x.decode("utf-8") for x in f["mutants"][:]]
    if h5_emb.ndim == 3:
        h5_emb = h5_emb.mean(dim=1)
    h5_idx     = {m: i for i, m in enumerate(h5_mutants)}
    order      = [h5_idx[m] for m in df["mutant"].tolist()]
    x_esm2_all = h5_emb[order].to(device)

    # ESM2 zero-shot scores
    df_zs     = pd.read_csv(ZS_PATH)[[ZS_COL, "mutant"]]
    df_merged = df[["mutant"]].merge(df_zs, on="mutant", how="left")
    df_merged = df_merged.groupby("mutant", sort=False).mean(numeric_only=True).reindex(
        df_merged["mutant"].drop_duplicates()
    ).reset_index()
    x_zero_all = torch.tensor(df_merged[ZS_COL].values, dtype=torch.float32).unsqueeze(1).to(device)

    # ProteinMPNN conditional probs + Ca coords
    cond_probs = torch.tensor(np.load(CPROBS_PATH), dtype=torch.float32).to(device)
    coords     = torch.tensor(np.load(COORDS_PATH), dtype=torch.float32).to(device)
    hell_full  = _hellinger(cond_probs)
    log_probs  = cond_probs.clamp(min=1e-8).log()

    mut_pos_t   = torch.tensor(mut_pos, dtype=torch.long, device=device)
    mut_aa_t    = torch.tensor(mut_aa,  dtype=torch.long, device=device)
    wt_aa_t     = torch.tensor(wt_aa,   dtype=torch.long, device=device)
    log_p_mut   = log_probs[mut_pos_t, mut_aa_t].unsqueeze(1)
    log_p_wt    = log_probs[mut_pos_t, wt_aa_t].unsqueeze(1)
    log_p_delta = log_p_mut - log_p_wt

    # Contact-map features (raw; standardized per-fold below)
    coords_np   = coords.cpu().numpy()
    dist_matrix = scipy_cdist(coords_np, coords_np)
    contact_feats = []
    for pos in mut_pos:
        dists = dist_matrix[pos]
        n_contacts    = np.sum(dists < 8.0) - 1
        mean_dist     = np.mean(dists)
        local_density = np.sum(dists < 10.0) - 1
        rsa_proxy     = 1.0 / (n_contacts + 1)
        contact_feats.append([n_contacts, mean_dist, local_density, rsa_proxy])
    contact_feats_raw = np.array(contact_feats, dtype=np.float32)

    # Extra ZS predictors (raw; median-filled per-fold below)
    df_zs_full   = pd.read_csv(ZS_FULL_PATH)[["mutant"] + EXTRA_ZS_COLS]
    df_zs_merged = df[["mutant"]].merge(df_zs_full, on="mutant", how="left")
    extra_zs_raw = df_zs_merged[EXTRA_ZS_COLS].values.astype(np.float32)

    print(f"Dataset: {len(df)} variants, {coords.shape[0]} residues")

    all_y_true_std, all_y_pred_std = [], []  # normalized space for MSE
    all_y_true_raw, all_y_pred_raw = [], []  # raw space for Spearman
    fold_spearmans = {}
    fold_mses      = {}

    for fold in range(5):
        train_mask = (df[CV_COL] != fold).values
        test_mask  = (df[CV_COL] == fold).values
        if test_mask.sum() == 0:
            print(f"\nFold {fold}/4 — skipped (empty test set)", flush=True)
            continue
        print(f"\nFold {fold}/4", flush=True)

        x_struct_tr = x_struct_all[train_mask]
        x_struct_te = x_struct_all[test_mask]
        x_zero_tr   = x_zero_all[train_mask]
        x_zero_te   = x_zero_all[test_mask]
        y_tr        = y_all[train_mask].to(device)
        y_te        = y_all[test_mask]

        # Region-aware per-sample noise weights
        train_region_ids = df.loc[train_mask, CV_COL].values.astype(np.int64)
        boundary_dist    = np.abs(train_region_ids - fold).astype(np.float32)
        region_weights   = 1.0 + REGION_ALPHA * np.exp(-boundary_dist / REGION_BETA)
        inv_w            = 1.0 / region_weights
        fixed_noise_np   = np.clip(REGION_NOISE_SCALE * (inv_w - inv_w.min()), 1e-4, None)
        fixed_noise      = torch.tensor(fixed_noise_np, dtype=torch.float32, device=device)

        # GP training targets: quantile-normalize to standard normal
        y_tr_np   = y_tr.cpu().numpy()
        ranks     = rankdata(y_tr_np, method="average")
        quantiles = (ranks - 0.5) / len(ranks)
        y_tr_std  = torch.tensor(norm.ppf(quantiles), dtype=torch.float32, device=device)

        # MSE targets: z-score normalize using train mean/std (matches kermut/ProteinGym)
        y_tr_mean = y_tr_np.mean()
        y_tr_sd   = y_tr_np.std() + 1e-8
        y_te_zscore = (y_te.cpu().numpy() - y_tr_mean) / y_tr_sd

        # Per-fold: standardize contact features using train statistics
        cf_train  = contact_feats_raw[train_mask]
        cf_mean   = cf_train.mean(axis=0, keepdims=True)
        cf_std    = cf_train.std(axis=0, keepdims=True) + 1e-8
        x_contact_all = torch.tensor(
            (contact_feats_raw - cf_mean) / cf_std, dtype=torch.float32, device=device
        )

        # Per-fold: median-fill extra ZS using train medians
        extra_zs_filled = extra_zs_raw.copy()
        for col_idx in range(extra_zs_raw.shape[1]):
            train_vals = extra_zs_raw[train_mask, col_idx]
            median_val = np.nanmedian(train_vals)
            nan_mask   = np.isnan(extra_zs_filled[:, col_idx])
            extra_zs_filled[nan_mask, col_idx] = median_val
        extra_zs = torch.tensor(extra_zs_filled, dtype=torch.float32, device=device)

        # Assemble and standardize scalar features (23 dims) with train stats
        x_scalars_all = torch.cat(
            [log_p_mut, log_p_wt, log_p_delta, x_zero_all, extra_zs, x_contact_all], dim=1
        )
        scalars_tr    = x_scalars_all[train_mask]
        scalars_mean  = scalars_tr.mean(dim=0, keepdim=True)
        scalars_std   = scalars_tr.std(dim=0, keepdim=True).clamp(min=1e-8)
        x_scalars_normed = (x_scalars_all - scalars_mean) / scalars_std

        # GP1: ESM2(1280) + scalars(23), Matern-3/2 + ARD
        x_emb_full    = torch.cat([x_esm2_all, x_scalars_normed], dim=1)
        # GP2: ESM2(1280) only, Matern-5/2 + ARD
        x_emb_esm     = x_esm2_all
        # GP3: compact 18 dims, Linear kernel
        x_emb_compact = torch.cat([
            x_scalars_normed[:, :14],
            x_scalars_normed[:, 19:],
        ], dim=1)

        feature_sets = [
            ("full_1303_matern32", x_emb_full,    "matern32"),
            ("esm_1280_matern52",  x_emb_esm,     "matern52"),
            ("compact_18_linear",  x_emb_compact, "linear"),
        ]

        ensemble_preds = []
        for gp_idx, (feat_name, x_emb_all_feat, kern_type) in enumerate(feature_sets):
            print(f"  GP{gp_idx+1} ({feat_name}, {x_emb_all_feat.shape[1]}d, {kern_type}):", flush=True)
            x_emb_tr = x_emb_all_feat[train_mask]
            x_emb_te = x_emb_all_feat[test_mask]

            torch.manual_seed(SEED)
            np.random.seed(SEED)

            struct_kernel = StructureKernelFast(hell_full, log_probs, coords)
            struct_kernel.set_train_cache(x_struct_tr[:, 0], x_struct_tr[:, 1])

            likelihood = FixedNoiseGaussianLikelihood(
                noise=fixed_noise,
                learn_additional_noise=True,
                noise_prior=HalfCauchyPrior(scale=0.3),
            ).to(device)
            ard_dims = x_emb_tr.shape[1] if kern_type != "linear" else None
            gp = KermutGP(
                train_x=(x_struct_tr, x_emb_tr, x_zero_tr),
                train_y=y_tr_std,
                likelihood=likelihood,
                struct_kernel=struct_kernel,
                seq_kernel_type=kern_type,
                ard_num_dims=ard_dims,
            ).to(device)

            try:
                train_gp(gp, likelihood, x_struct_tr, x_emb_tr, x_zero_tr, y_tr_std)
                y_pred_i = predict_gp(gp, likelihood, x_struct_te, x_emb_te, x_zero_te)
                print(f"    Individual Spearman: {spearmanr(y_te.cpu().numpy(), y_pred_i)[0]:.4f}", flush=True)
                ensemble_preds.append(y_pred_i)
            except (ValueError, RuntimeError) as e:
                print(f"    GP{gp_idx+1} failed ({e.__class__.__name__}: {e}), skipping.", flush=True)
            finally:
                del gp, likelihood, struct_kernel
                torch.cuda.empty_cache()

        if not ensemble_preds:
            print(f"  Fold {fold}: all GPs failed, skipping fold.", flush=True)
            continue

        # Ensemble predictions are in quantile-normalized (standard normal) space.
        # Spearman: rank-invariant, compute directly against raw test targets.
        y_pred    = np.mean(ensemble_preds, axis=0)
        y_te_np   = y_te.cpu().numpy()
        rho       = spearmanr(y_te_np, y_pred)[0]

        # MSE: predictions are on the quantile-normalized scale; test targets are
        # z-scored with train mean/std. Both transformations map to approximately
        # N(0,1) for well-behaved data, so MSE is computed between y_pred (quantile
        # normal) and y_te_zscore (z-score normal) — both in standardized space,
        # matching the ProteinGym/Kermut benchmark convention.
        # Clip predictions to guard against GP extrapolation blow-up on OOD folds.
        y_pred_clipped = np.clip(y_pred, -10, 10)
        mse = float(np.mean((y_te_zscore - y_pred_clipped) ** 2))

        fold_spearmans[fold] = rho
        fold_mses[fold]      = mse
        print(f"  Ensemble Fold {fold} Spearman: {rho:.4f}  MSE: {mse:.6f}", flush=True)

        all_y_true_raw.extend(y_te_np.tolist())
        all_y_pred_raw.extend(y_pred.tolist())
        all_y_true_std.extend(y_te_zscore.tolist())
        all_y_pred_std.extend(y_pred_clipped.tolist())

    overall_spearman = spearmanr(all_y_true_raw, all_y_pred_raw)[0]
    overall_mse      = float(np.mean((np.array(all_y_true_std) - np.array(all_y_pred_std)) ** 2))

    print(f"\n{'='*55}")
    print(f"Protein:   {PROTEIN}")
    print(f"CV split:  {CV_COL}")
    folds_run = sorted(fold_spearmans.keys())
    print(f"Per-fold Spearman: {' '.join(f'{fold_spearmans[f]:.4f}' for f in folds_run)}")
    print(f"Per-fold MSE:      {' '.join(f'{fold_mses[f]:.6f}' for f in folds_run)}")
    print(f"Overall Spearman: {overall_spearman:.4f}")
    print(f"Overall MSE: {overall_mse:.6f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
