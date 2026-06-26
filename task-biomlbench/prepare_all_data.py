#!/usr/bin/env python3
"""
Prepare all 24 biomlbench task datasets into their respective task data/ directories.

Run this script once to download and set up all data before agents start working.

Usage:
    python prepare_all_data.py [--biomlbench-dir PATH] [--output-dir PATH]
                               [--task TASK] [--category CATEGORY]

Arguments:
    --biomlbench-dir  Path to a biomlbench data directory (optional).
                      This is the directory populated by `biomlbench prepare --data-dir PATH`.
                      When provided, the script copies pre-prepared data from the local
                      cache instead of downloading from the internet. Speeds up setup
                      significantly for ProteinGym, single-cell, and Kaggle tasks.
    --output-dir      Write prepared data here instead of the default task-biomlbench tree.
                      Useful for testing reproducibility in a clean directory.
    --task            Prepare only this single task (by directory name, e.g.
                      proteingym-dms-SPIKE_SARS2_Starr_2020_binding).
    --category        Prepare only one category: protein_engineering, drug_discovery, single_cell_omics, biomedical_imaging.

Task categories and their data sources
---------------------------------------
  protein_engineering (ProteinGym DMS, 6 tasks)
    Cache path : <biomlbench-dir>/proteingym-dms/<DMS_ID>/prepared/public/data.csv
    Fallback   : raw ProteinGym CSVs searched recursively under
                 <biomlbench-dir>/proteingym-dms/ (e.g. in cv_folds_singles_substitutions/)

  drug_discovery (Polaris / TDCommons, 9 tasks)
    Cache path : <biomlbench-dir>/polarishub/<task>/prepared/public/train.csv
    Fallback   : downloads directly from Polaris Hub (requires `pip install polaris-client`
                 and a free Polaris Hub account: polaris login)

  single_cell_omics (OpenProblems, 5 tasks)
    Cache path : <biomlbench-dir>/manual/<task>/prepared/  (public + private)
    Fallback   : downloads from S3 via biomlbench pipeline (requires AWS CLI and
                 biomlbench Python package importable from --biomlbench-dir)

  biomedical_imaging (Kaggle, 4 tasks)
    Cache path : <biomlbench-dir>/kaggle/<competition>/prepared/public/
    Fallback   : downloads via `kaggle` CLI (requires ~/.kaggle/kaggle.json credentials
                 and competition rules accepted on kaggle.com)

Populate the cache with:
    cd biomlbench && biomlbench prepare --all --data-dir /path/to/data
Then pass /path/to/data as --biomlbench-dir to this script.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Globals set by main() via argparse
TASK_ROOT = Path(__file__).parent
BIOMLBENCH: Optional[Path] = None


# ===========================================================================
# ProteinGym DMS tasks
# ===========================================================================

PROTEINGYM_SUBSTITUTION_TASKS = [
    "SPIKE_SARS2_Starr_2020_binding",
    "SBI_STAAM_Tsuboyama_2023_2JVG",
    "PSAE_PICP2_Tsuboyama_2023_1PSE",
    "CBX4_HUMAN_Tsuboyama_2023_2K28",
]

PROTEINGYM_INDEL_TASKS = [
    "Q8EG35_SHEON_Campbell_2022_indels",
    "CSN4_MOUSE_Tsuboyama_2023_1UFM_indels",
]

# ===========================================================================
# Polaris/TDCommons tasks
# ===========================================================================

POLARIS_TASKS = [
    ("polaris-pkis2-egfr-wt-c-1",          "polaris/pkis2-egfr-wt-c-1"),
    ("polaris-adme-fang-hclint-1",          "polaris/adme-fang-hclint-1"),
    ("polaris-adme-fang-hppb-1",            "polaris/adme-fang-hppb-1"),
    ("polaris-adme-fang-solu-1",            "polaris/adme-fang-solu-1"),
    ("tdcommons-cyp2d6-substrate-carbonmangels", "tdcommons/cyp2d6-substrate-carbonmangels"),
    ("tdcommons-lipophilicity-astrazeneca", "tdcommons/lipophilicity-astrazeneca"),
    ("tdcommons-caco2-wang",                "tdcommons/caco2-wang"),
    ("tdcommons-herg",                      "tdcommons/herg"),
    ("tdcommons-bbb-martins",               "tdcommons/bbb-martins"),
]

# ===========================================================================
# OpenProblems single-cell tasks
# ===========================================================================

MANUAL_TASKS = [
    "open-problems-predict-modality",
    "open-problems-single-cell-perturbations",
    "open-problems-cell-cell-communication-ligand-target",
    "open-problems-spatially-variable-genes",
    "open-problems-label-projection",
]

# ===========================================================================
# Kaggle tasks
# ===========================================================================

KAGGLE_TASKS = [
    ("kaggle-osic-pulmonary-fibrosis-progression",               "osic-pulmonary-fibrosis-progression"),
    ("kaggle-histopathologic-cancer-detection",                   "histopathologic-cancer-detection"),
    ("kaggle-rsna-miccai-brain-tumor-radiogenomic-classification","rsna-miccai-brain-tumor-radiogenomic-classification"),
    ("kaggle-uw-madison-gi-tract-image-segmentation",             "uw-madison-gi-tract-image-segmentation"),
]


# ===========================================================================
# Helpers
# ===========================================================================

def add_murcko_cv_fold(df: "pd.DataFrame", smiles_col: str, n_folds: int = 5) -> "pd.DataFrame":
    """
    Append a cv_fold column (0..n_folds-1) computed via Murcko scaffold grouping.

    Scaffolds are sorted by descending group size and assigned round-robin so each
    fold gets roughly equal coverage (≈20% per fold for n_folds=5).
    Molecules with invalid SMILES get fold -1.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def get_scaffold(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return "__invalid__"
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    scaffolds = df[smiles_col].map(get_scaffold)
    scaffold_groups = scaffolds.groupby(scaffolds).apply(lambda g: list(g.index))
    scaffold_groups = sorted(scaffold_groups, key=len, reverse=True)

    fold_col = pd.Series(-1, index=df.index, dtype=int)
    for i, idx_list in enumerate(scaffold_groups):
        fold = i % n_folds
        for idx in idx_list:
            fold_col.at[idx] = fold

    df = df.copy()
    df["cv_fold"] = fold_col
    return df


def add_patient_cv_fold(df: "pd.DataFrame", patient_col: str, n_folds: int = 5, seed: int = 42) -> "pd.DataFrame":
    """
    Append a cv_fold column using patient-level KFold split.

    Patients are sorted, then KFold(shuffle=True, random_state=seed) is applied so
    the assignment is deterministic and each fold gets ~1/n_folds of patients.
    """
    from sklearn.model_selection import KFold

    patients_sorted = sorted(df[patient_col].unique())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    patient_fold = {}
    for fold_i, (_, val_idx) in enumerate(kf.split(patients_sorted)):
        for idx in val_idx:
            patient_fold[patients_sorted[idx]] = fold_i

    df = df.copy()
    df["cv_fold"] = df[patient_col].map(patient_fold)
    return df


# ===========================================================================
# Prepare functions
# ===========================================================================

def prepare_proteingym_task(dms_id: str, is_indel: bool = False) -> bool:
    """Prepare a ProteinGym DMS task."""
    task_dir = TASK_ROOT / "protein_engineering" / f"proteingym-dms-{dms_id}" / "data"
    task_dir.mkdir(parents=True, exist_ok=True)

    if (task_dir / "data.csv").exists():
        print(f"  [skip] {dms_id} already has data.csv")
        return True

    # Try copying from biomlbench prepared cache
    if BIOMLBENCH is not None:
        prep_pub = BIOMLBENCH / "proteingym-dms" / dms_id / "prepared" / "public"
        if (prep_pub / "data.csv").exists():
            shutil.copy(prep_pub / "data.csv", task_dir / "data.csv")
            if (prep_pub / "sample_submission.csv").exists():
                shutil.copy(prep_pub / "sample_submission.csv", task_dir / "sample_submission.csv")
            print(f"  [copy] {dms_id} -> copied from biomlbench cache")
            return True

    # Prepare from raw ProteinGym CSV files (search in biomlbench data dir)
    if BIOMLBENCH is None:
        print(f"  [fail] {dms_id}: --biomlbench-dir not provided and no prepared cache found.")
        print(f"         Provide --biomlbench-dir pointing to a biomlbench clone with ProteinGym data.")
        return False

    raw_dir = BIOMLBENCH / "proteingym-dms"
    candidates = list(raw_dir.rglob(f"{dms_id}.csv"))
    if not candidates:
        print(f"  [fail] {dms_id}: raw CSV not found in {raw_dir}.")
        print(f"         Run biomlbench's ingest_proteingym.py first to download the raw ProteinGym data.")
        return False
    raw_csv_file = candidates[0]

    try:
        df = pd.read_csv(raw_csv_file)

        if is_indel:
            metadata = pd.read_csv(raw_dir / "DMS_indels.csv", index_col="DMS_id")
            fold_columns = ["fold_random_5"]
        else:
            metadata = pd.read_csv(raw_dir / "DMS_substitutions.csv", index_col="DMS_id")
            if "fold_random_5" in df.columns:
                fold_columns = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
            elif "fold_rand_multiples" in df.columns:
                df.rename(columns={"fold_rand_multiples": "fold_random_5"}, inplace=True)
                fold_columns = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
                df["fold_modulo_5"] = df["fold_random_5"]
                df["fold_contiguous_5"] = df["fold_random_5"]
            else:
                print(f"  [warn] Unknown fold columns in {dms_id}: {list(df.columns)}")
                fold_columns = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]

        wt_sequence = metadata.loc[dms_id, "target_seq"]
        wt_seq_row = pd.DataFrame({"id": ["WT"], "sequence": [wt_sequence], "fitness_score": [np.nan]})
        wt_seq_row[fold_columns] = -1

        df["id"] = range(len(df))
        df.rename({"mutated_sequence": "sequence", "DMS_score": "fitness_score"}, axis=1, inplace=True)
        df = df[["id", "sequence", "fitness_score"] + fold_columns]
        df = pd.concat([wt_seq_row, df])
        df.to_csv(task_dir / "data.csv", index=False)

        if is_indel:
            sample_sub = pd.DataFrame({
                "id": df["id"].iloc[1:],
                "fitness_score": [0.0] * len(df.iloc[1:]),
            })
        else:
            sample_sub = pd.DataFrame({
                "id": df["id"].iloc[1:],
                "fitness_score_fold_random_5": [0.0] * len(df.iloc[1:]),
                "fitness_score_fold_modulo_5": [0.0] * len(df.iloc[1:]),
                "fitness_score_fold_contiguous_5": [0.0] * len(df.iloc[1:]),
            })
        sample_sub.to_csv(task_dir / "sample_submission.csv", index=False)

        print(f"  [prepare] {dms_id} -> prepared {len(df)-1} sequences")
        return True

    except Exception as e:
        print(f"  [fail] {dms_id}: {e}")
        return False


def prepare_polaris_task(task_name: str, benchmark_id: str) -> bool:
    """Prepare a Polaris/TDCommons task."""
    task_dir = TASK_ROOT / "drug_discovery" / task_name / "data"
    task_dir.mkdir(parents=True, exist_ok=True)

    if (task_dir / "train.csv").exists() and (task_dir / "test_features.csv").exists():
        print(f"  [skip] {task_name} already has train.csv + test_features.csv")
        return True

    # Try copying from biomlbench prepared cache
    if BIOMLBENCH is not None:
        biomlbench_task_name = benchmark_id.replace("/", "-")
        prep_pub = BIOMLBENCH / "polarishub" / biomlbench_task_name / "prepared" / "public"
        if (prep_pub / "train.csv").exists():
            for f in ["train.csv", "test_features.csv", "sample_submission.csv"]:
                if (prep_pub / f).exists():
                    shutil.copy(prep_pub / f, task_dir / f)
            print(f"  [copy] {task_name} -> copied from biomlbench cache")
            return True

    # Download directly from Polaris Hub
    try:
        import polaris as po

        print(f"  [download] {task_name}: fetching from Polaris Hub ({benchmark_id})...")
        benchmark = po.load_benchmark(benchmark_id)
        train, test = benchmark.get_train_test_split()

        train_df = train.as_dataframe()
        test_df = test.as_dataframe()

        target_col = next(iter(benchmark.target_cols))
        molecule_col = [c for c in train_df.columns if c != target_col][0]

        full_df = test.dataset.table
        test_rows_full = full_df.iloc[test.indices][[molecule_col, target_col]].reset_index(drop=True)
        test_targets = test_rows_full[target_col].values

        train_out = train_df[[molecule_col, target_col]].copy()
        train_out = add_murcko_cv_fold(train_out, smiles_col=molecule_col)
        train_out.to_csv(task_dir / "train.csv", index=False)

        test_features = test_df[[molecule_col]].copy().reset_index(drop=True)
        test_features.insert(0, "id", range(len(test_features)))
        test_features.to_csv(task_dir / "test_features.csv", index=False)

        sample_sub = pd.DataFrame({"id": range(len(test_features)), target_col: [0.0] * len(test_features)})
        sample_sub.to_csv(task_dir / "sample_submission.csv", index=False)

        private_dir = task_dir / "private"
        private_dir.mkdir(exist_ok=True)
        answers = pd.DataFrame({"id": range(len(test_features)), target_col: test_targets})
        answers.to_csv(private_dir / "answers.csv", index=False)

        print(f"  [download] {task_name} -> {len(train_df)} train, {len(test_df)} test")
        return True

    except ImportError:
        print(f"  [fail] {task_name}: polaris-client not installed.")
        print(f"         Install with: pip install polaris-client")
        print(f"         Then authenticate: polaris login")
        return False
    except Exception as e:
        print(f"  [fail] {task_name}: {e}")
        return False


def prepare_manual_task(task_name: str) -> bool:
    """Prepare an OpenProblems single-cell task."""
    task_dir = TASK_ROOT / "single_cell_omics" / task_name / "data"
    task_dir.mkdir(parents=True, exist_ok=True)

    existing = [f for f in task_dir.iterdir() if not f.name.startswith('.') and f.name != "private"] if task_dir.exists() else []
    if existing:
        print(f"  [skip] {task_name} already has data ({existing[0].name}, ...)")
        return True

    copied = False

    # Try copying from biomlbench prepared cache (public + private)
    if BIOMLBENCH is not None:
        prep_root = BIOMLBENCH / "manual" / task_name / "prepared"
        public_base = prep_root / "public"
        private_base = prep_root / "private"
        if public_base.exists() and any(p for p in public_base.iterdir() if not p.name.startswith('.')):
            for f in public_base.iterdir():
                dest = task_dir / f.name
                if not dest.exists():
                    if f.is_dir():
                        shutil.copytree(f, dest)
                    else:
                        shutil.copy(f, dest)
            if private_base.exists():
                priv_dest = task_dir / "private"
                priv_dest.mkdir(exist_ok=True)
                for f in private_base.iterdir():
                    dest = priv_dest / f.name
                    if not dest.exists():
                        if f.is_dir():
                            shutil.copytree(f, dest)
                        else:
                            shutil.copy(f, dest)
            print(f"  [copy] {task_name} -> copied from biomlbench cache")
            copied = True

    # Fallback: download from S3 via biomlbench pipeline (requires AWS CLI + biomlbench package)
    if not copied:
        if BIOMLBENCH is None:
            print(f"  [fail] {task_name}: --biomlbench-dir not provided and no prepared cache found.")
            print(f"         Provide --biomlbench-dir pointing to a biomlbench clone to download from S3.")
            return False
        try:
            sys.path.insert(0, str(BIOMLBENCH))
            from biomlbench.data import download_and_prepare_dataset
            from biomlbench.registry import registry

            task = registry.get_task(f"manual/{task_name}")
            print(f"  [download] {task_name}: downloading from S3 (requires AWS CLI)...")
            download_and_prepare_dataset(task)

            public_dir = task.public_dir
            if public_dir.exists():
                for f in public_dir.iterdir():
                    dest = task_dir / f.name
                    if not dest.exists():
                        if f.is_dir():
                            shutil.copytree(f, dest)
                        else:
                            shutil.copy(f, dest)
            private_dir = getattr(task, "private_dir", public_dir.parent / "private" if public_dir else None)
            if private_dir is not None and Path(private_dir).exists():
                priv_dest = task_dir / "private"
                priv_dest.mkdir(exist_ok=True)
                for f in Path(private_dir).iterdir():
                    dest = priv_dest / f.name
                    if not dest.exists():
                        if f.is_dir():
                            shutil.copytree(f, dest)
                        else:
                            shutil.copy(f, dest)
            print(f"  [prepare] {task_name} -> downloaded and copied")
            copied = True

        except Exception as e:
            print(f"  [fail] {task_name}: {e}")
            print(f"         Single-cell data requires AWS CLI (pip install awscli) and valid credentials.")
            return False

    # Task-specific post-processing
    if task_name == "open-problems-spatially-variable-genes":
        try:
            fix_svg_label_leak(task_dir)
        except Exception as e:
            print(f"  [warn] {task_name}: shuffle post-processor failed: {e}")
            return False

    return copied


def fix_svg_label_leak(task_dir: Path,
                       seed_cortex: int = 2026,
                       seed_cereb: int = 2027) -> None:
    """
    Mitigate the cross-tissue label leak in `open-problems-spatially-variable-genes`.

    The biomlbench source pipeline ships cortex (`train.h5ad` + `private/answers.csv`)
    and cerebellum (`cerebellum_train.h5ad` + `cerebellum_labels.h5ad`) anonymized as
    `GENE1..GENE210` in alpha-sorted positional order. As a result, the cortex and
    cerebellum label vectors are identical when aligned by `gene_id` (Kendall tau = 1.0),
    and any per-position fusion of cerebellum into cortex predictions is the leak.

    Fix: independently random-permute gene order in cortex and cerebellum BEFORE the
    positional `GENE_i` names are reused, then rewrite each H5AD/CSV in place. Originals
    are preserved under `data/private/original/` so the operation is idempotent — re-runs
    re-derive the shuffled files from the saved originals.
    """
    import anndata as ad
    from scipy.stats import kendalltau

    private = task_dir / "private"
    orig = private / "original"
    orig.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        task_dir / "train.h5ad",
        task_dir / "cerebellum_train.h5ad",
        task_dir / "cerebellum_labels.h5ad",
        private / "answers.csv",
    ]
    for f in files_to_backup:
        if not f.exists():
            raise FileNotFoundError(f"Expected SVG file missing: {f}")
        dst = orig / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

    cortex_train  = ad.read_h5ad(orig / "train.h5ad")
    cereb_train   = ad.read_h5ad(orig / "cerebellum_train.h5ad")
    cereb_labels  = ad.read_h5ad(orig / "cerebellum_labels.h5ad")
    answers       = pd.read_csv(orig / "answers.csv")

    n = cortex_train.n_vars
    if not (cereb_train.n_vars == cereb_labels.n_vars == len(answers) == n):
        raise ValueError(
            f"Gene-count mismatch: cortex={n}, cereb_train={cereb_train.n_vars}, "
            f"cereb_labels={cereb_labels.n_vars}, answers={len(answers)}"
        )

    tau_before, _ = kendalltau(
        answers.set_index("gene_id")["true_spatial_var_score"].loc[cereb_labels.var_names].values,
        cereb_labels.var["true_spatial_var_score"].values,
    )

    perm_cx = np.random.default_rng(seed_cortex).permutation(n)
    perm_cb = np.random.default_rng(seed_cereb).permutation(n)
    new_names = [f"GENE{i+1}" for i in range(n)]

    cortex_train_shuf = cortex_train[:, perm_cx].copy()
    cortex_train_shuf.var_names = new_names
    cortex_train_shuf.var["feature_id"]   = new_names
    cortex_train_shuf.var["feature_name"] = new_names

    ans_orig_in_order = answers.set_index("gene_id")["true_spatial_var_score"].loc[
        [f"GENE{i+1}" for i in range(n)]
    ].values
    answers_shuf = pd.DataFrame({
        "gene_id": new_names,
        "true_spatial_var_score": ans_orig_in_order[perm_cx],
    })

    cereb_train_shuf  = cereb_train[:, perm_cb].copy()
    cereb_labels_shuf = cereb_labels[:, perm_cb].copy()
    for adata in (cereb_train_shuf, cereb_labels_shuf):
        adata.var_names = new_names
        adata.var["feature_id"]   = new_names
        adata.var["feature_name"] = new_names

    tau_after, _ = kendalltau(
        answers_shuf.set_index("gene_id")["true_spatial_var_score"].loc[cereb_labels_shuf.var_names].values,
        cereb_labels_shuf.var["true_spatial_var_score"].values,
    )

    cortex_train_shuf.write_h5ad(task_dir / "train.h5ad")
    cereb_train_shuf.write_h5ad(task_dir / "cerebellum_train.h5ad")
    cereb_labels_shuf.write_h5ad(task_dir / "cerebellum_labels.h5ad")
    answers_shuf.to_csv(private / "answers.csv", index=False)

    print(f"  [shuffle] SVG leak fix applied: tau(answers, cereb_labels) {tau_before:.4f} -> {tau_after:.4f}")
    print(f"            originals preserved at {orig}")


def prepare_kaggle_task(task_name: str, competition_id: str) -> bool:
    """Prepare a Kaggle competition task."""
    task_dir = TASK_ROOT / "biomedical_imaging" / task_name / "data"
    task_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = [f for f in task_dir.iterdir() if f.suffix == ".csv" and f.name != "sample_submission.csv"] if task_dir.exists() else []
    if existing_csvs:
        print(f"  [skip] {task_name} already has data ({existing_csvs[0].name}, ...)")
        return True

    # Try copying from biomlbench prepared cache, then add cv_fold where needed
    if BIOMLBENCH is not None:
        prep_pub = BIOMLBENCH / "kaggle" / competition_id / "prepared" / "public"
        if prep_pub.exists() and any(prep_pub.iterdir()):
            for f in prep_pub.iterdir():
                dest = task_dir / f.name
                if not dest.exists():
                    if f.is_dir():
                        shutil.copytree(f, dest)
                    else:
                        shutil.copy(f, dest)
            print(f"  [copy] {task_name} -> copied from biomlbench cache")
            _add_kaggle_cv_fold(task_dir, competition_id)
            return True

    # Download via Kaggle CLI
    try:
        import subprocess
        raw_dir = task_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition_id, "-p", str(raw_dir)],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        module_name = competition_id.replace("-", "_")
        try:
            prepare_module = __import__(
                f"biomlbench.tasks.kaggle.{module_name}.prepare",
                fromlist=["prepare"]
            )
            prepare_module.prepare(raw_dir, task_dir, task_dir / "private")
        except ImportError:
            print(f"  [warn] No biomlbench prepare script for {task_name}; raw data in {raw_dir}")

        _add_kaggle_cv_fold(task_dir, competition_id)
        print(f"  [download] {task_name} -> Kaggle data downloaded")
        return True

    except FileNotFoundError:
        print(f"  [fail] {task_name}: 'kaggle' CLI not found.")
        print(f"         Install with: pip install kaggle")
        print(f"         Set up credentials: place API token at ~/.kaggle/kaggle.json")
        print(f"         Accept competition rules at: https://kaggle.com/competitions/{competition_id}")
        return False
    except Exception as e:
        print(f"  [fail] {task_name}: {e}")
        return False


def _add_kaggle_cv_fold(task_dir: Path, competition_id: str) -> None:
    """Add reproducible patient-level cv_fold to Kaggle tasks that need it."""
    if competition_id == "osic-pulmonary-fibrosis-progression":
        train_path = task_dir / "train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path)
            df = add_patient_cv_fold(df, patient_col="Patient")
            df.to_csv(train_path, index=False)
            print(f"  [cv_fold] OSIC: patient-level 5-fold CV (KFold sorted, seed=42)")

    elif competition_id == "rsna-miccai-brain-tumor-radiogenomic-classification":
        labels_path = task_dir / "train_labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            df = add_patient_cv_fold(df, patient_col="BraTS21ID")
            df.to_csv(labels_path, index=False)
            print(f"  [cv_fold] RSNA: patient-level 5-fold CV (KFold sorted, seed=42)")


# ===========================================================================
# Main
# ===========================================================================

def main():
    global TASK_ROOT, BIOMLBENCH

    parser = argparse.ArgumentParser(
        description="Prepare all 24 biomlbench task datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--biomlbench-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a biomlbench data directory — the directory populated by "
            "`biomlbench prepare --data-dir PATH`. "
            "When provided, the script copies pre-prepared data from the local cache "
            "instead of downloading from the internet. "
            "Expected subdirectories: proteingym-dms/, polarishub/, manual/, kaggle/. "
            "For single-cell tasks the biomlbench Python package must also be importable "
            "so the S3 download pipeline can be used as a fallback."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write prepared data here instead of the default task-biomlbench tree. "
            "Useful for testing reproducibility in a clean directory without touching "
            "existing data."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        metavar="TASK_NAME",
        help="Prepare only this specific task (directory name, e.g. tdcommons-caco2-wang).",
    )
    parser.add_argument(
        "--category",
        choices=["protein_engineering", "drug_discovery", "single_cell_omics", "biomedical_imaging", "all"],
        default="all",
        help="Prepare only tasks in this category (default: all). Categories match the task-biomlbench/ subdirectory names.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        TASK_ROOT = args.output_dir.resolve()
        TASK_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {TASK_ROOT}")

    if args.biomlbench_dir is not None:
        BIOMLBENCH = args.biomlbench_dir.resolve()
        if not BIOMLBENCH.exists():
            parser.error(f"--biomlbench-dir does not exist: {BIOMLBENCH}")
        print(f"biomlbench dir: {BIOMLBENCH}")

    results = {}

    def should_run(name: str) -> bool:
        return (args.task is None) or (name == args.task)

    if args.category in ("protein_engineering", "all"):
        print("\n=== ProteinGym Substitution Tasks ===")
        for dms_id in PROTEINGYM_SUBSTITUTION_TASKS:
            name = f"proteingym-dms-{dms_id}"
            if should_run(name):
                print(f"\n[{name}]")
                results[name] = prepare_proteingym_task(dms_id, is_indel=False)

        print("\n=== ProteinGym Indel Tasks ===")
        for dms_id in PROTEINGYM_INDEL_TASKS:
            name = f"proteingym-dms-{dms_id}"
            if should_run(name):
                print(f"\n[{name}]")
                results[name] = prepare_proteingym_task(dms_id, is_indel=True)

    if args.category in ("drug_discovery", "all"):
        print("\n=== Polaris/TDCommons Molecular Property Tasks ===")
        for task_name, benchmark_id in POLARIS_TASKS:
            if should_run(task_name):
                print(f"\n[{task_name}]")
                results[task_name] = prepare_polaris_task(task_name, benchmark_id)

    if args.category in ("single_cell_omics", "all"):
        print("\n=== OpenProblems Single-Cell Tasks ===")
        for task_name in MANUAL_TASKS:
            if should_run(task_name):
                print(f"\n[{task_name}]")
                results[task_name] = prepare_manual_task(task_name)

    if args.category in ("biomedical_imaging", "all"):
        print("\n=== Kaggle Competition Tasks ===")
        for task_name, competition_id in KAGGLE_TASKS:
            if should_run(task_name):
                print(f"\n[{task_name}]")
                results[task_name] = prepare_kaggle_task(task_name, competition_id)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success = [k for k, v in results.items() if v]
    failed  = [k for k, v in results.items() if not v]
    print(f"Success ({len(success)}): {', '.join(success) if success else 'none'}")
    if failed:
        print(f"Failed  ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    print(f"\nData written to: {TASK_ROOT}/")
    print(f"  biomedical_imaging/{{task}}/data/")
    print(f"  single_cell_omics/{{task}}/data/")
    print(f"  protein_engineering/{{task}}/data/")
    print(f"  drug_discovery/{{task}}/data/")


if __name__ == "__main__":
    main()
