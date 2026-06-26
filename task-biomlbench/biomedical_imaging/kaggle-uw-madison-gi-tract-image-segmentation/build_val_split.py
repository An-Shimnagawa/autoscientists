"""Build a deterministic 2-regime validation split from train_split.csv.

Mirrors the public Kaggle test-set design (some cases held out entirely, others
held out by later treatment days within shared cases) so the local val score
correlates with the leaderboard score.

Run once before iterating. Produces three files alongside train_split.csv:
    train_for_modeling.csv   -- training rows (train_split.csv minus val)
    val.csv                  -- val rows with labels
                                (schema: id, class, predicted — matches
                                train_split.csv minus the column rename)
    val_case_days.txt        -- list of (case, case_day) pairs in val

`data/` is a symlink to /n/netscratch/.../uw-madison-gi-tract-image-segmentation/;
this script writes to that netscratch path directly so the data/ symlink picks
up the new files.

Usage:
    /n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python build_val_split.py
"""

import random

import pandas as pd

DATA_DIR = "/n/netscratch/mzitnik_lab/Lab/afang/kaggle/uw-madison-gi-tract-image-segmentation"

# Knobs — tune to balance val size vs training signal.
RANDOM_SEED = 0
HOLDOUT_CASE_FRAC = 0.10            # fraction of train_split cases held out entirely
TEMPORAL_HOLDOUT_LAST_N_DAYS = 1    # last N days per multi-day case go to val
# Mirror the test split logic: only cases with ≥4 scan days had their later
# days moved to the test set by prepare.py. Matching that threshold here keeps
# the val regime-mix (~60% whole-case / ~40% temporal) close to the real test.
MIN_DAYS_FOR_TEMPORAL_SPLIT = 4


def main() -> None:
    train = pd.read_csv(f"{DATA_DIR}/train_split.csv")
    train["case"] = train["id"].apply(lambda x: x.split("_")[0])
    train["day"] = train["id"].apply(lambda x: x.split("_")[1])
    train["case_day"] = train["id"].apply(lambda x: x.rsplit("_slice_", 1)[0])

    cases = sorted(train["case"].unique())

    # Whole-case hold-out: deterministic random sample of cases.
    rng = random.Random(RANDOM_SEED)
    n_holdout = max(1, int(round(len(cases) * HOLDOUT_CASE_FRAC)))
    holdout_cases = set(rng.sample(cases, n_holdout))

    val_case_days: set[str] = set()

    # Every (case, day) from whole-case hold-out cases goes to val.
    for case in holdout_cases:
        for d in train.loc[train["case"] == case, "day"].unique():
            val_case_days.add(f"{case}_{d}")

    # For each remaining case with enough days, the LAST N days go to val.
    for case in cases:
        if case in holdout_cases:
            continue
        days = sorted(
            train.loc[train["case"] == case, "day"].unique(),
            key=lambda d: int(d.replace("day", "")),
        )
        if len(days) < MIN_DAYS_FOR_TEMPORAL_SPLIT:
            continue
        for d in days[-TEMPORAL_HOLDOUT_LAST_N_DAYS:]:
            val_case_days.add(f"{case}_{d}")

    is_val = train["case_day"].isin(val_case_days)
    val = train[is_val].copy()
    trn = train[~is_val].copy()

    # train_for_modeling keeps train_split.csv schema (id, class, segmentation).
    # val uses the same schema but renames segmentation -> predicted to match
    # submission.csv and test_split.csv conventions.
    val = val.rename(columns={"segmentation": "predicted"})
    val = val[["id", "class", "predicted"]]
    trn = trn[["id", "class", "segmentation"]]

    trn.to_csv(f"{DATA_DIR}/train_for_modeling.csv", index=False, na_rep="")
    val.to_csv(f"{DATA_DIR}/val.csv", index=False, na_rep="")
    with open(f"{DATA_DIR}/val_case_days.txt", "w") as f:
        for cd in sorted(val_case_days):
            f.write(cd + "\n")

    # Summary
    val_case = val["id"].apply(lambda x: x.split("_")[0])
    n_val_holdout_rows = int(val_case.isin(holdout_cases).sum())
    n_val_temporal_rows = len(val) - n_val_holdout_rows
    n_train_case_days = trn["id"].apply(lambda x: x.rsplit("_slice_", 1)[0]).nunique()

    total_rows = len(trn) + len(val)
    print(f"cases: total={len(cases)}, whole-case holdout={n_holdout}, "
          f"shared (split by day)={len(cases) - n_holdout}")
    print(f"(case,day) pairs: train_for_modeling={n_train_case_days}, val={len(val_case_days)}")
    print(f"rows: train_for_modeling={len(trn)} ({100*len(trn)//total_rows}%), "
          f"val={len(val)} ({100*len(val)//total_rows}%)")
    print(f"  whole-case rows in val: {n_val_holdout_rows}")
    print(f"  temporal rows in val:   {n_val_temporal_rows}")
    print(f"\noutputs written to {DATA_DIR}/")
    print(f"  train_for_modeling.csv ({len(trn)} rows)")
    print(f"  val.csv                ({len(val)} rows)")
    print(f"  val_case_days.txt      ({len(val_case_days)} entries)")


if __name__ == "__main__":
    main()
