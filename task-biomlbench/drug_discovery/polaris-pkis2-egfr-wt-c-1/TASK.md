---
task_type: biomlbench
name: polaris-pkis2-egfr-wt-c-1
description: >
  Binary classification: predict EGFR wild-type kinase inhibition (>80% inhibition) from SMILES.
  Maximize PR-AUC using the Polaris PKIS2 EGFR benchmark.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **Scaffold split:** Test set has different scaffolds than train — fingerprint similarity alone may underperform.
- **Compute:** See `program_biomlbench.md` for wall-clock limit and GPU/CPU settings.

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out ground-truth labels for `data/test_features.csv`.
> It is used **only** by the external grader to score your final `submission.csv`. **Never read, open,
> or use this file in `train.py` or any intermediate evaluation.** Doing so would invalidate your score.
>
> **How to evaluate during development:** Use the `cv_fold` column in `data/train.csv`.
> It assigns each molecule a fold number 0–4 computed via Murcko scaffold grouping (round-robin by
> descending group size). For each fold `k`, train on rows where `cv_fold != k` and validate on rows
> where `cv_fold == k`. Report the **mean score across all 5 folds** to the ClawLab leaderboard.
> See the **Iterative Leaderboard** section below for the exact code to do this.

---

# Polaris: EGFR Wild-Type Kinase Inhibition Classification (pkis2-egfr-wt-c-1)

**Metric:** PR-AUC (Precision-Recall AUC) — higher is better  
**Task type:** Binary classification  
**Awards medals:** Yes  
**Train:** 496, **Test:** 144 compounds  
**Source:** PKIS2 (DOI: 10.1038/s41589-019-0325-7)

---

## The Problem

EGFR (Epidermal Growth Factor Receptor) is a receptor tyrosine kinase overexpressed in many cancers. This benchmark classifies whether a compound achieves >80% inhibition of wild-type EGFR. The dataset uses scaffold-based splitting to test generalization to new chemical scaffolds.

**Input:** SMILES string  
**Output:** Binary — 1 (>80% inhibition, positive hit) or 0, or probability  
**Evaluation:** PR-AUC — higher is better (preferred over ROC-AUC for hit discovery)

---

## Data

### Location
```
data/
├── train.csv          # smiles, CLASS_EGFR (0/1)
├── test_features.csv  # id, smiles
└── sample_submission.csv
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/polarishub/polaris-pkis2-egfr-wt-c-1/prepared
cp $PREP/public/train.csv data/
cp $PREP/public/test_features.csv data/
cp $PREP/public/sample_submission.csv data/
```

### Data format
- `train.csv`: `smiles` (SMILES), `CLASS_EGFR` (0/1)
- `test_features.csv`: `id`, `smiles`

---

## How to Run

**`train.py` is NOT provided. You must write it from scratch.**

Your workflow:

1. **Write `train.py`** — implement a model that trains on the provided data split,
   evaluates on the val split, prints the val metric, and saves `submission.csv`.
2. **Run it** to get a baseline val score:
   ```bash
   PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
   cd $FOCUS_ROOT/task
   $PYTHON train.py
   ```
3. **Report the val score** to the ClawLab leaderboard (see Iterative Leaderboard section below).
4. **Iterate** — improve `train.py`, re-run, report each new score. Save each version as
   `train_v{N}.py` so you can compare and recover prior approaches.
5. **Select your best version** — once you have tried multiple approaches, pick the one
   with the highest val score, retrain it cleanly, and produce the final `submission.csv`.
6. **Save `submission.csv`** to `$FOCUS_ROOT/task/submission.csv` before the wall-clock
   limit expires — this is the file the grader scores.
7. **Save `train.py`** to `$FOCUS_ROOT/task/train.py` alongside `submission.csv` — this
   must be the exact script used to produce the final `submission.csv`, with all data paths
   hardcoded or resolved from the `data/` directory so the run is fully reproducible.

`train.py` must do three things on every run:
1. **Evaluate locally** — train on the data split, predict on the val split, print the val metric.
2. **Save `submission.csv`** — predictions for all compounds in `data/test_features.csv` (columns: `id`, `CLASS_EGFR`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** PR-AUC — higher is better

Use the `cv_fold` column in `data/train.csv` for 5-fold scaffold cross-validation. Each fold uses
~20% of molecules from structurally distinct scaffolds as the validation set. Fold sizes: 120 / 99 / 94 / 92 / 91.

> **Why CV instead of a single split?** A fixed val split can be overfitted through repeated
> hyperparameter tuning — after many iterations the chosen hyperparameters work well on that specific
> val set but not on unseen test scaffolds. CV averages over 5 different val partitions, making it
> much harder to overfit the validation signal.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

train_df = pd.read_csv("data/train.csv")
fold_scores = []

for k in range(5):
    tr  = train_df[train_df["cv_fold"] != k]
    val = train_df[train_df["cv_fold"] == k]
    val_labels = val["CLASS_EGFR"].values

    # ... train model on tr, predict on val ...
    # val_preds = model.predict(val_features)

    score = average_precision_score(val_labels, val_preds)
    fold_scores.append(score)

mean_score = np.mean(fold_scores)
std_score  = np.std(fold_scores)
print(f"Mean CV PR-AUC: {mean_score:.4f} ± {std_score:.4f} (per fold: {[f'{s:.4f}' for s in fold_scores]})")
```

Report the following to the ClawLab leaderboard after each iteration:
- **Primary metric:** mean CV PR-AUC (used to rank approaches and pick the champion)
- **Additional metrics:** per-fold PR-AUC for each of the 5 folds, and the standard deviation across folds

Only treat an improvement as real if it is consistent across most folds — a gain on a single fold
while others regress is noise, not signal. High standard deviation (relative to the mean) is a
sign of instability and should make you prefer a simpler, more consistent approach.

---

## Final Submission

After identifying your best approach via the iterative leaderboard, **retrain on ALL of `data/train.csv`** (no val split) before running inference on `data/test_features.csv`. This maximises the data available for the final model.

```python
# Final training — use full training set
full_train = pd.read_csv("data/train.csv")
# ... retrain model on full_train ...

# Run inference on test set
test_df = pd.read_csv("data/test_features.csv")
# ... predict ...

# Save submission
submission = pd.DataFrame({"id": test_df["id"], "CLASS_EGFR": predictions})
submission.to_csv("submission.csv", index=False)
```

The final submission is graded once against the held-out test labels using PR-AUC.

---

## Output Format

```csv
id,CLASS_EGFR
0,0.892
1,0.034
...
```

---



---

## Background

# pkis2-egfr-wt-c-1

![molprop](https://storage.googleapis.com/polaris-public/icons/icons8-fox-60-kinases.png)



## Background
**EGFR (Epidermal Growth Factor Receptor) kinase** is a type of receptor tyrosine kinase that plays a significant role in cell growth, proliferation, and survival. Mutations or overexpression of EGFR have been associated with various diseases, particularly cancer.

## Benchmarking
**EGFR Wild type**:  Targeting wild-type EGFR with small-molecule inhibitors, such as erlotinib, is an ongoing area of research in the treatment of glioblastoma. While early findings are promising, the complexity of glioblastoma biology presents challenges that require further investigation to improve treatment outcomes for patients.

**The goal** of this benchmark is to perform a single task, which is to the best predictive model for 
- Optimization of the bioactivity % inhibition for EGFR wile type.
- Discovery of potential hits in new chemical space.


## Description of readout 
- **Readouts**: `CLASS_EGFR`
- **Bioassay readout**: percentage of inhnibition.
- **Optimization objective**: postive label (1)
- **Number of data points**: train:  496 test:  144
- **Thresholds**:  > 80

## Data resource: 
- **Reference**: [PKIS2](https://www.ncbi.nlm.nih.gov/pubmed/28767711)

## Train/test split
Given the benchmarking goal, a scaffold-based splitting approach was applied to ensure training and test sets contain distinct chemical structures while maintaining the diversity of scaffolds.

**Distribution of the train/test in the chemical space**
![image](https://storage.googleapis.com/polaris-public/datasets/kinases/egfr/figures/drewry_egfr_wildtype_v1_tnse_scaffold_split.png)


## Related links
The full curation and creation process is documented -> [notebook](https://github.com/polaris-hub/polaris-recipes/blob/main/03_Kinases/EGFR)

## Related benchmarks
- polaris/drewry_egfr_wildtype_singletask_reg_v1
- polaris/egfr_wt_l858r_v1
> Note: It's recommanded to evaluate your methods agaisnt all the benchmarks related to this dataset. 


---

**Source:** [Polaris Hub - polaris/pkis2-egfr-wt-c-1](https://polarishub.io)  
**Task Type:** single_task  
**Main Metric:** pr_auc

## Data Format

This task uses the Polaris data source system:
- Data is downloaded from Polaris Hub using the PolarisDataSource
- Molecule column: `smiles`
- Target column: `{'CLASS_EGFR'}` (first target if multiple available)

## Files

- `train.csv`: Training data with molecules and targets
- `test_features.csv`: Test features with ID column
- `sample_submission.csv`: Example submission format

## Evaluation

Uses official Polaris evaluation system with benchmark `polaris/pkis2-egfr-wt-c-1`.
Main metric: **pr_auc**

## Source

Auto-generated from [Polaris Hub](https://polarishub.io/).

---
