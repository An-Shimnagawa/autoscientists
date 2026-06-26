---
task_type: biomlbench
name: tdcommons-herg
description: >
  Binary classification: predict hERG channel blocking (cardiotoxicity) from drug SMILES.
  Maximize ROC-AUC using the TDCommons hERG benchmark.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
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

# TDCommons: hERG Blocking Prediction

**Metric:** ROC-AUC — higher is better  
**Task type:** Binary classification  
**Awards medals:** Yes  
**Source:** TDCommons hERG dataset

---

## The Problem

The hERG (human Ether-a-go-go Related Gene) protein is crucial for the heart's electrical activity. Blocking hERG channels causes QT interval prolongation, leading to potentially fatal cardiac arrhythmia. Predicting hERG blocking is a critical safety filter in drug development.

**Input:** SMILES string of a drug molecule  
**Output:** Binary — 1 (hERG blocker, cardiotoxic) or 0 (non-blocker), or probability  
**Evaluation:** ROC-AUC — higher is better

---

## Data

### Location
```
data/
├── train.csv          # Drug (SMILES), Y (0/1)
├── test_features.csv  # id, Drug (SMILES)
└── sample_submission.csv
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/polarishub/tdcommons-herg/prepared
cp $PREP/public/train.csv data/
cp $PREP/public/test_features.csv data/
cp $PREP/public/sample_submission.csv data/
```

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
2. **Save `submission.csv`** — predictions for all compounds in `data/test_features.csv` (columns: `id`, `Y`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** ROC-AUC — higher is better

Use the `cv_fold` column in `data/train.csv` for 5-fold scaffold cross-validation. Each fold uses
~20% of molecules from structurally distinct scaffolds as the validation set. Fold sizes: 123 / 109 / 102 / 97 / 92.

> **Why CV instead of a single split?** A fixed val split can be overfitted through repeated
> hyperparameter tuning — after many iterations the chosen hyperparameters work well on that specific
> val set but not on unseen test scaffolds. CV averages over 5 different val partitions, making it
> much harder to overfit the validation signal.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train_df = pd.read_csv("data/train.csv")
fold_scores = []

for k in range(5):
    tr  = train_df[train_df["cv_fold"] != k]
    val = train_df[train_df["cv_fold"] == k]
    val_labels = val["Y"].values

    # ... train model on tr, predict on val ...
    # val_preds = model.predict(val_features)

    score = roc_auc_score(val_labels, val_preds)
    fold_scores.append(score)

mean_score = np.mean(fold_scores)
std_score  = np.std(fold_scores)
print(f"Mean CV ROC-AUC: {mean_score:.4f} ± {std_score:.4f} (per fold: {[f'{s:.4f}' for s in fold_scores]})")
```

Report the following to the ClawLab leaderboard after each iteration:
- **Primary metric:** mean CV ROC-AUC (used to rank approaches and pick the champion)
- **Additional metrics:** per-fold ROC-AUC for each of the 5 folds, and the standard deviation across folds

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
submission = pd.DataFrame({"id": test_df["id"], "Y": predictions})
submission.to_csv("submission.csv", index=False)
```

The final submission is graded once against the held-out test labels using ROC-AUC.

---

## Output Format

```csv
id,Y
0,0.823
1,0.124
...
```

---



---

## Background

# herg

## Background
Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. Therefore, reliable prediction of hERG liability in the early stages of drug design is quite important to reduce the risk of cardiotoxicity-related attritions in the later development stages.

## Description of readout
Task Description: Binary classification. Given a drug SMILES string, predict whether it blocks (1) or not blocks (0).

## Data resource
**Reference**: [1] [ADMET Evaluation in Drug Discovery. 16. Predicting hERG Blockers by Combining Multiple Pharmacophores and Machine Learning Approaches](https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.6b00471)

---

**Source:** [Polaris Hub - tdcommons/herg](https://polarishub.io)  
**Task Type:** single_task  
**Main Metric:** roc_auc

## Data Format

This task uses the Polaris data source system:
- Data is downloaded from Polaris Hub using the PolarisDataSource
- Molecule column: `Drug`
- Target column: `{'Y'}` (first target if multiple available)

## Files

- `train.csv`: Training data with molecules and targets
- `test_features.csv`: Test features with ID column
- `sample_submission.csv`: Example submission format

## Evaluation

Uses official Polaris evaluation system with benchmark `tdcommons/herg`.
Main metric: **roc_auc**

## Source

Auto-generated from [Polaris Hub](https://polarishub.io/).

---
