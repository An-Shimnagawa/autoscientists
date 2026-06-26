---
task_type: biomlbench
name: tdcommons-lipophilicity-astrazeneca
description: >
  Predict lipophilicity (regression) of drug molecules from SMILES strings.
  Minimize Mean Absolute Error using the AstraZeneca DMPK dataset from TDCommons.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **No test leakage:** Fit preprocessing on train only.
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

# TDCommons: Lipophilicity (AstraZeneca)

**Metric:** Mean Absolute Error (MAE) — lower is better  
**Task type:** Regression  
**Awards medals:** Yes  
**Source:** AstraZeneca DMPK dataset (MoleculeNet)

---

## The Problem

Lipophilicity measures how readily a compound dissolves in lipid environments versus water. High lipophilicity leads to high metabolism, poor aqueous solubility, high turnover, and low absorption — all critical drug-likeness factors. The AstraZeneca DMPK dataset is a standard benchmark from MoleculeNet.

**Input:** SMILES string of a drug molecule  
**Output:** Lipophilicity value (continuous regression)  
**Evaluation:** Mean Absolute Error (MAE) — lower is better

---

## Data

### Location
```
data/
├── train.csv          # smiles (SMILES), Y (lipophilicity target)
├── test_features.csv  # id, smiles
└── sample_submission.csv
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/polarishub/tdcommons-lipophilicity-astrazeneca/prepared
cp $PREP/public/train.csv data/
cp $PREP/public/test_features.csv data/
cp $PREP/public/sample_submission.csv data/
```

### Data format
- `train.csv`: SMILES column + lipophilicity target `Y`
- `test_features.csv`: `id`, SMILES column

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

**Metric:** MAE — lower is better

Use the `cv_fold` column in `data/train.csv` for 5-fold scaffold cross-validation. Each fold uses
~20% of molecules from structurally distinct scaffolds as the validation set. Fold sizes: 718 / 675 / 661 / 657 / 649.

> **Why CV instead of a single split?** A fixed val split can be overfitted through repeated
> hyperparameter tuning — after many iterations the chosen hyperparameters work well on that specific
> val set but not on unseen test scaffolds. CV averages over 5 different val partitions, making it
> much harder to overfit the validation signal.

```python
import pandas as pd
import numpy as np

train_df = pd.read_csv("data/train.csv")
fold_scores = []

for k in range(5):
    tr  = train_df[train_df["cv_fold"] != k]
    val = train_df[train_df["cv_fold"] == k]
    val_labels = val["Y"].values

    # ... train model on tr, predict on val ...
    # val_preds = model.predict(val_features)

    score = np.mean(np.abs(val_labels - val_preds))
    fold_scores.append(score)

mean_score = np.mean(fold_scores)
std_score  = np.std(fold_scores)
print(f"Mean CV MAE: {mean_score:.4f} ± {std_score:.4f} (per fold: {[f'{s:.4f}' for s in fold_scores]})")
```

Report the following to the ClawLab leaderboard after each iteration:
- **Primary metric:** mean CV MAE (used to rank approaches and pick the champion)
- **Additional metrics:** per-fold MAE for each of the 5 folds, and the standard deviation across folds

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

The final submission is graded once against the held-out test labels using MAE.

---

## Output Format

```csv
id,Y
0,2.456
1,1.234
...
```

---



---

## Background

# lipophilicity-astrazeneca

## Background
Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption. From MoleculeNet.

## Description of readout
Task Description: Regression. Given a drug SMILES string, predict the activity of lipophilicity.


## Data resource
**Reference**: 

- [AstraZeneca. Experimental in vitro Dmpk and physicochemical data on a set of publicly disclosed compounds (2016)](https://doi.org/10.6019/chembl3301361)

- [Wu, Zhenqin, et al. “MoleculeNet: a benchmark for molecular machine learning.” Chemical science 9.2 (2018): 513-530.](https://pubs.rsc.org/--/content/articlehtml/2018/sc/c7sc02664a)


---

**Source:** [Polaris Hub - tdcommons/lipophilicity-astrazeneca](https://polarishub.io)  
**Task Type:** single_task  
**Main Metric:** mean_absolute_error

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

Uses official Polaris evaluation system with benchmark `tdcommons/lipophilicity-astrazeneca`.
Main metric: **mean_absolute_error**

## Source

Auto-generated from [Polaris Hub](https://polarishub.io/).

---
