---
task_type: biomlbench
name: tdcommons-caco2-wang
description: >
  Predict Caco-2 cell effective permeability (regression) from drug SMILES strings.
  Minimize Mean Absolute Error using the Polaris/TDCommons caco2-wang benchmark.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **No test leakage:** Feature selection and scaling must be fitted on training data only.
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

# TDCommons: Caco-2 Cell Permeability (caco2-wang)

**Metric:** Mean Absolute Error (MAE) — lower is better  
**Task type:** Regression  
**Awards medals:** Yes  
**Source:** Wang et al. 2016 (DOI: 10.1021/ci300400a)

---

## The Problem

The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate human intestinal tissue. Predicting the rate of drug permeation through Caco-2 cells approximates intestinal absorption — a critical ADME property.

**Input:** SMILES string of a drug molecule  
**Output:** Caco-2 cell effective permeability (continuous regression)  
**Evaluation:** Mean Absolute Error (MAE) — lower is better

---

## Data

### Location
```
data/
├── train.csv          # training molecules + targets
├── test_features.csv  # test molecules (id + SMILES, no targets)
└── sample_submission.csv
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

$PYTHON << 'EOF'
import sys; sys.path.insert(0, '/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench')
from biomlbench.tasks.polarishub.tdcommons_caco2_wang.prepare import prepare
from pathlib import Path
import polaris as po

benchmark = po.load_benchmark("tdcommons/caco2-wang")
train, test = benchmark.get_train_test_split()

raw = Path('data/raw'); raw.mkdir(parents=True, exist_ok=True)
# Save polaris data
import pandas as pd
train_df = pd.DataFrame({'Drug': [x for x, _ in train], 'Y': [y for _, y in train]})
test_df = pd.DataFrame({'Drug': [x for x, _ in test], 'Y': [y for _, y in test]})
train_df.to_csv(raw / 'polaris_train_data.csv', index=False)
test_df.to_csv(raw / 'polaris_test_data.csv', index=False)

prepare(raw, Path('data'), Path('data/private'))
EOF
```

Or copy from biomlbench prepared data if available:
```bash
PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/polarishub/tdcommons-caco2-wang/prepared
cp $PREP/public/train.csv data/
cp $PREP/public/test_features.csv data/
cp $PREP/public/sample_submission.csv data/
```

### Data format
- `train.csv`: columns `Drug` (SMILES), `Y` (permeability target)
- `test_features.csv`: columns `id`, `Drug` (SMILES)
- `sample_submission.csv`: columns `id`, `Y`

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
~20% of molecules from structurally distinct scaffolds as the validation set. Fold sizes: 174 / 152 / 138 / 134 / 130.

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

Save predictions as `submission.csv`:
```csv
id,Y
0,0.456
1,-0.234
...
```

---



---

## Background

# caco2-wang

## Background
The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

## Description of readout
Task Description: Regression. Given a drug SMILES string, predict the Caco-2 cell effective permeability.


## Data resource
**Reference**: [Wang, NN et al, ADME Properties Evaluation in Drug Discovery: Prediction of Caco-2 Cell Permeability Using a Combination of NSGA-II and Boosting, Journal of Chemical Information and Modeling 2016 56 (4), 763-773](https://pubs.acs.org/doi/10.1021/ci300400a)


---

**Source:** [Polaris Hub - tdcommons/caco2-wang](https://polarishub.io)  
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

Uses official Polaris evaluation system with benchmark `tdcommons/caco2-wang`.
Main metric: **mean_absolute_error**

## Source

Auto-generated from [Polaris Hub](https://polarishub.io/).

---
