---
task_type: biomlbench
name: tdcommons-cyp2d6-substrate-carbonmangels
description: >
  Binary classification: predict whether a drug is a CYP2D6 substrate from its SMILES string.
  Maximize PR-AUC using the Polaris/TDCommons cyp2d6-substrate-carbonmangels benchmark.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **Imbalanced classes:** Dataset is imbalanced — use PR-AUC not ROC-AUC.
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

# TDCommons: CYP2D6 Substrate Classification

**Metric:** PR-AUC (Precision-Recall AUC) — higher is better  
**Task type:** Binary classification  
**Awards medals:** Yes  
**Source:** Carbonmangels et al., merged dataset from 6 publications

---

## The Problem

CYP2D6 is a cytochrome P450 enzyme primarily in the liver responsible for metabolizing ~25% of clinical drugs. Predicting whether a drug is a CYP2D6 substrate (1) or non-substrate (0) is critical for drug safety and drug-drug interaction prediction.

**Input:** SMILES string of a drug molecule  
**Output:** Binary classification — 1 (substrate) or 0 (non-substrate), or probability  
**Evaluation:** PR-AUC — higher is better

---

## Data

### Location
```
data/
├── train.csv          # Drug (SMILES), Y (0/1 substrate label)
├── test_features.csv  # id, Drug (SMILES)
└── sample_submission.csv
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

# Copy from biomlbench prepared data if available
PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/polarishub/tdcommons-cyp2d6-substrate-carbonmangels/prepared
cp $PREP/public/train.csv data/
cp $PREP/public/test_features.csv data/
cp $PREP/public/sample_submission.csv data/

# Or prepare via Polaris
$PYTHON -c "
import polaris as po, pandas as pd
from pathlib import Path
benchmark = po.load_benchmark('tdcommons/cyp2d6-substrate-carbonmangels')
train, test = benchmark.get_train_test_split()
Path('data').mkdir(exist_ok=True)
# Save train/test CSVs
"
```

### Data format
- `train.csv`: columns `Drug` (SMILES), `Y` (0/1 binary)
- `test_features.csv`: columns `id`, `Drug` (SMILES)

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

**Metric:** PR-AUC — higher is better

Use the `cv_fold` column in `data/train.csv` for 5-fold scaffold cross-validation. Each fold uses
~20% of molecules from structurally distinct scaffolds as the validation set. Fold sizes: 143 / 109 / 98 / 92 / 90 — class imbalance (~28% positive); per-fold PR-AUC can vary by ±0.05.

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
    val_labels = val["Y"].values

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
submission = pd.DataFrame({"id": test_df["id"], "Y": predictions})
submission.to_csv("submission.csv", index=False)
```

The final submission is graded once against the held-out test labels using PR-AUC.

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

# cyp2d6-substrate-carbonmangels

## Background
CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra. TDC used a dataset from [1], which merged information on substrates and nonsubstrates from six publications.

## Description of readout
Task Description: Binary Classification. Given a drug SMILES string, predict if it is a substrate to the enzyme.

## Data resource
**References**: [1] [Selecting relevant descriptors for classification by bayesian estimates: a comparison with decision trees and support vector machines approaches for disparate data sets.](https://doi.org/10.1002/minf.201100069)

[2] [admetSAR: a comprehensive source and free tool for assessment of chemical ADMET properties.](https://pubs.acs.org/doi/10.1021/ci300367a)

---

**Source:** [Polaris Hub - tdcommons/cyp2d6-substrate-carbonmangels](https://polarishub.io)  
**Task Type:** single_task  
**Main Metric:** pr_auc

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

Uses official Polaris evaluation system with benchmark `tdcommons/cyp2d6-substrate-carbonmangels`.
Main metric: **pr_auc**

## Source

Auto-generated from [Polaris Hub](https://polarishub.io/).

---
