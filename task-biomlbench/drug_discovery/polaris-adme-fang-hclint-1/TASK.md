---
task_type: biomlbench
name: polaris-adme-fang-hclint-1
description: >
  Predict human liver microsomal clearance (LOG HLM_CLint) from SMILES strings (regression).
  Maximize Pearson correlation using the Polaris Fang 2023 DMPK benchmark.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **Log-transformed target:** Data uses LOG HLM_CLint.
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

# Polaris: Human Liver Microsomal Clearance (polaris-adme-fang-hclint-1)

**Metric:** Pearson correlation — higher is better  
**Task type:** Regression  
**Awards medals:** Yes  
**Train:** 2,229, **Test:** 575 molecules  
**Source:** Fang et al. 2023 DMPK dataset

---

## The Problem

Human Liver Microsomal (HLM) intrinsic clearance (CLint) measures how rapidly the liver metabolizes a compound. High CLint means rapid clearance → short half-life → needs frequent dosing. Predicting LOG HLM_CLint is essential for ADME profiling in drug development.

**Input:** SMILES string  
**Output:** LOG HLM_CLint (continuous, log-transformed clearance)  
**Evaluation:** Pearson correlation — higher is better

---

## Data

### Location
```
data/
├── train.csv          # SMILES column, HLM_CLint target
├── test_features.csv  # id, SMILES column
└── sample_submission.csv
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/polarishub/polaris-adme-fang-hclint-1/prepared
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
2. **Save `submission.csv`** — predictions for all compounds in `data/test_features.csv` (columns: `id`, `LOG_HLM_CLint`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** Pearson r — higher is better

Use the `cv_fold` column in `data/train.csv` for 5-fold scaffold cross-validation. Each fold uses
~20% of molecules from structurally distinct scaffolds as the validation set. Fold sizes: 466 / 445 / 442 / 439 / 437.

> **Why CV instead of a single split?** A fixed val split can be overfitted through repeated
> hyperparameter tuning — after many iterations the chosen hyperparameters work well on that specific
> val set but not on unseen test scaffolds. CV averages over 5 different val partitions, making it
> much harder to overfit the validation signal.

```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

train_df = pd.read_csv("data/train.csv")
fold_scores = []

for k in range(5):
    tr  = train_df[train_df["cv_fold"] != k]
    val = train_df[train_df["cv_fold"] == k]
    val_labels = val["LOG_HLM_CLint"].values

    # ... train model on tr, predict on val ...
    # val_preds = model.predict(val_features)

    r, _ = pearsonr(val_labels, val_preds)
    fold_scores.append(r)

mean_score = np.mean(fold_scores)
std_score  = np.std(fold_scores)
print(f"Mean CV Pearson r: {mean_score:.4f} ± {std_score:.4f} (per fold: {[f'{s:.4f}' for s in fold_scores]})")
```

Report the following to the ClawLab leaderboard after each iteration:
- **Primary metric:** mean CV Pearson r (used to rank approaches and pick the champion)
- **Additional metrics:** per-fold Pearson r for each of the 5 folds, and the standard deviation across folds

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
submission = pd.DataFrame({"id": test_df["id"], "LOG_HLM_CLint": predictions})
submission.to_csv("submission.csv", index=False)
```

The final submission is graded once against the held-out test labels using Pearson r.

---

## Output Format

```csv
id,HLM_CLint
0,2.456
1,1.123
...
```

(Use the actual target column name from `train.csv`)

---



---

## Background

# adme-fang-HCLint-1

![ADME](https://storage.googleapis.com/polaris-public/icons/icons8-whale-96-ADME.png) 

## Background

The goal of accessing ADME properties is to understand how a potential drug candidate interacts with the human body, including absorption, distribution, metabolism, and excretion. This knowledge is crucial for evaluating efficacy, safety, and clinical potential, guiding drug development for optimal therapeutic outcomes. [Fang et al. 2023](https://doi.org/10.1021/acs.jcim.3c00160) has disclosed DMPK datasets collected over 20 months across six ADME in vitro endpoints, which are human and rat liver microsomal stability, MDR1-MDCK efflux ratio, solubility, and human and rat plasma protein binding. The dataset contains 885 to 3087 measures for the corresponding endpoints. 

## Benchmarking
**The goal** of this benchmark is to perform a single task, which is to the best predictive model for human liver microsomal stability. 


## Description of readout 
- **Readouts**: `LOG HLM_CLint (mL/min/kg)`
- **Bioassay readout**: Intrinsic clearance
- **Optimization objective**: Higher value
- **Number of data points**: train: 2229, test:  575

## Molecule data resource:
**Reference**: https://doi.org/10.1021/acs.jcim.3c00160

## Train/test split
In this benchmark set, the same train/test sets in the fang2023 paper were used for the 6 endpoints human and rat liver microsomal stability, MDR1-MDCK efflux ratio, solubility, and human and rat plasma protein binding, respectively. 
See more details at https://github.com/molecularinformatics/Computational-ADME/tree/main/MPNN.

**Distribution of the train/test in the chemical space**
![image](https://storage.googleapis.com/polaris-public/datasets/ADME/fang2023/figures/fang2023_ADME_public_v1_HLM_tsne_fang2023split.png)


## Related links
The full curation and creation process is documented [here](https://github.com/polaris-hub/polaris-recipes/blob/main/01_ADME).


---

**Source:** [Polaris Hub - polaris/adme-fang-hclint-1](https://polarishub.io)  
**Task Type:** single_task  
**Main Metric:** pearsonr

## Data Format

This task uses the Polaris data source system:
- Data is downloaded from Polaris Hub using the PolarisDataSource
- Molecule column: `smiles`
- Target column: `{'LOG_HLM_CLint'}` (first target if multiple available)

## Files

- `train.csv`: Training data with molecules and targets
- `test_features.csv`: Test features with ID column
- `sample_submission.csv`: Example submission format

## Evaluation

Uses official Polaris evaluation system with benchmark `polaris/adme-fang-hclint-1`.
Main metric: **pearsonr**

## Source

Auto-generated from [Polaris Hub](https://polarishub.io/).

---
