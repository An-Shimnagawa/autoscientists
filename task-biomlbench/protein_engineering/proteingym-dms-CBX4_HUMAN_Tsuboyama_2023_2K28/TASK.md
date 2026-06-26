---
task_type: biomlbench
name: proteingym-dms-CBX4_HUMAN_Tsuboyama_2023_2K28
description: >
  Predict stability fitness of unseen multi-substitution variants of E3 SUMO-protein ligase CBX4
  from Homo sapiens. Maximize Spearman correlation using 5-fold CV across 3 split strategies.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **No test leakage:** Train only on other folds.
- **Compute:** See `program_biomlbench.md` for wall-clock limit and GPU/CPU settings.

---

# ProteinGym DMS: CBX4_HUMAN_Tsuboyama_2023_2K28

**Metric:** Spearman correlation — higher is better  
**Difficulty:** Medium  
**Dataset size:** 2,282 sequences (multi-substitution variants)  
**Source:** "Mega-scale experimental analysis of protein folding stability in biology and design" (DOI: 10.1038/s41586-023-06328-6)

---

## The Problem

Deep mutational scanning dataset measuring **stability** fitness for **multi-substitution** variants of E3 SUMO-protein ligase CBX4 from *Homo sapiens* (Uniprot: CBX4_HUMAN). This is a human protein making it clinically relevant — CBX4 plays a role in chromatin remodeling and has implications in cancer biology.

**Input:** Amino acid sequence of the variant  
**Output:** Three predicted fitness scores (one per CV split strategy)  
**Dataset:** 2,282 sequences with measured stability scores

---

## Data

### Location
```
data/
└── data.csv
└── sample_submission.csv
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

# Copy from biomlbench prepared data if available
PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/proteingym-dms/CBX4_HUMAN_Tsuboyama_2023_2K28/prepared/public
cp $PREP/data.csv data/ 2>/dev/null || echo "Run biomlbench data preparation first"

# Or use biomlbench API to download and prepare
$PYTHON -c "
import sys; sys.path.insert(0, '$BIOMLBENCH')
from biomlbench.data_sources.proteingym import ProteinGymDMSDataSource
from pathlib import Path
src = ProteinGymDMSDataSource('CBX4_HUMAN_Tsuboyama_2023_2K28')
raw = Path('data/raw'); raw.mkdir(parents=True, exist_ok=True)
src.download(raw)
"
```

### Data format (`data/data.csv`)
- `id`: sequence index
- `sequence`: amino acid sequence of the variant
- `fitness_score`: measured stability (target variable)
- `fold_random_5`, `fold_modulo_5`, `fold_contiguous_5`: fold assignments (0–4)

First row is the wild-type sequence (fitness_score = NaN).

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
2. **Save `submission.csv`** — OOF fitness score predictions for all variants in `data/data.csv` across all 3 fold strategies (columns: `id`, `fitness_score_fold_random_5`, `fitness_score_fold_modulo_5`, `fitness_score_fold_contiguous_5`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** Mean Spearman correlation averaged across all 3 fold strategies — higher is better

Use the prescribed fold columns in `data/data.csv` for 5-fold CV. This IS the test evaluation — no separate held-out test set exists for ProteinGym tasks.

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

data = pd.read_csv("data/data.csv")
wt_row = data[data["id"] == "WT"]
data = data[data["id"] != "WT"].copy()  # exclude WT from scoring

fold_columns = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
all_preds = {col: np.zeros(len(data)) for col in fold_columns}

for col in fold_columns:
    for fold in range(5):
        val_mask = data[col] == fold
        tr, val = data[~val_mask], data[val_mask]
        # ... train on tr, predict on val ...
        all_preds[col][val_mask.values] = model.predict(val)

scores = [spearmanr(data["fitness_score"], all_preds[c]).correlation for c in fold_columns]
mean_spearman = np.mean(scores)
print(f"Mean Spearman: {mean_spearman:.4f}  (random={scores[0]:.4f}, modulo={scores[1]:.4f}, contiguous={scores[2]:.4f})")
```

Report `mean_spearman` to the ClawLab leaderboard after each iteration.

---

## Final Submission

For ProteinGym tasks, the iterative leaderboard score IS the final evaluation — there is no separate held-out test set. The prescribed fold splits define both development and final scoring.

**Do not retrain on all data for the submission.** The submission is the out-of-fold predictions from the 5-fold CV above, which already uses all data exactly once.

```python
# submission.csv = OOF predictions from the 5-fold CV
sub = data[["id"]].copy()
for col in fold_columns:
    sub[f"fitness_score_{col}"] = all_preds[col]
sub.to_csv("submission.csv", index=False)
```

## Output Format

```csv
id,fitness_score_fold_random_5,fitness_score_fold_modulo_5,fitness_score_fold_contiguous_5
0,0.123,0.134,0.119
...
```

---



---

## Background

# Proteingym-DMS dataset: CBX4_HUMAN_Tsuboyama_2023_2K28

## Description

This dataset is part of the ProteinGym DMS benchmark, which contains deep mutational scanning datasets that measure
protein fitness (in different contexts) for sequence variants of a wide range of proteins. This dataset contains
multi-substitution variants for the protein E3 SUMO-protein ligase CBX4 from the organism Homo sapiens. This protein has Uniprot ID: CBX4_HUMAN. 

The DMS selection assay was described as follows: 

    Stability

It was categorised as measuring the following (general) fitness attribute: Stability. Higher scores indicate better fitness.

The source publication for this dataset is titled: 

"Mega-scale experimental analysis of protein folding stability in biology and design"

and can be accessed at the following DOI: 10.1038/s41586-023-06328-6.

## Objective

The objective of this benchmark is to train a model that can predict the fitness of unseen single-substitution sequence variants of E3 SUMO-protein ligase CBX4.
To train your model, you will use 5-fold cross-validation on the sequences and fitness scores defined in the `data.csv` file. 

You will use the `fold_random_5`, `fold_modulo_5`, and `fold_contiguous_5` columns to split the data into training and test sets.
Each of these columns contains integer values from 0 to 4, which indicate the fold of the sequence in the corresponding 5-fold cross-validation split.

When predicting the fitness score for a given sequence, **you must use a model trained only on sequences from other folds**.
For example, to predict the fitness score for sequences in fold 0 for the `fold_random_5` split, you must use a model trained
only on the sequences with `fold_random_5` values other than 0.

You must repeat this process for each of the five folds in `fold_random_5` (so that all sequences in `data.csv` 
receive a predicted score). Then repeat the process separately for the other two cross-validation split columns
`fold_modulo_5` and `fold_contiguous_5`. Hence, each sequence should have three predicted fitness scores,
corresponding to the prediction for that sequence under models trained on the three different cross-validation splits.

Overall, your training and inference pseudocode loop should look like this:

```python
import pandas as pd
data = pd.read_csv("data.csv")
wt_sequence = data.iloc[0]["sequence"]

# remove the wild-type sequence from the data
data = data.iloc[1:]

## define your model here ##
model = ...

# initialize a dataframe to store the predictions
predictions = pd.DataFrame(columns=["id", "fold_random_5", "fold_modulo_5", "fold_contiguous_5"], index=data.index)
predictions["id"] = data["id"]

# loop over the different cross-validation splits
fold_columns = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
for column in fold_columns:  # different splits
    for fold in range(5):  # different folds per-split
        fold_mask = data[column] == fold
        train_data = data[~fold_mask]  # train on all folds except the current one
        test_data = data[fold_mask]  # test on the current fold

        # train the model **from scratch** on the training set 
        trained_model = model.fit(train_data["sequence"], train_data["fitness_score"]) 

        # predict the fitness score for the sequences in the current fold 
        predictions.loc[fold_mask, f"fitness_score_{column}"] = trained_model.predict(test_data["sequence"])
```

Hence, the output data frame should contain four columns:
- `id`: The ID of the sequence 
- `fitness_score_fold_random_5`: The predicted fitness score for that sequence predicted by the model trained on the 
  cross-validation split defined by the `fold_random_5` column
- `fitness_score_fold_modulo_5`: The predicted fitness score for that sequence predicted by the model trained on the 
  cross-validation split defined by the `fold_modulo_5` column
- `fitness_score_fold_contiguous_5`: The predicted fitness score for that sequence predicted by the model trained on the 
  cross-validation split defined by the `fold_contiguous_5` column

## Data Format

The `data.csv` file contains the following columns:
- `id`: The index of the sequence
- `sequence`: The amino acid sequence of the variant
- `fitness_score`: The fitness score of the variant
- `fold_random_5`: The fold of the variant (0-4) in the "random" 5-fold cross-validation split
- `fold_modulo_5`: The fold of the variant (0-4) in the "modulo" 5-fold cross-validation split
- `fold_contiguous_5`: The fold of the variant (0-4) in the "contiguous" 5-fold cross-validation split

The first row of the CSV file contains the wild-type sequence in the `sequence` field and missing values for the other columns.

## Files

- `data.csv`: File with sequences and fitness scores
- `sample_submission.csv`: Example submission format with ID column and the three fitness score columns

## Evaluation

Your model will be evaluated on the Spearman correlation between the predicted fitness scores and the true fitness scores for
each of the sequences in `data.csv`.

---
