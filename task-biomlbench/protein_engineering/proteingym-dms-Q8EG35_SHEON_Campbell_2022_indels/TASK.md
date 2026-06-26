---
task_type: biomlbench
name: proteingym-dms-Q8EG35_SHEON_Campbell_2022_indels
description: >
  Predict organismal fitness of unseen indel variants of MtrA protein from Shewanella oneidensis.
  Maximize Spearman correlation using 5-fold CV on fold_random_5 only (indel task, single split).
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **Indels only:** Sequences have insertions and deletions, variable length — models must handle this.
- **Single split:** Only `fold_random_5`, output is `fitness_score` (not three columns).
- **Compute:** See `program_biomlbench.md` for wall-clock limit and GPU/CPU settings.

---

# ProteinGym DMS: Q8EG35_SHEON_Campbell_2022_indels

**Metric:** Spearman correlation — higher is better  
**Difficulty:** Hard  
**Dataset size:** 331 sequences (indel variants)  
**Source:** "Determinants of Multiheme Cytochrome Extracellular Electron Transfer Uncovered by Systematic Peptide Insertion" (DOI: 10.1021/acs.biochem.2c00148)

---

## The Problem

Deep mutational scanning dataset measuring **extracellular electron transfer** (organismal fitness) for **indel** (insertion/deletion) variants of MtrA from *Shewanella oneidensis* (Uniprot: Q8EG35_SHEON). Indels change sequence length making this significantly harder than substitution tasks.

**Key difference from substitution tasks:** Only uses `fold_random_5` (not 3 split strategies). Output is a single `fitness_score` column.

**Input:** Amino acid sequence of the variant (variable length due to indels)  
**Output:** One predicted fitness score (fold_random_5 CV only)  
**Dataset:** 331 sequences

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

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/proteingym-dms/Q8EG35_SHEON_Campbell_2022_indels/prepared/public
cp $PREP/data.csv data/ 2>/dev/null || echo "Run biomlbench data preparation first"

# Or download via biomlbench
$PYTHON -c "
import sys; sys.path.insert(0, '$BIOMLBENCH')
from biomlbench.data_sources.proteingym import ProteinGymDMSDataSource
from pathlib import Path
src = ProteinGymDMSDataSource('Q8EG35_SHEON_Campbell_2022_indels')
raw = Path('data/raw'); raw.mkdir(parents=True, exist_ok=True)
src.download(raw)
"
```

### Data format (`data/data.csv`)
- `id`: sequence index
- `sequence`: amino acid sequence (variable length due to indels)
- `fitness_score`: measured extracellular electron transfer fitness
- `fold_random_5`: fold assignment (0–4) — **only one fold column for indel tasks**

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
2. **Save `submission.csv`** — OOF fitness score predictions for all variants in `data/data.csv` using `fold_random_5` (columns: `id`, `fitness_score`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** Spearman correlation on `fold_random_5` OOF predictions — higher is better

Use `fold_random_5` in `data/data.csv` for 5-fold CV. This IS the test evaluation — no separate held-out test set exists for ProteinGym tasks.

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

data = pd.read_csv("data/data.csv")
data = data[data["id"] != "WT"].copy()  # exclude WT from scoring

all_preds = np.zeros(len(data))

for fold in range(5):
    val_mask = data["fold_random_5"] == fold
    tr, val = data[~val_mask], data[val_mask]
    # ... train on tr, predict on val ...
    all_preds[val_mask.values] = model.predict(val)

spearman = spearmanr(data["fitness_score"], all_preds).correlation
print(f"Spearman: {spearman:.4f}")
```

Report `spearman` to the ClawLab leaderboard after each iteration.

---

## Final Submission

For ProteinGym tasks, the iterative leaderboard score IS the final evaluation — there is no separate held-out test set. The prescribed fold splits define both development and final scoring.

**Do not retrain on all data for the submission.** The submission is the out-of-fold predictions from the 5-fold CV above, which already uses all data exactly once.

```python
# submission.csv = OOF predictions from fold_random_5 5-fold CV
sub = data[["id"]].copy()
sub["fitness_score"] = all_preds
sub.to_csv("submission.csv", index=False)
```

## Output Format

```csv
id,fitness_score
0,0.123
1,-0.045
...
```

---



---

## Background

# Proteingym-DMS dataset: Q8EG35_SHEON_Campbell_2022_indels

## Description

This dataset is part of the ProteinGym DMS benchmark, which contains deep mutational scanning datasets that measure
protein fitness (in different contexts) for sequence variants of a wide range of proteins. This dataset contains
indel variants for the protein MtrA from the organism Shewanella oneidensis. This protein has Uniprot ID: Q8EG35_SHEON. 

The DMS selection assay was described as follows: 

    Extracellular electron transfer

It was categorised as measuring the following (general) fitness attribute: OrganismalFitness. Higher scores indicate better fitness.

The source publication for this dataset is titled: 

"Determinants of Multiheme Cytochrome Extracellular Electron Transfer Uncovered by Systematic Peptide Insertion"

and can be accessed at the following DOI: 10.1021/acs.biochem.2c00148.

## Objective

The objective of this benchmark is to train a model that can predict the fitness of unseen single-substitution sequence variants of MtrA.
To train your model, you will use 5-fold cross-validation on the sequences and fitness scores defined in the `data.csv` file. 

You will use the `fold_random_5` column to split the data into training and test sets. This column contains integer values from 0 to 4, 
which indicate the fold of the sequence in the corresponding 5-fold cross-validation split.

When predicting the fitness score for a given sequence, **you must use a model trained only on sequences from other folds**.
For example, to predict the fitness score for sequences with `fold_random_5 == 0`, you must use a model trained
only on the sequences with `fold_random_5` values other than 0.

You must repeat this process for each of the five folds in `fold_random_5` (so that all sequences in `data.csv` 
receive a predicted score).

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
predictions = pd.DataFrame(columns=["id", "fitness_score"], index=data.index)
predictions["id"] = data["id"]

# loop over the different folds
for fold in range(5): 
    fold_mask = data["fold_random_5"] == fold
    train_data = data[~fold_mask]  # train on all folds except the current one
    test_data = data[fold_mask]  # test on the current fold

    # train the model **from scratch** on the training set 
    trained_model = model.fit(train_data["sequence"], train_data["fitness_score"]) 

    # predict the fitness score for the sequences in the current fold 
    predictions.loc[fold_mask, "fitness_score"] = trained_model.predict(test_data["sequence"])
```

Hence, the output data frame should contain four columns:
- `id`: The ID of the sequence 
- `fitness_score`: The predicted fitness score for that sequence predicted by the model trained on the 
    cross-validation split defined by the `fold_random_5` column.

## Data Format

The `data.csv` file contains the following columns:
- `id`: The index of the sequence
- `sequence`: The amino acid sequence of the variant
- `fitness_score`: The fitness score of the variant
- `fold_random_5`: The fold of the variant (0-4) in the 5-fold cross-validation split

The first row of the CSV file contains the wild-type sequence in the `sequence` field and missing values for the other columns.

## Files

- `data.csv`: File with sequences and fitness scores
- `sample_submission.csv`: Example submission format with ID column and fitness score column

## Evaluation

Your model will be evaluated using the Spearman correlation between the predicted fitness scores and the true fitness scores for
each of the sequences in `data.csv`.

---
