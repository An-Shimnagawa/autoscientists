---
task_type: biomlbench
name: kaggle-osic-pulmonary-fibrosis-progression
description: >
  Predict lung function decline (FVC) and confidence from baseline CT scans for IPF patients.
  Maximize Modified Laplace Log Likelihood (higher = better).
  Prizes: 1st $30K, 2nd $15K, 3rd $10K.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **CT scans required:** Your model **must use the baseline CT scan images** (`data/train/{Patient}/`) as input features, not just the tabular metadata. Solutions that ignore the CT scans will not be accepted.
- **Uncertainty required:** Must predict both FVC and Confidence (sigma). Overconfident predictions (small sigma) are heavily penalized.
- **Sigma clipped at 70:** Never predict sigma < 70.
- **Compute:** NVIDIA A100-SXM4 GPU (40 GB VRAM, 503 GB RAM), 16-hour wall-clock limit. **`submission.csv` must be saved before the wall-clock limit expires.**

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out ground-truth FVC values for the **fixed test
> patients** (the 18 patients in `data/test.csv`). It is used **only** by the external grader to
> score your final `submission.csv`. **Never read, open, or use this file in `train.py` or any
> intermediate evaluation.** Doing so would invalidate your score.
>
> **How to evaluate during development:** Use the `cv_fold` column in `data/train.csv` for
> patient-level 5-fold cross-validation. Each fold contains ~31–32 patients; fold assignments are
> by patient (all rows for a given patient go to the same fold). For each fold `k`, train on rows
> where `cv_fold != k` and validate on rows where `cv_fold == k`. Report the **mean Laplace LL
> across all 5 folds** to the ClawLab leaderboard.
> The grader scores predictions for the **18 patients in `data/test.csv`** — your `submission.csv`
> must contain `Patient_Week` entries for all weeks (range -12 to 133) for those 18 patients.
> See `data/sample_submission.csv` for the exact 1,908 `Patient_Week` rows expected.

---

# Kaggle: OSIC Pulmonary Fibrosis Progression

**Metric:** Modified Laplace Log Likelihood — higher is better  
**Task type:** Regression with uncertainty  
**Awards medals:** Yes  
**Prizes:** 1st $30K, 2nd $15K, 3rd $10K  
**Competition:** https://kaggle.com/competitions/osic-pulmonary-fibrosis-progression

---

## The Problem

Idiopathic Pulmonary Fibrosis (IPF) causes progressive lung scarring with no known cure. Outcomes range widely — some patients are stable for years, others decline rapidly. Predict each patient's FVC (forced vital capacity, lung volume in mL) at future time points, plus a confidence measure.

**Input:** Baseline CT scan (DICOM) + clinical metadata (age, sex, smoking status, baseline FVC/percent)  
**Output:** FVC prediction + Confidence (standard deviation) for every possible week per patient  
**Evaluation:** Modified Laplace Log Likelihood (accounts for both accuracy and calibrated uncertainty)

### Metric formula

```
sigma_clipped = max(sigma, 70)
delta = min(|FVC_true - FVC_predicted|, 1000)
metric = -sqrt(2) * delta / sigma_clipped - ln(sqrt(2) * sigma_clipped)
```

Higher is better. Averaged across all test `Patient_Week` pairs.

---

## Data

### Location

Data is pre-downloaded. `data/` is a symlink to:
```
/n/netscratch/mzitnik_lab/Lab/afang/kaggle/osic-pulmonary-fibrosis-progression/
├── osic-pulmonary-fibrosis-progression.zip  # original download
├── train.csv              # Clinical metadata + full FVC history for 158 training patients
├── test.csv               # Baseline measurement for the 18 biomlbench test patients
├── train/                 # Baseline CT scans (DICOM) for train patients
├── test/                  # Baseline CT scans (DICOM) for test patients
└── sample_submission.csv  # 1,908 Patient_Week rows for the 18 test patients (use as template)
```

**Note:** Each patient folder in `train/` and `test/` contains the baseline CT scan which can have different numbers of slices (the number of .dcm files).

### Loading DICOM slices with pydicom

`pydicom` is installed in the project venv. Each patient folder contains numbered `.dcm` files (e.g. `10.dcm`, `11.dcm`, …). Sort by integer stem to get slices in anatomical order.

```python
import pydicom
import numpy as np
from pathlib import Path

TRAIN_SCANS = Path("data/train")  # or the absolute netscratch path

def load_ct_slices(patient_id: str) -> list[np.ndarray]:
    """Return CT slices as a list of float32 HU arrays, each (512, 512)."""
    patient_dir = TRAIN_SCANS / patient_id
    dcm_files = sorted(patient_dir.glob("*.dcm"), key=lambda p: int(p.stem))
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        hu = ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        slices.append(hu.astype(np.float32))
    return slices  # length varies per patient

# Example
slices = load_ct_slices("ID00007637202177411956430")
print(len(slices), slices[0].shape)  # e.g. 30, (512, 512) — shape and length vary per patient
```

**Key facts from the training data:**
- Slice shape varies: observed shapes include 512×512, 632×632, 768×768, and non-square sizes up to 1302×888 — always read from the DICOM, never assume
- Raw pixel dtype is `int16` or `uint16` depending on the patient
- `RescaleSlope=1` for all patients; `RescaleIntercept` is `-1024`, `-1000`, or `0` depending on the patient — always read from the DICOM
- Number of slices varies widely: min 12, max 1018, median ~98
- HU values after conversion range from roughly -31000 to +32000 (including scanner artefacts outside the body)
- One slice is unreadable: `data/train/ID00052637202186188008618/4.dcm` (JPEG decode error); that patient has 311 slices total so skipping it is fine — handle decode errors gracefully
- CT scans for the 18 **test** patients are in `data/test/{Patient}/` (separate from the training scans in `data/train/`)

Your `submission.csv` must contain predictions for all 1,908 `Patient_Week` rows in `data/sample_submission.csv`.
Do not submit to kaggle.com.

### How to prepare data
```bash
# Data is already available via the data/ symlink.
# To unzip if needed:
cd /n/netscratch/mzitnik_lab/Lab/afang/kaggle/osic-pulmonary-fibrosis-progression/
unzip -q osic-pulmonary-fibrosis-progression.zip

# Or run biomlbench prepare (requires Kaggle API credentials)
$PYTHON -c "
import sys; sys.path.insert(0, '$BIOMLBENCH')
from biomlbench.tasks.kaggle.osic_pulmonary_fibrosis_progression.prepare import prepare
from pathlib import Path
prepare(Path('data/raw'), Path('data'), Path('data/private'))
"
```

### Data columns

**`train.csv` and `test.csv`:**
- `Patient`: unique patient ID (also folder name for DICOM scans)
- `Weeks`: relative weeks from baseline CT (may be negative)
- `FVC`: forced vital capacity in mL
- `Percent`: FVC as % of expected for similar demographics
- `Age`, `Sex`, `SmokingStatus`

**`data/train.csv`:** 158 patients, complete FVC history. Includes `cv_fold` column (0–4) for patient-level 5-fold CV (generated via `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)` applied to the sorted list of unique patient IDs; all rows for a patient share the same fold). **`data/test.csv`:** 18 test patients, baseline measurement only (1 row per patient).

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
1. **Evaluate locally** — train on `data/train.csv` patients, estimate val score on a local hold-out, print the val metric.
2. **Save `submission.csv`** — FVC and Confidence predictions for all 1,908 Patient_Week rows in `data/sample_submission.csv` (columns: `Patient_Week`, `FVC`, `Confidence`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


**IMPORTANT:** The biomlbench grader scores `submission.csv` against the 18 patients in `data/test.csv`.
Your `submission.csv` must contain the **exact 1,908 `Patient_Week` rows from `data/sample_submission.csv`**
(all weeks -12 to 133 for each of the 18 test patients). The grader then scores only the 3 latest
weeks per patient (54 rows total).

**CV must mirror test conditions — two requirements:**

1. **Input: single baseline row only.** The test patients in `data/test.csv` each have **exactly one row**: their single baseline measurement (one week, one FVC value, plus static demographics). Your model receives no follow-up visits for test patients. Your CV must replicate this: when evaluating fold `k`, treat each val patient as if only their **single baseline row** is available as input (the row with the smallest `Weeks` value for that patient). Do **not** use any of the val patient's other observed visits as model input or context during evaluation.

2. **Scoring: last 3 visits only.** The grader scores only the **3 latest visits per patient** (54 rows total across 18 patients). Your CV score must do the same — score only each val patient's last 3 visits (by `Weeks`), not all their visits. Scoring all visits inflates the CV estimate and produces a misleading estimate of test performance.

```python
import pandas as pd, numpy as np

# Test patients and their baseline info
test = pd.read_csv("data/test.csv")  # 18 rows, 1 per test patient
sample = pd.read_csv("data/sample_submission.csv")  # 1,908 Patient_Week rows — use as submission template

# Training data
train = pd.read_csv("data/train.csv")  # 158 patients (do NOT include test patients here)

# For local val estimation, use 5-fold CV via the cv_fold column.
# IMPORTANT: val patients must be treated identically to test patients —
# use only each val patient's single baseline row as model input,
# and score only their last 3 visits (matching the grader).
fold_scores = []
for k in range(5):
    tr  = train[train["cv_fold"] != k]
    val = train[train["cv_fold"] == k]

    # Extract one baseline row per val patient (earliest visit = minimum Weeks).
    # This is the ONLY information your model may use about val patients as input.
    val_baseline = val.sort_values("Weeks").groupby("Patient").first().reset_index()

    # Score only the last 3 visits per val patient — matching the grader.
    val_last3 = val.sort_values("Weeks").groupby("Patient").tail(3)

    # ... train on tr patients using their full FVC history ...
    # ... predict FVC+Confidence for every val_last3 row using ONLY val_baseline as input ...
    # score = laplace_ll(val_last3["FVC"].values, val_preds_fvc, val_preds_conf)
    # fold_scores.append(score)
mean_score = np.mean(fold_scores)
std_score  = np.std(fold_scores)
print(f"Mean CV Laplace LL: {mean_score:.4f} ± {std_score:.4f}")
# submission.csv must cover test patients in data/test.csv, not val patients
```

The test set has **18 patients**. Train on `data/train.csv` (158 patients), predict all weeks for the
18 test patients in `data/test.csv` (their scans are in `data/train/{Patient}/`), save as `submission.csv`.

---

## Iterative Leaderboard (ClawLab API)

**Metric:** Modified Laplace Log Likelihood — higher is better (less negative)

Use the `cv_fold` column in `data/train.csv` for patient-level 5-fold cross-validation:

```python
import pandas as pd, numpy as np

def laplace_ll(fvc_true, fvc_pred, sigma):
    sigma_c = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000)
    return (-np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)).mean()

train = pd.read_csv("data/train.csv")  # 158 training patients, cv_fold column 0–4
fold_scores = []
for k in range(5):
    tr  = train[train["cv_fold"] != k]
    val = train[train["cv_fold"] == k]

    # INPUT: use only each val patient's single baseline row (earliest visit) —
    # matching what is available for test patients at inference time.
    val_baseline = val.sort_values("Weeks").groupby("Patient").first().reset_index()

    # SCORING: score only the last 3 visits per val patient — matching the grader.
    # Do NOT score all visits: that inflates CV and misrepresents test performance.
    val_last3 = val.sort_values("Weeks").groupby("Patient").tail(3)

    # Train on tr (full history). Predict FVC+Confidence for val_last3 rows
    # using ONLY val_baseline (one row per val patient) as input.
    # val_preds_fvc and val_preds_conf must align with val_last3 row order.
    score = laplace_ll(val_last3["FVC"].values, val_preds_fvc, val_preds_conf)
    fold_scores.append(score)
mean_score = np.mean(fold_scores)
std_score  = np.std(fold_scores)
print(f"Mean CV Laplace LL: {mean_score:.4f} ± {std_score:.4f} (per fold: {[f'{s:.4f}' for s in fold_scores]})")
```

Report the following to the ClawLab leaderboard after each iteration:
- **Primary metric:** mean CV Laplace LL (used to rank approaches and pick the champion)
- **Additional metrics:** per-fold Laplace LL for each of the 5 folds, and the standard deviation across folds

Only treat an improvement as real if it is consistent across most folds — a gain on a single fold
while others regress is noise, not signal. High standard deviation (relative to the mean) is a
sign of instability and should make you prefer a simpler, more consistent approach.

---

## Final Submission

Train on `data/train.csv` patients, predict ALL weeks for the 18 test patients in `data/test.csv`,
save as `submission.csv`. Use `data/sample_submission.csv` as the template (exactly 1,908 rows).

```python
import pandas as pd

test = pd.read_csv("data/test.csv")  # 18 test patients
sample = pd.read_csv("data/sample_submission.csv")  # 1,908 Patient_Week rows — use as template
test_patients = test["Patient"].tolist()

# ... train model on data/train.csv patients ...
# Generate predictions for ALL weeks for each test patient (weeks -12 to 133)
# The grader only scores the 3 latest weeks per patient, but you must supply all weeks
rows = []
for patient in test_patients:
    for week in range(-12, 134):
        fvc_pred, conf_pred = model.predict(patient, week)
        rows.append({"Patient_Week": f"{patient}_{week}", "FVC": fvc_pred, "Confidence": conf_pred})
submission = pd.DataFrame(rows)
# submission must have the same Patient_Week values as sample_submission.csv
submission.to_csv("submission.csv", index=False)
```

CT scans for the 18 test patients are in `data/train/{Patient}/` — same directory as training scans.

---

## Output Format

`submission.csv` — predict for ALL possible weeks per patient:
```csv
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2800,200
ID00002637202176704235138_2,2790,205
...
```

---



---

## Background

## **Description**

Imagine one day, your breathing became consistently labored and shallow. Months later you were finally diagnosed with pulmonary fibrosis, a disorder with no known cause and no known cure, created by scarring of the lungs. If that happened to you, you would want to know your prognosis. That’s where a troubling disease becomes frightening for the patient: outcomes can range from long-term stability to rapid deterioration, but doctors aren’t easily able to tell where an individual may fall on that spectrum. Your help, and data science, may be able to aid in this prediction, which would dramatically help both patients and clinicians.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/667f1cbf-826f-4641-a321-96054292638d/d4ebeea7-3764-4f7b-b4f7-35b2d0ecbeea/Untitled.png)

Current methods make fibrotic lung diseases difficult to treat, even with access to a chest CT scan. In addition, the wide range of varied prognoses create issues organizing clinical trials. Finally, patients suffer extreme anxiety—in addition to fibrosis-related symptoms—from the disease’s opaque path of progression.

[Open Source Imaging Consortium (OSIC)](https://www.osicild.org/) is a not-for-profit, co-operative effort between academia, industry and philanthropy. The group enables rapid advances in the fight against Idiopathic Pulmonary Fibrosis (IPF), fibrosing interstitial lung diseases (ILDs), and other respiratory diseases, including emphysematous conditions. Its mission is to bring together radiologists, clinicians and computational scientists from around the world to improve imaging-based treatments.

In this competition, you’ll predict a patient’s severity of decline in lung function based on a CT scan of their lungs. You’ll determine lung function based on output from a spirometer, which measures the volume of air inhaled and exhaled. The challenge is to use machine learning techniques to make a prediction with the image, metadata, and baseline FVC as input.

If successful, patients and their families would better understand their prognosis when they are first diagnosed with this incurable lung disease. Improved severity detection would also positively impact treatment trial design and accelerate the clinical development of novel treatments.

**This is a Code Competition. Refer to [Code Requirements](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression#Code-Requirements) for details.**

## **Evaluation**

This competition is evaluated on a modified version of the Laplace Log Likelihood. In medical applications, it is useful to evaluate a model's confidence in its decisions. Accordingly, the metric is designed to reflect both the accuracy and certainty of each prediction.

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). Note that metric values will be negative and higher is better.

## **Submission File**

For each `Patient_Week`, you must predict the `FVC` and a confidence. To avoid potential leakage in the timing of follow up visits, you are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## **Timeline**

- **September 29, 2020** - Entry deadline. You must accept the competition rules before this date in order to compete.
- **September 29, 2020** - Team Merger deadline. This is the last day participants may join or merge teams.
- **October 6, 2020** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## **Prizes**

- 1st Place - $30,000
- 2nd Place - $15,000
- 3rd Place - $10,000

## **Code Requirements**

### **This is a Code Competition**

Submissions to this competition must be made through Notebooks. In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 4 hours run-time
- [TPUs](https://www.kaggle.com/docs/tpu) will not be available for making submissions to this competition. You are still welcome to use them for training models.
- No internet access enabled
- External data, freely & publicly available, is allowed. This includes pre-trained models.
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit.

## **Citation**

Ahmed Shahin, Carmela Wegworth, David, Elizabeth Estes, Julia Elliott, Justin Zita, SimonWalsh, Slepetys, Will Cukierski. (2020). OSIC Pulmonary Fibrosis Progression. Kaggle. https://kaggle.com/competitions/osic-pulmonary-fibrosis-progression

# Data

## **Dataset Description**

The aim of this competition is to predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

There are around 200 cases in the public & private test sets, combined. This is split roughly 15-85 between public-private.

Since this is real medical data, you will notice the relative timing of `FVC` measurements varies widely. The timing of the initial measurement relative to the CT scan and the duration to the forecasted time points may be different for each patient. This is considered part of the challenge of the competition. To avoid potential leakage in the timing of follow up visits, you are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

### **Files**

This is a synchronous rerun code competition. The provided test set is a small representative set of files (copied from the training set) to demonstrate the format of the private test set. When you submit your notebook, Kaggle will rerun your code on the test set, which contains unseen images.

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

### **Columns**

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

---
