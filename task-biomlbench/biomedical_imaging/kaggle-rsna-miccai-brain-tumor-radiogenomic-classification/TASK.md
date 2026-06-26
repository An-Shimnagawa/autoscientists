---
task_type: biomlbench
name: kaggle-rsna-miccai-brain-tumor-radiogenomic-classification
description: >
  Predict MGMT promoter methylation status from multi-parametric brain MRI scans (4 sequences).
  Maximize ROC-AUC. ~609 training glioblastoma patients. Prizes: 1st $6K.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **DICOM format:** Use `pydicom` to load MRI slices.
- **4 MRI sequences:** FLAIR, T1w, T1wCE (T1 with contrast), T2w — all available for most patients.
- **Noisy labels:** MGMT prediction from MRI is inherently uncertain; top scores are ~0.64 ROC-AUC.
- **Compute:** NVIDIA A100-SXM4 GPU (40 GB VRAM, 503 GB RAM), 16-hour wall-clock limit. **`submission.csv` must be saved before the wall-clock limit expires.**
- **PyTorch/CUDA note:** The venv has torch 2.11.0+cu130 but system CUDA is 12.9. Install a compatible version if needed: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out MGMT labels for the **59 test patients** in
> `data/test/`. It is used **only** by the external grader to score your final `submission.csv`.
> **Never read, open, or use this file in `train.py` or any intermediate evaluation.**
> Doing so would invalidate your score.
>
> **How to evaluate during development:** Use the `cv_fold` column in `data/train_labels.csv` for
> patient-level 5-fold cross-validation. Each fold contains ~105–106 patients. For each fold `k`,
> train on rows where `cv_fold != k` and validate on rows where `cv_fold == k`. Report the **mean
> ROC-AUC across all 5 folds** to the ClawLab leaderboard.
> The grader scores predictions for the **59 patients in `data/test/`** — your `submission.csv`
> must contain exactly those BraTS21IDs. Use `data/sample_submission.csv` to get the correct list.

---

# Kaggle: RSNA-MICCAI Brain Tumor Radiogenomic Classification

**Metric:** ROC-AUC — higher is better  
**Task type:** Binary image classification (3D MRI)  
**Awards medals:** Yes  
**Prizes:** 1st $6K, 2nd $5K, 3rd $4K, 4th–8th $3K each  
**Competition:** https://kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification

---

## The Problem

Glioblastoma is the most common adult primary malignant brain tumor, with poor prognosis. MGMT (O6-methylguanine-DNA methyltransferase) promoter methylation is a key biomarker — methylated tumors respond better to chemotherapy (temozolomide). Currently requires brain biopsy; predicting from MRI would enable non-invasive assessment.

**Input:** Multi-parametric brain MRI — 4 sequences: FLAIR, T1w, T1Gd (contrast-enhanced), T2w  
**Output:** Probability of MGMT methylation (1 = methylated, better prognosis)  
**Evaluation:** ROC-AUC — higher is better

---

## Data

### Location

Data is pre-downloaded. `data/` is a symlink to:
```
/n/netscratch/mzitnik_lab/Lab/afang/kaggle/rsna-miccai-brain-tumor-radiogenomic-classification/
├── rsna-miccai-brain-tumor-radiogenomic-classification.zip  # original download
├── train/          # DICOM scan directories for 526 training patients (4 sequences each)
├── test/           # DICOM scan directories for 59 biomlbench test patients (predict these)
├── train_labels.csv # BraTS21ID, MGMT_value (0/1), cv_fold (0–4) — 526 training patients only (cv_fold generated via KFold(n_splits=5, shuffle=True, random_state=42) on sorted BraTS21IDs)
└── sample_submission.csv  # 59 rows with correct test BraTS21IDs (use as submission template)
```

**Note:** The biomlbench test set is `data/test/` — 59 patients whose scans are in `data/test/{BraTS21ID}/`.
`data/train_labels.csv` has labels for the 526 training patients in `data/train/` only (not the test patients).
Your `submission.csv` must contain predictions for all 59 BraTS21IDs in `data/sample_submission.csv`.
Do not submit to kaggle.com.

### How to prepare data
```bash
# Data is already available via the data/ symlink.
# To unzip if needed:
cd /n/netscratch/mzitnik_lab/Lab/afang/kaggle/rsna-miccai-brain-tumor-radiogenomic-classification/
unzip -q rsna-miccai-brain-tumor-radiogenomic-classification.zip
```

### Data format

```python
import pydicom, os
import pandas as pd

# Training data — 526 patients with known labels
labels = pd.read_csv("data/train_labels.csv")
# Columns: BraTS21ID (zero-padded 5-digit string), MGMT_value (0=unmethylated, 1=methylated)

# Load a training patient's scans (in data/train/)
patient_id = "00000"
for seq in ["FLAIR", "T1w", "T1wCE", "T2w"]:
    seq_dir = f"data/train/{patient_id}/{seq}/"
    # IMPORTANT: files are named Image-1.dcm, Image-2.dcm, ..., Image-N.dcm
    # Use numeric sort — alphabetical sort is wrong
    dicoms = sorted(os.listdir(seq_dir), key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))
    # Load: pydicom.dcmread(f"{seq_dir}/{dicoms[0]}").pixel_array

# Test data — 59 patients to predict (scans in data/test/, no labels)
sample = pd.read_csv("data/sample_submission.csv", dtype={"BraTS21ID": str})
# sample["BraTS21ID"] gives the 59 test patient IDs
for seq in ["FLAIR", "T1w", "T1wCE", "T2w"]:
    seq_dir = f"data/test/{sample['BraTS21ID'][0]}/{seq}/"
    # same DICOM format and naming as train/ — use the same numeric sort
```

### Key facts from the training data

**Slice counts per sequence** (585 train patients):

| Sequence | Min | Median | Max |
|----------|-----|--------|-----|
| FLAIR    | 15  | 60     | 514 |
| T1w      | 19  | 180    | 400 |
| T1wCE    | 19  | 192    | 400 |
| T2w      | 19  | 64     | 472 |

Note: slice counts differ across sequences within the same patient (e.g., FLAIR median 60 vs T1wCE median 192). All 585 patients have all 4 sequences.

**Image shape:** Not always 512×512 — observed shapes include `(256, 192)`, `(256, 256)`, `(320, 320)`, `(384, 336)`, `(512, 384)`, `(512, 512)` and others. Always read from the DICOM and resize explicitly; never assume a fixed shape.

**DICOM naming:** Files are named `Image-1.dcm`, `Image-2.dcm`, ..., `Image-N.dcm`. Sort by the integer in the filename.

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
1. **Evaluate locally** — train on `data/train/` scans, estimate val score on a local hold-out, print the val metric.
2. **Save `submission.csv`** — MGMT methylation predictions for all 59 test patients in `data/test/` (columns: `BraTS21ID`, `MGMT_value`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


**IMPORTANT:** The biomlbench grader scores `submission.csv` against the 59 patients in `data/test/`.
Your `submission.csv` must contain the **exact BraTS21IDs from `data/sample_submission.csv`** (59 rows).
Use `data/train/` scans + `data/train_labels.csv` for training and local validation only.

```python
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

# Training: 526 labeled patients — scans in data/train/
labels = pd.read_csv("data/train_labels.csv", dtype={"BraTS21ID": str})
# cv_fold column (0–4) enables patient-level 5-fold CV:
fold_scores = []
for k in range(5):
    tr  = labels[labels["cv_fold"] != k]
    val = labels[labels["cv_fold"] == k]
    # ... train on tr scans (data/train/{BraTS21ID}/), eval on val scans ...
    # score = roc_auc_score(val["MGMT_value"].values, val_preds)
    # fold_scores.append(score)
mean_score = np.mean(fold_scores)
std_score  = np.std(fold_scores)
print(f"Mean CV ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")

# Test: 59 patients — scans in data/test/
sample = pd.read_csv("data/sample_submission.csv", dtype={"BraTS21ID": str})
test_ids = sample["BraTS21ID"].tolist()
# ... predict probabilities for test scans (data/test/{BraTS21ID}/) ...
```

The test set has **59 patients** with scans in `data/test/{BraTS21ID}/`. Train on `data/train/` scans,
predict the test scans, save as `submission.csv`.

---

## Iterative Leaderboard (ClawLab API)

**Metric:** ROC-AUC — higher is better

Use the `cv_fold` column in `data/train_labels.csv` for patient-level 5-fold cross-validation:

```python
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

labels = pd.read_csv("data/train_labels.csv", dtype={"BraTS21ID": str})
# cv_fold column (0–4); all scans are in data/train/{BraTS21ID}/
fold_scores = []
for k in range(5):
    tr  = labels[labels["cv_fold"] != k]
    val = labels[labels["cv_fold"] == k]
    # ... train on tr scans (data/train/{BraTS21ID}/), predict val scans ...
    score = roc_auc_score(val["MGMT_value"].values, val_preds)
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

Train on `data/train/` scans (526 patients), predict the 59 test patients in `data/test/`,
save as `submission.csv`.

```python
import pandas as pd

sample = pd.read_csv("data/sample_submission.csv", dtype={"BraTS21ID": str})
test_ids = sample["BraTS21ID"].tolist()  # 59 five-digit zero-padded IDs

# ... train model on data/train/ scans with labels from data/train_labels.csv ...
# ... predict data/test/{BraTS21ID}/ scans ...

# BraTS21ID must be integer in the submission
submission = pd.DataFrame({"BraTS21ID": [int(i) for i in test_ids], "MGMT_value": test_preds})
submission.to_csv("submission.csv", index=False)
```

**Medal thresholds (1555 teams):** Gold ≈ top 156 (10%), Silver ≈ top 311 (20%), Bronze ≈ top 622 (40%).

Note: This is a challenging task — the correlation between imaging and MGMT status is inherently noisy.

---

## Output Format

```csv
BraTS21ID,MGMT_value
00001,0.823
00013,0.234
...
```

---



---

## Background

# RSNA-MICCAI Brain Tumor Radiogenomic Classification

## Description

A malignant tumor in the brain is a life-threatening condition. Known as glioblastoma, it's both the most common form of brain cancer in adults and the one with the worst prognosis, with median survival being less than a year. The presence of a specific genetic sequence in the tumor known as MGMT promoter methylation has been shown to be a favorable prognostic factor and a strong predictor of responsiveness to chemotherapy.

![](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2021/image2.png)

Currently, genetic analysis of cancer requires surgery to extract a tissue sample. Then it can take several weeks to determine the genetic characterization of the tumor. Depending upon the results and type of initial therapy chosen, a subsequent surgery may be necessary. If an accurate method to predict the genetics of the cancer through imaging (i.e., radiogenomics) alone could be developed, this would potentially minimize the number of surgeries and refine the type of therapy required.

The Radiological Society of North America (RSNA) has teamed up with the Medical Image Computing and Computer Assisted Intervention Society (the MICCAI Society) to improve diagnosis and treatment planning for patients with glioblastoma. In this competition you will predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to train and test your model to detect for the presence of MGMT promoter methylation.

If successful, you'll help brain cancer patients receive less invasive diagnoses and treatments. The introduction of new and customized treatment strategies before surgery has the potential to improve the management, survival, and prospects of patients with brain cancer.

### Acknowledgments

**The Radiological Society of North America (RSNA®)** is a non-profit organization that represents 31 radiologic subspecialties from 145 countries around the world. RSNA promotes excellence in patient care and health care delivery through education, research and technological innovation.

RSNA provides high-quality educational resources, publishes five top peer-reviewed journals, hosts the world’s largest radiology conference and is dedicated to building the future of the profession through the RSNA Research & Education (R&E) Foundation, which has funded $66 million in grants since its inception. RSNA also supports and facilitates artificial intelligence (AI) research in medical imaging by sponsoring an ongoing series of AI challenge competitions.

**The Medical Image Computing and Computer Assisted Intervention Society (the MICCAI Society)** is dedicated to the promotion, preservation and facilitation of research, education and practice in the field of medical image computing and computer assisted medical interventions including biomedical imaging and medical robotics. The Society achieves this aim through the organization and operation of annual high quality international conferences, workshops, tutorials and publications that promote and foster the exchange and dissemination of advanced knowledge, expertise and experience in the field produced by leading institutions and outstanding scientists, physicians and educators around the world.

[A full set of acknowledgments can be found on this page](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/overview/acknowledgments).

![](https://storage.googleapis.com/kaggle-media/competitions/RSNA-2021/sponsors.png)

## Evaluation

Submissions are evaluated on the [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

### Submission File

For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Timeline

- **July 13, 2021** - Start Date.
- **October 8, 2021** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **October 8, 2021** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **October 15, 2021** - Final Submission Deadline.
- **October 25, 2021** - Winners’ Requirements Deadline. This is the deadline for winners to submit to the host/Kaggle their training code, video, method description.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

- 1st Place - $6,000
- 2nd Place - $5,000
- 3rd Place - $4,000
- 4th - 8th Places - $3,000 each

Because this competition is being hosted in coordination with the Radiological Society of North America (RSNA®) Annual Meeting, winners will be invited and strongly encouraged to attend the conference with waived fees, contingent on review of solution and fulfillment of winners' obligations.

Note that, per the [competition rules](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/rules), in addition to the standard Kaggle Winners' Obligations (open-source licensing requirements, [solution packaging/delivery](https://www.kaggle.com/WinningModelDocumentationGuidelines), presentation to host), the host team also asks that you:

(i) create a short video presenting your approach and solution, and

(ii) publish a link to your open sourced code on the competition forum

## Code Requirements

![](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Acknowledgments

The dataset for this challenge has been collected from institutions around the world as part of a decade-long project to advance the use of AI in brain tumor diagnosis and treatment, the **Brain Tumor Segmentation (BraTS) challenge**. Running in parallel with this challenge, a challenge addressing segmentation represents the culmination of this effort.

- A comprehensive description of the both tasks of the RSNA-MICCAI Brain Tumor challenge can be found at: https://www.med.upenn.edu/cbica/brats2021/
- Participants interested in the segmentation task of this competition can find more info at: https://www.synapse.org/brats2021

### Challenge Organizing Team

*(in alphabetical order)*

- Spyridon Bakas, PhD - University of Pennsylvania, Philadelphia, PA, USA
- Ujjwal Baid, PhD - University of Pennsylvania, Philadelphia, PA, USA
- Evan Calabrese, MD, PhD - University of California San Francisco, CA, USA
- Christopher Carr - Radiological Society of North America (RSNA), Oak Brook, IL, USA
- Errol Colak, MD - Unity Health Toronto, Canada
- Keyvan Farahani, PhD - National Cancer Institute (NCI), National Institutes of Health (NIH), Bethesda, MD, USA
- Adam E. Flanders, MD - Thomas Jefferson University Hospital, Philadelphia, PA, USA
- Jayashree Kalpathy-Cramer, PhD - Massachusetts General Hospital, Boston, MA, USA
- Felipe C Kitamura, MD, MSc, PhD - Diagnósticos da América SA (Dasa) and Universidade Federal de São Paulo, Brazil
- Bjoern Menze, PhD - University of Zurich, Switzerland
- John Mongan, MD, PhD - University of California - San Francisco, CA, USA
- Luciano Prevedello, MD, MPH - The Ohio State University, Columbus, OH, USA
- Jeffrey Rudie, MD, PhD - University of California - San Francisco, CA, USA
- Russell Taki Shinohara, PhD - University of Pennsylvania, Philadelphia, PA, USA

### Data Contributors

*(in order of decreasing data contributions)*

- Christos Davatzikos, PhD, & Spyridon Bakas, PhD, & Chiharu Sako, PhD - University of Pennsylvania, Philadelphia, PA, USA
- John Mongan, MD, PhD, & Evan Calabrese, MD, PhD, & Jeff Rudie, MD, PhD, & Christopher Hess, MD, PhD, & Soonmee Cha, MD, & Javier Villanueva-Meyer, MD - University of California San Francisco, CA, USA
- John B. Freymann & Justin S. Kirby - on behalf of The Cancer Imaging Archive (TCIA), Cancer Imaging Program, NCI, NIH, USA
- Benedikt Wiestler, MD, & Bjoern Menze, PhD - Technical University of Munich, Germany
- Bjoern Menze, PhD - University of Zurich, Switzerland
- Errol Colak, MD & Priscila Crivellaro, MD - University of Toronto, Toronto, ON, Canada
- Rivka R. Colen, MD, & Aikaterini Kotrotsou, PhD - MD Anderson Cancer Center, TX, USA
- Daniel Marcus, PhD, & Mikhail Milchenko, PhD, & Arash Nazeri, MD - Washington University School of Medicine in St. Louis, MO, USA
- Hassan Fathallah-Shaykh, MD, PhD - University of Alabama at Birmingham, AL, USA
- Roland Wiest, MD - University of Bern, Switzerland
- Andras Jakab, MD, PhD - University of Debrecen, Hungary
- Marc-Andre Weber, MD - Heidelberg University, Germany
- Abhishek Mahajan, MD, & Ujjwal Baid, PhD - Tata Memorial Centre, Mumbai, India, & SGGS Institute of Engineering and Technology, Nanded, India

### Sponsors

Prize money for challenge winners is provided by:

- Intel Corporation
- Radiological Society of North American (RSNA)
- NeoSoma, Inc.

## Citation

Adam Flanders, Chris Carr, Evan Calabrese, FelipeKitamura, MD, PhD, inversion, JeffRudie, John Mongan, Julia Elliott, Luciano Prevedello, Michelle Riopel, sprint, Spyridon Bakas, Ujjwal. (2021). RSNA-MICCAI Brain Tumor Radiogenomic Classification. Kaggle. https://kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification

# Data

## Dataset Description

The competition data is defined by three cohorts: Training, Validation (Public), and Testing (Private). The “Training” and the “Validation” cohorts are provided to the participants, whereas the “Testing” cohort is kept hidden at all times, during and after the competition.

These 3 cohorts are structured as follows: Each independent case has a dedicated folder identified by a five-digit number. Within each of these “case” folders, there are four sub-folders, each of them corresponding to each of the structural multi-parametric MRI (mpMRI) scans, in DICOM format. The exact mpMRI scans included are:

- Fluid Attenuated Inversion Recovery (FLAIR)
- T1-weighted pre-contrast (T1w)
- T1-weighted post-contrast (T1Gd)
- T2-weighted (T2)

Exact folder structure:

```
Training/Validation/Testing
│
└─── 00000
│   │
│   └─── FLAIR
│   │   │ Image-1.dcm
│   │   │ Image-2.dcm
│   │   │ ...
│   │
│   └─── T1w
│   │   │ Image-1.dcm
│   │   │ Image-2.dcm
│   │   │ ...
│   │
│   └─── T1wCE
│   │   │ Image-1.dcm
│   │   │ Image-2.dcm
│   │   │ ...
│   │
│   └─── T2w
│   │   │ Image-1.dcm
│   │   │ Image-2.dcm
│   │   │ .....
│
└─── 00001
│   │ ...
│
│ ...
│
└─── 00002
│   │ ...

```

## Files

- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

## Citation

If you reference or use the dataset in any form, include the following citation:

> U.Baid, et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification”, arXiv:2107.02314, 2021.
>

# Required Output

- `submission.csv`: Submission file. Must be in the same format as `sample_submission.csv`.

---
