---
task_type: biomlbench
name: kaggle-histopathologic-cancer-detection
description: >
  Binary classification of metastatic cancer in 96x96px histopathology image patches.
  Maximize ROC-AUC. PatchCamelyon benchmark (PCam), ~220K training images.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **Center 32×32 matters:** Only central region has labels; surrounding context is additional information.
- **Large dataset:** ~220K training images — data loading efficiency matters.
- **Compute:** NVIDIA A100-SXM4 GPU (40 GB VRAM, 503 GB RAM), 16-hour wall-clock limit. **`submission.csv` must be saved before the wall-clock limit expires.**
- **PyTorch/CUDA note:** The venv has torch 2.11.0+cu130 but system CUDA is 12.9. Install a compatible version if needed: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out ground-truth labels for the **fixed test set**
> (the 45,561 images in `data/test/`). It is used **only** by the external grader to score your final
> `submission.csv`. **Never read, open, or use this file in `train.py` or any intermediate evaluation.**
> Doing so would invalidate your score.
>
> **How to evaluate during development:** Split `data/train_labels.csv` however you like to get a
> local validation estimate. The grader scores predictions for **all 45,561 images in `data/test/`** —
> your `submission.csv` must contain exactly those IDs. You can use `data/sample_submission.csv` to
> get the correct list of IDs (same 45,561 IDs with dummy labels).

---

# Kaggle: Histopathologic Cancer Detection

**Metric:** ROC-AUC — higher is better  
**Task type:** Binary image classification  
**Awards medals:** No (practice competition)  
**Competition:** https://kaggle.com/competitions/histopathologic-cancer-detection

---

## The Problem

Identify metastatic cancer in small pathology image patches (96×96px). Only the central 32×32px region of each patch is relevant — surrounding pixels provide context. Based on the PatchCamelyon (PCam) benchmark derived from Camelyon16 challenge. A positive label = at least one pixel in the center 32×32 contains tumor tissue.

**Input:** 96×96px RGB pathology image patches  
**Output:** Probability of metastatic tissue presence (0–1)  
**Evaluation:** ROC-AUC — higher is better

---

## Data

### Location

Data is pre-downloaded. `data/` is a symlink to:
```
/n/netscratch/mzitnik_lab/Lab/afang/kaggle/histopathologic-cancer-detection/
├── histopathologic-cancer-detection.zip   # original download
├── train/          # ~174K .tif image patches for training (96x96 RGB)
├── test/           # 45,561 .tif image patches — the biomlbench test set (predict these)
├── train_labels.csv # id, label (0/1) — training images only (174,464 rows)
└── sample_submission.csv  # 45,561 rows with correct test IDs (use for submission template)
```

**Note:** The biomlbench test set is `data/test/` — 45,561 images carved from the original Kaggle
training set (they have known labels in `data/private/answers.csv`). Your `submission.csv` must
contain predictions for all 45,561 IDs in `data/test/`. Do not use the `train_test_split` approach
described in the original Kaggle task — instead predict all images in `data/test/` directly.
Do not submit to kaggle.com.

### How to prepare data
```bash
# Data is already available via the data/ symlink.
# To unzip if needed:
cd /n/netscratch/mzitnik_lab/Lab/afang/kaggle/histopathologic-cancer-detection/
unzip -q histopathologic-cancer-detection.zip
```

### Data format

```python
from PIL import Image
import pandas as pd, numpy as np

# Training data
labels = pd.read_csv("data/train_labels.csv")
# Columns: id, label (0=no cancer, 1=cancer present in center 32x32)
# 174,464 rows — images in data/train/

# Test data (predict these 45,561 images)
sample = pd.read_csv("data/sample_submission.csv")
# Columns: id, label — use sample["id"] to get the list of test IDs

# Load a training image
img = Image.open(f"data/train/{labels['id'][0]}.tif")
# img.size = (96, 96), RGB

# Load a test image
img = Image.open(f"data/test/{sample['id'][0]}.tif")
# img.size = (96, 96), RGB
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
1. **Evaluate locally** — train on training data, predict on a local val fold, print the val metric.
2. **Save `submission.csv`** — cancer predictions for all 45,561 images in `data/test/` (columns: `id`, `label`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


**IMPORTANT:** The biomlbench grader scores `submission.csv` against `data/test/` — the 45,561 test
images. Your submission must contain **exactly the IDs from `data/sample_submission.csv`** (45,561
rows, correct test IDs). Use `data/train_labels.csv` for training only.

```python
import pandas as pd
from pathlib import Path

# Get the exact test IDs the grader expects
sample = pd.read_csv("data/sample_submission.csv")
test_ids = sample["id"].tolist()
# test_ids has 45,561 entries — these are the image stems in data/test/

# Train/val split for local evaluation — use train_labels.csv however you like
labels = pd.read_csv("data/train_labels.csv")  # 174,464 rows
# e.g. a simple 80/20 split for local validation:
from sklearn.model_selection import train_test_split
tr, val = train_test_split(labels, test_size=0.2, random_state=0)
# train on tr, estimate val ROC-AUC — but submission.csv must cover test_ids, not val
```

The test set has **45,561 images** in `data/test/`. Train on `data/train_labels.csv` images (in `data/train/`), predict probabilities for all test images (in `data/test/`), save as `submission.csv`.

---

## Iterative Leaderboard (ClawLab API)

**Metric:** ROC-AUC — higher is better

Use any train/val split of `data/train_labels.csv` for local evaluation. The example below uses a
simple 80/20 split — you can use any strategy you prefer (stratified, k-fold, etc.):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

labels = pd.read_csv("data/train_labels.csv")  # 174,464 rows, images in data/train/
tr, val = train_test_split(labels, test_size=0.2, random_state=0, stratify=labels["label"])
# ... train on tr images (data/train/{id}.tif), predict on val images (data/train/{id}.tif) ...
score = roc_auc_score(val["label"].values, val_preds)
print(f"Val ROC-AUC: {score:.4f}")
```

Report this val score to the ClawLab leaderboard after each iteration.

---

## Final Submission

Train on `data/train_labels.csv` images, predict all 45,561 images in `data/test/`, save `submission.csv`.

```python
import pandas as pd
from pathlib import Path

# Get the exact IDs expected by the grader
sample = pd.read_csv("data/sample_submission.csv")  # 45,561 rows
test_ids = sample["id"].tolist()

# ... train model on data/train_labels.csv images (data/train/{id}.tif) ...
# ... predict probabilities for all test images (data/test/{id}.tif) ...

submission = pd.DataFrame({"id": test_ids, "label": test_preds})
submission.to_csv("submission.csv", index=False)
```

---

## Output Format

```csv
id,label
f38a6374c348f90b587e046aac6079959adf3835,0.923
c18f2d887b7ae4f6742ee445113fa1aef383ed77,0.045
...
```

---



---

## Background

# Overview

## Overview

![Microscope](https://storage.googleapis.com/kaggle-media/competitions/playground/Microscope)

In this competition, you must create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The data for this competition is a slightly modified version of the PatchCamelyon (PCam) [benchmark dataset](https://github.com/basveeling/pcam) (the original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates).

PCam is highly interesting for both its size, simplicity to get started on, and approachability. In the authors' words:

> [PCam] packs the clinically-relevant task of metastasis detection into a straight-forward binary image classification task, akin to CIFAR-10 and MNIST. Models can easily be trained on a single GPU in a couple hours, and achieve competitive scores in the Camelyon16 tasks of tumor detection and whole-slide image diagnosis. Furthermore, the balance between task-difficulty and tractability makes it a prime suspect for fundamental machine learning research on topics as active learning, model uncertainty, and explainability. 

### Acknowledgements

Kaggle is hosting this competition for the machine learning community to use for fun and practice. This dataset was provided by Bas Veeling, with additional input from Babak Ehteshami Bejnordi, Geert Litjens, and Jeroen van der Laak.

You may view and download the official Pcam dataset from [GitHub](https://github.com/basveeling/pcam). The data is provided under the [CC0 License](https://choosealicense.com/licenses/cc0-1.0/), following the license of Camelyon16.

If you use PCam in a scientific publication, please reference the following papers:

[1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". [arXiv:1806.03962](http://arxiv.org/abs/1806.03962)

[2] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA: The Journal of the American Medical Association, 318(22), 2199–2210. [doi:jama.2017.14585](https://doi.org/10.1001/jama.2017.14585)

Photo by [Ousa Chea](https://unsplash.com/photos/gKUC4TMhOiY)

## Evaluation

Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

### Submission File

For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Prizes

At the conclusion of the competition, the top five most popular Kernels--as determined by number of upvotes at the deadline--will receive Kaggle Swag.

Happy modeling and thanks for being great Kernelers of the Kaggle community!

## Timeline

The competition will conclude **March 30, 2019** at 11:59 PM UTC.

## Citation

Will Cukierski. (2018). Histopathologic Cancer Detection. Kaggle. https://kaggle.com/competitions/histopathologic-cancer-detection

# Data

## Dataset Description

In this dataset, you are provided with a large number of small pathology images to classify. Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder. A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.

The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates. We have otherwise maintained the same data and splits as the PCam benchmark.

- `sample_submission.csv`: Example submission format. Use this to check your submission format.

## Required Output

- `submission.csv`: Submission file. Must be in the same format as `sample_submission.csv`.

---
