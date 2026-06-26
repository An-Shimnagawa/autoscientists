---
task_type: biomlbench
name: kaggle-uw-madison-gi-tract-image-segmentation
description: >
  Segment stomach and bowel on MR-Linac guided MRI scans (50 cancer patients).
  Maximize combined Dice + 3D Hausdorff distance metric. Prizes: 1st $12K.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **16-bit PNG images:** Use PIL/cv2 to load correctly (`Image.open(...).convert('I')` or keep as 16-bit).
- **3 classes:** stomach, small_bowel, large_bowel — predict all three.
- **RLE encoding:** Output is run-length encoded masks.
- **Compute:** NVIDIA A100-SXM4 GPU (40 GB VRAM, 503 GB RAM), 16-hour wall-clock limit. **`submission.csv` must be saved before the wall-clock limit expires.**
- **PyTorch/CUDA note:** The venv has torch 2.11.0+cu130 but system CUDA is 12.9. Install a compatible version if needed: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out ground-truth RLE masks for the **fixed test split**
> (20,400 slice-class rows in `data/test_split.csv`). It is used **only** by the external grader to
> score your final `submission.csv`. **Never read, open, or use this file in `train.py` or any
> intermediate evaluation.** Doing so would invalidate your score.
>
> **How to evaluate during development:** A held-out validation split has been pre-built for you.
> Train on `data/train_for_modeling.csv` (77,328 rows) and score your predictions against
> `data/val.csv` (17,760 rows — `id`, `class`, `predicted` ground-truth RLE). **Never use `data/test_split.csv` or
> `data/private/answers.csv` for development evaluation.** See the **Iterative Leaderboard**
> section below.

---

# Kaggle: UW-Madison GI Tract Image Segmentation

**Metric:** 0.4 × Dice + 0.6 × (1 − normalized 3D Hausdorff) — higher is better  
**Task type:** Medical image segmentation  
**Awards medals:** Yes  
**Prizes:** 1st $12K, 2nd $8K, 3rd $5K  
**Competition:** https://kaggle.com/competitions/uw-madison-gi-tract-image-segmentation

---

## The Problem

Cancer patients receiving radiation therapy need precise delineation of GI organs (stomach, small bowel, large bowel) to avoid irradiating healthy tissue. Automatic segmentation on daily MR-Linac scans (used for real-time adaptive radiation therapy) would enable oncologists to safely escalate radiation doses to tumors. 50 patients, 1–5 scans each across treatment days.

**Input:** 16-bit grayscale PNG MRI slices  
**Output:** RLE-encoded segmentation masks for stomach, small_bowel, large_bowel  
**Evaluation:** `0.4 × Dice + 0.6 × (1 − Hausdorff)` — higher is better

### Metric details

The two components are computed differently:

- **Dice** — computed **per (slice, class) row** (20,400 rows for the test set). When both prediction and ground truth are empty, Dice = **0** (not 1, not skipped). Final Dice is `np.mean` across all rows.
- **Hausdorff** — computed **per 3D case-day volume**. For each (case, day), all three class masks are **OR-merged into one combined binary mask per slice**, then slices are stacked into a single 3D volume. The symmetric Hausdorff distance is computed between predicted and ground-truth voxel coordinates, normalized per-axis by `volume.shape` so coordinates lie in [0, 1], then divided by the unit-cube diagonal (√3) to bound the result in [0, 1]. `nanmean` across case-days (pairs where both volumes are empty are skipped). If either volume is empty and the other is not, Hausdorff = 1.0 (maximum penalty).

Final score: `0.4 * np.mean(dice_per_slice) + 0.6 * (1 - np.nanmean(hausdorff_per_caseday))` — range [0, 1], higher is better.

```python
from scipy.spatial.distance import directed_hausdorff
import numpy as np

UNIT_CUBE_DIAGONAL = np.sqrt(3)

def dice_2d(pred, true):
    """pred, true: 2D binary arrays (H, W). Returns 0 when both empty."""
    inter = (pred & true).sum()
    denom = pred.sum() + true.sum()
    return float(2 * inter / denom) if denom else 0.0

def hausdorff_3d(pred_vol, true_vol):
    """pred_vol, true_vol: 3D binary arrays (slices, H, W) — all classes OR-merged."""
    if pred_vol.sum() == 0 and true_vol.sum() == 0:
        return np.nan          # skipped in nanmean
    if pred_vol.sum() == 0 or true_vol.sum() == 0:
        return 1.0             # maximum penalty
    if pred_vol.sum() > 10 * true_vol.sum():
        return 1.0             # overpredict guard
    tc = np.argwhere(true_vol) / np.array(true_vol.shape)   # per-axis normalisation
    pc = np.argwhere(pred_vol) / np.array(pred_vol.shape)
    h = max(directed_hausdorff(tc, pc)[0], directed_hausdorff(pc, tc)[0])
    return h / UNIT_CUBE_DIAGONAL

# --- grouping: OR all 3 class masks per slice, stack into 3D per case-day ---
# submission and answers each have one row per (slice_id, class);
# group_masks_by_day ORs the 3 class rows for the same slice_id, then
# stacks slices sorted by slice_id into a (n_slices, H, W) volume.

def score(pred_masks_2d, true_masks_2d, pred_vols_3d, true_vols_3d):
    dice_scores = [dice_2d(p, t) for p, t in zip(pred_masks_2d, true_masks_2d)]
    hd_scores   = [hausdorff_3d(p, t) for p, t in zip(pred_vols_3d, true_vols_3d)]
    return 0.4 * np.mean(dice_scores) + 0.6 * (1 - np.nanmean(hd_scores))
```

---

## Data

### Location

Data is pre-downloaded. `data/` is a symlink to:
```
/n/netscratch/mzitnik_lab/Lab/afang/kaggle/uw-madison-gi-tract-image-segmentation/
├── train/                    # MRI slices for training cases
│   └── case{id}/case{id}_day{day}/scans/   # PNG 16-bit grayscale slices
├── train_for_modeling.csv    # 77,328 rows — train on this during iteration
├── val.csv                   # 17,760 rows — held-out val with labels (id, class, predicted)
├── train_split.csv           # 95,088 rows — full labels (train_for_modeling + val combined;
│                             #   use only for the final submission retrain)
├── test_split.csv            # 20,400 rows — test IDs (id, class, no labels) — predict these
├── sample_submission.csv     # 20,400 rows with dummy predictions
└── val_case_days.txt         # list of (case, case_day) pairs held out in val
```

All training images are in `data/train/`. `submission.csv` must cover all 20,400 rows in
`data/test_split.csv` (6,800 slices × 3 classes). Do not submit to kaggle.com.

### Scan path and filename convention

Example path:
```
data/train/case7/case7_day13/scans/slice_0001_266_266_1.50_1.50.png
```

| Component | Example | Meaning |
|---|---|---|
| `case7` | `case{N}` | Patient case ID (anonymized patient) |
| `case7_day13` | `case{N}_day{D}` | Scan session: day D of treatment for patient N |
| `scans/` | — | Directory containing all 2D slice PNGs for this session |
| `slice_0001` | `slice_{NNNN}` | Slice index within the 3D volume (1-indexed, zero-padded to 4 digits); slices are ordered inferior → superior |
| `266` (first) | `{W}` | Image width in pixels |
| `266` (second) | `{H}` | Image height in pixels |
| `1.50` (first) | `{sx}` | Pixel spacing in x-direction (mm/pixel) |
| `1.50` (second) | `{sy}` | Pixel spacing in y-direction (mm/pixel) |

The slice index encoded in the filename (`0001`) matches the `slice_NNNN` suffix in the `id` column of the CSV files (e.g. `case7_day13_slice_0001`). Physical slice thickness in the superior-inferior direction (z) is always 3 mm.

### Data format

```python
import pandas as pd, numpy as np
from PIL import Image

# Iteration: use train_for_modeling + val
train = pd.read_csv("data/train_for_modeling.csv")  # 77,328 rows — id, class, segmentation
val   = pd.read_csv("data/val.csv")                  # 17,760 rows — id, class, predicted
# id format: case{case_id}_day{day_id}_slice_{slice_id}
# Columns in train_for_modeling: id, class, segmentation (RLE or empty if no organ present)
# Columns in val: id, class, predicted (same RLE, renamed to match submission convention)

# Decode RLE mask (column name is 'segmentation' in train files, 'predicted' in val/answers):
def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle) or mask_rle == "": return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends): img[lo:hi] = 1
    return img.reshape(shape, order="F")  # column-major: top-to-bottom then left-to-right

# Load image
img = np.array(Image.open("data/train/case1/case1_day0/scans/slice_0001_266_266_1.50_1.50.png"))
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
1. **Evaluate locally** — train on `data/train_for_modeling.csv`, predict on `data/val.csv` IDs,
   score against `data/val.csv` labels, print the val metric.
2. **Save `submission.csv`** — predictions for all 20,400 rows in `data/test_split.csv` (columns: id, class, predicted RLE).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


**IMPORTANT:** The biomlbench grader scores `submission.csv` against the 20,400-row test split.
See the **Data → Location** section above for the full file listing.

---

## Iterative Leaderboard (ClawLab API)

**Metric:** 0.4×Dice + 0.6×(1−normalized Hausdorff) — higher is better

A held-out validation split (`data/val.csv`) has been pre-built from `data/train_split.csv`.
It covers two generalization regimes matching the public Kaggle test-set design: a set of
cases held out entirely (the model never trains on these patients), and for other cases the
most recent treatment days are held out (the model sees earlier days of the same patient
during training but must predict later ones). Do **not** validate on `data/test_split.csv` —
that is the graded test set.

```python
import pandas as pd

# Training and validation data
train_df = pd.read_csv("data/train_for_modeling.csv")  # 77,328 rows — train on this
val_df   = pd.read_csv("data/val.csv")                  # 17,760 rows — val IDs + labels

# ... train model on train_df slices ...
# ... predict masks for all slice IDs in val_df["id"] ...

# Score: val_df has a 'predicted' column with ground-truth RLE.
# Image dimensions are encoded in each slice filename (see Scan path convention).
# Use your grading code (same as for submission.csv) against val_df.
# score = 0.4 * nanmean(dice_per_slice) + 0.6 * (1 - nanmean(hausdorff_per_caseday))
print(f"Val score: {score:.4f}")
```

Report this val score to the ClawLab leaderboard after each iteration.

For the **final submission run**: retrain on the full `data/train_split.csv`
(= `train_for_modeling.csv` + `val.csv` rows combined) using your best
hyperparameters, then predict on `data/test_split.csv` to produce `submission.csv`.
This maximises training signal for the held-out test set.

---

## Final Submission

Retrain on the full `data/train_split.csv` (all 95,088 rows — training + validation combined)
using your best hyperparameters from iteration, predict all slices in `data/test_split.csv`,
save `submission.csv`. **Do not use `data/private/answers.csv`** — the grader evaluates
`submission.csv` against that file; reading it during training invalidates your score.

```python
# Train on train_split.csv, predict test_split.csv IDs
# submission must have columns: id, class, predicted (RLE)
test_df = pd.read_csv("data/test_split.csv")  # 20,400 rows
# ... predict all slices in test_df ...
submission = pd.DataFrame({"id": ..., "class": ..., "predicted": ...})
submission.to_csv("submission.csv", index=False)
```

---

## Output Format

`submission.csv` must have exactly three columns: `id`, `class`, `predicted`.

**Important:** `train_split.csv` uses the column name `segmentation` for masks — your submission must
use `predicted` instead. The grader validates for `id`, `class`, `predicted` specifically.

```csv
id,class,predicted
case1_day0_slice_0001,large_bowel,1 10 20 5
case1_day0_slice_0001,small_bowel,
case1_day0_slice_0001,stomach,30 15
...
```

### `predicted` column requirements

The `predicted` column must contain a **valid segmentation RLE string** (or be left blank for empty
masks) for **every** test row. Invalid RLE will cause the grader to reject the submission.

- Space-separated `start length start length …` pairs, using the same RLE convention as
  `train_split.csv`'s `segmentation` column (1-indexed pixels, numbered top-to-bottom then left-to-right).
- `start` values must be strictly increasing (sorted).
- All `start` and `length` values must be positive integers (`> 0`).
- Decoded pixel ranges must not overlap within a single row.
- Empty/NaN `predicted` = no segmentation for that class. Leave the field blank — do **not** write
  `0`, `None`, `nan`, or `-1`.

### Coverage

- Must cover **all 20,400 rows** from `data/test_split.csv` (6,800 slices × 3 classes:
  `large_bowel`, `small_bowel`, `stomach`).
- Each `(id, class)` pair must appear exactly once. Missing or duplicate pairs will mis-score.
- Row order does not matter — the grader matches by `id` + `class`.

---



---

## Background

# Overview

## Description

In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day. In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment.

The UW-Madison Carbone Cancer Center is a pioneer in MR-Linac based radiotherapy, and has treated patients with MRI guided radiotherapy based on their daily anatomy since 2015. UW-Madison has generously agreed to support this project which provides anonymized MRIs of patients treated at the UW-Madison Carbone Cancer Center. The University of Wisconsin-Madison is a public land-grant research university in Madison, Wisconsin. The Wisconsin Idea is the university's pledge to the state, the nation, and the world that their endeavors will benefit all citizens.

In this competition, you’ll create a model to automatically segment the stomach and intestines on MRI scans. The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment. You'll base your algorithm on a dataset of these scans to come up with creative deep learning solutions that will help cancer patients get better care.

Cancer takes enough of a toll. If successful, you'll enable radiation oncologists to safely deliver higher doses of radiation to tumors while avoiding the stomach and intestines. This will make cancer patients' daily treatments faster and allow them to get more effective treatment with less side effects and better long-term cancer control.

## Acknowledgments

Sangjune Laurence Lee MSE MD FRCPC DABR
Poonam Yadav Ph.D., DABR
Yin Li PhD
Jason J. Meudt BS, RTT
Jessica Strang
Dustin Hebel
Alyx Alfson MS CMD, R.T.(T)
Stephanie J. Olson RTT (BS), CMD (MS)
Tera R. Kruser MS, RTT, CMD
Jennifer B Smilowitz, Ph.D., DABR, FAAPM
Kailee Borchert
Brianne Loritz
John Bayouth PhD
Michael Bassetti MD PhD

Work funded by the University of Wisconsin Carbone Cancer Center Pancreas Pilot Research Grant.

## Evaluation

This competition is evaluated on the mean Dice coefficient and 3D Hausdorff distance. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where $X$ is the predicted set of pixels and $Y$ is the ground truth. The Dice coefficient is defined to be 0 when both $X$ and $Y$ are empty. The leaderboard score is the mean of the Dice coefficients for each image in the test set.

Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (In this competition, the slice depth for all scans is set to 1.) The scipy code for Hausdorff is linked. The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance.

## Submission File

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
id,class,predicted
1,large_bowel,1 1 5 1
1,small_bowel,1 1
1,stomach,1 1
2,large_bowel,1 5 2 17
etc.
```

## Timeline

- April 14, 2021 - Start Date.
- July 7, 2022 - Entry Deadline. You must accept the competition rules before this date in order to compete.
- July 7, 2022 - Team Merger Deadline. This is the last day participants may join or merge teams.
- July 14, 2022 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

- 1st Place - \$12,000
- 2nd Place - \$8,000
- 3rd Place - \$5,000

## Code Requirements

### This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook `<= 9 hours run-time`
- GPU Notebook `<= 9 hours run-time`
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named submission.csv

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

## Citation

happyharrycn, Maggie, Phil Culliton, Poonam Yadav, Sangjune Laurence Lee. (2022). UW-Madison GI Tract Image Segmentation . Kaggle. https://kaggle.com/competitions/uw-madison-gi-tract-image-segmentation

# Dataset Description

In this competition we are segmenting organs cells in images. The training annotations are provided as RLE-encoded masks, and the images are in 16-bit grayscale PNG format.

Each case in this competition is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal of this competition is to be able to generalize to both partially and wholly unseen cases.

Note that, in this case, the test set is entirely unseen. It is roughly 50 cases, with a varying number of days and slices, as seen in the training set.

## How does an entirely hidden test set work?

The test set in this competition is only available when your code is submitted. The sample_submission.csv provided in the public set is an empty placeholder that shows the required submission format; you should perform your modeling, cross-validation, etc., using the training set, and write code to process a non-empty sample submission. It will contain rows with id, class and predicted columns as described in the Evaluation page.

When you submit your notebook, your code will be run against the non-hidden test set, which has the same folder format (<case>/<case_day>/<scans>) as the training data.

## Files

- train.csv - IDs and masks for all training objects.
- sample_submission.csv - a sample submission file in the correct format
- train - a folder of case/day folders, each containing slice images for a particular case on a given day.

Note that the image filenames include 4 numbers (ex. 276_276_1.63_1.63.png). These four numbers are slice width / height (integers in pixels) and width/height pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.

Physical pixel thickness in superior-inferior direction is 3mm.

## Columns

- `id` - unique identifier for object
- `class` - the predicted class for the object
- `segmentation` - RLE-encoded pixels for the identified object

# Required Output

- `submission.csv`: Submission file. Must be in the same format as `sample_submission.csv`.

---
