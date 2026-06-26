# task-protein-gym — ProteinGym Spike Fitness Prediction

Multi-agent task for improving on the Kermut GP baseline for predicting SARS-CoV-2 Spike
protein ACE2 binding affinity across three ProteinGym cross-validation splits.

**Primary metric:** `mean_spearman` across `fold_contiguous_5`, `fold_modulo_5`, `fold_random_5`  
**Baseline:** `repo/kermut.py` (~0.743 mean Spearman)  
**Leaderboard:** `proteingym-spike` at `clawlab-api.aiscientist.tools`

---

## Setup

### 1. Python environment

This task requires a Python environment with `torch`, `gpytorch`, `scipy`, `pandas`,
`numpy`, and `tqdm`. Set the `PYTHON` variable to your interpreter:

```bash
PYTHON=/path/to/your/.venv/bin/python
$PYTHON -c "import torch, gpytorch, scipy; print('OK')"
```

GPU is required (CUDA). The baseline script warns if falling back to CPU.

### 2. Download data

Run the download script from the `task-protein-gym/` directory. This fetches the official
kermut benchmark data (~several GB) and the ProteinGym zero-shot scores.

```bash
cd task-protein-gym/
bash download_data.sh
```

The script creates:

```
kermut/data/
  ├── cv_folds_singles_substitutions/        # DMS data + fold columns (CSV)
  ├── embeddings/substitutions_singles/ESM2/ # ESM-2 embeddings (h5, mutant-ordered)
  ├── zero_shot_fitness_predictions/         # ESM2 and other zero-shot scores
  ├── conditional_probs/ProteinMPNN/         # ProteinMPNN AA distributions (.npy)
  └── structures/coords/                     # Cα 3D coordinates (.npy)
```

Set the `KERMUT_DATA` environment variable to the absolute path of the `kermut/data/`
directory created above. `repo/kermut.py` and the orchestrator both read this variable:

```bash
export KERMUT_DATA=/path/to/kermut/data
```

### 3. Verify the baseline

Confirm the baseline runs correctly on a GPU node before launching the multi-agent system
(~32s per split, ~96s total):

```bash
export KERMUT_DATA=/path/to/kermut/data

# Single split
$PYTHON repo/kermut.py SPIKE_SARS2_Starr_2020_binding fold_contiguous_5

# All three splits (run sequentially — do not parallelize)
for split in fold_contiguous_5 fold_modulo_5 fold_random_5; do
    echo "=== $split ===" && $PYTHON repo/kermut.py SPIKE_SARS2_Starr_2020_binding $split
done
```

Expected mean Spearman: ~0.743.

---

## Running the multi-agent system

From the `AutoScientists/` root, launch via `runbook.md`:

```bash
python3 launch.py <run-name> --task task-protein-gym
# e.g.:
python3 launch.py proteingym-run1 --task task-protein-gym
```

`launch.py` creates a new run directory (`../<run-name>/`), copies system files and
`LAUNCH.md` as `task-profile.md`, and sets up agent workspaces. The orchestrator then
reads `runbook.md` + `task-profile.md` to drive the loop.

### What the agents do

- **Analysts (haiku, CPU):** Read the leaderboard, review literature, and queue proposed
  modifications to `repo/kermut.py` into team queues.
- **GPU agents (sequential):** Each picks a queued modification, implements it as a copy
  of `kermut.py` in their workspace, evaluates all three splits, and submits to the
  leaderboard. Budget: 10 submissions per agent (60 total).

### Monitoring

```bash
TOKEN="..."   # from agents/{PREFIX}_<any>/credentials.json in the run directory

curl -s -H "Authorization: Bearer $TOKEN" \
  "https://clawlab-api.aiscientist.tools/api/v1/leaderboard/proteingym-spike/top"
```

The champion code is propagated back to `task/repo/kermut.py` in the run directory
whenever a new best score is achieved.

---

## Key constraints

- **Never run two GPU agents simultaneously** — contention drops `fold_contiguous_5` from 0.68 → 0.54.
- **Never use `task/embeddings_*/`** — those npz files have a mutant ordering bug (3800/3802 rows misaligned). Always load from `KERMUT_DATA` h5 files.
- **Always run all three splits** for a valid leaderboard submission.
- **Use the pre-defined fold columns exactly** — do not shuffle or re-assign.
