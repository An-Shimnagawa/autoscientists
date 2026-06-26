---
name: autoresearch-nanogpt
task_type: optimization
metric: val_bpb
direction: minimize
---

# Autoresearch: nanoGPT `val_bpb` Optimization

Open-ended optimization of the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) nanoGPT pre-training loop. Agents iterate on `repo/train.py` (cloned from upstream by `download_repo.sh`) to drive validation bits-per-byte (`val_bpb`) down indefinitely. There is no wall-clock deadline — the loop runs until stagnation (0 KEEPs in the last 10 experiments) or user interrupt.

## Setup

```bash
bash task-autoresearch/download_repo.sh    # one-time: clones upstream into repo/
```

After cloning, the layout becomes:

```
task-autoresearch/
├── LAUNCH.md           Task profile (runbook hooks for optimization)
├── TASK.md             This file
├── README.md           Setup notes
├── download_repo.sh    Clones karpathy/autoresearch into repo/
├── .gitignore          Excludes repo/ from this repo
└── repo/               karpathy/autoresearch clone (after running download_repo.sh)
    └── train.py        The file the agents evolve
```

## Hardware

2× NVIDIA H100 (80 GB) recommended — GPU agents run sequentially per device, pinned to `CUDA_VISIBLE_DEVICES=0` and `=1`.

## Metric

- **`val_bpb`** (validation bits-per-byte): lower is better.
- A result is a KEEP if it strictly improves on the current champion.
- The system stops when the last 10 experiments produced 0 KEEPs.

## Run

```bash
claude -p "Read runbook.md and execute. Task: task-autoresearch. Run name: ar_baseline."
```
