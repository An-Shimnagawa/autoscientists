# task-autoresearch — nanoGPT `val_bpb` Optimization

Bundled wrapper around [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the AutoScientists multi-agent loop. The upstream repo is cloned on demand into `repo/` (kept out of this repo's git history via `.gitignore`); the wrapper itself just provides the `LAUNCH.md` profile and the `TASK.md` spec.

## Setup (one-time)

```bash
bash task-autoresearch/download_repo.sh
```

This clones `karpathy/autoresearch` into `task-autoresearch/repo/`. Follow any additional data-prep steps in `repo/README.md` (e.g. dataset downloads) before launching.

## Run

From the repository root:

```bash
claude -p "Read runbook.md and execute. Task: task-autoresearch. Run name: ar_baseline."
```

The orchestrator copies `task-autoresearch/` (including the cloned `repo/`) into the new run directory's `task/`, and GPU agents evolve `task/repo/train.py` over many cycles.

## Stop condition

Stagnation: 0 KEEPs in the last 10 experiments. See `LAUNCH.md` → `## Hook: stagnation_response`.

## Hardware

2× NVIDIA H100 (80 GB) recommended. The orchestrator pins `CUDA_VISIBLE_DEVICES=0` and `=1` and runs one GPU agent per device sequentially.
