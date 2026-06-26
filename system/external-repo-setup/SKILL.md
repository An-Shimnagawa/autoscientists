---
name: external-repo-setup
description: >
  Protocol for GPU agents to clone external GitHub repos, install their
  dependencies, download pretrained weights, and integrate them as featurizers
  or models inside a focus-area train.py.
  Use this when a proposal references a GitHub repo or pretrained checkpoint
  that is not already present in the focus-area workspace.
---

# External Repo Setup

This system addresses a recurring failure mode: agents propose experiments that
depend on external GitHub repos or pretrained weights, but never actually set
them up because no protocol exists for doing so. This system gives GPU agents a
concrete, step-by-step procedure so those proposals can be executed rather
than left as "future work."

---

## Storage Locations — ALWAYS Use Scratch

**All repos, venvs, checkpoints, and caches go to scratch, not to $HOME.**

```python
SCRATCH_ROOT = "${WORKSPACE_ROOT}/AnonAPI"

REPOS_DIR   = f"{SCRATCH_ROOT}/repos"          # cloned GitHub repos
CKPT_DIR    = f"{SCRATCH_ROOT}/checkpoints"    # pretrained weights
HF_CACHE    = "${WORKSPACE_ROOT}/huggingface_cache"
TORCH_CACHE = "${WORKSPACE_ROOT}/torch_cache"
UV_CACHE    = "${WORKSPACE_ROOT}/uv_cache"
```

**Why scratch?** $HOME has a strict quota. Model checkpoints and conda/venv
environments can easily consume tens of GB. Scratch is the designated location
for large intermediate files on this cluster.

Always set these environment variables when running any download or training:

```bash
export HF_HOME=${WORKSPACE_ROOT}/huggingface_cache
export TRANSFORMERS_CACHE=${WORKSPACE_ROOT}/huggingface_cache
export HF_DATASETS_CACHE=${WORKSPACE_ROOT}/huggingface_cache
export TORCH_HOME=${WORKSPACE_ROOT}/torch_cache
export UV_CACHE_DIR=${WORKSPACE_ROOT}/uv_cache
```

---

## When to Use This Skill

A GPU agent needs this system when its claimed experiment requires **any** of:

- A Python package that cannot be `pip/uv` installed from PyPI (i.e., it lives
  on GitHub and must be cloned + installed from source)
- A pretrained model checkpoint (e.g. a `.ckpt`, `.pt`, `.bin`, or
  `safetensors` file) that must be downloaded separately
- A featurizer or encoder that wraps a third-party model whose interface is
  defined in that model's own codebase

If the package IS on PyPI and has no checkpoint dependency, a normal
`uv add <package>` in `train.py` is sufficient — you do not need this system.

---

## Step 1 — Determine What Needs to Be Pulled

Before cloning anything, read the proposal carefully and identify:

1. **Repo URL** — the canonical GitHub URL (e.g. `https://github.com/org/repo`)
2. **Commit or tag** — always pin to a specific commit or release tag, never
   `main`/`HEAD`, to ensure reproducibility
3. **Checkpoint source** — HuggingFace model ID (e.g. `org/model-name`),
   Zenodo DOI, figshare URL, or direct link in the repo's README
4. **Interface** — what function/class from the repo you will call, and what
   it expects as input (SMILES strings? amino acid sequences? numpy arrays?)

Write these down before touching the filesystem.

---

## Step 2 — Clone into the Shared Scratch Cache

Clone the repo into a shared cache directory so all GPU agents on this machine
can reuse the same checkout without re-downloading:

```python
import subprocess, os
from pathlib import Path

REPOS_DIR = Path("${WORKSPACE_ROOT}/AnonAPI/repos")
REPOS_DIR.mkdir(parents=True, exist_ok=True)

REPO_URL    = "https://github.com/ORG/REPO"   # fill in
REPO_COMMIT = "abc1234"                        # pin to specific commit or tag
REPO_NAME   = "REPO"                           # short name for the dir
REPO_DIR    = REPOS_DIR / REPO_NAME

if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        check=True
    )
    # If pinning to a specific non-tag commit, fetch and checkout:
    if len(REPO_COMMIT) > 10:  # full hash, not a tag
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", REPO_COMMIT],
            check=True
        )
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "checkout", REPO_COMMIT],
            check=True
        )
else:
    print(f"Repo already cloned at {REPO_DIR}")
```

**Why a shared scratch directory?** Use storage that is visible to all
compute nodes. Never clone into `$FOCUS_ROOT` or `$HOME` — both have tight
quotas.

---

## Step 3 — Python Environment: Reuse the Existing Venv First

**Before creating any new environment, check if the existing venv already has
what you need.** Creating new venvs for every repo wastes gigabytes of disk.

The primary venv for all focus-area work is:

```
EXISTING_VENV = ${WORKSPACE_ROOT}/workspace/ai_scientists/.venv
PYTHON        = ${WORKSPACE_ROOT}/workspace/ai_scientists/.venv/bin/python
PIP           = ${WORKSPACE_ROOT}/workspace/ai_scientists/.venv/bin/pip
```

### 3a. Check if the existing venv is sufficient

```python
import subprocess

PYTHON = "${WORKSPACE_ROOT}/workspace/ai_scientists/.venv/bin/python"
PIP    = "${WORKSPACE_ROOT}/workspace/ai_scientists/.venv/bin/pip"

# 1. Check if the repo can be imported without any installs
result = subprocess.run(
    [PYTHON, "-c", f"import sys; sys.path.insert(0, '{REPO_DIR}'); import molbert"],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("Repo importable from existing venv — no install needed")
else:
    print(f"Import failed: {result.stderr[:200]}")
    # Proceed to 3b
```

### 3b. Try pip-installing missing deps into the existing venv

If the repo has a `requirements.txt` or `setup.py`, try installing its deps
into the existing venv first. Most scientific repos share deps (torch,
numpy, pandas, rdkit) that are already present.

```python
req_file = REPO_DIR / "requirements.txt"
if req_file.exists():
    # Install only packages not already present — pip handles duplicates
    subprocess.run(
        [PIP, "install", "-r", str(req_file), "--quiet"],
        check=False  # don't crash on conflict — inspect error instead
    )
elif (REPO_DIR / "setup.py").exists() or (REPO_DIR / "pyproject.toml").exists():
    subprocess.run(
        [PIP, "install", "-e", str(REPO_DIR), "--quiet", "--no-deps"],
        check=False  # --no-deps avoids overwriting existing packages
    )
```

**Conflict handling:** If a package conflict arises (e.g. repo pins
`torch==1.4.0` but venv has `torch==2.10.0`), do NOT install the old version.
Instead:
1. Add the repo to `sys.path` directly (no install needed for pure Python repos)
2. Patch the incompatible imports in the repo code (see Step 5)
3. Only as a last resort: create a new venv in scratch (Step 3c)

### 3c. Create a new venv in scratch (last resort)

Only if the existing venv absolutely cannot run the repo's code AND patching
is not feasible:

```python
VENV_DIR = Path("${WORKSPACE_ROOT}/AnonAPI") / f"{REPO_NAME}_venv"

if not VENV_DIR.exists():
    subprocess.run(["uv", "venv", str(VENV_DIR),
                    f"--cache-dir=${WORKSPACE_ROOT}/uv_cache"],
                   check=True)
    PYTHON_NEW = str(VENV_DIR / "bin" / "python")
    PIP_NEW    = str(VENV_DIR / "bin" / "pip")

    req_file = REPO_DIR / "requirements.txt"
    if req_file.exists():
        subprocess.run(
            [PIP_NEW, "install", "-r", str(req_file)],
            env={**os.environ,
                 "UV_CACHE_DIR": "${WORKSPACE_ROOT}/uv_cache",
                 "PIP_CACHE_DIR": "${WORKSPACE_ROOT}/uv_cache/pip"},
            check=True
        )
```

---

## Step 4 — Download Pretrained Weights

### 4a. HuggingFace Hub

```python
import os
os.environ["HF_HOME"] = "${WORKSPACE_ROOT}/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "${WORKSPACE_ROOT}/huggingface_cache"

from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="ORG/MODEL_NAME",
    cache_dir="${WORKSPACE_ROOT}/huggingface_cache",
    revision="main",
    ignore_patterns=["*.msgpack", "flax_model*"]
)
print(f"Model downloaded to: {local_dir}")
```

**Critical:** always set `HF_HOME` and `cache_dir` explicitly. If you forget,
`huggingface_hub` will default to `$HOME/.cache/huggingface` and fill up
your home directory quota.

### 4b. Direct URL download (figshare, Zenodo, GitHub releases)

```python
import urllib.request, hashlib
from pathlib import Path

CKPT_DIR = Path(f"${WORKSPACE_ROOT}/AnonAPI/checkpoints/{REPO_NAME}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_URL  = "https://ndownloader.figshare.com/files/XXXXX"
WEIGHT_PATH = CKPT_DIR / "model.ckpt"
WEIGHT_MD5  = "abc123..."   # from the paper or repo README (optional but recommended)

if not WEIGHT_PATH.exists():
    print(f"Downloading from {WEIGHT_URL} ...")
    urllib.request.urlretrieve(WEIGHT_URL, str(WEIGHT_PATH))
    print(f"Saved to {WEIGHT_PATH} ({WEIGHT_PATH.stat().st_size / 1e6:.0f} MB)")

# Verify MD5 if available
if WEIGHT_MD5:
    actual = hashlib.md5(WEIGHT_PATH.read_bytes()).hexdigest()
    if actual != WEIGHT_MD5:
        WEIGHT_PATH.unlink()
        raise RuntimeError(f"MD5 mismatch — re-download. Expected {WEIGHT_MD5}, got {actual}")
    print("MD5 verified OK")
```

### 4c. Zip archives

```python
import zipfile
if WEIGHT_PATH.suffix == ".zip":
    with zipfile.ZipFile(str(WEIGHT_PATH)) as zf:
        zf.extractall(str(CKPT_DIR))
    print(f"Extracted to {CKPT_DIR}")
    # Find the actual checkpoint
    ckpt_files = list(CKPT_DIR.rglob("*.ckpt")) + list(CKPT_DIR.rglob("*.pt"))
    print(f"Checkpoint files: {ckpt_files}")
```

---

## Step 5 — Patching Repos for Modern Dependencies

Many older repos use APIs that changed in newer versions of torch, transformers,
or numpy. **Rather than installing old package versions, patch the repo code.**
Patches live in the cloned repo directory and are documented in the setup note
(Step 7) so other agents don't redo them.

### Common patches needed

#### transformers 3.x → 4.x

| Old import | New import |
|-----------|-----------|
| `from transformers import AdamW` | `from torch.optim import AdamW` |
| `from transformers.modeling_bert import BertEncoder, BertPooler, BertLMPredictionHead` | `from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertLMPredictionHead` |
| `from transformers.modeling_transfo_xl import PositionalEmbedding` | Inline implementation (see below) |

Inline `PositionalEmbedding` replacement:
```python
import math, torch, torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1).unsqueeze(0)
        return pos_emb.expand(bsz, -1, -1) if bsz is not None else pos_emb
```

#### pytorch-lightning 0.x → 1.x

| Issue | Fix |
|-------|-----|
| `self.hparams = args` in `__init__` | Replace with `self.save_hyperparameters(vars(args)); self._args = args` |
| `model.load_from_checkpoint(path, hparam_overrides=...)` | Load weights directly: `ckpt = torch.load(path, weights_only=False); model.load_state_dict(ckpt['state_dict'])` |
| `model.freeze()` | `for p in model.parameters(): p.requires_grad = False` |

#### numpy 1.24+ (removed aliases)

| Old | New |
|-----|-----|
| `np.long` | `np.int64` |
| `np.float` | `np.float64` |
| `np.bool` | `np.bool_` |
| `np.complex` | `np.complex128` |

#### torch 2.6+ (weights_only default changed)

```python
# Old (crashes in torch 2.6+ for checkpoints with custom classes):
torch.load(path)

# Fix:
torch.load(path, weights_only=False)  # only for trusted checkpoints
```

#### transformers 4.36+ (BertModel API changes)

Custom `BertModel` subclasses that skip `BertModel.__init__` need:
```python
self.attn_implementation = getattr(config, '_attn_implementation', 'eager')
self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
```

Custom embedding `forward` signatures need `past_key_values_length=0` parameter.

BertModel now returns a dataclass by default — pass `return_dict=False` to
get tuple output for code that unpacks `sequence_output, pooled_output = outputs`.

---

## Step 6 — Extract Features (Inference Only)

Use the model as a **frozen featurizer** — pass molecules through it and
extract embeddings, then train your own model on top. Do not fine-tune.

```python
import sys, os
import numpy as np

# Add repo to path
sys.path.insert(0, str(REPO_DIR))

# Set all caches to scratch
os.environ["HF_HOME"] = "${WORKSPACE_ROOT}/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "${WORKSPACE_ROOT}/huggingface_cache"
os.environ["TORCH_HOME"] = "${WORKSPACE_ROOT}/torch_cache"

# Import featurizer (after patching if needed)
from repo_module import SomeFeaturizer

def get_embeddings(smiles_list: list[str], ckpt_path: str,
                   device: str = "cuda") -> np.ndarray:
    f = SomeFeaturizer(ckpt_path, device=device)
    embeddings, valid = f.transform(smiles_list)
    return embeddings, valid
```

---

## Step 7 — Cache the Embeddings

Extraction is slow (minutes for hundreds of molecules). After extracting once,
cache so subsequent runs load instantly:

```python
EMBED_CACHE = Path(f"${WORKSPACE_ROOT}/AnonAPI/embeddings/{REPO_NAME}")
EMBED_CACHE.mkdir(parents=True, exist_ok=True)

for split, smiles_list in [("train", train_smiles), ("val", val_smiles), ("test", test_smiles)]:
    cache_path = EMBED_CACHE / f"{split}.npy"
    if cache_path.exists():
        emb = np.load(str(cache_path))
        print(f"Loaded cached {split} embeddings: {emb.shape}")
    else:
        emb, valid = get_embeddings(smiles_list, ckpt_path=str(CKPT_PATH))
        np.save(str(cache_path), emb)
        print(f"Cached {split} embeddings: {emb.shape}")
```

---

## Step 8 — Document and Share

After successfully setting up the repo, write a setup note so other GPU
agents can load pre-cached embeddings without re-running Steps 1-7:

```python
from datetime import datetime, timezone
setup_note = f"""---
repo: {REPO_URL}
commit: {REPO_COMMIT}
ckpt_path: {CKPT_PATH}
embed_cache: ${WORKSPACE_ROOT}/AnonAPI/embeddings/{REPO_NAME}
embed_dim: {embeddings.shape[1]}
python: {PYTHON}
sys_path_insert: {REPO_DIR}
patches_applied: [list any files you patched and why]
author: {AGENT_NAME}
created: {datetime.now(timezone.utc).isoformat()}
---

# Setup: {REPO_NAME}

## How Other Agents Load Pre-Cached Embeddings

```python
import numpy as np
EMBED_CACHE = "${WORKSPACE_ROOT}/AnonAPI/embeddings/{REPO_NAME}"
emb_train = np.load(f"{{EMBED_CACHE}}/train.npy")   # shape (N_train, {embeddings.shape[1]})
emb_val   = np.load(f"{{EMBED_CACHE}}/val.npy")
emb_test  = np.load(f"{{EMBED_CACHE}}/test.npy")
```

## Notes

- Embedding dim: {embeddings.shape[1]}
- Patches applied: [document any patches here]
- Gotchas: [anything unusual discovered during setup]
"""
requests.put(
    f"{API}/workspaces/{TEAM_WS_ID}/files/knowledge/setup_{REPO_NAME}.md",
    headers=HEADERS, json={"content": setup_note}
)
```

Also post a [SUGGESTION] to the workshop so other teams can reuse the cache.

---

## Step 9 — Integrate into train.py

Add as an additional feature view — stack with existing features:

```python
# Load pre-cached embeddings
EMBED_CACHE = f"${WORKSPACE_ROOT}/AnonAPI/embeddings/{REPO_NAME}"
emb_train = np.load(f"{EMBED_CACHE}/train.npy")
emb_val   = np.load(f"{EMBED_CACHE}/val.npy")

# Stack with existing features
X_train = np.hstack([X_train_existing, emb_train])
X_val   = np.hstack([X_val_existing,   emb_val])
```

---

## Common Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| `ImportError: No module named 'X'` | Missing dep | Install into existing venv or add to sys.path |
| HF download goes to `~/.cache` | `HF_HOME` not set | Export `HF_HOME` to scratch path before import |
| `np.long` / `np.float` AttributeError | numpy 1.24+ removed aliases | Replace with `np.int64` / `np.float64` |
| `AdamW` ImportError from transformers | transformers 4.x moved AdamW to torch | `from torch.optim import AdamW` |
| `modeling_bert` ModuleNotFoundError | transformers 4.x renamed modules | `from transformers.models.bert.modeling_bert import ...` |
| `weights_only` UnpicklingError | torch 2.6+ default changed | `torch.load(path, weights_only=False)` |
| `attn_implementation` AttributeError | Custom BertModel subclass misses init | Set `self.attn_implementation = 'eager'` |
| `past_key_values_length` TypeError | Custom BertEmbeddings missing new arg | Add `past_key_values_length=0` to `forward` signature |
| `self.hparams = args` AttributeError | pytorch-lightning 1.x made hparams read-only | Use `save_hyperparameters(vars(args))` |
| Disk quota exceeded | Downloads going to $HOME | Always set `HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR` to scratch |
| New venv too large | Created in $HOME or FOCUS_ROOT | Always create venvs under a shared scratch directory |

---

## Worked Example: MolBERT

MolBERT (BenevolentAI/MolBERT) requires 5 patches to work with modern deps.
This serves as the reference example for older repos.

**Setup:**
```
Repo:     ${WORKSPACE_ROOT}/AnonAPI/repos/MolBERT
Commit:   HEAD of main (no releases; pin after clone)
Weights:  ${WORKSPACE_ROOT}/AnonAPI/checkpoints/MolBERT/
          molbert_100epochs/checkpoints/last.ckpt
Source:   https://ndownloader.figshare.com/files/25611290  (~923 MB zip)
Python:   ${WORKSPACE_ROOT}/workspace/ai_scientists/.venv/bin/python
Extra dep: pytorch-lightning==1.9.5 (installed into existing venv)
```

**Patches applied** (all in the cloned repo):

1. `molbert/models/base.py` — `AdamW` import from torch; inline `PositionalEmbedding`; `save_hyperparameters` for pl 1.x; `attn_implementation` + `position_embedding_type` on `SuperPositionalBertModel`; `past_key_values_length` on `SuperPositionalBertEmbeddings.forward`
2. `molbert/tasks/tasks.py` — `BertLMPredictionHead` import path
3. `molbert/utils/featurizer/molbert_featurizer.py` — skip `load_from_checkpoint` (use direct `load_state_dict`); `weights_only=False`; `np.long` → `np.int64`; `return_dict=False` on bert call

**Usage:**
```python
import sys, os
sys.path.insert(0, "${WORKSPACE_ROOT}/AnonAPI/repos/MolBERT")
os.environ["HF_HOME"] = "${WORKSPACE_ROOT}/huggingface_cache"

from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import warnings; warnings.filterwarnings('ignore')

CKPT = "${WORKSPACE_ROOT}/AnonAPI/checkpoints/MolBERT/molbert_100epochs/checkpoints/last.ckpt"
f = MolBertFeaturizer(CKPT, device='cuda')  # or 'cpu'
embeddings, valid = f.transform(smiles_list)
# embeddings: np.ndarray shape (N, 768), valid: bool array
```

---

## References

- `system/reference/LOGGING.md` — how to log experiment results
- `system/templates/ROLE-GPU.md` — how to claim an experiment and record KEEP/DISCARD
- HuggingFace Hub docs: huggingface.co/docs/huggingface_hub/guides/download
