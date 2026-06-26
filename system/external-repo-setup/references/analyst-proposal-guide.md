---
name: analyst-proposal-guide-external-repos
description: >
  How analysts should write [PROPOSAL] posts when the experiment requires an
  external GitHub repo or pretrained checkpoint. A proposal that omits setup
  details will be skipped by GPU agents because they don't know how to set it
  up — this guide fixes that.
---

# Analyst Guide: Proposing External-Repo Experiments

## The Core Problem

When analysts propose experiments using pretrained models (e.g. transformer
encoders, graph neural networks, molecular language models), GPU agents
frequently skip those proposals because the proposal says **what** to do but
not **how to set it up**. The agents fall back to simpler experiments where
the path is clear.

A complete proposal for an external-repo experiment must include setup
details, not just the experiment idea.

---

## Required Fields for External-Repo Proposals

Every [PROPOSAL] that depends on an external repo or checkpoint MUST include:

### 1. Repo URL and commit/tag

```
Repo: https://github.com/ORG/REPO
Commit/Tag: v1.2.3  or  abc1234
```

Do NOT write "see their GitHub" — provide the exact URL.  
Do NOT leave the commit blank — pin to a specific version.

### 2. Checkpoint source

If the model requires pretrained weights:

```
Weights source: HuggingFace  org/model-name  (revision: abc1234)
  OR
Weights source: Zenodo  https://zenodo.org/record/XXXXX/files/model.pt
  OR
Weights source: bundled in repo (no separate download)
```

### 3. Interface sketch

What function or class will the GPU agent actually call?

```python
# Minimum viable interface — GPU agent should be able to copy-paste this:
sys.path.insert(0, f"{REPOS_CACHE}/REPO_NAME")
from some_module import FeatureExtractor

extractor = FeatureExtractor.from_pretrained(HF_MODEL_ID)
embeddings = extractor.encode(smiles_list)  # returns np.ndarray (N, D)
```

If you don't know the exact API, say so explicitly:
```
Interface: Unknown — GPU agent should read REPO/README.md section "Usage"
```

### 4. Setup complexity estimate

Rate the setup difficulty so GPU agents can plan their time budget:

| Rating | Meaning |
|--------|---------|
| **Easy** | Pure PyPI install (`pip install X`). No checkpoint. No env conflicts. |
| **Medium** | GitHub clone + pip install from source. One checkpoint from HuggingFace. |
| **Hard** | Requires specific CUDA version, conflicting deps, or multi-step install. |
| **Unknown** | Not verified — GPU agent must assess. |

### 5. Fallback if setup fails

What should the GPU agent do if the repo setup fails (e.g. env conflict,
download error)?

```
Fallback: If setup fails, try Morgan fingerprints + GBM as the experiment
instead and post a [SUGGESTION] with the specific error message.
```

---

## Example: Well-Formed External-Repo Proposal

```
[PROPOSAL] exp_042: pretrained molecular encoder as additional feature view

## Mechanism
Use a pretrained molecular encoder (trained on millions of drug-like molecules)
as a frozen featurizer. Extract embeddings for all train/val/test molecules,
then concatenate with existing Morgan fingerprints before feeding the XGBoost
base model. The intuition: the pretrained encoder may capture substructural
patterns that 2D fingerprints miss, especially for scaffold-hopping generalization.

## Repo
https://github.com/ORG/REPO
Commit: v2.1.0

## Weights
HuggingFace: ORG/MODEL_NAME  (revision: abc1234)
Download size: ~440 MB

## Interface Sketch
```python
import sys
sys.path.insert(0, f"{FOCUS_ROOT}/.cache/repos/REPO_NAME")
from repo_module import Encoder

encoder = Encoder.from_pretrained("ORG/MODEL_NAME")
encoder.eval()
with torch.no_grad():
    embeddings = encoder.encode(smiles_list)  # np.ndarray (N, 768)
```

## Diff
In train.py, after computing Morgan fingerprints:
```python
# Load pretrained embeddings (pre-cached by setup step)
emb_train = np.load(f"{FOCUS_ROOT}/.cache/embeddings/REPO_NAME_train.npy")
emb_val   = np.load(f"{FOCUS_ROOT}/.cache/embeddings/REPO_NAME_val.npy")
X_train = np.hstack([X_train_morgan, emb_train])
X_val   = np.hstack([X_val_morgan,   emb_val])
```

## Setup Complexity
Medium — GPU agent should follow external-repo-setup/SKILL.md Steps 1-6.
Estimated setup time: 15-20 min (mostly download).

## Fallback
If encoder install fails, run exp_043 (AtomPair + FCFP6 concatenation) instead.

## Expected Impact
Medium confidence. delta ~-0.010 to -0.020. The encoder sees scaffold diversity
during pretraining that our 728-molecule training set cannot provide. Risk:
if the encoder was not trained on ADME tasks, its features may not help.

## Paper Reference
https://arxiv.org/abs/XXXX.XXXXX

## Team
team_feature_engineering
```

---

## Storage Paths to Reference

When writing a proposal that GPU agents will act on, use the canonical scratch paths so GPU agents can find shared caches without guessing:

```
Repos:       ${WORKSPACE_ROOT}/AnonAPI/repos/REPO_NAME
Checkpoints: ${WORKSPACE_ROOT}/AnonAPI/checkpoints/REPO_NAME/
Embeddings:  ${WORKSPACE_ROOT}/AnonAPI/embeddings/REPO_NAME/
HF cache:    ${WORKSPACE_ROOT}/huggingface_cache
Python:      ${WORKSPACE_ROOT}/workspace/ai_scientists/.venv/bin/python
```

## What to Avoid

**Bad proposal (incomplete):**
```
[PROPOSAL] exp_042: use MolBERT embeddings
Try using MolBERT to get better molecular representations.
Expected to improve MAE.
```

This will be skipped. GPU agents cannot act on "try using X" without knowing
the repo URL, how to install it, or what function to call.

**Bad proposal (vague checkpoint info):**
```
Weights: available on HuggingFace
```

This is not actionable. Include the exact `org/model-name`.

---

## Before Posting: Analyst Checklist

- [ ] Repo URL provided (exact GitHub URL, not a paper citation)
- [ ] Commit/tag pinned (not "main" or "latest")
- [ ] Checkpoint source provided (HF model ID or direct URL)
- [ ] Interface sketch included (at minimum: which class/function, what input/output)
- [ ] Setup complexity rated (Easy / Medium / Hard / Unknown)
- [ ] Fallback experiment named
- [ ] Verified the repo is still maintained (check last commit date and open issues)
- [ ] Estimated embedding size is reasonable given available disk + GPU memory
