---
name: multi-agent-focus-gpu
description: GPU agent protocol — claim from team queue, train, record results
---

# GPU Agent Protocol

**STOP. Did you go through HEARTBEAT Part 0 first?** If not, go back. This file is only for agents who have been routed into Part 4 (Normal Cycle) by the Mode Selector. If the Mode Selector sent you to Part 2 (Discussion) or Part 3 (No-Team), do NOT read or execute this file — follow that branch instead.

You run experiments on a dedicated GPU. You belong to a team.

## Two rules that override everything below

1. **No team → no work.** Enforced by HEARTBEAT Part 0. If you reach this file, `MY_TEAM` is set.
2. **Every experiment MUST have a complete AnonAPI API trail:** POST [PROPOSAL] → add to queue → claim → train → write result file → release claim → POST [RESULT]. If KEEP, also PUT champion.md. This applies whether the experiment came from an analyst's queue or you self-designed it. Skip any step → invisible work → forbidden.

## CRITICAL: YAML Frontmatter Parsing

The API does NOT parse YAML frontmatter. Always parse client-side:

```python
import yaml

def parse_frontmatter(api_response):
    content = api_response.get("content", "")
    parts = content.split("---")
    if len(parts) >= 3:
        return yaml.safe_load(parts[1]) or {}
    return {}
```

## Your Cycle

### Step 0 — Find Your Team (HARD GATE)

```python
# Read roster from main workspace (parse YAML client-side)
roster_raw = requests.get(f"{API}/workspaces/{MAIN_WS_ID}/files/teams/roster.md",
                          headers=HEADERS).json()
roster = parse_frontmatter(roster_raw).get("teams", {})

MY_TEAM = TEAM_WS_ID = None
for name, t in roster.items():
    if AGENT_NAME in t.get("members", []):
        MY_TEAM = name
        TEAM_WS_ID = t["workspace_id"]
        break

if MY_TEAM is None:
    # No team assigned. Per Rule 1, exit immediately. Do NOT run experiments.
    print(f"[EXIT] {AGENT_NAME}: no team in roster ({len(roster)} teams). "
          f"Waiting for monitor to form teams.")
    import sys; sys.exit(0)
```

**Do not** wrap this in a try/except that swallows the exit and continues. The only valid response to "no team" is to exit cleanly.

### Step 1 — Check GPU Availability

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
```
If >1000 MiB used on your GPU → do analyst work instead.

### Step 1.5 — Shared-Baseline Coordination — REQUIRED

If the champion file is in `awaiting_baseline` state (no metric_value
set yet), the WHOLE SYSTEM needs exactly ONE baseline run — not one per
team. Use claim-based coordination to avoid duplicated baselines.

```python
champ_raw = requests.get(f"{API}/workspaces/{MAIN_WS_ID}/files/champion.md",
                         headers=HEADERS).json()
champ = parse_frontmatter(champ_raw)

if champ.get("status") == "awaiting_baseline":
    # Try to claim the baseline lock with If-None-Match (atomic).
    # First GPU to arrive wins the lock and runs the baseline; every
    # other GPU reads a real experiment from queue instead.
    r = requests.put(f"{API}/workspaces/{MAIN_WS_ID}/files/baseline_lock.md",
                     headers={**HEADERS, "If-None-Match": "*"},
                     json={"content": f"holder: {AGENT_NAME}\nclaimed_at: {NOW}\n"})
    if r.status_code in (200, 201):
        # We got the lock — run champion unchanged as the shared baseline,
        # then seed champion.md for everyone.
        item = {"id": "baseline_shared",
                "axis": "seed",
                "direction": "none",
                "value": 0,
                "diff": "Run champion train.py unchanged (shared baseline)",
                "infrastructure_probe": True}
    else:
        # Someone else holds the lock — skip baseline, proceed to real
        # experiment from queue. If queue is empty, wait one rotation.
        print("[BASELINE] another agent is running the shared baseline; "
              "picking a real experiment from queue instead")
        # fall through to Step 3 (queue claim)
```

**Never run baseline on a team-by-team basis.** The champion metric is
global — one run is sufficient.

### Step 2 — Read Champion Config

```python
# Read champion config from main workspace
champ_raw = requests.get(f"{API}/workspaces/{MAIN_WS_ID}/files/champion.md",
                         headers=HEADERS).json()
champ = parse_frontmatter(champ_raw)
champ_version = champ_raw.get("version", 0)  # Save for race condition check later

# Read canonical champion train.py (SINGLE SOURCE OF TRUTH)
# Located at: {FOCUS_ROOT}/champion/train.py
# Copy it AND its runtime dependencies to your workspace before making changes.
#
# Copying only train.py is the #1 first-launch failure in this role:
# `uv run python train.py` will ModuleNotFoundError on `prepare.py` (or fail
# the `pyproject.toml` lookup, or pick a wrong dependency version without
# `uv.lock`). Both gpu5 and gpu6 hit this on 2026-05-26 — the auto-recovery
# costs 30-60s per agent and burns API budget. Just copy them all.
import shutil
from pathlib import Path
workdir = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/repo")
workdir.mkdir(parents=True, exist_ok=True)
for fname in ("train.py", "prepare.py", "pyproject.toml", "uv.lock"):
    src = Path(f"{FOCUS_ROOT}/champion") / fname
    if not src.exists():
        # Fallback to task/repo/ for files that aren't champion-tracked
        # (champion only tracks files that diff per experiment).
        src = Path(f"{FOCUS_ROOT}/task/repo") / fname
    shutil.copy(src, workdir / fname)
```

**Never read train.py from another agent's workspace.** Always use `{FOCUS_ROOT}/champion/train.py`.

### Step 2a — Biomlbench Experiment Priorities — READ IF BIOMLBENCH=true

If `BIOMLBENCH=true`, read this section in full before claiming or self-designing any experiment.

#### 2a-i. Register your approach (REQUIRED at start of every cycle)

Before you claim or design an experiment, register your intended approach in the shared
approach registry so other agents don't duplicate it:

```python
import json, fcntl
from pathlib import Path

reg_path = Path(f"{FOCUS_ROOT}/logs/approach_registry.json")
MY_APPROACH = "<one-line label, e.g. 'chemprop-GNN' or 'ChemBERTa-finetune' or 'Mordred+RF'>"

# Atomic read-modify-write with file lock
with open(reg_path, "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    reg = json.load(f)
    taken = reg.get("taken", [])
    if MY_APPROACH in taken:
        print(f"APPROACH CONFLICT: '{MY_APPROACH}' already taken — pick a different paradigm")
        # Choose a different approach and repeat this check before proceeding
    else:
        taken.append(MY_APPROACH)
        reg["taken"] = taken
        f.seek(0); json.dump(reg, f, indent=2); f.truncate()
        print(f"Registered approach: {MY_APPROACH}  (registry now: {taken})")
    fcntl.flock(f, fcntl.LOCK_UN)
```

If `approach_registry.json` does not exist yet, create it:
```python
reg_path.write_text(json.dumps({"cycle": 1, "taken": [MY_APPROACH]}, indent=2))
```

**Do not proceed with training if another agent has already registered the same approach
this cycle.** Pick a different paradigm and re-register.

#### 2a-ii. Declare your compute mode — REQUIRED if GPU_AVAILABLE=True

After registering your approach and **before starting any training**, write a one-line file
declaring whether your experiment needs the GPU or can run on CPU only. The orchestrator
polls this file to decide whether to serialize or parallelize the next agent.

```python
# Determine compute mode based on your chosen approach
MY_COMPUTE = "gpu"   # set to "cpu" if your experiment is CPU-only

claim_path = Path(f"{FOCUS_ROOT}/logs/{AGENT_NAME}.gpu_claim")
claim_path.write_text(MY_COMPUTE + "\n")
print(f"[COMPUTE CLAIM] {AGENT_NAME}: {MY_COMPUTE}")
```

**Write this file as early as possible** — ideally right after approach registration, before
you read the queue or write any code. The orchestrator waits up to 120 s for this file;
if it doesn't appear, it assumes `gpu` and blocks the next agent unnecessarily.

**GPU experiments** (write `gpu`): anything that calls `torch`, `tensorflow`, or a CUDA
kernel — GNN training, transformer fine-tuning, neural network training. These serialize
because only 1 GPU is available.

**CPU experiments** (write `cpu`): classical ML (XGBoost, LightGBM, SVR, RF), Gaussian
processes, AutoML (FLAML/TPOT), offline embedding inference followed by a linear head,
any sklearn-based pipeline. These run in parallel with each other and with the active GPU
agent — they add throughput at zero GPU cost.

**Aim for a mix across the team each cycle.** If the approach registry already shows 2+
GPU approaches registered, strongly prefer a CPU experiment for your slot (and vice versa).
This keeps the GPU busy on deep models while CPU agents explore classical/ensemble methods
simultaneously.

#### 2a-iii. GPU vs CPU approach selection

Read `GPU_AVAILABLE` from your launch prompt. It determines which method classes are practical.

**If `GPU_AVAILABLE=True`** — strongly prefer GPU-native methods:

| Domain | Preferred approaches |
|--------|---------------------|
| Small-molecule ADMET | Chemprop (MPNN), PyG/DGL GNN, ChemBERTa/MolBERT, UniMol, Graph Transformer |
| Protein fitness | ESM-2 embeddings + head, MSA Transformer, Chemprop on SMILES if applicable |
| Single-cell | scVI VAE, Geneformer/scGPT, GNN on cell graph, CLIP-style multimodal |
| Pathology imaging | ViT/ResNet fine-tune, pathology FM (UNI, CONCH), nnU-Net |

You can run pretrained foundation model embeddings for feature extraction and use them as features for a classical ML model.

CPU-friendly methods (XGBoost+RDKit etc.) are fine as ONE fallback team — not as the
default for every agent. If GPU is available and the queue only has classical ML entries,
self-design a GPU-native experiment instead.

**If `GPU_AVAILABLE=False` (CPU only)** — diversify across these CPU-friendly paradigms
(do NOT all pick the same approach)

Paradigms: LightGBM/XGBoost, SVR, RF/ExtraTrees, Gaussian Process, Offline pretrained foundation model embeddings 

**Pick the paradigm your team was assigned in queue.md. If the queue is empty, pick the
highest-value unclaimed paradigm NOT in the registry.**

#### 2a-iv. Low-value experiment types to avoid

The following have low expected value and should NOT be prioritized:

1. **More HP search trials on an already-tuned model** — extra Optuna trials on the same
   architecture tend to overfit the CV split on small datasets.
2. **Fine-bracket sweeps of one regularization coefficient** — usually inside the CV noise band.
3. **More ensemble seeds on an unchanged model** — reduces variance slightly but adds nothing new.
4. **Single-parameter tuning of a model that's already had a tuning pass** — CV noise dominates.

If the queue contains only these types, self-design something that meaningfully changes the
approach. Post a [SUGGESTION] if you skip queued items so analysts can reprioritize.

**Why this matters:** biomlbench tasks span small-molecule ADMET, protein fitness, single-cell
genomics, and medical imaging — all with finite wall-clock budgets. The highest-value experiments
test a qualitatively different approach, not re-tuning a model the team has already optimized.

### Step 2b — Verify Task Specifications

**Before claiming any experiment, verify you understand the task requirements:**

```python
# Read task specification
task_spec_path = f"{FOCUS_ROOT}/task/TASK.md"
with open(task_spec_path) as f:
    task_content = f.read()

# For BioML tasks (ProteinGym, TDC), verify the fold split specification
if "fold_contiguous" in task_content:
    # Ensure you use the correct fold column as specified in TASK.md
    # Example check for ProteinGym:
    import re
    fold_match = re.search(r'fold_([a-z_]+5)', task_content)
    if fold_match:
        required_fold = fold_match.group(0)  # e.g., "fold_contiguous_5"
        print(f"TASK VERIFICATION: Using split = {required_fold}")
        # Verify your code uses this exact fold column before training
```

**Why this matters:** Task specs may specify a particular data split (e.g., `fold_contiguous_5` instead of `fold_random_5`). Using the wrong split will invalidate all results.

### Step 3 — Claim Experiment from Team Queue (REQUIRED)

**Safety: abort if a prior unposted result still sits in `result_latest.json`** (HEARTBEAT Part 0 Check C should have caught this; verify once more to prevent orphaned results):

```python
import json
from pathlib import Path
_p = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/result_latest.json")
if _p.exists():
    _pend = json.loads(_p.read_text())
    if not _pend.get("posted_to_workshop") and _pend.get("status") == "complete":
        raise RuntimeError(f"[SAFETY] unposted result for {_pend.get('exp_id')} — re-enter HEARTBEAT, go to Part 5")
```

Check your team's queue for pending experiments.

```python
queue_raw = requests.get(f"{API}/workspaces/{TEAM_WS_ID}/files/queue.md",
                         headers=HEADERS).json()
queue = parse_frontmatter(queue_raw)
pending = queue.get("pending", [])

if pending:
    # Normal path: claim from queue
    item = pending[0]
else:
    # EMPTY QUEUE — self-propose a bold experiment within your team's
    # dimension. Read your team's strategy.md, dead_ends.md, and the
    # champion code to pick the highest-value untested change. Then:
    #   1. Post a [PROPOSAL] to the workshop (full rationale + diff)
    #   2. Add it to your team's queue.md
    #   3. Claim it below
    # This maintains the full API trail while not wasting GPU time.
    # Teams are HYPOTHESIS-based, not axis-based — propose any axis as
    # long as the change is consistent with your team's hypothesis.
    # Prefer changes that are:
    #   - Bold (ambition quota: ≥10% param change, or structural variant)
    #   - Not in dead_ends.md
    #   - Grounded in champion code analysis, not speculation
    item = self_designed_item  # you create this from your analysis
```

**Every experiment must have a full API trail:** [PROPOSAL] post → queue
entry → claim → training → result file → [RESULT] post. Self-designed
experiments follow the same trail; the only difference is the GPU agent
writes the proposal instead of an analyst.

**Every queue item and [PROPOSAL] MUST include axis / direction / value
tags.** These feed the empirical-priors ranking, direction-diversity
check, and failure-range check. Claiming or self-designing an item
without these tags is forbidden — if the queue item is missing them,
reject the claim and post a [SUGGESTION] asking the analyst to fix the
queue.

**Teams are hypothesis-based, not axis-based.** You may propose any
axis as long as the change is consistent with your team's hypothesis.
If another team's proposal looks promising and shares your hypothesis's
lens, you can claim it.
# **Discussion-gate check:** if the item is `discussion_pending: true`,
# verify its [PROPOSAL] post has at least one comment from a non-author
# before claiming. A comment from the proposer themselves does not count.
# Skip items that don't yet meet this bar and pick the next one.
#
# Two auto-clear overrides prevent the gate from starving the queue
# (observed in gpt-nano-agents 2026-05-26: cycles 7-12 had GPU agents
# posting near-empty "[GPU-REVIEW] acknowledged" comments just to satisfy
# the gate, burning API budget for no information value):
#
#   1. Time-based: if the proposal was posted more than DISCUSSION_GRACE
#      ago (default 15 min), claim it anyway. The discussion window has
#      passed; agents who wanted to comment had their chance.
#   2. Queue-starvation: if THIS is the only `discussion_pending: true`
#      item remaining and there are no non-pending items either, claim
#      it. A blocked GPU is worse than a thinly-discussed proposal.
import time
DISCUSSION_GRACE_SEC = 15 * 60

if item.get("discussion_pending"):
    proposal_id = item.get("proposal_post")
    cleared = False

    # Override 1: time-based grace
    proposed_at = item.get("proposed_at") or item.get("created_at")
    if proposed_at:
        try:
            from datetime import datetime, timezone
            t0 = datetime.fromisoformat(proposed_at.replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - t0).total_seconds() > DISCUSSION_GRACE_SEC:
                cleared = True  # waited long enough
        except Exception:
            pass

    # Override 2: starvation — this is the only claimable item
    if not cleared:
        other_claimable = [it for it in (queue.get("pending") or [])
                           if it.get("id") != item["id"]
                           and not it.get("discussion_pending")]
        if not other_claimable:
            cleared = True  # rather claim discussion-pending than idle the GPU

    # Default path: require a non-author comment
    if not cleared and proposal_id:
        comments = requests.get(f"{API}/posts/{proposal_id}/comments",
                                headers=HEADERS).json().get("data", [])
        proposer = item.get("proposed_by", "")
        non_author = [c for c in comments
                      if proposer not in str(c.get("author", ""))]
        if not non_author:
            # Not yet discussed — skip to next item
            item = None  # fall through to next pending item or self-design

# Claim via read-modify-PUT with If-Match (DO NOT use PATCH — it corrupts nested YAML
# frontmatter like pending: lists. Confirmed to destroy queue.md across teams.)
queue_version = queue_raw.get("version", 0)
fm = parse_frontmatter(queue_raw)
fm.setdefault("claims", {})[AGENT_NAME] = {"exp_id": item["id"], "claimed_at": now}
body = queue_raw.get("content", "").split("---", 2)[-1]
new_content = f"---\n{yaml.safe_dump(fm, sort_keys=False)}---{body}"
# Validate round-trip before writing
assert yaml.safe_load(new_content.split("---")[1]) == fm, "frontmatter round-trip failed"
r = requests.put(f"{API}/workspaces/{TEAM_WS_ID}/files/queue.md",
    headers={**HEADERS, "If-Match": str(queue_version)},
    json={"content": new_content})
if r.status_code == 409:
    # Conflict — another agent claimed concurrently. Re-read and retry or pick a different item.
    pass
```

If queue is empty, design your own experiment. Your only constraint
is **consistency with your team's hypothesis**: the change you propose
must be one your team's hypothesis predicts will improve the metric.
Any axis is fair game. This is the triangulation value of
hypothesis-based teams — the same experiment may be proposed by
different teams for different reasons.

```python
# Discover your team's context for self-designed experiments
team_files = requests.get(f"{API}/workspaces/{TEAM_WS_ID}/files",
                          headers=HEADERS).json()["files"]
# Read strategy.md, dead_ends.md, analysis/ files from YOUR team
# Design an experiment within YOUR dimension
```

### Step 3b — Dedup Check

Before training, verify this experiment hasn't already been run AND isn't already in the code:

```python
# 1. Search workspace results for similar experiments
hits = requests.get(f"{API}/workspaces/{MAIN_WS_ID}/search?q={mechanism_keyword}",
                    headers=HEADERS).json()["results"]
# If results/ files already cover this mechanism, skip it

# 2. Search team dead ends
team_hits = requests.get(f"{API}/workspaces/{TEAM_WS_ID}/search?q={mechanism_keyword}",
                         headers=HEADERS).json()["results"]
# If your mechanism appears in dead_ends or similar analysis, skip it

# 3. Check if the mechanism already exists in champion code
champion_code = open(f"{FOCUS_ROOT}/champion/train.py").read()
if mechanism_keyword.lower() in champion_code.lower():
    print(f"ALREADY IN CODE: {mechanism_keyword} — skip this experiment")
    # Release claim and pick next experiment

# 4. **Target validation** — if your change reads or writes a named variable or
#    collection in the target code (a list of params, a config dict, a feature
#    set), verify that collection is non-empty and actually wired into the code
#    path you expect. Helper variables are sometimes defined but never referenced
#    — tuning them produces noise-only deltas that look like real signal.
#    Catch this BEFORE running, not after cascades of dead hypotheses.
#
# Example pattern:
#   target_collection = f"{group_name}_items"
#   if f"{target_collection} = []" in code:
#       print(f"DEAD TARGET: {target_collection} is empty — change would be a no-op")
#       # Release claim, skip, and post a [SUGGESTION] for code cleanup
```

### Step 3c — External Repo Setup (if experiment requires it)

If the claimed experiment depends on a GitHub repo or pretrained checkpoint
that is not already installed in `{FOCUS_ROOT}/.cache/repos/`:

1. Read `{FOCUS_ROOT}/system/external-repo-setup/SKILL.md` — it is the
   complete protocol for cloning repos, installing deps, downloading weights,
   extracting embeddings, and caching them.
2. Check whether a teammate already did the setup:
   ```python
   # Search team workspace for setup notes
   team_hits = requests.get(
       f"{API}/workspaces/{TEAM_WS_ID}/search?q=setup_{REPO_NAME}",
       headers=HEADERS
   ).json()["results"]
   ```
   If `knowledge/setup_{REPO_NAME}.md` exists, load pre-cached embeddings
   instead of re-running extraction.
3. After successful setup, write `knowledge/setup_{REPO_NAME}.md` to the
   team workspace so other GPU agents can reuse the cached embeddings.

**Time budget:** factor in 15-30 min for first-time setup when deciding
whether to run this experiment or pick a lighter one from the queue instead.

### Step 4 — Apply Change and Train

Apply ONE change from the experiment's diff, then **block synchronously** on
training. Detached / fire-and-forget training is forbidden: round 19 showed
that when the agent's claude session ends before parsing `train.stdout`, the
real metric is computed but never recorded — the entire cycle's work
vanishes. The agent MUST wait for the training subprocess and then run
Steps 5–8 in the same session.

**Before training, verify the diff actually landed.** If the Edit tool said
`old_string not found`, `patch -p1` printed `FAILED` / `Hunk #N FAILED`, or the
resulting `train.py` is byte-identical to `champion/train.py`, the proposal
was NOT tested — training would just re-measure the baseline at noise. Set
`item["diff_applied"] = False` on the sentinel BEFORE launching training, or
better, skip training entirely and post `[RESULT] {exp_id}: FAILED` so the
proposal can be re-queued with a fresh diff. Phantom KEEPs from this exact
path (gpt-nano-pubrun round on 2026-05-26: `data_v7`, `0.979985`, diff
rejected) corrupted the champion lineage — never let baseline noise be
mistaken for evidence about a change.

```python
import filecmp
diff_applied = not filecmp.cmp(
    str(rep / "train.py"),
    f"{FOCUS_ROOT}/champion/train.py",
    shallow=False,
)
item["diff_applied"] = diff_applied
if not diff_applied:
    print(f"[STEP4] diff for {exp_id} did NOT apply — train.py matches champion. "
          f"Marking FAILED and skipping training.")
    # Jump to Step 5 with outcome="FAILED", our_metric=None.
```

**Pattern A (default, foreground shell).** Use when you just want to watch
training in the current shell:

```bash
cd $FOCUS_ROOT/agents/$AGENT_NAME/workspace/repo
CUDA_VISIBLE_DEVICES=$GPU_ID \
UV_CACHE_DIR=$FOCUS_ROOT/.cache/uv HF_HOME=$FOCUS_ROOT/.cache/huggingface \
TORCH_HOME=$FOCUS_ROOT/.cache/torch \
uv run python train.py
```

**Pattern B (blocking subprocess from Python, capture stdout/stderr to
files).** Use when you want stdout/stderr persisted on disk for later
inspection. This is still SYNCHRONOUS — `subprocess.run` waits for the
training process to exit. NEVER use `subprocess.Popen` without an
immediately-following `proc.wait()`; NEVER use `nohup ... &`; NEVER exit
the agent session while training is still running.

```python
import json, os, subprocess
from pathlib import Path
from datetime import datetime, timezone

ws  = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace")
rep = ws / "repo"
out, err = ws / f"train_{exp_id}.stdout", ws / f"train_{exp_id}.stderr"

sentinel = {
    "status": "running", "posted_to_workshop": False,
    "exp_id": exp_id, "agent": AGENT_NAME, "item": item, "queue_claimed": True,
    "direction": direction, "val_score": None,
    "submission_path": str(rep / f"submission_{exp_id}.csv"),
    "train_path":      str(rep / f"train_{exp_id}.py"),
    "stdout_path": str(out), "stderr_path": str(err),
    # Record OUR pid so HEARTBEAT Part 0 Check C can tell whether we died
    # ungracefully (rate limit, OOM, SIGKILL) vs. legitimately still training.
    # `pid: None` would make _alive() return False and incorrectly route a
    # live cycle to resume-and-post.
    "pid": os.getpid(), "monitor_id": None, "description": description,
    "launched_at": datetime.now(timezone.utc).isoformat(),
}
(ws / "result_latest.json").write_text(json.dumps(sentinel, indent=2, default=str))

# BLOCK until training finishes. 20-min hard cap matches the per-experiment
# budget; raise it locally if you genuinely need longer runs (and document why).
result = subprocess.run(
    ["uv", "run", "python", "train.py"],
    cwd=str(rep),
    capture_output=True,
    text=True,
    timeout=1200,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU_ID),
         "UV_CACHE_DIR": f"{FOCUS_ROOT}/.cache/uv",
         "HF_HOME":      f"{FOCUS_ROOT}/.cache/huggingface",
         "TORCH_HOME":   f"{FOCUS_ROOT}/.cache/torch"},
)
out.write_text(result.stdout)
err.write_text(result.stderr)
training_succeeded = result.returncode == 0
sentinel["status"] = "complete" if training_succeeded else "failed"
sentinel["returncode"] = result.returncode
(ws / "result_latest.json").write_text(json.dumps(sentinel, indent=2, default=str))
# Now parse the metric from result.stdout and continue to Step 4b → Step 5
# in this same session — do NOT exit until the result is posted.
```

If your training is too long to fit in one session, split it: run a shorter
config (fewer steps, smaller batch) so the metric still flows back this
cycle. A recorded partial result beats a perfect orphaned one.

After training, save outputs to **agent-local paths** (never `task/` or `champion/`):

```python
import shutil
from pathlib import Path

agent_workspace = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/repo")

# Save a stamped copy of the submission for this experiment
agent_sub = agent_workspace / f"submission_{exp_id}.csv"
shutil.copy(agent_workspace / "submission.csv", agent_sub)

# Save a stamped copy of the train script for this experiment
agent_train = agent_workspace / f"train_{exp_id}.py"
shutil.copy(agent_workspace / "train.py", agent_train)

print(f"[ISOLATION] saved submission → {agent_sub}")
print(f"[ISOLATION] saved train     → {agent_train}")
```

**Stamped files belong here in agent-local paths.** The shared `champion/train.py` is propagated by the KEEP-winning agent in Step 7b1 (see below) — not from this step and not by the orchestrator. The stamped copy must exist before Step 7b1 can copy it.

### Step 4b — Analyze Training Dynamics — REQUIRED

After training completes, analyze the training log before recording the
result. This takes 30 seconds and produces diagnostic signals that are
more informative than the final metric alone.

Check these three things from the training output:

1. **Was the loss still decreasing when training ended?** Compare the
   loss at the final step to the loss at ~80% of training. If the loss
   is still dropping meaningfully (>1% of its total range), the model
   is **undertrained** — it would benefit from more steps. Note this in
   the result file. This signal suggests step-increasing changes
   (smaller batch, shorter sequences, faster kernels) are productive.

2. **Did the loss plateau early?** If the loss flattened before ~60%
   of training, the model has **excess capacity** for this step count.
   Note this. This signal suggests capacity can be reduced (smaller
   model) or step count decreased (larger batch) without loss.

3. **How many training steps completed?** Record `num_steps` and
   `tokens_seen` in the result file. These are the key throughput
   metrics. Any experiment that reduces steps by >10% relative to
   champion is fighting an uphill battle in a fixed-time benchmark —
   flag this prominently so analysts can factor throughput into their
   proposals.

Include these diagnostics in every result file under a `## Training
Dynamics` section. Analysts use this information in Step 1b2 (post-KEEP
inductive reasoning) to understand WHY a KEEP worked, not just that it
did.

### Step 5 — Record Result

**Before recording: re-read champion.md to handle race conditions.**

```python
# Re-read champion (may have changed during our 5-min training)
fresh_raw = requests.get(f"{API}/workspaces/{MAIN_WS_ID}/files/champion.md",
                         headers=HEADERS).json()
fresh_champ = parse_frontmatter(fresh_raw)
# Generic metric handling (supports different optimization directions)
metric_name = fresh_champ["metric_name"]  # task defines this in champion.md
direction = fresh_champ.get("direction", "minimize")  # "minimize" or "maximize"
current_best = fresh_champ.get(metric_name, float("inf") if direction == "minimize" else float("-inf"))
fresh_version = fresh_raw.get("version", 0)

race_condition = (fresh_version != champ_version)
if race_condition:
    print(f"Champion changed during training (v{champ_version} → v{fresh_version})")

# Compare against CURRENT champion, not the one we read before training.
# Use < for minimize (smaller is better), > for maximize (larger is better).
#
# IMPORTANT: a result is only meaningful if the proposed diff actually applied.
# If Step 4's edit failed (Edit tool said old_string not found, patch -p1
# rejected hunks, etc.) and you trained on the untouched champion code anyway,
# the metric you measured is baseline noise — NOT evidence about the proposal.
# Recording it as KEEP corrupts the champion (a phantom that didn't test the
# proposed change); recording it as DISCARD wrongly refutes the proposal.
# Mark FAILED so the orchestrator skips champion promotion and analysts can
# re-queue the proposal with a fresh diff.
diff_applied = bool(item.get("diff_applied", True))  # default True for legacy items

if not diff_applied:
    outcome = "FAILED"
elif (direction == "minimize" and our_metric < current_best) or \
     (direction == "maximize" and our_metric > current_best):
    outcome = "KEEP"
else:
    outcome = "DISCARD"
```

Write to **main workspace** (visible to all teams):
```python
requests.put(f"{API}/workspaces/{MAIN_WS_ID}/files/results/{item['id']}.md",
    headers=HEADERS, json={"content": result_markdown})
```

### Step 6 — Release Claim AND Move Item to Completed

Atomically do BOTH in a single read-modify-PUT: drop the claim AND move
the experiment record from `pending:` → `completed:`. Doing only the first
(the historic pattern) leaves stale rows in `pending:`, forcing the next
analyst cycle to hand-prune the queue before they can propose. Observed in
gpt-nano-agents 2026-05-26: cycles 2-4 each had analysts spending several
turns cleaning up DISCARDed-but-still-pending items.

```python
# Read-modify-PUT with If-Match (NEVER PATCH — corrupts nested pending: list).
# Missing claim or 409 is benign on resume (monitor's 30-min sweep may have cleared it).
from datetime import datetime, timezone
q_raw = requests.get(f"{API}/workspaces/{TEAM_WS_ID}/files/queue.md", headers=HEADERS).json()
q_fm  = parse_frontmatter(q_raw)

claim_removed = q_fm.get("claims", {}).pop(AGENT_NAME, None) is not None

# Move the just-finished item from pending → completed in the same write.
pending = q_fm.get("pending", []) or []
completed = q_fm.get("completed", []) or []
remaining = []
for it in pending:
    if it.get("id") == exp_id:
        it = dict(it)
        it["completed_at"] = datetime.now(timezone.utc).isoformat()
        it["completed_by"] = AGENT_NAME
        it["outcome"]      = outcome  # KEEP / DISCARD / FAILED from Step 5
        it["val_score"]    = our_metric
        completed.append(it)
    else:
        remaining.append(it)
q_fm["pending"]   = remaining
q_fm["completed"] = completed

if claim_removed or len(remaining) != len(pending):
    q_body = q_raw.get("content", "").split("---", 2)[-1]
    q_new  = f"---\n{yaml.safe_dump(q_fm, sort_keys=False)}---{q_body}"
    requests.put(f"{API}/workspaces/{TEAM_WS_ID}/files/queue.md",
        headers={**HEADERS, "If-Match": str(q_raw.get("version", 0))},
        json={"content": q_new})  # 409 OK — continue to Step 7/8
```

### Step 7 — Update Champion (KEEP only)

If result is strictly better than current champion:

**CRITICAL: Before propagating, make ALL improvements unconditional in your train.py.**
Do NOT gate changes behind `if EXPERIMENT_ID == "exp_foo"` or similar checks.
Every improvement must be baked into the code as the default behavior.
If you find gated code from previous experiments, make it unconditional too.

```python
# Bad:  if EXPERIMENT_ID == "exp_my_change": value *= factor
# Good: value *= factor  (always active)
```

#### Step 7.0 — Multi-Seed Gate — REQUIRED

**Before writing the champion file, confirm the result is not a lucky
seed.** Read the team's empirically-measured seed standard deviation
from `knowledge/noise_floor.md` (or the team's canonical location for
it). Let `sigma` be that value.

- **If `|delta| > sigma * MARGIN`** (default `MARGIN = 2`), the result
  is outside the one-seed noise band. Propagate as normal.
- **If `|delta| <= sigma * MARGIN`**, the delta is inside a band where
  a lucky seed could account for the improvement. You MUST re-run the
  same code change on a **different random seed** before promoting.
  - If the second-seed result is also strictly better than champion,
    propagate.
  - If the second-seed result is not better, classify as near-miss,
    post a `[NEAR-MISS]` instead of promoting, and leave champion
    unchanged. Do NOT overwrite champion on half-confirmed evidence.

```python
# Read empirical noise pairs (see analyst Step 0.5). If n<3, use
# conservative 0.003 band.
nf_raw = requests.get(
    f"{API}/workspaces/{MAIN_WS_ID}/files/knowledge/noise_floor_data.md",
    headers=HEADERS).json()
pairs = parse_pairs(nf_raw.get("content", ""))
if len(pairs) >= 3:
    sigma = pooled_std(pairs)
    noise_floor = sigma
else:
    noise_floor = 0.0015  # conservative, implies 2σ band ≈ 0.003

MARGIN = 2.0
if abs(delta) > noise_floor * MARGIN:
    promote = True
else:
    # Borderline — re-run on a different seed before promoting, AND
    # append the (metric_a, metric_b, code_hash) triple to
    # knowledge/noise_floor_data.md so future runs get better σ for
    # free. This is the lazy-calibration source.
    second_seed_metric = run_training_on_fresh_seed(code=our_code)
    _append_noise_pair(MAIN_WS_ID, our_metric, second_seed_metric, code_hash=sha1_of_train_py)
    promote = (second_seed_metric is strictly better than current_best)
```

The append is REQUIRED on every second-seed invocation. Without it the
noise floor never accumulates and the conservative default (0.003)
stays active forever, which blocks real small-delta KEEPs long-term.

**3-seed confirmation for persistent NEAR-MISSes.** If the same
(axis, direction, value) has already produced 2 NEAR-MISS results
with a consistent pattern (same seed beats champion, other seed
doesn't), do NOT discard the result as noise. Instead, launch a
**third** seed on the same code. Promote only if ≥2 of the 3 seeds
beat champion. This resolves the case where a real sub-noise signal
keeps showing up but can never clear the 2-seed gate.

Look up prior same-tuple NEAR-MISSes in `knowledge/near_miss_ledger.md`
before classifying. If current run is the 3rd attempt on the same
tuple, run seed 3 immediately; don't require another full claim cycle.

**Why this exists:** the champion file is the baseline every subsequent
experiment is measured against. A single-seed lucky draw that slips
into champion corrupts every downstream comparison — every "it's
better by +0.0005" judgment is made relative to a fictional baseline.
One measurement showed the seed-variance band was 4-6× larger than
the previously-assumed noise floor, which means several past champion
updates may have been artifacts. A multi-seed confirmation gate
prevents this from continuing to accumulate.

**If `knowledge/noise_floor.md` doesn't exist yet**, your team has not
measured seed variance. Before promoting anything near noise, post a
`[SUGGESTION]` requesting a seed-variance infrastructure probe (or run
it yourself as a dormant-team activity), then apply this gate once a
measurement exists. Until then, treat any delta smaller than the
prior-champion-delta as "not confirmed" and do not promote.

#### Step 7a — Extract Reproduction Information

```python
import re, json

# 1. Read YOUR train.py docstring (experiment description)
with open(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/repo/train.py") as f:
    code = f.read()
docstring_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
experiment_description = docstring_match.group(1).strip() if docstring_match else "No description provided"

# 2. Parse JSON hyperparameters from training stdout
# train.py prints JSON between === markers. Extract it:
json_match = re.search(r'=+\n({.*?})\n=+', training_stdout, re.DOTALL)
hyperparameters_json = json.loads(json_match.group(1)) if json_match else {}
```

#### Step 7b — Build Complete champion.md

**champion.md must be a complete standalone reproduction recipe.** Include ALL information needed to reproduce without reading train.py.

**Recorded `metric_value` is the BEST seed observed.** When a multi-seed
gate fired in Step 7.0, two measurements of the same code exist
(`our_metric` from the proposal-default seed and `second_seed_metric`
from the confirmation seed). Use the optimization-direction-best of
the two so subsequent agents diff against the strongest evidence
this code can produce, not against the worse draw. The other seed's
value is preserved in `knowledge/noise_floor_data.md` as a
reproducibility ledger and contributes to σ.

```python
# Choose the BEST seed value across all multi-seed runs of this code.
# direction="minimize" → use min; direction="maximize" → use max.
seed_metrics = [our_metric] + ([second_seed_metric] if 'second_seed_metric' in dir() else [])
champion_metric = min(seed_metrics) if direction == "minimize" else max(seed_metrics)

# Read current champion version for If-Match
current_raw = requests.get(f"{API}/workspaces/{MAIN_WS_ID}/files/champion.md", headers=HEADERS).json()
current_version = current_raw.get("version", 0)

champion_content = f"""---
metric_name: {metric_name}
metric_value: {champion_metric}
seed_values: {seed_metrics}
direction: {direction}
experiment_id: {exp_id}
agent: {AGENT_NAME}
timestamp: {datetime.now(timezone.utc).isoformat()}
---

# Champion: {exp_id}

## Experiment Description

{experiment_description}

## Result

- **Recorded metric (best of {len(seed_metrics)} seeds):** {metric_name} = {champion_metric}
- **All seed values:** {seed_metrics}
- **Delta from previous:** {delta:+.6f}

## Complete Hyperparameters

```json
{json.dumps(hyperparameters_json, indent=2)}
```

## Reproduction

1. Copy `{FOCUS_ROOT}/champion/train.py`
2. Run: `CUDA_VISIBLE_DEVICES=0 uv run python train.py`
3. Expected: {metric_name} ∈ {seed_metrics} (recorded best = {champion_metric})

## Provenance

- Agent: {AGENT_NAME}
- Timestamp: {datetime.now(timezone.utc).isoformat()}
- Source: {FOCUS_ROOT}/champion/train.py
"""

requests.put(f"{API}/workspaces/{MAIN_WS_ID}/files/champion.md",
    headers={**HEADERS, "If-Match": str(current_version)},
    json={"content": champion_content})
```

#### Step 7b1 — Propagate champion/train.py — REQUIRED on KEEP

**Immediately after the champion.md PUT succeeds, you MUST copy your stamped train file
to `{FOCUS_ROOT}/champion/train.py` and append a SOURCE line.** The local champion file
is read by every subsequent rotation's GPU agent in Step 2 — leaving it stale corrupts
every downstream baseline. This step is the agent's responsibility, not the orchestrator's.

```python
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Atomic write: temp-then-rename so concurrent KEEPs cannot half-overwrite.
src = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/repo/train_{exp_id}.py")
dst = Path(f"{FOCUS_ROOT}/champion/train.py")
tmp = dst.with_suffix(".py.tmp")
shutil.copy(src, tmp)
tmp.replace(dst)  # atomic on POSIX

# Append provenance to champion/SOURCE (one line per promotion).
src_log = Path(f"{FOCUS_ROOT}/champion/SOURCE")
ts = datetime.now(timezone.utc).isoformat()
with src_log.open("a") as f:
    f.write(f"{exp_id} {our_metric:.6f} {AGENT_NAME} {ts}\n")
```

**Race-safety:** if multiple GPU agents land KEEPs in the same rotation, the champion.md
PUT serializes them via If-Match — only one wins. The losing agent's `our_metric < current_best`
check at the top of Step 7 will already have failed (champion changed during their training),
so they will not enter Step 7b1. The winning agent's `tmp.replace(dst)` is atomic.

**Why this exists:** the champion file is the baseline every subsequent experiment's
diff is applied against. A 4-KEEP-deep stale champion file means agents who don't know
to read the latest stamped train file in the winning workspace will silently regress
the codebase. This step replaces the prior "orchestrator promotes" model that was
unreliable in practice.

#### Step 7c — Write result_latest.json (agent-local sentinel)

`result_latest.json` is your post-training state record — it lets HEARTBEAT Part 0
resume an unposted result on the next session and is read by analysts who want to
know your last outcome. Champion propagation already happened in Step 7b1; this file
is purely a sentinel.

```python
import json
from pathlib import Path

# Merge with any Step 4 in-flight sentinel to preserve stdout_path/launched_at.
rl = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace") / "result_latest.json"
prior = json.loads(rl.read_text()) if rl.exists() else {}
rep = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/repo")

rl.write_text(json.dumps({**prior,
    "val_score": our_metric, "direction": direction,
    "exp_id": exp_id, "agent": AGENT_NAME,
    "submission_path": str(rep / f"submission_{exp_id}.csv"),
    "train_path":      str(rep / f"train_{exp_id}.py"),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    # Resume fields — HEARTBEAT Part 0 Check C reads these. REQUIRED.
    "status": "complete", "posted_to_workshop": False, "result_post_id": None,
    "item": prior.get("item") or item,
    "queue_claimed": prior.get("queue_claimed", True),
    "description": description,
}, indent=2, default=str))
```

**If DISCARD:** write the result to `dead_ends.md` in your team workspace so analysts and other
GPU agents skip this mechanism family. Use If-Match to avoid clobbering concurrent writes.

```python
if outcome == "DISCARD":
    de_raw = requests.get(f"{API}/workspaces/{TEAM_WS_ID}/files/dead_ends.md",
                          headers=HEADERS).json()
    de_content = de_raw.get("content", "# Dead Ends\n\n")
    de_version = de_raw.get("version", 0)

    # Structured entry — REQUIRED. Future proposals check whether their
    # (axis, direction, value) falls inside a recorded DISCARD range.
    # Unstructured free-text entries defeat the failure-range check and
    # are not permitted.
    axis = item.get("axis") or "UNKNOWN"
    direction = item.get("direction") or "UNKNOWN"
    value = item.get("value")
    fam = "_".join(exp_id.split("_")[:2])
    entry = (
        f"\n- exp_id: {exp_id}\n"
        f"  axis: {axis}\n"
        f"  direction: {direction}\n"
        f"  value: {value}\n"
        f"  delta: {delta:+.6f}\n"
        f"  family: {fam}\n"
        f"  date: {datetime.now(timezone.utc).date()}\n"
        f"  reason: {experiment_description[:160].replace(chr(10), ' ')}\n"
    )

    r = requests.put(f"{API}/workspaces/{TEAM_WS_ID}/files/dead_ends.md",
                     headers={**HEADERS, "If-Match": str(de_version)},
                     json={"content": de_content + entry})
    if r.status_code == 409:
        print("dead_ends.md conflict — skipping write (analyst will update next cycle)")
    else:
        print(f"Recorded DISCARD in dead_ends.md (HTTP {r.status_code})")
```

### Step 8 — Post Result to Workshop (MANDATORY)

**This step is required for EVERY experiment, KEEP or DISCARD.** A result file in the workspace is not enough — the workshop post is what notifies analysts and other teams. Skipping this step makes the experiment invisible to the rest of the system.

Post as a NEW workshop post (not a comment on the kickoff thread):

```python
r = requests.post(f"{API}/posts", headers=HEADERS, json={
    "workshop": WORKSHOP,
    "title": f"[RESULT] {item['id']}: {metric_name}={our_metric} ({outcome})",
    "content": f"## Experiment\n{description}\n\n## Result\n{metric_name}: {our_metric}\nDelta: {delta}\nOutcome: {outcome}\nRace condition: {race_condition}\n\n## Team\n{MY_TEAM}",
    "notify_agents": team_members,
    "tags": [f"team:{MY_TEAM}", "type:result", f"outcome:{outcome}"]
})
result_post_id = r.json().get("id") if r.ok else None
```

### Step 8b — Mark result as posted (REQUIRED — prevents duplicate [RESULT] next cycle)

```python
from datetime import datetime, timezone
rl_path = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace") / "result_latest.json"
rl = json.loads(rl_path.read_text())
rl.update({"status": "posted", "posted_to_workshop": True,
           "result_post_id": result_post_id,
           "posted_at": datetime.now(timezone.utc).isoformat()})
rl_path.write_text(json.dumps(rl, indent=2, default=str))
```

### Step 9 — Near-Miss Protocol

A "near-miss" only makes sense as a **signal-carrying DISCARD** — a result
close enough to champion that the underlying mechanism may still be
productive. It must be anchored to the **team's noise floor** (see the
analyst Step 1a noise-floor rule), not a fixed global delta threshold:

- **Delta inside the noise band:** NOT a near-miss. It's noise. Do NOT
  post a [NEAR-MISS] and do NOT trigger a cross-team follow-up. If it's
  the only point on its axis, leave the axis open; if it's part of a
  bracketed minimum already above the noise band, the axis is closed.
- **Delta clearly above the noise band but within a small multiple of
  it:** legitimate near-miss. Post a [NEAR-MISS] and let analysts apply
  the Step 1a far / opposite / 2-point rule before any follow-up.

```python
noise_floor = ...  # team's current estimate from knowledge/noise_floor.md
if (delta > noise_floor) and (delta < noise_floor * SMALL_MULTIPLE):
    requests.post(f"{API}/posts", headers=HEADERS, json={
        "workshop": WORKSHOP,
        "title": f"[NEAR-MISS] {item['id']}: delta=+{delta}",
        "content": f"Near-miss. Team: {MY_TEAM}. Description: {description}. "
                   f"Delta: {delta} (noise floor {noise_floor}).",
        "notify_agents": all_agent_names,
        "tags": ["type:near-miss"],
    })
```

**Do not** simultaneously label a result as a near-miss AND register it
as an axis-exhaustion trigger — if it qualifies for the former it is
signal, if it qualifies for the latter it is too high above the noise
floor for any refinement to reach KEEP and the axis is closed.

### Step 10 — Run Second Experiment

Go back to Step 2 for a second experiment before finishing your session.

