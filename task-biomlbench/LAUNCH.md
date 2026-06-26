# task-profile.md — biomlbench

This profile fills in the hooks from `runbook.md` for **fixed-deadline benchmark** tasks (`task_type: biomlbench`) — Kaggle-style tasks with a hard wall-clock limit and a mandatory `submission.csv` output.

**Key shape:**
- 1 GPU (or none for CPU-only tasks). GPU agents run sequentially on GPU; CPU experiments run fully in parallel.
- Wall-clock deadline (8 h CPU / 16 h GPU). Loop exits when the deadline is near.
- Champion = best `submission.csv` score, not `train.py` provenance.
- Mandatory dimension discussion with a domain-aware approach menu (forces method diversity).
- Meta-improvement is **disabled** (no editing of role templates during a benchmark run).
- Every cycle checks time remaining and triggers an emergency submission save if close to deadline.

**Loop exit reasons:**
1. Deadline is within `DEADLINE_BUFFER_MINUTES` (default 15) and `submission.csv` exists → stop.
2. Deadline is within `DEADLINE_BUFFER_MINUTES` and `submission.csv` does NOT exist → emergency submission, then stop.
3. User Ctrl+C.

---

## Hook: launch_command

```bash
cd THIS_DIR
python3 launch.py <run-name> --task task-biomlbench/<task-folder>
# e.g.:
python3 launch.py my-run --task task-biomlbench/kaggle-histopathologic-cancer-detection
```

`launch.py` reads `task_type: biomlbench` from `task/TASK.md` frontmatter and copies this file as `task-profile.md` into the new run directory.

---

## Hook: bootstrap_extras

Set up the deadline clock and GPU/CPU detection — these variables are referenced by every later hook.

```python
import subprocess
from datetime import timedelta

# ── Compute constraints from TASK.md ─────────────────────────────────────
# GPU tasks get more time; CPU-only gets 8 h.
DEADLINE_BUFFER_MINUTES = 15

def _gpu_is_available():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return any(l.strip() for l in r.stdout.splitlines())
    except Exception:
        return False

GPU_AVAILABLE      = _gpu_is_available()
WALL_CLOCK_HOURS   = 16 if ("A100" in task_md or GPU_AVAILABLE) else 8
_task_says_cpu_only = 'CUDA_VISIBLE_DEVICES=""' in task_md or "CPU-only" in task_md

# IS_CPU_ONLY is True only when TASK.md mandates it AND no GPU is available.
# If a GPU is present we always enable it — neural approaches require GPU to be
# practical within the time budget, and running CPU-only produces method monoculture.
if _task_says_cpu_only and not GPU_AVAILABLE:
    IS_CPU_ONLY  = True
    CUDA_SETTING = '""'
elif GPU_AVAILABLE:
    IS_CPU_ONLY  = False
    CUDA_SETTING = "0"
else:
    IS_CPU_ONLY  = True
    CUDA_SETTING = '""'

print(f"GPU_AVAILABLE={GPU_AVAILABLE}  IS_CPU_ONLY={IS_CPU_ONLY}  CUDA_SETTING={CUDA_SETTING}")

# ── Deadline clock (persist launch time so resumes share the same deadline) ─
launch_ts_file = FOCUS_ROOT / "logs" / "launch_timestamp.txt"
if launch_ts_file.exists():
    LAUNCH_TIME = datetime.fromisoformat(launch_ts_file.read_text().strip())
    print(f"Resuming. Original launch: {LAUNCH_TIME.isoformat()}")
else:
    LAUNCH_TIME = datetime.now(timezone.utc)
    launch_ts_file.parent.mkdir(parents=True, exist_ok=True)
    launch_ts_file.write_text(LAUNCH_TIME.isoformat())
    print(f"Fresh start. Launch time: {LAUNCH_TIME.isoformat()}")

_launch_utc = LAUNCH_TIME.replace(tzinfo=timezone.utc) if LAUNCH_TIME.tzinfo is None else LAUNCH_TIME
DEADLINE = _launch_utc + timedelta(hours=WALL_CLOCK_HOURS)

def time_remaining_minutes():
    return max(0, (DEADLINE - datetime.now(timezone.utc)).total_seconds() / 60)

def deadline_status():
    return f"{time_remaining_minutes():.0f} min remaining (deadline {DEADLINE.isoformat()})"

print(f"Deadline: {DEADLINE.isoformat()} — {deadline_status()}")
```

When reading `task/TASK.md` in Step 2, also identify from it:
- **Metric** (higher or lower is better)
- **Submission format** (`submission.csv` columns and encoding)
- **Data location** (`task/data/`)

---

## Hook: discussion_policy

**MANDATORY.** Method diversity is the single most important factor in benchmark performance — without explicit guidance, agents converge on the same baseline (e.g. RDKit+XGBoost for any molecular task). Discussion must run before team formation, with a domain-aware approach menu seeded into each agent's prompt.

Build the approach menu from `task_md`:

```python
def _domain_approach_menu(task_md, gpu_available):
    """Return a multi-line string listing diverse method paradigms for this task domain."""
    is_protein = any(k in task_md for k in [
        "ProteinGym", "DMS", "deep mutational", "deep mutational scan",
        "variant effect", "fitness landscape", "protein fitness",
        "amino acid", "mutation", "substitution", "UniProt", "Uniprot",
        "MSA", "multiple sequence alignment"])
    is_mol = any(k in task_md for k in [
        "SMILES", "RDKit", "fingerprint", "ADME", "lipophilicity",
        "clearance", "solubility", "PKIS"])
    is_image = any(k in task_md for k in [
        "image", "histopathologic", "tumor", "MRI", "segmentation"])
    is_sc = any(k in task_md for k in [
        "single-cell", "single cell", "scRNA", "CITE-seq", "CITE seq",
        "multimodal", "cross-modality", "cross modality",
        "RNA->protein", "protein abundance", "perturbation",
        "label_projection", "label projection",
        "gene expression", "cell line", "anndata", "h5ad", "BMMC", "PBMC"])

    if is_protein:
        gpu_methods = (
            "GPU-ENABLED methods (preferred if GPU_AVAILABLE):\n"
            "  - ESM2 / ESM-1v / ProtBert / ProtT5 (pretrained protein language model "
            "embeddings + supervised head, or zero-shot likelihood scoring)\n"
            "  - MSA Transformer / EVE / DeepSequence (alignment-based generative models)\n"
            "  - Tranception / ProGen / RITA (autoregressive protein LMs)\n"
            "  - SaProt / Ankh / ProtGPT2 (structure-aware or sequence-only transformers)\n"
            "  - Fine-tuned ESM with LoRA or supervised head for fitness regression\n"
            "  - Equivariant structure-aware GNN on AlphaFold predicted structure\n"
            "  - Kermut / ProteinNPT (non-parametric transformers)\n"
        ) if gpu_available else ""
        cpu_methods = (
            "CPU methods (diversify, do NOT all pick the same approach):\n"
            "  - One-hot sequence features + Ridge / Lasso / LightGBM\n"
            "  - Precomputed ESM / ProtT5 embeddings (offline) + Ridge / SVR / KNN / GBM\n"
            "  - Position-specific scoring matrix (PSSM) from MSA as feature\n"
            "  - Independent site model (per-position additive effects)\n"
            "  - EVE / DeepSequence zero-shot scores as features or direct predictor\n"
            "  - k-mer / substring features + tree ensemble\n"
            "  - Gaussian Process on sequence-similarity (Hamming / BLOSUM) kernel\n"
            "  - Stacking ensemble combining zero-shot scores and supervised predictors\n"
        )
        return gpu_methods + cpu_methods
    elif is_mol:
        gpu_methods = (
            "GPU-ENABLED methods (preferred if GPU_AVAILABLE):\n"
            "  - Chemprop (directed message-passing GNN)\n"
            "  - PyG/DGL GCN or GAT on molecular graphs\n"
            "  - ChemBERTa / MolBERT / MolFormer SMILES transformer (pretrained)\n"
            "  - UniMol (3D molecular transformer, pretrained)\n"
            "  - Graph Transformer (GT) with bond/atom features\n"
            "  - Equivariant GNN (e.g. SchNet, DimeNet) if 3D coords available\n"
        ) if gpu_available else ""
        cpu_methods = (
            "CPU methods (diversify, do NOT all pick the same approach):\n"
            "  - RDKit descriptors + LightGBM\n"
            "  - Morgan fingerprints + SVR (kernel SVM)\n"
            "  - Mordred descriptors + RF or ExtraTrees\n"
            "  - Tanimoto kernel GP (Gaussian Process on fingerprint similarity)\n"
            "  - MACCS + physchem + stacking ensemble (heterogeneous features)\n"
            "  - AutoML (FLAML/TPOT) over combined descriptor sets\n"
            "  - RDKit 3D descriptors (PMAPPER, shape, pharmacophore) if 3D available\n"
        )
        return gpu_methods + cpu_methods
    elif is_image:
        gpu_methods = (
            "GPU-ENABLED image methods (preferred if GPU_AVAILABLE):\n"
            "  - Fine-tuned ResNet/EfficientNet/ViT backbone\n"
            "  - Pathology foundation models\n"
            "  - DINO/MAE self-supervised pretraining + linear probe\n"
            "  - Ensemble of CNNs with TTA\n"
            "  - U-Net / nnU-Net for segmentation tasks\n"
        ) if gpu_available else ""
        cpu_methods = "CPU methods: histogram features + SVM, HOG + RF, handcrafted texture.\n"
        return gpu_methods + cpu_methods
    elif is_sc:
        principles = (
            "METHODOLOGICAL PRINCIPLES (apply to any single-cell task):\n"
            "  - Compute a trivial baseline (constant / mean / random) before any model.\n"
            "  - Identify what is being predicted and which structural axis the test holds out\n"
            "    (batch, donor, cell type, compound, spatial coordinate). Match CV to that axis.\n"
        )
        gpu_methods = (
            "GPU-ENABLED methods (available because GPU_AVAILABLE=True):\n"
            "  - scVI / totalVI (VAE) latent embeddings + linear or KNN readout\n"
            "  - Geneformer / scGPT / UCE (pretrained transformer over gene expression)\n"
            "  - Deep MLP with batch-conditional layers or gradient-reversal domain adaptation\n"
            "  - CLIP-style contrastive between paired modalities\n"
            "  - GNN on cell-cell k-NN or feature-feature graphs (PyG / DGL)\n"
            "  - scANVI / scArches for reference-based label transfer\n"
        ) if gpu_available else ""
        cpu_methods = (
            "CPU methods (diversify, do NOT all pick the same approach):\n"
            "  - Per-target Ridge / Lasso on gene features (scikit-learn)\n"
            "  - KNN regressor / classifier on TruncatedSVD or PCA embeddings\n"
            "  - LightGBM / XGBoost on dimensionality-reduced or per-stratum features\n"
            "  - Harmony / ComBat-seq / fastMNN batch correction + downstream head\n"
            "  - Statistical / unsupervised scoring using prior-knowledge resources\n"
            "    (ligand-receptor databases, reference atlases, spatial autocorrelation)\n"
            "  - Moran's I / SpatialDE / SPARK for spatial pattern scoring\n"
            "  - SingleR / SCimilarity for reference-atlas label lookup\n"
            "  - Stacking ensemble across heterogeneous features and model families\n"
        )
        return principles + gpu_methods + cpu_methods
    else:
        return (
            "Consider a range of approaches: classical ML (XGBoost, SVR), "
            "neural networks (MLP, CNN, Transformer), pretrained embeddings, ensembles.\n"
        )

approach_menu = _domain_approach_menu(task_md, GPU_AVAILABLE)
```

**`extra_discussion_instructions` (appended to each agent's prompt):**

```python
extra_discussion_instructions = (
    f"GPU_AVAILABLE={GPU_AVAILABLE}\n"
    f"\n"
    f"DIVERSITY INSTRUCTION: Before posting, read all existing [DISCUSSION] posts.\n"
    f"You must propose a model family or approach that NO other agent has already\n"
    f"claimed. If every obvious approach is already covered, find a gap — a different\n"
    f"featurisation, a different model class, or a fundamentally different framing.\n"
    f"Do NOT propose incremental HP tuning of whatever the simplest baseline is.\n"
    f"Your [DISCUSSION] post must include at least 2 concrete experiment proposals\n"
    f"within your chosen paradigm and explain why it is worth exploring for this task.\n"
    f"\n"
    f"APPROACH MENU (domain-specific options for this task; you can also propose others):\n"
    f"{approach_menu}\n"
    f"Pick ONE paradigm from the menu (or invent a novel one not listed).\n"
    f"Avoid claiming a paradigm already taken in existing [DISCUSSION] posts."
)
```

---

## Hook: seeding_policy

**Monitor-seeded with diversity instructions.** The monitor agent reads all `[DISCUSSION]` posts, forms teams, and seeds each team's `queue.md` with a **distinct starting approach** — so GPU agents have differentiated work on cycle 1 rather than converging on the same baseline. The orchestrator then *verifies* every queue has a seed and writes a fallback if any team was missed.

**`extra_monitor_instructions`:**

```python
extra_monitor_instructions = (
    f"GPU_AVAILABLE={GPU_AVAILABLE}\n"
    f"IS_CPU_ONLY={IS_CPU_ONLY}\n"
    f"\n"
    f"DIVERSITY INSTRUCTION: After forming teams, seed each team's queue.md with a\n"
    f"DIFFERENT starting experiment. Read all [DISCUSSION] posts to identify distinct\n"
    f"approaches, then assign one per team such that no two teams start from the same\n"
    f"model family or featurisation strategy.\n"
    f"\n"
    f"Rules for seeding:\n"
    f"  - Each team gets exactly one seed experiment, priority: high.\n"
    f"  - No two teams should share the same axis (model family / approach).\n"
    f"  - Prefer approaches explicitly proposed in [DISCUSSION] posts.\n"
    f"  - CRITICAL: Exactly ONE team gets the classical ML baseline (e.g. RDKit+XGBoost).\n"
    f"    All other teams MUST be seeded with qualitatively different paradigms.\n"
    f"    If GPU_AVAILABLE={GPU_AVAILABLE}, at least half of non-baseline teams should\n"
    f"    be seeded with GPU-native methods (GNNs, transformers, pretrained embeddings).\n"
    f"  - Order: give the simplest/fastest approach to the team most likely to produce\n"
    f"    a submission.csv quickly (ensures a fallback exists early).\n"
    f"  - Include in each queue entry: axis, a self-contained diff describing exactly\n"
    f"    what to implement, and a note that the team owns this approach and should\n"
    f"    iterate within this paradigm before proposing a switch.\n"
    f"  - Also POST a [PROPOSAL] to the workshop for each seed so all agents can see\n"
    f"    the full diversity plan.\n"
    f"  - After seeding, write {{FOCUS_ROOT}}/logs/approach_registry.json with the\n"
    f"    mapping {{\"cycle\": 0, \"taken\": [list of approach names assigned]}} so the\n"
    f"    orchestrator can show agents which paradigms are already covered."
)
```

After the monitor finishes, verify and fallback-seed:

```python
for team_name, team_info in teams.items():
    team_ws_id = team_info["workspace_id"]
    q_raw = requests.get(f"{API}/workspaces/{team_ws_id}/files/queue.md",
                         headers=HEADERS).json()
    if not parse_fm(q_raw).get("pending"):
        print(f"WARNING: {team_name} queue empty after monitor — writing fallback seed")
        # Read discussion posts, pick an unclaimed approach, PUT a seed to this team's queue.md
```

---

## Hook: pre_cycle_check

Mandatory deadline check at the top of every cycle. If time is short and no `submission.csv` exists, trigger an emergency submission save.

```python
def pre_cycle_check():
    """Return True if the loop should exit after this cycle."""
    rem = time_remaining_minutes()
    submission_exists = (FOCUS_ROOT / "task" / "submission.csv").exists()
    agent_submissions = list(FOCUS_ROOT.glob("agents/*/workspace/repo/submission.csv"))

    print(f"\n{'='*60}\nDEADLINE CHECK: {deadline_status()}")
    print(f"submission.csv exists: {submission_exists or bool(agent_submissions)}\n{'='*60}\n")

    if rem <= DEADLINE_BUFFER_MINUTES:
        if not submission_exists and not agent_submissions:
            print("EMERGENCY: deadline approaching with no submission.csv — emergency save")
            trigger_emergency_submission()
        else:
            print("Deadline approaching — final submission saved. Stopping.")
        return True

    if rem <= DEADLINE_BUFFER_MINUTES * 2:
        print(f"WARNING: {rem:.0f} min remaining — prioritise submission.csv this cycle")

    return False


def trigger_emergency_submission():
    """Launch one agent with explicit instruction to write submission.csv immediately.

    IMPORTANT: Do NOT route through Part 0 (Mode Selector) — in an emergency there may
    be no teams, the roster may be empty, or monitor may not have completed. Part 0 would
    route the agent to Part 3 (No-Team exit) and produce nothing. Give direct imperative
    instructions that bypass normal heartbeat routing.
    """
    agent_name = f"{PREFIX}_gpu1"

    existing_subs = list(FOCUS_ROOT.glob("agents/*/workspace/repo/submission.csv"))
    existing_sub_hint = (
        f"An existing submission.csv was found at: {existing_subs[0]}\n"
        f"  If you cannot produce a better one in time, copy it to "
        f"{FOCUS_ROOT}/task/submission.csv as a fallback.\n"
    ) if existing_subs else ""

    existing_trains = list(FOCUS_ROOT.glob("agents/*/workspace/repo/train.py"))
    existing_train_hint = (
        f"An existing train.py was found at: {existing_trains[0]}\n"
        f"  Copy it: cp {existing_trains[0]} "
        f"{FOCUS_ROOT}/agents/{agent_name}/workspace/repo/train.py\n"
    ) if existing_trains else ""

    Agent(
        description=f"EMERGENCY submission save — {deadline_status()}",
        prompt=(
            f"You are {agent_name}. This is an EMERGENCY session.\n"
            f"FOCUS_ROOT={FOCUS_ROOT}\n"
            f"CUDA_VISIBLE_DEVICES={CUDA_SETTING}\n"
            f"BIOMLBENCH=true\n"
            f"TIME_REMAINING_MINUTES={time_remaining_minutes():.0f}\n"
            f"DEADLINE_BUFFER_MINUTES={DEADLINE_BUFFER_MINUTES}\n"
            f"\n"
            f"DO NOT read HEARTBEAT.md or follow normal agent protocol.\n"
            f"DO NOT check roster, teams, or queue.\n"
            f"DO NOT run a second experiment.\n"
            f"DO NOT post to the workshop.\n"
            f"\n"
            f"YOUR ONLY JOB:\n"
            f"STEP 1: import json; creds = json.load(open('{FOCUS_ROOT}/agents/{agent_name}/credentials.json'))\n"
            f"STEP 2: Get a working train.py.\n"
            f"  {existing_train_hint}"
            f"  If none exists: read {FOCUS_ROOT}/task/TASK.md and write the simplest\n"
            f"  possible model that produces a valid submission.csv. Mean/zero predictions\n"
            f"  are acceptable in an emergency.\n"
            f"STEP 3: cd {FOCUS_ROOT}/agents/{agent_name}/workspace/repo\n"
            f"        CUDA_VISIBLE_DEVICES={CUDA_SETTING} python train.py\n"
            f"STEP 4: import shutil; shutil.copy(\n"
            f"          '{FOCUS_ROOT}/agents/{agent_name}/workspace/repo/submission.csv',\n"
            f"          '{FOCUS_ROOT}/task/submission.csv')\n"
            f"{existing_sub_hint}"
            f"STEP 5: Exit. Print: <promise>{agent_name} emergency submission complete</promise>\n"
        ),
    )
```

---

## Hook: analyst_prompt_extras

Append deadline awareness and approach diversity to every analyst prompt:

```python
def _approach_diversity_note():
    reg_path = FOCUS_ROOT / "logs" / "approach_registry.json"
    if not reg_path.exists():
        return ""
    try:
        taken = json.loads(reg_path.read_text()).get("taken", [])
    except Exception:
        return ""
    if not taken:
        return ""
    return (
        "APPROACH DIVERSITY: The following model families / featurizations are already\n"
        "running or completed this cycle — DO NOT duplicate them:\n"
        + "\n".join(f"  - {a}" for a in taken)
        + "\nClaim or self-design something from a DIFFERENT paradigm instead.\n"
    )

rem = time_remaining_minutes()
analyst_prompt_extras = (
    f"BIOMLBENCH=true\n"
    f"GPU_AVAILABLE={GPU_AVAILABLE}\n"
    f"TIME_REMAINING_MINUTES={rem:.0f}\n"
    f"DEADLINE_BUFFER_MINUTES={DEADLINE_BUFFER_MINUTES}\n"
    f"IMPORTANT: This is a fixed-deadline benchmark. Prioritise experiments that\n"
    f"can complete within the remaining time. If TIME_REMAINING_MINUTES < "
    f"{DEADLINE_BUFFER_MINUTES*3}, do NOT propose experiments longer than 20 minutes —\n"
    f"propose only fast hyperparameter changes or inference-only improvements.\n"
    f"{_approach_diversity_note()}"
)
```

---

## Hook: gpu_dispatch

Two dispatch modes depending on hardware:

### CPU-only tasks (`IS_CPU_ONLY=True`)

All 6 GPU agents share the CPU pool — launch analysts AND GPU agents together in a single message, fully in parallel.

```python
analysts   = [f"{PREFIX}_analyst{i}" for i in (1, 2, 3)]
gpu_agents = [f"{PREFIX}_gpu{i}" for i in range(1, 7)]

# (analysts already launched in Step 5b; here we add the 6 GPU agents in parallel)
for agent_name in gpu_agents:
    Agent(
        description=f"{agent_name} cycle (CPU parallel) — {deadline_status()}",
        prompt=_gpu_prompt(agent_name, time_remaining_minutes(), '""'),
        run_in_background=True,
    )
# All 9 agents run concurrently. Wait for all to complete before Step 5d.
```

### GPU tasks (`IS_CPU_ONLY=False`) — mixed dispatch

Only 1 GPU. GPU agents serialize on the GPU, but agents that choose a CPU-only experiment run in the background concurrently with the next GPU agent.

Each GPU agent declares its compute mode by writing `{FOCUS_ROOT}/logs/{agent_name}.gpu_claim` containing `gpu` or `cpu`. The orchestrator reads this before deciding whether to block or background.

```python
import time

GPU_CLAIM_TIMEOUT_S = 120
GPU_CLAIM_POLL_S    = 5

def _wait_for_gpu_claim(agent_name):
    claim_path = FOCUS_ROOT / "logs" / f"{agent_name}.gpu_claim"
    waited = 0
    while waited < GPU_CLAIM_TIMEOUT_S:
        if claim_path.exists():
            val = claim_path.read_text().strip().lower()
            return val if val in ("gpu", "cpu") else "gpu"
        time.sleep(GPU_CLAIM_POLL_S)
        waited += GPU_CLAIM_POLL_S
    print(f"  [{agent_name}] no gpu_claim after {GPU_CLAIM_TIMEOUT_S}s — assuming GPU")
    return "gpu"

for agent_name in gpu_agents:
    rem = time_remaining_minutes()
    if rem <= DEADLINE_BUFFER_MINUTES:
        print(f"Deadline imminent — not launching {agent_name}")
        break

    claim_path = FOCUS_ROOT / "logs" / f"{agent_name}.gpu_claim"
    claim_path.unlink(missing_ok=True)

    Agent(
        description=f"{agent_name} cycle (mixed dispatch) — {deadline_status()}",
        prompt=_gpu_prompt(agent_name, time_remaining_minutes(), "0"),
        run_in_background=True,
    )

    claim = _wait_for_gpu_claim(agent_name)
    print(f"  [{agent_name}] declared: {claim}")

    if claim == "gpu":
        # Block until this agent releases the GPU (writes result_latest.json or submission.csv)
        result_path = FOCUS_ROOT / "agents" / agent_name / "workspace" / "result_latest.json"
        sub_path    = FOCUS_ROOT / "agents" / agent_name / "workspace" / "repo" / "submission.csv"
        GPU_WAIT_MAX_S = min(60 * 60, int((time_remaining_minutes() - DEADLINE_BUFFER_MINUTES) * 60))
        waited = 0
        while waited < GPU_WAIT_MAX_S:
            if result_path.exists() or sub_path.exists():
                print(f"  [{agent_name}] GPU training complete (waited {waited}s)")
                break
            time.sleep(30)
            waited += 30
        else:
            print(f"  [{agent_name}] GPU wait timed out — proceeding")
    # else: CPU experiment, just launch the next agent immediately
```

### `_gpu_prompt`

```python
def _gpu_prompt(agent_name, rem, cuda):
    if cuda == '""':
        compute_note = (
            "COMPUTE: No GPU available. All training runs on CPU.\n"
            "You are running in parallel with other agents on the shared CPU pool.\n"
            "Do NOT all pick the same approach — see approach diversity rules in ROLE-GPU Step 2a.\n"
            "Good CPU methods: RDKit+GBM, Morgan+SVR, Mordred+RF, Tanimoto-GP, AutoML, "
            "offline embeddings + linear head.\n"
        )
    else:
        compute_note = (
            "COMPUTE: A GPU is available (CUDA_VISIBLE_DEVICES=0).\n\n"
            "MIXED DISPATCH — declare your compute mode BEFORE training:\n"
            f"  1. Decide whether your experiment needs the GPU or can run on CPU only.\n"
            f"  2. Write your claim IMMEDIATELY after deciding (before any training):\n"
            f"       echo 'gpu' > {FOCUS_ROOT}/logs/{agent_name}.gpu_claim   # GPU experiment\n"
            f"       echo 'cpu' > {FOCUS_ROOT}/logs/{agent_name}.gpu_claim   # CPU-only\n"
            f"  3. The orchestrator reads this to decide whether to block on you or launch\n"
            f"     the next agent in parallel. If you don't write it, the orchestrator\n"
            f"     assumes GPU and blocks — wasting wall-clock time.\n\n"
            "GPU experiments (declare 'gpu'): GNN (DGL/PyG/Chemprop), Transformer "
            "(ChemBERTa/MolBERT/UniMol), fine-tuned pretrained models, scVI, ViT/ResNet.\n"
            "CPU experiments (declare 'cpu'): RDKit+XGBoost/LightGBM, Morgan+SVR, "
            "Mordred+RF, Tanimoto-GP, AutoML, offline embeddings + linear head.\n\n"
            "The team should run a MIX: ideally 2-3 GPU and 2-3 CPU experiments per cycle.\n"
        )
    div_note = _approach_diversity_note()
    return (
        f"You are {agent_name}.\n"
        f"FOCUS_ROOT={FOCUS_ROOT}\n"
        f"CUDA_VISIBLE_DEVICES={cuda}\n"
        f"GPU_AVAILABLE={GPU_AVAILABLE}\n"
        f"MODE=execute\n"
        f"BIOMLBENCH=true\n"
        f"TIME_REMAINING_MINUTES={rem:.0f}\n"
        f"DEADLINE_BUFFER_MINUTES={DEADLINE_BUFFER_MINUTES}\n"
        f"Read {FOCUS_ROOT}/agents/{agent_name}/HEARTBEAT.md and follow it.\n"
        f"{compute_note}"
        f"{div_note}"
        f"IMPORTANT: If TIME_REMAINING_MINUTES < {DEADLINE_BUFFER_MINUTES + 20},\n"
        f"skip the second experiment (Step 10) and go directly to Part 5 to save\n"
        f"your best submission.csv to {FOCUS_ROOT}/task/submission.csv and exit.\n"
        f"Start at Part 0 (Mode Selector).\n"
        f"When done: <promise>{agent_name} cycle complete</promise>"
    )
```

---

## Hook: champion_promotion

**Champion = best `submission.csv` score across all agents this cycle.** Each agent writes `result_latest.json` (keys: `val_score`, `direction`, `exp_id`, `agent`, `submission_path`, `train_path`). Agents NEVER write directly to `task/submission.csv` or `champion/train.py` — this orchestrator block is the **sole** promoter.

Always propagate the best submission for deadline safety. Only update `champion.md` / `champion/train.py` / `SOURCE` when the new score strictly beats the previous champion.

```python
import shutil

best_score, best_agent, best_sub, best_train = None, None, None, None
best_direction = "maximize"

for agent_name in gpu_agents:
    result_path = FOCUS_ROOT / "agents" / agent_name / "workspace" / "result_latest.json"
    if not result_path.exists():
        # Fallback: bare submission.csv with no result summary — any submission beats none
        bare_sub = FOCUS_ROOT / "agents" / agent_name / "workspace" / "repo" / "submission.csv"
        if bare_sub.exists() and best_sub is None:
            best_sub, best_agent = bare_sub, agent_name
        continue

    result = json.loads(result_path.read_text())
    score, direction = result.get("val_score"), result.get("direction", "maximize")
    sub_path = Path(result["submission_path"]) if result.get("submission_path") else None
    if sub_path is None or not sub_path.exists():
        sub_path = FOCUS_ROOT / "agents" / agent_name / "workspace" / "repo" / "submission.csv"
    if not sub_path.exists():
        continue
    train_path = Path(result["train_path"]) if result.get("train_path") else \
                 FOCUS_ROOT / "agents" / agent_name / "workspace" / "repo" / "train.py"

    if score is None:
        if best_sub is None:
            best_sub, best_train, best_agent = sub_path, (train_path if train_path.exists() else None), agent_name
        continue

    is_better = (
        best_score is None or
        (direction == "maximize" and score > best_score) or
        (direction == "minimize" and score < best_score)
    )
    if is_better:
        best_score, best_agent, best_sub = score, agent_name, sub_path
        best_train = train_path if train_path.exists() else None
        best_direction = direction

if best_sub:
    canonical_sub = FOCUS_ROOT / "task" / "submission.csv"
    shutil.copy(best_sub, canonical_sub)
    print(f"submission.csv propagated from {best_agent} (score={best_score})")

    champ_raw = requests.get(f"{API}/workspaces/{WS_ID}/files/champion.md", headers=HEADERS).json()
    prev_fm    = parse_fm(champ_raw)
    prev_score = prev_fm.get("metric_value")
    metric_name = parse_fm(task_md).get("metric", "val_score")

    is_new_champion = (
        prev_score is None or best_score is None or
        (best_direction == "maximize" and best_score > prev_score) or
        (best_direction == "minimize" and best_score < prev_score)
    )
    if is_new_champion:
        (FOCUS_ROOT / "champion").mkdir(exist_ok=True)
        if best_train and best_train.exists():
            shutil.copy(best_train, FOCUS_ROOT / "champion" / "train.py")
        (FOCUS_ROOT / "champion" / "SOURCE").write_text(
            f"{best_agent} score={best_score} {datetime.now(timezone.utc).isoformat()}\n"
        )
        requests.put(f"{API}/workspaces/{WS_ID}/files/champion.md",
            headers=HEADERS, json={"content": (
                f"---\nmetric_name: {metric_name}\nmetric_value: {best_score}\n"
                f"direction: {best_direction}\nagent: {best_agent}\ncycle: {cycle_count}\n"
                f"timestamp: {datetime.now(timezone.utc).isoformat()}\nsubmission_saved: true\n"
                f"---\n\n# Champion (cycle {cycle_count})\n\n"
                f"- **Score:** {best_score}\n- **Agent:** {best_agent}\n"
                f"- **Previous:** {prev_score}\n- **submission.csv:** `task/submission.csv`\n"
            )})
        print(f"Champion updated: {best_agent} score={best_score} (prev={prev_score})")
    else:
        print(f"No improvement (best={best_score}, prev={prev_score}) — champion unchanged")
else:
    print("WARNING: No agent produced a submission.csv this cycle.")
```

---

## Hook: stagnation_response

**Do NOT stop the loop.** Time is finite and we always keep trying. Stagnation just triggers a regime change: analysts are asked to propose dramatically different approaches.

```python
def stagnation_response(cycle_count):
    if cycle_count < 3:
        return
    print(f"STAGNATION: 0 KEEPs in last 6 — posting [STUCK] to trigger pivot")
    requests.post(f"{API}/posts", headers=HEADERS, json={
        "workshop": WORKSHOP,
        "title":   f"[STUCK] Cycle {cycle_count}: 0 KEEPs in 6 experiments — need pivot",
        "content": (
            f"No improvements in last 6 experiments. {deadline_status()}.\n\n"
            f"Analysts: please propose a qualitatively different approach — different\n"
            f"model family, feature representation, or training strategy. Small variations\n"
            f"on the current champion are exhausted."
        ),
        "notify_agents": analysts,
        "tags": ["type:stuck", f"cycle:{cycle_count}"],
    })
```

(Base uses last 10 for stagnation; for biomlbench use last 6 since cycles are shorter.)

---

## Hook: periodic_hooks

**Meta-improvement is DISABLED for biomlbench** — modifying role templates mid-run risks regressing performance with no time to recover.

The only periodic action is resetting the approach registry at the start of each cycle so agents re-register their chosen paradigm for the cycle:

```python
def periodic_hooks(cycle_count):
    reg_path = FOCUS_ROOT / "logs" / "approach_registry.json"
    reg_path.write_text(json.dumps({"cycle": cycle_count, "taken": []}, indent=2))
    print(f"Approach registry reset for cycle {cycle_count}.")
```

---

## Hook: exit_condition

`pre_cycle_check` is the primary exit gate. This hook is a backstop:

```python
def exit_condition():
    return time_remaining_minutes() <= DEADLINE_BUFFER_MINUTES
```

---

## Hook: final_report

```python
def final_report():
    submission_path = FOCUS_ROOT / "task" / "submission.csv"
    champ = parse_fm(requests.get(
        f"{API}/workspaces/{WS_ID}/files/champion.md", headers=HEADERS).json())
    print()
    print("=" * 60)
    print("  BIOMLBENCH RUN COMPLETE")
    print("=" * 60)
    print(f"  Task:            {task_name}")
    print(f"  Total cycles:    {cycle_count}")
    print(f"  Time used:       {WALL_CLOCK_HOURS * 60 - time_remaining_minutes():.0f}"
          f" / {WALL_CLOCK_HOURS * 60:.0f} min")
    print(f"  Best score:      {champ.get('metric_value', 'none')}")
    print(f"  Best agent:      {champ.get('agent', 'none')}")
    print(f"  submission.csv:  {'EXISTS' if submission_path.exists() else 'MISSING — CRITICAL'}")
    print(f"  Location:        {submission_path}")
    print()
    if not submission_path.exists():
        print("  CRITICAL: No submission.csv was saved. Manual intervention required.")
    else:
        print("  submission.csv is ready for biomlbench grader.")
    print("=" * 60)
```

---

## Hook: never_do_extras

In addition to the universal "never do" list:

- Do NOT set `CUDA_VISIBLE_DEVICES=0` when `IS_CPU_ONLY=True` (use `""`).
- Do NOT launch GPU agents in parallel when `IS_CPU_ONLY=False` (only 1 GPU).
- Do NOT launch GPU agents sequentially when `IS_CPU_ONLY=True` (wastes wall-clock).
- Do NOT stop the loop due to stagnation — time is the only stopping criterion.
- Do NOT edit `system/templates/ROLE-GPU.md` or `ROLE-ANALYST.md` (meta-improvement is disabled).
- Do NOT use `champion/train.py` as the deliverable — the deliverable is `task/submission.csv`.

**If you feel the urge to "just run something quickly" because agents are struggling or time is running out — stop. The orchestrator never runs experiments. Period.**
