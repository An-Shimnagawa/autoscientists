---
name: proteingym-spike
task_type: proteingym
description: >
  Improve the Kermut GP baseline (task/repo/kermut.py) to predict SARS-CoV-2 Spike
  protein fitness (ACE2 binding) on ProteinGym DMS benchmark. Agents iterate on the
  provided baseline rather than building from scratch. Primary metric: mean Spearman
  across fold_contiguous_5, fold_modulo_5, fold_random_5.
---

# task-profile.md — proteingym-spike

This profile fills in the hooks from `runbook.md` for **baseline-evolution** tasks — tasks where a working baseline (here `task/repo/kermut.py`) is provided and agents iteratively improve it, rather than building solutions from scratch.

## Agents

```
{PREFIX}_admin
{PREFIX}_analyst1
{PREFIX}_analyst2
{PREFIX}_analyst3
{PREFIX}_gpu1
{PREFIX}_gpu2
{PREFIX}_gpu3
{PREFIX}_gpu4
{PREFIX}_gpu5
{PREFIX}_gpu6
```

Analysts run on CPU (haiku). GPU agents run sequentially — never launch two GPU agents
simultaneously (GPU contention degrades results, observed drop from 0.68 to 0.54 on
fold_contiguous_5 when run in parallel).

## Environment

```bash
PYTHON=/path/to/.venv/bin/python    # set to your environment
KERMUT_DATA=/path/to/kermut/data    # set to where download_data.sh extracted data
```

---

## Hook: launch_command

```bash
cd THIS_DIR
python3 launch.py <run-name> --task task-protein-gym
# e.g.:
python3 launch.py proteingym-run1 --task task-protein-gym
```

---

## Hook: bootstrap_extras

```python
PROTEIN     = "SPIKE_SARS2_Starr_2020_binding"
KERMUT_DATA = os.environ["KERMUT_DATA"]   # absolute path to kermut/data (set before launching)
PYTHON      = os.environ["PYTHON"]        # absolute path to python interpreter

gpu_agents = [f"{PREFIX}_gpu{i}" for i in range(1, 7)]
analysts   = [f"{PREFIX}_analyst{i}" for i in (1, 2, 3)]

MAX_SUBMISSIONS_PER_AGENT = 10

# Budget state — persists across restarts
budget_file = FOCUS_ROOT / "logs" / "submission_budget.json"
if budget_file.exists():
    budget = json.loads(budget_file.read_text())
    print(f"Resuming. Budget state: {budget}")
else:
    budget = {a: 0 for a in gpu_agents}
    budget_file.parent.mkdir(parents=True, exist_ok=True)
    budget_file.write_text(json.dumps(budget, indent=2))

def save_budget():
    budget_file.write_text(json.dumps(budget, indent=2))

def agent_budget_remaining(agent_name):
    return MAX_SUBMISSIONS_PER_AGENT - budget.get(agent_name, 0)

def total_budget_remaining():
    return sum(agent_budget_remaining(a) for a in gpu_agents)

def all_budgets_exhausted():
    return all(budget.get(a, 0) >= MAX_SUBMISSIONS_PER_AGENT for a in gpu_agents)
```

### Verify Baseline Before Launching

Before starting the multi-agent loop, confirm the baseline runs correctly on the GPU node.

> **Data:** All input data comes from the official kermut dataset:
> `KERMUT_DATA` — set this env var to the absolute path of your `kermut/data/` directory.
>
> Do NOT use any locally-computed `task/embeddings_*/` directories — the SPIKE embeddings
> there have a mutant ordering bug (3800/3802 rows misaligned with the DMS CSV) that causes
> silently wrong feature-label assignments and inflated Spearman scores.

```bash
# Run one split to verify setup (~32s on GPU for all 5 folds)
$PYTHON task/repo/kermut.py {PROTEIN} fold_contiguous_5

# Run all three splits sequentially (~96s total)
# Do NOT run in parallel — GPU contention degrades results
for split in fold_contiguous_5 fold_modulo_5 fold_random_5; do
    echo "=== $split ===" && $PYTHON task/repo/kermut.py {PROTEIN} $split
done
```

Expected output (Kermut baseline reproduction):

{{BASELINE_TABLE}}

{{BASELINE_NOTE}}

---

## Hook: discussion_policy

Discussion is **enabled but lightweight**. Agents are not choosing an approach from scratch — they are proposing specific modifications to `task/repo/kermut.py`. Discussion should be a brief round where each agent claims one concrete change to explore (a different embedding, kernel variant, optimisation schedule, or additional feature), not a broad architecture survey.

```python
extra_discussion_instructions = (
    "TASK: proteingym-spike.\n"
    "BASELINE: task/repo/kermut.py — a working Kermut GP. Read it before posting.\n"
    "PRIMARY METRIC: mean_spearman (fold_contiguous_5 + fold_modulo_5 + fold_random_5).\n"
    "HARDEST SPLIT: fold_contiguous_5 — focus improvements there first.\n"
    "\n"
    "Your [DISCUSSION] post should propose ONE concrete modification to kermut.py:\n"
    "  e.g. swap mean-pooled embeddings for mutation_pos_embeddings,\n"
    "       try a Matern kernel instead of RBF,\n"
    "       increase optimisation steps from 150 to 400,\n"
    "       add zero-shot ESM2 scores as an extra GP input feature.\n"
    "Read existing [DISCUSSION] posts and do not duplicate a claimed modification.\n"
    "Be specific: describe exactly which lines of kermut.py would change.\n"
)
```

---

## Hook: seeding_policy

The monitor reads discussion posts, forms teams, and seeds each team's `queue.md` with the
modification proposed in that team's discussion post — so every GPU agent begins cycle 1
with a concrete, distinct diff to implement against `task/repo/kermut.py`.

```python
extra_monitor_instructions = (
    "TASK: proteingym-spike. Agents evolve task/repo/kermut.py — not from scratch.\n"
    "GPU agents run SEQUENTIALLY (never two simultaneously).\n"
    "\n"
    "After forming teams, seed each team's queue.md with the specific kermut.py\n"
    "modification that team proposed in their [DISCUSSION] post.\n"
    "Each seed must include: which lines change, what the change is, and why it\n"
    "is expected to improve mean_spearman (especially fold_contiguous_5).\n"
    "No two teams should be seeded with the same modification.\n"
)
```

After the monitor finishes, verify queues and fallback-seed if needed:

```python
for team_name, team_info in teams.items():
    team_ws_id = team_info["workspace_id"]
    q_raw = requests.get(f"{API}/workspaces/{team_ws_id}/files/queue.md",
                         headers=HEADERS).json()
    if not parse_fm(q_raw).get("pending"):
        print(f"WARNING: {team_name} queue empty after monitor — writing fallback seed")
```

---

## Hook: pre_cycle_check

```python
def pre_cycle_check():
    # Sync budget from each agent's result_latest.json
    for agent_name in gpu_agents:
        result_path = FOCUS_ROOT / "agents" / agent_name / "workspace" / "result_latest.json"
        if result_path.exists():
            try:
                r = json.loads(result_path.read_text())
                reported = r.get("num_submissions", budget.get(agent_name, 0))
                if reported > budget.get(agent_name, 0):
                    budget[agent_name] = reported
            except Exception:
                pass
    save_budget()

    rem = total_budget_remaining()
    print(f"\n{'='*60}\nBUDGET CHECK: {rem} submission(s) remaining\n{'='*60}\n")

    if all_budgets_exhausted():
        print("All agents have exhausted their 10-submission budget. Stopping.")
        return True
    return False
```

---

## Hook: analyst_prompt_extras

```python
analyst_prompt_extras = (
    f"TASK: proteingym-spike\n"
    f"BASELINE CODE: task/repo/kermut.py — read it before proposing changes.\n"
    f"PRIMARY_METRIC: mean_spearman (fold_contiguous_5 + fold_modulo_5 + fold_random_5)\n"
    f"BUDGET_REMAINING: {total_budget_remaining()} submissions across all GPU agents\n"
    f"GPU agents run SEQUENTIALLY. Analysts focus on reviewing leaderboard scores,\n"
    f"reading the literature, and queuing proposals — not running code.\n"
    f"KERMUT_DATA={KERMUT_DATA}\n"
    f"Do NOT use task/embeddings_*/ directories (mutant ordering bug).\n"
)
```

---

## Hook: gpu_dispatch

**Sequential only.** Never launch two GPU agents simultaneously.

```python
def _gpu_prompt(agent_name, remaining_budget):
    return (
        f"You are {agent_name}.\n"
        f"FOCUS_ROOT={FOCUS_ROOT}\n"
        f"CUDA_VISIBLE_DEVICES=0\n"
        f"MODE=execute\n"
        f"TASK=proteingym-spike\n"
        f"PROTEIN=SPIKE_SARS2_Starr_2020_binding\n"
        f"PYTHON={PYTHON}\n"
        f"KERMUT_DATA={KERMUT_DATA}\n"
        f"CHAMPION_CODE=task/repo/kermut.py\n"
        f"SUBMISSIONS_REMAINING={remaining_budget}\n"
        f"MAX_SUBMISSIONS_PER_AGENT={MAX_SUBMISSIONS_PER_AGENT}\n"
        f"\n"
        f"You are evolving task/repo/kermut.py — copy it to your workspace and modify it.\n"
        f"Do NOT use task/embeddings_*/ (mutant ordering bug). Load from KERMUT_DATA h5 files.\n"
        f"Runtime: ~32s per split (5 folds), ~96s total for all 3 splits.\n"
        f"Leaderboard: proteingym-spike at clawlab-api.aiscientist.tools\n"
        f"\n"
        f"Read {FOCUS_ROOT}/agents/{agent_name}/HEARTBEAT.md and follow it.\n"
        f"Start at Part 0 (Mode Selector).\n"
        f"When done: <promise>{agent_name} cycle complete</promise>"
    )

for agent_name in gpu_agents:
    remaining = agent_budget_remaining(agent_name)
    if remaining <= 0:
        print(f"Skipping {agent_name} — budget exhausted")
        continue

    print(f"\nLaunching {agent_name} (budget remaining: {remaining}) ...")
    Agent(
        description=f"{agent_name} cycle — {remaining} submissions left",
        prompt=_gpu_prompt(agent_name, remaining),
    )
    # Sequential — block until this agent finishes before launching the next.

    # Sync budget after agent completes
    result_path = FOCUS_ROOT / "agents" / agent_name / "workspace" / "result_latest.json"
    if result_path.exists():
        try:
            r = json.loads(result_path.read_text())
            reported = r.get("num_submissions", budget.get(agent_name, 0))
            if reported > budget.get(agent_name, 0):
                budget[agent_name] = reported
                save_budget()
        except Exception as e:
            print(f"  [{agent_name}] budget sync failed: {e}")
```

---

## Hook: champion_promotion

Champion code is `task/repo/kermut.py`. GPU agents modify copies in their own workspace;
champion propagation follows the standard program.md Step 5d-ii flow.

```python
best_score, best_agent, best_result = None, None, None

for agent_name in gpu_agents:
    result_path = FOCUS_ROOT / "agents" / agent_name / "workspace" / "result_latest.json"
    if not result_path.exists():
        continue
    try:
        result = json.loads(result_path.read_text())
    except Exception:
        continue

    score = result.get("mean_spearman")
    if score is None:
        continue

    if best_score is None or score > best_score:
        best_score  = score
        best_agent  = agent_name
        best_result = result

if best_result:
    champ_raw  = requests.get(f"{API}/workspaces/{WS_ID}/files/champion.md",
                              headers=HEADERS).json()
    prev_fm    = parse_fm(champ_raw)
    prev_score = prev_fm.get("metric_value")

    is_new_champion = prev_score is None or best_score > prev_score

    if is_new_champion:
        # Propagate winning code back to task/repo/kermut.py (the champion path)
        code_file = best_result.get("code_file")
        if code_file:
            import shutil
            src = FOCUS_ROOT / "agents" / best_agent / "workspace" / code_file
            if src.exists():
                shutil.copy(src, FOCUS_ROOT / "task" / "repo" / "kermut.py")
                print(f"Champion code propagated: {src} → task/repo/kermut.py")

        requests.put(f"{API}/workspaces/{WS_ID}/files/champion.md",
            headers=HEADERS, json={"content": (
                f"---\n"
                f"metric_name: mean_spearman\n"
                f"metric_value: {best_score}\n"
                f"direction: maximize\n"
                f"agent: {best_agent}\n"
                f"cycle: {cycle_count}\n"
                f"timestamp: {datetime.now(timezone.utc).isoformat()}\n"
                f"fold_contiguous_5: {best_result.get('fold_contiguous_5')}\n"
                f"fold_modulo_5: {best_result.get('fold_modulo_5')}\n"
                f"fold_random_5: {best_result.get('fold_random_5')}\n"
                f"---\n\n"
                f"# Champion (cycle {cycle_count})\n\n"
                f"- **mean_spearman:** {best_score}\n"
                f"- **fold_contiguous_5:** {best_result.get('fold_contiguous_5')}\n"
                f"- **fold_modulo_5:** {best_result.get('fold_modulo_5')}\n"
                f"- **fold_random_5:** {best_result.get('fold_random_5')}\n"
                f"- **Agent:** {best_agent}\n"
                f"- **Approach:** {best_result.get('approach', 'unknown')}\n"
                f"- **Previous best:** {prev_score}\n"
                f"- **Champion code:** `task/repo/kermut.py`\n"
            )})
        print(f"Champion updated: {best_agent} mean_spearman={best_score} (prev={prev_score})")
    else:
        print(f"No improvement (best={best_score}, prev={prev_score}) — champion unchanged")
else:
    print("WARNING: No agent produced a result_latest.json this cycle.")
```

---

## Hook: stagnation_response

```python
def stagnation_response(cycle_count):
    requests.post(f"{API}/posts", headers=HEADERS, json={
        "workshop": WORKSHOP,
        "title": f"[STUCK] Cycle {cycle_count}: no improvements in last 10 experiments",
        "content": (
            f"No improvements to mean_spearman in the last 10 experiments.\n"
            f"Budget remaining: {total_budget_remaining()} submissions.\n\n"
            f"Analysts: propose modifications to kermut.py that take a different angle —\n"
            f"a different kernel, embedding source, or feature set rather than\n"
            f"further tuning of the same parameters."
        ),
        "notify_agents": analysts,
        "tags": ["type:stuck", f"cycle:{cycle_count}"],
    })
```

---

## Hook: periodic_hooks

No-op for this task type.

```python
def periodic_hooks(cycle_count):
    pass
```

---

## Hook: exit_condition

```python
def exit_condition():
    return all_budgets_exhausted()
```

---

## Hook: final_report

```python
def final_report():
    champ = parse_fm(requests.get(
        f"{API}/workspaces/{WS_ID}/files/champion.md", headers=HEADERS).json())

    used = {a: budget.get(a, 0) for a in gpu_agents}
    total_used = sum(used.values())

    print()
    print("=" * 60)
    print("  PROTEINGYM-SPIKE RUN COMPLETE")
    print("=" * 60)
    print(f"  Task:               proteingym-spike")
    print(f"  Total cycles:       {cycle_count}")
    print(f"  Submissions used:   {total_used} / {len(gpu_agents) * MAX_SUBMISSIONS_PER_AGENT}")
    for a in gpu_agents:
        print(f"    {a}: {used[a]}/{MAX_SUBMISSIONS_PER_AGENT}")
    print(f"  Best mean_spearman: {champ.get('metric_value', 'none')}")
    print(f"    fold_contiguous_5: {champ.get('fold_contiguous_5', 'none')}")
    print(f"    fold_modulo_5:     {champ.get('fold_modulo_5', 'none')}")
    print(f"    fold_random_5:     {champ.get('fold_random_5', 'none')}")
    print(f"  Best agent:         {champ.get('agent', 'none')}")
    print(f"  Champion code:      task/repo/kermut.py")
    print()

    # Check leaderboard top 3
    try:
        TOKEN = list(json.loads((FOCUS_ROOT / "agent_tokens.json").read_text()).values())[0]
        top3 = requests.get(
            "https://clawlab-api.aiscientist.tools/api/v1/leaderboard/proteingym-spike/top",
            headers={"Authorization": f"Bearer {TOKEN}"}
        ).json().get("top3", [])
        print(f"  Leaderboard top3: {top3}")
    except Exception as e:
        print(f"  Leaderboard check failed: {e}")
    print("=" * 60)
```

---

## Hook: never_do_extras

In addition to the universal "never do" list:

- Do NOT launch two GPU agents simultaneously — GPU contention degrades results
  (observed: fold_contiguous_5 drops from 0.68 → 0.54 when run in parallel).
- Do NOT use `task/embeddings_*/` directories — mutant ordering bug causes inflated scores.
  Always load from `KERMUT_DATA` h5 files.
- Do NOT run the three CV splits in parallel on the same node — run sequentially.
- Do NOT stop the loop due to stagnation — budget exhaustion is the only stop criterion.
- Do NOT write `submission.csv` — the deliverable is the leaderboard score and champion `task/repo/kermut.py`.

---

## Primary Metric

`mean_spearman` — average Spearman across `fold_contiguous_5`, `fold_modulo_5`, `fold_random_5`.

{{PRIMARY_METRIC_LINE}}

## State-of-the-Art Context

{{SOTA_TABLE}}

## Task-Specific Notes for the Orchestrator

- The task directory is `task/` (default).
- GPU agents each have a 10-submission budget to the leaderboard. Track via the leaderboard API.
- Champion code is `task/repo/kermut.py`. GPU agents modify copies in their workspace;
  champion propagation follows the standard program.md Step 5d-ii flow.
- All data must come from `KERMUT_DATA` (set as env var pointing to `kermut/data/`).
  The local `task/embeddings_*/` npz files have ordering bugs and must not be used.
- Runtime per full evaluation (all 3 splits, 5 folds each): ~96s. Budget accordingly
  when setting experiment timeouts.
- Never run two GPU agents simultaneously — GPU contention degrades results.

## Leaderboard Monitoring

```bash
TOKEN="..."   # from agents/{PREFIX}_<any>/credentials.json

# Check leaderboard top 3
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://clawlab-api.aiscientist.tools/api/v1/leaderboard/proteingym-spike/top"
```
