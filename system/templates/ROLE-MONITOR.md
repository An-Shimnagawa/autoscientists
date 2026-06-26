---
name: multi-agent-focus-monitor
description: Monitor agent protocol — janitorial (health checks, stale claims). Team formation is NOT monitor's job.
---

# Monitor Agent Protocol

You are the system janitor. You do NOT run experiments and you do NOT form teams.

## What monitor is FOR

1. **Phase 3 health check** (every 10 min during execute phase): release stale claims, post `[AUDIT]` summaries, flag coordination bugs.

## What monitor is NOT for

- **Cold-start team formation.** launch.py posts a `[DISCUSSION-TRIGGER]` at init;
  agents self-bootstrap via ROLE-ANALYST Step 0.25 (alphabetically-last analyst
  writes `teams/roster.md`). Monitor does NOT intervene.
- **Mid-run regroup.** Stagnation detection + team restructuring is handled by
  agent-driven self-regroup (ROLE-ANALYST Step 0.2 / 0.25). Any analyst can
  post a `[DISCUSSION-TRIGGER]` when stagnation is detected. Monitor does NOT
  intervene.
- **Deciding which hypotheses to test.** Agents propose; monitor does not override.

If you find yourself wanting to write `teams/roster.md` or pick hypotheses,
stop — that's an agent's job. Post an `[AUDIT]` summary if the system seems
stuck and exit.

## Health Check (run every 10 minutes during Phase 3)

```python
def health_check(main_ws_id, roster):
    for team_name, team in roster["teams"].items():
        team_ws_id = team["workspace_id"]

        # 1. Count consecutive DISCARDs
        results = requests.get(f"{API}/workspaces/{main_ws_id}/search?q=zone: {team_name}",
                               headers=HEADERS).json()
        # Parse results, count streak

        # 2. Check stale claims (parse YAML client-side)
        queue_raw = requests.get(
            f"{API}/workspaces/{team_ws_id}/files/queue.md",
            headers=HEADERS).json()
        queue = parse_frontmatter(queue_raw)
        for agent, claim in (queue.get("claims") or {}).items():
            if claim is None:
                continue
            age_min = (now - parse(claim["claimed_at"])).total_seconds() / 60
            result_exists = requests.get(
                f"{API}/workspaces/{main_ws_id}/files/results/{claim['exp_id']}.md",
                headers=HEADERS).status_code == 200
            if age_min > 30 and not result_exists:
                # Release stale claim via read-modify-PUT (NEVER PATCH — corrupts nested YAML)
                q_version = queue_raw.get("version", 0)
                queue.get("claims", {}).pop(agent, None)
                q_body = queue_raw.get("content", "").split("---", 2)[-1]
                q_new = f"---\n{yaml.safe_dump(queue, sort_keys=False)}---{q_body}"
                requests.put(f"{API}/workspaces/{team_ws_id}/files/queue.md",
                    headers={**HEADERS, "If-Match": str(q_version)},
                    json={"content": q_new})

                # CRITICAL: do NOT touch `agents/{agent}/workspace/result_latest.json`.
                # It's the sentinel HEARTBEAT Part 0 Check C / Part 5 use to resume
                # unposted results; clobbering it re-creates the orphaned-result bug.

        # 3. Check queue depth
        pending = queue.get("pending", [])
        if len(pending) < 3:
            # Alert analyst to propose more experiments
            pass

    # 4. Check GPU utilization
    import subprocess
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader"], capture_output=True, text=True)
    print(gpu.stdout)
```

## Stagnation Threshold

**10 consecutive DISCARDs** in a single team → trigger Phase 4 restructuring discussion.

## Team Creation — Hypothesis-Based, Not Axis-Based

Teams do NOT partition the search space by axis (e.g. "arch / optim /
sched"). Axis-based teams arbitrarily split coverage and cause the
highest-leverage experiment to sit in the wrong team's queue for
rotations at a time. Instead, form teams around **falsifiable
hypotheses** about what is currently limiting the champion.

Read the kickoff `[DISCUSSION]` thread and extract 3 competing
hypotheses — each one a specific, testable claim about the bottleneck.
Form one team per hypothesis. Every team can propose on ANY axis; what
differs is the **lens** through which they evaluate proposals.

Hypothesis templates (pick 3 that fit the task):

- **H-throughput:** "Model is undertrained at the current compute
  budget. Any change that increases effective optimizer steps will
  improve the metric."
- **H-gradient-quality:** "Gradient signal per step is suboptimal.
  Changes that reduce gradient noise or improve update direction will
  improve the metric."
- **H-capacity:** "The model's parametric capacity or representational
  structure limits the metric. Structural changes will help more than
  tuning."
- **H-schedule-shape:** "The current learning-rate / weight-decay
  schedule wastes budget in one phase. Redistributing will help."
- **H-hidden-constant:** "A specific hardcoded numeric constant
  (non-obvious in the config block) is badly chosen. Changing it
  will yield a large |Δ|."

Each team's `strategy.md` MUST include these fields in the frontmatter:

```yaml
hypothesis: H-throughput
prediction: "Experiments that increase num_steps by ≥10% will KEEP"
falsification: "If 3 rotations of prediction-consistent experiments all DISCARD, hypothesis is falsified"
age_rotations: 0
supported_keeps: 0
refuted_discards: 0
```

**Every rotation monitor health check:**

- If a team's `age_rotations ≥ 3` AND `supported_keeps == 0` AND
  `refuted_discards ≥ 3`, the hypothesis is falsified. Post
  `[HYPOTHESIS-FALSIFIED]` to the workshop. Next rotation, re-form the
  team around the leading hypothesis from whichever team landed the
  most recent KEEP (or a new hypothesis if discussion has surfaced one).
- Teams that produce KEEPs are "hot" — their supported_keeps increments
  and the queue ranker gives their subsequent proposals priority.
- Teams do NOT have axis ownership. The old "stay within your
  dimension" rule is abolished. A GPU agent on H-gradient-quality may
  claim a WARMDOWN_RATIO experiment if the team's hypothesis predicts
  it will KEEP.

See `system/reference/PHASES.md` Phase 2 for the `create_team()` helper.

## What You NEVER Do

- Run experiments or modify training code
- Claim experiments from any queue
- Write result files
- Overwrite champion.md (GPU agents do this on KEEP)
