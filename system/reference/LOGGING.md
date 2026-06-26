---
name: multi-agent-focus-logging
description: Structured logging for full experiment tracking and ablation analysis
---

# Logging & Tracking

Every action in the system is logged for full traceability.

## Single Canonical Log

**`{FOCUS_ROOT}/logs/experiments.jsonl`** is the SINGLE SOURCE OF TRUTH for all experiment results. The **orchestrator** writes this file — agents do NOT write to it directly.

### How it works

1. GPU agent runs experiment and reports result in its promise message
2. Orchestrator receives the result and writes ONE line to `experiments.jsonl`
3. This line contains everything needed for stagnation checks and analysis

### Format

```json
{
  "exp_id": "exp_swiglu",
  "agent": "run01_gpu1",
  "team": "architecture",
  "metric": 0.998097,
  "champion_before": 1.005071,
  "champion_after": 0.998097,
  "delta": -0.006974,
  "outcome": "KEEP",
  "description": "SwiGLU MLP replacement",
  "started_at": "2026-03-29T10:01:00Z",
  "completed_at": "2026-03-29T10:06:30Z",
  "training_seconds": 300.1,
  "race_condition": false
}
```

## Other Logs (secondary, not authoritative)

```
{FOCUS_ROOT}/
├── logs/
│   ├── experiments.jsonl       ← CANONICAL (orchestrator writes)
│   ├── sessions.jsonl          ← One line per agent session (orchestrator writes)
│   └── raw/
│       └── {agent}_{timestamp}.log  ← Raw stdout/stderr per session
│
├── agents/{name}/
│   └── actions.md              ← Human-readable session history per agent
│
└── Main workspace
    ├── results/{exp_id}.md     ← Structured result per experiment (agent writes)
    └── agents/{name}.md        ← Last-seen, session count, last outcome
```

These are all useful for context but `experiments.jsonl` is the one the stagnation check reads.

## 1. sessions.jsonl — Session Tracking

**Written by:** Orchestrator, after each agent session finishes.
**Format:** One JSON line per session.

```json
{
  "agent": "run01_gpu1",
  "role": "gpu",
  "team": "architecture",
  "session_id": "uuid",
  "started_at": "2026-03-29T10:00:00Z",
  "ended_at": "2026-03-29T10:08:30Z",
  "duration_seconds": 510,
  "status": "success",
  "promise_received": true,
  "experiments_run": 2,
  "experiments": [
    {"exp_id": "exp_kv_shift", "metric": 0.985, "outcome": "KEEP", "delta": -0.005},
    {"exp_id": "exp_gated_attn", "metric": 1.002, "outcome": "DISCARD", "delta": 0.012}
  ],
  "error": null
}
```

**Failed session:**
```json
{
  "agent": "run01_gpu2",
  "role": "gpu",
  "team": "optimizer",
  "started_at": "2026-03-29T10:00:05Z",
  "ended_at": "2026-03-29T10:20:05Z",
  "duration_seconds": 1200,
  "status": "timeout",
  "promise_received": false,
  "experiments_run": 1,
  "experiments": [
    {"exp_id": "exp_muon_warmup", "metric": null, "outcome": null, "delta": null}
  ],
  "error": "Agent timed out after 1200s — training may have completed but result not written"
}
```

**Orchestrator writes this:**
```python
import json, uuid
from datetime import datetime, timezone

def log_session(agent, role, team, started, status, experiments, error=None):
    entry = {
        "agent": agent,
        "role": role,
        "team": team,
        "session_id": str(uuid.uuid4()),
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": (datetime.now(timezone.utc) - parse(started)).total_seconds(),
        "status": status,
        "promise_received": status == "success",
        "experiments_run": len(experiments),
        "experiments": experiments,
        "error": error
    }
    with open(f"{FOCUS_ROOT}/logs/sessions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
```

## 2. experiments.jsonl — Experiment Tracking

**Written by:** GPU agents, after each experiment.
**Format:** One JSON line per experiment.

```json
{
  "exp_id": "exp_kv_shift_identity_init",
  "agent": "run01_gpu1",
  "team": "architecture",
  "run_id": "run_001_gpu1",
  "champion_baseline": 0.990,
  "metric": 0.985,
  "delta": -0.005,
  "outcome": "KEEP",
  "training_seconds": 300.1,
  "started_at": "2026-03-29T10:01:00Z",
  "completed_at": "2026-03-29T10:06:30Z",
  "description": "KV-shift attention with identity init"
}
```

**GPU agent writes this:**
```python
with open(f"{FOCUS_ROOT}/logs/experiments.jsonl", "a") as f:
    f.write(json.dumps(experiment_entry) + "\n")
```

## 3. Raw Logs

**Written by:** Orchestrator, capturing agent stdout/stderr.
**Location:** `{FOCUS_ROOT}/logs/raw/{agent}_{YYYYMMDD}_{HHMMSS}.log`

```bash
# Orchestrator captures:
claude -p "..." 2>&1 | tee logs/raw/${AGENT}_$(date +%Y%m%d_%H%M%S).log
```

## 4. agents/{name}/actions.md — Per-Agent History

**Written by:** Each agent at the end of its session (Step 4 in HEARTBEAT.md).
**Format:** Human-readable markdown, auto-truncated at 100 lines.

```markdown
## Session 2026-03-29T10:00:00Z — run01_gpu1

### State
- Champion: 0.990 (run_baseline)
- Team: architecture (2 pending, 0 claims)

### Experiments
1. exp_kv_shift → 0.985 KEEP (delta=-0.005) ← NEW CHAMPION
2. exp_gated_attn → 1.002 DISCARD (delta=+0.012)

### Duration: 8m 30s
```

## 5. Workspace Agent Files — Cross-Agent Visibility

**Written by:** Each agent via PATCH after session.
**Read by:** Orchestrator + other agents to see who's active.

```yaml
---
agent: run01_gpu1
last_seen: "2026-03-29T10:08:30Z"
status: idle
session_count: 5
last_experiment: exp_gated_attn
last_outcome: DISCARD
last_metric: 1.002
---
```

## Analysis Queries

### "How many experiments per team?"
```bash
cat logs/experiments.jsonl | python3 -c "
import sys, json
from collections import Counter
teams = Counter()
for line in sys.stdin:
    d = json.loads(line)
    teams[d['team']] += 1
for team, count in teams.most_common():
    print(f'{team}: {count}')
"
```

### "What's the KEEP rate per team?"
```bash
cat logs/experiments.jsonl | python3 -c "
import sys, json
from collections import defaultdict
stats = defaultdict(lambda: {'keep': 0, 'discard': 0})
for line in sys.stdin:
    d = json.loads(line)
    stats[d['team']][d['outcome'].lower()] += 1
for team, s in stats.items():
    total = s['keep'] + s['discard']
    rate = s['keep'] / total * 100 if total else 0
    print(f'{team}: {s[\"keep\"]}/{total} KEEPs ({rate:.0f}%)')
"
```

### "How many sessions timed out?"
```bash
cat logs/sessions.jsonl | python3 -c "
import sys, json
from collections import Counter
status = Counter()
for line in sys.stdin:
    d = json.loads(line)
    status[d['status']] += 1
for s, c in status.most_common():
    print(f'{s}: {c}')
"
```

### "Timeline of champion improvements"
```bash
cat logs/experiments.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if d['outcome'] == 'KEEP':
        print(f'{d[\"completed_at\"]} {d[\"exp_id\"]:40s} {d[\"metric\"]:.6f} (delta={d[\"delta\"]:+.6f}) by {d[\"agent\"]} ({d[\"team\"]})')
"
```

## Directory Setup

The orchestrator creates log directories before first launch:

```python
(FOCUS_ROOT / "logs" / "raw").mkdir(parents=True, exist_ok=True)
# sessions.jsonl and experiments.jsonl are created on first write (append mode)
```
