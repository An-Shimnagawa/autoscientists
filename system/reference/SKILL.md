---
name: multi-agent-focus
description: >
  Generic skill for self-organizing multi-agent teams that collaborate on an
  optimization problem. Agents discuss dimensions, form teams, run experiments,
  and adapt when stagnating. Uses AnonAPI posts for discussion and
  workspaces for shared state.
---

# Multi-Agent Focus Area

A **focus area** is a group of AI agents collaborating on an optimization problem. Agents self-organize into teams, each attacking a different dimension of the problem.

## Core Concepts

| Concept | What it is | AnonAPI feature |
|---|---|---|
| **Workshop** | The focus area container — all agents subscribe | `POST /workshops` |
| **Main workspace** | Shared state: champion config, all results, cross-team knowledge | `POST /workspaces` |
| **Team workspace** | Team-internal state: queue, hypotheses, dead ends, strategy | One per team |
| **Posts** | Discussion: proposals, results, strategy debates, votes | `POST /posts` |
| **Workspace files** | Structured data with YAML frontmatter, versioned, searchable | `PUT /workspaces/{id}/files/{path}` |

## How It Works

```
1. BOOTSTRAP    — Monitor creates workshop + main workspace + kickoff post
2. DISCUSS      — All agents propose dimensions, debate, vote on teams
3. EXECUTE      — Teams run experiments in parallel, share results
4. ADAPT        — Stagnating teams restructure via discussion + vote
```

See `reference/PHASES.md` for detailed lifecycle.

## Agent Roles

| Role | Count per team | What they do |
|---|---|---|
| **Monitor** | 1 (global) | Bootstrap, facilitate team formation, monitor health |
| **GPU Agent** | 2 per team | Claim experiments, train models, record results |
| **Analyst** | 1 per team | Research mechanisms, propose experiments, prune dead ends |

See `templates/ROLE-MONITOR.md`, `templates/ROLE-GPU.md`, `templates/ROLE-ANALYST.md`, `templates/ROLE-TEAM.md`.

## Main Workspace — Initial Files

These files are created during bootstrap. Agents may create additional files as needed — other agents discover them via LIST.

```
champion.md                — Current best config (ESSENTIAL ANCHOR — always read)
results/{exp_id}.md        — One file per experiment result (write-once)
teams/roster.md            — Team assignments and workspace IDs (ESSENTIAL ANCHOR)
```

Additional files are created organically by agents (e.g., `knowledge/lr-schedules.md`). Use `GET /files` to discover what exists.

## Team Workspace — Initial Files

```
queue.md                   — Pending experiments + active claims (ESSENTIAL ANCHOR)
dead_ends.md               — Mechanisms ruled out by this team
strategy.md                — Current team approach
```

Agents may create additional files (analysis docs, hypothesis lists, etc.). Use descriptive paths — see `templates/ROLE-TEAM.md` § File Naming Convention.

## Coordination Model

- **Discovery over prescription** — agents LIST workspace files each cycle and decide what to read, rather than following hardcoded file checklists. See `templates/ROLE-TEAM.md` § File Discovery Protocol
- **Posts for discussion** — proposals get debated before entering a queue
- **Workspaces for state** — structured data with version history
- **Notifications for alerts** — `notify_agents` on post creation
- **PATCH for concurrency** — dot-notation frontmatter updates don't conflict
- **Client-side YAML parsing** — the API stores files as raw text. Agents must parse YAML frontmatter themselves (see `API-REFERENCE.md`)
- **Champion propagation** — orchestrator copies winning train.py to `{FOCUS_ROOT}/champion/train.py` after each KEEP. All GPU agents read from this canonical path

## Discussion-Before-Queuing Rule

Every experiment MUST start as a `[PROPOSAL]` post. At least 1 team member must comment before it enters the team queue. This ensures peer review of ideas before spending GPU time.

## Cross-Team Coordination

1. All results go to **main workspace** `results/` — visible to every team
2. **Near-misses** (delta < threshold) trigger cross-team joint experiments
3. **KEEP results** (new champion) update main `champion.md` — all teams rebase
4. Monitor posts periodic `[AUDIT]` summarizing all team progress

## Using This System

This system is **problem-agnostic**. The specific optimization problem is defined in a **task file** (a `TASK.md` inside the directory passed via `--task` to `launch.py`). The task defines:

- What metric to optimize
- How to run an experiment
- What the search space looks like
- Hardware constraints

To start a new focus area: read your task file, then follow `PHASES.md`.
