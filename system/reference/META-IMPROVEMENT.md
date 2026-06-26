---
name: meta-improvement
description: Guide for the orchestrator to critically review and improve the multi-agent system
---

# Meta-Improvement

Every 3 execution cycles, the orchestrator pauses to critically examine how the agent team is operating and makes one concrete improvement. This is not an automated diagnostic — it requires genuine judgment about what is and isn't working.

## What This Is Not

Meta-improvement is not:
- Running a script and applying whatever it suggests
- Writing a report or checklist
- Checking boxes or logging observations without changing anything
- Declaring "system operating normally" without evidence

If you finish this step and no file has changed, you did not do meta-improvement.

## The Core Question

**"Is the team making the best possible use of its time and knowledge?"**

If the answer is anything other than a confident yes, find the most significant gap and fix it.

---

## Step 1 — Read the Evidence

Before forming any opinion, read what actually happened:

- `logs/experiments.jsonl` and individual `agents/*/cycle_result.json` files — what did agents try, what were the outcomes, is the metric improving or flat?
- `logs/sessions.jsonl` — did all agents complete? did any time out or fail silently?
- Each team's `queue.md` — are there pending experiments? are queues going empty? are the same ideas appearing repeatedly?
- Workshop posts — are agents posting substantive [RESULT] and [INSPIRATION] content, or formulaic boilerplate? are ideas being picked up across teams?
- `champion/SOURCE` — when did the champion last improve? how many cycles ago?
- `system/templates/ROLE-ANALYST.md` and `ROLE-GPU.md` — what are agents actually being asked to do?

Read the real files. Do not rely on memory or assumptions.

---

## Step 2 — Form a Diagnosis

Identify the single most significant dysfunction. Be specific and honest. Some things to watch for:

**Exploration problems**
- Analysts keep proposing small variations on the same idea — no diversity
- Agents are ignoring the hardest cases and only working on easy wins
- The same dead ends are being re-attempted by different agents

**Knowledge sharing problems**
- A KEEP happened several cycles ago but other agents haven't built on it
- [RESULT] posts are too thin to be useful — no mechanism explanation, no suggested follow-ups
- Teams are working in isolation when they should be learning from each other

**Pipeline problems**
- GPU agents are waiting for work because analyst queues ran dry
- Experiments are being proposed that are obviously redundant across teams
- Agents are spending time on infrastructure issues rather than experiments

**Quality problems**
- KEEP rate has been zero for many cycles — proposals are consistently weak
- Proposals lack a clear hypothesis about why they should improve the metric
- Agents aren't incorporating lessons from past failures

**Protocol problems**
- Agents are skipping steps in their role docs (often because instructions are unclear or too long)
- Role docs have accumulated contradictory or outdated instructions from past edits
- An earlier meta-improvement added something that turned out to add friction without benefit

Name the problem specifically. "Coordination seems weak" is not a diagnosis. "Team B has had 6 consecutive DISCARDs and has not looked at Team A's two recent KEEPs" is a diagnosis.

---

## Step 3 — Make One Targeted Change

Based on your diagnosis, make the most direct fix you can. Edit the relevant file — don't describe the change you would make, make it.

**What you can change:**
- `system/templates/ROLE-ANALYST.md` — how analysts propose and prioritize experiments
- `system/templates/ROLE-GPU.md` — how compute agents run experiments and share results
- `system/reference/*.md` — coordination protocols and shared guidelines
- `task/TASK.md` or `task/LAUNCH.md` — task clarity, hints, evaluation guidance
- Team `queue.md` files directly — seed experiments if queues are empty

**What you cannot change:**
- Agent workspaces (`agents/*/workspace/`)
- Logs (`logs/`)
- Champion code (`champion/`)
- Run metadata (`WORKSPACE_ID`, `run_metadata.json`)

**Principles for a good change:**
- It addresses the root cause, not a symptom
- It is specific — an agent reading the updated file will behave differently in a concrete way
- It does not add unnecessary steps or complexity
- If it fixes something, consider whether an earlier instruction caused the problem and remove it

One change only. If you see multiple problems, fix the most important one. Bundling changes makes it impossible to know what worked.

---

## Step 4 — Log What You Did and Why

Append a brief entry to `logs/meta_results.tsv`:

```
cycle   pattern_diagnosed           file_changed                    outcome
6       low_keep_rate               system/templates/ROLE-ANALYST.md applied
9       queue_empty                 agents/team_a/workspace/...     seeded_queue
12      duplicate_proposals         system/templates/ROLE-ANALYST.md applied
15      role_doc_contradiction      system/templates/ROLE-GPU.md     applied
```

Include a one-line note on what you changed and why. This record lets future meta-improvement steps see what was already tried.

---

## Judgment Heuristics

**If the champion hasn't improved in 5+ cycles:** The team is stuck. Look at whether proposals are genuinely diverse or converging on a local optimum. Consider whether the role docs are steering agents away from high-risk/high-reward ideas.

**If queues are repeatedly going empty:** Analysts aren't keeping up with GPU agents. Either increase proposal diversity expectations, or check whether analysts are getting stuck on infrastructure issues.

**If KEEP rate is high but metric gain per KEEP is small:** The team is making incremental progress but not exploring enough. The role docs may be too conservative — penalizing bold proposals.

**If agents are not building on each other's KEEPs:** Look at the [RESULT] post format. If posts don't include mechanism explanations and suggested follow-ups, other agents have nothing to build on.

**If the same experiment appears across multiple teams:** The deduplication guidance in ROLE-ANALYST.md is either missing or not being followed. Check whether it's clear and early in the instructions.

**If a recent meta-improvement made things worse:** Revert it. Read the current role doc, find the change you made, remove it, and try something different.

---

## Documented Patterns From Past Runs

### Task Specification Drift (2026-04-03)

All GPU agents used the wrong data split because example code in TASK.md contradicted the prose. Agents trusted the example over the text.

**Lesson:** When agents systematically make the same mistake, look at the role docs and task docs for an instruction that is ambiguous or contradictory. Fix the source, not the symptom.

### Orchestrator Autonomy Failure (2026-04-03)

Orchestrator paused after cycle 1 and asked "can you continue?" despite being told not to stop.

**Lesson:** Weak directives ("do not pause") are ignored. Strong prohibitions ("NEVER ASK PERMISSION — if you find yourself typing this, just continue") work better.

### Meta-Improvement Writing Reports Instead of Making Changes (2026-04-05)

After hundreds of cycles, meta_improvement/ contained thousands of identical report files. No role doc had ever been changed. The step ran but did nothing.

**Lesson:** The step is only complete when a file is different than it was before you started. If nothing changed, you did not do meta-improvement.
