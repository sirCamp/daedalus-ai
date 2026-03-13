---
description: Create or review a structured research plan with prioritized steps
---

# /daedalus:plan

Manage your research plan using Daedalus MCP tools.

## What This Command Does

1. Checks if a plan already exists (`daedalus_get_plan`)
2. If no plan: reviews context, suggests steps, creates a plan (`daedalus_create_plan`)
3. If plan exists: shows progress, autonomy mode, guardrails, next actionable step
4. Optionally updates step status, switches autonomy mode, or adds new steps

## When to Use

- **Starting a research direction** — create a structured plan before experimenting
- **Between experiments** — check what's next, update progress
- **After reflection** — mark steps done, add new steps based on findings
- **Adjusting guardrails** — change max_experiments, pause/resume, toggle autonomy mode

## Steps

1. Get full research context: `daedalus_get_context(mode="full")`
2. Check existing plan: `daedalus_get_plan`
3. If no plan exists:
   - Analyze the research goals from program.md
   - Search relevant papers: `daedalus_search_papers`
   - Create a plan with 3-7 steps, each with:
     - **Description**: what to test (one variable)
     - **Rationale**: why it matters
     - **Expected outcome**: what we predict
     - **Priority**: 1 (highest) to 5
     - **Dependencies**: which steps must complete first
     - **requires_confirmation**: true for expensive/risky steps
   - Set autonomy mode: `autonomous` (default) or `supervised`
   - Set guardrails: `max_experiments`, `max_consecutive_failures`
   - Save the plan: `daedalus_create_plan`
4. If plan exists:
   - Show current progress (done/failed/total)
   - Show autonomy mode and guardrail status
   - Highlight next actionable step
   - Ask if the user wants to:
     - Update any steps or add new ones
     - Switch autonomy mode: `daedalus_update_plan(autonomy="autonomous")`
     - Pause/resume: `daedalus_update_plan(paused=true/false)`

## Autonomy Modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `autonomous` (default) | Launches directly — Claude Code chat is already human-in-the-loop | Normal usage |
| `supervised` | Adds explicit code-level gate before each launch | Extra caution (CLI `run_loop`) |

Switch mode: `daedalus_update_plan(autonomy="autonomous")` or `daedalus_update_plan(autonomy="supervised")`

## Guardrails

| Guardrail | Default | Effect |
|-----------|---------|--------|
| `max_experiments` | unlimited | Auto-pause after N total experiments |
| `max_consecutive_failures` | 2 | Auto-pause after N failures in a row |
| Per-step `requires_confirmation` | false | Block launch even in autonomous mode |

When a guardrail triggers, the plan is **paused**. Resume with `daedalus_update_plan(paused=false)`.

## Example

```
User: /daedalus:plan

Agent: Let me check the current research plan...

Research Plan: Optimize calibration for QRC model
Mode: autonomous | Experiments: 3/10 | Failures: 0/2
Progress: 2/5 done

S01 [done] — Baseline with default parameters (exp_003)
S02 [done] — Test learning rate 1e-5 (exp_004)
S03 [pending] — Test IDK penalty -0.5 [NEXT]
S04 [pending] — Add FiSCoRe agreement (depends on S03)
S05 [pending] — Final evaluation battery [requires confirmation]

Next actionable: S03 — Test IDK penalty -0.5
Shall I design this experiment?

User: si, e poi mettilo in autonomous che vado a dormire

Agent: [designs and launches experiment with plan_step_id="S03"]
       [calls daedalus_update_plan(autonomy="autonomous")]
       Done. Plan switched to autonomous mode. I'll continue with the
       remaining steps overnight. Guardrails: max 10 experiments, pause
       after 2 consecutive failures.
```

## Related Commands

- `/daedalus:experiment` — Design and launch a single experiment
- `/daedalus:research-status` — Full project overview
- `/daedalus:analyze` — Analyze completed results
