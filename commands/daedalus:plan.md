---
description: Create or review a structured research plan with prioritized steps
---

# /daedalus:plan

Manage your research plan using Daedalus MCP tools.

## What This Command Does

1. Checks if a plan already exists (`daedalus_get_plan`)
2. If no plan: runs literature review, asks about approach level, creates a plan
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
   a. **Literature review** — search for relevant work before planning:
      - `daedalus_search_papers` for the task/model/technique
      - Add key papers: `daedalus_add_paper` with relevance notes
      - Look for recommended hyperparameters, known baselines, common pitfalls
      - This informs step design: realistic expected outcomes, proven techniques
   b. **Ask the user about approach level**:
      > What level of depth do you want?
      > - **Quick exploration**: baselines only, compare models, pick the best. ~3-5 experiments.
      > - **Thorough tuning**: baselines + hyperparameter search (lr, LoRA rank, epochs). ~10-20 experiments.
      > - **Production-ready**: full tuning + full dataset + eval battery + multiple seeds. ~20-30+ experiments.
   c. Create a plan with steps matching the chosen approach:
      - Each step: description, rationale, expected outcome, priority (1-5), dependencies
      - Mark expensive steps with `requires_confirmation: true`
      - Set `autonomy="autonomous"` (default)
      - Set guardrails: `max_experiments`, `max_consecutive_failures`
      - Save: `daedalus_create_plan`
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

Agent: [calls daedalus_get_context, daedalus_get_plan]
       No plan found. Let me search for relevant work first.

       [calls daedalus_search_papers("LoRA fine-tuning text classification")]
       Found 5 relevant papers. Adding key ones to library.
       [calls daedalus_add_paper for top 3]

       What level of depth do you want?
       - Quick exploration (~3-5 experiments)
       - Thorough tuning (~10-20 experiments)
       - Production-ready (~20-30+ experiments)

User: thorough tuning

Agent: [creates plan with 7 steps, max_experiments=20, autonomous mode]

       Research Plan: Optimize fine-tuning for text classification
       Mode: autonomous | Max experiments: 20 | Max failures: 2

       S01 [pending] — Baseline with default parameters [NEXT]
       S02 [pending] — Test learning rate 1e-5 (depends on S01)
       S03 [pending] — Test batch size 32 (depends on S01)
       S04 [pending] — Test LoRA rank 64 (depends on S01)
       S05 [pending] — HP tuning: lr grid search (depends on S02)
       S06 [pending] — HP tuning: batch size sweep (depends on S03)
       S07 [pending] — Final evaluation battery [requires confirmation]

       Ready to start with S01. Use /daedalus:experiment to launch.
```

## Related Commands

- `/daedalus:experiment` — Design and launch a single experiment
- `/daedalus:research-status` — Full project overview
- `/daedalus:analyze` — Analyze completed results
- `/daedalus:literature-review` — Search papers and build library
