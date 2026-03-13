---
description: Start the experiment workflow — design, launch, and monitor an ML experiment
---

# /daedalus:experiment

Run the full experiment workflow using Daedalus MCP tools.

## What This Command Does

1. Checks the current research context (`daedalus_get_context`)
2. Lists available scripts and parameters (`daedalus_list_scripts`)
3. Reviews recent experiments to pick a baseline
4. Checks if a plan exists — if so, picks the next actionable step
5. Asks what hypothesis to test (or uses the plan step)
6. Creates a draft experiment with predictions
7. Launches with `plan_step_id` if following a plan (enables auto-bookkeeping)
8. **Immediately starts monitoring** with background watchdog agents
9. Does productive work while experiments train (reflect, search papers, prepare next)
10. Analyzes results and records reflection on completion

## When to Use

- Starting a new training run
- Iterating on hyperparameters
- Running an ablation study
- Testing a new model configuration
- Executing the next step in a research plan

## How It Works

The research-assistant agent orchestrates the workflow:
- Uses Daedalus MCP tools for experiment management
- Uses Claude Code tools (Read, Edit, Bash) for code fixes
- Follows hypothesis-driven methodology (one variable at a time)
- Compares results against baseline automatically
- Auto-syncs files to remote for SSH runners
- Auto-updates plan steps on completion/failure (when using `plan_step_id`)

## Monitoring

After launch, immediately start monitoring with background watchdog agents:

1. Estimate remaining training time from launch response hints
2. Choose strategy: **short watch** (ETA < 30 min) or **long watch** (ETA > 30 min)
3. Launch one background Agent per experiment (see `/daedalus:watch` for details)
4. Do productive work while watchdogs run
5. Short watch: when watchdog returns with PROGRESS, relaunch. Long watch: watchdog sleeps first, then polls until done — no relaunch needed.

For overnight or long runs, always use **long watch** with an initial sleep of ~80% of ETA. This ensures the watchdog covers the full training without needing anyone to relaunch it.

NEVER use `Bash(sleep N)` in the main conversation. NEVER use CronCreate.

## Auto-Bookkeeping

When you pass `plan_step_id` to `launch_experiment`, Daedalus automatically:
- Links the experiment to the plan step
- Marks the step running/done/failed when the experiment transitions
- Resets or increments failure counters
- Copies reflection notes to the plan step (via `add_reflection`)

No manual `update_plan_step` needed for these transitions.

## Example

```
User: /daedalus:experiment
Agent: [calls daedalus_get_context, daedalus_list_experiments, daedalus_get_plan]
       You have 3 completed experiments. Best so far: exp_003 (acc=0.92).
       Plan step S03 is next: "Test learning rate 5e-5"

       [designs, creates, launches experiment with plan_step_id="S03"]
       [launches background watchdog Agent]
       [searches papers while waiting]

       [watchdog returns: PROGRESS]
       exp_006: step 120/375, loss=1.85, ~30min remaining — healthy

       [relaunches watchdog, prepares next experiment config]

       [watchdog returns: COMPLETED]
       exp_006: completed! eval_loss=0.28, accuracy=0.94
       [calls daedalus_add_reflection, daedalus_save_note]
```
