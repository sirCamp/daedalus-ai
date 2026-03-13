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
8. Polls periodically until completion (does productive work between polls)
9. Analyzes results and records reflection

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
- Auto-updates plan steps on completion/failure

## Monitoring

After launch, `launch_experiment` returns monitoring hints:
- `suggested_poll_interval_seconds` — how often to poll (60s SSH, 10s local)
- `while_waiting` — productive tasks to do between polls

The agent polls with `poll_experiment` and works on other tasks in between.

## Example

```
User: /daedalus:experiment
Agent: Let me check the current state of your research...
       [calls daedalus_get_context, daedalus_list_experiments, daedalus_get_plan]
       You have 3 completed experiments. Best so far: exp_003 (acc=0.92).
       Plan step S03 is next: "Test IDK penalty -0.5"
       Shall I design this experiment?
```
