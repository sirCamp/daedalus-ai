---
description: Monitor running experiments and report when they complete
---

# /daedalus:watch

Start monitoring running experiments with background watchdog agents.

## IMPORTANT

- NEVER use `Bash(sleep N)` to wait — it blocks everything
- NEVER use CronCreate or any invented polling mechanism
- ONLY use Claude Code's `Agent` tool with `run_in_background=true`

## What This Command Does

1. List running experiments via `daedalus_list_experiments(status="running")`
2. Launch one background watchdog Agent per experiment
3. Do productive work while watchdogs run
4. When a watchdog returns, show status, react, and relaunch if needed

## How to Launch a Watchdog

Use Claude Code's Agent tool:

```
Agent(
  description="Watch exp_abc123",
  prompt="Monitor experiment exp_abc123. Do 3-5 cycles of: (1) call daedalus_poll_experiment with exp_id exp_abc123, (2) call daedalus_get_experiment_logs with exp_id exp_abc123 and tail=30, (3) check for problems (NaN loss, OOM, process died). If completed or problem detected, return immediately with a report. Otherwise sleep 60 and repeat. After 5 cycles, return a progress report with current step, loss, and ETA.",
  subagent_type="general-purpose",
  run_in_background=true
)
```

## Watchdog Returns One Of

- **COMPLETED** — final metrics, ready for analysis
- **ALERT** — problem detected, diagnosis, suggested fix
- **PROGRESS** — still running, current step/loss/ETA

## Cycle

```
list running experiments
  |
launch one background watchdog Agent per experiment
  |
do productive work (reflect, papers, prepare next exp)
  |
watchdog returns --> show status to user
  |
react to alerts / relaunch watchdogs for still-running experiments
  |
repeat
```

## Example

```
User: /daedalus:watch

Agent: [daedalus_list_experiments(status="running")]
       2 running experiments.

       [Agent(description="Watch exp_006", ..., run_in_background=true)]
       [Agent(description="Watch exp_007", ..., run_in_background=true)]

       While they check, let me search for papers...
       [daedalus_search_papers, daedalus_save_note]

       [watchdog exp_006 returns: PROGRESS]
       exp_006: step 50/375, loss=2.31, ~45min — healthy

       [watchdog exp_007 returns: ALERT]
       exp_007: ALERT — loss NaN at step 23. Stopping.
       Preparing retry with lr=1e-4.

       [relaunch watchdog for exp_006, fix and relaunch exp_007]
       ...
```

## Arguments

Watch a specific experiment:
```
/daedalus:watch exp_006
```

Watch all running experiments:
```
/daedalus:watch
```
