---
description: Monitor running experiments and report when they complete
---

# /daedalus:watch

Start monitoring running experiments with background watchdog agents.

## IMPORTANT

- NEVER use `Bash(sleep N)` in the main conversation to wait — it blocks everything
- NEVER use CronCreate or any invented polling mechanism
- ONLY use Claude Code's `Agent` tool with `run_in_background=true`

## What This Command Does

1. List running experiments via `daedalus_list_experiments(status="running")`
2. Estimate remaining time for each experiment
3. Choose watchdog strategy: **short** (< 30 min ETA) or **long** (> 30 min ETA)
4. Launch one background watchdog Agent per experiment
5. Do productive work while watchdogs run
6. When a watchdog returns, show status, react, and relaunch if needed

## Two Watchdog Strategies

**Short watch** (ETA < 30 min, or interactive monitoring):

```
Agent(
  description="Watch exp_abc123",
  prompt="Monitor experiment exp_abc123. Do 3-5 cycles of: (1) call daedalus_poll_experiment with exp_id exp_abc123, (2) call daedalus_get_experiment_logs with exp_id exp_abc123 and tail=30, (3) check for problems (NaN loss, OOM, process died). If completed or problem detected, return immediately with a report. Otherwise sleep 60 and repeat. After 5 cycles, return a progress report with current step, loss, and ETA.",
  subagent_type="general-purpose",
  run_in_background=true
)
```

**Long watch** (ETA > 30 min, overnight, or unattended):

```
Agent(
  description="Watch exp_abc123 (long)",
  prompt="Monitor experiment exp_abc123 which has an ETA of ~3 hours. First sleep 9000 (2.5h) to avoid wasting poll cycles. Then do poll cycles every 60s until completed or problem detected (NaN loss, OOM, process died). On completion or problem, return immediately with a full report. Max 60 poll cycles after the initial sleep.",
  subagent_type="general-purpose",
  run_in_background=true
)
```

For long watch, calculate the initial sleep as ~80% of the ETA in seconds. The watchdog sleeps first, then polls until completion — no relaunch needed.

## When to Use Which

| Strategy | ETA | Use case | Relaunch needed? |
|----------|-----|----------|-----------------|
| Short | < 30 min | Interactive, want frequent updates | Yes, after each return |
| Long | > 30 min | Overnight, AFK, batch runs | No, covers full duration |

When in doubt, use **long watch** — it covers completion without requiring relaunches.

## Watchdog Returns One Of

- **COMPLETED** — final metrics, ready for analysis
- **ALERT** — problem detected, diagnosis, suggested fix
- **PROGRESS** — (short watch only) still running, current step/loss/ETA

## Cycle (short watch)

```
list running experiments
  |
launch one short watchdog per experiment
  |
do productive work (reflect, papers, prepare next exp)
  |
watchdog returns --> show status to user
  |
react to alerts / relaunch watchdogs for still-running experiments
  |
repeat
```

## Cycle (long watch)

```
list running experiments, estimate ETA
  |
launch one long watchdog per experiment (with initial sleep)
  |
do productive work / user goes AFK
  |
watchdog returns with COMPLETED or ALERT (hours later)
  |
analyze results / react to alerts
```

## Example

```
User: /daedalus:watch

Agent: [daedalus_list_experiments(status="running")]
       2 running experiments.

       exp_006: step 50/375, ~45min remaining — using short watch
       exp_007: step 10/375, ~3.5h remaining — using long watch

       [Agent(description="Watch exp_006", ...short..., run_in_background=true)]
       [Agent(description="Watch exp_007 (long)", ...sleep 10000 then poll..., run_in_background=true)]

       While they run, let me search for papers...
       [daedalus_search_papers, daedalus_save_note]

       [short watchdog exp_006 returns: COMPLETED]
       exp_006: completed! eval_loss=0.28, accuracy=0.94
       [daedalus_add_reflection, daedalus_save_note]

       exp_007 long watchdog still running — it will report when done.

--- (2.5 hours later, long watchdog returns) ---

       [long watchdog exp_007 returns: COMPLETED]
       exp_007: completed! eval_loss=0.31, accuracy=0.91
       [daedalus_compare_experiments, daedalus_add_reflection]
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
