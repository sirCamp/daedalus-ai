---
description: Monitor running experiments and report when they complete
---

# /daedalus:watch

Start monitoring running experiments with watchdog agents.

## What This Command Does

1. Checks for running experiments in the ledger
2. Launches a **background watchdog** per experiment
3. Watchdog does 3-5 polls (~5 min), then returns a report
4. Main agent shows status, does productive work, relaunches watchdog
5. Cycle repeats until all experiments complete or fail

## Watchdog Agent

Each watchdog runs in background and does 3-5 polls with `sleep 60` between them:

- **COMPLETED** → returns final metrics
- **ALERT** (NaN loss, OOM, process died) → stops early, returns diagnosis
- **PROGRESS** (still running after 3-5 polls) → returns step, loss, ETA

## Cycle

```
launch watchdog(s) in background
  ↓
wait for watchdog return (~5 min)
  ↓
show status to user
  ↓
react to alerts / do productive work
  ↓
relaunch watchdog(s) for still-running experiments
  ↓
repeat
```

## Example

```
User: /daedalus:watch

Agent: 2 running experiments. Launching watchdogs.
       [background watchdog for exp_006]
       [background watchdog for exp_007]

       [~5 min later, watchdogs return]
       exp_006: step 50/375, loss=2.31, ~45min — healthy
       exp_007: step 30/375, loss=3.12, ~50min — healthy

       Let me search for relevant papers while they train...
       [searches papers, saves notes]

       Relaunching watchdogs.
       [background watchdog for exp_006]
       [background watchdog for exp_007]

       [~5 min later]
       exp_006: step 110/375, loss=1.85, ~30min — healthy
       exp_007: ALERT — loss NaN at step 62. Stopping.
       Diagnosis: likely lr too high. Preparing retry with lr=1e-4.

       [relaunches only exp_006 watchdog, fixes and relaunches exp_007]
       ...

       [watchdog returns]
       exp_006: COMPLETED! cer=0.12, exact_match=0.58
       Running analysis...
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
