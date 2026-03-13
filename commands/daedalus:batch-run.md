---
description: Launch multiple experiments distributed across hosts
---

# /daedalus:batch-run

Launch a battery of experiments in parallel, optionally distributed across multiple hosts via round-robin.

## What to do

1. Use `daedalus_list_experiments` to find all DRAFT experiments, or ask the user which experiments to launch
2. If multiple hosts are available, check `runner_config.yaml` for host names
3. Use `daedalus_batch_run` with the list of experiment IDs and hosts
4. Report which experiments launched and which failed
5. **Immediately start monitoring** — launch one background watchdog Agent per experiment

## Monitoring After Launch

After batch launch, do NOT stop and wait for user input. Immediately start monitoring:

```
Agent(
  description="Watch exp_ID",
  prompt="Monitor experiment exp_ID. Do 3-5 cycles: (1) daedalus_poll_experiment, (2) daedalus_get_experiment_logs(tail=30), (3) check for NaN loss, OOM, process death. If completed or problem, return immediately. Otherwise sleep 60 and repeat. After 5 cycles return progress report.",
  subagent_type="general-purpose",
  run_in_background=true
)
```

Launch one watchdog per experiment. While watchdogs run, do productive work (reflect on previous results, search papers, prepare next configs).

NEVER use `Bash(sleep N)` or inline polling. NEVER use CronCreate.

## Example

User: "Launch all my pending experiments on both GPU servers"

1. List DRAFT experiments: [exp_001, exp_002, exp_003, exp_004]
2. Read runner_config.yaml: hosts: gpu-a100, gpu-h100
3. Call `daedalus_batch_run(exp_ids=["exp_001","exp_002","exp_003","exp_004"], hosts=["gpu-a100","gpu-h100"])`
4. Result: exp_001->gpu-a100, exp_002->gpu-h100, exp_003->gpu-a100, exp_004->gpu-h100
5. "All 4 experiments launched."
6. [Launch 4 background watchdog Agents, one per experiment]
7. While monitoring, search for relevant papers or prepare analysis.
