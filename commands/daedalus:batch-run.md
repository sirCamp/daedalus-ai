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

1. Estimate ETA for each experiment from launch response hints
2. Use **long watch** for experiments with ETA > 30 min (typical for batch runs)
3. Launch one background watchdog Agent per experiment

```
Agent(
  description="Watch exp_ID (long)",
  prompt="Monitor experiment exp_ID which has an ETA of ~3 hours. First sleep 9000 (2.5h) to avoid wasting poll cycles. Then do poll cycles every 60s until completed or problem detected (NaN loss, OOM, process died). On completion or problem, return immediately with a full report. Max 60 poll cycles after the initial sleep.",
  subagent_type="general-purpose",
  run_in_background=true
)
```

For short experiments (ETA < 30 min), use the short watch pattern instead (3-5 cycles, no initial sleep). See `/daedalus:watch` for details.

While watchdogs run, do productive work (reflect on previous results, search papers, prepare next configs).

NEVER use `Bash(sleep N)` in the main conversation. NEVER use CronCreate.

## Example

User: "Launch all my pending experiments on both GPU servers"

1. List DRAFT experiments: [exp_001, exp_002, exp_003, exp_004]
2. Read runner_config.yaml: hosts: gpu-a100, gpu-h100
3. Call `daedalus_batch_run(exp_ids=["exp_001","exp_002","exp_003","exp_004"], hosts=["gpu-a100","gpu-h100"])`
4. Result: exp_001->gpu-a100, exp_002->gpu-h100, exp_003->gpu-a100, exp_004->gpu-h100
5. "All 4 experiments launched."
6. [Launch 4 background watchdog Agents, one per experiment]
7. While monitoring, search for relevant papers or prepare analysis.
