---
description: Launch multiple experiments distributed across hosts
---

# Batch Run

Launch a battery of experiments in parallel, optionally distributed across multiple hosts via round-robin.

## What to do

1. Use `daedalus_list_experiments` to find all DRAFT experiments, or ask the user which experiments to launch
2. If multiple hosts are available, check `runner_config.yaml` for host names
3. Use `daedalus_batch_run` with the list of experiment IDs and hosts
4. Report which experiments launched and which failed
5. Suggest using `/daedalus:watch` to monitor progress

## Example

User: "Launch all my pending experiments on both GPU servers"

1. List DRAFT experiments → [exp_001, exp_002, exp_003, exp_004]
2. Read runner_config.yaml → hosts: gpu-a100, gpu-h100
3. Call `daedalus_batch_run(exp_ids=["exp_001","exp_002","exp_003","exp_004"], hosts=["gpu-a100","gpu-h100"])`
4. Result: exp_001→gpu-a100, exp_002→gpu-h100, exp_003→gpu-a100, exp_004→gpu-h100
5. "All 4 experiments launched. Use `/daedalus:watch` to monitor them."
