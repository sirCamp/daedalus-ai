---
description: Show full research status — experiments, progress, and next steps
---

# /daedalus:research-status

Get a comprehensive overview of your research project.

## What This Command Does

1. Loads the research program (`program.md`)
2. Summarizes experiment history by status
3. Highlights best results so far
4. Shows unanalyzed experiments
5. Lists currently running experiments
6. Suggests next steps based on history

## When to Use

- Starting a new work session
- Getting oriented after a break
- Before deciding what to do next
- Reviewing overall progress

## How It Works

```
User: /daedalus:research-status
Agent: [calls daedalus_get_context(mode="full"), daedalus_list_experiments]

       Project: GRPO Calibration
       Goal: Improve calibration without losing accuracy

       Experiments: 12 total
       - 8 completed, 6 analyzed
       - 2 running (exp_011, exp_012)
       - 1 failed, 1 abandoned

       Best result: exp_009 (acc=0.95, calibration=0.87)

       Unanalyzed: exp_010 (completed 2h ago)

       Suggested next: Try beta=0.1 for KL regularization
```
