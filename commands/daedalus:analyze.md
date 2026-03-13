---
description: Analyze completed experiment results and record reflections
---

# /daedalus:analyze

Analyze results from completed experiments.

## What This Command Does

1. Lists completed experiments that haven't been analyzed yet
2. Fetches full results and compares against baseline
3. Determines if hypothesis was confirmed, partially confirmed, or rejected
4. Records structured reflection with analysis and next steps
5. Suggests what to try next based on findings

## When to Use

- After an experiment finishes
- When reviewing a batch of completed experiments
- To generate a summary of research progress

## How It Works

```
User: /daedalus:analyze
Agent: [calls daedalus_list_experiments(status="completed")]
       Found 2 unanalyzed experiments: exp_004, exp_005.

       exp_004 vs baseline exp_003:
       - eval.acc: 0.92 → 0.94 (+2.2%) ✓
       - eval.loss: 0.31 → 0.28 (-9.7%) ✓
       Hypothesis CONFIRMED: higher LR improved convergence.

       Next suggestion: try LR warmup to stabilize early training.
```
