---
description: Monitor running experiments and report when they complete
---

# /daedalus:watch

Start monitoring running experiments.

## What This Command Does

1. Checks for running experiments in the ledger
2. Polls their status at regular intervals
3. Reports completion or failure
4. Fetches and displays results when done
5. Suggests analysis steps

## When to Use

- After launching a long-running experiment
- To check on experiments launched earlier
- When you want to be notified of completion

## How It Works

```
User: /daedalus:watch
Agent: [runs daedalus watch --all --json-output in background]
       Monitoring 2 running experiments: exp_006, exp_007

       [later]
       exp_006 completed! Results:
       - eval.acc: 0.95
       - eval.loss: 0.22

       Want me to analyze the results?
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
