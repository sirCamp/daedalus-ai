---
name: experiment-workflow
description: Hypothesis-driven experiment lifecycle management with Daedalus
origin: daedalus
version: 1.0.0
---

# Experiment Workflow

## When to Activate

- User wants to run ML experiments
- User asks to train a model or evaluate results
- User wants to compare different configurations
- User mentions hypothesis, ablation, or hyperparameter search

## Experiment Lifecycle

Every experiment follows this state machine:

```
DRAFT → RUNNING → COMPLETED → ANALYZED
                → FAILED
                → ABANDONED
```

### 1. Draft — Design the Experiment

Before creating an experiment, always:

1. **Check context**: `daedalus_get_context(mode="design")` to see what's been tried
2. **Check scripts**: `daedalus_list_scripts` to see available scripts and valid parameters
3. **Set a baseline**: Pick the most relevant completed experiment as `baseline_id`
4. **Make predictions**: What metric will change? By how much? Why?

```
daedalus_create_experiment(
  hypothesis_statement="Increasing learning rate from 1e-5 to 3e-5 will improve convergence speed",
  rationale="Current LR is conservative; model shows no signs of instability",
  script="train.py",
  script_args={"learning_rate": 3e-5},
  baseline_id="exp_001",
  predictions=[{"metric": "eval.loss", "direction": "lower", "confidence": 0.7}]
)
```

### 2. Running — Launch and Monitor

```
daedalus_launch_experiment(exp_id="exp_002")
daedalus_poll_experiment(exp_id="exp_002")
```

For long-running jobs, use the watcher:
```bash
daedalus watch --exp-id exp_002 --poll-interval 300
```

### 3. Completed — Analyze Results

Always compare against baseline:
```
daedalus_compare_experiments(exp_a="exp_001", exp_b="exp_002")
```

### 4. Analyzed — Record Reflection

```
daedalus_add_reflection(
  exp_id="exp_002",
  analysis="LR 3e-5 reduced eval loss by 12% but increased variance...",
  hypothesis_confirmed="partial",
  next_suggestions=["Try LR 2e-5 as middle ground", "Add LR warmup"]
)
```

## Configuration — daedalus.yaml

Every project needs a `daedalus.yaml` with:

```yaml
project:
  name: my-research
  description: "Fine-tuning experiments"

runner:
  type: local  # or ssh
  python: python

scripts:
  train:
    path: train.py
    description: "Main training script"
    parameters:
      learning_rate:
        type: float
        default: 1e-5
        range: [1e-6, 1e-3]
      batch_size:
        type: int
        default: 8
        choices: [4, 8, 16, 32]
```

## Best Practices

1. **Name experiments meaningfully** — The ID is auto-generated, but the hypothesis IS the name
2. **One variable at a time** — If you change LR and batch size together, you learn nothing
3. **Record negative results** — Failed experiments prevent others from repeating mistakes
4. **Use tags** — Tag experiments by theme: `["lr-sweep", "regularization", "architecture"]`
5. **Keep the program updated** — `program.md` should reflect current research direction
