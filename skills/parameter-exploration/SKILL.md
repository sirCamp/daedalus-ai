---
name: parameter-exploration
description: Data-driven hyperparameter exploration strategies for ML experiments
origin: daedalus
version: 1.0.0
---

# Parameter Exploration

## When to Activate

- User wants to optimize hyperparameters
- After baseline experiment completes successfully
- When multiple experiments exist and patterns can be extracted
- User asks "what should I try next?"

## Using suggest_next

The `daedalus_suggest_next` tool analyzes experiment history and suggests parameter values:

```
daedalus_suggest_next(
  target_metric="eval.acc",
  higher_is_better=true,
  focus_param="learning_rate"  # optional: focus on one param
)
```

Returns up to 5 suggestions, each with strategy name and rationale.

## Exploration Strategies

### 1. Baseline (no experiments yet)
Returns default values from the scripts registry. Always start here.

### 2. Explore Neighbor (1 data point)
With only one experiment, suggests small variations (e.g., 2x and 0.5x the current value).

### 3. Bisect (2+ data points)
When two experiments have different performance, bisects the parameter space between the best and second-best to find the optimum.

### 4. Trend Extrapolate (3+ monotonic points)
If performance improves monotonically with a parameter, extrapolates the trend to suggest the next value.

### 5. Untried Choice (categorical params)
For parameters with discrete choices (e.g., optimizer: adam/sgd/adamw), suggests values that haven't been tried yet.

### 6. Unexplored Range (never-varied params)
Identifies parameters that have never been changed across experiments and suggests exploring them.

## Strategy Selection

The tool automatically picks strategies based on available data:
- 0 experiments → baseline
- 1 experiment → explore_neighbor for all params
- 2+ experiments → bisect + untried_choice + unexplored_range
- 3+ monotonic → trend_extrapolate

## Best Practices

1. **Don't blindly follow suggestions** — They are data-driven heuristics, not optimal solutions
2. **One parameter at a time** — Even if multiple suggestions exist, run them separately
3. **Set stopping criteria** — Decide in advance when to stop (e.g., < 0.1% improvement)
4. **Consider interactions** — After single-param sweeps, try combinations of best values
5. **Watch for overfitting** — Compare train vs eval metrics, not just eval alone
6. **Budget awareness** — Each experiment costs compute. Prioritize high-impact parameters

## Parameter Priority Order

When starting a new project, explore parameters in this order:
1. **Learning rate** — Highest impact, always explore first
2. **Batch size** — Affects both speed and quality
3. **Architecture choices** — Model size, layers, etc.
4. **Regularization** — Dropout, weight decay, etc.
5. **Schedule** — Warmup, decay, epochs
6. **Data augmentation** — If applicable
