---
description: Compare two or more experiments side by side
---

# Compare Experiments

Compare experiment configs and results to understand what changed and what improved.

## What to do

1. If the user specifies experiment IDs, use those. Otherwise, use `daedalus_list_experiments` and pick the most relevant (e.g., best vs second-best, or latest vs baseline).
2. Call `daedalus_compare_experiments` with the two experiment IDs
3. Present a clear table showing:
   - Config differences (what changed)
   - Metric deltas (what improved/regressed)
   - Percentage change for each metric
4. Provide a brief analysis: which experiment is better and why

## Example

User: "Compare my last two experiments"

1. List experiments → exp_005 (completed), exp_004 (completed)
2. `daedalus_compare_experiments(exp_a="exp_004", exp_b="exp_005")`
3. Show: "exp_005 changed learning_rate from 2e-5 to 5e-5. Result: eval_loss improved 0.342 → 0.298 (-12.9%), accuracy improved 0.891 → 0.923 (+3.6%)"
