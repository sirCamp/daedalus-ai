You are designing the next experiment to test a specific hypothesis.

## Instructions

1. Review the hypothesis and its predictions.
2. Propose the **minimal config change** needed to test it.
3. Show the exact diff from the baseline config.
4. Estimate compute cost (GPU hours, approximate USD).
5. Identify potential confounds — what else could explain the result?

## Constraints

- Change ONE variable at a time (unless the hypothesis explicitly requires multiple changes).
- Keep all other settings identical to the baseline.
- If the change requires code modifications, specify exactly what to change.

## Output Format

```json
{
  "baseline_id": "exp_XXX",
  "config_diff": {"key": {"from": "old", "to": "new"}},
  "code_changes": [],
  "estimated_hours": 4.0,
  "estimated_cost_usd": 5.0,
  "confounds": ["Possible confound 1"],
  "ready_to_launch": true
}
```
