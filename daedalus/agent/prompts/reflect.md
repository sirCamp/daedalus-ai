You are analyzing the results of a completed experiment.

## Instructions

1. Compare actual results against the hypothesis predictions.
2. For each prediction, state whether it was confirmed, partially confirmed, or rejected.
3. Identify **surprises** — results you did not expect.
4. Analyze **why** the results occurred (not just what happened).
5. Propose 2-3 concrete next steps based on this analysis.

## Analysis Framework

- Was the hypothesis confirmed? (confirmed / partial / rejected)
- What changed relative to baseline? (deltas, significance)
- Are there trade-offs? (one metric improved, another degraded?)
- What does this tell us about the underlying mechanism?
- What is the most informative next experiment?

## Output Format

```json
{
  "hypothesis_confirmed": "confirmed|partial|rejected",
  "analysis": "Detailed analysis of what happened and why",
  "surprise": "What was unexpected, if anything",
  "next_suggestions": [
    "Concrete next step 1",
    "Concrete next step 2"
  ],
  "confidence": "How confident are you in this analysis (high/medium/low)"
}
```
