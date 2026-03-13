You are conducting a targeted literature review for the research project.

## Instructions

1. Based on the current research direction and recent experiment results, identify knowledge gaps.
2. Suggest 3-5 search queries for finding relevant papers.
3. For papers you find, extract:
   - Key findings relevant to our work
   - Specific techniques or hyperparameters we could adopt
   - Results that confirm or contradict our observations

## Focus Areas

- Papers that address the specific problem we're solving
- Papers with techniques we haven't tried yet
- Recent papers (last 12 months) that may change our approach
- Papers that report negative results similar to ours

## Output Format

```json
{
  "search_queries": ["query 1", "query 2"],
  "papers_found": [
    {
      "arxiv_id": "XXXX.XXXXX",
      "relevance_note": "Why this matters for us",
      "key_findings": ["Finding 1"],
      "actionable_insight": "What we should do based on this"
    }
  ]
}
```
