---
name: dataset-validation
description: Dataset inspection and validation for ML training pipelines
origin: daedalus
version: 1.0.0
---

# Dataset Validation

## When to Activate

- User provides a dataset for training
- Before any training run
- User asks about data quality or format
- Debugging training issues (bad data is the most common cause)

## Inspection First, Always

Before training on ANY dataset, run inspection:

```
daedalus_inspect_dataset(path="data/train.jsonl", sample_n=5000)
```

This returns:
- Row count and column names
- Per-column stats: dtype, nulls, uniques, top values
- Sample records (3 examples)
- Estimated token count and training cost

## Validation by Format

Daedalus supports 16 dataset formats:

| Format | Required Fields | Description |
|--------|----------------|-------------|
| `sft` | prompt, completion | Supervised fine-tuning |
| `chat` | messages[] | Chat format (role/content) |
| `dpo` | prompt, chosen, rejected | Direct preference optimization |
| `grpo` | prompt, chosen | Group relative policy optimization |
| `classification` | text, label | Text classification |
| `nli` | premise, hypothesis, label | Natural language inference |
| `sentiment` | text, label/sentiment | Sentiment analysis |
| `ner` | tokens[], ner_tags[] | Named entity recognition |
| `token_classification` | tokens[], tags[] | Token-level classification |
| `regression` | text, score/label | Regression tasks |
| `sts` | sentence1, sentence2, score | Semantic textual similarity |
| `qa` | question, answer/answers | Question answering |
| `extractive_qa` | question, context, answers | Extractive QA (SQuAD-like) |
| `retrieval` | query, positive | Information retrieval |
| `ranking` | query, candidates[] | Learning to rank |
| `text_pair` | text1/sentence1, text2/sentence2 | Generic text pair |

Run validation:
```
daedalus_validate_dataset(
  path="data/train.jsonl",
  format="sft",
  max_duplicates_pct=1.0,
  max_empty_pct=0.5
)
```

## What Gets Checked

1. **Required fields** — Are all mandatory columns present?
2. **Format validation** — Chat messages have role+content? DPO has chosen != rejected?
3. **Duplicates** — MD5 hash dedup on concatenated fields
4. **Empty fields** — Blank or whitespace-only values
5. **Class balance** — Label distribution for classification tasks (imbalance ratio)
6. **Split leakage** — Prompt overlap between train and eval sets

## Red Flags to Watch For

- **Duplicate rate > 5%** — Likely a data pipeline bug
- **Empty fields > 1%** — Missing data, check preprocessing
- **Imbalance ratio > 10:1** — Consider oversampling or weighted loss
- **Split leakage > 0%** — Data contamination, regenerate splits
- **Token estimate > 1B** — Consider sampling or multi-epoch strategy
- **Suspiciously clean data** — May be low-quality synthetic (check diversity)

## Integration with Training

The validation report has a `passed` field. If `False`, do NOT proceed to training.
Fix the data first, then revalidate.
