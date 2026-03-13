---
description: Inspect and validate a dataset before training
---

# /daedalus:inspect-data

Run full dataset inspection and validation.

## What This Command Does

1. Loads the dataset (JSONL, CSV, Parquet, or HuggingFace)
2. Computes statistics: row count, column types, nulls, distributions
3. Shows sample records for visual inspection
4. Estimates token count and training cost
5. Validates format (sft, dpo, grpo, classification, etc.)
6. Reports issues: duplicates, empty fields, class imbalance, leakage

## When to Use

- Before any training run (mandatory)
- When debugging training issues
- When receiving a new dataset
- After data preprocessing

## How It Works

```
User: /daedalus:inspect-data data/train.jsonl
Agent: [calls daedalus_inspect_dataset, daedalus_validate_dataset]

       Dataset: 15,432 rows, 3 columns (prompt, chosen, rejected)
       Format: DPO (valid)
       Tokens: ~2.1M (estimated)

       Issues found:
       WARNING: 127 duplicates (0.8%) — below threshold
       ERROR: 3 empty 'rejected' fields — MUST FIX

       Showing 3 samples...
```

## Arguments

Pass the dataset path after the command:
```
/daedalus:inspect-data path/to/dataset.jsonl
/daedalus:inspect-data hf://username/dataset-name
```
