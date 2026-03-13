# Research Program: Iris Classification

## Goal

Find the best RandomForest configuration for Iris classification.

## Metrics

- **accuracy** (higher is better) — primary metric
- **f1_macro** (higher is better) — secondary, handles class imbalance

## Approach

1. Start with default parameters as baseline
2. Explore `n_estimators` (number of trees)
3. Explore `max_depth` (tree complexity)
4. Fine-tune `min_samples_split`

## Notes

This is a toy example to demonstrate Daedalus. The dataset is small (150 samples)
so results will be noisy — that's fine, the point is to show the workflow.
