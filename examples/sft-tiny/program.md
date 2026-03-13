# Research Program: Minimal SFT

## Goal

Fine-tune SmolLM2-135M-Instruct on SmolTalk to demonstrate the Daedalus experiment workflow
with a real LLM training loop.

## Metrics

- **loss** (lower is better) — eval loss, primary metric
- **train_loss** (lower is better) — training loss
- **train_samples_per_second** (higher is better) — throughput

## Approach

1. Baseline: default parameters on 1000 samples, 1 epoch
2. Compare learning rates: 1e-5 vs 5e-5 vs 1e-4
3. Try longer training: 2-3 epochs
4. Experiment with batch size and gradient accumulation

## Notes

This uses SmolLM2-135M-Instruct (~135M parameters) — small enough to train on CPU in a few
minutes. The dataset is sliced to 1000 samples for speed. The goal is to
demonstrate the workflow, not achieve SOTA.

## Requirements

```bash
pip install transformers trl datasets torch
```
