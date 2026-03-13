---
description: Show full research status — experiments, progress, and next steps
---

# /daedalus:research-status

Get a comprehensive overview of your research project.

## What This Command Does

1. Loads the research program (`program.md`)
2. Summarizes experiment history by status
3. Highlights best results so far
4. Shows unanalyzed experiments
5. Lists currently running experiments
6. Shows research plan progress (if a plan exists)
7. Shows recent research memory (notes, decisions, dead ends)
8. Shows paper library summary
9. Gets data-driven insights and suggestions
10. Suggests next steps based on history

## When to Use

- Starting a new work session
- Getting oriented after a break
- Before deciding what to do next
- Reviewing overall progress

## How It Works

```
User: /daedalus:research-status
Agent: [calls daedalus_get_context(mode="full"), daedalus_list_experiments,
        daedalus_get_plan, daedalus_read_notes, daedalus_list_papers,
        daedalus_get_insights]

       Project: Text Classification Fine-tuning
       Goal: Maximize accuracy while keeping training efficient

       Experiments: 12 total
       - 8 completed, 6 analyzed
       - 2 running (exp_011, exp_012)
       - 1 failed, 1 abandoned

       Best result: exp_009 (accuracy=0.95, f1=0.93)
       Unanalyzed: exp_010 (completed 2h ago)

       Plan: 5/7 steps done (autonomous mode, 0 consecutive failures)
       Next step: S06 — HP tuning: batch size sweep

       Memory: 8 notes (3 insights, 2 decisions, 2 dead ends, 1 todo)
       Recent: "lr=1e-5 consistently outperforms 2e-5 across all configs"

       Papers: 6 in library (3 fine-tuning, 2 LoRA, 1 data augmentation)

       Insights:
       - Top config: lr=1e-5, batch_size=32, LoRA r=16 (exp_009)
       - Most sensitive parameter: learning_rate
       - Dead end: batch_size > 64 causes OOM on single GPU

       Suggested next: Try LoRA rank 32 on best config
```

## Related Commands

- `/daedalus:plan` — Create or review the research plan
- `/daedalus:experiment` — Design and launch experiments
- `/daedalus:analyze` — Analyze completed results
- `/daedalus:literature-review` — Search papers and build library
