---
description: Start the experiment workflow — design, launch, and monitor an ML experiment
---

# /daedalus:experiment

Run the full experiment workflow using Daedalus MCP tools.

## What This Command Does

1. Checks the current research context (`daedalus_get_context`)
2. Lists available scripts and parameters (`daedalus_list_scripts`)
3. Reviews recent experiments to pick a baseline
4. Asks what hypothesis to test
5. Creates a draft experiment with predictions
6. Launches and monitors until completion
7. Analyzes results and records reflection

## When to Use

- Starting a new training run
- Iterating on hyperparameters
- Running an ablation study
- Testing a new model configuration

## How It Works

The research-assistant agent orchestrates the workflow:
- Uses Daedalus MCP tools for experiment management
- Uses Claude Code tools (Read, Edit, Bash) for code fixes
- Follows hypothesis-driven methodology (one variable at a time)
- Compares results against baseline automatically

## Example

```
User: /daedalus:experiment
Agent: Let me check the current state of your research...
       [calls daedalus_get_context, daedalus_list_experiments]
       You have 3 completed experiments. Best so far: exp_003 (acc=0.92).
       What would you like to test next?
```
