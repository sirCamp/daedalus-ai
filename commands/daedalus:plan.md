---
description: Create or review a structured research plan with prioritized steps
---

# /daedalus:plan

Manage your research plan using Daedalus MCP tools.

## What This Command Does

1. Checks if a plan already exists (`daedalus_get_plan`)
2. If no plan: reviews context, suggests steps, creates a plan (`daedalus_create_plan`)
3. If plan exists: shows progress, highlights next actionable step
4. Optionally updates step status or adds new steps

## When to Use

- **Starting a research direction** — create a structured plan before experimenting
- **Between experiments** — check what's next, update progress
- **After reflection** — mark steps done, add new steps based on findings

## Steps

1. Get full research context: `daedalus_get_context(mode="full")`
2. Check existing plan: `daedalus_get_plan`
3. If no plan exists:
   - Analyze the research goals from program.md
   - Search relevant papers: `daedalus_search_papers`
   - Create a plan with 3-7 steps, each with:
     - **Description**: what to test (one variable)
     - **Rationale**: why it matters
     - **Expected outcome**: what we predict
     - **Priority**: 1 (highest) to 5
     - **Dependencies**: which steps must complete first
   - Save the plan: `daedalus_create_plan`
4. If plan exists:
   - Show current progress (done/total)
   - Highlight next actionable step
   - Ask if the user wants to update any steps or add new ones

## Example

```
User: /daedalus:plan

Agent: Let me check the current research plan...

📋 Research Plan: Optimize calibration for QRC model
Progress: 2/5 done

✅ S01 — Baseline with default parameters (exp_003)
✅ S02 — Test learning rate 1e-5 (exp_004)
⬜ S03 — Test IDK penalty -0.5 [NEXT]
⬜ S04 — Add FiSCoRe agreement (depends on S03)
⬜ S05 — Final evaluation battery

Next actionable: S03 — Test IDK penalty -0.5
Shall I design this experiment?
```

## Related Commands

- `/daedalus:experiment` — Design and launch a single experiment
- `/daedalus:research-status` — Full project overview
- `/daedalus:analyze` — Analyze completed results
