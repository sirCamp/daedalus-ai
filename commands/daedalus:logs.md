---
description: Show training logs for a running or completed experiment
---

# View Logs

Show recent training output (stdout + stderr) for an experiment.

## What to do

1. If user provides an experiment ID, use it. Otherwise find running experiments via `daedalus_list_experiments(status="running")`
2. Call `daedalus_get_experiment_logs(exp_id=..., tail=100)`
3. Highlight important information:
   - Current metrics (loss, accuracy, learning rate)
   - Progress (epoch, step, percentage)
   - Any warnings or errors
4. If the experiment looks stuck or has errors, suggest next steps
