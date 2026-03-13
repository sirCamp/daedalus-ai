# Daedalus AI

**AI research assistant for hypothesis-driven ML experiments.**

Daedalus AI is a Claude Code MCP plugin that manages the full experiment lifecycle: hypothesis formation, experiment design, training execution, log monitoring, result analysis, and autonomous parameter exploration. It works as a set of tools that Claude Code can use alongside its native code editing capabilities.

## Why Daedalus?

Running ML experiments involves a repetitive loop: change a parameter, launch training, wait hours, analyze results, decide what to try next. Daedalus automates the bookkeeping and gives Claude Code the tools to manage this loop — while you focus on the research questions.

- **Hypothesis-driven**: Every experiment starts with a hypothesis and predictions, not just "try this"
- **One variable at a time**: The system enforces scientific discipline
- **Full lifecycle tracking**: Draft → Running → Completed → Analyzed, with structured reflections
- **Persistent memory**: Research notes, decisions, and dead ends survive across sessions
- **Literature integration**: Search papers, build a library, generate structured reviews
- **Smart exploration**: Convergence detection, parameter sensitivity, data-driven suggestions
- **Works with any framework**: HuggingFace, PyTorch, TensorFlow, custom scripts — Daedalus doesn't care what you train with
- **Local or remote**: Run on your machine or SSH to GPU servers with automatic sentinel monitoring
- **Multi-GPU**: torchrun, accelerate, deepspeed launchers with GPU selection
- **Batch execution**: Distribute experiments across hosts with round-robin scheduling
- **Claude Code integration**: MCP tools + slash commands + code editing = the AI can fix bugs, not just track experiments

## Installation

```bash
# Install the package
pip install -e .

# Register MCP server globally + install agents, commands, skills + auto-approve tools
daedalus install-mcp

# Restart Claude Code
```

### Scopes

| Scope | Flag | Where it writes | Use case |
|-------|------|----------------|----------|
| `user` (default) | `--scope user` or omit | `~/.claude.json` | **Recommended**: available in ALL projects |
| `project` | `--scope project` | `.mcp.json` in project dir | Shared via git, per-project |
| `local` | `--scope local` | `~/.claude.json` (local entry) | This project only, not shared |

```bash
# Global install (recommended, default)
daedalus install-mcp

# Project-only install (creates .mcp.json, committable to git)
daedalus install-mcp --scope project
```

### Updating

Since it's installed in editable mode, Python code changes are immediate. But plugin files (commands, skills, agents) are copied to `~/.claude/`, so after updating:

```bash
# Re-install plugin files + restart Claude Code
daedalus install-mcp
```

## Quick Start

```bash
# 1. Create a project
daedalus init my-research
cd my-research

# 2. Edit daedalus.yaml — define your scripts, parameters, stack
# 3. Edit program.md — define your research goals and metrics

# 4. Register MCP (global, works from any project with daedalus.yaml)
daedalus install-mcp

# 5. Open Claude Code and start experimenting
#    Use /daedalus:experiment, /daedalus:research-status, or just ask in natural language
```

## Project Structure

```
my-research/
├── daedalus.yaml          # Project config: runner, scripts, stack
├── program.md             # Research goals and metrics (editable by agent)
├── runner_config.yaml     # SSH host configuration
├── ledger/
│   ├── experiments.jsonl  # Experiment log (append-only)
│   ├── papers.jsonl       # Literature library
│   ├── notes.jsonl        # Research memory (decisions, insights, dead ends)
│   └── narrative.md       # Auto-generated narrative
└── runs/                  # Work directories (auto-created)
    └── exp_abc123/
        ├── logs/
        │   ├── stdout.log
        │   └── stderr.log
        └── results.json
```

## Configuration

### daedalus.yaml

```yaml
project_name: my-research

runner:
  type: local              # or: ssh

higher_is_better:
  accuracy: true
  loss: false

# Stack — libraries and frameworks (shown to Claude Code for context)
stack:
  python: "3.11"
  requirements: requirements.txt
  libraries:
    - transformers
    - datasets
    - torch
  docs:
    - https://huggingface.co/docs/transformers/

# Scripts registry — what scripts exist and their parameters
scripts:
  train:
    path: train.py
    description: "Main training script"
    parameters:
      model_name:
        type: str
        default: "bert-base-uncased"
        description: "HuggingFace model ID"
      learning_rate:
        type: float
        default: 2e-5
        range: [1e-6, 1e-3]
      batch_size:
        type: int
        default: 16
        choices: [8, 16, 32, 64]
      epochs:
        type: int
        default: 3
  eval:
    path: evaluate.py
    description: "Run evaluation"
    parameters:
      model_path:
        type: str
        required: true
```

### Multi-GPU Support

Experiments support multi-GPU training via the `launcher` field in the experiment config:

```yaml
# In the experiment config (set via create_experiment tool or CLI)
launcher: accelerate   # "python" (default), "torchrun", "accelerate", "deepspeed"
num_gpus: 4            # Number of GPUs
gpu_ids: [0, 1, 2, 3]  # Specific GPUs (sets CUDA_VISIBLE_DEVICES)
```

| Launcher | Command generated |
|----------|-------------------|
| `python` | `python train.py --args` |
| `torchrun` | `torchrun --nproc_per_node=4 train.py --args` |
| `accelerate` | `accelerate launch --num_processes=4 train.py --args` |
| `deepspeed` | `deepspeed --num_gpus=4 train.py --args` |

To use a single specific GPU: `gpu_ids: [2]` → sets `CUDA_VISIBLE_DEVICES=2`.

### Multi-Host SSH (runner_config.yaml)

```yaml
hosts:
  gpu-a100:
    host: 10.0.0.1
    user: ubuntu
    key_path: ~/.ssh/id_rsa
    remote_work_dir: /data/daedalus
    python_path: /opt/conda/bin/python
  gpu-h100:
    host: 10.0.0.2
    user: admin
    password: mypassword      # sshpass-based auth (alternative to key_path)
    remote_work_dir: /science/daedalus
```

## CLI Commands

### Project Management

| Command | Description |
|---------|-------------|
| `daedalus init NAME` | Create a new research project |
| `daedalus status` | Show experiment counts, active runs, best result |
| `daedalus ledger [-n N]` | Show last N experiments |
| `daedalus narrative` | Auto-generated research narrative |
| `daedalus context [-m MODE]` | Dump context (reason/design/reflect/review/full) |

### Experiment Lifecycle

| Command | Description |
|---------|-------------|
| `daedalus hypothesis STATEMENT -r RATIONALE` | Create draft experiment |
| `daedalus run EXP_ID [-H HOST]` | Launch experiment |
| `daedalus run-batch [EXP_IDS] [-H HOST ...]` | Launch multiple experiments across hosts (round-robin) |
| `daedalus poll EXP_ID [-H HOST]` | Check running status |
| `daedalus logs EXP_ID [-n TAIL]` | View training logs (stdout + stderr) |
| `daedalus cancel EXP_ID [-H HOST]` | Cancel running experiment |
| `daedalus record EXP_ID -r METRIC=VALUE` | Record results manually |
| `daedalus reflect EXP_ID -a ANALYSIS -c STATUS` | Add analysis |
| `daedalus diff EXP_A EXP_B` | Compare two experiments |

### Data Tools

| Command | Description |
|---------|-------------|
| `daedalus data inspect PATH` | Dataset stats, distributions, samples, token estimates |
| `daedalus data validate PATH -f FORMAT` | Validate against format requirements |

### Literature

| Command | Description |
|---------|-------------|
| `daedalus papers search QUERY` | Search Semantic Scholar |
| `daedalus papers add ARXIV_ID` | Add paper to library |
| `daedalus papers list [-t TAG]` | Browse library |

### Remote Management

| Command | Description |
|---------|-------------|
| `daedalus sync [-H HOST]` | Upload scripts and requirements to remote |
| `daedalus setup-env [-H HOST]` | Install dependencies on remote |

### Monitoring

| Command | Description |
|---------|-------------|
| `daedalus watch EXP_ID` | Wait for experiment to complete |
| `daedalus watch --all` | Wait for all running experiments |
| `daedalus watch --stream` | JSONL event stream (for Claude Code) |
| `daedalus watch --verbose` | Human-readable progress to stderr |

### Agent & Plan

| Command | Description |
|---------|-------------|
| `daedalus agent reason` | Run one reasoning cycle |
| `daedalus agent design` | Design the next experiment |
| `daedalus agent reflect` | Reflect on latest results |
| `daedalus agent plan` | Create a structured research plan |
| `daedalus agent loop [-n N] [--plan]` | Autonomous research loop (optionally plan-driven) |
| `daedalus plan show` | Show current plan with step statuses |
| `daedalus plan next` | Show next actionable step |

### MCP Integration

| Command | Description |
|---------|-------------|
| `daedalus install-mcp` | Register MCP server globally + install plugin files (default) |
| `daedalus install-mcp --scope project` | Register MCP server for this project only (.mcp.json) |
| `daedalus uninstall-mcp` | Remove MCP registration |

## MCP Tools

When registered as an MCP server, Daedalus exposes 30 tools to Claude Code:

### Context & Analytics
| Tool | Description |
|------|-------------|
| `daedalus_get_context` | Research context: goals, history, papers, memory, suggestions |
| `daedalus_suggest_next` | Data-driven parameter suggestions with convergence detection |
| `daedalus_get_insights` | Aggregated insights: top configs, parameter sensitivity, dead ends, explored ranges |

### Experiment Management
| Tool | Description |
|------|-------------|
| `daedalus_list_experiments` | List experiments by status |
| `daedalus_get_experiment` | Full details of one experiment |
| `daedalus_create_experiment` | Create draft with hypothesis + config |
| `daedalus_launch_experiment` | Start training |
| `daedalus_poll_experiment` | Check status, auto-fetch results on completion |
| `daedalus_record_results` | Manually record results |
| `daedalus_add_reflection` | Record analysis + auto-save insight to memory |

### Comparison & Logs
| Tool | Description |
|------|-------------|
| `daedalus_compare_experiments` | Config diff + metric deltas |
| `daedalus_get_experiment_logs` | Recent training output (stdout + stderr) |

### Research Memory
| Tool | Description |
|------|-------------|
| `daedalus_save_note` | Save a research note (decision, insight, dead_end, convergence, todo) |
| `daedalus_read_notes` | Read notes — filter by category, experiment, or keyword |
| `daedalus_update_program` | Edit program.md — append, replace section, or rewrite |

### Literature
| Tool | Description |
|------|-------------|
| `daedalus_search_papers` | Query Semantic Scholar and ACL Anthology |
| `daedalus_add_paper` | Add to library by arXiv ID or ACL ID, with notes and tags |
| `daedalus_list_papers` | Browse library by tag |
| `daedalus_get_paper_details` | Full paper info from library, Semantic Scholar, or ACL Anthology |
| `daedalus_literature_review` | Structured review: papers vs experiments, gaps, new papers |

### Data & Scripts
| Tool | Description |
|------|-------------|
| `daedalus_inspect_dataset` | Stats, distributions, token estimates |
| `daedalus_validate_dataset` | Format checks (18 formats supported) |
| `daedalus_list_scripts` | Available scripts and parameters |

### Batch & Remote
| Tool | Description |
|------|-------------|
| `daedalus_batch_run` | Launch multiple experiments across hosts (round-robin) |
| `daedalus_remote_exec` | Execute a command on the remote SSH host (read-only direct, mutating needs confirmation) |
| `daedalus_human_confirm` | Ask researcher for approval |

### Research Plan
| Tool | Description |
|------|-------------|
| `daedalus_get_plan` | Get the current research plan with all steps, autonomy mode, and guardrail status |
| `daedalus_create_plan` | Create a plan with autonomy mode and guardrails (3-7 steps, priorities, dependencies) |
| `daedalus_update_plan_step` | Mark steps done/failed, change priority, set requires_confirmation |
| `daedalus_update_plan` | Switch autonomy mode, adjust guardrails, pause/resume the plan |

## Slash Commands

After installing (`daedalus install-mcp` copies everything), these commands are available in Claude Code:

| Command | Description |
|---------|-------------|
| `/daedalus:experiment` | Full experiment workflow: design → launch → monitor → analyze |
| `/daedalus:batch-run` | Launch multiple experiments across hosts |
| `/daedalus:analyze` | Analyze completed results and record reflections |
| `/daedalus:compare` | Compare two experiments side by side |
| `/daedalus:logs` | Show training logs (stdout + stderr) |
| `/daedalus:inspect-data PATH` | Inspect and validate a dataset |
| `/daedalus:watch [EXP_ID]` | Monitor running experiments |
| `/daedalus:research-status` | Project overview: progress, best results, next steps |
| `/daedalus:literature-review [TOPIC]` | Structured literature review with gap analysis |
| `/daedalus:plan` | Create or review a structured research plan |

## Research Memory

Daedalus maintains persistent research memory across Claude Code sessions via `ledger/notes.jsonl`. Notes are categorized:

| Category | Use case |
|----------|----------|
| `decision` | Key research decisions and their rationale |
| `insight` | Learned patterns from experiments |
| `dead_end` | Approaches that don't work (auto-saved on rejected hypotheses) |
| `convergence` | Parameter convergence and saturation detections (auto-saved) |
| `todo` | Next steps and ideas to try |
| `general` | Uncategorized notes |

Memory is integrated at three levels:
- **Context**: `get_context` includes recent notes in reason/design/full modes
- **Auto-save**: `add_reflection` automatically saves insights and dead ends
- **Agent loop**: The reasoning step reads notes first to recall previous sessions

## Plan Mode

Daedalus supports structured research planning — like Claude Code's plan mode, but for experiments.

```bash
# Agent creates a multi-step plan
daedalus agent plan

# Run the loop following the plan
daedalus agent loop --plan --autonomous -n 5

# View the plan
daedalus plan show

# See what's next
daedalus plan next
```

Each plan step has:
- **Description**: what to test (one variable)
- **Rationale**: why it matters
- **Expected outcome**: what we predict
- **Priority**: 1 (highest) to 5
- **Dependencies**: which steps must complete first
- **Status**: pending → running → done/failed/skipped
- **Requires confirmation**: override for risky steps (even in autonomous mode)

The autonomous loop follows the plan: picks the next actionable step (highest priority with dependencies met), designs the experiment, runs it, reflects, marks it done, and moves to the next step. If all steps complete, the loop stops. The agent can also add new steps during reflection as results evolve.

### Autonomy Modes

Plans support two execution modes:

| Mode | Behavior | Use case |
|------|----------|----------|
| `autonomous` (default) | Launches directly — Claude Code chat is already human-in-the-loop | Normal usage via MCP |
| `supervised` | Adds explicit code-level confirmation gate before each launch | CLI `run_loop` or extra caution |

```bash
# Create an autonomous plan via MCP
# (Claude Code tool call)
create_plan(
    goal="Find optimal learning rate",
    steps=[...],
    autonomy="autonomous",
    max_experiments=10,
    max_consecutive_failures=2,
)

# Switch mode mid-plan
update_plan(autonomy="autonomous")  # "go overnight"
update_plan(autonomy="supervised")  # "I'm back, let me review"
```

### Auto-Bookkeeping

When experiments are linked to plan steps (via `plan_step_id` in `launch_experiment`), Daedalus automatically:

- Links the experiment to the plan step on launch
- Marks the step as done/failed when the experiment completes
- Resets/increments failure counters
- Copies reflection notes to the plan step

No manual `update_plan_step` calls needed for these transitions.

### Guardrails

Safety rails prevent runaway execution in autonomous mode:

| Guardrail | Default | Effect |
|-----------|---------|--------|
| `max_experiments` | unlimited | Auto-pause after N total experiments |
| `max_consecutive_failures` | 2 | Auto-pause after N failures in a row |
| Per-step `requires_confirmation` | false | Block launch even in autonomous mode |

When a guardrail triggers, the plan is **paused** (not stopped). Resume with `update_plan(paused=false)`.

### MCP Tools for Plan Management

| Tool | Description |
|------|-------------|
| `daedalus_get_plan` | Full plan state: steps, autonomy mode, guardrail counters, next actionable |
| `daedalus_create_plan` | Create plan with autonomy mode and guardrails |
| `daedalus_update_plan_step` | Update step status, priority, notes, requires_confirmation |
| `daedalus_update_plan` | Switch autonomy mode, adjust guardrails, pause/resume |

## Smart Context & Convergence

### Context Compression

When experiment count exceeds 20, older experiments are automatically compressed into aggregated statistics (status distribution, parameter ranges, hypothesis outcomes) while recent experiments keep full detail. This prevents context overflow while preserving all relevant information.

### Convergence Detection

`suggest_next` detects when parameters have converged and stops suggesting them:

| Check | Trigger | What it means |
|-------|---------|---------------|
| Plateau | Last 3+ values show <2% improvement | Metric has stabilized |
| Range saturation | >80% of declared range covered, optimum is interior | No room to explore |
| Categorical exhaustion | All choices tested | Best choice identified |

### Insights Generator

`get_insights` produces a structured analysis:
- **Top configurations** ranked by target metric
- **Parameter sensitivity** — which parameters have the most impact
- **Dead ends** — rejected hypotheses and failed experiments
- **Explored ranges** — coverage of declared parameter ranges

## Literature Management

### Paper Library

Papers are stored locally in `ledger/papers.jsonl` with:
- Title, authors, year, abstract, citation count
- **Key findings** — curated bullet points relevant to your research
- **Relevance note** — why this paper matters
- **Tags** — for filtering

### Literature Review

The `literature_review` tool generates a structured review:
- Papers in library with key findings
- **Papers vs experiments** — which papers support which experimental outcomes
- **Gaps** — experiments without paper backing, rejected hypotheses needing literature
- **New papers** — optionally search Semantic Scholar for recent work

## Dataset Validation

Daedalus validates datasets against 18 format specifications:

| Format | Required Fields | Use Case |
|--------|----------------|----------|
| `sft` | prompt, completion | Supervised fine-tuning |
| `chat` | messages (role/content) | Chat format |
| `dpo` | prompt, chosen, rejected | Direct preference optimization |
| `grpo` | prompt | Group relative policy optimization |
| `classification` | text, label | Text classification |
| `nli` | premise, hypothesis, label | Natural language inference |
| `sentiment` | text, label | Sentiment analysis |
| `multi_label` | text, labels | Multi-label classification |
| `ner` | tokens, tags | Named entity recognition |
| `token_classification` | tokens, tags | Token-level classification |
| `regression` | text, score | Regression tasks |
| `sts` | sentence1, sentence2, score | Semantic textual similarity |
| `qa` | question, answer | Question answering (open-domain) |
| `extractive_qa` | question, context, answers | Extractive QA (SQuAD-like) |
| `retrieval` | query, positive | Information retrieval |
| `ranking` | query, candidates | Learning to rank |
| `tabular` | target | Tabular/sklearn (features auto-detected) |
| `text_pair` | text1, text2 | Generic text pair |

Checks performed: required fields, format validation, duplicates (MD5), empty fields, class balance, split leakage.

## Parameter Exploration

The `suggest_next` tool analyzes experiment history and suggests what to try:

| Strategy | When | What it does |
|----------|------|-------------|
| Baseline | 0 experiments | Returns defaults from scripts registry |
| Explore neighbor | 1 experiment | Suggests 2x and 0.5x of current values |
| Bisect | 2+ experiments | Binary search between best and second-best |
| Trend extrapolate | 3+ monotonic points | Extrapolates observed trend |
| Untried choice | Categorical params | Suggests values never tested |
| Unexplored range | Never-varied params | Identifies parameters worth exploring |
| Converged | 3+ plateau points | Reports convergence, stops suggesting |

## Experiment Monitoring

The watcher tails training logs in real-time and emits structured events:

```bash
daedalus watch exp_007 --stream --log-interval 10
```

```jsonl
{"event": "WATCHING", "exp_id": "exp_007"}
{"event": "PROGRESS", "exp_id": "exp_007", "key": "epoch", "value": "1/10"}
{"event": "METRIC", "exp_id": "exp_007", "metric": "loss", "value": 0.345}
{"event": "METRIC", "exp_id": "exp_007", "metric": "eval_loss", "value": 0.287}
{"event": "ALERT", "exp_id": "exp_007", "alert_type": "loss_spike", "detail": "Loss jumped 5.2x"}
{"event": "COMPLETED", "exp_id": "exp_007", "results": {"eval": {"acc": 0.92}}}
```

### Detected Patterns

- **Metrics**: loss, eval_loss, accuracy, learning_rate (HF Trainer dict + key=value formats)
- **Progress**: epoch N/M, step N/M, tqdm percentage bars
- **Alerts**: NaN, Inf, CUDA OOM, loss spike (configurable threshold), training stall (no output for N minutes)

### Remote Sentinel

For SSH experiments, a sentinel script runs inside `screen` on the remote host:
- Survives SSH disconnects
- Parses both stdout and stderr logs
- 5-level completion detection: results.json, trainer_state.json, model artifacts, exit code, log heuristics
- Writes structured events to `events.jsonl`
- Local watcher reads events via single SSH calls (no persistent connection)

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Claude Code                     │
│  (the brain — reasons, edits code, decides)      │
│                                                  │
│  Native tools: Read, Write, Edit, Bash, Grep     │
│  Daedalus tools: daedalus_* (via MCP)            │
└──────────┬──────────────────────────┬────────────┘
           │ MCP (JSON-RPC/stdio)     │ direct
           ▼                          ▼
┌──────────────────┐    ┌──────────────────────────┐
│  Daedalus MCP    │    │  Your codebase           │
│  Server          │    │  (training scripts,      │
│                  │    │   data, configs)          │
│  - ToolExecutor  │    └──────────────────────────┘
│  - 30 tools      │
│  - Ledger (JSONL)│
│  - Memory (JSONL)│
│  - Runners       │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Runners                                         │
│                                                  │
│  LocalRunner          SSHRunner + Sentinel        │
│  - subprocess         - ssh/scp (no paramiko)     │
│  - logs/stdout.log    - sentinel.sh in screen     │
│  - results.json       - events.jsonl on remote    │
│                       - sshpass for password auth  │
└──────────────────────────────────────────────────┘
```

**Key design principle**: Claude Code is the brain. Daedalus is the hands. The watcher is the eyes. No AI runs inside Daedalus — it's pure bookkeeping, polling, and log parsing. All intelligence comes from Claude Code, which can both manage experiments AND edit the code.

## Results Auto-Detection

Daedalus automatically detects result formats:
- `results.json` — Daedalus native (nested `{"eval": {"metric": value}}`) or flat JSON
- `eval_results.json` / `trainer_state.json` — HuggingFace Trainer
- `results.csv` — CSV with metric,value columns

## Examples

Two ready-to-run examples are included in `examples/`:

### sklearn-iris — Classical ML

RandomForest on Iris dataset. Runs in seconds, no GPU needed.

```bash
cd examples/sklearn-iris
daedalus init .          # Initialize the project
# Open Claude Code and ask: "Run a baseline experiment"
```

Demonstrates: experiment lifecycle, parameter exploration, convergence detection.

### sft-tiny — LLM Fine-tuning

SFT with TRL on SmolLM2-135M (135M params). Runs in a few minutes on CPU.

```bash
cd examples/sft-tiny
pip install -r requirements.txt
daedalus init .
# Open Claude Code and ask: "Create a plan to find the best learning rate"
```

Demonstrates: plan mode, HuggingFace integration, learning rate search, reflection.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (472 tests)
python -m pytest tests/ -q

# Run with coverage
python -m pytest tests/ --cov=daedalus --cov-report=term-missing
```

## License

MIT
