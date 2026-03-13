"""Tool definitions for the Daedalus research agent.

Each tool wraps a Daedalus operation as a callable that the agent can invoke
via Anthropic's tool-use API. Tools are defined as JSON schemas for the API
and as Python functions for execution.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from ..core.config import ExperimentConfig, config_diff
from ..core.experiment import Experiment, ExperimentStatus, Reflection
from ..core.hypothesis import Hypothesis, Prediction
from ..core.ledger import Ledger
from ..evaluators.metric import compare_results
from ..literature.library import Library
from ..literature.paper import Paper
from ..literature.search import PaperSearcher
from ..runners.factory import create_runner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_context",
        "description": "Get the full research context: program goals, experiment history, papers, and suggested next steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["reason", "design", "reflect", "review", "full"],
                    "description": "Context mode. 'reason' for forming hypotheses, 'design' for experiment planning, 'reflect' for result analysis, 'full' for everything.",
                },
                "recent_n": {
                    "type": "integer",
                    "description": "Number of recent experiments to include (default: 10).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_experiments",
        "description": "List experiments from the ledger, optionally filtered by status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["draft", "queued", "running", "completed", "analyzed", "failed", "abandoned"],
                    "description": "Filter by status. Omit to list all.",
                },
                "last_n": {
                    "type": "integer",
                    "description": "Return only the last N experiments.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_experiment",
        "description": "Get full details of a specific experiment by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exp_id": {"type": "string", "description": "Experiment ID."},
            },
            "required": ["exp_id"],
        },
    },
    {
        "name": "create_experiment",
        "description": "Create a new draft experiment with a hypothesis and config.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis_statement": {"type": "string", "description": "Clear, falsifiable hypothesis."},
                "rationale": {"type": "string", "description": "Why you expect this to work."},
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string"},
                            "direction": {"type": "string", "enum": ["increase", "decrease", "stable"]},
                            "expected_value": {"type": "number"},
                        },
                        "required": ["metric", "direction"],
                    },
                    "description": "Falsifiable predictions.",
                },
                "papers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ArXiv IDs supporting this hypothesis.",
                },
                "script": {"type": "string", "description": "Training script path."},
                "script_args": {"type": "object", "description": "Script arguments as key-value pairs."},
                "eval_script": {"type": "string", "description": "Optional separate evaluation script (runs after training)."},
                "eval_script_args": {"type": "object", "description": "Evaluation script arguments."},
                "results_adapter": {
                    "type": "string",
                    "enum": ["auto", "daedalus", "hf_trainer", "csv", "json_flat"],
                    "description": "Results format. 'auto' detects from files (default).",
                },
                "env": {"type": "object", "description": "Environment variables."},
                "baseline_id": {"type": "string", "description": "Baseline experiment ID for comparison."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for this experiment.",
                },
            },
            "required": ["hypothesis_statement", "rationale", "script"],
        },
    },
    {
        "name": "launch_experiment",
        "description": "Launch a draft experiment. Auto-syncs files to remote for SSH runners. Returns monitoring hints (poll interval, suggested tasks while waiting). If a plan exists and is paused or guardrails are exceeded, the launch is blocked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exp_id": {"type": "string", "description": "Experiment ID to launch."},
                "host": {"type": "string", "description": "Target host name for multi-host SSH. Omit to use default."},
                "plan_step_id": {"type": "string", "description": "Plan step ID to link this experiment to. Enables auto-bookkeeping."},
            },
            "required": ["exp_id"],
        },
    },
    {
        "name": "poll_experiment",
        "description": "Check the status of a running experiment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exp_id": {"type": "string", "description": "Experiment ID to poll."},
            },
            "required": ["exp_id"],
        },
    },
    {
        "name": "record_results",
        "description": "Record evaluation results for a completed experiment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exp_id": {"type": "string", "description": "Experiment ID."},
                "results": {
                    "type": "object",
                    "description": "Results as {eval_name: {metric: value}}.",
                },
            },
            "required": ["exp_id", "results"],
        },
    },
    {
        "name": "add_reflection",
        "description": "Add a reflection/analysis to a completed experiment. Transitions it to analyzed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exp_id": {"type": "string", "description": "Experiment ID."},
                "hypothesis_confirmed": {
                    "type": "string",
                    "enum": ["confirmed", "partial", "rejected"],
                },
                "analysis": {"type": "string", "description": "Detailed analysis of what happened and why."},
                "surprise": {"type": "string", "description": "What was unexpected."},
                "next_suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete next steps.",
                },
            },
            "required": ["exp_id", "hypothesis_confirmed", "analysis"],
        },
    },
    {
        "name": "compare_experiments",
        "description": "Compare two experiments: config diff and metric deltas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exp_a": {"type": "string", "description": "First experiment ID (baseline)."},
                "exp_b": {"type": "string", "description": "Second experiment ID (current)."},
            },
            "required": ["exp_a", "exp_b"],
        },
    },
    {
        "name": "search_papers",
        "description": "Search for papers on Semantic Scholar and ACL Anthology.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "description": "Max results (default: 5)."},
                "source": {
                    "type": "string",
                    "enum": ["all", "semantic_scholar", "acl"],
                    "description": "Search source: 'all' (default), 'semantic_scholar', or 'acl' (ACL Anthology).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_paper",
        "description": "Add a paper to the local library with notes and tags. Provide arxiv_id OR acl_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "ArXiv ID (e.g., '2601.20126')."},
                "acl_id": {"type": "string", "description": "ACL Anthology ID (e.g., '2024.acl-long.1')."},
                "relevance_note": {"type": "string", "description": "Why this paper matters for our research."},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key findings relevant to our work.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_papers",
        "description": "List papers in the local library, optionally filtered by tag.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Filter by tag."},
            },
            "required": [],
        },
    },
    {
        "name": "get_experiment_logs",
        "description": "Get recent training logs for an experiment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exp_id": {"type": "string"},
                "tail": {"type": "integer", "description": "Number of lines (default: 50)."},
            },
            "required": ["exp_id"],
        },
    },
    {
        "name": "suggest_next",
        "description": "Analyze experiment history and suggest which parameter values to try next. Uses strategies like bisection, trend extrapolation, and unexplored choices.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_metric": {
                    "type": "string",
                    "description": "Metric to optimize (e.g., 'simpleqa.not_attempted', 'eval.accuracy').",
                },
                "higher_is_better": {
                    "type": "boolean",
                    "description": "True if higher metric values are better.",
                },
                "focus_param": {
                    "type": "string",
                    "description": "Focus suggestions on this parameter only. Omit to explore all.",
                },
            },
            "required": ["target_metric"],
        },
    },
    {
        "name": "list_scripts",
        "description": "List available scripts and their parameters. Use this to understand what scripts you can use in create_experiment and what arguments they accept.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Script name to get details for. Omit to list all."},
            },
            "required": [],
        },
    },
    {
        "name": "inspect_dataset",
        "description": "Inspect a dataset file: compute stats, field types, value distributions, token estimates, and show sample records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to dataset file (JSONL/CSV/Parquet) or HuggingFace dataset ID."},
                "sample_n": {"type": "integer", "description": "Sample size for large datasets. Omit to load all."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "validate_dataset",
        "description": "Validate a dataset against configurable checks: required fields, duplicates, empty fields, format compliance, class balance, split leakage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to dataset file."},
                "required_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields that must be present in every record.",
                },
                "format": {
                    "type": "string",
                    "enum": ["sft", "chat", "dpo", "grpo", "classification", "nli", "sentiment", "multi_label", "ner", "token_classification", "regression", "sts", "qa", "extractive_qa", "retrieval", "ranking", "tabular", "text_pair"],
                    "description": "Predefined format to validate against.",
                },
                "max_duplicate_ratio": {"type": "number", "description": "Max allowed duplicate ratio (default: 0.01)."},
                "max_empty_ratio": {"type": "number", "description": "Max allowed empty field ratio (default: 0.05)."},
                "check_class_balance": {"type": "boolean", "description": "Check class distribution balance."},
                "label_field": {"type": "string", "description": "Field name for class balance check."},
                "eval_path": {"type": "string", "description": "Path to eval split for leakage check."},
                "prompt_field": {"type": "string", "description": "Field to compare for split leakage (default: 'prompt'). Use 'text' for classification, 'question' for QA."},
                "sample_n": {"type": "integer", "description": "Sample size for large datasets."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "human_confirm",
        "description": "Ask the human researcher for confirmation or input before proceeding with an action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "What to ask the human."},
                "context": {"type": "string", "description": "Relevant context for the decision."},
            },
            "required": ["question"],
        },
    },
    {
        "name": "save_note",
        "description": (
            "Save a research note to persistent memory. Notes survive across sessions "
            "and help maintain continuity. Use after key decisions, insights, or dead ends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The note content."},
                "category": {
                    "type": "string",
                    "enum": ["decision", "insight", "dead_end", "convergence", "todo", "general"],
                    "description": "Note category (default: general).",
                },
                "experiment_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Experiment IDs this note relates to.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for filtering.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "read_notes",
        "description": (
            "Read research notes from persistent memory. Use at session start to "
            "recall previous decisions, insights, and dead ends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["decision", "insight", "dead_end", "convergence", "todo", "general"],
                    "description": "Filter by category. Omit for all.",
                },
                "experiment_id": {
                    "type": "string",
                    "description": "Filter notes linked to this experiment.",
                },
                "search": {
                    "type": "string",
                    "description": "Search notes by keyword.",
                },
                "last_n": {
                    "type": "integer",
                    "description": "Return only the last N notes (default: 20).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "update_program",
        "description": (
            "Update the research program (program.md). Use when research direction changes, "
            "goals are refined, or milestones are reached. Supports append, replace section, or full rewrite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["append", "replace_section", "rewrite"],
                    "description": "How to update: append to end, replace a section, or full rewrite.",
                },
                "content": {
                    "type": "string",
                    "description": "New content to add or replace with.",
                },
                "section_heading": {
                    "type": "string",
                    "description": "Section heading to replace (for replace_section action).",
                },
            },
            "required": ["action", "content"],
        },
    },
    {
        "name": "get_paper_details",
        "description": (
            "Get full details of a paper from the local library or fetch from Semantic Scholar / ACL Anthology. "
            "Returns abstract, key findings, relevance notes, citation count, and URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "ArXiv ID (e.g., '2601.20126').",
                },
                "acl_id": {
                    "type": "string",
                    "description": "ACL Anthology ID (e.g., '2024.acl-long.1').",
                },
                "query": {
                    "type": "string",
                    "description": "Search local library by keyword (alternative to arxiv_id/acl_id).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "literature_review",
        "description": (
            "Generate a structured literature review comparing papers in the library "
            "against experiment results. Identifies gaps, validates approaches, "
            "and suggests papers to read."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Focus topic (e.g., 'calibration', 'reward design'). Omit for broad review.",
                },
                "include_search": {
                    "type": "boolean",
                    "description": "Also search Semantic Scholar for new papers (default: false).",
                },
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Queries for Semantic Scholar if include_search is true.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_insights",
        "description": (
            "Generate aggregated research insights from experiment history: "
            "top configurations, parameter sensitivity, dead ends, and explored ranges."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_metric": {
                    "type": "string",
                    "description": "Metric to rank by (e.g., 'simpleqa.not_attempted').",
                },
                "higher_is_better": {
                    "type": "boolean",
                    "description": "True if higher metric values are better.",
                },
                "save_path": {
                    "type": "string",
                    "description": "Optional path to save insights as Markdown file.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "batch_run",
        "description": (
            "Launch multiple experiments distributed across hosts (round-robin). "
            "Use this to run a battery of experiments in parallel on different machines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "exp_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of experiment IDs to launch.",
                },
                "hosts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Host names for round-robin distribution. "
                        "Omit to use default runner for all."
                    ),
                },
            },
            "required": ["exp_ids"],
        },
    },
    # --- Remote ---
    {
        "name": "remote_exec",
        "description": (
            "Execute a shell command on the remote SSH host. "
            "Use ONLY for operations that have NO dedicated tool: nvidia-smi, disk space, "
            "pip list, environment checks, process inspection, etc. "
            "Do NOT use for experiment status (use poll_experiment) or training logs "
            "(use get_experiment_logs). "
            "Read-only commands run directly. Mutating commands (pip install, kill, rm) "
            "use a two-step flow: (1) call with mutating=true — returns needs_confirmation; "
            "(2) after user approves, call again with mutating=true AND confirmed=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute on the remote host.",
                },
                "host": {
                    "type": "string",
                    "description": "Target host name. Omit to use default.",
                },
                "mutating": {
                    "type": "boolean",
                    "description": (
                        "Set true if the command modifies state (pip install, kill, rm, apt, etc.). "
                        "Mutating commands require confirmed=true after human approval."
                    ),
                },
                "confirmed": {
                    "type": "boolean",
                    "description": (
                        "Set true ONLY on the second call, after showing the "
                        "needs_confirmation response to the human and getting approval. "
                        "NEVER set confirmed=true on the first call."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30, max: 600). Use 120-600 for pip install.",
                },
            },
            "required": ["command"],
        },
    },
    # --- Plan ---
    {
        "name": "get_plan",
        "description": (
            "Get the current research plan. Returns the full plan with all steps, "
            "statuses, priorities, and the next actionable step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_plan",
        "description": (
            "Create a structured multi-step research plan. Replaces any existing plan. "
            "Each step should test ONE variable with a clear expected outcome. "
            "Order by priority (1=highest) and set depends_on for sequential steps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "High-level objective for this plan.",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "rationale": {"type": "string"},
                            "expected_outcome": {"type": "string"},
                            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Step IDs (e.g. 'S01') that must complete first.",
                            },
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "requires_confirmation": {
                                "type": "boolean",
                                "description": "If true, require human confirm even in autonomous mode (for risky/expensive steps).",
                            },
                        },
                        "required": ["description", "rationale", "expected_outcome"],
                    },
                    "description": "List of plan steps (3-7 recommended).",
                },
                "autonomy": {
                    "type": "string",
                    "enum": ["supervised", "autonomous"],
                    "description": "Launch mode. 'autonomous' launches directly (default). 'supervised' adds explicit confirmation gate before each launch.",
                },
                "max_experiments": {
                    "type": "integer",
                    "description": "Auto-pause after this many experiments. Omit for unlimited.",
                },
                "max_consecutive_failures": {
                    "type": "integer",
                    "description": "Auto-pause after N consecutive failures (default: 2).",
                },
            },
            "required": ["goal", "steps"],
        },
    },
    {
        "name": "update_plan_step",
        "description": (
            "Update a step in the research plan. Use to mark steps done, "
            "change priority, add notes, or link an experiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step_id": {
                    "type": "string",
                    "description": "Step ID (e.g. 'S01').",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "running", "done", "failed", "skipped", "blocked"],
                },
                "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                "experiment_id": {"type": "string"},
                "notes": {"type": "string"},
                "requires_confirmation": {
                    "type": "boolean",
                    "description": "If true, require human confirm even in autonomous mode.",
                },
            },
            "required": ["step_id"],
        },
    },
    {
        "name": "update_plan",
        "description": (
            "Update plan-level settings: switch between supervised/autonomous mode, "
            "adjust guardrails, or pause/resume the plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "autonomy": {
                    "type": "string",
                    "enum": ["supervised", "autonomous"],
                    "description": "Launch mode. 'autonomous' for overnight runs.",
                },
                "max_experiments": {
                    "type": "integer",
                    "description": "Auto-pause after this many experiments.",
                },
                "max_consecutive_failures": {
                    "type": "integer",
                    "description": "Auto-pause after N consecutive failures.",
                },
                "paused": {
                    "type": "boolean",
                    "description": "Set true to pause, false to resume.",
                },
                "pause_reason": {
                    "type": "string",
                    "description": "Reason for pausing (when paused=true).",
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Execute tools on behalf of the agent.

    Wraps Daedalus operations as callable tools. Each tool returns a
    JSON-serializable result string.
    """

    def __init__(self, project_path: Path) -> None:
        self.project_path = Path(project_path)
        self.ledger = Ledger(self.project_path / "ledger")
        self.library = Library(self.project_path / "ledger" / "papers.jsonl")
        self.searcher = PaperSearcher()
        self._pending_confirmations: list[str] = []

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**tool_input)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return json.dumps({"error": f"Tool '{tool_name}' failed: {e}"})

    def _tool_get_context(self, mode: str = "full", recent_n: int = 10) -> dict:
        from .context import ContextBuilder
        builder = ContextBuilder(self.project_path)
        return {"context": builder.build(mode=mode, recent_n=recent_n)}

    def _tool_list_experiments(self, status: str | None = None, last_n: int | None = None) -> dict:
        if status:
            exps = self.ledger.by_status(ExperimentStatus(status))
        else:
            exps = self.ledger.all()
        if last_n:
            exps = exps[-last_n:]

        return {"experiments": [
            {
                "id": e.id,
                "status": e.status.value,
                "hypothesis": e.hypothesis.statement,
                "has_results": e.results is not None,
                "reflection": e.reflection.hypothesis_confirmed if e.reflection else None,
            }
            for e in exps
        ]}

    def _tool_get_experiment(self, exp_id: str) -> dict:
        exp = self.ledger.get(exp_id)
        if not exp:
            return {"error": f"Experiment {exp_id} not found"}
        return {"experiment": json.loads(exp.model_dump_json())}

    def _resolve_script_path(self, name: str) -> str:
        """Resolve a script name to its path using the scripts registry.

        If ``name`` matches a registered script name (e.g. "train"),
        returns the registered path (e.g. "train.py"). If not found
        in the registry, returns the name as-is (assumed to be a path).
        """
        import yaml
        from ..core.scripts_registry import ScriptsRegistry

        config_file = self.project_path / "daedalus.yaml"
        if not config_file.exists():
            return name

        config = yaml.safe_load(config_file.read_text()) or {}
        scripts_config = config.get("scripts", {})
        if not scripts_config:
            return name

        registry = ScriptsRegistry(scripts_config)
        spec = registry.get(name)
        if spec and spec.path:
            return spec.path
        return name

    def _tool_create_experiment(
        self,
        hypothesis_statement: str,
        rationale: str,
        script: str,
        predictions: list[dict] | None = None,
        papers: list[str] | None = None,
        script_args: dict | None = None,
        eval_script: str | None = None,
        eval_script_args: dict | None = None,
        results_adapter: str = "auto",
        env: dict | None = None,
        baseline_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        # Resolve script names (e.g. "train") to paths (e.g. "train.py")
        script = self._resolve_script_path(script)
        if eval_script:
            eval_script = self._resolve_script_path(eval_script)

        preds = [
            Prediction(
                metric=p["metric"],
                direction=p["direction"],
                expected_value=p.get("expected_value"),
            )
            for p in (predictions or [])
        ]

        hyp = Hypothesis(
            statement=hypothesis_statement,
            rationale=rationale,
            predictions=preds,
            papers=papers or [],
        )

        config = ExperimentConfig(
            script=script,
            script_args=script_args or {},
            eval_script=eval_script,
            eval_script_args=eval_script_args or {},
            results_adapter=results_adapter,
            env=env or {},
        )

        diff = None
        if baseline_id:
            base = self.ledger.get(baseline_id)
            if base:
                diff = config_diff(base.config, config)

        exp = Experiment(
            hypothesis=hyp,
            config=config,
            config_diff=diff,
            baseline_id=baseline_id,
            tags=tags or [],
        )

        self.ledger.append(exp)
        return {"created": exp.id, "status": "draft"}

    def _tool_launch_experiment(
        self, exp_id: str, host: str | None = None, plan_step_id: str | None = None,
    ) -> dict:
        exp = self.ledger.get(exp_id)
        if not exp:
            return {"error": f"Experiment {exp_id} not found"}

        if exp.status not in (ExperimentStatus.DRAFT, ExperimentStatus.QUEUED):
            return {"error": f"Cannot launch: status is {exp.status.value}"}

        # --- Plan launch gate ---
        from .plan import PlanManager
        plan_mgr = PlanManager(self.project_path / "ledger")
        plan = plan_mgr.load()

        if plan is not None:
            # Check guardrails
            ok, reason = plan_mgr.check_guardrails()
            if not ok:
                return {"error": f"Launch blocked: {reason}", "plan_paused": True}

            # Determine confirmation requirement
            needs_confirm = False
            if plan.autonomy == "supervised":
                needs_confirm = True
            elif plan_step_id:
                step = next((s for s in plan.steps if s.id == plan_step_id), None)
                if step and step.requires_confirmation:
                    needs_confirm = True

            if needs_confirm:
                return {
                    "needs_confirmation": True,
                    "exp_id": exp_id,
                    "plan_step_id": plan_step_id,
                    "autonomy": plan.autonomy,
                    "message": (
                        "Human confirmation required before launch. "
                        "Call human_confirm to approve, or switch to autonomous mode "
                        "with update_plan(autonomy='autonomous')."
                    ),
                }

        # --- Launch ---
        runner = create_runner(self.project_path, host=host)
        work_dir = self.project_path / "runs" / exp_id
        work_dir.mkdir(parents=True, exist_ok=True)

        if exp.status == ExperimentStatus.DRAFT:
            exp = exp.transition(ExperimentStatus.QUEUED)
        exp = exp.transition(ExperimentStatus.RUNNING)

        run_id = runner.launch(exp, work_dir)
        exp = exp.model_copy(update={"run_id": run_id})
        self.ledger.update(exp)

        # Detect runner type for response hints
        is_ssh = run_id.startswith("ssh:")
        result: dict[str, Any] = {
            "launched": exp_id,
            "run_id": run_id,
            "work_dir": str(work_dir),
            "runner": "ssh" if is_ssh else "local",
            "auto_synced": is_ssh,
            "monitoring": {
                "poll_with": "poll_experiment",
                "suggested_poll_interval_seconds": 60 if is_ssh else 10,
                "while_waiting": [
                    "Review results of previous experiments",
                    "Search for related papers with search_papers",
                    "Prepare the next experiment in the plan",
                    "Check experiment logs with get_experiment_logs",
                ],
            },
        }

        # --- Auto-bookkeeping: link to plan step ---
        if plan is not None and plan_step_id:
            try:
                plan_mgr.update_step(plan_step_id, status="running", experiment_id=exp_id)
                plan_mgr.increment_experiments_run()
                result["plan_step_linked"] = plan_step_id
            except ValueError as e:
                logger.warning("Plan auto-link failed: %s", e)

        return result

    def _tool_poll_experiment(self, exp_id: str) -> dict:
        exp = self.ledger.get(exp_id)
        if not exp:
            return {"error": f"Experiment {exp_id} not found"}
        if not exp.run_id:
            return {"error": f"Experiment {exp_id} has no run_id"}

        runner = create_runner(self.project_path)
        status = runner.poll(exp.run_id)

        result: dict[str, Any] = {"exp_id": exp_id, "state": status.state}
        if status.progress:
            result["progress"] = status.progress
        if status.error:
            result["error"] = status.error

        # Auto-transition on completion
        if status.state == "completed" and exp.status == ExperimentStatus.RUNNING:
            exp = exp.transition(ExperimentStatus.COMPLETED)
            try:
                results = runner.fetch_results(exp.run_id)
                exp = exp.model_copy(update={"results": results})
                result["results"] = results
            except Exception as e:
                result["fetch_error"] = str(e)
            self.ledger.update(exp)

            # Auto-bookkeeping: update plan step
            self._plan_on_experiment_done(exp_id, "done")

        elif status.state == "failed" and exp.status == ExperimentStatus.RUNNING:
            exp = exp.transition(ExperimentStatus.FAILED)
            self.ledger.update(exp)

            # Auto-bookkeeping: update plan step
            paused, reason = self._plan_on_experiment_done(exp_id, "failed")
            if paused:
                result["plan_paused"] = True
                result["plan_pause_reason"] = reason

        return result

    def _plan_on_experiment_done(
        self, exp_id: str, outcome: str,
    ) -> tuple[bool, str | None]:
        """Auto-update plan step when experiment completes/fails."""
        from .plan import PlanManager
        plan_mgr = PlanManager(self.project_path / "ledger")
        step = plan_mgr.find_step_by_experiment(exp_id)
        if step is None:
            return False, None

        try:
            plan_mgr.update_step(step.id, status=outcome)
            if outcome == "failed":
                return plan_mgr.record_failure()
            else:
                plan_mgr.record_success()
        except Exception as e:
            logger.warning("Plan auto-bookkeeping failed: %s", e)
        return False, None

    def _tool_record_results(self, exp_id: str, results: dict) -> dict:
        exp = self.ledger.get(exp_id)
        if not exp:
            return {"error": f"Experiment {exp_id} not found"}

        # Auto-transition to completed
        if exp.status == ExperimentStatus.DRAFT:
            exp = exp.transition(ExperimentStatus.QUEUED)
        if exp.status == ExperimentStatus.QUEUED:
            exp = exp.transition(ExperimentStatus.RUNNING)
        if exp.status == ExperimentStatus.RUNNING:
            exp = exp.transition(ExperimentStatus.COMPLETED)

        exp = exp.model_copy(update={"results": results})
        self.ledger.update(exp)
        return {"recorded": exp_id, "status": exp.status.value}

    def _tool_add_reflection(
        self,
        exp_id: str,
        hypothesis_confirmed: str,
        analysis: str,
        surprise: str | None = None,
        next_suggestions: list[str] | None = None,
    ) -> dict:
        exp = self.ledger.get(exp_id)
        if not exp:
            return {"error": f"Experiment {exp_id} not found"}

        reflection = Reflection(
            hypothesis_confirmed=hypothesis_confirmed,
            analysis=analysis,
            surprise=surprise,
            next_suggestions=next_suggestions or [],
        )

        if exp.status == ExperimentStatus.COMPLETED:
            exp = exp.transition(ExperimentStatus.ANALYZED)

        exp = exp.model_copy(update={"reflection": reflection})

        # Auto-evaluate hypothesis
        if exp.results:
            flat = {}
            for en, metrics in exp.results.items():
                for k, v in metrics.items():
                    flat[f"{en}.{k}"] = v
            flat_base = None
            if exp.baseline_id:
                base = self.ledger.get(exp.baseline_id)
                if base and base.results:
                    flat_base = {}
                    for en, metrics in base.results.items():
                        for k, v in metrics.items():
                            flat_base[f"{en}.{k}"] = v
            exp = exp.model_copy(update={
                "hypothesis": exp.hypothesis.evaluate(flat, flat_base)
            })

        self.ledger.update(exp)

        # Auto-save key insight to research memory
        try:
            from .memory import ResearchMemory, ResearchNote
            memory = ResearchMemory(self.project_path / "ledger")
            category = "dead_end" if hypothesis_confirmed == "rejected" else "insight"
            summary = f"[{hypothesis_confirmed}] {exp.hypothesis.statement}: {analysis[:200]}"
            memory.save(ResearchNote(
                content=summary,
                category=category,
                experiment_ids=[exp_id],
            ))
        except Exception as e:
            logger.debug(f"Auto-save note failed: {e}")

        # Auto-bookkeeping: copy reflection to plan step notes
        try:
            from .plan import PlanManager
            plan_mgr = PlanManager(self.project_path / "ledger")
            step = plan_mgr.find_step_by_experiment(exp_id)
            if step is not None:
                summary = f"[{hypothesis_confirmed}] {analysis[:200]}"
                existing = step.notes
                new_notes = f"{existing}\n{summary}".strip() if existing else summary
                plan_mgr.update_step(step.id, notes=new_notes)
        except Exception as e:
            logger.debug("Plan reflection auto-copy failed: %s", e)

        return {"reflected": exp_id, "hypothesis_status": exp.hypothesis.status}

    def _tool_compare_experiments(self, exp_a: str, exp_b: str) -> dict:
        a = self.ledger.get(exp_a)
        b = self.ledger.get(exp_b)
        if not a:
            return {"error": f"Experiment {exp_a} not found"}
        if not b:
            return {"error": f"Experiment {exp_b} not found"}

        result: dict[str, Any] = {"exp_a": exp_a, "exp_b": exp_b}

        cdiff = config_diff(a.config, b.config)
        result["config_diff"] = cdiff

        if a.results and b.results:
            comps = compare_results(b.results, a.results)
            result["metric_comparison"] = [
                {
                    "metric": c.metric,
                    "baseline": c.baseline_value,
                    "current": c.current_value,
                    "delta": c.delta,
                    "improved": c.improved,
                }
                for c in comps
            ]

        return result

    def _tool_search_papers(self, query: str, limit: int = 5, source: str = "all") -> dict:
        results = self.searcher.search(query, limit=limit, source=source)
        return {"papers": [
            {
                "arxiv_id": p.arxiv_id,
                "acl_id": p.acl_id,
                "title": p.title,
                "year": p.year,
                "authors": p.authors[:3],
                "citation_count": p.citation_count,
                "abstract": p.abstract[:300],
                "venue": p.venue,
                "url": p.url,
            }
            for p in results
        ]}

    def _tool_add_paper(
        self,
        arxiv_id: str | None = None,
        acl_id: str | None = None,
        relevance_note: str = "",
        key_findings: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        if not arxiv_id and not acl_id:
            return {"error": "Provide either arxiv_id or acl_id"}

        paper = None
        if arxiv_id:
            paper = self.searcher.get_paper(f"ARXIV:{arxiv_id}")
            if not paper:
                return {"error": f"Paper arXiv:{arxiv_id} not found on Semantic Scholar"}
        elif acl_id:
            paper = self.searcher.get_acl_paper(acl_id)
            if not paper:
                return {"error": f"Paper ACL:{acl_id} not found on ACL Anthology"}

        paper.relevance_note = relevance_note
        paper.key_findings = key_findings or []
        paper.tags = tags or []
        self.library.add(paper)
        return {
            "added": paper.title,
            "arxiv_id": paper.arxiv_id,
            "acl_id": paper.acl_id,
        }

    def _tool_list_papers(self, tag: str | None = None) -> dict:
        papers = self.library.by_tag(tag) if tag else self.library.all()
        return {"papers": [
            {
                "arxiv_id": p.arxiv_id,
                "acl_id": p.acl_id,
                "title": p.title,
                "tags": p.tags,
                "relevance_note": p.relevance_note,
                "venue": p.venue,
            }
            for p in papers
        ]}

    def _tool_get_experiment_logs(self, exp_id: str, tail: int = 50) -> dict:
        exp = self.ledger.get(exp_id)
        if not exp:
            return {"error": f"Experiment {exp_id} not found"}
        if not exp.run_id:
            return {"error": "No run_id"}

        runner = create_runner(self.project_path)
        log_text = runner.logs(exp.run_id, tail=tail)
        return {"logs": log_text}

    def _tool_suggest_next(
        self,
        target_metric: str,
        higher_is_better: bool = True,
        focus_param: str | None = None,
    ) -> dict:
        import yaml
        from ..core.scripts_registry import ScriptsRegistry
        from .strategies import suggest_next_params

        config_file = self.project_path / "daedalus.yaml"
        if not config_file.exists():
            return {"error": "No daedalus.yaml found"}

        config = yaml.safe_load(config_file.read_text()) or {}
        scripts_config = config.get("scripts", {})
        registry = ScriptsRegistry(scripts_config)

        experiments = self.ledger.all()
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric=target_metric,
            higher_is_better=higher_is_better,
            focus_param=focus_param,
        )

        # Separate suggestions from convergence notices
        active_suggestions = [s for s in suggestions if not s.get("converged")]
        converged_params = [s for s in suggestions if s.get("converged")]

        # Auto-save convergences to memory (important for future sessions)
        if converged_params:
            try:
                from .memory import ResearchMemory, ResearchNote
                memory = ResearchMemory(self.project_path / "ledger")
                for cp in converged_params:
                    memory.save(ResearchNote(
                        content=(
                            f"Parameter '{cp['parameter']}' converged "
                            f"({cp['reason']}): {cp.get('detail', '')}. "
                            f"Best: {cp.get('best_value')}→{cp.get('best_metric')}"
                        ),
                        category="convergence",
                    ))
            except Exception as e:
                logger.debug(f"Auto-save convergence note failed: {e}")

        return {
            "suggestions": active_suggestions,
            "converged_params": converged_params,
            "experiments_analyzed": len([
                e for e in experiments
                if e.results and e.status.value in ("completed", "analyzed")
            ]),
        }

    def _tool_list_scripts(self, name: str | None = None) -> dict:
        import yaml
        from ..core.scripts_registry import ScriptsRegistry
        config_file = self.project_path / "daedalus.yaml"
        if not config_file.exists():
            return {"error": "No daedalus.yaml found"}
        config = yaml.safe_load(config_file.read_text()) or {}
        scripts_config = config.get("scripts", {})
        if not scripts_config:
            return {"error": "No scripts registered in daedalus.yaml"}

        registry = ScriptsRegistry(scripts_config)

        if name:
            spec = registry.get(name)
            if not spec:
                return {"error": f"Script '{name}' not found. Available: {list(registry.scripts.keys())}"}
            return {
                "name": name,
                "path": spec.path,
                "description": spec.description,
                "parameters": {
                    pname: {
                        "type": param.type,
                        "default": param.default,
                        "required": param.required,
                        "description": param.description,
                        "choices": param.choices,
                        "range": param.range,
                    }
                    for pname, param in spec.parameters.items()
                },
                "defaults": registry.get_defaults(name),
            }

        return {
            "scripts": {
                sname: {
                    "path": spec.path,
                    "description": spec.description,
                    "parameter_count": len(spec.parameters),
                }
                for sname, spec in registry.scripts.items()
            }
        }

    def _tool_inspect_dataset(self, path: str, sample_n: int | None = None) -> dict:
        from ..data.inspector import DatasetInspector
        inspector = DatasetInspector()
        report = inspector.inspect(path, sample_n=sample_n)
        return json.loads(report.model_dump_json())

    def _tool_validate_dataset(
        self,
        path: str,
        required_fields: list[str] | None = None,
        format: str | None = None,
        max_duplicate_ratio: float = 0.01,
        max_empty_ratio: float = 0.05,
        check_class_balance: bool = False,
        label_field: str | None = None,
        eval_path: str | None = None,
        prompt_field: str = "prompt",
        sample_n: int | None = None,
    ) -> dict:
        from ..data.validator import DatasetValidator, ValidatorConfig
        config = ValidatorConfig(
            required_fields=required_fields,
            format=format,
            max_duplicate_ratio=max_duplicate_ratio,
            max_empty_ratio=max_empty_ratio,
            check_class_balance=check_class_balance,
            label_field=label_field,
            eval_path=eval_path,
            prompt_field=prompt_field,
        )
        validator = DatasetValidator(config)
        report = validator.validate(path, sample_n=sample_n)
        return json.loads(report.model_dump_json())

    def _tool_human_confirm(self, question: str, context: str = "") -> dict:
        self._pending_confirmations.append(question)
        return {"status": "pending", "question": question, "context": context}

    def _tool_batch_run(
        self, exp_ids: list[str], hosts: list[str] | None = None,
    ) -> dict:
        """Launch multiple experiments distributed across hosts."""
        hosts_list = hosts or [None]
        launched = []
        failed = []

        for i, exp_id in enumerate(exp_ids):
            exp = self.ledger.get(exp_id)
            if not exp:
                failed.append({"exp_id": exp_id, "error": "not found"})
                continue
            if exp.status not in (ExperimentStatus.DRAFT, ExperimentStatus.QUEUED):
                failed.append({"exp_id": exp_id, "error": f"status is {exp.status.value}"})
                continue

            target_host = hosts_list[i % len(hosts_list)]
            try:
                runner = create_runner(self.project_path, host=target_host)
                work_dir = self.project_path / "runs" / exp_id
                work_dir.mkdir(parents=True, exist_ok=True)

                if exp.status == ExperimentStatus.DRAFT:
                    exp = exp.transition(ExperimentStatus.QUEUED)
                exp = exp.transition(ExperimentStatus.RUNNING)

                run_id = runner.launch(exp, work_dir)
                exp = exp.model_copy(update={"run_id": run_id})
                self.ledger.update(exp)

                launched.append({
                    "exp_id": exp_id,
                    "host": target_host or "default",
                    "run_id": run_id,
                })
            except Exception as e:
                exp = exp.transition(ExperimentStatus.FAILED)
                self.ledger.update(exp)
                failed.append({"exp_id": exp_id, "error": str(e)})

        return {
            "launched": launched,
            "failed": failed,
            "total_launched": len(launched),
            "total_failed": len(failed),
        }

    def _tool_save_note(
        self,
        content: str,
        category: str = "general",
        experiment_ids: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Save a research note to persistent memory."""
        from .memory import ResearchMemory, ResearchNote

        memory = ResearchMemory(self.project_path / "ledger")
        note = ResearchNote(
            content=content,
            category=category,
            experiment_ids=experiment_ids or [],
            tags=tags or [],
        )
        memory.save(note)
        return {
            "saved": True,
            "category": category,
            "total_notes": len(memory.all()),
        }

    def _tool_read_notes(
        self,
        category: str | None = None,
        experiment_id: str | None = None,
        search: str | None = None,
        last_n: int = 20,
    ) -> dict:
        """Read research notes from persistent memory."""
        from .memory import ResearchMemory

        memory = ResearchMemory(self.project_path / "ledger")

        if category:
            notes = memory.by_category(category)
        elif experiment_id:
            notes = memory.by_experiment(experiment_id)
        elif search:
            notes = memory.search(search)
        else:
            notes = memory.recent(last_n)

        return {
            "notes": [
                {
                    "timestamp": n.timestamp.isoformat(),
                    "category": n.category,
                    "content": n.content,
                    "experiment_ids": n.experiment_ids,
                    "tags": n.tags,
                }
                for n in notes[-last_n:]
            ],
            "total_notes": len(memory.all()),
            "formatted": memory.format_for_context(limit=last_n),
        }

    def _tool_update_program(
        self,
        action: str,
        content: str,
        section_heading: str | None = None,
    ) -> dict:
        """Update program.md — the research goals document."""
        program_file = self.project_path / "program.md"

        if action == "rewrite":
            program_file.write_text(content)
            return {"updated": True, "action": "rewrite", "size": len(content)}

        if action == "append":
            existing = program_file.read_text() if program_file.exists() else ""
            new_content = existing.rstrip() + "\n\n" + content + "\n"
            program_file.write_text(new_content)
            return {"updated": True, "action": "append", "size": len(new_content)}

        if action == "replace_section":
            if not section_heading:
                return {"error": "section_heading required for replace_section action"}
            if not program_file.exists():
                return {"error": "program.md does not exist"}

            existing = program_file.read_text()
            lines = existing.split("\n")
            new_lines: list[str] = []
            in_section = False
            section_level = 0
            replaced = False

            for line in lines:
                # Check if this is a heading
                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    heading_text = line.lstrip("#").strip()

                    if heading_text == section_heading:
                        # Found the section to replace
                        in_section = True
                        section_level = level
                        new_lines.append(line)  # keep the heading
                        new_lines.append("")
                        new_lines.append(content)
                        new_lines.append("")
                        replaced = True
                        continue
                    elif in_section and level <= section_level:
                        # Next section at same or higher level — stop replacing
                        in_section = False

                if not in_section:
                    new_lines.append(line)

            if not replaced:
                return {"error": f"Section '{section_heading}' not found in program.md"}

            program_file.write_text("\n".join(new_lines))
            return {"updated": True, "action": "replace_section", "section": section_heading}

        return {"error": f"Unknown action: {action}"}

    def _tool_get_paper_details(
        self,
        arxiv_id: str | None = None,
        acl_id: str | None = None,
        query: str | None = None,
    ) -> dict:
        """Get full paper details from library, Semantic Scholar, or ACL Anthology."""
        paper_id = arxiv_id or acl_id
        if paper_id:
            # Try local library first (searches by any ID type)
            paper = self.library.get(paper_id)
            if paper:
                return {
                    "source": "library",
                    "arxiv_id": paper.arxiv_id,
                    "acl_id": paper.acl_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "abstract": paper.abstract,
                    "key_findings": paper.key_findings,
                    "relevance_note": paper.relevance_note,
                    "tags": paper.tags,
                    "url": paper.url,
                    "citation_count": paper.citation_count,
                    "venue": paper.venue,
                }
            # Fetch from external API
            if arxiv_id:
                paper = self.searcher.get_paper(f"ARXIV:{arxiv_id}")
            else:
                paper = self.searcher.get_acl_paper(acl_id)

            if paper:
                return {
                    "source": "semantic_scholar" if arxiv_id else "acl_anthology",
                    "arxiv_id": paper.arxiv_id,
                    "acl_id": paper.acl_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "abstract": paper.abstract,
                    "url": paper.url,
                    "citation_count": paper.citation_count,
                    "venue": paper.venue,
                    "note": "Paper not in local library. Use add_paper to save it.",
                }
            id_label = f"arXiv:{arxiv_id}" if arxiv_id else f"ACL:{acl_id}"
            return {"error": f"Paper {id_label} not found"}

        if query:
            results = self.library.search_local(query)
            if results:
                return {
                    "source": "library",
                    "papers": [
                        {
                            "arxiv_id": p.arxiv_id,
                            "acl_id": p.acl_id,
                            "title": p.title,
                            "abstract": p.abstract[:300],
                            "key_findings": p.key_findings,
                            "relevance_note": p.relevance_note,
                            "tags": p.tags,
                        }
                        for p in results
                    ],
                }
            return {"error": f"No papers matching '{query}' in library"}

        return {"error": "Provide arxiv_id, acl_id, or query"}

    def _tool_literature_review(
        self,
        focus: str | None = None,
        include_search: bool = False,
        search_queries: list[str] | None = None,
    ) -> dict:
        """Generate a structured literature review."""
        papers = self.library.all()
        experiments = self.ledger.all()
        completed = [
            e for e in experiments
            if e.status.value in ("completed", "analyzed") and e.results
        ]

        # Filter papers by focus topic if given
        if focus and papers:
            focus_lower = focus.lower()
            papers = [
                p for p in papers
                if focus_lower in p.title.lower()
                or focus_lower in p.abstract.lower()
                or focus_lower in " ".join(p.tags).lower()
                or focus_lower in p.relevance_note.lower()
                or focus_lower in " ".join(p.key_findings).lower()
            ]

        # Search for new papers — auto-enable if library is empty
        auto_search = include_search or (not papers and completed)
        new_papers: list[dict] = []
        if auto_search:
            # Build queries from focus, user queries, and experiment hypotheses
            queries = list(search_queries or [])
            if focus and focus not in queries:
                queries.append(focus)
            # Auto-generate queries from experiment hypotheses if no explicit queries
            if not queries and completed:
                # Extract key terms from recent hypotheses
                seen = set()
                for exp in completed[-5:]:
                    # Use hypothesis statement as search query (truncated)
                    q = exp.hypothesis.statement[:100]
                    if q not in seen:
                        queries.append(q)
                        seen.add(q)
            for q in queries[:5]:  # limit to 5 queries
                try:
                    results = self.searcher.search(q, limit=5)
                    for p in results:
                        # Skip if already in library or already found
                        if any(lib_p.uid == p.uid for lib_p in self.library.all()):
                            continue
                        if any(np.get("arxiv_id") == p.arxiv_id for np in new_papers if p.arxiv_id):
                            continue
                        new_papers.append({
                            "arxiv_id": p.arxiv_id,
                            "title": p.title,
                            "year": p.year,
                            "abstract": p.abstract[:200],
                            "citation_count": p.citation_count,
                        })
                except Exception as e:
                    logger.warning(f"Paper search failed for '{q}': {e}")

        # Build review
        review_sections: list[str] = []

        # Section 1: Papers overview
        if papers:
            review_sections.append(f"## Papers in Library ({len(papers)})\n")
            for p in papers:
                line = f"- **{p.title}**"
                if p.year:
                    line += f" ({p.year})"
                if p.arxiv_id:
                    line += f" [arXiv:{p.arxiv_id}]"
                if p.key_findings:
                    for kf in p.key_findings:
                        line += f"\n  - {kf}"
                review_sections.append(line)
        else:
            review_sections.append("## Papers\n\nNo papers in library" +
                                   (f" matching '{focus}'" if focus else "") + ".")

        # Section 2: Papers vs experiments
        if papers and completed:
            review_sections.append("\n## Papers vs Experiments\n")

            # Find which papers support which experiment outcomes
            for p in papers:
                linked_exps = []
                for exp in completed:
                    if exp.hypothesis.papers and p.arxiv_id in exp.hypothesis.papers:
                        status = "confirmed" if exp.reflection and exp.reflection.hypothesis_confirmed == "confirmed" else \
                                 "rejected" if exp.reflection and exp.reflection.hypothesis_confirmed == "rejected" else \
                                 "untested"
                        linked_exps.append(f"{exp.id} ({status})")

                if linked_exps:
                    review_sections.append(
                        f"- **{p.title}**: linked to {', '.join(linked_exps)}"
                    )

            # Papers not linked to any experiment
            unlinked = [
                p for p in papers
                if not any(
                    p.arxiv_id and p.arxiv_id in (exp.hypothesis.papers or [])
                    for exp in completed
                )
            ]
            if unlinked:
                review_sections.append("\n**Papers not yet tested experimentally**:")
                for p in unlinked:
                    review_sections.append(f"- {p.title}")
                    if p.key_findings:
                        review_sections.append(f"  Key finding: {p.key_findings[0]}")

        # Section 3: Gaps
        review_sections.append("\n## Gaps & Suggestions\n")

        # Experiments without paper backing
        no_papers = [
            e for e in completed
            if not e.hypothesis.papers
        ]
        if no_papers:
            review_sections.append(f"- **{len(no_papers)} experiments** have no cited papers")

        # Rejected hypotheses that might need more literature
        rejected = [
            e for e in completed
            if e.reflection and e.reflection.hypothesis_confirmed == "rejected"
        ]
        if rejected:
            review_sections.append(
                f"- **{len(rejected)} rejected hypotheses** — consider searching "
                f"for papers explaining why these approaches fail"
            )

        if not papers:
            review_sections.append(
                "- **No papers in library** — use search_papers to find relevant work"
            )

        # Section 4: New papers found
        if new_papers:
            review_sections.append(f"\n## New Papers Found ({len(new_papers)})\n")
            for p in new_papers[:10]:
                line = f"- **{p['title']}**"
                if p.get("year"):
                    line += f" ({p['year']})"
                if p.get("arxiv_id"):
                    line += f" [arXiv:{p['arxiv_id']}]"
                if p.get("citation_count"):
                    line += f" — {p['citation_count']} citations"
                review_sections.append(line)
            review_sections.append(
                "\nUse `add_paper` to save relevant papers to the library."
            )

        review_text = "\n".join(review_sections)

        # Auto-save literature review to research memory
        try:
            from .memory import ResearchMemory, ResearchNote
            memory = ResearchMemory(self.project_path / "ledger")

            # Save review summary
            summary_parts = []
            if papers:
                summary_parts.append(f"{len(papers)} papers in library")
            if new_papers:
                summary_parts.append(f"{len(new_papers)} new papers found")
            if no_papers:
                summary_parts.append(f"{len(no_papers)} experiments without paper backing")
            if rejected:
                summary_parts.append(f"{len(rejected)} rejected hypotheses need literature")

            if summary_parts:
                memory.save(ResearchNote(
                    content=f"[literature review] {'; '.join(summary_parts)}."
                            + (f" Focus: {focus}" if focus else ""),
                    category="insight",
                ))

            # Save each new paper found as a separate note for discoverability
            if new_papers:
                top_papers = sorted(
                    new_papers,
                    key=lambda p: p.get("citation_count") or 0,
                    reverse=True,
                )[:5]
                titles = [p["title"][:80] for p in top_papers]
                memory.save(ResearchNote(
                    content=(
                        f"[literature review] Top new papers found: "
                        + "; ".join(titles)
                    ),
                    category="insight",
                ))
        except Exception as e:
            logger.debug(f"Auto-save literature review note failed: {e}")

        return {
            "review": review_text,
            "papers_in_library": len(self.library.all()),
            "papers_matched": len(papers),
            "new_papers_found": len(new_papers),
            "experiments_analyzed": len(completed),
        }

    def _tool_get_insights(
        self,
        target_metric: str = "",
        higher_is_better: bool = True,
        save_path: str | None = None,
    ) -> dict:
        """Generate aggregated research insights from experiment history."""
        import yaml
        from ..core.scripts_registry import ScriptsRegistry
        from .insights import InsightsGenerator

        registry = None
        config_file = self.project_path / "daedalus.yaml"
        if config_file.exists():
            config = yaml.safe_load(config_file.read_text()) or {}
            scripts_config = config.get("scripts", {})
            if scripts_config:
                registry = ScriptsRegistry(scripts_config)

        experiments = self.ledger.all()
        generator = InsightsGenerator(
            experiments=experiments,
            registry=registry,
            target_metric=target_metric,
            higher_is_better=higher_is_better,
        )

        insights = generator.generate()

        if save_path:
            generator.save(Path(save_path))

        # Auto-save key insights to research memory
        try:
            from .memory import ResearchMemory, ResearchNote
            memory = ResearchMemory(self.project_path / "ledger")

            # Save top config insight
            if generator.completed and target_metric:
                from .strategies import _get_metric, _find_best
                best = _find_best(generator.completed, target_metric, higher_is_better)
                best_val = _get_metric(best, target_metric)
                if best_val is not None:
                    args_str = ", ".join(f"{k}={v}" for k, v in best.config.script_args.items())
                    memory.save(ResearchNote(
                        content=(
                            f"[insights] Best config for {target_metric}: "
                            f"{best.id} ({target_metric}={best_val:.4f}). "
                            f"Config: {args_str}"
                        ),
                        category="insight",
                        experiment_ids=[best.id],
                    ))

            # Save dead ends summary
            rejected = [
                e for e in generator.completed
                if e.reflection and e.reflection.hypothesis_confirmed == "rejected"
            ]
            if rejected:
                dead_end_ids = [e.id for e in rejected[-3:]]
                memory.save(ResearchNote(
                    content=(
                        f"[insights] {len(rejected)} rejected hypotheses identified. "
                        f"Recent: {', '.join(dead_end_ids)}"
                    ),
                    category="dead_end",
                    experiment_ids=dead_end_ids,
                ))
        except Exception as e:
            logger.debug(f"Auto-save insights note failed: {e}")

        return {
            "insights": insights,
            "experiments_total": len(experiments),
            "experiments_completed": len(generator.completed),
        }

    # --- Remote tools ---

    def _tool_remote_exec(
        self,
        command: str,
        host: str | None = None,
        mutating: bool = False,
        confirmed: bool = False,
        timeout: int = 30,
    ) -> dict:
        """Execute a command on the remote SSH host."""
        from ..runners.factory import create_runner as _create_runner
        from ..runners.ssh import SSHRunner

        runner = _create_runner(self.project_path, host=host)
        if not isinstance(runner, SSHRunner):
            return {"error": "remote_exec is only available for SSH runners"}

        # Safety: mutating commands require explicit confirmation
        if mutating and not confirmed:
            return {
                "needs_confirmation": True,
                "command": command,
                "host": host or "default",
                "message": (
                    "This command modifies remote state. "
                    "Show it to the user and get approval, then call again "
                    "with confirmed=true."
                ),
            }

        # Cap timeout
        timeout = min(max(timeout, 5), 600)

        try:
            result = runner._run_ssh(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {
                "error": f"Command timed out after {timeout}s",
                "hint": f"Retry with a higher timeout (max 600s): timeout={min(timeout * 2, 600)}",
                "host": runner.config.host,
            }

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "host": runner.config.host,
        }

    # --- Plan tools ---

    def _tool_get_plan(self) -> dict:
        """Get the current research plan."""
        from .plan import PlanManager

        manager = PlanManager(self.project_path / "ledger")
        plan = manager.load()
        if plan is None:
            return {"plan": None, "message": "No plan exists. Use create_plan to create one."}

        next_step = manager.next_actionable()
        return {
            "goal": plan.goal,
            "version": plan.version,
            "autonomy": plan.autonomy,
            "paused": plan.paused,
            "pause_reason": plan.pause_reason,
            "experiments_run": plan.experiments_run,
            "max_experiments": plan.max_experiments,
            "consecutive_failures": plan.consecutive_failures,
            "max_consecutive_failures": plan.max_consecutive_failures,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "rationale": s.rationale,
                    "expected_outcome": s.expected_outcome,
                    "priority": s.priority,
                    "status": s.status.value,
                    "experiment_id": s.experiment_id,
                    "depends_on": s.depends_on,
                    "notes": s.notes,
                    "requires_confirmation": s.requires_confirmation,
                }
                for s in plan.steps
            ],
            "next_actionable": next_step.id if next_step else None,
            "progress": {
                "total": len(plan.steps),
                "done": sum(1 for s in plan.steps if s.status.value == "done"),
                "failed": sum(1 for s in plan.steps if s.status.value == "failed"),
                "pending": sum(1 for s in plan.steps if s.status.value == "pending"),
                "skipped": sum(1 for s in plan.steps if s.status.value == "skipped"),
            },
        }

    def _tool_create_plan(
        self,
        goal: str,
        steps: list[dict],
        autonomy: str = "autonomous",
        max_experiments: int | None = None,
        max_consecutive_failures: int = 2,
    ) -> dict:
        """Create a new research plan."""
        from .plan import PlanManager

        if not steps:
            return {"error": "Plan must have at least one step."}

        manager = PlanManager(self.project_path / "ledger")
        plan = manager.create(
            goal, steps,
            autonomy=autonomy,
            max_experiments=max_experiments,
            max_consecutive_failures=max_consecutive_failures,
        )

        # Auto-save a note about the plan
        try:
            from .memory import ResearchMemory, ResearchNote
            memory = ResearchMemory(self.project_path / "ledger")
            step_summaries = "; ".join(s.description for s in plan.steps[:5])
            memory.save(ResearchNote(
                content=f"Plan created ({len(plan.steps)} steps): {plan.goal}. Steps: {step_summaries}",
                category="decision",
            ))
        except Exception as e:
            logger.debug(f"Auto-save plan note failed: {e}")

        return {
            "created": True,
            "goal": plan.goal,
            "steps_count": len(plan.steps),
            "step_ids": [s.id for s in plan.steps],
            "autonomy": plan.autonomy,
            "max_experiments": plan.max_experiments,
            "max_consecutive_failures": plan.max_consecutive_failures,
        }

    def _tool_update_plan_step(
        self,
        step_id: str,
        status: str | None = None,
        priority: int | None = None,
        experiment_id: str | None = None,
        notes: str | None = None,
        requires_confirmation: bool | None = None,
    ) -> dict:
        """Update a step in the research plan."""
        from .plan import PlanManager

        manager = PlanManager(self.project_path / "ledger")

        kwargs: dict[str, Any] = {}
        if status is not None:
            kwargs["status"] = status
        if priority is not None:
            kwargs["priority"] = priority
        if experiment_id is not None:
            kwargs["experiment_id"] = experiment_id
        if notes is not None:
            kwargs["notes"] = notes
        if requires_confirmation is not None:
            kwargs["requires_confirmation"] = requires_confirmation

        if not kwargs:
            return {"error": "No fields to update. Provide at least one of: status, priority, experiment_id, notes, requires_confirmation."}

        try:
            step = manager.update_step(step_id, **kwargs)
            return {
                "updated": True,
                "step_id": step.id,
                "status": step.status.value,
                "priority": step.priority,
                "experiment_id": step.experiment_id,
                "notes": step.notes,
                "requires_confirmation": step.requires_confirmation,
            }
        except ValueError as e:
            return {"error": str(e)}

    def _tool_update_plan(
        self,
        autonomy: str | None = None,
        max_experiments: int | None = None,
        max_consecutive_failures: int | None = None,
        paused: bool | None = None,
        pause_reason: str | None = None,
    ) -> dict:
        """Update plan-level settings."""
        from .plan import PlanManager

        manager = PlanManager(self.project_path / "ledger")
        plan = manager.load()
        if plan is None:
            return {"error": "No plan exists. Use create_plan first."}

        kwargs: dict[str, Any] = {}
        if autonomy is not None:
            kwargs["autonomy"] = autonomy
        if max_experiments is not None:
            kwargs["max_experiments"] = max_experiments
        if max_consecutive_failures is not None:
            kwargs["max_consecutive_failures"] = max_consecutive_failures
        if paused is not None:
            kwargs["paused"] = paused
            # Auto-clear pause reason when resuming
            if not paused:
                kwargs["pause_reason"] = None
                kwargs["consecutive_failures"] = 0
        if pause_reason is not None:
            kwargs["pause_reason"] = pause_reason

        if not kwargs:
            return {"error": "No fields to update."}

        try:
            plan = manager.update_plan_settings(**kwargs)
            return {
                "updated": True,
                "autonomy": plan.autonomy,
                "paused": plan.paused,
                "pause_reason": plan.pause_reason,
                "max_experiments": plan.max_experiments,
                "experiments_run": plan.experiments_run,
                "max_consecutive_failures": plan.max_consecutive_failures,
                "consecutive_failures": plan.consecutive_failures,
            }
        except ValueError as e:
            return {"error": str(e)}
