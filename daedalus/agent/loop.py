"""Autonomous research agent loop using Anthropic API with tool-use.

The agent reads the research context, reasons about the next experiment,
designs it, launches it, waits for results, and reflects. Each step uses
a specialized system prompt and a subset of tools.

The loop supports three modes:
- interactive: pauses for human confirmation before launching experiments
- autonomous: runs without confirmation (with optional cost gate)
- single_step: runs one reasoning step and returns (for Claude Code integration)
"""

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from anthropic import Anthropic

from .tools import TOOL_SCHEMAS, ToolExecutor

logger = logging.getLogger(__name__)

# Default model for agent reasoning
DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AgentMode(str, Enum):
    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"
    SINGLE_STEP = "single_step"


class ResearchAgent:
    """Autonomous ML research agent powered by Claude.

    Uses Daedalus tools to manage the full research loop:
    reason → design → run → evaluate → reflect → repeat.
    """

    def __init__(
        self,
        project_path: Path,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        mode: AgentMode = AgentMode.INTERACTIVE,
        max_cost_usd: float | None = None,
        on_human_confirm: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.project_path = Path(project_path)
        self.model = model
        self.max_tokens = max_tokens
        self.mode = mode
        self.max_cost_usd = max_cost_usd
        self._on_human_confirm = on_human_confirm or self._default_confirm

        self.client = Anthropic()
        self.executor = ToolExecutor(self.project_path)

        # Load system prompt
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt from program.md, scripts registry, and agent identity."""
        program_file = self.project_path / "program.md"
        program = program_file.read_text() if program_file.exists() else ""

        # Load scripts registry
        from ..core.scripts_registry import ScriptsRegistry
        import yaml

        scripts_section = ""
        config_file = self.project_path / "daedalus.yaml"
        if config_file.exists():
            config = yaml.safe_load(config_file.read_text()) or {}
            scripts_config = config.get("scripts", {})
            if scripts_config:
                registry = ScriptsRegistry(scripts_config)
                scripts_section = f"""
## Available Scripts

These are the scripts you can use with create_experiment. Use the exact paths
and parameter names listed here. Only change ONE parameter at a time from the defaults.

{registry.format_for_context()}
"""

        return f"""You are Daedalus, an autonomous AI research assistant specialized in ML experiments.

Your role is to help a researcher by:
1. Analyzing experiment history and forming hypotheses
2. Designing experiments (one variable at a time)
3. Launching and monitoring experiments
4. Analyzing results and reflecting on outcomes
5. Searching relevant papers to inform decisions
6. Creating and following structured research plans

## Research Principles
- Always start with the simplest baseline
- Change ONE variable at a time
- Make falsifiable predictions before running experiments
- Analyze WHY results happened, not just WHAT happened
- Cite papers to justify decisions
- When uncertain, ask the human researcher

## Current Research Program
{program}
{scripts_section}
## Research Plan

Use structured plans to organize multi-step research:
- `get_plan` — Check if a plan exists, see progress, autonomy mode, and guardrail status
- `create_plan` — Create a plan with autonomy mode and guardrails (3-7 steps)
- `update_plan_step` — Mark steps done/failed/skipped, add notes, set requires_confirmation
- `update_plan` — Switch autonomy mode (supervised/autonomous), adjust guardrails, pause/resume

When following a plan, pick the next actionable step (highest priority with all dependencies met),
design the experiment, run it with `plan_step_id` to enable auto-bookkeeping, poll until done,
reflect, then move on. Auto-bookkeeping handles step status updates automatically.

Autonomy modes:
- `autonomous` (default for MCP): launches proceed directly without extra confirmation
- `supervised`: launch_experiment returns needs_confirmation — for explicit human gate

Always create plans with `autonomy="autonomous"` unless the user explicitly asks for supervised mode.
Claude Code already provides human-in-the-loop via the chat interface.

Guardrails: `max_experiments`, `max_consecutive_failures`, per-step `requires_confirmation`.
When triggered, the plan pauses. Resume with `update_plan(paused=false)`.

## How to Design Experiments

When creating an experiment with create_experiment:
1. Set `script` to the script path from the Available Scripts section
2. Set `script_args` with the parameters you want to change (only override what differs from defaults)
3. If the project has a separate eval script, set `eval_script` and `eval_script_args`
4. Always set a `baseline_id` to compare against the most relevant previous experiment
5. Make specific, falsifiable predictions about which metrics will change and in which direction
6. Use `inspect_dataset` and `validate_dataset` to check data BEFORE launching training
7. If a research plan exists, pass `plan_step_id` when launching to enable auto-bookkeeping

## How to Analyze Results

After an experiment completes:
1. Use `get_experiment` to see full details
2. Use `compare_experiments` to see metric deltas vs baseline
3. Use `add_reflection` to record your analysis — be specific about WHY, not just WHAT
   (auto-bookkeeping copies the reflection to the linked plan step)
4. Suggest concrete next steps based on the results
5. If following a plan, check progress with `get_plan` and adjust priorities if needed

## Guidelines
- Use get_context to understand the full project state before making decisions
- Use compare_experiments to understand what changed between runs
- Always create a hypothesis with predictions BEFORE launching an experiment
- After results come in, ALWAYS add a reflection
- If you're unsure about a decision, use human_confirm to ask the researcher
- When starting a new research direction, consider creating a plan first
- Keep analyses concise but insightful
"""

    @staticmethod
    def _default_confirm(question: str, context: str) -> bool:
        """Default human confirmation via stdin."""
        print(f"\n{'='*60}")
        print(f"AGENT ASKS: {question}")
        if context:
            print(f"Context: {context}")
        print(f"{'='*60}")
        response = input("Approve? (y/n): ").strip().lower()
        return response in ("y", "yes")

    def _call_api(self, messages: list[dict]) -> Any:
        """Make a single API call with tools."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        return response

    def _process_tool_calls(self, response: Any, messages: list[dict]) -> tuple[Any, list[dict]]:
        """Process tool calls in a response, executing them and continuing the conversation."""
        while response.stop_reason == "tool_use":
            # Collect assistant message
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    logger.info(f"Tool call: {block.name}({json.dumps(block.input)[:200]})")

                    # Handle human_confirm specially
                    if block.name == "human_confirm":
                        question = block.input.get("question", "")
                        context = block.input.get("context", "")

                        if self.mode == AgentMode.AUTONOMOUS:
                            result = json.dumps({"approved": True, "note": "auto-approved (autonomous mode)"})
                        else:
                            approved = self._on_human_confirm(question, context)
                            result = json.dumps({"approved": approved})
                    else:
                        result = self.executor.execute(block.name, block.input)

                    logger.info(f"Tool result: {result[:200]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

            # Continue conversation
            response = self._call_api(messages)

        return response, messages

    def plan(self, prior_context: str = "") -> str:
        """Create a structured multi-step research plan.

        Analyzes the full research state and produces an ordered plan with
        3-7 concrete steps. Each step tests one variable with a clear
        expected outcome.

        Returns the agent's planning reasoning as text.
        """
        prompt = (
            "Create a structured research plan.\n\n"
            "1. Use read_notes to check research memory from previous sessions.\n"
            "2. Use get_context with mode='full' to understand the complete state.\n"
            "3. Use get_insights to see parameter sensitivity and explored ranges.\n"
            "4. Use suggest_next to see data-driven parameter suggestions.\n"
            "5. Search for relevant papers if it helps inform the plan.\n"
            "6. Create a plan with create_plan that has 3-7 concrete steps:\n"
            "   - Each step tests ONE variable with a clear expected outcome\n"
            "   - Order by priority (1=highest) and set depends_on for sequential steps\n"
            "   - Include the rationale (WHY this matters) for each step\n"
            "   - First step should be the simplest/most impactful change\n"
            "7. Explain your reasoning for the plan structure."
        )

        if prior_context:
            prompt = f"## Context from previous iteration\n{prior_context}\n\n{prompt}"

        messages = [{"role": "user", "content": prompt}]

        response = self._call_api(messages)
        response, messages = self._process_tool_calls(response, messages)

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_parts)

    def reason(self, prior_context: str = "") -> str:
        """Run a single reasoning step: analyze state and propose next experiment.

        Args:
            prior_context: Optional context from previous iterations (reflection, etc.)

        Returns the agent's reasoning as text.
        """
        prompt = (
            "Analyze the current research state. "
            "1. Use read_notes to check research memory from previous sessions.\n"
            "2. Use get_context with mode='reason' to understand where we are.\n"
            "3. Use suggest_next to see data-driven parameter suggestions.\n"
            "4. Use list_scripts to see available scripts and their parameters.\n"
            "5. Propose the next hypothesis to test. Be specific about:\n"
            "   - Which SINGLE parameter you want to change\n"
            "   - What value to try and why\n"
            "   - What metric you expect to improve and by roughly how much\n"
            "6. If you think a literature search would help, do one first."
        )

        if prior_context:
            prompt = f"## Context from previous iteration\n{prior_context}\n\n{prompt}"

        messages = [{"role": "user", "content": prompt}]

        response = self._call_api(messages)
        response, messages = self._process_tool_calls(response, messages)

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_parts)

    def design(self, hypothesis: str | None = None, prior_context: str = "") -> str:
        """Design the next experiment based on current state or a given hypothesis."""
        prompt = "Design the next experiment.\n\n"
        if hypothesis:
            prompt += f"The hypothesis is: {hypothesis}\n\n"

        prompt += (
            "Steps:\n"
            "1. Use list_scripts to see available scripts and their parameter specs.\n"
            "2. Use get_context with mode='design' to see recent experiments and their configs.\n"
            "3. Pick the most relevant completed experiment as baseline_id.\n"
            "4. Create the experiment with create_experiment:\n"
            "   - Set `script` to the exact path from list_scripts\n"
            "   - Set `script_args` with ONLY the parameters that differ from the baseline\n"
            "   - Set `baseline_id` to compare against\n"
            "   - Add specific predictions for each metric you expect to change\n"
            "5. Show what you changed and why."
        )

        if prior_context:
            prompt = f"## Context from previous iteration\n{prior_context}\n\n{prompt}"

        messages = [{"role": "user", "content": prompt}]
        response = self._call_api(messages)
        response, messages = self._process_tool_calls(response, messages)

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_parts)

    def reflect(self, exp_id: str) -> str:
        """Reflect on a completed experiment's results.

        Returns the reflection text, which is passed to the next iteration.
        """
        prompt = (
            f"Analyze the results of experiment {exp_id}.\n\n"
            f"Steps:\n"
            f"1. Use get_experiment to see full details.\n"
            f"2. Use compare_experiments to see metric deltas vs baseline.\n"
            f"3. Use suggest_next to see what the data suggests trying next.\n"
            f"4. Add a reflection with add_reflection. Be specific about:\n"
            f"   - Was the hypothesis confirmed, partially confirmed, or rejected?\n"
            f"   - WHY did the results turn out this way?\n"
            f"   - What does this tell us about the parameter's effect?\n"
            f"   - Concrete next_suggestions: what specific parameter value to try next\n"
            f"5. If this was a key decision or dead end, use save_note to record it.\n"
            f"6. If a research plan exists, use get_plan to review it.\n"
            f"   - Use update_plan_step to adjust priorities of future steps if needed.\n"
            f"   - If results suggest new experiments not in the plan, mention them.\n"
        )

        messages = [{"role": "user", "content": prompt}]
        response = self._call_api(messages)
        response, messages = self._process_tool_calls(response, messages)

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_parts)

    def run_loop(
        self,
        max_experiments: int = 3,
        poll_interval: int = 300,
        on_step: Callable[[str, str], None] | None = None,
        use_plan: bool = False,
    ) -> list[str]:
        """Run the full autonomous research loop.

        The loop maintains context between iterations: each reflection
        is passed to the next reason step, so the agent learns from
        its own experiments and iterates intelligently.

        When use_plan=True, the loop follows the research plan:
        - If no plan exists, creates one first
        - Picks the next actionable step as the hypothesis
        - After reflection, marks the step done

        Args:
            max_experiments: Maximum experiments to run before stopping.
            poll_interval: Seconds between status polls for running experiments.
            on_step: Optional callback(step_name, detail) for progress reporting.
            use_plan: If True, follow the research plan instead of free-form reasoning.

        Returns:
            List of experiment IDs created during this loop.
        """
        def _report(step: str, detail: str = "") -> None:
            logger.info(f"[{step}] {detail}")
            if on_step:
                on_step(step, detail)

        # Plan mode: ensure plan exists
        plan_manager = None
        if use_plan:
            from .plan import PlanManager
            plan_manager = PlanManager(self.project_path / "ledger")
            if plan_manager.load() is None:
                _report("PLAN", "No plan found, creating one...")
                plan_output = self.plan()
                _report("PLAN_DONE", plan_output[:200])

        experiment_ids: list[str] = []
        # Carry context between iterations
        iteration_context = ""
        current_step = None  # Track which plan step we're executing

        for i in range(max_experiments):
            _report("REASON", f"Experiment {i+1}/{max_experiments}")

            if use_plan and plan_manager:
                # Plan mode: pick next step from plan
                current_step = plan_manager.next_actionable()
                if current_step is None:
                    _report("PLAN_COMPLETE", "All plan steps done or skipped.")
                    break

                _report("PLAN_STEP", f"{current_step.id}: {current_step.description}")
                plan_manager.update_step(current_step.id, status="running")

                # Design using the plan step as hypothesis
                design_output = self.design(
                    hypothesis=current_step.description,
                    prior_context=(
                        f"Plan step {current_step.id} (P{current_step.priority}):\n"
                        f"Rationale: {current_step.rationale}\n"
                        f"Expected outcome: {current_step.expected_outcome}\n"
                        + (f"\n{iteration_context}" if iteration_context else "")
                    ),
                )
                _report("DESIGN_DONE", design_output[:200])
            else:
                # Free-form mode: reason first, then design
                reasoning = self.reason(prior_context=iteration_context)
                _report("REASON_DONE", reasoning[:200])

                design_output = self.design(prior_context=iteration_context)
                _report("DESIGN_DONE", design_output[:200])

            # Find the new draft experiment
            drafts = self.executor.ledger.by_status(ExperimentStatus.DRAFT)
            if not drafts:
                _report("NO_DRAFT", "Agent did not create a draft experiment. Stopping.")
                break

            draft = drafts[-1]  # most recent draft
            experiment_ids.append(draft.id)
            _report("DRAFT_CREATED", f"{draft.id}: {draft.hypothesis.statement}")

            # Step 3: Confirm (if interactive mode)
            if self.mode == AgentMode.INTERACTIVE:
                cost_info = ""
                if draft.config.resources and draft.config.resources.estimated_cost_usd:
                    cost_info = f" (est. ${draft.config.resources.estimated_cost_usd})"

                approved = self._on_human_confirm(
                    f"Launch experiment {draft.id}?{cost_info}",
                    f"Hypothesis: {draft.hypothesis.statement}\n"
                    f"Config: {json.dumps(draft.config.script_args, indent=2)}",
                )
                if not approved:
                    _report("SKIPPED", f"Human declined {draft.id}")
                    continue

            # Step 4: Launch
            _report("LAUNCHING", draft.id)
            launch_result = self.executor.execute("launch_experiment", {"exp_id": draft.id})
            launch_data = json.loads(launch_result)

            if "error" in launch_data:
                _report("LAUNCH_FAILED", launch_data["error"])
                continue

            _report("LAUNCHED", f"{draft.id} → {launch_data.get('run_id', '?')}")

            # Step 5: Poll until done
            _report("WAITING", f"Polling every {poll_interval}s")
            while True:
                poll_result = json.loads(
                    self.executor.execute("poll_experiment", {"exp_id": draft.id})
                )
                state = poll_result.get("state", "unknown")
                _report("POLL", f"{draft.id}: {state}")

                if state in ("completed", "failed"):
                    break

                time.sleep(poll_interval)

            if state == "failed":
                _report("FAILED", f"{draft.id}: {poll_result.get('error', 'unknown')}")
                iteration_context = (
                    f"Experiment {draft.id} FAILED: {poll_result.get('error', 'unknown')}. "
                    f"Hypothesis was: {draft.hypothesis.statement}. "
                    f"Consider a different approach or fixing the issue."
                )
                # Mark plan step as done with failure note
                if use_plan and plan_manager and current_step:
                    plan_manager.update_step(
                        current_step.id,
                        status="done",
                        experiment_id=draft.id,
                        notes=f"FAILED: {poll_result.get('error', 'unknown')[:200]}",
                    )
                continue

            # Step 6: Reflect — produces context for next iteration
            _report("REFLECTING", draft.id)
            reflection = self.reflect(draft.id)
            _report("REFLECT_DONE", reflection[:200])

            # Mark plan step as done
            if use_plan and plan_manager and current_step:
                plan_manager.update_step(
                    current_step.id,
                    status="done",
                    experiment_id=draft.id,
                    notes=reflection[:200],
                )

            # Carry reflection forward to next iteration
            iteration_context = (
                f"Previous experiment {draft.id}: {draft.hypothesis.statement}\n"
                f"Reflection:\n{reflection}\n"
            )

        _report("LOOP_DONE", f"Completed {len(experiment_ids)} experiments: {experiment_ids}")
        return experiment_ids


# Need this import here to avoid circular
from ..core.experiment import ExperimentStatus
