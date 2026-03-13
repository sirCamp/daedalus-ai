"""Research plan — structured, persistent, multi-step experiment planning.

Stores a research plan in `ledger/plan.json` (source of truth) and
auto-generates `ledger/plan.md` (human-readable table view) on every save.

The plan guides the autonomous loop: instead of free-form reasoning,
the agent picks the next actionable step from the plan, executes it,
reflects, and updates the plan.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class PlanStep(BaseModel):
    """A single step in the research plan."""

    id: str  # "S01", "S02", etc.
    description: str  # what to test
    rationale: str  # why this matters
    expected_outcome: str  # what we expect to see
    priority: int = Field(ge=1, le=5, default=3)  # 1 = highest
    status: PlanStepStatus = PlanStepStatus.PENDING
    experiment_id: str | None = None  # linked after launch
    depends_on: list[str] = []  # step IDs that must complete first
    tags: list[str] = []
    notes: str = ""  # post-hoc notes from agent
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchPlan(BaseModel):
    """A structured, multi-step research plan."""

    goal: str
    steps: list[PlanStep] = []
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlanManager:
    """Manages the persistent research plan.

    Storage: single JSON file (plan is one object, not a stream).
    Auto-generates plan.md for human reading.
    """

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = Path(ledger_path)
        self._plan_path = self.ledger_path / "plan.json"
        self._markdown_path = self.ledger_path / "plan.md"
        self.ledger_path.mkdir(parents=True, exist_ok=True)

    def load(self) -> ResearchPlan | None:
        """Load the current plan, or None if no plan exists."""
        if not self._plan_path.exists():
            return None
        try:
            data = json.loads(self._plan_path.read_text())
            return ResearchPlan.model_validate(data)
        except Exception as e:
            logger.error("Failed to load plan: %s", e)
            return None

    def save(self, plan: ResearchPlan) -> None:
        """Save the plan (full rewrite) and regenerate Markdown."""
        plan.updated_at = datetime.now(timezone.utc)
        self._plan_path.write_text(plan.model_dump_json(indent=2))
        self._sync_markdown(plan)
        logger.info("Plan saved: %d steps, version %d", len(plan.steps), plan.version)

    def create(self, goal: str, steps: list[dict[str, Any]]) -> ResearchPlan:
        """Create a new plan from scratch, replacing any existing one."""
        plan_steps = []
        for i, s in enumerate(steps, 1):
            step = PlanStep(
                id=f"S{i:02d}",
                description=s.get("description", ""),
                rationale=s.get("rationale", ""),
                expected_outcome=s.get("expected_outcome", ""),
                priority=s.get("priority", 3),
                depends_on=s.get("depends_on", []),
                tags=s.get("tags", []),
            )
            plan_steps.append(step)

        plan = ResearchPlan(goal=goal, steps=plan_steps)
        self.save(plan)
        return plan

    def add_step(self, step_data: dict[str, Any]) -> PlanStep:
        """Add a new step to the existing plan."""
        plan = self.load()
        if plan is None:
            raise ValueError("No plan exists. Use create() first.")

        # Auto-generate ID
        existing_ids = {s.id for s in plan.steps}
        num = len(plan.steps) + 1
        while f"S{num:02d}" in existing_ids:
            num += 1

        step = PlanStep(
            id=f"S{num:02d}",
            description=step_data.get("description", ""),
            rationale=step_data.get("rationale", ""),
            expected_outcome=step_data.get("expected_outcome", ""),
            priority=step_data.get("priority", 3),
            depends_on=step_data.get("depends_on", []),
            tags=step_data.get("tags", []),
        )
        plan.steps.append(step)
        plan.version += 1
        self.save(plan)
        return step

    def update_step(self, step_id: str, **kwargs: Any) -> PlanStep:
        """Update fields on a plan step."""
        plan = self.load()
        if plan is None:
            raise ValueError("No plan exists.")

        step = next((s for s in plan.steps if s.id == step_id), None)
        if step is None:
            raise ValueError(f"Step {step_id} not found in plan.")

        for key, value in kwargs.items():
            if key == "status":
                value = PlanStepStatus(value)
            if hasattr(step, key):
                setattr(step, key, value)

        step.updated_at = datetime.now(timezone.utc)
        self.save(plan)
        return step

    def next_actionable(self) -> PlanStep | None:
        """Return the highest-priority pending step whose dependencies are met.

        Dependencies are met when all depends_on steps are done or skipped.
        If a cycle or deadlock is detected, picks the highest-priority pending
        step ignoring dependencies.
        """
        plan = self.load()
        if plan is None:
            return None

        done_or_skipped = {
            s.id for s in plan.steps
            if s.status in (PlanStepStatus.DONE, PlanStepStatus.SKIPPED)
        }
        pending = [s for s in plan.steps if s.status == PlanStepStatus.PENDING]

        if not pending:
            return None

        # Find steps with all dependencies met
        actionable = [
            s for s in pending
            if all(dep in done_or_skipped for dep in s.depends_on)
        ]

        if actionable:
            # Sort by priority (1 = highest), then by creation order
            actionable.sort(key=lambda s: (s.priority, s.id))
            return actionable[0]

        # Deadlock: no actionable steps but pending steps exist
        logger.warning(
            "Plan deadlock: %d pending steps but none have met dependencies. "
            "Picking highest priority ignoring dependencies.",
            len(pending),
        )
        pending.sort(key=lambda s: (s.priority, s.id))
        return pending[0]

    def format_for_context(self) -> str:
        """Format the plan for agent context injection."""
        plan = self.load()
        if plan is None:
            return "(No research plan. Use create_plan to define one.)"

        lines = [f"**Research Plan**: {plan.goal} (v{plan.version})\n"]

        status_icons = {
            PlanStepStatus.PENDING: "[ ]",
            PlanStepStatus.RUNNING: "[~]",
            PlanStepStatus.DONE: "[x]",
            PlanStepStatus.SKIPPED: "[-]",
            PlanStepStatus.BLOCKED: "[!]",
        }

        for step in plan.steps:
            icon = status_icons.get(step.status, "[ ]")
            exp_ref = f" → {step.experiment_id}" if step.experiment_id else ""
            deps = f" (after {', '.join(step.depends_on)})" if step.depends_on else ""
            lines.append(f"{icon} **{step.id}** [P{step.priority}]{deps}: {step.description}{exp_ref}")
            if step.notes:
                lines.append(f"    ↳ {step.notes}")

        # Summary
        total = len(plan.steps)
        done = sum(1 for s in plan.steps if s.status == PlanStepStatus.DONE)
        skipped = sum(1 for s in plan.steps if s.status == PlanStepStatus.SKIPPED)
        pending = sum(1 for s in plan.steps if s.status == PlanStepStatus.PENDING)
        lines.append(f"\nProgress: {done}/{total} done, {skipped} skipped, {pending} pending")

        return "\n".join(lines)

    def _sync_markdown(self, plan: ResearchPlan) -> None:
        """Regenerate plan.md from the plan object."""
        lines = [
            "# Research Plan\n",
            f"**Goal**: {plan.goal}",
            f"**Version**: {plan.version}",
            f"**Updated**: {plan.updated_at.strftime('%Y-%m-%d %H:%M UTC')}\n",
            "| ID | P | Status | Description | Experiment | Dependencies |",
            "|----|---|--------|-------------|------------|--------------|",
        ]

        for step in plan.steps:
            exp = step.experiment_id or ""
            deps = ", ".join(step.depends_on) or ""
            lines.append(
                f"| {step.id} | {step.priority} | {step.status.value} | "
                f"{step.description} | {exp} | {deps} |"
            )

        lines.append("")

        # Detailed view
        for step in plan.steps:
            lines.append(f"## {step.id}: {step.description}\n")
            lines.append(f"- **Priority**: {step.priority}")
            lines.append(f"- **Status**: {step.status.value}")
            lines.append(f"- **Rationale**: {step.rationale}")
            lines.append(f"- **Expected outcome**: {step.expected_outcome}")
            if step.experiment_id:
                lines.append(f"- **Experiment**: {step.experiment_id}")
            if step.depends_on:
                lines.append(f"- **Depends on**: {', '.join(step.depends_on)}")
            if step.notes:
                lines.append(f"- **Notes**: {step.notes}")
            lines.append("")

        self._markdown_path.write_text("\n".join(lines))
