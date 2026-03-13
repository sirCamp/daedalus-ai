"""Tests for the research plan — PlanStep, ResearchPlan, PlanManager."""

from __future__ import annotations

import json

import pytest

from daedalus.agent.plan import (
    PlanManager,
    PlanStep,
    PlanStepStatus,
    ResearchPlan,
)


def _step(id: str, priority: int = 3, status: str = "pending", depends_on: list[str] | None = None) -> dict:
    return {
        "description": f"Test step {id}",
        "rationale": f"Rationale for {id}",
        "expected_outcome": f"Outcome for {id}",
        "priority": priority,
        "depends_on": depends_on or [],
    }


class TestPlanStep:
    def test_defaults(self):
        step = PlanStep(id="S01", description="Test", rationale="Why", expected_outcome="What")
        assert step.status == PlanStepStatus.PENDING
        assert step.priority == 3
        assert step.experiment_id is None
        assert step.depends_on == []
        assert step.notes == ""

    def test_priority_bounds(self):
        step = PlanStep(id="S01", description="T", rationale="R", expected_outcome="E", priority=1)
        assert step.priority == 1

        with pytest.raises(Exception):
            PlanStep(id="S01", description="T", rationale="R", expected_outcome="E", priority=0)

        with pytest.raises(Exception):
            PlanStep(id="S01", description="T", rationale="R", expected_outcome="E", priority=6)


class TestPlanManager:
    def test_load_empty(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        assert mgr.load() is None

    def test_create_and_load(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        plan = mgr.create("Test research", [_step("S01"), _step("S02")])

        assert plan.goal == "Test research"
        assert len(plan.steps) == 2
        assert plan.steps[0].id == "S01"
        assert plan.steps[1].id == "S02"

        # Reload from disk
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.goal == "Test research"
        assert len(loaded.steps) == 2

    def test_create_generates_markdown(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])

        md_path = tmp_path / "ledger" / "plan.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Research Plan" in content
        assert "Test step S01" in content

    def test_create_replaces_existing(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Plan 1", [_step("S01")])
        mgr.create("Plan 2", [_step("S01"), _step("S02"), _step("S03")])

        plan = mgr.load()
        assert plan.goal == "Plan 2"
        assert len(plan.steps) == 3

    def test_update_step_status(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01"), _step("S02")])

        step = mgr.update_step("S01", status="done", notes="Completed successfully")
        assert step.status == PlanStepStatus.DONE
        assert step.notes == "Completed successfully"

        # Verify persisted
        plan = mgr.load()
        assert plan.steps[0].status == PlanStepStatus.DONE

    def test_update_step_priority(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01", priority=3)])

        step = mgr.update_step("S01", priority=1)
        assert step.priority == 1

    def test_update_step_experiment_id(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])

        step = mgr.update_step("S01", experiment_id="exp_001")
        assert step.experiment_id == "exp_001"

    def test_update_nonexistent_step(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])

        with pytest.raises(ValueError, match="S99 not found"):
            mgr.update_step("S99", status="done")

    def test_update_no_plan(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        with pytest.raises(ValueError, match="No plan exists"):
            mgr.update_step("S01", status="done")

    def test_add_step(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])

        new_step = mgr.add_step({
            "description": "New step",
            "rationale": "New rationale",
            "expected_outcome": "New outcome",
            "priority": 2,
        })

        assert new_step.id == "S02"
        plan = mgr.load()
        assert len(plan.steps) == 2
        assert plan.version == 2

    def test_add_step_no_plan(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        with pytest.raises(ValueError, match="No plan exists"):
            mgr.add_step({"description": "X", "rationale": "Y", "expected_outcome": "Z"})


class TestNextActionable:
    def test_picks_highest_priority(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [
            _step("S01", priority=3),
            _step("S02", priority=1),
            _step("S03", priority=2),
        ])

        step = mgr.next_actionable()
        assert step.id == "S02"  # P1 is highest

    def test_respects_dependencies(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [
            _step("S01", priority=2),
            _step("S02", priority=1, depends_on=["S01"]),
        ])

        # S02 has higher priority but depends on S01
        step = mgr.next_actionable()
        assert step.id == "S01"

        # After S01 is done, S02 becomes actionable
        mgr.update_step("S01", status="done")
        step = mgr.next_actionable()
        assert step.id == "S02"

    def test_skipped_deps_unblock(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [
            _step("S01", priority=2),
            _step("S02", priority=1, depends_on=["S01"]),
        ])

        # Skip S01 — should unblock S02
        mgr.update_step("S01", status="skipped")
        step = mgr.next_actionable()
        assert step.id == "S02"

    def test_all_done_returns_none(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        mgr.update_step("S01", status="done")

        assert mgr.next_actionable() is None

    def test_no_plan_returns_none(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        assert mgr.next_actionable() is None

    def test_deadlock_fallback(self, tmp_path):
        """Circular dependencies should not hang — picks highest priority."""
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [
            _step("S01", priority=2, depends_on=["S02"]),
            _step("S02", priority=1, depends_on=["S01"]),
        ])

        # Both depend on each other — deadlock
        step = mgr.next_actionable()
        assert step is not None  # Should not hang
        assert step.id == "S02"  # P1 wins in deadlock


class TestFormatForContext:
    def test_no_plan(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        ctx = mgr.format_for_context()
        assert "No research plan" in ctx

    def test_with_plan(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test goal", [
            _step("S01", priority=1),
            _step("S02", priority=2),
        ])
        mgr.update_step("S01", status="done", experiment_id="exp_001")

        ctx = mgr.format_for_context()
        assert "Test goal" in ctx
        assert "[x]" in ctx  # done
        assert "[ ]" in ctx  # pending
        assert "exp_001" in ctx
        assert "1/2 done" in ctx

    def test_shows_notes(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        mgr.update_step("S01", notes="Important finding")

        ctx = mgr.format_for_context()
        assert "Important finding" in ctx
