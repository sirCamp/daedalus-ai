"""Tests for autonomous plan execution — guardrails, launch gate, auto-bookkeeping."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from daedalus.agent.plan import PlanManager, PlanStepStatus
from daedalus.agent.tools import ToolExecutor
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.ledger import Ledger
from daedalus.runners.local import LocalRunner


def _step(id: str, priority: int = 3, **kwargs) -> dict:
    return {
        "description": f"Test step {id}",
        "rationale": f"Rationale for {id}",
        "expected_outcome": f"Outcome for {id}",
        "priority": priority,
        **kwargs,
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    config = {
        "project_name": "test",
        "runner": {"type": "local"},
        "scripts": {"train": {"path": "train.py", "parameters": {}}},
    }
    (tmp_path / "daedalus.yaml").write_text(yaml.dump(config))
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "experiments.jsonl").touch()
    (tmp_path / "ledger" / "papers.jsonl").touch()
    (tmp_path / "program.md").write_text("# Test\n")
    return tmp_path


# ---------------------------------------------------------------------------
# PlanManager guardrail tests
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_check_guardrails_ok(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")], autonomy="autonomous")
        ok, reason = mgr.check_guardrails()
        assert ok is True
        assert reason is None

    def test_check_guardrails_paused(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        mgr.update_plan_settings(paused=True, pause_reason="Manual pause")
        ok, reason = mgr.check_guardrails()
        assert ok is False
        assert "Manual pause" in reason

    def test_check_guardrails_max_experiments(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")], max_experiments=3)
        plan = mgr.load()
        plan.experiments_run = 3
        mgr.save(plan)

        ok, reason = mgr.check_guardrails()
        assert ok is False
        assert "max experiments" in reason

    def test_record_success_resets_failures(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        plan = mgr.load()
        plan.consecutive_failures = 1
        mgr.save(plan)

        mgr.record_success()
        plan = mgr.load()
        assert plan.consecutive_failures == 0

    def test_record_failure_increments(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")], max_consecutive_failures=3)
        paused, _ = mgr.record_failure()
        assert paused is False
        plan = mgr.load()
        assert plan.consecutive_failures == 1

    def test_record_failure_triggers_pause(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")], max_consecutive_failures=2)
        mgr.record_failure()  # 1
        paused, reason = mgr.record_failure()  # 2 = limit
        assert paused is True
        assert "consecutive failures" in reason
        plan = mgr.load()
        assert plan.paused is True

    def test_increment_experiments_run(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        mgr.increment_experiments_run()
        plan = mgr.load()
        assert plan.experiments_run == 1


class TestFindStepByExperiment:
    def test_found(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01"), _step("S02")])
        mgr.update_step("S01", experiment_id="exp_007")
        step = mgr.find_step_by_experiment("exp_007")
        assert step is not None
        assert step.id == "S01"

    def test_not_found(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        assert mgr.find_step_by_experiment("exp_999") is None

    def test_no_plan(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        assert mgr.find_step_by_experiment("exp_001") is None


class TestUpdatePlanSettings:
    def test_update_autonomy(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        plan = mgr.update_plan_settings(autonomy="autonomous")
        assert plan.autonomy == "autonomous"
        assert plan.version == 2

    def test_pause_and_resume(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        mgr.update_plan_settings(paused=True, pause_reason="Testing")
        plan = mgr.load()
        assert plan.paused is True

        mgr.update_plan_settings(paused=False)
        plan = mgr.load()
        assert plan.paused is False

    def test_no_plan(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        with pytest.raises(ValueError, match="No plan exists"):
            mgr.update_plan_settings(autonomy="autonomous")


class TestRequiresConfirmation:
    def test_default_false(self):
        from daedalus.agent.plan import PlanStep
        step = PlanStep(id="S01", description="T", rationale="R", expected_outcome="E")
        assert step.requires_confirmation is False

    def test_create_with_confirmation(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [
            _step("S01", requires_confirmation=True),
            _step("S02"),
        ])
        plan = mgr.load()
        assert plan.steps[0].requires_confirmation is True
        assert plan.steps[1].requires_confirmation is False


class TestFailedStatus:
    def test_failed_status_exists(self):
        assert PlanStepStatus.FAILED.value == "failed"

    def test_failed_unblocks_dependents(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [
            _step("S01", priority=2),
            _step("S02", priority=1, depends_on=["S01"]),
        ])
        mgr.update_step("S01", status="failed")
        step = mgr.next_actionable()
        assert step.id == "S02"


class TestFormatContextAutonomy:
    def test_shows_mode(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")], autonomy="autonomous")
        ctx = mgr.format_for_context()
        assert "AUTONOMOUS" in ctx

    def test_shows_paused(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")])
        mgr.update_plan_settings(paused=True, pause_reason="OOM errors")
        ctx = mgr.format_for_context()
        assert "PAUSED" in ctx
        assert "OOM errors" in ctx

    def test_shows_experiment_counter(self, tmp_path):
        mgr = PlanManager(tmp_path / "ledger")
        mgr.create("Test", [_step("S01")], max_experiments=10)
        mgr.increment_experiments_run()
        ctx = mgr.format_for_context()
        assert "1/10" in ctx


# ---------------------------------------------------------------------------
# Tool-level tests — launch gate, auto-bookkeeping
# ---------------------------------------------------------------------------


class TestLaunchGate:
    def test_supervised_blocks_launch(self, project):
        """In supervised mode, launch returns needs_confirmation."""
        executor = ToolExecutor(project)
        # Create plan in supervised mode (explicit)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
            "autonomy": "supervised",
        })
        # Create a draft experiment
        executor.execute("create_experiment", {
            "hypothesis_statement": "Test",
            "rationale": "Test",
            "script": "train.py",
        })
        ledger = Ledger(project / "ledger")
        exp = ledger.all()[-1]

        result = json.loads(executor.execute("launch_experiment", {
            "exp_id": exp.id,
            "plan_step_id": "S01",
        }))
        assert result.get("needs_confirmation") is True
        assert result["autonomy"] == "supervised"

    def test_autonomous_allows_launch(self, project):
        """In autonomous mode, launch proceeds directly."""
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
            "autonomy": "autonomous",
        })

        # Create experiment with a real script
        work_dir = project / "runs" / "test_exp"
        work_dir.mkdir(parents=True)
        (work_dir / "train.py").write_text("import json; json.dump({'eval': {'acc': 0.9}}, open('results.json', 'w'))")

        executor.execute("create_experiment", {
            "hypothesis_statement": "Test",
            "rationale": "Test",
            "script": "train.py",
        })
        ledger = Ledger(project / "ledger")
        exp = ledger.all()[-1]

        result = json.loads(executor.execute("launch_experiment", {
            "exp_id": exp.id,
            "plan_step_id": "S01",
        }))
        assert "launched" in result
        assert result.get("plan_step_linked") == "S01"

    def test_paused_plan_blocks_launch(self, project):
        """Paused plan blocks all launches."""
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
            "autonomy": "autonomous",
        })
        executor.execute("update_plan", {"paused": True, "pause_reason": "Manual pause"})

        executor.execute("create_experiment", {
            "hypothesis_statement": "Test",
            "rationale": "Test",
            "script": "train.py",
        })
        ledger = Ledger(project / "ledger")
        exp = ledger.all()[-1]

        result = json.loads(executor.execute("launch_experiment", {
            "exp_id": exp.id,
        }))
        assert "error" in result
        assert result.get("plan_paused") is True

    def test_requires_confirmation_step_blocks(self, project):
        """Step with requires_confirmation blocks even in autonomous mode."""
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01", requires_confirmation=True)],
            "autonomy": "autonomous",
        })

        executor.execute("create_experiment", {
            "hypothesis_statement": "Test",
            "rationale": "Test",
            "script": "train.py",
        })
        ledger = Ledger(project / "ledger")
        exp = ledger.all()[-1]

        result = json.loads(executor.execute("launch_experiment", {
            "exp_id": exp.id,
            "plan_step_id": "S01",
        }))
        assert result.get("needs_confirmation") is True

    def test_no_plan_allows_launch(self, project):
        """Without a plan, launch works normally (backward compat)."""
        executor = ToolExecutor(project)
        executor.execute("create_experiment", {
            "hypothesis_statement": "Test",
            "rationale": "Test",
            "script": "train.py",
        })
        ledger = Ledger(project / "ledger")
        exp = ledger.all()[-1]

        result = json.loads(executor.execute("launch_experiment", {
            "exp_id": exp.id,
        }))
        assert "launched" in result

    def test_max_experiments_blocks(self, project):
        """Max experiments guardrail blocks launch."""
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
            "autonomy": "autonomous",
            "max_experiments": 1,
        })
        # Set counter to max
        mgr = PlanManager(project / "ledger")
        plan = mgr.load()
        plan.experiments_run = 1
        mgr.save(plan)

        executor.execute("create_experiment", {
            "hypothesis_statement": "Test",
            "rationale": "Test",
            "script": "train.py",
        })
        ledger = Ledger(project / "ledger")
        exp = ledger.all()[-1]

        result = json.loads(executor.execute("launch_experiment", {
            "exp_id": exp.id,
        }))
        assert "error" in result
        assert result.get("plan_paused") is True


class TestAutoBookkeeping:
    def _launch_and_complete(self, project: Path) -> str:
        """Helper: create, launch, and wait for experiment completion."""
        executor = ToolExecutor(project)

        # Create plan
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
            "autonomy": "autonomous",
        })

        # Create experiment
        executor.execute("create_experiment", {
            "hypothesis_statement": "Test hypothesis",
            "rationale": "Test rationale",
            "script": "run.py",
        })
        ledger = Ledger(project / "ledger")
        exp = ledger.all()[-1]

        # Write script
        work_dir = project / "runs" / exp.id
        work_dir.mkdir(parents=True)
        (work_dir / "run.py").write_text(
            "import json; json.dump({'eval': {'acc': 0.9}}, open('results.json', 'w'))"
        )

        # Launch with plan step link
        result = json.loads(executor.execute("launch_experiment", {
            "exp_id": exp.id,
            "plan_step_id": "S01",
        }))
        assert "launched" in result

        # Wait for completion
        import time
        for _ in range(40):
            poll = json.loads(executor.execute("poll_experiment", {"exp_id": exp.id}))
            if poll["state"] != "running":
                break
            time.sleep(0.25)

        return exp.id

    def test_poll_completed_marks_step_done(self, project):
        exp_id = self._launch_and_complete(project)

        mgr = PlanManager(project / "ledger")
        step = mgr.find_step_by_experiment(exp_id)
        assert step is not None
        assert step.status == PlanStepStatus.DONE

    def test_poll_completed_resets_failures(self, project):
        exp_id = self._launch_and_complete(project)

        mgr = PlanManager(project / "ledger")
        plan = mgr.load()
        assert plan.consecutive_failures == 0

    def test_reflection_copies_to_step_notes(self, project):
        exp_id = self._launch_and_complete(project)

        executor = ToolExecutor(project)
        executor.execute("add_reflection", {
            "exp_id": exp_id,
            "hypothesis_confirmed": "confirmed",
            "analysis": "The accuracy improved as expected.",
        })

        mgr = PlanManager(project / "ledger")
        step = mgr.find_step_by_experiment(exp_id)
        assert step is not None
        assert "confirmed" in step.notes
        assert "accuracy improved" in step.notes


class TestUpdatePlanTool:
    def test_update_autonomy(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
        })
        result = json.loads(executor.execute("update_plan", {
            "autonomy": "autonomous",
        }))
        assert result["updated"] is True
        assert result["autonomy"] == "autonomous"

    def test_pause_plan(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
        })
        result = json.loads(executor.execute("update_plan", {
            "paused": True,
            "pause_reason": "Need to check results",
        }))
        assert result["paused"] is True
        assert result["pause_reason"] == "Need to check results"

    def test_resume_plan_clears_failures(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
        })
        # Simulate failures
        mgr = PlanManager(project / "ledger")
        plan = mgr.load()
        plan.consecutive_failures = 2
        plan.paused = True
        plan.pause_reason = "Too many failures"
        mgr.save(plan)

        result = json.loads(executor.execute("update_plan", {"paused": False}))
        assert result["paused"] is False
        assert result["consecutive_failures"] == 0

    def test_no_plan_error(self, project):
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("update_plan", {
            "autonomy": "autonomous",
        }))
        assert "error" in result

    def test_no_fields_error(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
        })
        result = json.loads(executor.execute("update_plan", {}))
        assert "error" in result


class TestCreatePlanAutonomy:
    def test_create_autonomous(self, project):
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("create_plan", {
            "goal": "Overnight run",
            "steps": [_step("S01")],
            "autonomy": "autonomous",
            "max_experiments": 5,
            "max_consecutive_failures": 3,
        }))
        assert result["autonomy"] == "autonomous"
        assert result["max_experiments"] == 5
        assert result["max_consecutive_failures"] == 3

    def test_get_plan_shows_autonomy(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
            "autonomy": "autonomous",
            "max_experiments": 10,
        })
        result = json.loads(executor.execute("get_plan", {}))
        assert result["autonomy"] == "autonomous"
        assert result["max_experiments"] == 10
        assert result["experiments_run"] == 0
        assert result["paused"] is False


class TestUpdatePlanStepConfirmation:
    def test_set_requires_confirmation(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [_step("S01")],
        })
        result = json.loads(executor.execute("update_plan_step", {
            "step_id": "S01",
            "requires_confirmation": True,
        }))
        assert result["updated"] is True
        assert result["requires_confirmation"] is True


class TestBackwardCompat:
    def test_old_plan_json_loads(self, tmp_path):
        """Old plan.json without new fields should load fine."""
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        old_plan = {
            "goal": "Old plan",
            "steps": [
                {
                    "id": "S01",
                    "description": "Old step",
                    "rationale": "R",
                    "expected_outcome": "E",
                    "priority": 1,
                    "status": "pending",
                    "experiment_id": None,
                    "depends_on": [],
                    "tags": [],
                    "notes": "",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "updated_at": "2025-01-01T00:00:00+00:00",
                }
            ],
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        }
        (ledger / "plan.json").write_text(json.dumps(old_plan))

        mgr = PlanManager(ledger)
        plan = mgr.load()
        assert plan is not None
        assert plan.autonomy == "autonomous"  # Pydantic default is now autonomous
        assert plan.max_experiments is None
        assert plan.paused is False
        assert plan.steps[0].requires_confirmation is False
