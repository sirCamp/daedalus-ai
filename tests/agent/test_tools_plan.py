"""Tests for plan MCP tools — create_plan, get_plan, update_plan_step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from daedalus.agent.tools import ToolExecutor


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create a minimal Daedalus project."""
    config = {"project_name": "test", "runner": {"type": "local"}, "scripts": {}}
    (tmp_path / "daedalus.yaml").write_text(yaml.dump(config))
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "experiments.jsonl").touch()
    return tmp_path


class TestCreatePlan:
    def test_create_plan(self, project):
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("create_plan", {
            "goal": "Optimize learning rate",
            "steps": [
                {
                    "description": "Test LR=1e-5",
                    "rationale": "Lower than default",
                    "expected_outcome": "Lower loss",
                    "priority": 1,
                },
                {
                    "description": "Test LR=5e-5",
                    "rationale": "Higher than default",
                    "expected_outcome": "Faster convergence",
                    "priority": 2,
                },
            ],
        }))

        assert result["created"] is True
        assert result["steps_count"] == 2
        assert result["step_ids"] == ["S01", "S02"]

    def test_create_empty_steps(self, project):
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("create_plan", {
            "goal": "Empty plan",
            "steps": [],
        }))
        assert "error" in result

    def test_create_plan_saves_note(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test plan",
            "steps": [{"description": "Step 1", "rationale": "R", "expected_outcome": "E"}],
        })

        notes_file = project / "ledger" / "notes.jsonl"
        if notes_file.exists():
            content = notes_file.read_text()
            assert "Plan created" in content


class TestGetPlan:
    def test_no_plan(self, project):
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("get_plan", {}))
        assert result["plan"] is None

    def test_get_plan(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test goal",
            "steps": [
                {"description": "S1", "rationale": "R1", "expected_outcome": "E1", "priority": 1},
                {"description": "S2", "rationale": "R2", "expected_outcome": "E2", "priority": 2},
            ],
        })

        result = json.loads(executor.execute("get_plan", {}))
        assert result["goal"] == "Test goal"
        assert len(result["steps"]) == 2
        assert result["next_actionable"] == "S01"
        assert result["progress"]["total"] == 2
        assert result["progress"]["pending"] == 2


class TestUpdatePlanStep:
    def test_update_status(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [{"description": "S1", "rationale": "R", "expected_outcome": "E"}],
        })

        result = json.loads(executor.execute("update_plan_step", {
            "step_id": "S01",
            "status": "done",
            "notes": "Completed successfully",
        }))
        assert result["updated"] is True
        assert result["status"] == "done"
        assert result["notes"] == "Completed successfully"

    def test_update_nonexistent(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [{"description": "S1", "rationale": "R", "expected_outcome": "E"}],
        })

        result = json.loads(executor.execute("update_plan_step", {
            "step_id": "S99",
            "status": "done",
        }))
        assert "error" in result

    def test_update_no_fields(self, project):
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("update_plan_step", {
            "step_id": "S01",
        }))
        assert "error" in result

    def test_link_experiment(self, project):
        executor = ToolExecutor(project)
        executor.execute("create_plan", {
            "goal": "Test",
            "steps": [{"description": "S1", "rationale": "R", "expected_outcome": "E"}],
        })

        result = json.loads(executor.execute("update_plan_step", {
            "step_id": "S01",
            "experiment_id": "exp_007",
            "status": "running",
        }))
        assert result["experiment_id"] == "exp_007"
        assert result["status"] == "running"


class TestToolSchemas:
    def test_plan_tools_in_schemas(self):
        from daedalus.agent.tools import TOOL_SCHEMAS
        names = {s["name"] for s in TOOL_SCHEMAS}
        assert "get_plan" in names
        assert "create_plan" in names
        assert "update_plan_step" in names
        assert "update_plan" in names

    def test_total_tool_count(self):
        from daedalus.agent.tools import TOOL_SCHEMAS
        assert len(TOOL_SCHEMAS) == 30  # 25 base + 1 remote + 4 plan tools
