"""Tests for agent tool executor."""

import json
import pytest

from daedalus.agent.tools import ToolExecutor, TOOL_SCHEMAS
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus, Reflection
from daedalus.core.hypothesis import Hypothesis, Prediction
from daedalus.core.ledger import Ledger
from daedalus.literature.library import Library
from daedalus.literature.paper import Paper


def _setup_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    (project / "program.md").write_text("# Test\n\nGoals here.\n")
    return project


class TestToolSchemas:
    def test_all_schemas_have_name(self):
        for schema in TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema

    def test_no_duplicate_names(self):
        names = [s["name"] for s in TOOL_SCHEMAS]
        assert len(names) == len(set(names))


class TestToolExecutor:
    def test_get_context(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("get_context", {"mode": "full"}))
        assert "context" in result
        assert "Test" in result["context"]

    def test_create_and_list_experiment(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)

        # Create
        result = json.loads(executor.execute("create_experiment", {
            "hypothesis_statement": "penalty -0.5 reduces IDK rate",
            "rationale": "EV analysis",
            "script": "train.py",
            "script_args": {"idk_knowable_reward": -0.5},
            "predictions": [
                {"metric": "simpleqa.not_attempted", "direction": "decrease"},
            ],
            "tags": ["calibration"],
        }))
        assert "created" in result
        exp_id = result["created"]

        # List
        result = json.loads(executor.execute("list_experiments", {}))
        assert len(result["experiments"]) == 1
        assert result["experiments"][0]["id"] == exp_id

    def test_get_experiment(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)

        executor.execute("create_experiment", {
            "hypothesis_statement": "test",
            "rationale": "test",
            "script": "train.py",
        })

        exps = json.loads(executor.execute("list_experiments", {}))
        exp_id = exps["experiments"][0]["id"]

        result = json.loads(executor.execute("get_experiment", {"exp_id": exp_id}))
        assert "experiment" in result
        assert result["experiment"]["hypothesis"]["statement"] == "test"

    def test_record_results(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)

        executor.execute("create_experiment", {
            "hypothesis_statement": "test",
            "rationale": "test",
            "script": "train.py",
        })
        exps = json.loads(executor.execute("list_experiments", {}))
        exp_id = exps["experiments"][0]["id"]

        result = json.loads(executor.execute("record_results", {
            "exp_id": exp_id,
            "results": {"simpleqa": {"correct": 3.5, "not_attempted": 41.0}},
        }))
        assert result["status"] == "completed"

    def test_add_reflection(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)

        # Create + record results
        executor.execute("create_experiment", {
            "hypothesis_statement": "test",
            "rationale": "test",
            "script": "train.py",
        })
        exps = json.loads(executor.execute("list_experiments", {}))
        exp_id = exps["experiments"][0]["id"]

        executor.execute("record_results", {
            "exp_id": exp_id,
            "results": {"simpleqa": {"correct": 3.5}},
        })

        # Reflect
        result = json.loads(executor.execute("add_reflection", {
            "exp_id": exp_id,
            "hypothesis_confirmed": "confirmed",
            "analysis": "Results matched expectations",
            "next_suggestions": ["Try penalty=-0.3"],
        }))
        assert "reflected" in result

    def test_compare_experiments(self, tmp_path):
        project = _setup_project(tmp_path)
        ledger = Ledger(project / "ledger")

        # Add two experiments with results
        for exp_id, correct in [("exp_a", 3.0), ("exp_b", 3.5)]:
            ledger.append(Experiment(
                id=exp_id,
                hypothesis=Hypothesis(statement="test", rationale="test"),
                config=ExperimentConfig(script="train.py", script_args={"lr": 5e-6 if exp_id == "exp_a" else 1e-5}),
                status=ExperimentStatus.COMPLETED,
                results={"simpleqa": {"correct": correct}},
            ))

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("compare_experiments", {
            "exp_a": "exp_a",
            "exp_b": "exp_b",
        }))

        assert "config_diff" in result
        assert "metric_comparison" in result
        assert len(result["metric_comparison"]) == 1
        assert result["metric_comparison"][0]["delta"] == pytest.approx(0.5)

    def test_list_papers(self, tmp_path):
        project = _setup_project(tmp_path)
        lib = Library(project / "ledger" / "papers.jsonl")
        lib.add(Paper(arxiv_id="1234.5678", title="Test Paper", tags=["grpo"]))

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("list_papers", {}))
        assert len(result["papers"]) == 1
        assert result["papers"][0]["arxiv_id"] == "1234.5678"

    def test_unknown_tool(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("nonexistent_tool", {}))
        assert "error" in result

    def test_suggest_next_with_experiments(self, tmp_path):
        project = _setup_project(tmp_path)
        import yaml
        (project / "daedalus.yaml").write_text(yaml.dump({
            "project_name": "test",
            "scripts": {
                "train": {
                    "path": "train.py",
                    "parameters": {
                        "lr": {"type": "float", "default": 1e-5, "range": [1e-6, 1e-3]},
                    },
                },
            },
        }))

        executor = ToolExecutor(project)
        # Create two experiments with results
        executor.execute("create_experiment", {
            "hypothesis_statement": "baseline",
            "rationale": "test",
            "script": "train.py",
            "script_args": {"lr": 1e-5},
        })
        exps = json.loads(executor.execute("list_experiments", {}))
        exp_id = exps["experiments"][0]["id"]
        executor.execute("record_results", {
            "exp_id": exp_id,
            "results": {"eval": {"accuracy": 0.85}},
        })

        result = json.loads(executor.execute("suggest_next", {
            "target_metric": "eval.accuracy",
            "higher_is_better": True,
            "focus_param": "lr",
        }))
        assert "suggestions" in result
        assert result["experiments_analyzed"] == 1
        assert len(result["suggestions"]) >= 1

    def test_list_scripts(self, tmp_path):
        project = _setup_project(tmp_path)
        import yaml
        (project / "daedalus.yaml").write_text(yaml.dump({
            "project_name": "test",
            "scripts": {
                "train": {
                    "path": "train.py",
                    "description": "Training script",
                    "parameters": {
                        "lr": {"type": "float", "default": 1e-5},
                        "epochs": {"type": "int", "default": 3},
                    },
                },
            },
        }))

        executor = ToolExecutor(project)
        # List all
        result = json.loads(executor.execute("list_scripts", {}))
        assert "scripts" in result
        assert "train" in result["scripts"]

        # Get specific
        result = json.loads(executor.execute("list_scripts", {"name": "train"}))
        assert result["path"] == "train.py"
        assert "lr" in result["parameters"]
        assert result["defaults"]["lr"] == 1e-5

    def test_list_scripts_not_found(self, tmp_path):
        project = _setup_project(tmp_path)
        import yaml
        (project / "daedalus.yaml").write_text(yaml.dump({
            "project_name": "test",
            "scripts": {"train": {"path": "train.py"}},
        }))

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("list_scripts", {"name": "nonexistent"}))
        assert "error" in result

    def test_inspect_dataset(self, tmp_path):
        project = _setup_project(tmp_path)
        # Write a test dataset
        data_file = project / "train.jsonl"
        import json as _json
        records = [
            {"prompt": "What is AI?", "completion": "Artificial Intelligence"},
            {"prompt": "What is ML?", "completion": "Machine Learning"},
            {"prompt": "What is NLP?", "completion": "Natural Language Processing"},
        ]
        data_file.write_text("\n".join(_json.dumps(r) for r in records))

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("inspect_dataset", {
            "path": str(data_file),
        }))
        assert result["row_count"] == 3
        assert result["column_count"] == 2
        assert result["format"] == "jsonl"
        assert len(result["sample_records"]) == 3

    def test_validate_dataset_pass(self, tmp_path):
        project = _setup_project(tmp_path)
        data_file = project / "train.jsonl"
        import json as _json
        records = [
            {"prompt": "q1", "completion": "a1"},
            {"prompt": "q2", "completion": "a2"},
        ]
        data_file.write_text("\n".join(_json.dumps(r) for r in records))

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("validate_dataset", {
            "path": str(data_file),
            "format": "sft",
        }))
        assert result["passed"] is True

    def test_validate_dataset_fail(self, tmp_path):
        project = _setup_project(tmp_path)
        data_file = project / "train.jsonl"
        import json as _json
        records = [
            {"prompt": "q1"},  # missing completion
        ]
        data_file.write_text("\n".join(_json.dumps(r) for r in records))

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("validate_dataset", {
            "path": str(data_file),
            "required_fields": ["prompt", "completion"],
        }))
        assert result["passed"] is False
        assert result["error_count"] == 1

    def test_human_confirm(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("human_confirm", {
            "question": "Launch expensive experiment?",
            "context": "Estimated cost: $50",
        }))
        assert result["status"] == "pending"
