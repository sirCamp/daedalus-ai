"""Tests for the Daedalus CLI."""

import json

import pytest
from click.testing import CliRunner

from daedalus.cli import cli
from daedalus.core.ledger import Ledger
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus
from daedalus.core.hypothesis import Hypothesis


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def project_dir(tmp_path):
    """Create a project and return its path."""
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    (project / "program.md").write_text("# Test\n\nGoals here.\n")
    (project / "daedalus.yaml").write_text("project_name: test\n")
    return project


class TestInit:
    def test_init_creates_project(self, runner, tmp_path):
        result = runner.invoke(cli, ["init", "my_project", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "initialized" in result.output

        project = tmp_path / "my_project"
        assert (project / "program.md").exists()
        assert (project / "daedalus.yaml").exists()
        assert (project / "ledger" / "experiments.jsonl").exists()
        assert (project / "runner_config.yaml").exists()
        assert (project / "CLAUDE.md").exists()

    def test_init_claude_md_content(self, runner, tmp_path):
        runner.invoke(cli, ["init", "my_project", "--path", str(tmp_path)])
        content = (tmp_path / "my_project" / "CLAUDE.md").read_text()
        assert "daedalus_*" in content
        assert "MCP tools" in content
        assert "ledger/" in content

    def test_init_existing_dir_creates_missing_files(self, runner, tmp_path):
        """Init on existing dir creates only missing files, never overwrites."""
        project = tmp_path / "existing"
        project.mkdir()
        # Pre-create program.md with custom content
        (project / "program.md").write_text("my custom program")

        result = runner.invoke(cli, ["init", "existing", "--path", str(tmp_path)])
        assert result.exit_code == 0

        # Custom file preserved
        assert (project / "program.md").read_text() == "my custom program"
        # Missing files created
        assert (project / "daedalus.yaml").exists()
        assert (project / "CLAUDE.md").exists()
        assert (project / "ledger" / "experiments.jsonl").exists()
        # Skipped message shown
        assert "Skipped" in result.output or "existing" in result.output

    def test_init_fully_initialized_project(self, runner, tmp_path):
        """Running init twice reports already initialized."""
        runner.invoke(cli, ["init", "myproj", "--path", str(tmp_path)])
        result = runner.invoke(cli, ["init", "myproj", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "already fully initialized" in result.output


class TestStatus:
    def test_empty_project(self, runner, project_dir):
        result = runner.invoke(cli, ["-p", str(project_dir), "status"])
        assert result.exit_code == 0
        assert "No experiments" in result.output

    def test_with_experiments(self, runner, project_dir):
        ledger = Ledger(project_dir / "ledger")
        ledger.append(Experiment(
            id="exp_001",
            hypothesis=Hypothesis(statement="test hyp", rationale="because"),
            config=ExperimentConfig(script="train.py"),
        ))

        result = runner.invoke(cli, ["-p", str(project_dir), "status"])
        assert result.exit_code == 0
        assert "draft" in result.output


class TestHypothesis:
    def test_create_hypothesis(self, runner, project_dir):
        result = runner.invoke(cli, [
            "-p", str(project_dir),
            "hypothesis", "penalty reduces IDK rate",
            "--rationale", "EV analysis shows model avoids risk",
            "--predict", "simpleqa.not_attempted:decrease:30.0",
            "--paper", "2601.20126",
            "--script", "train.py",
            "--arg", "idk_knowable_reward=-0.5",
        ])
        assert result.exit_code == 0
        assert "created" in result.output

        # Verify in ledger
        ledger = Ledger(project_dir / "ledger")
        assert len(ledger) == 1
        exp = ledger.all()[0]
        assert exp.hypothesis.statement == "penalty reduces IDK rate"
        assert len(exp.hypothesis.predictions) == 1
        assert exp.config.script_args["idk_knowable_reward"] == -0.5


class TestRecord:
    def test_record_results(self, runner, project_dir):
        # Create experiment first
        ledger = Ledger(project_dir / "ledger")
        ledger.append(Experiment(
            id="exp_001",
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="train.py"),
        ))

        result = runner.invoke(cli, [
            "-p", str(project_dir),
            "record", "exp_001",
            "-r", "simpleqa.correct=3.5",
            "-r", "simpleqa.not_attempted=41.0",
            "-r", "popqa.em=10.2",
        ])
        assert result.exit_code == 0

        exp = ledger.get("exp_001")
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.results["simpleqa"]["correct"] == 3.5
        assert exp.results["popqa"]["em"] == 10.2


class TestReflect:
    def test_add_reflection(self, runner, project_dir):
        ledger = Ledger(project_dir / "ledger")
        exp = Experiment(
            id="exp_001",
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="train.py"),
            status=ExperimentStatus.COMPLETED,
            results={"simpleqa": {"not_attempted": 41.0}},
        )
        ledger.append(exp)

        result = runner.invoke(cli, [
            "-p", str(project_dir),
            "reflect", "exp_001",
            "--analysis", "Not_attempted reduced as expected",
            "--confirmed", "confirmed",
            "--surprise", "PopQA also improved",
            "--next", "Try penalty=-0.3",
            "--next", "Run FiSCoRe",
        ])
        assert result.exit_code == 0

        exp = ledger.get("exp_001")
        assert exp.status == ExperimentStatus.ANALYZED
        assert exp.reflection.hypothesis_confirmed == "confirmed"
        assert len(exp.reflection.next_suggestions) == 2


class TestLedgerCommand:
    def test_show_ledger(self, runner, project_dir):
        ledger = Ledger(project_dir / "ledger")
        ledger.append(Experiment(
            id="exp_001",
            hypothesis=Hypothesis(statement="first experiment", rationale="baseline"),
            config=ExperimentConfig(script="train.py"),
        ))

        result = runner.invoke(cli, ["-p", str(project_dir), "ledger"])
        assert result.exit_code == 0
        assert "exp_001" in result.output


class TestContext:
    def test_full_context(self, runner, project_dir):
        result = runner.invoke(cli, ["-p", str(project_dir), "context"])
        assert result.exit_code == 0
        # Should include program.md content
        assert "Test" in result.output or "Goals" in result.output


class TestDiff:
    def test_diff_experiments(self, runner, project_dir):
        ledger = Ledger(project_dir / "ledger")
        ledger.append(Experiment(
            id="exp_001",
            hypothesis=Hypothesis(statement="baseline", rationale="test"),
            config=ExperimentConfig(script="train.py", script_args={"lr": 1e-4}),
            status=ExperimentStatus.ANALYZED,
            results={"simpleqa": {"correct": 3.0}},
        ))
        ledger.append(Experiment(
            id="exp_002",
            hypothesis=Hypothesis(statement="lr change", rationale="test"),
            config=ExperimentConfig(script="train.py", script_args={"lr": 5e-6}),
            status=ExperimentStatus.ANALYZED,
            results={"simpleqa": {"correct": 3.5}},
        ))

        result = runner.invoke(cli, ["-p", str(project_dir), "diff", "exp_001", "exp_002"])
        assert result.exit_code == 0
        assert "lr" in result.output
