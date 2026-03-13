"""Tests for run, poll, logs, cancel CLI commands."""

import json
import sys
import time

import pytest
import yaml
from click.testing import CliRunner

from daedalus.cli import cli
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.ledger import Ledger


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def project_dir(tmp_path):
    """Create a project configured with local runner."""
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    (project / "program.md").write_text("# Test\n")
    (project / "daedalus.yaml").write_text(yaml.dump({
        "project_name": "test",
        "runner": {"type": "local"},
    }))
    (project / "runner_config.yaml").write_text(yaml.dump({
        "local": {"python": sys.executable},
    }))
    return project


def _add_draft_experiment(project_dir, exp_id="exp_test", script_content=None):
    """Add a draft experiment with a dummy training script."""
    # Write training script
    script_path = project_dir / "train_dummy.py"
    if script_content is None:
        script_content = (
            'import json, pathlib\n'
            'pathlib.Path("results.json").write_text('
            'json.dumps({"eval": {"metric": 42.0}}))\n'
        )
    script_path.write_text(script_content)

    ledger = Ledger(project_dir / "ledger")
    exp = Experiment(
        id=exp_id,
        hypothesis=Hypothesis(statement="test hypothesis", rationale="testing"),
        config=ExperimentConfig(
            script=str(script_path),
            script_args={},
        ),
    )
    ledger.append(exp)
    return exp


class TestRun:
    def test_run_and_poll(self, runner, project_dir):
        """Full lifecycle: run → poll → completed."""
        _add_draft_experiment(project_dir)

        # Launch with --confirm to skip prompt
        result = runner.invoke(cli, [
            "-p", str(project_dir), "run", "exp_test", "--confirm",
        ])
        assert result.exit_code == 0
        assert "Launched" in result.output

        # Wait for script to complete and poll until done
        for _ in range(20):
            time.sleep(0.5)
            result = runner.invoke(cli, ["-p", str(project_dir), "poll", "exp_test"])
            if "completed" in result.output:
                break

        assert result.exit_code == 0
        assert "completed" in result.output

        # Verify results in ledger
        ledger = Ledger(project_dir / "ledger")
        exp = ledger.get("exp_test")
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.results is not None
        assert exp.results["eval"]["metric"] == 42.0

    def test_run_nonexistent(self, runner, project_dir):
        result = runner.invoke(cli, [
            "-p", str(project_dir), "run", "nope", "--confirm",
        ])
        assert result.exit_code != 0

    def test_run_already_completed(self, runner, project_dir):
        ledger = Ledger(project_dir / "ledger")
        ledger.append(Experiment(
            id="done",
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="train.py"),
            status=ExperimentStatus.COMPLETED,
        ))

        result = runner.invoke(cli, [
            "-p", str(project_dir), "run", "done", "--confirm",
        ])
        assert result.exit_code != 0
        assert "cannot run" in result.output.lower()


class TestLogs:
    def test_logs_after_run(self, runner, project_dir):
        _add_draft_experiment(
            project_dir,
            script_content='print("hello from experiment")\n',
        )

        runner.invoke(cli, ["-p", str(project_dir), "run", "exp_test", "--confirm"])

        # Wait for script to finish
        for _ in range(20):
            time.sleep(0.5)
            poll_result = runner.invoke(cli, ["-p", str(project_dir), "poll", "exp_test"])
            if "running" not in poll_result.output:
                break

        result = runner.invoke(cli, ["-p", str(project_dir), "logs", "exp_test"])
        assert result.exit_code == 0
        assert "hello from experiment" in result.output

    def test_logs_no_run_id(self, runner, project_dir):
        _add_draft_experiment(project_dir)
        result = runner.invoke(cli, ["-p", str(project_dir), "logs", "exp_test"])
        assert result.exit_code == 0
        assert "No run_id" in result.output


class TestCancel:
    def test_cancel_running(self, runner, project_dir):
        _add_draft_experiment(
            project_dir,
            script_content='import time; time.sleep(60)\n',
        )

        # Launch
        runner.invoke(cli, ["-p", str(project_dir), "run", "exp_test", "--confirm"])
        time.sleep(0.5)

        # Cancel
        result = runner.invoke(cli, [
            "-p", str(project_dir), "cancel", "exp_test", "--confirm",
        ])
        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()

        # Verify status
        ledger = Ledger(project_dir / "ledger")
        exp = ledger.get("exp_test")
        assert exp.status == ExperimentStatus.ABANDONED
