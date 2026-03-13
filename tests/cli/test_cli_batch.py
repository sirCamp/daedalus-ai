"""Tests for run-batch CLI command."""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from daedalus.cli import cli
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus
from daedalus.core.hypothesis import Hypothesis


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def project_dir(tmp_path):
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    (project / "program.md").write_text("# Test\n")
    (project / "daedalus.yaml").write_text(
        "project_name: test\nrunner:\n  type: local\n"
    )
    return project


def _make_experiment(exp_id: str, status: ExperimentStatus = ExperimentStatus.DRAFT) -> Experiment:
    return Experiment(
        id=exp_id,
        hypothesis=Hypothesis(
            statement=f"Test hypothesis for {exp_id}",
            rationale="Test rationale",
        ),
        config=ExperimentConfig(script="train.py", script_args={"lr": 0.001}),
        status=status,
    )


def _seed_experiments(project_dir, experiments: list[Experiment]):
    ledger_file = project_dir / "ledger" / "experiments.jsonl"
    with open(ledger_file, "w") as f:
        for exp in experiments:
            f.write(exp.model_dump_json() + "\n")


class TestRunBatch:
    def test_no_args_shows_error(self, runner, project_dir):
        result = runner.invoke(cli, ["-p", str(project_dir), "run-batch"])
        assert result.exit_code != 0

    def test_all_draft(self, runner, project_dir):
        exps = [
            _make_experiment("exp_001"),
            _make_experiment("exp_002"),
            _make_experiment("exp_003", ExperimentStatus.COMPLETED),
        ]
        _seed_experiments(project_dir, exps)

        with patch("daedalus.runners.factory.create_runner") as mock_create:
            mock_runner = MagicMock()
            mock_runner.launch.return_value = "local:/tmp/test"
            mock_create.return_value = mock_runner

            result = runner.invoke(
                cli,
                ["-p", str(project_dir), "run-batch", "--all-draft", "-y"],
            )
            assert result.exit_code == 0
            # Should launch 2 (exp_001 and exp_002), not exp_003
            assert mock_runner.launch.call_count == 2

    def test_specific_experiments(self, runner, project_dir):
        exps = [_make_experiment("exp_001"), _make_experiment("exp_002")]
        _seed_experiments(project_dir, exps)

        with patch("daedalus.runners.factory.create_runner") as mock_create:
            mock_runner = MagicMock()
            mock_runner.launch.return_value = "local:/tmp/test"
            mock_create.return_value = mock_runner

            result = runner.invoke(
                cli,
                ["-p", str(project_dir), "run-batch", "exp_001", "exp_002", "-y"],
            )
            assert result.exit_code == 0
            assert mock_runner.launch.call_count == 2

    def test_round_robin_hosts(self, runner, project_dir):
        exps = [
            _make_experiment("exp_001"),
            _make_experiment("exp_002"),
            _make_experiment("exp_003"),
        ]
        _seed_experiments(project_dir, exps)

        hosts_used = []

        with patch("daedalus.runners.factory.create_runner") as mock_create:
            mock_runner = MagicMock()
            mock_runner.launch.return_value = "ssh:host:/tmp/test"
            mock_create.return_value = mock_runner

            def capture_host(project, host=None):
                hosts_used.append(host)
                return mock_runner

            mock_create.side_effect = capture_host

            result = runner.invoke(
                cli,
                [
                    "-p", str(project_dir), "run-batch",
                    "exp_001", "exp_002", "exp_003",
                    "-H", "gpu-a100", "-H", "gpu-h100",
                    "-y",
                ],
            )
            assert result.exit_code == 0
            # Round-robin: exp_001→gpu-a100, exp_002→gpu-h100, exp_003→gpu-a100
            assert hosts_used == ["gpu-a100", "gpu-h100", "gpu-a100"]

    def test_no_draft_experiments(self, runner, project_dir):
        _seed_experiments(project_dir, [])
        result = runner.invoke(
            cli, ["-p", str(project_dir), "run-batch", "--all-draft", "-y"],
        )
        assert "No DRAFT" in result.output
