"""Tests for sync and setup-env CLI commands."""

import json

import pytest
import yaml
from click.testing import CliRunner

from daedalus.cli import cli


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
    return project


class TestSync:
    def test_sync_local_runner_warns(self, runner, project_dir):
        """sync with local runner prints a warning."""
        (project_dir / "daedalus.yaml").write_text(
            yaml.dump({"runner": {"type": "local"}})
        )

        result = runner.invoke(cli, ["-p", str(project_dir), "sync"])
        assert result.exit_code == 0
        assert "only needed for SSH" in result.output


class TestSetupEnv:
    def test_setup_env_local_runner_warns(self, runner, project_dir):
        """setup-env with local runner prints a warning."""
        (project_dir / "daedalus.yaml").write_text(
            yaml.dump({"runner": {"type": "local"}})
        )

        result = runner.invoke(cli, ["-p", str(project_dir), "setup-env"])
        assert result.exit_code == 0
        assert "remote hosts only" in result.output


class TestInitStack:
    def test_init_creates_stack_section(self, runner, tmp_path):
        """daedalus init includes stack section in daedalus.yaml."""
        result = runner.invoke(cli, ["init", "test-proj", "--path", str(tmp_path)])
        assert result.exit_code == 0

        config_file = tmp_path / "test-proj" / "daedalus.yaml"
        assert config_file.exists()

        config = yaml.safe_load(config_file.read_text())
        assert "stack" in config
        assert "python" in config["stack"]
        assert "libraries" in config["stack"]

    def test_init_creates_stack_in_program(self, runner, tmp_path):
        """daedalus init includes Stack section in program.md."""
        result = runner.invoke(cli, ["init", "test-proj", "--path", str(tmp_path)])
        assert result.exit_code == 0

        program = (tmp_path / "test-proj" / "program.md").read_text()
        assert "## Stack" in program
        assert "Training" in program
