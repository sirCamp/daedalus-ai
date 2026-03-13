"""Tests for install-mcp and uninstall-mcp CLI commands."""

import json
import os
from unittest.mock import patch

import pytest
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
    (project / "daedalus.yaml").write_text("project_name: test\n")
    return project


def _mock_subprocess_run(returncode: int = 0, stderr: str = ""):
    """Create a mock for subprocess.run that returns given results."""
    from unittest.mock import MagicMock
    mock_result = MagicMock()
    mock_result.returncode = returncode
    mock_result.stdout = ""
    mock_result.stderr = stderr
    return MagicMock(return_value=mock_result)


class TestInstallMCP:
    def test_install_calls_claude_mcp_add(self, runner, project_dir, tmp_path):
        """install-mcp calls `claude mcp add` with correct args."""
        mock_run = _mock_subprocess_run()
        fake_home = tmp_path / "fakehome"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / "settings.json").write_text("{}")

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", mock_run), \
             patch("pathlib.Path.home", return_value=fake_home):
            result = runner.invoke(cli, ["-p", str(project_dir), "install-mcp"])

        assert result.exit_code == 0
        # Should have called subprocess.run twice: remove + add
        assert mock_run.call_count == 2

        # Second call is the `add`
        add_call = mock_run.call_args_list[1]
        cmd = add_call[0][0]
        assert "mcp" in cmd
        assert "add" in cmd
        assert "-s" in cmd
        assert "user" in cmd  # default scope
        assert "daedalus" in cmd
        assert "-m" in cmd
        assert "daedalus.mcp_server" in cmd

    def test_install_project_scope(self, runner, project_dir, tmp_path):
        """install-mcp --scope project passes project scope."""
        mock_run = _mock_subprocess_run()
        fake_home = tmp_path / "fakehome"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / "settings.json").write_text("{}")

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", mock_run), \
             patch("pathlib.Path.home", return_value=fake_home):
            result = runner.invoke(cli, [
                "-p", str(project_dir), "install-mcp", "--scope", "project",
            ])

        assert result.exit_code == 0
        add_call = mock_run.call_args_list[1]
        cmd = add_call[0][0]
        assert "project" in cmd

    def test_install_custom_python(self, runner, project_dir, tmp_path):
        """install-mcp --python uses custom python path."""
        mock_run = _mock_subprocess_run()
        fake_home = tmp_path / "fakehome"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / "settings.json").write_text("{}")

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", mock_run), \
             patch("pathlib.Path.home", return_value=fake_home):
            result = runner.invoke(cli, [
                "-p", str(project_dir),
                "install-mcp", "--python", "/usr/bin/python3.12",
            ])

        assert result.exit_code == 0
        add_call = mock_run.call_args_list[1]
        cmd = add_call[0][0]
        assert "/usr/bin/python3.12" in cmd

    def test_install_no_claude_cli(self, runner, project_dir):
        """install-mcp fails gracefully when claude CLI is not found."""
        with patch("shutil.which", return_value=None):
            result = runner.invoke(cli, ["-p", str(project_dir), "install-mcp"])

        assert result.exit_code != 0
        assert "claude" in result.output.lower()

    def test_install_auto_approves_tools(self, runner, project_dir, tmp_path):
        """install-mcp adds mcp__daedalus__* to allowed permissions."""
        mock_run = _mock_subprocess_run()
        fake_home = tmp_path / "fakehome"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / "settings.json").write_text("{}")

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", mock_run), \
             patch("pathlib.Path.home", return_value=fake_home):
            runner.invoke(cli, ["-p", str(project_dir), "install-mcp"])

        settings = json.loads((fake_home / ".claude" / "settings.json").read_text())
        assert "mcp__daedalus__*" in settings["permissions"]["allow"]


class TestUninstallMCP:
    def test_uninstall_calls_claude_mcp_remove(self, runner, project_dir):
        """uninstall-mcp calls `claude mcp remove`."""
        mock_run = _mock_subprocess_run()

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", mock_run):
            result = runner.invoke(cli, ["-p", str(project_dir), "uninstall-mcp"])

        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]
        assert "remove" in cmd
        assert "daedalus" in cmd

    def test_uninstall_not_registered(self, runner, project_dir):
        """uninstall-mcp handles not-registered gracefully."""
        mock_run = _mock_subprocess_run(returncode=1, stderr="Server not found")

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", mock_run):
            result = runner.invoke(cli, ["-p", str(project_dir), "uninstall-mcp"])

        assert result.exit_code == 0
        assert "not registered" in result.output.lower()

    def test_uninstall_no_claude_cli(self, runner, project_dir):
        """uninstall-mcp fails gracefully when claude CLI is not found."""
        with patch("shutil.which", return_value=None):
            result = runner.invoke(cli, ["-p", str(project_dir), "uninstall-mcp"])

        assert result.exit_code != 0
