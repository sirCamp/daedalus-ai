"""Tests for remote_exec MCP tool."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from daedalus.agent.tools import TOOL_SCHEMAS, ToolExecutor


@pytest.fixture
def ssh_project(tmp_path: Path) -> Path:
    """Create a project configured with SSH runner."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()

    (project / "daedalus.yaml").write_text(yaml.dump({
        "project_name": "test",
        "runner": {"type": "ssh", "default_host": "gpu-1"},
    }))
    (project / "runner_config.yaml").write_text(yaml.dump({
        "hosts": {
            "gpu-1": {
                "host": "192.168.0.1",
                "user": "testuser",
                "remote_work_dir": "/data/test",
                "python_path": "python3",
            },
        },
    }))
    return project


@pytest.fixture
def local_project(tmp_path: Path) -> Path:
    """Create a project configured with local runner."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()

    (project / "daedalus.yaml").write_text(yaml.dump({
        "project_name": "test",
        "runner": {"type": "local"},
    }))
    return project


class TestRemoteExecSchema:
    def test_schema_exists(self):
        names = {s["name"] for s in TOOL_SCHEMAS}
        assert "remote_exec" in names

    def test_schema_fields(self):
        schema = next(s for s in TOOL_SCHEMAS if s["name"] == "remote_exec")
        props = schema["input_schema"]["properties"]
        assert "command" in props
        assert "host" in props
        assert "mutating" in props
        assert "confirmed" in props
        assert "timeout" in props
        assert schema["input_schema"]["required"] == ["command"]


class TestRemoteExecReadOnly:
    @patch("daedalus.runners.ssh.subprocess.run")
    def test_read_only_executes_directly(self, mock_run, ssh_project):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="GPU 0: RTX 3090\n", stderr=""
        )
        executor = ToolExecutor(ssh_project)
        result = json.loads(executor.execute("remote_exec", {
            "command": "nvidia-smi --list-gpus",
        }))

        assert result["exit_code"] == 0
        assert "RTX 3090" in result["stdout"]
        assert result["host"] == "192.168.0.1"

    @patch("daedalus.runners.ssh.subprocess.run")
    def test_read_only_with_host(self, mock_run, ssh_project):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="100G\n", stderr=""
        )
        executor = ToolExecutor(ssh_project)
        result = json.loads(executor.execute("remote_exec", {
            "command": "df -h /data",
            "host": "gpu-1",
        }))

        assert result["exit_code"] == 0

    @patch("daedalus.runners.ssh.subprocess.run")
    def test_command_failure(self, mock_run, ssh_project):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="command not found"
        )
        executor = ToolExecutor(ssh_project)
        result = json.loads(executor.execute("remote_exec", {
            "command": "nonexistent_cmd",
        }))

        assert result["exit_code"] == 1
        assert "command not found" in result["stderr"]


class TestRemoteExecMutating:
    def test_mutating_without_confirm_blocks(self, ssh_project):
        executor = ToolExecutor(ssh_project)
        result = json.loads(executor.execute("remote_exec", {
            "command": "pip install transformers",
            "mutating": True,
        }))

        assert result["needs_confirmation"] is True
        assert "pip install transformers" in result["command"]
        assert "approval" in result["message"].lower()

    @patch("daedalus.runners.ssh.subprocess.run")
    def test_mutating_with_confirm_executes(self, mock_run, ssh_project):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Successfully installed\n", stderr=""
        )
        executor = ToolExecutor(ssh_project)
        result = json.loads(executor.execute("remote_exec", {
            "command": "pip install transformers",
            "mutating": True,
            "confirmed": True,
        }))

        assert result["exit_code"] == 0
        assert "Successfully installed" in result["stdout"]


class TestRemoteExecLocalRunner:
    def test_local_runner_rejected(self, local_project):
        executor = ToolExecutor(local_project)
        result = json.loads(executor.execute("remote_exec", {
            "command": "ls",
        }))

        assert "error" in result
        assert "SSH" in result["error"]


class TestRemoteExecTimeout:
    @patch("daedalus.runners.ssh.subprocess.run")
    def test_custom_timeout(self, mock_run, ssh_project):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok\n", stderr=""
        )
        executor = ToolExecutor(ssh_project)
        executor.execute("remote_exec", {
            "command": "long_running_cmd",
            "timeout": 120,
        })

        # Verify timeout was passed to subprocess.run
        call_args = mock_run.call_args
        assert call_args.kwargs.get("timeout") == 120 or call_args[1].get("timeout") == 120

    @patch("daedalus.runners.ssh.subprocess.run")
    def test_timeout_capped_at_600(self, mock_run, ssh_project):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok\n", stderr=""
        )
        executor = ToolExecutor(ssh_project)
        executor.execute("remote_exec", {
            "command": "cmd",
            "timeout": 999,
        })

        call_args = mock_run.call_args
        assert call_args.kwargs.get("timeout") == 600 or call_args[1].get("timeout") == 600

    @patch("daedalus.runners.ssh.subprocess.run")
    def test_timeout_expired_returns_error(self, mock_run, ssh_project):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        executor = ToolExecutor(ssh_project)
        result = json.loads(executor.execute("remote_exec", {
            "command": "pip install transformers",
            "mutating": True,
            "confirmed": True,
            "timeout": 30,
        }))

        assert "error" in result
        assert "timed out" in result["error"]
        assert "hint" in result
