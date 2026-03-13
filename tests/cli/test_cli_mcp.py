"""Tests for install-mcp and uninstall-mcp CLI commands."""

import json

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


class TestInstallMCP:
    def test_install_project_level(self, runner, project_dir):
        """install-mcp creates .claude/settings.json in project dir."""
        result = runner.invoke(cli, ["-p", str(project_dir), "install-mcp"])
        assert result.exit_code == 0

        settings_file = project_dir / ".claude" / "settings.json"
        assert settings_file.exists()

        settings = json.loads(settings_file.read_text())
        assert "mcpServers" in settings
        assert "daedalus" in settings["mcpServers"]

        server = settings["mcpServers"]["daedalus"]
        assert server["args"][1] == "daedalus.mcp_server"
        assert server["args"][3] == str(project_dir)

    def test_install_global(self, runner, project_dir, tmp_path, monkeypatch):
        """install-mcp --global writes to ~/.claude/settings.json."""
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        result = runner.invoke(cli, ["-p", str(project_dir), "install-mcp", "--global"])
        assert result.exit_code == 0

        settings_file = fake_home / ".claude" / "settings.json"
        assert settings_file.exists()

        settings = json.loads(settings_file.read_text())
        # Global install uses "." so the server auto-detects from workspace cwd
        assert settings["mcpServers"]["daedalus"]["args"][3] == "."

    def test_install_preserves_existing_settings(self, runner, project_dir):
        """install-mcp preserves existing settings."""
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "theme": "dark",
            "mcpServers": {
                "other": {"command": "node", "args": ["other.js"]},
            },
        }))

        result = runner.invoke(cli, ["-p", str(project_dir), "install-mcp"])
        assert result.exit_code == 0

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert settings["theme"] == "dark"
        assert "other" in settings["mcpServers"]
        assert "daedalus" in settings["mcpServers"]

    def test_install_custom_python(self, runner, project_dir):
        """install-mcp --python uses custom python path."""
        result = runner.invoke(cli, [
            "-p", str(project_dir),
            "install-mcp", "--python", "/usr/bin/python3.12",
        ])
        assert result.exit_code == 0

        settings = json.loads(
            (project_dir / ".claude" / "settings.json").read_text()
        )
        assert settings["mcpServers"]["daedalus"]["command"] == "/usr/bin/python3.12"


class TestUninstallMCP:
    def test_uninstall_removes_daedalus(self, runner, project_dir):
        """uninstall-mcp removes daedalus from settings."""
        # First install
        runner.invoke(cli, ["-p", str(project_dir), "install-mcp"])
        # Then uninstall
        result = runner.invoke(cli, ["-p", str(project_dir), "uninstall-mcp"])
        assert result.exit_code == 0

        settings = json.loads(
            (project_dir / ".claude" / "settings.json").read_text()
        )
        assert "daedalus" not in settings.get("mcpServers", {})

    def test_uninstall_preserves_other_servers(self, runner, project_dir):
        """uninstall-mcp keeps other MCP servers."""
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "mcpServers": {
                "daedalus": {"command": "python", "args": []},
                "other": {"command": "node", "args": []},
            },
        }))

        result = runner.invoke(cli, ["-p", str(project_dir), "uninstall-mcp"])
        assert result.exit_code == 0

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert "other" in settings["mcpServers"]
        assert "daedalus" not in settings["mcpServers"]

    def test_uninstall_no_settings_file(self, runner, project_dir):
        """uninstall-mcp handles missing settings file gracefully."""
        result = runner.invoke(cli, ["-p", str(project_dir), "uninstall-mcp"])
        assert result.exit_code == 0
        assert "No settings file" in result.output

    def test_uninstall_not_registered(self, runner, project_dir):
        """uninstall-mcp handles not-registered gracefully."""
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({"mcpServers": {}}))

        result = runner.invoke(cli, ["-p", str(project_dir), "uninstall-mcp"])
        assert result.exit_code == 0
        assert "not registered" in result.output
