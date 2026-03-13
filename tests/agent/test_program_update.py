"""Tests for program.md update tool."""

import pytest

from daedalus.agent.tools import ToolExecutor


def _setup_project(tmp_path):
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    return project


class TestUpdateProgram:
    def test_rewrite(self, tmp_path):
        project = _setup_project(tmp_path)
        (project / "program.md").write_text("# Old Program\n\nOld content.\n")

        executor = ToolExecutor(project)
        result = executor.execute("update_program", {
            "action": "rewrite",
            "content": "# New Program\n\nNew content.\n",
        })
        import json
        data = json.loads(result)
        assert data["updated"] is True
        assert "New content" in (project / "program.md").read_text()
        assert "Old content" not in (project / "program.md").read_text()

    def test_append(self, tmp_path):
        project = _setup_project(tmp_path)
        (project / "program.md").write_text("# Research\n\nGoal 1.\n")

        executor = ToolExecutor(project)
        result = executor.execute("update_program", {
            "action": "append",
            "content": "## New Section\n\nGoal 2.",
        })
        import json
        data = json.loads(result)
        assert data["updated"] is True
        content = (project / "program.md").read_text()
        assert "Goal 1" in content
        assert "Goal 2" in content

    def test_replace_section(self, tmp_path):
        project = _setup_project(tmp_path)
        (project / "program.md").write_text(
            "# Research\n\n## Goals\n\nOld goals.\n\n## Methods\n\nOld methods.\n"
        )

        executor = ToolExecutor(project)
        result = executor.execute("update_program", {
            "action": "replace_section",
            "content": "New goals after pivot.",
            "section_heading": "Goals",
        })
        import json
        data = json.loads(result)
        assert data["updated"] is True
        content = (project / "program.md").read_text()
        assert "New goals after pivot" in content
        assert "Old goals" not in content
        assert "Old methods" in content  # other section preserved

    def test_replace_section_not_found(self, tmp_path):
        project = _setup_project(tmp_path)
        (project / "program.md").write_text("# Research\n\n## Goals\n\nGoals.\n")

        executor = ToolExecutor(project)
        result = executor.execute("update_program", {
            "action": "replace_section",
            "content": "New stuff",
            "section_heading": "Nonexistent",
        })
        import json
        data = json.loads(result)
        assert "error" in data

    def test_replace_section_requires_heading(self, tmp_path):
        project = _setup_project(tmp_path)
        (project / "program.md").write_text("# Research\n")

        executor = ToolExecutor(project)
        result = executor.execute("update_program", {
            "action": "replace_section",
            "content": "New stuff",
        })
        import json
        data = json.loads(result)
        assert "error" in data

    def test_append_creates_file(self, tmp_path):
        project = _setup_project(tmp_path)
        # No program.md exists

        executor = ToolExecutor(project)
        result = executor.execute("update_program", {
            "action": "append",
            "content": "# New Program\n\nStarting fresh.",
        })
        import json
        data = json.loads(result)
        assert data["updated"] is True
        assert (project / "program.md").exists()
