"""Tests for the MCP server — project discovery and server build.

Protocol compliance is verified by `claude mcp list` showing ✓ Connected,
which uses the official MCP SDK (mcp>=1.0) for transport handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daedalus.mcp_server import (
    _build_mcp_server,
    _ensure_project,
    _find_project_root,
)


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------

class TestFindProjectRoot:
    def test_finds_in_current_dir(self, tmp_path):
        (tmp_path / "daedalus.yaml").write_text("project_name: test")
        assert _find_project_root(tmp_path) == tmp_path

    def test_finds_in_parent(self, tmp_path):
        (tmp_path / "daedalus.yaml").write_text("project_name: test")
        child = tmp_path / "subdir" / "deep"
        child.mkdir(parents=True)
        assert _find_project_root(child) == tmp_path

    def test_returns_none_when_not_found(self, tmp_path):
        assert _find_project_root(tmp_path) is None


class TestEnsureProject:
    def test_found(self, tmp_path):
        (tmp_path / "daedalus.yaml").write_text("project_name: test")
        assert _ensure_project(tmp_path) == tmp_path

    def test_not_found(self, tmp_path):
        assert _ensure_project(tmp_path) is None


# ---------------------------------------------------------------------------
# Server build (tools registration)
# ---------------------------------------------------------------------------

class TestBuildMCPServer:
    def test_builds_without_project(self):
        """Server builds even without a project path."""
        mcp = _build_mcp_server(None)
        assert mcp is not None

    def test_builds_with_project(self, tmp_path):
        """Server builds with a valid project path."""
        (tmp_path / "daedalus.yaml").write_text("project_name: test")
        (tmp_path / "ledger").mkdir()
        (tmp_path / "ledger" / "experiments.jsonl").touch()
        (tmp_path / "ledger" / "papers.jsonl").touch()
        mcp = _build_mcp_server(tmp_path)
        assert mcp is not None
