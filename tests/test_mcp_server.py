"""Tests for the MCP server — protocol compliance, stdio framing, project discovery."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from daedalus.mcp_server import (
    _build_json_schema,
    _ensure_project,
    _find_project_root,
    _handle_request,
    run_server,
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
# Schema conversion
# ---------------------------------------------------------------------------

class TestBuildJsonSchema:
    def test_adds_prefix(self):
        tool = {
            "name": "get_context",
            "description": "Get context",
            "input_schema": {"type": "object", "properties": {}},
        }
        result = _build_json_schema(tool)
        assert result["name"] == "daedalus_get_context"
        assert result["description"] == "Get context"
        assert result["inputSchema"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------

class TestHandleRequest:
    def test_initialize(self):
        resp = _handle_request({"method": "initialize", "id": 1}, executor=None)
        assert resp["id"] == 1
        assert resp["result"]["serverInfo"]["name"] == "daedalus"
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert "tools" in resp["result"]["capabilities"]

    def test_initialized_notification(self):
        resp = _handle_request({"method": "notifications/initialized", "id": None}, executor=None)
        assert resp is None

    def test_tools_list_no_executor(self):
        resp = _handle_request({"method": "tools/list", "id": 2}, executor=None)
        tools = resp["result"]["tools"]
        assert len(tools) > 0
        # All tools should have daedalus_ prefix
        for t in tools:
            assert t["name"].startswith("daedalus_")

    def test_tools_list_has_all_25(self):
        resp = _handle_request({"method": "tools/list", "id": 3}, executor=None)
        tools = resp["result"]["tools"]
        assert len(tools) == 28

    def test_tools_call_no_executor(self):
        resp = _handle_request(
            {
                "method": "tools/call",
                "id": 4,
                "params": {"name": "daedalus_get_context", "arguments": {}},
            },
            executor=None,
        )
        text = resp["result"]["content"][0]["text"]
        assert "ERROR" in text
        assert "daedalus init" in text

    def test_tools_call_with_executor(self):
        executor = MagicMock()
        executor.execute.return_value = '{"context": "test"}'

        resp = _handle_request(
            {
                "method": "tools/call",
                "id": 5,
                "params": {"name": "daedalus_get_context", "arguments": {"mode": "full"}},
            },
            executor=executor,
        )
        executor.execute.assert_called_once_with("get_context", {"mode": "full"})
        text = resp["result"]["content"][0]["text"]
        assert "test" in text

    def test_tools_call_strips_prefix(self):
        executor = MagicMock()
        executor.execute.return_value = "{}"

        _handle_request(
            {
                "method": "tools/call",
                "id": 6,
                "params": {"name": "daedalus_search_papers", "arguments": {"query": "test"}},
            },
            executor=executor,
        )
        executor.execute.assert_called_once_with("search_papers", {"query": "test"})

    def test_unknown_method(self):
        resp = _handle_request({"method": "foo/bar", "id": 7}, executor=None)
        assert "error" in resp
        assert resp["error"]["code"] == -32601
        assert "foo/bar" in resp["error"]["message"]

    def test_missing_method(self):
        resp = _handle_request({"id": 8}, executor=None)
        assert "error" in resp


# ---------------------------------------------------------------------------
# Content-Length framing (run_server)
# ---------------------------------------------------------------------------

class TestRunServer:
    def _make_request(self, method: str, req_id: int = 1, params: dict | None = None) -> str:
        """Build a Content-Length framed JSON-RPC message."""
        body = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id,
            "params": params or {},
        })
        return f"Content-Length: {len(body)}\r\n\r\n{body}"

    def _make_raw_json(self, method: str, req_id: int = 1) -> str:
        """Build a raw JSON line (no Content-Length framing)."""
        return json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id,
        })

    def test_content_length_framing(self, tmp_path):
        """Server should handle Content-Length framed messages."""
        (tmp_path / "daedalus.yaml").write_text("project_name: test")
        (tmp_path / "ledger").mkdir()

        request = self._make_request("initialize")
        stdin = io.StringIO(request + "\n")
        stdout_buffer = io.BytesIO()

        with patch.object(sys, "stdin", stdin):
            stdout_mock = MagicMock()
            stdout_mock.buffer = stdout_buffer
            with patch.object(sys, "stdout", stdout_mock):
                run_server(tmp_path)

        output = stdout_buffer.getvalue().decode("utf-8")
        assert "Content-Length:" in output
        # Extract JSON body after headers
        parts = output.split("\r\n\r\n", 1)
        assert len(parts) == 2
        response = json.loads(parts[1])
        assert response["result"]["serverInfo"]["name"] == "daedalus"

    def test_raw_json_line(self, tmp_path):
        """Server should also handle raw JSON lines."""
        request = self._make_raw_json("initialize")
        stdin = io.StringIO(request + "\n")
        stdout_buffer = io.BytesIO()

        with patch.object(sys, "stdin", stdin):
            stdout_mock = MagicMock()
            stdout_mock.buffer = stdout_buffer
            with patch.object(sys, "stdout", stdout_mock):
                run_server(None)

        output = stdout_buffer.getvalue().decode("utf-8")
        assert "Content-Length:" in output

    def test_eof_exits_cleanly(self):
        """Server should exit on EOF."""
        stdin = io.StringIO("")  # Immediately EOF
        with patch.object(sys, "stdin", stdin):
            stdout_mock = MagicMock()
            stdout_mock.buffer = io.BytesIO()
            with patch.object(sys, "stdout", stdout_mock):
                run_server(None)  # Should not hang or crash

    def test_notification_no_response(self):
        """Notifications should not produce a response."""
        request = self._make_raw_json("notifications/initialized")
        stdin = io.StringIO(request + "\n")
        stdout_buffer = io.BytesIO()

        with patch.object(sys, "stdin", stdin):
            stdout_mock = MagicMock()
            stdout_mock.buffer = stdout_buffer
            with patch.object(sys, "stdout", stdout_mock):
                run_server(None)

        assert stdout_buffer.getvalue() == b""

    def test_malformed_json_skipped(self):
        """Malformed input should be skipped without crashing."""
        lines = "not json\n" + self._make_raw_json("initialize") + "\n"
        stdin = io.StringIO(lines)
        stdout_buffer = io.BytesIO()

        with patch.object(sys, "stdin", stdin):
            stdout_mock = MagicMock()
            stdout_mock.buffer = stdout_buffer
            with patch.object(sys, "stdout", stdout_mock):
                run_server(None)

        output = stdout_buffer.getvalue().decode("utf-8")
        # Should still process the valid request
        assert "daedalus" in output

    def test_multiple_requests(self):
        """Server should handle multiple sequential requests."""
        req1 = self._make_raw_json("initialize", req_id=1)
        req2 = self._make_raw_json("tools/list", req_id=2)
        stdin = io.StringIO(req1 + "\n" + req2 + "\n")
        stdout_buffer = io.BytesIO()

        with patch.object(sys, "stdin", stdin):
            stdout_mock = MagicMock()
            stdout_mock.buffer = stdout_buffer
            with patch.object(sys, "stdout", stdout_mock):
                run_server(None)

        output = stdout_buffer.getvalue().decode("utf-8")
        # Should have two Content-Length responses
        assert output.count("Content-Length:") == 2
