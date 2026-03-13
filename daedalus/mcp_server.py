"""Daedalus MCP server — exposes tools to Claude Code.

Usage:
    python -m daedalus.mcp_server --project /path/to/project

Register globally in ~/.claude/settings.json:
    {
        "mcpServers": {
            "daedalus": {
                "command": "python",
                "args": ["-m", "daedalus.mcp_server", "--project", "."]
            }
        }
    }

The server auto-discovers the project root by searching for daedalus.yaml
starting from --project and walking up. If no project is found, it creates
a minimal structure in the current directory on first tool call.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_project_root(start: Path) -> Path | None:
    """Find the project root by looking for daedalus.yaml upwards."""
    current = start.resolve()
    for parent in [current] + list(current.parents):
        if (parent / "daedalus.yaml").exists():
            return parent
    return None


def _ensure_project(path: Path) -> Path | None:
    """Find a valid Daedalus project starting from path.

    Searches for daedalus.yaml by walking up from path.
    Returns None if no project is found (caller should handle gracefully).
    """
    found = _find_project_root(path)
    if found:
        logger.info(f"Found project at: {found}")
        return found

    logger.warning(
        f"No daedalus.yaml found starting from {path}. "
        "Run 'daedalus init <name>' to create a project."
    )
    return None


def _build_json_schema(tool_schema: dict) -> dict:
    """Convert Anthropic tool schema to MCP/JSON-RPC tool schema."""
    return {
        "name": f"daedalus_{tool_schema['name']}",
        "description": tool_schema["description"],
        "inputSchema": tool_schema["input_schema"],
    }


def _handle_request(request: dict, executor) -> dict:
    """Handle a JSON-RPC request.

    If executor is None (no project found), tools/list returns an empty list
    and tools/call returns a helpful error message.
    """
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "daedalus",
                    "version": "0.1.0",
                },
            },
        }

    if method == "notifications/initialized":
        # No response needed for notifications
        return None  # type: ignore

    if method == "tools/list":
        from .agent.tools import TOOL_SCHEMAS
        # Always list tools even if no project — so Claude Code sees them
        tools = [_build_json_schema(s) for s in TOOL_SCHEMAS]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Strip daedalus_ prefix
        if tool_name.startswith("daedalus_"):
            tool_name = tool_name[9:]

        if executor is None:
            result_str = (
                "ERROR: No Daedalus project found. "
                "No daedalus.yaml was found in the current directory or any parent. "
                "Run 'daedalus init <name>' to create a project, then restart Claude Code."
            )
        else:
            result_str = executor.execute(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {"type": "text", "text": result_str},
                ],
            },
        }

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    }


def run_server(project_path: Path | None) -> None:
    """Run the MCP server on stdio."""
    if project_path is not None:
        from .agent.tools import ToolExecutor
        executor = ToolExecutor(project_path)
        logger.info(f"Daedalus MCP server started for project: {project_path}")
    else:
        executor = None
        logger.warning("Daedalus MCP server started WITHOUT a project (tools will return errors)")

    # Read JSON-RPC messages from stdin, write responses to stdout.
    # IMPORTANT: Use readline() in a loop, NOT `for line in sys.stdin:`.
    # The iterator uses a read-ahead buffer that conflicts with read(N)
    # for Content-Length framed messages, causing lost/corrupted data.
    while True:
        line = sys.stdin.readline()
        if not line:
            break  # EOF
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            # Content-Length header based protocol (MCP standard)
            if line.startswith("Content-Length:"):
                length = int(line.split(":", 1)[1].strip())
                # Skip header lines until empty line
                while True:
                    header_line = sys.stdin.readline().strip()
                    if not header_line:
                        break
                # Read body
                body = sys.stdin.read(length)
                try:
                    request = json.loads(body)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse body: {body[:200]}")
                    continue
            else:
                logger.warning(f"Unparseable input: {line[:200]}")
                continue

        response = _handle_request(request, executor)

        if response is not None:
            response_bytes = json.dumps(response).encode("utf-8")
            # Write using Content-Length framing (MCP standard)
            sys.stdout.buffer.write(
                f"Content-Length: {len(response_bytes)}\r\n\r\n".encode("utf-8")
            )
            sys.stdout.buffer.write(response_bytes)
            sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daedalus MCP server")
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=".",
        help="Path to Daedalus project directory (default: auto-detect from cwd)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # MCP uses stderr for logs
    )

    start_path = Path(args.project).resolve()
    logger.info(f"MCP server starting, cwd={Path.cwd()}, --project={start_path}")
    project_path = _ensure_project(start_path)
    if project_path:
        logger.info(f"Using project: {project_path}")
    run_server(project_path)


if __name__ == "__main__":
    main()
