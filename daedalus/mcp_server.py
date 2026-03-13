"""Daedalus MCP server — exposes tools to Claude Code.

Usage:
    python -u -m daedalus.mcp_server --project /path/to/project

Register globally (recommended):
    daedalus install-mcp

Or manually:
    claude mcp add -s user daedalus -- python -u -m daedalus.mcp_server --project .

Uses the official MCP Python SDK (mcp>=1.0) for protocol handling.
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
    """Find a valid Daedalus project starting from path."""
    found = _find_project_root(path)
    if found:
        logger.info("Found project at: %s", found)
        return found

    logger.warning(
        "No daedalus.yaml found starting from %s. "
        "Run 'daedalus init <name>' to create a project.",
        path,
    )
    return None


def _build_mcp_server(project_path: Path | None):
    """Build and return an MCP server with all Daedalus tools registered.

    Uses the low-level ``mcp.server.Server`` class so that pre-defined
    JSON schemas from ``TOOL_SCHEMAS`` are forwarded verbatim to the
    MCP protocol (FastMCP infers schemas from function signatures, which
    doesn't work for our ``**kwargs`` handlers).
    """
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    from .agent.tools import TOOL_SCHEMAS

    executor = None
    if project_path is not None:
        from .agent.tools import ToolExecutor
        executor = ToolExecutor(project_path)

    server = Server("daedalus")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=f"daedalus_{schema['name']}",
                description=schema["description"],
                inputSchema=schema["input_schema"],
            )
            for schema in TOOL_SCHEMAS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        tool_name = name.removeprefix("daedalus_")
        if executor is None:
            text = (
                "ERROR: No Daedalus project found. "
                "No daedalus.yaml was found in the current directory or any parent. "
                "Run 'daedalus init <name>' to create a project, then restart Claude Code."
            )
        else:
            text = executor.execute(tool_name, arguments or {})
        return [TextContent(type="text", text=text)]

    return server


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
        stream=sys.stderr,
    )

    start_path = Path(args.project).resolve()
    logger.info("MCP server starting, cwd=%s, --project=%s", Path.cwd(), start_path)
    project_path = _ensure_project(start_path)
    if project_path:
        logger.info("Using project: %s", project_path)

    server = _build_mcp_server(project_path)

    # Run with stdio transport (MCP SDK handles Content-Length framing)
    import asyncio
    from mcp.server.stdio import stdio_server

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
