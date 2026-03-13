"""Runner factory — create runners from project config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .local import LocalRunner
from .ssh import SSHRunner, SSHConfig


def create_runner(
    project_path: Path,
    host: str | None = None,
) -> LocalRunner | SSHRunner:
    """Create a runner from the project's daedalus.yaml and runner_config.yaml.

    Reads ``daedalus.yaml`` for the runner type, then ``runner_config.yaml``
    for backend-specific settings.

    Args:
        project_path: Path to the project directory.
        host: For SSH, the named host to use. Overrides default_host in config.
    """
    # Load project config
    daedalus_yaml = project_path / "daedalus.yaml"
    if not daedalus_yaml.exists():
        return LocalRunner()

    project_config = yaml.safe_load(daedalus_yaml.read_text()) or {}
    runner_spec = project_config.get("runner", {})
    runner_type = runner_spec.get("type", "local")

    # Load runner config
    runner_config_file = project_path / "runner_config.yaml"
    runner_config: dict[str, Any] = {}
    if runner_config_file.exists():
        runner_config = yaml.safe_load(runner_config_file.read_text()) or {}

    if runner_type == "local":
        local_config = runner_config.get("local", {})
        return LocalRunner(python=local_config.get("python", "python"))

    if runner_type == "ssh":
        ssh_config = _resolve_ssh_config(runner_spec, runner_config, host)
        return SSHRunner(SSHConfig(**ssh_config))

    raise ValueError(f"Unknown runner type: {runner_type}")


def _resolve_ssh_config(
    runner_spec: dict,
    runner_config: dict,
    host: str | None,
) -> dict[str, Any]:
    """Resolve SSH config, supporting both legacy and multi-host formats.

    Multi-host format (preferred):
        runner_config.yaml:
            hosts:
                gpu-a100: {host: ..., user: ...}
                gpu-h100: {host: ..., user: ...}

        daedalus.yaml:
            runner:
                type: ssh
                default_host: gpu-a100

    Legacy format (still supported):
        runner_config.yaml:
            ssh: {host: ..., user: ...}

        daedalus.yaml:
            runner:
                type: ssh
                config_key: ssh
    """
    hosts = runner_config.get("hosts")

    if hosts and isinstance(hosts, dict):
        # Multi-host format
        if host:
            if host not in hosts:
                available = ", ".join(hosts.keys())
                raise ValueError(
                    f"Unknown host '{host}'. Available: {available}"
                )
            return hosts[host]

        # Use default_host from daedalus.yaml
        default_host = runner_spec.get("default_host")
        if default_host:
            if default_host not in hosts:
                available = ", ".join(hosts.keys())
                raise ValueError(
                    f"Default host '{default_host}' not found. Available: {available}"
                )
            return hosts[default_host]

        # Fall back to first host
        return next(iter(hosts.values()))

    # Legacy single-host format
    config_key = runner_spec.get("config_key", "ssh")
    return runner_config.get(config_key, {})
