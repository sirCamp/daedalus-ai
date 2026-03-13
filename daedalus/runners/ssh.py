"""SSH runner — launches experiments on remote hosts with sentinel monitoring.

The sentinel is a bash script deployed to the remote that:
1. Runs inside screen (survives SSH disconnects)
2. Monitors the training process (PID alive check)
3. Tails stdout.log and parses metrics/alerts
4. Writes structured events to events.jsonl

The local watcher reads events.jsonl via single SSH calls —
no persistent connection needed.
"""

from __future__ import annotations

import json
import logging
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..core.experiment import Experiment
from .base import RunStatus, build_env_vars, build_launch_cmd

logger = logging.getLogger(__name__)


class SSHConfig(BaseModel):
    """SSH connection configuration.

    Authentication: use ``key_path`` for key-based auth, or ``password``
    for password auth (requires ``sshpass`` installed on the local machine).
    If neither is set, SSH uses the default agent/config.
    """

    host: str
    user: str
    key_path: Path | None = None
    password: str | None = None
    remote_work_dir: str = "/tmp/daedalus"
    python_path: str = "python"
    port: int = 22


class SSHRunner:
    """Run experiments on a remote host via SSH.

    Uses plain ``ssh`` and ``scp`` commands (no paramiko dependency).
    The training script must already exist on the remote, or be synced
    via ``sync_files`` before launching.

    Run ID format: ``ssh:{host}:{remote_dir}``
    This encodes everything needed for stateless cross-session recovery.
    """

    def __init__(self, config: SSHConfig) -> None:
        self.config = config

    def _sshpass_prefix(self) -> list[str]:
        """Build sshpass prefix for password auth."""
        if self.config.password:
            return ["sshpass", "-p", self.config.password]
        return []

    def _ssh_base(self) -> list[str]:
        """Build base SSH command with options."""
        cmd = self._sshpass_prefix()
        cmd.extend(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"])
        if self.config.key_path:
            cmd.extend(["-i", str(self.config.key_path)])
        if self.config.port != 22:
            cmd.extend(["-p", str(self.config.port)])
        cmd.append(f"{self.config.user}@{self.config.host}")
        return cmd

    def _scp_base(self) -> list[str]:
        """Build base SCP command."""
        cmd = self._sshpass_prefix()
        cmd.extend(["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"])
        if self.config.key_path:
            cmd.extend(["-i", str(self.config.key_path)])
        if self.config.port != 22:
            cmd.extend(["-P", str(self.config.port)])
        return cmd

    def _run_ssh(self, remote_cmd: str, timeout: float = 30) -> subprocess.CompletedProcess:
        """Execute a command on the remote host."""
        cmd = self._ssh_base() + [remote_cmd]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _encode_run_id(self, remote_dir: str) -> str:
        """Encode host + remote_dir as stateless run_id."""
        return f"ssh:{self.config.host}:{remote_dir}"

    @staticmethod
    def _decode_run_id(run_id: str) -> tuple[str, str]:
        """Decode run_id to (host, remote_dir).

        Supports both new format (ssh:host:path) and legacy (ssh_pid).
        """
        if run_id.startswith("ssh:"):
            parts = run_id.split(":", 2)
            if len(parts) == 3:
                return parts[1], parts[2]
        # Legacy format: ssh_{pid} — can't recover remote_dir
        raise ValueError(f"Cannot decode run_id: {run_id}")

    def sync_files(self, local_dir: Path, remote_dir: str) -> None:
        """Sync local files to remote directory."""
        remote_path = f"{self.config.user}@{self.config.host}:{remote_dir}"
        self._run_ssh(f"mkdir -p {remote_dir}")
        cmd = self._scp_base() + ["-r", str(local_dir) + "/.", remote_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"SCP sync failed: {result.stderr}")

    def _deploy_sentinel(self, remote_dir: str) -> None:
        """Copy sentinel.sh to the remote work directory."""
        sentinel_src = Path(__file__).parent / "sentinel.sh"
        remote_path = f"{self.config.user}@{self.config.host}:{remote_dir}/sentinel.sh"
        cmd = self._scp_base() + [str(sentinel_src), remote_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to deploy sentinel: {result.stderr}")
        self._run_ssh(f"chmod +x {remote_dir}/sentinel.sh")

    def launch(self, experiment: Experiment, work_dir: Path) -> str:
        """Launch the experiment on the remote host.

        Deploys the sentinel script and starts both the training process
        and the sentinel inside a screen session.

        Args:
            experiment: The experiment to run.
            work_dir: Local work directory (used for config sync).

        Returns:
            Stateless run_id: ``ssh:{host}:{remote_dir}``.
        """
        remote_dir = f"{self.config.remote_work_dir}/{experiment.id}"

        # Create remote directory
        self._run_ssh(f"mkdir -p {remote_dir}/logs")

        # Write config to remote
        config_json = experiment.config.model_dump_json()
        self._run_ssh(
            f"cat > {remote_dir}/config.json << 'DAEDALUS_EOF'\n{config_json}\nDAEDALUS_EOF"
        )

        # Deploy sentinel
        self._deploy_sentinel(remote_dir)

        # Build training command using launcher helper
        cmd_parts = build_launch_cmd(experiment.config, python=self.config.python_path)
        train_cmd = " ".join(str(p) for p in cmd_parts)
        env_vars = build_env_vars(experiment.config)
        env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())

        # Launch training process via a launcher script to avoid SSH
        # hanging on file descriptor close. Write a script, then execute it.
        launcher_script = (
            f"#!/bin/bash\n"
            f"cd {remote_dir}\n"
            f"{env_str + ' ' if env_str else ''}nohup {train_cmd} "
            f"> logs/stdout.log 2> logs/stderr.log &\n"
            f"echo $!\n"
        )
        # Write launcher script
        escaped = launcher_script.replace("'", "'\\''")
        self._run_ssh(
            f"cat > {remote_dir}/_launch.sh << 'DAEDALUS_LAUNCH_EOF'\n{launcher_script}DAEDALUS_LAUNCH_EOF\n"
            f"chmod +x {remote_dir}/_launch.sh"
        )
        # Execute launcher — it backgrounds the process and prints PID
        result = self._run_ssh(f"bash {remote_dir}/_launch.sh", timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"SSH launch failed: {result.stderr}")

        remote_pid = result.stdout.strip()

        # Write initial status
        self._run_ssh(
            f'echo \'{{"state":"running","pid":{remote_pid}}}\' > {remote_dir}/status.json'
        )

        # Start sentinel in screen session
        screen_name = f"daedalus_{experiment.id}"
        sentinel_cmd = (
            f"screen -dmS {screen_name} bash {remote_dir}/sentinel.sh "
            f"{remote_dir} {remote_pid} 30 10"
        )
        sentinel_result = self._run_ssh(sentinel_cmd, timeout=10)
        if sentinel_result.returncode != 0:
            # Fallback: try without screen (tmux, or just nohup)
            logger.warning(
                f"screen not available, trying nohup: {sentinel_result.stderr}"
            )
            fallback_cmd = (
                f"nohup bash {remote_dir}/sentinel.sh "
                f"{remote_dir} {remote_pid} 30 10 "
                f"> /dev/null 2>&1 & echo $!"
            )
            self._run_ssh(fallback_cmd, timeout=10)

        run_id = self._encode_run_id(remote_dir)
        logger.info(f"Launched {run_id} (PID {remote_pid}) with sentinel")
        return run_id

    def poll(self, run_id: str) -> RunStatus:
        """Check experiment status by reading sentinel events.

        Single SSH call: reads status.json and last events.
        """
        try:
            _, remote_dir = self._decode_run_id(run_id)
        except ValueError as e:
            return RunStatus(state="failed", error=str(e))

        # Read status.json and last 5 events in one SSH call
        result = self._run_ssh(
            f"cat {remote_dir}/status.json 2>/dev/null; "
            f"echo '---SEPARATOR---'; "
            f"tail -5 {remote_dir}/events.jsonl 2>/dev/null"
        )
        if result.returncode != 0:
            return RunStatus(state="failed", error=f"SSH failed: {result.stderr}")

        parts = result.stdout.split("---SEPARATOR---")
        status_str = parts[0].strip() if len(parts) > 0 else ""
        events_str = parts[1].strip() if len(parts) > 1 else ""

        # Parse status.json
        state = "running"
        progress = None
        error = None

        if status_str:
            try:
                status_data = json.loads(status_str)
                state = status_data.get("state", "running")
                progress = status_data.get("progress")
            except json.JSONDecodeError:
                pass

        # Extract progress from recent events
        if events_str:
            for line in reversed(events_str.splitlines()):
                try:
                    event = json.loads(line)
                    etype = event.get("event", "")

                    if etype == "COMPLETED":
                        return RunStatus(state="completed")
                    if etype == "FAILED":
                        return RunStatus(
                            state="failed",
                            error=event.get("error", "unknown"),
                        )
                    if etype == "PROGRESS" and progress is None:
                        progress = f"{event.get('key', '')}: {event.get('value', '')}"
                    if etype == "METRIC" and progress is None:
                        metric = event.get("metric", "")
                        value = event.get("value", "")
                        progress = f"{metric}={value}"
                except json.JSONDecodeError:
                    continue

        if state == "completed":
            return RunStatus(state="completed")
        if state == "failed":
            return RunStatus(state="failed", error=error or "Process failed")

        return RunStatus(state="running", progress=progress)

    def fetch_events(self, run_id: str, since_line: int = 0) -> list[dict]:
        """Fetch events from remote events.jsonl.

        Args:
            run_id: The run identifier.
            since_line: Skip first N lines (for incremental reads).

        Returns:
            List of event dicts.
        """
        try:
            _, remote_dir = self._decode_run_id(run_id)
        except ValueError:
            return []

        if since_line > 0:
            cmd = f"tail -n +{since_line + 1} {remote_dir}/events.jsonl 2>/dev/null"
        else:
            cmd = f"cat {remote_dir}/events.jsonl 2>/dev/null"

        result = self._run_ssh(cmd)
        if result.returncode != 0 or not result.stdout.strip():
            return []

        events = []
        for line in result.stdout.strip().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def fetch_results(self, run_id: str, adapter: str = "auto") -> dict[str, Any]:
        """Fetch results from remote, trying multiple formats."""
        try:
            _, remote_dir = self._decode_run_id(run_id)
        except ValueError as e:
            raise ValueError(str(e))

        # Try results.json first (most common)
        result = self._run_ssh(f"cat {remote_dir}/results.json 2>/dev/null")
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                return data
            if isinstance(data, dict):
                return {"eval": {k: float(v) for k, v in data.items()
                        if isinstance(v, (int, float))}}

        # Try HF Trainer format
        result = self._run_ssh(
            f"cat {remote_dir}/eval_results.json 2>/dev/null || "
            f"find {remote_dir} -name eval_results.json -print -quit 2>/dev/null "
            f"| head -1 | xargs cat 2>/dev/null"
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            metrics = {
                k.removeprefix("eval_"): float(v)
                for k, v in data.items()
                if isinstance(v, (int, float))
            }
            return {"eval": metrics}

        raise FileNotFoundError(f"No results found on remote: {remote_dir}")

    def cancel(self, run_id: str) -> None:
        """Kill the training process and sentinel."""
        try:
            _, remote_dir = self._decode_run_id(run_id)
        except ValueError:
            return

        # Kill training process
        self._run_ssh(
            f"cat {remote_dir}/status.json 2>/dev/null | "
            f"python3 -c \"import sys,json; print(json.load(sys.stdin).get('pid',''))\" 2>/dev/null | "
            f"xargs -r kill 2>/dev/null; "
            # Kill sentinel
            f"cat {remote_dir}/sentinel.pid 2>/dev/null | xargs -r kill 2>/dev/null"
        )
        logger.info(f"Cancelled {run_id}")

    def logs(self, run_id: str, tail: int = 50) -> str:
        """Fetch last N lines from both stdout and stderr on remote.

        Most ML frameworks (HF Trainer, tqdm) write to stderr, so we
        merge both streams for a complete picture.
        """
        try:
            _, remote_dir = self._decode_run_id(run_id)
        except ValueError as e:
            return str(e)

        # Read both logs in one SSH call
        result = self._run_ssh(
            f"echo '=== stdout ===' && "
            f"tail -n {tail} {remote_dir}/logs/stdout.log 2>/dev/null; "
            f"echo '=== stderr ===' && "
            f"tail -n {tail} {remote_dir}/logs/stderr.log 2>/dev/null"
        )
        return result.stdout if result.returncode == 0 else "No logs available."
