"""Local runner — launches experiments as subprocesses."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from ..core.experiment import Experiment
from .base import RunStatus, build_env_vars, build_launch_cmd

logger = logging.getLogger(__name__)


class LocalRunner:
    """Run experiments as local subprocesses.

    The runner launches the training script, writes a PID file, and polls
    the process status. Results are expected in ``{work_dir}/results.json``.

    The run_id encodes the work directory path so the runner is stateless
    between CLI invocations.
    """

    def __init__(self, python: str = "python") -> None:
        self.python = python
        self._processes: dict[str, subprocess.Popen] = {}

    @staticmethod
    def _encode_run_id(work_dir: Path) -> str:
        """Encode work_dir path as run_id."""
        return f"local:{work_dir}"

    @staticmethod
    def _decode_run_id(run_id: str) -> Path:
        """Decode run_id back to work_dir path."""
        if run_id.startswith("local:"):
            return Path(run_id[6:])
        # Legacy format: local_{pid}
        raise ValueError(f"Cannot decode run_id: {run_id}")

    def launch(self, experiment: Experiment, work_dir: Path) -> str:
        """Launch the experiment script as a subprocess.

        If the experiment config has an ``eval_script``, a wrapper script
        is generated that runs the training script first, then the eval script.
        """
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = work_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Build command
        if experiment.config.eval_script:
            # Multi-step: train then eval
            cmd = self._build_pipeline_cmd(experiment, work_dir)
        else:
            cmd = build_launch_cmd(experiment.config, python=self.python)

        # Environment
        env = os.environ.copy()
        env.update(build_env_vars(experiment.config))

        # Launch
        stdout_log = open(logs_dir / "stdout.log", "w")
        stderr_log = open(logs_dir / "stderr.log", "w")

        proc = subprocess.Popen(
            cmd,
            stdout=stdout_log,
            stderr=stderr_log,
            env=env,
            cwd=str(work_dir),
        )

        run_id = self._encode_run_id(work_dir)

        # Save PID and status
        (work_dir / "pid").write_text(str(proc.pid))
        (work_dir / "status.json").write_text(
            json.dumps({"state": "running", "pid": proc.pid})
        )

        # Save config for results adapter detection
        (work_dir / "config.json").write_text(
            experiment.config.model_dump_json()
        )

        self._processes[run_id] = proc

        logger.info(f"Launched {run_id} (PID {proc.pid}): {' '.join(cmd)}")
        return run_id

    def _build_pipeline_cmd(self, experiment: Experiment, work_dir: Path) -> list[str]:
        """Build a bash wrapper that runs train then eval."""
        train_cmd_parts = build_launch_cmd(experiment.config, python=self.python)

        eval_config = experiment.config.model_copy(update={
            "script": experiment.config.eval_script,
            "script_args": experiment.config.eval_script_args,
            "launcher": "python",  # eval always runs on single GPU
        })
        eval_cmd_parts = [self.python, eval_config.script]
        for key, value in eval_config.script_args.items():
            eval_cmd_parts.append(f"--{key}")
            eval_cmd_parts.append(str(value))

        train_cmd = " ".join(str(p) for p in train_cmd_parts)
        eval_cmd = " ".join(str(p) for p in eval_cmd_parts)

        # Write pipeline script
        pipeline_script = work_dir / "_daedalus_pipeline.sh"
        pipeline_script.write_text(
            f"#!/bin/bash\nset -e\n"
            f"echo '[daedalus] Step 1: Training'\n"
            f"{train_cmd}\n"
            f"echo '[daedalus] Step 2: Evaluation'\n"
            f"{eval_cmd}\n"
        )
        pipeline_script.chmod(0o755)

        return ["bash", str(pipeline_script)]

    def _get_pid(self, work_dir: Path) -> int | None:
        """Read PID from work directory."""
        pid_file = work_dir / "pid"
        if pid_file.exists():
            try:
                return int(pid_file.read_text().strip())
            except (ValueError, OSError):
                return None
        return None

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def poll(self, run_id: str) -> RunStatus:
        """Check process status."""
        try:
            work_dir = self._decode_run_id(run_id)
        except ValueError:
            return RunStatus(state="failed", error=f"Invalid run_id: {run_id}")

        if not work_dir.exists():
            return RunStatus(state="failed", error=f"Work dir not found: {work_dir}")

        # Check in-memory process first (same session)
        proc = self._processes.get(run_id)
        if proc is not None:
            retcode = proc.poll()
            if retcode is None:
                progress = self._read_progress(work_dir)
                return RunStatus(state="running", progress=progress)
            if retcode == 0:
                return RunStatus(state="completed")
            error = self._read_stderr(work_dir)
            return RunStatus(state="failed", error=error or f"Exit code {retcode}")

        # Cross-session: check results first (process may be zombie but finished)
        results_file = work_dir / "results.json"
        if results_file.exists():
            return RunStatus(state="completed")

        # No results yet — check if process is still running
        pid = self._get_pid(work_dir)
        if pid is None:
            return RunStatus(state="failed", error="No PID file found")

        if self._is_process_alive(pid):
            progress = self._read_progress(work_dir)
            return RunStatus(state="running", progress=progress)

        # Process dead, no results
        error = self._read_stderr(work_dir)
        return RunStatus(state="failed", error=error or "Process exited without results")

    def _read_progress(self, work_dir: Path) -> str | None:
        """Read progress from status.json."""
        status_file = work_dir / "status.json"
        if status_file.exists():
            try:
                data = json.loads(status_file.read_text())
                return data.get("progress")
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _read_stderr(self, work_dir: Path) -> str | None:
        """Read last lines of stderr."""
        stderr_path = work_dir / "logs" / "stderr.log"
        if stderr_path.exists():
            lines = stderr_path.read_text().strip().splitlines()
            return "\n".join(lines[-10:]) if lines else None
        return None

    def fetch_results(self, run_id: str, adapter: str = "auto") -> dict[str, Any]:
        """Read results from the work directory using auto-detection.

        Supports multiple result formats:
        - results.json (Daedalus native or flat JSON)
        - eval_results.json / trainer_state.json (HuggingFace Trainer)
        - results.csv (CSV with metric,value columns)

        Args:
            run_id: The run identifier.
            adapter: Results adapter ('auto', 'daedalus', 'hf_trainer', 'csv', 'json_flat').
        """
        work_dir = self._decode_run_id(run_id)

        from ..evaluators.results_adapter import read_results
        return read_results(work_dir, adapter=adapter)

    def cancel(self, run_id: str) -> None:
        """Send SIGTERM to the process, SIGKILL after 10s."""
        work_dir = self._decode_run_id(run_id)

        # Try in-memory process first
        proc = self._processes.get(run_id)
        if proc is not None:
            logger.info(f"Cancelling {run_id} (PID {proc.pid})")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return

        # Cross-session: use PID file
        pid = self._get_pid(work_dir)
        if pid is None:
            logger.warning(f"Cannot cancel {run_id}: no PID file")
            return

        logger.info(f"Cancelling {run_id} (PID {pid})")
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def logs(self, run_id: str, tail: int = 50) -> str:
        """Return last N lines from both stdout and stderr.

        Most ML frameworks (HF Trainer, tqdm) write to stderr, so we
        merge both streams for a complete picture.
        """
        try:
            work_dir = self._decode_run_id(run_id)
        except ValueError:
            return f"Invalid run_id: {run_id}"

        parts = []
        for name in ("stdout.log", "stderr.log"):
            log_path = work_dir / "logs" / name
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                if lines:
                    parts.append(f"=== {name} ===")
                    parts.extend(lines[-tail:])

        return "\n".join(parts) if parts else "No logs yet."
