"""Daedalus watcher — poll, tail logs, and stream events.

The watcher is intentionally DUMB: it polls running experiments,
tails their logs, detects patterns, and emits structured events.
It does NOT make AI decisions.

The brain is Claude Code, which:
- Has Daedalus MCP tools (experiments, data, suggestions)
- Has code tools (read, write, bash, grep)
- Can fix bugs, modify scripts, analyze results, design experiments
- Gets notified by the watcher when things happen

Event types emitted:
- WATCHING    — started monitoring an experiment
- PROGRESS    — periodic status update (epoch, step, loss)
- METRIC      — parsed metric from training log
- ALERT       — anomaly detected (NaN, OOM, loss spike, stall)
- LOG         — raw log lines (when --verbose)
- COMPLETED   — experiment finished successfully
- FAILED      — experiment failed
- POLL_ERROR  — error while polling

Usage:
    # Stream events as JSONL (for Claude Code):
    daedalus watch --exp-id exp_007 --stream

    # Watch all, human-readable output:
    daedalus watch --all --verbose

    # Quick status check:
    daedalus watch --exp-id exp_007
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

from .core.experiment import ExperimentStatus
from .core.ledger import Ledger
from .runners.factory import create_runner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log Monitor — tails log files and detects patterns
# ---------------------------------------------------------------------------

@dataclass
class LogEvent:
    """A parsed event from training logs."""

    type: str  # metric, alert, progress
    key: str
    value: float | str
    raw_line: str = ""


# Common training log patterns
_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # HF Trainer dict format: {'loss': 0.345, 'learning_rate': 1e-5, 'epoch': 1.5}
    ("hf_log", re.compile(
        r"\{['\"]loss['\"]:\s*([\d.e+-]+).*?['\"]epoch['\"]:\s*([\d.]+)"
    ), "metric"),
    # HF eval: {'eval_loss': 0.287, 'eval_accuracy': 0.92, ...}
    ("hf_eval", re.compile(
        r"\{['\"]eval_loss['\"]:\s*([\d.e+-]+)"
    ), "metric"),
    # Generic key=value: loss=0.345 or loss: 0.345
    # NOTE: eval_loss must come BEFORE loss to avoid false match
    ("kv_eval_loss", re.compile(
        r"eval[_/]loss[\s=:]+(\d+\.?\d*(?:e[+-]?\d+)?)", re.IGNORECASE
    ), "metric"),
    ("kv_loss", re.compile(
        r"(?:train[_/])?loss[\s=:]+(\d+\.?\d*(?:e[+-]?\d+)?)", re.IGNORECASE
    ), "metric"),
    ("kv_lr", re.compile(
        r"(?:learning_rate|lr)[\s=:]+(\d+\.?\d*(?:e[+-]?\d+)?)", re.IGNORECASE
    ), "metric"),
    ("kv_acc", re.compile(
        r"(?:eval[_/])?(?:accuracy|acc)[\s=:]+(\d+\.?\d*(?:e[+-]?\d+)?)", re.IGNORECASE
    ), "metric"),
    # Epoch progress: Epoch 3/10, epoch: 3
    ("epoch", re.compile(
        r"[Ee]poch[\s:]+(\d+)(?:/(\d+))?"
    ), "progress"),
    # Step progress: Step 500/5000, step: 500
    ("step", re.compile(
        r"[Ss]tep[\s:]+(\d+)(?:/(\d+))?"
    ), "progress"),
    # NaN / Inf detection
    ("nan", re.compile(
        r"\b(?:nan|NaN|NAN)\b"
    ), "alert"),
    ("inf", re.compile(
        r"(?:loss|grad).*\b(?:inf|Inf|INF)\b"
    ), "alert"),
    # OOM detection
    ("oom", re.compile(
        r"(?:CUDA out of memory|OutOfMemoryError|OOM|torch\.cuda\.OutOfMemoryError)",
        re.IGNORECASE,
    ), "alert"),
    # Generic error
    ("error", re.compile(
        r"(?:RuntimeError|ValueError|KeyError|FileNotFoundError|ModuleNotFoundError):\s*(.+)"
    ), "alert"),
]


class LogMonitor:
    """Tails a log file and extracts structured events.

    Tracks file position between reads so it only processes new lines.
    Detects loss spikes and training stalls.
    """

    def __init__(
        self,
        log_path: Path,
        loss_spike_threshold: float = 5.0,
        stall_timeout: int = 1800,
    ) -> None:
        self.log_path = Path(log_path)
        self._file_pos: int = 0
        self._loss_spike_threshold = loss_spike_threshold
        self._stall_timeout = stall_timeout
        self._last_losses: list[float] = []
        self._last_event_time: float = time.time()
        self._last_epoch: str | None = None
        self._last_step: str | None = None

    def read_new_lines(self) -> list[str]:
        """Read new lines since last check."""
        if not self.log_path.exists():
            return []

        try:
            with open(self.log_path, "r") as f:
                f.seek(self._file_pos)
                new_content = f.read()
                self._file_pos = f.tell()
        except OSError:
            return []

        if not new_content:
            return []

        return new_content.splitlines()

    def parse_lines(self, lines: list[str]) -> list[LogEvent]:
        """Parse log lines and extract structured events."""
        events: list[LogEvent] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for name, pattern, event_type in _PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue

                if event_type == "metric":
                    if name == "hf_log":
                        loss = float(match.group(1))
                        epoch = float(match.group(2))
                        events.append(LogEvent("metric", "loss", loss, line))
                        events.append(LogEvent("progress", "epoch", epoch, line))
                        self._track_loss(loss, events)
                    elif name == "hf_eval":
                        events.append(LogEvent("metric", "eval_loss", float(match.group(1)), line))
                    elif name in ("kv_loss",):
                        loss = float(match.group(1))
                        events.append(LogEvent("metric", "loss", loss, line))
                        self._track_loss(loss, events)
                    elif name == "kv_lr":
                        events.append(LogEvent("metric", "learning_rate", float(match.group(1)), line))
                    elif name == "kv_eval_loss":
                        events.append(LogEvent("metric", "eval_loss", float(match.group(1)), line))
                    elif name == "kv_acc":
                        events.append(LogEvent("metric", "accuracy", float(match.group(1)), line))

                elif event_type == "progress":
                    if name == "epoch":
                        current = match.group(1)
                        total = match.group(2) or "?"
                        epoch_str = f"{current}/{total}"
                        if epoch_str != self._last_epoch:
                            self._last_epoch = epoch_str
                            events.append(LogEvent("progress", "epoch", epoch_str, line))
                    elif name == "step":
                        current = match.group(1)
                        total = match.group(2) or "?"
                        step_str = f"{current}/{total}"
                        if step_str != self._last_step:
                            self._last_step = step_str
                            events.append(LogEvent("progress", "step", step_str, line))

                elif event_type == "alert":
                    detail = match.group(1) if match.lastindex else name.upper()
                    events.append(LogEvent("alert", name, detail, line))

                self._last_event_time = time.time()
                break  # Only match first pattern per line

        return events

    def _track_loss(self, loss: float, events: list[LogEvent]) -> None:
        """Track loss values and detect spikes."""
        self._last_losses.append(loss)

        # Keep last 10 values
        if len(self._last_losses) > 10:
            self._last_losses = self._last_losses[-10:]

        # Detect spike (requires at least 3 data points)
        if len(self._last_losses) >= 3:
            recent_avg = sum(self._last_losses[-3:]) / 3
            prev_avg = sum(self._last_losses[:-1]) / (len(self._last_losses) - 1)
            if prev_avg > 0 and recent_avg / prev_avg > self._loss_spike_threshold:
                events.append(LogEvent(
                    "alert", "loss_spike",
                    f"Loss jumped {recent_avg/prev_avg:.1f}x (avg {prev_avg:.4f} → {recent_avg:.4f})",
                ))

    def check_stall(self) -> LogEvent | None:
        """Check if training appears stalled (no log output)."""
        elapsed = time.time() - self._last_event_time
        if elapsed > self._stall_timeout:
            self._last_event_time = time.time()  # Reset to avoid repeated alerts
            return LogEvent(
                "alert", "stall",
                f"No log output for {elapsed/60:.0f} minutes",
            )
        return None

    def tail(self) -> list[LogEvent]:
        """Read new lines and parse them. Also checks for stalls."""
        lines = self.read_new_lines()
        events = self.parse_lines(lines)

        stall = self.check_stall()
        if stall:
            events.append(stall)

        return events


# ---------------------------------------------------------------------------
# Experiment Watcher — polls status + monitors logs
# ---------------------------------------------------------------------------

class ExperimentWatcher:
    """Lightweight experiment monitor. Polls, tails logs, emits events.

    No AI, no Anthropic API. Just polling + log parsing + notification.
    The AI decisions are made by Claude Code using MCP tools.
    """

    def __init__(
        self,
        project_path: Path,
        poll_interval: int = 300,
        on_event: Callable[[dict], None] | None = None,
        stream: bool = False,
        stream_output: TextIO | None = None,
        verbose: bool = False,
        log_poll_interval: int = 10,
        stall_timeout: int = 1800,
    ) -> None:
        self.project_path = Path(project_path)
        self.poll_interval = poll_interval
        self.log_poll_interval = log_poll_interval
        self.stall_timeout = stall_timeout
        self.stream = stream
        self.verbose = verbose
        self._stream_out = stream_output or sys.stdout
        self._on_event = on_event or self._default_event_handler
        self.ledger = Ledger(self.project_path / "ledger")
        self._log_monitors: dict[str, LogMonitor] = {}
        self._remote_event_cursors: dict[str, int] = {}  # exp_id -> last line read

    def _emit(self, event: dict) -> None:
        """Emit an event through all channels."""
        logger.info(f"[{event.get('event')}] {event.get('detail', '')}")
        self._on_event(event)

        if self.stream:
            self._stream_out.write(json.dumps(event, default=str) + "\n")
            self._stream_out.flush()

    def _default_event_handler(self, event: dict) -> None:
        """Pretty-print events to stderr for human consumption."""
        if not self.verbose and not self.stream:
            return

        if self.stream:
            return  # JSONL goes to stdout, don't duplicate

        ts = time.strftime("%H:%M:%S")
        etype = event.get("event", "?")
        exp_id = event.get("exp_id", "")
        detail = event.get("detail", "")

        # Color-code by event type
        if etype == "COMPLETED":
            print(f"[{ts}] \033[32m[{etype}]\033[0m {exp_id} {detail}", file=sys.stderr)
        elif etype == "FAILED":
            print(f"[{ts}] \033[31m[{etype}]\033[0m {exp_id} {detail}", file=sys.stderr)
        elif etype == "ALERT":
            print(f"[{ts}] \033[33m[{etype}]\033[0m {exp_id} {detail}", file=sys.stderr)
        elif etype == "METRIC":
            print(f"[{ts}] \033[36m[{etype}]\033[0m {exp_id} {detail}", file=sys.stderr)
        else:
            print(f"[{ts}] [{etype}] {exp_id} {detail}", file=sys.stderr)

    def _get_log_path(self, exp_id: str, run_id: str) -> Path | None:
        """Resolve log path for an experiment."""
        if run_id.startswith("local:"):
            work_dir = Path(run_id[6:])
            stdout_log = work_dir / "logs" / "stdout.log"
            if stdout_log.exists():
                return stdout_log
            # Also check stderr for error detection
            stderr_log = work_dir / "logs" / "stderr.log"
            if stderr_log.exists():
                return stderr_log
        return None

    def _get_log_monitor(self, exp_id: str, run_id: str) -> LogMonitor | None:
        """Get or create a LogMonitor for an experiment."""
        if exp_id in self._log_monitors:
            return self._log_monitors[exp_id]

        log_path = self._get_log_path(exp_id, run_id)
        if log_path is None:
            return None

        monitor = LogMonitor(
            log_path,
            stall_timeout=self.stall_timeout,
        )
        self._log_monitors[exp_id] = monitor
        return monitor

    def _fetch_remote_events(self, exp_id: str, run_id: str) -> None:
        """Fetch and forward events from remote sentinel's events.jsonl."""
        if not run_id.startswith("ssh:"):
            return

        try:
            from .runners.ssh import SSHRunner, SSHConfig
            runner = create_runner(self.project_path)
            if not hasattr(runner, "fetch_events"):
                return

            cursor = self._remote_event_cursors.get(exp_id, 0)
            events = runner.fetch_events(run_id, since_line=cursor)

            for event in events:
                cursor += 1
                etype = event.get("event", "")

                # Forward sentinel events, adding exp_id
                if etype in ("METRIC", "PROGRESS", "ALERT"):
                    event["exp_id"] = exp_id
                    if "detail" not in event:
                        # Build detail string
                        if etype == "METRIC":
                            event["detail"] = f"{event.get('metric', '')}={event.get('value', '')}"
                        elif etype == "PROGRESS":
                            event["detail"] = f"{event.get('key', '')}: {event.get('value', '')}"
                        elif etype == "ALERT":
                            event["detail"] = event.get("detail", "")
                    self._emit(event)

            self._remote_event_cursors[exp_id] = cursor
        except Exception as e:
            logger.debug(f"Failed to fetch remote events for {exp_id}: {e}")

    def _monitor_experiment(self, exp_id: str, run_id: str) -> None:
        """Tail logs (local) or fetch events (remote) for an experiment."""
        if run_id.startswith("ssh:"):
            self._fetch_remote_events(exp_id, run_id)
        else:
            monitor = self._get_log_monitor(exp_id, run_id)
            if monitor:
                log_events = monitor.tail()
                self._process_log_events(exp_id, log_events)

    def _process_log_events(self, exp_id: str, log_events: list[LogEvent]) -> None:
        """Convert LogEvents to watcher events and emit them."""
        for le in log_events:
            if le.type == "metric":
                self._emit({
                    "event": "METRIC",
                    "exp_id": exp_id,
                    "metric": le.key,
                    "value": le.value,
                    "detail": f"{le.key}={le.value}",
                })
            elif le.type == "progress":
                self._emit({
                    "event": "PROGRESS",
                    "exp_id": exp_id,
                    "key": le.key,
                    "value": le.value,
                    "detail": f"{le.key}: {le.value}",
                })
            elif le.type == "alert":
                self._emit({
                    "event": "ALERT",
                    "exp_id": exp_id,
                    "alert_type": le.key,
                    "detail": str(le.value),
                    "raw_line": le.raw_line,
                })

    def wait_for(self, exp_id: str) -> dict:
        """Block until a specific experiment completes or fails.

        While waiting, tails logs and emits METRIC/PROGRESS/ALERT events.
        Returns a dict with status and results (if any).
        """
        self._emit({"event": "WATCHING", "exp_id": exp_id, "detail": "Waiting for completion..."})

        while True:
            self.ledger = Ledger(self.project_path / "ledger")
            exp = self.ledger.get(exp_id)

            if not exp:
                return {"error": f"Experiment {exp_id} not found"}

            if not exp.run_id:
                return {"error": f"Experiment {exp_id} has no run_id"}

            if exp.status in (ExperimentStatus.COMPLETED, ExperimentStatus.ANALYZED):
                self._emit({"event": "COMPLETED", "exp_id": exp_id, "detail": "Already completed"})
                return {
                    "exp_id": exp_id,
                    "status": "completed",
                    "results": exp.results,
                }

            if exp.status in (ExperimentStatus.FAILED, ExperimentStatus.ABANDONED):
                self._emit({"event": "FAILED", "exp_id": exp_id, "detail": exp.status.value})
                return {
                    "exp_id": exp_id,
                    "status": exp.status.value,
                    "notes": exp.notes,
                }

            # Still running — poll the runner and tail logs
            try:
                runner = create_runner(self.project_path)
                run_status = runner.poll(exp.run_id)

                if run_status.state == "completed":
                    try:
                        results = runner.fetch_results(exp.run_id)
                        exp = exp.transition(ExperimentStatus.COMPLETED)
                        exp = exp.model_copy(update={"results": results})
                        self.ledger.update(exp)
                        self._emit({
                            "event": "COMPLETED",
                            "exp_id": exp_id,
                            "detail": "Results recorded",
                            "results": results,
                        })
                        return {
                            "exp_id": exp_id,
                            "status": "completed",
                            "results": results,
                        }
                    except Exception as e:
                        self._emit({"event": "ALERT", "exp_id": exp_id,
                                    "alert_type": "fetch_error", "detail": str(e)})
                        exp = exp.transition(ExperimentStatus.COMPLETED)
                        self.ledger.update(exp)
                        return {
                            "exp_id": exp_id,
                            "status": "completed",
                            "fetch_error": str(e),
                        }

                elif run_status.state == "failed":
                    exp = exp.transition(ExperimentStatus.FAILED)
                    error = run_status.error or "unknown"
                    exp = exp.model_copy(update={
                        "notes": exp.notes + f"\nFailed: {error}"
                    })
                    self.ledger.update(exp)
                    self._emit({"event": "FAILED", "exp_id": exp_id, "detail": error})
                    return {
                        "exp_id": exp_id,
                        "status": "failed",
                        "error": error,
                    }

                # Still running — monitor logs/events
                self._monitor_experiment(exp_id, exp.run_id)

                if run_status.progress:
                    self._emit({
                        "event": "PROGRESS",
                        "exp_id": exp_id,
                        "detail": run_status.progress,
                    })

            except Exception as e:
                self._emit({"event": "POLL_ERROR", "exp_id": exp_id, "detail": str(e)})

            # Inner loop: monitor more frequently than status poll
            if self.log_poll_interval < self.poll_interval:
                remaining = self.poll_interval
                while remaining > 0:
                    sleep_time = min(self.log_poll_interval, remaining)
                    time.sleep(sleep_time)
                    remaining -= sleep_time

                    if remaining > 0:
                        self._monitor_experiment(exp_id, exp.run_id)
            else:
                time.sleep(self.poll_interval)

    def watch_all(self, wait_all: bool = False) -> list[dict]:
        """Watch all running experiments.

        Args:
            wait_all: If True, wait until ALL running experiments finish.
                      If False, return as soon as ANY one finishes.

        Returns:
            List of completion dicts.
        """
        self._emit({"event": "START", "detail": "Watching all running experiments..."})
        completed_results: list[dict] = []

        while True:
            self.ledger = Ledger(self.project_path / "ledger")
            running = self.ledger.by_status(ExperimentStatus.RUNNING)

            if not running:
                if completed_results:
                    break
                self._emit({"event": "NO_RUNNING", "detail": "No running experiments found."})
                return []

            runner = create_runner(self.project_path)

            for exp in running:
                if not exp.run_id:
                    continue

                # Monitor logs/events
                self._monitor_experiment(exp.id, exp.run_id)

                run_status = runner.poll(exp.run_id)

                if run_status.state == "completed":
                    try:
                        results = runner.fetch_results(exp.run_id)
                        exp = exp.transition(ExperimentStatus.COMPLETED)
                        exp = exp.model_copy(update={"results": results})
                        self.ledger.update(exp)
                        self._emit({
                            "event": "COMPLETED",
                            "exp_id": exp.id,
                            "detail": "Results recorded",
                            "results": results,
                        })
                        completed_results.append({
                            "exp_id": exp.id,
                            "status": "completed",
                            "results": results,
                        })
                    except Exception as e:
                        exp = exp.transition(ExperimentStatus.COMPLETED)
                        self.ledger.update(exp)
                        completed_results.append({
                            "exp_id": exp.id,
                            "status": "completed",
                            "fetch_error": str(e),
                        })

                    if not wait_all:
                        return completed_results

                elif run_status.state == "failed":
                    error = run_status.error or "unknown"
                    exp = exp.transition(ExperimentStatus.FAILED)
                    exp = exp.model_copy(update={
                        "notes": exp.notes + f"\nFailed: {error}"
                    })
                    self.ledger.update(exp)
                    self._emit({"event": "FAILED", "exp_id": exp.id, "detail": error})
                    completed_results.append({
                        "exp_id": exp.id,
                        "status": "failed",
                        "error": error,
                    })

                    if not wait_all:
                        return completed_results

            # Check if all done
            self.ledger = Ledger(self.project_path / "ledger")
            still_running = self.ledger.by_status(ExperimentStatus.RUNNING)
            if not still_running:
                break

            # Inner loop: tail logs between status polls
            if self.log_poll_interval < self.poll_interval:
                remaining = self.poll_interval
                while remaining > 0:
                    sleep_time = min(self.log_poll_interval, remaining)
                    time.sleep(sleep_time)
                    remaining -= sleep_time

                    if remaining > 0:
                        for exp in running:
                            if not exp.run_id:
                                continue
                            self._monitor_experiment(exp.id, exp.run_id)
            else:
                time.sleep(self.poll_interval)

        return completed_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for standalone watcher."""
    import argparse

    parser = argparse.ArgumentParser(description="Watch Daedalus experiments")
    parser.add_argument("--project", "-p", required=True, help="Project path")
    parser.add_argument("--exp-id", type=str, default=None, help="Watch specific experiment")
    parser.add_argument("--all", action="store_true", help="Wait for ALL experiments")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Seconds between status polls (default: 300)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Seconds between log tails (default: 10)")
    parser.add_argument("--stream", action="store_true",
                        help="Output JSONL events to stdout (for Claude Code)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print human-readable events to stderr")
    parser.add_argument("--stall-timeout", type=int, default=1800,
                        help="Alert if no log output for N seconds (default: 1800)")
    args = parser.parse_args()

    watcher = ExperimentWatcher(
        Path(args.project),
        poll_interval=args.poll_interval,
        log_poll_interval=args.log_interval,
        stream=args.stream,
        verbose=args.verbose,
        stall_timeout=args.stall_timeout,
    )

    if args.exp_id:
        result = watcher.wait_for(args.exp_id)
    else:
        result = watcher.watch_all(wait_all=args.all)

    # Final output as JSON (always, even in stream mode)
    if not args.stream:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
