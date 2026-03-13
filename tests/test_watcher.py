"""Tests for experiment watcher and log monitor."""

import json
import sys
import time
import io
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.ledger import Ledger
from daedalus.runners.local import LocalRunner
from daedalus.watcher import ExperimentWatcher, LogMonitor, LogEvent


def _setup_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    (project / "program.md").write_text("# Test\n")
    return project


# ---------------------------------------------------------------------------
# LogMonitor tests
# ---------------------------------------------------------------------------


class TestLogMonitor:
    def test_parse_hf_trainer_log(self, tmp_path):
        """Parses HuggingFace Trainer dict format."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text(
            "{'loss': 0.345, 'learning_rate': 1e-05, 'epoch': 1.5}\n"
            "{'loss': 0.312, 'learning_rate': 9e-06, 'epoch': 2.0}\n"
        )

        monitor = LogMonitor(log_file)
        events = monitor.tail()

        metric_events = [e for e in events if e.type == "metric"]
        progress_events = [e for e in events if e.type == "progress"]

        assert len(metric_events) >= 2  # at least 2 loss values
        assert any(e.key == "loss" and e.value == 0.345 for e in metric_events)
        assert any(e.key == "loss" and e.value == 0.312 for e in metric_events)
        assert len(progress_events) >= 1

    def test_parse_kv_format(self, tmp_path):
        """Parses key=value and key: value log formats."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text(
            "loss=0.456\n"
            "eval_loss: 0.289\n"
            "accuracy: 0.92\n"
            "lr=3e-5\n"
        )

        monitor = LogMonitor(log_file)
        events = monitor.tail()

        keys = {e.key for e in events if e.type == "metric"}
        assert "loss" in keys
        assert "eval_loss" in keys
        assert "accuracy" in keys
        assert "learning_rate" in keys

    def test_parse_epoch_progress(self, tmp_path):
        """Parses epoch and step progress."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text(
            "Epoch 1/10\n"
            "Step 500/5000\n"
            "Epoch 2/10\n"
        )

        monitor = LogMonitor(log_file)
        events = monitor.tail()

        progress = [e for e in events if e.type == "progress"]
        assert len(progress) == 3
        assert any(e.key == "epoch" and e.value == "1/10" for e in progress)
        assert any(e.key == "step" and e.value == "500/5000" for e in progress)
        assert any(e.key == "epoch" and e.value == "2/10" for e in progress)

    def test_detect_nan(self, tmp_path):
        """Detects NaN in training output."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text("loss: nan\n")

        monitor = LogMonitor(log_file)
        events = monitor.tail()

        alerts = [e for e in events if e.type == "alert"]
        assert len(alerts) >= 1
        assert any(e.key == "nan" for e in alerts)

    def test_detect_oom(self, tmp_path):
        """Detects CUDA out of memory errors."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text(
            "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        )

        monitor = LogMonitor(log_file)
        events = monitor.tail()

        alerts = [e for e in events if e.type == "alert"]
        assert len(alerts) >= 1
        assert any(e.key == "oom" for e in alerts)

    def test_detect_loss_spike(self, tmp_path):
        """Detects sudden loss spike."""
        log_file = tmp_path / "stdout.log"
        # Normal training then sudden spike
        log_file.write_text(
            "loss=0.5\n"
            "loss=0.45\n"
            "loss=0.43\n"
            "loss=0.40\n"
            "loss=50.0\n"  # 100x spike
        )

        monitor = LogMonitor(log_file, loss_spike_threshold=3.0)
        events = monitor.tail()

        alerts = [e for e in events if e.type == "alert" and e.key == "loss_spike"]
        assert len(alerts) >= 1

    def test_incremental_reads(self, tmp_path):
        """Only reads new lines on subsequent calls."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text("loss=0.5\n")

        monitor = LogMonitor(log_file)
        events1 = monitor.tail()
        assert len([e for e in events1 if e.type == "metric"]) == 1

        # Append more data
        with open(log_file, "a") as f:
            f.write("loss=0.4\n")

        events2 = monitor.tail()
        assert len([e for e in events2 if e.type == "metric"]) == 1

    def test_stall_detection(self, tmp_path):
        """Detects training stall when no output for too long."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text("loss=0.5\n")

        monitor = LogMonitor(log_file, stall_timeout=1)
        monitor.tail()  # Read initial content

        # Simulate time passing
        monitor._last_event_time = time.time() - 2

        stall = monitor.check_stall()
        assert stall is not None
        assert stall.key == "stall"

    def test_nonexistent_log_file(self, tmp_path):
        """Handles missing log file gracefully."""
        monitor = LogMonitor(tmp_path / "nonexistent.log")
        events = monitor.tail()
        assert events == [] or all(e.type == "alert" and e.key == "stall" for e in events)

    def test_dedup_epoch_events(self, tmp_path):
        """Doesn't emit duplicate epoch events."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text(
            "Epoch 1/10\n"
            "loss=0.5\n"
            "Epoch 1/10\n"  # Same epoch again
            "loss=0.4\n"
            "Epoch 2/10\n"  # New epoch
        )

        monitor = LogMonitor(log_file)
        events = monitor.tail()

        epoch_events = [e for e in events if e.type == "progress" and e.key == "epoch"]
        assert len(epoch_events) == 2  # Only 1/10 and 2/10, no duplicate

    def test_runtime_error_detection(self, tmp_path):
        """Detects generic runtime errors."""
        log_file = tmp_path / "stdout.log"
        log_file.write_text(
            "ValueError: invalid literal for int()\n"
        )

        monitor = LogMonitor(log_file)
        events = monitor.tail()

        alerts = [e for e in events if e.type == "alert" and e.key == "error"]
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# ExperimentWatcher tests
# ---------------------------------------------------------------------------


class TestWaitFor:
    def test_wait_for_completed_experiment(self, tmp_path):
        """wait_for returns immediately if experiment is already completed."""
        project = _setup_project(tmp_path)
        ledger = Ledger(project / "ledger")

        exp = Experiment(
            id="done_exp",
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="train.py"),
            status=ExperimentStatus.COMPLETED,
            results={"eval": {"acc": 0.9}},
            run_id="local:fake",
        )
        ledger.append(exp)

        watcher = ExperimentWatcher(project, poll_interval=1)
        result = watcher.wait_for("done_exp")
        assert result["status"] == "completed"
        assert result["results"]["eval"]["acc"] == 0.9

    def test_wait_for_running_experiment(self, tmp_path):
        """wait_for polls until a running experiment completes."""
        project = _setup_project(tmp_path)
        runner = LocalRunner(python=sys.executable)

        exp = Experiment(
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="run.py"),
            status=ExperimentStatus.RUNNING,
        )

        work_dir = project / "runs" / exp.id
        work_dir.mkdir(parents=True)
        (work_dir / "run.py").write_text(
            "import json, time; time.sleep(0.5); "
            "json.dump({'eval': {'acc': 0.92}}, open('results.json', 'w'))"
        )

        run_id = runner.launch(exp, work_dir)
        exp = exp.model_copy(update={"run_id": run_id})

        ledger = Ledger(project / "ledger")
        ledger.append(exp)

        watcher = ExperimentWatcher(project, poll_interval=1, log_poll_interval=1)
        result = watcher.wait_for(exp.id)

        assert result["status"] == "completed"
        assert result["results"]["eval"]["acc"] == 0.92

    def test_wait_for_failed_experiment(self, tmp_path):
        """wait_for detects failed experiments."""
        project = _setup_project(tmp_path)
        ledger = Ledger(project / "ledger")

        exp = Experiment(
            id="failed_exp",
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="train.py"),
            status=ExperimentStatus.FAILED,
            run_id="local:fake",
            notes="Error: OOM",
        )
        ledger.append(exp)

        watcher = ExperimentWatcher(project, poll_interval=1)
        result = watcher.wait_for("failed_exp")
        assert result["status"] == "failed"

    def test_wait_for_nonexistent(self, tmp_path):
        project = _setup_project(tmp_path)
        watcher = ExperimentWatcher(project, poll_interval=1)
        result = watcher.wait_for("nonexistent")
        assert "error" in result

    def test_wait_for_stream_mode(self, tmp_path):
        """Stream mode outputs JSONL events to stdout."""
        project = _setup_project(tmp_path)
        ledger = Ledger(project / "ledger")

        exp = Experiment(
            id="stream_exp",
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="train.py"),
            status=ExperimentStatus.COMPLETED,
            results={"eval": {"acc": 0.95}},
            run_id="local:fake",
        )
        ledger.append(exp)

        stream_buf = io.StringIO()
        watcher = ExperimentWatcher(
            project, poll_interval=1,
            stream=True, stream_output=stream_buf,
        )
        result = watcher.wait_for("stream_exp")

        assert result["status"] == "completed"

        # Check JSONL output
        stream_buf.seek(0)
        events = [json.loads(line) for line in stream_buf if line.strip()]
        assert len(events) >= 1
        event_types = {e["event"] for e in events}
        assert "WATCHING" in event_types or "COMPLETED" in event_types

    def test_wait_for_with_log_monitoring(self, tmp_path):
        """wait_for tails logs and emits metric events."""
        project = _setup_project(tmp_path)
        runner = LocalRunner(python=sys.executable)

        exp = Experiment(
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="run.py"),
            status=ExperimentStatus.RUNNING,
        )

        work_dir = project / "runs" / exp.id
        work_dir.mkdir(parents=True)
        # Script that writes training-like logs before producing results
        (work_dir / "run.py").write_text(
            "import json, time, sys\n"
            "print('Epoch 1/2', flush=True)\n"
            "print('loss=0.5', flush=True)\n"
            "time.sleep(0.3)\n"
            "print('Epoch 2/2', flush=True)\n"
            "print('loss=0.3', flush=True)\n"
            "json.dump({'eval': {'acc': 0.88}}, open('results.json', 'w'))\n"
        )

        run_id = runner.launch(exp, work_dir)
        exp = exp.model_copy(update={"run_id": run_id})

        ledger = Ledger(project / "ledger")
        ledger.append(exp)

        collected_events: list[dict] = []
        def capture_event(event: dict) -> None:
            collected_events.append(event)

        watcher = ExperimentWatcher(
            project, poll_interval=1, log_poll_interval=1,
            on_event=capture_event,
        )
        result = watcher.wait_for(exp.id)

        assert result["status"] == "completed"

        # Should have captured some metric/progress events from logs
        event_types = {e.get("event") for e in collected_events}
        assert "COMPLETED" in event_types


# ---------------------------------------------------------------------------
# WatchAll tests
# ---------------------------------------------------------------------------


class TestWatchAll:
    def test_watch_all_no_running(self, tmp_path):
        project = _setup_project(tmp_path)
        watcher = ExperimentWatcher(project, poll_interval=1)
        result = watcher.watch_all()
        assert result == []

    def test_watch_all_detects_completion(self, tmp_path):
        project = _setup_project(tmp_path)
        runner = LocalRunner(python=sys.executable)

        exp = Experiment(
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="run.py"),
            status=ExperimentStatus.RUNNING,
        )

        work_dir = project / "runs" / exp.id
        work_dir.mkdir(parents=True)
        (work_dir / "run.py").write_text(
            "import json; json.dump({'eval': {'acc': 0.88}}, open('results.json', 'w'))"
        )

        run_id = runner.launch(exp, work_dir)
        exp = exp.model_copy(update={"run_id": run_id})

        ledger = Ledger(project / "ledger")
        ledger.append(exp)

        # Wait for script to finish
        for _ in range(40):
            if runner.poll(run_id).state != "running":
                break
            time.sleep(0.25)

        watcher = ExperimentWatcher(project, poll_interval=1)
        results = watcher.watch_all()

        assert len(results) == 1
        assert results[0]["status"] == "completed"
        assert results[0]["results"]["eval"]["acc"] == 0.88


# ---------------------------------------------------------------------------
# MCP Server tests (moved to tests/test_mcp_server.py)
# ---------------------------------------------------------------------------
# MCP protocol tests now use the official MCP SDK and subprocess-based
# integration tests. See tests/test_mcp_server.py for full coverage.
