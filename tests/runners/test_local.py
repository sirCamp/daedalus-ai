"""Tests for the local runner."""

import json
import sys
import time

import pytest

from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment
from daedalus.core.hypothesis import Hypothesis
from daedalus.runners.local import LocalRunner


def _make_experiment(script: str = "train.py", **script_args) -> Experiment:
    return Experiment(
        hypothesis=Hypothesis(statement="test", rationale="test"),
        config=ExperimentConfig(script=script, script_args=script_args),
    )


class TestLocalRunner:
    def test_launch_and_complete(self, tmp_path):
        """Launch a dummy script that writes results.json."""
        # Write a dummy training script
        script = tmp_path / "train.py"
        script.write_text(
            'import json, pathlib\n'
            'pathlib.Path("results.json").write_text('
            'json.dumps({"eval": {"metric": 42.0}}))\n'
        )

        runner = LocalRunner(python=sys.executable)
        exp = _make_experiment(script=str(script))

        work_dir = tmp_path / "work"
        run_id = runner.launch(exp, work_dir)

        assert run_id.startswith("local:")
        assert (work_dir / "pid").exists()

        # Wait for completion
        for _ in range(50):
            status = runner.poll(run_id)
            if status.state != "running":
                break
            time.sleep(0.1)

        assert status.state == "completed"

        results = runner.fetch_results(run_id)
        assert results["eval"]["metric"] == 42.0

    def test_launch_failing_script(self, tmp_path):
        """Script that exits with error."""
        script = tmp_path / "bad.py"
        script.write_text('raise RuntimeError("boom")\n')

        runner = LocalRunner(python=sys.executable)
        exp = _make_experiment(script=str(script))

        work_dir = tmp_path / "work"
        run_id = runner.launch(exp, work_dir)

        for _ in range(50):
            status = runner.poll(run_id)
            if status.state != "running":
                break
            time.sleep(0.1)

        assert status.state == "failed"
        assert status.error is not None

    def test_cancel(self, tmp_path):
        """Launch a long script and cancel it."""
        script = tmp_path / "slow.py"
        script.write_text('import time; time.sleep(60)\n')

        runner = LocalRunner(python=sys.executable)
        exp = _make_experiment(script=str(script))

        work_dir = tmp_path / "work"
        run_id = runner.launch(exp, work_dir)

        # Verify running
        time.sleep(0.2)
        status = runner.poll(run_id)
        assert status.state == "running"

        # Cancel
        runner.cancel(run_id)
        time.sleep(0.5)
        status = runner.poll(run_id)
        assert status.state == "failed"

    def test_logs(self, tmp_path):
        """Verify log capture."""
        script = tmp_path / "log.py"
        script.write_text('print("hello from training")\n')

        runner = LocalRunner(python=sys.executable)
        exp = _make_experiment(script=str(script))

        work_dir = tmp_path / "work"
        run_id = runner.launch(exp, work_dir)

        for _ in range(50):
            status = runner.poll(run_id)
            if status.state != "running":
                break
            time.sleep(0.1)

        log_output = runner.logs(run_id)
        assert "hello from training" in log_output

    def test_unknown_run_id(self):
        runner = LocalRunner()
        status = runner.poll("nonexistent")
        assert status.state == "failed"
