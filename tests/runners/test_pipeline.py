"""Tests for multi-step pipeline and results adapter integration."""

import json
import sys
import time
import pytest

from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment
from daedalus.core.hypothesis import Hypothesis
from daedalus.runners.local import LocalRunner


class TestPipelineConfig:
    def test_eval_script_in_config(self):
        """ExperimentConfig supports eval_script."""
        config = ExperimentConfig(
            script="train_bert.py",
            script_args={"model": "bert-base-uncased", "epochs": 3},
            eval_script="evaluate.py",
            eval_script_args={"split": "test"},
            results_adapter="hf_trainer",
        )
        assert config.eval_script == "evaluate.py"
        assert config.eval_script_args == {"split": "test"}
        assert config.results_adapter == "hf_trainer"

    def test_config_diff_with_eval_script(self):
        """config_diff detects eval_script changes."""
        from daedalus.core.config import config_diff

        old = ExperimentConfig(script="train.py")
        new = ExperimentConfig(
            script="train.py",
            eval_script="eval.py",
            eval_script_args={"split": "test"},
        )
        diff = config_diff(old, new)
        assert "eval_script" in diff

    def test_pipeline_script_generated(self, tmp_path):
        """Multi-step pipeline generates a bash wrapper."""
        runner = LocalRunner(python=sys.executable)

        exp = Experiment(
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(
                script="train.py",
                script_args={"lr": "1e-5"},
                eval_script="eval.py",
                eval_script_args={"split": "test"},
            ),
        )

        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # Write dummy scripts
        (work_dir / "train.py").write_text("print('training')")
        (work_dir / "eval.py").write_text(
            "import json; json.dump({'accuracy': 0.92}, open('results.json', 'w'))"
        )

        run_id = runner.launch(exp, work_dir)

        # Wait for completion
        for _ in range(40):
            status = runner.poll(run_id)
            if status.state != "running":
                break
            time.sleep(0.25)

        assert status.state == "completed"
        results = runner.fetch_results(run_id)
        assert results["eval"]["accuracy"] == 0.92

    def test_single_script_still_works(self, tmp_path):
        """Single script (no eval_script) still works as before."""
        runner = LocalRunner(python=sys.executable)

        exp = Experiment(
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="run.py"),
        )

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "run.py").write_text(
            "import json; json.dump({'eval': {'acc': 0.9}}, open('results.json', 'w'))"
        )

        run_id = runner.launch(exp, work_dir)
        for _ in range(40):
            status = runner.poll(run_id)
            if status.state != "running":
                break
            time.sleep(0.25)

        assert status.state == "completed"
        results = runner.fetch_results(run_id)
        assert results["eval"]["acc"] == 0.9


class TestLocalRunnerResultsAdapter:
    def test_fetch_hf_trainer_results(self, tmp_path):
        """LocalRunner.fetch_results reads HF Trainer format."""
        runner = LocalRunner(python=sys.executable)

        exp = Experiment(
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="train.py"),
        )

        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # Write HF Trainer output instead of results.json
        (work_dir / "train.py").write_text(
            "import json\n"
            "json.dump({'eval_accuracy': 0.92, 'eval_f1': 0.89, 'eval_runtime': 5.0}, "
            "open('eval_results.json', 'w'))"
        )

        run_id = runner.launch(exp, work_dir)
        for _ in range(40):
            status = runner.poll(run_id)
            if status.state != "running":
                break
            time.sleep(0.25)

        assert status.state == "completed"
        results = runner.fetch_results(run_id)
        assert results["eval"]["accuracy"] == 0.92

    def test_fetch_csv_results(self, tmp_path):
        """LocalRunner.fetch_results reads CSV format."""
        runner = LocalRunner(python=sys.executable)

        exp = Experiment(
            hypothesis=Hypothesis(statement="test", rationale="test"),
            config=ExperimentConfig(script="train.py"),
        )

        work_dir = tmp_path / "work"
        work_dir.mkdir()

        (work_dir / "train.py").write_text(
            "with open('results.csv', 'w') as f:\n"
            "    f.write('metric,value\\n')\n"
            "    f.write('accuracy,0.92\\n')\n"
            "    f.write('f1,0.89\\n')\n"
        )

        run_id = runner.launch(exp, work_dir)
        for _ in range(40):
            status = runner.poll(run_id)
            if status.state != "running":
                break
            time.sleep(0.25)

        assert status.state == "completed"
        results = runner.fetch_results(run_id)
        assert results["eval"]["accuracy"] == 0.92
