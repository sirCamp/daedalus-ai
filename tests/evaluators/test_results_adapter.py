"""Tests for results adapter."""

import json
import pytest

from daedalus.evaluators.results_adapter import read_results


class TestResultsAdapter:
    def test_daedalus_format(self, tmp_path):
        """Standard Daedalus format: {eval_name: {metric: value}}."""
        (tmp_path / "results.json").write_text(json.dumps({
            "simpleqa": {"correct": 3.5, "not_attempted": 41.0},
            "popqa": {"em": 10.54, "f1": 0.35},
        }))

        results = read_results(tmp_path)
        assert "simpleqa" in results
        assert results["simpleqa"]["correct"] == 3.5
        assert results["popqa"]["em"] == 10.54

    def test_json_flat_format(self, tmp_path):
        """Flat JSON: {metric: value} — common for simple scripts."""
        (tmp_path / "results.json").write_text(json.dumps({
            "accuracy": 0.92,
            "f1": 0.89,
            "loss": 0.34,
        }))

        results = read_results(tmp_path, adapter="json_flat")
        assert "eval" in results
        assert results["eval"]["accuracy"] == 0.92
        assert results["eval"]["f1"] == 0.89

    def test_json_flat_auto_detect(self, tmp_path):
        """Auto-detect flat JSON when values are numbers."""
        (tmp_path / "results.json").write_text(json.dumps({
            "accuracy": 0.92,
            "f1": 0.89,
        }))

        results = read_results(tmp_path, adapter="auto")
        assert "eval" in results
        assert results["eval"]["accuracy"] == 0.92

    def test_hf_trainer_eval_results(self, tmp_path):
        """HuggingFace Trainer eval_results.json."""
        (tmp_path / "eval_results.json").write_text(json.dumps({
            "eval_loss": 0.34,
            "eval_accuracy": 0.92,
            "eval_f1": 0.89,
            "eval_runtime": 12.5,
            "eval_samples_per_second": 100.0,
        }))

        results = read_results(tmp_path)
        assert "eval" in results
        # eval_ prefix should be stripped
        assert results["eval"]["accuracy"] == 0.92
        assert results["eval"]["f1"] == 0.89
        assert results["eval"]["loss"] == 0.34

    def test_hf_trainer_state(self, tmp_path):
        """HuggingFace trainer_state.json with log_history."""
        (tmp_path / "trainer_state.json").write_text(json.dumps({
            "log_history": [
                {"loss": 1.2, "epoch": 1, "step": 100},
                {"loss": 0.8, "epoch": 2, "step": 200},
                {"eval_loss": 0.5, "eval_accuracy": 0.85, "epoch": 2, "step": 200},
                {"loss": 0.4, "epoch": 3, "step": 300},
                {"eval_loss": 0.34, "eval_accuracy": 0.92, "epoch": 3, "step": 300},
            ]
        }))

        results = read_results(tmp_path)
        assert "eval" in results
        assert results["eval"]["accuracy"] == 0.92  # last eval entry
        assert "train" in results
        assert results["train"]["loss"] == 0.4  # last train entry

    def test_csv_format(self, tmp_path):
        """CSV with metric,value columns."""
        (tmp_path / "results.csv").write_text(
            "metric,value\n"
            "accuracy,0.92\n"
            "f1,0.89\n"
            "precision,0.91\n"
        )

        results = read_results(tmp_path, adapter="csv")
        assert results["eval"]["accuracy"] == 0.92
        assert results["eval"]["f1"] == 0.89

    def test_csv_auto_detect(self, tmp_path):
        """Auto-detect CSV when only results.csv exists."""
        (tmp_path / "results.csv").write_text(
            "metric,value\n"
            "accuracy,0.92\n"
        )

        results = read_results(tmp_path)
        assert results["eval"]["accuracy"] == 0.92

    def test_no_results_raises(self, tmp_path):
        """No results files should raise."""
        with pytest.raises(FileNotFoundError, match="No recognized"):
            read_results(tmp_path)

    def test_unknown_adapter_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown results adapter"):
            read_results(tmp_path, adapter="nonexistent")

    def test_hf_trainer_in_checkpoint_subdir(self, tmp_path):
        """HF Trainer sometimes puts eval_results in checkpoint dirs."""
        checkpoint = tmp_path / "checkpoint-1000"
        checkpoint.mkdir()
        (checkpoint / "eval_results.json").write_text(json.dumps({
            "eval_accuracy": 0.88,
        }))

        results = read_results(tmp_path)
        assert results["eval"]["accuracy"] == 0.88

    def test_custom_eval_name(self, tmp_path):
        """Flat results with custom eval name."""
        (tmp_path / "results.json").write_text(json.dumps({
            "accuracy": 0.92,
        }))

        results = read_results(tmp_path, adapter="json_flat", eval_name="test_set")
        assert "test_set" in results

    def test_daedalus_ignores_non_numeric(self, tmp_path):
        """Daedalus format should skip non-numeric values."""
        (tmp_path / "results.json").write_text(json.dumps({
            "eval": {"accuracy": 0.92, "model_name": "bert-base"},
        }))

        results = read_results(tmp_path)
        assert "accuracy" in results["eval"]
        assert "model_name" not in results["eval"]
