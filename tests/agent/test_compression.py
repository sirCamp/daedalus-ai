"""Tests for smart context compression."""

import pytest

from daedalus.agent.context import ContextBuilder
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus, Reflection
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.ledger import Ledger


def _setup_project(tmp_path):
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    (project / "program.md").write_text("# Test\n")
    return project


def _add_experiments(project, count, start_id=1, results=None, reflection=None):
    ledger = Ledger(project / "ledger")
    for i in range(start_id, start_id + count):
        exp = Experiment(
            id=f"exp_{i:03d}",
            hypothesis=Hypothesis(
                statement=f"Hypothesis {i}",
                rationale="Test",
            ),
            config=ExperimentConfig(
                script="train.py",
                script_args={"lr": i * 1e-6, "batch_size": 8},
            ),
            status=ExperimentStatus.ANALYZED,
            results=results or {"eval": {"accuracy": 0.5 + i * 0.01}},
            reflection=reflection,
        )
        ledger.append(exp)


class TestContextCompression:
    def test_few_experiments_no_compression(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiments(project, 5)

        builder = ContextBuilder(project)
        summary = builder._format_experiment_summary()

        # All 5 should be detailed (no "Summary of")
        assert "Summary of" not in summary
        assert "exp_001" in summary
        assert "exp_005" in summary

    def test_many_experiments_triggers_compression(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiments(project, 30)

        builder = ContextBuilder(project)
        summary = builder._format_experiment_summary(compression_threshold=20)

        # Old ones compressed, recent ones detailed
        assert "Summary of 10 earlier experiments" in summary
        assert "Recent 20 experiments" in summary

    def test_compressed_summary_includes_status(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiments(project, 25)

        builder = ContextBuilder(project)
        summary = builder._format_experiment_summary(compression_threshold=20)

        assert "analyzed:" in summary

    def test_compressed_summary_includes_param_ranges(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiments(project, 25)

        builder = ContextBuilder(project)
        summary = builder._format_experiment_summary(compression_threshold=20)

        assert "Parameter ranges explored" in summary
        assert "lr" in summary

    def test_compressed_summary_with_reflections(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiments(
            project, 25,
            reflection=Reflection(
                hypothesis_confirmed="rejected",
                analysis="Did not work because XYZ",
            ),
        )

        builder = ContextBuilder(project)
        summary = builder._format_experiment_summary(compression_threshold=20)

        assert "Hypothesis outcomes" in summary
        assert "Dead ends" in summary

    def test_limit_overrides_compression(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiments(project, 30)

        builder = ContextBuilder(project)
        summary = builder._format_experiment_summary(limit=10)

        # With limit, no compression
        assert "Summary of" not in summary

    def test_compression_threshold_customizable(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiments(project, 15)

        builder = ContextBuilder(project)
        # threshold=10 → 15 > 10 → compress
        summary = builder._format_experiment_summary(compression_threshold=10)
        assert "Summary of 5 earlier experiments" in summary

    def test_empty_experiments(self, tmp_path):
        project = _setup_project(tmp_path)
        builder = ContextBuilder(project)
        summary = builder._format_experiment_summary()
        assert "No experiments recorded" in summary
