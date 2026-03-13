"""Tests for insights generator."""

import pytest

from daedalus.agent.insights import InsightsGenerator
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus, Reflection
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.scripts_registry import ScriptsRegistry


def _make_exp(
    exp_id, script_args, results,
    status=ExperimentStatus.COMPLETED,
    reflection=None,
    config_diff=None,
):
    return Experiment(
        id=exp_id,
        hypothesis=Hypothesis(statement=f"H for {exp_id}", rationale="test"),
        config=ExperimentConfig(script="train.py", script_args=script_args),
        status=status,
        results=results,
        reflection=reflection,
        config_diff=config_diff,
    )


def _make_registry():
    return ScriptsRegistry({
        "train": {
            "path": "train.py",
            "parameters": {
                "lr": {"type": "float", "default": 1e-5, "range": [1e-6, 1e-3]},
                "reward_type": {
                    "type": "str",
                    "default": "simple",
                    "choices": ["simple", "hybrid", "fiscore"],
                },
            },
        },
    })


class TestInsightsGenerator:
    def test_no_completed_experiments(self):
        gen = InsightsGenerator(experiments=[], target_metric="eval.acc")
        result = gen.generate()
        assert "No completed experiments" in result

    def test_top_configs_ranked(self):
        experiments = [
            _make_exp("e1", {"lr": 1e-5}, {"eval": {"acc": 0.80}}),
            _make_exp("e2", {"lr": 2e-5}, {"eval": {"acc": 0.90}}),
            _make_exp("e3", {"lr": 3e-5}, {"eval": {"acc": 0.85}}),
        ]
        gen = InsightsGenerator(
            experiments=experiments,
            target_metric="eval.acc",
            higher_is_better=True,
        )
        result = gen.generate()
        assert "Top" in result
        # e2 should be ranked first
        e2_pos = result.index("e2")
        e3_pos = result.index("e3")
        e1_pos = result.index("e1")
        assert e2_pos < e3_pos < e1_pos

    def test_top_configs_lower_is_better(self):
        experiments = [
            _make_exp("e1", {"lr": 1e-5}, {"eval": {"loss": 0.50}}),
            _make_exp("e2", {"lr": 2e-5}, {"eval": {"loss": 0.30}}),
        ]
        gen = InsightsGenerator(
            experiments=experiments,
            target_metric="eval.loss",
            higher_is_better=False,
        )
        result = gen.generate()
        e2_pos = result.index("e2")
        e1_pos = result.index("e1")
        assert e2_pos < e1_pos

    def test_parameter_sensitivity(self):
        experiments = [
            _make_exp("e1", {"lr": 1e-5}, {"eval": {"acc": 0.50}}),
            _make_exp("e2", {"lr": 5e-5}, {"eval": {"acc": 0.90}}),
            _make_exp("e3", {"lr": 1e-4}, {"eval": {"acc": 0.70}}),
        ]
        gen = InsightsGenerator(
            experiments=experiments,
            target_metric="eval.acc",
        )
        result = gen.generate()
        assert "Parameter Sensitivity" in result
        assert "lr" in result

    def test_dead_ends_rejected(self):
        experiments = [
            _make_exp(
                "e1", {"lr": 1e-5}, {"eval": {"acc": 0.80}},
                reflection=Reflection(
                    hypothesis_confirmed="rejected",
                    analysis="This approach did not work at all.",
                ),
                status=ExperimentStatus.ANALYZED,
            ),
        ]
        gen = InsightsGenerator(experiments=experiments, target_metric="eval.acc")
        result = gen.generate()
        assert "Dead Ends" in result
        assert "Rejected hypotheses" in result

    def test_dead_ends_failed(self):
        experiments = [
            _make_exp("e0", {"lr": 2e-5}, {"eval": {"acc": 0.80}}),
            _make_exp(
                "e1", {"lr": 1e-5}, None,
                status=ExperimentStatus.FAILED,
            ),
        ]
        gen = InsightsGenerator(experiments=experiments, target_metric="eval.acc")
        result = gen.generate()
        assert "Failed experiments" in result

    def test_dead_ends_none(self):
        experiments = [
            _make_exp(
                "e1", {"lr": 1e-5}, {"eval": {"acc": 0.80}},
                reflection=Reflection(
                    hypothesis_confirmed="confirmed",
                    analysis="Worked well.",
                ),
                status=ExperimentStatus.ANALYZED,
            ),
        ]
        gen = InsightsGenerator(experiments=experiments, target_metric="eval.acc")
        result = gen.generate()
        assert "None found" in result

    def test_explored_ranges(self):
        registry = _make_registry()
        experiments = [
            _make_exp("e1", {"lr": 1e-5}, {"eval": {"acc": 0.80}}),
            _make_exp("e2", {"lr": 5e-5}, {"eval": {"acc": 0.85}}),
            _make_exp("e3", {"lr": 1e-4}, {"eval": {"acc": 0.82}}),
        ]
        gen = InsightsGenerator(
            experiments=experiments,
            registry=registry,
            target_metric="eval.acc",
        )
        result = gen.generate()
        assert "Explored Ranges" in result
        assert "lr" in result

    def test_explored_ranges_with_coverage(self):
        registry = _make_registry()
        experiments = [
            _make_exp("e1", {"lr": 1e-6}, {"eval": {"acc": 0.70}}),
            _make_exp("e2", {"lr": 1e-3}, {"eval": {"acc": 0.90}}),
        ]
        gen = InsightsGenerator(
            experiments=experiments,
            registry=registry,
            target_metric="eval.acc",
        )
        result = gen.generate()
        assert "100%" in result  # full range coverage

    def test_save_to_file(self, tmp_path):
        experiments = [
            _make_exp("e1", {"lr": 1e-5}, {"eval": {"acc": 0.80}}),
        ]
        gen = InsightsGenerator(
            experiments=experiments,
            target_metric="eval.acc",
        )
        output_path = tmp_path / "insights.md"
        gen.save(output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert "Research Insights" in content

    def test_no_target_metric_minimal_output(self):
        experiments = [
            _make_exp("e1", {"lr": 1e-5}, {"eval": {"acc": 0.80}}),
        ]
        gen = InsightsGenerator(experiments=experiments, target_metric="")
        result = gen.generate()
        # Should still include basic info but no rankings
        assert "Research Insights" in result
        assert "Top" not in result

    def test_filters_non_completed(self):
        experiments = [
            _make_exp("e1", {"lr": 1e-5}, None, status=ExperimentStatus.RUNNING),
            _make_exp("e2", {"lr": 2e-5}, {"eval": {"acc": 0.85}}),
        ]
        gen = InsightsGenerator(
            experiments=experiments,
            target_metric="eval.acc",
        )
        assert len(gen.completed) == 1
        assert gen.completed[0].id == "e2"

    def test_config_diff_in_dead_ends(self):
        experiments = [
            _make_exp(
                "e1", {"lr": 1e-5}, {"eval": {"acc": 0.50}},
                reflection=Reflection(
                    hypothesis_confirmed="rejected",
                    analysis="Failed due to low lr.",
                ),
                status=ExperimentStatus.ANALYZED,
                config_diff={"lr": {"from": 1e-4, "to": 1e-5}},
            ),
        ]
        gen = InsightsGenerator(experiments=experiments, target_metric="eval.acc")
        result = gen.generate()
        assert "1e-05" in result or "1e-5" in result
