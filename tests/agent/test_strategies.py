"""Tests for exploration strategies."""

import pytest

from daedalus.agent.strategies import suggest_next_params
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.scripts_registry import ScriptsRegistry


def _make_exp(exp_id, script_args, results, status=ExperimentStatus.COMPLETED):
    return Experiment(
        id=exp_id,
        hypothesis=Hypothesis(statement="test", rationale="test"),
        config=ExperimentConfig(script="train.py", script_args=script_args),
        status=status,
        results=results,
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
                "epochs": {"type": "int", "default": 3},
            },
        },
    })


class TestSuggestNextParams:
    def test_no_experiments_suggests_baseline(self):
        registry = _make_registry()
        suggestions = suggest_next_params(
            experiments=[],
            registry=registry,
            target_metric="eval.accuracy",
        )
        assert len(suggestions) >= 1
        assert suggestions[0]["strategy"] == "baseline"

    def test_single_experiment_suggests_neighbors(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"lr": 1e-5}, {"eval": {"accuracy": 0.85}}),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.accuracy",
            focus_param="lr",
        )
        # Should suggest exploring neighbors of 1e-5
        assert any(s["strategy"] == "explore_neighbor" for s in suggestions)

    def test_two_experiments_bisects(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"lr": 1e-5}, {"eval": {"accuracy": 0.85}}),
            _make_exp("exp_2", {"lr": 1e-4}, {"eval": {"accuracy": 0.80}}),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.accuracy",
            higher_is_better=True,
            focus_param="lr",
        )
        # Best is exp_1 (0.85). Should suggest bisecting between 1e-5 and 1e-4
        bisect_suggestions = [s for s in suggestions if s["strategy"] == "bisect"]
        assert len(bisect_suggestions) >= 1

    def test_monotonic_trend_extrapolates(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"lr": 1e-5}, {"eval": {"accuracy": 0.80}}),
            _make_exp("exp_2", {"lr": 2e-5}, {"eval": {"accuracy": 0.85}}),
            _make_exp("exp_3", {"lr": 3e-5}, {"eval": {"accuracy": 0.90}}),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.accuracy",
            higher_is_better=True,
            focus_param="lr",
        )
        trend = [s for s in suggestions if s["strategy"] == "trend_extrapolate"]
        assert len(trend) >= 1
        # Should suggest 4e-5
        assert trend[0]["value"] == pytest.approx(4e-5, rel=0.1)

    def test_categorical_suggests_untried(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"reward_type": "simple"}, {"eval": {"accuracy": 0.85}}),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.accuracy",
            focus_param="reward_type",
        )
        untried = [s for s in suggestions if s["strategy"] == "untried_choice"]
        assert len(untried) >= 1
        assert untried[0]["value"] in ("hybrid", "fiscore")

    def test_ignores_incomplete_experiments(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"lr": 1e-5}, None, status=ExperimentStatus.RUNNING),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.accuracy",
        )
        # Should suggest baseline since no completed experiments
        assert suggestions[0]["strategy"] == "baseline"

    def test_focus_param_filters(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"lr": 1e-5, "epochs": 3}, {"eval": {"accuracy": 0.85}}),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.accuracy",
            focus_param="lr",
        )
        # All suggestions should be about lr
        for s in suggestions:
            if "parameter" in s:
                assert s["parameter"] == "lr"

    def test_unexplored_param_suggested(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"lr": 1e-5}, {"eval": {"accuracy": 0.85}}),
        ]
        # Don't focus on any param — should suggest trying untouched params too
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.accuracy",
        )
        # reward_type has never been varied, should suggest it
        untried = [s for s in suggestions if s.get("parameter") == "reward_type"]
        assert len(untried) >= 1

    def test_dot_notation_metric(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"lr": 1e-5}, {"simpleqa": {"correct": 3.5}}),
            _make_exp("exp_2", {"lr": 2e-5}, {"simpleqa": {"correct": 4.0}}),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="simpleqa.correct",
            higher_is_better=True,
            focus_param="lr",
        )
        assert len(suggestions) >= 1

    def test_max_5_suggestions(self):
        registry = _make_registry()
        experiments = [
            _make_exp(f"exp_{i}", {"lr": i * 1e-5}, {"eval": {"accuracy": 0.5 + i * 0.05}})
            for i in range(1, 10)
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.accuracy",
        )
        assert len(suggestions) <= 5
