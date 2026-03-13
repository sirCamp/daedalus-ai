"""Tests for convergence detection in strategies."""

import pytest

from daedalus.agent.strategies import _check_convergence, suggest_next_params
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.scripts_registry import ParameterSpec, ScriptsRegistry


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
                "epochs": {"type": "int", "default": 3, "range": [1, 10]},
            },
        },
    })


class TestCheckConvergence:
    def test_too_few_points_returns_none(self):
        param = ParameterSpec(type="float", default=1e-5, range=[1e-6, 1e-3])
        scored = {1e-5: 0.85, 2e-5: 0.87}
        result = _check_convergence("lr", param, scored, higher_is_better=True)
        assert result is None

    def test_plateau_detected(self):
        param = ParameterSpec(type="float", default=1e-5, range=[1e-6, 1e-3])
        # All values very close — plateau
        scored = {1e-5: 0.850, 2e-5: 0.851, 3e-5: 0.851, 4e-5: 0.852}
        result = _check_convergence("lr", param, scored, higher_is_better=True)
        assert result is not None
        assert result["converged"] is True
        assert result["reason"] == "plateau"

    def test_no_plateau_when_improving(self):
        param = ParameterSpec(type="float", default=1e-5, range=[1e-6, 1e-3])
        # Clear improvement
        scored = {1e-5: 0.70, 2e-5: 0.80, 3e-5: 0.90}
        result = _check_convergence("lr", param, scored, higher_is_better=True)
        # Should NOT detect plateau — big improvements
        if result is not None:
            assert result["reason"] != "plateau"

    def test_categorical_exhaustion(self):
        param = ParameterSpec(
            type="str", default="simple",
            choices=["simple", "hybrid", "fiscore"],
        )
        scored = {"simple": 0.85, "hybrid": 0.80, "fiscore": 0.88}
        result = _check_convergence("reward_type", param, scored, higher_is_better=True)
        assert result is not None
        assert result["converged"] is True
        assert result["reason"] == "all_choices_tried"
        assert result["best_value"] == "fiscore"

    def test_categorical_not_exhausted(self):
        param = ParameterSpec(
            type="str", default="simple",
            choices=["simple", "hybrid", "fiscore"],
        )
        scored = {"simple": 0.85, "hybrid": 0.80}
        result = _check_convergence("reward_type", param, scored, higher_is_better=True)
        assert result is None

    def test_range_saturation(self):
        param = ParameterSpec(type="float", default=5.0, range=[0.0, 10.0])
        # Covered 90% of range, best is interior (not boundary)
        scored = {0.5: 0.70, 3.0: 0.85, 5.0: 0.90, 7.0: 0.88, 9.5: 0.75}
        result = _check_convergence("lr", param, scored, higher_is_better=True)
        assert result is not None
        assert result["converged"] is True
        assert result["reason"] == "range_saturated"

    def test_range_not_saturated_at_boundary(self):
        param = ParameterSpec(type="float", default=5.0, range=[0.0, 10.0])
        # Best value clearly at boundary — might improve further
        scored = {0.5: 0.70, 3.0: 0.85, 5.0: 0.88, 9.8: 0.95}
        result = _check_convergence("lr", param, scored, higher_is_better=True)
        # Best is at boundary (9.8 within 2% of 10.0), should NOT converge as range_saturated
        if result is not None:
            assert result["reason"] != "range_saturated"

    def test_lower_is_better_plateau(self):
        param = ParameterSpec(type="float", default=1e-5, range=[1e-6, 1e-3])
        # Values barely change — clear plateau for lower-is-better
        scored = {1e-5: 0.050, 2e-5: 0.0499, 3e-5: 0.0498, 4e-5: 0.0498}
        result = _check_convergence("lr", param, scored, higher_is_better=False)
        assert result is not None
        assert result["converged"] is True

    def test_non_numeric_type_returns_none(self):
        param = ParameterSpec(type="str", default="test")
        scored = {"a": 0.85, "b": 0.80, "c": 0.88}
        result = _check_convergence("p", param, scored, higher_is_better=True)
        assert result is None


class TestSuggestNextWithConvergence:
    def test_converged_params_in_output(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"reward_type": "simple"}, {"eval": {"acc": 0.85}}),
            _make_exp("exp_2", {"reward_type": "hybrid"}, {"eval": {"acc": 0.80}}),
            _make_exp("exp_3", {"reward_type": "fiscore"}, {"eval": {"acc": 0.88}}),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.acc",
            focus_param="reward_type",
        )
        converged = [s for s in suggestions if s.get("converged")]
        assert len(converged) >= 1
        assert converged[0]["parameter"] == "reward_type"

    def test_converged_param_excluded_from_suggestions(self):
        registry = _make_registry()
        experiments = [
            _make_exp("exp_1", {"reward_type": "simple"}, {"eval": {"acc": 0.85}}),
            _make_exp("exp_2", {"reward_type": "hybrid"}, {"eval": {"acc": 0.80}}),
            _make_exp("exp_3", {"reward_type": "fiscore"}, {"eval": {"acc": 0.88}}),
        ]
        suggestions = suggest_next_params(
            experiments=experiments,
            registry=registry,
            target_metric="eval.acc",
            focus_param="reward_type",
        )
        active = [s for s in suggestions if not s.get("converged")]
        for s in active:
            if "parameter" in s:
                assert s["parameter"] != "reward_type"
