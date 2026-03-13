"""Tests for Experiment lifecycle and state transitions."""

import pytest

from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus
from daedalus.core.hypothesis import Hypothesis


def _make_experiment(**kwargs) -> Experiment:
    defaults = dict(
        hypothesis=Hypothesis(statement="test", rationale="test"),
        config=ExperimentConfig(script="train.py"),
    )
    defaults.update(kwargs)
    return Experiment(**defaults)


class TestExperimentTransitions:
    @pytest.mark.parametrize(
        "from_status,to_status",
        [
            (ExperimentStatus.DRAFT, ExperimentStatus.QUEUED),
            (ExperimentStatus.DRAFT, ExperimentStatus.ABANDONED),
            (ExperimentStatus.QUEUED, ExperimentStatus.RUNNING),
            (ExperimentStatus.QUEUED, ExperimentStatus.FAILED),
            (ExperimentStatus.QUEUED, ExperimentStatus.ABANDONED),
            (ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED),
            (ExperimentStatus.RUNNING, ExperimentStatus.FAILED),
            (ExperimentStatus.RUNNING, ExperimentStatus.ABANDONED),
            (ExperimentStatus.COMPLETED, ExperimentStatus.ANALYZED),
            (ExperimentStatus.COMPLETED, ExperimentStatus.ABANDONED),
        ],
    )
    def test_valid_transitions(self, from_status, to_status):
        exp = _make_experiment(status=from_status)
        new_exp = exp.transition(to_status)
        assert new_exp.status == to_status
        assert new_exp.updated_at >= exp.updated_at

    @pytest.mark.parametrize(
        "from_status,to_status",
        [
            (ExperimentStatus.DRAFT, ExperimentStatus.RUNNING),
            (ExperimentStatus.DRAFT, ExperimentStatus.COMPLETED),
            (ExperimentStatus.QUEUED, ExperimentStatus.COMPLETED),
            (ExperimentStatus.RUNNING, ExperimentStatus.QUEUED),
            (ExperimentStatus.COMPLETED, ExperimentStatus.RUNNING),
            (ExperimentStatus.ANALYZED, ExperimentStatus.RUNNING),
            (ExperimentStatus.FAILED, ExperimentStatus.RUNNING),
            (ExperimentStatus.ABANDONED, ExperimentStatus.RUNNING),
        ],
    )
    def test_invalid_transitions(self, from_status, to_status):
        exp = _make_experiment(status=from_status)
        with pytest.raises(ValueError, match="Cannot transition"):
            exp.transition(to_status)


class TestExperimentProperties:
    def test_is_terminal(self):
        for status in [ExperimentStatus.ANALYZED, ExperimentStatus.FAILED, ExperimentStatus.ABANDONED]:
            exp = _make_experiment(status=status)
            assert exp.is_terminal is True

    def test_not_terminal(self):
        for status in [ExperimentStatus.DRAFT, ExperimentStatus.QUEUED, ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED]:
            exp = _make_experiment(status=status)
            assert exp.is_terminal is False

    def test_serialization_roundtrip(self):
        exp = _make_experiment(
            tags=["grpo", "calibration"],
            notes="First run with new dataset",
        )
        dumped = exp.model_dump_json()
        restored = Experiment.model_validate_json(dumped)
        assert restored.id == exp.id
        assert restored.tags == exp.tags
        assert restored.hypothesis.statement == "test"
