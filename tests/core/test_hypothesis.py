"""Tests for Hypothesis and Prediction evaluation."""

import pytest

from daedalus.core.hypothesis import Hypothesis, Prediction


class TestPrediction:
    def test_increase_confirmed(self):
        pred = Prediction(metric="accuracy", direction="increase")
        assert pred.evaluate(actual=0.85, baseline=0.80) is True

    def test_increase_rejected(self):
        pred = Prediction(metric="accuracy", direction="increase")
        assert pred.evaluate(actual=0.75, baseline=0.80) is False

    def test_decrease_confirmed(self):
        pred = Prediction(metric="not_attempted", direction="decrease")
        assert pred.evaluate(actual=0.30, baseline=0.50) is True

    def test_stable_confirmed(self):
        pred = Prediction(metric="similarity", direction="stable", tolerance=0.05)
        assert pred.evaluate(actual=0.462, baseline=0.460) is True

    def test_stable_rejected(self):
        pred = Prediction(metric="similarity", direction="stable", tolerance=0.05)
        assert pred.evaluate(actual=0.50, baseline=0.40) is False

    def test_expected_value_confirmed(self):
        pred = Prediction(
            metric="not_attempted",
            direction="decrease",
            expected_value=30.0,
            tolerance=0.15,
        )
        assert pred.evaluate(actual=32.0) is True  # within 15%

    def test_expected_value_rejected(self):
        pred = Prediction(
            metric="not_attempted",
            direction="decrease",
            expected_value=30.0,
            tolerance=0.05,
        )
        assert pred.evaluate(actual=40.0) is False

    def test_no_baseline_no_expected(self):
        pred = Prediction(metric="accuracy", direction="increase")
        assert pred.evaluate(actual=0.85) is False


class TestHypothesis:
    def test_all_confirmed(self):
        hyp = Hypothesis(
            statement="penalty reduces not_attempted",
            rationale="EV analysis",
            predictions=[
                Prediction(metric="not_attempted", direction="decrease"),
                Prediction(metric="similarity", direction="stable", tolerance=0.1),
            ],
        )
        results = {"not_attempted": 30.0, "similarity": 0.46}
        baseline = {"not_attempted": 50.0, "similarity": 0.46}
        evaluated = hyp.evaluate(results, baseline)
        assert evaluated.status == "confirmed"

    def test_partial(self):
        hyp = Hypothesis(
            statement="test",
            rationale="test",
            predictions=[
                Prediction(metric="accuracy", direction="increase"),
                Prediction(metric="loss", direction="decrease"),
            ],
        )
        results = {"accuracy": 0.9, "loss": 0.6}
        baseline = {"accuracy": 0.8, "loss": 0.5}
        evaluated = hyp.evaluate(results, baseline)
        assert evaluated.status == "partial"

    def test_rejected(self):
        hyp = Hypothesis(
            statement="test",
            rationale="test",
            predictions=[
                Prediction(metric="accuracy", direction="increase"),
            ],
        )
        results = {"accuracy": 0.7}
        baseline = {"accuracy": 0.8}
        evaluated = hyp.evaluate(results, baseline)
        assert evaluated.status == "rejected"

    def test_no_predictions_stays_pending(self):
        hyp = Hypothesis(statement="test", rationale="test")
        evaluated = hyp.evaluate({"accuracy": 0.9})
        assert evaluated.status == "pending"

    def test_missing_metric_counts_as_failed(self):
        hyp = Hypothesis(
            statement="test",
            rationale="test",
            predictions=[
                Prediction(metric="nonexistent", direction="increase"),
            ],
        )
        evaluated = hyp.evaluate({"accuracy": 0.9}, {"accuracy": 0.8})
        assert evaluated.status == "rejected"
