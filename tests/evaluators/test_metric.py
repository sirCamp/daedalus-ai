"""Tests for metric comparison."""

import pytest

from daedalus.evaluators.metric import compare_results, MetricComparison


class TestCompareResults:
    def test_basic_comparison(self):
        current = {"simpleqa": {"correct": 3.5, "not_attempted": 41.0}}
        baseline = {"simpleqa": {"correct": 3.0, "not_attempted": 50.0}}

        comps = compare_results(current, baseline)
        assert len(comps) == 2

        correct_comp = next(c for c in comps if c.metric == "simpleqa.correct")
        assert correct_comp.delta == pytest.approx(0.5)
        assert correct_comp.improved is True

        na_comp = next(c for c in comps if c.metric == "simpleqa.not_attempted")
        assert na_comp.delta == pytest.approx(-9.0)
        # Default: higher_is_better=True, so decrease is NOT improved
        assert na_comp.improved is False

    def test_higher_is_better_override(self):
        current = {"simpleqa": {"not_attempted": 41.0}}
        baseline = {"simpleqa": {"not_attempted": 50.0}}

        comps = compare_results(
            current,
            baseline,
            higher_is_better={"simpleqa.not_attempted": False},
        )
        assert comps[0].improved is True  # decrease is good

    def test_missing_metric_skipped(self):
        current = {"simpleqa": {"correct": 3.5}}
        baseline = {"simpleqa": {"correct": 3.0, "new_metric": 1.0}}

        comps = compare_results(current, baseline)
        # Only 'correct' has both values
        assert len(comps) == 1
        assert comps[0].metric == "simpleqa.correct"

    def test_multiple_eval_names(self):
        current = {
            "simpleqa": {"correct": 3.5},
            "popqa": {"em": 10.5, "similarity": 0.46},
        }
        baseline = {
            "simpleqa": {"correct": 3.0},
            "popqa": {"em": 10.0, "similarity": 0.45},
        }

        comps = compare_results(current, baseline)
        assert len(comps) == 3
        metrics = {c.metric for c in comps}
        assert metrics == {"simpleqa.correct", "popqa.em", "popqa.similarity"}

    def test_empty_results(self):
        assert compare_results({}, {}) == []

    def test_relative_delta(self):
        current = {"eval": {"metric": 1.5}}
        baseline = {"eval": {"metric": 1.0}}

        comps = compare_results(current, baseline)
        assert comps[0].relative_delta == pytest.approx(0.5)
