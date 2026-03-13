"""Evaluation battery — run multiple evaluators and aggregate results."""

from __future__ import annotations

import logging
from typing import Any

from ..core.ledger import Ledger
from .base import Evaluator, EvalResult
from .metric import MetricComparison, compare_results

logger = logging.getLogger(__name__)


class EvalBattery:
    """Run a set of evaluators and aggregate results."""

    def __init__(
        self,
        evaluators: list[Evaluator],
        higher_is_better: dict[str, bool] | None = None,
    ) -> None:
        self.evaluators = evaluators
        self.higher_is_better = higher_is_better or {}

    def run_all(self, model_path: str, **kwargs: Any) -> dict[str, dict[str, float]]:
        """Run all evaluators and return aggregated results.

        Returns:
            ``{evaluator_name: {metric_name: value}}``.
        """
        results: dict[str, dict[str, float]] = {}

        for evaluator in self.evaluators:
            logger.info(f"Running evaluator: {evaluator.name}")
            try:
                eval_results = evaluator.run(model_path, **kwargs)
                results[evaluator.name] = {
                    r.metric_name: r.value for r in eval_results
                }
            except Exception as e:
                logger.error(f"Evaluator {evaluator.name} failed: {e}")
                results[evaluator.name] = {"error": -1.0}

        return results

    def compare(
        self,
        current: dict[str, dict[str, float]],
        baseline_id: str,
        ledger: Ledger,
    ) -> list[MetricComparison]:
        """Compare current results against a baseline from the ledger."""
        baseline_exp = ledger.get(baseline_id)
        if baseline_exp is None:
            raise ValueError(f"Baseline experiment {baseline_id} not found in ledger.")
        if baseline_exp.results is None:
            raise ValueError(f"Baseline experiment {baseline_id} has no results.")

        return compare_results(
            current,
            baseline_exp.results,
            higher_is_better=self.higher_is_better,
        )
