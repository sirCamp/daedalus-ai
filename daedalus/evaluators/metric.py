"""Metric comparison between experiments."""

from __future__ import annotations

from pydantic import BaseModel


class MetricComparison(BaseModel):
    """Comparison of a single metric between two experiments."""

    metric: str
    baseline_value: float
    current_value: float
    delta: float
    relative_delta: float
    improved: bool


def compare_results(
    current: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    higher_is_better: dict[str, bool] | None = None,
) -> list[MetricComparison]:
    """Compare metrics between current and baseline results.

    Args:
        current: Current experiment results ``{eval_name: {metric: value}}``.
        baseline: Baseline experiment results, same structure.
        higher_is_better: Optional dict ``{metric_name: bool}`` indicating
            direction. Defaults to ``True`` for all metrics.

    Returns:
        List of metric comparisons, sorted by eval_name.metric.
    """
    if higher_is_better is None:
        higher_is_better = {}

    comparisons: list[MetricComparison] = []

    all_eval_names = sorted(set(current) | set(baseline))

    for eval_name in all_eval_names:
        cur_metrics = current.get(eval_name, {})
        base_metrics = baseline.get(eval_name, {})
        all_metrics = sorted(set(cur_metrics) | set(base_metrics))

        for metric in all_metrics:
            if metric not in cur_metrics or metric not in base_metrics:
                continue

            cur_val = cur_metrics[metric]
            base_val = base_metrics[metric]
            delta = cur_val - base_val
            rel_delta = delta / max(abs(base_val), 1e-9)

            hib = higher_is_better.get(f"{eval_name}.{metric}", True)
            improved = (delta > 0) if hib else (delta < 0)

            comparisons.append(
                MetricComparison(
                    metric=f"{eval_name}.{metric}",
                    baseline_value=base_val,
                    current_value=cur_val,
                    delta=delta,
                    relative_delta=rel_delta,
                    improved=improved,
                )
            )

    return comparisons
