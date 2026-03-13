"""Exploration strategies — help the agent decide what to try next.

Given a history of experiments and their results, strategies suggest
which parameter values to try next. The agent can use these suggestions
or override them with its own reasoning.

Strategies:
- grid: enumerate a predefined set of values
- sweep: linear sweep across a range
- bisect: binary search between best-so-far and a bound
- trend: extrapolate from observed trends
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.experiment import Experiment, ExperimentStatus
from ..core.scripts_registry import ScriptsRegistry, ParameterSpec

logger = logging.getLogger(__name__)


def suggest_next_params(
    experiments: list[Experiment],
    registry: ScriptsRegistry,
    target_metric: str,
    higher_is_better: bool = True,
    focus_param: str | None = None,
) -> list[dict[str, Any]]:
    """Suggest parameter configurations to try next.

    Analyzes completed experiments and suggests configurations that
    explore the parameter space intelligently.

    Args:
        experiments: All experiments (including incomplete ones).
        registry: Scripts registry with parameter specs.
        target_metric: Metric to optimize (e.g., "simpleqa.not_attempted").
        higher_is_better: Direction of improvement.
        focus_param: If set, only suggest variations of this parameter.

    Returns:
        List of suggested configurations with rationale.
    """
    # Only use completed/analyzed experiments with results
    completed = [
        e for e in experiments
        if e.status in (ExperimentStatus.COMPLETED, ExperimentStatus.ANALYZED)
        and e.results
    ]

    if not completed:
        return _suggest_baseline(registry)

    # Extract parameter-metric relationships
    param_history = _extract_param_history(completed, target_metric)
    suggestions: list[dict[str, Any]] = []

    # Find the best experiment so far
    best_exp = _find_best(completed, target_metric, higher_is_better)

    # Determine which script to use
    script_name = _guess_script_name(best_exp, registry)
    if not script_name:
        return []

    spec = registry.get(script_name)
    if not spec:
        return []

    # Generate suggestions for each tunable parameter
    params_to_explore = [focus_param] if focus_param else list(spec.parameters.keys())
    converged_params: list[dict[str, Any]] = []

    for pname in params_to_explore:
        param = spec.parameters.get(pname)
        if not param:
            continue

        tried_values = param_history.get(pname, {})
        if not tried_values:
            continue

        # Check convergence before generating suggestions
        scored = {k: v for k, v in tried_values.items() if v is not None}
        conv = _check_convergence(pname, param, scored, higher_is_better)
        if conv:
            converged_params.append(conv)
            continue  # skip this parameter

        new_suggestions = _suggest_for_param(
            pname, param, tried_values,
            best_exp, target_metric, higher_is_better,
        )
        suggestions.extend(new_suggestions)

    # Also suggest untried parameters
    for pname, param in spec.parameters.items():
        if focus_param and pname != focus_param:
            continue
        if pname not in param_history and param.default is not None:
            if param.choices and len(param.choices) > 1:
                for choice in param.choices:
                    if choice != param.default:
                        suggestions.append({
                            "parameter": pname,
                            "value": choice,
                            "strategy": "unexplored_choice",
                            "rationale": f"'{pname}' has never been varied. Try {choice} (default: {param.default}).",
                        })
                        break  # suggest only one untried choice
            elif param.range and param.type in ("float", "int"):
                # Suggest midpoint away from default
                low, high = param.range
                if param.default == low:
                    suggested = (low + high) / 2
                elif param.default == high:
                    suggested = (low + high) / 2
                else:
                    suggested = high if param.default < (low + high) / 2 else low
                suggestions.append({
                    "parameter": pname,
                    "value": suggested,
                    "strategy": "unexplored_range",
                    "rationale": f"'{pname}' has never been varied. Try {suggested} (default: {param.default}, range: [{low}, {high}]).",
                })

    # Attach convergence info to the result
    result = suggestions[:5]  # top 5 suggestions
    if converged_params:
        for cp in converged_params:
            result.append(cp)
    return result


def _suggest_baseline(registry: ScriptsRegistry) -> list[dict[str, Any]]:
    """When no experiments exist, suggest running with defaults."""
    suggestions = []
    for name, spec in registry.scripts.items():
        defaults = {
            pname: param.default
            for pname, param in spec.parameters.items()
            if param.default is not None
        }
        suggestions.append({
            "script": spec.path,
            "script_args": defaults,
            "strategy": "baseline",
            "rationale": f"No experiments yet. Run baseline with default parameters for '{name}'.",
        })
    return suggestions


def _extract_param_history(
    experiments: list[Experiment],
    target_metric: str,
) -> dict[str, dict[Any, float | None]]:
    """Extract {param_name: {value: metric_value}} from experiment history."""
    history: dict[str, dict[Any, float | None]] = {}

    for exp in experiments:
        metric_val = _get_metric(exp, target_metric)
        for param, value in exp.config.script_args.items():
            if param not in history:
                history[param] = {}
            history[param][value] = metric_val

    return history


def _get_metric(exp: Experiment, metric_key: str) -> float | None:
    """Get a metric value from an experiment, handling dot notation."""
    if not exp.results:
        return None

    if "." in metric_key:
        eval_name, metric_name = metric_key.split(".", 1)
        return exp.results.get(eval_name, {}).get(metric_name)

    # Search all eval names
    for metrics in exp.results.values():
        if metric_key in metrics:
            return metrics[metric_key]
    return None


def _find_best(
    experiments: list[Experiment],
    target_metric: str,
    higher_is_better: bool,
) -> Experiment:
    """Find the experiment with the best target metric."""
    best = experiments[0]
    best_val = _get_metric(best, target_metric)

    for exp in experiments[1:]:
        val = _get_metric(exp, target_metric)
        if val is None:
            continue
        if best_val is None:
            best, best_val = exp, val
            continue
        if higher_is_better and val > best_val:
            best, best_val = exp, val
        elif not higher_is_better and val < best_val:
            best, best_val = exp, val

    return best


def _guess_script_name(exp: Experiment, registry: ScriptsRegistry) -> str | None:
    """Guess which registry script an experiment used."""
    for name, spec in registry.scripts.items():
        if spec.path == exp.config.script:
            return name
    # Fall back to first script
    if registry.scripts:
        return next(iter(registry.scripts))
    return None


def _check_convergence(
    pname: str,
    param: ParameterSpec,
    scored: dict[Any, float | None],
    higher_is_better: bool,
    min_points: int = 3,
    plateau_threshold: float = 0.02,
    range_coverage_threshold: float = 0.8,
) -> dict[str, Any] | None:
    """Check if a parameter has converged (further exploration unlikely to help).

    Three convergence checks:
    1. **Plateau**: last min_points experiments show <plateau_threshold relative improvement
    2. **Range saturation**: tried values cover >range_coverage_threshold of declared range
       AND best value is not at the boundary
    3. **Categorical exhaustion**: all choices have been tried

    Returns a convergence report dict, or None if not converged.
    """
    filtered = {k: v for k, v in scored.items() if v is not None}
    if len(filtered) < min_points:
        return None

    # --- Categorical exhaustion ---
    if param.choices:
        tried = set(str(k) for k in filtered.keys())
        all_choices = set(str(c) for c in param.choices)
        if tried >= all_choices:
            best_k = max(filtered, key=filtered.get) if higher_is_better else min(filtered, key=filtered.get)  # type: ignore
            return {
                "parameter": pname,
                "converged": True,
                "reason": "all_choices_tried",
                "detail": f"All {len(param.choices)} choices tested. Best: {best_k}→{filtered[best_k]:.4f}",
                "best_value": best_k,
                "best_metric": filtered[best_k],
            }
        return None

    # --- Numeric checks ---
    if param.type not in ("float", "int"):
        return None

    sorted_points = sorted(
        [(float(k), v) for k, v in filtered.items()],
        key=lambda x: x[0],
    )
    values = [v for _, v in sorted_points]

    # Find running best
    if higher_is_better:
        running_best = []
        best_so_far = float("-inf")
        for v in values:
            best_so_far = max(best_so_far, v)
            running_best.append(best_so_far)
        overall_best_val, overall_best_metric = max(sorted_points, key=lambda x: x[1])
    else:
        running_best = []
        best_so_far = float("inf")
        for v in values:
            best_so_far = min(best_so_far, v)
            running_best.append(best_so_far)
        overall_best_val, overall_best_metric = min(sorted_points, key=lambda x: x[1])

    # --- Plateau detection ---
    # Check if the last min_points additions improved <threshold
    if len(running_best) >= min_points:
        recent_improvements = []
        for i in range(len(running_best) - min_points, len(running_best)):
            if i < 1:
                continue
            prev = running_best[i - 1]
            curr = running_best[i]
            denom = max(abs(prev), 1e-9)
            rel_improvement = abs(curr - prev) / denom
            recent_improvements.append(rel_improvement)

        if recent_improvements and all(imp < plateau_threshold for imp in recent_improvements):
            detail_vals = [f"{v:.4f}" for _, v in sorted_points[-min_points:]]
            return {
                "parameter": pname,
                "converged": True,
                "reason": "plateau",
                "detail": f"Last {min_points} values show <{plateau_threshold*100:.0f}% improvement: [{', '.join(detail_vals)}]",
                "best_value": overall_best_val,
                "best_metric": overall_best_metric,
            }

    # --- Range saturation ---
    if param.range:
        range_low, range_high = param.range
        range_span = range_high - range_low
        if range_span > 0:
            tried_low = sorted_points[0][0]
            tried_high = sorted_points[-1][0]
            coverage = (tried_high - tried_low) / range_span

            # Best not at boundary = we've likely found the optimum region
            at_boundary = (
                abs(overall_best_val - range_low) < range_span * 0.05
                or abs(overall_best_val - range_high) < range_span * 0.05
            )

            if coverage >= range_coverage_threshold and not at_boundary:
                return {
                    "parameter": pname,
                    "converged": True,
                    "reason": "range_saturated",
                    "detail": (
                        f"Tried [{tried_low}, {tried_high}] covering {coverage*100:.0f}% "
                        f"of range [{range_low}, {range_high}]. "
                        f"Best {overall_best_val}→{overall_best_metric:.4f} is interior."
                    ),
                    "best_value": overall_best_val,
                    "best_metric": overall_best_metric,
                }

    return None


def _suggest_for_param(
    pname: str,
    param: ParameterSpec,
    tried: dict[Any, float | None],
    best_exp: Experiment,
    target_metric: str,
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    """Suggest next values for a specific parameter."""
    suggestions = []

    # Filter to entries with metrics
    scored = {k: v for k, v in tried.items() if v is not None}
    if not scored:
        return suggestions

    if param.type in ("float", "int"):
        suggestions.extend(_suggest_numeric(pname, param, scored, higher_is_better))
    elif param.choices:
        suggestions.extend(_suggest_categorical(pname, param, scored, higher_is_better))

    return suggestions


def _suggest_numeric(
    pname: str,
    param: ParameterSpec,
    scored: dict[Any, float | None],
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    """Suggest next numeric values using bisection and trend analysis."""
    suggestions = []

    # Sort by parameter value
    sorted_points = sorted(
        [(float(k), v) for k, v in scored.items() if v is not None],
        key=lambda x: x[0],
    )

    if len(sorted_points) < 2:
        # Only one point — suggest neighbors
        val, metric = sorted_points[0]
        if param.range:
            low, high = param.range
            # Suggest one step in each direction
            step = (high - low) / 4
            if val + step <= high:
                suggestions.append({
                    "parameter": pname,
                    "value": round(val + step, 6),
                    "strategy": "explore_neighbor",
                    "rationale": f"Only one value tried ({val}→{metric:.4f}). Explore {val + step:.6f}.",
                })
            if val - step >= low:
                suggestions.append({
                    "parameter": pname,
                    "value": round(val - step, 6),
                    "strategy": "explore_neighbor",
                    "rationale": f"Only one value tried ({val}→{metric:.4f}). Explore {val - step:.6f}.",
                })
        return suggestions

    # Multiple points — find best and bisect
    if higher_is_better:
        best_val, best_metric = max(sorted_points, key=lambda x: x[1])
    else:
        best_val, best_metric = min(sorted_points, key=lambda x: x[1])

    # Find adjacent points to best
    best_idx = next(i for i, (v, _) in enumerate(sorted_points) if v == best_val)

    # Bisect between best and neighbors
    if best_idx > 0:
        neighbor_val, neighbor_metric = sorted_points[best_idx - 1]
        midpoint = round((best_val + neighbor_val) / 2, 6)
        if midpoint not in [v for v, _ in sorted_points]:
            suggestions.append({
                "parameter": pname,
                "value": midpoint,
                "strategy": "bisect",
                "rationale": (
                    f"Best {pname}={best_val}→{best_metric:.4f}, "
                    f"neighbor {neighbor_val}→{neighbor_metric:.4f}. "
                    f"Bisect at {midpoint}."
                ),
            })

    if best_idx < len(sorted_points) - 1:
        neighbor_val, neighbor_metric = sorted_points[best_idx + 1]
        midpoint = round((best_val + neighbor_val) / 2, 6)
        if midpoint not in [v for v, _ in sorted_points]:
            suggestions.append({
                "parameter": pname,
                "value": midpoint,
                "strategy": "bisect",
                "rationale": (
                    f"Best {pname}={best_val}→{best_metric:.4f}, "
                    f"neighbor {neighbor_val}→{neighbor_metric:.4f}. "
                    f"Bisect at {midpoint}."
                ),
            })

    # Trend: if improving monotonically, suggest extrapolation
    metrics_ordered = [m for _, m in sorted_points]
    if len(metrics_ordered) >= 2:
        diffs = [metrics_ordered[i+1] - metrics_ordered[i] for i in range(len(metrics_ordered)-1)]
        all_improving = all(d > 0 for d in diffs) if higher_is_better else all(d < 0 for d in diffs)

        if all_improving:
            # Extrapolate in the improving direction
            last_val = sorted_points[-1][0] if higher_is_better == (diffs[-1] > 0) else sorted_points[0][0]
            step = sorted_points[-1][0] - sorted_points[-2][0]
            extrapolated = round(last_val + step, 6)

            # Clamp to range
            if param.range:
                extrapolated = max(param.range[0], min(param.range[1], extrapolated))

            if extrapolated not in [v for v, _ in sorted_points]:
                suggestions.append({
                    "parameter": pname,
                    "value": extrapolated,
                    "strategy": "trend_extrapolate",
                    "rationale": f"Monotonic trend detected for {pname}. Extrapolate to {extrapolated}.",
                })

    return suggestions


def _suggest_categorical(
    pname: str,
    param: ParameterSpec,
    scored: dict[Any, float | None],
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    """Suggest untried categorical values."""
    suggestions = []

    if not param.choices:
        return suggestions

    tried = set(str(k) for k in scored.keys())
    untried = [c for c in param.choices if str(c) not in tried]

    for choice in untried[:2]:  # suggest up to 2 untried choices
        suggestions.append({
            "parameter": pname,
            "value": choice,
            "strategy": "untried_choice",
            "rationale": f"'{choice}' has not been tried yet for {pname}. Tried: {list(scored.keys())}.",
        })

    return suggestions
