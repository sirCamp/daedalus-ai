"""Auto-generated research insights from experiment history.

Analyzes the ledger to produce a structured summary of:
- Top configurations and why they work
- Parameter sensitivity (which params matter most)
- Dead ends and known failures
- Explored ranges per parameter
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..core.experiment import Experiment, ExperimentStatus
from ..core.scripts_registry import ScriptsRegistry
from .strategies import _extract_param_history, _get_metric, _find_best

logger = logging.getLogger(__name__)


class InsightsGenerator:
    """Generate aggregated research insights from experiment history."""

    def __init__(
        self,
        experiments: list[Experiment],
        registry: ScriptsRegistry | None = None,
        target_metric: str = "",
        higher_is_better: bool = True,
    ) -> None:
        self.experiments = experiments
        self.registry = registry
        self.target_metric = target_metric
        self.higher_is_better = higher_is_better

        # Only use completed/analyzed with results
        self.completed = [
            e for e in experiments
            if e.status in (ExperimentStatus.COMPLETED, ExperimentStatus.ANALYZED)
            and e.results
        ]

    def generate(self) -> str:
        """Generate full insights document as Markdown."""
        if not self.completed:
            return "# Research Insights\n\nNo completed experiments with results yet."

        sections = ["# Research Insights\n"]
        sections.append(f"Based on {len(self.completed)} completed experiments.\n")

        sections.append(self._top_configs())
        sections.append(self._parameter_sensitivity())
        sections.append(self._dead_ends())
        sections.append(self._explored_ranges())

        return "\n".join(sections)

    def save(self, path: Path) -> None:
        """Generate and save insights to a file."""
        content = self.generate()
        path.write_text(content)
        logger.info(f"Insights saved to {path}")

    def _top_configs(self, n: int = 5) -> str:
        """Rank experiments by target metric, show top N."""
        if not self.target_metric:
            return ""

        scored = []
        for exp in self.completed:
            val = _get_metric(exp, self.target_metric)
            if val is not None:
                scored.append((exp, val))

        if not scored:
            return ""

        scored.sort(key=lambda x: x[1], reverse=self.higher_is_better)
        top = scored[:n]

        lines = [f"## Top {min(n, len(top))} Configurations\n"]
        for rank, (exp, val) in enumerate(top, 1):
            args_str = ", ".join(f"{k}={v}" for k, v in exp.config.script_args.items())
            lines.append(f"{rank}. **{exp.id}** — {self.target_metric}={val:.4f}")
            lines.append(f"   Config: {args_str}")
            if exp.reflection and exp.reflection.analysis:
                # First sentence of analysis
                analysis = exp.reflection.analysis.split(".")[0] + "."
                lines.append(f"   Why: {analysis}")
            lines.append("")

        return "\n".join(lines)

    def _parameter_sensitivity(self) -> str:
        """Compute metric variance per parameter to find high-impact params."""
        if not self.target_metric:
            return ""

        param_history = _extract_param_history(self.completed, self.target_metric)
        if not param_history:
            return ""

        sensitivities: list[tuple[str, float, int]] = []

        for pname, tried in param_history.items():
            values = [v for v in tried.values() if v is not None]
            if len(values) < 2:
                continue

            # Use range (max-min) as sensitivity measure
            metric_range = max(values) - min(values)
            mean = sum(values) / len(values)
            # Relative sensitivity (avoid division by zero)
            rel_sensitivity = metric_range / max(abs(mean), 1e-9)
            sensitivities.append((pname, rel_sensitivity, len(values)))

        if not sensitivities:
            return ""

        sensitivities.sort(key=lambda x: x[1], reverse=True)

        lines = ["## Parameter Sensitivity\n"]
        lines.append(f"Ranking by impact on `{self.target_metric}`:\n")

        for pname, sens, n_points in sensitivities:
            if n_points < 2:
                continue
            bar = "█" * min(int(sens * 20), 20)
            label = "HIGH" if sens > 0.1 else "medium" if sens > 0.02 else "low"
            lines.append(f"- **{pname}** ({label}, {n_points} values): {bar} ({sens:.1%})")

        return "\n".join(lines) + "\n"

    def _dead_ends(self) -> str:
        """Show rejected hypotheses and known failures."""
        rejected = [
            e for e in self.completed
            if e.reflection and e.reflection.hypothesis_confirmed == "rejected"
        ]
        failed = [
            e for e in self.experiments
            if e.status == ExperimentStatus.FAILED
        ]

        if not rejected and not failed:
            return "## Dead Ends\n\nNone found — all experiments confirmed or partially confirmed.\n"

        lines = ["## Dead Ends\n"]

        if rejected:
            lines.append("**Rejected hypotheses**:\n")
            for exp in rejected:
                diff_str = ""
                if exp.config_diff:
                    diff_str = " (" + ", ".join(
                        f"{k}: {v.get('from')}→{v.get('to')}"
                        for k, v in exp.config_diff.items()
                    ) + ")"
                analysis = exp.reflection.analysis[:150] if exp.reflection else "No analysis"
                lines.append(f"- **{exp.id}**{diff_str}: {analysis}")
            lines.append("")

        if failed:
            lines.append(f"**Failed experiments**: {len(failed)} total\n")
            for exp in failed[-5:]:  # last 5
                lines.append(f"- **{exp.id}**: {exp.hypothesis.statement[:80]}")

        return "\n".join(lines) + "\n"

    def _explored_ranges(self) -> str:
        """Show tried ranges per parameter vs declared ranges."""
        param_history = _extract_param_history(self.completed, self.target_metric)
        if not param_history:
            return ""

        lines = ["## Explored Ranges\n"]

        # Get parameter specs from registry
        specs: dict[str, Any] = {}
        if self.registry:
            for _, script_spec in self.registry.scripts.items():
                for pname, pspec in script_spec.parameters.items():
                    specs[pname] = pspec

        for pname, tried in sorted(param_history.items()):
            tried_vals = [v for v in tried.keys() if isinstance(v, (int, float))]
            if not tried_vals:
                # Categorical
                lines.append(f"- **{pname}**: tried {list(tried.keys())}")
                continue

            tried_min = min(tried_vals)
            tried_max = max(tried_vals)
            n_unique = len(set(tried_vals))

            pspec = specs.get(pname)
            if pspec and pspec.range:
                range_low, range_high = pspec.range
                range_span = range_high - range_low
                coverage = (tried_max - tried_min) / range_span * 100 if range_span > 0 else 0
                lines.append(
                    f"- **{pname}**: [{tried_min}, {tried_max}] "
                    f"({n_unique} values, {coverage:.0f}% of [{range_low}, {range_high}])"
                )
            else:
                lines.append(f"- **{pname}**: [{tried_min}, {tried_max}] ({n_unique} values)")

        return "\n".join(lines) + "\n"
