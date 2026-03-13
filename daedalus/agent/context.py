"""Context assembly for agent consumption."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from ..core.experiment import ExperimentStatus
from ..core.ledger import Ledger
from ..core.scripts_registry import ScriptsRegistry
from ..literature.library import Library


class ContextBuilder:
    """Assemble structured context from project state for agent reasoning.

    Each mode includes different context sections optimized for the
    reasoning task at hand.
    """

    def __init__(self, project_path: Path) -> None:
        self.project_path = Path(project_path)
        self.ledger = Ledger(self.project_path / "ledger")
        self.library = Library(self.project_path / "ledger" / "papers.jsonl")
        self.registry = self._load_registry()

    def _load_registry(self) -> ScriptsRegistry:
        """Load scripts registry from daedalus.yaml."""
        config_file = self.project_path / "daedalus.yaml"
        if config_file.exists():
            config = yaml.safe_load(config_file.read_text()) or {}
            scripts_config = config.get("scripts", {})
            if scripts_config:
                return ScriptsRegistry(scripts_config)
        return ScriptsRegistry()

    def _format_stack(self) -> str:
        """Format stack/framework info for context."""
        config_file = self.project_path / "daedalus.yaml"
        if not config_file.exists():
            return ""

        config = yaml.safe_load(config_file.read_text()) or {}
        stack = config.get("stack", {})
        if not stack:
            return ""

        lines: list[str] = []
        if stack.get("python"):
            lines.append(f"- Python: {stack['python']}")
        if stack.get("libraries"):
            lines.append(f"- Libraries: {', '.join(stack['libraries'])}")
        if stack.get("requirements"):
            lines.append(f"- Requirements: {stack['requirements']}")
        if stack.get("docs"):
            lines.append("- Reference docs:")
            for doc in stack["docs"]:
                lines.append(f"  - {doc}")

        return "\n".join(lines) if lines else ""

    def _read_program(self) -> str:
        """Read the research program document."""
        program_file = self.project_path / "program.md"
        if program_file.exists():
            return program_file.read_text()
        return "(No program.md found — define your research goals.)"

    def _format_experiment_detail(self, exp) -> str:
        """Format a single experiment with full detail."""
        status_icon = {
            "draft": "📝", "queued": "⏳", "running": "🔄",
            "completed": "✅", "analyzed": "📊",
            "failed": "❌", "abandoned": "🚫",
        }.get(exp.status.value, "?")

        line = f"{status_icon} **{exp.id}** ({exp.status.value}): {exp.hypothesis.statement}"

        if exp.config_diff:
            changes = ", ".join(
                f"`{k}`: {v['from']}→{v['to']}"
                for k, v in exp.config_diff.items()
            )
            line += f"\n  Changes: {changes}"

        if exp.results:
            for eval_name, metrics in exp.results.items():
                metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                line += f"\n  {eval_name}: {metrics_str}"

        if exp.reflection:
            line += f"\n  Reflection ({exp.reflection.hypothesis_confirmed}): {exp.reflection.analysis}"

        return line

    def _format_compressed_summary(self, experiments: list) -> str:
        """Compress older experiments into aggregated statistics.

        Instead of showing every experiment in detail, this shows:
        - Status distribution
        - Per-parameter: tried values → best metric (leaderboard)
        - Confirmed vs rejected hypotheses
        - Key dead ends
        """
        if not experiments:
            return ""

        from collections import Counter

        # Status distribution
        status_counts = Counter(e.status.value for e in experiments)
        status_line = ", ".join(f"{s}: {c}" for s, c in sorted(status_counts.items()))

        lines = [
            f"**Summary of {len(experiments)} earlier experiments** ({status_line})\n",
        ]

        # Per-parameter leaderboard
        param_best: dict[str, list[tuple[Any, float, str]]] = {}
        for exp in experiments:
            if not exp.results:
                continue
            # Get first metric value as a summary
            for eval_name, metrics in exp.results.items():
                for metric_name, metric_val in metrics.items():
                    if not isinstance(metric_val, (int, float)):
                        continue
                    for param, val in exp.config.script_args.items():
                        if param not in param_best:
                            param_best[param] = []
                        param_best[param].append((val, metric_val, f"{eval_name}.{metric_name}"))

        if param_best:
            lines.append("**Parameter ranges explored**:")
            for param, entries in sorted(param_best.items()):
                values = sorted(set(v for v, _, _ in entries), key=lambda x: (isinstance(x, str), x))
                if len(values) <= 5:
                    lines.append(f"- `{param}`: {values}")
                else:
                    lines.append(f"- `{param}`: {len(values)} values in [{values[0]}, {values[-1]}]")

        # Hypothesis outcomes
        confirmed = [e for e in experiments if e.reflection and e.reflection.hypothesis_confirmed == "confirmed"]
        rejected = [e for e in experiments if e.reflection and e.reflection.hypothesis_confirmed == "rejected"]

        if confirmed or rejected:
            lines.append(f"\n**Hypothesis outcomes**: {len(confirmed)} confirmed, {len(rejected)} rejected")

        # Key dead ends (rejected experiments)
        if rejected:
            lines.append("\n**Dead ends**:")
            for exp in rejected[-5:]:  # last 5 rejections
                diff_str = ""
                if exp.config_diff:
                    diff_str = " (" + ", ".join(f"{k}={v.get('to')}" for k, v in exp.config_diff.items()) + ")"
                lines.append(f"- {exp.id}{diff_str}: {exp.reflection.analysis[:100] if exp.reflection else '?'}")

        return "\n".join(lines)

    def _format_experiment_summary(
        self,
        limit: int | None = None,
        compression_threshold: int = 20,
    ) -> str:
        """Format experiment summaries for context.

        When experiment count exceeds compression_threshold, older experiments
        are compressed into aggregated statistics while recent ones keep full detail.
        """
        experiments = self.ledger.all()

        if limit:
            experiments = experiments[-limit:]

        if not experiments:
            return "No experiments recorded yet."

        # Smart compression: if too many, compress old ones
        if limit is None and len(experiments) > compression_threshold:
            recent_n = compression_threshold
            old_exps = experiments[:-recent_n]
            recent_exps = experiments[-recent_n:]

            parts = [self._format_compressed_summary(old_exps)]
            parts.append(f"\n**Recent {recent_n} experiments (detailed)**:\n")
            for exp in recent_exps:
                parts.append(self._format_experiment_detail(exp))
            return "\n\n".join(parts)

        # Normal: all detailed
        return "\n\n".join(self._format_experiment_detail(exp) for exp in experiments)

    def _format_papers(self) -> str:
        """Format paper library for context."""
        papers = self.library.all()
        if not papers:
            return "No papers in library."

        lines: list[str] = []
        for p in papers:
            ref = f"**{p.title}**"
            if p.authors:
                ref += f" — {p.authors[0]}"
                if len(p.authors) > 1:
                    ref += " et al."
            if p.year:
                ref += f" ({p.year})"
            if p.arxiv_id:
                ref += f" [arXiv:{p.arxiv_id}]"
            if p.relevance_note:
                ref += f"\n  > {p.relevance_note}"
            if p.key_findings:
                for f in p.key_findings:
                    ref += f"\n  - {f}"
            lines.append(ref)

        return "\n\n".join(lines)

    def _active_experiments(self) -> str:
        """Show currently running or queued experiments."""
        active = (
            self.ledger.by_status(ExperimentStatus.RUNNING)
            + self.ledger.by_status(ExperimentStatus.QUEUED)
        )
        if not active:
            return "No active experiments."

        lines = []
        for exp in active:
            lines.append(f"- **{exp.id}** ({exp.status.value}): {exp.hypothesis.statement}")
        return "\n".join(lines)

    def build(
        self,
        mode: Literal["reason", "design", "reflect", "review", "full"] = "full",
        recent_n: int = 10,
    ) -> str:
        """Build context string for agent consumption.

        Modes:
            reason: Program + experiment history + papers → form new hypothesis
            design: Recent experiments + last config → design next experiment
            reflect: Latest completed experiment + baseline → analyze results
            review: Full narrative + paper library → literature review
            full: Everything (default, for Claude Code sessions)
        """
        sections: list[str] = []

        if mode in ("reason", "full"):
            sections.append("# Research Program\n")
            sections.append(self._read_program())

        if mode in ("reason", "design", "reflect", "full"):
            sections.append("\n# Experiment History\n")
            limit = recent_n if mode != "full" else None
            sections.append(self._format_experiment_summary(limit=limit))

        if mode in ("design", "full"):
            stack_info = self._format_stack()
            if stack_info:
                sections.append("\n# Project Stack\n")
                sections.append(stack_info)

            sections.append("\n# Available Scripts\n")
            sections.append(self.registry.format_for_context())

            sections.append("\n# Active Experiments\n")
            sections.append(self._active_experiments())

        if mode == "reflect":
            # Include the latest completed experiment in detail
            completed = self.ledger.by_status(ExperimentStatus.COMPLETED)
            if completed:
                latest = completed[-1]
                sections.append(f"\n# Experiment to Analyze: {latest.id}\n")
                sections.append(f"**Hypothesis**: {latest.hypothesis.statement}")
                sections.append(f"**Rationale**: {latest.hypothesis.rationale}")
                if latest.hypothesis.predictions:
                    sections.append("\n**Predictions**:")
                    for pred in latest.hypothesis.predictions:
                        sections.append(
                            f"- {pred.metric}: expect {pred.direction}"
                            + (f" (≈{pred.expected_value})" if pred.expected_value else "")
                        )
                if latest.results:
                    sections.append("\n**Results**:")
                    for eval_name, metrics in latest.results.items():
                        for k, v in metrics.items():
                            sections.append(f"- {eval_name}.{k} = {v}")

                # Include baseline for comparison
                if latest.baseline_id:
                    baseline = self.ledger.get(latest.baseline_id)
                    if baseline and baseline.results:
                        sections.append(f"\n**Baseline ({baseline.id}) results**:")
                        for eval_name, metrics in baseline.results.items():
                            for k, v in metrics.items():
                                sections.append(f"- {eval_name}.{k} = {v}")

        if mode in ("reason", "review", "full"):
            sections.append("\n# Literature\n")
            sections.append(self._format_papers())

        if mode in ("reason", "full"):
            # Include latest reflection's next_suggestions
            analyzed = self.ledger.by_status(ExperimentStatus.ANALYZED)
            if analyzed:
                latest = analyzed[-1]
                if latest.reflection and latest.reflection.next_suggestions:
                    sections.append("\n# Suggested Next Steps\n")
                    for s in latest.reflection.next_suggestions:
                        sections.append(f"- {s}")

        # Include research plan in reason, design, and full modes
        if mode in ("reason", "design", "full"):
            from .plan import PlanManager
            plan_mgr = PlanManager(self.project_path / "ledger")
            plan_ctx = plan_mgr.format_for_context()
            if "(No research plan" not in plan_ctx:
                sections.append("\n# Research Plan\n")
                sections.append(plan_ctx)

        # Include research memory in reason and full modes
        if mode in ("reason", "design", "full"):
            from .memory import ResearchMemory
            memory = ResearchMemory(self.project_path / "ledger")
            notes_ctx = memory.format_for_context(limit=15)
            if notes_ctx and "No research notes" not in notes_ctx:
                sections.append("\n# Research Memory\n")
                sections.append(notes_ctx)

        return "\n".join(sections)
