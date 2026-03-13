"""Append-only experiment ledger with dual storage (JSONL + narrative)."""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Iterator

from .experiment import Experiment, ExperimentStatus

logger = logging.getLogger(__name__)


class Ledger:
    """Persistent experiment ledger backed by JSONL + Markdown narrative.

    Thread-safe via file locking. The JSONL file is the source of truth;
    the narrative Markdown is regenerated on every write.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.path / "experiments.jsonl"
        self._narrative = self.path / "narrative.md"

        # Ensure files exist
        self._jsonl.touch(exist_ok=True)

    def _read_all(self) -> list[Experiment]:
        """Read all experiments from JSONL."""
        experiments: list[Experiment] = []
        if not self._jsonl.exists():
            return experiments

        for line_no, line in enumerate(self._jsonl.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                experiments.append(Experiment.model_validate_json(line))
            except Exception as e:
                logger.warning(f"Failed to parse ledger line {line_no}: {e}")
        return experiments

    def _write_all(self, experiments: list[Experiment]) -> None:
        """Write all experiments to JSONL with file locking."""
        with open(self._jsonl, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for exp in experiments:
                    f.write(exp.model_dump_json() + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        self._sync_narrative(experiments)

    def append(self, experiment: Experiment) -> None:
        """Append a new experiment to the ledger."""
        existing = self._read_all()
        if any(e.id == experiment.id for e in existing):
            raise ValueError(f"Experiment {experiment.id} already exists. Use update().")
        existing.append(experiment)
        self._write_all(existing)
        logger.info(f"Appended experiment {experiment.id} ({experiment.status.value})")

    def update(self, experiment: Experiment) -> None:
        """Update an existing experiment in place."""
        experiments = self._read_all()
        found = False
        for i, e in enumerate(experiments):
            if e.id == experiment.id:
                experiments[i] = experiment
                found = True
                break
        if not found:
            raise ValueError(f"Experiment {experiment.id} not found in ledger.")
        self._write_all(experiments)
        logger.info(f"Updated experiment {experiment.id} ({experiment.status.value})")

    def get(self, exp_id: str) -> Experiment | None:
        """Retrieve an experiment by ID."""
        for exp in self._read_all():
            if exp.id == exp_id:
                return exp
        return None

    def latest(self, n: int = 1) -> list[Experiment]:
        """Return the N most recent experiments (by created_at)."""
        experiments = self._read_all()
        experiments.sort(key=lambda e: e.created_at, reverse=True)
        return experiments[:n]

    def by_status(self, status: ExperimentStatus) -> list[Experiment]:
        """Return all experiments with a given status."""
        return [e for e in self._read_all() if e.status == status]

    def baseline(self, exp_id: str) -> Experiment | None:
        """Return the baseline experiment for a given experiment."""
        exp = self.get(exp_id)
        if exp is None or exp.baseline_id is None:
            return None
        return self.get(exp.baseline_id)

    def all(self) -> list[Experiment]:
        """Return all experiments, ordered by created_at."""
        experiments = self._read_all()
        experiments.sort(key=lambda e: e.created_at)
        return experiments

    def __len__(self) -> int:
        return len(self._read_all())

    def __iter__(self) -> Iterator[Experiment]:
        return iter(self.all())

    def narrative(self) -> str:
        """Return the current narrative markdown."""
        if self._narrative.exists():
            return self._narrative.read_text()
        return ""

    def _sync_narrative(self, experiments: list[Experiment]) -> None:
        """Regenerate the narrative markdown from current experiments."""
        if not experiments:
            self._narrative.write_text("# Research Narrative\n\nNo experiments yet.\n")
            return

        lines: list[str] = ["# Research Narrative\n"]
        lines.append(f"Total experiments: {len(experiments)}\n")

        # Summary by status
        status_counts: dict[str, int] = {}
        for exp in experiments:
            status_counts[exp.status.value] = status_counts.get(exp.status.value, 0) + 1
        lines.append("## Status Summary\n")
        for status, count in sorted(status_counts.items()):
            lines.append(f"- **{status}**: {count}")
        lines.append("")

        # Experiments (chronological)
        lines.append("## Experiment Log\n")
        sorted_exps = sorted(experiments, key=lambda e: e.created_at)

        for exp in sorted_exps:
            ts = exp.created_at.strftime("%Y-%m-%d %H:%M")
            lines.append(f"### {exp.id} ({exp.status.value}) — {ts}\n")
            lines.append(f"**Hypothesis**: {exp.hypothesis.statement}")
            if exp.hypothesis.rationale:
                lines.append(f"**Rationale**: {exp.hypothesis.rationale}")

            if exp.config_diff:
                lines.append("\n**Config changes**:")
                for key, change in exp.config_diff.items():
                    lines.append(f"- `{key}`: {change.get('from')} → {change.get('to')}")

            if exp.results:
                lines.append("\n**Results**:")
                for eval_name, metrics in exp.results.items():
                    metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                    lines.append(f"- {eval_name}: {metrics_str}")

            if exp.reflection:
                lines.append(f"\n**Reflection** ({exp.reflection.hypothesis_confirmed}):")
                if exp.reflection.analysis:
                    lines.append(f"> {exp.reflection.analysis}")
                if exp.reflection.surprise:
                    lines.append(f"> Surprise: {exp.reflection.surprise}")
                if exp.reflection.next_suggestions:
                    lines.append("\nNext steps:")
                    for s in exp.reflection.next_suggestions:
                        lines.append(f"- {s}")

            lines.append("")

        self._narrative.write_text("\n".join(lines))
