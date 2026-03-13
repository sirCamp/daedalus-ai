"""Hypothesis tracking with falsifiable predictions."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """A single falsifiable prediction attached to a hypothesis."""

    metric: str
    direction: Literal["increase", "decrease", "stable"]
    expected_value: float | None = None
    tolerance: float = 0.1  # relative tolerance for confirmation

    def evaluate(self, actual: float, baseline: float | None = None) -> bool:
        """Check whether the actual result matches this prediction.

        Uses ``expected_value`` when available; otherwise checks direction
        relative to ``baseline``.
        """
        if self.expected_value is not None:
            rel = abs(actual - self.expected_value) / max(abs(self.expected_value), 1e-9)
            return rel <= self.tolerance

        if baseline is None:
            return False

        delta = actual - baseline
        if self.direction == "increase":
            return delta > 0
        if self.direction == "decrease":
            return delta < 0
        # stable
        rel = abs(delta) / max(abs(baseline), 1e-9)
        return rel <= self.tolerance


class Hypothesis(BaseModel):
    """A research hypothesis with falsifiable predictions."""

    id: str = Field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:6]}")
    statement: str
    rationale: str
    predictions: list[Prediction] = []
    papers: list[str] = []  # arxiv IDs or references supporting this
    status: Literal["pending", "confirmed", "partial", "rejected"] = "pending"

    def evaluate(
        self,
        results: dict[str, float],
        baseline: dict[str, float] | None = None,
    ) -> "Hypothesis":
        """Evaluate predictions against actual results and update status.

        Returns a copy with updated ``status``.
        """
        if not self.predictions:
            return self.model_copy(update={"status": "pending"})

        outcomes: list[bool] = []
        for pred in self.predictions:
            actual = results.get(pred.metric)
            if actual is None:
                outcomes.append(False)
                continue
            base_val = baseline.get(pred.metric) if baseline else None
            outcomes.append(pred.evaluate(actual, base_val))

        confirmed = sum(outcomes)
        total = len(outcomes)

        if confirmed == total:
            new_status = "confirmed"
        elif confirmed > 0:
            new_status = "partial"
        else:
            new_status = "rejected"

        return self.model_copy(update={"status": new_status})
