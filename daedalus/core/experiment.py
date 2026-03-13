"""Experiment lifecycle model."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .config import ExperimentConfig
from .hypothesis import Hypothesis


class ExperimentStatus(str, Enum):
    """Experiment lifecycle states."""

    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ANALYZED = "analyzed"
    FAILED = "failed"
    ABANDONED = "abandoned"


# Valid state transitions
_TRANSITIONS: dict[ExperimentStatus, set[ExperimentStatus]] = {
    ExperimentStatus.DRAFT: {ExperimentStatus.QUEUED, ExperimentStatus.ABANDONED},
    ExperimentStatus.QUEUED: {
        ExperimentStatus.RUNNING,
        ExperimentStatus.FAILED,
        ExperimentStatus.ABANDONED,
    },
    ExperimentStatus.RUNNING: {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.ABANDONED,
    },
    ExperimentStatus.COMPLETED: {ExperimentStatus.ANALYZED, ExperimentStatus.ABANDONED},
    ExperimentStatus.ANALYZED: set(),
    ExperimentStatus.FAILED: set(),
    ExperimentStatus.ABANDONED: set(),
}


class Reflection(BaseModel):
    """Post-experiment analysis."""

    hypothesis_confirmed: str = "pending"  # confirmed / partial / rejected / pending
    analysis: str = ""
    surprise: str | None = None
    next_suggestions: list[str] = []


class Experiment(BaseModel):
    """A single research experiment with full lifecycle tracking."""

    id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:6]}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    hypothesis: Hypothesis
    config: ExperimentConfig
    config_diff: dict[str, Any] | None = None
    baseline_id: str | None = None

    status: ExperimentStatus = ExperimentStatus.DRAFT
    run_id: str | None = None  # opaque ID from the runner

    results: dict[str, dict[str, float]] | None = None
    reflection: Reflection | None = None

    tags: list[str] = []
    notes: str = ""

    def transition(self, new_status: ExperimentStatus) -> "Experiment":
        """Transition to a new status, enforcing valid transitions.

        Returns a copy with updated status and timestamp.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed = _TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition from {self.status.value} to {new_status.value}. "
                f"Allowed: {sorted(s.value for s in allowed)}"
            )
        return self.model_copy(
            update={
                "status": new_status,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    @property
    def is_terminal(self) -> bool:
        """Whether the experiment is in a terminal state."""
        return self.status in {
            ExperimentStatus.ANALYZED,
            ExperimentStatus.FAILED,
            ExperimentStatus.ABANDONED,
        }
