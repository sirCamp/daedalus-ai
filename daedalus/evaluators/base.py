"""Abstract evaluator interface."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class EvalResult(BaseModel):
    """A single evaluation metric result."""

    metric_name: str
    value: float
    metadata: dict[str, Any] = {}


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for experiment evaluators."""

    name: str

    def run(self, model_path: str, **kwargs: Any) -> list[EvalResult]:
        """Run evaluation and return metric results."""
        ...
