"""Pydantic models for dataset inspection and validation reports."""

from __future__ import annotations

from pydantic import BaseModel, computed_field


class ColumnStats(BaseModel):
    """Statistics for a single column/field."""

    name: str
    dtype: str  # "str", "int", "float", "list", "dict", "bool", "null", "mixed"
    null_count: int = 0
    total_count: int = 0
    unique_count: int = 0
    top_values: list[tuple[str, int]] = []  # (value, count)
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    empty_string_count: int = 0


class InspectionReport(BaseModel):
    """Report from dataset inspection."""

    row_count: int
    column_count: int
    columns: list[ColumnStats]
    sample_records: list[dict]
    estimated_tokens: int | None = None
    file_size_bytes: int | None = None
    format: str  # "jsonl", "csv", "parquet", "json", "huggingface"
    sampled: bool = False  # True if sample_n was applied
    warnings: list[str] = []


class ValidationCheck(BaseModel):
    """Result of a single validation check."""

    name: str
    passed: bool
    severity: str  # "error", "warning", "info"
    message: str
    details: dict | None = None


class ValidationReport(BaseModel):
    """Report from dataset validation."""

    checks: list[ValidationCheck] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """True if no error-level checks failed."""
        return all(c.passed for c in self.checks if c.severity == "error")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == "error" and not c.passed)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == "warning" and not c.passed)
