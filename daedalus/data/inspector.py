"""Dataset inspector — compute statistics and profiles."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .loader import load_dataset
from .models import ColumnStats, InspectionReport

logger = logging.getLogger(__name__)


class DatasetInspector:
    """Inspect datasets and produce statistical reports."""

    def inspect(
        self,
        path: str,
        sample_n: int | None = None,
        num_samples: int = 3,
    ) -> InspectionReport:
        """Inspect a dataset and return a report.

        Args:
            path: Path to dataset file or HuggingFace ID.
            sample_n: Sample size for large datasets.
            num_samples: Number of sample records to include.

        Returns:
            InspectionReport with statistics and samples.
        """
        records, fmt = load_dataset(path, sample_n=sample_n)

        if not records:
            return InspectionReport(
                row_count=0,
                column_count=0,
                columns=[],
                sample_records=[],
                format=fmt,
                sampled=sample_n is not None,
            )

        # Collect all field names
        all_fields: list[str] = []
        field_set: set[str] = set()
        for rec in records:
            for k in rec:
                if k not in field_set:
                    all_fields.append(k)
                    field_set.add(k)

        # Compute column stats
        columns = [self._compute_column(name, records) for name in all_fields]

        # Sample records
        samples = records[:num_samples]

        # Token estimate (chars / 4 rough heuristic)
        total_chars = sum(len(str(rec)) for rec in records)
        estimated_tokens = total_chars // 4

        # File size
        file_size = None
        p = Path(path)
        if p.exists():
            file_size = p.stat().st_size

        warnings: list[str] = []
        if sample_n and sample_n < len(records):
            warnings.append(f"Dataset sampled to {sample_n} rows (original size unknown)")
        elif sample_n:
            pass  # sample_n >= actual size, no warning

        return InspectionReport(
            row_count=len(records),
            column_count=len(all_fields),
            columns=columns,
            sample_records=samples,
            estimated_tokens=estimated_tokens,
            file_size_bytes=file_size,
            format=fmt,
            sampled=sample_n is not None and sample_n < len(records),
            warnings=warnings,
        )

    def _compute_column(self, name: str, records: list[dict]) -> ColumnStats:
        """Compute statistics for a single column."""
        values: list[Any] = []
        null_count = 0
        empty_count = 0
        type_counter: Counter[str] = Counter()

        for rec in records:
            v = rec.get(name)
            if v is None:
                null_count += 1
                type_counter["null"] += 1
            else:
                values.append(v)
                t = type(v).__name__
                type_counter[t] += 1
                if isinstance(v, str) and v.strip() == "":
                    empty_count += 1

        total = len(records)

        # Determine dominant type
        non_null_types = {k: v for k, v in type_counter.items() if k != "null"}
        if not non_null_types:
            dtype = "null"
        elif len(non_null_types) == 1:
            dtype = next(iter(non_null_types))
        else:
            dtype = "mixed"

        # Unique count (for hashable values only)
        try:
            unique_count = len(set(str(v) for v in values))
        except TypeError:
            unique_count = len(values)

        # Top values (for non-collection types)
        top_values: list[tuple[str, int]] = []
        if dtype in ("str", "int", "float", "bool"):
            counter = Counter(str(v) for v in values)
            top_values = counter.most_common(10)

        # Numeric stats
        min_val = max_val = mean_val = None
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if numeric_values:
            min_val = float(min(numeric_values))
            max_val = float(max(numeric_values))
            mean_val = sum(numeric_values) / len(numeric_values)

        return ColumnStats(
            name=name,
            dtype=dtype,
            null_count=null_count,
            total_count=total,
            unique_count=unique_count,
            top_values=top_values,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            empty_string_count=empty_count,
        )
