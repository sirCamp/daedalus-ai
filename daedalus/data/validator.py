"""Dataset validator — configurable checks for ML datasets."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loader import load_dataset
from .models import ValidationCheck, ValidationReport

logger = logging.getLogger(__name__)

# Predefined format requirements
FORMAT_REQUIREMENTS: dict[str, list[str]] = {
    # Generative / alignment
    "sft": ["prompt", "completion"],
    "chat": ["messages"],
    "dpo": ["prompt", "chosen", "rejected"],
    "grpo": ["prompt"],
    # Classification
    "classification": ["text", "label"],
    "nli": ["premise", "hypothesis", "label"],
    "sentiment": ["text", "label"],
    "multi_label": ["text", "labels"],
    # Sequence labeling
    "ner": ["tokens", "tags"],
    "token_classification": ["tokens", "tags"],
    # Regression
    "regression": ["text", "score"],
    "sts": ["sentence1", "sentence2", "score"],
    # Question answering
    "qa": ["question", "answer"],
    "extractive_qa": ["question", "context", "answers"],
    # Retrieval / ranking
    "retrieval": ["query", "positive"],
    "ranking": ["query", "candidates"],
    # Tabular / sklearn
    "tabular": ["target"],
    # Generic
    "text_pair": ["text1", "text2"],
}

# Formats where class balance is automatically checked (as warning)
_CLASSIFICATION_FORMATS: dict[str, str] = {
    "classification": "label",
    "nli": "label",
    "sentiment": "label",
}


@dataclass
class ValidatorConfig:
    """Configuration for dataset validation."""

    required_fields: list[str] | None = None
    format: str | None = None
    max_duplicate_ratio: float = 0.01
    max_empty_ratio: float = 0.05
    check_class_balance: bool = False
    label_field: str | None = None
    eval_path: str | None = None  # For split leakage check
    prompt_field: str = "prompt"  # Field to use for leakage comparison


class DatasetValidator:
    """Validate datasets against configurable checks."""

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self.config = config or ValidatorConfig()

    def validate(
        self,
        path: str,
        sample_n: int | None = None,
    ) -> ValidationReport:
        """Run all configured checks and return a validation report."""
        records, fmt = load_dataset(path, sample_n=sample_n)

        checks: list[ValidationCheck] = []

        if not records:
            checks.append(ValidationCheck(
                name="non_empty",
                passed=False,
                severity="error",
                message="Dataset is empty (0 records).",
            ))
            return ValidationReport(checks=checks)

        # Non-empty check
        checks.append(ValidationCheck(
            name="non_empty",
            passed=True,
            severity="info",
            message=f"Dataset has {len(records)} records.",
        ))

        # Required fields
        checks.extend(self._check_required_fields(records))

        # Format validation
        if self.config.format:
            checks.extend(self._check_format(records, self.config.format))

        # Duplicates
        checks.extend(self._check_duplicates(records))

        # Empty fields
        checks.extend(self._check_empty_fields(records))

        # Type consistency
        checks.extend(self._check_type_consistency(records))

        # Class balance — explicit or auto for classification formats
        label_field = self.config.label_field
        do_balance = self.config.check_class_balance

        if not do_balance and not label_field and self.config.format in _CLASSIFICATION_FORMATS:
            do_balance = True
            label_field = _CLASSIFICATION_FORMATS[self.config.format]

        if do_balance and label_field:
            checks.extend(self._check_class_balance(records, label_field))

        # Split leakage
        if self.config.eval_path:
            checks.extend(self._check_split_leakage(records, self.config.eval_path))

        return ValidationReport(checks=checks)

    def _check_required_fields(self, records: list[dict]) -> list[ValidationCheck]:
        """Check that all required fields are present in every record."""
        checks = []
        required = list(self.config.required_fields or [])

        # Add format-based requirements
        if self.config.format and self.config.format in FORMAT_REQUIREMENTS:
            for f in FORMAT_REQUIREMENTS[self.config.format]:
                if f not in required:
                    required.append(f)

        if not required:
            return checks

        for field_name in required:
            missing_count = sum(1 for r in records if field_name not in r)
            if missing_count == 0:
                checks.append(ValidationCheck(
                    name=f"required_field:{field_name}",
                    passed=True,
                    severity="error",
                    message=f"Field '{field_name}' present in all records.",
                ))
            else:
                checks.append(ValidationCheck(
                    name=f"required_field:{field_name}",
                    passed=False,
                    severity="error",
                    message=f"Field '{field_name}' missing in {missing_count}/{len(records)} records.",
                    details={"missing_count": missing_count, "total": len(records)},
                ))

        return checks

    def _check_format(self, records: list[dict], fmt: str) -> list[ValidationCheck]:
        """Validate format-specific structural constraints."""
        checks = []

        if fmt == "chat":
            bad_count = 0
            for rec in records:
                msgs = rec.get("messages")
                if not isinstance(msgs, list):
                    bad_count += 1
                    continue
                for msg in msgs:
                    if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                        bad_count += 1
                        break

            checks.append(ValidationCheck(
                name="format:chat_messages",
                passed=bad_count == 0,
                severity="error",
                message=(
                    "All records have valid chat messages."
                    if bad_count == 0
                    else f"{bad_count}/{len(records)} records have invalid messages format."
                ),
                details={"bad_count": bad_count} if bad_count else None,
            ))

        if fmt == "dpo":
            bad = 0
            for rec in records:
                if _is_empty(rec.get("chosen")) or _is_empty(rec.get("rejected")):
                    bad += 1

            checks.append(ValidationCheck(
                name="format:dpo_pairs",
                passed=bad == 0,
                severity="error",
                message=(
                    "All DPO records have non-empty chosen/rejected."
                    if bad == 0
                    else f"{bad}/{len(records)} records have empty chosen or rejected."
                ),
            ))

        if fmt in ("ner", "token_classification"):
            bad = 0
            for rec in records:
                tokens = rec.get("tokens")
                tags = rec.get("tags")
                if isinstance(tokens, list) and isinstance(tags, list) and len(tokens) != len(tags):
                    bad += 1

            if bad > 0:
                checks.append(ValidationCheck(
                    name=f"format:{fmt}_length_mismatch",
                    passed=False,
                    severity="error",
                    message=f"{bad}/{len(records)} records have tokens/tags length mismatch.",
                    details={"mismatched_count": bad},
                ))

        if fmt == "multi_label":
            bad = sum(1 for rec in records if not isinstance(rec.get("labels"), list))
            if bad > 0:
                checks.append(ValidationCheck(
                    name="format:multi_label_type",
                    passed=False,
                    severity="warning",
                    message=f"{bad}/{len(records)} records have 'labels' that is not a list.",
                    details={"bad_count": bad},
                ))

        if fmt == "ranking":
            bad = sum(1 for rec in records if not isinstance(rec.get("candidates"), list))
            if bad > 0:
                checks.append(ValidationCheck(
                    name="format:ranking_candidates_type",
                    passed=False,
                    severity="warning",
                    message=f"{bad}/{len(records)} records have 'candidates' that is not a list.",
                    details={"bad_count": bad},
                ))

        if fmt == "extractive_qa":
            bad = 0
            for rec in records:
                answers = rec.get("answers")
                if not isinstance(answers, (list, dict)):
                    bad += 1
            if bad > 0:
                checks.append(ValidationCheck(
                    name="format:extractive_qa_answers_type",
                    passed=False,
                    severity="warning",
                    message=f"{bad}/{len(records)} records have 'answers' that is not a list or dict.",
                    details={"bad_count": bad},
                ))

        if fmt == "tabular":
            # Report feature count (all fields except target)
            all_fields: set[str] = set()
            for rec in records:
                all_fields.update(rec.keys())
            feature_fields = all_fields - {"target"}
            checks.append(ValidationCheck(
                name="format:tabular_features",
                passed=True,
                severity="info",
                message=f"Tabular dataset with {len(feature_fields)} feature columns.",
                details={"features": sorted(feature_fields)},
            ))

        return checks

    def _check_duplicates(self, records: list[dict]) -> list[ValidationCheck]:
        """Check for duplicate records."""
        hashes = []
        for rec in records:
            h = hashlib.md5(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
            hashes.append(h)

        total = len(hashes)
        unique = len(set(hashes))
        dup_count = total - unique
        dup_ratio = dup_count / total if total > 0 else 0.0

        passed = dup_ratio <= self.config.max_duplicate_ratio

        return [ValidationCheck(
            name="duplicates",
            passed=passed,
            severity="error" if not passed else "info",
            message=(
                f"No significant duplicates ({dup_count}/{total}, {dup_ratio:.1%})."
                if passed
                else f"High duplicate ratio: {dup_count}/{total} ({dup_ratio:.1%}) > {self.config.max_duplicate_ratio:.1%} threshold."
            ),
            details={"duplicate_count": dup_count, "ratio": round(dup_ratio, 4)},
        )]

    def _check_empty_fields(self, records: list[dict]) -> list[ValidationCheck]:
        """Check for fields with too many empty/null values."""
        checks = []
        if not records:
            return checks

        all_fields: set[str] = set()
        for rec in records:
            all_fields.update(rec.keys())

        for field_name in sorted(all_fields):
            empty_count = 0
            for rec in records:
                v = rec.get(field_name)
                if _is_empty(v):
                    empty_count += 1

            ratio = empty_count / len(records)
            passed = ratio <= self.config.max_empty_ratio

            if not passed:
                checks.append(ValidationCheck(
                    name=f"empty_field:{field_name}",
                    passed=False,
                    severity="warning",
                    message=f"Field '{field_name}' has {empty_count}/{len(records)} ({ratio:.1%}) empty values.",
                    details={"empty_count": empty_count, "ratio": round(ratio, 4)},
                ))

        return checks

    def _check_type_consistency(self, records: list[dict]) -> list[ValidationCheck]:
        """Warn when a field has mixed types across records."""
        checks = []
        if not records:
            return checks

        all_fields: set[str] = set()
        for rec in records:
            all_fields.update(rec.keys())

        for field_name in sorted(all_fields):
            type_counts: Counter[str] = Counter()
            for rec in records:
                if field_name in rec:
                    v = rec[field_name]
                    type_counts[type(v).__name__] += 1

            if len(type_counts) > 1:
                total = sum(type_counts.values())
                dominant_type, dominant_count = type_counts.most_common(1)[0]
                minority_count = total - dominant_count
                minority_ratio = minority_count / total

                # Only warn if minority is significant (>1%)
                if minority_ratio > 0.01:
                    checks.append(ValidationCheck(
                        name=f"type_consistency:{field_name}",
                        passed=False,
                        severity="warning",
                        message=(
                            f"Field '{field_name}' has mixed types: "
                            f"{dict(type_counts.most_common())} "
                            f"({minority_ratio:.1%} minority)."
                        ),
                        details={
                            "type_distribution": dict(type_counts.most_common()),
                            "minority_ratio": round(minority_ratio, 4),
                        },
                    ))

        return checks

    def _check_class_balance(
        self, records: list[dict], label_field: str | None = None,
    ) -> list[ValidationCheck]:
        """Check class distribution balance."""
        label_field = label_field or self.config.label_field
        if not label_field:
            return []

        counter: Counter[str] = Counter()
        missing = 0
        for rec in records:
            v = rec.get(label_field)
            if v is None:
                missing += 1
            else:
                counter[str(v)] += 1

        if not counter:
            return [ValidationCheck(
                name="class_balance",
                passed=False,
                severity="warning",
                message=f"No values found for label field '{label_field}'.",
            )]

        min_count = min(counter.values())
        max_count = max(counter.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        passed = imbalance_ratio <= 10.0

        return [ValidationCheck(
            name="class_balance",
            passed=passed,
            severity="warning" if not passed else "info",
            message=(
                f"Class distribution OK (ratio {imbalance_ratio:.1f}x, {len(counter)} classes)."
                if passed
                else f"Class imbalance: {imbalance_ratio:.1f}x ratio ({dict(counter.most_common())})."
            ),
            details={"distribution": dict(counter.most_common()), "imbalance_ratio": round(imbalance_ratio, 2)},
        )]

    def _check_split_leakage(
        self,
        train_records: list[dict],
        eval_path: str,
    ) -> list[ValidationCheck]:
        """Check for data leakage between train and eval splits."""
        try:
            eval_records, _ = load_dataset(eval_path)
        except Exception as e:
            return [ValidationCheck(
                name="split_leakage",
                passed=False,
                severity="error",
                message=f"Could not load eval split: {e}",
            )]

        prompt_field = self.config.prompt_field

        train_prompts = {
            rec.get(prompt_field, "").strip().lower()
            for rec in train_records
            if rec.get(prompt_field)
        }
        eval_prompts = {
            rec.get(prompt_field, "").strip().lower()
            for rec in eval_records
            if rec.get(prompt_field)
        }

        overlap = train_prompts & eval_prompts
        overlap_count = len(overlap)

        passed = overlap_count == 0

        return [ValidationCheck(
            name="split_leakage",
            passed=passed,
            severity="error" if not passed else "info",
            message=(
                "No split leakage detected."
                if passed
                else f"Split leakage: {overlap_count} overlapping prompts between train and eval."
            ),
            details={
                "overlap_count": overlap_count,
                "examples": list(overlap)[:5],
            } if not passed else None,
        )]


def _is_empty(v: Any) -> bool:
    """Check if a value is considered empty (None, empty string, empty list/dict).

    Numeric zero (0, 0.0) and False are NOT considered empty,
    as they are valid values in ML datasets.
    """
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False
