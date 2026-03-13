"""Tests for dataset validator."""

import json
import pytest

from daedalus.data.validator import DatasetValidator, ValidatorConfig


class TestDatasetValidator:
    def _write_jsonl(self, tmp_path, records, name="data.jsonl"):
        f = tmp_path / name
        f.write_text("\n".join(json.dumps(r) for r in records))
        return str(f)

    def test_required_fields_pass(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"prompt": "q1", "completion": "a1"},
            {"prompt": "q2", "completion": "a2"},
        ])

        config = ValidatorConfig(required_fields=["prompt", "completion"])
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed

    def test_required_fields_fail(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"prompt": "q1", "completion": "a1"},
            {"prompt": "q2"},  # missing completion
        ])

        config = ValidatorConfig(required_fields=["prompt", "completion"])
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert not report.passed
        assert report.error_count == 1

    def test_format_sft_valid(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"prompt": "q1", "completion": "a1"},
        ])

        config = ValidatorConfig(format="sft")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed

    def test_format_dpo_missing_rejected(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"prompt": "q1", "chosen": "a1"},  # missing rejected
        ])

        config = ValidatorConfig(format="dpo")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert not report.passed

    def test_format_chat_valid(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        ])

        config = ValidatorConfig(format="chat")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed

    def test_format_chat_invalid(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"messages": "not a list"},
        ])

        config = ValidatorConfig(format="chat")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert not report.passed

    def test_duplicates_pass(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"prompt": f"q{i}", "completion": f"a{i}"} for i in range(100)
        ])

        config = ValidatorConfig(max_duplicate_ratio=0.01)
        validator = DatasetValidator(config)
        report = validator.validate(path)
        dup_check = [c for c in report.checks if c.name == "duplicates"][0]
        assert dup_check.passed

    def test_duplicates_fail(self, tmp_path):
        records = [{"prompt": "q1", "completion": "a1"}] * 10
        records.extend([{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(10)])
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(max_duplicate_ratio=0.01)
        validator = DatasetValidator(config)
        report = validator.validate(path)
        dup_check = [c for c in report.checks if c.name == "duplicates"][0]
        assert not dup_check.passed

    def test_empty_fields_warning(self, tmp_path):
        records = [{"prompt": "q1", "completion": "a1"}] * 8
        records.extend([{"prompt": "q", "completion": ""}] * 2)
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(max_empty_ratio=0.05)
        validator = DatasetValidator(config)
        report = validator.validate(path)
        empty_checks = [c for c in report.checks if c.name.startswith("empty_field:")]
        # completion has 20% empty, should warn
        comp_check = [c for c in empty_checks if "completion" in c.name]
        assert len(comp_check) == 1
        assert not comp_check[0].passed

    def test_class_balance_ok(self, tmp_path):
        records = [{"text": "x", "label": "pos"}] * 5 + [{"text": "x", "label": "neg"}] * 5
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(check_class_balance=True, label_field="label")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        balance_check = [c for c in report.checks if c.name == "class_balance"][0]
        assert balance_check.passed

    def test_class_balance_imbalanced(self, tmp_path):
        records = [{"text": "x", "label": "pos"}] * 100 + [{"text": "x", "label": "neg"}] * 2
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(check_class_balance=True, label_field="label")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        balance_check = [c for c in report.checks if c.name == "class_balance"][0]
        assert not balance_check.passed

    def test_split_leakage_detected(self, tmp_path):
        train = self._write_jsonl(tmp_path, [
            {"prompt": "What is AI?", "completion": "a"},
            {"prompt": "What is ML?", "completion": "b"},
        ], name="train.jsonl")
        eval_path = self._write_jsonl(tmp_path, [
            {"prompt": "What is AI?", "completion": "c"},  # overlap!
            {"prompt": "What is NLP?", "completion": "d"},
        ], name="eval.jsonl")

        config = ValidatorConfig(eval_path=eval_path)
        validator = DatasetValidator(config)
        report = validator.validate(train)
        leak_check = [c for c in report.checks if c.name == "split_leakage"][0]
        assert not leak_check.passed

    def test_split_leakage_clean(self, tmp_path):
        train = self._write_jsonl(tmp_path, [
            {"prompt": "What is AI?", "completion": "a"},
        ], name="train.jsonl")
        eval_path = self._write_jsonl(tmp_path, [
            {"prompt": "What is NLP?", "completion": "b"},
        ], name="eval.jsonl")

        config = ValidatorConfig(eval_path=eval_path)
        validator = DatasetValidator(config)
        report = validator.validate(train)
        leak_check = [c for c in report.checks if c.name == "split_leakage"][0]
        assert leak_check.passed

    def test_split_leakage_custom_prompt_field(self, tmp_path):
        train = self._write_jsonl(tmp_path, [
            {"question": "What is AI?", "answer": "a"},
        ], name="train.jsonl")
        eval_path = self._write_jsonl(tmp_path, [
            {"question": "What is AI?", "answer": "b"},
        ], name="eval.jsonl")
        config = ValidatorConfig(eval_path=eval_path, prompt_field="question")
        report = DatasetValidator(config).validate(train)
        leak_check = [c for c in report.checks if c.name == "split_leakage"][0]
        assert not leak_check.passed

    def test_report_passed_property(self, tmp_path):
        path = self._write_jsonl(tmp_path, [{"prompt": "q"}])

        # All pass
        config = ValidatorConfig(required_fields=["prompt"])
        report = DatasetValidator(config).validate(path)
        assert report.passed

        # One error
        config = ValidatorConfig(required_fields=["prompt", "missing_field"])
        report = DatasetValidator(config).validate(path)
        assert not report.passed

    def test_empty_dataset(self, tmp_path):
        path = self._write_jsonl(tmp_path, [])

        config = ValidatorConfig()
        report = DatasetValidator(config).validate(path)
        assert not report.passed
        assert report.error_count == 1

    def test_format_classification(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"text": "Great product!", "label": "positive"},
            {"text": "Terrible service", "label": "negative"},
        ])

        config = ValidatorConfig(format="classification")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed

    def test_format_classification_missing_label(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"text": "Great product!"},  # missing label
        ])

        config = ValidatorConfig(format="classification")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert not report.passed

    def test_format_nli(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"premise": "A man walks", "hypothesis": "A person moves", "label": "entailment"},
        ])

        config = ValidatorConfig(format="nli")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed

    def test_format_qa(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
        ])

        config = ValidatorConfig(format="qa")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed

    def test_format_qa_with_context(self, tmp_path):
        """QA with optional context should also pass."""
        path = self._write_jsonl(tmp_path, [
            {"question": "What is AI?", "context": "AI is...", "answer": "Artificial Intelligence"},
        ])

        config = ValidatorConfig(format="qa")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed

    def test_format_sts(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"sentence1": "Hello world", "sentence2": "Hi earth", "score": 0.8},
        ])

        config = ValidatorConfig(format="sts")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed

    def test_format_grpo(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"prompt": "q1"},
            {"prompt": "q2"},
        ])

        config = ValidatorConfig(format="grpo")
        validator = DatasetValidator(config)
        report = validator.validate(path)
        assert report.passed


class TestFormatSpecificChecks:
    """Tests for structural validation of specific formats."""

    def _write_jsonl(self, tmp_path, records, name="data.jsonl"):
        f = tmp_path / name
        f.write_text("\n".join(json.dumps(r) for r in records))
        return str(f)

    def test_ner_length_mismatch(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"tokens": ["Hello", "world"], "tags": ["O", "O"]},
            {"tokens": ["Hello", "world", "!"], "tags": ["O", "O"]},  # mismatch
        ])
        config = ValidatorConfig(format="ner")
        report = DatasetValidator(config).validate(path)
        mismatch = [c for c in report.checks if "length_mismatch" in c.name]
        assert len(mismatch) == 1
        assert not mismatch[0].passed
        assert mismatch[0].severity == "error"

    def test_ner_length_ok(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"tokens": ["Hello", "world"], "tags": ["O", "O"]},
        ])
        config = ValidatorConfig(format="ner")
        report = DatasetValidator(config).validate(path)
        mismatch = [c for c in report.checks if "length_mismatch" in c.name]
        assert len(mismatch) == 0

    def test_multi_label_list_check(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"text": "foo", "labels": ["a", "b"]},
            {"text": "bar", "labels": "not_a_list"},
        ])
        config = ValidatorConfig(format="multi_label")
        report = DatasetValidator(config).validate(path)
        type_check = [c for c in report.checks if "multi_label_type" in c.name]
        assert len(type_check) == 1
        assert not type_check[0].passed

    def test_multi_label_valid(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"text": "foo", "labels": ["a", "b"]},
            {"text": "bar", "labels": ["c"]},
        ])
        config = ValidatorConfig(format="multi_label")
        report = DatasetValidator(config).validate(path)
        type_check = [c for c in report.checks if "multi_label_type" in c.name]
        assert len(type_check) == 0

    def test_ranking_candidates_type(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"query": "q1", "candidates": ["a", "b"]},
            {"query": "q2", "candidates": "not_a_list"},
        ])
        config = ValidatorConfig(format="ranking")
        report = DatasetValidator(config).validate(path)
        check = [c for c in report.checks if "ranking_candidates" in c.name]
        assert len(check) == 1
        assert not check[0].passed

    def test_extractive_qa_answers_type(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"question": "q", "context": "c", "answers": {"text": ["a"], "answer_start": [0]}},
            {"question": "q2", "context": "c2", "answers": "wrong"},
        ])
        config = ValidatorConfig(format="extractive_qa")
        report = DatasetValidator(config).validate(path)
        check = [c for c in report.checks if "extractive_qa_answers" in c.name]
        assert len(check) == 1
        assert not check[0].passed

    def test_extractive_qa_valid(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"question": "q", "context": "c", "answers": {"text": ["a"], "answer_start": [0]}},
            {"question": "q2", "context": "c2", "answers": ["answer"]},
        ])
        config = ValidatorConfig(format="extractive_qa")
        report = DatasetValidator(config).validate(path)
        check = [c for c in report.checks if "extractive_qa_answers" in c.name]
        assert len(check) == 0

    def test_tabular_feature_report(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"age": 25, "income": 50000, "target": 1},
            {"age": 30, "income": 60000, "target": 0},
        ])
        config = ValidatorConfig(format="tabular")
        report = DatasetValidator(config).validate(path)
        feat_check = [c for c in report.checks if "tabular_features" in c.name]
        assert len(feat_check) == 1
        assert feat_check[0].passed
        assert "age" in feat_check[0].details["features"]
        assert "target" not in feat_check[0].details["features"]


class TestAutoClassBalance:
    """Class balance auto-activation for classification formats."""

    def _write_jsonl(self, tmp_path, records, name="data.jsonl"):
        f = tmp_path / name
        f.write_text("\n".join(json.dumps(r) for r in records))
        return str(f)

    def test_classification_auto_balance(self, tmp_path):
        """Classification format auto-checks class balance without explicit flag."""
        records = [{"text": "x", "label": "pos"}] * 50 + [{"text": "x", "label": "neg"}] * 1
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(format="classification")
        report = DatasetValidator(config).validate(path)
        balance = [c for c in report.checks if c.name == "class_balance"]
        assert len(balance) == 1
        assert not balance[0].passed
        assert balance[0].severity == "warning"

    def test_nli_auto_balance(self, tmp_path):
        records = (
            [{"premise": "p", "hypothesis": "h", "label": "entailment"}] * 10
            + [{"premise": "p", "hypothesis": "h", "label": "contradiction"}] * 10
            + [{"premise": "p", "hypothesis": "h", "label": "neutral"}] * 10
        )
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(format="nli")
        report = DatasetValidator(config).validate(path)
        balance = [c for c in report.checks if c.name == "class_balance"]
        assert len(balance) == 1
        assert balance[0].passed

    def test_sentiment_auto_balance(self, tmp_path):
        records = [{"text": "x", "label": "pos"}] * 5 + [{"text": "x", "label": "neg"}] * 5
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(format="sentiment")
        report = DatasetValidator(config).validate(path)
        balance = [c for c in report.checks if c.name == "class_balance"]
        assert len(balance) == 1
        assert balance[0].passed

    def test_no_auto_balance_for_non_classification(self, tmp_path):
        """SFT format should NOT auto-check class balance."""
        path = self._write_jsonl(tmp_path, [
            {"prompt": "q1", "completion": "a1"},
        ])
        config = ValidatorConfig(format="sft")
        report = DatasetValidator(config).validate(path)
        balance = [c for c in report.checks if c.name == "class_balance"]
        assert len(balance) == 0

    def test_explicit_overrides_auto(self, tmp_path):
        """Explicit check_class_balance + label_field overrides auto."""
        records = [{"text": "x", "label": "pos", "category": "A"}] * 5
        records += [{"text": "x", "label": "neg", "category": "B"}] * 5
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(
            format="classification",
            check_class_balance=True,
            label_field="category",
        )
        report = DatasetValidator(config).validate(path)
        balance = [c for c in report.checks if c.name == "class_balance"]
        assert len(balance) == 1


class TestEmptyValues:
    """Empty check should catch [], {}, None, and empty strings."""

    def _write_jsonl(self, tmp_path, records, name="data.jsonl"):
        f = tmp_path / name
        f.write_text("\n".join(json.dumps(r) for r in records))
        return str(f)

    def test_empty_list_detected(self, tmp_path):
        records = [{"text": "x", "tags": []}] * 10
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(max_empty_ratio=0.05)
        report = DatasetValidator(config).validate(path)
        empty = [c for c in report.checks if "empty_field:tags" in c.name]
        assert len(empty) == 1
        assert not empty[0].passed

    def test_empty_dict_detected(self, tmp_path):
        records = [{"text": "x", "meta": {}}] * 10
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(max_empty_ratio=0.05)
        report = DatasetValidator(config).validate(path)
        empty = [c for c in report.checks if "empty_field:meta" in c.name]
        assert len(empty) == 1

    def test_non_empty_list_ok(self, tmp_path):
        records = [{"text": "x", "tags": ["O"]}] * 10
        path = self._write_jsonl(tmp_path, records)

        config = ValidatorConfig(max_empty_ratio=0.05)
        report = DatasetValidator(config).validate(path)
        empty = [c for c in report.checks if "empty_field:tags" in c.name]
        assert len(empty) == 0


class TestTypeConsistency:
    """Type consistency warnings for mixed-type fields."""

    def _write_jsonl(self, tmp_path, records, name="data.jsonl"):
        f = tmp_path / name
        f.write_text("\n".join(json.dumps(r) for r in records))
        return str(f)

    def test_mixed_types_warns(self, tmp_path):
        records = [{"score": 0.5}] * 8 + [{"score": "high"}] * 2
        path = self._write_jsonl(tmp_path, records)

        report = DatasetValidator().validate(path)
        type_check = [c for c in report.checks if "type_consistency:score" in c.name]
        assert len(type_check) == 1
        assert not type_check[0].passed
        assert type_check[0].severity == "warning"

    def test_consistent_types_no_warning(self, tmp_path):
        records = [{"score": float(i)} for i in range(10)]
        path = self._write_jsonl(tmp_path, records)

        report = DatasetValidator().validate(path)
        type_check = [c for c in report.checks if "type_consistency" in c.name]
        assert len(type_check) == 0

    def test_tiny_minority_ignored(self, tmp_path):
        """Less than 1% minority type should not trigger warning."""
        records = [{"val": 1}] * 200 + [{"val": "x"}] * 1
        path = self._write_jsonl(tmp_path, records)

        report = DatasetValidator().validate(path)
        type_check = [c for c in report.checks if "type_consistency:val" in c.name]
        assert len(type_check) == 0
