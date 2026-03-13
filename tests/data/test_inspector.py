"""Tests for dataset inspector."""

import json
import pytest

from daedalus.data.inspector import DatasetInspector


class TestDatasetInspector:
    def _write_jsonl(self, tmp_path, records):
        f = tmp_path / "data.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in records))
        return str(f)

    def test_basic_inspect(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"prompt": "q1", "score": 1.0},
            {"prompt": "q2", "score": 2.0},
            {"prompt": "q3", "score": 3.0},
        ])

        inspector = DatasetInspector()
        report = inspector.inspect(path)

        assert report.row_count == 3
        assert report.column_count == 2
        assert report.format == "jsonl"
        assert len(report.columns) == 2

    def test_column_names(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"a": 1, "b": "x", "c": [1, 2]},
        ])

        inspector = DatasetInspector()
        report = inspector.inspect(path)
        names = [c.name for c in report.columns]
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_numeric_stats(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"val": 10},
            {"val": 20},
            {"val": 30},
        ])

        inspector = DatasetInspector()
        report = inspector.inspect(path)
        col = [c for c in report.columns if c.name == "val"][0]

        assert col.min_value == 10.0
        assert col.max_value == 30.0
        assert col.mean_value == 20.0
        assert col.dtype == "int"

    def test_string_stats(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"label": "pos"},
            {"label": "neg"},
            {"label": "pos"},
            {"label": "pos"},
        ])

        inspector = DatasetInspector()
        report = inspector.inspect(path)
        col = [c for c in report.columns if c.name == "label"][0]

        assert col.unique_count == 2
        assert col.dtype == "str"
        # Top values should show "pos" as most common
        assert col.top_values[0] == ("pos", 3)

    def test_null_detection(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"a": 1, "b": "x"},
            {"a": 2},           # b is missing
            {"a": None, "b": "y"},  # a is null
        ])

        inspector = DatasetInspector()
        report = inspector.inspect(path)

        col_a = [c for c in report.columns if c.name == "a"][0]
        col_b = [c for c in report.columns if c.name == "b"][0]

        assert col_a.null_count == 1
        assert col_b.null_count == 1

    def test_sample_records(self, tmp_path):
        records = [{"i": i} for i in range(20)]
        path = self._write_jsonl(tmp_path, records)

        inspector = DatasetInspector()
        report = inspector.inspect(path, num_samples=5)

        assert len(report.sample_records) == 5
        assert report.sample_records[0]["i"] == 0

    def test_token_estimate(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"text": "Hello world " * 100},
        ])

        inspector = DatasetInspector()
        report = inspector.inspect(path)
        assert report.estimated_tokens is not None
        assert report.estimated_tokens > 0

    def test_file_size(self, tmp_path):
        path = self._write_jsonl(tmp_path, [{"a": 1}])

        inspector = DatasetInspector()
        report = inspector.inspect(path)
        assert report.file_size_bytes is not None
        assert report.file_size_bytes > 0

    def test_empty_dataset(self, tmp_path):
        path = self._write_jsonl(tmp_path, [])

        inspector = DatasetInspector()
        report = inspector.inspect(path)
        assert report.row_count == 0
        assert report.column_count == 0

    def test_mixed_types(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"val": 1},
            {"val": "text"},
            {"val": 3.14},
        ])

        inspector = DatasetInspector()
        report = inspector.inspect(path)
        col = [c for c in report.columns if c.name == "val"][0]
        assert col.dtype == "mixed"

    def test_empty_strings_counted(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            {"text": "hello"},
            {"text": ""},
            {"text": "  "},
        ])

        inspector = DatasetInspector()
        report = inspector.inspect(path)
        col = [c for c in report.columns if c.name == "text"][0]
        assert col.empty_string_count == 2
