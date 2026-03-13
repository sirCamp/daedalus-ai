"""Tests for dataset loader."""

import json
import pytest

from daedalus.data.loader import load_dataset


class TestLoadDataset:
    def test_load_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        records = [{"prompt": "q1", "completion": "a1"}, {"prompt": "q2", "completion": "a2"}]
        f.write_text("\n".join(json.dumps(r) for r in records))

        result, fmt = load_dataset(str(f))
        assert fmt == "jsonl"
        assert len(result) == 2
        assert result[0]["prompt"] == "q1"

    def test_load_json(self, tmp_path):
        f = tmp_path / "data.json"
        records = [{"a": 1}, {"a": 2}, {"a": 3}]
        f.write_text(json.dumps(records))

        result, fmt = load_dataset(str(f))
        assert fmt == "json"
        assert len(result) == 3

    def test_load_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")

        result, fmt = load_dataset(str(f))
        assert fmt == "csv"
        assert len(result) == 2
        assert result[0]["name"] == "Alice"

    def test_sample_n(self, tmp_path):
        f = tmp_path / "data.jsonl"
        records = [{"i": i} for i in range(100)]
        f.write_text("\n".join(json.dumps(r) for r in records))

        result, fmt = load_dataset(str(f), sample_n=10)
        assert len(result) == 10

    def test_sample_n_larger_than_dataset(self, tmp_path):
        f = tmp_path / "data.jsonl"
        records = [{"i": i} for i in range(5)]
        f.write_text("\n".join(json.dumps(r) for r in records))

        result, fmt = load_dataset(str(f), sample_n=100)
        assert len(result) == 5  # Returns all, no sampling needed

    def test_unknown_format_raises(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("hello")

        with pytest.raises(ValueError, match="Unsupported format"):
            load_dataset(str(f))

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path.jsonl")

    def test_parquet_not_installed(self, tmp_path, monkeypatch):
        f = tmp_path / "data.parquet"
        f.write_text("fake")

        import daedalus.data.loader as loader_mod
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyarrow.parquet":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="pyarrow"):
            load_dataset(str(f))

    def test_empty_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text("")

        result, fmt = load_dataset(str(f))
        assert result == []
        assert fmt == "jsonl"
