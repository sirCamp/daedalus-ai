"""Unified dataset loader for JSONL, CSV, JSON, Parquet, and HuggingFace."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


def load_dataset(
    path: str,
    sample_n: int | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], str]:
    """Load a dataset from file or HuggingFace Hub.

    Args:
        path: File path (JSONL/CSV/JSON/Parquet) or HuggingFace dataset ID.
        sample_n: If set, randomly sample N records.
        seed: Random seed for sampling.

    Returns:
        Tuple of (records, format_name).
    """
    # HuggingFace Hub
    if path.startswith("hf://") or (not Path(path).suffix and "/" in path):
        return _load_huggingface(path, sample_n, seed)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        records = _load_jsonl(p)
        fmt = "jsonl"
    elif suffix == ".json":
        records = _load_json(p)
        fmt = "json"
    elif suffix == ".csv":
        records = _load_csv(p)
        fmt = "csv"
    elif suffix == ".parquet":
        records = _load_parquet(p)
        fmt = "parquet"
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .jsonl, .json, .csv, or .parquet")

    if sample_n and len(records) > sample_n:
        rng = random.Random(seed)
        records = rng.sample(records, sample_n)

    return records, fmt


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("JSON file must contain an array of objects")


def _load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet files. "
            "Install with: pip install daedalus[data]"
        )
    table = pq.read_table(str(path))
    return table.to_pylist()


def _load_huggingface(
    path: str,
    sample_n: int | None,
    seed: int,
) -> tuple[list[dict], str]:
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for HuggingFace datasets. "
            "Install with: pip install daedalus[data]"
        )

    dataset_id = path.removeprefix("hf://")
    ds = hf_load(dataset_id)

    # Get the first split
    if hasattr(ds, "keys"):
        split_name = list(ds.keys())[0]
        ds = ds[split_name]

    records = [dict(row) for row in ds]

    if sample_n and len(records) > sample_n:
        rng = random.Random(seed)
        records = rng.sample(records, sample_n)

    return records, "huggingface"
