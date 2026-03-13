"""Results adapters — read results from different training frameworks.

Training scripts produce results in different formats. Adapters normalize
them to the Daedalus standard: ``{eval_name: {metric: value}}``.

Supported formats:
- daedalus: ``results.json`` with ``{eval_name: {metric: value}}``
- hf_trainer: HuggingFace Trainer's ``eval_results.json`` or ``trainer_state.json``
- csv: A CSV file with ``metric,value`` columns
- json_flat: A flat JSON dict ``{metric: value}`` (single eval)
- tensorboard: Read scalar summaries from TensorBoard event files
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_results(
    work_dir: Path,
    adapter: str = "auto",
    eval_name: str = "eval",
) -> dict[str, dict[str, float]]:
    """Read experiment results from a work directory.

    Args:
        work_dir: Experiment work directory.
        adapter: Adapter type. 'auto' tries to detect the format.
        eval_name: Default eval name for flat formats.

    Returns:
        Normalized results: ``{eval_name: {metric: value}}``.
    """
    work_dir = Path(work_dir)

    if adapter == "auto":
        adapter = _detect_adapter(work_dir)

    adapters = {
        "daedalus": _read_daedalus,
        "hf_trainer": _read_hf_trainer,
        "csv": _read_csv,
        "json_flat": _read_json_flat,
    }

    reader = adapters.get(adapter)
    if reader is None:
        raise ValueError(
            f"Unknown results adapter: '{adapter}'. "
            f"Available: {', '.join(adapters.keys())}"
        )

    return reader(work_dir, eval_name)


def _detect_adapter(work_dir: Path) -> str:
    """Auto-detect the results format from files in work_dir."""
    if (work_dir / "results.json").exists():
        # Could be daedalus or json_flat — check structure
        try:
            data = json.loads((work_dir / "results.json").read_text())
            if isinstance(data, dict):
                # If all values are dicts, it's daedalus format
                if all(isinstance(v, dict) for v in data.values()):
                    return "daedalus"
                # If values are numbers, it's json_flat
                if any(isinstance(v, (int, float)) for v in data.values()):
                    return "json_flat"
            return "daedalus"
        except (json.JSONDecodeError, OSError):
            return "daedalus"

    # HuggingFace Trainer outputs
    if (work_dir / "eval_results.json").exists():
        return "hf_trainer"
    if (work_dir / "trainer_state.json").exists():
        return "hf_trainer"

    # Check subdirectories (HF Trainer often puts results in checkpoint dirs)
    for subdir in sorted(work_dir.iterdir()):
        if subdir.is_dir() and (subdir / "eval_results.json").exists():
            return "hf_trainer"

    if (work_dir / "results.csv").exists():
        return "csv"

    raise FileNotFoundError(
        f"No recognized results files in {work_dir}. "
        f"Expected one of: results.json, eval_results.json, trainer_state.json, results.csv"
    )


def _read_daedalus(work_dir: Path, eval_name: str) -> dict[str, dict[str, float]]:
    """Read Daedalus native format: {eval_name: {metric: value}}."""
    data = json.loads((work_dir / "results.json").read_text())
    # Validate and coerce values to float
    results: dict[str, dict[str, float]] = {}
    for en, metrics in data.items():
        if isinstance(metrics, dict):
            results[en] = {k: float(v) for k, v in metrics.items() if _is_numeric(v)}
    return results


def _read_hf_trainer(work_dir: Path, eval_name: str) -> dict[str, dict[str, float]]:
    """Read HuggingFace Trainer results.

    HF Trainer produces:
    - eval_results.json: {eval_metric: value} (prefixed with eval_)
    - trainer_state.json: training history with log_history
    """
    results: dict[str, dict[str, float]] = {}

    # Try eval_results.json first (in work_dir or any checkpoint subdir)
    eval_files = list(work_dir.glob("**/eval_results.json"))
    if eval_files:
        # Use the most recent one
        eval_file = sorted(eval_files)[-1]
        data = json.loads(eval_file.read_text())
        metrics: dict[str, float] = {}
        for k, v in data.items():
            if _is_numeric(v):
                # Strip eval_ prefix that HF adds
                clean_key = k.removeprefix("eval_")
                metrics[clean_key] = float(v)
        if metrics:
            results[eval_name] = metrics

    # Also try trainer_state.json for training metrics
    state_file = work_dir / "trainer_state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            log_history = state.get("log_history", [])
            if log_history:
                # Get the last eval entry
                eval_entries = [e for e in log_history if any(k.startswith("eval_") for k in e)]
                if eval_entries:
                    last_eval = eval_entries[-1]
                    eval_metrics = {
                        k.removeprefix("eval_"): float(v)
                        for k, v in last_eval.items()
                        if k.startswith("eval_") and _is_numeric(v)
                    }
                    if eval_metrics and eval_name not in results:
                        results[eval_name] = eval_metrics

                # Get final training metrics
                train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
                if train_entries:
                    last_train = train_entries[-1]
                    train_metrics = {
                        k: float(v) for k, v in last_train.items()
                        if _is_numeric(v) and k not in ("epoch", "step")
                    }
                    if train_metrics:
                        results["train"] = train_metrics
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not parse trainer_state.json: {e}")

    if not results:
        raise FileNotFoundError(f"No eval results found in {work_dir}")

    return results


def _read_csv(work_dir: Path, eval_name: str) -> dict[str, dict[str, float]]:
    """Read CSV results: metric,value columns."""
    csv_file = work_dir / "results.csv"
    metrics: dict[str, float] = {}

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("metric", row.get("name", ""))
            value = row.get("value", row.get("score", ""))
            if name and _is_numeric(value):
                metrics[name] = float(value)

    return {eval_name: metrics} if metrics else {}


def _read_json_flat(work_dir: Path, eval_name: str) -> dict[str, dict[str, float]]:
    """Read flat JSON: {metric: value}."""
    data = json.loads((work_dir / "results.json").read_text())
    metrics = {k: float(v) for k, v in data.items() if _is_numeric(v)}
    return {eval_name: metrics}


def _is_numeric(v: Any) -> bool:
    """Check if a value is numeric."""
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False
