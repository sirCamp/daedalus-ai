"""Experiment configuration with deep diff support."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ResourceSpec(BaseModel):
    """Estimated resource requirements for an experiment."""

    gpus: int = 1
    gpu_type: str | None = None
    estimated_hours: float | None = None
    estimated_cost_usd: float | None = None


class ExperimentConfig(BaseModel):
    """Flexible experiment configuration.

    Stores the training script path, its arguments, environment variables,
    and optional resource estimates. Extra fields are allowed for
    project-specific configuration.

    For multi-step workflows (train → eval), use ``eval_script`` and
    ``eval_script_args`` to define a separate evaluation step.

    Multi-GPU support via ``launcher``:
    - ``"python"`` (default): plain ``python script.py``
    - ``"torchrun"``: ``torchrun --nproc_per_node=N script.py``
    - ``"accelerate"``: ``accelerate launch --num_processes=N script.py``
    - ``"deepspeed"``: ``deepspeed --num_gpus=N script.py``
    """

    model_config = ConfigDict(extra="allow")

    script: str
    script_args: dict[str, Any] = {}
    env: dict[str, str] = {}
    resources: ResourceSpec | None = None
    # Multi-GPU launcher
    launcher: str = "python"  # "python", "torchrun", "accelerate", "deepspeed"
    num_gpus: int | None = None  # None = 1 GPU (or all for accelerate auto)
    gpu_ids: list[int] | None = None  # e.g. [0, 2] → CUDA_VISIBLE_DEVICES=0,2
    # Optional separate evaluation step
    eval_script: str | None = None
    eval_script_args: dict[str, Any] = {}
    # Results format hint for auto-detection
    results_adapter: str = "auto"  # "auto", "daedalus", "hf_trainer", "csv", "json_flat"


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict with dot-separated keys."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items


def config_diff(
    old: ExperimentConfig,
    new: ExperimentConfig,
) -> dict[str, dict[str, Any]]:
    """Compute a JSON-serializable diff between two configs.

    Returns a dict of changed keys with ``{"from": old_value, "to": new_value}``
    for every leaf value that differs, using dot notation for nested paths.
    """
    old_flat = _flatten(old.model_dump())
    new_flat = _flatten(new.model_dump())

    all_keys = set(old_flat) | set(new_flat)
    diff: dict[str, dict[str, Any]] = {}

    for key in sorted(all_keys):
        old_val = old_flat.get(key)
        new_val = new_flat.get(key)
        if old_val != new_val:
            diff[key] = {"from": old_val, "to": new_val}

    return diff
