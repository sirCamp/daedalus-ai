"""Abstract runner interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from ..core.config import ExperimentConfig
from ..core.experiment import Experiment


class RunStatus(BaseModel):
    """Status of a running experiment."""

    state: Literal["queued", "running", "completed", "failed"]
    progress: str | None = None
    eta: str | None = None
    error: str | None = None


def build_launch_cmd(
    config: ExperimentConfig,
    python: str = "python",
) -> list[str]:
    """Build the launch command respecting launcher and GPU settings.

    Returns command tokens list. The caller is responsible for joining
    them as needed (subprocess list or shell string).

    Launcher modes:
    - python (default): ``python script.py --args``
    - torchrun: ``torchrun --nproc_per_node=N script.py --args``
    - accelerate: ``accelerate launch [--num_processes=N] script.py --args``
    - deepspeed: ``deepspeed [--num_gpus=N] script.py --args``
    """
    launcher = config.launcher or "python"
    num_gpus = config.num_gpus
    script = config.script
    script_args = []
    for key, value in config.script_args.items():
        script_args.append(f"--{key}")
        script_args.append(str(value))

    if launcher == "python":
        return [python, script] + script_args

    if launcher == "torchrun":
        cmd = ["torchrun"]
        cmd.append(f"--nproc_per_node={num_gpus or 1}")
        return cmd + [script] + script_args

    if launcher == "accelerate":
        cmd = ["accelerate", "launch"]
        if num_gpus is not None:
            cmd.append(f"--num_processes={num_gpus}")
        return cmd + [script] + script_args

    if launcher == "deepspeed":
        cmd = ["deepspeed"]
        if num_gpus is not None:
            cmd.append(f"--num_gpus={num_gpus}")
        return cmd + [script] + script_args

    raise ValueError(f"Unknown launcher: {launcher}")


def build_env_vars(config: ExperimentConfig) -> dict[str, str]:
    """Build environment variables dict, including CUDA_VISIBLE_DEVICES."""
    env = dict(config.env)
    if config.gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpu_ids)
    return env


@runtime_checkable
class Runner(Protocol):
    """Protocol for experiment runners.

    Implementations must handle launching a training command,
    polling its status, fetching results, and cancellation.
    """

    def launch(self, experiment: Experiment, work_dir: Path) -> str:
        """Launch an experiment. Returns an opaque run_id."""
        ...

    def poll(self, run_id: str) -> RunStatus:
        """Check the status of a running experiment."""
        ...

    def fetch_results(self, run_id: str) -> dict[str, Any]:
        """Fetch results after experiment completion."""
        ...

    def cancel(self, run_id: str) -> None:
        """Cancel a running experiment."""
        ...

    def logs(self, run_id: str, tail: int = 50) -> str:
        """Retrieve recent log lines."""
        ...
