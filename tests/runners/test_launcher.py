"""Tests for multi-GPU launcher command building."""

import pytest

from daedalus.core.config import ExperimentConfig
from daedalus.runners.base import build_env_vars, build_launch_cmd


class TestBuildLaunchCmd:
    def test_default_python(self):
        config = ExperimentConfig(script="train.py", script_args={"lr": 0.001})
        cmd = build_launch_cmd(config)
        assert cmd == ["python", "train.py", "--lr", "0.001"]

    def test_custom_python(self):
        config = ExperimentConfig(script="train.py")
        cmd = build_launch_cmd(config, python="/opt/conda/bin/python")
        assert cmd == ["/opt/conda/bin/python", "train.py"]

    def test_torchrun(self):
        config = ExperimentConfig(
            script="train.py", launcher="torchrun", num_gpus=4,
            script_args={"epochs": 10},
        )
        cmd = build_launch_cmd(config)
        assert cmd == ["torchrun", "--nproc_per_node=4", "train.py", "--epochs", "10"]

    def test_torchrun_default_gpus(self):
        config = ExperimentConfig(script="train.py", launcher="torchrun")
        cmd = build_launch_cmd(config)
        assert "--nproc_per_node=1" in cmd

    def test_accelerate(self):
        config = ExperimentConfig(
            script="train.py", launcher="accelerate", num_gpus=8,
        )
        cmd = build_launch_cmd(config)
        assert cmd == ["accelerate", "launch", "--num_processes=8", "train.py"]

    def test_accelerate_auto(self):
        """accelerate without num_gpus uses all available."""
        config = ExperimentConfig(script="train.py", launcher="accelerate")
        cmd = build_launch_cmd(config)
        assert cmd == ["accelerate", "launch", "train.py"]

    def test_deepspeed(self):
        config = ExperimentConfig(
            script="train.py", launcher="deepspeed", num_gpus=2,
            script_args={"config": "ds_config.json"},
        )
        cmd = build_launch_cmd(config)
        assert cmd == ["deepspeed", "--num_gpus=2", "train.py", "--config", "ds_config.json"]

    def test_unknown_launcher_raises(self):
        config = ExperimentConfig(script="train.py", launcher="unknown")
        with pytest.raises(ValueError, match="Unknown launcher"):
            build_launch_cmd(config)


class TestBuildEnvVars:
    def test_no_gpu_ids(self):
        config = ExperimentConfig(script="train.py", env={"WANDB_PROJECT": "test"})
        env = build_env_vars(config)
        assert env == {"WANDB_PROJECT": "test"}
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_gpu_ids(self):
        config = ExperimentConfig(script="train.py", gpu_ids=[0, 2, 3])
        env = build_env_vars(config)
        assert env["CUDA_VISIBLE_DEVICES"] == "0,2,3"

    def test_gpu_ids_single(self):
        config = ExperimentConfig(script="train.py", gpu_ids=[1])
        env = build_env_vars(config)
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_gpu_ids_with_existing_env(self):
        config = ExperimentConfig(
            script="train.py",
            env={"WANDB_MODE": "offline"},
            gpu_ids=[0, 1],
        )
        env = build_env_vars(config)
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert env["WANDB_MODE"] == "offline"
