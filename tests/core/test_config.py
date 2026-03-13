"""Tests for ExperimentConfig and config_diff."""

import pytest

from daedalus.core.config import ExperimentConfig, ResourceSpec, config_diff


class TestExperimentConfig:
    def test_minimal(self):
        cfg = ExperimentConfig(script="train.py")
        assert cfg.script == "train.py"
        assert cfg.script_args == {}
        assert cfg.env == {}
        assert cfg.resources is None

    def test_full(self):
        cfg = ExperimentConfig(
            script="train.py",
            script_args={"lr": 5e-6, "batch_size": 16},
            env={"CUDA_VISIBLE_DEVICES": "0"},
            resources=ResourceSpec(gpus=2, gpu_type="A100", estimated_hours=4.0),
        )
        assert cfg.script_args["lr"] == 5e-6
        assert cfg.resources.gpu_type == "A100"

    def test_extra_fields_allowed(self):
        cfg = ExperimentConfig(script="train.py", custom_field="hello")
        assert cfg.custom_field == "hello"

    def test_serialization_roundtrip(self):
        cfg = ExperimentConfig(
            script="train.py",
            script_args={"nested": {"a": 1, "b": [2, 3]}},
        )
        dumped = cfg.model_dump_json()
        restored = ExperimentConfig.model_validate_json(dumped)
        assert restored.script_args == cfg.script_args


class TestConfigDiff:
    def test_no_diff(self):
        cfg = ExperimentConfig(script="train.py", script_args={"lr": 1e-4})
        assert config_diff(cfg, cfg) == {}

    def test_simple_diff(self):
        old = ExperimentConfig(script="train.py", script_args={"lr": 1e-4})
        new = ExperimentConfig(script="train.py", script_args={"lr": 5e-6})
        diff = config_diff(old, new)
        assert "script_args.lr" in diff
        assert diff["script_args.lr"]["from"] == 1e-4
        assert diff["script_args.lr"]["to"] == 5e-6

    def test_added_key(self):
        old = ExperimentConfig(script="train.py", script_args={})
        new = ExperimentConfig(script="train.py", script_args={"lr": 1e-4})
        diff = config_diff(old, new)
        assert "script_args.lr" in diff
        assert diff["script_args.lr"]["from"] is None
        assert diff["script_args.lr"]["to"] == 1e-4

    def test_removed_key(self):
        old = ExperimentConfig(script="train.py", script_args={"lr": 1e-4})
        new = ExperimentConfig(script="train.py", script_args={})
        diff = config_diff(old, new)
        assert "script_args.lr" in diff
        assert diff["script_args.lr"]["to"] is None

    def test_nested_diff(self):
        old = ExperimentConfig(
            script="train.py",
            script_args={"reward": {"penalty": -0.5, "bonus": 1.0}},
        )
        new = ExperimentConfig(
            script="train.py",
            script_args={"reward": {"penalty": -0.8, "bonus": 1.0}},
        )
        diff = config_diff(old, new)
        assert "script_args.reward.penalty" in diff
        assert "script_args.reward.bonus" not in diff
