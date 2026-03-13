"""Tests for scripts registry."""

import pytest

from daedalus.core.scripts_registry import ScriptsRegistry, ScriptSpec, ParameterSpec


class TestScriptsRegistry:
    def test_load_from_config(self):
        config = {
            "train": {
                "path": "train.py",
                "description": "Main training script",
                "parameters": {
                    "lr": {"type": "float", "default": 1e-5, "range": [1e-6, 1e-3]},
                    "epochs": {"type": "int", "default": 3},
                    "model": {"type": "str", "default": "bert-base", "choices": ["bert-base", "bert-large"]},
                },
            },
            "eval": {
                "path": "evaluate.py",
                "description": "Run evaluation",
                "parameters": {
                    "model_path": {"type": "str", "required": True},
                },
            },
        }
        registry = ScriptsRegistry(config)

        assert "train" in registry.scripts
        assert "eval" in registry.scripts
        assert registry.scripts["train"].path == "train.py"
        assert len(registry.scripts["train"].parameters) == 3

    def test_get_script(self):
        registry = ScriptsRegistry({
            "train": {"path": "train.py", "parameters": {"lr": {"default": 1e-5}}},
        })
        spec = registry.get("train")
        assert spec is not None
        assert spec.path == "train.py"

        assert registry.get("nonexistent") is None

    def test_get_defaults(self):
        registry = ScriptsRegistry({
            "train": {
                "path": "train.py",
                "parameters": {
                    "lr": {"default": 1e-5},
                    "model": {"default": "bert-base"},
                    "output_dir": {"required": True},  # no default
                },
            },
        })
        defaults = registry.get_defaults("train")
        assert defaults == {"lr": 1e-5, "model": "bert-base"}
        assert "output_dir" not in defaults

    def test_format_for_context(self):
        registry = ScriptsRegistry({
            "train": {
                "path": "train.py",
                "description": "Train a model",
                "parameters": {
                    "lr": {"type": "float", "default": 1e-5, "description": "Learning rate"},
                },
            },
        })
        text = registry.format_for_context()
        assert "train.py" in text
        assert "Train a model" in text
        assert "lr" in text
        assert "Learning rate" in text

    def test_format_empty_registry(self):
        registry = ScriptsRegistry()
        text = registry.format_for_context()
        assert "No scripts registered" in text

    def test_simple_value_as_default(self):
        """If a parameter spec is just a value, treat it as default."""
        registry = ScriptsRegistry({
            "train": {
                "path": "train.py",
                "parameters": {"lr": 1e-5, "epochs": 3},
            },
        })
        assert registry.scripts["train"].parameters["lr"].default == 1e-5
        assert registry.scripts["train"].parameters["epochs"].default == 3

    def test_choices_in_format(self):
        registry = ScriptsRegistry({
            "train": {
                "path": "train.py",
                "parameters": {
                    "reward": {"type": "str", "choices": ["simple", "hybrid"]},
                },
            },
        })
        text = registry.format_for_context()
        assert "simple" in text
        assert "hybrid" in text

    def test_range_in_format(self):
        registry = ScriptsRegistry({
            "train": {
                "path": "train.py",
                "parameters": {
                    "lr": {"type": "float", "range": [1e-6, 1e-3]},
                },
            },
        })
        text = registry.format_for_context()
        assert "Range:" in text
