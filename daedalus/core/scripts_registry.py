"""Scripts registry — declares available scripts and their parameters.

The registry is defined in ``daedalus.yaml`` under the ``scripts:`` key.
It tells the agent what scripts exist, what they do, and what parameters
they accept. This is critical for the agent to design experiments autonomously.

Example daedalus.yaml:

    scripts:
      train:
        path: training_grpo_unified.py
        description: "GRPO training with configurable reward"
        parameters:
          model_name:
            type: str
            default: "Qwen/Qwen2.5-7B-Instruct"
            description: "Base model"
          reward_type:
            type: str
            default: "similarity_hybrid"
            choices: [simple, hybrid_llm_v2, similarity_hybrid, fiscore]
          idk_knowable_reward:
            type: float
            default: -0.5
            range: [-1.0, 0.0]
            description: "Penalty for IDK on knowable questions"
          learning_rate:
            type: float
            default: 5e-6

      eval:
        path: evaluate.py
        description: "Run SimpleQA + PopQA evaluation"
        parameters:
          model_path:
            type: str
            required: true
          benchmarks:
            type: str
            default: "simpleqa,popqa"
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """Specification for a script parameter."""

    type: str = "str"  # str, int, float, bool
    default: Any = None
    required: bool = False
    description: str = ""
    choices: list[Any] | None = None
    range: list[float] | None = None  # [min, max] for numeric


class ScriptSpec(BaseModel):
    """Specification for a registered script."""

    path: str
    description: str = ""
    parameters: dict[str, ParameterSpec] = {}


class ScriptsRegistry:
    """Registry of available scripts and their parameters."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.scripts: dict[str, ScriptSpec] = {}
        if config:
            for name, spec in config.items():
                params = {}
                for pname, pspec in spec.get("parameters", {}).items():
                    if isinstance(pspec, dict):
                        params[pname] = ParameterSpec(**pspec)
                    else:
                        # Simple value — treat as default
                        params[pname] = ParameterSpec(default=pspec)
                self.scripts[name] = ScriptSpec(
                    path=spec.get("path", ""),
                    description=spec.get("description", ""),
                    parameters=params,
                )

    def get(self, name: str) -> ScriptSpec | None:
        return self.scripts.get(name)

    def format_for_context(self) -> str:
        """Format registry as text for agent context injection."""
        if not self.scripts:
            return "(No scripts registered. Define them in daedalus.yaml under 'scripts:'.)"

        lines: list[str] = []
        for name, spec in self.scripts.items():
            lines.append(f"### `{name}` — {spec.path}")
            if spec.description:
                lines.append(f"{spec.description}")
            lines.append("")

            if spec.parameters:
                lines.append("| Parameter | Type | Default | Description |")
                lines.append("|-----------|------|---------|-------------|")
                for pname, param in spec.parameters.items():
                    default = str(param.default) if param.default is not None else ("**required**" if param.required else "—")
                    desc = param.description
                    if param.choices:
                        desc += f" Choices: {param.choices}"
                    if param.range:
                        desc += f" Range: [{param.range[0]}, {param.range[1]}]"
                    lines.append(f"| `{pname}` | {param.type} | {default} | {desc} |")
                lines.append("")

        return "\n".join(lines)

    def get_defaults(self, name: str) -> dict[str, Any]:
        """Get default parameter values for a script."""
        spec = self.scripts.get(name)
        if not spec:
            return {}
        return {
            pname: param.default
            for pname, param in spec.parameters.items()
            if param.default is not None
        }
