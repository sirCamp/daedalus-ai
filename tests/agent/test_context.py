"""Tests for agent context builder."""

import pytest

from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus, Reflection
from daedalus.core.hypothesis import Hypothesis, Prediction
from daedalus.core.ledger import Ledger
from daedalus.agent.context import ContextBuilder
from daedalus.literature.library import Library
from daedalus.literature.paper import Paper


def _setup_project(tmp_path):
    """Create a minimal project structure for testing."""
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    (project / "program.md").write_text(
        "# Test Project\n\n## Goals\n\nImprove calibration.\n"
    )
    return project


def _add_experiment(project, exp_id, status=ExperimentStatus.ANALYZED, results=None, reflection=None):
    ledger = Ledger(project / "ledger")
    exp = Experiment(
        id=exp_id,
        hypothesis=Hypothesis(
            statement=f"Hypothesis for {exp_id}",
            rationale="Test rationale",
            predictions=[Prediction(metric="simpleqa.not_attempted", direction="decrease")],
        ),
        config=ExperimentConfig(script="train.py", script_args={"lr": 5e-6}),
        status=status,
        results=results,
        reflection=reflection,
    )
    ledger.append(exp)
    return exp


class TestContextBuilder:
    def test_full_context_includes_all_sections(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiment(project, "exp_001", results={"simpleqa": {"correct": 3.5}})

        builder = ContextBuilder(project)
        ctx = builder.build(mode="full")

        assert "Test Project" in ctx
        assert "exp_001" in ctx
        assert "Hypothesis for exp_001" in ctx

    def test_reason_mode(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiment(
            project, "exp_001",
            reflection=Reflection(
                hypothesis_confirmed="partial",
                analysis="Partial improvement",
                next_suggestions=["Try penalty=-0.3"],
            ),
        )

        builder = ContextBuilder(project)
        ctx = builder.build(mode="reason")

        assert "Research Program" in ctx
        assert "Experiment History" in ctx
        assert "Try penalty=-0.3" in ctx

    def test_reflect_mode_includes_details(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiment(
            project, "exp_001",
            status=ExperimentStatus.COMPLETED,
            results={"simpleqa": {"correct": 3.5, "not_attempted": 41.0}},
        )

        builder = ContextBuilder(project)
        ctx = builder.build(mode="reflect")

        assert "Experiment to Analyze" in ctx
        assert "simpleqa" in ctx

    def test_with_papers(self, tmp_path):
        project = _setup_project(tmp_path)
        lib = Library(project / "ledger" / "papers.jsonl")
        lib.add(Paper(
            arxiv_id="2601.20126",
            title="Rewarding Intellectual Humility",
            authors=["Smith"],
            year=2026,
            relevance_note="Abstention reward design",
        ))

        builder = ContextBuilder(project)
        ctx = builder.build(mode="full")

        assert "Rewarding Intellectual Humility" in ctx
        assert "Abstention reward design" in ctx

    def test_empty_project(self, tmp_path):
        project = _setup_project(tmp_path)
        builder = ContextBuilder(project)
        ctx = builder.build(mode="full")

        assert "No experiments recorded yet" in ctx

    def test_no_program_md(self, tmp_path):
        project = tmp_path / "bare_project"
        project.mkdir()
        (project / "ledger").mkdir()
        (project / "ledger" / "experiments.jsonl").touch()
        (project / "ledger" / "papers.jsonl").touch()

        builder = ContextBuilder(project)
        ctx = builder.build(mode="reason")
        assert "No program.md found" in ctx
