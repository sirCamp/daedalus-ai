"""Tests for literature review and paper details tools."""

import json
import pytest

from daedalus.agent.tools import ToolExecutor
from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus, Reflection
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.ledger import Ledger
from daedalus.literature.library import Library
from daedalus.literature.paper import Paper


def _setup_project(tmp_path):
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "ledger").mkdir()
    (project / "ledger" / "experiments.jsonl").touch()
    (project / "ledger" / "papers.jsonl").touch()
    return project


def _add_paper(project, arxiv_id, title, **kwargs):
    lib = Library(project / "ledger" / "papers.jsonl")
    lib.add(Paper(arxiv_id=arxiv_id, title=title, **kwargs))


def _add_experiment(project, exp_id, papers=None, results=None, reflection=None):
    ledger = Ledger(project / "ledger")
    exp = Experiment(
        id=exp_id,
        hypothesis=Hypothesis(
            statement=f"H for {exp_id}",
            rationale="test",
            papers=papers or [],
        ),
        config=ExperimentConfig(script="train.py", script_args={"lr": 1e-5}),
        status=ExperimentStatus.ANALYZED,
        results=results or {"eval": {"acc": 0.85}},
        reflection=reflection,
    )
    ledger.append(exp)


class TestGetPaperDetails:
    def test_from_library(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_paper(
            project, "2601.20126", "Rewarding Intellectual Humility",
            key_findings=["Ternary abstention reward works"],
            relevance_note="Calibration reward design",
            tags=["calibration"],
        )

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("get_paper_details", {
            "arxiv_id": "2601.20126",
        }))

        assert result["source"] == "library"
        assert result["title"] == "Rewarding Intellectual Humility"
        assert "Ternary abstention" in result["key_findings"][0]
        assert result["tags"] == ["calibration"]

    def test_search_local(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_paper(project, "2601.20126", "Rewarding Intellectual Humility",
                   abstract="A paper about abstention and calibration.")
        _add_paper(project, "2503.02623", "Rewarding Doubt",
                   abstract="Log scoring rule for calibration.")

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("get_paper_details", {
            "query": "calibration",
        }))

        assert result["source"] == "library"
        assert len(result["papers"]) == 2

    def test_not_found(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("get_paper_details", {
            "query": "nonexistent",
        }))
        assert "error" in result

    def test_no_params(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("get_paper_details", {}))
        assert "error" in result


class TestLiteratureReview:
    def test_basic_review(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_paper(project, "2601.20126", "Rewarding Intellectual Humility",
                   key_findings=["Ternary reward"])
        _add_experiment(project, "exp_001")

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("literature_review", {}))

        assert "review" in result
        assert "Rewarding Intellectual Humility" in result["review"]
        assert result["papers_in_library"] == 1

    def test_review_with_focus(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_paper(project, "2601.20126", "Rewarding Intellectual Humility",
                   tags=["calibration"])
        _add_paper(project, "9999.99999", "Unrelated Paper on Vision",
                   tags=["vision"])

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("literature_review", {
            "focus": "calibration",
        }))

        assert result["papers_matched"] == 1
        assert "Rewarding Intellectual Humility" in result["review"]

    def test_papers_vs_experiments(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_paper(project, "2601.20126", "Rewarding Intellectual Humility")
        _add_experiment(
            project, "exp_001",
            papers=["2601.20126"],
            reflection=Reflection(
                hypothesis_confirmed="confirmed",
                analysis="Worked well",
            ),
        )

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("literature_review", {}))

        assert "exp_001 (confirmed)" in result["review"]

    def test_unlinked_papers(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_paper(project, "2601.20126", "Rewarding Intellectual Humility",
                   key_findings=["Ternary reward"])
        _add_experiment(project, "exp_001")  # no papers linked

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("literature_review", {}))

        assert "not yet tested experimentally" in result["review"]

    def test_gaps_section(self, tmp_path):
        project = _setup_project(tmp_path)
        _add_experiment(project, "exp_001")  # no papers
        _add_experiment(
            project, "exp_002",
            reflection=Reflection(
                hypothesis_confirmed="rejected",
                analysis="Did not work",
            ),
        )

        executor = ToolExecutor(project)
        result = json.loads(executor.execute("literature_review", {}))

        assert "no cited papers" in result["review"]
        assert "rejected hypotheses" in result["review"]

    def test_empty_library(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("literature_review", {}))

        assert "No papers in library" in result["review"]
        assert result["papers_in_library"] == 0

    def test_no_external_search_by_default(self, tmp_path):
        project = _setup_project(tmp_path)
        executor = ToolExecutor(project)
        result = json.loads(executor.execute("literature_review", {}))

        assert result["new_papers_found"] == 0
