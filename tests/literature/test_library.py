"""Tests for the local paper library."""

import pytest

from daedalus.literature.paper import Paper
from daedalus.literature.library import Library


def _make_paper(**kwargs) -> Paper:
    defaults = dict(
        arxiv_id="2601.20126",
        title="Rewarding Intellectual Humility",
        authors=["Smith", "Jones"],
        year=2026,
        abstract="We study abstention rewards in RLHF.",
        tags=["calibration", "grpo"],
    )
    defaults.update(kwargs)
    return Paper(**defaults)


class TestLibrary:
    def test_add_and_get(self, tmp_path):
        lib = Library(tmp_path / "papers.jsonl")
        paper = _make_paper()
        lib.add(paper)
        assert lib.get("2601.20126") is not None
        assert lib.get("2601.20126").title == "Rewarding Intellectual Humility"

    def test_dedup(self, tmp_path):
        lib = Library(tmp_path / "papers.jsonl")
        paper = _make_paper()
        lib.add(paper)
        lib.add(paper)  # should not duplicate
        assert len(lib) == 1

    def test_search_local(self, tmp_path):
        lib = Library(tmp_path / "papers.jsonl")
        lib.add(_make_paper())
        lib.add(_make_paper(
            arxiv_id="2503.06639",
            title="GRPO Effective Loss",
            abstract="Analysis of GRPO loss landscape.",
        ))

        results = lib.search_local("abstention")
        assert len(results) == 1
        assert results[0].arxiv_id == "2601.20126"

        results = lib.search_local("GRPO")
        assert len(results) == 1
        assert results[0].arxiv_id == "2503.06639"

    def test_by_tag(self, tmp_path):
        lib = Library(tmp_path / "papers.jsonl")
        lib.add(_make_paper(tags=["calibration"]))
        lib.add(_make_paper(
            arxiv_id="other",
            title="Other paper",
            tags=["unrelated"],
        ))

        results = lib.by_tag("calibration")
        assert len(results) == 1

    def test_empty_library(self, tmp_path):
        lib = Library(tmp_path / "papers.jsonl")
        assert len(lib) == 0
        assert lib.all() == []
        assert lib.get("nope") is None
