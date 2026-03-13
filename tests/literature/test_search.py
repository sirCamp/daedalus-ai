"""Tests for PaperSearcher — Semantic Scholar + ACL Anthology with mocked HTTP."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from daedalus.literature.paper import Paper
from daedalus.literature.search import (
    PaperSearcher,
    _parse_acl_paper,
    _parse_ss_paper,
)


# ---------------------------------------------------------------------------
# Fixtures: mock API responses
# ---------------------------------------------------------------------------

SS_PAPER_RESPONSE = {
    "paperId": "abc123",
    "title": "Attention Is All You Need",
    "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
    "year": 2017,
    "abstract": "We propose a new simple network architecture.",
    "citationCount": 90000,
    "externalIds": {"ArXiv": "1706.03762", "ACL": "P17-1017"},
    "url": "https://semanticscholar.org/paper/abc123",
    "venue": "NeurIPS",
}

SS_SEARCH_RESPONSE = {
    "data": [SS_PAPER_RESPONSE],
}

ACL_PAPER_RESPONSE = {
    "anthology_id": "2024.acl-long.1",
    "title": "Some ACL Paper",
    "author": [{"full": "Alice Smith"}, {"full": "Bob Jones"}],
    "year": 2024,
    "abstract": "We study something interesting.",
    "venue": "ACL",
    "url": "https://aclanthology.org/2024.acl-long.1",
}

ACL_SEARCH_RESPONSE_LIST = [ACL_PAPER_RESPONSE]

ACL_SEARCH_RESPONSE_DICT = {
    "results": [ACL_PAPER_RESPONSE],
}


def _mock_response(json_data: dict | list, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParseSSPaper:
    def test_basic_fields(self):
        paper = _parse_ss_paper(SS_PAPER_RESPONSE)
        assert paper.title == "Attention Is All You Need"
        assert paper.arxiv_id == "1706.03762"
        assert paper.acl_id == "P17-1017"
        assert paper.semantic_scholar_id == "abc123"
        assert paper.year == 2017
        assert paper.citation_count == 90000
        assert paper.venue == "NeurIPS"
        assert len(paper.authors) == 2

    def test_missing_fields(self):
        paper = _parse_ss_paper({"title": "Minimal"})
        assert paper.title == "Minimal"
        assert paper.arxiv_id is None
        assert paper.acl_id is None
        assert paper.authors == []
        assert paper.abstract == ""

    def test_null_external_ids(self):
        paper = _parse_ss_paper({"title": "No IDs", "externalIds": None})
        assert paper.arxiv_id is None


class TestParseACLPaper:
    def test_basic_fields(self):
        paper = _parse_acl_paper(ACL_PAPER_RESPONSE)
        assert paper.acl_id == "2024.acl-long.1"
        assert paper.title == "Some ACL Paper"
        assert paper.venue == "ACL"
        assert paper.year == 2024
        assert len(paper.authors) == 2
        assert paper.authors[0] == "Alice Smith"

    def test_fallback_id_field(self):
        data = {"id": "2024.emnlp-main.5", "title": "EMNLP Paper"}
        paper = _parse_acl_paper(data)
        assert paper.acl_id == "2024.emnlp-main.5"

    def test_authors_as_strings(self):
        data = {"title": "Test", "author": ["Alice", "Bob"]}
        paper = _parse_acl_paper(data)
        assert paper.authors == ["Alice", "Bob"]

    def test_url_generated_from_acl_id(self):
        paper = _parse_acl_paper(ACL_PAPER_RESPONSE)
        assert paper.url == "https://aclanthology.org/2024.acl-long.1"


# ---------------------------------------------------------------------------
# PaperSearcher tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestSearchSemanticScholar:
    def test_search_returns_papers(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response(SS_SEARCH_RESPONSE)

        results = searcher.search("attention", limit=5, source="semantic_scholar")
        assert len(results) == 1
        assert results[0].title == "Attention Is All You Need"

    def test_search_http_error_returns_empty(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response({}, status_code=500)

        results = searcher.search("attention", source="semantic_scholar")
        assert results == []

    def test_search_empty_data(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response({"data": []})

        results = searcher.search("nothing", source="semantic_scholar")
        assert results == []


class TestSearchACL:
    def test_search_list_response(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response(ACL_SEARCH_RESPONSE_LIST)

        results = searcher.search("NLP", limit=5, source="acl")
        assert len(results) == 1
        assert results[0].acl_id == "2024.acl-long.1"

    def test_search_dict_response(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response(ACL_SEARCH_RESPONSE_DICT)

        results = searcher.search("NLP", limit=5, source="acl")
        assert len(results) == 1

    def test_search_http_error_returns_empty(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response({}, status_code=404)

        results = searcher.search("NLP", source="acl")
        assert results == []


class TestSearchAll:
    def test_combined_search_deduplicates(self):
        """Same paper from both sources should be deduped."""
        searcher = PaperSearcher()
        searcher._client = MagicMock()

        # Both return the same title
        ss_data = {
            "data": [{
                "paperId": "ss1",
                "title": "Shared Paper Title",
                "authors": [],
                "year": 2024,
                "externalIds": {"ArXiv": "2401.00001"},
            }]
        }
        acl_data = [{
            "anthology_id": "2024.acl-long.99",
            "title": "Shared Paper Title",
            "author": [],
            "year": 2024,
        }]

        # First call = SS, second call = ACL
        searcher._client.get.side_effect = [
            _mock_response(ss_data),
            _mock_response(acl_data),
        ]

        results = searcher.search("shared", limit=10, source="all")
        assert len(results) == 1  # deduped by title

    def test_combined_different_papers(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()

        ss_data = {"data": [SS_PAPER_RESPONSE]}
        acl_data = [ACL_PAPER_RESPONSE]

        searcher._client.get.side_effect = [
            _mock_response(ss_data),
            _mock_response(acl_data),
        ]

        results = searcher.search("papers", limit=10, source="all")
        assert len(results) == 2

    def test_limit_applied_after_merge(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()

        # 3 papers from SS + 3 from ACL, limit=4
        ss_papers = [
            {"paperId": f"ss{i}", "title": f"SS Paper {i}", "authors": [], "externalIds": {}}
            for i in range(3)
        ]
        acl_papers = [
            {"anthology_id": f"acl{i}", "title": f"ACL Paper {i}", "author": []}
            for i in range(3)
        ]

        searcher._client.get.side_effect = [
            _mock_response({"data": ss_papers}),
            _mock_response(acl_papers),
        ]

        results = searcher.search("test", limit=4, source="all")
        assert len(results) == 4


# ---------------------------------------------------------------------------
# Dedup tests
# ---------------------------------------------------------------------------

class TestDedup:
    def test_dedup_by_uid(self):
        papers = [
            Paper(arxiv_id="123", title="Paper A"),
            Paper(arxiv_id="123", title="Paper A (copy)"),
        ]
        result = PaperSearcher._dedup_papers(papers)
        assert len(result) == 1

    def test_dedup_by_normalized_title(self):
        papers = [
            Paper(arxiv_id="1", title="Some Paper."),
            Paper(arxiv_id="2", title="some paper"),
        ]
        result = PaperSearcher._dedup_papers(papers)
        assert len(result) == 1

    def test_empty_titles_not_collapsed(self):
        papers = [
            Paper(semantic_scholar_id="a", title=""),
            Paper(semantic_scholar_id="b", title=""),
        ]
        result = PaperSearcher._dedup_papers(papers)
        assert len(result) == 2

    def test_no_dedup_different_papers(self):
        papers = [
            Paper(arxiv_id="1", title="Paper One"),
            Paper(arxiv_id="2", title="Paper Two"),
        ]
        result = PaperSearcher._dedup_papers(papers)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# get_paper / get_acl_paper / related
# ---------------------------------------------------------------------------

class TestGetPaper:
    def test_get_by_arxiv_id(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response(SS_PAPER_RESPONSE)

        paper = searcher.get_paper("ARXIV:1706.03762")
        assert paper is not None
        assert paper.title == "Attention Is All You Need"

    def test_get_by_raw_id(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response(SS_PAPER_RESPONSE)

        paper = searcher.get_paper("abc123")
        assert paper is not None
        # Verify URL was called with the raw ID, not ARXIV: prefix
        call_args = searcher._client.get.call_args
        assert "abc123" in call_args[0][0]

    def test_get_not_found(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response({}, status_code=404)

        paper = searcher.get_paper("nonexistent")
        assert paper is None


class TestGetACLPaper:
    def test_get_by_acl_id(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response(ACL_PAPER_RESPONSE)

        paper = searcher.get_acl_paper("2024.acl-long.1")
        assert paper is not None
        assert paper.acl_id == "2024.acl-long.1"

    def test_get_not_found(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response({}, status_code=404)

        paper = searcher.get_acl_paper("nonexistent")
        assert paper is None


class TestRelated:
    def test_related_papers(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response({
            "recommendedPapers": [SS_PAPER_RESPONSE],
        })

        results = searcher.related("abc123", limit=5)
        assert len(results) == 1

    def test_related_http_error(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher._client.get.return_value = _mock_response({}, status_code=500)

        results = searcher.related("abc123")
        assert results == []


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_close(self):
        searcher = PaperSearcher()
        searcher._client = MagicMock()
        searcher.close()
        searcher._client.close.assert_called_once()

    def test_context_manager(self):
        with PaperSearcher() as searcher:
            searcher._client = MagicMock()
        # After exiting, close would have been called on the original client
