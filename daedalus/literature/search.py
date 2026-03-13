"""Paper search via Semantic Scholar and ACL Anthology APIs."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .paper import Paper

logger = logging.getLogger(__name__)

# --- Semantic Scholar ---
_SS_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_SS_FIELDS = "title,authors,year,abstract,citationCount,externalIds,url,venue"


def _parse_ss_paper(data: dict[str, Any]) -> Paper:
    """Parse a Semantic Scholar API response into a Paper."""
    external_ids = data.get("externalIds") or {}
    authors = [a.get("name", "") for a in (data.get("authors") or [])]
    acl_id = external_ids.get("ACL")

    return Paper(
        arxiv_id=external_ids.get("ArXiv"),
        acl_id=acl_id,
        semantic_scholar_id=data.get("paperId"),
        title=data.get("title", ""),
        authors=authors,
        year=data.get("year"),
        abstract=data.get("abstract") or "",
        url=data.get("url") or "",
        citation_count=data.get("citationCount"),
        venue=data.get("venue") or "",
    )


# --- ACL Anthology ---
_ACL_SEARCH_URL = "https://aclanthology.org/api/search"


def _parse_acl_paper(data: dict[str, Any]) -> Paper:
    """Parse an ACL Anthology search result into a Paper."""
    authors = data.get("author", [])
    if isinstance(authors, list):
        authors = [a.get("full", a) if isinstance(a, dict) else str(a) for a in authors]

    acl_id = data.get("anthology_id") or data.get("id", "")
    url = f"https://aclanthology.org/{acl_id}" if acl_id else data.get("url", "")

    return Paper(
        acl_id=acl_id,
        title=data.get("title", ""),
        authors=authors,
        year=data.get("year"),
        abstract=data.get("abstract") or "",
        url=url,
        venue=data.get("venue") or "",
    )


class PaperSearcher:
    """Search for papers across Semantic Scholar and ACL Anthology.

    Semantic Scholar: broad coverage, citation counts, arXiv IDs.
    ACL Anthology: NLP/CL conference papers (ACL, EMNLP, NAACL, EACL, etc.).
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> PaperSearcher:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def search(
        self,
        query: str,
        limit: int = 10,
        source: str = "all",
    ) -> list[Paper]:
        """Search for papers matching a query string.

        Args:
            query: Search query.
            limit: Max results per source.
            source: "semantic_scholar", "acl", or "all" (default).
        """
        results: list[Paper] = []

        if source in ("semantic_scholar", "all"):
            results.extend(self._search_semantic_scholar(query, limit))

        if source in ("acl", "all"):
            results.extend(self._search_acl(query, limit))

        # Deduplicate by title similarity
        if source == "all":
            results = self._dedup_papers(results)

        return results[:limit]

    def _search_semantic_scholar(self, query: str, limit: int) -> list[Paper]:
        """Search Semantic Scholar."""
        try:
            resp = self._client.get(
                f"{_SS_BASE_URL}/paper/search",
                params={"query": query, "limit": limit, "fields": _SS_FIELDS},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return [_parse_ss_paper(d) for d in data]
        except httpx.HTTPError as e:
            logger.error("Semantic Scholar search failed: %s", e)
            return []
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Unexpected Semantic Scholar response format: %s", e)
            return []

    def _search_acl(self, query: str, limit: int) -> list[Paper]:
        """Search ACL Anthology.

        Uses the ACL Anthology search endpoint. Falls back gracefully
        if the API is unavailable.
        """
        try:
            resp = self._client.get(
                _ACL_SEARCH_URL,
                params={"q": query, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle different response formats
            if isinstance(data, list):
                papers = data
            elif isinstance(data, dict):
                papers = data.get("results", data.get("papers", []))
            else:
                return []

            return [_parse_acl_paper(p) for p in papers[:limit]]
        except httpx.HTTPError as e:
            logger.warning("ACL Anthology search failed (non-critical): %s", e)
            return []
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Unexpected ACL Anthology response format: %s", e)
            return []

    @staticmethod
    def _dedup_papers(papers: list[Paper]) -> list[Paper]:
        """Remove duplicate papers by matching UIDs and normalized titles."""
        seen_uids: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[Paper] = []

        for p in papers:
            uid = p.uid
            title_norm = p.title.lower().strip().rstrip(".")

            if uid in seen_uids:
                continue
            if title_norm and title_norm in seen_titles:
                continue

            seen_uids.add(uid)
            if title_norm:
                seen_titles.add(title_norm)
            unique.append(p)

        return unique

    def get_paper(self, paper_id: str) -> Paper | None:
        """Fetch a specific paper by Semantic Scholar ID or arxiv ID.

        Accepts formats: "arxiv:2601.20126", "ARXIV:2601.20126", or raw SS ID.
        """
        if not paper_id.lower().startswith("arxiv:"):
            paper_id_query = paper_id
        else:
            paper_id_query = f"ARXIV:{paper_id.split(':')[-1]}"

        try:
            resp = self._client.get(
                f"{_SS_BASE_URL}/paper/{paper_id_query}",
                params={"fields": _SS_FIELDS},
            )
            resp.raise_for_status()
            return _parse_ss_paper(resp.json())
        except httpx.HTTPError as e:
            logger.error("Failed to fetch paper %s: %s", paper_id, e)
            return None
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Unexpected response for paper %s: %s", paper_id, e)
            return None

    def get_acl_paper(self, acl_id: str) -> Paper | None:
        """Fetch a paper from ACL Anthology by its ID (e.g., '2024.acl-long.1')."""
        try:
            resp = self._client.get(
                f"https://aclanthology.org/{acl_id}.json",
            )
            resp.raise_for_status()
            return _parse_acl_paper(resp.json())
        except httpx.HTTPError as e:
            logger.error("Failed to fetch ACL paper %s: %s", acl_id, e)
            return None
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Unexpected response for ACL paper %s: %s", acl_id, e)
            return None

    def related(self, paper_id: str, limit: int = 5) -> list[Paper]:
        """Find papers related to a given paper (Semantic Scholar only)."""
        try:
            resp = self._client.get(
                f"{_SS_BASE_URL}/paper/{paper_id}/recommendations",
                params={"limit": limit, "fields": _SS_FIELDS},
            )
            resp.raise_for_status()
            data = resp.json().get("recommendedPapers", [])
            return [_parse_ss_paper(d) for d in data]
        except httpx.HTTPError as e:
            logger.error("Failed to fetch related papers: %s", e)
            return []
