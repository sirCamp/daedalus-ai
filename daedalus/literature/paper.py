"""Paper data model."""

from __future__ import annotations

from pydantic import BaseModel


class Paper(BaseModel):
    """A research paper with metadata and curated notes."""

    arxiv_id: str | None = None
    acl_id: str | None = None
    semantic_scholar_id: str | None = None
    title: str
    authors: list[str] = []
    year: int | None = None
    abstract: str = ""
    key_findings: list[str] = []
    relevance_note: str = ""
    tags: list[str] = []
    url: str = ""
    citation_count: int | None = None
    venue: str = ""

    @property
    def uid(self) -> str:
        """Unique identifier: arxiv_id if available, else acl_id, else semantic_scholar_id."""
        return self.arxiv_id or self.acl_id or self.semantic_scholar_id or self.title
