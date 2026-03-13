"""Local paper library backed by JSONL."""

from __future__ import annotations

import logging
from pathlib import Path

from .paper import Paper

logger = logging.getLogger(__name__)


class Library:
    """Local paper library with JSONL persistence."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _read_all(self) -> list[Paper]:
        papers: list[Paper] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                papers.append(Paper.model_validate_json(line))
            except Exception as e:
                logger.warning(f"Failed to parse paper entry: {e}")
        return papers

    def _write_all(self, papers: list[Paper]) -> None:
        with open(self.path, "w") as f:
            for p in papers:
                f.write(p.model_dump_json() + "\n")

    def add(self, paper: Paper) -> None:
        """Add a paper, deduplicating by uid."""
        papers = self._read_all()
        if any(p.uid == paper.uid for p in papers):
            logger.info(f"Paper {paper.uid} already in library, skipping.")
            return
        papers.append(paper)
        self._write_all(papers)

    def get(self, paper_id: str) -> Paper | None:
        """Look up a paper by arxiv ID, ACL ID, or Semantic Scholar ID."""
        for p in self._read_all():
            if paper_id in (p.arxiv_id, p.acl_id, p.semantic_scholar_id):
                return p
        return None

    def search_local(self, query: str) -> list[Paper]:
        """Simple substring search across title and abstract."""
        query_lower = query.lower()
        return [
            p
            for p in self._read_all()
            if query_lower in p.title.lower() or query_lower in p.abstract.lower()
        ]

    def by_tag(self, tag: str) -> list[Paper]:
        """Filter papers by tag."""
        return [p for p in self._read_all() if tag in p.tags]

    def all(self) -> list[Paper]:
        """Return all papers."""
        return self._read_all()

    def __len__(self) -> int:
        return len(self._read_all())
