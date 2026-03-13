"""Citation formatting helpers."""

from __future__ import annotations

from .paper import Paper


def format_citation(paper: Paper) -> str:
    """Format a single paper citation.

    Example: "Campese et al. (2025) — Self-Calibrating GRPO"
    """
    if paper.authors:
        first_author = paper.authors[0].split()[-1]  # last name
        if len(paper.authors) > 1:
            author_str = f"{first_author} et al."
        else:
            author_str = first_author
    else:
        author_str = "Unknown"

    year_str = f" ({paper.year})" if paper.year else ""
    return f"{author_str}{year_str} — {paper.title}"


def cite_for_hypothesis(papers: list[Paper]) -> str:
    """Format a list of papers for use in hypothesis justification."""
    if not papers:
        return "No supporting literature cited."

    lines = ["**Supporting literature:**"]
    for p in papers:
        ref = format_citation(p)
        if p.arxiv_id:
            ref += f" [arXiv:{p.arxiv_id}]"
        if p.relevance_note:
            ref += f"\n  > {p.relevance_note}"
        lines.append(f"- {ref}")

    return "\n".join(lines)
