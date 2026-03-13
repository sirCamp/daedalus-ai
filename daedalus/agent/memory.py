"""Persistent research memory — notes and decisions across sessions.

Stores structured research notes in `ledger/notes.jsonl` (source of truth)
and auto-generates `ledger/notes.md` (human-readable, greppable) on every save.

Each note has:
- A category (decision, insight, dead_end, convergence, todo, general)
- Free-text content
- Optional linked experiment IDs and tags
- Timestamp

The Markdown file has dedicated sections per category so you can grep:
  grep "Dead Ends" -A 50 ledger/notes.md
  grep "Convergence" -A 20 ledger/notes.md
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResearchNote(BaseModel):
    """A single research note."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    category: str = "general"  # decision, insight, dead_end, convergence, todo, general
    content: str
    experiment_ids: list[str] = []
    tags: list[str] = []


# Section ordering and labels for Markdown output
CATEGORY_ORDER = ["decision", "insight", "dead_end", "convergence", "todo", "general"]
CATEGORY_LABELS = {
    "decision": "Key Decisions",
    "insight": "Insights",
    "dead_end": "Dead Ends",
    "convergence": "Convergence & Saturation",
    "todo": "TODOs & Next Steps",
    "general": "General Notes",
}


class ResearchMemory:
    """Persistent research memory backed by JSONL + Markdown.

    The JSONL file is the source of truth. The Markdown file is
    auto-regenerated on every save with dedicated sections per category,
    designed for human reading and grepping.
    """

    VALID_CATEGORIES = set(CATEGORY_ORDER)

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.path = self.ledger_path / "notes.jsonl"
        self._markdown_path = self.ledger_path / "notes.md"
        self.ledger_path.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def save(self, note: ResearchNote, dedup: bool = True) -> bool:
        """Append a note to the memory.

        Args:
            note: The note to save.
            dedup: If True, skip saving if a note with identical content
                   already exists (prevents repeated auto-saves).

        Returns:
            True if saved, False if skipped (duplicate).
        """
        if dedup:
            existing = self.all()
            if any(
                n.content == note.content and n.category == note.category
                for n in existing
            ):
                logger.debug(f"Skipping duplicate note: {note.content[:60]}")
                return False

        with open(self.path, "a") as f:
            f.write(note.model_dump_json() + "\n")
        logger.info(f"Saved note [{note.category}]: {note.content[:80]}")

        # Regenerate Markdown
        self._sync_markdown()
        return True

    def all(self) -> list[ResearchNote]:
        """Read all notes, ordered by timestamp."""
        notes: list[ResearchNote] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                notes.append(ResearchNote.model_validate_json(line))
            except Exception as e:
                logger.warning(f"Failed to parse note: {e}")
        notes.sort(key=lambda n: n.timestamp)
        return notes

    def by_category(self, category: str) -> list[ResearchNote]:
        """Filter notes by category."""
        return [n for n in self.all() if n.category == category]

    def by_experiment(self, exp_id: str) -> list[ResearchNote]:
        """Get all notes linked to a specific experiment."""
        return [n for n in self.all() if exp_id in n.experiment_ids]

    def recent(self, n: int = 10) -> list[ResearchNote]:
        """Return the N most recent notes."""
        return self.all()[-n:]

    def search(self, query: str) -> list[ResearchNote]:
        """Simple substring search across note content."""
        query_lower = query.lower()
        return [n for n in self.all() if query_lower in n.content.lower()]

    def format_for_context(self, limit: int = 20) -> str:
        """Format recent notes as context for the agent.

        Groups by category for readability.
        """
        notes = self.all()
        if not notes:
            return "No research notes saved yet."

        recent = notes[-limit:]

        by_cat: dict[str, list[ResearchNote]] = {}
        for note in recent:
            by_cat.setdefault(note.category, []).append(note)

        if len(recent) < len(notes):
            lines = [f"**Research Memory** ({len(notes)} total, showing last {len(recent)}):\n"]
        else:
            lines = [f"**Research Memory** ({len(notes)} notes):\n"]

        # Include all categories, including unknown ones
        all_cats = list(CATEGORY_ORDER) + [
            c for c in by_cat if c not in CATEGORY_ORDER
        ]
        for cat in all_cats:
            cat_notes = by_cat.get(cat, [])
            if not cat_notes:
                continue
            lines.append(f"**{CATEGORY_LABELS.get(cat, cat.replace('_', ' ').title())}**:")
            for note in cat_notes:
                ts = note.timestamp.strftime("%Y-%m-%d")
                exp_ref = f" (exp: {', '.join(note.experiment_ids)})" if note.experiment_ids else ""
                lines.append(f"- [{ts}]{exp_ref} {note.content}")
            lines.append("")

        return "\n".join(lines)

    def _sync_markdown(self) -> None:
        """Regenerate notes.md from notes.jsonl.

        Produces a human-readable Markdown file with dedicated sections
        per category, suitable for grep and direct reading.
        """
        notes = self.all()
        if not notes:
            self._markdown_path.write_text(
                "# Research Memory\n\nNo notes yet.\n"
            )
            return

        lines = ["# Research Memory\n"]
        lines.append(f"Auto-generated from `notes.jsonl` — {len(notes)} notes total.\n")

        by_cat: dict[str, list[ResearchNote]] = {}
        for note in notes:
            by_cat.setdefault(note.category, []).append(note)

        all_cats = list(CATEGORY_ORDER) + [
            c for c in by_cat if c not in CATEGORY_ORDER
        ]
        for cat in all_cats:
            cat_notes = by_cat.get(cat, [])
            label = CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())
            lines.append(f"## {label}\n")

            if not cat_notes:
                lines.append("_(none)_\n")
                continue

            for note in cat_notes:
                ts = note.timestamp.strftime("%Y-%m-%d %H:%M")
                # Main line
                line = f"- **[{ts}]** {note.content}"
                lines.append(line)

                # Metadata on indented sub-lines
                if note.experiment_ids:
                    lines.append(f"  - Experiments: {', '.join(note.experiment_ids)}")
                if note.tags:
                    lines.append(f"  - Tags: {', '.join(note.tags)}")

            lines.append("")

        # Cross-reference section: experiments mentioned in notes
        all_exp_ids: dict[str, list[str]] = {}
        for note in notes:
            for eid in note.experiment_ids:
                all_exp_ids.setdefault(eid, []).append(
                    f"{note.category}: {note.content[:60]}"
                )

        if all_exp_ids:
            lines.append("## Experiment Cross-Reference\n")
            for eid in sorted(all_exp_ids.keys()):
                lines.append(f"### {eid}\n")
                for ref in all_exp_ids[eid]:
                    lines.append(f"- {ref}")
                lines.append("")

        self._markdown_path.write_text("\n".join(lines))
