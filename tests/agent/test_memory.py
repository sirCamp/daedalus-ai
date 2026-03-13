"""Tests for research memory persistence."""

import pytest

from daedalus.agent.memory import ResearchMemory, ResearchNote


class TestResearchMemory:
    def test_save_and_read(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        note = ResearchNote(content="Test note", category="insight")
        memory.save(note)

        notes = memory.all()
        assert len(notes) == 1
        assert notes[0].content == "Test note"
        assert notes[0].category == "insight"

    def test_multiple_notes_ordered(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        memory.save(ResearchNote(content="First"))
        memory.save(ResearchNote(content="Second"))
        memory.save(ResearchNote(content="Third"))

        notes = memory.all()
        assert len(notes) == 3
        assert notes[0].content == "First"
        assert notes[2].content == "Third"

    def test_by_category(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        memory.save(ResearchNote(content="Decision A", category="decision"))
        memory.save(ResearchNote(content="Insight B", category="insight"))
        memory.save(ResearchNote(content="Decision C", category="decision"))

        decisions = memory.by_category("decision")
        assert len(decisions) == 2
        insights = memory.by_category("insight")
        assert len(insights) == 1

    def test_by_experiment(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        memory.save(ResearchNote(content="Note 1", experiment_ids=["exp_001"]))
        memory.save(ResearchNote(content="Note 2", experiment_ids=["exp_002"]))
        memory.save(ResearchNote(content="Note 3", experiment_ids=["exp_001", "exp_003"]))

        notes = memory.by_experiment("exp_001")
        assert len(notes) == 2

    def test_search(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        memory.save(ResearchNote(content="Learning rate is important"))
        memory.save(ResearchNote(content="Batch size doesn't matter"))

        results = memory.search("learning rate")
        assert len(results) == 1
        assert "Learning rate" in results[0].content

    def test_recent(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        for i in range(10):
            memory.save(ResearchNote(content=f"Note {i}"))

        recent = memory.recent(3)
        assert len(recent) == 3
        assert recent[-1].content == "Note 9"

    def test_format_for_context(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        memory.save(ResearchNote(content="Key decision", category="decision"))
        memory.save(ResearchNote(content="Dead end found", category="dead_end"))

        formatted = memory.format_for_context()
        assert "Key Decisions" in formatted
        assert "Dead Ends" in formatted
        assert "Key decision" in formatted

    def test_empty_format(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        formatted = memory.format_for_context()
        assert "No research notes" in formatted

    def test_persistence_across_instances(self, tmp_path):
        ledger_path = tmp_path / "ledger"
        memory1 = ResearchMemory(ledger_path)
        memory1.save(ResearchNote(content="Persisted note"))

        memory2 = ResearchMemory(ledger_path)
        notes = memory2.all()
        assert len(notes) == 1
        assert notes[0].content == "Persisted note"

    def test_note_with_tags(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        memory.save(ResearchNote(
            content="Test", tags=["calibration", "grpo"],
        ))
        notes = memory.all()
        assert notes[0].tags == ["calibration", "grpo"]

    def test_dedup_skips_duplicate(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        assert memory.save(ResearchNote(content="Same note")) is True
        assert memory.save(ResearchNote(content="Same note")) is False
        assert len(memory.all()) == 1

    def test_dedup_disabled(self, tmp_path):
        memory = ResearchMemory(tmp_path / "ledger")
        memory.save(ResearchNote(content="Same note"), dedup=False)
        memory.save(ResearchNote(content="Same note"), dedup=False)
        assert len(memory.all()) == 2

    def test_markdown_generated_on_save(self, tmp_path):
        ledger_path = tmp_path / "ledger"
        memory = ResearchMemory(ledger_path)
        memory.save(ResearchNote(content="Key decision", category="decision"))
        memory.save(ResearchNote(content="LR converged at 3e-5", category="convergence"))
        memory.save(ResearchNote(content="SC-GRPO failed", category="dead_end",
                                 experiment_ids=["exp_004"]))

        md_path = ledger_path / "notes.md"
        assert md_path.exists()
        content = md_path.read_text()

        # Check sections exist
        assert "## Key Decisions" in content
        assert "## Dead Ends" in content
        assert "## Convergence & Saturation" in content
        assert "## Insights" in content
        assert "## TODOs & Next Steps" in content

        # Check content is in correct sections
        assert "Key decision" in content
        assert "LR converged" in content
        assert "SC-GRPO failed" in content

        # Check cross-reference
        assert "## Experiment Cross-Reference" in content
        assert "exp_004" in content

    def test_markdown_sections_greppable(self, tmp_path):
        ledger_path = tmp_path / "ledger"
        memory = ResearchMemory(ledger_path)
        memory.save(ResearchNote(content="reward_type exhausted all choices", category="convergence"))
        memory.save(ResearchNote(content="lr plateau at 3e-5", category="convergence"))

        content = (ledger_path / "notes.md").read_text()

        # Both convergence notes should be under the same section
        conv_section_start = content.index("## Convergence & Saturation")
        next_section = content.index("##", conv_section_start + 1)
        conv_section = content[conv_section_start:next_section]

        assert "reward_type exhausted" in conv_section
        assert "lr plateau" in conv_section

    def test_markdown_empty(self, tmp_path):
        ledger_path = tmp_path / "ledger"
        memory = ResearchMemory(ledger_path)
        # No saves — markdown should say "No notes"
        # Trigger markdown sync manually since no save happened
        memory._sync_markdown()
        content = (ledger_path / "notes.md").read_text()
        assert "No notes yet" in content

    def test_markdown_has_tags_and_experiments(self, tmp_path):
        ledger_path = tmp_path / "ledger"
        memory = ResearchMemory(ledger_path)
        memory.save(ResearchNote(
            content="Important finding",
            category="insight",
            experiment_ids=["exp_001", "exp_002"],
            tags=["calibration", "grpo"],
        ))
        content = (ledger_path / "notes.md").read_text()
        assert "exp_001, exp_002" in content
        assert "calibration, grpo" in content
