"""Tests for the experiment ledger."""

import pytest

from daedalus.core.config import ExperimentConfig
from daedalus.core.experiment import Experiment, ExperimentStatus, Reflection
from daedalus.core.hypothesis import Hypothesis
from daedalus.core.ledger import Ledger


def _make_experiment(exp_id: str = "exp_001", **kwargs) -> Experiment:
    defaults = dict(
        id=exp_id,
        hypothesis=Hypothesis(statement="test hypothesis", rationale="because"),
        config=ExperimentConfig(script="train.py"),
    )
    defaults.update(kwargs)
    return Experiment(**defaults)


class TestLedgerCRUD:
    def test_append_and_get(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        exp = _make_experiment("exp_001")
        ledger.append(exp)
        assert ledger.get("exp_001") is not None
        assert ledger.get("exp_001").id == "exp_001"

    def test_append_duplicate_raises(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        exp = _make_experiment("exp_001")
        ledger.append(exp)
        with pytest.raises(ValueError, match="already exists"):
            ledger.append(exp)

    def test_update(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        exp = _make_experiment("exp_001")
        ledger.append(exp)

        updated = exp.transition(ExperimentStatus.QUEUED)
        ledger.update(updated)

        fetched = ledger.get("exp_001")
        assert fetched.status == ExperimentStatus.QUEUED

    def test_update_nonexistent_raises(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        exp = _make_experiment("exp_999")
        with pytest.raises(ValueError, match="not found"):
            ledger.update(exp)

    def test_get_nonexistent(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        assert ledger.get("nope") is None

    def test_len(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        assert len(ledger) == 0
        ledger.append(_make_experiment("exp_001"))
        ledger.append(_make_experiment("exp_002"))
        assert len(ledger) == 2


class TestLedgerQueries:
    def test_latest(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        ledger.append(_make_experiment("exp_001"))
        ledger.append(_make_experiment("exp_002"))
        ledger.append(_make_experiment("exp_003"))

        latest = ledger.latest(2)
        assert len(latest) == 2
        # Most recent first
        assert latest[0].created_at >= latest[1].created_at

    def test_by_status(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        exp1 = _make_experiment("exp_001")
        exp2 = _make_experiment("exp_002", status=ExperimentStatus.COMPLETED)
        ledger.append(exp1)
        ledger.append(exp2)

        drafts = ledger.by_status(ExperimentStatus.DRAFT)
        assert len(drafts) == 1
        assert drafts[0].id == "exp_001"

    def test_baseline(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        base = _make_experiment("exp_001")
        child = _make_experiment("exp_002", baseline_id="exp_001")
        ledger.append(base)
        ledger.append(child)

        assert ledger.baseline("exp_002").id == "exp_001"
        assert ledger.baseline("exp_001") is None

    def test_all_ordered(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        ledger.append(_make_experiment("exp_002"))
        ledger.append(_make_experiment("exp_001"))
        ledger.append(_make_experiment("exp_003"))

        all_exps = ledger.all()
        assert len(all_exps) == 3
        for i in range(len(all_exps) - 1):
            assert all_exps[i].created_at <= all_exps[i + 1].created_at


class TestLedgerNarrative:
    def test_narrative_generated(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        exp = _make_experiment(
            "exp_001",
            config_diff={"lr": {"from": 1e-4, "to": 5e-6}},
            results={"simpleqa": {"correct": 3.5, "not_attempted": 41.0}},
            reflection=Reflection(
                hypothesis_confirmed="partial",
                analysis="Not_attempted reduced but correct flat.",
                next_suggestions=["Try penalty=-0.3"],
            ),
            status=ExperimentStatus.ANALYZED,
        )
        ledger.append(exp)

        narrative = ledger.narrative()
        assert "exp_001" in narrative
        assert "test hypothesis" in narrative
        assert "Not_attempted reduced" in narrative
        assert "Try penalty=-0.3" in narrative

    def test_empty_narrative(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        narrative = ledger.narrative()
        # File not generated yet until first write
        assert narrative == ""

    def test_iteration(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        ledger.append(_make_experiment("exp_001"))
        ledger.append(_make_experiment("exp_002"))

        ids = [e.id for e in ledger]
        assert "exp_001" in ids
        assert "exp_002" in ids
