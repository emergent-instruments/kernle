"""Regression tests for TrustComponent, BeliefRevisionComponent, and
on_save advisory metadata logging (#818 audit findings).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from kernle.stack.components.belief_revision import (
    BeliefRevisionComponent,
    detect_opposition,
)
from kernle.stack.components.trust import TrustComponent
from kernle.types import TRUST_THRESHOLDS, TrustAssessment

# ---------------------------------------------------------------------------
# TrustComponent.on_save — threshold key regression (#818 finding 1)
# ---------------------------------------------------------------------------


class TestTrustComponentOnSave:
    """TrustComponent.on_save must fire for normal save paths using the
    lowest TRUST_THRESHOLDS value as baseline, not nonexistent action keys.
    """

    def _make_component(self, trust_score=0.1):
        """Create a TrustComponent with a mock storage returning a fixed trust."""
        comp = TrustComponent()
        comp.attach("test-stack")

        assessment = TrustAssessment(
            id="ta-1",
            stack_id="test-stack",
            entity="untrusted-source",
            dimensions={"general": {"score": trust_score}},
            authority=[],
        )

        storage = MagicMock()
        storage.get_trust_assessment.return_value = assessment
        comp.set_storage(storage)
        return comp

    def test_on_save_fires_for_low_trust_episode(self):
        """on_save returns warning when source entity trust is below threshold."""
        comp = self._make_component(trust_score=0.1)
        memory = SimpleNamespace(source_entity="untrusted-source")

        result = comp.on_save("episode", "ep-1", memory)

        assert result is not None
        assert "trust_warning" in result
        assert result["trust_level"] == 0.1
        assert "untrusted-source" in result["trust_warning"]

    def test_on_save_fires_for_low_trust_note(self):
        """on_save works for note memory type (not just episodes)."""
        comp = self._make_component(trust_score=0.1)
        memory = SimpleNamespace(source_entity="untrusted-source")

        result = comp.on_save("note", "n-1", memory)

        assert result is not None
        assert "trust_warning" in result

    def test_on_save_passes_for_high_trust(self):
        """on_save returns None when trust meets threshold."""
        threshold = min(TRUST_THRESHOLDS.values())
        comp = self._make_component(trust_score=threshold + 0.1)
        memory = SimpleNamespace(source_entity="untrusted-source")

        result = comp.on_save("episode", "ep-1", memory)

        assert result is None

    def test_on_save_uses_defined_thresholds(self):
        """on_save threshold derives from TRUST_THRESHOLDS, not hardcoded keys."""
        comp = self._make_component(trust_score=0.25)
        memory = SimpleNamespace(source_entity="untrusted-source")

        # 0.25 < 0.3 (min threshold = suggest_belief)
        result = comp.on_save("episode", "ep-1", memory)
        assert result is not None

    def test_on_save_no_source_entity(self):
        """on_save returns None for memories without source_entity."""
        comp = self._make_component()
        memory = SimpleNamespace()  # no source_entity

        result = comp.on_save("episode", "ep-1", memory)
        assert result is None

    def test_on_save_no_storage(self):
        """on_save returns None when storage is unavailable."""
        comp = TrustComponent()
        comp.attach("test-stack")
        memory = SimpleNamespace(source_entity="someone")

        result = comp.on_save("episode", "ep-1", memory)
        assert result is None

    def test_on_save_unknown_entity_warns(self):
        """on_save warns when no assessment exists for the source entity."""
        comp = TrustComponent()
        comp.attach("test-stack")
        storage = MagicMock()
        storage.get_trust_assessment.return_value = None
        comp.set_storage(storage)

        memory = SimpleNamespace(source_entity="stranger")
        result = comp.on_save("episode", "ep-1", memory)

        assert result is not None
        assert result["trust_level"] == 0.0
        assert "stranger" in result["trust_warning"]

    def test_on_save_all_authority_passes(self):
        """on_save returns None for entities with 'all' authority scope."""
        comp = TrustComponent()
        comp.attach("test-stack")

        assessment = TrustAssessment(
            id="ta-1",
            stack_id="test-stack",
            entity="admin",
            dimensions={"general": {"score": 0.0}},  # Low score but has authority
            authority=[{"scope": "all"}],
        )

        storage = MagicMock()
        storage.get_trust_assessment.return_value = assessment
        comp.set_storage(storage)

        memory = SimpleNamespace(source_entity="admin")
        result = comp.on_save("episode", "ep-1", memory)
        assert result is None


# ---------------------------------------------------------------------------
# Advisory metadata logging (#818 finding 2)
# ---------------------------------------------------------------------------


class TestAdvisoryMetadataLogging:
    """_persist_on_save_metadata must log advisory keys from Trust and
    BeliefRevision components rather than silently dropping them.
    """

    def test_trust_advisory_is_logged(self, tmp_path, caplog):
        """Trust warning metadata is logged at info level, not dropped."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-advisory",
            db_path=tmp_path / "test.db",
            enforce_provenance=False,
        )

        metadata = {
            "trust_warning": "Low trust (0.10) for stranger",
            "trust_level": 0.1,
            "domain": "general",
        }

        with caplog.at_level(logging.INFO, logger="kernle.stack.sqlite_stack"):
            stack._persist_on_save_metadata("episode", "ep-fake-id", metadata)

        assert any("trust_warning" in record.message for record in caplog.records)

    def test_contradiction_advisory_is_logged(self, tmp_path, caplog):
        """BeliefRevision contradiction metadata is logged, not dropped."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-advisory",
            db_path=tmp_path / "test.db",
            enforce_provenance=False,
        )

        metadata = {
            "contradictions": [
                {
                    "belief_id": "b-1",
                    "statement": "Testing is good",
                    "opposition_score": 0.7,
                }
            ]
        }

        with caplog.at_level(logging.INFO, logger="kernle.stack.sqlite_stack"):
            stack._persist_on_save_metadata("belief", "b-new", metadata)

        assert any("contradictions" in record.message for record in caplog.records)

    def test_emotional_fields_still_persisted(self, tmp_path):
        """Schema-backed emotional fields are still persisted (not regressed)."""
        from kernle.stack.sqlite_stack import Stack
        from kernle.storage import Episode

        stack = Stack.from_sqlite(
            "test-persist",
            db_path=tmp_path / "test.db",
            enforce_provenance=False,
        )

        # Save an episode first
        ep = Episode(
            id="ep-emotion-test",
            stack_id="test-persist",
            objective="Test episode",
            outcome="Success",
            outcome_type="success",
            source_type="direct",
        )
        stack.save_episode(ep)

        # Persist emotional metadata (simulating EmotionalTaggingComponent output)
        metadata = {
            "emotional_valence": 0.8,
            "emotional_arousal": 0.6,
            "emotional_tags": ["joy", "pride"],
        }
        stack._persist_on_save_metadata("episode", "ep-emotion-test", metadata)

        # Verify it was persisted by reading back from storage
        with stack._backend._connect() as conn:
            row = conn.execute(
                "SELECT emotional_valence, emotional_arousal FROM episodes "
                "WHERE id = ? AND stack_id = ?",
                ("ep-emotion-test", "test-persist"),
            ).fetchone()
        assert row is not None
        assert row[0] == 0.8
        assert row[1] == 0.6


# ---------------------------------------------------------------------------
# BeliefRevisionComponent.on_save — detect_opposition function
# ---------------------------------------------------------------------------


class TestBeliefRevisionOnSave:
    def test_detect_opposition_with_opposing_words(self):
        """detect_opposition identifies opposition word pairs."""
        result = detect_opposition(
            "testing is always important for quality",
            "testing is never important for quality",
        )
        assert result["score"] > 0
        assert result["type"] == "opposition_words"

    def test_detect_opposition_no_overlap(self):
        """detect_opposition returns score 0 when topics don't overlap."""
        result = detect_opposition(
            "python is good",
            "weather is bad",
        )
        assert result["score"] == 0.0

    def test_on_save_only_for_beliefs(self):
        """on_save returns None for non-belief memory types."""
        comp = BeliefRevisionComponent()
        comp.attach("test-stack")
        comp.set_storage(MagicMock())

        memory = SimpleNamespace(statement="testing is good")
        assert comp.on_save("episode", "ep-1", memory) is None
        assert comp.on_save("note", "n-1", memory) is None

    def test_on_save_returns_contradictions(self):
        """on_save returns contradictions when opposing beliefs found."""
        comp = BeliefRevisionComponent()
        comp.attach("test-stack")

        existing_belief = SimpleNamespace(
            id="b-existing",
            statement="Testing is never important",
            confidence=0.8,
            times_reinforced=2,
            is_active=True,
        )
        search_result = SimpleNamespace(
            record_type="belief",
            record=existing_belief,
            score=0.9,
        )

        storage = MagicMock()
        storage.search.return_value = [search_result]
        comp.set_storage(storage)

        new_belief = SimpleNamespace(statement="Testing is always important for quality")
        result = comp.on_save("belief", "b-new", new_belief)

        assert result is not None
        assert "contradictions" in result
        assert len(result["contradictions"]) > 0
        assert result["contradictions"][0]["belief_id"] == "b-existing"

    def test_on_save_no_contradictions_returns_none(self):
        """on_save returns None when no contradictions found."""
        comp = BeliefRevisionComponent()
        comp.attach("test-stack")

        storage = MagicMock()
        storage.search.return_value = []
        comp.set_storage(storage)

        memory = SimpleNamespace(statement="Python is a programming language")
        result = comp.on_save("belief", "b-1", memory)
        assert result is None
