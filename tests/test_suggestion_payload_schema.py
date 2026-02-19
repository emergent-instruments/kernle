"""Tests for suggestion payload key contracts.

Verifies that _make_suggestion produces exactly the expected keys for each
memory type, preventing accidental key duplication or schema drift.

Addresses: #711 (FIND-STK-01)
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kernle.stack.components.suggestions import SuggestionComponent


@pytest.fixture
def component():
    """SuggestionComponent attached to a test stack."""
    c = SuggestionComponent()
    c.attach(stack_id="test-schema")
    return c


def _raw_entry(content: str, entry_id: str | None = None):
    """Build a minimal raw entry mock."""
    entry = MagicMock()
    entry.id = entry_id or str(uuid.uuid4())
    entry.blob = content
    entry.content = content
    return entry


# ============================================================================
# Exact key contracts per memory type
# ============================================================================


class TestSuggestionPayloadKeys:
    """Each memory_type must produce exactly the expected content dict keys."""

    def test_episode_content_keys(self, component):
        """Episode suggestion content has exactly: objective, outcome, outcome_type."""
        entry = _raw_entry("I completed the deployment and it succeeded with all tests passing")
        suggestion = component._make_suggestion(entry, "episode", 0.8)
        assert suggestion is not None
        assert set(suggestion.content.keys()) == {"objective", "outcome", "outcome_type"}

    def test_belief_content_keys(self, component):
        """Belief suggestion content has exactly: statement, belief_type, confidence."""
        entry = _raw_entry(
            "I believe that test-driven development always leads to better code quality"
        )
        suggestion = component._make_suggestion(entry, "belief", 0.7)
        assert suggestion is not None
        assert set(suggestion.content.keys()) == {"statement", "belief_type", "confidence"}

    def test_note_content_keys(self, component):
        """Note suggestion content has exactly: content, note_type.

        note_type inside content is the NOTE SUB-TYPE (observation, decision,
        quote, etc.) and is distinct from the suggestion-level memory_type
        which is always "note" for note suggestions.
        """
        entry = _raw_entry("This is an interesting observation about the system architecture")
        suggestion = component._make_suggestion(entry, "note", 0.6)
        assert suggestion is not None
        assert set(suggestion.content.keys()) == {"content", "note_type"}

    def test_note_type_vs_memory_type_distinction(self, component):
        """note_type in content is the sub-type, memory_type is the category."""
        entry = _raw_entry("This is a noteworthy observation about system behavior patterns")
        suggestion = component._make_suggestion(entry, "note", 0.6)
        assert suggestion is not None
        # memory_type is the suggestion category
        assert suggestion.memory_type == "note"
        # note_type is the note sub-type (always "note" from pattern extraction)
        assert suggestion.content["note_type"] == "note"


# ============================================================================
# No duplicate keys in content dict
# ============================================================================


class TestNoDuplicateKeys:
    """Content dicts must not have overlapping/duplicate keys."""

    def test_episode_no_type_key(self, component):
        """Episode content should not have a 'type' key (uses outcome_type instead)."""
        entry = _raw_entry("Successfully deployed the release after careful testing and review")
        suggestion = component._make_suggestion(entry, "episode", 0.8)
        assert suggestion is not None
        assert "type" not in suggestion.content

    def test_belief_no_type_key(self, component):
        """Belief content should not have a 'type' key (uses belief_type instead)."""
        entry = _raw_entry("I think that continuous integration is the best approach for teams")
        suggestion = component._make_suggestion(entry, "belief", 0.7)
        assert suggestion is not None
        assert "type" not in suggestion.content

    def test_note_no_memory_type_key(self, component):
        """Note content should not have 'memory_type' key (that's at suggestion level)."""
        entry = _raw_entry("An important observation about how the cache invalidation works")
        suggestion = component._make_suggestion(entry, "note", 0.6)
        assert suggestion is not None
        assert "memory_type" not in suggestion.content


# ============================================================================
# Short content rejection
# ============================================================================


class TestShortContentRejection:
    """Content shorter than 10 chars should return None."""

    def test_short_content_returns_none(self, component):
        entry = _raw_entry("hi")
        assert component._make_suggestion(entry, "note", 0.5) is None

    def test_whitespace_only_returns_none(self, component):
        entry = _raw_entry("         ")
        assert component._make_suggestion(entry, "note", 0.5) is None


# ============================================================================
# Accept suggestion key consumption
# ============================================================================


class TestAcceptSuggestionKeyConsumption:
    """accept_suggestion correctly reads keys from each content dict type."""

    @pytest.fixture
    def stack(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        return Stack.from_sqlite(
            stack_id="test-accept",
            db_path=tmp_path / "accept.db",
            components=[],
            enforce_provenance=False,
        )

    def test_accept_note_reads_note_type_from_content(self, stack):
        """accept_suggestion reads note_type from content dict, not memory_type."""
        from kernle.types import MemorySuggestion

        suggestion = MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id="test-accept",
            memory_type="note",
            content={"content": "A test observation", "note_type": "observation"},
            confidence=0.8,
            source_raw_ids=["raw:test-1"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        stack._backend.save_suggestion(suggestion)

        memory_id = stack.accept_suggestion(suggestion.id)
        assert memory_id is not None

        # Verify the created note has the correct note_type
        notes = stack._backend.get_notes(limit=10)
        created = [n for n in notes if n.id == memory_id]
        assert len(created) == 1
        assert created[0].note_type == "observation"

    def test_accept_episode_reads_content_keys(self, stack):
        """accept_suggestion reads objective/outcome from episode content."""
        from kernle.types import MemorySuggestion

        suggestion = MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id="test-accept",
            memory_type="episode",
            content={
                "objective": "Deploy the feature",
                "outcome": "Deployment succeeded",
                "outcome_type": "success",
            },
            confidence=0.9,
            source_raw_ids=["raw:test-2"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        stack._backend.save_suggestion(suggestion)

        memory_id = stack.accept_suggestion(suggestion.id)
        assert memory_id is not None

        episodes = stack._backend.get_episodes(limit=10)
        created = [e for e in episodes if e.id == memory_id]
        assert len(created) == 1
        assert created[0].objective == "Deploy the feature"
        assert created[0].outcome == "Deployment succeeded"

    def test_accept_belief_reads_content_keys(self, stack):
        """accept_suggestion reads statement/belief_type from belief content."""
        from kernle.types import MemorySuggestion

        suggestion = MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id="test-accept",
            memory_type="belief",
            content={
                "statement": "TDD leads to better code",
                "belief_type": "principle",
                "confidence": 0.85,
            },
            confidence=0.9,
            source_raw_ids=["raw:test-3"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        stack._backend.save_suggestion(suggestion)

        memory_id = stack.accept_suggestion(suggestion.id)
        assert memory_id is not None

        beliefs = stack._backend.get_beliefs(limit=10)
        created = [b for b in beliefs if b.id == memory_id]
        assert len(created) == 1
        assert created[0].statement == "TDD leads to better code"
        assert created[0].belief_type == "principle"
