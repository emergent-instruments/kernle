"""Unit tests for Stack complex methods.

These tests cover:
- accept_suggestion() with different memory types (episode, belief, note, goal, value, drive, relationship)
- accept_suggestion() rejecting invalid/unsupported types
- accept_suggestion() returning None for non-pending suggestions
- accept_suggestion() applying modifications to content
- _validate_provenance() rejecting missing source references
- _validate_provenance() allowing INITIALIZING state writes
- _validate_provenance() blocking MAINTENANCE mode writes
- _normalize_suggestion_provenance_refs() normalizing typed and untyped refs
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kernle.protocols import MaintenanceModeError, ProvenanceError
from kernle.stack.sqlite_stack import (
    Stack,
    _normalize_suggestion_provenance_refs,
)
from kernle.types import MemorySuggestion

# =========================================================================
# Helpers
# =========================================================================


def _uid():
    """Generate a unique ID for test data."""
    return str(uuid.uuid4())


def _now():
    """Get the current UTC timestamp."""
    return datetime.now(timezone.utc)


def _make_suggestion(
    memory_type="episode",
    content=None,
    status="pending",
    source_raw_ids=None,
):
    """Create a MemorySuggestion with sensible defaults for testing."""
    if content is None:
        # Build default content based on memory_type
        content_map = {
            "episode": {"objective": "Test objective", "outcome": "Test outcome"},
            "belief": {"statement": "Test belief statement"},
            "note": {"content": "Test note content"},
            "goal": {"title": "Test goal title"},
            "value": {"name": "Test value", "statement": "Test value statement"},
            "relationship": {
                "entity_name": "TestEntity",
                "entity_type": "person",
                "relationship_type": "colleague",
            },
            "drive": {"drive_type": "curiosity", "intensity": 0.5},
        }
        content = content_map.get(memory_type, {"data": "test"})

    return MemorySuggestion(
        id=_uid(),
        stack_id="test-stack",
        memory_type=memory_type,
        content=content,
        confidence=0.8,
        source_raw_ids=source_raw_ids or [f"raw:{_uid()}"],
        status=status,
        created_at=_now(),
    )


@pytest.fixture
def stack(tmp_path):
    """Create a Stack in INITIALIZING state for testing.

    Uses a temporary directory for the database file.
    Components are set to bare (empty list) to avoid
    needing real component implementations during unit tests.
    """
    db_path = tmp_path / "test.db"
    stack = Stack.from_sqlite(
        stack_id="test-stack",
        db_path=db_path,
        enforce_provenance=False,
        components=[],
    )
    return stack


@pytest.fixture
def active_stack(tmp_path):
    """Create a Stack in ACTIVE state with provenance enforcement.

    This fixture transitions the stack to ACTIVE by attaching a mock core.
    """
    db_path = tmp_path / "test_active.db"
    stack = Stack.from_sqlite(
        stack_id="test-active-stack",
        db_path=db_path,
        enforce_provenance=True,
        components=[],
    )
    # Transition to ACTIVE state
    stack.on_attach("test-core-id")
    return stack


# =========================================================================
# accept_suggestion: Different memory types
# =========================================================================


class TestAcceptSuggestionMemoryTypes:
    """accept_suggestion should promote suggestions into the correct memory type."""

    def test_accept_episode_suggestion(self, stack):
        """Accepting an episode suggestion should create an Episode."""
        suggestion = _make_suggestion(memory_type="episode")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        memory_id = stack.accept_suggestion(suggestion.id)

        assert memory_id is not None
        # Verify the suggestion status was updated
        stack._backend.update_suggestion_status.assert_called_once()
        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["status"] in ("promoted", "modified")
        assert call_kwargs["promoted_to"].startswith("episode:")

    def test_accept_belief_suggestion(self, stack):
        """Accepting a belief suggestion should create a Belief."""
        suggestion = _make_suggestion(memory_type="belief")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        memory_id = stack.accept_suggestion(suggestion.id)

        assert memory_id is not None
        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["promoted_to"].startswith("belief:")

    def test_accept_note_suggestion(self, stack):
        """Accepting a note suggestion should create a Note."""
        suggestion = _make_suggestion(memory_type="note")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        memory_id = stack.accept_suggestion(suggestion.id)

        assert memory_id is not None
        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["promoted_to"].startswith("note:")

    def test_accept_goal_suggestion(self, stack):
        """Accepting a goal suggestion should create a Goal."""
        suggestion = _make_suggestion(memory_type="goal")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        memory_id = stack.accept_suggestion(suggestion.id)

        assert memory_id is not None
        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["promoted_to"].startswith("goal:")

    def test_accept_value_suggestion(self, stack):
        """Accepting a value suggestion should create a Value."""
        suggestion = _make_suggestion(memory_type="value")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        memory_id = stack.accept_suggestion(suggestion.id)

        assert memory_id is not None
        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["promoted_to"].startswith("value:")

    def test_accept_relationship_suggestion(self, stack):
        """Accepting a relationship suggestion should create a Relationship."""
        suggestion = _make_suggestion(memory_type="relationship")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        memory_id = stack.accept_suggestion(suggestion.id)

        assert memory_id is not None
        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["promoted_to"].startswith("relationship:")

    def test_accept_drive_suggestion(self, stack):
        """Accepting a drive suggestion should create a Drive."""
        suggestion = _make_suggestion(memory_type="drive")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        memory_id = stack.accept_suggestion(suggestion.id)

        assert memory_id is not None
        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["promoted_to"].startswith("drive:")


# =========================================================================
# accept_suggestion: Edge cases
# =========================================================================


class TestAcceptSuggestionEdgeCases:
    """accept_suggestion should handle invalid inputs gracefully."""

    def test_returns_none_for_nonexistent_suggestion(self, stack):
        """Should return None when the suggestion doesn't exist."""
        stack._backend.get_suggestion = MagicMock(return_value=None)

        result = stack.accept_suggestion("nonexistent-id")

        assert result is None

    def test_returns_none_for_non_pending_suggestion(self, stack):
        """Should return None when the suggestion is not in 'pending' status."""
        suggestion = _make_suggestion(status="promoted")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)

        result = stack.accept_suggestion(suggestion.id)

        assert result is None

    def test_unsupported_type_raises_value_error(self, stack):
        """Should raise ValueError for unsupported memory types."""
        suggestion = _make_suggestion(memory_type="unknown_type")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)

        with pytest.raises(ValueError, match="Unsupported suggestion type"):
            stack.accept_suggestion(suggestion.id)

    def test_modifications_applied_to_content(self, stack):
        """Modifications should be merged into the suggestion content."""
        suggestion = _make_suggestion(
            memory_type="episode",
            content={"objective": "Original objective", "outcome": "Original outcome"},
        )
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        memory_id = stack.accept_suggestion(
            suggestion.id,
            modifications={"outcome": "Modified outcome"},
        )

        assert memory_id is not None
        # Status should be 'modified' since modifications were provided
        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["status"] == "modified"

    def test_status_is_promoted_without_modifications(self, stack):
        """Status should be 'promoted' when no modifications are provided."""
        suggestion = _make_suggestion(memory_type="episode")
        stack._backend.get_suggestion = MagicMock(return_value=suggestion)
        stack._backend.update_suggestion_status = MagicMock(return_value=True)
        stack._backend.mark_raw_processed = MagicMock()

        stack.accept_suggestion(suggestion.id)

        call_kwargs = stack._backend.update_suggestion_status.call_args[1]
        assert call_kwargs["status"] == "promoted"


# =========================================================================
# _validate_provenance
# =========================================================================


class TestValidateProvenance:
    """Tests for Stack._validate_provenance."""

    def test_initializing_state_allows_any_write(self, stack):
        """In INITIALIZING state, provenance is not required."""
        # stack fixture starts in INITIALIZING state
        # This should NOT raise, even with no derived_from
        stack._validate_provenance("belief", None)

    def test_maintenance_mode_blocks_writes(self, active_stack):
        """In MAINTENANCE state, all writes should be blocked."""
        active_stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError, match="maintenance mode"):
            active_stack._validate_provenance("episode", ["raw:abc123"])

    def test_missing_derived_from_raises_provenance_error(self, active_stack):
        """In ACTIVE state, missing derived_from should raise ProvenanceError."""
        with pytest.raises(ProvenanceError, match="without provenance"):
            active_stack._validate_provenance("episode", None)

    def test_empty_derived_from_raises_provenance_error(self, active_stack):
        """In ACTIVE state, empty derived_from should raise ProvenanceError."""
        with pytest.raises(ProvenanceError, match="without provenance"):
            active_stack._validate_provenance("belief", [])

    def test_invalid_ref_format_raises_provenance_error(self, active_stack):
        """Refs without ':' separator should raise ProvenanceError."""
        with pytest.raises(ProvenanceError, match="Invalid provenance reference"):
            active_stack._validate_provenance("episode", ["no-colon-here"])

    def test_invalid_source_type_raises_provenance_error(self, active_stack):
        """Refs with invalid source types should raise ProvenanceError."""
        # Episodes can only come from raw entries (per PROVENANCE_RULES)
        # Mocking memory_exists to return True so it passes the existence check
        active_stack._backend.memory_exists = MagicMock(return_value=True)

        with pytest.raises(ProvenanceError, match="not an allowed source"):
            active_stack._validate_provenance("episode", ["belief:some-id"])

    def test_annotation_refs_alone_raises_provenance_error(self, active_stack):
        """Only annotation refs (context:, kernle:) should raise ProvenanceError."""
        with pytest.raises(ProvenanceError, match="only annotation refs"):
            active_stack._validate_provenance("episode", ["context:cli-session"])

    def test_system_source_entity_bypasses_provenance(self, active_stack):
        """Writes from kernle:* source_entity should bypass provenance."""
        # This should NOT raise despite missing derived_from
        active_stack._validate_provenance(
            "episode", None, source_entity="kernle:suggestion-promotion"
        )

    def test_registered_plugin_bypasses_provenance(self, active_stack):
        """Registered plugins should bypass provenance checks."""
        active_stack.register_plugin("my-plugin")

        # This should NOT raise despite missing derived_from
        active_stack._validate_provenance("episode", None, source_entity="plugin:my-plugin")

    def test_unregistered_plugin_does_not_bypass(self, active_stack):
        """Unregistered plugins should NOT bypass provenance checks."""
        with pytest.raises(ProvenanceError, match="without provenance"):
            active_stack._validate_provenance("episode", None, source_entity="plugin:unregistered")

    def test_pre_v09_annotation_bypasses_provenance(self, active_stack):
        """Pre-v0.9 migrated memories should bypass provenance."""
        # This should NOT raise
        active_stack._validate_provenance("belief", ["kernle:pre-v0.9:migrated"])

    def test_provenance_disabled_allows_any_write(self, active_stack):
        """When enforce_provenance is False, any write should be allowed."""
        active_stack._enforce_provenance = False

        # This should NOT raise even with no derived_from
        active_stack._validate_provenance("belief", None)


# =========================================================================
# _normalize_suggestion_provenance_refs
# =========================================================================


class TestNormalizeSuggestionProvenanceRefs:
    """Tests for the _normalize_suggestion_provenance_refs helper."""

    def test_plain_ids_get_raw_prefix(self):
        """Plain IDs without a colon should be prefixed with 'raw:'."""
        result = _normalize_suggestion_provenance_refs(["abc123"])
        assert result == ["raw:abc123"]

    def test_typed_refs_preserved(self):
        """Already-typed refs should be preserved as-is."""
        result = _normalize_suggestion_provenance_refs(["episode:abc123"])
        assert result == ["episode:abc123"]

    def test_duplicates_removed(self):
        """Duplicate refs should be deduplicated."""
        result = _normalize_suggestion_provenance_refs(["raw:abc", "raw:abc"])
        assert result == ["raw:abc"]

    def test_empty_strings_skipped(self):
        """Empty strings should be filtered out."""
        result = _normalize_suggestion_provenance_refs(["", "  ", "raw:valid"])
        assert result == ["raw:valid"]

    def test_non_string_values_skipped(self):
        """Non-string values should be skipped."""
        result = _normalize_suggestion_provenance_refs([123, None, "raw:valid"])
        assert result == ["raw:valid"]

    def test_none_input_returns_empty_list(self):
        """None input should return an empty list."""
        result = _normalize_suggestion_provenance_refs(None)
        assert result == []

    def test_mixed_typed_and_untyped_refs(self):
        """Mixed typed and untyped refs should be handled correctly."""
        result = _normalize_suggestion_provenance_refs(
            ["plain-id", "episode:ep-123", "belief:bel-456"]
        )
        assert "raw:plain-id" in result
        assert "episode:ep-123" in result
        assert "belief:bel-456" in result
        assert len(result) == 3
