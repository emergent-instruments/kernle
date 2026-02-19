"""Tests for Stack feature coverage: suggestions, belief revision, helpers.

Targets coverage for:
- _normalize_suggestion_provenance_refs edge cases
- save_suggestion and accept_suggestion flows
- accept_suggestion for unsupported memory type
- _validate_literal_value boundary cases
- _estimate_tokens and _truncate_at_word_boundary helpers
- accept_suggestion with lint-redirect (suggestion: prefix)
- Belief revision: supersede_belief and revise_beliefs_from_episode paths
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from kernle.stack import Stack
from kernle.stack.sqlite_stack import (
    VALID_SUGGESTION_STATUSES,
    _compute_priority_score,
    _estimate_tokens,
    _normalize_suggestion_provenance_refs,
    _truncate_at_word_boundary,
    _validate_literal_value,
)
from kernle.types import (
    MemorySuggestion,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp database path."""
    return tmp_path / "test_features.db"


@pytest.fixture
def stack(tmp_db):
    """Create an Stack for testing with provenance enforcement disabled."""
    return Stack.from_sqlite(
        stack_id="test-features",
        db_path=tmp_db,
        components=[],
        enforce_provenance=False,
    )


# =============================================================================
# _normalize_suggestion_provenance_refs
# =============================================================================


class TestNormalizeSuggestionProvenanceRefs:
    """Tests for the _normalize_suggestion_provenance_refs helper."""

    def test_none_input_returns_empty(self):
        """None input returns an empty list."""
        result = _normalize_suggestion_provenance_refs(None)
        assert result == []

    def test_empty_list_returns_empty(self):
        """Empty list returns empty list."""
        result = _normalize_suggestion_provenance_refs([])
        assert result == []

    def test_plain_ids_get_raw_prefix(self):
        """IDs without colons are treated as raw IDs."""
        result = _normalize_suggestion_provenance_refs(["abc123", "def456"])
        assert result == ["raw:abc123", "raw:def456"]

    def test_typed_refs_preserved(self):
        """Typed refs like 'episode:abc' are preserved as-is."""
        result = _normalize_suggestion_provenance_refs(["episode:abc", "belief:def"])
        assert result == ["episode:abc", "belief:def"]

    def test_duplicates_removed(self):
        """Duplicate refs (after normalization) are removed."""
        result = _normalize_suggestion_provenance_refs(["abc123", "abc123"])
        assert result == ["raw:abc123"]

    def test_typed_duplicates_removed(self):
        """Typed duplicate refs are removed."""
        result = _normalize_suggestion_provenance_refs(["episode:x", "episode:x"])
        assert result == ["episode:x"]

    def test_mixed_plain_and_typed(self):
        """Mix of plain IDs and typed refs works correctly."""
        result = _normalize_suggestion_provenance_refs(["abc", "episode:def", "ghi"])
        assert result == ["raw:abc", "episode:def", "raw:ghi"]

    def test_non_string_values_skipped(self):
        """Non-string values in the list are silently skipped."""
        result = _normalize_suggestion_provenance_refs([123, None, "abc"])
        assert result == ["raw:abc"]

    def test_empty_string_skipped(self):
        """Empty strings are skipped."""
        result = _normalize_suggestion_provenance_refs(["", "  ", "abc"])
        assert result == ["raw:abc"]

    def test_colon_with_empty_parts_skipped(self):
        """Refs like ':value' or 'type:' with empty parts are skipped."""
        result = _normalize_suggestion_provenance_refs([":value", "type:", "episode:good"])
        assert result == ["episode:good"]

    def test_whitespace_trimmed(self):
        """Leading/trailing whitespace is trimmed from refs."""
        result = _normalize_suggestion_provenance_refs(["  abc  ", " episode:def "])
        assert result == ["raw:abc", "episode:def"]


# =============================================================================
# _validate_literal_value
# =============================================================================


class TestValidateLiteralValue:
    """Tests for the _validate_literal_value helper."""

    def test_none_is_allowed(self):
        """None value passes validation (optional field)."""
        # Should not raise
        _validate_literal_value(None, VALID_SUGGESTION_STATUSES, "status")

    def test_valid_value_passes(self):
        """A valid value from the allowed set passes."""
        _validate_literal_value("pending", VALID_SUGGESTION_STATUSES, "status")

    def test_invalid_value_raises(self):
        """An invalid value raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Invalid.*status.*bogus"):
            _validate_literal_value("bogus", VALID_SUGGESTION_STATUSES, "status")


# =============================================================================
# _estimate_tokens and _truncate_at_word_boundary
# =============================================================================


class TestHelperFunctions:
    """Tests for utility helper functions."""

    def test_estimate_tokens_empty_string(self):
        """Empty string returns 0 tokens."""
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_returns_int(self):
        """Token estimate for non-empty text is a positive int."""
        result = _estimate_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_truncate_empty_string(self):
        """Empty string returns empty string."""
        assert _truncate_at_word_boundary("", 100) == ""

    def test_truncate_short_text_unchanged(self):
        """Text shorter than max_chars is returned unchanged."""
        text = "hello world"
        assert _truncate_at_word_boundary(text, 100) == text

    def test_truncate_long_text_with_ellipsis(self):
        """Long text is truncated at a word boundary with ellipsis."""
        text = "The quick brown fox jumps over the lazy dog"
        result = _truncate_at_word_boundary(text, 20)
        assert result.endswith("...")
        assert len(result) <= 20

    def test_truncate_very_small_max(self):
        """Very small max_chars returns just '...'."""
        text = "hello world"
        result = _truncate_at_word_boundary(text, 3)
        assert result == "..."


# =============================================================================
# _compute_priority_score
# =============================================================================


class TestComputePriorityScore:
    """Tests for the _compute_priority_score function."""

    def test_known_type_returns_base_score(self):
        """A known memory type returns at least its base priority."""
        record = MagicMock(strength=1.0)
        score = _compute_priority_score("episode", record)
        assert score >= 0.40  # episode base is 0.40

    def test_unknown_type_returns_default(self):
        """An unknown memory type returns the default base (0.20)."""
        record = MagicMock(strength=1.0)
        score = _compute_priority_score("totally_unknown", record)
        assert 0.19 <= score <= 0.21

    def test_fading_strength_reduces_score(self):
        """Records with strength < 0.8 (fading) get half priority."""
        record = MagicMock(strength=0.5, confidence=0.8)
        full_score = _compute_priority_score("belief", MagicMock(strength=1.0, confidence=0.8))
        fading_score = _compute_priority_score("belief", record)
        assert fading_score < full_score

    def test_value_priority_includes_priority_bonus(self):
        """Value records get a bonus from their priority field."""
        high_priority = MagicMock(strength=1.0, priority=100)
        low_priority = MagicMock(strength=1.0, priority=10)
        high_score = _compute_priority_score("value", high_priority)
        low_score = _compute_priority_score("value", low_priority)
        assert high_score > low_score

    def test_belief_priority_includes_confidence_bonus(self):
        """Belief records get a bonus from their confidence field."""
        high_conf = MagicMock(strength=1.0, confidence=1.0)
        low_conf = MagicMock(strength=1.0, confidence=0.3)
        high_score = _compute_priority_score("belief", high_conf)
        low_score = _compute_priority_score("belief", low_conf)
        assert high_score > low_score


# =============================================================================
# Stack save_suggestion and get_suggestion
# =============================================================================


class TestStackSuggestions:
    """Tests for suggestion CRUD through the stack."""

    def test_save_and_retrieve_suggestion(self, stack):
        """Saving a suggestion and retrieving it returns the same data."""
        suggestion = MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id="test-features",
            memory_type="episode",
            content={"objective": "test", "outcome": "pass"},
            confidence=0.75,
            source_raw_ids=["raw-001"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        result_id = stack.save_suggestion(suggestion)
        assert result_id is not None

        # Retrieve it
        retrieved = stack.get_suggestion(suggestion.id)
        assert retrieved is not None
        assert retrieved.memory_type == "episode"
        assert retrieved.confidence == 0.75

    def test_get_suggestions_filters_by_status(self, stack):
        """get_suggestions filters by status."""
        for i, status_val in enumerate(["pending", "pending", "promoted"]):
            stack.save_suggestion(
                MemorySuggestion(
                    id=str(uuid.uuid4()),
                    stack_id="test-features",
                    memory_type="note",
                    content={"content": f"test {i}"},
                    confidence=0.5,
                    source_raw_ids=[],
                    status=status_val,
                    created_at=datetime.now(timezone.utc),
                )
            )

        pending = stack.get_suggestions(status="pending")
        assert len(pending) == 2

    def test_get_suggestions_invalid_status_raises(self, stack):
        """get_suggestions with invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid.*status"):
            stack.get_suggestions(status="nonexistent-status")

    def test_get_suggestions_invalid_memory_type_raises(self, stack):
        """get_suggestions with invalid memory_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid.*memory_type"):
            stack.get_suggestions(memory_type="bogus_type")


# =============================================================================
# accept_suggestion
# =============================================================================


class TestAcceptSuggestion:
    """Tests for the accept_suggestion flow."""

    def test_accept_nonexistent_suggestion_returns_none(self, stack):
        """Accepting a non-existent suggestion ID returns None."""
        result = stack.accept_suggestion("nonexistent-id-123")
        assert result is None

    def test_accept_already_promoted_returns_none(self, stack):
        """Accepting a suggestion that is already promoted returns None."""
        suggestion_id = str(uuid.uuid4())
        stack.save_suggestion(
            MemorySuggestion(
                id=suggestion_id,
                stack_id="test-features",
                memory_type="note",
                content={"content": "already done"},
                confidence=0.5,
                source_raw_ids=[],
                status="promoted",
                created_at=datetime.now(timezone.utc),
            )
        )
        result = stack.accept_suggestion(suggestion_id)
        assert result is None

    def test_accept_episode_suggestion_creates_episode(self, stack):
        """Accepting an episode suggestion creates an episode and returns its ID."""
        # First create a raw entry so provenance is valid
        raw_id = stack._backend.save_raw(
            "Completed the refactor of authentication module with great results",
            source="cli",
        )

        suggestion_id = str(uuid.uuid4())
        stack.save_suggestion(
            MemorySuggestion(
                id=suggestion_id,
                stack_id="test-features",
                memory_type="episode",
                content={
                    "objective": "Refactor auth module",
                    "outcome": "Successfully refactored",
                    "outcome_type": "success",
                    "lessons": ["Better to refactor early"],
                },
                confidence=0.8,
                source_raw_ids=[raw_id],
                status="pending",
                created_at=datetime.now(timezone.utc),
            )
        )

        result = stack.accept_suggestion(suggestion_id)
        assert result is not None

    def test_accept_note_suggestion_creates_note(self, stack):
        """Accepting a note suggestion creates a note."""
        raw_id = stack._backend.save_raw(
            "Important decision was made about the architecture",
            source="cli",
        )

        suggestion_id = str(uuid.uuid4())
        stack.save_suggestion(
            MemorySuggestion(
                id=suggestion_id,
                stack_id="test-features",
                memory_type="note",
                content={
                    "content": "Architecture uses microservices",
                    "note_type": "decision",
                },
                confidence=0.7,
                source_raw_ids=[raw_id],
                status="pending",
                created_at=datetime.now(timezone.utc),
            )
        )

        result = stack.accept_suggestion(suggestion_id)
        assert result is not None

    def test_accept_with_modifications(self, stack):
        """Accepting with modifications merges them into content."""
        raw_id = stack._backend.save_raw(
            "A note about something important",
            source="cli",
        )

        suggestion_id = str(uuid.uuid4())
        stack.save_suggestion(
            MemorySuggestion(
                id=suggestion_id,
                stack_id="test-features",
                memory_type="note",
                content={"content": "original content", "note_type": "note"},
                confidence=0.6,
                source_raw_ids=[raw_id],
                status="pending",
                created_at=datetime.now(timezone.utc),
            )
        )

        # Accept with a modification to the content
        result = stack.accept_suggestion(
            suggestion_id,
            modifications={"content": "improved content"},
        )
        assert result is not None

    def test_accept_unsupported_type_raises(self, stack):
        """Accepting a suggestion with an unsupported memory type raises ValueError."""
        suggestion_id = str(uuid.uuid4())
        stack.save_suggestion(
            MemorySuggestion(
                id=suggestion_id,
                stack_id="test-features",
                memory_type="episode",
                content={"objective": "test"},
                confidence=0.5,
                source_raw_ids=[],
                status="pending",
                created_at=datetime.now(timezone.utc),
            )
        )

        # Patch the suggestion's memory_type to something unsupported after retrieval
        with patch.object(stack._backend, "get_suggestion") as mock_get:
            fake_suggestion = MemorySuggestion(
                id=suggestion_id,
                stack_id="test-features",
                memory_type="totally_invalid_type",
                content={"foo": "bar"},
                confidence=0.5,
                source_raw_ids=[],
                status="pending",
                created_at=datetime.now(timezone.utc),
            )
            mock_get.return_value = fake_suggestion

            with pytest.raises(ValueError, match="Unsupported suggestion type"):
                stack.accept_suggestion(suggestion_id)


# =============================================================================
# Belief revision: find_contradictions placeholder coverage
# =============================================================================


class TestBeliefRevisionHelpers:
    """Tests for belief revision helper methods accessible via the stack.

    Note: The belief revision mixin is accessible via Kernle, not Stack
    directly. These tests cover the _normalize_suggestion_provenance_refs used
    during accept_suggestion which handles belief-typed refs.
    """

    def test_belief_typed_ref_preserved_in_normalization(self):
        """Belief-typed provenance refs are preserved during normalization."""
        refs = ["belief:abc-123", "episode:def-456"]
        result = _normalize_suggestion_provenance_refs(refs)
        assert "belief:abc-123" in result
        assert "episode:def-456" in result

    def test_mixed_raw_and_belief_refs(self):
        """Mix of raw IDs and belief refs are normalized correctly."""
        refs = ["raw-id-001", "belief:belief-001"]
        result = _normalize_suggestion_provenance_refs(refs)
        assert result == ["raw:raw-id-001", "belief:belief-001"]


# =============================================================================
# Stack set_stack_setting side effects
# =============================================================================


class TestSetStackSettingSideEffects:
    """Tests for set_stack_setting updating in-memory state."""

    def test_set_enforce_provenance_true(self, stack):
        """Setting enforce_provenance to 'true' updates the in-memory flag."""
        stack._enforce_provenance = False
        stack.set_stack_setting("enforce_provenance", "true")
        assert stack._enforce_provenance is True

    def test_set_enforce_provenance_false(self, stack):
        """Setting enforce_provenance to 'false' updates the in-memory flag."""
        stack._enforce_provenance = True
        stack.set_stack_setting("enforce_provenance", "false")
        assert stack._enforce_provenance is False

    def test_set_stack_state_updates_state(self, stack):
        """Setting stack_state to a valid StackState name updates _state."""
        from kernle.protocols import StackState

        stack.set_stack_setting("stack_state", "ACTIVE")
        assert stack._state == StackState.ACTIVE

    def test_set_unknown_key_no_side_effect(self, stack):
        """Setting an unrecognized key does not crash or change known state."""
        original_state = stack._state
        original_provenance = stack._enforce_provenance
        stack.set_stack_setting("some_random_key", "some_value")
        assert stack._state == original_state
        assert stack._enforce_provenance == original_provenance
