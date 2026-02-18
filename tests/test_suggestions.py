"""Tests for memory suggestion system."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from kernle import Kernle
from kernle.storage import MemorySuggestion, RawEntry
from tests.conftest import bind_noop_model


class TestSuggestionExtraction:
    """Test suggestion extraction — returns empty without inference."""

    def test_no_inference_returns_empty(self, tmp_path):
        """Without inference, extract_suggestions returns empty list."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)

        raw_entry = RawEntry(
            id="raw-123",
            stack_id="test-agent",
            content="Completed the authentication module. It was successful. Lesson learned: testing early saves time.",
            timestamp=datetime.now(timezone.utc),
            source="cli",
        )

        suggestions = k.extract_suggestions(raw_entry, auto_save=False)
        assert suggestions == []

    def test_auto_save_no_inference_no_calls(self, tmp_path):
        """Without inference, auto_save=True does nothing (no suggestions to save)."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        k._storage.save_suggestion = MagicMock(return_value="saved-id")

        raw_entry = RawEntry(
            id="raw-123",
            stack_id="test-agent",
            content="Completed the task successfully. This was a great learning experience.",
            timestamp=datetime.now(timezone.utc),
            source="cli",
        )

        suggestions = k.extract_suggestions(raw_entry, auto_save=True)
        assert suggestions == []
        k._storage.save_suggestion.assert_not_called()

    def test_extract_suggestions_from_unprocessed(self, tmp_path):
        """Batch extraction should process each unprocessed raw entry."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        now = datetime.now(timezone.utc)
        raw_entries = [
            RawEntry(
                id="raw-a",
                stack_id="test-agent",
                content="first",
                timestamp=now,
                source="test",
            ),
            RawEntry(
                id="raw-b",
                stack_id="test-agent",
                content="second",
                timestamp=now,
                source="test",
            ),
        ]
        k._storage.list_raw = MagicMock(return_value=raw_entries)

        suggestion_a = MemorySuggestion(
            id="s-a",
            stack_id="test-agent",
            memory_type="note",
            content={"content": "a"},
            confidence=0.6,
            source_raw_ids=["raw-a"],
            status="pending",
            created_at=now,
        )
        suggestion_b = MemorySuggestion(
            id="s-b",
            stack_id="test-agent",
            memory_type="belief",
            content={"statement": "b", "belief_type": "fact", "confidence": 0.7},
            confidence=0.7,
            source_raw_ids=["raw-b"],
            status="pending",
            created_at=now,
        )
        k.extract_suggestions = MagicMock(side_effect=[[suggestion_a], [suggestion_b]])

        result = k.extract_suggestions_from_unprocessed(limit=2)

        assert len(result) == 2
        assert {item["id"] for item in result} == {"s-a", "s-b"}
        k._storage.list_raw.assert_called_once_with(processed=False, limit=2)
        k.extract_suggestions.assert_any_call(raw_entries[0], auto_save=True)
        k.extract_suggestions.assert_any_call(raw_entries[1], auto_save=True)


class TestSuggestionStorage:
    """Test storage operations for suggestions."""

    def test_save_and_get_suggestion(self, tmp_path):
        """Should save and retrieve a suggestion."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        suggestion = MemorySuggestion(
            id="sug-123",
            stack_id="test-agent",
            memory_type="episode",
            content={"objective": "Test objective", "outcome": "Test outcome"},
            confidence=0.75,
            source_raw_ids=["raw-1", "raw-2"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

        # Save
        saved_id = storage.save_suggestion(suggestion)
        assert saved_id == "sug-123"

        # Retrieve
        retrieved = storage.get_suggestion("sug-123")
        assert retrieved is not None
        assert retrieved.memory_type == "episode"
        assert retrieved.content["objective"] == "Test objective"
        assert retrieved.confidence == 0.75
        assert retrieved.source_raw_ids == ["raw-1", "raw-2"]
        assert retrieved.status == "pending"

        storage.close()

    def test_get_suggestions_filtered(self, tmp_path):
        """Should filter suggestions by status and type."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        # Create test suggestions
        suggestions = [
            MemorySuggestion(
                id="sug-1",
                stack_id="test-agent",
                memory_type="episode",
                content={},
                confidence=0.8,
                source_raw_ids=["raw-1"],
                status="pending",
                created_at=datetime.now(timezone.utc),
            ),
            MemorySuggestion(
                id="sug-2",
                stack_id="test-agent",
                memory_type="belief",
                content={},
                confidence=0.7,
                source_raw_ids=["raw-2"],
                status="pending",
                created_at=datetime.now(timezone.utc),
            ),
            MemorySuggestion(
                id="sug-3",
                stack_id="test-agent",
                memory_type="episode",
                content={},
                confidence=0.6,
                source_raw_ids=["raw-3"],
                status="promoted",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        for s in suggestions:
            storage.save_suggestion(s)

        # Filter by status
        pending = storage.get_suggestions(status="pending")
        assert len(pending) == 2

        promoted = storage.get_suggestions(status="promoted")
        assert len(promoted) == 1
        assert promoted[0].id == "sug-3"

        # Filter by type
        episodes = storage.get_suggestions(memory_type="episode")
        assert len(episodes) == 2

        # Filter by both
        pending_beliefs = storage.get_suggestions(status="pending", memory_type="belief")
        assert len(pending_beliefs) == 1
        assert pending_beliefs[0].id == "sug-2"

        storage.close()

    def test_update_suggestion_status(self, tmp_path):
        """Should update suggestion status."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        suggestion = MemorySuggestion(
            id="sug-update",
            stack_id="test-agent",
            memory_type="episode",
            content={},
            confidence=0.8,
            source_raw_ids=["raw-1"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Update status
        result = storage.update_suggestion_status(
            suggestion_id="sug-update",
            status="promoted",
            resolution_reason="Approved by user",
            promoted_to="episode:ep-123",
        )
        assert result is True

        # Verify update
        updated = storage.get_suggestion("sug-update")
        assert updated.status == "promoted"
        assert updated.resolution_reason == "Approved by user"
        assert updated.promoted_to == "episode:ep-123"
        assert updated.resolved_at is not None

        storage.close()

    def test_delete_suggestion(self, tmp_path):
        """Should soft delete a suggestion."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        suggestion = MemorySuggestion(
            id="sug-delete",
            stack_id="test-agent",
            memory_type="note",
            content={},
            confidence=0.5,
            source_raw_ids=["raw-1"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Delete
        result = storage.delete_suggestion("sug-delete")
        assert result is True

        # Should not be retrievable
        deleted = storage.get_suggestion("sug-delete")
        assert deleted is None

        storage.close()


class TestPromotionWorkflow:
    """Test the suggestion promotion workflow."""

    def test_promote_episode_suggestion(self, tmp_path):
        """Should promote episode suggestion to actual episode."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage, strict=False)

        # Create a suggestion
        suggestion = MemorySuggestion(
            id="sug-promote-ep",
            stack_id="test-agent",
            memory_type="episode",
            content={
                "objective": "Implement feature X",
                "outcome": "Successfully deployed",
                "outcome_type": "success",
                "lessons": ["Small commits are better"],
            },
            confidence=0.8,
            source_raw_ids=["raw-1"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Create a raw entry that would be marked as processed
        storage.save_raw("Test content", source="test")
        raw_entries = storage.list_raw(limit=1)
        if raw_entries:
            # Update the suggestion to reference a real raw entry
            suggestion.source_raw_ids = [raw_entries[0].id]
            storage.save_suggestion(suggestion)

        # Promote
        memory_id = k.promote_suggestion("sug-promote-ep")

        assert memory_id is not None

        # Verify episode was created
        episode = storage.get_episode(memory_id)
        assert episode is not None
        assert episode.objective == "Implement feature X"
        assert episode.outcome == "Successfully deployed"

        # Verify suggestion was updated
        updated_suggestion = storage.get_suggestion("sug-promote-ep")
        assert updated_suggestion.status == "promoted"
        assert f"episode:{memory_id}" in updated_suggestion.promoted_to

        storage.close()

    def test_promote_with_modifications(self, tmp_path):
        """Should apply modifications when promoting."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage, strict=False)
        bind_noop_model(k)

        suggestion = MemorySuggestion(
            id="sug-modify",
            stack_id="test-agent",
            memory_type="belief",
            content={
                "statement": "Original statement",
                "belief_type": "fact",
                "confidence": 0.7,
            },
            confidence=0.6,
            source_raw_ids=[],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Promote with modifications
        memory_id = k.promote_suggestion(
            "sug-modify",
            modifications={"statement": "Modified statement", "confidence": 0.9},
        )

        assert memory_id is not None

        # Verify belief has modified content
        beliefs = storage.get_beliefs(limit=100)
        created_belief = next((b for b in beliefs if b.id == memory_id), None)
        assert created_belief is not None
        assert created_belief.statement == "Modified statement"

        # Verify status is "modified" not "promoted"
        updated_suggestion = storage.get_suggestion("sug-modify")
        assert updated_suggestion.status == "modified"

        storage.close()

    def test_reject_suggestion(self, tmp_path):
        """Should reject a suggestion with reason."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage, strict=False)

        suggestion = MemorySuggestion(
            id="sug-reject",
            stack_id="test-agent",
            memory_type="note",
            content={"content": "Not useful"},
            confidence=0.3,
            source_raw_ids=[],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Reject
        result = k.reject_suggestion("sug-reject", reason="Low quality suggestion")
        assert result is True

        # Verify
        rejected = storage.get_suggestion("sug-reject")
        assert rejected.status == "rejected"
        assert rejected.resolution_reason == "Low quality suggestion"

        storage.close()

    def test_promote_non_pending_fails(self, tmp_path):
        """Should not promote already-resolved suggestions."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage, strict=False)

        suggestion = MemorySuggestion(
            id="sug-already-promoted",
            stack_id="test-agent",
            memory_type="episode",
            content={"objective": "Test", "outcome": "Done"},
            confidence=0.8,
            source_raw_ids=[],
            status="promoted",  # Already promoted
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Try to promote again
        result = k.promote_suggestion("sug-already-promoted")
        assert result is None

        storage.close()

    def test_promote_note_suggestion(self, tmp_path):
        """Note suggestions should promote into notes."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage, strict=False)

        suggestion = MemorySuggestion(
            id="sug-note",
            stack_id="test-agent",
            memory_type="note",
            content={
                "content": "Remember to pin dependency versions.",
                "note_type": "insight",
                "speaker": "alice",
                "reason": "build reproducibility",
            },
            confidence=0.75,
            source_raw_ids=[],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        memory_id = k.promote_suggestion("sug-note")
        assert memory_id is not None
        note = next((n for n in storage.get_notes(limit=100) if n.id == memory_id), None)
        assert note is not None
        assert "Remember to pin dependency versions." in note.content
        assert note.note_type in {"insight", "note"}

        storage.close()

    def test_promote_unknown_type_returns_none(self, tmp_path):
        """Unknown memory type should not promote."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage, strict=False)

        suggestion = MemorySuggestion(
            id="sug-unknown",
            stack_id="test-agent",
            memory_type="unknown",
            content={"foo": "bar"},
            confidence=0.75,
            source_raw_ids=[],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        assert k.promote_suggestion("sug-unknown") is None
        assert storage.get_suggestion("sug-unknown").status == "pending"

        storage.close()

    def test_strict_mode_delegates_to_stack_accept(self, tmp_path):
        """Strict mode should delegate promotion to stack.accept_suggestion."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage, strict=False)
        k._strict = True
        k._stack = MagicMock()
        k._stack.accept_suggestion = MagicMock(return_value="delegated-id")
        k._storage.get_suggestion = MagicMock(side_effect=AssertionError("should not be called"))

        assert k.promote_suggestion("any-id") == "delegated-id"
        k._stack.accept_suggestion.assert_called_once_with("any-id", None)

    def test_promote_preserves_typed_episode_provenance_and_skips_raw_mark(self, tmp_path):
        """Typed refs should stay typed and must not mark matching raw IDs processed."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage, strict=False)
        bind_noop_model(k)

        raw_id = storage.save_raw("raw content", source="test")
        suggestion = MemorySuggestion(
            id="sug-typed-episode-ref",
            stack_id="test-agent",
            memory_type="belief",
            content={
                "statement": "Belief from an episode source",
                "belief_type": "fact",
                "confidence": 0.8,
            },
            confidence=0.8,
            source_raw_ids=[f"episode:{raw_id}"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        memory_id = k.promote_suggestion(suggestion.id)
        assert memory_id is not None

        belief = next((b for b in storage.get_beliefs(limit=100) if b.id == memory_id), None)
        assert belief is not None
        assert belief.derived_from == [f"episode:{raw_id}"]

        raw_entry = storage.get_raw(raw_id)
        assert raw_entry is not None
        assert raw_entry.processed is False

        storage.close()


class TestHelperMethods:
    """Test helper extraction methods."""

    def test_extract_first_sentence(self):
        """Should extract first meaningful sentence."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)

        content = "This is the first sentence. Here is another one."
        result = k._extract_first_sentence(content)
        assert result == "This is the first sentence"

        # Should handle newlines
        content2 = "First line\nSecond line"
        result2 = k._extract_first_sentence(content2)
        assert result2 == "First line"

    def test_extract_first_sentence_short_fallback(self):
        """When no long sentence exists, helper should return truncated fallback."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        assert k._extract_first_sentence("short") == "short"

    def test_extract_outcome_none_when_no_pattern(self):
        """Outcome extraction should return None if no pattern matches."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        assert k._extract_outcome("This narrative has no completion marker") is None

    def test_extract_lessons_deduplicates(self):
        """Duplicate lesson matches should be deduplicated."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        content = (
            "Lesson: check assumptions carefully. "
            "takeaway: check assumptions carefully. "
            "insight: validate edge cases first."
        )
        lessons = k._extract_lessons(content)
        assert "check assumptions carefully" in lessons
        assert len(lessons) == 2

    def test_extract_belief_statement_falls_back_to_first_sentence(self):
        """Belief statement extraction should fallback when no opinion phrase exists."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        content = "Fallback sentence from plain statement. Second sentence."
        assert k._extract_belief_statement(content) == "Fallback sentence from plain statement"

    def test_infer_outcome_type(self):
        """Should infer outcome type from content."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)

        assert k._infer_outcome_type("The task was successful") == "success"
        assert k._infer_outcome_type("It failed completely") == "failure"
        assert k._infer_outcome_type("Only partially done") == "partial"
        assert k._infer_outcome_type("Random content") == "unknown"

    def test_infer_belief_type(self):
        """Should infer belief type from content."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)

        assert k._infer_belief_type("You should always test") == "rule"
        assert k._infer_belief_type("I prefer Python") == "preference"
        assert k._infer_belief_type("The limit is 100") == "constraint"
        assert k._infer_belief_type("I learned this today") == "learned"
        assert k._infer_belief_type("The sky is blue") == "fact"

    def test_infer_note_type(self):
        """Should infer note type from content."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)

        assert k._infer_note_type('"Quote" said John') == "quote"
        assert k._infer_note_type("I decided to use Python") == "decision"
        assert k._infer_note_type("Interesting insight about X") == "insight"
        assert k._infer_note_type("Random note content") == "note"

    def test_extract_speaker_from_quote_and_fallback(self):
        """Speaker extraction should parse known forms and return None otherwise."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        assert k._extract_speaker('Alice said "let us simplify this"') == "Alice"
        assert k._extract_speaker('"Just a quote"') is None

    def test_extract_reason_from_decision_and_fallback(self):
        """Reason extraction should parse because-clause and return None if absent."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        assert (
            k._extract_reason("I decided to switch databases because migrations were failing")
            == "migrations were failing"
        )
        assert k._extract_reason("I decided to switch databases") is None

    def test_create_suggestions_return_none_for_short_content(self):
        """Creation helpers should guard against very short content."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        raw = RawEntry(
            id="raw-short",
            stack_id="test-agent",
            content="short",
            timestamp=datetime.now(timezone.utc),
            source="test",
        )
        assert k._create_episode_suggestion(raw, 0.7) is None
        assert k._create_belief_suggestion(raw, 0.7) is None
        assert k._create_note_suggestion(raw, 0.7) is None

    def test_create_note_suggestion_extracts_quote_and_decision_fields(self):
        """Quote and decision notes should populate speaker and reason fields."""
        k = Kernle("test-agent", storage=MagicMock(), strict=False)
        quote_raw = RawEntry(
            id="raw-quote",
            stack_id="test-agent",
            content='Alice said "we should validate inputs first"',
            timestamp=datetime.now(timezone.utc),
            source="test",
        )
        decision_raw = RawEntry(
            id="raw-decision",
            stack_id="test-agent",
            content="I decided to keep SQLite because setup is simple",
            timestamp=datetime.now(timezone.utc),
            source="test",
        )

        quote_suggestion = k._create_note_suggestion(quote_raw, 0.8)
        decision_suggestion = k._create_note_suggestion(decision_raw, 0.8)

        assert quote_suggestion is not None
        assert quote_suggestion.content["note_type"] == "quote"
        assert quote_suggestion.content["speaker"] == "Alice"
        assert decision_suggestion is not None
        assert decision_suggestion.content["note_type"] == "decision"
        assert decision_suggestion.content["reason"] == "setup is simple"


class TestStatsIncludesSuggestions:
    """Test that stats include suggestion counts."""

    def test_stats_include_suggestions(self, tmp_path):
        """Stats should include total and pending suggestion counts."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        # Create some suggestions
        for i in range(3):
            suggestion = MemorySuggestion(
                id=f"sug-stats-{i}",
                stack_id="test-agent",
                memory_type="episode",
                content={},
                confidence=0.7,
                source_raw_ids=[],
                status="pending" if i < 2 else "promoted",
                created_at=datetime.now(timezone.utc),
            )
            storage.save_suggestion(suggestion)

        stats = storage.get_stats()

        assert "suggestions" in stats
        assert stats["suggestions"] == 3
        assert "pending_suggestions" in stats
        assert stats["pending_suggestions"] == 2

        storage.close()
