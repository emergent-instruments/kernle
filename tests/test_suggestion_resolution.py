"""Tests for suggestion resolution workflow and persistence semantics.

Tests the full lifecycle: create -> list/filter -> accept/dismiss/expire -> verify state.
Covers storage, stack, Kernle compat, CLI, and MCP layers.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kernle import Kernle
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import MemorySuggestion
from tests.conftest import bind_noop_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path):
    """Create a fresh SQLiteStorage."""
    db_path = tmp_path / "test.db"
    s = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    return s


@pytest.fixture
def kernle_instance(tmp_path):
    """Create a fresh Kernle instance for compat-layer tests."""
    db_path = tmp_path / "test.db"
    s = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    k = Kernle("test-stack", storage=s, strict=False)
    bind_noop_model(k)
    return k


def _make_suggestion(
    storage,
    memory_type="belief",
    confidence=0.8,
    status="pending",
    content=None,
    created_at=None,
    source_raw_ids=None,
):
    """Helper to create and save a suggestion."""
    sid = str(uuid.uuid4())
    suggestion = MemorySuggestion(
        id=sid,
        stack_id=storage.stack_id,
        memory_type=memory_type,
        content=content or {"statement": "Test belief", "belief_type": "fact", "confidence": 0.8},
        confidence=confidence,
        source_raw_ids=source_raw_ids or [],
        status=status,
        created_at=created_at or datetime.now(timezone.utc),
    )
    storage.save_suggestion(suggestion)
    return sid


def _make_raw(storage, content="Test raw entry"):
    """Helper to create and save a raw entry."""
    return storage.save_raw(content, source="test")


# ---------------------------------------------------------------------------
# Storage layer tests
# ---------------------------------------------------------------------------


class TestStorageGetSuggestions:
    """Test enhanced get_suggestions with new filters."""

    def test_filter_by_status(self, storage):
        _make_suggestion(storage, status="pending")
        _make_suggestion(storage, status="pending")
        _make_suggestion(storage, status="promoted")

        pending = storage.get_suggestions(status="pending")
        assert len(pending) == 2

        promoted = storage.get_suggestions(status="promoted")
        assert len(promoted) == 1

    def test_filter_by_memory_type(self, storage):
        _make_suggestion(storage, memory_type="belief")
        _make_suggestion(
            storage, memory_type="episode", content={"objective": "Test", "outcome": "Done"}
        )
        _make_suggestion(storage, memory_type="belief")

        beliefs = storage.get_suggestions(memory_type="belief")
        assert len(beliefs) == 2

        episodes = storage.get_suggestions(memory_type="episode")
        assert len(episodes) == 1

    def test_filter_by_min_confidence(self, storage):
        _make_suggestion(storage, confidence=0.3)
        _make_suggestion(storage, confidence=0.6)
        _make_suggestion(storage, confidence=0.9)

        high = storage.get_suggestions(min_confidence=0.5)
        assert len(high) == 2

        very_high = storage.get_suggestions(min_confidence=0.8)
        assert len(very_high) == 1

    def test_filter_by_max_age_hours(self, storage):
        # Recent suggestion
        _make_suggestion(storage, created_at=datetime.now(timezone.utc))
        # Old suggestion
        _make_suggestion(
            storage,
            created_at=datetime.now(timezone.utc) - timedelta(hours=48),
        )

        recent = storage.get_suggestions(max_age_hours=24)
        assert len(recent) == 1

        all_suggestions = storage.get_suggestions(max_age_hours=72)
        assert len(all_suggestions) == 2

    def test_filter_by_source_raw_id(self, storage):
        raw_id = str(uuid.uuid4())
        _make_suggestion(storage, source_raw_ids=[raw_id])
        _make_suggestion(storage, source_raw_ids=["other-id"])

        filtered = storage.get_suggestions(source_raw_id=raw_id)
        assert len(filtered) == 1
        assert filtered[0].source_raw_ids == [raw_id]

    def test_combined_filters(self, storage):
        _make_suggestion(storage, confidence=0.9, memory_type="belief", status="pending")
        _make_suggestion(storage, confidence=0.3, memory_type="belief", status="pending")
        _make_suggestion(
            storage,
            confidence=0.9,
            memory_type="episode",
            content={"objective": "Test", "outcome": "Done"},
            status="pending",
        )

        results = storage.get_suggestions(
            status="pending", memory_type="belief", min_confidence=0.5
        )
        assert len(results) == 1
        assert results[0].confidence == 0.9


class TestStorageExpireSuggestions:
    """Test expire_suggestions storage method."""

    def test_expire_old_pending(self, storage):
        # Old pending suggestion
        old_id = _make_suggestion(
            storage,
            created_at=datetime.now(timezone.utc) - timedelta(hours=200),
        )
        # Recent pending suggestion
        recent_id = _make_suggestion(storage)

        expired = storage.expire_suggestions(max_age_hours=168)
        assert len(expired) == 1
        assert old_id in expired

        # Verify status changed
        old = storage.get_suggestion(old_id)
        assert old.status == "expired"
        assert old.resolved_at is not None
        assert "auto-expired" in old.resolution_reason

        # Recent should still be pending
        recent = storage.get_suggestion(recent_id)
        assert recent.status == "pending"

    def test_expire_only_pending(self, storage):
        """Already-resolved suggestions should not be expired."""
        old_promoted_id = _make_suggestion(
            storage,
            status="promoted",
            created_at=datetime.now(timezone.utc) - timedelta(hours=200),
        )

        expired = storage.expire_suggestions(max_age_hours=168)
        assert len(expired) == 0

        # Status should remain promoted
        s = storage.get_suggestion(old_promoted_id)
        assert s.status == "promoted"

    def test_expire_custom_threshold(self, storage):
        s1 = _make_suggestion(
            storage,
            created_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        _make_suggestion(
            storage,
            created_at=datetime.now(timezone.utc) - timedelta(hours=5),
        )

        expired = storage.expire_suggestions(max_age_hours=24)
        assert len(expired) == 1
        assert s1 in expired

    def test_expire_returns_empty_when_no_match(self, storage):
        _make_suggestion(storage)
        expired = storage.expire_suggestions(max_age_hours=168)
        assert expired == []


# ---------------------------------------------------------------------------
# Stack layer tests
# ---------------------------------------------------------------------------


class TestStackAcceptSuggestion:
    """Test accept_suggestion on Stack."""

    def test_accept_belief_suggestion(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        raw_id = _make_raw(stack._backend)
        sid = _make_suggestion(
            stack._backend,
            memory_type="belief",
            content={
                "statement": "Testing is important",
                "belief_type": "learned",
                "confidence": 0.85,
            },
            source_raw_ids=[raw_id],
        )

        memory_id = stack.accept_suggestion(sid)
        assert memory_id is not None

        # Verify suggestion status updated
        suggestion = stack.get_suggestion(sid)
        assert suggestion.status == "promoted"
        assert suggestion.promoted_to == f"belief:{memory_id}"

        # Verify belief was created
        beliefs = stack._backend.get_beliefs(limit=10)
        assert len(beliefs) == 1
        assert beliefs[0].statement == "Testing is important"
        assert beliefs[0].source_type == "processing"
        assert beliefs[0].derived_from == [f"raw:{raw_id}"]

        # Verify audit log
        audit = stack.get_audit_log(
            memory_type="suggestion", memory_id=sid, operation="suggestion.resolved"
        )
        assert len(audit) == 1

    def test_accept_episode_suggestion(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        sid = _make_suggestion(
            stack._backend,
            memory_type="episode",
            content={
                "objective": "Deploy v2",
                "outcome": "Shipped successfully",
                "outcome_type": "success",
            },
        )

        memory_id = stack.accept_suggestion(sid)
        assert memory_id is not None

        episodes = stack._backend.get_episodes(limit=10)
        assert len(episodes) == 1
        assert episodes[0].objective == "Deploy v2"

    def test_accept_note_suggestion(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        sid = _make_suggestion(
            stack._backend,
            memory_type="note",
            content={"content": "Important observation", "note_type": "insight"},
        )

        memory_id = stack.accept_suggestion(sid)
        assert memory_id is not None

        notes = stack._backend.get_notes(limit=10)
        assert len(notes) == 1
        assert notes[0].content == "Important observation"

    def test_accept_preserves_typed_episode_provenance_and_skips_raw_mark(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        raw_id = _make_raw(stack._backend)
        sid = _make_suggestion(
            stack._backend,
            memory_type="belief",
            content={
                "statement": "Typed provenance should survive promotion",
                "belief_type": "factual",
                "confidence": 0.8,
            },
            source_raw_ids=[f"episode:{raw_id}"],
        )

        memory_id = stack.accept_suggestion(sid)
        assert memory_id is not None

        beliefs = stack._backend.get_beliefs(limit=10)
        assert len(beliefs) == 1
        assert beliefs[0].derived_from == [f"episode:{raw_id}"]

        raw_entry = stack._backend.get_raw(raw_id)
        assert raw_entry is not None
        assert raw_entry.processed is False

    def test_accept_with_modifications(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        sid = _make_suggestion(
            stack._backend,
            memory_type="belief",
            content={"statement": "Original statement", "belief_type": "fact", "confidence": 0.7},
        )

        memory_id = stack.accept_suggestion(sid, modifications={"statement": "Refined statement"})
        assert memory_id is not None

        suggestion = stack.get_suggestion(sid)
        assert suggestion.status == "modified"

        beliefs = stack._backend.get_beliefs(limit=10)
        assert beliefs[0].statement == "Refined statement"

    def test_accept_nonexistent_returns_none(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        result = stack.accept_suggestion("nonexistent-id")
        assert result is None

    def test_accept_already_resolved_returns_none(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        sid = _make_suggestion(stack._backend, status="promoted")
        result = stack.accept_suggestion(sid)
        assert result is None


class TestStackAcceptSuggestionWithProvenance:
    """Test accept_suggestion with enforce_provenance=True (P0 fix)."""

    def test_accept_belief_with_provenance_enforced(self, tmp_path):
        """Accepting a belief suggestion must not raise ProvenanceError."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )

        raw_id = _make_raw(stack._backend)
        sid = _make_suggestion(
            stack._backend,
            memory_type="belief",
            content={
                "statement": "Code reviews improve quality",
                "belief_type": "learned",
                "confidence": 0.85,
            },
            source_raw_ids=[raw_id],
        )

        # This used to raise ProvenanceError
        memory_id = stack.accept_suggestion(sid)
        assert memory_id is not None

        # Verify provenance fields
        beliefs = stack._backend.get_beliefs(limit=10)
        assert len(beliefs) == 1
        assert beliefs[0].source_type == "processing"
        assert beliefs[0].derived_from == [f"raw:{raw_id}"]
        assert beliefs[0].source_entity == "kernle:suggestion-promotion"

    def test_accept_episode_with_provenance_enforced(self, tmp_path):
        """Accepting an episode suggestion must not raise ProvenanceError."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )

        raw_id = _make_raw(stack._backend)
        sid = _make_suggestion(
            stack._backend,
            memory_type="episode",
            content={
                "objective": "Deploy v2",
                "outcome": "Shipped successfully",
                "outcome_type": "success",
            },
            source_raw_ids=[raw_id],
        )

        memory_id = stack.accept_suggestion(sid)
        assert memory_id is not None

        episodes = stack._backend.get_episodes(limit=10)
        assert len(episodes) == 1
        assert episodes[0].source_type == "processing"
        assert episodes[0].derived_from == [f"raw:{raw_id}"]
        assert episodes[0].source_entity == "kernle:suggestion-promotion"

    def test_accept_note_with_provenance_enforced(self, tmp_path):
        """Accepting a note suggestion must not raise ProvenanceError."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )

        raw_id = _make_raw(stack._backend)
        sid = _make_suggestion(
            stack._backend,
            memory_type="note",
            content={"content": "Important observation", "note_type": "insight"},
            source_raw_ids=[raw_id],
        )

        memory_id = stack.accept_suggestion(sid)
        assert memory_id is not None

        notes = stack._backend.get_notes(limit=10)
        assert len(notes) == 1
        assert notes[0].source_type == "processing"
        assert notes[0].derived_from == [f"raw:{raw_id}"]
        assert notes[0].source_entity == "kernle:suggestion-promotion"

    def test_accept_with_modifications_provenance_enforced(self, tmp_path):
        """Accepting with modifications must not raise ProvenanceError."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )

        raw_id = _make_raw(stack._backend)
        sid = _make_suggestion(
            stack._backend,
            memory_type="belief",
            content={"statement": "Original", "belief_type": "fact", "confidence": 0.7},
            source_raw_ids=[raw_id],
        )

        memory_id = stack.accept_suggestion(sid, modifications={"statement": "Refined statement"})
        assert memory_id is not None

        suggestion = stack.get_suggestion(sid)
        assert suggestion.status == "modified"

        beliefs = stack._backend.get_beliefs(limit=10)
        assert beliefs[0].statement == "Refined statement"
        assert beliefs[0].source_type == "processing"

    def test_accept_multiple_source_raws_provenance_enforced(self, tmp_path):
        """Suggestion with multiple source_raw_ids should all get raw: prefix."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )

        raw_id1 = _make_raw(stack._backend, content="First observation")
        raw_id2 = _make_raw(stack._backend, content="Second observation")
        sid = _make_suggestion(
            stack._backend,
            memory_type="episode",
            content={
                "objective": "Multi-source task",
                "outcome": "Combined insights",
                "outcome_type": "success",
            },
            source_raw_ids=[raw_id1, raw_id2],
        )

        memory_id = stack.accept_suggestion(sid)
        assert memory_id is not None

        episodes = stack._backend.get_episodes(limit=10)
        assert episodes[0].derived_from == [f"raw:{raw_id1}", f"raw:{raw_id2}"]


class TestKernleAcceptWithStrictMode:
    """Test accept_suggestion on Kernle compat layer with strict=True (P0 fix)."""

    def test_accept_belief_strict_mode(self, tmp_path):
        """Kernle(strict=True).accept_suggestion() must not raise ProvenanceError."""
        db_path = tmp_path / "test.db"
        s = SQLiteStorage(stack_id="test-stack", db_path=db_path)
        k = Kernle("test-stack", storage=s, strict=True)

        raw_id = _make_raw(s)
        sid = _make_suggestion(
            s,
            memory_type="belief",
            content={
                "statement": "Strict mode works",
                "belief_type": "fact",
                "confidence": 0.9,
            },
            source_raw_ids=[raw_id],
        )

        memory_id = k.accept_suggestion(sid)
        assert memory_id is not None

    def test_accept_episode_strict_mode(self, tmp_path):
        """Kernle(strict=True).accept_suggestion() for episodes."""
        db_path = tmp_path / "test.db"
        s = SQLiteStorage(stack_id="test-stack", db_path=db_path)
        k = Kernle("test-stack", storage=s, strict=True)

        raw_id = _make_raw(s)
        sid = _make_suggestion(
            s,
            memory_type="episode",
            content={
                "objective": "Test strict episode",
                "outcome": "Success in strict mode",
                "outcome_type": "success",
            },
            source_raw_ids=[raw_id],
        )

        memory_id = k.accept_suggestion(sid)
        assert memory_id is not None

    def test_accept_note_strict_mode(self, tmp_path):
        """Kernle(strict=True).accept_suggestion() for notes."""
        db_path = tmp_path / "test.db"
        s = SQLiteStorage(stack_id="test-stack", db_path=db_path)
        k = Kernle("test-stack", storage=s, strict=True)

        raw_id = _make_raw(s)
        sid = _make_suggestion(
            s,
            memory_type="note",
            content={"content": "Strict note test", "note_type": "insight"},
            source_raw_ids=[raw_id],
        )

        memory_id = k.accept_suggestion(sid)
        assert memory_id is not None


class TestStackDismissSuggestion:
    """Test dismiss_suggestion on Stack."""

    def test_dismiss_pending(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        sid = _make_suggestion(stack._backend)
        result = stack.dismiss_suggestion(sid, reason="Not relevant")
        assert result is True

        suggestion = stack.get_suggestion(sid)
        assert suggestion.status == "dismissed"
        assert suggestion.resolution_reason == "Not relevant"

        # Verify audit log
        audit = stack.get_audit_log(
            memory_type="suggestion", memory_id=sid, operation="suggestion.resolved"
        )
        assert len(audit) == 1

    def test_dismiss_already_resolved_returns_false(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        sid = _make_suggestion(stack._backend, status="promoted")
        result = stack.dismiss_suggestion(sid)
        assert result is False


class TestStackExpireSuggestions:
    """Test expire_suggestions on Stack."""

    def test_expire_logs_audit(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )

        sid = _make_suggestion(
            stack._backend,
            created_at=datetime.now(timezone.utc) - timedelta(hours=200),
        )

        expired = stack.expire_suggestions(max_age_hours=168)
        assert len(expired) == 1

        audit = stack.get_audit_log(
            memory_type="suggestion", memory_id=sid, operation="suggestion.resolved"
        )
        assert len(audit) == 1


# ---------------------------------------------------------------------------
# Kernle compat layer tests
# ---------------------------------------------------------------------------


class TestKernleAcceptDismissExpire:
    """Test accept/dismiss/expire on Kernle compat layer."""

    def test_accept_suggestion(self, kernle_instance):
        k = kernle_instance
        sid = _make_suggestion(k._storage)
        memory_id = k.accept_suggestion(sid)
        assert memory_id is not None

    def test_dismiss_suggestion(self, kernle_instance):
        k = kernle_instance
        sid = _make_suggestion(k._storage)
        result = k.dismiss_suggestion(sid, reason="Duplicate")
        assert result is True

        suggestion = k.get_suggestion(sid)
        assert suggestion["status"] == "dismissed"

    def test_expire_suggestions(self, kernle_instance):
        k = kernle_instance
        _make_suggestion(
            k._storage,
            created_at=datetime.now(timezone.utc) - timedelta(hours=200),
        )
        expired = k.expire_suggestions(max_age_hours=168)
        assert len(expired) == 1

    def test_get_suggestions_with_filters(self, kernle_instance):
        k = kernle_instance
        _make_suggestion(k._storage, confidence=0.9)
        _make_suggestion(k._storage, confidence=0.3)

        high = k.get_suggestions(min_confidence=0.5)
        assert len(high) == 1
        assert high[0]["confidence"] == 0.9


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLISuggestionCommands:
    """Test CLI accept/dismiss/expire commands."""

    def test_cli_accept(self, kernle_instance, capsys):
        from kernle.cli.commands.suggestions import cmd_suggestions

        k = kernle_instance
        sid = _make_suggestion(k._storage)

        args = SimpleNamespace(
            suggestions_action="accept",
            id=sid[:8],
            objective=None,
            outcome=None,
            statement=None,
            content=None,
        )
        cmd_suggestions(args, k)
        output = capsys.readouterr().out
        assert "accepted" in output.lower()

    def test_cli_dismiss(self, kernle_instance, capsys):
        from kernle.cli.commands.suggestions import cmd_suggestions

        k = kernle_instance
        sid = _make_suggestion(k._storage)

        args = SimpleNamespace(
            suggestions_action="dismiss",
            id=sid[:8],
            reason="Not useful",
        )
        cmd_suggestions(args, k)
        output = capsys.readouterr().out
        assert "dismissed" in output.lower()

    def test_cli_expire(self, kernle_instance, capsys):
        from kernle.cli.commands.suggestions import cmd_suggestions

        k = kernle_instance
        _make_suggestion(
            k._storage,
            created_at=datetime.now(timezone.utc) - timedelta(hours=200),
        )

        args = SimpleNamespace(
            suggestions_action="expire",
            max_age_hours=168.0,
        )
        cmd_suggestions(args, k)
        output = capsys.readouterr().out
        assert "expired" in output.lower() or "Expired" in output

    def test_cli_list_dismissed(self, kernle_instance, capsys):
        from kernle.cli.commands.suggestions import cmd_suggestions

        k = kernle_instance
        sid = _make_suggestion(k._storage)
        k.dismiss_suggestion(sid, reason="test")

        args = SimpleNamespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            dismissed=True,
            expired=False,
            type=None,
            min_confidence=None,
            max_age_hours=None,
            source=None,
            limit=50,
            json=False,
        )
        cmd_suggestions(args, k)
        output = capsys.readouterr().out
        assert "dismissed" in output.lower()

    def test_cli_list_with_min_confidence(self, kernle_instance, capsys):
        from kernle.cli.commands.suggestions import cmd_suggestions

        k = kernle_instance
        _make_suggestion(k._storage, confidence=0.9)
        _make_suggestion(k._storage, confidence=0.3)

        args = SimpleNamespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            dismissed=False,
            expired=False,
            type=None,
            min_confidence=0.5,
            max_age_hours=None,
            source=None,
            limit=50,
            json=False,
        )
        cmd_suggestions(args, k)
        output = capsys.readouterr().out
        # Should only show the high-confidence one
        assert "90%" in output


# ---------------------------------------------------------------------------
# MCP handler tests
# ---------------------------------------------------------------------------


class TestMCPSuggestionTools:
    """Test new MCP suggestion_list/accept/dismiss handlers."""

    def test_suggestion_list_handler(self, kernle_instance):
        from kernle.mcp.handlers.sync import handle_suggestion_list

        k = kernle_instance
        _make_suggestion(k._storage, confidence=0.9)
        _make_suggestion(k._storage, confidence=0.3)

        result = handle_suggestion_list({"status": "pending", "limit": 20}, k)
        data = json.loads(result)
        assert len(data) == 2

    def test_suggestion_list_with_filters(self, kernle_instance):
        from kernle.mcp.handlers.sync import handle_suggestion_list

        k = kernle_instance
        _make_suggestion(k._storage, confidence=0.9)
        _make_suggestion(k._storage, confidence=0.3)

        result = handle_suggestion_list(
            {"status": "pending", "min_confidence": 0.5, "limit": 20}, k
        )
        data = json.loads(result)
        assert len(data) == 1

    def test_suggestion_accept_handler(self, kernle_instance):
        from kernle.mcp.handlers.sync import handle_suggestion_accept

        k = kernle_instance
        sid = _make_suggestion(k._storage)

        result = handle_suggestion_accept({"suggestion_id": sid}, k)
        assert "accepted" in result

    def test_suggestion_accept_with_modifications(self, kernle_instance):
        from kernle.mcp.handlers.sync import handle_suggestion_accept

        k = kernle_instance
        sid = _make_suggestion(k._storage)

        result = handle_suggestion_accept({"suggestion_id": sid, "statement": "Modified belief"}, k)
        assert "modified" in result

    def test_suggestion_accept_nonexistent(self, kernle_instance):
        from kernle.mcp.handlers.sync import handle_suggestion_accept

        k = kernle_instance
        result = handle_suggestion_accept({"suggestion_id": "nonexistent"}, k)
        assert "Could not" in result

    def test_suggestion_dismiss_handler(self, kernle_instance):
        from kernle.mcp.handlers.sync import handle_suggestion_dismiss

        k = kernle_instance
        sid = _make_suggestion(k._storage)

        result = handle_suggestion_dismiss({"suggestion_id": sid, "reason": "Not relevant"}, k)
        assert "dismissed" in result
        assert "Not relevant" in result

    def test_suggestion_dismiss_nonexistent(self, kernle_instance):
        from kernle.mcp.handlers.sync import handle_suggestion_dismiss

        k = kernle_instance
        result = handle_suggestion_dismiss({"suggestion_id": "nonexistent"}, k)
        assert "Could not" in result

    def test_validator_suggestion_list(self):
        from kernle.mcp.handlers.sync import validate_suggestion_list

        result = validate_suggestion_list({"status": "pending", "min_confidence": 0.5})
        assert result["status"] == "pending"
        assert result["min_confidence"] == 0.5

    def test_validator_suggestion_accept(self):
        from kernle.mcp.handlers.sync import validate_suggestion_accept

        result = validate_suggestion_accept({"suggestion_id": "test-id", "statement": "Override"})
        assert result["suggestion_id"] == "test-id"
        assert result["statement"] == "Override"

    def test_validator_suggestion_dismiss(self):
        from kernle.mcp.handlers.sync import validate_suggestion_dismiss

        result = validate_suggestion_dismiss({"suggestion_id": "test-id", "reason": "Bad"})
        assert result["suggestion_id"] == "test-id"
        assert result["reason"] == "Bad"


# ---------------------------------------------------------------------------
# Export-full tests
# ---------------------------------------------------------------------------


class TestExportFullSuggestions:
    """Test suggestion history in export-full."""

    def test_export_full_markdown_includes_suggestions(self, kernle_instance):
        k = kernle_instance
        sid = _make_suggestion(k._storage, confidence=0.9)
        k.dismiss_suggestion(sid, reason="test dismissal")
        _make_suggestion(k._storage, confidence=0.7)  # pending

        content = k.export_full(format="markdown")
        assert "## Suggestions" in content
        assert "Pending" in content
        assert "Resolved" in content

    def test_export_full_json_includes_suggestions(self, kernle_instance):
        k = kernle_instance
        sid = _make_suggestion(k._storage)
        k.accept_suggestion(sid)

        content = k.export_full(format="json")
        data = json.loads(content)
        assert "suggestions" in data
        assert len(data["suggestions"]) >= 1
        # Should have the promoted suggestion
        promoted = [s for s in data["suggestions"] if s["status"] in ("promoted", "modified")]
        assert len(promoted) == 1


# ---------------------------------------------------------------------------
# Full lifecycle test
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end lifecycle: create -> list -> accept/dismiss -> verify state."""

    def test_full_accept_lifecycle(self, kernle_instance):
        """Create suggestion, list it, accept it, verify memory created."""
        k = kernle_instance

        # Create a raw entry and suggestion
        raw_id = _make_raw(k._storage)
        sid = _make_suggestion(
            k._storage,
            memory_type="belief",
            content={
                "statement": "TDD leads to better design",
                "belief_type": "learned",
                "confidence": 0.85,
            },
            source_raw_ids=[raw_id],
        )

        # List pending
        pending = k.get_suggestions(status="pending")
        assert len(pending) == 1
        assert pending[0]["id"] == sid

        # Accept
        memory_id = k.accept_suggestion(sid)
        assert memory_id is not None

        # Verify suggestion resolved
        suggestion = k.get_suggestion(sid)
        assert suggestion["status"] in ("promoted", "modified")
        assert suggestion["promoted_to"] == f"belief:{memory_id}"

        # Verify no more pending
        pending = k.get_suggestions(status="pending")
        assert len(pending) == 0

    def test_full_dismiss_lifecycle(self, kernle_instance):
        """Create suggestion, dismiss it, verify state."""
        k = kernle_instance
        sid = _make_suggestion(k._storage)

        # Dismiss
        result = k.dismiss_suggestion(sid, reason="Low quality")
        assert result is True

        # Verify
        suggestion = k.get_suggestion(sid)
        assert suggestion["status"] == "dismissed"
        assert suggestion["resolution_reason"] == "Low quality"

    def test_full_expire_lifecycle(self, kernle_instance):
        """Create old suggestions, expire them, verify state."""
        k = kernle_instance

        old_sid = _make_suggestion(
            k._storage,
            created_at=datetime.now(timezone.utc) - timedelta(hours=200),
        )
        recent_sid = _make_suggestion(k._storage)

        # Expire
        expired = k.expire_suggestions(max_age_hours=168)
        assert len(expired) == 1
        assert old_sid in expired

        # Verify old expired
        old = k.get_suggestion(old_sid)
        assert old["status"] == "expired"

        # Verify recent still pending
        recent = k.get_suggestion(recent_sid)
        assert recent["status"] == "pending"

    def test_mixed_resolution_lifecycle(self, kernle_instance):
        """Create multiple suggestions, resolve them differently, verify list filters."""
        k = kernle_instance

        s1 = _make_suggestion(k._storage, confidence=0.9)
        s2 = _make_suggestion(k._storage, confidence=0.5)
        _make_suggestion(
            k._storage,
            confidence=0.3,
            created_at=datetime.now(timezone.utc) - timedelta(hours=200),
        )

        # Accept s1
        k.accept_suggestion(s1)
        # Dismiss s2
        k.dismiss_suggestion(s2, reason="duplicate")
        # Expire s3
        k.expire_suggestions(max_age_hours=168)

        # All should have non-pending statuses
        pending = k.get_suggestions(status="pending")
        assert len(pending) == 0

        # Filter by status
        dismissed = k.get_suggestions(status="dismissed")
        assert len(dismissed) == 1

        expired = k.get_suggestions(status="expired")
        assert len(expired) == 1

        promoted = k.get_suggestions(status="promoted")
        assert len(promoted) == 1
