"""Tests for suggestion type resolution — all produced types must be accepted.

Verifies that accept_suggestion() handles every memory type that the
processing pipeline can produce, and that a shared constant ensures
producer and resolver can't drift apart.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle import Kernle
from kernle.stack import SQLiteStack
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import (
    SUGGESTION_MEMORY_TYPES,
    MemorySuggestion,
)
from tests.conftest import bind_noop_model


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def stack(tmp_path):
    stack_id = f"type-resolution-{uuid.uuid4().hex[:8]}"
    db_path = tmp_path / "type_resolution.db"
    return SQLiteStack(stack_id=stack_id, db_path=db_path, components=[], enforce_provenance=False)


@pytest.fixture
def strict_stack(tmp_path):
    stack_id = f"type-resolution-strict-{uuid.uuid4().hex[:8]}"
    db_path = tmp_path / "type_resolution_strict.db"
    return SQLiteStack(stack_id=stack_id, db_path=db_path, components=[], enforce_provenance=True)


def _make_suggestion(memory_type: str, content: dict, stack_id: str) -> MemorySuggestion:
    return MemorySuggestion(
        id=_uid(),
        stack_id=stack_id,
        memory_type=memory_type,
        content=content,
        confidence=0.8,
        source_raw_ids=["raw-1"],
        created_at=_now(),
    )


class TestSuggestionTypeResolution:
    """Verify that accept_suggestion handles all types the producer emits."""

    def test_accept_goal_suggestion_creates_goal(self, stack):
        s = _make_suggestion(
            "goal",
            {
                "title": "Ship v2.0",
                "description": "Release the next version",
                "goal_type": "task",
                "priority": "high",
                "status": "active",
            },
            stack.stack_id,
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        goals = stack.get_goals(limit=100)
        assert any(g.id == memory_id for g in goals)

    def test_accept_value_suggestion_creates_value(self, stack):
        s = _make_suggestion(
            "value",
            {
                "name": "Reliability",
                "statement": "Systems should be dependable",
                "priority": 75,
            },
            stack.stack_id,
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        values = stack.get_values(limit=100)
        assert any(v.id == memory_id for v in values)

    def test_accept_relationship_suggestion_creates_relationship(self, stack):
        s = _make_suggestion(
            "relationship",
            {
                "entity_name": "alice",
                "entity_type": "human",
                "relationship_type": "collaborator",
                "notes": "Good partner",
            },
            stack.stack_id,
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        rels = stack.get_relationships()
        assert any(r.id == memory_id for r in rels)

    def test_accept_drive_suggestion_creates_drive(self, stack):
        s = _make_suggestion(
            "drive",
            {
                "drive_type": "curiosity",
                "intensity": 0.7,
                "focus_areas": ["learning"],
            },
            stack.stack_id,
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        drives = stack.get_drives()
        assert any(d.id == memory_id for d in drives)

    def test_accept_unknown_type_raises_value_error(self, stack):
        s = _make_suggestion("widget", {"foo": "bar"}, stack.stack_id)
        stack.save_suggestion(s)
        with pytest.raises(ValueError, match="[Uu]nsupported.*widget"):
            stack.accept_suggestion(s.id)

    def test_produced_types_exactly_match_resolved_types(self):
        """Hard assertion: the shared constant covers all 7 types."""
        expected = {"episode", "belief", "note", "goal", "relationship", "value", "drive"}
        assert SUGGESTION_MEMORY_TYPES == expected


class TestSuggestionTypeResolutionStrictMode:
    """Verify accept_suggestion works with enforce_provenance=True.

    The source_entity='kernle:suggestion-promotion' bypass must be set
    on all type handlers, otherwise strict mode raises ProvenanceError.
    """

    def test_accept_goal_strict_mode(self, strict_stack):
        s = _make_suggestion(
            "goal",
            {
                "title": "Ship v2.0",
                "description": "Release the next version",
                "goal_type": "task",
                "priority": "high",
                "status": "active",
            },
            strict_stack.stack_id,
        )
        strict_stack.save_suggestion(s)
        memory_id = strict_stack.accept_suggestion(s.id)
        assert memory_id is not None

    def test_accept_value_strict_mode(self, strict_stack):
        s = _make_suggestion(
            "value",
            {
                "name": "Reliability",
                "statement": "Systems should be dependable",
                "priority": 75,
            },
            strict_stack.stack_id,
        )
        strict_stack.save_suggestion(s)
        memory_id = strict_stack.accept_suggestion(s.id)
        assert memory_id is not None

    def test_accept_relationship_strict_mode(self, strict_stack):
        s = _make_suggestion(
            "relationship",
            {
                "entity_name": "alice",
                "entity_type": "human",
                "relationship_type": "collaborator",
                "notes": "Good partner",
            },
            strict_stack.stack_id,
        )
        strict_stack.save_suggestion(s)
        memory_id = strict_stack.accept_suggestion(s.id)
        assert memory_id is not None

    def test_accept_drive_strict_mode(self, strict_stack):
        s = _make_suggestion(
            "drive",
            {
                "drive_type": "curiosity",
                "intensity": 0.7,
                "focus_areas": ["learning"],
            },
            strict_stack.stack_id,
        )
        strict_stack.save_suggestion(s)
        memory_id = strict_stack.accept_suggestion(s.id)
        assert memory_id is not None


class TestSuggestionsMixinTypeResolution:
    """Verify promote_suggestion (non-strict Kernle compat) handles all 7 types.

    The SuggestionsMixin.promote_suggestion() is the legacy path used when
    strict=False. It must handle the same types as SQLiteStack.accept_suggestion().
    """

    @pytest.fixture
    def kernle_instance(self, tmp_path):
        db_path = tmp_path / "mixin_test.db"
        storage = SQLiteStorage("mixin-test", db_path=db_path)
        k = Kernle("mixin-test", storage=storage, strict=False)
        bind_noop_model(k)
        return k

    def _save_and_promote(self, k, memory_type, content):
        """Helper: save a suggestion and promote it via the Kernle compat layer."""
        s = MemorySuggestion(
            id=_uid(),
            stack_id=k.stack_id,
            memory_type=memory_type,
            content=content,
            confidence=0.8,
            source_raw_ids=["raw-1"],
            created_at=_now(),
        )
        k._storage.save_suggestion(s)
        return k.accept_suggestion(s.id)

    def test_promote_goal_creates_goal(self, kernle_instance):
        memory_id = self._save_and_promote(
            kernle_instance,
            "goal",
            {
                "title": "Ship v2.0",
                "description": "Release the next version",
                "goal_type": "task",
                "priority": "high",
            },
        )
        assert memory_id is not None
        goals = kernle_instance._storage.get_goals()
        assert any(g.id == memory_id for g in goals)

    def test_promote_value_creates_value(self, kernle_instance):
        memory_id = self._save_and_promote(
            kernle_instance,
            "value",
            {
                "name": "Reliability",
                "statement": "Systems should be dependable",
                "priority": 75,
            },
        )
        assert memory_id is not None
        values = kernle_instance._storage.get_values()
        assert any(v.id == memory_id for v in values)

    def test_promote_relationship_creates_relationship(self, kernle_instance):
        memory_id = self._save_and_promote(
            kernle_instance,
            "relationship",
            {
                "entity_name": "alice",
                "entity_type": "human",
                "notes": "Good partner",
            },
        )
        assert memory_id is not None
        rels = kernle_instance._storage.get_relationships()
        assert any(r.id == memory_id for r in rels)

    def test_promote_relationship_preserves_type(self, kernle_instance):
        """relationship_type from suggestion must map to interaction_type."""
        memory_id = self._save_and_promote(
            kernle_instance,
            "relationship",
            {
                "entity_name": "bob",
                "entity_type": "human",
                "relationship_type": "collaborator",
                "notes": "Works on kernle",
            },
        )
        assert memory_id is not None
        rels = kernle_instance._storage.get_relationships()
        rel = next(r for r in rels if r.id == memory_id)
        assert rel.relationship_type == "collaborator"

    def test_promote_drive_creates_drive(self, kernle_instance):
        memory_id = self._save_and_promote(
            kernle_instance,
            "drive",
            {
                "drive_type": "curiosity",
                "intensity": 0.7,
                "focus_areas": ["learning"],
            },
        )
        assert memory_id is not None
        drives = kernle_instance._storage.get_drives()
        assert any(d.id == memory_id for d in drives)

    def test_promote_drive_missing_type_does_not_raise(self, kernle_instance):
        """drive_type missing from content should not raise ValueError."""
        memory_id = self._save_and_promote(
            kernle_instance,
            "drive",
            {
                "intensity": 0.6,
            },
        )
        assert memory_id is not None
