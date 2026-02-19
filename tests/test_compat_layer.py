"""Tests for the Kernle compatibility layer (v0.4.0).

Verifies that the Kernle class exposes Entity and Stack
via lazy properties, while preserving full backward compatibility
with the existing _storage-based code paths.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import Kernle
from kernle.entity import Entity
from kernle.stack.sqlite_stack import Stack
from kernle.storage import Note, SQLiteStorage


@pytest.fixture
def kernle_sqlite(tmp_path):
    """Kernle instance backed by SQLite for compat-layer tests."""
    db_path = tmp_path / "compat_test.db"
    storage = SQLiteStorage(stack_id="compat_agent", db_path=db_path)
    k = Kernle(
        stack_id="compat_agent", storage=storage, checkpoint_dir=tmp_path / "cp", strict=False
    )
    yield k
    storage.close()


class TestEntityProperty:
    """Tests for Kernle.entity lazy property."""

    def test_entity_returns_entity_instance(self, kernle_sqlite):
        e = kernle_sqlite.entity
        assert isinstance(e, Entity)

    def test_entity_core_id_matches_stack_id(self, kernle_sqlite):
        e = kernle_sqlite.entity
        assert e.core_id == kernle_sqlite.stack_id

    def test_entity_is_lazily_created(self, kernle_sqlite):
        assert not hasattr(kernle_sqlite, "_entity")
        _ = kernle_sqlite.entity
        assert hasattr(kernle_sqlite, "_entity")

    def test_entity_is_cached(self, kernle_sqlite):
        e1 = kernle_sqlite.entity
        e2 = kernle_sqlite.entity
        assert e1 is e2


class TestStackProperty:
    """Tests for Kernle.stack lazy property."""

    def test_stack_returns_sqlite_stack(self, kernle_sqlite):
        s = kernle_sqlite.stack
        assert isinstance(s, Stack)

    def test_stack_uses_same_db_path(self, kernle_sqlite):
        s = kernle_sqlite.stack
        assert s._backend.db_path == kernle_sqlite._storage.db_path

    def test_stack_is_eagerly_created(self, kernle_sqlite):
        """Stack is created eagerly during __init__ (Phase 3)."""
        assert hasattr(kernle_sqlite, "_stack")
        assert kernle_sqlite._stack is not None

    def test_stack_is_cached(self, kernle_sqlite):
        s1 = kernle_sqlite.stack
        s2 = kernle_sqlite.stack
        assert s1 is s2

    def test_stack_works_with_non_sqlite_storage(self, tmp_path):
        """Stack is created eagerly with any storage backend (Phase 3)."""
        from unittest.mock import MagicMock

        mock_storage = MagicMock()
        mock_storage.stack_id = "mock_agent"
        mock_storage.is_online.return_value = False
        mock_storage.get_pending_sync_count.return_value = 0
        mock_storage.get_trust_assessment.return_value = None
        mock_storage.get_all_stack_settings.return_value = {}

        k = Kernle(
            stack_id="mock_agent",
            storage=mock_storage,
            checkpoint_dir=tmp_path / "cp",
            strict=False,
        )
        assert k.stack is not None
        assert k._storage is k.stack._backend


class TestEntityStackIntegration:
    """Tests for entity + stack working together."""

    def test_entity_then_stack_attaches_automatically(self, kernle_sqlite):
        """Accessing .entity first, then .stack, auto-attaches the stack."""
        e = kernle_sqlite.entity
        s = kernle_sqlite.stack
        assert e.active_stack is s

    def test_entity_always_gets_stack_attached(self, kernle_sqlite):
        """Entity always auto-attaches the eager Stack (Phase 3)."""
        e = kernle_sqlite.entity
        assert e.active_stack is kernle_sqlite.stack

    def test_entity_shares_stack_with_kernle(self, kernle_sqlite):
        """Entity's active stack is the same Stack Kernle created eagerly."""
        e = kernle_sqlite.entity
        assert e.active_stack is kernle_sqlite.stack
        assert e.active_stack._backend is kernle_sqlite._storage

    def test_entity_can_write_through_stack(self, kernle_sqlite):
        """Full round-trip: entity writes to stack, data visible in storage."""
        _ = kernle_sqlite.entity  # create entity first
        _ = kernle_sqlite.stack  # auto-attaches

        ep_id = kernle_sqlite.entity.episode(
            objective="Test compat layer",
            outcome="Verified round-trip works",
        )
        assert ep_id is not None

        # The episode should be retrievable from the stack
        episodes = kernle_sqlite.stack.get_episodes(limit=10)
        assert any(ep.id == ep_id for ep in episodes)


class TestBackwardCompatibility:
    """Verify that existing code paths are not affected."""

    def test_storage_property_still_works(self, kernle_sqlite):
        assert kernle_sqlite.storage is kernle_sqlite._storage

    def test_internal_storage_unaffected(self, kernle_sqlite):
        """Internal _storage attribute still works for all operations."""
        note = Note(
            id=str(uuid.uuid4()),
            stack_id="compat_agent",
            content="test note",
            note_type="observation",
            created_at=datetime.now(timezone.utc),
        )
        note_id = kernle_sqlite._storage.save_note(note)
        assert note_id is not None
        notes = kernle_sqlite._storage.get_notes()
        assert len(notes) >= 1

    def test_existing_methods_work(self, kernle_sqlite):
        """Core methods that use self._storage still work."""
        ep_id = kernle_sqlite.episode(
            objective="Test backward compat",
            outcome="Methods still work",
        )
        assert ep_id is not None

    def test_entity_and_stack_do_not_interfere_with_storage(self, kernle_sqlite):
        """Creating entity/stack doesn't break _storage operations."""
        # Create entity and stack
        _ = kernle_sqlite.entity
        _ = kernle_sqlite.stack

        # _storage should still work fine
        note = Note(
            id=str(uuid.uuid4()),
            stack_id="compat_agent",
            content="after entity/stack creation",
            note_type="observation",
            created_at=datetime.now(timezone.utc),
        )
        note_id = kernle_sqlite._storage.save_note(note)
        assert note_id is not None
        notes = kernle_sqlite._storage.get_notes()
        assert any(n.content == "after entity/stack creation" for n in notes)
