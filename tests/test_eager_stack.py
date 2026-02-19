"""Tests for Phase 3: Eager Stack creation in Kernle class (#880).

Verifies:
- Stack is created eagerly in __init__ (not lazily)
- Single shared storage instance (no double-storage)
- stack property always returns a Stack (never None)
- _write_backend always returns the Stack
- Entity auto-attaches the eager Stack
- Custom storage injection flows through to Stack
- Strict/non-strict both route through Stack
"""

import pytest

from kernle.core import Kernle
from kernle.storage.sqlite import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance for testing."""
    db_path = tmp_path / "test_eager_stack.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    storage = SQLiteStorage(stack_id="test-eager", db_path=db_path)
    inst = Kernle(
        stack_id="test-eager",
        storage=storage,
        checkpoint_dir=checkpoint_dir,
        strict=True,
    )
    bind_noop_model(inst)
    return inst


@pytest.fixture
def k_nonstrict(tmp_path):
    """Create a non-strict Kernle instance."""
    db_path = tmp_path / "test_eager_nonstrict.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    storage = SQLiteStorage(stack_id="test-nonstrict", db_path=db_path)
    inst = Kernle(
        stack_id="test-nonstrict",
        storage=storage,
        checkpoint_dir=checkpoint_dir,
        strict=False,
    )
    bind_noop_model(inst)
    return inst


class TestSharedStorageInstance:
    """Stack and Kernle share the same storage — no double instantiation."""

    def test_storage_is_same_object(self, k):
        """k._storage is k.stack._backend — exact same instance."""
        assert k._storage is k.stack._backend

    def test_no_second_storage_created(self, tmp_path):
        """Creating Kernle should not produce two SQLiteStorage instances."""
        db_path = tmp_path / "single.db"
        checkpoint_dir = tmp_path / "cp"
        checkpoint_dir.mkdir()
        storage = SQLiteStorage(stack_id="single", db_path=db_path)
        inst = Kernle(
            stack_id="single",
            storage=storage,
            checkpoint_dir=checkpoint_dir,
            strict=True,
        )
        # The storage passed in should be the SAME one the stack uses
        assert inst._storage is storage
        assert inst.stack._backend is storage


class TestEagerStackCreation:
    """Stack is created during __init__, not lazily on first access."""

    def test_stack_exists_immediately(self, tmp_path):
        """_stack attribute should exist right after __init__."""
        db_path = tmp_path / "eager.db"
        checkpoint_dir = tmp_path / "cp"
        checkpoint_dir.mkdir()
        storage = SQLiteStorage(stack_id="eager", db_path=db_path)
        inst = Kernle(
            stack_id="eager",
            storage=storage,
            checkpoint_dir=checkpoint_dir,
        )
        # _stack should be set directly, not via lazy property
        assert hasattr(inst, "_stack")
        assert inst._stack is not None

    def test_stack_property_returns_stack(self, k):
        """stack property should never return None for any valid Kernle."""
        from kernle.stack import Stack

        assert k.stack is not None
        assert isinstance(k.stack, Stack)

    def test_stack_is_active(self, k):
        """Eager stack should be in ACTIVE state after init."""
        from kernle.protocols import StackState

        assert k.stack.state == StackState.ACTIVE


class TestWriteBackendAlwaysStack:
    """_write_backend should always return the Stack."""

    def test_write_backend_is_stack_strict(self, k):
        """In strict mode, _write_backend is the stack."""
        assert k._write_backend is k.stack

    def test_write_backend_is_stack_nonstrict(self, k_nonstrict):
        """In non-strict mode, _write_backend is ALSO the stack."""
        assert k_nonstrict._write_backend is k_nonstrict.stack

    def test_write_backend_never_raises(self, k):
        """_write_backend should not raise ValueError."""
        # Previously it could raise if stack was None in strict mode
        backend = k._write_backend
        assert backend is not None


class TestEntityAutoAttach:
    """Entity should auto-attach the eager Stack when first accessed."""

    def test_entity_gets_stack_attached(self, k):
        """Accessing k.entity should auto-attach k.stack."""
        entity = k.entity
        assert entity.active_stack is k.stack

    def test_entity_attach_is_idempotent(self, k):
        """Multiple accesses to k.entity don't re-attach."""
        entity1 = k.entity
        entity2 = k.entity
        assert entity1 is entity2
        assert entity1.active_stack is k.stack


class TestCustomStorageInjection:
    """Custom storage backends work with eager Stack."""

    def test_injected_storage_used_by_stack(self, tmp_path):
        """Stack should use the same storage instance passed to Kernle."""
        db_path = tmp_path / "custom.db"
        checkpoint_dir = tmp_path / "cp"
        checkpoint_dir.mkdir()
        custom_storage = SQLiteStorage(stack_id="custom", db_path=db_path)
        inst = Kernle(
            stack_id="custom",
            storage=custom_storage,
            checkpoint_dir=checkpoint_dir,
        )
        assert inst.stack._backend is custom_storage

    def test_default_storage_created_when_none(self, tmp_path):
        """When no storage is passed, Kernle creates SQLiteStorage."""
        checkpoint_dir = tmp_path / "cp"
        checkpoint_dir.mkdir()
        inst = Kernle(
            stack_id="auto-storage",
            checkpoint_dir=checkpoint_dir,
        )
        assert isinstance(inst._storage, SQLiteStorage)
        assert inst._storage is inst.stack._backend


class TestStrictModeViaStack:
    """Strict mode enforcement flows through Stack."""

    def test_strict_stack_enforces_provenance(self, k):
        """Strict Kernle creates Stack with enforce_provenance=True."""
        assert k.stack._enforce_provenance is True

    def test_nonstrict_stack_no_provenance(self, k_nonstrict):
        """Non-strict Kernle creates Stack with enforce_provenance=False."""
        assert k_nonstrict.stack._enforce_provenance is False


class TestWritesThroughStack:
    """Writes should go through Stack regardless of strict mode."""

    def test_episode_write_nonstrict(self, k_nonstrict):
        """Episodes are saved through Stack in non-strict mode."""
        ep_id = k_nonstrict.episode(
            objective="Test nonstrict write",
            outcome="completed",
        )
        assert ep_id is not None
        episodes = k_nonstrict.stack.get_episodes()
        assert any(e.id == ep_id for e in episodes)

    def test_note_write_nonstrict(self, k_nonstrict):
        """Notes go through Stack in non-strict mode."""
        note_id = k_nonstrict.note("Test note via stack", type="insight")
        assert note_id is not None
        notes = k_nonstrict.stack.get_notes()
        assert any(n.id == note_id for n in notes)

    def test_raw_write_through_stack(self, k):
        """Raw entries go through Stack in strict mode."""
        raw_id = k.raw("test blob content")
        assert raw_id is not None


class TestLoadWithEagerStack:
    """load() uses eager Stack — no fallback path needed."""

    def test_load_uses_stack(self, k_nonstrict):
        """load() delegates to stack.load()."""
        k_nonstrict.episode(objective="Seed", outcome="done")
        result = k_nonstrict.load()
        assert isinstance(result, dict)
        assert "values" in result
        assert "_meta" in result

    def test_load_nonstrict(self, k_nonstrict):
        """load() works in non-strict mode too."""
        result = k_nonstrict.load()
        assert isinstance(result, dict)
        assert "values" in result
