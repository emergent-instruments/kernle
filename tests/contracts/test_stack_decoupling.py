"""Tests for Phase 2: Stack decoupled from SQLiteStorage.

Verifies that Stack accepts injected Storage, from_sqlite() factory works,
and the class is importable from the expected paths.
"""

from unittest.mock import MagicMock

import pytest

from kernle.storage.sqlite import SQLiteStorage


class TestStackImports:
    """Stack is importable from both kernle.stack and kernle.stack.sqlite_stack."""

    def test_import_from_stack_package(self):
        from kernle.stack import Stack

        assert Stack is not None

    def test_import_from_sqlite_stack_module(self):
        from kernle.stack.sqlite_stack import Stack

        assert Stack is not None

    def test_stack_in_all(self):
        import kernle.stack

        assert "Stack" in kernle.stack.__all__


class TestStackConstructorAcceptsStorage:
    """Stack(stack_id, storage=...) accepts any Storage implementation."""

    def test_construct_with_sqlite_storage(self, tmp_path):
        from kernle.stack import Stack

        storage = SQLiteStorage(stack_id="test", db_path=tmp_path / "test.db")
        stack = Stack(stack_id="test", storage=storage, components=[], enforce_provenance=False)
        assert stack._backend is storage
        assert stack._storage is storage

    def test_construct_with_mock_storage(self):
        """A mock that satisfies Storage protocol can be injected."""
        from kernle.stack import Stack

        mock_storage = MagicMock(spec=SQLiteStorage)
        mock_storage.stack_id = "test"
        mock_storage.get_stack_setting.return_value = None
        mock_storage.get_trust_assessment.return_value = None
        mock_storage.save_trust_assessment.return_value = "trust-id"

        stack = Stack(
            stack_id="test", storage=mock_storage, components=[], enforce_provenance=False
        )
        assert stack._backend is mock_storage

    def test_storage_is_required(self):
        """Stack() without storage raises TypeError."""
        from kernle.stack import Stack

        with pytest.raises(TypeError):
            Stack(stack_id="test")


class TestFromSqliteFactory:
    """Stack.from_sqlite() is a convenience factory for the SQLite path."""

    def test_from_sqlite_basic(self, tmp_path):
        from kernle.stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test", db_path=tmp_path / "test.db", components=[], enforce_provenance=False
        )
        assert isinstance(stack._backend, SQLiteStorage)

    def test_from_sqlite_default_components(self, tmp_path):
        """from_sqlite without components= loads default components."""
        from kernle.stack import Stack

        stack = Stack.from_sqlite(stack_id="test", db_path=tmp_path / "test.db")
        assert len(stack._components) > 0  # default components loaded

    def test_from_sqlite_bare_components(self, tmp_path):
        """from_sqlite with components=[] gives bare stack."""
        from kernle.stack import Stack

        stack = Stack.from_sqlite(stack_id="test", db_path=tmp_path / "test.db", components=[])
        assert len(stack._components) == 0

    def test_from_sqlite_passes_cloud_storage(self, tmp_path):
        """from_sqlite passes cloud_storage to SQLiteStorage."""
        from kernle.stack import Stack

        mock_cloud = MagicMock()
        stack = Stack.from_sqlite(
            stack_id="test",
            db_path=tmp_path / "test.db",
            cloud_storage=mock_cloud,
            components=[],
            enforce_provenance=False,
        )
        assert stack._backend.cloud_storage is mock_cloud

    def test_from_sqlite_passes_embedder(self, tmp_path):
        """from_sqlite passes embedder to SQLiteStorage."""
        from kernle.stack import Stack

        mock_embedder = MagicMock()
        stack = Stack.from_sqlite(
            stack_id="test",
            db_path=tmp_path / "test.db",
            embedder=mock_embedder,
            components=[],
            enforce_provenance=False,
        )
        # Embedder is set on the storage
        assert stack._backend._embedder is mock_embedder


class TestStackInitSideEffects:
    """Init side effects are preserved regardless of how storage is provided."""

    def test_self_trust_bootstrapped(self, tmp_path):
        """Stack creates self-trust assessment on init."""
        from kernle.stack import Stack

        storage = SQLiteStorage(stack_id="test", db_path=tmp_path / "test.db")
        _stack = Stack(
            stack_id="test", storage=storage, components=[], enforce_provenance=False
        )  # noqa: F841
        trust = storage.get_trust_assessment("identity")
        assert trust is not None

    def test_stack_state_initializing(self, tmp_path):
        """New stack starts in INITIALIZING state."""
        from kernle.protocols import StackState
        from kernle.stack import Stack

        storage = SQLiteStorage(stack_id="test", db_path=tmp_path / "test.db")
        stack = Stack(stack_id="test", storage=storage, components=[], enforce_provenance=False)
        assert stack.state == StackState.INITIALIZING

    def test_default_components_loaded(self, tmp_path):
        """Stack with components=None loads 8 default components."""
        from kernle.stack import Stack

        storage = SQLiteStorage(stack_id="test", db_path=tmp_path / "test.db")
        stack = Stack(stack_id="test", storage=storage, enforce_provenance=False)
        assert len(stack._components) >= 8  # at least 8 default components

    def test_persisted_state_restored(self, tmp_path):
        """Stack restores persisted state from storage."""
        from kernle.protocols import StackState
        from kernle.stack import Stack

        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(stack_id="test", db_path=db_path)
        storage.set_stack_setting("stack_state", "ACTIVE")

        stack = Stack(stack_id="test", storage=storage, components=[], enforce_provenance=False)
        assert stack.state == StackState.ACTIVE


class TestStackProperties:
    """Stack exposes expected properties."""

    def test_stack_id(self, tmp_path):
        from kernle.stack import Stack

        stack = Stack.from_sqlite(
            stack_id="my-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        assert stack.stack_id == "my-stack"

    def test_backend_property(self, tmp_path):
        from kernle.stack import Stack

        storage = SQLiteStorage(stack_id="test", db_path=tmp_path / "test.db")
        stack = Stack(stack_id="test", storage=storage, components=[], enforce_provenance=False)
        assert stack._backend is storage
