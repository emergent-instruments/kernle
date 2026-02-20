"""Tests for storage backend discovery and factory.

Phase 5: Verifies entry point discovery for storage backends,
the create_storage factory function, and KERNLE_STORAGE_BACKEND env var.
"""

import pytest


class TestEntryPointConstant:
    """Verify the entry point group constant exists."""

    def test_storage_entry_point_group_defined(self):
        from kernle.protocols import ENTRY_POINT_GROUP_STORAGE

        assert ENTRY_POINT_GROUP_STORAGE == "kernle.storage"


class TestDiscoverStorage:
    """Verify discover_storage() finds registered backends."""

    def test_discover_finds_sqlite(self):
        from kernle.discovery import discover_storage

        backends = discover_storage()
        names = {b.name for b in backends}
        assert "sqlite" in names

    def test_discover_returns_discovered_components(self):
        from kernle.discovery import DiscoveredComponent, discover_storage

        backends = discover_storage()
        assert all(isinstance(b, DiscoveredComponent) for b in backends)

    def test_sqlite_component_metadata(self):
        from kernle.discovery import discover_storage

        backends = discover_storage()
        sqlite = next(b for b in backends if b.name == "sqlite")
        assert sqlite.module == "kernle.storage.sqlite"
        assert sqlite.attr == "SQLiteStorage"

    def test_discover_all_includes_storage(self):
        from kernle.discovery import discover_all
        from kernle.protocols import ENTRY_POINT_GROUP_STORAGE

        all_components = discover_all()
        assert ENTRY_POINT_GROUP_STORAGE in all_components


class TestCreateStorage:
    """Verify the create_storage factory function."""

    def test_create_sqlite_default(self, tmp_path):
        from kernle.storage.factory import create_storage

        storage = create_storage(
            backend="sqlite",
            stack_id="factory-test",
            db_path=tmp_path / "test.db",
        )
        from kernle.storage.sqlite import SQLiteStorage

        assert isinstance(storage, SQLiteStorage)
        assert storage.stack_id == "factory-test"

    def test_create_sqlite_explicit(self, tmp_path):
        from kernle.storage.factory import create_storage

        storage = create_storage(
            backend="sqlite",
            stack_id="explicit-test",
            db_path=tmp_path / "explicit.db",
        )
        assert storage.stack_id == "explicit-test"

    def test_create_nonexistent_raises(self):
        from kernle.storage.factory import create_storage

        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_storage(backend="nonexistent", stack_id="test")

    def test_create_nonexistent_lists_available(self):
        from kernle.storage.factory import create_storage

        with pytest.raises(ValueError, match="sqlite"):
            create_storage(backend="nonexistent", stack_id="test")

    def test_default_backend_is_sqlite(self, tmp_path):
        from kernle.storage.factory import create_storage

        storage = create_storage(stack_id="default-test", db_path=tmp_path / "default.db")
        from kernle.storage.sqlite import SQLiteStorage

        assert isinstance(storage, SQLiteStorage)


class TestKernleEnvVar:
    """Verify KERNLE_STORAGE_BACKEND env var in Kernle.__init__."""

    def test_env_var_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KERNLE_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("KERNLE_HOME", str(tmp_path))

        from kernle import Kernle
        from kernle.storage.sqlite import SQLiteStorage

        k = Kernle(stack_id="env-test", strict=False)
        assert isinstance(k._storage, SQLiteStorage)

    def test_env_var_nonexistent_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KERNLE_STORAGE_BACKEND", "nonexistent")
        monkeypatch.setenv("KERNLE_HOME", str(tmp_path))

        from kernle import Kernle

        with pytest.raises(ValueError, match="Unknown storage backend"):
            Kernle(stack_id="env-test", strict=False)

    def test_no_env_var_defaults_to_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KERNLE_STORAGE_BACKEND", raising=False)
        monkeypatch.setenv("KERNLE_HOME", str(tmp_path))

        from kernle import Kernle
        from kernle.storage.sqlite import SQLiteStorage

        k = Kernle(stack_id="no-env-test", strict=False)
        assert isinstance(k._storage, SQLiteStorage)
