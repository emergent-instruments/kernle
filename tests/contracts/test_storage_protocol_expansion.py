"""Tests for Phase 1: Storage Protocol expansion.

Verifies that all 5 new sub-protocols and 2 expanded sub-protocols
are declared in the Storage protocol and implemented by SQLiteStorage.
"""

from typing import Protocol

import pytest

from kernle.storage.base import (
    AtomicUpdateStorage,
    EmbeddingStorage,
    LineageStorage,
    MetaMemoryStorage,
    ProcessingStorage,
    SettingsStorage,
    Storage,
    SuggestionStorage,
)
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Belief, Drive, Episode, Goal, Relationship

# === Sub-protocol existence and runtime_checkable ===


def _is_runtime_checkable_protocol(cls) -> bool:
    """Verify a class is a @runtime_checkable Protocol.

    Checks that:
    1. It's a subclass of Protocol (is a protocol)
    2. It has _is_runtime_protocol = True (is runtime_checkable)
    """
    return issubclass(cls, Protocol) and getattr(cls, "_is_runtime_protocol", False)


class TestNewSubProtocolsExist:
    """Verify that new sub-protocols are @runtime_checkable Protocols."""

    def test_settings_storage_is_runtime_checkable_protocol(self):
        assert _is_runtime_checkable_protocol(SettingsStorage)

    def test_atomic_update_storage_is_runtime_checkable_protocol(self):
        assert _is_runtime_checkable_protocol(AtomicUpdateStorage)

    def test_processing_storage_is_runtime_checkable_protocol(self):
        assert _is_runtime_checkable_protocol(ProcessingStorage)

    def test_lineage_storage_is_runtime_checkable_protocol(self):
        assert _is_runtime_checkable_protocol(LineageStorage)

    def test_embedding_storage_is_runtime_checkable_protocol(self):
        assert _is_runtime_checkable_protocol(EmbeddingStorage)


class TestRuntimeCheckableConformance:
    """Verify that isinstance() works correctly for duck-typed classes.

    A class that implements the right methods should pass isinstance()
    against the runtime_checkable protocol, even without inheriting from it.
    A class missing methods should fail.
    """

    def test_duck_typed_settings_passes(self):
        class MySettings:
            def get_stack_setting(self, key): ...
            def set_stack_setting(self, key, value): ...
            def get_all_stack_settings(self): ...

        assert isinstance(MySettings(), SettingsStorage)

    def test_missing_settings_method_fails(self):
        class IncompleteSettings:
            def get_stack_setting(self, key): ...

            # missing set_stack_setting and get_all_stack_settings

        assert not isinstance(IncompleteSettings(), SettingsStorage)

    def test_duck_typed_processing_passes(self):
        class MyProcessing:
            def mark_episode_processed(self, episode_id): ...
            def mark_note_processed(self, note_id): ...
            def mark_belief_processed(self, belief_id): ...
            def get_processing_config(self): ...
            def set_processing_config(self, layer_transition, **kwargs): ...

        assert isinstance(MyProcessing(), ProcessingStorage)

    def test_duck_typed_lineage_passes(self):
        class MyLineage:
            def get_memories_derived_from(self, memory_type, memory_id): ...
            def get_ungrounded_memories(self, stack_id): ...
            def log_belief_revision(
                self, old_id, new_id, reason=None, actor="system", correlation_id=None
            ): ...
            def boost_memory_strength(self, memory_type, memory_id, amount): ...

        assert isinstance(MyLineage(), LineageStorage)

    def test_duck_typed_embedding_passes(self):
        class MyEmbedding:
            def get_embedding_stats(self): ...

        assert isinstance(MyEmbedding(), EmbeddingStorage)


# === Default implementations (no-op) ===


class TestSettingsStorageDefaults:
    """SettingsStorage methods have sensible defaults."""

    class MinimalSettings(SettingsStorage):
        """Minimal class relying on defaults — should not need any method impl."""

        pass

    def test_get_stack_setting_default_returns_none(self):
        obj = self.MinimalSettings()
        assert obj.get_stack_setting("any_key") is None

    def test_set_stack_setting_default_is_noop(self):
        obj = self.MinimalSettings()
        obj.set_stack_setting("key", "value")  # should not raise

    def test_get_all_stack_settings_default_returns_empty_dict(self):
        obj = self.MinimalSettings()
        assert obj.get_all_stack_settings() == {}


class TestAtomicUpdateStorageDefaults:
    """AtomicUpdateStorage methods have sensible defaults."""

    class MinimalAtomic(AtomicUpdateStorage):
        pass

    def test_update_belief_atomic_default_returns_false(self):
        obj = self.MinimalAtomic()
        belief = Belief(id="b-1", stack_id="test", statement="test")
        assert obj.update_belief_atomic(belief) is False

    def test_update_goal_atomic_default_returns_false(self):
        obj = self.MinimalAtomic()
        goal = Goal(id="g-1", stack_id="test", title="test")
        assert obj.update_goal_atomic(goal) is False

    def test_update_drive_atomic_default_returns_false(self):
        obj = self.MinimalAtomic()
        drive = Drive(id="d-1", stack_id="test", drive_type="test")
        assert obj.update_drive_atomic(drive) is False

    def test_update_relationship_atomic_default_returns_false(self):
        obj = self.MinimalAtomic()
        rel = Relationship(
            id="r-1",
            stack_id="test",
            entity_name="test",
            entity_type="person",
            relationship_type="peer",
        )
        assert obj.update_relationship_atomic(rel) is False

    def test_update_episode_atomic_default_returns_false(self):
        obj = self.MinimalAtomic()
        ep = Episode(id="ep-1", stack_id="test", objective="test", outcome="test")
        assert obj.update_episode_atomic(ep) is False


class TestProcessingStorageDefaults:
    """ProcessingStorage methods have sensible defaults."""

    class MinimalProcessing(ProcessingStorage):
        pass

    def test_mark_episode_processed_default_returns_false(self):
        obj = self.MinimalProcessing()
        assert obj.mark_episode_processed("ep-1") is False

    def test_mark_note_processed_default_returns_false(self):
        obj = self.MinimalProcessing()
        assert obj.mark_note_processed("note-1") is False

    def test_mark_belief_processed_default_returns_false(self):
        obj = self.MinimalProcessing()
        assert obj.mark_belief_processed("belief-1") is False

    def test_get_processing_config_default_returns_empty_list(self):
        obj = self.MinimalProcessing()
        assert obj.get_processing_config() == []

    def test_set_processing_config_default_returns_false(self):
        obj = self.MinimalProcessing()
        assert obj.set_processing_config("raw_to_episode") is False


class TestLineageStorageDefaults:
    """LineageStorage methods have sensible defaults."""

    class MinimalLineage(LineageStorage):
        pass

    def test_get_memories_derived_from_default_returns_empty(self):
        obj = self.MinimalLineage()
        assert obj.get_memories_derived_from("episode", "ep-1") == []

    def test_get_ungrounded_memories_default_returns_empty(self):
        obj = self.MinimalLineage()
        assert obj.get_ungrounded_memories("stack-1") == []

    def test_log_belief_revision_default_returns_tuple(self):
        obj = self.MinimalLineage()
        result = obj.log_belief_revision("old-1", "new-1")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_boost_memory_strength_default_returns_false(self):
        obj = self.MinimalLineage()
        assert obj.boost_memory_strength("belief", "b-1", 0.05) is False


class TestEmbeddingStorageDefaults:
    """EmbeddingStorage methods have sensible defaults."""

    class MinimalEmbedding(EmbeddingStorage):
        pass

    def test_get_embedding_stats_default_returns_dict(self):
        obj = self.MinimalEmbedding()
        stats = obj.get_embedding_stats()
        assert isinstance(stats, dict)
        assert stats["total"] == 0


# === Expanded existing sub-protocols ===


class TestMetaMemoryStorageExpanded:
    """MetaMemoryStorage now includes get_all_active_memories and update_strength_batch."""

    def test_has_get_all_active_memories(self):
        assert hasattr(MetaMemoryStorage, "get_all_active_memories")

    def test_has_update_strength_batch(self):
        assert hasattr(MetaMemoryStorage, "update_strength_batch")

    def test_has_update_memory_metadata(self):
        assert hasattr(MetaMemoryStorage, "update_memory_metadata")


class TestSuggestionStorageExpanded:
    """SuggestionStorage now includes expire_suggestions."""

    def test_has_expire_suggestions(self):
        assert hasattr(SuggestionStorage, "expire_suggestions")


# === Storage composite includes new sub-protocols ===


class TestStorageComposite:
    """Storage composite protocol includes all new sub-protocols."""

    def test_storage_includes_settings(self):
        assert issubclass(Storage, SettingsStorage)

    def test_storage_includes_atomic_update(self):
        assert issubclass(Storage, AtomicUpdateStorage)

    def test_storage_includes_processing(self):
        assert issubclass(Storage, ProcessingStorage)

    def test_storage_includes_lineage(self):
        assert issubclass(Storage, LineageStorage)

    def test_storage_includes_embedding(self):
        assert issubclass(Storage, EmbeddingStorage)


# === SQLiteStorage implements all new protocol methods ===


class TestSQLiteStorageImplementsNew:
    """SQLiteStorage has concrete implementations for all new protocol methods."""

    def test_has_get_stack_setting(self):
        assert hasattr(SQLiteStorage, "get_stack_setting")

    def test_has_set_stack_setting(self):
        assert hasattr(SQLiteStorage, "set_stack_setting")

    def test_has_get_all_stack_settings(self):
        assert hasattr(SQLiteStorage, "get_all_stack_settings")

    def test_has_update_belief_atomic(self):
        assert hasattr(SQLiteStorage, "update_belief_atomic")

    def test_has_update_goal_atomic(self):
        assert hasattr(SQLiteStorage, "update_goal_atomic")

    def test_has_update_drive_atomic(self):
        assert hasattr(SQLiteStorage, "update_drive_atomic")

    def test_has_update_relationship_atomic(self):
        assert hasattr(SQLiteStorage, "update_relationship_atomic")

    def test_has_update_episode_atomic(self):
        assert hasattr(SQLiteStorage, "update_episode_atomic")

    def test_has_mark_episode_processed(self):
        assert hasattr(SQLiteStorage, "mark_episode_processed")

    def test_has_mark_note_processed(self):
        assert hasattr(SQLiteStorage, "mark_note_processed")

    def test_has_mark_belief_processed(self):
        assert hasattr(SQLiteStorage, "mark_belief_processed")

    def test_has_get_processing_config(self):
        assert hasattr(SQLiteStorage, "get_processing_config")

    def test_has_set_processing_config(self):
        assert hasattr(SQLiteStorage, "set_processing_config")

    def test_has_get_memories_derived_from(self):
        assert hasattr(SQLiteStorage, "get_memories_derived_from")

    def test_has_get_ungrounded_memories(self):
        assert hasattr(SQLiteStorage, "get_ungrounded_memories")

    def test_has_log_belief_revision(self):
        assert hasattr(SQLiteStorage, "log_belief_revision")

    def test_has_boost_memory_strength(self):
        assert hasattr(SQLiteStorage, "boost_memory_strength")

    def test_has_get_all_active_memories(self):
        assert hasattr(SQLiteStorage, "get_all_active_memories")

    def test_has_update_strength_batch(self):
        assert hasattr(SQLiteStorage, "update_strength_batch")

    def test_has_expire_suggestions(self):
        assert hasattr(SQLiteStorage, "expire_suggestions")

    def test_has_get_embedding_stats(self):
        assert hasattr(SQLiteStorage, "get_embedding_stats")

    def test_has_update_memory_metadata(self):
        assert hasattr(SQLiteStorage, "update_memory_metadata")


# === Integration: SQLiteStorage passes isinstance check ===


class TestSQLiteStorageConformance:
    """SQLiteStorage conforms to the expanded Storage protocol."""

    @pytest.fixture
    def storage(self, tmp_path):
        return SQLiteStorage(stack_id="test", db_path=tmp_path / "test.db")

    def test_isinstance_storage(self, storage):
        assert isinstance(storage, Storage)

    def test_isinstance_settings_storage(self, storage):
        assert isinstance(storage, SettingsStorage)

    def test_isinstance_atomic_update_storage(self, storage):
        assert isinstance(storage, AtomicUpdateStorage)

    def test_isinstance_processing_storage(self, storage):
        assert isinstance(storage, ProcessingStorage)

    def test_isinstance_lineage_storage(self, storage):
        assert isinstance(storage, LineageStorage)

    def test_isinstance_embedding_storage(self, storage):
        assert isinstance(storage, EmbeddingStorage)


# === _connect() leak replacements ===


class TestConnectLeakReplacement:
    """Verify that update_memory_metadata and update_strength_batch are on Storage protocol."""

    def test_update_memory_metadata_in_metamemory(self):
        """update_memory_metadata should be in MetaMemoryStorage (not requiring _connect)."""
        assert hasattr(MetaMemoryStorage, "update_memory_metadata")

    def test_update_strength_batch_in_metamemory(self):
        """update_strength_batch should be in MetaMemoryStorage (not requiring _connect)."""
        assert hasattr(MetaMemoryStorage, "update_strength_batch")
