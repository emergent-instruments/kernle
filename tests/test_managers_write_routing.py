"""Tests for ManagersMixin write routing through Stack.

Phase 4: Verifies that memory-adjacent writes in ManagersMixin route
through self._write_backend (Stack) instead of self._storage directly.
"""

import pytest

from kernle import Kernle
from kernle.protocols import MaintenanceModeError
from kernle.storage.sqlite import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def k_strict(tmp_path):
    """Kernle in strict mode with noop model."""
    db_path = tmp_path / "managers_test.db"
    storage = SQLiteStorage(stack_id="mgr-test", db_path=db_path)
    k = Kernle(
        stack_id="mgr-test",
        storage=storage,
        checkpoint_dir=tmp_path / "cp",
        strict=True,
    )
    bind_noop_model(k)
    return k


@pytest.fixture
def k_nonstrict(tmp_path):
    """Kernle in non-strict mode with noop model."""
    db_path = tmp_path / "managers_test.db"
    storage = SQLiteStorage(stack_id="mgr-test", db_path=db_path)
    k = Kernle(
        stack_id="mgr-test",
        storage=storage,
        checkpoint_dir=tmp_path / "cp",
        strict=False,
    )
    bind_noop_model(k)
    return k


class TestWriteRoutingMaintenance:
    """Verify writes are blocked in maintenance mode (proves Stack routing)."""

    def test_epoch_create_blocked_in_maintenance(self, k_nonstrict):
        """epoch_create writes should go through Stack."""
        k_nonstrict.stack.enter_maintenance()
        with pytest.raises(MaintenanceModeError):
            k_nonstrict.epoch_create("test-epoch")

    def test_epoch_close_blocked_in_maintenance(self, k_nonstrict):
        """epoch_close writes should go through Stack."""
        # Create an epoch first (before maintenance)
        epoch_id = k_nonstrict.epoch_create("test-epoch")
        k_nonstrict.stack.enter_maintenance()
        with pytest.raises(MaintenanceModeError):
            k_nonstrict.epoch_close(epoch_id)

    def test_summary_save_blocked_in_maintenance(self, k_nonstrict):
        """summary_save writes should go through Stack."""
        k_nonstrict.stack.enter_maintenance()
        with pytest.raises(MaintenanceModeError):
            k_nonstrict.summary_save(
                content="Test summary",
                scope="month",
                period_start="2026-01-01",
                period_end="2026-01-31",
            )

    def test_narrative_save_blocked_in_maintenance(self, k_nonstrict):
        """narrative_save writes should go through Stack."""
        k_nonstrict.stack.enter_maintenance()
        with pytest.raises(MaintenanceModeError):
            k_nonstrict.narrative_save(content="I am a test agent")

    def test_entity_model_blocked_in_maintenance(self, k_nonstrict):
        """add_entity_model writes should go through Stack."""
        k_nonstrict.stack.enter_maintenance()
        with pytest.raises(MaintenanceModeError):
            k_nonstrict.add_entity_model(
                entity_name="alice",
                model_type="behavioral",
                observation="Alice prefers direct communication",
            )


class TestBootConfigInfrastructure:
    """Boot config is infrastructure — stays on self._storage, works during maintenance."""

    def test_boot_set_works_in_maintenance(self, k_nonstrict):
        """Boot config changes should work even in maintenance mode."""
        k_nonstrict.stack.enter_maintenance()
        k_nonstrict.boot_set("model", "anthropic:claude-3")
        assert k_nonstrict.boot_get("model") == "anthropic:claude-3"

    def test_boot_delete_works_in_maintenance(self, k_nonstrict):
        """Boot config deletion should work even in maintenance mode."""
        k_nonstrict.boot_set("key", "value")
        k_nonstrict.stack.enter_maintenance()
        assert k_nonstrict.boot_delete("key") is True

    def test_boot_clear_works_in_maintenance(self, k_nonstrict):
        """Boot config clear should work even in maintenance mode."""
        k_nonstrict.boot_set("a", "1")
        k_nonstrict.stack.enter_maintenance()
        assert k_nonstrict.boot_clear() >= 1


class TestWriteRoutingNormal:
    """Verify write operations work normally (not in maintenance)."""

    def test_epoch_create_roundtrip(self, k_nonstrict):
        """epoch_create through Stack stores correctly."""
        epoch_id = k_nonstrict.epoch_create("test-epoch", trigger_type="declared")
        assert epoch_id is not None
        epoch = k_nonstrict.get_epoch(epoch_id)
        assert epoch.name == "test-epoch"

    def test_epoch_close_roundtrip(self, k_nonstrict):
        """epoch_close through Stack works correctly."""
        epoch_id = k_nonstrict.epoch_create("to-close")
        result = k_nonstrict.epoch_close(epoch_id, summary="Done")
        assert result is True
        epoch = k_nonstrict.get_epoch(epoch_id)
        assert epoch.ended_at is not None

    def test_summary_save_roundtrip(self, k_nonstrict):
        """summary_save through Stack stores correctly."""
        summary_id = k_nonstrict.summary_save(
            content="Monthly summary content",
            scope="month",
            period_start="2026-01-01",
            period_end="2026-01-31",
        )
        assert summary_id is not None
        summary = k_nonstrict.summary_get(summary_id)
        assert summary.content == "Monthly summary content"

    def test_narrative_save_roundtrip(self, k_nonstrict):
        """narrative_save through Stack stores correctly."""
        narrative_id = k_nonstrict.narrative_save(
            content="I value clarity and precision",
            narrative_type="identity",
        )
        assert narrative_id is not None
        active = k_nonstrict.narrative_get_active("identity")
        assert active is not None
        assert active.content == "I value clarity and precision"

    def test_entity_model_roundtrip(self, k_nonstrict):
        """add_entity_model through Stack stores correctly."""
        model_id = k_nonstrict.add_entity_model(
            entity_name="bob",
            model_type="capability",
            observation="Bob is skilled at Python",
            confidence=0.85,
        )
        assert model_id is not None
        models = k_nonstrict.get_entity_models(entity_name="bob")
        assert len(models) >= 1
        assert models[0]["observation"] == "Bob is skilled at Python"


class TestReadsDirect:
    """Verify read operations still go directly to storage."""

    def test_reads_work_in_maintenance(self, k_nonstrict):
        """Read operations should work even in maintenance mode."""
        # Create data first
        k_nonstrict.epoch_create("readable-epoch")
        k_nonstrict.stack.enter_maintenance()

        # Reads should still work
        epochs = k_nonstrict.get_epochs(limit=10)
        assert len(epochs) >= 1
        # get_current_epoch should also work (may be None after maintenance)
        k_nonstrict.get_current_epoch()
        stats = k_nonstrict.status()
        assert "stack_id" in stats

    def test_search_works_in_maintenance(self, k_nonstrict):
        """Search should work in maintenance mode."""
        k_nonstrict.stack.enter_maintenance()
        results = k_nonstrict.search("test query")
        assert isinstance(results, list)

    def test_load_drives_works_in_maintenance(self, k_nonstrict):
        """load_drives should work in maintenance mode."""
        k_nonstrict.stack.enter_maintenance()
        drives = k_nonstrict.load_drives()
        assert isinstance(drives, list)

    def test_load_relationships_works_in_maintenance(self, k_nonstrict):
        """load_relationships should work in maintenance mode."""
        k_nonstrict.stack.enter_maintenance()
        rels = k_nonstrict.load_relationships()
        assert isinstance(rels, list)
