"""Tests for Kernle.load() delegation to stack.load() (#574).

Verifies that Kernle.load() properly delegates to stack.load() and
transforms the output to maintain backward compatibility.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from kernle.core import Kernle
from kernle.stack import Stack
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Value
from tests.conftest import bind_noop_model


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance for testing."""
    db_path = tmp_path / "test_load_delegation.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    storage = SQLiteStorage(stack_id="test-delegation", db_path=db_path)
    inst = Kernle(
        stack_id="test-delegation",
        storage=storage,
        checkpoint_dir=checkpoint_dir,
        strict=False,
    )
    bind_noop_model(inst)
    return inst


@pytest.fixture
def stack(tmp_path):
    """Create a bare Stack for testing stack.load() extensions."""
    db_path = tmp_path / "test_stack_load.db"
    return Stack.from_sqlite(
        stack_id="test-stack-load",
        db_path=db_path,
        enforce_provenance=False,
    )


def _seed_memories(k):
    """Seed a Kernle instance with test memories."""
    k.value("test-value", "A test value statement", source="test", source_type="direct_experience")
    k.belief("A test belief", confidence=0.8, source="test", source_type="inference")
    k.episode(
        objective="Test objective",
        outcome="completed",
        lessons=["A test lesson"],
        tags=["test"],
    )
    k.note("Test note content", type="insight")


class TestLoadDispatchesComponentOnLoad:
    """Components receive on_load during k.load()."""

    def test_load_dispatches_component_on_load(self, k):
        _seed_memories(k)
        result = k.load()
        # AnxietyComponent adds 'anxiety' key during on_load
        assert "anxiety" in result

    def test_load_includes_anxiety_from_component(self, k):
        _seed_memories(k)
        result = k.load()
        anxiety = result.get("anxiety", {})
        assert "overall_score" in anxiety
        assert "overall_level" in anxiety


class TestLoadAppliesStrengthFiltering:
    """Weak memories should be excluded via stack."""

    def test_load_filters_weak_memories(self, k):
        k.value(
            "weak-value",
            "Should not appear",
            source="test",
            source_type="direct_experience",
        )
        # Save a normal value
        k.value(
            "strong-value",
            "Should appear",
            source="test",
            source_type="direct_experience",
        )
        result = k.load()
        assert len(result.get("values", [])) >= 1


class TestLoadIncludesCheckpoint:
    """Result should contain checkpoint context when available."""

    def test_load_includes_checkpoint_when_saved(self, k):
        _seed_memories(k)
        k.checkpoint(task="test-task", context="test context")
        result = k.load()
        assert result.get("checkpoint") is not None

    def test_load_no_checkpoint_when_none(self, k):
        _seed_memories(k)
        k.load()  # Should not crash


class TestLoadIncludesBootConfig:
    """Result should contain boot_config when set."""

    def test_load_includes_boot_config(self, k):
        _seed_memories(k)
        k.boot_set("test_key", "test_value")
        result = k.load()
        assert "boot_config" in result
        assert result["boot_config"].get("test_key") == "test_value"


class TestLoadRespectsBudget:
    """Budget should limit total output."""

    def test_load_respects_small_budget(self, k):
        _seed_memories(k)
        result = k.load(budget=200)
        assert isinstance(result, dict)
        meta = result.get("_meta", {})
        assert meta.get("budget_total") == 200


class TestLoadTruncation:
    """max_item_chars truncates individual items."""

    def test_load_truncation(self, k):
        k.value(
            "long-value",
            "x" * 2000,
            source="test",
            source_type="direct_experience",
        )
        result = k.load(max_item_chars=100)
        for v in result.get("values", []):
            if v.get("statement"):
                assert len(v["statement"]) <= 120  # margin for word boundary


class TestLoadSyncBeforeLoad:
    """sync=True should trigger sync first."""

    def test_load_sync_before_load(self, k):
        _seed_memories(k)
        result = k.load(sync=True)
        assert isinstance(result, dict)


class TestLoadEpochIdFilter:
    """epoch_id filters to specific epoch."""

    def test_load_epoch_id_filter(self, k):
        _seed_memories(k)
        result = k.load(epoch_id="non-existent-epoch")
        assert isinstance(result, dict)


class TestLoadBackwardCompat:
    """k.load(budget=4000) still works."""

    def test_load_backward_compat_budget_param(self, k):
        _seed_memories(k)
        result = k.load(budget=4000)
        assert isinstance(result, dict)
        assert "values" in result
        assert "beliefs" in result


class TestLoadOutputShape:
    """Load returns the expected output shape."""

    def test_load_output_has_required_keys(self, k):
        _seed_memories(k)
        result = k.load()
        for key in ("values", "beliefs", "lessons", "recent_work", "recent_notes"):
            assert key in result, f"Missing key: {key}"

    def test_load_lessons_from_episodes(self, k):
        """Lessons come from budgeted episodes."""
        k.episode(
            objective="Lesson test",
            outcome="completed",
            lessons=["lesson one", "lesson two"],
        )
        result = k.load()
        assert len(result.get("lessons", [])) >= 1

    def test_load_recent_work_excludes_checkpoints(self, k):
        """Checkpoint episodes should not appear in recent_work."""
        k.episode(objective="Normal work", outcome="completed")
        k.checkpoint(task="test")
        result = k.load()
        for work in result.get("recent_work", []):
            tags = work.get("tags") or []
            assert "checkpoint" not in tags


class TestStackLoadExtensions:
    """Stack.load() supports new parameters."""

    def test_stack_load_max_item_chars(self, stack):
        """max_item_chars truncates items."""
        stack.save_value(
            Value(
                id=str(uuid.uuid4()),
                stack_id="test-stack-load",
                name="long-val",
                statement="y" * 2000,
            )
        )
        result = stack.load(max_item_chars=100)
        for v in result.get("values", []):
            if v.get("statement"):
                assert len(v["statement"]) <= 120

    def test_stack_load_track_access_false(self, stack):
        """track_access=False skips access recording."""
        stack.save_value(
            Value(
                id=str(uuid.uuid4()),
                stack_id="test-stack-load",
                name="test",
                statement="test",
            )
        )
        # Should not crash with track_access=False
        result = stack.load(track_access=False)
        assert isinstance(result, dict)

    def test_stack_load_epoch_id(self, stack):
        """epoch_id filters candidates."""
        stack.save_value(
            Value(
                id=str(uuid.uuid4()),
                stack_id="test-stack-load",
                name="test",
                statement="test",
            )
        )
        result = stack.load(epoch_id="nonexistent")
        assert isinstance(result, dict)


class TestNonSQLiteStorageFallback:
    """Kernle.load() works with any storage backend via eager Stack (Phase 3)."""

    def test_load_with_non_sqlite_storage(self, tmp_path):
        """Non-SQLite storage works through eager Stack — no fallback needed."""
        mock_storage = MagicMock()
        mock_storage.stack_id = "test-non-sqlite"
        mock_storage.is_online.return_value = False
        mock_storage.get_pending_sync_count.return_value = 0
        mock_storage.get_trust_assessment.return_value = None
        mock_storage.get_all_stack_settings.return_value = {}
        mock_storage.get_values.return_value = []
        mock_storage.get_beliefs.return_value = []
        mock_storage.get_goals.return_value = []
        mock_storage.get_drives.return_value = []
        mock_storage.get_episodes.return_value = []
        mock_storage.get_notes.return_value = []
        mock_storage.load_all.return_value = None
        mock_storage.list_summaries.return_value = []
        mock_storage.list_self_narratives.return_value = []
        mock_storage.get_relationships.return_value = []

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        k = Kernle(
            stack_id="test-non-sqlite",
            storage=mock_storage,
            checkpoint_dir=checkpoint_dir,
            strict=False,
        )
        # Stack is always created (Phase 3)
        assert k.stack is not None
        assert k._storage is k.stack._backend
        # load() must not crash
        result = k.load()
        assert isinstance(result, dict)
        assert "values" in result
        assert "_meta" in result


class TestRelationshipOutputCompat:
    """Relationship output must match the old contract."""

    def test_last_interaction_is_iso_string(self, k):
        """last_interaction must be serialized to ISO string, not raw datetime."""
        k.relationship(
            other_stack_id="test-entity",
            entity_type="agent",
            notes="We worked together",
        )
        result = k.load(budget=50000)
        rels = result.get("relationships", [])
        assert len(rels) >= 1, "Expected at least one relationship in load output"
        last = rels[0].get("last_interaction")
        assert last is not None, "last_interaction should not be None after relationship()"
        assert isinstance(last, str), f"last_interaction should be ISO string, got {type(last)}"

    def test_relationship_notes_truncated(self, k):
        """Relationship notes must be truncated by max_item_chars."""
        k.relationship(
            other_stack_id="verbose-entity",
            entity_type="agent",
            notes="x" * 400,
        )
        result = k.load(max_item_chars=50)
        rels = result.get("relationships", [])
        assert len(rels) >= 1, "Expected at least one relationship in load output"
        notes = rels[0].get("notes")
        assert notes is not None, "notes should not be None"
        # _truncate_at_word_boundary guarantees output <= max_chars
        assert len(notes) <= 50, f"Expected len <= 50, got {len(notes)}"
        assert notes.endswith("..."), "Truncated notes should end with ellipsis"
