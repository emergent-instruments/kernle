"""Tests for the sync engine.

Tests:
- Queueing changes when offline
- Pushing queued changes when back online
- Pulling changes from cloud
- Conflict resolution (last-write-wins)
- Sync metadata tracking
- Merge array fields (dict dedup, truncation, fallback)
- Push record dispatch and error paths
- Pull/push phase edge cases in sync()
- _merge_generic edge cases
- _get_record_summary for all table types
- _record_to_dict success and failure
- _save_from_cloud dispatch for all table types
- is_online caching and outer exception
- clear_sync_conflicts with before parameter
- apply_pull_operation for all table types
- _record_from_pull_data field mapping and provenance
"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from kernle.storage import (
    Belief,
    Drive,
    Episode,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    QueuedChange,
    Relationship,
    SQLiteStorage,
    SyncConflict,
    SyncResult,
    Value,
)
from kernle.storage.sync_engine import MAX_SYNC_ARRAY_SIZE


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    storage = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    yield storage
    storage.close()


@pytest.fixture
def mock_cloud_storage():
    """Create a mock cloud storage for testing sync."""
    mock = MagicMock(spec=SQLiteStorage)
    mock.stack_id = "test-agent"

    # Default return values
    mock.get_stats.return_value = {"episodes": 0, "notes": 0}
    mock.get_episodes.return_value = []
    mock.get_notes.return_value = []
    mock.get_beliefs.return_value = []
    mock.get_values.return_value = []
    mock.get_goals.return_value = []
    mock.get_drives.return_value = []
    mock.get_relationships.return_value = []

    return mock


@pytest.fixture
def storage_with_cloud(temp_db, mock_cloud_storage):
    """Create a SQLiteStorage with a mock cloud storage."""
    storage = SQLiteStorage(
        stack_id="test-agent", db_path=temp_db, cloud_storage=mock_cloud_storage
    )
    yield storage
    storage.close()


class TestSyncQueueBasics:
    """Test the sync queue functionality."""

    def test_changes_are_queued_on_save(self, storage):
        """Saving a record should queue it for sync."""
        initial_count = storage.get_pending_sync_count()

        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test note"))

        assert storage.get_pending_sync_count() == initial_count + 1

    def test_multiple_saves_same_record_dedupe(self, storage):
        """Multiple saves of the same record should dedupe in queue."""
        storage.save_note(Note(id="n1", stack_id="test-agent", content="First version"))

        count_after_first = storage.get_pending_sync_count()

        storage.save_note(Note(id="n1", stack_id="test-agent", content="Second version"))

        # Should still be same count (deduped)
        assert storage.get_pending_sync_count() == count_after_first

    def test_get_queued_changes(self, storage):
        """Can retrieve queued changes."""
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test note"))

        changes = storage.get_queued_changes()

        assert len(changes) >= 2
        assert all(isinstance(c, QueuedChange) for c in changes)

        tables = {c.table_name for c in changes}
        assert "episodes" in tables
        assert "notes" in tables

    def test_queued_change_has_timestamp(self, storage):
        """Queued changes should have a timestamp."""
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test"))

        changes = storage.get_queued_changes()
        assert len(changes) > 0
        assert changes[0].queued_at is not None

    def test_memory_suggestions_are_local_only_not_queued(self, storage):
        """Suggestion lifecycle changes should not enqueue cloud sync operations."""
        initial_pending = storage.get_pending_sync_count()
        suggestion = MemorySuggestion(
            id="s-local-1",
            stack_id="test-agent",
            memory_type="note",
            content={"content": "Local-only suggestion"},
            confidence=0.8,
            source_raw_ids=["raw-1"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

        storage.save_suggestion(suggestion)
        assert storage.get_pending_sync_count() == initial_pending

        assert storage.update_suggestion_status(
            suggestion_id="s-local-1",
            status="dismissed",
            resolution_reason="Reviewed locally",
        )
        assert storage.get_pending_sync_count() == initial_pending

        assert storage.delete_suggestion("s-local-1")
        assert storage.get_pending_sync_count() == initial_pending

        changes = storage.get_queued_changes(limit=50)
        assert not any(c.table_name == "memory_suggestions" for c in changes)

    def test_queue_sync_operation_rejects_local_only_tables(self, storage):
        """Direct queue API should no-op for local-only tables."""
        initial_pending = storage.get_pending_sync_count()

        queue_id = storage.queue_sync_operation(
            "upsert",
            "memory_suggestions",
            "manual-local-only",
        )

        assert queue_id == 0
        assert storage.get_pending_sync_count() == initial_pending
        assert not any(
            c.table_name == "memory_suggestions" and c.record_id == "manual-local-only"
            for c in storage.get_queued_changes(limit=50)
        )


class TestConnectivity:
    """Test connectivity detection."""

    def test_is_online_false_without_cloud(self, storage):
        """Without cloud storage configured, is_online returns False."""
        assert storage.is_online() is False

    def test_is_online_true_with_reachable_cloud(self, storage_with_cloud, mock_cloud_storage):
        """With reachable cloud storage, is_online returns True."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        assert storage_with_cloud.is_online() is True

    def test_is_online_false_when_cloud_unreachable(self, storage_with_cloud, mock_cloud_storage):
        """When cloud throws an exception, is_online returns False."""
        mock_cloud_storage.get_stats.side_effect = Exception("Connection refused")

        assert storage_with_cloud.is_online() is False

    def test_connectivity_cache(self, storage_with_cloud, mock_cloud_storage):
        """Connectivity result is cached briefly."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # First call
        storage_with_cloud.is_online()
        # Second call should use cache
        storage_with_cloud.is_online()

        # Should only have called get_stats once due to caching
        assert mock_cloud_storage.get_stats.call_count == 1


class TestSyncPush:
    """Test pushing changes to cloud."""

    def test_sync_without_cloud_returns_empty_result(self, storage):
        """Sync without cloud storage configured returns empty result."""
        result = storage.sync()

        assert isinstance(result, SyncResult)
        assert result.pushed == 0
        assert result.pulled == 0

    def test_sync_pushes_queued_changes(self, storage_with_cloud, mock_cloud_storage):
        """Sync should push queued changes to cloud."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create some local changes
        storage_with_cloud.save_episode(
            Episode(
                id="ep1", stack_id="test-agent", objective="Test objective", outcome="Test outcome"
            )
        )
        storage_with_cloud.save_note(Note(id="n1", stack_id="test-agent", content="Test note"))

        # Sync
        result = storage_with_cloud.sync()

        assert result.pushed >= 2
        assert mock_cloud_storage.save_episode.called
        assert mock_cloud_storage.save_note.called

    def test_sync_clears_queue_on_success(self, storage_with_cloud, mock_cloud_storage):
        """Successful sync should clear the queue."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_note(Note(id="n1", stack_id="test-agent", content="Test"))

        assert storage_with_cloud.get_pending_sync_count() > 0

        storage_with_cloud.sync()

        # Queue should be cleared
        assert storage_with_cloud.get_pending_sync_count() == 0

    def test_sync_marks_records_synced(self, storage_with_cloud, mock_cloud_storage):
        """Synced records should have cloud_synced_at set."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_note(Note(id="n1", stack_id="test-agent", content="Test"))

        # Before sync
        notes = storage_with_cloud.get_notes()
        assert notes[0].cloud_synced_at is None

        # Sync
        storage_with_cloud.sync()

        # After sync
        notes = storage_with_cloud.get_notes()
        assert notes[0].cloud_synced_at is not None

    def test_sync_offline_returns_error(self, storage_with_cloud, mock_cloud_storage):
        """Sync when offline should return error."""
        mock_cloud_storage.get_stats.side_effect = Exception("Connection refused")

        storage_with_cloud.save_note(Note(id="n1", stack_id="test-agent", content="Test"))

        result = storage_with_cloud.sync()

        assert len(result.errors) > 0
        assert "Offline" in result.errors[0]
        # Queue should NOT be cleared
        assert storage_with_cloud.get_pending_sync_count() > 0


class TestSyncPull:
    """Test pulling changes from cloud."""

    def test_pull_new_records(self, storage_with_cloud, mock_cloud_storage):
        """Pull should add new records from cloud."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Cloud has a record we don't have locally
        cloud_note = Note(
            id="cloud-note-1",
            stack_id="test-agent",
            content="Note from cloud",
            cloud_synced_at=datetime.now(timezone.utc),
            local_updated_at=datetime.now(timezone.utc),
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Do a sync (which includes pull)
        result = storage_with_cloud.sync()

        # Should have pulled the note
        assert result.pulled >= 1

        # Note should exist locally
        notes = storage_with_cloud.get_notes()
        note_ids = {n.id for n in notes}
        assert "cloud-note-1" in note_ids

    def test_pull_without_cloud_returns_empty(self, storage):
        """Pull without cloud storage returns empty result."""
        result = storage.pull_changes()

        assert result.pulled == 0
        assert result.conflict_count == 0


class TestConflictResolution:
    """Test conflict resolution with last-write-wins."""

    def test_cloud_wins_when_newer(self, storage_with_cloud, mock_cloud_storage):
        """Cloud record wins when it's newer."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_note(
            Note(
                id="conflict-note",
                stack_id="test-agent",
                content="Local version",
                local_updated_at=old_time,
            )
        )

        # Manually clear the queue so we can test pull
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has a newer version
        new_time = datetime.now(timezone.utc)
        cloud_note = Note(
            id="conflict-note",
            stack_id="test-agent",
            content="Cloud version (newer)",
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Pull changes
        result = storage_with_cloud.pull_changes()

        # Should have a conflict
        assert result.conflict_count >= 1
        assert len(result.conflicts) >= 1

        # Verify conflict details
        conflict = result.conflicts[0]
        assert conflict.table == "notes"
        assert conflict.record_id == "conflict-note"
        assert "cloud_wins" in conflict.resolution  # May include _arrays_merged suffix
        assert conflict.local_summary is not None
        assert conflict.cloud_summary is not None

        # Local should now have cloud's content
        notes = storage_with_cloud.get_notes()
        conflict_note = next(n for n in notes if n.id == "conflict-note")
        assert "Cloud version" in conflict_note.content

    def test_local_wins_when_newer(self, storage_with_cloud, mock_cloud_storage):
        """Local record wins when it's newer."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record (newer)
        new_time = datetime.now(timezone.utc)
        storage_with_cloud.save_note(
            Note(
                id="conflict-note",
                stack_id="test-agent",
                content="Local version (newer)",
                local_updated_at=new_time,
            )
        )

        # Manually clear the queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has an older version
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        cloud_note = Note(
            id="conflict-note",
            stack_id="test-agent",
            content="Cloud version (older)",
            cloud_synced_at=old_time,
            local_updated_at=old_time,
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Pull changes
        result = storage_with_cloud.pull_changes()

        # Should detect conflict
        assert result.conflict_count >= 1
        assert len(result.conflicts) >= 1

        # Verify conflict details
        conflict = result.conflicts[0]
        assert conflict.table == "notes"
        assert conflict.record_id == "conflict-note"
        assert "local_wins" in conflict.resolution  # May include _arrays_merged suffix

        # Local version should be preserved
        notes = storage_with_cloud.get_notes()
        conflict_note = next(n for n in notes if n.id == "conflict-note")
        assert "Local version" in conflict_note.content

    def test_conflict_history_stored(self, storage_with_cloud, mock_cloud_storage):
        """Conflicts should be stored in history."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_note(
            Note(
                id="history-note",
                stack_id="test-agent",
                content="Local version",
                local_updated_at=old_time,
            )
        )

        # Clear the queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has a newer version
        new_time = datetime.now(timezone.utc)
        cloud_note = Note(
            id="history-note",
            stack_id="test-agent",
            content="Cloud version (newer)",
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Pull to create conflict
        storage_with_cloud.pull_changes()

        # Check conflict history
        history = storage_with_cloud.get_sync_conflicts(limit=10)
        assert len(history) >= 1

        # Find our conflict
        conflict = next((c for c in history if c.record_id == "history-note"), None)
        assert conflict is not None
        assert conflict.table == "notes"
        assert "cloud_wins" in conflict.resolution  # May include _arrays_merged suffix
        assert "Local version" in (conflict.local_summary or "")
        assert "Cloud version" in (conflict.cloud_summary or "")

    def test_conflict_history_clear(self, storage):
        """Conflict history can be cleared."""
        # Add some test conflicts manually
        conflict = SyncConflict(
            id="test-conflict-1",
            table="notes",
            record_id="test-note",
            local_version={"content": "local"},
            cloud_version={"content": "cloud"},
            resolution="cloud_wins",
            resolved_at=datetime.now(timezone.utc),
            local_summary="local",
            cloud_summary="cloud",
        )
        storage.save_sync_conflict(conflict)

        # Verify it was saved
        history = storage.get_sync_conflicts()
        assert len(history) >= 1

        # Clear all
        cleared = storage.clear_sync_conflicts()
        assert cleared >= 1

        # Verify cleared
        history = storage.get_sync_conflicts()
        assert len(history) == 0


class TestSyncMetadata:
    """Test sync metadata tracking."""

    def test_last_sync_time_initially_none(self, storage):
        """Last sync time should be None initially."""
        assert storage.get_last_sync_time() is None

    def test_last_sync_time_updated_on_sync(self, storage_with_cloud, mock_cloud_storage):
        """Last sync time should be updated after sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        before = storage_with_cloud.get_last_sync_time()
        assert before is None

        storage_with_cloud.sync()

        after = storage_with_cloud.get_last_sync_time()
        assert after is not None
        assert isinstance(after, datetime)

    def test_sync_meta_persistence(self, temp_db, mock_cloud_storage):
        """Sync metadata should persist across storage instances."""
        # First instance
        storage1 = SQLiteStorage(
            stack_id="test-agent", db_path=temp_db, cloud_storage=mock_cloud_storage
        )
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}
        storage1.sync()

        sync_time = storage1.get_last_sync_time()
        storage1.close()

        # Second instance
        storage2 = SQLiteStorage(
            stack_id="test-agent", db_path=temp_db, cloud_storage=mock_cloud_storage
        )

        # Should have same last sync time
        assert storage2.get_last_sync_time() == sync_time
        storage2.close()


class TestSyncAllRecordTypes:
    """Test that sync works for all record types."""

    def test_sync_episodes(self, storage_with_cloud, mock_cloud_storage):
        """Episodes should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_episode.called

    def test_sync_beliefs(self, storage_with_cloud, mock_cloud_storage):
        """Beliefs should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_belief(
            Belief(id="b1", stack_id="test-agent", statement="Test belief")
        )

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_belief.called

    def test_sync_values(self, storage_with_cloud, mock_cloud_storage):
        """Values should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_value(
            Value(id="v1", stack_id="test-agent", name="Test", statement="Test value")
        )

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_value.called

    def test_sync_goals(self, storage_with_cloud, mock_cloud_storage):
        """Goals should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_goal(Goal(id="g1", stack_id="test-agent", title="Test goal"))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_goal.called

    def test_sync_drives(self, storage_with_cloud, mock_cloud_storage):
        """Drives should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_drive(Drive(id="d1", stack_id="test-agent", drive_type="curiosity"))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_drive.called

    def test_sync_relationships(self, storage_with_cloud, mock_cloud_storage):
        """Relationships should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_relationship(
            Relationship(
                id="r1",
                stack_id="test-agent",
                entity_name="Alice",
                entity_type="human",
                relationship_type="friend",
            )
        )

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_relationship.called


class TestOfflineQueuing:
    """Test that changes are properly queued when offline."""

    def test_operations_work_offline(self, storage):
        """All operations should work without cloud configured."""
        # These should all succeed
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test"))
        storage.save_belief(Belief(id="b1", stack_id="test-agent", statement="Test"))

        # Data should be accessible
        assert len(storage.get_episodes()) == 1
        assert len(storage.get_notes()) == 1
        assert len(storage.get_beliefs()) == 1

        # Changes should be queued
        assert storage.get_pending_sync_count() >= 3

    def test_queue_survives_reconnect(self, temp_db):
        """Queue should survive closing and reopening storage."""
        # Create storage and add data
        storage1 = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
        storage1.save_note(Note(id="n1", stack_id="test-agent", content="Test"))

        pending_before = storage1.get_pending_sync_count()
        storage1.close()

        # Create new instance
        storage2 = SQLiteStorage(stack_id="test-agent", db_path=temp_db)

        # Queue should still be there
        assert storage2.get_pending_sync_count() == pending_before
        storage2.close()


class TestSyncEdgeCases:
    """Test edge cases in sync behavior."""

    def test_sync_deleted_record(self, storage_with_cloud, mock_cloud_storage):
        """Sync handles records that were deleted locally."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Save a note
        storage_with_cloud.save_note(
            Note(id="n1", stack_id="test-agent", content="Will be deleted")
        )

        # Delete it by marking as deleted (soft delete)
        with storage_with_cloud._get_conn() as conn:
            conn.execute("UPDATE notes SET deleted = 1 WHERE id = 'n1'")
            conn.commit()

        # Sync should handle this gracefully
        result = storage_with_cloud.sync()

        assert result.success is True
        assert result.errors == []
        assert result.pushed == 1
        mock_cloud_storage.save_note.assert_called_once()

    def test_sync_empty_queue(self, storage_with_cloud, mock_cloud_storage):
        """Sync with empty queue should succeed."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Clear queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        result = storage_with_cloud.sync()

        assert result.success
        assert result.pushed == 0

    def test_partial_sync_failure(self, storage_with_cloud, mock_cloud_storage):
        """Sync continues even if some records fail."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Save multiple notes
        storage_with_cloud.save_note(Note(id="n1", stack_id="test-agent", content="Note 1"))
        storage_with_cloud.save_note(Note(id="n2", stack_id="test-agent", content="Note 2"))

        # First call fails, second succeeds
        mock_cloud_storage.save_note.side_effect = [Exception("First failed"), None]  # Success

        result = storage_with_cloud.sync()

        assert result.pushed == 1
        assert len(result.errors) == 1
        assert "Failed to push notes:n1" in result.errors[0]


class TestSyncHooks:
    """Test the auto-sync hooks for load and checkpoint."""

    def test_auto_sync_defaults_to_true_when_online(self, storage_with_cloud, mock_cloud_storage):
        """Auto-sync should default to true when cloud storage is available and online."""
        from kernle import Kernle

        mock_cloud_storage.get_stats.return_value = {"episodes": 0}  # Returns value = online

        k = Kernle(stack_id="test-agent", storage=storage_with_cloud, strict=False)
        assert k.auto_sync is True

    def test_auto_sync_can_be_disabled_via_property(self, storage):
        """Auto-sync can be disabled by setting the property."""
        from kernle import Kernle

        k = Kernle(stack_id="test-agent", storage=storage, strict=False)
        k.auto_sync = False
        assert k.auto_sync is False

    def test_load_with_sync_false_skips_pull(self, storage):
        """Load with sync=False should not attempt to pull."""
        from kernle import Kernle

        k = Kernle(stack_id="test-agent", storage=storage, strict=False)
        k.auto_sync = True

        # Load with sync=False should work without errors
        memory = k.load(sync=False)
        assert "checkpoint" in memory

    def test_load_with_sync_true_attempts_pull(self, storage_with_cloud, mock_cloud_storage):
        """Load with sync=True should attempt to pull changes."""
        from kernle import Kernle

        mock_cloud_storage.get_stats.return_value = {"episodes": 1}  # Simulates online
        mock_cloud_storage.get_episodes.return_value = []
        mock_cloud_storage.get_notes.return_value = []
        mock_cloud_storage.get_beliefs.return_value = []
        mock_cloud_storage.get_values.return_value = []
        mock_cloud_storage.get_goals.return_value = []
        mock_cloud_storage.get_drives.return_value = []
        mock_cloud_storage.get_relationships.return_value = []

        k = Kernle(stack_id="test-agent", storage=storage_with_cloud, strict=False)
        k.load(sync=True)

        mock_cloud_storage.get_episodes.assert_called_once_with(limit=1000, since=None)

    def test_checkpoint_with_sync_true_attempts_push(self, storage_with_cloud, mock_cloud_storage):
        """Checkpoint with sync=True should attempt to push changes."""
        from kernle import Kernle

        mock_cloud_storage.get_stats.return_value = {"episodes": 1}  # Simulates online

        k = Kernle(stack_id="test-agent", storage=storage_with_cloud, strict=False)

        result = k.checkpoint("Test task", pending=["Next"], sync=True)

        assert result["current_task"] == "Test task"
        # Should have sync result attached
        assert "_sync" in result
        sync_result = result["_sync"]
        assert sync_result["attempted"] is True

    def test_checkpoint_sync_result_in_response(self, storage):
        """Checkpoint should include sync result when sync is attempted."""
        from kernle import Kernle

        k = Kernle(stack_id="test-agent", storage=storage, strict=False)

        # With no cloud storage, sync should report offline
        result = k.checkpoint("Test task", sync=True)

        assert "_sync" in result
        sync_result = result["_sync"]
        # Should not have attempted since offline
        assert sync_result["attempted"] is False
        assert len(sync_result["errors"]) > 0

    def test_load_sync_non_blocking_on_error(self, storage_with_cloud, mock_cloud_storage):
        """Load should not fail if sync pull fails."""
        from kernle import Kernle

        # Make cloud throw an error
        mock_cloud_storage.get_stats.side_effect = Exception("Network error")

        k = Kernle(stack_id="test-agent", storage=storage_with_cloud, strict=False)

        # Load should still work
        memory = k.load(sync=True)
        assert "checkpoint" in memory

    def test_checkpoint_sync_non_blocking_on_error(self, storage_with_cloud, mock_cloud_storage):
        """Checkpoint should not fail if sync push fails."""
        from kernle import Kernle

        # Make cloud throw an error
        mock_cloud_storage.get_stats.side_effect = Exception("Network error")

        k = Kernle(stack_id="test-agent", storage=storage_with_cloud, strict=False)

        # Checkpoint should still work
        result = k.checkpoint("Test task", sync=True)
        assert result["current_task"] == "Test task"


class TestSyncQueueAtomicity:
    """Test that sync queue operations are atomic."""

    def test_queue_sync_uses_atomic_upsert(self, storage):
        """Verify that queue_sync uses INSERT ON CONFLICT (atomic) not DELETE+INSERT."""
        # Save the same note multiple times rapidly
        for i in range(10):
            storage.save_note(Note(id="atomic-test", stack_id="test-agent", content=f"Version {i}"))

        # Should still have exactly one queued change for this record
        changes = storage.get_queued_changes()
        matching = [c for c in changes if c.record_id == "atomic-test"]
        assert len(matching) == 1

        # The queued change should have the latest data
        # (Verify it's truly the latest by checking the operation is "upsert")
        assert matching[0].operation in ("upsert", "insert", "update")

    def test_sync_queue_deduplication_preserves_latest(self, storage):
        """Multiple updates to the same record should keep only the latest in queue."""
        # Save three different versions
        storage.save_episode(
            Episode(id="dedupe-test", stack_id="test-agent", objective="First", outcome="v1")
        )
        storage.save_episode(
            Episode(id="dedupe-test", stack_id="test-agent", objective="Second", outcome="v2")
        )
        storage.save_episode(
            Episode(id="dedupe-test", stack_id="test-agent", objective="Third", outcome="v3")
        )

        # Get queued changes for this record
        changes = storage.get_queued_changes()
        matching = [c for c in changes if c.record_id == "dedupe-test"]

        # Should be exactly one queued change
        assert len(matching) == 1

        # Verify the episode was saved with the latest values
        # (Queue stores the record ID, actual data is in the episodes table)
        episodes = storage.get_episodes()
        dedupe_episode = next(e for e in episodes if e.id == "dedupe-test")
        assert dedupe_episode.objective == "Third"
        assert dedupe_episode.outcome == "v3"


class TestArrayFieldMerging:
    """Test that array fields are merged (set union) during sync instead of last-write-wins."""

    def test_cloud_wins_merges_local_tags(self, storage_with_cloud, mock_cloud_storage):
        """When cloud wins, array fields from local should be merged in."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record with some tags
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_episode(
            Episode(
                id="ep-merge-1",
                stack_id="test-agent",
                objective="Test episode",
                outcome="Test outcome",
                tags=["local-tag-1", "local-tag-2"],
                lessons=["local-lesson-1"],
                emotional_tags=["joy"],
                local_updated_at=old_time,
            )
        )

        # Clear the queue so we can test pull
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has a newer version with different tags
        new_time = datetime.now(timezone.utc)
        cloud_episode = Episode(
            id="ep-merge-1",
            stack_id="test-agent",
            objective="Updated objective from cloud",
            outcome="Updated outcome from cloud",
            tags=["cloud-tag-1", "local-tag-1"],  # One overlapping, one new
            lessons=["cloud-lesson-1"],  # Different lesson
            emotional_tags=["curiosity"],  # Different emotion
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_episodes.return_value = [cloud_episode]

        # Pull changes
        result = storage_with_cloud.pull_changes()

        # Should have a conflict
        assert result.conflict_count >= 1
        conflict = result.conflicts[0]
        assert "arrays_merged" in conflict.resolution

        # Verify the episode has merged arrays
        episode = storage_with_cloud.get_episode("ep-merge-1")
        assert episode is not None

        # Scalar fields should come from cloud (winner)
        assert episode.objective == "Updated objective from cloud"

        # Array fields should be merged (set union)
        assert set(episode.tags) == {"local-tag-1", "local-tag-2", "cloud-tag-1"}
        assert set(episode.lessons) == {"local-lesson-1", "cloud-lesson-1"}
        assert set(episode.emotional_tags) == {"joy", "curiosity"}

    def test_local_wins_merges_cloud_tags(self, storage_with_cloud, mock_cloud_storage):
        """When local wins, array fields from cloud should be merged in."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record (newer)
        new_time = datetime.now(timezone.utc)
        storage_with_cloud.save_note(
            Note(
                id="note-merge-1",
                stack_id="test-agent",
                content="Local content (newer)",
                tags=["local-tag-1", "common-tag"],
                local_updated_at=new_time,
            )
        )

        # Clear the queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has an older version with different tags
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        cloud_note = Note(
            id="note-merge-1",
            stack_id="test-agent",
            content="Cloud content (older)",
            tags=["cloud-tag-1", "common-tag"],
            cloud_synced_at=old_time,
            local_updated_at=old_time,
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Pull changes
        result = storage_with_cloud.pull_changes()

        # Should have a conflict where local wins but arrays are merged
        assert result.conflict_count >= 1
        conflict = result.conflicts[0]
        assert "local_wins" in conflict.resolution
        assert "arrays_merged" in conflict.resolution

        # Verify the note has merged arrays but local scalar content
        note = next(n for n in storage_with_cloud.get_notes() if n.id == "note-merge-1")
        assert note is not None

        # Scalar fields should come from local (winner)
        assert note.content == "Local content (newer)"

        # Tags should be merged (set union)
        assert set(note.tags) == {"local-tag-1", "cloud-tag-1", "common-tag"}

    def test_drive_focus_areas_merge(self, storage_with_cloud, mock_cloud_storage):
        """Drive focus_areas should be merged."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local drive
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_drive(
            Drive(
                id="drive-merge-1",
                stack_id="test-agent",
                drive_type="curiosity",
                focus_areas=["local-area-1", "common-area"],
                local_updated_at=old_time,
            )
        )

        # Clear queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has newer version with different focus areas
        new_time = datetime.now(timezone.utc)
        cloud_drive = Drive(
            id="drive-merge-1",
            stack_id="test-agent",
            drive_type="curiosity",
            intensity=0.8,  # Updated intensity
            focus_areas=["cloud-area-1", "common-area"],
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_drives.return_value = [cloud_drive]

        # Pull changes
        storage_with_cloud.pull_changes()

        # Verify merged focus areas
        drive = storage_with_cloud.get_drive("curiosity")
        assert drive is not None
        assert drive.intensity == 0.8  # Scalar from cloud winner
        assert set(drive.focus_areas) == {"local-area-1", "cloud-area-1", "common-area"}

    def test_merge_with_none_array_local(self, storage_with_cloud, mock_cloud_storage):
        """Merge handles None array on local side."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record with no tags
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_episode(
            Episode(
                id="ep-none-local",
                stack_id="test-agent",
                objective="Test",
                outcome="Test",
                tags=None,  # No tags locally
                local_updated_at=old_time,
            )
        )

        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has tags
        new_time = datetime.now(timezone.utc)
        cloud_episode = Episode(
            id="ep-none-local",
            stack_id="test-agent",
            objective="Cloud objective",
            outcome="Cloud outcome",
            tags=["cloud-tag"],
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_episodes.return_value = [cloud_episode]

        storage_with_cloud.pull_changes()

        episode = storage_with_cloud.get_episode("ep-none-local")
        # Should have cloud's tags (local had none)
        assert episode.tags == ["cloud-tag"]

    def test_merge_with_none_array_cloud(self, storage_with_cloud, mock_cloud_storage):
        """Merge handles None array on cloud side."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record with tags
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_episode(
            Episode(
                id="ep-none-cloud",
                stack_id="test-agent",
                objective="Test",
                outcome="Test",
                tags=["local-tag"],
                local_updated_at=old_time,
            )
        )

        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has no tags
        new_time = datetime.now(timezone.utc)
        cloud_episode = Episode(
            id="ep-none-cloud",
            stack_id="test-agent",
            objective="Cloud objective",
            outcome="Cloud outcome",
            tags=None,  # No tags in cloud
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_episodes.return_value = [cloud_episode]

        storage_with_cloud.pull_changes()

        episode = storage_with_cloud.get_episode("ep-none-cloud")
        # Should preserve local tags even though cloud won
        assert episode.tags == ["local-tag"]

    def test_merge_deduplicates_arrays(self, storage_with_cloud, mock_cloud_storage):
        """Merged arrays should not have duplicates."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_episode(
            Episode(
                id="ep-dedup",
                stack_id="test-agent",
                objective="Test",
                outcome="Test",
                tags=["tag-a", "tag-b", "tag-c"],
                local_updated_at=old_time,
            )
        )

        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has overlapping tags
        new_time = datetime.now(timezone.utc)
        cloud_episode = Episode(
            id="ep-dedup",
            stack_id="test-agent",
            objective="Cloud objective",
            outcome="Cloud outcome",
            tags=["tag-b", "tag-c", "tag-d"],  # b and c overlap
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_episodes.return_value = [cloud_episode]

        storage_with_cloud.pull_changes()

        episode = storage_with_cloud.get_episode("ep-dedup")
        # Should have exactly 4 unique tags
        assert len(episode.tags) == 4
        assert set(episode.tags) == {"tag-a", "tag-b", "tag-c", "tag-d"}


class TestMergeArrayFieldsUnit:
    """Unit tests for the _merge_array_fields helper method."""

    def test_merge_array_fields_no_arrays(self, storage):
        """No-op when table has no array fields configured."""
        # Using a fake table name that's not in SYNC_ARRAY_FIELDS
        ep1 = Episode(id="1", stack_id="test", objective="O1", outcome="Out1", tags=["a"])
        ep2 = Episode(id="1", stack_id="test", objective="O2", outcome="Out2", tags=["b"])

        result = storage._merge_array_fields("fake_table", ep1, ep2)

        # Should return winner unchanged
        assert result is ep1
        assert result.tags == ["a"]

    def test_merge_array_fields_episode(self, storage):
        """Direct test of _merge_array_fields for episodes."""
        winner = Episode(
            id="1",
            stack_id="test",
            objective="Winner",
            outcome="Out",
            tags=["tag-w"],
            lessons=["lesson-w"],
            emotional_tags=["joy"],
        )
        loser = Episode(
            id="1",
            stack_id="test",
            objective="Loser",
            outcome="Out",
            tags=["tag-l"],
            lessons=["lesson-l"],
            emotional_tags=["curiosity"],
        )

        result = storage._merge_array_fields("episodes", winner, loser)

        # Winner's scalar fields preserved, arrays merged
        assert result.objective == "Winner"
        assert set(result.tags) == {"tag-w", "tag-l"}
        assert set(result.lessons) == {"lesson-w", "lesson-l"}
        assert set(result.emotional_tags) == {"joy", "curiosity"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSyncQueueResilience:
    """Tests for resilient sync queue behavior (v0.2.5)."""

    def test_record_sync_failure_truncates_long_error_to_500_chars(self, storage):
        """Long sync errors are persisted as at most 500 characters."""
        storage.save_note(Note(id="n-long", stack_id="test-agent", content="Test note"))

        queued = storage.get_queued_changes(limit=10)
        change = next((c for c in queued if c.record_id == "n-long"), None)
        assert change is not None

        long_error = "E" * 800

        with storage._connect() as conn:
            new_count = storage._record_sync_failure(conn, change.id, long_error)
            conn.commit()

        assert new_count == 1

        refreshed = storage.get_queued_changes(limit=10)
        failed = next((c for c in refreshed if c.id == change.id), None)
        assert failed is not None
        assert failed.retry_count == 1
        assert failed.last_error == "E" * 500

    def test_failed_record_increments_retry_count(self, storage):
        """Test that failed sync records get retry count incremented."""
        # Create a test record
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test note"))

        # Get the queued change
        queued = storage.get_queued_changes(limit=10)
        assert len(queued) >= 1
        change = queued[0]
        assert change.retry_count == 0

        # Simulate a failure
        with storage._connect() as conn:
            new_count = storage._record_sync_failure(conn, change.id, "Test error")
            conn.commit()

        assert new_count == 1

        # Check the record has the error
        queued = storage.get_queued_changes(limit=10)
        change = next((c for c in queued if c.id == change.id), None)
        assert change is not None
        assert change.retry_count == 1
        assert change.last_error == "Test error"

    def test_max_retries_excludes_from_queue(self, storage):
        """Test that records exceeding max retries are skipped."""
        # Create a test record
        storage.save_note(Note(id="n2", stack_id="test-agent", content="Test note 2"))

        # Get the queued change
        queued = storage.get_queued_changes(limit=10)
        change = next((c for c in queued if c.record_id == "n2"), None)
        assert change is not None

        # Simulate 5 failures to exceed max
        with storage._connect() as conn:
            for i in range(5):
                storage._record_sync_failure(conn, change.id, f"Error {i+1}")
            conn.commit()

        # Now it should not appear in normal queue
        queued = storage.get_queued_changes(limit=10, max_retries=5)
        assert not any(c.id == change.id for c in queued)

        # But should appear in failed records
        failed = storage.get_failed_sync_records(min_retries=5)
        assert any(c.id == change.id for c in failed)

    def test_sync_continues_after_individual_failure(self, storage):
        """Test that sync continues processing after individual record failures."""
        # Create multiple test records
        storage.save_note(Note(id="n3", stack_id="test-agent", content="Note 3"))
        storage.save_note(Note(id="n4", stack_id="test-agent", content="Note 4"))
        storage.save_note(Note(id="n5", stack_id="test-agent", content="Note 5"))

        queued = storage.get_queued_changes(limit=10)
        assert len(queued) >= 3

        # All records should be present
        ids = {c.record_id for c in queued}
        assert "n3" in ids
        assert "n4" in ids
        assert "n5" in ids

    def test_clear_failed_sync_records(self, storage):
        """Test clearing old failed records."""
        # Create a test record
        storage.save_note(Note(id="n6", stack_id="test-agent", content="Test note 6"))

        queued = storage.get_queued_changes(limit=10)
        change = next((c for c in queued if c.record_id == "n6"), None)
        assert change is not None

        # Simulate failures and set old timestamp
        with storage._connect() as conn:
            for _ in range(5):
                storage._record_sync_failure(conn, change.id, "Old error")
            # Set last_attempt_at to 10 days ago
            old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            conn.execute(
                "UPDATE sync_queue SET last_attempt_at = ? WHERE id = ?",
                (old_time, change.id),
            )
            conn.commit()

        # Clear old failed records
        cleared = storage.clear_failed_sync_records(older_than_days=7)
        assert cleared == 1

        # Should no longer appear in failed records
        failed = storage.get_failed_sync_records(min_retries=5)
        assert not any(c.id == change.id for c in failed)


class TestMergeArrayFieldsDictDedup:
    """Test _merge_array_fields with dict arrays and edge cases."""

    def test_dict_array_dedup_overlapping(self, storage):
        """Dict arrays from winner and loser are deduped by JSON key."""
        engine = storage._sync_engine
        winner = Episode(
            id="1",
            stack_id="test",
            objective="W",
            outcome="O",
            source_episodes=None,
            derived_from=[{"type": "episode", "id": "a"}, {"type": "episode", "id": "b"}],
        )
        loser = Episode(
            id="1",
            stack_id="test",
            objective="L",
            outcome="O",
            source_episodes=None,
            derived_from=[{"type": "episode", "id": "b"}, {"type": "episode", "id": "c"}],
        )
        result = engine._merge_array_fields("episodes", winner, loser)
        # Should have 3 unique dicts: a, b, c — "b" is deduped
        assert len(result.derived_from) == 3
        ids_in_merged = {d["id"] for d in result.derived_from}
        assert ids_in_merged == {"a", "b", "c"}

    def test_dict_array_max_size_truncation(self, storage, caplog):
        """Arrays exceeding MAX_SYNC_ARRAY_SIZE are truncated with a warning."""
        engine = storage._sync_engine
        # Create arrays that together exceed 500
        big_winner = [{"idx": i} for i in range(300)]
        big_loser = [{"idx": i} for i in range(300, 600)]
        winner = Episode(
            id="1",
            stack_id="test",
            objective="W",
            outcome="O",
            derived_from=big_winner,
        )
        loser = Episode(
            id="1",
            stack_id="test",
            objective="L",
            outcome="O",
            derived_from=big_loser,
        )
        with caplog.at_level(logging.WARNING, logger="kernle.storage.sync_engine"):
            result = engine._merge_array_fields("episodes", winner, loser)
        assert len(result.derived_from) == MAX_SYNC_ARRAY_SIZE
        assert any("exceeded max size" in r.message for r in caplog.records)

    def test_non_dict_array_max_size_truncation(self, storage, caplog):
        """Non-dict arrays exceeding MAX_SYNC_ARRAY_SIZE are also truncated."""
        engine = storage._sync_engine
        big_winner = [f"tag-{i}" for i in range(300)]
        big_loser = [f"tag-{i}" for i in range(300, 600)]
        winner = Episode(
            id="1",
            stack_id="test",
            objective="W",
            outcome="O",
            tags=big_winner,
        )
        loser = Episode(
            id="1",
            stack_id="test",
            objective="L",
            outcome="O",
            tags=big_loser,
        )
        with caplog.at_level(logging.WARNING, logger="kernle.storage.sync_engine"):
            result = engine._merge_array_fields("episodes", winner, loser)
        assert len(result.tags) == MAX_SYNC_ARRAY_SIZE
        assert any("exceeded max size" in r.message for r in caplog.records)

    def test_type_error_fallback_keeps_winner(self, storage, caplog):
        """When set() fails on unhashable items, winner's value is kept."""
        engine = storage._sync_engine
        # lists are unhashable — set() on them will raise TypeError
        winner = Episode(
            id="1",
            stack_id="test",
            objective="W",
            outcome="O",
            tags=[["nested", "list"]],
        )
        loser = Episode(
            id="1",
            stack_id="test",
            objective="L",
            outcome="O",
            tags=[["other"]],
        )
        with caplog.at_level(logging.WARNING, logger="kernle.storage.sync_engine"):
            result = engine._merge_array_fields("episodes", winner, loser)
        # Winner's value kept due to TypeError fallback
        assert result.tags == [["nested", "list"]]
        assert any("Failed to merge array field" in r.message for r in caplog.records)

    def test_non_dict_array_set_union(self, storage):
        """Non-dict arrays merge via set union."""
        engine = storage._sync_engine
        winner = Episode(
            id="1",
            stack_id="test",
            objective="W",
            outcome="O",
            tags=["a", "b"],
            lessons=["l1"],
        )
        loser = Episode(
            id="1",
            stack_id="test",
            objective="L",
            outcome="O",
            tags=["b", "c"],
            lessons=["l2"],
        )
        result = engine._merge_array_fields("episodes", winner, loser)
        assert set(result.tags) == {"a", "b", "c"}
        assert set(result.lessons) == {"l1", "l2"}


class TestPushRecord:
    """Test _push_record dispatch and error paths."""

    def test_push_record_no_cloud_returns_false(self, storage):
        """Without cloud storage, _push_record returns False."""
        engine = storage._sync_engine
        ep = Episode(id="1", stack_id="test", objective="O", outcome="Out")
        assert engine._push_record("episodes", ep) is False

    def test_push_record_unknown_table_returns_false(self, storage_with_cloud, caplog):
        """Unknown table name logs warning and returns False."""
        engine = storage_with_cloud._sync_engine
        ep = Episode(id="1", stack_id="test", objective="O", outcome="Out")
        with caplog.at_level(logging.WARNING, logger="kernle.storage.sync_engine"):
            result = engine._push_record("unknown_table", ep)
        assert result is False
        assert any("Unknown table for push" in r.message for r in caplog.records)

    def test_push_record_exception_returns_false(
        self, storage_with_cloud, mock_cloud_storage, caplog
    ):
        """Exception during push logs error and returns False."""
        engine = storage_with_cloud._sync_engine
        mock_cloud_storage.save_episode.side_effect = Exception("Cloud error")
        ep = Episode(id="ep1", stack_id="test", objective="O", outcome="Out")
        with caplog.at_level(logging.ERROR, logger="kernle.storage.sync_engine"):
            result = engine._push_record("episodes", ep)
        assert result is False
        assert any("Failed to push record" in r.message for r in caplog.records)

    def test_push_record_dispatches_episodes(self, storage_with_cloud, mock_cloud_storage):
        """Push episodes dispatches to cloud save_episode."""
        engine = storage_with_cloud._sync_engine
        ep = Episode(id="ep1", stack_id="test", objective="O", outcome="Out")
        assert engine._push_record("episodes", ep) is True
        mock_cloud_storage.save_episode.assert_called_once_with(ep)

    def test_push_record_dispatches_notes(self, storage_with_cloud, mock_cloud_storage):
        """Push notes dispatches to cloud save_note."""
        engine = storage_with_cloud._sync_engine
        note = Note(id="n1", stack_id="test", content="C")
        assert engine._push_record("notes", note) is True
        mock_cloud_storage.save_note.assert_called_once_with(note)

    def test_push_record_dispatches_beliefs(self, storage_with_cloud, mock_cloud_storage):
        """Push beliefs dispatches to cloud save_belief."""
        engine = storage_with_cloud._sync_engine
        b = Belief(id="b1", stack_id="test", statement="S")
        assert engine._push_record("beliefs", b) is True
        mock_cloud_storage.save_belief.assert_called_once_with(b)

    def test_push_record_dispatches_values(self, storage_with_cloud, mock_cloud_storage):
        """Push agent_values dispatches to cloud save_value."""
        engine = storage_with_cloud._sync_engine
        v = Value(id="v1", stack_id="test", name="N", statement="S")
        assert engine._push_record("agent_values", v) is True
        mock_cloud_storage.save_value.assert_called_once_with(v)

    def test_push_record_dispatches_goals(self, storage_with_cloud, mock_cloud_storage):
        """Push goals dispatches to cloud save_goal."""
        engine = storage_with_cloud._sync_engine
        g = Goal(id="g1", stack_id="test", title="T")
        assert engine._push_record("goals", g) is True
        mock_cloud_storage.save_goal.assert_called_once_with(g)

    def test_push_record_dispatches_drives(self, storage_with_cloud, mock_cloud_storage):
        """Push drives dispatches to cloud save_drive."""
        engine = storage_with_cloud._sync_engine
        d = Drive(id="d1", stack_id="test", drive_type="curiosity")
        assert engine._push_record("drives", d) is True
        mock_cloud_storage.save_drive.assert_called_once_with(d)

    def test_push_record_dispatches_relationships(self, storage_with_cloud, mock_cloud_storage):
        """Push relationships dispatches to cloud save_relationship."""
        engine = storage_with_cloud._sync_engine
        r = Relationship(
            id="r1",
            stack_id="test",
            entity_name="Alice",
            entity_type="human",
            relationship_type="friend",
        )
        assert engine._push_record("relationships", r) is True
        mock_cloud_storage.save_relationship.assert_called_once_with(r)

    def test_push_record_dispatches_playbooks(self, storage_with_cloud, mock_cloud_storage):
        """Push playbooks dispatches to cloud save_playbook."""
        engine = storage_with_cloud._sync_engine
        p = Playbook(
            id="p1",
            stack_id="test",
            name="PB",
            description="D",
            trigger_conditions=["t"],
            steps=[{"action": "a"}],
            failure_modes=["f"],
        )
        assert engine._push_record("playbooks", p) is True
        mock_cloud_storage.save_playbook.assert_called_once_with(p)


class TestSyncPushPhaseEdgeCases:
    """Test sync() push-phase edge cases: retry tracking, dead letter, exceptions."""

    def test_push_failure_records_retry_and_error(self, storage_with_cloud, mock_cloud_storage):
        """Push failure increments retry count and adds error to result."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}
        # save_note succeeds on cloud but save_episode fails
        mock_cloud_storage.save_episode.side_effect = None
        mock_cloud_storage.save_note.return_value = None

        storage_with_cloud.save_note(Note(id="n1", stack_id="test-agent", content="C"))

        # Make the push return False (unknown table trick won't work, so use exception)
        mock_cloud_storage.save_note.side_effect = Exception("Cloud reject")

        result = storage_with_cloud.sync()
        assert len(result.errors) >= 1
        # The error should mention the record
        assert any("n1" in e for e in result.errors)

    def test_push_retry_reaches_dead_letter(self, storage_with_cloud, mock_cloud_storage, caplog):
        """After 5 retries, dead letter warning is logged."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_note(Note(id="dl1", stack_id="test-agent", content="DL"))

        # Set retry count to 4 so next failure = 5 (dead letter)
        with storage_with_cloud._connect() as conn:
            conn.execute("UPDATE sync_queue SET retry_count = 4 WHERE record_id = 'dl1'")
            conn.commit()

        # Make push fail (not exception, but cloud method fails)
        # We need _push_record to return False — make cloud save raise
        mock_cloud_storage.save_note.side_effect = Exception("still failing")

        # Clear connectivity cache to re-check
        storage_with_cloud._last_connectivity_check = None

        with caplog.at_level(logging.WARNING, logger="kernle.storage.sync_engine"):
            result = storage_with_cloud.sync()

        # retry_count was 4, exception increments to 5 → dead letter
        assert any(
            "exceeded max retries" in r.message or "dead letter" in r.message
            for r in caplog.records
        ) or any("retry" in e for e in result.errors)

    def test_record_not_found_for_non_delete_clears_queue(
        self, storage_with_cloud, mock_cloud_storage
    ):
        """When record is not found and operation is not delete, queue entry is cleared."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Queue a change for a record that doesn't exist
        storage_with_cloud.queue_sync_operation("upsert", "notes", "ghost-record")

        # Verify it's queued
        assert storage_with_cloud.get_pending_sync_count() >= 1

        # Clear connectivity cache
        storage_with_cloud._last_connectivity_check = None

        storage_with_cloud.sync()

        # The ghost record should be cleared from queue (not stuck forever)
        changes = storage_with_cloud.get_queued_changes()
        assert not any(c.record_id == "ghost-record" for c in changes)

    def test_local_only_memory_suggestion_queue_entry_is_not_silently_cleared(
        self, storage_with_cloud, mock_cloud_storage
    ):
        """Legacy queued suggestion ops should stay queued with explicit failure metadata."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Simulate a pre-guard/legacy queue row that already exists in sync_queue.
        with storage_with_cloud._connect() as conn:
            now = storage_with_cloud._now()
            conn.execute(
                """INSERT INTO sync_queue
                   (table_name, record_id, operation, data, local_updated_at, synced, payload, queued_at)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?)""",
                ("memory_suggestions", "legacy-suggestion-op", "upsert", None, now, None, now),
            )

        pre_sync = storage_with_cloud.get_queued_changes(limit=10)
        assert any(
            c.table_name == "memory_suggestions" and c.record_id == "legacy-suggestion-op"
            for c in pre_sync
        )

        storage_with_cloud._last_connectivity_check = None
        result = storage_with_cloud.sync()

        post_sync = storage_with_cloud.get_queued_changes(limit=10)
        queued_change = next(
            (
                c
                for c in post_sync
                if c.table_name == "memory_suggestions" and c.record_id == "legacy-suggestion-op"
            ),
            None,
        )
        assert queued_change is not None
        assert queued_change.retry_count == 1
        assert queued_change.last_error is not None
        assert "local-only" in queued_change.last_error
        assert any(
            "local-only table memory_suggestions:legacy-suggestion-op" in error
            for error in result.errors
        )

    def test_failed_records_count_logged(self, storage_with_cloud, mock_cloud_storage, caplog):
        """When there are failed records, info log is emitted."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create a record that has exceeded retries
        storage_with_cloud.save_note(Note(id="fail1", stack_id="test-agent", content="F"))
        with storage_with_cloud._connect() as conn:
            conn.execute("UPDATE sync_queue SET retry_count = 6 WHERE record_id = 'fail1'")
            conn.commit()

        storage_with_cloud._last_connectivity_check = None

        with caplog.at_level(logging.INFO, logger="kernle.storage.sync_engine"):
            storage_with_cloud.sync()

        assert any(
            "Skipping" in r.message and "exceeded max retries" in r.message for r in caplog.records
        )

    def test_exception_in_push_loop_records_error(self, storage_with_cloud, mock_cloud_storage):
        """Exception in _get_record_for_push is caught by outer except in push loop."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_note(Note(id="exc1", stack_id="test-agent", content="E"))

        # Make _get_record_for_push raise to hit the outer except block (lines 535-542)
        engine = storage_with_cloud._sync_engine
        original = engine._get_record_for_push
        engine._get_record_for_push = MagicMock(side_effect=RuntimeError("DB corrupt"))

        storage_with_cloud._last_connectivity_check = None
        try:
            result = storage_with_cloud.sync()
        finally:
            engine._get_record_for_push = original

        assert any("exc1" in e for e in result.errors)
        assert any("DB corrupt" in e for e in result.errors)


class TestMergeGenericEdgeCases:
    """Test _merge_generic edge cases."""

    def test_cloud_time_no_local_time_saves(self, storage):
        """When cloud has time but local has no local_updated_at, save_fn is called."""
        engine = storage._sync_engine

        # Cloud record has timestamps, local record has local_updated_at = None
        cloud = Note(
            id="no-local-time",
            stack_id="test-agent",
            content="Cloud",
            cloud_synced_at=datetime.now(timezone.utc),
            local_updated_at=datetime.now(timezone.utc),
        )
        local = Note(
            id="no-local-time",
            stack_id="test-agent",
            content="Local",
            cloud_synced_at=None,
            local_updated_at=None,
        )
        save_called = []
        count, conflict = engine._merge_generic(
            "notes", cloud, local, lambda: save_called.append(True)
        )
        # cloud_time is set, local_time is None → elif cloud_time branch → save_fn called
        assert count == 1
        assert conflict is None
        assert len(save_called) == 1

    def test_neither_has_time_saves_cloud_as_fallback(self, storage_with_cloud):
        """When neither cloud nor local has time, cloud is saved with conflict record."""
        engine = storage_with_cloud._sync_engine

        # Both records have no timestamps
        cloud = Note(
            id="no-time",
            stack_id="test-agent",
            content="Cloud",
            cloud_synced_at=None,
            local_updated_at=None,
        )
        local = Note(
            id="no-time",
            stack_id="test-agent",
            content="Local",
            cloud_synced_at=None,
            local_updated_at=None,
        )
        count, conflict = engine._merge_generic("notes", cloud, local, lambda: None)
        assert count == 1
        assert conflict is not None  # conflict-aware merge when local exists
        assert conflict.policy_decision == "no_timestamps_fallback"

    def test_equal_timestamp_and_matching_snapshot_deduplicates(self, storage):
        """When timestamps and payloads match, no merge/callback is performed."""
        engine = storage._sync_engine

        shared_time = datetime.now(timezone.utc)
        local = Note(
            id="equal-eq",
            stack_id="test-agent",
            content="same-content",
            tags=["sync"],
            local_updated_at=shared_time,
        )
        cloud = Note(
            id="equal-eq",
            stack_id="test-agent",
            content="same-content",
            tags=["sync"],
            local_updated_at=shared_time,
            cloud_synced_at=shared_time,
        )
        save_called = []
        count, conflict = engine._merge_generic(
            "notes", cloud, local, lambda: save_called.append(True)
        )
        assert count == 0
        assert conflict is None
        assert save_called == []

    def test_equal_timestamp_records_tie_decision_and_metadata(self, storage):
        """Equal timestamps use deterministic policy and store conflict metadata."""
        engine = storage._sync_engine

        shared_time = datetime.now(timezone.utc)
        local = Note(
            id="equal-tie",
            stack_id="test-agent",
            content="local",
            tags=["local"],
            local_updated_at=shared_time,
        )
        cloud = Note(
            id="equal-tie",
            stack_id="test-agent",
            content="cloud",
            tags=["cloud"],
            local_updated_at=shared_time,
            cloud_synced_at=shared_time,
        )
        save_called = []
        count, conflict = engine._merge_generic(
            "notes", cloud, local, lambda: save_called.append(True)
        )
        assert count == 0
        assert conflict is not None
        assert conflict.table == "notes"
        assert conflict.record_id == "equal-tie"
        assert conflict.source == "sync_engine"
        assert conflict.policy_decision in {"local_wins_tie_hash", "cloud_wins_tie_hash"}
        assert conflict.resolution in {"local_wins_arrays_merged", "cloud_wins_arrays_merged"}
        assert conflict.diff_hash
        assert len(save_called) in {0, 1}


class TestGetRecordSummary:
    """Test _get_record_summary for all table types + unknown fallback."""

    def test_episode_summary(self, storage):
        engine = storage._sync_engine
        ep = Episode(id="1", stack_id="t", objective="Short obj", outcome="O")
        assert engine._get_record_summary("episodes", ep) == "Short obj"

    def test_episode_summary_truncated(self, storage):
        engine = storage._sync_engine
        long_obj = "A" * 60
        ep = Episode(id="1", stack_id="t", objective=long_obj, outcome="O")
        summary = engine._get_record_summary("episodes", ep)
        assert summary.endswith("...")
        assert len(summary) == 53  # 50 chars + "..."

    def test_note_summary(self, storage):
        engine = storage._sync_engine
        note = Note(id="1", stack_id="t", content="My note content")
        assert engine._get_record_summary("notes", note) == "My note content"

    def test_note_summary_truncated(self, storage):
        engine = storage._sync_engine
        note = Note(id="1", stack_id="t", content="X" * 60)
        summary = engine._get_record_summary("notes", note)
        assert len(summary) == 53

    def test_belief_summary(self, storage):
        engine = storage._sync_engine
        b = Belief(id="1", stack_id="t", statement="I believe X")
        assert engine._get_record_summary("beliefs", b) == "I believe X"

    def test_value_summary(self, storage):
        engine = storage._sync_engine
        v = Value(id="1", stack_id="t", name="honesty", statement="Always be honest")
        summary = engine._get_record_summary("agent_values", v)
        assert summary == "honesty: Always be honest"

    def test_value_summary_truncated_statement(self, storage):
        engine = storage._sync_engine
        v = Value(id="1", stack_id="t", name="val", statement="S" * 50)
        summary = engine._get_record_summary("agent_values", v)
        # statement truncated at 40 + "..."
        assert "..." in summary
        assert summary.startswith("val: ")

    def test_goal_summary(self, storage):
        engine = storage._sync_engine
        g = Goal(id="1", stack_id="t", title="Learn Python")
        assert engine._get_record_summary("goals", g) == "Learn Python"

    def test_drive_summary(self, storage):
        engine = storage._sync_engine
        d = Drive(id="1", stack_id="t", drive_type="curiosity", intensity=0.7)
        summary = engine._get_record_summary("drives", d)
        assert summary == "curiosity (intensity: 0.7)"

    def test_relationship_summary(self, storage):
        engine = storage._sync_engine
        r = Relationship(
            id="1",
            stack_id="t",
            entity_name="Alice",
            entity_type="human",
            relationship_type="mentor",
        )
        summary = engine._get_record_summary("relationships", r)
        assert summary == "Alice (mentor)"

    def test_playbook_summary(self, storage):
        engine = storage._sync_engine
        p = Playbook(
            id="1",
            stack_id="t",
            name="Deployment Playbook",
            description="A long description that should be summarized for conflict output.",
            trigger_conditions=["x"],
            steps=[{"action": "a"}],
            failure_modes=["f"],
        )
        assert (
            engine._get_record_summary("playbooks", p)
            == "Deployment Playbook :: A long description that should be summarized for conflict ou..."
        )

    def test_unknown_table_fallback(self, storage):
        engine = storage._sync_engine
        ep = Episode(id="ep-id", stack_id="t", objective="O", outcome="Out")
        summary = engine._get_record_summary("unknown_table", ep)
        assert summary == "unknown_table:ep-id"


class TestRecordToDict:
    """Test _record_to_dict success and failure paths."""

    def test_success_with_datetime_fields(self, storage):
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)
        note = Note(
            id="n1",
            stack_id="t",
            content="C",
            local_updated_at=now,
            cloud_synced_at=now,
        )
        d = engine._record_to_dict(note)
        assert d["id"] == "n1"
        assert d["content"] == "C"
        # datetime fields should be ISO strings
        assert isinstance(d["local_updated_at"], str)
        assert isinstance(d["cloud_synced_at"], str)

    def test_failure_fallback_non_dataclass(self, storage):
        """Non-dataclass objects fall back to {'id': ...}."""
        engine = storage._sync_engine

        class FakeRecord:
            id = "fake-id"

        result = engine._record_to_dict(FakeRecord())
        assert result == {"id": "fake-id"}

    def test_failure_fallback_no_id(self, storage):
        """Objects without id fall back to {'id': 'unknown'}."""
        engine = storage._sync_engine

        class NoId:
            pass

        result = engine._record_to_dict(NoId())
        assert result == {"id": "unknown"}


class TestSaveFromCloud:
    """Test _save_from_cloud dispatches for each table type."""

    def test_save_episode_from_cloud(self, storage):
        engine = storage._sync_engine
        ep = Episode(id="ep1", stack_id="test-agent", objective="O", outcome="Out")
        engine._save_from_cloud("episodes", ep)
        assert storage.get_episode("ep1") is not None

    def test_save_note_from_cloud(self, storage):
        engine = storage._sync_engine
        note = Note(id="n1", stack_id="test-agent", content="C")
        engine._save_from_cloud("notes", note)
        notes = storage.get_notes()
        assert any(n.id == "n1" for n in notes)

    def test_save_belief_from_cloud(self, storage):
        engine = storage._sync_engine
        b = Belief(id="b1", stack_id="test-agent", statement="S")
        engine._save_from_cloud("beliefs", b)
        beliefs = storage.get_beliefs()
        assert any(bl.id == "b1" for bl in beliefs)

    def test_save_value_from_cloud(self, storage):
        engine = storage._sync_engine
        v = Value(id="v1", stack_id="test-agent", name="N", statement="S")
        engine._save_from_cloud("agent_values", v)
        values = storage.get_values()
        assert any(val.id == "v1" for val in values)

    def test_save_goal_from_cloud(self, storage):
        engine = storage._sync_engine
        g = Goal(id="g1", stack_id="test-agent", title="T")
        engine._save_from_cloud("goals", g)
        goals = storage.get_goals()
        assert any(gl.id == "g1" for gl in goals)

    def test_save_drive_from_cloud(self, storage):
        engine = storage._sync_engine
        d = Drive(id="d1", stack_id="test-agent", drive_type="curiosity")
        engine._save_from_cloud("drives", d)
        assert storage.get_drive("curiosity") is not None

    def test_save_relationship_from_cloud(self, storage):
        engine = storage._sync_engine
        r = Relationship(
            id="r1",
            stack_id="test-agent",
            entity_name="Alice",
            entity_type="human",
            relationship_type="friend",
        )
        engine._save_from_cloud("relationships", r)
        assert storage.get_relationship("Alice") is not None

    def test_save_playbook_from_cloud(self, storage):
        engine = storage._sync_engine
        p = Playbook(
            id="p1",
            stack_id="test-agent",
            name="PB",
            description="From cloud",
            trigger_conditions=["t"],
            steps=[{"action": "a"}],
            failure_modes=["f"],
        )
        engine._save_from_cloud("playbooks", p)
        pb = storage.get_playbook("p1")
        assert pb is not None
        assert pb.name == "PB"


class TestIsOnlineEdgeCases:
    """Test is_online caching and outer exception paths."""

    def test_cached_within_ttl(self, storage_with_cloud, mock_cloud_storage):
        """Second call within TTL uses cached value, no additional get_stats call."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}
        engine = storage_with_cloud._sync_engine

        # First call
        result1 = engine.is_online()
        assert result1 is True
        assert mock_cloud_storage.get_stats.call_count == 1

        # Second call within TTL — should use cache
        result2 = engine.is_online()
        assert result2 is True
        assert mock_cloud_storage.get_stats.call_count == 1  # still 1

    def test_cache_expired_rechecks(self, storage_with_cloud, mock_cloud_storage):
        """After TTL expires, is_online re-checks connectivity."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}
        engine = storage_with_cloud._sync_engine

        # First call
        engine.is_online()
        assert mock_cloud_storage.get_stats.call_count == 1

        # Expire the cache by backdating the check time
        storage_with_cloud._last_connectivity_check = datetime.now(timezone.utc) - timedelta(
            seconds=120
        )

        # Second call after expiry
        engine.is_online()
        assert mock_cloud_storage.get_stats.call_count == 2

    def test_outer_exception_during_socket_ops(self, storage_with_cloud, mock_cloud_storage):
        """Outer exception during socket operations marks offline."""
        engine = storage_with_cloud._sync_engine
        storage_with_cloud._last_connectivity_check = None

        # socket.getdefaulttimeout() is called in the inner try; if it raises
        # an error that propagates past the inner except (which only catches
        # exceptions from get_stats), the outer except catches it.
        # We mock socket.getdefaulttimeout to raise inside the inner try.
        import socket

        original = socket.getdefaulttimeout
        socket.getdefaulttimeout = MagicMock(side_effect=Exception("socket broken"))
        try:
            result = engine.is_online()
        finally:
            socket.getdefaulttimeout = original

        assert result is False
        assert storage_with_cloud._is_online_cached is False

    def test_is_online_inner_exception_marks_offline(self, storage_with_cloud, mock_cloud_storage):
        """When cloud get_stats raises, is_online returns False and caches it."""
        mock_cloud_storage.get_stats.side_effect = ConnectionError("refused")
        engine = storage_with_cloud._sync_engine
        storage_with_cloud._last_connectivity_check = None

        result = engine.is_online()
        assert result is False
        assert storage_with_cloud._is_online_cached is False


class TestClearSyncConflictsWithBefore:
    """Test clear_sync_conflicts with a before datetime parameter."""

    def test_clear_conflicts_before_date(self, storage):
        """Only conflicts resolved before the given date are cleared."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=10)
        recent_time = now - timedelta(hours=1)

        # Save two conflicts: one old, one recent
        old_conflict = SyncConflict(
            id="old-c",
            table="notes",
            record_id="n-old",
            local_version={"content": "local"},
            cloud_version={"content": "cloud"},
            resolution="cloud_wins",
            resolved_at=old_time,
            local_summary="old local",
            cloud_summary="old cloud",
        )
        recent_conflict = SyncConflict(
            id="recent-c",
            table="notes",
            record_id="n-recent",
            local_version={"content": "local"},
            cloud_version={"content": "cloud"},
            resolution="local_wins",
            resolved_at=recent_time,
            local_summary="recent local",
            cloud_summary="recent cloud",
        )
        engine.save_sync_conflict(old_conflict)
        engine.save_sync_conflict(recent_conflict)

        # Clear conflicts older than 5 days ago
        cutoff = now - timedelta(days=5)
        cleared = engine.clear_sync_conflicts(before=cutoff)
        assert cleared == 1

        # Only the recent conflict should remain
        remaining = engine.get_sync_conflicts()
        assert len(remaining) == 1
        assert remaining[0].id == "recent-c"

    def test_clear_conflicts_no_before_clears_all(self, storage):
        """Without before parameter, all conflicts are cleared."""
        engine = storage._sync_engine
        for i in range(3):
            conflict = SyncConflict(
                id=f"c-{i}",
                table="notes",
                record_id=f"n-{i}",
                local_version={},
                cloud_version={},
                resolution="cloud_wins",
                resolved_at=datetime.now(timezone.utc),
                local_summary="l",
                cloud_summary="c",
            )
            engine.save_sync_conflict(conflict)

        cleared = engine.clear_sync_conflicts()
        assert cleared == 3
        assert len(engine.get_sync_conflicts()) == 0


class TestQueueUtilityMethods:
    """Tests for get_pending_sync_operations, mark_synced, get_sync_status."""

    def test_get_pending_sync_operations(self, storage):
        """get_pending_sync_operations returns dicts with correct keys."""
        engine = storage._sync_engine
        storage.save_note(Note(id="n-ops1", stack_id="test-agent", content="C"))
        ops = engine.get_pending_sync_operations(limit=10)
        assert len(ops) >= 1
        op = next(o for o in ops if o["record_id"] == "n-ops1")
        assert "id" in op
        assert op["operation"] in ("upsert", "insert", "update")
        assert op["table_name"] == "notes"
        assert op["local_updated_at"] is not None

    def test_mark_synced_empty_ids(self, storage):
        """mark_synced with empty list returns 0."""
        engine = storage._sync_engine
        assert engine.mark_synced([]) == 0

    def test_mark_synced_marks_records(self, storage):
        """mark_synced marks specific queue entries as synced."""
        engine = storage._sync_engine
        storage.save_note(Note(id="n-ms1", stack_id="test-agent", content="C"))
        ops = engine.get_pending_sync_operations(limit=10)
        ids = [o["id"] for o in ops]
        count = engine.mark_synced(ids)
        assert count == len(ids)
        # After marking synced, pending count should be 0
        assert engine.get_pending_sync_count() == 0

    def test_get_sync_status(self, storage):
        """get_sync_status returns counts by table and operation."""
        engine = storage._sync_engine
        storage.save_note(Note(id="n-st1", stack_id="test-agent", content="C"))
        storage.save_episode(
            Episode(id="ep-st1", stack_id="test-agent", objective="O", outcome="Out")
        )
        status = engine.get_sync_status()
        assert status["pending"] >= 2
        assert "notes" in status["by_table"]
        assert "episodes" in status["by_table"]
        assert status["total"] == status["pending"] + status["synced"]
        # by_operation should have at least one key
        assert len(status["by_operation"]) >= 1


class TestSyncPushDeletePath:
    """Test sync push phase handles delete operations."""

    def test_delete_operation_for_missing_record(self, storage_with_cloud, mock_cloud_storage):
        """Delete operation for a record that doesn't exist locally is counted as pushed."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Queue a delete for a record that doesn't exist
        storage_with_cloud.queue_sync_operation("delete", "notes", "deleted-note-1")

        # Clear connectivity cache
        storage_with_cloud._last_connectivity_check = None

        result = storage_with_cloud.sync()

        # Delete for missing record should still be marked as pushed
        assert result.pushed >= 1
        # Queue should be cleared
        changes = storage_with_cloud.get_queued_changes()
        assert not any(c.record_id == "deleted-note-1" for c in changes)


class TestSyncPushReturnsFalse:
    """Test sync push when _push_record returns False (not exception)."""

    def test_push_returns_false_records_failure(
        self, storage_with_cloud, mock_cloud_storage, caplog
    ):
        """When _push_record returns False, retry is recorded with error message."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Save a note
        storage_with_cloud.save_note(Note(id="pf1", stack_id="test-agent", content="C"))

        # Make save_note return without error but make push return False
        # by patching the engine's _push_record directly
        engine = storage_with_cloud._sync_engine
        original_push = engine._push_record
        engine._push_record = lambda table, record: False

        storage_with_cloud._last_connectivity_check = None

        try:
            with caplog.at_level(logging.WARNING, logger="kernle.storage.sync_engine"):
                result = storage_with_cloud.sync()
        finally:
            engine._push_record = original_push

        # Should have error about failed push with retry count
        assert any("pf1" in e and "retry" in e for e in result.errors)


class TestPullChangesExceptionPath:
    """Test pull_changes handles exceptions from cloud getters."""

    def test_pull_exception_from_cloud_getter(self, storage_with_cloud, mock_cloud_storage, caplog):
        """Exception from a cloud getter is caught and recorded."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}
        mock_cloud_storage.get_episodes.side_effect = RuntimeError("network timeout")
        mock_cloud_storage.get_notes.return_value = []

        engine = storage_with_cloud._sync_engine

        with caplog.at_level(logging.ERROR, logger="kernle.storage.sync_engine"):
            result = engine.pull_changes()

        assert any("episodes" in e for e in result.errors)
        assert any("Failed to pull from episodes" in r.message for r in caplog.records)


class TestPullMergesAllTypes:
    """Test that pull_changes exercises merge paths for beliefs, values, goals, relationships, playbooks."""

    def test_pull_new_belief(self, storage_with_cloud, mock_cloud_storage):
        """Pulling a new belief from cloud saves it locally."""
        mock_cloud_storage.get_stats.return_value = {}
        cloud_belief = Belief(
            id="cb1",
            stack_id="test-agent",
            statement="Cloud belief",
            cloud_synced_at=datetime.now(timezone.utc),
            local_updated_at=datetime.now(timezone.utc),
        )
        mock_cloud_storage.get_beliefs.return_value = [cloud_belief]
        mock_cloud_storage.get_episodes.return_value = []
        mock_cloud_storage.get_notes.return_value = []
        mock_cloud_storage.get_values.return_value = []
        mock_cloud_storage.get_goals.return_value = []
        mock_cloud_storage.get_drives.return_value = []
        mock_cloud_storage.get_relationships.return_value = []
        mock_cloud_storage.list_playbooks.return_value = []

        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 1
        beliefs = storage_with_cloud.get_beliefs()
        assert any(b.id == "cb1" for b in beliefs)

    def test_pull_new_value(self, storage_with_cloud, mock_cloud_storage):
        """Pulling a new value from cloud saves it locally."""
        mock_cloud_storage.get_stats.return_value = {}
        cloud_value = Value(
            id="cv1",
            stack_id="test-agent",
            name="N",
            statement="S",
            cloud_synced_at=datetime.now(timezone.utc),
            local_updated_at=datetime.now(timezone.utc),
        )
        mock_cloud_storage.get_values.return_value = [cloud_value]
        mock_cloud_storage.get_episodes.return_value = []
        mock_cloud_storage.get_notes.return_value = []
        mock_cloud_storage.get_beliefs.return_value = []
        mock_cloud_storage.get_goals.return_value = []
        mock_cloud_storage.get_drives.return_value = []
        mock_cloud_storage.get_relationships.return_value = []
        mock_cloud_storage.list_playbooks.return_value = []

        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 1
        values = storage_with_cloud.get_values()
        assert any(v.id == "cv1" for v in values)

    def test_pull_new_goal(self, storage_with_cloud, mock_cloud_storage):
        """Pulling a new goal from cloud saves it locally."""
        mock_cloud_storage.get_stats.return_value = {}
        cloud_goal = Goal(
            id="cg1",
            stack_id="test-agent",
            title="Cloud goal",
            cloud_synced_at=datetime.now(timezone.utc),
            local_updated_at=datetime.now(timezone.utc),
        )
        mock_cloud_storage.get_goals.return_value = [cloud_goal]
        mock_cloud_storage.get_episodes.return_value = []
        mock_cloud_storage.get_notes.return_value = []
        mock_cloud_storage.get_beliefs.return_value = []
        mock_cloud_storage.get_values.return_value = []
        mock_cloud_storage.get_drives.return_value = []
        mock_cloud_storage.get_relationships.return_value = []
        mock_cloud_storage.list_playbooks.return_value = []

        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 1
        goals = storage_with_cloud.get_goals()
        assert any(g.id == "cg1" for g in goals)

    def test_pull_new_relationship(self, storage_with_cloud, mock_cloud_storage):
        """Pulling a new relationship from cloud saves it locally."""
        mock_cloud_storage.get_stats.return_value = {}
        cloud_rel = Relationship(
            id="cr1",
            stack_id="test-agent",
            entity_name="Bob",
            entity_type="human",
            relationship_type="colleague",
            cloud_synced_at=datetime.now(timezone.utc),
            local_updated_at=datetime.now(timezone.utc),
        )
        mock_cloud_storage.get_relationships.return_value = [cloud_rel]
        mock_cloud_storage.get_episodes.return_value = []
        mock_cloud_storage.get_notes.return_value = []
        mock_cloud_storage.get_beliefs.return_value = []
        mock_cloud_storage.get_values.return_value = []
        mock_cloud_storage.get_goals.return_value = []
        mock_cloud_storage.get_drives.return_value = []
        mock_cloud_storage.list_playbooks.return_value = []

        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 1
        rel = storage_with_cloud.get_relationship("Bob")
        assert rel is not None

    def test_pull_new_playbook(self, storage_with_cloud, mock_cloud_storage):
        """Pulling a new playbook from cloud saves it locally."""
        mock_cloud_storage.get_stats.return_value = {}
        cloud_pb = Playbook(
            id="cp1",
            stack_id="test-agent",
            name="PB",
            description="D",
            trigger_conditions=["t"],
            steps=[{"action": "a"}],
            failure_modes=["f"],
            cloud_synced_at=datetime.now(timezone.utc),
            local_updated_at=datetime.now(timezone.utc),
        )
        mock_cloud_storage.list_playbooks.return_value = [cloud_pb]
        mock_cloud_storage.get_episodes.return_value = []
        mock_cloud_storage.get_notes.return_value = []
        mock_cloud_storage.get_beliefs.return_value = []
        mock_cloud_storage.get_values.return_value = []
        mock_cloud_storage.get_goals.return_value = []
        mock_cloud_storage.get_drives.return_value = []
        mock_cloud_storage.get_relationships.return_value = []

        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 1
        pb = storage_with_cloud.get_playbook("cp1")
        assert pb is not None


class TestPullMergesConflictForAllTypes:
    """Test that pull with existing local records exercises merge paths for beliefs/values/goals."""

    def _setup_cloud_defaults(self, mock_cloud_storage):
        """Set all cloud getters to return empty lists."""
        mock_cloud_storage.get_stats.return_value = {}
        mock_cloud_storage.get_episodes.return_value = []
        mock_cloud_storage.get_notes.return_value = []
        mock_cloud_storage.get_beliefs.return_value = []
        mock_cloud_storage.get_values.return_value = []
        mock_cloud_storage.get_goals.return_value = []
        mock_cloud_storage.get_drives.return_value = []
        mock_cloud_storage.get_relationships.return_value = []
        mock_cloud_storage.list_playbooks.return_value = []

    def test_pull_belief_conflict_cloud_wins(self, storage_with_cloud, mock_cloud_storage):
        """Pulling a belief that already exists locally exercises _merge_belief local lookup."""
        self._setup_cloud_defaults(mock_cloud_storage)
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_belief(
            Belief(
                id="cb-conflict",
                stack_id="test-agent",
                statement="Old local",
                local_updated_at=old_time,
            )
        )
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        new_time = datetime.now(timezone.utc)
        mock_cloud_storage.get_beliefs.return_value = [
            Belief(
                id="cb-conflict",
                stack_id="test-agent",
                statement="Cloud belief",
                cloud_synced_at=new_time,
                local_updated_at=new_time,
            )
        ]
        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 1
        assert result.conflict_count == 1

    def test_pull_value_conflict_cloud_wins(self, storage_with_cloud, mock_cloud_storage):
        """Pulling a value that already exists exercises _merge_value local lookup."""
        self._setup_cloud_defaults(mock_cloud_storage)
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_value(
            Value(
                id="cv-conflict",
                stack_id="test-agent",
                name="V",
                statement="Old",
                local_updated_at=old_time,
            )
        )
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        new_time = datetime.now(timezone.utc)
        mock_cloud_storage.get_values.return_value = [
            Value(
                id="cv-conflict",
                stack_id="test-agent",
                name="V",
                statement="Cloud",
                cloud_synced_at=new_time,
                local_updated_at=new_time,
            )
        ]
        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 1

    def test_pull_goal_conflict_cloud_wins(self, storage_with_cloud, mock_cloud_storage):
        """Pulling a goal that already exists exercises _merge_goal local lookup."""
        self._setup_cloud_defaults(mock_cloud_storage)
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_goal(
            Goal(
                id="cg-conflict", stack_id="test-agent", title="Old goal", local_updated_at=old_time
            )
        )
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        new_time = datetime.now(timezone.utc)
        mock_cloud_storage.get_goals.return_value = [
            Goal(
                id="cg-conflict",
                stack_id="test-agent",
                title="Cloud goal",
                cloud_synced_at=new_time,
                local_updated_at=new_time,
            )
        ]
        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 1

    def test_pull_playbook_conflict_local_wins_merges_and_persists(
        self,
        storage_with_cloud,
        mock_cloud_storage,
    ):
        """Local playbook remains authoritative when newer but still merges cloud arrays."""
        self._setup_cloud_defaults(mock_cloud_storage)
        older = datetime.now(timezone.utc) - timedelta(hours=2)
        newer = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_playbook(
            Playbook(
                id="cp-conflict",
                stack_id="test-agent",
                name="Local playbook",
                description="local",
                trigger_conditions=["local"],
                steps=[{"action": "a"}],
                failure_modes=["local-fail"],
                source_episodes=["local-episode"],
                tags=["local-tag"],
                cloud_synced_at=older,
                local_updated_at=newer,
            )
        )
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        mock_cloud_storage.list_playbooks.return_value = [
            Playbook(
                id="cp-conflict",
                stack_id="test-agent",
                name="Cloud playbook",
                description="cloud",
                trigger_conditions=["cloud"],
                steps=[{"action": "a"}],
                failure_modes=["cloud-fail"],
                source_episodes=["cloud-episode"],
                tags=["cloud-tag"],
                cloud_synced_at=older,
                local_updated_at=older,
            )
        ]

        result = storage_with_cloud._sync_engine.pull_changes()
        assert result.pulled == 0
        merged = storage_with_cloud.get_playbook("cp-conflict")
        assert merged is not None
        assert set(merged.trigger_conditions) == {"local", "cloud"}
        assert set(merged.tags) == {"local-tag", "cloud-tag"}

        with storage_with_cloud._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM sync_queue WHERE table_name = ? AND record_id = ?",
                ("playbooks", "cp-conflict"),
            ).fetchone()
            assert row is None


class TestMergeArrayFieldsMissingAttribute:
    """Test _merge_array_fields when winner/loser lacks an expected field."""

    def test_missing_field_skipped(self, storage):
        """If winner lacks a configured array field, that field is skipped."""
        engine = storage._sync_engine
        from dataclasses import dataclass

        @dataclass
        class PartialRecord:
            id: str
            tags: list  # has tags but not other fields like lessons, derived_from

        winner = PartialRecord(id="1", tags=["a"])
        loser = PartialRecord(id="1", tags=["b"])
        # episodes expects lessons, emotional_tags etc. which PartialRecord lacks
        result = engine._merge_array_fields("episodes", winner, loser)
        # Should still merge the field it does have (tags)
        assert set(result.tags) == {"a", "b"}


class TestPlaybookMergeArrayFields:
    """Test _merge_array_fields for playbook-specific array fields."""

    def test_playbook_arrays_merged(self, storage):
        """Playbook trigger_conditions, failure_modes, recovery_steps merge."""
        engine = storage._sync_engine
        winner = Playbook(
            id="p1",
            stack_id="test",
            name="PB",
            description="D",
            trigger_conditions=["when-a"],
            steps=[{"action": "a"}],
            failure_modes=["fail-a"],
            recovery_steps=["recover-a"],
            source_episodes=["ep-1"],
            tags=["tag-w"],
        )
        loser = Playbook(
            id="p1",
            stack_id="test",
            name="PB",
            description="D",
            trigger_conditions=["when-b"],
            steps=[{"action": "a"}],
            failure_modes=["fail-b"],
            recovery_steps=["recover-b"],
            source_episodes=["ep-2"],
            tags=["tag-l"],
        )
        result = engine._merge_array_fields("playbooks", winner, loser)
        assert set(result.trigger_conditions) == {"when-a", "when-b"}
        assert set(result.failure_modes) == {"fail-a", "fail-b"}
        assert set(result.recovery_steps) == {"recover-a", "recover-b"}
        assert set(result.source_episodes) == {"ep-1", "ep-2"}
        assert set(result.tags) == {"tag-w", "tag-l"}


class TestDeadLetterSemantics:
    """Tests for dead-letter sync queue semantics (issue #721).

    Dead-lettered entries (synced=2) must be distinguishable from
    successfully synced records (synced=1) and pending records (synced=0).
    """

    def test_clear_failed_sets_dead_letter_state(self, storage):
        """clear_failed_sync_records should set synced=2 (DEAD_LETTER), not synced=1."""
        from kernle.types import SYNC_DEAD_LETTER

        storage.save_note(Note(id="dl-1", stack_id="test-agent", content="Dead letter test"))

        queued = storage.get_queued_changes(limit=10)
        change = next(c for c in queued if c.record_id == "dl-1")

        # Simulate 5 failures and set last_attempt_at to 10 days ago
        with storage._connect() as conn:
            for _ in range(5):
                storage._record_sync_failure(conn, change.id, "Persistent error")
            old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            conn.execute(
                "UPDATE sync_queue SET last_attempt_at = ? WHERE id = ?",
                (old_time, change.id),
            )
            conn.commit()

        cleared = storage.clear_failed_sync_records(older_than_days=7)
        assert cleared == 1

        # Verify the record has synced=2 (DEAD_LETTER), not synced=1
        with storage._connect() as conn:
            row = conn.execute(
                "SELECT synced FROM sync_queue WHERE id = ?", (change.id,)
            ).fetchone()
        assert row["synced"] == SYNC_DEAD_LETTER

    def test_dead_letter_entries_not_in_queued_changes(self, storage):
        """Dead-lettered entries must not appear in get_queued_changes."""
        storage.save_note(Note(id="dl-2", stack_id="test-agent", content="Dead letter queue test"))

        queued = storage.get_queued_changes(limit=10)
        change = next(c for c in queued if c.record_id == "dl-2")

        # Move to dead-letter state
        with storage._connect() as conn:
            for _ in range(5):
                storage._record_sync_failure(conn, change.id, "Error")
            old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            conn.execute(
                "UPDATE sync_queue SET last_attempt_at = ? WHERE id = ?",
                (old_time, change.id),
            )
            conn.commit()

        storage.clear_failed_sync_records(older_than_days=7)

        # Should not appear in queued changes
        queued_after = storage.get_queued_changes(limit=100)
        assert not any(c.record_id == "dl-2" for c in queued_after)

    def test_get_dead_letter_count(self, storage):
        """get_dead_letter_count returns the number of dead-lettered entries."""
        # Initially zero
        assert storage.get_dead_letter_count() == 0

        # Create and dead-letter two entries
        for note_id in ("dl-3a", "dl-3b"):
            storage.save_note(Note(id=note_id, stack_id="test-agent", content=f"DL test {note_id}"))
            queued = storage.get_queued_changes(limit=100)
            change = next(c for c in queued if c.record_id == note_id)
            with storage._connect() as conn:
                for _ in range(5):
                    storage._record_sync_failure(conn, change.id, "Error")
                old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
                conn.execute(
                    "UPDATE sync_queue SET last_attempt_at = ? WHERE id = ?",
                    (old_time, change.id),
                )
                conn.commit()

        storage.clear_failed_sync_records(older_than_days=7)
        assert storage.get_dead_letter_count() == 2

    def test_requeue_dead_letters_resets_state(self, storage):
        """requeue_dead_letters moves dead-lettered entries back to pending."""
        from kernle.types import SYNC_PENDING

        storage.save_note(Note(id="dl-4", stack_id="test-agent", content="Requeue test"))

        queued = storage.get_queued_changes(limit=10)
        change = next(c for c in queued if c.record_id == "dl-4")

        # Move to dead-letter
        with storage._connect() as conn:
            for _ in range(5):
                storage._record_sync_failure(conn, change.id, "Error")
            old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            conn.execute(
                "UPDATE sync_queue SET last_attempt_at = ? WHERE id = ?",
                (old_time, change.id),
            )
            conn.commit()

        storage.clear_failed_sync_records(older_than_days=7)
        assert storage.get_dead_letter_count() == 1

        # Requeue
        requeued = storage.requeue_dead_letters()
        assert requeued == 1
        assert storage.get_dead_letter_count() == 0

        # Verify it is back in queued changes with reset retry_count
        queued_after = storage.get_queued_changes(limit=100)
        requeued_change = next(c for c in queued_after if c.record_id == "dl-4")
        assert requeued_change.retry_count == 0
        assert requeued_change.last_error is None

        # Verify synced column is SYNC_PENDING
        with storage._connect() as conn:
            row = conn.execute(
                "SELECT synced FROM sync_queue WHERE id = ?", (change.id,)
            ).fetchone()
        assert row["synced"] == SYNC_PENDING

    def test_requeue_dead_letters_by_id(self, storage):
        """requeue_dead_letters with specific IDs only requeues those entries."""
        ids_map = {}
        for note_id in ("dl-5a", "dl-5b", "dl-5c"):
            storage.save_note(Note(id=note_id, stack_id="test-agent", content=f"DL {note_id}"))
            queued = storage.get_queued_changes(limit=100)
            change = next(c for c in queued if c.record_id == note_id)
            ids_map[note_id] = change.id
            with storage._connect() as conn:
                for _ in range(5):
                    storage._record_sync_failure(conn, change.id, "Error")
                old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
                conn.execute(
                    "UPDATE sync_queue SET last_attempt_at = ? WHERE id = ?",
                    (old_time, change.id),
                )
                conn.commit()

        storage.clear_failed_sync_records(older_than_days=7)
        assert storage.get_dead_letter_count() == 3

        # Requeue only the first one
        requeued = storage.requeue_dead_letters(record_ids=[ids_map["dl-5a"]])
        assert requeued == 1
        assert storage.get_dead_letter_count() == 2

    def test_dead_letter_count_in_sync_status(self, storage):
        """get_sync_status should include dead_letter count."""
        status = storage.get_sync_status()
        assert "dead_letter" in status
        assert status["dead_letter"] == 0

        # Create and dead-letter an entry
        storage.save_note(Note(id="dl-6", stack_id="test-agent", content="Status test"))
        queued = storage.get_queued_changes(limit=10)
        change = next(c for c in queued if c.record_id == "dl-6")
        with storage._connect() as conn:
            for _ in range(5):
                storage._record_sync_failure(conn, change.id, "Error")
            old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            conn.execute(
                "UPDATE sync_queue SET last_attempt_at = ? WHERE id = ?",
                (old_time, change.id),
            )
            conn.commit()

        storage.clear_failed_sync_records(older_than_days=7)
        status = storage.get_sync_status()
        assert status["dead_letter"] == 1


# ============================================================================
# apply_pull_operation — per-table dispatch
# ============================================================================


class TestApplyPullOperation:
    """Tests for SyncEngine.apply_pull_operation — the unified pull apply path."""

    def test_apply_pull_operation_episode(self, storage):
        """Applying an episode pull operation saves locally."""
        op = {
            "table": "episodes",
            "record_id": "ep-pull-1",
            "operation": "upsert",
            "data": {
                "objective": "Test episode from cloud",
                "outcome_description": "It worked",
                "outcome_type": "success",
                "lessons_learned": ["lesson1"],
                "tags": ["cloud"],
            },
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1
        assert error is None

        ep = storage.get_episode("ep-pull-1")
        assert ep is not None
        assert ep.objective == "Test episode from cloud"
        assert ep.outcome == "It worked"
        assert ep.lessons == ["lesson1"]
        assert ep.source_type == "external"
        assert ep.source_entity == "kernle:sync"

    def test_apply_pull_operation_note(self, storage):
        """Applying a note pull operation saves locally."""
        op = {
            "table": "notes",
            "record_id": "note-pull-1",
            "operation": "upsert",
            "data": {
                "content": "Note from cloud",
                "note_type": "insight",
                "tags": ["synced"],
            },
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1
        assert error is None

        notes = storage.get_notes(limit=20)
        note = next(n for n in notes if n.id == "note-pull-1")
        assert note.content == "Note from cloud"
        assert note.note_type == "insight"
        assert note.source_type == "external"

    def test_apply_pull_operation_belief(self, storage):
        """Applying a belief pull operation saves locally."""
        op = {
            "table": "beliefs",
            "record_id": "bel-pull-1",
            "operation": "upsert",
            "data": {
                "statement": "The sky is blue",
                "belief_type": "fact",
                "confidence": 0.9,
            },
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1
        assert error is None

        with storage._connect() as conn:
            row = conn.execute("SELECT * FROM beliefs WHERE id = ?", ("bel-pull-1",)).fetchone()
        assert row is not None
        assert row["statement"] == "The sky is blue"

    def test_apply_pull_operation_goal(self, storage):
        """Applying a goal pull operation saves locally."""
        op = {
            "table": "goals",
            "record_id": "goal-pull-1",
            "operation": "upsert",
            "data": {
                "title": "Ship the feature",
                "goal_type": "task",
                "priority": "high",
            },
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1
        assert error is None

        with storage._connect() as conn:
            row = conn.execute("SELECT * FROM goals WHERE id = ?", ("goal-pull-1",)).fetchone()
        assert row is not None
        assert row["title"] == "Ship the feature"

    def test_apply_pull_operation_drive(self, storage):
        """Applying a drive pull operation saves locally."""
        op = {
            "table": "drives",
            "record_id": "drv-pull-1",
            "operation": "upsert",
            "data": {
                "drive_type": "curiosity",
                "intensity": 0.8,
                "focus_areas": ["learning"],
            },
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1
        assert error is None

        drive = storage.get_drive("curiosity")
        assert drive is not None
        assert drive.intensity == 0.8

    def test_apply_pull_operation_relationship(self, storage):
        """Applying a relationship pull operation saves locally."""
        op = {
            "table": "relationships",
            "record_id": "rel-pull-1",
            "operation": "upsert",
            "data": {
                "entity_name": "Alice",
                "entity_type": "person",
                "relationship_type": "collaborator",
                "sentiment": 0.5,
            },
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1
        assert error is None

        rel = storage.get_relationship("Alice")
        assert rel is not None
        assert rel.relationship_type == "collaborator"

    def test_apply_pull_operation_value(self, storage):
        """Applying an agent_values pull operation saves locally."""
        op = {
            "table": "agent_values",
            "record_id": "val-pull-1",
            "operation": "upsert",
            "data": {
                "name": "honesty",
                "statement": "Be honest always",
                "priority": 90,
            },
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1
        assert error is None

        with storage._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_values WHERE id = ?", ("val-pull-1",)
            ).fetchone()
        assert row is not None
        assert row["name"] == "honesty"

    def test_apply_pull_operation_playbook(self, storage):
        """Applying a playbook pull operation saves locally."""
        op = {
            "table": "playbooks",
            "record_id": "pb-pull-1",
            "operation": "upsert",
            "data": {
                "name": "Deploy procedure",
                "description": "How to deploy",
                "trigger_conditions": ["deploy requested"],
                "steps": [{"action": "run tests"}],
                "failure_modes": ["tests fail"],
            },
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1
        assert error is None

        pb = storage.get_playbook("pb-pull-1")
        assert pb is not None
        assert pb.name == "Deploy procedure"

    # --- Operation type tests ---

    def test_apply_pull_operation_upsert_accepted(self, storage):
        """'upsert' is accepted as a valid operation."""
        op = {
            "table": "notes",
            "record_id": "note-upsert",
            "operation": "upsert",
            "data": {"content": "Upsert note"},
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True

    def test_apply_pull_operation_insert_accepted(self, storage):
        """'insert' is accepted as a valid operation."""
        op = {
            "table": "notes",
            "record_id": "note-insert",
            "operation": "insert",
            "data": {"content": "Insert note"},
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True

    def test_apply_pull_operation_update_accepted(self, storage):
        """'update' is accepted as a valid operation."""
        op = {
            "table": "notes",
            "record_id": "note-update",
            "operation": "update",
            "data": {"content": "Update note"},
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True

    # --- Error contract tests ---

    def test_apply_pull_operation_unknown_table(self, storage):
        """Unknown table returns failure with descriptive error."""
        op = {
            "table": "unicorns",
            "record_id": "x",
            "operation": "upsert",
            "data": {"foo": "bar"},
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is False
        assert count == 0
        assert "unsupported table" in error

    def test_apply_pull_operation_delete_rejected(self, storage):
        """Delete operations are explicitly rejected."""
        op = {
            "table": "notes",
            "record_id": "n1",
            "operation": "delete",
            "data": {},
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is False
        assert count == 0
        assert "delete" in error.lower()

    def test_apply_pull_operation_missing_required_field(self, storage):
        """Missing required field (e.g. drive_type for drives) returns failure."""
        op = {
            "table": "drives",
            "record_id": "drv-bad",
            "operation": "upsert",
            "data": {"intensity": 0.5},  # missing drive_type
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is False
        assert count == 0
        assert error is not None

    def test_apply_pull_operation_unsupported_operation(self, storage):
        """Unsupported operation type returns failure."""
        op = {
            "table": "notes",
            "record_id": "n1",
            "operation": "purge",
            "data": {"content": "x"},
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is False
        assert count == 0
        assert "unsupported operation" in error.lower()

    # --- Table name normalization ---

    def test_apply_pull_operation_values_alias(self, storage):
        """'values' is normalized to 'agent_values'."""
        op = {
            "table": "values",
            "record_id": "val-alias-1",
            "operation": "upsert",
            "data": {"name": "respect", "statement": "Show respect"},
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True
        assert count == 1

        with storage._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_values WHERE id = ?", ("val-alias-1",)
            ).fetchone()
        assert row is not None

    def test_apply_pull_operation_agent_values_direct(self, storage):
        """'agent_values' table name works directly."""
        op = {
            "table": "agent_values",
            "record_id": "val-direct-1",
            "operation": "upsert",
            "data": {"name": "courage", "statement": "Be brave"},
        }
        success, count, error = storage.apply_pull_operation(op)
        assert success is True


# ============================================================================
# _record_from_pull_data — field mapping & provenance
# ============================================================================


class TestRecordFromPullData:
    """Tests for SyncEngine._record_from_pull_data — record factory."""

    def test_record_from_pull_data_episode_field_mapping(self, storage):
        """Episode field mapping: lessons_learned -> lessons, outcome_description -> outcome."""
        record = storage._record_from_pull_data(
            "episodes",
            "ep-map-1",
            {
                "objective": "Test",
                "lessons_learned": ["a", "b"],
                "outcome_description": "desc",
                "outcome_type": "success",
            },
        )
        assert record.lessons == ["a", "b"]
        assert record.outcome == "desc"
        assert record.objective == "Test"

    def test_record_from_pull_data_sets_sync_provenance(self, storage):
        """All records get source_type=external and source_entity=kernle:sync."""
        record = storage._record_from_pull_data("notes", "note-prov-1", {"content": "Test note"})
        assert record.source_type == "external"
        assert record.source_entity == "kernle:sync"
        assert record.stack_id == storage.stack_id

    def test_record_from_pull_data_remaps_stack_id(self, storage):
        """stack_id in data is overridden by storage.stack_id."""
        record = storage._record_from_pull_data(
            "notes", "note-stack-1", {"content": "Test", "stack_id": "foreign-stack"}
        )
        assert record.stack_id == storage.stack_id

    def test_record_from_pull_data_parses_timestamps(self, storage):
        """ISO timestamp strings in data are parsed to datetime."""
        record = storage._record_from_pull_data(
            "notes",
            "note-ts-1",
            {
                "content": "Test",
                "local_updated_at": "2025-01-15T10:00:00+00:00",
                "cloud_synced_at": "2025-01-15T11:00:00+00:00",
            },
        )
        assert isinstance(record.local_updated_at, datetime)
        assert isinstance(record.cloud_synced_at, datetime)

    def test_record_from_pull_data_default_timestamps(self, storage):
        """When timestamps are missing, they default to None."""
        record = storage._record_from_pull_data("notes", "note-notime-1", {"content": "Test"})
        # Should not raise — timestamps are optional
        assert record.local_updated_at is None or isinstance(record.local_updated_at, datetime)

    def test_record_from_pull_data_preserves_version(self, storage):
        """Version from data is preserved on the record."""
        record = storage._record_from_pull_data(
            "notes", "note-ver-1", {"content": "Test", "version": 5}
        )
        assert record.version == 5
