"""Tests verifying strict-mode write path routes through _write_backend.

These tests ensure that write methods in WritersMixin read existing data
through _write_backend (which respects strict mode) rather than bypassing
it by going directly to _storage.

The key invariant: in strict mode, _write_backend returns a Stack
which enforces maintenance mode blocking, provenance validation, and
stack component hooks. If a method reads via _storage directly, those
enforcement layers are silently bypassed.
"""

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kernle.core import Kernle
from kernle.storage import Drive, Episode, Goal

# =========================================================================
# Real strict-mode guard failures (no stack attached)
# =========================================================================


@pytest.fixture
def strict_kernle_no_stack():
    """Kernle in strict=True mode with a non-SQLite storage so stack is None.

    The _write_backend property raises ValueError when strict=True and
    the stack property returns None (which happens when the underlying
    storage is not SQLiteStorage).

    This IS the real runtime failure path — it fires when a caller
    passes a custom storage backend (not SQLiteStorage) while requesting
    strict=True. There is no other way stack returns None with Kernle.

    Note: _require_inference is patched out because these tests verify
    the strict-mode guard (ValueError), not the inference gate.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mock_storage = MagicMock()
        # Prevent auto-sync detection from failing
        mock_storage.is_online.return_value = False
        mock_storage.get_pending_sync_count.return_value = 0
        k = Kernle(
            stack_id="test-strict-guard",
            storage=mock_storage,
            checkpoint_dir=Path(tmp) / "cp",
            strict=True,
        )
        # Verify the precondition: stack IS None with non-SQLite storage
        assert k.stack is None
        with patch.object(type(k), "_require_inference", lambda self, op: None):
            yield k


class TestStrictModeGuardFailures:
    """Verify that write operations raise ValueError when strict=True and no stack is available.

    This covers the real runtime path: strict mode with non-SQLite storage.
    """

    def test_write_backend_property_raises_directly(self, strict_kernle_no_stack):
        """The _write_backend property itself raises ValueError."""
        with pytest.raises(ValueError, match="strict=True requires"):
            _ = strict_kernle_no_stack._write_backend

    def test_update_episode_raises_without_stack(self, strict_kernle_no_stack):
        with pytest.raises(ValueError, match="strict=True requires"):
            strict_kernle_no_stack.update_episode("fake-id", outcome="x")

    def test_update_goal_raises_without_stack(self, strict_kernle_no_stack):
        with pytest.raises(ValueError, match="strict=True requires"):
            strict_kernle_no_stack.update_goal("fake-id", status="completed")

    def test_goal_raises_without_stack(self, strict_kernle_no_stack):
        with pytest.raises(ValueError, match="strict=True requires"):
            strict_kernle_no_stack.goal("test goal", "desc")

    def test_drive_raises_without_stack(self, strict_kernle_no_stack):
        with pytest.raises(ValueError, match="strict=True requires"):
            strict_kernle_no_stack.drive("curiosity", intensity=0.8)

    def test_satisfy_drive_raises_without_stack(self, strict_kernle_no_stack):
        with pytest.raises(ValueError, match="strict=True requires"):
            strict_kernle_no_stack.satisfy_drive("curiosity")


class TestStrictModeWithRealSQLite:
    """Verify strict mode works end-to-end with real SQLiteStorage.

    When strict=True with SQLiteStorage, _write_backend returns a real
    Stack. Writes should succeed and route through the stack.
    """

    def test_strict_mode_creates_stack_with_real_storage(self, tmp_path):
        """strict=True + SQLiteStorage auto-creates a Stack."""
        from kernle.stack.sqlite_stack import Stack
        from kernle.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(stack_id="test-real-strict", db_path=tmp_path / "test.db")
        try:
            k = Kernle(
                stack_id="test-real-strict",
                storage=storage,
                checkpoint_dir=tmp_path / "cp",
                strict=True,
            )
            assert k.stack is not None
            assert isinstance(k.stack, Stack)
            assert k._write_backend is k.stack
        finally:
            storage.close()

    def test_strict_mode_episode_writes_through_stack(self, tmp_path):
        """Episode writes succeed and go through the stack enforcement layer."""
        from kernle.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(stack_id="test-real-strict", db_path=tmp_path / "test.db")
        try:
            k = Kernle(
                stack_id="test-real-strict",
                storage=storage,
                checkpoint_dir=tmp_path / "cp",
                strict=True,
            )
            # First save a raw entry to satisfy provenance requirements
            raw_id = storage.save_raw("test content for provenance", source="test")

            ep_id = k.episode(
                objective="verify strict write path",
                outcome="episode saved through stack",
                derived_from=[f"raw:{raw_id}"],
            )
            assert ep_id is not None
            # Verify it was persisted
            episodes = storage.get_episodes(limit=10)
            assert any(e.id == ep_id for e in episodes)
        finally:
            storage.close()


def _uid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def mocked_kernle():
    """Create a Kernle with separately mockable _write_backend and _storage.

    Yields (kernle, mock_write_backend, mock_storage).

    Uses patch.object on the Kernle class to override the _write_backend
    property safely, ensuring cleanup even if the test fails.

    Note: _require_inference is patched out because these tests verify
    the write-backend routing, not the inference gate.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Create a real Kernle in non-strict mode (to avoid stack creation)
        k = Kernle(stack_id="test-strict", checkpoint_dir=Path(tmp) / "cp", strict=False)

        mock_storage = MagicMock()
        mock_write_backend = MagicMock()

        k._storage = mock_storage

        # Patch the _write_backend property on the class; patch.object handles cleanup
        with (
            patch.object(
                type(k),
                "_write_backend",
                new_callable=PropertyMock,
                return_value=mock_write_backend,
            ),
            patch.object(type(k), "_require_inference", lambda self, op: None),
        ):
            yield k, mock_write_backend, mock_storage


# =========================================================================
# update_episode
# =========================================================================


class TestUpdateEpisodeStrictPath:
    """update_episode must read the episode through _write_backend."""

    def test_uses_write_backend_get_memory_for_lookup(self, mocked_kernle):
        """update_episode reads episode via _write_backend.get_memory, not _storage.get_episode."""
        k, mock_wb, mock_storage = mocked_kernle

        episode_id = _uid()
        existing_episode = Episode(
            id=episode_id,
            stack_id="test-strict",
            objective="test objective",
            outcome="test outcome",
            outcome_type="partial",
            tags=["test"],
            created_at=_now(),
            confidence=0.8,
        )
        mock_wb.get_memory.return_value = existing_episode

        result = k.update_episode(episode_id, outcome="updated success outcome")

        assert result is True
        mock_wb.get_memory.assert_called_once_with("episode", episode_id)
        mock_storage.get_episode.assert_not_called()

    def test_returns_false_when_write_backend_returns_none(self, mocked_kernle):
        """update_episode returns False when _write_backend.get_memory returns None."""
        k, mock_wb, mock_storage = mocked_kernle

        episode_id = _uid()
        mock_wb.get_memory.return_value = None

        result = k.update_episode(episode_id, outcome="new outcome")

        assert result is False
        mock_wb.get_memory.assert_called_once_with("episode", episode_id)
        mock_storage.get_episode.assert_not_called()

    def test_still_uses_storage_for_atomic_update(self, mocked_kernle):
        """update_episode uses _storage.update_episode_atomic after the _write_backend guard.

        The atomic update bypasses the normal save path by design -- it needs
        the low-level storage for optimistic concurrency control.
        """
        k, mock_wb, mock_storage = mocked_kernle

        episode_id = _uid()
        existing_episode = Episode(
            id=episode_id,
            stack_id="test-strict",
            objective="test objective",
            outcome="old outcome",
            outcome_type="partial",
            tags=["test"],
            created_at=_now(),
            confidence=0.8,
        )
        mock_wb.get_memory.return_value = existing_episode

        result = k.update_episode(episode_id, outcome="success done")

        assert result is True
        # The guard goes through _write_backend
        mock_wb.get_memory.assert_called_once()
        # The atomic update goes through _storage (by design)
        mock_storage.update_episode_atomic.assert_called_once_with(existing_episode)


# =========================================================================
# update_goal
# =========================================================================


class TestUpdateGoalStrictPath:
    """update_goal must read goals through _write_backend."""

    def test_uses_write_backend_get_goals_for_lookup(self, mocked_kernle):
        """update_goal reads goals via _write_backend.get_goals, not _storage.get_goals."""
        k, mock_wb, mock_storage = mocked_kernle

        goal_id = _uid()
        existing_goal = Goal(
            id=goal_id,
            stack_id="test-strict",
            title="test goal",
            description="test goal description",
            goal_type="task",
            priority="medium",
            status="active",
            created_at=_now(),
        )
        mock_wb.get_goals.return_value = [existing_goal]

        result = k.update_goal(goal_id, status="completed")

        assert result is True
        mock_wb.get_goals.assert_called_once_with(status=None, limit=1000)
        mock_storage.get_goals.assert_not_called()

    def test_returns_false_when_not_found_in_write_backend(self, mocked_kernle):
        """update_goal returns False when goal not found via _write_backend.get_goals."""
        k, mock_wb, mock_storage = mocked_kernle

        mock_wb.get_goals.return_value = []

        result = k.update_goal(_uid(), status="completed")

        assert result is False
        mock_wb.get_goals.assert_called_once()
        mock_storage.get_goals.assert_not_called()

    def test_save_goes_through_write_backend(self, mocked_kernle):
        """update_goal saves the modified goal via _write_backend.update_goal_atomic."""
        k, mock_wb, mock_storage = mocked_kernle

        goal_id = _uid()
        existing_goal = Goal(
            id=goal_id,
            stack_id="test-strict",
            title="test",
            description="test",
            goal_type="task",
            priority="medium",
            status="active",
            created_at=_now(),
        )
        mock_wb.get_goals.return_value = [existing_goal]

        result = k.update_goal(goal_id, status="completed")

        assert result is True
        mock_wb.update_goal_atomic.assert_called_once()
        mock_storage.save_goal.assert_not_called()


# =========================================================================
# goal (protect_memory)
# =========================================================================


class TestGoalProtectMemoryStrictPath:
    """goal() must call protect_memory through _write_backend for protected goal types."""

    def test_aspiration_uses_write_backend_protect_memory(self, mocked_kernle):
        """goal() calls _write_backend.protect_memory for aspiration goals."""
        k, mock_wb, mock_storage = mocked_kernle

        goal_id = k.goal(
            title="long-term aspiration",
            goal_type="aspiration",
        )

        assert goal_id is not None
        mock_wb.protect_memory.assert_called_once_with("goal", goal_id, protected=True)
        mock_storage.protect_memory.assert_not_called()

    def test_commitment_uses_write_backend_protect_memory(self, mocked_kernle):
        """goal() calls _write_backend.protect_memory for commitment goals."""
        k, mock_wb, mock_storage = mocked_kernle

        goal_id = k.goal(
            title="commitment goal",
            goal_type="commitment",
        )

        mock_wb.protect_memory.assert_called_once_with("goal", goal_id, protected=True)
        mock_storage.protect_memory.assert_not_called()

    def test_task_does_not_protect(self, mocked_kernle):
        """goal() does not call protect_memory for task goals."""
        k, mock_wb, mock_storage = mocked_kernle

        k.goal(title="regular task", goal_type="task")

        mock_wb.protect_memory.assert_not_called()
        mock_storage.protect_memory.assert_not_called()

    def test_exploration_does_not_protect(self, mocked_kernle):
        """goal() does not call protect_memory for exploration goals."""
        k, mock_wb, mock_storage = mocked_kernle

        k.goal(title="exploration goal", goal_type="exploration")

        mock_wb.protect_memory.assert_not_called()
        mock_storage.protect_memory.assert_not_called()


# =========================================================================
# drive
# =========================================================================


class TestDriveStrictPath:
    """drive() must read existing drives through _write_backend."""

    def test_new_drive_uses_write_backend_for_lookup(self, mocked_kernle):
        """drive() reads drives via _write_backend.get_drives, not _storage.get_drive."""
        k, mock_wb, mock_storage = mocked_kernle

        mock_wb.get_drives.return_value = []

        drive_id = k.drive("curiosity", intensity=0.7)

        assert drive_id is not None
        mock_wb.get_drives.assert_called_once()
        mock_storage.get_drive.assert_not_called()

    def test_update_existing_uses_write_backend(self, mocked_kernle):
        """drive() updates an existing drive found via _write_backend.get_drives."""
        k, mock_wb, mock_storage = mocked_kernle

        existing_drive = Drive(
            id=_uid(),
            stack_id="test-strict",
            drive_type="curiosity",
            intensity=0.5,
            focus_areas=["learning"],
            created_at=_now(),
            updated_at=_now(),
        )
        mock_wb.get_drives.return_value = [existing_drive]

        result_id = k.drive("curiosity", intensity=0.9)

        assert result_id == existing_drive.id
        mock_wb.get_drives.assert_called_once()
        mock_storage.get_drive.assert_not_called()
        mock_wb.update_drive_atomic.assert_called_once()

    def test_filters_by_drive_type(self, mocked_kernle):
        """drive() correctly filters by drive_type when multiple drives exist."""
        k, mock_wb, mock_storage = mocked_kernle

        other_drive = Drive(
            id=_uid(),
            stack_id="test-strict",
            drive_type="growth",
            intensity=0.3,
            focus_areas=[],
            created_at=_now(),
            updated_at=_now(),
        )
        target_drive = Drive(
            id=_uid(),
            stack_id="test-strict",
            drive_type="curiosity",
            intensity=0.5,
            focus_areas=[],
            created_at=_now(),
            updated_at=_now(),
        )
        mock_wb.get_drives.return_value = [other_drive, target_drive]

        result_id = k.drive("curiosity", intensity=0.9)

        assert result_id == target_drive.id
        # Should update the target, not create new
        mock_wb.update_drive_atomic.assert_called_once()


# =========================================================================
# satisfy_drive
# =========================================================================


class TestSatisfyDriveStrictPath:
    """satisfy_drive must read existing drives through _write_backend."""

    def test_uses_write_backend_for_lookup(self, mocked_kernle):
        """satisfy_drive reads drives via _write_backend.get_drives, not _storage.get_drive."""
        k, mock_wb, mock_storage = mocked_kernle

        existing_drive = Drive(
            id=_uid(),
            stack_id="test-strict",
            drive_type="growth",
            intensity=0.8,
            focus_areas=[],
            created_at=_now(),
            updated_at=_now(),
        )
        mock_wb.get_drives.return_value = [existing_drive]

        result = k.satisfy_drive("growth", amount=0.3)

        assert result is True
        mock_wb.get_drives.assert_called_once()
        mock_storage.get_drive.assert_not_called()

    def test_returns_false_when_not_found(self, mocked_kernle):
        """satisfy_drive returns False when drive not found via _write_backend."""
        k, mock_wb, mock_storage = mocked_kernle

        mock_wb.get_drives.return_value = []

        result = k.satisfy_drive("connection")

        assert result is False
        mock_wb.get_drives.assert_called_once()
        mock_storage.get_drive.assert_not_called()

    def test_saves_via_write_backend(self, mocked_kernle):
        """satisfy_drive saves the updated drive through _write_backend.save_drive."""
        k, mock_wb, mock_storage = mocked_kernle

        existing_drive = Drive(
            id=_uid(),
            stack_id="test-strict",
            drive_type="existence",
            intensity=0.6,
            focus_areas=[],
            created_at=_now(),
            updated_at=_now(),
        )
        mock_wb.get_drives.return_value = [existing_drive]

        k.satisfy_drive("existence", amount=0.2)

        mock_wb.update_drive_atomic.assert_called_once()
        saved_drive = mock_wb.update_drive_atomic.call_args[0][0]
        assert saved_drive.intensity == pytest.approx(0.4)


# =========================================================================
# Sanity checks: save paths already correct
# =========================================================================


class TestSavePathsSanity:
    """Sanity checks that save operations continue to route through _write_backend."""

    def test_episode_save_goes_through_write_backend(self, mocked_kernle):
        """episode() saves via _write_backend.save_episode."""
        k, mock_wb, mock_storage = mocked_kernle

        k.episode(
            objective="test objective",
            outcome="completed successfully",
        )

        mock_wb.save_episode.assert_called_once()
        mock_storage.save_episode.assert_not_called()

    def test_goal_save_goes_through_write_backend(self, mocked_kernle):
        """goal() saves via _write_backend.save_goal."""
        k, mock_wb, mock_storage = mocked_kernle

        k.goal(title="test goal")

        mock_wb.save_goal.assert_called_once()
        mock_storage.save_goal.assert_not_called()

    def test_drive_new_save_goes_through_write_backend(self, mocked_kernle):
        """drive() saves new drives via _write_backend.save_drive."""
        k, mock_wb, mock_storage = mocked_kernle

        mock_wb.get_drives.return_value = []

        k.drive("curiosity", intensity=0.5)

        mock_wb.save_drive.assert_called_once()
        mock_storage.save_drive.assert_not_called()
