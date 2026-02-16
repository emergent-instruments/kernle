"""Tests for strict-mode error propagation in checkpoint and sync operations.

Verifies that:
- strict=True re-raises persistence errors after logging
- strict=False (default) catches and warns without raising
- CLI output reflects sync errors instead of showing unconditional success
"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kernle.core import Kernle


def _make_storage():
    """Create a mock storage that satisfies Kernle init."""
    storage = MagicMock()
    storage.get_pending_sync_count.return_value = 0
    storage.is_online.return_value = False
    return storage


def _make_kernle(tmp_path, *, strict, storage=None):
    """Create a Kernle instance with the given strict setting.

    For strict=True tests, we patch _write_backend to avoid the
    SQLiteStack requirement when using mock storage.
    """
    if storage is None:
        storage = _make_storage()
    k = Kernle(
        "test-agent",
        storage=storage,
        checkpoint_dir=tmp_path / "cp",
        strict=strict,
    )
    return k, storage


class TestCheckpointEpisodeSaveStrict:
    """Episode save within checkpoint -- strict vs. permissive."""

    def test_checkpoint_episode_save_failure_strict_raises(self, tmp_path):
        """strict=True: episode save failure propagates the exception."""
        storage = _make_storage()
        storage.save_episode.side_effect = sqlite3.OperationalError("disk I/O error")

        k, _ = _make_kernle(tmp_path, strict=True, storage=storage)

        # Patch _write_backend so checkpoint can reach save_episode on the mock
        with patch.object(
            type(k), "_write_backend", new_callable=PropertyMock, return_value=storage
        ):
            with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
                k.checkpoint(task="test task")

    def test_checkpoint_episode_save_failure_permissive_warns(self, tmp_path):
        """strict=False: episode save failure is caught and logged as warning."""
        storage = _make_storage()
        storage.save_episode.side_effect = sqlite3.OperationalError("disk I/O error")

        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        # Should NOT raise
        result = k.checkpoint(task="test task")
        assert result["current_task"] == "test task"

    def test_checkpoint_episode_save_failure_permissive_logs_warning(self, tmp_path, caplog):
        """strict=False: episode save failure produces a warning log."""
        import logging

        storage = _make_storage()
        storage.save_episode.side_effect = OSError("write failed")

        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        with caplog.at_level(logging.WARNING):
            k.checkpoint(task="test task")

        assert any("Failed to save checkpoint to database" in rec.message for rec in caplog.records)

    def test_checkpoint_episode_save_ioerror_strict_raises(self, tmp_path):
        """strict=True: IOError propagates."""
        storage = _make_storage()
        storage.save_episode.side_effect = IOError("unexpected FS error")

        k, _ = _make_kernle(tmp_path, strict=True, storage=storage)

        with patch.object(
            type(k), "_write_backend", new_callable=PropertyMock, return_value=storage
        ):
            with pytest.raises(IOError, match="unexpected FS error"):
                k.checkpoint(task="test task")


class TestCheckpointEpisodeProvenance:
    """Checkpoint episode carries correct source_type and source_entity."""

    def test_checkpoint_episode_has_observation_source_type(self, tmp_path):
        """Checkpoint sidecar episode has source_type='observation'."""
        storage = _make_storage()
        captured_episodes = []
        storage.save_episode.side_effect = lambda ep: captured_episodes.append(ep) or "ep-id"

        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        k.checkpoint(task="provenance test")

        assert len(captured_episodes) == 1
        ep = captured_episodes[0]
        assert ep.source_type == "observation"

    def test_checkpoint_episode_has_kernle_checkpoint_source_entity(self, tmp_path):
        """Checkpoint sidecar episode has source_entity='kernle:checkpoint'."""
        storage = _make_storage()
        captured_episodes = []
        storage.save_episode.side_effect = lambda ep: captured_episodes.append(ep) or "ep-id"

        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        k.checkpoint(task="entity test")

        assert len(captured_episodes) == 1
        ep = captured_episodes[0]
        assert ep.source_entity == "kernle:checkpoint"


class TestCheckpointBootExportStrict:
    """Boot file export within checkpoint -- strict vs. permissive."""

    def test_checkpoint_boot_export_failure_strict_raises(self, tmp_path):
        """strict=True: boot file export failure propagates the exception."""
        storage = _make_storage()
        k, _ = _make_kernle(tmp_path, strict=True, storage=storage)

        with patch.object(
            type(k), "_write_backend", new_callable=PropertyMock, return_value=storage
        ):
            with patch.object(k, "_export_boot_file", side_effect=OSError("permission denied")):
                with pytest.raises(OSError, match="permission denied"):
                    k.checkpoint(task="test task")

    def test_checkpoint_boot_export_failure_permissive_warns(self, tmp_path, caplog):
        """strict=False: boot file export failure is caught and logged."""
        import logging

        storage = _make_storage()
        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        with patch.object(k, "_export_boot_file", side_effect=OSError("permission denied")):
            with caplog.at_level(logging.WARNING):
                result = k.checkpoint(task="test task")

        assert result["current_task"] == "test task"
        assert any(
            "Failed to export boot file on checkpoint" in rec.message for rec in caplog.records
        )


class TestSyncBeforeLoadStrict:
    """_sync_before_load -- strict vs. permissive."""

    def test_sync_before_load_failure_strict_raises(self, tmp_path):
        """strict=True: pull failure propagates the exception."""
        storage = _make_storage()
        storage.is_online.return_value = True
        storage.pull_changes.side_effect = ConnectionError("network down")

        k, _ = _make_kernle(tmp_path, strict=True, storage=storage)

        with pytest.raises(ConnectionError, match="network down"):
            k._sync_before_load()

    def test_sync_before_load_failure_permissive_returns_errors(self, tmp_path):
        """strict=False: pull failure is caught and returned in errors list."""
        storage = _make_storage()
        storage.is_online.return_value = True
        storage.pull_changes.side_effect = ConnectionError("network down")

        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        result = k._sync_before_load()
        assert any("network down" in err for err in result["errors"])
        assert result["attempted"] is True

    def test_sync_before_load_timeout_strict_raises(self, tmp_path):
        """strict=True: TimeoutError propagates."""
        storage = _make_storage()
        storage.is_online.return_value = True
        storage.pull_changes.side_effect = TimeoutError("request timed out")

        k, _ = _make_kernle(tmp_path, strict=True, storage=storage)

        with pytest.raises(TimeoutError, match="request timed out"):
            k._sync_before_load()


class TestSyncAfterCheckpointStrict:
    """_sync_after_checkpoint -- strict vs. permissive."""

    def test_sync_after_checkpoint_failure_strict_raises(self, tmp_path):
        """strict=True: sync failure propagates the exception."""
        storage = _make_storage()
        storage.is_online.return_value = True
        storage.sync.side_effect = OSError("sync filesystem error")

        k, _ = _make_kernle(tmp_path, strict=True, storage=storage)

        with pytest.raises(OSError, match="sync filesystem error"):
            k._sync_after_checkpoint()

    def test_sync_after_checkpoint_failure_permissive_returns_errors(self, tmp_path):
        """strict=False: sync failure is caught and returned in errors list."""
        storage = _make_storage()
        storage.is_online.return_value = True
        storage.sync.side_effect = ValueError("bad data")

        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        result = k._sync_after_checkpoint()
        assert any("bad data" in err for err in result["errors"])
        assert result["attempted"] is True

    def test_sync_after_checkpoint_connection_error_strict_raises(self, tmp_path):
        """strict=True: ConnectionError propagates."""
        storage = _make_storage()
        storage.is_online.return_value = True
        storage.sync.side_effect = ConnectionError("refused")

        k, _ = _make_kernle(tmp_path, strict=True, storage=storage)

        with pytest.raises(ConnectionError, match="refused"):
            k._sync_after_checkpoint()


class TestCLICheckpointSyncWarning:
    """CLI output should reflect sync errors instead of clean checkmark."""

    def test_cli_checkpoint_shows_warning_on_sync_error(self, tmp_path, capsys):
        """When sync errors are present, CLI shows warning instead of clean success."""
        from kernle.cli.commands.memory import cmd_checkpoint

        storage = _make_storage()
        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        sync_result = {
            "attempted": True,
            "pushed": 0,
            "conflicts": 0,
            "errors": ["Connection refused", "Timeout waiting for response"],
        }
        checkpoint_result = {
            "current_task": "test task",
            "pending": [],
            "_sync": sync_result,
        }

        args = SimpleNamespace(
            checkpoint_action="save",
            task="test task",
            pending=None,
            context=None,
            progress=None,
            next=None,
            blocker=None,
            no_sync=False,
            sync=False,
        )

        with patch.object(k, "checkpoint", return_value=checkpoint_result):
            cmd_checkpoint(args, k)

        captured = capsys.readouterr()
        # Should show warning indicator, not clean checkmark
        assert "\u26a0" in captured.out
        assert "Checkpoint saved" in captured.out
        # Should show ALL errors, not just first one truncated
        assert "Connection refused" in captured.out
        assert "Timeout waiting for response" in captured.out

    def test_cli_checkpoint_shows_clean_success_when_no_errors(self, tmp_path, capsys):
        """When no sync errors, CLI shows clean checkmark."""
        from kernle.cli.commands.memory import cmd_checkpoint

        storage = _make_storage()
        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        checkpoint_result = {
            "current_task": "test task",
            "pending": [],
            "_sync": {
                "attempted": True,
                "pushed": 3,
                "conflicts": 0,
                "errors": [],
            },
        }

        args = SimpleNamespace(
            checkpoint_action="save",
            task="test task",
            pending=None,
            context=None,
            progress=None,
            next=None,
            blocker=None,
            no_sync=False,
            sync=False,
        )

        with patch.object(k, "checkpoint", return_value=checkpoint_result):
            cmd_checkpoint(args, k)

        captured = capsys.readouterr()
        assert "\u2713 Checkpoint saved" in captured.out

    def test_cli_checkpoint_no_sync_shows_clean_success(self, tmp_path, capsys):
        """When sync was not attempted, CLI shows clean checkmark."""
        from kernle.cli.commands.memory import cmd_checkpoint

        storage = _make_storage()
        k, _ = _make_kernle(tmp_path, strict=False, storage=storage)

        checkpoint_result = {
            "current_task": "test task",
            "pending": [],
        }

        args = SimpleNamespace(
            checkpoint_action="save",
            task="test task",
            pending=None,
            context=None,
            progress=None,
            next=None,
            blocker=None,
            no_sync=False,
            sync=False,
        )

        with patch.object(k, "checkpoint", return_value=checkpoint_result):
            cmd_checkpoint(args, k)

        captured = capsys.readouterr()
        assert "\u2713 Checkpoint saved" in captured.out
