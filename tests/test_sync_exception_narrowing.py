"""Tests for narrowed exception handling in kernle.cli.commands.sync.

Verifies that:
- Conflict persistence catches sqlite3/OS errors, not all exceptions.
- Programming errors (AttributeError, etc.) propagate instead of being swallowed.
- Log level is WARNING (not DEBUG) for conflict persistence failures.
- Structured extra fields (table, record_id, error_type) are present on log records.
- _apply_single_pull_operation catches data errors cleanly.
"""

import argparse
import json
import logging
import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from kernle import Kernle
from kernle.cli.commands.sync import cmd_sync

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path, sqlite_storage_factory):
    return sqlite_storage_factory(stack_id="test-exc", db_path=tmp_path / "exc.db")


@pytest.fixture
def k(storage):
    inst = Kernle(stack_id="test-exc", storage=storage, strict=False)
    yield inst


def _args(**kwargs):
    """Build an argparse.Namespace with defaults for sync commands."""
    defaults = {
        "command": "sync",
        "sync_action": "status",
        "json": False,
        "limit": 100,
        "full": False,
        "clear": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_response(status_code=200, json_data=None, text=""):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def _mock_httpx_module(get_response=None, post_response=None):
    """Create a mock httpx module with configurable responses."""
    mock = MagicMock()
    if get_response:
        mock.get.return_value = get_response
    if post_response:
        mock.post.return_value = post_response
    return mock


def _setup_creds(tmp_path, subdir="creds"):
    """Create credentials dir and file, return path."""
    creds = {"backend_url": "https://api.test.com", "auth_token": "tok123", "user_id": "u1"}
    creds_path = tmp_path / subdir
    creds_path.mkdir(exist_ok=True)
    (creds_path / "credentials.json").write_text(json.dumps(creds))
    return creds_path


def _run_sync(creds_path, mock_httpx, k, args):
    """Run cmd_sync with standard env/mock patching."""
    with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            cmd_sync(args, k)


# ============================================================================
# Pull conflict persistence
# ============================================================================


class TestPullConflictPersistence:
    """Tests for _save_pull_apply_conflict exception narrowing."""

    def test_pull_conflict_persistence_catches_sqlite_error(self, k, capsys, tmp_path, caplog):
        """sqlite3.OperationalError during save_sync_conflict is caught and logged as WARNING."""
        creds_path = _setup_creds(tmp_path, "pull_sqlite_err")

        # Pull returns an operation for an unhandled table so it triggers a conflict
        # that goes through _save_pull_apply_conflict
        ops = [
            {
                "table": "unknown_table",
                "record_id": "bad-rec-1",
                "operation": "upsert",
                "data": {"foo": "bar"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        # Make save_sync_conflict raise sqlite3.OperationalError
        def raise_sqlite_error(conflict):
            raise sqlite3.OperationalError("disk I/O error")

        with patch.object(k._storage, "save_sync_conflict", side_effect=raise_sqlite_error):
            with caplog.at_level(logging.WARNING):
                _run_sync(creds_path, mock_httpx, k, _args(sync_action="pull"))

        # The WARNING should have been logged (not swallowed silently at DEBUG)
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "persist pull apply conflict" in r.message
        ]
        assert len(warning_records) >= 1, (
            f"Expected WARNING log for pull conflict persistence failure, "
            f"got: {[r.message for r in caplog.records]}"
        )

    def test_pull_conflict_persistence_does_not_catch_programming_error(
        self, k, capsys, tmp_path, caplog
    ):
        """Non-sqlite, non-OS errors (like AttributeError) are not silently caught by the
        narrowed handler.  They propagate out to the top-level dispatch which prints an
        error and exits, rather than being quietly logged as a WARNING."""
        creds_path = _setup_creds(tmp_path, "pull_prog_err")

        ops = [
            {
                "table": "unknown_table",
                "record_id": "bad-rec-2",
                "operation": "upsert",
                "data": {"foo": "bar"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        def raise_attribute_error(conflict):
            raise AttributeError("something is very wrong")

        with patch.object(k._storage, "save_sync_conflict", side_effect=raise_attribute_error):
            with caplog.at_level(logging.WARNING):
                # The error propagates past the narrowed handler and hits the outer
                # user-facing catch-all which calls sys.exit(1).
                with pytest.raises(SystemExit):
                    _run_sync(creds_path, mock_httpx, k, _args(sync_action="pull"))

        # The narrowed handler should NOT have caught this -- no WARNING about
        # "persist pull apply conflict" should appear in the logs.
        conflict_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "persist pull apply conflict" in r.message
        ]
        assert (
            len(conflict_warnings) == 0
        ), "AttributeError should NOT be caught by the narrowed handler"

        # The error should have surfaced in stdout via the top-level handler
        captured = capsys.readouterr().out
        assert "something is very wrong" in captured


# ============================================================================
# Push conflict persistence
# ============================================================================


class TestPushConflictPersistence:
    """Tests for _save_push_apply_conflict exception narrowing."""

    def test_push_conflict_persistence_logs_warning_not_debug(self, k, capsys, tmp_path, caplog):
        """Conflict persistence failures log at WARNING level, not DEBUG."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-push-warn", stack_id="test-exc", content="push test"))

        creds_path = _setup_creds(tmp_path, "push_warn")

        # Backend returns a conflict so _save_push_apply_conflict is called
        conflicts = [{"table": "notes", "record_id": "n-push-warn", "error": "version mismatch"}]
        push_resp = _make_response(200, json_data={"synced": 0, "conflicts": conflicts})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        def raise_sqlite_error(conflict):
            raise sqlite3.OperationalError("database is locked")

        with patch.object(k._storage, "save_sync_conflict", side_effect=raise_sqlite_error):
            with caplog.at_level(logging.DEBUG):
                _run_sync(creds_path, mock_httpx, k, _args(sync_action="push"))

        # Verify WARNING level, not DEBUG
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "persist push conflict" in r.message
        ]
        debug_records = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "persist push conflict" in r.message
        ]
        assert len(warning_records) >= 1, "Expected at least one WARNING log for push conflict"
        assert len(debug_records) == 0, "Should not log push conflict failures at DEBUG"

    def test_push_conflict_persistence_does_not_catch_programming_error(
        self, k, capsys, tmp_path, caplog
    ):
        """AttributeError during push conflict persistence is not caught by the
        narrowed handler.  It propagates to the outer dispatch which exits."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-push-attr", stack_id="test-exc", content="test"))

        creds_path = _setup_creds(tmp_path, "push_attr_err")

        conflicts = [{"table": "notes", "record_id": "n-push-attr", "error": "stale"}]
        push_resp = _make_response(200, json_data={"synced": 0, "conflicts": conflicts})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        def raise_attribute_error(conflict):
            raise AttributeError("broken attribute access")

        with patch.object(k._storage, "save_sync_conflict", side_effect=raise_attribute_error):
            with caplog.at_level(logging.WARNING):
                with pytest.raises(SystemExit):
                    _run_sync(creds_path, mock_httpx, k, _args(sync_action="push"))

        # The narrowed handler should NOT have caught this
        conflict_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "persist push conflict" in r.message
        ]
        assert (
            len(conflict_warnings) == 0
        ), "AttributeError should NOT be caught by the narrowed handler"

        captured = capsys.readouterr().out
        assert "broken attribute access" in captured


# ============================================================================
# Structured log fields
# ============================================================================


class TestStructuredLogFields:
    """Tests that extra dict fields are present on conflict failure log records."""

    def test_structured_log_fields_on_pull_conflict_failure(self, k, capsys, tmp_path, caplog):
        """Extra fields (table, record_id, error_type) are present on log records."""
        creds_path = _setup_creds(tmp_path, "pull_extra")

        ops = [
            {
                "table": "unknown_table",
                "record_id": "structured-rec",
                "operation": "upsert",
                "data": {"foo": "bar"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        def raise_os_error(conflict):
            raise OSError("Permission denied")

        with patch.object(k._storage, "save_sync_conflict", side_effect=raise_os_error):
            with caplog.at_level(logging.WARNING):
                _run_sync(creds_path, mock_httpx, k, _args(sync_action="pull"))

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "persist pull apply conflict" in r.message
        ]
        assert len(warning_records) >= 1

        record = warning_records[0]
        assert record.table == "unknown_table"
        assert record.record_id == "structured-rec"
        assert record.error_type == "OSError"

    def test_structured_log_fields_on_push_conflict_failure(self, k, capsys, tmp_path, caplog):
        """Extra fields on push conflict persistence failure logs."""
        from kernle.storage import Note

        k._storage.save_note(
            Note(id="n-extra-push", stack_id="test-exc", content="extra fields test")
        )

        creds_path = _setup_creds(tmp_path, "push_extra")

        conflicts = [{"table": "notes", "record_id": "n-extra-push", "error": "version conflict"}]
        push_resp = _make_response(200, json_data={"synced": 0, "conflicts": conflicts})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        def raise_sqlite_error(conflict):
            raise sqlite3.IntegrityError("UNIQUE constraint failed")

        with patch.object(k._storage, "save_sync_conflict", side_effect=raise_sqlite_error):
            with caplog.at_level(logging.WARNING):
                _run_sync(creds_path, mock_httpx, k, _args(sync_action="push"))

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "persist push conflict" in r.message
        ]
        assert len(warning_records) >= 1

        record = warning_records[0]
        assert record.table == "notes"
        assert record.record_id == "n-extra-push"
        assert record.error_type == "IntegrityError"


# ============================================================================
# _apply_single_pull_operation narrowing
# ============================================================================


class TestApplyPullOperationNarrowing:
    """Tests for narrowed exception handling in _apply_single_pull_operation."""

    def test_apply_pull_operation_catches_data_errors(self, k, capsys, tmp_path):
        """Malformed data errors (KeyError, ValueError, TypeError) are caught cleanly."""
        creds_path = _setup_creds(tmp_path, "apply_data_err")

        # Force a TypeError inside the episode construction by making save_episode
        # raise TypeError (simulating malformed data)
        ops = [
            {
                "table": "episodes",
                "record_id": "ep-bad-data",
                "operation": "upsert",
                "data": {"objective": "test"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        def raise_type_error(ep):
            raise TypeError("unexpected type in field")

        with patch.object(k._storage, "save_episode", side_effect=raise_type_error):
            # Should NOT raise -- the error should be caught and turned into a conflict
            _run_sync(creds_path, mock_httpx, k, _args(sync_action="pull"))

        captured = capsys.readouterr().out
        assert "conflicts" in captured.lower() or "Pulled 0 changes" in captured

    def test_apply_pull_operation_catches_sqlite_error(self, k, capsys, tmp_path):
        """sqlite3 errors during apply are caught and result in conflict, not crash."""
        creds_path = _setup_creds(tmp_path, "apply_sqlite_err")

        ops = [
            {
                "table": "notes",
                "record_id": "note-db-fail",
                "operation": "upsert",
                "data": {"content": "This will fail", "note_type": "note"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        def raise_db_error(note):
            raise sqlite3.OperationalError("table agent_notes has no column named xyz")

        with patch.object(k._storage, "save_note", side_effect=raise_db_error):
            _run_sync(creds_path, mock_httpx, k, _args(sync_action="pull"))

        captured = capsys.readouterr().out
        assert "conflicts" in captured.lower() or "Pulled 0 changes" in captured

    def test_apply_pull_operation_quarantines_programming_errors(self, k, capsys, tmp_path):
        """Programming errors like AttributeError are caught and quarantined
        by SyncEngine delegation rather than crashing the entire pull.

        Post-unification, all exceptions during apply are caught and reported
        as failed operations (quarantined for retry), rather than propagating
        and aborting the pull mid-batch.
        """
        creds_path = _setup_creds(tmp_path, "apply_prog_err")

        ops = [
            {
                "table": "notes",
                "record_id": "note-prog-err",
                "operation": "upsert",
                "data": {"content": "Will trigger bug", "note_type": "note"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        def raise_attribute_error(note):
            raise AttributeError("'NoneType' object has no attribute 'foo'")

        with patch.object(k._storage, "save_note", side_effect=raise_attribute_error):
            # Post-unification: errors are caught and quarantined, not propagated
            _run_sync(creds_path, mock_httpx, k, _args(sync_action="pull"))

        # Verify the error surfaced in output (not silently swallowed)
        captured = capsys.readouterr().out
        assert "NoneType" in captured
        assert "conflicts during apply" in captured

    def test_apply_pull_operation_error_message_includes_type(self, k, capsys, tmp_path):
        """Error messages from caught exceptions include the exception type name."""
        creds_path = _setup_creds(tmp_path, "apply_err_type")

        ops = [
            {
                "table": "episodes",
                "record_id": "ep-val-err",
                "operation": "upsert",
                "data": {"objective": "test"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        def raise_value_error(ep):
            raise ValueError("invalid outcome_type")

        with patch.object(k._storage, "save_episode", side_effect=raise_value_error):
            _run_sync(creds_path, mock_httpx, k, _args(sync_action="pull", json=True))

        output_text = capsys.readouterr().out
        # Find the JSON portion
        start = output_text.index("{")
        output = json.loads(output_text[start:])
        # The failed operation should include the error detail
        failed = output.get("failed_operations", [])
        assert len(failed) >= 1
        assert "invalid outcome_type" in failed[0].get("error", "")
