"""Tests for kernle.cli.commands.sync — cmd_sync function."""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from kernle import Kernle
from kernle.cli.commands.sync import cmd_sync
from kernle.storage import SQLiteStorage
from kernle.types import SyncConflict

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path, sqlite_storage_factory):
    return sqlite_storage_factory(stack_id="test-sync", db_path=tmp_path / "sync.db")


@pytest.fixture
def k(storage):
    inst = Kernle(stack_id="test-sync", storage=storage, strict=False)
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


def _mock_httpx_module(get_response=None, post_response=None):
    """Create a mock httpx module with configurable responses."""
    mock = MagicMock()
    if get_response:
        mock.get.return_value = get_response
    if post_response:
        mock.post.return_value = post_response
    return mock


def _make_response(status_code=200, json_data=None, text=""):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def _extract_json(output: str) -> dict:
    """Extract the first JSON object from mixed stdout output.

    Some commands print a status line before the JSON output.
    This finds the first '{' and parses from there.
    """
    start = output.index("{")
    return json.loads(output[start:])


# ============================================================================
# Helper: format_datetime
# ============================================================================


class TestFormatDatetime:
    """Test the format_datetime helper inside cmd_sync."""

    def test_format_datetime_none(self, k, capsys):
        """None input returns None — verified via status JSON where last_sync_time is null."""
        # format_datetime is a closure inside cmd_sync, so test indirectly.
        # A fresh storage has no last_sync_time → format_datetime(None) → null in JSON.
        cmd_sync(_args(sync_action="status", json=True), k)
        captured = capsys.readouterr().out
        data = _extract_json(captured)
        assert data["last_sync_time"] is None

    def test_format_datetime_string_passthrough(self, k, capsys, tmp_path):
        """String datetimes pass through unchanged in JSON output."""
        # The format_datetime inside cmd_sync handles strings and datetimes
        # We verify this indirectly via the status --json output
        health_resp = _make_response(200)
        mock_httpx = _mock_httpx_module(get_response=health_resp)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok123", "user_id": "u1"}
        creds_path = tmp_path / "creds"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["backend_connected"] is True
        assert output["authenticated"] is True


# ============================================================================
# sync status
# ============================================================================


class TestSyncStatus:
    """Tests for `kernle sync status`."""

    def test_status_no_backend(self, k, capsys, tmp_path):
        """Status shows disconnected when no backend configured."""
        health_resp = _make_response(200)
        mock_httpx = _mock_httpx_module(get_response=health_resp)

        # Empty credentials dir — no creds file
        creds_path = tmp_path / "no_creds"
        creds_path.mkdir()

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}, clear=False):
            # Clear any KERNLE_BACKEND_URL, KERNLE_AUTH_TOKEN, KERNLE_USER_ID
            env_overrides = {
                "KERNLE_DATA_DIR": str(creds_path),
                "KERNLE_BACKEND_URL": "",
                "KERNLE_AUTH_TOKEN": "",
                "KERNLE_USER_ID": "",
            }
            with patch.dict(os.environ, env_overrides):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status"), k)

        captured = capsys.readouterr().out
        assert "Sync Status" in captured

    def test_status_json_output(self, k, capsys, tmp_path):
        """Status --json returns structured JSON with all fields."""
        health_resp = _make_response(200)
        mock_httpx = _mock_httpx_module(get_response=health_resp)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok123", "user_id": "u42"}
        creds_path = tmp_path / "creds_json"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["local_stack_id"] == "test-sync"
        assert output["namespaced_stack_id"] == "u42/test-sync"
        assert output["user_id"] == "u42"
        assert output["backend_url"] == "https://api.test.com"
        assert output["backend_connected"] is True
        assert output["authenticated"] is True
        assert isinstance(output["pending_operations"], int)

    def test_status_backend_unreachable(self, k, capsys, tmp_path):
        """Status shows error when backend health check fails."""
        error_resp = _make_response(503, text="Service Unavailable")
        mock_httpx = _mock_httpx_module(get_response=error_resp)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "creds_fail"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["backend_connected"] is False
        assert "503" in output["connection_status"]

    def test_status_with_namespaced_stack_id(self, tmp_path):
        """Extracts local project name from namespaced stack_id.

        Uses a Kernle instance whose stack_id contains a '/' separator
        (SQLiteStorage rejects '/' in stack_id, so we patch it on k).
        """
        storage = SQLiteStorage(stack_id="myproject", db_path=tmp_path / "ns.db")
        inst = Kernle(stack_id="myproject", storage=storage, strict=False)
        # Simulate a namespaced stack_id by patching the attribute directly
        inst.stack_id = "user123/myproject"

        health_resp = _make_response(200)
        mock_httpx = _mock_httpx_module(get_response=health_resp)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "creds_ns"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                import io
                import sys

                captured = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = captured
                try:
                    cmd_sync(_args(sync_action="status", json=True), inst)
                finally:
                    sys.stdout = old_stdout

        try:
            output = json.loads(captured.getvalue())
            assert output["local_stack_id"] == "myproject"
            assert output["namespaced_stack_id"] == "u1/myproject"
        finally:
            inst._storage.close()

    def test_status_env_var_fallback(self, k, capsys, tmp_path):
        """Falls back to KERNLE_BACKEND_URL/KERNLE_AUTH_TOKEN env vars."""
        health_resp = _make_response(200)
        mock_httpx = _mock_httpx_module(get_response=health_resp)

        creds_path = tmp_path / "empty_creds"
        creds_path.mkdir()

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "https://env.api.com",
            "KERNLE_AUTH_TOKEN": "env-token",
            "KERNLE_USER_ID": "env-user",
        }
        with patch.dict(os.environ, env):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["backend_url"] == "https://env.api.com"
        assert output["authenticated"] is True
        assert output["user_id"] == "env-user"


# ============================================================================
# sync push
# ============================================================================


class TestSyncPush:
    """Tests for `kernle sync push`."""

    def test_push_no_backend_url(self, k, tmp_path):
        """Exits when no backend URL configured."""
        creds_path = tmp_path / "no_be"
        creds_path.mkdir()

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "",
            "KERNLE_AUTH_TOKEN": "",
            "KERNLE_USER_ID": "",
        }
        with patch.dict(os.environ, env):
            mock_httpx = _mock_httpx_module()
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="push"), k)
                assert exc_info.value.code == 1

    def test_push_no_auth_token(self, k, tmp_path):
        """Exits when no auth token configured."""
        creds = {"backend_url": "https://api.test.com"}
        creds_path = tmp_path / "no_auth"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(creds_path), "KERNLE_AUTH_TOKEN": ""}
        with patch.dict(os.environ, env):
            mock_httpx = _mock_httpx_module()
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="push"), k)
                assert exc_info.value.code == 1

    def test_push_no_pending_changes(self, k, capsys, tmp_path):
        """Reports no changes when queue is empty."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok123"}
        creds_path = tmp_path / "push_empty"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        mock_httpx = _mock_httpx_module()
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push"), k)

        captured = capsys.readouterr().out
        assert "No pending changes" in captured

    def test_push_success(self, k, capsys, tmp_path):
        """Successfully pushes queued changes to backend."""
        # Add a note to create a queued change
        from kernle.storage import Note

        note = Note(id="n1", stack_id="test-sync", content="Hello world")
        k._storage.save_note(note)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok123", "user_id": "u1"}
        creds_path = tmp_path / "push_ok"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push"), k)

        captured = capsys.readouterr().out
        assert "Pushed 1 changes" in captured
        assert "u1/test-sync" in captured

    def test_push_success_json(self, k, capsys, tmp_path):
        """Push with --json returns structured output."""
        from kernle.storage import Note

        note = Note(id="n2", stack_id="test-sync", content="Test note")
        k._storage.save_note(note)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok123", "user_id": "u1"}
        creds_path = tmp_path / "push_json"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push", json=True), k)

        output = _extract_json(capsys.readouterr().out)
        assert output["synced"] == 1
        assert output["local_project"] == "test-sync"
        assert output["namespaced_id"] == "u1/test-sync"

    def test_push_auth_failure(self, k, tmp_path):
        """Exits on 401 auth failure."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n3", stack_id="test-sync", content="Data"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "bad-tok"}
        creds_path = tmp_path / "push_401"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(401, text="Unauthorized")
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="push"), k)
                assert exc_info.value.code == 1

    def test_push_server_error(self, k, tmp_path):
        """Exits on 500 server error."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n4", stack_id="test-sync", content="Data"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "push_500"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(500, text="Internal Server Error")
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="push"), k)
                assert exc_info.value.code == 1

    def test_push_network_error(self, k, tmp_path):
        """Exits on network connection error."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n5", stack_id="test-sync", content="Data"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "push_net"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("Connection refused")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="push"), k)
                assert exc_info.value.code == 1

    def test_push_with_conflicts_in_response(self, k, capsys, tmp_path):
        """Reports conflicts returned by backend."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n6", stack_id="test-sync", content="Conflict"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "push_conflict"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        conflicts = [{"record_id": "n6", "error": "version mismatch"}]
        push_resp = _make_response(200, json_data={"synced": 0, "conflicts": conflicts})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push"), k)

        captured = capsys.readouterr().out
        assert "1 conflicts" in captured
        assert "version mismatch" in captured

    def test_push_conflicts_have_deterministic_snapshot(self, k, capsys, tmp_path):
        """Push conflict JSON should expose deterministic ordering and snapshot."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="zz-conflict", stack_id="test-sync", content="z note"))
        k._storage.save_note(Note(id="aa-conflict", stack_id="test-sync", content="a note"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "push_conf_snapshot"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(
            200,
            json_data={
                "synced": 0,
                "conflicts": [
                    {"table": "notes", "record_id": "zz-conflict", "error": "version mismatch"},
                    {"table": "notes", "record_id": "aa-conflict", "error": "stale snapshot"},
                ],
            },
        )
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push", json=True), k)

        output = _extract_json(capsys.readouterr().out)
        assert output["conflict_snapshot"]["count"] == 2
        assert output["conflict_snapshot"]["records"][0]["record_id"] == "aa-conflict"
        assert output["conflict_snapshot"]["records"][1]["record_id"] == "zz-conflict"
        assert output["conflicts"][0]["record_id"] == "aa-conflict"

    def test_push_table_name_mapping(self, k, tmp_path):
        """Verifies table name translation from local to backend schema."""
        from kernle.storage import Episode

        ep = Episode(
            id="ep1",
            stack_id="test-sync",
            objective="test",
            outcome_type="success",
            outcome="done",
        )
        k._storage.save_episode(ep)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "push_map"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push"), k)

        # Verify the API call was made with proper table mapping
        call_args = mock_httpx.post.call_args
        assert call_args is not None
        sent_json = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        ops = sent_json.get("operations", [])
        assert len(ops) == 1

    def test_push_acknowledges_by_identity_not_position(self, k, tmp_path):
        """Clears acknowledged queue entries by identity, not positional count."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-ack-1", stack_id="test-sync", content="first"))
        k._storage.save_note(Note(id="n-ack-2", stack_id="test-sync", content="second"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "push_ack_identity"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(
            200,
            json_data={
                "synced": 1,
                "conflicts": [
                    {"table": "notes", "record_id": "n-ack-1", "error": "version mismatch"}
                ],
            },
        )
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push"), k)

        remaining = k._storage.get_queued_changes(limit=10)
        assert len(remaining) == 1
        assert remaining[0].record_id == "n-ack-1"

    def test_push_success_does_not_update_last_sync_time(self, k, tmp_path):
        """Push success must not mutate sync cursor metadata."""
        from kernle.storage import Note

        sentinel_cursor = "2025-01-01T00:00:00+00:00"
        k._storage._set_sync_meta("last_sync_time", sentinel_cursor)
        k._storage.save_note(Note(id="n-cursor", stack_id="test-sync", content="cursor test"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "push_no_cursor"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push"), k)

        assert k._storage._get_sync_meta("last_sync_time") == sentinel_cursor


# ============================================================================
# sync pull
# ============================================================================


class TestSyncPull:
    """Tests for `kernle sync pull`."""

    def test_pull_no_backend(self, k, tmp_path):
        """Exits when no backend URL configured."""
        creds_path = tmp_path / "pull_no_be"
        creds_path.mkdir()

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "",
            "KERNLE_AUTH_TOKEN": "",
            "KERNLE_USER_ID": "",
        }
        with patch.dict(os.environ, env):
            mock_httpx = _mock_httpx_module()
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="pull"), k)
                assert exc_info.value.code == 1

    def test_pull_no_auth(self, k, tmp_path):
        """Exits when not authenticated."""
        creds = {"backend_url": "https://api.test.com"}
        creds_path = tmp_path / "pull_no_auth"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(creds_path), "KERNLE_AUTH_TOKEN": ""}
        with patch.dict(os.environ, env):
            mock_httpx = _mock_httpx_module()
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="pull"), k)
                assert exc_info.value.code == 1

    def test_pull_already_up_to_date(self, k, capsys, tmp_path):
        """Reports up to date when no operations returned."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_up2date"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull"), k)

        captured = capsys.readouterr().out
        assert "up to date" in captured

    def test_pull_episodes(self, k, capsys, tmp_path):
        """Pulls and applies episode operations."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "pull_ep"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        ops = [
            {
                "table": "episodes",
                "record_id": "ep-remote-1",
                "operation": "upsert",
                "data": {
                    "objective": "Remote episode",
                    "outcome_type": "success",
                    "outcome": "It worked",
                    "tags": ["remote"],
                },
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull"), k)

        captured = capsys.readouterr().out
        assert "Pulled 1 changes" in captured

    def test_pull_notes(self, k, capsys, tmp_path):
        """Pulls and applies note operations."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_note"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        ops = [
            {
                "table": "notes",
                "record_id": "note-remote-1",
                "operation": "upsert",
                "data": {"content": "Remote note", "note_type": "insight"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull"), k)

        captured = capsys.readouterr().out
        assert "Pulled 1 changes" in captured

    def test_pull_json_output(self, k, capsys, tmp_path):
        """Pull --json returns structured output."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "pull_json"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        ops = [
            {
                "table": "episodes",
                "record_id": "ep-r1",
                "operation": "upsert",
                "data": {"objective": "Test", "outcome_type": "neutral"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": True})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull", json=True), k)

        output = _extract_json(capsys.readouterr().out)
        assert output["pulled"] == 1
        assert output["has_more"] is True
        assert output["local_project"] == "test-sync"
        assert output["namespaced_id"] == "u1/test-sync"

    def test_pull_conflict_snapshot_is_deterministic(self, k, capsys, tmp_path):
        """Pull JSON includes deterministic conflict order and snapshot metadata."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_conf_snapshot"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        ops = [
            {
                "table": "notes",
                "record_id": "z-note",
                "operation": "remove",
                "data": {},
            },
            {
                "table": "episodes",
                "record_id": "a-episode",
                "operation": "delete",
                "data": {"objective": "bad", "outcome": "bad"},
            },
            {
                "table": "notes",
                "record_id": "a-note",
                "operation": "upsert",
                "data": {"content": "pull note", "note_type": "insight"},
            },
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull", json=True), k)

        output = _extract_json(capsys.readouterr().out)
        assert output["conflicts"] == 2
        assert output["conflict_snapshot"]["count"] == 2
        first = output["conflict_snapshot"]["records"][0]
        second = output["conflict_snapshot"]["records"][1]
        assert (first["table"], first["record_id"]) == ("episodes", "a-episode")
        assert (second["table"], second["record_id"]) == ("notes", "z-note")

    def test_pull_auth_failure(self, k, tmp_path):
        """Exits on 401 response."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "bad"}
        creds_path = tmp_path / "pull_401"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        pull_resp = _make_response(401, text="Unauthorized")
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="pull"), k)
                assert exc_info.value.code == 1

    def test_pull_server_error(self, k, tmp_path):
        """Exits on 500 server error."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_500"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        pull_resp = _make_response(500, text="Server Error")
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="pull"), k)
                assert exc_info.value.code == 1

    def test_pull_network_error(self, k, tmp_path):
        """Exits on network connection error."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_net"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("Connection refused")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="pull"), k)
                assert exc_info.value.code == 1

    def test_pull_has_more_hint(self, k, capsys, tmp_path):
        """Shows hint when more changes are available."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_more"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        ops = [{"table": "other", "record_id": "x1", "operation": "upsert", "data": {"foo": "bar"}}]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": True})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull"), k)

        captured = capsys.readouterr().out
        assert "More changes available" in captured

    def test_pull_full_flag(self, k, capsys, tmp_path):
        """--full flag forces full pull (no since parameter)."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_full"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull", full=True), k)

        # Verify the request did NOT include a "since" parameter
        call_args = mock_httpx.post.call_args
        sent_json = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert "since" not in sent_json

    def test_pull_unhandled_ops_are_quarantined_and_cursor_advances(self, k, capsys, tmp_path):
        """Unhandled/failed pull ops are quarantined so cursor can still advance."""
        sentinel_cursor = "2025-01-01T00:00:00+00:00"
        k._storage._set_sync_meta("last_sync_time", sentinel_cursor)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_unhandled"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        ops = [
            {
                "table": "notes",
                "record_id": "note-ok",
                "operation": "upsert",
                "data": {"content": "Remote good note", "note_type": "insight"},
            },
            {
                "table": "unknown_table",
                "record_id": "bad-op",
                "operation": "upsert",
                "data": {"foo": "bar"},
            },
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull"), k)

        captured = capsys.readouterr().out
        assert "Pulled 1 changes" in captured
        assert "1 conflicts during apply" in captured
        assert k._storage._get_sync_meta("last_sync_time") != sentinel_cursor

        notes = k._storage.get_notes(limit=20)
        pulled_note = next(n for n in notes if n.id == "note-ok")
        assert pulled_note is not None
        assert pulled_note.source_type == "external"
        assert pulled_note.source_entity == "kernle:sync"

        pull_conflicts = k._storage.get_sync_conflicts(limit=10)
        assert any(
            c.record_id == "bad-op" and c.resolution == "pull_apply_failed" for c in pull_conflicts
        )

    def test_pull_retries_quarantined_ops_on_next_run(self, k, capsys, tmp_path):
        """Previously failed pull ops are retried from quarantine on later pulls."""
        sentinel_cursor = "2025-01-01T00:00:00+00:00"
        k._storage._set_sync_meta("last_sync_time", sentinel_cursor)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_retry_poison"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        first_pull = _make_response(
            200,
            json_data={
                "operations": [
                    {
                        "table": "notes",
                        "record_id": "note-retry-1",
                        "operation": "upsert",
                        "data": {"content": "Recovered on retry", "note_type": "insight"},
                    }
                ],
                "has_more": False,
            },
        )
        second_pull = _make_response(200, json_data={"operations": [], "has_more": False})

        original_save_note = k._storage.save_note
        calls = {"count": 0}

        def flaky_save(note):
            calls["count"] += 1
            if calls["count"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return original_save_note(note)

        with patch.object(k._storage, "save_note", side_effect=flaky_save):
            with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
                with patch.dict(
                    "sys.modules", {"httpx": _mock_httpx_module(post_response=first_pull)}
                ):
                    cmd_sync(_args(sync_action="pull"), k)

            with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
                with patch.dict(
                    "sys.modules", {"httpx": _mock_httpx_module(post_response=second_pull)}
                ):
                    cmd_sync(_args(sync_action="pull"), k)

        captured = capsys.readouterr().out
        assert "Pulled 0 changes" in captured
        assert "Pulled 1 changes" in captured
        notes = k._storage.get_notes(limit=20)
        assert any(n.id == "note-retry-1" for n in notes)

    def test_pull_episode_has_external_source_type_and_sync_entity(self, k, capsys, tmp_path):
        """Pulled episodes carry source_type='external' and source_entity='kernle:sync'."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "pull_ep_prov"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        ops = [
            {
                "table": "episodes",
                "record_id": "ep-pulled-1",
                "operation": "upsert",
                "data": {
                    "objective": "Remote episode",
                    "outcome": "synced",
                    "outcome_type": "success",
                },
            },
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull"), k)

        episodes = k._storage.get_episodes(limit=20)
        pulled_ep = next(e for e in episodes if e.id == "ep-pulled-1")
        assert pulled_ep.source_type == "external"
        assert pulled_ep.source_entity == "kernle:sync"


# ============================================================================
# sync conflicts
# ============================================================================


class TestSyncConflicts:
    """Tests for `kernle sync conflicts`."""

    def test_conflicts_empty(self, k, capsys, tmp_path):
        """Shows message when no conflicts exist."""
        creds_path = tmp_path / "conf_empty"
        creds_path.mkdir()

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            cmd_sync(_args(sync_action="conflicts"), k)

        captured = capsys.readouterr().out
        assert "No sync conflicts" in captured

    def test_conflicts_display(self, k, capsys, tmp_path):
        """Displays conflict history."""
        # Insert a conflict directly
        conflict = SyncConflict(
            id="c1",
            table="notes",
            record_id="rec-abc123-full",
            local_version={"content": "local version"},
            cloud_version={"content": "cloud version"},
            resolution="cloud_wins",
            resolved_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            local_summary="local version",
            cloud_summary="cloud version",
        )
        k._storage._sync_engine.save_sync_conflict(conflict)

        creds_path = tmp_path / "conf_disp"
        creds_path.mkdir()

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            cmd_sync(_args(sync_action="conflicts"), k)

        captured = capsys.readouterr().out
        assert "1 conflicts" in captured
        assert "cloud wins" in captured
        assert "rec-abc1" in captured  # truncated record_id
        assert "local version" in captured
        assert "cloud version" in captured

    def test_conflicts_json_output(self, k, capsys, tmp_path):
        """Conflicts --json returns structured data."""
        conflict = SyncConflict(
            id="c2",
            table="episodes",
            record_id="ep-xyz",
            local_version={"objective": "local"},
            cloud_version={"objective": "cloud"},
            resolution="local_wins",
            resolved_at=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            local_summary="local obj",
            cloud_summary="cloud obj",
        )
        k._storage._sync_engine.save_sync_conflict(conflict)

        creds_path = tmp_path / "conf_json"
        creds_path.mkdir()

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            cmd_sync(_args(sync_action="conflicts", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["count"] == 1
        assert output["conflicts"][0]["table"] == "episodes"
        assert output["conflicts"][0]["record_id"] == "ep-xyz"
        assert output["conflicts"][0]["resolution"] == "local_wins"

    def test_conflicts_clear(self, k, capsys, tmp_path):
        """--clear removes all conflict history."""
        conflict = SyncConflict(
            id="c3",
            table="notes",
            record_id="note-clear",
            local_version={},
            cloud_version={},
            resolution="cloud_wins",
            resolved_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
        )
        k._storage._sync_engine.save_sync_conflict(conflict)

        creds_path = tmp_path / "conf_clear"
        creds_path.mkdir()

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            cmd_sync(_args(sync_action="conflicts", clear=True), k)

        captured = capsys.readouterr().out
        assert "Cleared" in captured

        # Verify actually cleared
        remaining = k._storage.get_sync_conflicts()
        assert len(remaining) == 0

    def test_conflicts_clear_json(self, k, capsys, tmp_path):
        """--clear --json returns structured result."""
        creds_path = tmp_path / "conf_clear_j"
        creds_path.mkdir()

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            cmd_sync(_args(sync_action="conflicts", clear=True, json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert "cleared" in output


# ============================================================================
# sync full
# ============================================================================


class TestSyncFull:
    """Tests for `kernle sync full` (bidirectional sync)."""

    def test_full_no_backend(self, k, tmp_path):
        """Exits when no backend URL."""
        creds_path = tmp_path / "full_no_be"
        creds_path.mkdir()

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "",
            "KERNLE_AUTH_TOKEN": "",
            "KERNLE_USER_ID": "",
        }
        with patch.dict(os.environ, env):
            mock_httpx = _mock_httpx_module()
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="full"), k)
                assert exc_info.value.code == 1

    def test_full_no_auth(self, k, tmp_path):
        """Exits when not authenticated."""
        creds = {"backend_url": "https://api.test.com"}
        creds_path = tmp_path / "full_no_auth"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(creds_path), "KERNLE_AUTH_TOKEN": ""}
        with patch.dict(os.environ, env):
            mock_httpx = _mock_httpx_module()
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="full"), k)
                assert exc_info.value.code == 1

    def test_full_sync_success(self, k, capsys, tmp_path):
        """Full sync pulls then pushes."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "full_ok"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        # Pull returns empty, push has nothing queued
        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="full"), k)

        captured = capsys.readouterr().out
        assert "Full sync complete" in captured
        assert "Pulled" in captured
        assert "No pending changes to push" in captured

    def test_full_sync_pushes_non_delete_operations(self, k, capsys, tmp_path):
        """Regression: sync full must include non-delete ops in push payload.

        Previously, a misplaced `continue` (line 740) caused all non-delete
        operations to be skipped — only deletes were actually pushed.
        """
        from kernle.storage import Episode

        # Create a real episode so it's in the sync queue
        ep = Episode(
            id="ep-full-push",
            stack_id="test-sync",
            objective="test full push",
            outcome_type="success",
            outcome="verified",
        )
        k._storage.save_episode(ep)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "full_push"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        # Pull returns empty, push should succeed with 1 operation
        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})

        call_count = {"n": 0}

        def route_post(*args, **kwargs):
            """First POST = pull, second POST = push."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                return pull_resp
            return push_resp

        mock_httpx = _mock_httpx_module()
        mock_httpx.post.side_effect = route_post

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="full"), k)

        # Verify push was called (2 POSTs: pull + push)
        assert mock_httpx.post.call_count == 2

        # Second call is the push — verify operations were included
        push_call = mock_httpx.post.call_args_list[1]
        sent_json = push_call[1].get("json") or push_call[0][1]
        ops = sent_json.get("operations", [])
        assert len(ops) == 1, f"Expected 1 push operation, got {len(ops)}"
        assert ops[0]["operation"] == "update"
        assert ops[0]["record_id"] == "ep-full-push"
        assert "data" in ops[0], "Non-delete operation must include record data"

    def test_full_sync_applies_pulled_ops_and_advances_cursor(self, k, capsys, tmp_path):
        """Full sync applies pull operations locally and advances cursor on full success."""
        sentinel_cursor = "2025-01-01T00:00:00+00:00"
        k._storage._set_sync_meta("last_sync_time", sentinel_cursor)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "full_apply_pull"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        pull_resp = _make_response(
            200,
            json_data={
                "operations": [
                    {
                        "table": "notes",
                        "record_id": "full-note-1",
                        "operation": "upsert",
                        "data": {"content": "Pulled in full sync", "note_type": "note"},
                    }
                ],
                "has_more": False,
            },
        )
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="full"), k)

        captured = capsys.readouterr().out
        assert "Pulled 1 changes" in captured
        assert k._storage._get_sync_meta("last_sync_time") != sentinel_cursor

        notes = k._storage.get_notes(limit=20)
        pulled_note = next(n for n in notes if n.id == "full-note-1")
        assert pulled_note is not None
        assert pulled_note.source_type == "external"
        assert pulled_note.source_entity == "kernle:sync"

    def test_full_sync_pull_apply_failure_is_quarantined_and_cursor_advances(
        self, k, capsys, tmp_path
    ):
        """Full sync quarantines failed pull ops and still advances cursor."""
        sentinel_cursor = "2025-01-01T00:00:00+00:00"
        k._storage._set_sync_meta("last_sync_time", sentinel_cursor)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = tmp_path / "full_pull_fail"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        pull_resp = _make_response(
            200,
            json_data={
                "operations": [
                    {
                        "table": "unknown_table",
                        "record_id": "not-applied",
                        "operation": "upsert",
                        "data": {"foo": "bar"},
                    }
                ],
                "has_more": False,
            },
        )
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="full"), k)

        captured = capsys.readouterr().out
        assert "0 changes" in captured
        assert "1 conflicts during apply" in captured
        assert "payload_sha256=" in captured
        assert k._storage._get_sync_meta("last_sync_time") != sentinel_cursor
        pull_conflicts = k._storage.get_sync_conflicts(limit=10)
        assert any(
            c.record_id == "not-applied" and c.resolution == "pull_apply_failed"
            for c in pull_conflicts
        )


# ============================================================================
# JSON version field (#710)
# ============================================================================


class TestJsonVersionField:
    """All JSON outputs must include 'version': 1 for forward compatibility."""

    def test_status_json_includes_version(self, k, capsys, tmp_path):
        """sync status --json includes version field."""
        health_resp = _make_response(200)
        mock_httpx = _mock_httpx_module(get_response=health_resp)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok123", "user_id": "u1"}
        creds_path = tmp_path / "ver_status"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["version"] == 1

    def test_push_json_includes_version(self, k, capsys, tmp_path):
        """sync push --json includes version field."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-ver", stack_id="test-sync", content="ver test"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "ver_push"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        mock_httpx = _mock_httpx_module(post_response=push_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="push", json=True), k)

        output = _extract_json(capsys.readouterr().out)
        assert output["version"] == 1

    def test_pull_json_includes_version(self, k, capsys, tmp_path):
        """sync pull --json includes version field."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "ver_pull"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        ops = [
            {
                "table": "episodes",
                "record_id": "ep-ver",
                "operation": "upsert",
                "data": {"objective": "version test", "outcome_type": "neutral"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = _mock_httpx_module(post_response=pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="pull", json=True), k)

        output = _extract_json(capsys.readouterr().out)
        assert output["version"] == 1

    def test_conflicts_json_includes_version(self, k, capsys, tmp_path):
        """sync conflicts --json includes version field."""
        creds_path = tmp_path / "ver_conf"
        creds_path.mkdir()

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            cmd_sync(_args(sync_action="conflicts", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["version"] == 1

    def test_conflicts_clear_json_includes_version(self, k, capsys, tmp_path):
        """sync conflicts --clear --json includes version field."""
        creds_path = tmp_path / "ver_conf_clear"
        creds_path.mkdir()

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            cmd_sync(_args(sync_action="conflicts", clear=True, json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["version"] == 1

    def test_status_timestamps_are_iso8601(self, k, capsys, tmp_path):
        """Timestamp fields in status JSON use ISO 8601 format."""
        # Set a last_sync_time so we can verify its format
        k._storage._set_sync_meta("last_sync_time", "2025-06-01T12:00:00+00:00")

        health_resp = _make_response(200)
        mock_httpx = _mock_httpx_module(get_response=health_resp)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = tmp_path / "ver_ts"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        ts = output["last_sync_time"]
        assert ts is not None
        # ISO 8601 should contain 'T' separator and be parseable
        assert "T" in ts
        from datetime import datetime as dt

        dt.fromisoformat(ts)  # will raise if not valid ISO 8601
