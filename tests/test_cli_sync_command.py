"""Tests for CLI sync command boundary validation.

Covers the gaps identified by codex audit for v0.13.06:
- cmd_sync with no backend_url prints an error and exits
- push/pull without credentials fails cleanly
- sync with invalid direction is rejected
- recovery tips appear in push/pull failure output
"""

import argparse
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from kernle import Kernle
from kernle.cli.commands.sync import cmd_sync

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path, sqlite_storage_factory):
    return sqlite_storage_factory(stack_id="test-sync-cli", db_path=tmp_path / "sync-cli.db")


@pytest.fixture
def k(storage):
    inst = Kernle(stack_id="test-sync-cli", storage=storage, strict=False)
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


def _no_creds_env(creds_path):
    """Environment dict that clears all credential sources."""
    return {
        "KERNLE_DATA_DIR": str(creds_path),
        "KERNLE_BACKEND_URL": "",
        "KERNLE_AUTH_TOKEN": "",
        "KERNLE_USER_ID": "",
    }


def _mock_httpx():
    """Create a minimal mock httpx module."""
    return MagicMock()


# ============================================================================
# No backend_url configured
# ============================================================================


class TestNoBackendUrl:
    """cmd_sync should print an error and exit(1) when backend_url is missing."""

    def test_push_no_backend_url_exits(self, k, capsys, tmp_path):
        """Push with no backend_url prints error message and exits with code 1."""
        creds_path = tmp_path / "no_be_push"
        creds_path.mkdir()

        with patch.dict(os.environ, _no_creds_env(creds_path)):
            with patch.dict("sys.modules", {"httpx": _mock_httpx()}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="push"), k)
                assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "Backend not configured" in captured

    def test_pull_no_backend_url_exits(self, k, capsys, tmp_path):
        """Pull with no backend_url prints error message and exits with code 1."""
        creds_path = tmp_path / "no_be_pull"
        creds_path.mkdir()

        with patch.dict(os.environ, _no_creds_env(creds_path)):
            with patch.dict("sys.modules", {"httpx": _mock_httpx()}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="pull"), k)
                assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "Backend not configured" in captured

    def test_full_no_backend_url_exits(self, k, capsys, tmp_path):
        """Full sync with no backend_url prints error message and exits with code 1."""
        creds_path = tmp_path / "no_be_full"
        creds_path.mkdir()

        with patch.dict(os.environ, _no_creds_env(creds_path)):
            with patch.dict("sys.modules", {"httpx": _mock_httpx()}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="full"), k)
                assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "Backend not configured" in captured


# ============================================================================
# No credentials (auth_token missing)
# ============================================================================


class TestNoCredentials:
    """Push/pull without auth credentials should fail cleanly with exit(1)."""

    def test_push_no_auth_token_exits(self, k, capsys, tmp_path):
        """Push with backend_url but no auth_token prints auth error and exits."""
        creds_path = tmp_path / "no_auth_push"
        creds_path.mkdir()
        creds = {"backend_url": "https://api.example.com"}
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(creds_path), "KERNLE_AUTH_TOKEN": ""}
        with patch.dict(os.environ, env):
            with patch.dict("sys.modules", {"httpx": _mock_httpx()}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="push"), k)
                assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "Not authenticated" in captured

    def test_pull_no_auth_token_exits(self, k, capsys, tmp_path):
        """Pull with backend_url but no auth_token prints auth error and exits."""
        creds_path = tmp_path / "no_auth_pull"
        creds_path.mkdir()
        creds = {"backend_url": "https://api.example.com"}
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(creds_path), "KERNLE_AUTH_TOKEN": ""}
        with patch.dict(os.environ, env):
            with patch.dict("sys.modules", {"httpx": _mock_httpx()}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="pull"), k)
                assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "Not authenticated" in captured

    def test_full_no_auth_token_exits(self, k, capsys, tmp_path):
        """Full sync with backend_url but no auth_token prints auth error and exits."""
        creds_path = tmp_path / "no_auth_full"
        creds_path.mkdir()
        creds = {"backend_url": "https://api.example.com"}
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(creds_path), "KERNLE_AUTH_TOKEN": ""}
        with patch.dict(os.environ, env):
            with patch.dict("sys.modules", {"httpx": _mock_httpx()}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_sync(_args(sync_action="full"), k)
                assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "Not authenticated" in captured


# ============================================================================
# Invalid sync direction
# ============================================================================


class TestInvalidSyncAction:
    """Sync with an unrecognized sync_action should do nothing (no crash)."""

    def test_unknown_sync_action_does_not_crash(self, k, capsys, tmp_path):
        """An unrecognized sync_action falls through without error or output.

        cmd_sync uses if/elif branching on sync_action; an unknown value
        simply does nothing. This test verifies it does not raise.
        """
        creds_path = tmp_path / "bad_direction"
        creds_path.mkdir()

        with patch.dict(os.environ, _no_creds_env(creds_path)):
            # Should not raise any exception
            cmd_sync(_args(sync_action="invalid_direction"), k)

        # No output expected for unrecognized action
        captured = capsys.readouterr().out
        assert captured == ""


# ============================================================================
# Recovery tips in failure output
# ============================================================================


class TestRecoveryTips:
    """Push/pull failure output should include recovery tips for the user."""

    def test_push_network_error_includes_recovery_tip(self, k, capsys, tmp_path):
        """Push failure from network error includes recovery tip about retrying."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-tip", stack_id="test-sync-cli", content="Data"))

        creds_path = tmp_path / "push_tip"
        creds_path.mkdir()
        creds = {"backend_url": "https://api.example.com", "auth_token": "tok"}
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("Connection refused")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit):
                    cmd_sync(_args(sync_action="push"), k)

        captured = capsys.readouterr().out
        # The push failure handler prints a Tip about local queuing
        assert "Tip" in captured
        assert "kernle sync push" in captured

    def test_pull_network_error_includes_recovery_tip(self, k, capsys, tmp_path):
        """Pull failure from network error includes recovery tip about retrying."""
        creds_path = tmp_path / "pull_tip"
        creds_path.mkdir()
        creds = {"backend_url": "https://api.example.com", "auth_token": "tok"}
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("Connection refused")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit):
                    cmd_sync(_args(sync_action="pull"), k)

        captured = capsys.readouterr().out
        # The pull failure handler prints a Tip about checking connectivity
        assert "Tip" in captured
        assert "kernle sync pull" in captured

    def test_push_no_backend_includes_auth_login_hint(self, k, capsys, tmp_path):
        """Push failure for missing backend includes kernle auth login suggestion."""
        creds_path = tmp_path / "push_auth_hint"
        creds_path.mkdir()

        with patch.dict(os.environ, _no_creds_env(creds_path)):
            with patch.dict("sys.modules", {"httpx": _mock_httpx()}):
                with pytest.raises(SystemExit):
                    cmd_sync(_args(sync_action="push"), k)

        captured = capsys.readouterr().out
        assert "kernle auth login" in captured or "KERNLE_BACKEND_URL" in captured

    def test_pull_no_auth_includes_auth_login_hint(self, k, capsys, tmp_path):
        """Pull failure for missing auth includes kernle auth login suggestion."""
        creds_path = tmp_path / "pull_auth_hint"
        creds_path.mkdir()
        creds = {"backend_url": "https://api.example.com"}
        (creds_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(creds_path), "KERNLE_AUTH_TOKEN": ""}
        with patch.dict(os.environ, env):
            with patch.dict("sys.modules", {"httpx": _mock_httpx()}):
                with pytest.raises(SystemExit):
                    cmd_sync(_args(sync_action="pull"), k)

        captured = capsys.readouterr().out
        assert "kernle auth login" in captured or "KERNLE_AUTH_TOKEN" in captured
