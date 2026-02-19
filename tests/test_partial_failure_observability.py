"""Tests for partial-failure observability — log shape and recovery guidance.

Verifies that component hook failures log structured fields and that
CLI sync failure paths include recovery guidance.
"""

import logging
from unittest.mock import MagicMock

import pytest

from kernle.stack.sqlite_stack import Stack

# ============================================================================
# Component hook failure log structure
# ============================================================================


class TestPartialFailureLogShape:
    """Component hook failures must log structured extra fields."""

    @pytest.fixture
    def stack(self, tmp_path):
        """Stack with a failing component."""
        s = Stack.from_sqlite(
            stack_id="test-pf",
            db_path=tmp_path / "pf.db",
            components=[],
            enforce_provenance=False,
        )
        return s

    def _get_partial_failure_record(self, caplog):
        """Find the partial-failure warning record in captured logs."""
        for record in caplog.records:
            if record.levelno == logging.WARNING and hasattr(record, "component"):
                return record
        return None

    def test_log_partial_failure_contains_component_field(self, stack, caplog):
        with caplog.at_level(logging.WARNING):
            stack._log_partial_failure("test-component", "on_save", ValueError("boom"))
        assert "test-component" in caplog.text
        assert "on_save" in caplog.text
        assert "boom" in caplog.text
        # Assert structured extra fields directly
        rec = self._get_partial_failure_record(caplog)
        assert rec is not None
        assert rec.component == "test-component"
        assert rec.hook == "on_save"
        assert rec.error_type == "ValueError"

    def test_log_partial_failure_contains_continues_message(self, stack, caplog):
        with caplog.at_level(logging.WARNING):
            stack._log_partial_failure("comp", "on_load", RuntimeError("fail"))
        assert "continues without component" in caplog.text

    def test_dispatch_on_save_uses_structured_log(self, stack, caplog):
        """on_save failure goes through _log_partial_failure with structured extras."""
        mock_component = MagicMock()
        mock_component.name = "test-comp"
        mock_component.on_save.side_effect = RuntimeError("save failed")
        stack._components["test-comp"] = mock_component

        with caplog.at_level(logging.WARNING):
            stack._dispatch_on_save("belief", "id-123", MagicMock())

        rec = self._get_partial_failure_record(caplog)
        assert rec is not None
        assert rec.component == "test-comp"
        assert rec.hook == "on_save"
        assert rec.error_type == "RuntimeError"

    def test_dispatch_on_search_uses_structured_log(self, stack, caplog):
        """on_search failure goes through _log_partial_failure with structured extras."""
        mock_component = MagicMock()
        mock_component.name = "search-comp"
        mock_component.on_search.side_effect = RuntimeError("search failed")
        stack._components["search-comp"] = mock_component

        with caplog.at_level(logging.WARNING):
            results = stack._dispatch_on_search("query", [])

        rec = self._get_partial_failure_record(caplog)
        assert rec is not None
        assert rec.component == "search-comp"
        assert rec.hook == "on_search"
        assert rec.error_type == "RuntimeError"
        assert results == []  # Falls back to original

    def test_dispatch_on_load_uses_structured_log(self, stack, caplog):
        """on_load failure goes through _log_partial_failure with structured extras."""
        mock_component = MagicMock()
        mock_component.name = "load-comp"
        mock_component.on_load.side_effect = RuntimeError("load failed")
        stack._components["load-comp"] = mock_component

        with caplog.at_level(logging.WARNING):
            stack._dispatch_on_load({"key": "value"})

        rec = self._get_partial_failure_record(caplog)
        assert rec is not None
        assert rec.component == "load-comp"
        assert rec.hook == "on_load"
        assert rec.error_type == "RuntimeError"

    def test_maintenance_uses_structured_log(self, stack, caplog):
        """maintenance failure goes through _log_partial_failure with structured extras."""
        mock_component = MagicMock()
        mock_component.name = "maint-comp"
        mock_component.on_maintenance.side_effect = RuntimeError("maint failed")
        stack._components["maint-comp"] = mock_component

        with caplog.at_level(logging.WARNING):
            results = stack.maintenance()

        rec = self._get_partial_failure_record(caplog)
        assert rec is not None
        assert rec.component == "maint-comp"
        assert rec.hook == "on_maintenance"
        assert rec.error_type == "RuntimeError"
        assert results["maint-comp"]["error"] == "maint failed"


# ============================================================================
# CLI sync recovery guidance
# ============================================================================


class TestSyncRecoveryGuidance:
    """CLI sync partial-failure output includes recovery guidance.

    Tests verify the real cmd_sync exception handlers emit recovery tips
    by triggering failures at the HTTP operation level.
    """

    def test_push_failure_includes_tip(self, capsys, monkeypatch):
        """Push failure through real cmd_sync emits recovery tip."""
        from argparse import Namespace
        from unittest.mock import patch

        from kernle.cli.commands.sync import cmd_sync

        args = Namespace(sync_action="push", json=False, force=False, limit=1000)
        mock_k = MagicMock()
        mock_k.stack_id = "test-push-tip"
        mock_k._storage = MagicMock()

        # Build a mock change with real-enough attributes for _build_push_operations
        mock_change = MagicMock()
        mock_change.payload = '{"content": "test data"}'
        mock_change.operation = "insert"
        mock_change.table_name = "notes"
        mock_change.record_id = "test-push-123"
        mock_change.queued_at = "2025-01-01T00:00:00"
        mock_change.id = 1
        mock_k._storage.get_queued_changes.return_value = [mock_change]

        monkeypatch.setenv("KERNLE_BACKEND_URL", "https://fake-backend.example.com")
        monkeypatch.setenv("KERNLE_AUTH_TOKEN", "fake-token")
        monkeypatch.setenv("KERNLE_USER_ID", "fake-user")

        # Mock httpx so that post() raises inside the try block (after _build_push_operations)
        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("connection refused")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with pytest.raises(SystemExit):
                cmd_sync(args, mock_k)

        captured = capsys.readouterr()
        assert "Tip:" in captured.out
        assert "kernle sync push" in captured.out

    def test_pull_failure_includes_tip(self, capsys, monkeypatch):
        """Pull failure through real cmd_sync emits recovery tip."""
        from argparse import Namespace
        from unittest.mock import patch

        from kernle.cli.commands.sync import cmd_sync

        args = Namespace(sync_action="pull", json=False, force=False, full=False)
        mock_k = MagicMock()
        mock_k.stack_id = "test-pull-tip"
        mock_k._storage = MagicMock()
        mock_k._storage.get_last_sync_time.return_value = None

        monkeypatch.setenv("KERNLE_BACKEND_URL", "https://fake-backend.example.com")
        monkeypatch.setenv("KERNLE_AUTH_TOKEN", "fake-token")
        monkeypatch.setenv("KERNLE_USER_ID", "fake-user")

        # Mock httpx import that raises on post (pull uses httpx.post too)
        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = TimeoutError("request timed out")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with pytest.raises(SystemExit):
                cmd_sync(args, mock_k)

        captured = capsys.readouterr()
        assert "Tip:" in captured.out
        assert "kernle sync pull" in captured.out
