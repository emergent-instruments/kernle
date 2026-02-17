"""Tests for CLI raw command module."""

from argparse import Namespace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from kernle.cli.commands.raw import cmd_raw, resolve_raw_id


class TestResolveRawId:
    """Test raw ID resolution."""

    def test_exact_match(self):
        """Exact ID should resolve directly."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc123", "content": "test"}

        result = resolve_raw_id(k, "abc123")
        assert result == "abc123"
        k.get_raw.assert_called_once_with("abc123")

    def test_prefix_match_single(self):
        """Single prefix match should resolve to full ID."""
        k = MagicMock()
        k.get_raw.return_value = None  # No exact match
        match_entry = MagicMock()
        match_entry.id = "abc123456789"
        k._storage.find_raw_by_prefix.return_value = [match_entry]

        result = resolve_raw_id(k, "abc")
        assert result == "abc123456789"

    def test_prefix_match_ambiguous(self):
        """Multiple prefix matches should raise error."""
        k = MagicMock()
        k.get_raw.return_value = None
        e1, e2 = MagicMock(), MagicMock()
        e1.id = "abc123456789"
        e2.id = "abc987654321"
        k._storage.find_raw_by_prefix.return_value = [e1, e2]

        with pytest.raises(ValueError, match="Ambiguous ID"):
            resolve_raw_id(k, "abc")

    def test_no_match(self):
        """No matches should raise error."""
        k = MagicMock()
        k.get_raw.return_value = None
        k._storage.find_raw_by_prefix.return_value = []

        with pytest.raises(ValueError, match="not found"):
            resolve_raw_id(k, "nonexistent")

    def test_prefix_match_many(self):
        """Many prefix matches should show truncated list."""
        k = MagicMock()
        k.get_raw.return_value = None
        entries = []
        for i in range(6):
            e = MagicMock()
            e.id = f"abc{i:010d}"
            entries.append(e)
        k._storage.find_raw_by_prefix.return_value = entries

        with pytest.raises(ValueError, match=r"Ambiguous ID.*\.\.\."):
            resolve_raw_id(k, "abc")

    def test_prefix_match_uses_db_level_query(self):
        """resolve_raw_id calls find_raw_by_prefix for database-level prefix search."""
        k = MagicMock()
        k.get_raw.return_value = None
        match_entry = MagicMock()
        match_entry.id = "def456789012"
        k._storage.find_raw_by_prefix.return_value = [match_entry]

        result = resolve_raw_id(k, "def")
        assert result == "def456789012"
        k._storage.find_raw_by_prefix.assert_called_once_with("def", limit=6)

    def test_fallback_to_list_raw_without_prefix_method(self):
        """Gracefully falls back to list_raw if find_raw_by_prefix is unavailable."""
        k = MagicMock()
        k.get_raw.return_value = None
        # Remove the find_raw_by_prefix method
        del k._storage.find_raw_by_prefix
        k.list_raw.return_value = [
            {"id": "abc123456789", "content": "test"},
        ]

        result = resolve_raw_id(k, "abc")
        assert result == "abc123456789"


class TestCmdRawCapture:
    """Test raw capture command."""

    def test_capture_basic(self, capsys):
        """Basic capture should work."""
        k = MagicMock()
        k.raw.return_value = "raw-id-12345678"

        args = Namespace(
            raw_action="capture",
            content="test content",
            tags=None,
            source=None,
            quiet=False,
            stdin=False,
        )

        cmd_raw(args, k)

        k.raw.assert_called_once()
        captured = capsys.readouterr()
        assert "✓ Raw entry captured" in captured.out

    def test_capture_with_tags(self, capsys):
        """Capture with tags should pass them through."""
        k = MagicMock()
        k.raw.return_value = "raw-id-12345678"

        args = Namespace(
            raw_action="capture",
            content="test content",
            tags="tag1,tag2",
            source=None,
            quiet=False,
            stdin=False,
        )

        cmd_raw(args, k)

        call_kwargs = k.raw.call_args
        # Tags are folded into blob text (tags param removed from raw API)
        blob_value = call_kwargs[1].get("blob", call_kwargs[0][0] if call_kwargs[0] else "")
        assert "tag1" in blob_value
        assert "tag2" in blob_value

    def test_capture_with_source(self, capsys):
        """Capture with custom source."""
        k = MagicMock()
        k.raw.return_value = "raw-id-12345678"

        args = Namespace(
            raw_action="capture",
            content="test content",
            tags=None,
            source="voice",
            quiet=False,
            stdin=False,
        )

        cmd_raw(args, k)

        call_kwargs = k.raw.call_args
        assert call_kwargs[1]["source"] == "voice"
        captured = capsys.readouterr()
        assert "Source: voice" in captured.out

    def test_capture_default_action(self, capsys):
        """Default action (None) should capture."""
        k = MagicMock()
        k.raw.return_value = "raw-id-12345678"

        args = Namespace(
            raw_action=None,
            content="test content",
            tags=None,
            source=None,
            quiet=False,
            stdin=False,
        )

        cmd_raw(args, k)

        k.raw.assert_called_once()
        captured = capsys.readouterr()
        assert "✓ Raw entry captured" in captured.out

    def test_capture_quiet_mode(self, capsys):
        """Quiet mode should only print ID."""
        k = MagicMock()
        k.raw.return_value = "raw-id-12345678"

        args = Namespace(
            raw_action="capture",
            content="test content",
            tags=None,
            source="hook-session-end",
            quiet=True,
            stdin=False,
        )

        cmd_raw(args, k)

        k.raw.assert_called_once()
        captured = capsys.readouterr()
        # Quiet mode should only print the short ID
        assert captured.out.strip() == "raw-id-1"
        assert "✓" not in captured.out

    def test_capture_stdin_mode(self, capsys, monkeypatch):
        """Stdin mode should read from stdin."""
        import io

        k = MagicMock()
        k.raw.return_value = "raw-id-12345678"

        # Mock stdin
        monkeypatch.setattr("sys.stdin", io.StringIO("content from stdin"))

        args = Namespace(
            raw_action="capture",
            content=None,  # No content argument
            tags=None,
            source="hook-session-end",
            quiet=False,
            stdin=True,
        )

        cmd_raw(args, k)

        # Verify content came from stdin (new API uses keyword arguments)
        call_args = k.raw.call_args
        assert "content from stdin" in call_args.kwargs.get("blob", "")
        captured = capsys.readouterr()
        assert "✓ Raw entry captured" in captured.out

    def test_capture_stdin_empty(self, capsys, monkeypatch):
        """Empty stdin should show error."""
        import io

        k = MagicMock()

        # Mock empty stdin
        monkeypatch.setattr("sys.stdin", io.StringIO(""))

        args = Namespace(
            raw_action="capture",
            content=None,
            tags=None,
            source=None,
            quiet=False,
            stdin=True,
        )

        cmd_raw(args, k)

        # Should not call k.raw
        k.raw.assert_not_called()
        captured = capsys.readouterr()
        assert "No content received from stdin" in captured.out

    def test_capture_stdin_quiet_error(self, capsys, monkeypatch):
        """Quiet mode with empty stdin should produce no output."""
        import io

        k = MagicMock()

        # Mock empty stdin
        monkeypatch.setattr("sys.stdin", io.StringIO(""))

        args = Namespace(
            raw_action="capture",
            content=None,
            tags=None,
            source=None,
            quiet=True,
            stdin=True,
        )

        cmd_raw(args, k)

        k.raw.assert_not_called()
        captured = capsys.readouterr()
        # Quiet mode should suppress error output
        assert captured.out.strip() == ""

    def test_capture_no_content_no_stdin(self, capsys):
        """Missing content without stdin should show error."""
        k = MagicMock()

        args = Namespace(
            raw_action="capture",
            content=None,
            tags=None,
            source=None,
            quiet=False,
            stdin=False,
        )

        cmd_raw(args, k)

        k.raw.assert_not_called()
        captured = capsys.readouterr()
        assert "Content is required" in captured.out


class TestCmdRawList:
    """Test raw list command."""

    def test_list_empty(self, capsys):
        """Empty list should show message."""
        k = MagicMock()
        k.list_raw.return_value = []

        args = Namespace(
            raw_action="list",
            unprocessed=False,
            processed=False,
            limit=20,
            json=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "No raw entries found" in captured.out

    def test_list_unprocessed_filter(self):
        """Unprocessed filter should be passed."""
        k = MagicMock()
        k.list_raw.return_value = []

        args = Namespace(
            raw_action="list",
            unprocessed=True,
            processed=False,
            limit=20,
            json=False,
        )

        cmd_raw(args, k)

        k.list_raw.assert_called_once_with(processed=False, limit=20)

    def test_list_processed_filter(self):
        """Processed filter should be passed."""
        k = MagicMock()
        k.list_raw.return_value = []

        args = Namespace(
            raw_action="list",
            unprocessed=False,
            processed=True,
            limit=20,
            json=False,
        )

        cmd_raw(args, k)

        k.list_raw.assert_called_once_with(processed=True, limit=20)

    def test_list_with_entries(self, capsys):
        """List with entries should display them."""
        k = MagicMock()
        k.list_raw.return_value = [
            {
                "id": "abc12345",
                "content": "This is a test raw entry with lots of content that might be long",
                "timestamp": "2026-01-28T10:00:00Z",
                "processed": False,
                "tags": ["test", "example"],
                "processed_into": None,
            },
            {
                "id": "def67890",
                "content": "Already processed entry",
                "timestamp": "2026-01-27T09:00:00Z",
                "processed": True,
                "tags": [],
                "processed_into": ["episode:ep123"],
            },
        ]

        args = Namespace(
            raw_action="list",
            unprocessed=False,
            processed=False,
            limit=20,
            json=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Raw Entries" in captured.out
        assert "2 total" in captured.out
        assert "1 unprocessed" in captured.out
        assert "abc12345" in captured.out
        assert "test, example" in captured.out
        assert "episode:ep123" in captured.out

    def test_list_json(self, capsys):
        """List JSON output."""
        k = MagicMock()
        k.list_raw.return_value = [
            {
                "id": "abc123",
                "content": "test",
                "timestamp": "2026-01-28",
                "processed": False,
                "tags": [],
            }
        ]

        args = Namespace(
            raw_action="list",
            unprocessed=False,
            processed=False,
            limit=20,
            json=True,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert '"id"' in captured.out
        assert '"abc123"' in captured.out


class TestCmdRawShow:
    """Test raw show command."""

    def test_show_not_found(self, capsys):
        """Show non-existent ID should error."""
        k = MagicMock()
        k.get_raw.return_value = None
        k.list_raw.return_value = []

        args = Namespace(
            raw_action="show",
            id="nonexistent",
            json=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_show_found(self, capsys):
        """Show existing entry should display details."""
        k = MagicMock()
        # First call for resolve_raw_id, second for actual get
        k.get_raw.side_effect = [
            {"id": "abc12345", "content": "test"},  # resolve check
            {
                "id": "abc12345",
                "content": "This is the full content\nWith multiple lines",
                "timestamp": "2026-01-28T10:00:00Z",
                "source": "cli",
                "processed": False,
                "tags": ["important"],
                "processed_into": None,
            },
        ]

        args = Namespace(
            raw_action="show",
            id="abc12345",
            json=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Raw Entry: abc12345" in captured.out
        assert "Unprocessed" in captured.out
        assert "cli" in captured.out
        assert "important" in captured.out
        assert "This is the full content" in captured.out

    def test_show_processed_entry(self, capsys):
        """Show processed entry should display details."""
        k = MagicMock()
        k.get_raw.side_effect = [
            {"id": "abc12345", "content": "test"},
            {
                "id": "abc12345",
                "content": "Content",
                "timestamp": "2026-01-28T10:00:00Z",
                "source": "cli",
                "processed": True,
                "tags": [],
                "processed_into": ["episode:ep123", "note:n456"],
            },
        ]

        args = Namespace(
            raw_action="show",
            id="abc12345",
            json=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Processed" in captured.out
        assert "episode:ep123" in captured.out

    def test_show_json(self, capsys):
        """Show JSON output."""
        k = MagicMock()
        k.get_raw.side_effect = [
            {"id": "abc12345", "content": "test"},
            {
                "id": "abc12345",
                "content": "Content",
                "timestamp": "2026-01-28T10:00:00Z",
                "source": "cli",
                "processed": False,
                "tags": [],
                "processed_into": None,
            },
        ]

        args = Namespace(
            raw_action="show",
            id="abc12345",
            json=True,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert '"id"' in captured.out
        assert '"abc12345"' in captured.out


class TestCmdRawProcess:
    """Test raw process command."""

    def test_process_success(self, capsys):
        """Successful processing."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc12345", "content": "test"}
        k.process_raw.return_value = "ep123456"

        args = Namespace(
            raw_action="process",
            id="abc12345",
            type="episode",
            objective="Test objective",
            outcome="Test outcome",
        )

        cmd_raw(args, k)

        k.process_raw.assert_called_once()
        captured = capsys.readouterr()
        assert "✓ Processed" in captured.out
        assert "episode:ep12345" in captured.out

    def test_process_not_found(self, capsys):
        """Process non-existent entry."""
        k = MagicMock()
        k.get_raw.return_value = None
        k.list_raw.return_value = []

        args = Namespace(
            raw_action="process",
            id="nonexistent",
            type="episode",
            objective=None,
            outcome=None,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "not found" in captured.out

    def test_process_batch(self, capsys):
        """Process multiple entries."""
        k = MagicMock()
        k.get_raw.side_effect = [
            {"id": "abc12345", "content": "test1"},
            {"id": "def67890", "content": "test2"},
        ]
        k.process_raw.side_effect = ["ep1", "ep2"]

        args = Namespace(
            raw_action="process",
            id="abc12345,def67890",
            type="episode",
            objective=None,
            outcome=None,
        )

        cmd_raw(args, k)

        assert k.process_raw.call_count == 2
        captured = capsys.readouterr()
        assert "Processed 2/2" in captured.out


class TestCmdRawReview:
    """Test raw review command."""

    def test_review_empty(self, capsys):
        """Review with no unprocessed entries."""
        k = MagicMock()
        k.list_raw.return_value = []

        args = Namespace(
            raw_action="review",
            limit=10,
            json=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "No unprocessed raw entries" in captured.out
        assert "memory is up to date" in captured.out

    def test_review_with_entries(self, capsys):
        """Review with unprocessed entries."""
        k = MagicMock()
        k.list_raw.return_value = [
            {
                "id": "abc12345",
                "content": "I learned that testing is important",
                "timestamp": "2026-01-28T10:00:00Z",
                "tags": ["work"],
            },
            {
                "id": "def67890",
                "content": "I decided to use Python for this project",
                "timestamp": "2026-01-27T09:00:00Z",
                "tags": [],
            },
        ]

        args = Namespace(
            raw_action="review",
            limit=10,
            json=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Raw Entry Review" in captured.out
        assert "2 unprocessed entries" in captured.out
        assert "episode (contains learning)" in captured.out
        assert "note (contains decision)" in captured.out

    def test_review_json(self, capsys):
        """Review JSON output."""
        k = MagicMock()
        k.list_raw.return_value = [{"id": "abc123", "content": "test"}]

        args = Namespace(
            raw_action="review",
            limit=10,
            json=True,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert '"id"' in captured.out


class TestCmdRawClean:
    """Test raw clean command."""

    def test_clean_no_targets(self, capsys):
        """No targets should show success message."""
        k = MagicMock()
        k.list_raw.return_value = []

        args = Namespace(
            raw_action="clean",
            age=7,
            junk=False,
            confirm=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "No unprocessed raw entries" in captured.out

    def test_clean_junk_detection(self, capsys):
        """Junk mode should detect short entries."""
        k = MagicMock()
        k.list_raw.return_value = [
            {"id": "abc123", "content": "test", "timestamp": "2026-01-01T00:00:00Z"},
            {"id": "def456", "content": "real content here", "timestamp": "2026-01-01T00:00:00Z"},
        ]

        args = Namespace(
            raw_action="clean",
            age=7,
            junk=True,
            confirm=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        # "test" is <10 chars, should be detected as junk
        assert "junk" in captured.out.lower()

    def test_clean_stale_entries(self, capsys):
        """Clean stale entries (older than age days)."""
        k = MagicMock()
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        k.list_raw.return_value = [
            {"id": "abc123", "content": "old entry content", "timestamp": old_timestamp},
        ]

        args = Namespace(
            raw_action="clean",
            age=7,
            junk=False,
            confirm=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "older than 7 days" in captured.out
        assert "DRY RUN" in captured.out

    def test_clean_with_confirm(self, capsys):
        """Clean with confirm should actually delete."""
        k = MagicMock()
        k.list_raw.return_value = [
            {"id": "abc123", "content": "test", "timestamp": "2026-01-01T00:00:00Z"},
        ]
        k._storage.delete_raw.return_value = True

        args = Namespace(
            raw_action="clean",
            age=7,
            junk=True,
            confirm=True,
        )

        cmd_raw(args, k)

        k._storage.delete_raw.assert_called_once_with("abc123")
        captured = capsys.readouterr()
        assert "Deleted 1" in captured.out

    def test_clean_junk_keywords(self, capsys):
        """Clean should detect junk keywords."""
        k = MagicMock()
        k.list_raw.return_value = [
            {"id": "abc123", "content": "testing 123", "timestamp": "2026-01-01T00:00:00Z"},
            {"id": "def456", "content": "hello", "timestamp": "2026-01-01T00:00:00Z"},
            {"id": "ghi789", "content": "foo", "timestamp": "2026-01-01T00:00:00Z"},
        ]

        args = Namespace(
            raw_action="clean",
            age=7,
            junk=True,
            confirm=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "junk" in captured.out.lower()

    def test_clean_many_entries(self, capsys):
        """Clean with many entries should show truncated list."""
        k = MagicMock()
        k.list_raw.return_value = [
            {"id": f"entry{i:05d}", "content": "x", "timestamp": "2026-01-01T00:00:00Z"}
            for i in range(20)
        ]

        args = Namespace(
            raw_action="clean",
            age=7,
            junk=True,
            confirm=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "... and" in captured.out  # Shows truncation


class TestCmdRawPromote:
    """Test raw promote command — should delegate to process_raw (#835)."""

    def test_promote_delegates_to_process_raw(self, capsys):
        """Promote should call k.process_raw(), not manual k.episode/note/belief."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc12345", "content": "test"}
        k.process_raw.return_value = "ep123456"

        args = Namespace(
            raw_action="promote",
            id="abc12345",
            type="episode",
            objective="Test obj",
            outcome="Test out",
        )

        cmd_raw(args, k)

        # Should delegate to process_raw
        k.process_raw.assert_called_once()
        call_kwargs = k.process_raw.call_args
        assert (
            call_kwargs.kwargs.get("raw_id") == "abc12345"
            or call_kwargs[1].get("raw_id") == "abc12345"
        )
        assert (
            call_kwargs.kwargs.get("as_type") == "episode"
            or call_kwargs[1].get("as_type") == "episode"
        )
        # Should NOT call manual creation methods
        k.episode.assert_not_called()
        k.note.assert_not_called()
        k.belief.assert_not_called()
        # Should NOT call mark_raw_processed manually
        k._storage.mark_raw_processed.assert_not_called()

        captured = capsys.readouterr()
        assert "Promoted to episode" in captured.out

    def test_promote_note_delegates_to_process_raw(self, capsys):
        """Promote to note should also delegate to process_raw."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc12345", "content": "test"}
        k.process_raw.return_value = "note123"

        args = Namespace(
            raw_action="promote",
            id="abc12345",
            type="note",
            objective=None,
            outcome=None,
        )

        cmd_raw(args, k)

        k.process_raw.assert_called_once()
        k.note.assert_not_called()
        k._storage.mark_raw_processed.assert_not_called()
        captured = capsys.readouterr()
        assert "Promoted to note" in captured.out

    def test_promote_belief_delegates_to_process_raw(self, capsys):
        """Promote to belief should also delegate to process_raw."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc12345", "content": "test"}
        k.process_raw.return_value = "belief123"

        args = Namespace(
            raw_action="promote",
            id="abc12345",
            type="belief",
            objective=None,
            outcome=None,
        )

        cmd_raw(args, k)

        k.process_raw.assert_called_once()
        k.belief.assert_not_called()
        k._storage.mark_raw_processed.assert_not_called()
        captured = capsys.readouterr()
        assert "Promoted to belief" in captured.out

    def test_promote_episode_passes_objective_and_outcome(self, capsys):
        """Episode promote should pass objective and outcome kwargs."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc12345", "content": "test"}
        k.process_raw.return_value = "ep123456"

        args = Namespace(
            raw_action="promote",
            id="abc12345",
            type="episode",
            objective="My objective",
            outcome="My outcome",
        )

        cmd_raw(args, k)

        call_kwargs = k.process_raw.call_args[1]
        assert call_kwargs.get("objective") == "My objective"
        assert call_kwargs.get("outcome") == "My outcome"

    def test_promote_already_processed_fails(self, capsys):
        """Already-processed raw entry should show error message."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc12345", "content": "test"}
        k.process_raw.side_effect = ValueError("Raw entry abc12345 already processed")

        args = Namespace(
            raw_action="promote",
            id="abc12345",
            type="note",
            objective=None,
            outcome=None,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "already processed" in captured.out

    def test_promote_unknown_type_fails(self, capsys):
        """Unknown target type should show error from process_raw."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc12345", "content": "test"}
        k.process_raw.side_effect = ValueError("Invalid as_type: bogus")

        args = Namespace(
            raw_action="promote",
            id="abc12345",
            type="bogus",
            objective=None,
            outcome=None,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Invalid as_type" in captured.out

    def test_promote_missing_raw_entry_fails(self, capsys):
        """Nonexistent ID should show resolve error."""
        k = MagicMock()
        k.get_raw.return_value = None
        k._storage.find_raw_by_prefix.return_value = []

        args = Namespace(
            raw_action="promote",
            id="nonexistent",
            type="episode",
            objective=None,
            outcome=None,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_promote_shows_processed_status(self, capsys):
        """Successful promote should show that raw entry was marked as processed."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc12345", "content": "test"}
        k.process_raw.return_value = "note12345"

        args = Namespace(
            raw_action="promote",
            id="abc12345",
            type="note",
            objective=None,
            outcome=None,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Raw entry marked as processed" in captured.out


class TestCmdRawTriage:
    """Test raw triage command."""

    def test_triage_empty(self, capsys):
        """Triage with no entries."""
        k = MagicMock()
        k.list_raw.return_value = []

        args = Namespace(
            raw_action="triage",
            limit=10,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "No unprocessed raw entries" in captured.out

    def test_triage_with_entries(self, capsys):
        """Triage with entries should suggest actions."""
        k = MagicMock()
        k.list_raw.return_value = [
            {
                "id": "abc12345",
                "content": "test",  # Short = junk
                "timestamp": "2026-01-28T10:00:00Z",
            },
            {
                "id": "def67890",
                "content": "Session completed: implemented new feature and shipped it",
                "timestamp": "2026-01-27T09:00:00Z",
            },
            {
                "id": "ghi11111",
                "content": "Important insight: testing saves time in the long run",
                "timestamp": "2026-01-26T08:00:00Z",
            },
        ]

        args = Namespace(
            raw_action="triage",
            limit=10,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Triage" in captured.out
        assert "Delete" in captured.out  # "test" is junk
        assert "Episode" in captured.out  # Session/shipped = episode
        assert "Note" in captured.out  # Insight = note


class TestCmdRawFiles:
    """Test raw files command."""

    def test_files_empty(self, capsys):
        """Files with empty directory."""
        from pathlib import Path

        k = MagicMock()
        mock_path = MagicMock(spec=Path)
        mock_path.__str__ = lambda self: "/home/test/.kernle/raw"
        mock_path.parent = Path("/home/test/.kernle")
        k._storage.get_raw_dir.return_value = mock_path
        k._storage.get_raw_files.return_value = []

        args = Namespace(
            raw_action="files",
            open=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Raw Flat Files Directory" in captured.out
        assert "No raw files yet" in captured.out

    def test_files_with_entries(self, capsys):
        """Files with some entries."""
        from pathlib import Path

        k = MagicMock()
        mock_path = MagicMock(spec=Path)
        mock_path.__str__ = lambda self: "/home/test/.kernle/raw"
        mock_path.parent = Path("/home/test/.kernle")
        k._storage.get_raw_dir.return_value = mock_path

        # Create mock file objects
        mock_file1 = MagicMock()
        mock_file1.name = "2026-01-28.md"
        mock_file1.stat.return_value.st_size = 1024

        mock_file2 = MagicMock()
        mock_file2.name = "2026-01-27.md"
        mock_file2.stat.return_value.st_size = 512

        k._storage.get_raw_files.return_value = [mock_file1, mock_file2]

        args = Namespace(
            raw_action="files",
            open=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "2 total" in captured.out
        assert "2026-01-28.md" in captured.out
        assert "1,536 bytes" in captured.out  # Total size

    def test_files_many_entries(self, capsys):
        """Files with more than 10 entries."""
        from pathlib import Path

        k = MagicMock()
        mock_path = MagicMock(spec=Path)
        mock_path.__str__ = lambda self: "/home/test/.kernle/raw"
        mock_path.parent = Path("/home/test/.kernle")
        k._storage.get_raw_dir.return_value = mock_path

        # Create 15 mock file objects
        mock_files = []
        for i in range(15):
            mock_file = MagicMock()
            mock_file.name = f"2026-01-{15 - i:02d}.md"
            mock_file.stat.return_value.st_size = 100
            mock_files.append(mock_file)

        k._storage.get_raw_files.return_value = mock_files

        args = Namespace(
            raw_action="files",
            open=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "15 total" in captured.out
        assert "... and 5 more" in captured.out


class TestCmdRawSync:
    """Test raw sync command."""

    def test_sync_no_entries(self, capsys):
        """Sync with nothing to import."""
        k = MagicMock()
        k._storage.sync_raw_from_files.return_value = {
            "files_processed": 5,
            "imported": 0,
            "skipped": 10,
            "errors": [],
        }

        args = Namespace(
            raw_action="sync",
            dry_run=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Files processed: 5" in captured.out
        assert "Entries imported: 0" in captured.out
        assert "All entries already indexed" in captured.out

    def test_sync_with_imports(self, capsys):
        """Sync with entries to import."""
        k = MagicMock()
        k._storage.sync_raw_from_files.return_value = {
            "files_processed": 3,
            "imported": 5,
            "skipped": 2,
            "errors": [],
        }

        args = Namespace(
            raw_action="sync",
            dry_run=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Imported 5 entries" in captured.out

    def test_sync_with_errors(self, capsys):
        """Sync with some errors."""
        k = MagicMock()
        k._storage.sync_raw_from_files.return_value = {
            "files_processed": 3,
            "imported": 2,
            "skipped": 0,
            "errors": ["Error parsing file1.md", "Error parsing file2.md"],
        }

        args = Namespace(
            raw_action="sync",
            dry_run=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Errors (2)" in captured.out
        assert "Error parsing file1.md" in captured.out

    def test_sync_empty(self, capsys):
        """Sync with no files and no entries."""
        k = MagicMock()
        k._storage.sync_raw_from_files.return_value = {
            "files_processed": 0,
            "imported": 0,
            "skipped": 0,
            "errors": [],
        }

        args = Namespace(
            raw_action="sync",
            dry_run=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "No entries to import" in captured.out

    def test_sync_many_errors(self, capsys):
        """Sync with many errors."""
        k = MagicMock()
        k._storage.sync_raw_from_files.return_value = {
            "files_processed": 10,
            "imported": 0,
            "skipped": 0,
            "errors": [f"Error {i}" for i in range(10)],
        }

        args = Namespace(
            raw_action="sync",
            dry_run=False,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "Errors (10)" in captured.out
        assert "... and 5 more" in captured.out

    def test_sync_dry_run(self, capsys):
        """Sync with dry run mode."""
        k = MagicMock()
        k._storage.sync_raw_from_files.return_value = {
            "files_processed": 5,
            "imported": 3,
            "skipped": 0,
            "errors": [],
        }

        args = Namespace(
            raw_action="sync",
            dry_run=True,
        )

        cmd_raw(args, k)

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
