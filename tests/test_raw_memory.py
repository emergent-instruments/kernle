"""
Tests for raw memory layer and dump/export functionality.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from kernle.core import Kernle
from kernle.storage import RawEntry
from kernle.storage.sqlite import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    s = SQLiteStorage(stack_id="test_agent", db_path=temp_db)
    yield s
    s.close()


@pytest.fixture
def kernle(temp_db):
    """Create a Kernle instance for testing."""
    storage = SQLiteStorage(stack_id="test_agent", db_path=temp_db)
    k = Kernle(stack_id="test_agent", storage=storage, strict=False)
    bind_noop_model(k)
    return k


class TestRawEntryDataclass:
    """Test the RawEntry dataclass."""

    def test_raw_entry_defaults(self):
        """Test RawEntry has correct defaults."""
        entry = RawEntry(
            id="test-id",
            stack_id="test-agent",
            content="Test content",
            timestamp=datetime.now(timezone.utc),
        )

        # Source default changed from "manual" to "unknown" in raw layer refactor
        assert entry.source == "unknown"
        assert entry.processed is False
        assert entry.processed_into is None
        assert entry.tags is None
        assert entry.confidence == 1.0
        assert entry.source_type == "direct_experience"
        assert entry.version == 1
        assert entry.deleted is False
        # Verify backward compat: content/timestamp populate blob/captured_at
        assert entry.blob == "Test content"
        assert entry.captured_at is not None

    def test_raw_entry_full(self):
        """Test RawEntry with all fields."""
        now = datetime.now(timezone.utc)
        entry = RawEntry(
            id="test-id",
            stack_id="test-agent",
            content="Test content",
            timestamp=now,
            source="voice",
            processed=True,
            processed_into=["note:abc123"],
            tags=["dev", "idea"],
            confidence=0.9,
            source_type="direct_experience",
        )

        assert entry.id == "test-id"
        assert entry.source == "voice"
        assert entry.processed is True
        assert entry.processed_into == ["note:abc123"]
        assert entry.tags == ["dev", "idea"]


class TestSQLiteStorageRaw:
    """Test SQLiteStorage raw entry methods."""

    def test_save_raw_returns_id(self, storage):
        """Test save_raw returns a valid ID."""
        raw_id = storage.save_raw("Test content")

        assert raw_id is not None
        assert len(raw_id) == 36  # UUID format

    def test_save_raw_with_source(self, storage):
        """Test save_raw with custom source (valid enum value)."""
        # Note: source is now normalized to valid enum values: cli, mcp, sdk, import, unknown
        # "voice" is not a valid enum value, so use "cli" instead
        raw_id = storage.save_raw("Test content", source="cli")
        entry = storage.get_raw(raw_id)

        assert entry.source == "cli"

    def test_save_raw_with_invalid_source(self, storage):
        """Test save_raw normalizes invalid source to unknown."""
        raw_id = storage.save_raw("Test content", source="voice")
        entry = storage.get_raw(raw_id)

        # Invalid source values are normalized to "unknown"
        assert entry.source == "unknown"

    def test_get_raw_not_found(self, storage):
        """Test get_raw returns None for non-existent entry."""
        entry = storage.get_raw("nonexistent-id")
        assert entry is None

    def test_list_raw_all(self, storage):
        """Test list_raw returns all entries."""
        storage.save_raw("Entry 1")
        storage.save_raw("Entry 2")
        storage.save_raw("Entry 3")

        entries = storage.list_raw()
        assert len(entries) == 3

    def test_list_raw_unprocessed(self, storage):
        """Test list_raw with processed=False filter."""
        id1 = storage.save_raw("Entry 1")
        storage.save_raw("Entry 2")
        storage.mark_raw_processed(id1, ["note:abc123"])

        unprocessed = storage.list_raw(processed=False)
        assert len(unprocessed) == 1
        assert unprocessed[0].content == "Entry 2"

    def test_list_raw_processed(self, storage):
        """Test list_raw with processed=True filter."""
        id1 = storage.save_raw("Entry 1")
        storage.save_raw("Entry 2")
        storage.mark_raw_processed(id1, ["note:abc123"])

        processed = storage.list_raw(processed=True)
        assert len(processed) == 1
        assert processed[0].content == "Entry 1"

    def test_list_raw_limit(self, storage):
        """Test list_raw respects limit."""
        for i in range(10):
            storage.save_raw(f"Entry {i}")

        entries = storage.list_raw(limit=5)
        assert len(entries) == 5

    def test_mark_raw_processed(self, storage):
        """Test mark_raw_processed updates entry."""
        raw_id = storage.save_raw("Test content")

        result = storage.mark_raw_processed(raw_id, ["episode:abc123", "note:def456"])
        assert result is True

        entry = storage.get_raw(raw_id)
        assert entry.processed is True
        assert entry.processed_into == ["episode:abc123", "note:def456"]

    def test_mark_raw_processed_not_found(self, storage):
        """Test mark_raw_processed returns False for non-existent entry."""
        result = storage.mark_raw_processed("nonexistent-id", ["note:abc123"])
        assert result is False

    def test_stats_includes_raw(self, storage):
        """Test get_stats includes raw entry count."""
        storage.save_raw("Entry 1")
        storage.save_raw("Entry 2")

        stats = storage.get_stats()
        assert "raw" in stats
        assert stats["raw"] == 2


class TestKernleRaw:
    """Test Kernle raw entry methods."""

    def test_raw_capture(self, kernle):
        """Test raw() captures content."""
        raw_id = kernle.raw("Quick thought about sync")

        assert raw_id is not None
        entry = kernle.get_raw(raw_id)
        assert entry["content"] == "Quick thought about sync"

    def test_raw_with_source(self, kernle):
        """Test raw() with custom source (valid enum value)."""
        # Note: source is normalized to valid enum values: cli, mcp, sdk, import, unknown
        raw_id = kernle.raw("SDK note", source="sdk")

        entry = kernle.get_raw(raw_id)
        assert entry["source"] == "sdk"

    def test_list_raw_returns_dicts(self, kernle):
        """Test list_raw returns list of dicts."""
        kernle.raw("Entry 1")
        kernle.raw("Entry 2")

        entries = kernle.list_raw()
        assert len(entries) == 2
        assert isinstance(entries[0], dict)
        assert "id" in entries[0]
        assert "content" in entries[0]

    def test_list_raw_unprocessed(self, kernle):
        """Test list_raw with processed filter."""
        id1 = kernle.raw("Entry 1")
        kernle.raw("Entry 2")
        kernle.process_raw(id1, "note")

        unprocessed = kernle.list_raw(processed=False)
        assert len(unprocessed) == 1

    def test_get_raw_not_found(self, kernle):
        """Test get_raw returns None for non-existent."""
        entry = kernle.get_raw("nonexistent-id")
        assert entry is None

    def test_process_raw_to_note(self, kernle):
        """Test process_raw converts to note."""
        raw_id = kernle.raw("This is an important insight")

        note_id = kernle.process_raw(raw_id, "note", type="insight")

        assert note_id is not None

        # Check raw entry is marked processed
        entry = kernle.get_raw(raw_id)
        assert entry["processed"] is True
        assert f"note:{note_id}" in entry["processed_into"]

    def test_process_raw_to_episode(self, kernle):
        """Test process_raw converts to episode."""
        raw_id = kernle.raw("Completed the sync feature implementation")

        episode_id = kernle.process_raw(
            raw_id, "episode", objective="Implement sync feature", outcome="completed"
        )

        assert episode_id is not None

        entry = kernle.get_raw(raw_id)
        assert entry["processed"] is True
        assert f"episode:{episode_id}" in entry["processed_into"]

    def test_process_raw_to_belief_raises_error(self, kernle):
        """process_raw rejects belief as target type."""
        raw_id = kernle.raw("Testing is essential for quality software")

        with pytest.raises(ValueError, match="Must be one of: episode, note"):
            kernle.process_raw(raw_id, "belief")

    def test_process_raw_not_found(self, kernle):
        """Test process_raw raises for non-existent entry."""
        with pytest.raises(ValueError, match="not found"):
            kernle.process_raw("nonexistent-id", "note")

    def test_process_raw_already_processed(self, kernle):
        """Test process_raw raises for already processed entry."""
        raw_id = kernle.raw("Test content")
        kernle.process_raw(raw_id, "note")

        with pytest.raises(ValueError, match="already processed"):
            kernle.process_raw(raw_id, "note")

    def test_process_raw_invalid_type(self, kernle):
        """Test process_raw raises for invalid type."""
        raw_id = kernle.raw("Test content")

        with pytest.raises(ValueError, match="Invalid as_type"):
            kernle.process_raw(raw_id, "invalid_type")


class TestDumpExport:
    """Test dump and export functionality."""

    def test_dump_markdown(self, kernle):
        """Test dump() returns markdown format."""
        kernle.raw("Test raw entry")
        kernle.note("Test note")
        kernle.belief("Test belief", confidence=0.8)

        output = kernle.dump(format="markdown")

        assert "# Memory Dump" in output
        assert "test_agent" in output
        assert "## Raw Entries" in output
        assert "Test raw entry" in output
        assert "## Notes" in output
        assert "## Beliefs" in output

    def test_dump_json(self, kernle):
        """Test dump() returns JSON format."""
        kernle.raw("Test raw entry")
        kernle.note("Test note")

        output = kernle.dump(format="json")

        # Should be valid JSON
        data = json.loads(output)
        assert "stack_id" in data
        assert "raw_entries" in data
        assert "notes" in data
        assert len(data["raw_entries"]) == 1

    def test_dump_without_raw(self, kernle):
        """Test dump() excludes raw when include_raw=False."""
        kernle.raw("Test raw entry")
        kernle.note("Test note")

        output = kernle.dump(include_raw=False, format="markdown")

        assert "## Raw Entries" not in output
        assert "Test raw entry" not in output

    def test_dump_json_without_raw(self, kernle):
        """Test dump() JSON excludes raw when include_raw=False."""
        kernle.raw("Test raw entry")

        output = kernle.dump(include_raw=False, format="json")
        data = json.loads(output)

        assert "raw_entries" not in data

    def test_export_markdown(self, kernle, tmp_path):
        """Test export() creates markdown file."""
        kernle.raw("Test raw entry")
        kernle.note("Test note")

        export_path = tmp_path / "memory.md"
        kernle.export(str(export_path))

        assert export_path.exists()
        content = export_path.read_text()
        assert "# Memory Dump" in content

    def test_export_json(self, kernle, tmp_path):
        """Test export() creates JSON file."""
        kernle.raw("Test raw entry")

        export_path = tmp_path / "memory.json"
        kernle.export(str(export_path), format="json")

        assert export_path.exists()
        content = export_path.read_text()
        data = json.loads(content)
        assert "stack_id" in data

    def test_export_auto_detect_format(self, kernle, tmp_path):
        """Test export() auto-detects format from extension."""
        kernle.raw("Test raw entry")

        # JSON extension
        json_path = tmp_path / "memory.json"
        kernle.export(str(json_path))

        content = json_path.read_text()
        data = json.loads(content)  # Should be valid JSON
        assert "stack_id" in data

        # Markdown extension
        md_path = tmp_path / "memory.md"
        kernle.export(str(md_path))

        content = md_path.read_text()
        assert "# Memory Dump" in content

    def test_export_creates_parent_dirs(self, kernle, tmp_path):
        """Test export() creates parent directories."""
        export_path = tmp_path / "nested" / "dir" / "memory.md"
        kernle.export(str(export_path))

        assert export_path.exists()


class TestCLIRaw:
    """Test CLI commands for raw entries."""

    @pytest.fixture
    def mock_kernle(self):
        """Create a mock Kernle for CLI testing."""
        k = Mock(spec=Kernle)
        k.raw.return_value = "raw123"
        k.list_raw.return_value = [
            {
                "id": "raw123",
                "content": "Test content",
                "timestamp": "2024-01-01T12:00:00+00:00",
                "source": "cli",
                "processed": False,
                "processed_into": None,
                "tags": ["dev"],
            }
        ]
        k.get_raw.return_value = {
            "id": "raw123",
            "content": "Test content here",
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source": "cli",
            "processed": False,
            "processed_into": None,
            "tags": ["dev"],
        }
        k.process_raw.return_value = "note456"
        k.dump.return_value = "# Memory Dump\nTest content"
        return k

    def test_cmd_raw_capture(self, mock_kernle):
        """Test raw capture command."""
        import argparse

        from kernle.cli.commands.raw import cmd_raw

        args = argparse.Namespace(
            raw_action=None,
            content="Quick thought",
            tags="dev,idea",
            source=None,
            quiet=False,
            stdin=False,
        )

        with patch("sys.stdout"):
            cmd_raw(args, mock_kernle)

        mock_kernle.raw.assert_called_once()
        call_args = mock_kernle.raw.call_args
        # Tags are folded into blob text (tags param removed from raw API)
        blob_value = call_args.kwargs.get("blob", "")
        assert "Quick thought" in blob_value
        assert "dev" in blob_value
        assert "idea" in blob_value

    def test_cmd_raw_list(self, mock_kernle):
        """Test raw list command."""
        import argparse
        from io import StringIO

        from kernle.cli.__main__ import cmd_raw

        args = argparse.Namespace(
            raw_action="list",
            unprocessed=False,
            processed=False,
            limit=50,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_raw(args, mock_kernle)

        mock_kernle.list_raw.assert_called_once_with(processed=None, limit=50)
        output = fake_out.getvalue()
        assert "Raw Entries" in output
        assert "raw123" in output

    def test_cmd_raw_show(self, mock_kernle):
        """Test raw show command."""
        import argparse
        from io import StringIO

        from kernle.cli.__main__ import cmd_raw

        args = argparse.Namespace(
            raw_action="show",
            id="raw123",
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_raw(args, mock_kernle)

        # Called twice: once for ID resolution (exact match check), once to fetch entry
        assert mock_kernle.get_raw.call_count == 2
        mock_kernle.get_raw.assert_called_with("raw123")
        output = fake_out.getvalue()
        assert "Raw Entry: raw123" in output
        assert "Test content here" in output

    def test_cmd_raw_show_partial_id(self, mock_kernle):
        """Test raw show command with partial ID (prefix match)."""
        import argparse
        from io import StringIO

        from kernle.cli.__main__ import cmd_raw

        # Set up _storage with find_raw_by_prefix for database-level prefix search
        mock_match = Mock()
        mock_match.id = "raw123-full-uuid-here"
        mock_storage = Mock()
        mock_storage.find_raw_by_prefix.return_value = [mock_match]
        mock_kernle._storage = mock_storage

        # Set up mock: first get_raw returns None (no exact match),
        # then get_raw returns the entry after prefix resolution
        mock_kernle.get_raw.side_effect = [
            None,  # No exact match for partial ID
            {  # Found via prefix match, fetch full entry
                "id": "raw123-full-uuid-here",
                "content": "Test content here",
                "timestamp": "2024-01-01T12:00:00+00:00",
                "source": "cli",
                "processed": False,
                "processed_into": None,
                "tags": ["dev"],
            },
        ]

        args = argparse.Namespace(
            raw_action="show",
            id="raw123",  # Partial ID
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_raw(args, mock_kernle)

        # Should have used database-level prefix search
        mock_storage.find_raw_by_prefix.assert_called_once_with("raw123", limit=6)
        output = fake_out.getvalue()
        assert "Raw Entry: raw123-full-uuid-here" in output
        assert "Test content here" in output

    def test_cmd_raw_process(self, mock_kernle):
        """Test raw process command."""
        import argparse
        from io import StringIO

        from kernle.cli.__main__ import cmd_raw

        args = argparse.Namespace(
            raw_action="process",
            id="raw123",
            type="note",
            objective=None,
            outcome=None,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_raw(args, mock_kernle)

        mock_kernle.process_raw.assert_called_once_with(
            raw_id="raw123",
            as_type="note",
            objective=None,
            outcome=None,
        )
        output = fake_out.getvalue()
        # New format: "✓ Processed raw123... → note:note456..."
        assert "Processed" in output
        assert "raw123" in output
        assert "note" in output

    def test_cmd_dump(self, mock_kernle):
        """Test dump command."""
        import argparse
        from io import StringIO

        from kernle.cli.__main__ import cmd_dump

        args = argparse.Namespace(
            format="markdown",
            include_raw=True,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_dump(args, mock_kernle)

        mock_kernle.dump.assert_called_once_with(include_raw=True, format="markdown")
        output = fake_out.getvalue()
        assert "Memory Dump" in output

    def test_cmd_export(self, mock_kernle, tmp_path):
        """Test export command."""
        import argparse
        from io import StringIO

        from kernle.cli.__main__ import cmd_export

        export_path = tmp_path / "memory.md"

        args = argparse.Namespace(
            path=str(export_path),
            format="markdown",
            include_raw=True,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_export(args, mock_kernle)

        mock_kernle.export.assert_called_once_with(
            str(export_path), include_raw=True, format="markdown"
        )
        output = fake_out.getvalue()
        assert "Exported memory" in output


class TestRawMemoryIntegration:
    """Integration tests for raw memory workflow."""

    def test_full_workflow(self, kernle):
        """Test complete raw entry workflow."""
        # 1. Capture raw thoughts
        raw1 = kernle.raw("Realized sync queue needs deduplication")
        kernle.raw("Feeling good about progress today")
        raw3 = kernle.raw("Sean suggested raw dump layer - great idea")

        # 2. Check they're captured
        entries = kernle.list_raw()
        assert len(entries) == 3

        # 3. List unprocessed
        unprocessed = kernle.list_raw(processed=False)
        assert len(unprocessed) == 3

        # 4. Process one into an episode
        kernle.process_raw(
            raw1, "episode", objective="Analyze sync queue", outcome="Found deduplication issue"
        )

        # 5. Process another into a note
        kernle.process_raw(raw3, "note", type="insight")

        # 6. Check processed count
        unprocessed = kernle.list_raw(processed=False)
        assert len(unprocessed) == 1

        processed = kernle.list_raw(processed=True)
        assert len(processed) == 2

        # 7. Verify dump includes everything
        dump = kernle.dump(format="markdown")
        assert "sync queue" in dump.lower() or "Sync queue" in dump
        assert "Raw Entries" in dump

    def test_dump_then_import_json(self, kernle, tmp_path):
        """Test that exported JSON can be re-imported (future feature)."""
        # Add some data
        kernle.raw("Test raw entry")
        kernle.note("Test note")
        kernle.belief("Test belief", confidence=0.8)

        # Export
        export_path = tmp_path / "backup.json"
        kernle.export(str(export_path), format="json")

        # Verify it's valid JSON
        content = export_path.read_text()
        data = json.loads(content)

        assert data["stack_id"] == "test_agent"
        assert len(data["raw_entries"]) == 1
        assert len(data["notes"]) == 1
        assert len(data["beliefs"]) == 1
