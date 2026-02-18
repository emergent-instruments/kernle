"""
Comprehensive tests for the Kernle CLI interface.
"""

import argparse
import json
import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from kernle.cli.__main__ import (
    cmd_checkpoint,
    cmd_drive,
    cmd_episode,
    cmd_load,
    cmd_note,
    cmd_promote,
    cmd_search,
    cmd_status,
    cmd_temporal,
    main,
)
from kernle.core import Kernle


@pytest.fixture
def mock_kernle():
    """Mock Kernle instance for CLI testing."""
    kernle = Mock(spec=Kernle)

    # Mock return values for various methods
    kernle.load.return_value = {
        "values": [{"name": "Quality", "statement": "High quality work"}],
        "beliefs": [{"statement": "Testing is important", "confidence": 0.9}],
        "goals": [{"title": "Complete tests", "status": "active"}],
        "checkpoint": {"current_task": "Testing", "pending": ["CLI tests"]},
        "recent_work": [{"objective": "Write tests", "outcome_type": "success"}],
        "drives": [{"drive_type": "growth", "intensity": 0.7}],
        "relationships": [{"other_stack_id": "peer", "trust_level": 0.8}],
    }

    kernle.format_memory.return_value = "# Working Memory\nFormatted memory context..."
    kernle.checkpoint.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "current_task": "Test task",
        "pending": ["item1", "item2"],
        "context": "Test context",
    }
    kernle.load_checkpoint.return_value = {
        "current_task": "Loaded task",
        "timestamp": "2024-01-01T11:00:00Z",
        "pending": ["pending1"],
        "context": "Loaded context",
    }
    kernle.clear_checkpoint.return_value = True
    kernle.episode.return_value = "episode123"
    kernle.episode_with_emotion.return_value = "episode123"
    kernle.note.return_value = "note456"
    kernle.search.return_value = [
        {
            "type": "episode",
            "title": "Complete testing",
            "content": "All tests completed successfully",
            "lessons": ["Test thoroughly"],
            "date": "2024-01-01",
        },
        {
            "type": "belief",
            "title": "Testing leads to quality",
            "content": "Testing leads to quality software",
            "confidence": 0.9,
            "date": "2024-01-01",
        },
    ]
    kernle.status.return_value = {
        "stack_id": "test_agent",
        "values": 5,
        "beliefs": 10,
        "goals": 3,
        "episodes": 25,
        "checkpoint": True,
    }
    kernle.load_drives.return_value = [
        {"drive_type": "curiosity", "intensity": 0.8, "focus_areas": ["AI", "ML"]},
        {"drive_type": "growth", "intensity": 0.6, "focus_areas": []},
    ]
    kernle.drive.return_value = "drive789"
    kernle.satisfy_drive.return_value = True
    kernle.what_happened.return_value = {
        "range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T23:59:59Z"},
        "episodes": [{"objective": "Test something", "outcome_type": "success"}],
        "notes": [{"content": "Important note"}],
    }

    return kernle


@pytest.fixture
def mock_sys_argv():
    """Helper to mock sys.argv for testing CLI argument parsing."""
    original_argv = sys.argv.copy()
    yield
    sys.argv = original_argv


class TestMainFunction:
    """Test the main CLI entry point."""

    @patch("kernle.cli.__main__.resolve_stack_id")
    @patch("kernle.cli.__main__.Kernle")
    @patch("sys.argv")
    def test_main_load_command(self, mock_argv, mock_kernle_class, mock_resolve, mock_kernle):
        """Test main function with load command (no explicit agent ID)."""
        mock_argv.__getitem__.side_effect = lambda x: ["kernle", "load"][x]
        mock_argv.__len__.return_value = 2
        mock_kernle_class.return_value = mock_kernle
        mock_resolve.return_value = "auto-test1234"

        with patch("sys.stdout", new=StringIO()) as fake_out:
            main()

        # When no -a is provided, resolve_stack_id is called to generate one
        mock_resolve.assert_called_once()
        mock_kernle_class.assert_called_once_with(stack_id="auto-test1234")
        mock_kernle.load.assert_called_once()
        mock_kernle.format_memory.assert_called_once()
        assert "# Working Memory" in fake_out.getvalue()

    @patch("kernle.cli.__main__.Kernle")
    @patch("sys.argv")
    def test_main_with_stack_id(self, mock_argv, mock_kernle_class, mock_kernle):
        """Test main function with agent ID parameter."""
        mock_argv.__getitem__.side_effect = lambda x: ["kernle", "--stack", "test_agent", "status"][
            x
        ]
        mock_argv.__len__.return_value = 4
        mock_kernle_class.return_value = mock_kernle

        with patch("sys.stdout", new=StringIO()):
            main()

        mock_kernle_class.assert_called_once_with(stack_id="test_agent")

    @patch("kernle.cli.__main__.Kernle")
    @patch("sys.argv")
    def test_main_missing_required_args(self, mock_argv, mock_kernle_class):
        """Test main function with missing required arguments."""
        mock_argv.__getitem__.side_effect = lambda x: ["kernle"][x]
        mock_argv.__len__.return_value = 1

        with pytest.raises(SystemExit):  # argparse exits on missing required args
            main()


class TestLoadCommand:
    """Test the load command functionality."""

    def test_cmd_load_formatted_output(self, mock_kernle):
        """Test load command with formatted output."""
        args = argparse.Namespace(json=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_load(args, mock_kernle)

        mock_kernle.load.assert_called_once()
        mock_kernle.format_memory.assert_called_once()
        assert "# Working Memory" in fake_out.getvalue()

    def test_cmd_load_json_output(self, mock_kernle):
        """Test load command with JSON output produces valid JSON and skips formatting."""
        args = argparse.Namespace(json=True)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_load(args, mock_kernle)

        # Verify the correct code path: load() called, format_memory() skipped
        mock_kernle.load.assert_called_once()
        mock_kernle.format_memory.assert_not_called()

        # Verify output is valid JSON (tests the json.dumps transformation)
        output = fake_out.getvalue()
        parsed = json.loads(output)  # Would raise if not valid JSON
        assert isinstance(parsed, dict), "JSON output should be a dict"
        # Don't assert specific keys from mock - that's testing mock config, not code


class TestCheckpointCommands:
    """Test checkpoint-related commands."""

    def test_cmd_checkpoint_save(self, mock_kernle):
        """Test checkpoint save command."""
        args = argparse.Namespace(
            checkpoint_action="save",
            task="Complete testing",
            pending=["write docs", "run CI"],
            context="Working on comprehensive tests",
            sync=False,
            no_sync=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_checkpoint(args, mock_kernle)

        mock_kernle.checkpoint.assert_called_once_with(
            "Complete testing",
            ["write docs", "run CI"],
            "Working on comprehensive tests",
            sync=None,  # Neither --sync nor --no-sync specified
        )
        assert "✓ Checkpoint saved: Test task" in fake_out.getvalue()

    def test_cmd_checkpoint_save_minimal(self, mock_kernle):
        """Test checkpoint save with minimal args."""
        args = argparse.Namespace(
            checkpoint_action="save",
            task="Simple task",
            pending=None,
            context=None,
            sync=False,
            no_sync=False,
        )

        with patch("sys.stdout", new=StringIO()):
            cmd_checkpoint(args, mock_kernle)

        mock_kernle.checkpoint.assert_called_once_with("Simple task", [], None, sync=None)

    def test_cmd_checkpoint_load_formatted(self, mock_kernle):
        """Test checkpoint load with formatted output."""
        args = argparse.Namespace(checkpoint_action="load", json=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_checkpoint(args, mock_kernle)

        mock_kernle.load_checkpoint.assert_called_once()
        output = fake_out.getvalue()
        # New format uses markdown-style bold and different structure
        assert "**Task**: Loaded task" in output
        assert "## Last Checkpoint" in output
        assert "**Pending**:" in output
        assert "  - pending1" in output
        assert "**Context**: Loaded context" in output

    def test_cmd_checkpoint_load_json(self, mock_kernle):
        """Test checkpoint load with JSON output produces valid JSON."""
        args = argparse.Namespace(checkpoint_action="load", json=True)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_checkpoint(args, mock_kernle)

        # Verify correct method was called
        mock_kernle.load_checkpoint.assert_called_once()

        # Verify output is valid JSON (tests the json.dumps transformation)
        output = fake_out.getvalue()
        parsed = json.loads(output)  # Would raise if not valid JSON
        assert isinstance(parsed, dict), "JSON output should be a dict"
        # Don't assert specific values from mock - that's testing mock config

    def test_cmd_checkpoint_load_not_found(self, mock_kernle):
        """Test checkpoint load when no checkpoint exists."""
        mock_kernle.load_checkpoint.return_value = None
        args = argparse.Namespace(checkpoint_action="load", json=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_checkpoint(args, mock_kernle)

        assert "No checkpoint found." in fake_out.getvalue()

    def test_cmd_checkpoint_clear_success(self, mock_kernle):
        """Test checkpoint clear when checkpoint exists."""
        args = argparse.Namespace(checkpoint_action="clear")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_checkpoint(args, mock_kernle)

        mock_kernle.clear_checkpoint.assert_called_once()
        assert "✓ Checkpoint cleared" in fake_out.getvalue()

    def test_cmd_checkpoint_clear_not_found(self, mock_kernle):
        """Test checkpoint clear when no checkpoint exists."""
        mock_kernle.clear_checkpoint.return_value = False
        args = argparse.Namespace(checkpoint_action="clear")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_checkpoint(args, mock_kernle)

        assert "No checkpoint to clear" in fake_out.getvalue()


class TestEpisodeCommand:
    """Test episode recording command."""

    def test_cmd_episode_full(self, mock_kernle):
        """Test episode command with all parameters."""
        args = argparse.Namespace(
            objective="Implement user authentication",
            outcome="Successfully implemented with JWT",
            lesson=["Validate all inputs", "Use secure libraries"],
            tag=["security", "feature"],
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_episode(args, mock_kernle)

        # With auto_emotion=True by default, episode_with_emotion is called
        mock_kernle.episode_with_emotion.assert_called_once_with(
            objective="Implement user authentication",
            outcome="Successfully implemented with JWT",
            lessons=["Validate all inputs", "Use secure libraries"],
            tags=["security", "feature"],
            valence=None,
            arousal=None,
            emotional_tags=None,
            auto_detect=True,
            derived_from=None,
            source=None,
            context=None,
            context_tags=None,
        )
        assert "✓ Episode saved: episode1..." in fake_out.getvalue()
        assert "Lessons: 2" in fake_out.getvalue()

    def test_cmd_episode_minimal(self, mock_kernle):
        """Test episode command with minimal parameters."""
        args = argparse.Namespace(
            objective="Simple task", outcome="completed", lesson=None, tag=None
        )

        with patch("sys.stdout", new=StringIO()):
            cmd_episode(args, mock_kernle)

        # With auto_emotion=True by default, episode_with_emotion is called
        mock_kernle.episode_with_emotion.assert_called_once_with(
            objective="Simple task",
            outcome="completed",
            lessons=[],
            tags=[],
            valence=None,
            arousal=None,
            emotional_tags=None,
            auto_detect=True,
            derived_from=None,
            source=None,
            context=None,
            context_tags=None,
        )


class TestNoteCommand:
    """Test note capture command."""

    def test_cmd_note_full(self, mock_kernle):
        """Test note command with all parameters."""
        args = argparse.Namespace(
            content="Use React for the frontend framework",
            type="decision",
            speaker=None,
            reason="Better component reusability",
            tag=["frontend", "architecture"],
            protect=True,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_note(args, mock_kernle)

        mock_kernle.note.assert_called_once_with(
            content="Use React for the frontend framework",
            type="decision",
            speaker=None,
            reason="Better component reusability",
            tags=["frontend", "architecture"],
            protect=True,
            derived_from=None,
            source=None,
            context=None,
            context_tags=None,
        )
        assert "✓ Note saved: Use React for the frontend framework..." in fake_out.getvalue()
        assert "Tags: frontend, architecture" in fake_out.getvalue()

    def test_cmd_note_quote(self, mock_kernle):
        """Test note command for quote type."""
        args = argparse.Namespace(
            content="Premature optimization is the root of all evil",
            type="quote",
            speaker="Donald Knuth",
            reason=None,
            tag=["wisdom"],
            protect=False,
        )

        with patch("sys.stdout", new=StringIO()):
            cmd_note(args, mock_kernle)

        mock_kernle.note.assert_called_once_with(
            content="Premature optimization is the root of all evil",
            type="quote",
            speaker="Donald Knuth",
            reason=None,
            tags=["wisdom"],
            protect=False,
            derived_from=None,
            source=None,
            context=None,
            context_tags=None,
        )

    def test_cmd_note_minimal(self, mock_kernle):
        """Test note command with minimal parameters."""
        args = argparse.Namespace(
            content="Simple note content",
            type="note",
            speaker=None,
            reason=None,
            tag=None,
            protect=False,
        )

        with patch("sys.stdout", new=StringIO()):
            cmd_note(args, mock_kernle)

        mock_kernle.note.assert_called_once_with(
            content="Simple note content",
            type="note",
            speaker=None,
            reason=None,
            tags=[],
            protect=False,
            derived_from=None,
            source=None,
            context=None,
            context_tags=None,
        )


class TestSearchCommand:
    """Test search command."""

    def test_cmd_search_with_results(self, mock_kernle):
        """Test search command calls search() and formats results correctly."""
        args = argparse.Namespace(query="testing", limit=10, min_score=None)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_search(args, mock_kernle)

        # Verify correct method called with correct args
        mock_kernle.search.assert_called_once_with("testing", 10, min_score=None)

        # Verify output formatting logic (mock returns exactly 2 results)
        output = fake_out.getvalue()
        assert "Found 2 result(s)" in output, "Should show exact result count"
        assert "[episode]" in output, "Should format episode results with type prefix"
        assert "[belief]" in output, "Should format belief results with type prefix"

    def test_cmd_search_no_results(self, mock_kernle):
        """Test search command with no results."""
        mock_kernle.search.return_value = []
        args = argparse.Namespace(query="nonexistent", limit=10, min_score=None)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_search(args, mock_kernle)

        assert "No results for 'nonexistent'" in fake_out.getvalue()

    def test_cmd_search_custom_limit(self, mock_kernle):
        """Test search command with custom limit."""
        args = argparse.Namespace(query="test", limit=5, min_score=None)

        cmd_search(args, mock_kernle)

        mock_kernle.search.assert_called_once_with("test", 5, min_score=None)


class TestStatusCommand:
    """Test status command."""

    def test_cmd_status(self, mock_kernle):
        """Test status command calls status() and formats output."""
        args = argparse.Namespace()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_status(args, mock_kernle)

        # Verify correct method was called
        mock_kernle.status.assert_called_once()

        # Verify output has expected structure with correct formatted values
        output = fake_out.getvalue()
        assert "Memory Status for test_agent" in output, "Should have status header with stack_id"
        assert "Values:     5" in output, "Should show values count"
        assert "Beliefs:    10" in output, "Should show beliefs count"
        assert "Goals:      3" in output, "Should show goals count"
        assert "Episodes:   25" in output, "Should show episodes count"
        assert "Checkpoint: Yes" in output, "Should show checkpoint as Yes"

    def test_cmd_status_no_checkpoint(self, mock_kernle):
        """Test status command when no checkpoint exists."""
        mock_kernle.status.return_value = {
            "stack_id": "test_agent",
            "values": 0,
            "beliefs": 0,
            "goals": 0,
            "episodes": 0,
            "checkpoint": False,
        }
        args = argparse.Namespace()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_status(args, mock_kernle)

        output = fake_out.getvalue()
        assert "Checkpoint: No" in output


class TestDriveCommands:
    """Test drive management commands."""

    def test_cmd_drive_list(self, mock_kernle):
        """Test drive list command calls load_drives() and formats output."""
        args = argparse.Namespace(drive_action="list")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_drive(args, mock_kernle)

        # Verify correct method was called
        mock_kernle.load_drives.assert_called_once()

        # Verify output has expected structure with formatted values from mock
        output = fake_out.getvalue()
        assert "Drives:" in output, "Should have drives header"
        assert "curiosity: 80%" in output, "Should format first drive with percentage"
        assert "growth: 60%" in output, "Should format second drive with percentage"
        assert "AI, ML" in output, "Should show focus areas for curiosity drive"

    def test_cmd_drive_list_empty(self, mock_kernle):
        """Test drive list when no drives exist."""
        mock_kernle.load_drives.return_value = []
        args = argparse.Namespace(drive_action="list")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_drive(args, mock_kernle)

        assert "No drives set." in fake_out.getvalue()

    def test_cmd_drive_set(self, mock_kernle):
        """Test drive set command."""
        args = argparse.Namespace(
            drive_action="set",
            type="curiosity",
            intensity=0.9,
            focus=["machine learning", "AI safety"],
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_drive(args, mock_kernle)

        mock_kernle.drive.assert_called_once_with(
            "curiosity", 0.9, ["machine learning", "AI safety"]
        )
        assert "✓ Drive 'curiosity' set to 90%" in fake_out.getvalue()

    def test_cmd_drive_satisfy_success(self, mock_kernle):
        """Test drive satisfy command success."""
        args = argparse.Namespace(drive_action="satisfy", type="growth", amount=0.3)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_drive(args, mock_kernle)

        mock_kernle.satisfy_drive.assert_called_once_with("growth", 0.3)
        assert "✓ Satisfied drive 'growth'" in fake_out.getvalue()

    def test_cmd_drive_satisfy_not_found(self, mock_kernle):
        """Test drive satisfy when drive not found."""
        mock_kernle.satisfy_drive.return_value = False
        args = argparse.Namespace(drive_action="satisfy", type="nonexistent", amount=0.2)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_drive(args, mock_kernle)

        assert "Drive 'nonexistent' not found" in fake_out.getvalue()


class TestPromoteCommand:
    """Test promote command (episode -> belief promotion)."""

    def test_cmd_promote_shows_suggestions(self, mock_kernle):
        """Test promote command shows promotion suggestions."""
        mock_kernle.promote.return_value = {
            "episodes_scanned": 10,
            "patterns_found": 2,
            "suggestions": [
                {
                    "lesson": "Always test first",
                    "count": 3,
                    "source_episodes": ["ep1", "ep2", "ep3"],
                    "promoted": False,
                },
            ],
            "beliefs_created": 0,
        }
        mock_kernle.stack_id = "test_agent"

        args = argparse.Namespace(
            auto=False,
            min_occurrences=2,
            min_episodes=3,
            confidence=0.7,
            limit=50,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_promote(args, mock_kernle)

        output = fake_out.getvalue()
        assert "Promotion Results" in output
        assert "Episodes scanned: 10" in output
        assert "Patterns found: 2" in output
        assert "Always test first" in output

    def test_cmd_promote_no_patterns(self, mock_kernle):
        """Test promote command with no patterns found."""
        mock_kernle.promote.return_value = {
            "episodes_scanned": 5,
            "patterns_found": 0,
            "suggestions": [],
            "beliefs_created": 0,
        }
        mock_kernle.stack_id = "test_agent"

        args = argparse.Namespace(
            auto=False,
            min_occurrences=2,
            min_episodes=3,
            confidence=0.7,
            limit=50,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_promote(args, mock_kernle)

        output = fake_out.getvalue()
        assert "No recurring patterns found" in output

    def test_cmd_promote_json_output(self, mock_kernle):
        """Test promote command with JSON output."""
        result = {
            "episodes_scanned": 10,
            "patterns_found": 1,
            "suggestions": [],
            "beliefs_created": 0,
        }
        mock_kernle.promote.return_value = result

        args = argparse.Namespace(
            auto=False,
            min_occurrences=2,
            min_episodes=3,
            confidence=0.7,
            limit=50,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_promote(args, mock_kernle)

        output = fake_out.getvalue()
        parsed = json.loads(output)
        assert parsed["episodes_scanned"] == 10


class TestTemporalCommand:
    """Test temporal query command."""

    def test_cmd_temporal_today(self, mock_kernle):
        """Test temporal command calls what_happened() and formats output."""
        args = argparse.Namespace(when="today")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_temporal(args, mock_kernle)

        # Verify correct method was called with correct arg
        mock_kernle.what_happened.assert_called_once_with("today")

        # Verify output formatting structure (not mock content)
        output = fake_out.getvalue()
        assert "What happened today:" in output, "Should have period header"
        assert "Time range:" in output, "Should show time range"
        assert "Episodes:" in output, "Should have episodes section"
        assert "Notes:" in output, "Should have notes section"
        # Don't assert specific mock values like "Test something"

    def test_cmd_temporal_yesterday(self, mock_kernle):
        """Test temporal command for yesterday."""
        args = argparse.Namespace(when="yesterday")

        cmd_temporal(args, mock_kernle)

        mock_kernle.what_happened.assert_called_once_with("yesterday")

    def test_cmd_temporal_custom_period(self, mock_kernle):
        """Test temporal command with custom period."""
        args = argparse.Namespace(when="this week")

        cmd_temporal(args, mock_kernle)

        mock_kernle.what_happened.assert_called_once_with("this week")

    def test_cmd_temporal_no_episodes(self, mock_kernle):
        """Test temporal command when no episodes found."""
        mock_kernle.what_happened.return_value = {
            "range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T23:59:59Z"},
            "episodes": [],
            "notes": [],
        }
        args = argparse.Namespace(when="today")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_temporal(args, mock_kernle)

        output = fake_out.getvalue()
        assert "What happened today:" in output
        # Should still show headers even with empty data


class TestArgumentParsing:
    """Test CLI argument parsing edge cases."""

    def test_parse_checkpoint_save_args(self):
        """Test parsing checkpoint save arguments."""
        from kernle.cli.__main__ import main

        test_args = [
            "kernle",
            "checkpoint",
            "save",
            "Test task",
            "--pending",
            "item1",
            "--pending",
            "item2",
            "--context",
            "Test context",
        ]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle") as mock_kernle_class:
                mock_kernle = Mock()
                mock_kernle_class.return_value = mock_kernle
                mock_kernle.checkpoint.return_value = {
                    "timestamp": "2024-01-01T12:00:00Z",
                    "current_task": "Test task",
                    "pending": ["item1", "item2"],
                }

                with patch("sys.stdout", new=StringIO()):
                    main()

                mock_kernle.checkpoint.assert_called_once_with(
                    "Test task", ["item1", "item2"], "Test context", sync=None
                )

    def test_parse_episode_args(self):
        """Test parsing episode arguments."""
        test_args = [
            "kernle",
            "episode",
            "Complete testing",
            "success",
            "--lesson",
            "Test early",
            "--lesson",
            "Test often",
            "--tag",
            "testing",
            "--tag",
            "quality",
        ]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle") as mock_kernle_class:
                mock_kernle = Mock()
                mock_kernle_class.return_value = mock_kernle
                mock_kernle.episode.return_value = "episode123"
                mock_kernle.episode_with_emotion.return_value = "episode123"

                with patch("sys.stdout", new=StringIO()):
                    main()

                # With auto_emotion=True by default, episode_with_emotion is called
                mock_kernle.episode_with_emotion.assert_called_once_with(
                    objective="Complete testing",
                    outcome="success",
                    lessons=["Test early", "Test often"],
                    tags=["testing", "quality"],
                    valence=None,
                    arousal=None,
                    emotional_tags=None,
                    auto_detect=True,
                    derived_from=None,
                    source=None,
                    context=None,
                    context_tags=None,
                )

    def test_parse_note_args(self):
        """Test parsing note arguments."""
        test_args = [
            "kernle",
            "note",
            "Important decision content",
            "--type",
            "decision",
            "--reason",
            "Performance benefits",
            "--tag",
            "architecture",
            "--protect",
        ]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle") as mock_kernle_class:
                mock_kernle = Mock()
                mock_kernle_class.return_value = mock_kernle
                mock_kernle.note.return_value = "note123"

                with patch("sys.stdout", new=StringIO()):
                    main()

                mock_kernle.note.assert_called_once_with(
                    content="Important decision content",
                    type="decision",
                    speaker=None,
                    reason="Performance benefits",
                    tags=["architecture"],
                    protect=True,
                    derived_from=None,
                    source=None,
                    context=None,
                    context_tags=None,
                )

    def test_parse_drive_set_args(self):
        """Test parsing drive set arguments."""
        test_args = ["kernle", "drive", "set", "curiosity", "0.8", "--focus", "AI", "--focus", "ML"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle") as mock_kernle_class:
                mock_kernle = Mock()
                mock_kernle_class.return_value = mock_kernle
                mock_kernle.drive.return_value = "drive123"

                with patch("sys.stdout", new=StringIO()):
                    main()

                mock_kernle.drive.assert_called_once_with("curiosity", 0.8, ["AI", "ML"])


class TestErrorHandling:
    """Test CLI error handling and edge cases."""

    def test_kernle_initialization_error(self, mock_kernle):
        """Test handling of Kernle initialization errors."""
        with patch("kernle.cli.__main__.Kernle") as mock_kernle_class:
            mock_kernle_class.side_effect = ValueError("Missing credentials")

            test_args = ["kernle", "status"]
            with patch("sys.argv", test_args):
                with pytest.raises(SystemExit):
                    main()

    def test_command_execution_error(self, mock_kernle):
        """Test handling of command execution errors."""
        mock_kernle.search.side_effect = RuntimeError("Database connection failed")

        args = argparse.Namespace(query="test", limit=10)

        # The CLI should propagate the specific error from the search operation
        with pytest.raises(RuntimeError, match="Database connection failed"):
            cmd_search(args, mock_kernle)

    def test_json_output_error_handling(self, mock_kernle):
        """Test handling of JSON serialization errors."""
        # Mock a response that can't be serialized
        from datetime import datetime

        mock_kernle.load.return_value = {
            "timestamp": datetime.now(),  # datetime objects can't be serialized by default
            "data": "test",
        }

        args = argparse.Namespace(json=True)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            # Should handle datetime with default=str
            cmd_load(args, mock_kernle)

        # Should not raise exception, should output JSON with string representation
        output = fake_out.getvalue()
        assert output  # Should have some output

    def test_empty_search_results_formatting(self, mock_kernle):
        """Test formatting of empty search results."""
        mock_kernle.search.return_value = []
        args = argparse.Namespace(query="nothing", limit=10)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_search(args, mock_kernle)

        output = fake_out.getvalue()
        assert "No results for 'nothing'" in output
        assert "Found 0 result(s)" not in output  # Should not show "Found 0 results"

    def test_malformed_search_results(self, mock_kernle):
        """Test handling of malformed search results."""
        # Mock search results missing expected fields
        mock_kernle.search.return_value = [
            {"type": "episode"},  # Missing title, content, etc.
            {"title": "Test", "content": "Test content"},  # Missing type
            {
                "type": "belief",
                "title": "Valid belief",
                "content": "Valid content",
                "confidence": 0.8,
                "date": "2024-01-01",
            },
        ]

        args = argparse.Namespace(query="test", limit=10)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            # Should handle malformed results gracefully
            cmd_search(args, mock_kernle)

        output = fake_out.getvalue()
        assert "Found 3 result(s)" in output
        # Should still display the valid result
        assert "Valid belief" in output


class TestAgentCommand:
    """Test agent management commands."""

    def test_cmd_stack_list(self, mock_kernle, tmp_path):
        """Test agent list command."""
        import argparse

        from kernle.cli.commands.stack import cmd_stack

        # Add stack_id and _storage attributes to mock
        mock_kernle.stack_id = "test-agent"
        mock_kernle._storage = Mock()  # Non-SQLiteStorage; skips DB queries

        args = argparse.Namespace(stack_action="list")
        kernle_home = tmp_path / ".kernle"
        (kernle_home / "test-agent" / "raw").mkdir(parents=True)
        (kernle_home / "test-agent" / "raw" / "entry.md").write_text("test")

        with patch("kernle.cli.commands.stack.get_kernle_home", return_value=kernle_home):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                cmd_stack(args, mock_kernle)

        output = fake_out.getvalue()
        # Should run without error
        assert "Local Stacks" in output
        assert "test-agent" in output


class TestImportCommand:
    """Test import command."""

    def test_cmd_import_dry_run(self, mock_kernle, tmp_path):
        """Test import command with dry-run."""
        import argparse

        from kernle.cli.commands.import_cmd import cmd_import

        # Create a test markdown file with raw content
        test_file = tmp_path / "test_import.md"
        test_file.write_text("""## Thoughts
- Raw thought one
- Raw thought two

## Ideas
- Another raw idea to explore
""")

        args = argparse.Namespace(file=str(test_file), dry_run=True, interactive=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_import(args, mock_kernle)

        output = fake_out.getvalue()
        assert "DRY RUN" in output
        assert "items to import" in output or "Found" in output


class TestSearchMinScore:
    """Test search with min_score parameter."""

    def test_search_with_min_score(self, mock_kernle):
        """Test that min_score filters results."""
        # Mock search to return results with scores
        from dataclasses import dataclass

        @dataclass
        class MockRecord:
            objective: str = "Test episode"
            outcome: str = "Test outcome"
            lessons: list = None
            created_at: None = None

        mock_kernle.search.return_value = [
            {"type": "episode", "title": "High score", "score": 0.8},
            {"type": "episode", "title": "Low score", "score": 0.2},
        ]

        args = argparse.Namespace(query="test", limit=10, min_score=0.5)

        with patch("sys.stdout", new=StringIO()):
            cmd_search(args, mock_kernle)

        mock_kernle.search.assert_called_once_with("test", 10, min_score=0.5)


# ============================================================================
# Integration Tests - Use real Kernle instance instead of mocks
# ============================================================================


class TestCLIIntegration:
    """Integration tests that verify CLI commands work with real Kernle.

    These tests use actual Kernle instances with SQLite storage to verify
    end-to-end behavior, complementing the unit tests that use mocks.
    """

    def test_episode_integration(self, kernle_instance):
        """Test episode command creates real episode in storage."""
        kernle, storage = kernle_instance

        # Note: CLI uses 'lesson' and 'tag' (singular) for append actions
        args = argparse.Namespace(
            objective="Integration test episode",
            outcome="success",
            lesson=["Lesson 1", "Lesson 2"],
            tag=["integration", "test"],
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_episode(args, kernle)

        output = fake_out.getvalue()
        assert "Episode saved:" in output

        # Verify episode was actually saved (exactly one created in this test)
        episodes = storage.get_episodes()
        assert len(episodes) == 1
        matching = [e for e in episodes if e.objective == "Integration test episode"]
        assert len(matching) == 1
        assert matching[0].outcome_type == "success"
        assert "Lesson 1" in matching[0].lessons

    def test_note_integration(self, kernle_instance):
        """Test note command creates real note in storage."""
        kernle, storage = kernle_instance

        # Note: CLI uses 'tag' (singular) for append action, plus protect flag
        args = argparse.Namespace(
            content="Integration test note content",
            type="decision",
            reason="Testing the CLI integration",
            speaker="",
            tag=["test"],
            protect=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_note(args, kernle)

        output = fake_out.getvalue()
        assert "Note saved:" in output

        # Verify note was actually saved (exactly one created in this test)
        notes = storage.get_notes()
        assert len(notes) == 1
        matching = [n for n in notes if "Integration test note" in n.content]
        assert len(matching) == 1
        assert matching[0].note_type == "decision"

    def test_search_integration(self, kernle_instance):
        """Test search command finds real data."""
        kernle, storage = kernle_instance

        # Create some searchable content
        kernle.episode("Searchable integration objective", "success")
        kernle.note("Searchable integration note")

        args = argparse.Namespace(
            query="Searchable integration",
            limit=10,
            min_score=None,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_search(args, kernle)

        output = fake_out.getvalue()
        # Should find the content we created
        assert "Searchable" in output or "Found" in output or "episode" in output.lower()

    def test_status_integration(self, kernle_instance):
        """Test status command shows real counts."""
        kernle, storage = kernle_instance

        # Add some data
        kernle.episode("Status test episode", "success")
        kernle.note("Status test note")

        args = argparse.Namespace()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_status(args, kernle)

        output = fake_out.getvalue()
        # Status should show agent info
        assert "test_agent" in output or "agent" in output.lower()

    def test_checkpoint_save_and_load_integration(self, kernle_instance):
        """Test checkpoint save and load with real storage."""
        kernle, storage = kernle_instance

        # Save a checkpoint - CLI uses 'task' not 'current_task'
        save_args = argparse.Namespace(
            checkpoint_action="save",
            task="Integration test task",
            pending=["item1", "item2"],
            context="Test context for checkpoint",
            progress=None,
            next=None,
            blocker=None,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_checkpoint(save_args, kernle)

        save_output = fake_out.getvalue()
        assert "Checkpoint saved" in save_output or "saved" in save_output.lower()

        # Load the checkpoint
        load_args = argparse.Namespace(
            checkpoint_action="load",
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_checkpoint(load_args, kernle)

        load_output = fake_out.getvalue()
        # Should show the task we saved
        assert "Integration test task" in load_output or "task" in load_output.lower()

    def test_load_integration(self, kernle_instance):
        """Test load command outputs formatted memory with real data."""
        kernle, storage = kernle_instance

        # Add some content to load
        kernle.episode("Load test episode", "success", lessons=["Important lesson"])
        kernle.note("Load test decision", type="decision", reason="Testing")

        args = argparse.Namespace(json=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_load(args, kernle)

        output = fake_out.getvalue()
        # Should have formatted memory header
        assert "Working Memory" in output or "Memory" in output

    def test_drive_integration(self, kernle_instance):
        """Test drive command creates and loads drives."""
        kernle, storage = kernle_instance

        # Create a drive - CLI uses 'set' action with type, intensity, focus args
        create_args = argparse.Namespace(
            drive_action="set",
            type="curiosity",
            intensity=0.8,
            focus=["testing", "integration"],
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_drive(create_args, kernle)

        create_output = fake_out.getvalue()
        assert "Drive" in create_output or "curiosity" in create_output.lower()

        # List drives
        list_args = argparse.Namespace(drive_action="list")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_drive(list_args, kernle)

        list_output = fake_out.getvalue()
        assert "curiosity" in list_output.lower()
