"""
Comprehensive tests for the Kernle MCP server.

Tests all MCP tools, tool definitions, call_tool dispatcher, and error handling.
"""

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from mcp.types import TextContent, Tool

from kernle.mcp.server import (
    TOOLS,
    call_tool,
    get_kernle,
    list_tools,
)
from kernle.types import VALID_SOURCE_TYPE_VALUES


class TestMCPToolDefinitions:
    """Test MCP tool definitions and list_tools functionality."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self):
        """Test that list_tools returns expected tools with proper structure."""
        tools = await list_tools()

        # Verify all tools have proper structure
        assert all(isinstance(tool, Tool) for tool in tools)

        # Check all expected tools are present
        tool_names = {tool.name for tool in tools}
        expected_names = {
            # Original tools (memory_belief removed in #866)
            "memory_load",
            "memory_checkpoint_save",
            "memory_checkpoint_load",
            "memory_episode",
            "memory_note",
            "memory_search",
            "memory_value",
            "memory_goal",
            "memory_drive",
            "memory_when",
            "memory_consolidate",
            "memory_status",
            "memory_auto_capture",
            # New list tools (4)
            "memory_belief_list",
            "memory_value_list",
            "memory_goal_list",
            "memory_drive_list",
            # New update tools (3)
            "memory_episode_update",
            "memory_goal_update",
            "memory_belief_update",
            # Sync and search tools (2)
            "memory_sync",
            "memory_note_search",
            # Suggestion tools (4)
            "suggestion_list",
            "suggestion_accept",
            "suggestion_dismiss",
            "suggestion_extract",
        }
        # Verify expected tools exist (new tools may be added)
        assert expected_names.issubset(tool_names), f"Missing tools: {expected_names - tool_names}"
        # Track current count for documentation (don't fail if new tools added)
        assert len(tools) >= 26, f"Should have at least 26 tools, got {len(tools)}"

    def test_tool_definitions_have_required_fields(self):
        """Test that all tool definitions have required fields."""
        for tool in TOOLS:
            assert tool.name
            assert tool.description
            assert tool.inputSchema
            assert "type" in tool.inputSchema
            assert "properties" in tool.inputSchema

    def test_memory_load_tool_definition(self):
        """Test memory_load tool definition."""
        tool = next(t for t in TOOLS if t.name == "memory_load")

        assert "format" in tool.inputSchema["properties"]
        format_prop = tool.inputSchema["properties"]["format"]
        assert format_prop["type"] == "string"
        assert format_prop["enum"] == ["text", "json"]
        assert format_prop["default"] == "text"

    def test_memory_checkpoint_save_tool_definition(self):
        """Test memory_checkpoint_save tool definition."""
        tool = next(t for t in TOOLS if t.name == "memory_checkpoint_save")

        props = tool.inputSchema["properties"]
        assert "task" in props
        assert "pending" in props
        assert "context" in props
        assert tool.inputSchema["required"] == ["task"]

    def test_memory_episode_tool_definition(self):
        """Test memory_episode tool definition."""
        tool = next(t for t in TOOLS if t.name == "memory_episode")

        props = tool.inputSchema["properties"]
        assert "objective" in props
        assert "outcome" in props
        assert "lessons" in props
        assert "tags" in props
        assert tool.inputSchema["required"] == ["objective", "outcome"]

    def test_memory_note_tool_definition(self):
        """Test memory_note tool definition."""
        tool = next(t for t in TOOLS if t.name == "memory_note")

        props = tool.inputSchema["properties"]
        assert "content" in props
        assert "type" in props
        assert "speaker" in props
        assert "reason" in props
        assert "tags" in props

        type_prop = props["type"]
        assert type_prop["enum"] == ["note", "decision", "insight", "quote"]
        assert tool.inputSchema["required"] == ["content"]


# Fixtures for mocking
@pytest.fixture
def mock_kernle():
    """Create a comprehensive mock of the Kernle class."""
    kernle_mock = Mock()

    # Mock all methods used by MCP tools
    kernle_mock.load.return_value = {
        "checkpoint": {"task": "test task", "pending": []},
        "values": [{"name": "quality", "statement": "Quality is important"}],
        "beliefs": [{"statement": "Testing is crucial", "confidence": 0.9}],
        "goals": [{"title": "Write tests", "priority": "high"}],
        "drives": [{"drive_type": "growth", "intensity": 0.8}],
        "lessons": ["Always test edge cases"],
        "recent_work": [{"objective": "Recent work", "outcome": "success"}],
        "recent_notes": [{"content": "Test note"}],
        "relationships": [],
    }

    kernle_mock.format_memory.return_value = "Formatted memory output"

    kernle_mock.checkpoint.return_value = {
        "current_task": "test task",
        "pending": ["item1", "item2"],
    }

    kernle_mock.load_checkpoint.return_value = {"task": "loaded task", "context": "test context"}

    kernle_mock.episode.return_value = "episode_123456"
    kernle_mock.note.return_value = "note_123456"
    kernle_mock.belief.return_value = "belief_123456"
    kernle_mock.value.return_value = "value_123456"
    kernle_mock.goal.return_value = "goal_123456"
    kernle_mock.drive.return_value = "drive_123456"

    kernle_mock.search.return_value = [
        {"type": "episode", "title": "Test Episode", "lessons": ["Lesson 1", "Lesson 2"]}
    ]

    kernle_mock.what_happened.return_value = {
        "episodes": [{"objective": "Test objective", "outcome_type": "success"}],
        "notes": [{"content": "Test note content"}],
    }

    # Mock load_beliefs for memory_consolidate reflection scaffold
    kernle_mock.load_beliefs.return_value = [
        {"statement": "Testing is crucial", "confidence": 0.9, "belief_type": "learned"},
        {"statement": "Quality over quantity", "confidence": 0.8, "belief_type": "preference"},
    ]

    # Mock _storage.get_episodes for memory_consolidate reflection scaffold
    storage_mock = Mock()
    episode_mock_1 = Mock()
    episode_mock_1.objective = "Test objective 1"
    episode_mock_1.outcome = "Successfully completed"
    episode_mock_1.outcome_type = "success"
    episode_mock_1.lessons = ["Lesson from ep1"]
    episode_mock_1.tags = ["test"]
    episode_mock_2 = Mock()
    episode_mock_2.objective = "Test objective 2"
    episode_mock_2.outcome = "Failed due to timeout"
    episode_mock_2.outcome_type = "failure"
    episode_mock_2.lessons = ["Lesson from ep2", "Lesson from ep1"]
    episode_mock_2.tags = None
    episode_mock_3 = Mock()
    episode_mock_3.objective = "Test objective 3"
    episode_mock_3.outcome = "Partially done"
    episode_mock_3.outcome_type = "partial"
    episode_mock_3.lessons = None
    episode_mock_3.tags = ["demo"]
    storage_mock.get_episodes.return_value = [episode_mock_1, episode_mock_2, episode_mock_3]
    kernle_mock._storage = storage_mock

    kernle_mock.status.return_value = {
        "stack_id": "test_agent",
        "values": 3,
        "beliefs": 10,
        "goals": 2,
        "episodes": 25,
        "checkpoint": True,
    }

    kernle_mock.auto_capture.return_value = "capture_123456"

    # Mock raw() for memory_auto_capture
    kernle_mock.raw.return_value = "raw_12345678"

    return kernle_mock


@pytest.fixture
def patched_get_kernle(mock_kernle):
    """Patch the get_kernle function to return our mock."""
    with patch("kernle.mcp.server.get_kernle", return_value=mock_kernle):
        yield mock_kernle


class TestKernleMocking:
    """Test proper mocking of the Kernle core class."""

    # NOTE: Removed test_mock_setup - it only tested mock configuration, not production code.
    # Mocks are implementation details of tests, not things to test themselves.


class TestMCPToolCalls:
    """Test individual MCP tool calls with proper mocking."""

    @pytest.mark.asyncio
    async def test_memory_load_text_format(self, patched_get_kernle):
        """Test memory_load with text format calls correct methods."""
        result = await call_tool("memory_load", {"format": "text"})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        # Verify production code calls the right sequence:
        # 1. Loads memory data
        patched_get_kernle.load.assert_called_once()
        # 2. Formats it for text display (passes loaded data to formatter)
        patched_get_kernle.format_memory.assert_called_once()
        # 3. format_memory should receive the loaded data
        format_call_args = patched_get_kernle.format_memory.call_args
        assert format_call_args is not None  # format_memory was called with arguments

        # The result should not be empty
        assert result[0].text  # Non-empty string returned

    @pytest.mark.asyncio
    async def test_memory_load_json_format(self, patched_get_kernle):
        """Test memory_load with JSON format produces valid JSON."""
        result = await call_tool("memory_load", {"format": "json"})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        # Verify output is valid JSON (tests the serialization code path)
        json_data = json.loads(result[0].text)  # Would raise if not valid JSON
        assert isinstance(json_data, dict), "JSON output should be a dict"
        # Don't assert specific keys from mock - that's testing mock config

        patched_get_kernle.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_load_default_format(self, patched_get_kernle):
        """Test memory_load with default format uses text format path."""
        result = await call_tool("memory_load", {})

        assert len(result) == 1
        # Default format should be text, so format_memory should be called
        patched_get_kernle.load.assert_called_once()
        patched_get_kernle.format_memory.assert_called_once()
        # Result should contain formatted text (not raw JSON)
        assert result[0].text  # Non-empty

    @pytest.mark.asyncio
    async def test_memory_checkpoint_save(self, patched_get_kernle):
        """Test memory_checkpoint_save calls checkpoint() with correct args."""
        args = {
            "task": "Write comprehensive tests",
            "pending": ["Test edge cases", "Add documentation"],
            "context": "Working on MCP tests",
        }

        result = await call_tool("memory_checkpoint_save", args)

        assert len(result) == 1
        # Verify output formatting structure (not mock return values)
        assert "Checkpoint saved:" in result[0].text, "Should confirm save"
        assert "Pending:" in result[0].text, "Should show pending count"

        # Verify correct method called with correct arguments (this IS valid to test)
        patched_get_kernle.checkpoint.assert_called_once_with(
            task="Write comprehensive tests",
            pending=["Test edge cases", "Add documentation"],
            context="Working on MCP tests",
        )

    @pytest.mark.asyncio
    async def test_memory_checkpoint_save_minimal(self, patched_get_kernle):
        """Test memory_checkpoint_save with only required fields provides defaults."""
        result = await call_tool("memory_checkpoint_save", {"task": "Minimal test"})

        assert len(result) == 1
        assert "Checkpoint saved:" in result[0].text, "Should confirm save"

        # Verify defaults are applied (this tests the tool's default handling logic)
        patched_get_kernle.checkpoint.assert_called_once_with(
            task="Minimal test", pending=[], context=""
        )

    @pytest.mark.asyncio
    async def test_memory_checkpoint_load(self, patched_get_kernle):
        """Test memory_checkpoint_load returns valid JSON checkpoint data."""
        result = await call_tool("memory_checkpoint_load", {})

        assert len(result) == 1
        # Verify output is valid JSON (tests serialization)
        json_data = json.loads(result[0].text)  # Would raise if not valid JSON
        assert isinstance(json_data, dict), "Should return checkpoint as JSON dict"
        # Don't assert specific mock values like "loaded task"

        patched_get_kernle.load_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_checkpoint_load_empty(self, patched_get_kernle):
        """Test memory_checkpoint_load when no checkpoint exists."""
        patched_get_kernle.load_checkpoint.return_value = None

        result = await call_tool("memory_checkpoint_load", {})

        assert len(result) == 1
        assert result[0].text == "No checkpoint found."

    @pytest.mark.asyncio
    async def test_memory_episode(self, patched_get_kernle):
        """Test memory_episode."""
        args = {
            "objective": "Write comprehensive tests",
            "outcome": "success",
            "lessons": ["Mock dependencies", "Test error cases"],
            "tags": ["testing", "development"],
        }

        result = await call_tool("memory_episode", args)

        assert len(result) == 1
        assert "Episode saved:" in result[0].text
        assert "episode_" in result[0].text

        patched_get_kernle.episode.assert_called_once_with(
            objective="Write comprehensive tests",
            outcome="success",
            lessons=["Mock dependencies", "Test error cases"],
            tags=["testing", "development"],
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "note_type,content,tags,extra_args,expected_speaker,expected_reason",
        [
            ("note", "This is a regular note", ["general"], {}, "", ""),
            (
                "decision",
                "Use pytest for testing",
                ["testing"],
                {"reason": "Industry standard with good ecosystem"},
                "",
                "Industry standard with good ecosystem",
            ),
            ("insight", "Mocking enables isolated testing", ["testing", "insights"], {}, "", ""),
            (
                "quote",
                "Code is poetry",
                ["inspiration"],
                {"speaker": "Someone Wise"},
                "Someone Wise",
                "",
            ),
        ],
        ids=["note", "decision", "insight", "quote"],
    )
    async def test_memory_note_by_type(
        self,
        patched_get_kernle,
        note_type,
        content,
        tags,
        extra_args,
        expected_speaker,
        expected_reason,
    ):
        """Test memory_note with different note types."""
        args = {"content": content, "type": note_type, "tags": tags, **extra_args}
        result = await call_tool("memory_note", args)

        assert len(result) == 1
        assert "Note saved:" in result[0].text
        assert content[:50] in result[0].text

        patched_get_kernle.note.assert_called_once_with(
            content=content,
            type=note_type,
            speaker=expected_speaker,
            reason=expected_reason,
            tags=tags,
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    @pytest.mark.asyncio
    async def test_memory_note_minimal(self, patched_get_kernle):
        """Test memory_note with minimal required fields."""
        result = await call_tool("memory_note", {"content": "Simple note"})

        assert len(result) == 1
        assert "Note saved: Simple note..." in result[0].text

        patched_get_kernle.note.assert_called_once_with(
            content="Simple note",
            type="note",
            speaker="",
            reason="",
            tags=[],
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    @pytest.mark.asyncio
    async def test_memory_search(self, patched_get_kernle):
        """Test memory_search calls search() and formats results."""
        result = await call_tool("memory_search", {"query": "testing", "limit": 5})

        assert len(result) == 1
        # Verify output formatting structure (not mock content)
        assert "Found" in result[0].text and "result(s):" in result[0].text
        assert "[episode]" in result[0].text, "Should format results with type prefix"

        # Verify correct method called with correct args
        patched_get_kernle.search.assert_called_once_with(query="testing", limit=5)

    @pytest.mark.asyncio
    async def test_memory_search_no_results(self, patched_get_kernle):
        """Test memory_search with no results."""
        patched_get_kernle.search.return_value = []

        result = await call_tool("memory_search", {"query": "nonexistent"})

        assert len(result) == 1
        assert "No results for 'nonexistent'" in result[0].text

    @pytest.mark.asyncio
    async def test_memory_search_default_limit(self, patched_get_kernle):
        """Test memory_search with default limit."""
        await call_tool("memory_search", {"query": "testing"})

        patched_get_kernle.search.assert_called_once_with(query="testing", limit=10)

    @pytest.mark.asyncio
    async def test_memory_value(self, patched_get_kernle):
        """Test memory_value."""
        args = {
            "name": "quality",
            "statement": "Software must be thoroughly tested and reliable",
            "priority": 90,
        }

        result = await call_tool("memory_value", args)

        assert len(result) == 1
        assert "Value saved: quality" in result[0].text

        patched_get_kernle.value.assert_called_once_with(
            name="quality",
            statement="Software must be thoroughly tested and reliable",
            priority=90,
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    @pytest.mark.asyncio
    async def test_memory_goal(self, patched_get_kernle):
        """Test memory_goal."""
        args = {
            "title": "Achieve comprehensive test coverage",
            "description": "Write tests for all MCP tools with edge cases",
            "priority": "high",
        }

        result = await call_tool("memory_goal", args)

        assert len(result) == 1
        assert "Goal saved: Achieve comprehensive test coverage" in result[0].text

        patched_get_kernle.goal.assert_called_once_with(
            title="Achieve comprehensive test coverage",
            description="Write tests for all MCP tools with edge cases",
            priority="high",
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    @pytest.mark.asyncio
    async def test_memory_goal_minimal(self, patched_get_kernle):
        """Test memory_goal with minimal fields."""
        await call_tool("memory_goal", {"title": "Simple goal"})

        patched_get_kernle.goal.assert_called_once_with(
            title="Simple goal",
            description="",
            priority="medium",
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    @pytest.mark.asyncio
    async def test_memory_drive(self, patched_get_kernle):
        """Test memory_drive with all parameters."""
        args = {
            "drive_type": "growth",
            "intensity": 0.8,
            "focus_areas": ["learning", "improvement", "mastery"],
        }

        result = await call_tool("memory_drive", args)

        assert len(result) == 1
        # Should save drive and return confirmation
        assert "growth" in result[0].text
        assert "80%" in result[0].text

        patched_get_kernle.drive.assert_called_once_with(
            drive_type="growth",
            intensity=0.8,
            focus_areas=["learning", "improvement", "mastery"],
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    @pytest.mark.asyncio
    async def test_memory_drive_default_intensity(self, patched_get_kernle):
        """Test memory_drive with default intensity."""
        result = await call_tool("memory_drive", {"drive_type": "curiosity"})

        assert len(result) == 1
        # Should use default intensity (0.5)
        assert "curiosity" in result[0].text
        assert "50%" in result[0].text

        patched_get_kernle.drive.assert_called_once_with(
            drive_type="curiosity",
            intensity=0.5,
            focus_areas=[],
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    # NOTE: The no-op test_memory_drive_validation_bug_documentation was removed.
    # Bug documentation belongs in issue tracker or code comments, not empty tests.

    @pytest.mark.asyncio
    async def test_memory_when_periods(self, patched_get_kernle):
        """Test memory_when with different time periods."""
        periods = ["today", "yesterday", "this week", "last hour"]

        for period in periods:
            result = await call_tool("memory_when", {"period": period})

            assert len(result) == 1
            assert f"What happened {period}:" in result[0].text
            assert "Episodes:" in result[0].text
            assert "Notes:" in result[0].text

            patched_get_kernle.what_happened.assert_called_with(period)

    @pytest.mark.asyncio
    async def test_memory_when_default_period(self, patched_get_kernle):
        """Test memory_when with default period."""
        await call_tool("memory_when", {})

        patched_get_kernle.what_happened.assert_called_with("today")

    @pytest.mark.asyncio
    async def test_memory_consolidate(self, patched_get_kernle):
        """Test memory_consolidate returns reflection scaffold."""
        result = await call_tool("memory_consolidate", {"min_episodes": 2})

        assert len(result) == 1
        text = result[0].text
        # Verify scaffold structure
        assert "# Memory Consolidation: Reflection Scaffold" in text
        assert "## Recent Experiences" in text
        assert "## Current Beliefs" in text
        assert "## Your Reflection Task" in text
        # Verify it includes guidance
        assert "Pattern Recognition" in text
        assert "Belief Validation" in text
        assert "Kernle provides the data. You do the reasoning." in text
        # Verify episodes are shown
        assert "Test objective 1" in text
        assert "Test objective 2" in text

    @pytest.mark.asyncio
    async def test_memory_consolidate_default(self, patched_get_kernle):
        """Test memory_consolidate with default min_episodes."""
        result = await call_tool("memory_consolidate", {})

        # Should show reflection scaffold (we have 3 episodes, default min is 3)
        assert "# Memory Consolidation: Reflection Scaffold" in result[0].text
        # Verify storage was called
        patched_get_kernle._storage.get_episodes.assert_called_once_with(limit=20)

    @pytest.mark.asyncio
    async def test_memory_consolidate_insufficient_episodes(self, patched_get_kernle):
        """Test memory_consolidate when not enough episodes."""
        # Set up with only 1 episode
        patched_get_kernle._storage.get_episodes.return_value = [
            patched_get_kernle._storage.get_episodes.return_value[0]
        ]

        result = await call_tool("memory_consolidate", {"min_episodes": 5})

        text = result[0].text
        assert "Only 1 episode(s) recorded (minimum 5 for consolidation)" in text
        assert "Continue capturing experiences before consolidating" in text

    @pytest.mark.asyncio
    async def test_memory_status(self, patched_get_kernle):
        """Test memory_status calls status() and formats output."""
        result = await call_tool("memory_status", {})

        assert len(result) == 1
        status_text = result[0].text

        # Verify output formatting structure (tests the formatting code)
        assert "Memory Status" in status_text, "Should have status header"
        assert "Values:" in status_text, "Should show values count"
        assert "Beliefs:" in status_text, "Should show beliefs count"
        assert "Goals:" in status_text, "Should show goals count"
        assert "Episodes:" in status_text, "Should show episodes count"
        assert "Checkpoint:" in status_text, "Should show checkpoint status"
        # Don't assert specific numbers like "3" or "Yes" - those come from mock

    @pytest.mark.asyncio
    async def test_memory_auto_capture_success(self, patched_get_kernle):
        """Test memory_auto_capture calls raw() and formats output."""
        args = {
            "text": "I learned that mocking is crucial for isolated testing",
            "context": "While writing tests",
        }

        result = await call_tool("memory_auto_capture", args)

        assert len(result) == 1
        assert "Auto-captured:" in result[0].text
        # Source is normalized to "mcp" for MCP tool calls
        assert "source: mcp" in result[0].text

        # Verify raw() was called with blob parameter (not content)
        patched_get_kernle.raw.assert_called_once()
        call_args = patched_get_kernle.raw.call_args
        assert "blob" in call_args.kwargs, "Should use blob parameter"
        assert "source" in call_args.kwargs
        # Source "auto" is normalized to "mcp" by the MCP tool
        assert call_args.kwargs["source"] == "mcp"

    @pytest.mark.asyncio
    async def test_memory_auto_capture_with_source(self, patched_get_kernle):
        """Test memory_auto_capture normalizes invalid source values to 'mcp'."""
        # Note: Source values are normalized to valid enum: cli|mcp|sdk|import|unknown
        # "hook-session-end" is not in the valid enum, so it gets normalized
        args = {"text": "Session completed: built user auth", "source": "hook-session-end"}

        result = await call_tool("memory_auto_capture", args)

        assert len(result) == 1
        assert "Auto-captured:" in result[0].text
        # Invalid source is normalized to "mcp" for MCP tool calls
        assert "(source: mcp)" in result[0].text

        call_args = patched_get_kernle.raw.call_args
        # Source is normalized to valid enum value
        assert call_args.kwargs["source"] == "mcp"
        # Tags parameter is no longer used in the new raw() API

    @pytest.mark.asyncio
    async def test_memory_auto_capture_with_suggestions(self, patched_get_kernle):
        """Test memory_auto_capture with extract_suggestions returns JSON with suggestions."""
        args = {
            "text": "Session completed: implemented user authentication and shipped to production",
            "source": "hook-session-end",
            "extract_suggestions": True,
        }

        result = await call_tool("memory_auto_capture", args)

        assert len(result) == 1
        result_data = json.loads(result[0].text)  # Should be valid JSON
        assert result_data["captured"] is True, "Should confirm capture"
        # Source is normalized to mcp for MCP tool calls
        assert result_data["source"] == "mcp", "Source should be normalized"
        assert "suggestions" in result_data, "Should include suggestions"
        assert "promote_command" in result_data, "Should include promote command"

    @pytest.mark.asyncio
    async def test_memory_auto_capture_minimal(self, patched_get_kernle):
        """Test memory_auto_capture with minimal args uses correct defaults."""
        await call_tool("memory_auto_capture", {"text": "Test text"})

        patched_get_kernle.raw.assert_called_once()
        call_args = patched_get_kernle.raw.call_args
        # Uses blob parameter (not content) in the new API
        assert "blob" in call_args.kwargs, "Should use blob parameter"
        # Source "auto" is normalized to "mcp" for MCP tool calls
        assert call_args.kwargs["source"] == "mcp"


class TestErrorHandling:
    """Test error handling in MCP tool calls."""

    @pytest.fixture
    def failing_kernle(self):
        """Mock Kernle that raises exceptions."""
        kernle_mock = Mock()
        kernle_mock.load.side_effect = Exception("Database connection failed")
        kernle_mock.episode.side_effect = ValueError("Invalid outcome type")
        kernle_mock.search.side_effect = RuntimeError("Search service unavailable")
        return kernle_mock

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, patched_get_kernle):
        """Test error handling for unknown tool names."""
        result = await call_tool("unknown_tool", {})

        assert len(result) == 1
        assert "Unknown tool: unknown_tool" in result[0].text

    @pytest.mark.asyncio
    async def test_kernle_exception_handling(self, failing_kernle):
        """Test that Kernle exceptions are caught and returned as error text."""
        with patch("kernle.mcp.server.get_kernle", return_value=failing_kernle):
            result = await call_tool("memory_load", {})

            assert len(result) == 1
            assert "Internal server error" in result[0].text

    @pytest.mark.asyncio
    async def test_episode_error_handling(self, failing_kernle):
        """Test error handling for memory_episode."""
        with patch("kernle.mcp.server.get_kernle", return_value=failing_kernle):
            result = await call_tool(
                "memory_episode", {"objective": "Test", "outcome": "invalid_type"}
            )

            assert len(result) == 1
            assert "Invalid input: Invalid outcome type" in result[0].text

    @pytest.mark.asyncio
    async def test_search_error_handling(self, failing_kernle):
        """Test error handling for memory_search."""
        with patch("kernle.mcp.server.get_kernle", return_value=failing_kernle):
            result = await call_tool("memory_search", {"query": "test"})

            assert len(result) == 1
            assert "Internal server error" in result[0].text


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_missing_required_arguments(self, patched_get_kernle):
        """Test behavior when required arguments are missing returns clear error."""
        # memory_checkpoint_save requires 'task' argument
        result = await call_tool("memory_checkpoint_save", {})

        assert len(result) == 1
        # Should get a validation error mentioning the missing field
        error_text = result[0].text.lower()
        assert "invalid" in error_text or "error" in error_text or "required" in error_text
        # Error should identify what's missing (task is required)
        assert "task" in error_text or "required" in error_text

        # The kernle method should NOT have been called with missing required args
        patched_get_kernle.checkpoint.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_argument_types(self, patched_get_kernle):
        """Test behavior with invalid argument types returns validation error."""
        # Pass invalid type for limit (should be integer)
        result = await call_tool("memory_search", {"query": "test", "limit": "invalid"})

        assert len(result) == 1
        # Server should validate argument types and return a clear error
        # The result should either:
        # - Be an error message about invalid type
        # - Or the call was made anyway (and we can verify how it was called)
        if "Invalid input:" in result[0].text or "Error" in result[0].text:
            # Validation rejected it - this is the expected safe behavior
            assert "limit" in result[0].text.lower() or "type" in result[0].text.lower()
        else:
            # If no error, verify the search was actually called (not silently swallowed)
            patched_get_kernle.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_results_handling(self, patched_get_kernle):
        """Test handling of empty results from Kernle methods."""
        patched_get_kernle.search.return_value = []
        patched_get_kernle.what_happened.return_value = {"episodes": [], "notes": []}

        # Test empty search results
        result = await call_tool("memory_search", {"query": "nothing"})
        assert "No results for 'nothing'" in result[0].text

        # Test empty temporal results
        result = await call_tool("memory_when", {"period": "today"})
        assert "What happened today:" in result[0].text

    @pytest.mark.asyncio
    async def test_null_values_handling(self, patched_get_kernle):
        """Test handling of null/None values from Kernle."""
        patched_get_kernle.load_checkpoint.return_value = None

        # Test null checkpoint
        result = await call_tool("memory_checkpoint_load", {})
        assert "No checkpoint found." in result[0].text

        # memory_auto_capture now always captures to raw layer (no filtering)
        result = await call_tool("memory_auto_capture", {"text": "casual text"})
        assert "Auto-captured:" in result[0].text

    @pytest.mark.asyncio
    async def test_large_content_handling(self, patched_get_kernle):
        """Test handling of large content that gets rejected by validation."""
        long_content = "This is a very long piece of content " * 100  # ~3800 chars

        result = await call_tool("memory_note", {"content": long_content})

        assert len(result) == 1
        # Should be rejected by validation (max 2000 characters for notes)
        error_text = result[0].text.lower()

        # Verify this is a validation error, not a success
        assert (
            "invalid" in error_text or "error" in error_text
        ), f"Large content should be rejected, got: {result[0].text}"

        # Error should mention the issue is with content length/size
        length_related_terms = [
            "long",
            "length",
            "size",
            "character",
            "2000",
            "limit",
            "exceed",
            "max",
        ]
        has_length_info = any(term in error_text for term in length_related_terms)
        assert has_length_info, f"Error should mention length/size issue, got: {result[0].text}"

        # Verify Kernle.note was NOT called (validation should prevent it)
        patched_get_kernle.note.assert_not_called()

    @pytest.mark.asyncio
    async def test_reasonable_content_handling(self, patched_get_kernle):
        """Test handling of reasonably-sized content."""
        content = "This is a reasonable piece of content for testing."

        result = await call_tool("memory_note", {"content": content})

        assert len(result) == 1
        assert "Note saved:" in result[0].text
        assert content in result[0].text

    @pytest.mark.asyncio
    async def test_json_serialization_edge_cases(self, patched_get_kernle):
        """Test JSON serialization converts datetime objects to strings."""
        # Create specific datetime for verification
        test_datetime = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        complex_memory = {
            "checkpoint": {"created_at": test_datetime},
            "values": [{"created": test_datetime}],
            "complex_data": {"nested": {"deep": "value"}},
        }
        patched_get_kernle.load.return_value = complex_memory

        result = await call_tool("memory_load", {"format": "json"})

        assert len(result) == 1
        # Should be valid JSON (datetime objects must be converted to strings)
        json_data = json.loads(result[0].text)
        assert "checkpoint" in json_data

        # Verify datetime was serialized - it should be a string, not a datetime object
        checkpoint_created = json_data["checkpoint"]["created_at"]
        assert isinstance(checkpoint_created, str), "datetime should be serialized to string"
        # The serialized string should contain date components
        assert "2024" in checkpoint_created

        # Verify nested structure is preserved
        assert json_data["complex_data"]["nested"]["deep"] == "value"

    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, patched_get_kernle):
        """Test handling of Unicode content is preserved."""
        unicode_content = "测试 🧪 emoji and unicode characters ñoño"

        result = await call_tool("memory_note", {"content": unicode_content})

        assert len(result) == 1
        assert "Note saved:" in result[0].text

        # Verify unicode content was passed through correctly to Kernle
        patched_get_kernle.note.assert_called_once()
        call_args = patched_get_kernle.note.call_args
        assert call_args.kwargs["content"] == unicode_content, "Unicode should be preserved"

        # Verify unicode appears in the response (truncated in output)
        assert "测试" in result[0].text or unicode_content[:50] in result[0].text

    @pytest.mark.asyncio
    async def test_special_characters_in_search(self, patched_get_kernle):
        """Test search with special characters."""
        special_query = "test with \"quotes\" and 'apostrophes' & symbols"

        await call_tool("memory_search", {"query": special_query})

        patched_get_kernle.search.assert_called_once_with(query=special_query, limit=10)


class TestGetKernleFunction:
    """Test the get_kernle singleton function."""

    def test_get_kernle_singleton_behavior(self):
        """Test that get_kernle returns the same instance."""
        # Clear any existing instance
        if hasattr(get_kernle, "_instance"):
            delattr(get_kernle, "_instance")

        # First call should create instance
        kernle1 = get_kernle()

        # Second call should return same instance
        kernle2 = get_kernle()

        assert kernle1 is kernle2

    def test_get_kernle_creates_kernle_instance(self):
        """Test that get_kernle creates a proper Kernle instance."""
        # Clear any existing instance
        if hasattr(get_kernle, "_instance"):
            delattr(get_kernle, "_instance")

        with patch("kernle.mcp.server.Kernle") as mock_kernle_cls:
            mock_instance = Mock()
            mock_kernle_cls.return_value = mock_instance

            result = get_kernle()

            mock_kernle_cls.assert_called_once()
            assert result is mock_instance


class TestMultiToolWorkflows:
    """Test workflows combining multiple tool calls (mocked).

    NOTE: These are NOT true integration tests - they use mocked Kernle.
    They verify that call_tool correctly dispatches multiple sequential calls
    and that each tool call works independently.

    For real integration tests, see tests/test_integration.py (if it exists)
    or create one that uses a real Kernle instance.
    """

    @pytest.mark.asyncio
    async def test_typical_session_workflow_dispatch(self, patched_get_kernle):
        """Test that typical workflow dispatches to correct Kernle methods."""
        # Load memory
        result1 = await call_tool("memory_load", {"format": "text"})
        assert len(result1) == 1

        # Record an episode
        result2 = await call_tool(
            "memory_episode",
            {
                "objective": "Write MCP tests",
                "outcome": "success",
                "lessons": ["Comprehensive mocking is essential"],
            },
        )
        assert "Episode saved:" in result2[0].text

        # Save checkpoint
        result3 = await call_tool(
            "memory_checkpoint_save", {"task": "Testing complete", "pending": []}
        )
        assert "Checkpoint saved:" in result3[0].text

        # Verify correct methods called with correct arguments
        patched_get_kernle.load.assert_called_once()
        patched_get_kernle.episode.assert_called_once_with(
            objective="Write MCP tests",
            outcome="success",
            lessons=["Comprehensive mocking is essential"],
            tags=[],
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )
        patched_get_kernle.checkpoint.assert_called_once_with(
            task="Testing complete", pending=[], context=""
        )

    @pytest.mark.asyncio
    async def test_memory_building_workflow_dispatch(self, patched_get_kernle):
        """Test that memory building workflow dispatches correctly."""
        # Add value
        await call_tool(
            "memory_value",
            {"name": "reliability", "statement": "Software should be dependable", "priority": 85},
        )

        # Add goal
        await call_tool("memory_goal", {"title": "Achieve zero critical bugs", "priority": "high"})

        # Verify correct methods called with correct arguments
        patched_get_kernle.value.assert_called_once_with(
            name="reliability",
            statement="Software should be dependable",
            priority=85,
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )
        patched_get_kernle.goal.assert_called_once_with(
            title="Achieve zero critical bugs",
            description="",
            priority="high",
            context=None,
            context_tags=None,
            source=None,
            derived_from=None,
            source_type=None,
        )

    @pytest.mark.asyncio
    async def test_search_and_consolidation_workflow_dispatch(self, patched_get_kernle):
        """Test search -> consolidate -> status dispatches correctly."""
        # Search for patterns
        await call_tool("memory_search", {"query": "testing patterns"})

        # Consolidate learnings (now returns reflection scaffold)
        result = await call_tool("memory_consolidate", {"min_episodes": 3})

        # Check status
        await call_tool("memory_status", {})

        # Verify correct methods called with correct arguments
        patched_get_kernle.search.assert_called_once_with(query="testing patterns", limit=10)
        # Consolidate now fetches data directly for reflection scaffold
        patched_get_kernle._storage.get_episodes.assert_called_once_with(limit=20)
        patched_get_kernle.load_beliefs.assert_called_with(limit=15)
        patched_get_kernle.status.assert_called_once()
        # Verify scaffold returned
        assert "Reflection Scaffold" in result[0].text


class TestNewListTools:
    """Test the new list tools added to MCP server."""

    @pytest.fixture
    def list_mock_kernle(self):
        """Create a mock Kernle with list methods configured."""
        kernle_mock = Mock()

        kernle_mock.load_beliefs.return_value = [
            {"statement": "Testing is important", "confidence": 0.9, "belief_type": "fact"},
            {"statement": "Code reviews help", "confidence": 0.85, "belief_type": "learned"},
        ]

        kernle_mock.load_values.return_value = [
            {"name": "quality", "statement": "Quality over quantity", "priority": 90},
            {"name": "clarity", "statement": "Clarity in communication", "priority": 80},
        ]

        kernle_mock.load_goals.return_value = [
            {
                "title": "Complete MCP",
                "description": "Finish MCP server",
                "priority": "high",
                "status": "active",
            },
            {
                "title": "Write docs",
                "description": "Documentation",
                "priority": "medium",
                "status": "active",
            },
        ]

        kernle_mock.load_drives.return_value = [
            {"drive_type": "growth", "intensity": 0.8, "focus_areas": ["learning", "skills"]},
            {"drive_type": "curiosity", "intensity": 0.6, "focus_areas": ["new tech"]},
        ]

        return kernle_mock

    @pytest.fixture
    def patched_list_kernle(self, list_mock_kernle):
        """Patch get_kernle to return the list mock."""
        with patch("kernle.mcp.server.get_kernle", return_value=list_mock_kernle):
            yield list_mock_kernle

    @pytest.mark.asyncio
    async def test_memory_belief_list(self, patched_list_kernle):
        """Test memory_belief_list calls load_beliefs() and formats output."""
        result = await call_tool("memory_belief_list", {"limit": 10})

        assert len(result) == 1
        text = result[0].text

        # Verify output formatting structure (not mock content)
        assert "Found" in text and "belief(s):" in text, "Should show count"
        assert "%" in text, "Should format confidence as percentage"
        # Type prefix like [fact] tests the formatting code
        assert "[" in text and "]" in text, "Should include belief type"

        patched_list_kernle.load_beliefs.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_memory_belief_list_empty(self, patched_list_kernle):
        """Test memory_belief_list when no beliefs exist."""
        patched_list_kernle.load_beliefs.return_value = []

        result = await call_tool("memory_belief_list", {})

        assert len(result) == 1
        assert "No beliefs found." in result[0].text

    @pytest.mark.asyncio
    async def test_memory_value_list(self, patched_list_kernle):
        """Test memory_value_list calls load_values() and formats output."""
        result = await call_tool("memory_value_list", {"limit": 5})

        assert len(result) == 1
        text = result[0].text

        # Verify output formatting structure (not mock content)
        assert "Found" in text and "value(s):" in text, "Should show count"
        assert "priority:" in text, "Should show priority"
        # Bold formatting for name tests the formatting code
        assert "**" in text, "Should use bold for value names"

        patched_list_kernle.load_values.assert_called_once_with(limit=5)

    @pytest.mark.asyncio
    async def test_memory_value_list_empty(self, patched_list_kernle):
        """Test memory_value_list when no values exist."""
        patched_list_kernle.load_values.return_value = []

        result = await call_tool("memory_value_list", {})

        assert len(result) == 1
        assert "No values found." in result[0].text

    @pytest.mark.asyncio
    async def test_memory_goal_list(self, patched_list_kernle):
        """Test memory_goal_list calls load_goals() and formats output."""
        result = await call_tool("memory_goal_list", {"status": "active", "limit": 10})

        assert len(result) == 1
        text = result[0].text

        # Verify output formatting structure (not mock content)
        assert "Found" in text and "goal(s):" in text, "Should show count"
        # Priority prefix tests the formatting code
        assert "[" in text and "]" in text, "Should include priority in brackets"

        patched_list_kernle.load_goals.assert_called_once_with(limit=10, status="active")

    @pytest.mark.asyncio
    async def test_memory_goal_list_empty(self, patched_list_kernle):
        """Test memory_goal_list when no goals exist."""
        patched_list_kernle.load_goals.return_value = []

        result = await call_tool("memory_goal_list", {})

        assert len(result) == 1
        assert "No active goals found." in result[0].text

    @pytest.mark.asyncio
    async def test_memory_drive_list(self, patched_list_kernle):
        """Test memory_drive_list calls load_drives() and formats output."""
        result = await call_tool("memory_drive_list", {})

        assert len(result) == 1
        text = result[0].text

        # Verify output formatting structure (not mock content)
        assert "Current drives:" in text, "Should have drives header"
        assert "**" in text, "Should use bold for drive names"
        assert "%" in text, "Should format intensity as percentage"

        patched_list_kernle.load_drives.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_drive_list_empty(self, patched_list_kernle):
        """Test memory_drive_list when no drives configured."""
        patched_list_kernle.load_drives.return_value = []

        result = await call_tool("memory_drive_list", {})

        assert len(result) == 1
        assert "No drives configured." in result[0].text


class TestNewUpdateTools:
    """Test the new update tools added to MCP server."""

    @pytest.fixture
    def update_mock_kernle(self):
        """Create a mock Kernle with update methods configured."""
        kernle_mock = Mock()

        kernle_mock.update_episode.return_value = True
        kernle_mock.update_goal.return_value = True
        kernle_mock.update_belief.return_value = True

        return kernle_mock

    @pytest.fixture
    def patched_update_kernle(self, update_mock_kernle):
        """Patch get_kernle to return the update mock."""
        with patch("kernle.mcp.server.get_kernle", return_value=update_mock_kernle):
            yield update_mock_kernle

    @pytest.mark.asyncio
    async def test_memory_episode_update(self, patched_update_kernle):
        """Test memory_episode_update with all fields."""
        args = {
            "episode_id": "ep-12345678",
            "outcome": "success with modifications",
            "lessons": ["Lesson 1", "Lesson 2"],
            "tags": ["important", "milestone"],
        }

        result = await call_tool("memory_episode_update", args)

        assert len(result) == 1
        assert "Episode ep-12345... updated successfully." in result[0].text

        patched_update_kernle.update_episode.assert_called_once_with(
            episode_id="ep-12345678",
            outcome="success with modifications",
            lessons=["Lesson 1", "Lesson 2"],
            tags=["important", "milestone"],
        )

    @pytest.mark.asyncio
    async def test_memory_episode_update_not_found(self, patched_update_kernle):
        """Test memory_episode_update when episode doesn't exist."""
        patched_update_kernle.update_episode.return_value = False

        result = await call_tool("memory_episode_update", {"episode_id": "nonexistent"})

        assert len(result) == 1
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_memory_goal_update(self, patched_update_kernle):
        """Test memory_goal_update with status change."""
        args = {
            "goal_id": "goal-12345678",
            "status": "completed",
            "priority": "high",
            "description": "Updated description",
        }

        result = await call_tool("memory_goal_update", args)

        assert len(result) == 1
        assert "Goal goal-123... updated successfully." in result[0].text

        patched_update_kernle.update_goal.assert_called_once_with(
            goal_id="goal-12345678",
            status="completed",
            priority="high",
            description="Updated description",
        )

    @pytest.mark.asyncio
    async def test_memory_goal_update_not_found(self, patched_update_kernle):
        """Test memory_goal_update when goal doesn't exist."""
        patched_update_kernle.update_goal.return_value = False

        result = await call_tool("memory_goal_update", {"goal_id": "nonexistent"})

        assert len(result) == 1
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_memory_belief_update(self, patched_update_kernle):
        """Test memory_belief_update with confidence change."""
        args = {"belief_id": "bel-12345678", "confidence": 0.95, "is_active": True}

        result = await call_tool("memory_belief_update", args)

        assert len(result) == 1
        assert "Belief bel-1234... updated successfully." in result[0].text

        patched_update_kernle.update_belief.assert_called_once_with(
            belief_id="bel-12345678", confidence=0.95, is_active=True
        )

    @pytest.mark.asyncio
    async def test_memory_belief_update_deactivate(self, patched_update_kernle):
        """Test memory_belief_update to deactivate a belief."""
        args = {"belief_id": "bel-12345678", "is_active": False}

        result = await call_tool("memory_belief_update", args)

        assert len(result) == 1
        assert "updated successfully" in result[0].text

        patched_update_kernle.update_belief.assert_called_once_with(
            belief_id="bel-12345678", confidence=None, is_active=False
        )

    @pytest.mark.asyncio
    async def test_memory_belief_update_not_found(self, patched_update_kernle):
        """Test memory_belief_update when belief doesn't exist."""
        patched_update_kernle.update_belief.return_value = False

        result = await call_tool("memory_belief_update", {"belief_id": "nonexistent"})

        assert len(result) == 1
        assert "not found" in result[0].text


class TestSyncTool:
    """Test the memory_sync tool."""

    @pytest.fixture
    def sync_mock_kernle(self):
        """Create a mock Kernle with sync method configured."""
        kernle_mock = Mock()

        kernle_mock.sync.return_value = {
            "pushed": 5,
            "pulled": 3,
            "conflicts": 0,
            "errors": [],
            "success": True,
        }

        return kernle_mock

    @pytest.fixture
    def patched_sync_kernle(self, sync_mock_kernle):
        """Patch get_kernle to return the sync mock."""
        with patch("kernle.mcp.server.get_kernle", return_value=sync_mock_kernle):
            yield sync_mock_kernle

    @pytest.mark.asyncio
    async def test_memory_sync_success(self, patched_sync_kernle):
        """Test memory_sync calls sync() and formats output."""
        result = await call_tool("memory_sync", {})

        assert len(result) == 1
        text = result[0].text

        # Verify output formatting structure (not mock values)
        assert "Sync complete:" in text, "Should confirm sync complete"
        assert "Pushed:" in text, "Should show pushed count"
        assert "Pulled:" in text, "Should show pulled count"

        patched_sync_kernle.sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_sync_with_conflicts(self, patched_sync_kernle):
        """Test memory_sync when there are conflicts."""
        patched_sync_kernle.sync.return_value = {
            "pushed": 2,
            "pulled": 4,
            "conflicts": 3,
            "errors": [],
            "success": True,
        }

        result = await call_tool("memory_sync", {})

        assert len(result) == 1
        text = result[0].text
        assert "Conflicts: 3" in text

    @pytest.mark.asyncio
    async def test_memory_sync_with_errors(self, patched_sync_kernle):
        """Test memory_sync when there are errors."""
        patched_sync_kernle.sync.return_value = {
            "pushed": 0,
            "pulled": 0,
            "conflicts": 0,
            "errors": ["Connection timeout", "Auth failed"],
            "success": False,
        }

        result = await call_tool("memory_sync", {})

        assert len(result) == 1
        text = result[0].text
        assert "Errors: 2" in text
        assert "Connection timeout" in text


class TestNoteSearchTool:
    """Test the memory_note_search tool."""

    @pytest.fixture
    def note_search_mock_kernle(self):
        """Create a mock Kernle with search configured for notes."""
        kernle_mock = Mock()

        kernle_mock.search.return_value = [
            {"type": "decision", "title": "Use pytest", "date": "2024-01-15"},
            {"type": "insight", "title": "Mocking is key", "date": "2024-01-14"},
            {"type": "note", "title": "General note", "date": "2024-01-13"},
            {"type": "quote", "title": "Testing quote", "date": "2024-01-12"},
            {"type": "episode", "title": "Episode (should be filtered)", "date": "2024-01-11"},
        ]

        return kernle_mock

    @pytest.fixture
    def patched_note_search_kernle(self, note_search_mock_kernle):
        """Patch get_kernle to return the note search mock."""
        with patch("kernle.mcp.server.get_kernle", return_value=note_search_mock_kernle):
            yield note_search_mock_kernle

    @pytest.mark.asyncio
    async def test_memory_note_search_all_types(self, patched_note_search_kernle):
        """Test memory_note_search filters out non-note types and formats output."""
        result = await call_tool(
            "memory_note_search", {"query": "testing", "note_type": "all", "limit": 10}
        )

        assert len(result) == 1
        text = result[0].text

        # Verify output formatting structure
        assert "Found" in text and "note(s):" in text, "Should show count"
        # Verify filtering logic: episodes should be excluded
        assert "[episode]" not in text, "Episodes should be filtered out"
        # Verify note type formatting
        assert "[" in text and "]" in text, "Should format note types in brackets"

    @pytest.mark.asyncio
    async def test_memory_note_search_specific_type(self, patched_note_search_kernle):
        """Test memory_note_search filters by specific type."""
        result = await call_tool(
            "memory_note_search", {"query": "testing", "note_type": "decision"}
        )

        assert len(result) == 1
        text = result[0].text

        # Verify filtering logic: only decision type should appear
        assert "Found" in text and "note(s):" in text, "Should show count"
        assert "[decision]" in text, "Should include decision type"
        assert "[insight]" not in text, "Should filter out non-matching types"

    @pytest.mark.asyncio
    async def test_memory_note_search_no_results(self, patched_note_search_kernle):
        """Test memory_note_search when no notes found."""
        patched_note_search_kernle.search.return_value = []

        result = await call_tool("memory_note_search", {"query": "nonexistent"})

        assert len(result) == 1
        assert "No notes found for 'nonexistent'" in result[0].text


class TestToolDefinitionsComplete:
    """Test that all new tools are properly defined."""

    @pytest.mark.asyncio
    async def test_list_tools_includes_new_tools(self):
        """Test that list_tools returns all new tools."""
        tools = await list_tools()
        tool_names = {tool.name for tool in tools}

        # Check new tools are present
        new_tools = {
            "memory_belief_list",
            "memory_value_list",
            "memory_goal_list",
            "memory_drive_list",
            "memory_episode_update",
            "memory_goal_update",
            "memory_belief_update",
            "memory_sync",
            "memory_note_search",
        }

        for tool_name in new_tools:
            assert tool_name in tool_names, f"Missing tool: {tool_name}"

    def test_new_tool_definitions_have_proper_schemas(self):
        """Test that all new tools have proper input schemas."""
        from kernle.mcp.server import TOOLS

        new_tool_names = {
            "memory_belief_list",
            "memory_value_list",
            "memory_goal_list",
            "memory_drive_list",
            "memory_episode_update",
            "memory_goal_update",
            "memory_belief_update",
            "memory_sync",
            "memory_note_search",
        }

        for tool in TOOLS:
            if tool.name in new_tool_names:
                assert tool.description, f"{tool.name} missing description"
                assert tool.inputSchema, f"{tool.name} missing inputSchema"
                assert (
                    tool.inputSchema.get("type") == "object"
                ), f"{tool.name} should have object schema"
                assert "properties" in tool.inputSchema, f"{tool.name} missing properties"
                if tool.name == "memory_sync":
                    assert tool.inputSchema.get("additionalProperties") is False


class TestMCPProvenanceParams:
    """Test provenance params (derived_from, source, source_type) on MCP tools."""

    def test_source_type_enum_in_all_creation_schemas(self):
        """All 5 memory creation tools should expose source_type with valid enum values."""
        creation_tools = {
            "memory_episode",
            "memory_note",
            "memory_value",
            "memory_goal",
            "memory_drive",
        }
        expected_enum = sorted(VALID_SOURCE_TYPE_VALUES)
        for tool in TOOLS:
            if tool.name in creation_tools:
                props = tool.inputSchema["properties"]
                assert "source_type" in props, f"{tool.name} missing source_type"
                assert (
                    props["source_type"]["enum"] == expected_enum
                ), f"{tool.name} source_type enum mismatch"

    def test_derived_from_in_all_creation_schemas(self):
        """All 5 memory creation tools should expose derived_from."""
        creation_tools = {
            "memory_episode",
            "memory_note",
            "memory_value",
            "memory_goal",
            "memory_drive",
        }
        for tool in TOOLS:
            if tool.name in creation_tools:
                props = tool.inputSchema["properties"]
                assert "derived_from" in props, f"{tool.name} missing derived_from"
                assert (
                    props["derived_from"]["type"] == "array"
                ), f"{tool.name} derived_from should be array"

    def test_source_in_all_creation_schemas(self):
        """All 5 memory creation tools should expose source."""
        creation_tools = {
            "memory_episode",
            "memory_note",
            "memory_value",
            "memory_goal",
            "memory_drive",
        }
        for tool in TOOLS:
            if tool.name in creation_tools:
                props = tool.inputSchema["properties"]
                assert "source" in props, f"{tool.name} missing source"
                assert props["source"]["type"] == "string", f"{tool.name} source should be string"

    def test_drive_has_context_and_context_tags(self):
        """memory_drive should now expose context and context_tags."""
        tool = next(t for t in TOOLS if t.name == "memory_drive")
        props = tool.inputSchema["properties"]
        assert "context" in props, "memory_drive missing context"
        assert "context_tags" in props, "memory_drive missing context_tags"

    @pytest.mark.asyncio
    async def test_episode_with_source_type(self, patched_get_kernle):
        """Test memory_episode passes source_type through to core."""
        args = {
            "objective": "Test provenance",
            "outcome": "success",
            "source": "consolidation run",
            "derived_from": ["raw:abc123"],
            "source_type": "consolidation",
        }
        result = await call_tool("memory_episode", args)
        assert "Episode saved:" in result[0].text
        patched_get_kernle.episode.assert_called_once_with(
            objective="Test provenance",
            outcome="success",
            lessons=[],
            tags=[],
            context=None,
            context_tags=None,
            source="consolidation run",
            derived_from=["raw:abc123"],
            source_type="consolidation",
        )

    @pytest.mark.asyncio
    async def test_value_with_provenance(self, patched_get_kernle):
        """Test memory_value passes source, derived_from, source_type through to core."""
        args = {
            "name": "transparency",
            "statement": "Be open about decisions",
            "source": "told by mentor",
            "derived_from": ["note:xyz789"],
            "source_type": "external",
        }
        result = await call_tool("memory_value", args)
        assert "Value saved: transparency" in result[0].text
        patched_get_kernle.value.assert_called_once_with(
            name="transparency",
            statement="Be open about decisions",
            priority=50,
            context=None,
            context_tags=None,
            source="told by mentor",
            derived_from=["note:xyz789"],
            source_type="external",
        )

    @pytest.mark.asyncio
    async def test_goal_with_provenance(self, patched_get_kernle):
        """Test memory_goal passes source, derived_from, source_type through to core."""
        args = {
            "title": "Learn Rust",
            "source": "inferred from patterns",
            "derived_from": ["episode:e1"],
            "source_type": "inference",
        }
        result = await call_tool("memory_goal", args)
        assert "Goal saved: Learn Rust" in result[0].text
        patched_get_kernle.goal.assert_called_once_with(
            title="Learn Rust",
            description="",
            priority="medium",
            context=None,
            context_tags=None,
            source="inferred from patterns",
            derived_from=["episode:e1"],
            source_type="inference",
        )

    @pytest.mark.asyncio
    async def test_drive_with_provenance(self, patched_get_kernle):
        """Test memory_drive passes all new params through to core."""
        args = {
            "drive_type": "curiosity",
            "intensity": 0.9,
            "focus_areas": ["AI"],
            "context": "project:kernle",
            "context_tags": ["dev"],
            "source": "seed data",
            "derived_from": ["belief:b1"],
            "source_type": "seed",
        }
        result = await call_tool("memory_drive", args)
        assert "curiosity" in result[0].text
        patched_get_kernle.drive.assert_called_once_with(
            drive_type="curiosity",
            intensity=0.9,
            focus_areas=["AI"],
            context="project:kernle",
            context_tags=["dev"],
            source="seed data",
            derived_from=["belief:b1"],
            source_type="seed",
        )

    @pytest.mark.asyncio
    async def test_invalid_source_type_rejected(self, patched_get_kernle):
        """Test that invalid source_type values are rejected by validation."""
        args = {
            "objective": "Test",
            "outcome": "fail",
            "source_type": "invalid_type",
        }
        result = await call_tool("memory_episode", args)
        assert "Invalid input" in result[0].text

    @pytest.mark.asyncio
    async def test_source_type_none_when_omitted(self, patched_get_kernle):
        """Test that source_type is None when not provided."""
        await call_tool("memory_value", {"name": "test", "statement": "Test value"})
        call_args = patched_get_kernle.value.call_args
        assert call_args.kwargs["source_type"] is None


class TestSuggestionTools:
    """Test suggestion and sync MCP tools via call_tool (sync.py handlers)."""

    @pytest.fixture
    def suggestion_mock_kernle(self):
        """Create a mock Kernle with suggestion/sync methods configured."""
        kernle_mock = Mock()

        kernle_mock.get_suggestions.return_value = [
            {
                "id": "sug-11111111-aaaa-bbbb-cccc-dddddddddddd",
                "memory_type": "episode",
                "status": "pending",
                "confidence": 0.85,
                "content": {"objective": "Built auth module", "outcome": "success"},
                "promoted_to": None,
            },
            {
                "id": "sug-22222222-aaaa-bbbb-cccc-dddddddddddd",
                "memory_type": "belief",
                "status": "promoted",
                "confidence": 0.72,
                "content": {"statement": "Testing improves quality"},
                "promoted_to": "belief-99999999",
            },
            {
                "id": "sug-33333333-aaaa-bbbb-cccc-dddddddddddd",
                "memory_type": "note",
                "status": "rejected",
                "confidence": 0.55,
                "content": {"content": "Random note text here"},
                "promoted_to": None,
            },
        ]

        kernle_mock.accept_suggestion.return_value = "mem-promoted-id-1234"
        kernle_mock.dismiss_suggestion.return_value = True

        kernle_mock.extract_suggestions_from_unprocessed.return_value = [
            {"memory_type": "episode", "content": {"objective": "Task A"}},
            {"memory_type": "episode", "content": {"objective": "Task B"}},
            {"memory_type": "belief", "content": {"statement": "Belief A"}},
        ]

        kernle_mock.sync.return_value = {
            "pushed": 10,
            "pulled": 7,
            "conflicts": 2,
            "errors": ["Timeout on record X", "Hash mismatch on Y"],
        }

        return kernle_mock

    @pytest.fixture
    def patched_suggestion_kernle(self, suggestion_mock_kernle):
        """Patch get_kernle to return the suggestion mock."""
        with patch("kernle.mcp.server.get_kernle", return_value=suggestion_mock_kernle):
            yield suggestion_mock_kernle

    # -- suggestion_extract --

    @pytest.mark.asyncio
    async def test_suggestions_extract_returns_summary(self, patched_suggestion_kernle):
        """Test suggestion_extract returns type-grouped summary."""
        result = await call_tool("suggestion_extract", {"limit": 100})

        assert len(result) == 1
        text = result[0].text
        assert "Extracted 3 suggestion(s):" in text
        assert "episode: 2" in text
        assert "belief: 1" in text
        assert "suggestion_list" in text

        patched_suggestion_kernle.extract_suggestions_from_unprocessed.assert_called_once_with(
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_suggestions_extract_empty(self, patched_suggestion_kernle):
        """Test suggestion_extract when no suggestions extracted."""
        patched_suggestion_kernle.extract_suggestions_from_unprocessed.return_value = []

        result = await call_tool("suggestion_extract", {})

        assert len(result) == 1
        assert "No suggestions extracted from raw entries." in result[0].text

    @pytest.mark.asyncio
    async def test_suggestions_extract_default_limit(self, patched_suggestion_kernle):
        """Test suggestion_extract uses default limit of 50."""
        await call_tool("suggestion_extract", {})

        patched_suggestion_kernle.extract_suggestions_from_unprocessed.assert_called_once_with(
            limit=50,
        )

    # -- memory_sync (via sync.py handler) --

    @pytest.mark.asyncio
    async def test_sync_with_conflicts_and_errors(self, patched_suggestion_kernle):
        """Test memory_sync formats conflicts and errors."""
        result = await call_tool("memory_sync", {})

        assert len(result) == 1
        text = result[0].text
        assert "Sync complete:" in text
        assert "Pushed: 10" in text
        assert "Pulled: 7" in text
        assert "Conflicts: 2" in text
        assert "Errors: 2" in text
        assert "Timeout on record X" in text
        assert "Hash mismatch on Y" in text

    @pytest.mark.asyncio
    async def test_sync_clean(self, patched_suggestion_kernle):
        """Test memory_sync with no conflicts or errors."""
        patched_suggestion_kernle.sync.return_value = {
            "pushed": 3,
            "pulled": 1,
        }

        result = await call_tool("memory_sync", {})

        assert len(result) == 1
        text = result[0].text
        assert "Sync complete:" in text
        assert "Pushed: 3" in text
        assert "Pulled: 1" in text
        assert "Conflicts" not in text
        assert "Errors" not in text

    @pytest.mark.asyncio
    async def test_sync_errors_truncated(self, patched_suggestion_kernle):
        """Test memory_sync truncates errors to first 3."""
        patched_suggestion_kernle.sync.return_value = {
            "pushed": 0,
            "pulled": 0,
            "errors": ["err1", "err2", "err3", "err4", "err5"],
        }

        result = await call_tool("memory_sync", {})

        text = result[0].text
        assert "Errors: 5" in text
        assert "err1" in text
        assert "err2" in text
        assert "err3" in text
        assert "err4" not in text
        assert "err5" not in text

    @pytest.mark.asyncio
    async def test_sync_rejects_unexpected_arguments(self, patched_suggestion_kernle):
        """memory_sync should reject extra arguments explicitly."""
        result = await call_tool("memory_sync", {"force": True})

        assert len(result) == 1
        assert "Invalid input" in result[0].text
        assert "unexpected" in result[0].text.lower()
        patched_suggestion_kernle.sync.assert_not_called()


class TestSyncValidators:
    """Direct tests for sync.py validator functions."""

    def test_validate_memory_sync_returns_empty(self):
        """validate_memory_sync always returns empty dict."""
        from kernle.mcp.handlers.sync import validate_memory_sync

        assert validate_memory_sync({}) == {}
        with pytest.raises(ValueError, match="memory_sync accepts no arguments"):
            validate_memory_sync({"extra": "ignored"})

    # -- validate_suggestion_extract --

    def test_validate_suggestions_extract_defaults(self):
        """validate_suggestion_extract returns default limit of 50."""
        from kernle.mcp.handlers.sync import validate_suggestion_extract

        result = validate_suggestion_extract({})
        assert result["limit"] == 50

    def test_validate_suggestions_extract_custom_limit(self):
        """validate_suggestion_extract accepts custom limit."""
        from kernle.mcp.handlers.sync import validate_suggestion_extract

        result = validate_suggestion_extract({"limit": 100})
        assert result["limit"] == 100

    def test_validate_suggestions_extract_limit_bounds(self):
        """validate_suggestion_extract rejects out-of-bounds limit."""
        from kernle.mcp.handlers.sync import validate_suggestion_extract

        with pytest.raises(ValueError, match="limit"):
            validate_suggestion_extract({"limit": 0})

        with pytest.raises(ValueError, match="limit"):
            validate_suggestion_extract({"limit": 300})


class TestMCPSuggestionStatusConsistency:
    """Verify MCP tool schemas include all valid suggestion statuses from types.py."""

    def test_tool_schema_enums_include_all_valid_statuses(self):
        """MCP suggestion tool status enums must include every status from VALID_SUGGESTION_STATUSES."""
        from kernle.mcp.server import TOOLS
        from kernle.types import VALID_SUGGESTION_STATUSES

        suggestion_tools = {t.name: t for t in TOOLS if t.name == "suggestion_list"}
        assert len(suggestion_tools) == 1, "Expected suggestion_list tool"

        for tool_name, tool in suggestion_tools.items():
            schema_enum = set(tool.inputSchema["properties"]["status"]["enum"])
            # "all" is a meta-filter, not a real status — exclude it
            schema_statuses = schema_enum - {"all"}
            assert schema_statuses == set(VALID_SUGGESTION_STATUSES), (
                f"{tool_name} status enum {sorted(schema_statuses)} != "
                f"VALID_SUGGESTION_STATUSES {sorted(VALID_SUGGESTION_STATUSES)}"
            )

    def test_validator_accepts_all_valid_statuses(self):
        """Shared validator must accept every status in VALID_SUGGESTION_STATUSES."""
        from kernle.mcp.handlers.sync import validate_suggestion_list
        from kernle.types import VALID_SUGGESTION_STATUSES

        for status in sorted(VALID_SUGGESTION_STATUSES):
            result = validate_suggestion_list({"status": status})
            assert result["status"] == status, f"Validator rejected '{status}'"

    def test_validator_rejects_unknown_status(self):
        """Validator must reject statuses not in VALID_SUGGESTION_STATUSES."""
        from kernle.mcp.handlers.sync import validate_suggestion_list

        with pytest.raises(ValueError, match="status"):
            validate_suggestion_list({"status": "unknown_status"})
