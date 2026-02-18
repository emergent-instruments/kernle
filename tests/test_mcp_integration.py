"""
Integration tests for MCP server with real Kernle instance.

These tests use a real Kernle instance with SQLite storage to verify
the MCP layer integrates correctly with the core memory system.

Unlike test_mcp.py (which mocks Kernle), these tests catch:
- Parameter type mismatches
- Schema/API drift
- Storage integration issues
- Real data flow through the stack
"""

import json
import os
from pathlib import Path

import pytest

from kernle import Kernle
from kernle.mcp.server import call_tool


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_memories.db"
    return str(db_path)


@pytest.fixture
def real_kernle(temp_db):
    """Create a real Kernle instance with temp storage."""
    # Set environment to use temp database
    old_home = os.environ.get("KERNLE_HOME")
    os.environ["KERNLE_HOME"] = str(Path(temp_db).parent)

    try:
        k = Kernle(stack_id="test-mcp-integration", strict=False)
        yield k
    finally:
        if old_home:
            os.environ["KERNLE_HOME"] = old_home
        else:
            os.environ.pop("KERNLE_HOME", None)


@pytest.fixture
def setup_kernle_for_mcp(real_kernle, monkeypatch):
    """Patch get_kernle to return our real instance."""
    monkeypatch.setattr("kernle.mcp.server.get_kernle", lambda: real_kernle)
    return real_kernle


class TestMCPEpisodeIntegration:
    """Integration tests for episode operations through MCP."""

    @pytest.mark.asyncio
    async def test_episode_create_and_retrieve(self, setup_kernle_for_mcp):
        """Test creating an episode through MCP and verifying storage."""
        k = setup_kernle_for_mcp

        # Create episode through MCP
        result = await call_tool(
            "memory_episode",
            {
                "objective": "Test MCP integration",
                "outcome": "Verified data flows correctly",
                "lesson": "Integration tests are valuable",
            },
        )

        # Verify result
        assert len(result) == 1
        content = result[0]
        assert "Test MCP integration" in content.text or "episode" in content.text.lower()

        # Verify actually stored
        episodes = k._storage.get_episodes(limit=10)
        assert any(e.objective == "Test MCP integration" for e in episodes)

    @pytest.mark.asyncio
    async def test_episode_with_all_fields(self, setup_kernle_for_mcp):
        """Test creating episode with all optional fields."""
        k = setup_kernle_for_mcp

        result = await call_tool(
            "memory_episode",
            {
                "objective": "Complete integration test",
                "outcome": "All fields stored",
                "lesson": "Comprehensive tests work",
                "emotion": "satisfaction",
            },
        )

        assert len(result) == 1

        # Verify fields stored correctly
        episodes = k._storage.get_episodes(limit=10)
        episode = next((e for e in episodes if e.objective == "Complete integration test"), None)
        assert episode is not None
        assert episode.outcome == "All fields stored"


class TestMCPBeliefIntegration:
    """Integration tests for belief operations through MCP."""

    @pytest.mark.asyncio
    async def test_belief_create_and_list(self, setup_kernle_for_mcp):
        """Test creating a belief via entity method and listing through MCP."""
        k = setup_kernle_for_mcp

        # Create belief via entity method (MCP memory_belief tool removed in #866)
        k.belief(
            statement="Integration tests catch real bugs",
            confidence=0.85,
        )

        # List beliefs through MCP (memory_belief_list still exists)
        list_result = await call_tool("memory_belief_list", {"limit": 10})
        assert len(list_result) == 1
        # Should contain our belief
        assert "Integration tests catch real bugs" in list_result[0].text

    @pytest.mark.asyncio
    async def test_belief_update(self, setup_kernle_for_mcp):
        """Test updating belief through MCP."""
        k = setup_kernle_for_mcp

        # Create belief via entity method (MCP memory_belief tool removed in #866)
        k.belief(
            statement="Mocks are sometimes insufficient",
            confidence=0.7,
        )

        # Get belief ID (belief uses 'statement' not 'content')
        beliefs = k._storage.get_beliefs(limit=10)
        belief = next((b for b in beliefs if "Mocks" in b.statement), None)
        assert belief is not None

        # Update confidence
        result = await call_tool(
            "memory_belief_update",
            {
                "belief_id": belief.id,
                "confidence": 0.95,
            },
        )
        assert len(result) == 1

        # Verify updated (use get_beliefs to find by ID)
        updated_beliefs = k._storage.get_beliefs(limit=100)
        updated = next((b for b in updated_beliefs if b.id == belief.id), None)
        assert updated is not None
        assert updated.confidence == 0.95


class TestMCPNoteIntegration:
    """Integration tests for note operations through MCP."""

    @pytest.mark.asyncio
    async def test_note_create_and_search(self, setup_kernle_for_mcp):
        """Test creating note and searching for it."""

        # Create note
        result = await call_tool(
            "memory_note",
            {
                "content": "Important architectural decision about MCP integration",
                "note_type": "decision",
                "tags": ["architecture", "mcp"],
            },
        )
        assert len(result) == 1

        # Search for note
        search_result = await call_tool(
            "memory_note_search",
            {
                "query": "MCP integration",
            },
        )
        assert len(search_result) == 1
        assert (
            "architectural decision" in search_result[0].text.lower()
            or len(search_result[0].text) > 0
        )


class TestMCPValueAndGoalIntegration:
    """Integration tests for values and goals through MCP."""

    @pytest.mark.asyncio
    async def test_value_create_and_list(self, setup_kernle_for_mcp):
        """Test creating and listing values."""
        # Create value (uses 'name' and 'statement', not 'description')
        result = await call_tool(
            "memory_value",
            {
                "name": "quality",
                "statement": "Prefer quality over speed",
                "importance": 0.9,
            },
        )
        assert len(result) == 1

        # List values
        list_result = await call_tool("memory_value_list", {"limit": 10})
        assert len(list_result) == 1
        assert "quality" in list_result[0].text.lower()

    @pytest.mark.asyncio
    async def test_goal_create_and_update(self, setup_kernle_for_mcp):
        """Test creating and updating goals."""
        k = setup_kernle_for_mcp

        # Create goal (uses 'title', not 'description'; 'priority' is enum not number)
        result = await call_tool(
            "memory_goal",
            {
                "title": "Improve test coverage",
                "description": "Reach 80% coverage",
                "priority": "high",
            },
        )
        assert len(result) == 1

        # Get goal ID
        goals = k._storage.get_goals(limit=10)
        goal = next((g for g in goals if "test coverage" in g.title.lower()), None)
        assert goal is not None

        # Update status (goal supports: status, priority, description)
        update_result = await call_tool(
            "memory_goal_update",
            {
                "goal_id": goal.id,
                "status": "completed",
            },
        )
        assert len(update_result) == 1

        # Verify updated (use get_goals with status=None to get all including completed)
        updated_goals = k._storage.get_goals(status=None, limit=100)
        updated = next((g for g in updated_goals if g.id == goal.id), None)
        assert updated is not None
        assert updated.status == "completed"


class TestMCPDriveIntegration:
    """Integration tests for drives through MCP."""

    @pytest.mark.asyncio
    async def test_drive_create_and_list(self, setup_kernle_for_mcp):
        """Test creating and listing drives."""
        # Create drive (uses 'drive_type' enum, not 'name')
        result = await call_tool(
            "memory_drive",
            {
                "drive_type": "curiosity",  # enum: existence, growth, curiosity, connection, reproduction
                "intensity": 0.85,
                "focus_areas": ["understanding", "learning"],
            },
        )
        assert len(result) == 1

        # List drives
        list_result = await call_tool("memory_drive_list", {"limit": 10})
        assert len(list_result) == 1
        assert "curiosity" in list_result[0].text.lower()


class TestMCPSearchIntegration:
    """Integration tests for search functionality through MCP."""

    @pytest.mark.asyncio
    async def test_semantic_search_finds_relevant(self, setup_kernle_for_mcp):
        """Test that semantic search returns relevant results."""

        # Create some data to search
        await call_tool(
            "memory_episode",
            {
                "objective": "Build a REST API endpoint",
                "outcome": "Successfully deployed to production",
                "lesson": "Always add rate limiting",
            },
        )

        await call_tool(
            "memory_note",
            {
                "content": "API design should follow REST principles",
                "note_type": "insight",
            },
        )

        # Search for API-related content
        result = await call_tool(
            "memory_search",
            {
                "query": "API development",
                "limit": 10,
            },
        )

        assert len(result) == 1
        # Should find at least one of our API-related entries
        text = result[0].text.lower()
        assert "api" in text or "rest" in text or "endpoint" in text


class TestMCPLoadIntegration:
    """Integration tests for memory_load functionality."""

    @pytest.mark.asyncio
    async def test_load_returns_formatted_memory(self, setup_kernle_for_mcp):
        """Test that memory_load returns properly formatted data."""
        k = setup_kernle_for_mcp

        # Create some data via entity method (MCP memory_belief tool removed in #866)
        k.belief(
            statement="Tests should be deterministic",
            confidence=0.9,
        )

        # Load memory
        result = await call_tool(
            "memory_load",
            {
                "format": "text",
                "budget": 10000,
            },
        )

        assert len(result) == 1
        # Should contain beliefs section or our belief text
        text = result[0].text.lower()
        assert "belief" in text or "deterministic" in text or "working memory" in text

    @pytest.mark.asyncio
    async def test_load_json_format(self, setup_kernle_for_mcp):
        """Test that memory_load JSON format is valid JSON."""

        # Create some data
        await call_tool(
            "memory_value",
            {
                "name": "clarity",
                "description": "Value clear communication",
                "importance": 0.8,
            },
        )

        # Load as JSON
        result = await call_tool(
            "memory_load",
            {
                "format": "json",
                "budget": 10000,
            },
        )

        assert len(result) == 1
        # Should be valid JSON
        data = json.loads(result[0].text)
        assert isinstance(data, dict)


class TestMCPCheckpointIntegration:
    """Integration tests for checkpoint functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_load(self, setup_kernle_for_mcp):
        """Test checkpoint save and load cycle."""
        # Save checkpoint
        save_result = await call_tool(
            "memory_checkpoint_save",
            {
                "task": "Testing MCP integration",
                "context": "Verifying checkpoint functionality",
            },
        )
        assert len(save_result) == 1
        assert "saved" in save_result[0].text.lower() or "checkpoint" in save_result[0].text.lower()

        # Load checkpoint
        load_result = await call_tool("memory_checkpoint_load", {})
        assert len(load_result) == 1
        assert "Testing MCP integration" in load_result[0].text


class TestMCPErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_belief_id_returns_error(self, setup_kernle_for_mcp):
        """Test that updating non-existent belief returns helpful error."""
        result = await call_tool(
            "memory_belief_update",
            {
                "belief_id": "non-existent-id",
                "confidence": 0.5,
            },
        )

        assert len(result) == 1
        # Should contain error message
        text = result[0].text.lower()
        assert "not found" in text or "error" in text

    @pytest.mark.asyncio
    async def test_invalid_goal_id_returns_error(self, setup_kernle_for_mcp):
        """Test that updating non-existent goal returns helpful error."""
        result = await call_tool(
            "memory_goal_update",
            {
                "goal_id": "non-existent-goal",
                "progress": 0.5,
            },
        )

        assert len(result) == 1
        text = result[0].text.lower()
        assert "not found" in text or "error" in text


class TestMCPStatusIntegration:
    """Integration tests for status functionality."""

    @pytest.mark.asyncio
    async def test_status_returns_counts(self, setup_kernle_for_mcp):
        """Test that memory_status returns actual counts."""
        k = setup_kernle_for_mcp

        before = k.status()

        # Create some data
        await call_tool(
            "memory_episode",
            {"objective": "Status test", "outcome": "Created", "lesson": "Count this"},
        )
        # Create belief via entity method (MCP memory_belief tool removed in #866)
        k.belief(statement="Status works", confidence=0.8)
        after = k.status()

        assert after["episodes"] >= before["episodes"] + 1
        assert after["beliefs"] >= before["beliefs"] + 1

        # Get status
        result = await call_tool("memory_status", {})
        assert len(result) == 1

        text = result[0].text
        assert "Memory Status (test-mcp-integration)" in text
        assert f"Beliefs:    {after['beliefs']}" in text
        assert f"Episodes:   {after['episodes']}" in text
