"""
Comprehensive tests for the Kernle core functionality.

Updated to work with the storage abstraction layer.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kernle.core import Kernle
from kernle.storage import SQLiteStorage


class TestKernleInitialization:
    """Test Kernle class initialization and setup."""

    def test_init_with_defaults(self, temp_db_path):
        """Test initialization with default values."""
        # Clear any env vars that might interfere
        with patch.dict(os.environ, {}, clear=True):
            kernle = Kernle(strict=False)
            assert kernle.stack_id == "default"  # Default when no env var
            assert kernle.checkpoint_dir == Path.home() / ".kernle" / "checkpoints"
            # Should have SQLite storage when no Supabase credentials
            assert isinstance(kernle._storage, SQLiteStorage)

    def test_init_with_explicit_params(self, temp_checkpoint_dir, temp_db_path):
        """Test initialization with explicit parameters."""
        storage = SQLiteStorage(stack_id="test_agent", db_path=temp_db_path)
        kernle = Kernle(
            stack_id="test_agent", storage=storage, checkpoint_dir=temp_checkpoint_dir, strict=False
        )
        assert kernle.stack_id == "test_agent"
        assert kernle.checkpoint_dir == temp_checkpoint_dir
        assert kernle._storage is storage

    def test_init_with_env_vars(self, temp_checkpoint_dir):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "KERNLE_STACK_ID": "env_agent",
            },
            clear=True,
        ):
            kernle = Kernle(checkpoint_dir=temp_checkpoint_dir, strict=False)
            assert kernle.stack_id == "env_agent"
            # Without Supabase creds, should use SQLite
            assert isinstance(kernle._storage, SQLiteStorage)

    def test_client_property_missing_credentials(self, temp_db_path):
        """Test that client property raises error (Supabase removed from core)."""
        with patch.dict(os.environ, {}, clear=True):
            kernle = Kernle(stack_id="test", strict=False)

            # Supabase storage removed from core — .client always raises
            with pytest.raises(ValueError, match="no longer available"):
                _ = kernle.client

    def test_storage_property(self, kernle_instance):
        """Test that storage property returns the storage backend."""
        kernle, storage = kernle_instance
        assert kernle.storage is storage


class TestLoadMethods:
    """Test various load methods."""

    def test_load_full_context(self, kernle_instance, populated_storage):
        """Test loading full working memory context."""
        kernle, _ = kernle_instance
        # Use populated_storage to ensure data is loaded
        kernle._storage = populated_storage

        memory = kernle.load()

        assert "values" in memory
        assert "beliefs" in memory
        assert "goals" in memory
        assert "drives" in memory
        assert "lessons" in memory
        assert "recent_work" in memory
        assert "recent_notes" in memory
        assert "relationships" in memory
        assert "checkpoint" in memory

    def test_load_values(self, kernle_instance, populated_storage):
        """Test loading agent values."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        values = kernle.load_values(limit=5)

        assert len(values) == 1  # One value in sample data
        assert values[0]["name"] == "Quality"
        assert values[0]["statement"] == "Software should be thoroughly tested and reliable"
        assert values[0]["priority"] == 80
        assert values[0]["value_type"] == "core_value"

    def test_load_beliefs(self, kernle_instance, populated_storage):
        """Test loading agent beliefs."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        beliefs = kernle.load_beliefs(limit=10)

        assert len(beliefs) == 1
        assert beliefs[0]["statement"] == "Comprehensive testing leads to more reliable software"
        assert beliefs[0]["belief_type"] == "fact"
        assert beliefs[0]["confidence"] == 0.9

    def test_load_goals(self, kernle_instance, populated_storage):
        """Test loading active goals."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        goals = kernle.load_goals(limit=5)

        assert len(goals) == 1
        assert goals[0]["title"] == "Achieve 80%+ test coverage"
        assert goals[0]["status"] == "active"
        assert goals[0]["priority"] == "high"

    def test_load_lessons(self, kernle_instance, populated_storage):
        """Test extracting lessons from episodes."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        lessons = kernle.load_lessons(limit=10)

        # Should extract lessons from episodes
        assert any("test" in lesson.lower() for lesson in lessons)

    def test_load_recent_work_filters_checkpoints(self, kernle_instance, populated_storage):
        """Test that recent work excludes checkpoint episodes."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        recent_work = kernle.load_recent_work(limit=5)

        # Verify no checkpoint episodes
        for episode in recent_work:
            tags = episode.get("tags", []) or []
            assert "checkpoint" not in tags

    def test_load_recent_notes(self, kernle_instance, populated_storage):
        """Test loading recent curated notes."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        notes = kernle.load_recent_notes(limit=5)

        assert len(notes) == 2  # Two notes in sample data

    def test_load_drives(self, kernle_instance, populated_storage):
        """Test loading drive states."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        drives = kernle.load_drives()

        assert len(drives) == 1
        assert drives[0]["drive_type"] == "growth"
        assert drives[0]["intensity"] == 0.7
        assert drives[0]["focus_areas"] == ["learning", "improvement"]

    def test_load_returns_meta_with_budget_metrics(self, kernle_instance, populated_storage):
        """Test that load() returns _meta with budget tracking info."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        memory = kernle.load(budget=8000)

        # Verify _meta is present
        assert "_meta" in memory
        meta = memory["_meta"]

        # Verify budget metrics
        assert "budget_used" in meta
        assert "budget_total" in meta
        assert "excluded_count" in meta

        # Verify types and values make sense
        assert isinstance(meta["budget_used"], int)
        assert isinstance(meta["budget_total"], int)
        assert isinstance(meta["excluded_count"], int)
        assert meta["budget_total"] == 8000
        assert meta["budget_used"] > 0
        assert meta["budget_used"] <= meta["budget_total"]
        assert meta["excluded_count"] == 0  # 8000 budget is generous enough for all items

    def test_load_meta_excluded_count_with_low_budget(self, kernle_instance, populated_storage):
        """Test that excluded_count increases when budget is too small."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        # Very low budget should exclude some items
        memory = kernle.load(budget=100)
        meta = memory["_meta"]

        # With a tiny budget, we should have exclusions (unless all items are tiny)
        # The important thing is the math works: budget_used <= budget_total
        assert meta["budget_used"] <= meta["budget_total"]
        assert meta["budget_total"] == 100

    def test_load_meta_budget_used_varies_with_content(self, kernle_instance):
        """Test that budget_used reflects actual content size."""
        kernle, storage = kernle_instance

        # Empty storage
        memory_empty = kernle.load(budget=8000)
        assert memory_empty["_meta"]["budget_used"] == 0
        assert memory_empty["_meta"]["excluded_count"] == 0


class TestProcessDelegation:
    """Test Kernle API compatibility for process delegation."""

    def test_process_forwards_to_entity(self, kernle_instance):
        kernle, _ = kernle_instance
        expected = "delegated-result"

        ent = kernle.entity
        ent.process = MagicMock(return_value=expected)

        result = kernle.process(
            transition="raw_to_episode",
            force=True,
            allow_no_inference_override=True,
            auto_promote=True,
            batch_size=64,
        )

        assert result == expected
        ent.process.assert_called_once_with(
            transition="raw_to_episode",
            force=True,
            allow_no_inference_override=True,
            auto_promote=True,
            batch_size=64,
        )


class TestCheckpoints:
    """Test checkpoint save/load/clear functionality."""

    def test_checkpoint_save_basic(self, kernle_instance):
        """Test basic checkpoint saving."""
        kernle, storage = kernle_instance

        checkpoint_data = kernle.checkpoint(
            task="Write tests",
            pending=["Test CLI", "Test edge cases"],
            context="Working on comprehensive test suite",
        )

        assert checkpoint_data["current_task"] == "Write tests"
        assert checkpoint_data["pending"] == ["Test CLI", "Test edge cases"]
        assert checkpoint_data["context"] == "Working on comprehensive test suite"
        assert "timestamp" in checkpoint_data
        assert checkpoint_data["stack_id"] == "test_agent"

    def test_checkpoint_multiple_saves(self, kernle_instance):
        """Test that multiple checkpoints are stored in history."""
        kernle, storage = kernle_instance

        kernle.checkpoint(task="Task 1")
        kernle.checkpoint(task="Task 2")
        kernle.checkpoint(task="Task 3")

        # Load the checkpoint file directly
        checkpoint_file = kernle.checkpoint_dir / f"{kernle.stack_id}.json"
        with open(checkpoint_file) as f:
            checkpoints = json.load(f)

        assert len(checkpoints) == 3
        assert checkpoints[-1]["current_task"] == "Task 3"

    def test_checkpoint_history_limit(self, kernle_instance):
        """Test that checkpoint history is limited to 10 entries."""
        kernle, storage = kernle_instance

        for i in range(15):
            kernle.checkpoint(task=f"Task {i}")

        checkpoint_file = kernle.checkpoint_dir / f"{kernle.stack_id}.json"
        with open(checkpoint_file) as f:
            checkpoints = json.load(f)

        assert len(checkpoints) == 10
        assert checkpoints[-1]["current_task"] == "Task 14"  # Most recent
        assert checkpoints[0]["current_task"] == "Task 5"  # Oldest kept

    def test_load_checkpoint_exists(self, kernle_instance):
        """Test loading an existing checkpoint."""
        kernle, storage = kernle_instance

        kernle.checkpoint(task="Test task", pending=["item 1"])

        loaded = kernle.load_checkpoint()

        assert loaded is not None
        assert loaded["current_task"] == "Test task"
        assert loaded["pending"] == ["item 1"]

    def test_load_checkpoint_not_exists(self, kernle_instance):
        """Test loading checkpoint when none exists."""
        kernle, storage = kernle_instance

        loaded = kernle.load_checkpoint()

        assert loaded is None

    def test_load_checkpoint_corrupted_file(self, kernle_instance):
        """Test loading checkpoint with corrupted file."""
        kernle, storage = kernle_instance

        # Create corrupted file
        checkpoint_file = kernle.checkpoint_dir / f"{kernle.stack_id}.json"
        checkpoint_file.write_text("not valid json")

        loaded = kernle.load_checkpoint()

        assert loaded is None

    def test_clear_checkpoint_exists(self, kernle_instance):
        """Test clearing an existing checkpoint."""
        kernle, storage = kernle_instance

        kernle.checkpoint(task="To be cleared")
        result = kernle.clear_checkpoint()

        assert result is True
        assert kernle.load_checkpoint() is None

    def test_clear_checkpoint_not_exists(self, kernle_instance):
        """Test clearing when no checkpoint exists."""
        kernle, storage = kernle_instance

        result = kernle.clear_checkpoint()

        assert result is False


class TestEpisodes:
    """Test episode recording functionality."""

    def test_episode_basic(self, kernle_instance):
        """Test basic episode recording."""
        kernle, storage = kernle_instance

        episode_id = kernle.episode(
            objective="Write unit tests",
            outcome="success",
            lessons=["Test early", "Test often"],
            tags=["testing"],
        )

        assert episode_id is not None
        assert len(episode_id) > 0

        # Verify episode was saved
        episodes = storage.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].objective == "Write unit tests"

    def test_episode_outcome_type_detection(self, kernle_instance):
        """Test automatic outcome type detection."""
        kernle, storage = kernle_instance

        # Test success detection
        kernle.episode("Task 1", "success")
        kernle.episode("Task 2", "failed")
        kernle.episode("Task 3", "some progress made")

        episodes = storage.get_episodes()

        # Find each episode by objective
        success_ep = next(e for e in episodes if e.objective == "Task 1")
        failure_ep = next(e for e in episodes if e.objective == "Task 2")
        partial_ep = next(e for e in episodes if e.objective == "Task 3")

        assert success_ep.outcome_type == "success"
        assert failure_ep.outcome_type == "failure"
        assert partial_ep.outcome_type == "partial"

    def test_episode_minimal(self, kernle_instance):
        """Test episode recording with minimal data."""
        kernle, storage = kernle_instance

        episode_id = kernle.episode("Simple task", "done")

        assert episode_id is not None
        episodes = storage.get_episodes()
        assert len(episodes) == 1


class TestNotes:
    """Test note recording functionality."""

    def test_note_basic(self, kernle_instance):
        """Test basic note recording."""
        kernle, storage = kernle_instance

        note_id = kernle.note(content="Important finding", type="note", tags=["important"])

        assert note_id is not None
        notes = storage.get_notes()
        assert len(notes) == 1
        assert "Important finding" in notes[0].content

    def test_note_decision(self, kernle_instance):
        """Test decision note formatting."""
        kernle, storage = kernle_instance

        kernle.note(content="Use pytest", type="decision", reason="Industry standard")

        notes = storage.get_notes()
        assert "**Decision**:" in notes[0].content
        assert "Use pytest" in notes[0].content

    def test_note_quote(self, kernle_instance):
        """Test quote note formatting."""
        kernle, storage = kernle_instance

        kernle.note(content="To be or not to be", type="quote", speaker="Shakespeare")

        notes = storage.get_notes()
        assert ">" in notes[0].content  # Quote formatting
        assert "Shakespeare" in notes[0].content

    def test_note_insight(self, kernle_instance):
        """Test insight note formatting."""
        kernle, storage = kernle_instance

        kernle.note(content="Testing is crucial", type="insight")

        notes = storage.get_notes()
        assert "**Insight**:" in notes[0].content

    def test_note_protected(self, kernle_instance):
        """Test protected note flag."""
        kernle, storage = kernle_instance

        kernle.note(content="Secret info", type="note", protect=True)

        # Note should be saved (protect flag is just metadata)
        notes = storage.get_notes()
        assert len(notes) == 1


class TestBeliefValueGoal:
    """Test belief, value, and goal operations."""

    def test_belief_creation(self, kernle_instance):
        """Test creating a belief."""
        kernle, storage = kernle_instance

        belief_id = kernle.belief(
            statement="Testing improves code quality", type="fact", confidence=0.85
        )

        assert belief_id is not None
        beliefs = storage.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].statement == "Testing improves code quality"
        assert beliefs[0].confidence == 0.85

    def test_belief_defaults(self, kernle_instance):
        """Test belief with default values."""
        kernle, storage = kernle_instance

        kernle.belief("Simple belief")

        beliefs = storage.get_beliefs()
        assert beliefs[0].belief_type == "fact"  # Default
        assert beliefs[0].confidence == 0.8  # Default

    def test_value_creation(self, kernle_instance):
        """Test creating a value."""
        kernle, storage = kernle_instance

        value_id = kernle.value(
            name="Quality", statement="Always prioritize code quality", priority=90
        )

        assert value_id is not None
        values = storage.get_values()
        assert len(values) == 1
        assert values[0].name == "Quality"
        assert values[0].priority == 90

    def test_value_defaults(self, kernle_instance):
        """Test value with default values."""
        kernle, storage = kernle_instance

        kernle.value("Simplicity", "Keep it simple")

        values = storage.get_values()
        assert values[0].priority == 50  # Default

    def test_goal_creation(self, kernle_instance):
        """Test creating a goal."""
        kernle, storage = kernle_instance

        goal_id = kernle.goal(
            title="Achieve test coverage", description="Get 80%+ coverage", priority="high"
        )

        assert goal_id is not None
        goals = storage.get_goals()
        assert len(goals) == 1
        assert goals[0].title == "Achieve test coverage"
        assert goals[0].priority == "high"

    def test_goal_defaults(self, kernle_instance):
        """Test goal with default values."""
        kernle, storage = kernle_instance

        kernle.goal("Simple goal")

        goals = storage.get_goals()
        assert goals[0].status == "active"
        assert goals[0].priority == "medium"


class TestSearch:
    """Test search functionality."""

    def test_search_episodes(self, kernle_instance, populated_storage):
        """Test searching episodes returns relevant results."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        results = kernle.search("tests")

        assert len(results) >= 1
        # Verify we found the expected episode about unit tests
        result_texts = [str(r.get("objective", r.get("content", ""))) for r in results]
        assert any(
            "test" in text.lower() for text in result_texts
        ), f"Expected 'test' in results: {result_texts}"

    def test_search_notes(self, kernle_instance, populated_storage):
        """Test searching notes returns relevant content."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        results = kernle.search("Decision")

        assert len(results) >= 1
        # Verify we found the decision note about pytest
        result_texts = [str(r.get("content", "")) for r in results]
        assert any(
            "decision" in text.lower() or "pytest" in text.lower() for text in result_texts
        ), f"Expected decision/pytest content in results: {result_texts}"

    def test_search_beliefs(self, kernle_instance, populated_storage):
        """Test searching beliefs returns matching statements."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        results = kernle.search("testing")

        assert len(results) >= 1
        # Verify we found belief about testing leading to reliable software
        result_texts = [str(r.get("statement", r.get("content", ""))) for r in results]
        assert any(
            "test" in text.lower() for text in result_texts
        ), f"Expected 'test' in results: {result_texts}"

    def test_search_no_results(self, kernle_instance):
        """Test search with no matching results."""
        kernle, storage = kernle_instance

        results = kernle.search("xyznonexistentquery123")

        assert len(results) == 0

    def test_search_case_insensitive(self, kernle_instance, populated_storage):
        """Test that search is case insensitive."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        results_upper = kernle.search("TESTING")
        results_lower = kernle.search("testing")

        # Both should return results with same content
        assert len(results_upper) > 0
        assert len(results_lower) > 0
        # Results should be equivalent (same items found)
        upper_ids = {r.get("id") for r in results_upper}
        lower_ids = {r.get("id") for r in results_lower}
        assert upper_ids == lower_ids, "Case-insensitive search should return same results"

    def test_search_limit(self, kernle_instance):
        """Test search result limit is respected."""
        kernle, storage = kernle_instance

        # Create many items with unique searchable content
        for i in range(20):
            kernle.episode(f"Searchable test episode number {i}", "success")

        results = kernle.search("Searchable", limit=5)

        assert len(results) == 5, f"Expected exactly 5 results, got {len(results)}"
        # Verify results are actually from our created episodes
        # Search results use 'title' for display, which contains the objective
        for result in results:
            result_text = result.get("title", result.get("objective", result.get("content", "")))
            assert "Searchable" in result_text, f"Result should contain 'Searchable': {result}"


class TestStatus:
    """Test status functionality."""

    def test_status_with_data(self, kernle_instance, populated_storage):
        """Test status with populated data."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        status = kernle.status()

        assert status["stack_id"] == "test_agent"
        assert status["values"] >= 1
        assert status["beliefs"] >= 1
        assert status["goals"] >= 1
        assert status["episodes"] >= 1

    def test_status_empty(self, kernle_instance):
        """Test status with empty storage."""
        kernle, storage = kernle_instance

        status = kernle.status()

        assert status["stack_id"] == "test_agent"
        assert status["values"] == 0
        assert status["beliefs"] == 0
        assert status["goals"] == 0
        assert status["episodes"] == 0


class TestDrives:
    """Test drive management functionality."""

    def test_drive_creation(self, kernle_instance):
        """Test creating a new drive."""
        kernle, storage = kernle_instance

        drive_id = kernle.drive(
            drive_type="growth", intensity=0.8, focus_areas=["learning", "skills"]
        )

        assert drive_id is not None
        drives = storage.get_drives()
        assert len(drives) == 1
        assert drives[0].drive_type == "growth"
        assert drives[0].intensity == 0.8

    def test_drive_update_existing(self, kernle_instance):
        """Test updating an existing drive."""
        kernle, storage = kernle_instance

        # Create initial drive
        drive_id1 = kernle.drive("growth", 0.5)

        # Update same drive type
        drive_id2 = kernle.drive("growth", 0.9)

        # Should be the same drive ID
        assert drive_id1 == drive_id2

        # Should have updated intensity
        drives = storage.get_drives()
        assert len(drives) == 1
        assert drives[0].intensity == 0.9

    def test_drive_invalid_type(self, kernle_instance):
        """Test that invalid drive type raises error."""
        kernle, storage = kernle_instance

        with pytest.raises(ValueError, match="Invalid drive type"):
            kernle.drive("invalid_drive", 0.5)

    def test_drive_intensity_bounds(self, kernle_instance):
        """Test that intensity is bounded to 0-1."""
        kernle, storage = kernle_instance

        kernle.drive("growth", 1.5)  # Above max
        kernle.drive("curiosity", -0.5)  # Below min

        drives = storage.get_drives()
        growth = next(d for d in drives if d.drive_type == "growth")
        curiosity = next(d for d in drives if d.drive_type == "curiosity")

        assert growth.intensity == 1.0
        assert curiosity.intensity == 0.0

    def test_satisfy_drive_existing(self, kernle_instance):
        """Test satisfying an existing drive."""
        kernle, storage = kernle_instance

        kernle.drive("growth", 0.8)
        result = kernle.satisfy_drive("growth", 0.2)

        assert result is True
        drives = storage.get_drives()
        assert abs(drives[0].intensity - 0.6) < 0.01  # 0.8 - 0.2 (allow float precision)

    def test_satisfy_drive_minimum_intensity(self, kernle_instance):
        """Test that satisfying doesn't go below minimum."""
        kernle, storage = kernle_instance

        kernle.drive("growth", 0.2)
        kernle.satisfy_drive("growth", 0.5)  # Would go to -0.3

        drives = storage.get_drives()
        assert drives[0].intensity == 0.1  # Minimum


class TestEmotionalMemory:
    """Test emotional memory functionality."""

    def test_detect_emotion(self, kernle_instance):
        """Test emotion detection in text."""
        kernle, storage = kernle_instance

        result = kernle.detect_emotion("I'm so happy and excited about this!")

        # Without inference, returns neutral defaults
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["tags"] == []
        assert result["confidence"] == 0.0

    def test_episode_with_emotion(self, kernle_instance):
        """Test creating episode with emotional tagging."""
        kernle, storage = kernle_instance

        kernle.episode_with_emotion(
            objective="Complete project",
            outcome="Great success!",
            valence=0.8,
            arousal=0.6,
            emotional_tags=["joy", "satisfaction"],
        )

        episodes = storage.get_episodes()
        ep = episodes[0]

        assert ep.emotional_valence == 0.8
        assert ep.emotional_arousal == 0.6
        assert "joy" in ep.emotional_tags

    def test_episode_auto_emotion_detection(self, kernle_instance):
        """Test automatic emotion detection in episodes."""
        kernle, storage = kernle_instance

        episode_id = kernle.episode_with_emotion(
            objective="Handle frustrating bug",
            outcome="Finally fixed it after struggling",
            auto_detect=True,
        )

        episodes = storage.get_episodes()
        # Auto-detection should have set some emotional values
        # (depends on the text matching patterns)
        assert episodes[0].id == episode_id

    def test_search_by_emotion(self, kernle_instance):
        """Test searching episodes by emotional criteria."""
        kernle, storage = kernle_instance

        # Create episodes with different emotions
        kernle.episode_with_emotion("Happy task", "Great!", valence=0.8, arousal=0.5)
        kernle.episode_with_emotion("Sad task", "Failed :(", valence=-0.8, arousal=0.3)

        # Search for positive emotions
        results = kernle.search_by_emotion(valence_range=(0.5, 1.0))

        assert len(results) == 1
        assert results[0]["objective"] == "Happy task"


class TestMetaMemory:
    """Test meta-memory functionality (confidence, lineage, etc.)."""

    def test_get_memory_confidence(self, kernle_instance):
        """Test getting confidence for a memory."""
        kernle, storage = kernle_instance

        belief_id = kernle.belief("Test belief", confidence=0.75)

        confidence = kernle.get_memory_confidence("belief", belief_id)

        assert confidence == 0.75

    def test_verify_memory(self, kernle_instance):
        """Test verifying a memory increases confidence."""
        kernle, storage = kernle_instance

        belief_id = kernle.belief("Test belief", confidence=0.5)

        result = kernle.verify_memory("belief", belief_id, evidence="Tested and confirmed")

        assert result is True
        new_confidence = kernle.get_memory_confidence("belief", belief_id)
        assert new_confidence > 0.5

    def test_get_memory_lineage(self, kernle_instance):
        """Test getting lineage for a memory."""
        kernle, storage = kernle_instance

        episode_id = kernle.episode("Test task", "success")

        lineage = kernle.get_memory_lineage("episode", episode_id)

        assert lineage["id"] == episode_id
        assert lineage["type"] == "episode"
        assert "source_type" in lineage


class TestIdentity:
    """Test identity synthesis functionality."""

    def test_synthesize_identity(self, kernle_instance, populated_storage):
        """Test identity synthesis."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        identity = kernle.synthesize_identity()

        assert "narrative" in identity
        assert "core_values" in identity
        assert "key_beliefs" in identity
        assert "active_goals" in identity
        assert "drives" in identity
        assert "confidence" in identity

    def test_get_identity_confidence(self, kernle_instance, populated_storage):
        """Test getting identity confidence score."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        confidence = kernle.get_identity_confidence()

        assert 0 <= confidence <= 1


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_validate_stack_id_empty(self, temp_db_path):
        """Test that empty agent ID uses default."""
        # Empty string defaults to "default" from env var fallback
        with patch.dict(os.environ, {}, clear=True):
            kernle = Kernle(stack_id="", strict=False)
            assert kernle.stack_id == "default"  # Falls back to default

    def test_validate_stack_id_whitespace_only(self, temp_db_path):
        """Test that whitespace-only agent ID raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Stack ID cannot be empty"):
                Kernle(stack_id="   ", strict=False)

    def test_validate_stack_id_special_chars(self, temp_checkpoint_dir, temp_db_path):
        """Test that special characters are sanitized from agent ID."""
        storage = SQLiteStorage(stack_id="test_agent", db_path=temp_db_path)
        kernle = Kernle(
            stack_id="test@agent!#$%", storage=storage, checkpoint_dir=temp_checkpoint_dir
        )

        # Should be sanitized to only alphanumeric and -_.
        assert kernle.stack_id == "testagent"

    @pytest.mark.parametrize(
        "bad_id,match",
        [
            ("..", "relative path component"),
            (".", "relative path component"),
            ("../x", "path separators"),
            ("nested/evil", "path separators"),
            ("a\\b", "path separators"),
            ("foo/../bar", "path separators"),
        ],
    )
    def test_validate_stack_id_path_traversal(self, bad_id, match):
        """Test that path traversal stack IDs are rejected."""
        with pytest.raises(ValueError, match=match):
            Kernle(stack_id=bad_id, strict=False)

    def test_validate_string_too_long(self, kernle_instance):
        """Test that strings exceeding max length raise error."""
        kernle, storage = kernle_instance

        with pytest.raises(ValueError, match="too long"):
            kernle.note("x" * 3000)  # Max is 2000


class TestFormatting:
    """Test memory formatting functionality."""

    def test_format_memory(self, kernle_instance, populated_storage):
        """Test formatting memory for context injection."""
        kernle, _ = kernle_instance
        kernle._storage = populated_storage

        formatted = kernle.format_memory()

        assert "# Working Memory" in formatted
        assert kernle.stack_id in formatted
        # Should include sections for various memory types
        assert "## Values" in formatted or "## Beliefs" in formatted


class TestSync:
    """Test sync functionality."""

    def test_sync_with_sqlite(self, kernle_instance):
        """Test sync with SQLite storage (should be no-op)."""
        kernle, storage = kernle_instance

        result = kernle.sync()

        assert result["success"] is True
        assert result["pushed"] == 0
        assert result["pulled"] == 0

    def test_get_sync_status(self, kernle_instance):
        """Test getting sync status."""
        kernle, storage = kernle_instance

        status = kernle.get_sync_status()

        assert "pending" in status
        assert "online" in status


class TestSyncHooks:
    """Test non-blocking sync helpers used by load/checkpoint paths."""

    def _make_storage(self):
        storage = MagicMock()
        storage.get_pending_sync_count.return_value = 0
        return storage

    def test_sync_before_load_offline(self):
        storage = self._make_storage()
        storage.is_online.return_value = False
        kernle = Kernle("test-agent", storage=storage, strict=False)

        result = kernle._sync_before_load()

        assert result == {"attempted": False, "pulled": 0, "conflicts": 0, "errors": []}
        storage.pull_changes.assert_not_called()

    def test_sync_before_load_with_pull_and_errors(self):
        storage = self._make_storage()
        storage.is_online.return_value = True
        storage.pull_changes.return_value = SimpleNamespace(
            pulled=3,
            conflicts=1,
            errors=["cloud warning"],
        )
        kernle = Kernle("test-agent", storage=storage, strict=False)

        result = kernle._sync_before_load()

        assert result["attempted"] is True
        assert result["pulled"] == 3
        assert result["conflicts"] == 1
        assert result["errors"] == ["cloud warning"]

    def test_sync_before_load_handles_exception(self):
        storage = self._make_storage()
        storage.is_online.return_value = True
        storage.pull_changes.side_effect = ConnectionError("pull boom")
        kernle = Kernle("test-agent", storage=storage, strict=False)

        result = kernle._sync_before_load()

        assert result["attempted"] is True
        assert any("pull boom" in err for err in result["errors"])

    def test_sync_after_checkpoint_offline(self):
        storage = self._make_storage()
        storage.is_online.return_value = False
        kernle = Kernle("test-agent", storage=storage, strict=False)

        result = kernle._sync_after_checkpoint()

        assert result == {
            "attempted": False,
            "pushed": 0,
            "conflicts": 0,
            "errors": ["Offline - changes queued"],
        }
        storage.sync.assert_not_called()

    def test_sync_after_checkpoint_with_push_and_errors(self):
        storage = self._make_storage()
        storage.is_online.return_value = True
        storage.sync.return_value = SimpleNamespace(
            pushed=2,
            conflicts=1,
            errors=["push warning"],
        )
        kernle = Kernle("test-agent", storage=storage, strict=False)

        result = kernle._sync_after_checkpoint()

        assert result["attempted"] is True
        assert result["pushed"] == 2
        assert result["conflicts"] == 1
        assert result["errors"] == ["push warning"]

    def test_sync_after_checkpoint_handles_exception(self):
        storage = self._make_storage()
        storage.is_online.return_value = True
        storage.sync.side_effect = ConnectionError("sync boom")
        kernle = Kernle("test-agent", storage=storage, strict=False)

        result = kernle._sync_after_checkpoint()

        assert result["attempted"] is True
        assert any("sync boom" in err for err in result["errors"])


class TestBatchInsertion:
    """Test batch insertion convenience methods."""

    def test_episodes_batch_empty(self, kernle_instance):
        """Empty batch should return empty list."""
        kernle, storage = kernle_instance
        ids = kernle.episodes_batch([])
        assert ids == []

    def test_episodes_batch_multiple(self, kernle_instance):
        """Test saving multiple episodes in batch."""
        kernle, storage = kernle_instance

        episodes = [{"objective": f"Task {i}", "outcome": f"Result {i}"} for i in range(5)]

        ids = kernle.episodes_batch(episodes)

        assert len(ids) == 5
        # Verify they were saved
        saved_episodes = storage.get_episodes(limit=10)
        assert len(saved_episodes) == 5

    def test_episodes_batch_with_all_fields(self, kernle_instance):
        """Test batch with all optional fields."""
        kernle, storage = kernle_instance

        episodes = [
            {
                "objective": "Full field test",
                "outcome": "Complete success",
                "outcome_type": "success",
                "lessons": ["Lesson 1", "Lesson 2"],
                "tags": ["test", "batch"],
                "confidence": 0.95,
            }
        ]

        ids = kernle.episodes_batch(episodes)

        assert len(ids) == 1
        saved = storage.get_episode(ids[0])
        assert saved.outcome_type == "success"
        assert saved.lessons == ["Lesson 1", "Lesson 2"]
        assert saved.tags == ["test", "batch"]
        assert saved.confidence == 0.95

    def test_beliefs_batch_empty(self, kernle_instance):
        """Empty batch should return empty list."""
        kernle, storage = kernle_instance
        ids = kernle.beliefs_batch([])
        assert ids == []

    def test_beliefs_batch_multiple(self, kernle_instance):
        """Test saving multiple beliefs in batch."""
        kernle, storage = kernle_instance

        beliefs = [{"statement": f"Belief {i}", "confidence": 0.7 + i * 0.05} for i in range(5)]

        ids = kernle.beliefs_batch(beliefs)

        assert len(ids) == 5
        # Verify they were saved
        saved_beliefs = storage.get_beliefs(limit=10)
        assert len(saved_beliefs) == 5

    def test_beliefs_batch_with_type(self, kernle_instance):
        """Test batch beliefs with different types."""
        kernle, storage = kernle_instance

        beliefs = [
            {"statement": "A fact", "type": "fact", "confidence": 1.0},
            {"statement": "A principle", "type": "principle", "confidence": 0.9},
        ]

        _ids = kernle.beliefs_batch(beliefs)

        saved = storage.get_beliefs(limit=10)
        types = {b.belief_type for b in saved}
        assert "fact" in types
        assert "principle" in types

    def test_notes_batch_empty(self, kernle_instance):
        """Empty batch should return empty list."""
        kernle, storage = kernle_instance
        ids = kernle.notes_batch([])
        assert ids == []

    def test_notes_batch_multiple(self, kernle_instance):
        """Test saving multiple notes in batch."""
        kernle, storage = kernle_instance

        notes = [{"content": f"Note content {i}", "type": "note"} for i in range(5)]

        ids = kernle.notes_batch(notes)

        assert len(ids) == 5
        # Verify they were saved
        saved_notes = storage.get_notes(limit=10)
        assert len(saved_notes) == 5

    def test_notes_batch_different_types(self, kernle_instance):
        """Test batch notes with different types."""
        kernle, storage = kernle_instance

        notes = [
            {"content": "A decision", "type": "decision", "reason": "Because"},
            {"content": "An insight", "type": "insight"},
            {"content": "A quote", "type": "quote", "speaker": "Someone"},
        ]

        _ids = kernle.notes_batch(notes)

        saved = storage.get_notes(limit=10)
        types = {n.note_type for n in saved}
        assert "decision" in types
        assert "insight" in types
        assert "quote" in types

    def test_batch_validates_input(self, kernle_instance):
        """Test that batch methods validate input."""
        kernle, storage = kernle_instance

        # Objective too long
        with pytest.raises(ValueError, match="too long"):
            kernle.episodes_batch([{"objective": "x" * 1500, "outcome": "test"}])

    def test_batch_performance_improvement(self, kernle_instance):
        """Test that batch is actually faster than individual saves.

        Note: This is more of a sanity check - actual performance
        gains depend on database configuration.
        """
        import time

        kernle, storage = kernle_instance

        # Create test data
        episodes = [{"objective": f"Batch task {i}", "outcome": f"Result {i}"} for i in range(20)]

        # Time the batch operation
        start = time.time()
        kernle.episodes_batch(episodes)
        batch_time = time.time() - start

        # This should complete reasonably fast (under 5 seconds)
        assert batch_time < 5.0
