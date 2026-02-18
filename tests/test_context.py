"""Tests for context/scope functionality (Issue #6).

Tests that context and context_tags fields are properly saved and retrieved
for all memory types, enabling project-specific memory isolation.
"""

from pathlib import Path

import pytest

from kernle.core import Kernle
from kernle.storage import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    Relationship,
    SQLiteStorage,
    Value,
)
from tests.conftest import bind_noop_model


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    storage = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    yield storage
    storage.close()


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary checkpoint directory."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def kernle_instance(temp_db, temp_checkpoint_dir):
    """Create a Kernle instance with SQLite storage for testing."""
    storage = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    kernle = Kernle(
        stack_id="test-agent",
        storage=storage,
        checkpoint_dir=temp_checkpoint_dir,
        strict=False,
    )
    bind_noop_model(kernle)
    yield kernle, storage
    storage.close()


class TestEpisodeContext:
    """Test context fields on Episode storage."""

    def test_save_episode_with_context(self, storage):
        """Episode context and context_tags should be saved and retrieved."""
        episode = Episode(
            id="ep-ctx-1",
            stack_id="test-agent",
            objective="Build API endpoint",
            outcome="Successfully deployed",
            context="project:api-service",
            context_tags=["backend", "python", "fastapi"],
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-ctx-1")

        assert retrieved is not None
        assert retrieved.context == "project:api-service"
        assert retrieved.context_tags == ["backend", "python", "fastapi"]

    def test_save_episode_without_context(self, storage):
        """Episode without context should work (backwards compatible)."""
        episode = Episode(
            id="ep-no-ctx",
            stack_id="test-agent",
            objective="Generic task",
            outcome="Done",
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-no-ctx")

        assert retrieved is not None
        assert retrieved.context is None
        assert retrieved.context_tags is None

    def test_episode_context_via_core(self, kernle_instance):
        """Test creating episode with context through Kernle core."""
        kernle, storage = kernle_instance

        episode_id = kernle.episode(
            objective="Implement user auth",
            outcome="success",
            context="repo:myorg/auth-service",
            context_tags=["security", "auth"],
        )

        retrieved = storage.get_episode(episode_id)
        assert retrieved.context == "repo:myorg/auth-service"
        assert retrieved.context_tags == ["security", "auth"]


class TestNoteContext:
    """Test context fields on Note storage."""

    def test_save_note_with_context(self, storage):
        """Note context and context_tags should be saved and retrieved."""
        note = Note(
            id="note-ctx-1",
            stack_id="test-agent",
            content="Important API design decision",
            note_type="decision",
            context="project:api-service",
            context_tags=["architecture", "design"],
        )

        storage.save_note(note)
        notes = storage.get_notes(limit=10)

        assert len(notes) == 1
        assert notes[0].context == "project:api-service"
        assert notes[0].context_tags == ["architecture", "design"]

    def test_save_note_without_context(self, storage):
        """Note without context should work (backwards compatible)."""
        note = Note(
            id="note-no-ctx",
            stack_id="test-agent",
            content="General note",
        )

        storage.save_note(note)
        notes = storage.get_notes(limit=10)

        assert len(notes) == 1
        assert notes[0].context is None
        assert notes[0].context_tags is None

    def test_note_context_via_core(self, kernle_instance):
        """Test creating note with context through Kernle core."""
        kernle, storage = kernle_instance

        _note_id = kernle.note(
            content="Database schema updated",
            type="note",
            context="project:data-pipeline",
            context_tags=["database", "schema"],
        )

        notes = storage.get_notes(limit=10)
        assert len(notes) == 1
        assert notes[0].context == "project:data-pipeline"
        assert notes[0].context_tags == ["database", "schema"]


class TestBeliefContext:
    """Test context fields on Belief storage."""

    def test_save_belief_with_context(self, storage):
        """Belief context and context_tags should be saved and retrieved."""
        belief = Belief(
            id="belief-ctx-1",
            stack_id="test-agent",
            statement="This API uses REST conventions",
            belief_type="fact",
            confidence=0.9,
            context="project:api-service",
            context_tags=["api", "rest"],
        )

        storage.save_belief(belief)
        beliefs = storage.get_beliefs(limit=10)

        assert len(beliefs) == 1
        assert beliefs[0].context == "project:api-service"
        assert beliefs[0].context_tags == ["api", "rest"]

    def test_save_belief_without_context(self, storage):
        """Belief without context should work (backwards compatible)."""
        belief = Belief(
            id="belief-no-ctx",
            stack_id="test-agent",
            statement="Testing is important",
        )

        storage.save_belief(belief)
        beliefs = storage.get_beliefs(limit=10)

        assert len(beliefs) == 1
        assert beliefs[0].context is None
        assert beliefs[0].context_tags is None

    def test_belief_context_via_core(self, kernle_instance):
        """Test creating belief with context through Kernle core."""
        kernle, storage = kernle_instance

        _belief_id = kernle.belief(
            statement="The payment API requires OAuth2",
            type="fact",
            confidence=0.95,
            context="project:payments",
            context_tags=["api", "auth", "oauth"],
        )

        beliefs = storage.get_beliefs(limit=10)
        assert len(beliefs) == 1
        assert beliefs[0].context == "project:payments"
        assert beliefs[0].context_tags == ["api", "auth", "oauth"]


class TestValueContext:
    """Test context fields on Value storage."""

    def test_save_value_with_context(self, storage):
        """Value context and context_tags should be saved and retrieved."""
        value = Value(
            id="value-ctx-1",
            stack_id="test-agent",
            name="Code Coverage",
            statement="Maintain 80% test coverage",
            priority=85,
            context="project:api-service",
            context_tags=["testing", "quality"],
        )

        storage.save_value(value)
        values = storage.get_values(limit=10)

        assert len(values) == 1
        assert values[0].context == "project:api-service"
        assert values[0].context_tags == ["testing", "quality"]

    def test_save_value_without_context(self, storage):
        """Value without context should work (backwards compatible)."""
        value = Value(
            id="value-no-ctx",
            stack_id="test-agent",
            name="Quality",
            statement="Quality matters",
        )

        storage.save_value(value)
        values = storage.get_values(limit=10)

        assert len(values) == 1
        assert values[0].context is None
        assert values[0].context_tags is None

    def test_value_context_via_core(self, kernle_instance):
        """Test creating value with context through Kernle core."""
        kernle, storage = kernle_instance

        _value_id = kernle.value(
            name="Security First",
            statement="Security is the top priority",
            priority=100,
            context="project:auth-service",
            context_tags=["security"],
        )

        values = storage.get_values(limit=10)
        assert len(values) == 1
        assert values[0].context == "project:auth-service"
        assert values[0].context_tags == ["security"]


class TestGoalContext:
    """Test context fields on Goal storage."""

    def test_save_goal_with_context(self, storage):
        """Goal context and context_tags should be saved and retrieved."""
        goal = Goal(
            id="goal-ctx-1",
            stack_id="test-agent",
            title="Complete API v2",
            description="Finish all v2 endpoints",
            priority="high",
            status="active",
            context="project:api-service",
            context_tags=["milestone", "api"],
        )

        storage.save_goal(goal)
        goals = storage.get_goals(limit=10)

        assert len(goals) == 1
        assert goals[0].context == "project:api-service"
        assert goals[0].context_tags == ["milestone", "api"]

    def test_save_goal_without_context(self, storage):
        """Goal without context should work (backwards compatible)."""
        goal = Goal(
            id="goal-no-ctx",
            stack_id="test-agent",
            title="General goal",
            description="Some goal",
        )

        storage.save_goal(goal)
        goals = storage.get_goals(limit=10)

        assert len(goals) == 1
        assert goals[0].context is None
        assert goals[0].context_tags is None

    def test_goal_context_via_core(self, kernle_instance):
        """Test creating goal with context through Kernle core."""
        kernle, storage = kernle_instance

        _goal_id = kernle.goal(
            title="Launch MVP",
            description="Ship minimum viable product",
            priority="high",
            context="project:startup-app",
            context_tags=["launch", "mvp"],
        )

        goals = storage.get_goals(limit=10)
        assert len(goals) == 1
        assert goals[0].context == "project:startup-app"
        assert goals[0].context_tags == ["launch", "mvp"]


class TestDriveContext:
    """Test context fields on Drive storage."""

    def test_save_drive_with_context(self, storage):
        """Drive context and context_tags should be saved and retrieved."""
        drive = Drive(
            id="drive-ctx-1",
            stack_id="test-agent",
            drive_type="growth",
            intensity=0.8,
            focus_areas=["learning", "improvement"],
            context="project:api-service",
            context_tags=["development"],
        )

        storage.save_drive(drive)
        drives = storage.get_drives()

        assert len(drives) == 1
        assert drives[0].context == "project:api-service"
        assert drives[0].context_tags == ["development"]

    def test_save_drive_without_context(self, storage):
        """Drive without context should work (backwards compatible)."""
        drive = Drive(
            id="drive-no-ctx",
            stack_id="test-agent",
            drive_type="curiosity",
            intensity=0.5,
        )

        storage.save_drive(drive)
        drives = storage.get_drives()

        assert len(drives) == 1
        assert drives[0].context is None
        assert drives[0].context_tags is None

    def test_drive_context_via_core(self, kernle_instance):
        """Test creating drive with context through Kernle core."""
        kernle, storage = kernle_instance

        _drive_id = kernle.drive(
            drive_type="growth",
            intensity=0.9,
            focus_areas=["performance", "scalability"],
            context="project:high-traffic-api",
            context_tags=["optimization"],
        )

        drives = storage.get_drives()
        assert len(drives) == 1
        assert drives[0].context == "project:high-traffic-api"
        assert drives[0].context_tags == ["optimization"]


class TestRelationshipContext:
    """Test context fields on Relationship storage."""

    def test_save_relationship_with_context(self, storage):
        """Relationship context and context_tags should be saved and retrieved."""
        rel = Relationship(
            id="rel-ctx-1",
            stack_id="test-agent",
            entity_name="Alice",
            entity_type="human",
            relationship_type="colleague",
            sentiment=0.8,
            context="project:api-service",
            context_tags=["team", "backend"],
        )

        storage.save_relationship(rel)
        rels = storage.get_relationships()

        assert len(rels) == 1
        assert rels[0].context == "project:api-service"
        assert rels[0].context_tags == ["team", "backend"]

    def test_save_relationship_without_context(self, storage):
        """Relationship without context should work (backwards compatible)."""
        rel = Relationship(
            id="rel-no-ctx",
            stack_id="test-agent",
            entity_name="Bob",
            entity_type="human",
            relationship_type="friend",
        )

        storage.save_relationship(rel)
        rels = storage.get_relationships()

        assert len(rels) == 1
        assert rels[0].context is None
        assert rels[0].context_tags is None


class TestBatchInsertionWithContext:
    """Test that batch insertion preserves context fields."""

    def test_episodes_batch_with_context(self, storage):
        """Batch episode insert should preserve context fields."""
        episodes = [
            Episode(
                id=f"ep-batch-ctx-{i}",
                stack_id="test-agent",
                objective=f"Task {i}",
                outcome="Done",
                context="project:batch-test",
                context_tags=["batch", f"tag-{i}"],
            )
            for i in range(3)
        ]

        storage.save_episodes_batch(episodes)

        for i in range(3):
            retrieved = storage.get_episode(f"ep-batch-ctx-{i}")
            assert retrieved.context == "project:batch-test"
            assert "batch" in retrieved.context_tags
            assert f"tag-{i}" in retrieved.context_tags

    def test_beliefs_batch_with_context(self, storage):
        """Batch belief insert should preserve context fields."""
        beliefs = [
            Belief(
                id=f"belief-batch-ctx-{i}",
                stack_id="test-agent",
                statement=f"Belief {i}",
                context="project:batch-test",
                context_tags=["batch"],
            )
            for i in range(3)
        ]

        storage.save_beliefs_batch(beliefs)
        all_beliefs = storage.get_beliefs(limit=10)

        assert len(all_beliefs) == 3
        for belief in all_beliefs:
            assert belief.context == "project:batch-test"
            assert belief.context_tags == ["batch"]

    def test_notes_batch_with_context(self, storage):
        """Batch note insert should preserve context fields."""
        notes = [
            Note(
                id=f"note-batch-ctx-{i}",
                stack_id="test-agent",
                content=f"Note {i}",
                context="project:batch-test",
                context_tags=["batch"],
            )
            for i in range(3)
        ]

        storage.save_notes_batch(notes)
        all_notes = storage.get_notes(limit=10)

        assert len(all_notes) == 3
        for note in all_notes:
            assert note.context == "project:batch-test"
            assert note.context_tags == ["batch"]


class TestContextTagSerialization:
    """Test that context_tags list is properly serialized/deserialized."""

    def test_empty_context_tags(self, storage):
        """Empty context_tags list should be preserved."""
        episode = Episode(
            id="ep-empty-tags",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            context="project:test",
            context_tags=[],
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-empty-tags")

        assert retrieved.context == "project:test"
        assert retrieved.context_tags == []

    def test_single_context_tag(self, storage):
        """Single context tag should be preserved."""
        episode = Episode(
            id="ep-single-tag",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            context="project:test",
            context_tags=["only-one"],
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-single-tag")

        assert retrieved.context_tags == ["only-one"]

    def test_many_context_tags(self, storage):
        """Many context tags should be preserved."""
        tags = [f"tag-{i}" for i in range(20)]
        episode = Episode(
            id="ep-many-tags",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            context="project:test",
            context_tags=tags,
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-many-tags")

        assert retrieved.context_tags == tags
        assert len(retrieved.context_tags) == 20

    def test_context_tags_with_special_chars(self, storage):
        """Context tags with special characters should be preserved."""
        tags = ["tag:with:colons", "tag/with/slashes", "tag-with-dashes", "tag_with_underscores"]
        episode = Episode(
            id="ep-special-tags",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            context="project:test",
            context_tags=tags,
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-special-tags")

        assert retrieved.context_tags == tags


class TestContextFormats:
    """Test different context string formats."""

    def test_project_context(self, storage):
        """project: prefix context format."""
        episode = Episode(
            id="ep-project",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            context="project:my-api-service",
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-project")

        assert retrieved.context == "project:my-api-service"

    def test_repo_context(self, storage):
        """repo: prefix context format."""
        episode = Episode(
            id="ep-repo",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            context="repo:myorg/myrepo",
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-repo")

        assert retrieved.context == "repo:myorg/myrepo"

    def test_custom_context(self, storage):
        """Custom context format without standard prefix."""
        episode = Episode(
            id="ep-custom",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            context="custom-context-value",
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-custom")

        assert retrieved.context == "custom-context-value"


class TestEmotionalEpisodeWithContext:
    """Test context fields with emotional memory features."""

    def test_episode_with_emotion_and_context(self, kernle_instance):
        """Test episode_with_emotion preserves context fields."""
        kernle, storage = kernle_instance

        episode_id = kernle.episode_with_emotion(
            objective="Complete exciting project",
            outcome="Great success!",
            valence=0.8,
            arousal=0.7,
            emotional_tags=["joy", "excitement"],
            context="project:exciting-app",
            context_tags=["milestone", "launch"],
        )

        retrieved = storage.get_episode(episode_id)

        assert retrieved.context == "project:exciting-app"
        assert retrieved.context_tags == ["milestone", "launch"]
        assert retrieved.emotional_valence == 0.8
        assert retrieved.emotional_arousal == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
