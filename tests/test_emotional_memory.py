"""Tests for emotional memory system.

Tests the emotional memory features:
- Emotional fields on episodes
- Emotional signal detection
- Emotional search and retrieval
- Mood-congruent memory retrieval
"""

from datetime import datetime, timedelta, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import (
    Episode,
    SQLiteStorage,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    s = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    yield s
    s.close()


class TestEmotionalFieldsOnEpisode:
    """Test that Episode dataclass has emotional fields."""

    def test_episode_has_emotional_valence(self):
        episode = Episode(
            id="ep-1",
            stack_id="test",
            objective="Test",
            outcome="Done",
            emotional_valence=0.8,
        )
        assert episode.emotional_valence == 0.8

    def test_episode_has_emotional_arousal(self):
        episode = Episode(
            id="ep-1",
            stack_id="test",
            objective="Test",
            outcome="Done",
            emotional_arousal=0.5,
        )
        assert episode.emotional_arousal == 0.5

    def test_episode_has_emotional_tags(self):
        episode = Episode(
            id="ep-1",
            stack_id="test",
            objective="Test",
            outcome="Done",
            emotional_tags=["joy", "excitement"],
        )
        assert episode.emotional_tags == ["joy", "excitement"]

    def test_episode_emotional_defaults(self):
        episode = Episode(
            id="ep-1",
            stack_id="test",
            objective="Test",
            outcome="Done",
        )
        assert episode.emotional_valence == 0.0
        assert episode.emotional_arousal == 0.0
        assert episode.emotional_tags is None


class TestSQLiteEmotionalStorage:
    """Test SQLite storage for emotional fields."""

    def test_save_episode_with_emotion(self, storage):
        episode = Episode(
            id="ep-emotional",
            stack_id="test-agent",
            objective="Emotional test",
            outcome="Feeling great!",
            emotional_valence=0.9,
            emotional_arousal=0.7,
            emotional_tags=["joy", "excitement"],
        )

        saved_id = storage.save_episode(episode)
        assert saved_id == "ep-emotional"

        # Retrieve and verify
        retrieved = storage.get_episode("ep-emotional")
        assert retrieved is not None
        assert retrieved.emotional_valence == 0.9
        assert retrieved.emotional_arousal == 0.7
        assert retrieved.emotional_tags == ["joy", "excitement"]

    def test_update_episode_emotion(self, storage):
        # Create episode without emotion
        episode = Episode(
            id="ep-update",
            stack_id="test-agent",
            objective="Update test",
            outcome="Done",
        )
        storage.save_episode(episode)

        # Update with emotion
        result = storage.update_episode_emotion(
            "ep-update",
            valence=0.5,
            arousal=0.3,
            tags=["satisfaction"],
        )
        assert result is True

        # Verify update
        retrieved = storage.get_episode("ep-update")
        assert retrieved.emotional_valence == 0.5
        assert retrieved.emotional_arousal == 0.3
        assert retrieved.emotional_tags == ["satisfaction"]

    def test_update_nonexistent_episode(self, storage):
        result = storage.update_episode_emotion(
            "nonexistent-id",
            valence=0.5,
            arousal=0.3,
        )
        assert result is False

    def test_search_by_emotion_valence(self, storage):
        # Create episodes with different valences
        for i, (v, label) in enumerate([(-0.8, "negative"), (0.0, "neutral"), (0.8, "positive")]):
            storage.save_episode(
                Episode(
                    id=f"ep-{label}",
                    stack_id="test-agent",
                    objective=f"{label} experience",
                    outcome=f"Outcome {i}",
                    emotional_valence=v,
                    emotional_arousal=0.5,
                )
            )

        # Search for positive episodes
        positive = storage.search_by_emotion(valence_range=(0.5, 1.0))
        assert len(positive) == 1
        assert positive[0].id == "ep-positive"

        # Search for negative episodes
        negative = storage.search_by_emotion(valence_range=(-1.0, -0.5))
        assert len(negative) == 1
        assert negative[0].id == "ep-negative"

    def test_search_by_emotion_arousal(self, storage):
        # Create episodes with different arousal levels
        storage.save_episode(
            Episode(
                id="ep-calm",
                stack_id="test-agent",
                objective="Calm experience",
                outcome="Peaceful",
                emotional_valence=0.3,
                emotional_arousal=0.1,
            )
        )
        storage.save_episode(
            Episode(
                id="ep-intense",
                stack_id="test-agent",
                objective="Intense experience",
                outcome="Exciting",
                emotional_valence=0.3,
                emotional_arousal=0.9,
            )
        )

        # Search for calm episodes
        calm = storage.search_by_emotion(arousal_range=(0.0, 0.3))
        assert len(calm) == 1
        assert calm[0].id == "ep-calm"

        # Search for intense episodes
        intense = storage.search_by_emotion(arousal_range=(0.7, 1.0))
        assert len(intense) == 1
        assert intense[0].id == "ep-intense"

    def test_search_by_emotion_tags(self, storage):
        storage.save_episode(
            Episode(
                id="ep-joy",
                stack_id="test-agent",
                objective="Joyful moment",
                outcome="Happy",
                emotional_tags=["joy", "excitement"],
            )
        )
        storage.save_episode(
            Episode(
                id="ep-sad",
                stack_id="test-agent",
                objective="Sad moment",
                outcome="Disappointed",
                emotional_tags=["sadness", "disappointment"],
            )
        )

        # Search by tag
        joyful = storage.search_by_emotion(tags=["joy"])
        assert len(joyful) == 1
        assert joyful[0].id == "ep-joy"

    def test_get_emotional_episodes(self, storage):
        # Create mix of emotional and non-emotional episodes
        storage.save_episode(
            Episode(
                id="ep-with-emotion",
                stack_id="test-agent",
                objective="Emotional",
                outcome="Done",
                emotional_valence=0.5,
            )
        )
        storage.save_episode(
            Episode(
                id="ep-no-emotion",
                stack_id="test-agent",
                objective="Plain",
                outcome="Done",
            )
        )

        emotional = storage.get_emotional_episodes(days=7)
        assert len(emotional) == 1
        assert emotional[0].id == "ep-with-emotion"


class TestEmotionDetection:
    """Test automatic emotion detection from text."""

    @pytest.fixture
    def kernle_instance(self, temp_db, temp_checkpoint_dir):
        """Create Kernle with SQLite storage."""
        storage = SQLiteStorage(stack_id="test_agent", db_path=temp_db)
        k = Kernle(
            stack_id="test_agent",
            storage=storage,
            checkpoint_dir=temp_checkpoint_dir,
            strict=False,
        )
        return k

    def test_detect_emotion_returns_neutral_without_inference(self, kernle_instance):
        """Without inference, detect_emotion returns neutral defaults."""
        result = kernle_instance.detect_emotion("I'm so happy and excited about this!")
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["tags"] == []
        assert result["confidence"] == 0.0

    def test_detect_neutral_text(self, kernle_instance):
        result = kernle_instance.detect_emotion("The system processes data.")
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["tags"] == []
        assert result["confidence"] == 0.0


class TestMoodCongruentRetrieval:
    """Test mood-congruent memory retrieval."""

    @pytest.fixture
    def kernle_with_memories(self, temp_db, temp_checkpoint_dir):
        """Create Kernle with pre-populated emotional memories using SQLite."""
        storage = SQLiteStorage(stack_id="test_agent", db_path=temp_db)

        # Add some emotional episodes
        episodes = [
            Episode(
                id="ep-happy-1",
                stack_id="test_agent",
                objective="Celebrate success",
                outcome="Great achievement!",
                outcome_type="success",
                emotional_valence=0.8,
                emotional_arousal=0.6,
                emotional_tags=["joy", "pride"],
                lessons=["Success feels great"],
                created_at=datetime.now(timezone.utc),
            ),
            Episode(
                id="ep-sad-1",
                stack_id="test_agent",
                objective="Handle setback",
                outcome="Didn't work out",
                outcome_type="failure",
                emotional_valence=-0.6,
                emotional_arousal=0.3,
                emotional_tags=["disappointment"],
                lessons=["Learn from failures"],
                created_at=datetime.now(timezone.utc),
            ),
            Episode(
                id="ep-neutral-1",
                stack_id="test_agent",
                objective="Regular task",
                outcome="Done",
                outcome_type="success",
                emotional_valence=0.0,
                emotional_arousal=0.1,
                emotional_tags=[],
                lessons=[],
                created_at=datetime.now(timezone.utc),
            ),
        ]
        for ep in episodes:
            storage.save_episode(ep)

        k = Kernle(
            stack_id="test_agent",
            storage=storage,
            checkpoint_dir=temp_checkpoint_dir,
            strict=False,
        )
        return k, storage

    def test_positive_mood_retrieves_positive_memories(self, kernle_with_memories):
        k, _ = kernle_with_memories

        memories = k.get_mood_relevant_memories(
            current_valence=0.7,
            current_arousal=0.5,
            limit=5,
        )

        # Should return memories when queried
        assert isinstance(memories, list)
        # Should prefer positive memories when in positive mood
        if memories:
            # At least the first result should have positive or neutral valence
            # (mood-congruent retrieval favors similar emotional states)
            first_valence = memories[0].get("emotional_valence", 0)
            # First result should be closer to our positive mood than to negative
            assert (
                first_valence >= -0.3
            ), f"Expected positive-leaning result, got valence={first_valence}"

    def test_negative_mood_retrieves_negative_memories(self, kernle_with_memories):
        k, _ = kernle_with_memories

        memories = k.get_mood_relevant_memories(
            current_valence=-0.5,
            current_arousal=0.3,
            limit=5,
        )

        # Should return memories when queried
        assert isinstance(memories, list)
        # Should prefer negative memories when in negative mood
        if memories:
            first_valence = memories[0].get("emotional_valence", 0)
            # First result should be closer to our negative mood than to positive
            assert (
                first_valence <= 0.3
            ), f"Expected negative-leaning result, got valence={first_valence}"


class TestEmotionalSummary:
    """Test emotional summary generation."""

    @pytest.fixture
    def kernle_with_trajectory(self, temp_db, temp_checkpoint_dir):
        """Create Kernle with emotional history using SQLite."""
        storage = SQLiteStorage(stack_id="test_agent", db_path=temp_db)

        # Create episodes over several days with different emotions
        base_time = datetime.now(timezone.utc)
        for i in range(5):
            day_offset = timedelta(days=i)
            valence = 0.2 * (i - 2)  # -0.4 to 0.4
            episode = Episode(
                id=f"ep-day-{i}",
                stack_id="test_agent",
                objective=f"Day {i} task",
                outcome=f"Result {i}",
                outcome_type="success",
                emotional_valence=valence,
                emotional_arousal=0.5,
                emotional_tags=["curiosity"] if valence > 0 else ["frustration"],
                lessons=[],
                created_at=base_time - day_offset,
            )
            storage.save_episode(episode)

        k = Kernle(
            stack_id="test_agent",
            storage=storage,
            checkpoint_dir=temp_checkpoint_dir,
            strict=False,
        )
        return k

    def test_emotional_summary_averages(self, kernle_with_trajectory):
        k = kernle_with_trajectory
        summary = k.get_emotional_summary(days=7)

        assert "average_valence" in summary
        assert "average_arousal" in summary
        assert "episode_count" in summary
        assert summary["episode_count"] == 5

    def test_emotional_summary_dominant_emotions(self, kernle_with_trajectory):
        k = kernle_with_trajectory
        summary = k.get_emotional_summary(days=7)

        assert "dominant_emotions" in summary
        assert len(summary["dominant_emotions"]) > 0

    def test_emotional_summary_trajectory(self, kernle_with_trajectory):
        k = kernle_with_trajectory
        summary = k.get_emotional_summary(days=7)

        assert "emotional_trajectory" in summary
        # Should have daily data points
        assert len(summary["emotional_trajectory"]) > 0


class TestEpisodeWithEmotion:
    """Test creating episodes with automatic emotion detection."""

    @pytest.fixture
    def kernle_instance(self, temp_db, temp_checkpoint_dir):
        """Create a Kernle instance with SQLite storage."""
        storage = SQLiteStorage(stack_id="test_agent", db_path=temp_db)
        k = Kernle(
            stack_id="test_agent",
            storage=storage,
            checkpoint_dir=temp_checkpoint_dir,
            strict=False,
        )
        return k, storage

    def test_episode_with_explicit_emotion(self, kernle_instance):
        k, storage = kernle_instance

        episode_id = k.episode_with_emotion(
            objective="Test with explicit emotion",
            outcome="Success!",
            valence=0.9,
            arousal=0.7,
            emotional_tags=["pride"],
            auto_detect=False,
        )

        assert episode_id is not None
        # Check it was stored
        episodes = storage.get_episodes()
        assert len(episodes) > 0

    def test_episode_with_auto_detection(self, kernle_instance):
        k, storage = kernle_instance

        episode_id = k.episode_with_emotion(
            objective="This is amazing and I'm so excited!",
            outcome="Fantastic results, really happy!",
            auto_detect=True,
        )

        assert episode_id is not None


class TestValenceArousalBounds:
    """Test that valence and arousal are properly bounded."""

    def test_valence_clamped_high(self, storage):
        episode = Episode(
            id="ep-high-v",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            emotional_valence=2.0,  # Out of bounds
        )
        storage.save_episode(episode)

        # Should be clamped when updating
        storage.update_episode_emotion("ep-high-v", valence=2.0, arousal=0.5)
        retrieved = storage.get_episode("ep-high-v")
        assert retrieved.emotional_valence <= 1.0

    def test_valence_clamped_low(self, storage):
        # Create an episode to test clamping on
        episode = Episode(
            id="ep-low-v",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
        )
        storage.save_episode(episode)

        # Try to set valence below bounds
        storage.update_episode_emotion("ep-low-v", valence=-2.0, arousal=0.5)
        retrieved = storage.get_episode("ep-low-v")
        assert retrieved.emotional_valence >= -1.0, "Valence should be clamped to >= -1.0"

    def test_arousal_clamped_high(self, storage):
        episode = Episode(
            id="ep-high-a",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
        )
        storage.save_episode(episode)

        storage.update_episode_emotion("ep-high-a", valence=0.0, arousal=1.5)
        retrieved = storage.get_episode("ep-high-a")
        assert retrieved.emotional_arousal <= 1.0

    def test_arousal_clamped_low(self, storage):
        episode = Episode(
            id="ep-low-a",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
        )
        storage.save_episode(episode)

        storage.update_episode_emotion("ep-low-a", valence=0.0, arousal=-0.5)
        retrieved = storage.get_episode("ep-low-a")
        assert retrieved.emotional_arousal >= 0.0
