"""Emotional memory mixin for Kernle.

This module provides emotional tagging and mood-aware recall capabilities,
enabling mood-congruent memory retrieval.
"""

import logging
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kernle.core.writers import WritersMixin as _WritersMixin
from kernle.storage import Episode
from kernle.types import SourceType

if TYPE_CHECKING:
    from kernle.core import Kernle

logger = logging.getLogger(__name__)


class EmotionsMixin:
    """Mixin providing emotional memory capabilities.

    Enables:
    - Emotional tagging of episodes (valence/arousal model)
    - Automatic emotion detection in text
    - Mood-congruent memory recall
    - Emotional pattern analysis over time
    """

    # Emotional signal patterns for automatic tagging
    EMOTION_PATTERNS = {
        # Positive emotions (high valence)
        "joy": {
            "keywords": [
                "happy",
                "joy",
                "delighted",
                "wonderful",
                "amazing",
                "fantastic",
                "love it",
                "excited",
            ],
            "valence": 0.8,
            "arousal": 0.6,
        },
        "satisfaction": {
            "keywords": ["satisfied", "pleased", "content", "glad", "good", "nice", "well done"],
            "valence": 0.6,
            "arousal": 0.3,
        },
        "excitement": {
            "keywords": ["excited", "thrilled", "pumped", "can't wait", "awesome", "incredible"],
            "valence": 0.7,
            "arousal": 0.9,
        },
        "curiosity": {
            "keywords": [
                "curious",
                "interesting",
                "fascinating",
                "wonder",
                "intriguing",
                "want to know",
            ],
            "valence": 0.3,
            "arousal": 0.5,
        },
        "pride": {
            "keywords": ["proud", "accomplished", "achieved", "nailed it", "crushed it"],
            "valence": 0.7,
            "arousal": 0.5,
        },
        "gratitude": {
            "keywords": ["grateful", "thankful", "appreciate", "thanks so much", "means a lot"],
            "valence": 0.7,
            "arousal": 0.3,
        },
        # Negative emotions (low valence)
        "frustration": {
            "keywords": [
                "frustrated",
                "annoying",
                "irritated",
                "ugh",
                "argh",
                "why won't",
                "doesn't work",
            ],
            "valence": -0.6,
            "arousal": 0.7,
        },
        "disappointment": {
            "keywords": ["disappointed", "let down", "expected better", "unfortunate", "bummer"],
            "valence": -0.5,
            "arousal": 0.3,
        },
        "anxiety": {
            "keywords": ["worried", "anxious", "nervous", "concerned", "stressed", "overwhelmed"],
            "valence": -0.4,
            "arousal": 0.7,
        },
        "confusion": {
            "keywords": ["confused", "don't understand", "unclear", "lost", "what do you mean"],
            "valence": -0.2,
            "arousal": 0.4,
        },
        "sadness": {
            "keywords": ["sad", "unhappy", "depressed", "down", "terrible", "awful"],
            "valence": -0.7,
            "arousal": 0.2,
        },
        "anger": {
            "keywords": ["angry", "furious", "mad", "hate", "outraged", "unacceptable"],
            "valence": -0.8,
            "arousal": 0.9,
        },
    }

    def detect_emotion(self: "Kernle", text: str) -> Dict[str, Any]:
        """Detect emotional signals in text.

        Args:
            text: Text to analyze for emotional content

        Returns:
            dict with:
            - valence: float (-1.0 to 1.0)
            - arousal: float (0.0 to 1.0)
            - tags: list[str] - detected emotion labels
            - confidence: float - how confident we are
        """
        # Handle None or empty input defensively
        if not text:
            return {"valence": 0.0, "arousal": 0.0, "tags": [], "confidence": 0.0}

        text_lower = text.lower()
        detected_emotions = []
        valence_sum = 0.0
        arousal_sum = 0.0

        for emotion_name, pattern in self.EMOTION_PATTERNS.items():
            for keyword in pattern["keywords"]:
                if keyword in text_lower:
                    detected_emotions.append(emotion_name)
                    valence_sum += pattern["valence"]
                    arousal_sum += pattern["arousal"]
                    break  # One match per emotion is enough

        if detected_emotions:
            # Average the emotional values
            count = len(detected_emotions)
            avg_valence = max(-1.0, min(1.0, valence_sum / count))
            avg_arousal = max(0.0, min(1.0, arousal_sum / count))
            confidence = min(1.0, 0.3 + (count * 0.2))  # More matches = higher confidence
        else:
            avg_valence = 0.0
            avg_arousal = 0.0
            confidence = 0.0

        return {
            "valence": avg_valence,
            "arousal": avg_arousal,
            "tags": detected_emotions,
            "confidence": confidence,
        }

    def add_emotional_association(
        self: "Kernle",
        episode_id: str,
        valence: float,
        arousal: float,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Add or update emotional associations for an episode.

        Args:
            episode_id: The episode to update
            valence: Emotional valence (-1.0 negative to 1.0 positive)
            arousal: Emotional arousal (0.0 calm to 1.0 intense)
            tags: Emotional labels (e.g., ["joy", "excitement"])

        Returns:
            True if successful, False otherwise
        """
        # Clamp values
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))

        try:
            return self._storage.update_episode_emotion(
                episode_id=episode_id,
                valence=valence,
                arousal=arousal,
                tags=tags,
            )
        except Exception as e:
            logger.warning(f"Failed to add emotional association: {e}", exc_info=True)
            return False

    def get_emotional_summary(self: "Kernle", days: int = 7) -> Dict[str, Any]:
        """Get emotional pattern summary over time period.

        Args:
            days: Number of days to analyze

        Returns:
            dict with:
            - average_valence: float
            - average_arousal: float
            - dominant_emotions: list[str]
            - emotional_trajectory: list - trend over time
            - episode_count: int - number of emotional episodes
        """
        # Get episodes with emotional data
        emotional_episodes = self._storage.get_emotional_episodes(days=days, limit=100)

        if not emotional_episodes:
            return {
                "average_valence": 0.0,
                "average_arousal": 0.0,
                "dominant_emotions": [],
                "emotional_trajectory": [],
                "episode_count": 0,
            }

        # Calculate averages
        valences = [ep.emotional_valence or 0.0 for ep in emotional_episodes]
        arousals = [ep.emotional_arousal or 0.0 for ep in emotional_episodes]

        avg_valence = sum(valences) / len(valences)
        avg_arousal = sum(arousals) / len(arousals)

        # Count emotion tags
        all_tags = []
        for ep in emotional_episodes:
            tags = ep.emotional_tags or []
            all_tags.extend(tags)

        tag_counts = Counter(all_tags)
        dominant_emotions = [tag for tag, count in tag_counts.most_common(5)]

        # Build trajectory (grouped by day)
        daily_data = defaultdict(lambda: {"valences": [], "arousals": []})

        for ep in emotional_episodes:
            if ep.created_at:
                date_str = ep.created_at.strftime("%Y-%m-%d")
                daily_data[date_str]["valences"].append(ep.emotional_valence or 0.0)
                daily_data[date_str]["arousals"].append(ep.emotional_arousal or 0.0)

        trajectory = []
        for date_str in sorted(daily_data.keys()):
            data = daily_data[date_str]
            trajectory.append(
                {
                    "date": date_str,
                    "valence": sum(data["valences"]) / len(data["valences"]),
                    "arousal": sum(data["arousals"]) / len(data["arousals"]),
                }
            )

        return {
            "average_valence": round(avg_valence, 3),
            "average_arousal": round(avg_arousal, 3),
            "dominant_emotions": dominant_emotions,
            "emotional_trajectory": trajectory,
            "episode_count": len(emotional_episodes),
        }

    def search_by_emotion(
        self: "Kernle",
        valence_range: Optional[tuple] = None,
        arousal_range: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find episodes matching emotional criteria.

        Args:
            valence_range: (min, max) valence filter, e.g. (0.5, 1.0) for positive
            arousal_range: (min, max) arousal filter, e.g. (0.7, 1.0) for high arousal
            tags: Emotional tags to match (matches any)
            limit: Maximum results

        Returns:
            List of matching episodes as dicts
        """
        episodes = self._storage.search_by_emotion(
            valence_range=valence_range,
            arousal_range=arousal_range,
            tags=tags,
            limit=limit,
        )

        return [
            {
                "id": ep.id,
                "objective": ep.objective,
                "outcome_type": ep.outcome_type,
                "outcome_description": ep.outcome,
                "emotional_valence": ep.emotional_valence,
                "emotional_arousal": ep.emotional_arousal,
                "emotional_tags": ep.emotional_tags,
                "created_at": ep.created_at.isoformat() if ep.created_at else "",
            }
            for ep in episodes
        ]

    def episode_with_emotion(
        self: "Kernle",
        objective: str,
        outcome: str,
        lessons: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        emotional_tags: Optional[List[str]] = None,
        auto_detect: bool = True,
        derived_from: Optional[List[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        source_type: Optional[str | SourceType] = None,
    ) -> str:
        """Record an episode with emotional tagging.

        Args:
            objective: What was the goal?
            outcome: What happened?
            lessons: Lessons learned
            tags: General tags
            valence: Emotional valence (-1.0 to 1.0), auto-detected if None
            arousal: Emotional arousal (0.0 to 1.0), auto-detected if None
            emotional_tags: Emotion labels, auto-detected if None
            auto_detect: If True and no emotion args given, detect from text
            derived_from: List of memory IDs this episode was derived from
            source: Source context (e.g., 'session with Sean', 'heartbeat check')
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering

        Returns:
            Episode ID
        """
        # Validate inputs
        objective = self._validate_string_input(objective, "objective", 1000)
        outcome = self._validate_string_input(outcome, "outcome", 1000)

        if lessons:
            lessons = [self._validate_string_input(lesson, "lesson", 500) for lesson in lessons]
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]
        if emotional_tags:
            emotional_tags = [
                self._validate_string_input(e, "emotion_tag", 50) for e in emotional_tags
            ]

        # Auto-detect emotions if not provided
        if auto_detect and valence is None and arousal is None and not emotional_tags:
            detection = self.detect_emotion(f"{objective} {outcome}")
            if detection["confidence"] > 0:
                valence = detection["valence"]
                arousal = detection["arousal"]
                emotional_tags = detection["tags"]

        episode_id = str(uuid.uuid4())

        # Determine outcome type using substring matching for flexibility
        outcome_lower = outcome.lower().strip()
        if any(
            word in outcome_lower
            for word in ("success", "done", "completed", "finished", "accomplished")
        ):
            outcome_type = "success"
        elif any(
            word in outcome_lower for word in ("fail", "error", "broke", "unable", "couldn't")
        ):
            outcome_type = "failure"
        else:
            outcome_type = "partial"

        resolved_source_type = _WritersMixin._normalize_source_type(source_type)

        # Keep explicit lineage and include source marker as annotation metadata.
        derived_from_value = list(derived_from) if derived_from else []
        if source:
            derived_from_value.append(f"context:{source}")

        episode = Episode(
            id=episode_id,
            stack_id=self.stack_id,
            objective=objective,
            outcome=outcome,
            outcome_type=outcome_type,
            lessons=lessons,
            tags=tags or ["manual"],
            created_at=datetime.now(timezone.utc),
            emotional_valence=valence or 0.0,
            emotional_arousal=arousal or 0.0,
            emotional_tags=emotional_tags,
            confidence=0.8,
            source_type=resolved_source_type.value,
            source_episodes=derived_from,  # Link to source memories
            derived_from=derived_from_value if derived_from_value else None,
            # Context/scope fields
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_episode(episode)
        return episode_id

    def get_mood_relevant_memories(
        self: "Kernle",
        current_valence: float,
        current_arousal: float,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to current emotional state.

        Useful for mood-congruent recall - we tend to remember
        experiences that match our current emotional state.

        Args:
            current_valence: Current valence (-1.0 to 1.0)
            current_arousal: Current arousal (0.0 to 1.0)
            limit: Maximum results

        Returns:
            List of mood-relevant episodes
        """
        # Get episodes with matching emotional range
        valence_range = (max(-1.0, current_valence - 0.3), min(1.0, current_valence + 0.3))
        arousal_range = (max(0.0, current_arousal - 0.3), min(1.0, current_arousal + 0.3))

        return self.search_by_emotion(
            valence_range=valence_range,
            arousal_range=arousal_range,
            limit=limit,
        )
