"""Emotional tagging stack component.

Provides automatic emotion detection from text and emotional pattern
analysis. Can enhance memories with emotional metadata on save.
When inference is available, uses the model for richer detection;
falls back to keyword-based detection otherwise.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

# Emotional signal patterns for automatic tagging
EMOTION_PATTERNS = {
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


class EmotionalTaggingComponent:
    """Emotional tagging component.

    Detects emotions in text using keyword patterns. During on_save,
    can annotate episodes with detected emotional valence/arousal.
    When inference is available, uses the model for richer detection;
    falls back to keyword-based detection when inference is unavailable
    or returns invalid data.
    """

    name = "emotions"
    version = "1.0.0"
    required = False
    needs_inference = True
    inference_scope = "fast"
    priority = 100

    def __init__(self) -> None:
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._storage: Optional[Any] = None

    def attach(self, stack_id: str, inference: Optional[InferenceService] = None) -> None:
        self._stack_id = stack_id
        self._inference = inference

    def detach(self) -> None:
        self._stack_id = None
        self._inference = None
        self._storage = None

    def set_inference(self, inference: Optional[InferenceService]) -> None:
        self._inference = inference

    def set_storage(self, storage: Any) -> None:
        """Called by SQLiteStack after attach to provide storage access."""
        self._storage = storage

    # ---- Lifecycle Hooks ----

    def on_save(self, memory_type: str, memory_id: str, memory: Any) -> Any:
        """Detect emotions on episode save and annotate if possible."""
        if memory_type != "episode":
            return None

        text = ""
        objective = getattr(memory, "objective", "")
        outcome = getattr(memory, "outcome", "")
        if objective:
            text += objective
        if outcome:
            text += " " + outcome

        if not text.strip():
            return None

        detection = self.detect_emotion(text)
        if detection["confidence"] > 0:
            # Return detected emotional metadata; the stack can use this
            return {
                "emotional_valence": detection["valence"],
                "emotional_arousal": detection["arousal"],
                "emotional_tags": detection["tags"],
            }
        return None

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        return results

    def on_load(self, context: Dict[str, Any]) -> None:
        pass

    def on_maintenance(self) -> Dict[str, Any]:
        """Report emotional summary during maintenance."""
        if self._storage is None:
            logger.debug("EmotionalTaggingComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        # Get recent emotional episodes
        episodes = self._storage.get_episodes(limit=50)
        valences = []
        arousals = []
        tag_counts: Counter = Counter()

        for ep in episodes:
            v = getattr(ep, "emotional_valence", None)
            a = getattr(ep, "emotional_arousal", None)
            if v is not None and v != 0.0:
                valences.append(v)
            if a is not None and a != 0.0:
                arousals.append(a)
            tags = getattr(ep, "emotional_tags", None) or []
            tag_counts.update(tags)

        return {
            "episodes_with_emotions": len(valences),
            "avg_valence": round(sum(valences) / len(valences), 3) if valences else 0.0,
            "avg_arousal": round(sum(arousals) / len(arousals), 3) if arousals else 0.0,
            "dominant_emotions": [tag for tag, _ in tag_counts.most_common(3)],
        }

    # ---- Core Logic ----

    def _detect_emotion_via_inference(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt emotion detection using inference model.

        Returns a detection dict on success, or None if inference is
        unavailable or returns invalid data.
        """
        from kernle.core.inference_utils import parse_inference_json

        if self._inference is None:
            return None

        try:
            response = self._inference.infer(
                prompt=(
                    "Analyze the emotional content of this experience and return JSON:\n\n"
                    f"{text}\n\n"
                    'Return: {"valence": float (-1.0 to 1.0), '
                    '"arousal": float (0.0 to 1.0), '
                    '"emotions": [list of emotion strings]}'
                ),
                system="You are an emotion analysis system. Return only valid JSON.",
            )
        except Exception:
            logger.debug(
                "EmotionalTaggingComponent: inference call failed",
                exc_info=True,
            )
            return None

        result = parse_inference_json(
            response,
            required_fields=["valence", "arousal"],
            fallback={"valence": 0.0, "arousal": 0.0, "emotions": []},
            logger=logger,
        )

        if result.fallback_used:
            return None

        valence = max(-1.0, min(1.0, float(result.data.get("valence", 0.0))))
        arousal = max(0.0, min(1.0, float(result.data.get("arousal", 0.0))))
        emotions = result.data.get("emotions", [])
        if not isinstance(emotions, list):
            return None
        emotions = [str(e) for e in emotions if isinstance(e, str)]

        return {
            "valence": valence,
            "arousal": arousal,
            "tags": emotions,
            "confidence": 0.9,
        }

    _NEUTRAL_EMOTION = {"valence": 0.0, "arousal": 0.0, "tags": [], "confidence": 0.0}

    def _use_legacy_heuristics(self) -> bool:
        """Check if legacy heuristics mode is enabled."""
        if self._storage is None:
            return True
        setting = self._storage.get_stack_setting("use_legacy_heuristics")
        if setting is None:
            return True
        return setting.lower() == "true"

    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """Detect emotional signals in text.

        Gating:
        - ``use_legacy_heuristics=true``: keyword-based detection (unchanged)
        - ``use_legacy_heuristics=false`` + no model: neutral defaults
        - ``use_legacy_heuristics=false`` + model: inference-based detection
        """
        if not text:
            return dict(self._NEUTRAL_EMOTION)

        if self._use_legacy_heuristics():
            # Legacy: try inference first, fall back to keywords
            inference_result = self._detect_emotion_via_inference(text)
            if inference_result is not None:
                return inference_result
            return self._detect_emotion_keywords(text)

        # Non-legacy: inference or neutral defaults (no keyword fallback)
        inference_result = self._detect_emotion_via_inference(text)
        if inference_result is not None:
            return inference_result
        return dict(self._NEUTRAL_EMOTION)

    def _detect_emotion_keywords(self, text: str) -> Dict[str, Any]:
        """Detect emotional signals using keyword patterns."""
        text_lower = text.lower()
        detected_emotions = []
        valence_sum = 0.0
        arousal_sum = 0.0

        for emotion_name, pattern in EMOTION_PATTERNS.items():
            for keyword in pattern["keywords"]:
                if keyword in text_lower:
                    detected_emotions.append(emotion_name)
                    valence_sum += pattern["valence"]
                    arousal_sum += pattern["arousal"]
                    break

        if detected_emotions:
            count = len(detected_emotions)
            avg_valence = max(-1.0, min(1.0, valence_sum / count))
            avg_arousal = max(0.0, min(1.0, arousal_sum / count))
            confidence = min(1.0, 0.3 + (count * 0.2))
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
