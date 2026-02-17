"""Tests for Phase 4: Replace Heuristic Emotion Detection (#841).

TDD tests for:
- No-model path returns neutral defaults
- Inference path for emotion detection
- episode_with_emotion auto-detection with gating
"""

from unittest.mock import MagicMock

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def k_inference(tmp_path):
    """Create a Kernle without a bound model (safe defaults path)."""
    db_path = tmp_path / "test_emotion_inference.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    return Kernle(stack_id="test-stack", storage=storage, strict=False)


# ---------------------------------------------------------------------------
# No-model path — neutral defaults
# ---------------------------------------------------------------------------


class TestNoModelReturnsNeutral:
    def test_no_model_returns_neutral(self, k_inference):
        """Without a bound model, detect_emotion returns neutral defaults."""
        # k_inference has use_legacy_heuristics=false but no model bound
        result = k_inference.detect_emotion("I'm so happy!")
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["tags"] == []
        assert result["confidence"] == 0.0

    def test_full_episode_save_no_model(self, k_inference):
        """Episode saved without model: valence=0, arousal=0, no error."""
        episode_id = k_inference.episode_with_emotion(
            objective="Test something",
            outcome="It worked",
        )
        assert episode_id is not None

        # Episode should be saved with neutral emotion values
        episode = k_inference._storage.get_episode(episode_id)
        assert episode is not None
        assert episode.emotional_valence == 0.0
        assert episode.emotional_arousal == 0.0


# ---------------------------------------------------------------------------
# Inference path tests (mocked inference)
# ---------------------------------------------------------------------------


class TestInferenceEmotionDetection:
    def test_inference_path_with_model(self, k_inference):
        """With model bound, detect_emotion uses inference."""
        import json

        mock_inference = MagicMock()
        mock_inference.infer.return_value = json.dumps(
            {"valence": 0.7, "arousal": 0.5, "emotions": ["joy", "satisfaction"]}
        )

        # Bind mock inference to the stack
        stack = k_inference.stack
        stack._inference = mock_inference

        result = k_inference.detect_emotion("I'm really happy about this outcome!")
        assert result["valence"] == 0.7
        assert result["arousal"] == 0.5
        assert "joy" in result["tags"]
        assert result["confidence"] > 0

    def test_inference_fallback_on_failure(self, k_inference):
        """If inference fails, returns neutral defaults (not keywords)."""
        mock_inference = MagicMock()
        mock_inference.infer.return_value = "not valid json {{{"

        stack = k_inference.stack
        stack._inference = mock_inference

        result = k_inference.detect_emotion("I'm frustrated")
        # Should return neutral (fallback), not keyword-based result
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["confidence"] == 0.0

    def test_inference_missing_fields_returns_neutral(self, k_inference):
        """Inference response missing required fields returns neutral."""
        import json

        mock_inference = MagicMock()
        mock_inference.infer.return_value = json.dumps({"valence": 0.5})  # Missing arousal

        stack = k_inference.stack
        stack._inference = mock_inference

        result = k_inference.detect_emotion("Some text")
        assert result["valence"] == 0.0
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Component integration
# ---------------------------------------------------------------------------


class TestComponentIntegration:
    def test_component_uses_parse_inference_json(self, tmp_path):
        """EmotionalTaggingComponent uses parse_inference_json for validation."""
        import json

        from kernle.stack.components.emotions import EmotionalTaggingComponent

        component = EmotionalTaggingComponent()
        mock_inference = MagicMock()
        mock_inference.infer.return_value = json.dumps(
            {"valence": 0.8, "arousal": 0.6, "emotions": ["joy"]}
        )
        component._inference = mock_inference

        result = component.detect_emotion("I'm very happy!")
        assert result["valence"] == 0.8
        assert result["arousal"] == 0.6
        assert "joy" in result["tags"]
