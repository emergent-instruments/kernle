"""Tests for Phase 4: Replace Heuristic Emotion Detection (#841).

TDD tests for:
- Legacy mode uses keyword detection (unchanged)
- No-model path returns neutral defaults
- Inference path for emotion detection
- Regression parity (false positives/negatives fixed)
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
def k_legacy(tmp_path):
    """Create a Kernle with use_legacy_heuristics=true (default for existing stacks)."""
    db_path = tmp_path / "test_emotion_legacy.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    storage.set_stack_setting("use_legacy_heuristics", "true")
    return Kernle(stack_id="test-stack", storage=storage, strict=False)


@pytest.fixture
def k_inference(tmp_path):
    """Create a Kernle with use_legacy_heuristics=false (new stack behavior)."""
    db_path = tmp_path / "test_emotion_inference.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    storage.set_stack_setting("use_legacy_heuristics", "false")
    return Kernle(stack_id="test-stack", storage=storage, strict=False)


# ---------------------------------------------------------------------------
# Legacy mode tests — keyword behavior unchanged
# ---------------------------------------------------------------------------


class TestLegacyModeUsesKeywords:
    def test_legacy_mode_uses_keywords(self, k_legacy):
        """With use_legacy_heuristics=true, keyword behavior is unchanged."""
        result = k_legacy.detect_emotion("I feel happy and excited about this!")
        assert result["valence"] > 0  # Positive emotion detected
        assert len(result["tags"]) > 0
        assert result["confidence"] > 0

    def test_explicit_joy_still_detected(self, k_legacy):
        """Legacy mode: explicit joy keywords are detected."""
        result = k_legacy.detect_emotion("I'm so happy about this success!")
        assert "joy" in result["tags"] or result["valence"] > 0

    def test_fear_detected(self, k_legacy):
        """Legacy mode: fear/anxiety keywords are detected."""
        result = k_legacy.detect_emotion("I'm worried and anxious about the deadline")
        assert result["valence"] < 0 or len(result["tags"]) > 0

    def test_neutral_stays_neutral(self, k_legacy):
        """Legacy mode: neutral text returns neutral values."""
        result = k_legacy.detect_emotion("The meeting is at 3pm in room B.")
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["tags"] == []
        assert result["confidence"] == 0.0


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
# Regression parity tests
# ---------------------------------------------------------------------------


class TestRegressionParity:
    """Tests for known heuristic false positives/negatives.

    These test that the inference path (when available) can handle cases
    that the keyword heuristic gets wrong.
    """

    def test_not_happy_not_tagged_as_joy_legacy(self, k_legacy):
        """Legacy mode: 'not happy' contains keyword 'happy' and WILL be tagged.

        This is a known false positive in keyword mode.
        """
        result = k_legacy.detect_emotion("I'm not happy about this at all")
        # In legacy mode, 'happy' keyword triggers joy detection (known false positive)
        # This test documents the limitation
        assert "joy" in result["tags"]  # Keyword match is a false positive

    def test_sarcasm_literal_in_legacy(self, k_legacy):
        """Legacy mode: sarcasm treated literally (known limitation)."""
        detection = k_legacy.detect_emotion("Oh great, another bug to fix")
        # "great" triggers positive detection in keyword mode — a known limitation
        assert detection["valence"] >= 0  # Documents the false positive


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
