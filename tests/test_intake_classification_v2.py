"""Tests for Phase 5: Replace Heuristic Intake Classification (#842).

TDD tests for:
- extract_suggestions() gated by use_legacy_heuristics
- detect_significance() gated by use_legacy_heuristics
- infer_outcome_type() gated by use_legacy_heuristics
- No-model path returns empty/neutral
- Legacy mode uses patterns (unchanged)
"""

import json
from unittest.mock import MagicMock

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def k_legacy(tmp_path):
    """Kernle with use_legacy_heuristics=true."""
    db_path = tmp_path / "test_classify_legacy.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    storage.set_stack_setting("use_legacy_heuristics", "true")
    return Kernle(stack_id="test-stack", storage=storage, strict=False)


@pytest.fixture
def k_inference(tmp_path):
    """Kernle with use_legacy_heuristics=false."""
    db_path = tmp_path / "test_classify_inference.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    storage.set_stack_setting("use_legacy_heuristics", "false")
    return Kernle(stack_id="test-stack", storage=storage, strict=False)


# ---------------------------------------------------------------------------
# extract_suggestions() tests
# ---------------------------------------------------------------------------


class TestExtractSuggestionsLegacy:
    def test_legacy_mode_uses_patterns(self, k_legacy):
        """With legacy=true, pattern-based extraction works as before."""
        from kernle.types import RawEntry

        raw = RawEntry(
            id="raw-1",
            stack_id="test-stack",
            blob="I completed the task and learned that testing is essential",
            source="cli",
        )
        suggestions = k_legacy.extract_suggestions(raw, auto_save=False)
        # Pattern-based: "completed" and "learned" should trigger suggestions
        assert len(suggestions) >= 1

    def test_completed_task_is_episode(self, k_legacy):
        """Legacy mode: episode keywords trigger episode suggestion."""
        from kernle.types import RawEntry

        # Need multiple pattern groups to exceed 0.4 threshold:
        # "completed" (0.7) + "success" (0.7) + "learned" (0.6) = strong signal
        raw = RawEntry(
            id="raw-2",
            stack_id="test-stack",
            blob="I completed the migration, it was a success, and learned a lot",
            source="cli",
        )
        suggestions = k_legacy.extract_suggestions(raw, auto_save=False)
        episode_suggestions = [s for s in suggestions if s.memory_type == "episode"]
        assert len(episode_suggestions) >= 1

    def test_believe_statement_is_belief(self, k_legacy):
        """Legacy mode: 'I believe' triggers belief suggestion."""
        from kernle.types import RawEntry

        raw = RawEntry(
            id="raw-3",
            stack_id="test-stack",
            blob="I believe that automated testing is the best way to ensure quality",
            source="cli",
        )
        suggestions = k_legacy.extract_suggestions(raw, auto_save=False)
        belief_suggestions = [s for s in suggestions if s.memory_type == "belief"]
        assert len(belief_suggestions) >= 1


class TestExtractSuggestionsNoModel:
    def test_no_model_returns_empty(self, k_inference):
        """Without a model, extract_suggestions returns empty list."""
        from kernle.types import RawEntry

        raw = RawEntry(
            id="raw-4",
            stack_id="test-stack",
            blob="I completed a major task, it was a success, and I learned valuable lessons",
            source="cli",
        )
        # In legacy mode this would produce suggestions, but with legacy=false + no model → empty
        suggestions = k_inference.extract_suggestions(raw, auto_save=False)
        assert suggestions == []

    def test_full_raw_to_suggestion_pipeline_no_model(self, k_inference):
        """Full pipeline: raw entries stored, no suggestions, no error."""
        raw_id = k_inference._storage.save_raw(
            "I believe testing is essential. I completed the feature successfully",
            source="cli",
        )
        assert raw_id is not None

        raw_entries = k_inference._storage.list_raw(limit=10)
        assert len(raw_entries) >= 1

        suggestions = k_inference.extract_suggestions(raw_entries[0], auto_save=False)
        assert suggestions == []


class TestExtractSuggestionsInference:
    def test_inference_path_with_model(self, k_inference):
        """With model bound, extract_suggestions uses inference."""
        from kernle.types import RawEntry

        mock_inference = MagicMock()
        mock_inference.infer.return_value = json.dumps(
            {
                "suggestions": [
                    {
                        "memory_type": "episode",
                        "confidence": 0.9,
                        "reason": "Describes a completed task with outcome",
                    }
                ]
            }
        )

        stack = k_inference.stack
        stack._inference = mock_inference

        raw = RawEntry(
            id="raw-5",
            stack_id="test-stack",
            blob="I finished the migration project with great results",
            source="cli",
        )
        suggestions = k_inference.extract_suggestions(raw, auto_save=False)
        assert len(suggestions) >= 1
        assert suggestions[0].memory_type == "episode"


# ---------------------------------------------------------------------------
# detect_significance() tests
# ---------------------------------------------------------------------------


class TestDetectSignificanceLegacy:
    def test_legacy_mode_uses_keywords(self, k_legacy):
        """Legacy mode: keyword-based significance detection."""
        result = k_legacy.detect_significance("I completed the migration successfully")
        assert result["significant"] is True
        assert result["score"] > 0

    def test_legacy_neutral_text_not_significant(self, k_legacy):
        """Legacy mode: neutral text is not significant."""
        result = k_legacy.detect_significance("The meeting is at 3pm")
        assert result["significant"] is False


class TestDetectSignificanceNoModel:
    def test_no_model_returns_not_significant(self, k_inference):
        """Without a model, detect_significance returns not significant."""
        result = k_inference.detect_significance("I completed a major task")
        assert result["significant"] is False
        assert result["score"] == 0.0
        assert result["signals"] == []


class TestDetectSignificanceInference:
    def test_inference_path_with_model(self, k_inference):
        """With model bound, detect_significance uses inference."""
        mock_inference = MagicMock()
        mock_inference.infer.return_value = json.dumps(
            {
                "significant": True,
                "score": 0.85,
                "signals": [{"signal": "lesson", "type": "lesson", "weight": 0.9}],
            }
        )

        stack = k_inference.stack
        stack._inference = mock_inference

        result = k_inference.detect_significance("I learned that testing prevents regressions")
        assert result["significant"] is True
        assert result["score"] > 0


# ---------------------------------------------------------------------------
# infer_outcome_type() gating tests
# ---------------------------------------------------------------------------


class TestInferOutcomeType:
    def test_legacy_outcome_type_keywords(self, k_legacy):
        """Legacy mode: outcome_type uses keyword inference."""
        from kernle.core.enrichment import infer_outcome_type

        assert infer_outcome_type("The deployment was a success") == "success"
        assert infer_outcome_type("The test failed with errors") == "failure"
        assert infer_outcome_type("Made some progress") == "partial"
