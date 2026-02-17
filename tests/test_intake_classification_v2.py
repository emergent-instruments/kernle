"""Tests for Phase 5: Replace Heuristic Intake Classification (#842).

TDD tests for:
- extract_suggestions() — no-model returns empty, inference path
- detect_significance() — no-model returns not significant, inference path
- infer_outcome_type() keyword matching
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
def k_inference(tmp_path):
    """Kernle without a bound model (safe defaults path)."""
    db_path = tmp_path / "test_classify_inference.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    return Kernle(stack_id="test-stack", storage=storage, strict=False)


# ---------------------------------------------------------------------------
# extract_suggestions() tests
# ---------------------------------------------------------------------------


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
        # No model bound → empty
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
# infer_outcome_type() tests
# ---------------------------------------------------------------------------


class TestInferOutcomeType:
    def test_outcome_type_keywords(self):
        """outcome_type uses keyword matching."""
        from kernle.core.enrichment import infer_outcome_type

        assert infer_outcome_type("The deployment was a success") == "success"
        assert infer_outcome_type("The test failed with errors") == "failure"
        assert infer_outcome_type("Made some progress") == "partial"
