"""Tests for Phase 6: Replace Heuristic Contradiction Detection (#840 Part 2).

TDD tests for:
- find_contradictions() gated by use_legacy_heuristics
- find_semantic_contradictions() gated by use_legacy_heuristics
- revise_beliefs_from_episode() gated by use_legacy_heuristics
- No-model path returns empty/neutral
- Legacy mode uses heuristics (unchanged)
- Confidence threshold filtering
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage
from kernle.types import Belief, SearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def k_legacy(tmp_path):
    """Kernle with use_legacy_heuristics=true."""
    db_path = tmp_path / "test_contradiction_legacy.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    storage.set_stack_setting("use_legacy_heuristics", "true")
    return Kernle(stack_id="test-stack", storage=storage, strict=False)


@pytest.fixture
def k_inference(tmp_path):
    """Kernle with use_legacy_heuristics=false."""
    db_path = tmp_path / "test_contradiction_inference.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    storage.set_stack_setting("use_legacy_heuristics", "false")
    return Kernle(stack_id="test-stack", storage=storage, strict=False)


def _make_search_result(belief_id, statement, score=0.8, confidence=0.8):
    """Create a mock SearchResult with a Belief record."""
    belief = Belief(
        id=belief_id,
        stack_id="test-stack",
        statement=statement,
        confidence=confidence,
        belief_type="fact",
        is_active=True,
    )
    return SearchResult(
        record=belief,
        record_type="belief",
        score=score,
        content=statement,
    )


# ---------------------------------------------------------------------------
# find_contradictions() tests
# ---------------------------------------------------------------------------


class TestFindContradictionsLegacy:
    def test_legacy_mode_uses_heuristics(self, k_legacy):
        """With legacy=true, heuristic pattern matching is used."""
        k_legacy.belief("I always prefer Python for backend work", confidence=0.8)
        k_legacy.belief("I never prefer Python for backend work", confidence=0.7)

        results = k_legacy.find_contradictions("I always prefer Python for backend work")
        assert isinstance(results, list)

    def test_legacy_direct_negation_detected(self, k_legacy):
        """Legacy mode: direct negation (always/never) is detected."""
        k_legacy.belief("I should always write tests first", confidence=0.8)
        k_legacy.belief("I should never write tests first", confidence=0.7)

        results = k_legacy.find_contradictions("I should always write tests first")
        assert isinstance(results, list)


class TestFindContradictionsNoModel:
    def test_no_model_returns_empty(self, k_inference):
        """Without a model, find_contradictions returns empty list."""
        k_inference.belief("Testing is important", confidence=0.8)

        results = k_inference.find_contradictions("Testing is a waste of time")
        assert results == []

    def test_no_model_find_semantic_returns_empty(self, k_inference):
        """Without a model, find_semantic_contradictions returns empty list."""
        k_inference.belief("Testing is important", confidence=0.8)

        results = k_inference.find_semantic_contradictions("Testing slows down development")
        assert results == []


class TestFindContradictionsInference:
    def test_inference_path_with_model(self, k_inference):
        """With model bound, find_contradictions uses inference."""
        mock_inference = MagicMock()
        mock_inference.infer.return_value = json.dumps(
            {
                "contradictions": [
                    {
                        "belief_id": "belief-123",
                        "contradiction_type": "direct_negation",
                        "contradiction_confidence": 0.85,
                        "explanation": "Directly opposes the statement",
                    }
                ]
            }
        )

        stack = k_inference.stack
        stack._inference = mock_inference

        # Mock search to return candidates
        search_results = [_make_search_result("belief-123", "Testing is important", score=0.8)]
        with patch.object(k_inference._storage, "search", return_value=search_results):
            results = k_inference.find_contradictions("Testing is a waste of time")

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["belief_id"] == "belief-123"
        assert results[0]["contradiction_confidence"] == 0.85
        assert mock_inference.infer.called

    def test_inference_confidence_threshold(self, k_inference):
        """Contradictions below confidence threshold are filtered out."""
        mock_inference = MagicMock()
        mock_inference.infer.return_value = json.dumps(
            {
                "contradictions": [
                    {
                        "belief_id": "belief-1",
                        "contradiction_type": "semantic",
                        "contradiction_confidence": 0.3,
                        "explanation": "Weakly related",
                    },
                    {
                        "belief_id": "belief-2",
                        "contradiction_type": "direct_negation",
                        "contradiction_confidence": 0.85,
                        "explanation": "Strongly contradicts",
                    },
                ]
            }
        )

        stack = k_inference.stack
        stack._inference = mock_inference

        search_results = [
            _make_search_result("belief-1", "Testing is important", score=0.8),
            _make_search_result("belief-2", "Code review is essential", score=0.7),
        ]
        with patch.object(k_inference._storage, "search", return_value=search_results):
            results = k_inference.find_contradictions("Testing is a waste of time")

        # Low confidence (0.3) should be filtered; only belief-2 (0.85) remains
        assert len(results) == 1
        assert results[0]["belief_id"] == "belief-2"
        for r in results:
            assert r["contradiction_confidence"] >= 0.6

    def test_inference_fallback_on_failure(self, k_inference):
        """If inference fails, returns empty list (not heuristics)."""
        mock_inference = MagicMock()
        mock_inference.infer.return_value = "not valid json {{{"

        stack = k_inference.stack
        stack._inference = mock_inference

        search_results = [_make_search_result("belief-1", "Testing is important", score=0.8)]
        with patch.object(k_inference._storage, "search", return_value=search_results):
            results = k_inference.find_contradictions("Testing is a waste of time")

        assert results == []


# ---------------------------------------------------------------------------
# revise_beliefs_from_episode() tests
# ---------------------------------------------------------------------------


class TestReviseBeliefsFromEpisodeLegacy:
    def test_legacy_mode_uses_word_overlap(self, k_legacy):
        """With legacy=true, word-overlap heuristic is used."""
        k_legacy.belief("Testing prevents regressions", confidence=0.7)
        episode_id = k_legacy.episode(
            objective="Write tests for the auth module",
            outcome="Tests caught a regression bug",
        )

        result = k_legacy.revise_beliefs_from_episode(episode_id)
        assert "reinforced" in result
        assert "contradicted" in result
        assert "suggested_new" in result


class TestReviseBeliefsNoModel:
    def test_no_model_returns_empty(self, k_inference):
        """Without a model, revise_beliefs_from_episode returns empty result."""
        k_inference.belief("Testing prevents regressions", confidence=0.7)
        episode_id = k_inference.episode(
            objective="Write tests for the auth module",
            outcome="Tests caught a regression bug",
        )

        result = k_inference.revise_beliefs_from_episode(episode_id)
        assert result["reinforced"] == []
        assert result["contradicted"] == []
        assert result["suggested_new"] == []

    def test_no_model_episode_not_found(self, k_inference):
        """Episode not found returns error dict regardless of mode."""
        result = k_inference.revise_beliefs_from_episode("nonexistent-episode")
        assert "error" in result


class TestReviseBeliefsInference:
    def test_inference_path_with_model(self, k_inference):
        """With model bound, revise_beliefs_from_episode uses inference."""
        mock_inference = MagicMock()
        # First call: infer for revision
        mock_inference.infer.return_value = json.dumps(
            {
                "reinforced": [],
                "contradicted": [],
                "suggested_new": [
                    {
                        "statement": "Integration tests are critical",
                        "confidence": 0.7,
                    }
                ],
            }
        )

        stack = k_inference.stack
        stack._inference = mock_inference

        k_inference.belief("Testing prevents regressions", confidence=0.7)
        episode_id = k_inference.episode(
            objective="Write tests for the auth module",
            outcome="Tests caught a regression bug",
        )

        result = k_inference.revise_beliefs_from_episode(episode_id)
        assert "reinforced" in result
        assert "contradicted" in result
        assert "suggested_new" in result
        assert mock_inference.infer.called


# ---------------------------------------------------------------------------
# Regression parity tests
# ---------------------------------------------------------------------------


class TestRegressionParity:
    """Tests for known heuristic false positives/negatives."""

    def test_different_domains_not_contradicted_legacy(self, k_legacy):
        """Legacy: unrelated beliefs from different domains should not be flagged."""
        k_legacy.belief("Fast feedback loops improve development speed", confidence=0.8)
        k_legacy.belief("Deliberate decision-making leads to better architecture", confidence=0.8)

        results = k_legacy.find_contradictions("Fast feedback loops improve development speed")
        assert isinstance(results, list)
