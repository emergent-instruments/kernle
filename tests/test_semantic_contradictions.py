"""Tests for semantic contradiction detection.

Tests the find_semantic_contradictions method which uses embeddings
to find beliefs that are semantically similar but may contradict.
"""

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def kernle_instance(tmp_path):
    """Create a Kernle instance with SQLite storage."""
    db_path = tmp_path / "test_semantic.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    k = Kernle(stack_id="test_agent", storage=storage, strict=False)
    bind_noop_model(k)
    return k


@pytest.fixture
def kernle_with_beliefs(tmp_path):
    """Create a Kernle instance with pre-populated beliefs."""
    db_path = tmp_path / "test_semantic_beliefs.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    k = Kernle(stack_id="test_agent", storage=storage, strict=False)
    bind_noop_model(k)

    # Add beliefs covering various contradiction scenarios
    k.belief("Testing is essential for code quality", confidence=0.9)
    k.belief("I prefer Python for data science work", confidence=0.85)
    k.belief("Code reviews improve team knowledge sharing", confidence=0.8)
    k.belief("Documentation should be written alongside code", confidence=0.75)
    k.belief("Fast iteration is important for product development", confidence=0.8)

    return k


class TestFindSemanticContradictions:
    """Tests for find_semantic_contradictions — returns [] without inference."""

    def test_no_inference_returns_empty(self, kernle_instance):
        """Without inference, find_semantic_contradictions returns empty list."""
        k = kernle_instance
        k.belief("testing is unnecessary", confidence=0.6)

        contradictions = k.find_semantic_contradictions(
            "testing is essential",
            similarity_threshold=0.0,
        )
        assert contradictions == []

    def test_empty_database(self, kernle_instance):
        """Should return empty list for empty database."""
        k = kernle_instance
        contradictions = k.find_semantic_contradictions(
            "Testing is important", similarity_threshold=0.1
        )
        assert contradictions == []


class TestIntegrationWithExistingContradictions:
    """Tests ensuring both methods return safe defaults without inference."""

    def test_both_methods_return_empty_without_inference(self, kernle_with_beliefs):
        """Both methods return empty lists without inference bound."""
        k = kernle_with_beliefs
        k.belief("Testing is never worth the effort", confidence=0.5)

        old_contradictions = k.find_contradictions("Testing is always worth the effort")
        new_contradictions = k.find_semantic_contradictions(
            "Testing is valuable and important", similarity_threshold=0.1
        )

        assert old_contradictions == []
        assert new_contradictions == []


class TestTokenizedSearchFallback:
    """Regression tests for tokenized non-vec search fallback (#214)."""

    def test_tokenized_belief_search_matches_shared_words(self, kernle_instance):
        """Searching 'never validate input' should match belief 'always validate user input'."""
        k = kernle_instance
        k.storage._has_vec = False

        k.belief("always validate user input", confidence=0.9)

        # Multi-word query shares "validate" and "input" with the belief
        results = k.storage.search("never validate input", record_types=["belief"])
        assert len(results) >= 1
        assert any("validate" in r.record.statement for r in results)

    def test_tokenized_search_skips_short_words(self, kernle_instance):
        """Short words (< 3 chars) like 'is', 'to', 'an' should be skipped."""
        k = kernle_instance
        k.storage._has_vec = False

        k.belief("AI is transforming industries", confidence=0.8)

        # "is" and "an" are too short, but "transforming" should match
        results = k.storage.search("an transforming is", record_types=["belief"])
        assert len(results) >= 1

    def test_tokenized_search_all_short_words_uses_full_phrase(self, kernle_instance):
        """If all words are < 3 chars, fall back to full-phrase match."""
        k = kernle_instance
        k.storage._has_vec = False

        k.belief("go do it", confidence=0.7)

        # All words < 3 chars, should use full phrase "go do it"
        results = k.storage.search("go do it", record_types=["belief"])
        assert len(results) >= 1

    def test_tokenized_search_scores_by_token_coverage(self, kernle_instance):
        """Results matching more query tokens should score higher."""
        k = kernle_instance
        k.storage._has_vec = False

        # This belief matches "validate" and "input" (2 tokens)
        k.belief("always validate user input before processing", confidence=0.9)
        # This belief only matches "validate" (1 token)
        k.belief("always validate configuration files", confidence=0.8)

        results = k.storage.search("never validate input", record_types=["belief"])
        assert len(results) == 2
        # First result should match more tokens (higher score)
        assert "input" in results[0].record.statement


class TestFindContradictionsThreshold:
    """Without inference, find_contradictions returns [] regardless of threshold."""

    def test_no_inference_returns_empty(self, kernle_instance):
        """Without inference, find_contradictions returns empty list."""
        k = kernle_instance
        k.belief("Testing should never be done on code", confidence=0.8)

        results = k.find_contradictions(
            "Testing should always be done on code",
            similarity_threshold=0.1,
        )
        assert results == []
