"""Tests for the promote (episodes → beliefs) feature."""

import tempfile
from pathlib import Path

import pytest

from kernle import Kernle
from kernle.storage.sqlite import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def kernle_instance():
    """Create an isolated Kernle instance with temp storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteStorage(stack_id="test-promote", db_path=db_path)
        k = Kernle(
            stack_id="test-promote",
            storage=storage,
            checkpoint_dir=Path(tmpdir),
            strict=False,
        )
        bind_noop_model(k)
        yield k


class TestPromoteBasic:
    """Test basic promote functionality."""

    def test_promote_no_episodes(self, kernle_instance):
        """Promote with no episodes returns empty."""
        result = kernle_instance.promote()
        assert result["episodes_scanned"] == 0
        assert result["patterns_found"] == 0
        assert result["suggestions"] == []
        assert result["beliefs_created"] == 0
        assert "message" in result

    def test_promote_few_episodes(self, kernle_instance):
        """Promote with fewer than min_episodes returns message."""
        kernle_instance.episode("task1", "done", lessons=["lesson A"])
        result = kernle_instance.promote(min_episodes=3)
        assert result["episodes_scanned"] == 1
        assert "Need at least 3" in result["message"]

    def test_promote_no_patterns(self, kernle_instance):
        """Episodes with unique lessons produce no patterns."""
        kernle_instance.episode("task1", "done", lessons=["unique lesson 1"])
        kernle_instance.episode("task2", "done", lessons=["unique lesson 2"])
        kernle_instance.episode("task3", "done", lessons=["unique lesson 3"])
        result = kernle_instance.promote()
        assert result["episodes_scanned"] == 3
        assert result["patterns_found"] == 0
        assert result["suggestions"] == []

    def test_promote_finds_patterns(self, kernle_instance):
        """Recurring lessons are detected as patterns."""
        kernle_instance.episode("task1", "done", lessons=["recurring lesson"])
        kernle_instance.episode("task2", "done", lessons=["recurring lesson"])
        kernle_instance.episode("task3", "done", lessons=["unique lesson"])
        result = kernle_instance.promote()
        assert result["patterns_found"] == 1
        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0]["lesson"] == "recurring lesson"
        assert result["suggestions"][0]["count"] == 2

    def test_promote_multiple_patterns(self, kernle_instance):
        """Multiple recurring patterns are detected and sorted by frequency."""
        kernle_instance.episode("t1", "done", lessons=["pattern A", "pattern B"])
        kernle_instance.episode("t2", "done", lessons=["pattern A", "pattern B"])
        kernle_instance.episode("t3", "done", lessons=["pattern A"])
        result = kernle_instance.promote()
        assert result["patterns_found"] == 2
        # pattern A appears 3x, pattern B appears 2x — sorted by frequency
        assert result["suggestions"][0]["lesson"] == "pattern A"
        assert result["suggestions"][0]["count"] == 3
        assert result["suggestions"][1]["lesson"] == "pattern B"
        assert result["suggestions"][1]["count"] == 2


class TestPromoteSuggestionMode:
    """Test default (suggestion-only) mode."""

    def test_suggestions_not_promoted(self, kernle_instance):
        """Default mode returns suggestions without creating beliefs."""
        kernle_instance.episode("t1", "done", lessons=["recurring"])
        kernle_instance.episode("t2", "done", lessons=["recurring"])
        kernle_instance.episode("t3", "done", lessons=["other"])
        result = kernle_instance.promote(auto=False)
        assert result["beliefs_created"] == 0
        assert result["suggestions"][0]["promoted"] is False
        assert result["suggestions"][0]["belief_id"] is None

        # Verify no beliefs were created
        beliefs = kernle_instance._storage.get_beliefs()
        assert len(beliefs) == 0

    def test_suggestions_include_source_episodes(self, kernle_instance):
        """Suggestions include source episode IDs."""
        ep1 = kernle_instance.episode("t1", "done", lessons=["pattern X"])
        ep2 = kernle_instance.episode("t2", "done", lessons=["pattern X"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        result = kernle_instance.promote()
        source_ids = result["suggestions"][0]["source_episodes"]
        assert ep1 in source_ids
        assert ep2 in source_ids


class TestPromoteAutoMode:
    """Test auto-promotion mode."""

    def test_auto_creates_beliefs(self, kernle_instance):
        """Auto mode creates beliefs from recurring patterns."""
        kernle_instance.episode("t1", "done", lessons=["auto pattern"])
        kernle_instance.episode("t2", "done", lessons=["auto pattern"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        result = kernle_instance.promote(auto=True)
        assert result["beliefs_created"] == 1
        assert result["suggestions"][0]["promoted"] is True
        assert result["suggestions"][0]["belief_id"] is not None

        # Verify belief was actually created
        beliefs = kernle_instance._storage.get_beliefs()
        created = [b for b in beliefs if b.statement == "auto pattern"]
        assert len(created) == 1
        assert created[0].confidence == 0.7
        assert created[0].belief_type == "pattern"
        assert created[0].source_type == "consolidation"  # "promotion" maps to consolidation

    def test_auto_sets_provenance(self, kernle_instance):
        """Auto-created beliefs have proper provenance."""
        ep1 = kernle_instance.episode("t1", "done", lessons=["prov test"])
        ep2 = kernle_instance.episode("t2", "done", lessons=["prov test"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        result = kernle_instance.promote(auto=True)
        belief_id = result["suggestions"][0]["belief_id"]

        beliefs = kernle_instance._storage.get_beliefs()
        belief = [b for b in beliefs if b.id == belief_id][0]
        # derived_from should include episode references
        assert any(f"episode:{ep1}" in ref for ref in belief.derived_from)
        assert any(f"episode:{ep2}" in ref for ref in belief.derived_from)

    def test_auto_custom_confidence(self, kernle_instance):
        """Auto mode respects custom confidence."""
        kernle_instance.episode("t1", "done", lessons=["conf test"])
        kernle_instance.episode("t2", "done", lessons=["conf test"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        result = kernle_instance.promote(auto=True, confidence=0.5)
        belief_id = result["suggestions"][0]["belief_id"]

        beliefs = kernle_instance._storage.get_beliefs()
        belief = [b for b in beliefs if b.id == belief_id][0]
        assert belief.confidence == 0.5

    def test_auto_skips_existing_beliefs(self, kernle_instance):
        """Auto mode skips patterns that match existing beliefs."""
        # Create a belief first
        kernle_instance.belief("already known")
        # Create episodes with matching lesson
        kernle_instance.episode("t1", "done", lessons=["already known"])
        kernle_instance.episode("t2", "done", lessons=["already known"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        result = kernle_instance.promote(auto=True)
        assert result["beliefs_created"] == 0
        assert result["suggestions"][0]["skipped"] == "similar_belief_exists"

    def test_auto_no_duplicates_within_run(self, kernle_instance):
        """Auto mode doesn't create duplicate beliefs within a single run."""
        # Same lesson appears in many episodes
        kernle_instance.episode("t1", "done", lessons=["dup test"])
        kernle_instance.episode("t2", "done", lessons=["dup test"])
        kernle_instance.episode("t3", "done", lessons=["dup test"])
        result = kernle_instance.promote(auto=True)
        # Should only create 1 belief, not 3
        assert result["beliefs_created"] == 1

        beliefs = kernle_instance._storage.get_beliefs()
        matching = [b for b in beliefs if b.statement == "dup test"]
        assert len(matching) == 1


class TestPromoteParameters:
    """Test parameter validation and edge cases."""

    def test_confidence_clamped(self, kernle_instance):
        """Confidence is clamped to 0.1-0.95."""
        kernle_instance.episode("t1", "done", lessons=["clamp test"])
        kernle_instance.episode("t2", "done", lessons=["clamp test"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        # Over max
        result = kernle_instance.promote(auto=True, confidence=1.5)
        belief_id = result["suggestions"][0]["belief_id"]
        beliefs = kernle_instance._storage.get_beliefs()
        belief = [b for b in beliefs if b.id == belief_id][0]
        assert belief.confidence == 0.95

    def test_min_occurrences_respected(self, kernle_instance):
        """Higher min_occurrences threshold filters out less common patterns."""
        kernle_instance.episode("t1", "done", lessons=["twice"])
        kernle_instance.episode("t2", "done", lessons=["twice"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        # Require 3 occurrences
        result = kernle_instance.promote(min_occurrences=3)
        assert result["patterns_found"] == 0

    def test_empty_lessons_ignored(self, kernle_instance):
        """Episodes with no lessons don't cause issues."""
        kernle_instance.episode("t1", "done")  # no lessons
        kernle_instance.episode("t2", "done", lessons=["something"])
        kernle_instance.episode("t3", "done", lessons=["something"])
        result = kernle_instance.promote()
        assert result["episodes_scanned"] == 3
        assert result["patterns_found"] == 1

    def test_whitespace_lessons_ignored(self, kernle_instance):
        """Whitespace-only lessons are skipped."""
        kernle_instance.episode("t1", "done", lessons=["  ", "real lesson"])
        kernle_instance.episode("t2", "done", lessons=["real lesson"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        result = kernle_instance.promote()
        assert result["patterns_found"] == 1
        assert result["suggestions"][0]["lesson"] == "real lesson"

    def test_forgotten_episodes_excluded(self, kernle_instance):
        """Forgotten episodes are not included in promotion."""
        ep1 = kernle_instance.episode("t1", "done", lessons=["forgotten pattern"])
        kernle_instance.episode("t2", "done", lessons=["forgotten pattern"])
        kernle_instance.episode("t3", "done", lessons=["filler"])
        # Forget one of the episodes
        kernle_instance.forget("episode", ep1, reason="test")
        result = kernle_instance.promote()
        # Only 1 occurrence of "forgotten pattern" remains — below threshold
        assert result["patterns_found"] == 0
