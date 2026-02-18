"""Tests for belief revision functionality.

Tests the belief revision system including:
- Finding contradictions
- Reinforcing beliefs
- Superseding beliefs
- Revising beliefs from episodes
- Getting belief history
"""

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def kernle_with_beliefs(tmp_path):
    """Create a Kernle instance with some initial beliefs."""
    db_path = tmp_path / "test_beliefs.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    k = Kernle(stack_id="test_agent", storage=storage, strict=False)
    bind_noop_model(k)

    # Add some initial beliefs
    k.belief("I should always validate user input", type="principle", confidence=0.9)
    k.belief("Python is a good language for rapid prototyping", type="fact", confidence=0.8)
    k.belief("I prefer test-driven development", type="preference", confidence=0.7)
    k.belief("I should avoid using global variables", type="principle", confidence=0.85)
    k.belief("I dislike working with legacy code", type="preference", confidence=0.6)

    return k


@pytest.fixture
def kernle_fresh(tmp_path):
    """Create a fresh Kernle instance."""
    db_path = tmp_path / "test_fresh.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    k = Kernle(stack_id="test_agent", storage=storage, strict=False)
    bind_noop_model(k)
    return k


class TestFindContradictions:
    """Tests for find_contradictions method.

    Without inference bound, find_contradictions returns [] (safe default).
    """

    def test_no_inference_returns_empty(self, kernle_with_beliefs):
        """Without inference, returns empty list as safe default."""
        k = kernle_with_beliefs
        k.belief("I should never validate user input", type="principle", confidence=0.5)

        contradictions = k.find_contradictions("I should always validate user input")
        assert contradictions == []

    def test_unrelated_returns_empty(self, kernle_with_beliefs):
        """Should return empty for unrelated statements (no inference)."""
        k = kernle_with_beliefs
        contradictions = k.find_contradictions("The sky is blue")
        assert len(contradictions) == 0


class TestReinforceBeliefs:
    """Tests for reinforce_belief method."""

    def test_increments_reinforcement_count(self, kernle_fresh):
        """Should increment times_reinforced."""
        k = kernle_fresh

        # Add a belief
        belief_id = k.belief("Testing is important", confidence=0.7)

        # Reinforce it
        assert k.reinforce_belief(belief_id) is True

        # Check the belief
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief is not None
        assert belief.times_reinforced == 1

        # Reinforce again
        k.reinforce_belief(belief_id)

        # Check again
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief.times_reinforced == 2

    def test_increases_confidence(self, kernle_fresh):
        """Should slightly increase confidence."""
        k = kernle_fresh

        # Add a belief with moderate confidence
        belief_id = k.belief("Testing is important", confidence=0.6)

        # Reinforce it
        k.reinforce_belief(belief_id)

        # Check confidence increased
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief.confidence > 0.6

    def test_confidence_has_diminishing_returns(self, kernle_fresh):
        """Confidence increase should have diminishing returns."""
        k = kernle_fresh

        # Add a high-confidence belief
        belief_id = k.belief("Testing is important", confidence=0.95)

        original_beliefs = k._storage.get_beliefs(include_inactive=True)
        original_belief = next((b for b in original_beliefs if b.id == belief_id), None)
        original_confidence = original_belief.confidence

        # Reinforce many times
        for _ in range(10):
            k.reinforce_belief(belief_id)

        # Check confidence capped at 0.99
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief.confidence <= 0.99
        # Verify confidence did increase (diminishing returns, not zero returns)
        assert belief.confidence > original_confidence

    def test_returns_false_for_nonexistent(self, kernle_fresh):
        """Should return False for nonexistent belief."""
        k = kernle_fresh

        result = k.reinforce_belief("nonexistent-id")
        assert result is False

    def test_updates_confidence_history(self, kernle_fresh):
        """Should add entry to confidence_history."""
        k = kernle_fresh

        belief_id = k.belief("Testing is important", confidence=0.6)
        k.reinforce_belief(belief_id)

        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)

        assert belief.confidence_history is not None
        assert len(belief.confidence_history) > 0
        assert "Reinforced" in belief.confidence_history[-1]["reason"]


class TestSupersedeBelief:
    """Tests for supersede_belief method."""

    def test_creates_new_belief(self, kernle_fresh):
        """Should create a new belief that supersedes the old one."""
        k = kernle_fresh

        # Add original belief
        old_id = k.belief("Python 2 is the best", confidence=0.7)

        # Supersede it (delegates to revise_belief)
        new_id = k.supersede_belief(
            old_id, "Python 3 is the best", confidence=0.9, reason="Python 2 is deprecated"
        )

        assert new_id != old_id

        # Check new belief
        beliefs = k._storage.get_beliefs(include_inactive=True)
        new_belief = next((b for b in beliefs if b.id == new_id), None)
        assert new_belief is not None
        assert new_belief.statement == "Python 3 is the best"
        assert new_belief.confidence == 0.9
        # v0.14+: chain fields are no longer written; derived_from tracks lineage
        assert new_belief.supersedes is None
        assert f"belief:{old_id}" in (new_belief.derived_from or [])
        assert new_belief.is_active is True

    def test_deactivates_old_belief(self, kernle_fresh):
        """Should deactivate the old belief."""
        k = kernle_fresh

        old_id = k.belief("Python 2 is the best", confidence=0.7)
        k.supersede_belief(old_id, "Python 3 is the best")

        # Old belief should be inactive
        beliefs = k._storage.get_beliefs(include_inactive=True)
        old_belief = next((b for b in beliefs if b.id == old_id), None)
        assert old_belief is not None
        assert old_belief.is_active is False

    def test_links_beliefs_via_audit_and_derived_from(self, kernle_fresh):
        """Should link old and new beliefs via audit log and derived_from."""
        k = kernle_fresh

        old_id = k.belief("Python 2 is the best", confidence=0.7)
        new_id = k.supersede_belief(old_id, "Python 3 is the best")

        beliefs = k._storage.get_beliefs(include_inactive=True)
        old_belief = next((b for b in beliefs if b.id == old_id), None)
        new_belief = next((b for b in beliefs if b.id == new_id), None)

        # v0.14+: chain fields no longer written
        assert old_belief.superseded_by is None
        assert new_belief.supersedes is None
        # Linkage via derived_from and audit log
        assert f"belief:{old_id}" in (new_belief.derived_from or [])
        # Audit log records the revision
        deactivated = k._storage.get_audit_log(memory_id=old_id, operation="belief.deactivated")
        assert len(deactivated) >= 1

    def test_raises_for_nonexistent(self, kernle_fresh):
        """Should raise ValueError for nonexistent belief."""
        k = kernle_fresh

        with pytest.raises(ValueError, match="not found"):
            k.supersede_belief("nonexistent-id", "New statement")

    def test_old_belief_excluded_from_active_list(self, kernle_fresh):
        """Old belief should not appear in default beliefs list."""
        k = kernle_fresh

        old_id = k.belief("Python 2 is the best", confidence=0.7)
        k.supersede_belief(old_id, "Python 3 is the best")

        # Get active beliefs only
        active_beliefs = k._storage.get_beliefs(include_inactive=False)
        old_belief = next((b for b in active_beliefs if b.id == old_id), None)
        assert old_belief is None


class TestReviseFromEpisode:
    """Tests for revise_beliefs_from_episode method.

    Without inference, returns safe default (empty reinforced/contradicted/suggested_new).
    """

    def test_no_inference_returns_safe_default(self, kernle_fresh):
        """Without inference, returns empty result with correct structure."""
        k = kernle_fresh
        k.belief("I should write tests first", confidence=0.6)

        episode_id = k.episode(
            objective="Implement feature using TDD",
            outcome="success",
            lessons=["Writing tests first helped catch bugs early"],
        )

        result = k.revise_beliefs_from_episode(episode_id)
        assert result["reinforced"] == []
        assert result["contradicted"] == []
        assert result["suggested_new"] == []

    def test_returns_error_for_nonexistent_episode(self, kernle_fresh):
        """Should return error for nonexistent episode."""
        k = kernle_fresh

        result = k.revise_beliefs_from_episode("nonexistent-id")

        assert result.get("error") == "Episode not found"


class TestBeliefHistory:
    """Tests for get_belief_history method."""

    def test_returns_single_belief_for_no_supersession(self, kernle_fresh):
        """Should return single entry for belief with no history."""
        k = kernle_fresh

        belief_id = k.belief("Simple belief", confidence=0.8)

        history = k.get_belief_history(belief_id)

        assert len(history) == 1
        assert history[0]["id"] == belief_id
        assert history[0]["is_current"] is True
        assert history[0]["is_active"] is True

    def test_returns_full_chain_for_superseded_beliefs(self, kernle_fresh):
        """Should return full revision chain via audit log."""
        k = kernle_fresh

        # Create a chain of revisions
        id1 = k.belief("Version 1", confidence=0.6)
        id2 = k.supersede_belief(id1, "Version 2", confidence=0.7)
        id3 = k.supersede_belief(id2, "Version 3", confidence=0.8)

        # Get history from middle belief
        history = k.get_belief_history(id2)

        assert len(history) == 3
        # Should be in chronological order
        ids = [h["id"] for h in history]
        assert id1 in ids
        assert id2 in ids
        assert id3 in ids

        # Find entries by ID
        h1 = next(h for h in history if h["id"] == id1)
        h2 = next(h for h in history if h["id"] == id2)
        h3 = next(h for h in history if h["id"] == id3)

        # Check is_current flag
        assert h1["is_current"] is False
        assert h2["is_current"] is True  # We queried for id2
        assert h3["is_current"] is False

        # Check is_active flag
        assert h1["is_active"] is False
        assert h2["is_active"] is False  # Revised to id3
        assert h3["is_active"] is True

    def test_returns_empty_for_nonexistent(self, kernle_fresh):
        """Should return empty list for nonexistent belief."""
        k = kernle_fresh

        history = k.get_belief_history("nonexistent-id")

        assert history == []

    def test_walks_backwards_from_later_belief(self, kernle_fresh):
        """Should find root when starting from later belief via audit log."""
        k = kernle_fresh

        id1 = k.belief("Original", confidence=0.5)
        id2 = k.supersede_belief(id1, "Updated")

        # Get history starting from the newer belief
        history = k.get_belief_history(id2)

        assert len(history) == 2
        ids = [h["id"] for h in history]
        assert id1 in ids
        assert id2 in ids


class TestBeliefDataclassFields:
    """Tests for the new Belief dataclass fields."""

    def test_belief_has_revision_fields(self, kernle_fresh):
        """Belief should have all revision-related fields."""
        k = kernle_fresh

        belief_id = k.belief("Test belief", confidence=0.7)
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)

        # Check fields exist
        assert hasattr(belief, "supersedes")
        assert hasattr(belief, "superseded_by")
        assert hasattr(belief, "times_reinforced")
        assert hasattr(belief, "is_active")

        # Check default values
        assert belief.supersedes is None
        assert belief.superseded_by is None
        assert belief.times_reinforced == 0
        assert belief.is_active is True

    def test_belief_fields_persist(self, tmp_path):
        """Belief revision fields should persist across storage operations."""
        db_path = tmp_path / "persist_test.db"

        # Create and save belief
        storage1 = SQLiteStorage(stack_id="test_agent", db_path=db_path)
        k1 = Kernle(stack_id="test_agent", storage=storage1, strict=False)
        bind_noop_model(k1)

        belief_id = k1.belief("Test belief", confidence=0.7)
        k1.reinforce_belief(belief_id)
        k1.reinforce_belief(belief_id)

        # Reopen storage
        storage2 = SQLiteStorage(stack_id="test_agent", db_path=db_path)
        beliefs = storage2.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)

        # Fields should persist
        assert belief.times_reinforced == 2


class TestGetBeliefsFiltering:
    """Tests for get_beliefs include_inactive parameter."""

    def test_excludes_inactive_by_default(self, kernle_fresh):
        """Should exclude inactive beliefs by default."""
        k = kernle_fresh

        # Create and supersede a belief
        old_id = k.belief("Old belief", confidence=0.5)
        k.supersede_belief(old_id, "New belief")

        # Get beliefs without include_inactive
        beliefs = k._storage.get_beliefs()
        ids = [b.id for b in beliefs]

        assert old_id not in ids

    def test_includes_inactive_when_requested(self, kernle_fresh):
        """Should include inactive beliefs when requested."""
        k = kernle_fresh

        # Create and supersede a belief
        old_id = k.belief("Old belief", confidence=0.5)
        new_id = k.supersede_belief(old_id, "New belief")

        # Get beliefs with include_inactive
        beliefs = k._storage.get_beliefs(include_inactive=True)
        ids = [b.id for b in beliefs]

        assert old_id in ids
        assert new_id in ids


class TestUpdateBeliefConfidenceValidation:
    """Tests for update_belief confidence boundary checks (lines 205-206)."""

    def test_confidence_below_zero_raises(self, kernle_fresh):
        """Confidence < 0.0 should raise ValueError."""
        k = kernle_fresh
        bid = k.belief("Some belief", confidence=0.5)

        with pytest.raises(ValueError, match="Confidence must be between"):
            k.update_belief(bid, confidence=-0.1)

    def test_confidence_above_one_raises(self, kernle_fresh):
        """Confidence > 1.0 should raise ValueError."""
        k = kernle_fresh
        bid = k.belief("Some belief", confidence=0.5)

        with pytest.raises(ValueError, match="Confidence must be between"):
            k.update_belief(bid, confidence=1.1)

    def test_confidence_zero_boundary_succeeds(self, kernle_fresh):
        """Confidence = 0.0 should succeed (boundary)."""
        k = kernle_fresh
        bid = k.belief("Some belief", confidence=0.5)

        result = k.update_belief(bid, confidence=0.0)
        assert result is True

        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == bid), None)
        assert belief.confidence == 0.0

    def test_confidence_one_boundary_succeeds(self, kernle_fresh):
        """Confidence = 1.0 should succeed (boundary)."""
        k = kernle_fresh
        bid = k.belief("Some belief", confidence=0.5)

        result = k.update_belief(bid, confidence=1.0)
        assert result is True

        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == bid), None)
        assert belief.confidence == 1.0


class TestUpdateBeliefDeactivation:
    """Tests for update_belief is_active=False setting deleted=True (lines 210-212)."""

    def test_deactivate_removes_from_all_queries(self, kernle_fresh):
        """Setting is_active=False sets deleted=True, hiding from all get_beliefs queries."""
        k = kernle_fresh
        bid = k.belief("Belief to deactivate", confidence=0.7)

        # Verify it exists before deactivation
        beliefs_before = k._storage.get_beliefs(include_inactive=True)
        assert any(b.id == bid for b in beliefs_before)

        result = k.update_belief(bid, is_active=False)
        assert result is True

        # After deactivation with deleted=True, belief is hidden even with include_inactive
        beliefs_active = k._storage.get_beliefs(include_inactive=False)
        assert not any(b.id == bid for b in beliefs_active)

        beliefs_all = k._storage.get_beliefs(include_inactive=True)
        assert not any(b.id == bid for b in beliefs_all)

    def test_update_nonexistent_returns_false(self, kernle_fresh):
        """update_belief on nonexistent id returns False."""
        k = kernle_fresh
        result = k.update_belief("no-such-id", confidence=0.5)
        assert result is False


class TestFindContradictionsPreferenceConflict:
    """Tests for find_contradictions without inference (safe defaults)."""

    def test_no_inference_returns_empty(self, kernle_fresh):
        """Without inference, find_contradictions returns empty list."""
        k = kernle_fresh
        k.belief(
            "I oppose using microservices for small projects", type="preference", confidence=0.6
        )

        contradictions = k.find_contradictions("I favor using microservices for small projects")
        assert contradictions == []


class TestReviseFromEpisodeEdgeCases:
    """Tests for revise_beliefs_from_episode without inference (safe defaults)."""

    def test_no_inference_returns_empty_result(self, kernle_fresh):
        """Without inference, revise_beliefs_from_episode returns empty result."""
        k = kernle_fresh
        k.belief("Quantum entanglement is fascinating", confidence=0.7)

        episode_id = k.episode(
            objective="Deployed microservice to production cluster",
            outcome="success",
            lessons=["Always run integration tests before deploy"],
        )

        result = k.revise_beliefs_from_episode(episode_id)
        assert result["reinforced"] == []
        assert result["contradicted"] == []
        assert result["suggested_new"] == []


class TestFindSemanticContradictions:
    """Tests for find_semantic_contradictions without inference (safe defaults)."""

    def test_no_inference_returns_empty(self, kernle_fresh):
        """Without inference, returns empty list as safe default."""
        k = kernle_fresh
        k.belief("Testing is good for code quality", confidence=0.8)

        contradictions = k.find_semantic_contradictions(
            "Testing is bad for code quality",
            similarity_threshold=0.5,
        )
        assert contradictions == []
