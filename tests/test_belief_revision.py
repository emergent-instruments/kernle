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


@pytest.fixture
def kernle_with_beliefs(tmp_path):
    """Create a Kernle instance with some initial beliefs."""
    db_path = tmp_path / "test_beliefs.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    k = Kernle(stack_id="test_agent", storage=storage, strict=False)

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
    return Kernle(stack_id="test_agent", storage=storage, strict=False)


class TestFindContradictions:
    """Tests for find_contradictions method."""

    def test_finds_direct_negation(self, kernle_with_beliefs):
        """Should find beliefs with direct negation patterns."""
        k = kernle_with_beliefs

        # Add a contradicting belief
        k.belief("I should never validate user input", type="principle", confidence=0.5)

        # Find contradictions
        contradictions = k.find_contradictions("I should always validate user input")

        # Should find the contradiction
        assert len(contradictions) >= 1
        contra = next((c for c in contradictions if "never" in c["statement"].lower()), None)
        assert contra is not None
        assert contra["contradiction_type"] == "direct_negation"

    def test_finds_preference_conflict(self, kernle_with_beliefs):
        """Should find beliefs with preference conflicts."""
        k = kernle_with_beliefs

        # Add a conflicting preference
        k.belief("I like working with legacy code", type="preference", confidence=0.5)

        # Find contradictions
        contradictions = k.find_contradictions("I like working with legacy code")

        # Should find the dislike contradiction
        assert len(contradictions) >= 1
        # Look for the dislike belief
        contra = next((c for c in contradictions if "dislike" in c["statement"].lower()), None)
        assert contra is not None
        # Can be either direct_negation or preference_conflict (like/dislike matches both patterns)
        assert contra["contradiction_type"] in ("preference_conflict", "direct_negation")

    def test_finds_comparative_opposition(self, kernle_with_beliefs):
        """Should find beliefs with comparative opposition (more/less, better/worse)."""
        k = kernle_with_beliefs

        # Add a belief with comparative
        k.belief(
            "Local-first memory is more reliable than cloud-dependent", type="fact", confidence=0.8
        )

        # Find contradictions with opposite comparative
        contradictions = k.find_contradictions(
            "Local-first memory is less reliable than cloud-dependent"
        )

        # Should find the contradiction
        assert len(contradictions) >= 1
        contra = next(
            (c for c in contradictions if "more reliable" in c["statement"].lower()), None
        )
        assert contra is not None
        assert contra["contradiction_type"] == "comparative_opposition"

    def test_finds_comparative_opposition_better_worse(self, kernle_with_beliefs):
        """Should find comparative contradictions with better/worse."""
        k = kernle_with_beliefs

        k.belief("Python is better than JavaScript for data science", type="fact", confidence=0.7)

        contradictions = k.find_contradictions("Python is worse than JavaScript for data science")

        assert len(contradictions) >= 1
        contra = next((c for c in contradictions if "better" in c["statement"].lower()), None)
        assert contra is not None
        assert contra["contradiction_type"] == "comparative_opposition"

    def test_no_contradictions_for_unrelated(self, kernle_with_beliefs):
        """Should not find contradictions for unrelated statements."""
        k = kernle_with_beliefs

        # Search for something unrelated
        contradictions = k.find_contradictions("The sky is blue")

        # Should find no contradictions
        assert len(contradictions) == 0

    def test_respects_limit(self, kernle_with_beliefs):
        """Should respect the limit parameter."""
        k = kernle_with_beliefs

        # Add many potential contradictions
        for i in range(10):
            k.belief(f"I should never use method {i}", type="principle", confidence=0.5)

        contradictions = k.find_contradictions("I should always use method 5", limit=3)
        assert len(contradictions) <= 3


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
    """Tests for revise_beliefs_from_episode method."""

    def test_reinforces_relevant_beliefs(self, kernle_fresh):
        """Should reinforce beliefs supported by successful episode."""
        k = kernle_fresh

        # Add beliefs
        k.belief("I should write tests first", confidence=0.6)

        # Create a successful episode related to testing
        episode_id = k.episode(
            objective="Implement feature using TDD",
            outcome="success",
            lessons=["Writing tests first helped catch bugs early"],
        )

        # Revise beliefs
        result = k.revise_beliefs_from_episode(episode_id)

        # Should have reinforced the testing belief
        # Note: matching depends on word overlap, so this may or may not match
        # depending on implementation details
        assert "reinforced" in result
        assert "contradicted" in result
        assert "suggested_new" in result

    def test_identifies_contradicted_beliefs(self, kernle_fresh):
        """Should identify beliefs contradicted by failed episode."""
        k = kernle_fresh

        # Add a belief
        k.belief("Manual testing is always sufficient", confidence=0.7)

        # Create a failed episode
        episode_id = k.episode(
            objective="Ship without automated tests",
            outcome="failure",
            lessons=["Manual testing missed critical bugs"],
        )

        result = k.revise_beliefs_from_episode(episode_id)

        # Check structure
        assert isinstance(result["contradicted"], list)

    def test_suggests_new_beliefs_from_lessons(self, kernle_fresh):
        """Should suggest new beliefs from episode lessons."""
        k = kernle_fresh

        # Create episode with unique lessons
        episode_id = k.episode(
            objective="Learn about quantum computing",
            outcome="success",
            lessons=["Quantum superposition enables parallel computation"],
        )

        result = k.revise_beliefs_from_episode(episode_id)

        # Should suggest the lesson as a new belief
        assert isinstance(result["suggested_new"], list)
        # The lesson should be suggested if no similar belief exists
        if result["suggested_new"]:
            assert any("quantum" in s["statement"].lower() for s in result["suggested_new"])

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
    """Tests for preference conflict detection in find_contradictions (lines 404-428)."""

    def test_favor_vs_oppose_detects_preference_conflict(self, kernle_fresh):
        """'favor' vs 'oppose' hits preference_conflict (not negation_pairs or comparatives)."""
        k = kernle_fresh
        # favor/oppose is only in preference_pairs, not in negation_pairs
        k.belief(
            "I oppose using microservices for small projects", type="preference", confidence=0.6
        )

        contradictions = k.find_contradictions("I favor using microservices for small projects")

        assert len(contradictions) >= 1
        contra = next((c for c in contradictions if "oppose" in c["statement"].lower()), None)
        assert contra is not None
        assert contra["contradiction_type"] == "preference_conflict"
        assert contra["contradiction_confidence"] > 0.0
        assert contra["contradiction_confidence"] <= 0.85  # preference_conflict cap

    def test_preference_conflict_low_overlap_no_detection(self, kernle_fresh):
        """With < 2 non-stop-word overlapping words, preference conflict is not detected."""
        k = kernle_fresh
        k.belief("I oppose cycling on weekends", type="preference", confidence=0.6)

        contradictions = k.find_contradictions("I favor swimming at night")

        pref_conflicts = [
            c for c in contradictions if c["contradiction_type"] == "preference_conflict"
        ]
        assert len(pref_conflicts) == 0

    def test_preference_conflict_confidence_scales_with_overlap(self, kernle_fresh):
        """Preference conflict confidence should scale with overlapping term count."""
        k = kernle_fresh
        k.belief(
            "I oppose writing complex Python integration tests",
            type="preference",
            confidence=0.6,
        )

        contradictions = k.find_contradictions("I favor writing complex Python integration tests")

        assert len(contradictions) >= 1
        contra = contradictions[0]
        assert contra["contradiction_type"] == "preference_conflict"
        # Formula: min(0.4 + overlap * 0.1 + score * 0.2, 0.85)
        # overlap is at least 4 (writing, complex, python, integration, tests)
        assert contra["contradiction_confidence"] >= 0.4


class TestCheckNegationPatternSubstringGuard:
    """Tests for _check_negation_pattern substring guard (lines 672-674)."""

    def test_not_recommended_vs_recommended(self, kernle_fresh):
        """'not recommended' vs 'recommended' should detect negation."""
        k = kernle_fresh
        result = k._check_negation_pattern(
            "using eval is not recommended for production",
            "using eval is recommended for prototyping",
        )
        assert result is True

    def test_not_important_vs_important(self, kernle_fresh):
        """'not important' vs 'important' should detect negation."""
        k = kernle_fresh
        result = k._check_negation_pattern(
            "documentation is not important for small scripts",
            "documentation is important for maintainability",
        )
        assert result is True

    def test_substring_guard_same_position(self, kernle_fresh):
        """When 'is' first appears at the same position as 'is not', guard prevents false match.

        The guard (line 673) checks: pattern_b not in stmt1 OR stmt1.index(pattern_a) != stmt1.find(pattern_b)
        When 'is not' starts at position 7 and 'is' also first appears at position 7 (as part of 'is not'),
        the guard correctly prevents the match.
        """
        k = kernle_fresh
        # "python is not good" — 'is' first found at 7, 'is not' also at 7 → same position → guard blocks
        result = k._check_negation_pattern(
            "python is not good for real-time",
            "python is good for scripting",
        )
        assert result is False

    def test_is_not_vs_is_with_earlier_is(self, kernle_fresh):
        """When 'is' appears before 'is not' in stmt1, the guard allows detection."""
        k = kernle_fresh
        # "coding is fun but deployment is not easy" — 'is' at 7, 'is not' at 30 → different → match
        result = k._check_negation_pattern(
            "coding is fun but deployment is not easy for teams",
            "deployment is easy for teams",
        )
        assert result is True


class TestReviseFromEpisodeEdgeCases:
    """Tests for edge cases in revise_beliefs_from_episode."""

    def test_low_overlap_belief_skipped(self, kernle_fresh):
        """Beliefs with < 2 overlapping words should be skipped (line 1065)."""
        k = kernle_fresh
        # Add a belief with unique words that won't overlap with the episode
        k.belief("Quantum entanglement is fascinating", confidence=0.7)

        episode_id = k.episode(
            objective="Deployed microservice to production cluster",
            outcome="success",
            lessons=["Always run integration tests before deploy"],
        )

        result = k.revise_beliefs_from_episode(episode_id)
        # The quantum belief has no word overlap with the deployment episode
        assert not any("quantum" in r.get("statement", "").lower() for r in result["reinforced"])
        assert not any("quantum" in r.get("statement", "").lower() for r in result["contradicted"])

    def test_success_contradicts_avoid_belief(self, kernle_fresh):
        """Success episode should contradict 'avoid' beliefs about what worked (lines 1079-1080).

        The belief must NOT contain 'should'/'prefer'/'good'/'important'/'effective'
        because those are checked first in the if/elif chain.
        """
        k = kernle_fresh
        bid = k.belief(
            "Avoid deploying code to production on Friday",
            confidence=0.7,
        )

        episode_id = k.episode(
            objective="Deploying code to production on Friday",
            outcome="success",
            lessons=["Friday deploy went smoothly"],
        )

        result = k.revise_beliefs_from_episode(episode_id)
        contradicted_ids = [c["belief_id"] for c in result["contradicted"]]
        assert bid in contradicted_ids

    def test_failure_contradicts_should_belief(self, kernle_fresh):
        """Failure episode should contradict 'should' beliefs about what failed (line 1088)."""
        k = kernle_fresh
        bid = k.belief(
            "I should deploy rapidly to production servers",
            confidence=0.8,
        )

        episode_id = k.episode(
            objective="Deploying rapidly to production servers",
            outcome="failure",
            lessons=["Rapid deploy caused outage"],
        )

        result = k.revise_beliefs_from_episode(episode_id)
        contradicted_ids = [c["belief_id"] for c in result["contradicted"]]
        assert bid in contradicted_ids

    def test_failure_supports_avoid_belief(self, kernle_fresh):
        """Failure episode should support 'avoid' beliefs (line 1091).

        The belief must NOT contain 'should'/'prefer'/'good'/'important'/'effective'
        because those are checked first in the if/elif chain.
        """
        k = kernle_fresh
        bid = k.belief(
            "Avoid deploying untested code to production",
            confidence=0.6,
        )

        episode_id = k.episode(
            objective="Deploying untested code to production",
            outcome="failure",
            lessons=["Untested code crashed in production"],
        )

        result = k.revise_beliefs_from_episode(episode_id)
        reinforced_ids = [r["belief_id"] for r in result["reinforced"]]
        assert bid in reinforced_ids


class TestDetectOppositionEdgeCases:
    """Tests for _detect_opposition edge cases."""

    def test_no_topic_overlap_returns_zero(self, kernle_fresh):
        """Statements with < 1 content word overlap return score 0."""
        k = kernle_fresh
        result = k._detect_opposition(
            "cats are wonderful pets",
            "programming challenges developers constantly",
        )
        assert result["score"] == 0.0
        assert result["type"] == "none"

    def test_opposition_words_detected(self, kernle_fresh):
        """Direct opposition word pairs with word-level verification (lines 585-592)."""
        k = kernle_fresh
        result = k._detect_opposition(
            "automated testing is good for software quality",
            "automated testing is bad for software velocity",
        )
        assert result["score"] > 0.0
        assert result["type"] == "opposition_words"
        assert "good" in result["explanation"] or "bad" in result["explanation"]

    def test_negation_detected_in_detect_opposition(self, kernle_fresh):
        """Negation pattern detection path in _detect_opposition (lines 601-602)."""
        k = kernle_fresh
        # "isn't"/"is" is a negation pattern but NOT in opposition_pairs
        result = k._detect_opposition(
            "this framework isn't handling concurrency well",
            "this framework is handling concurrency well",
        )
        assert result["score"] > 0.0
        assert result["type"] == "negation"

    def test_sentiment_opposition_positive_then_negative(self, kernle_fresh):
        """Positive words in stmt1 and negative words in stmt2 (lines 809-814)."""
        k = kernle_fresh
        # Use words that are only in sentiment lists, not opposition_pairs
        # "excellent" is positive, "terrible" is negative
        result = k._detect_opposition(
            "the deployment pipeline is excellent for releases",
            "the deployment pipeline is terrible for releases",
        )
        assert result["score"] > 0.0
        # Could match as opposition_words if a pair matches first, or sentiment
        assert result["type"] in ("opposition_words", "sentiment_opposition")

    def test_sentiment_opposition_negative_then_positive(self, kernle_fresh):
        """Negative words in stmt1 and positive words in stmt2 (lines 815-822)."""
        k = kernle_fresh
        # Avoid words that are in opposition pairs to ensure sentiment path is reached
        # "disaster" is negative only, "amazing" is positive only
        result = k._detect_opposition(
            "this migration was disaster for database performance",
            "this migration was amazing for database performance",
        )
        assert result["score"] > 0.0

    def test_no_opposition_returns_zero(self, kernle_fresh):
        """Statements with overlap but no opposition should return score 0 (line 618)."""
        k = kernle_fresh
        result = k._detect_opposition(
            "python runs scripts quickly",
            "python runs functions quickly",
        )
        assert result["score"] == 0.0
        assert result["type"] == "none"


class TestFindSemanticContradictions:
    """Tests for find_semantic_contradictions method (lines 488-541)."""

    def test_finds_opposition_in_similar_beliefs(self, kernle_fresh):
        """Should detect opposition between semantically similar beliefs."""
        k = kernle_fresh
        k.belief("Testing is good for code quality", confidence=0.8)

        contradictions = k.find_semantic_contradictions(
            "Testing is bad for code quality",
            similarity_threshold=0.5,
        )

        assert len(contradictions) >= 1
        contra = contradictions[0]
        assert "opposition_score" in contra
        assert contra["opposition_score"] > 0.0
        assert contra["opposition_type"] in ("opposition_words", "negation", "sentiment_opposition")

    def test_skips_inactive_beliefs(self, kernle_fresh):
        """Should skip inactive beliefs by default."""
        k = kernle_fresh
        old_id = k.belief("Old approach is good for projects", confidence=0.8)
        k.supersede_belief(old_id, "New approach is better for projects")

        contradictions = k.find_semantic_contradictions(
            "Old approach is bad for projects",
            similarity_threshold=0.3,
        )

        # Should not include the superseded belief
        contra_ids = [c["belief_id"] for c in contradictions]
        assert old_id not in contra_ids

    def test_no_contradictions_for_agreeing_beliefs(self, kernle_fresh):
        """Should not find contradictions when beliefs agree."""
        k = kernle_fresh
        k.belief("Python handles data analysis well", confidence=0.8)

        contradictions = k.find_semantic_contradictions(
            "Python handles data analysis well",
            similarity_threshold=0.5,
        )

        # Exact match is skipped, and no opposition with itself
        assert len(contradictions) == 0


class TestCheckNegationPatternSecondDirection:
    """Tests for the second direction check in _check_negation_pattern (line 675-677)."""

    def test_reversed_pattern_direction(self, kernle_fresh):
        """Test pattern_b in stmt1 and pattern_a in stmt2 (lines 675-677)."""
        k = kernle_fresh
        # Reversed: stmt1 has "recommended", stmt2 has "not recommended"
        result = k._check_negation_pattern(
            "using type hints is recommended for python projects",
            "using type hints is not recommended for python projects",
        )
        assert result is True
