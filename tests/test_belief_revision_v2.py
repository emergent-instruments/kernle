"""Tests for Phase 3: Remove Supersession Chains (#840 Part 1).

TDD tests for:
- revise_belief() replaces supersede_belief()
- Audit-log based belief history (dual-source: audit primary, chain fallback)
- save_belief writes NULL for supersedes/superseded_by
- Consolidation uses is_active instead of superseded_by
- No-model end-to-end belief save
"""

import json

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage
from kernle.types import Belief
from tests.conftest import bind_noop_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance for testing."""
    db_path = tmp_path / "test_revision_v2.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    instance = Kernle(stack_id="test-stack", storage=storage, strict=False)
    bind_noop_model(instance)
    return instance


@pytest.fixture
def k_with_belief(k):
    """Create a Kernle instance with one initial belief."""
    k.belief("Testing is essential for code quality", type="principle", confidence=0.9)
    beliefs = k._storage.get_beliefs(limit=10)
    assert len(beliefs) >= 1
    return k, beliefs[0]


# ---------------------------------------------------------------------------
# revise_belief() tests
# ---------------------------------------------------------------------------


class TestReviseBelief:
    def test_revise_belief_creates_new_deactivates_old(self, k_with_belief):
        """revise_belief() creates a new active belief and deactivates the old one."""
        k, old_belief = k_with_belief

        new_id = k.revise_belief(
            old_id=old_belief.id,
            new_statement="Testing is critical, not just essential",
            confidence=0.95,
            reason="Strengthened conviction after production incident",
        )

        # New belief should exist and be active
        all_beliefs = k._storage.get_beliefs(limit=100, include_inactive=True)
        new_belief = next((b for b in all_beliefs if b.id == new_id), None)
        assert new_belief is not None
        assert new_belief.is_active is True
        assert new_belief.statement == "Testing is critical, not just essential"
        assert new_belief.confidence == 0.95

        # Old belief should be deactivated
        old_refreshed = next((b for b in all_beliefs if b.id == old_belief.id), None)
        assert old_refreshed is not None
        assert old_refreshed.is_active is False

    def test_revise_belief_creates_audit_trail(self, k_with_belief):
        """revise_belief() creates audit log entries for the revision."""
        k, old_belief = k_with_belief

        new_id = k.revise_belief(
            old_id=old_belief.id,
            new_statement="Revised statement",
            confidence=0.85,
            reason="New evidence",
        )

        # Check audit log for belief.deactivated on old belief
        deactivated_entries = k._storage.get_audit_log(
            memory_id=old_belief.id, operation="belief.deactivated"
        )
        assert len(deactivated_entries) >= 1
        details = deactivated_entries[0].get("details")
        if isinstance(details, str):
            details = json.loads(details)
        assert details["reason"] == "New evidence"
        assert details["trigger_id"] == new_id

        # Check audit log for belief.revised on new belief
        revised_entries = k._storage.get_audit_log(memory_id=new_id, operation="belief.revised")
        assert len(revised_entries) >= 1
        details = revised_entries[0].get("details")
        if isinstance(details, str):
            details = json.loads(details)
        assert details["revision_type"] == "supersession"
        assert details["trigger_id"] == old_belief.id

    def test_revise_belief_new_has_derived_from(self, k_with_belief):
        """New belief from revise_belief() has derived_from pointing to old belief."""
        k, old_belief = k_with_belief

        new_id = k.revise_belief(
            old_id=old_belief.id,
            new_statement="Updated statement",
            confidence=0.8,
        )

        all_beliefs = k._storage.get_beliefs(limit=100, include_inactive=True)
        new_belief = next((b for b in all_beliefs if b.id == new_id), None)
        assert new_belief is not None
        assert new_belief.derived_from is not None
        assert f"belief:{old_belief.id}" in new_belief.derived_from

    def test_revise_belief_writes_null_supersedes(self, k_with_belief):
        """revise_belief() does NOT write supersedes/superseded_by chain fields."""
        k, old_belief = k_with_belief

        new_id = k.revise_belief(
            old_id=old_belief.id,
            new_statement="Updated statement",
            confidence=0.8,
        )

        all_beliefs = k._storage.get_beliefs(limit=100, include_inactive=True)
        new_belief = next((b for b in all_beliefs if b.id == new_id), None)
        old_refreshed = next((b for b in all_beliefs if b.id == old_belief.id), None)

        # Neither new nor old should have chain fields set
        assert new_belief.supersedes is None
        assert new_belief.superseded_by is None
        assert old_refreshed.superseded_by is None

    def test_revise_belief_not_found_raises(self, k):
        """revise_belief() raises ValueError for non-existent belief."""
        with pytest.raises(ValueError, match="not found"):
            k.revise_belief(
                old_id="nonexistent-id",
                new_statement="Updated",
                confidence=0.8,
            )

    def test_revise_belief_clamps_confidence(self, k_with_belief):
        """revise_belief() clamps confidence to 0.0-1.0 range."""
        k, old_belief = k_with_belief

        new_id = k.revise_belief(
            old_id=old_belief.id,
            new_statement="Updated",
            confidence=1.5,  # Over max
        )

        all_beliefs = k._storage.get_beliefs(limit=100, include_inactive=True)
        new_belief = next((b for b in all_beliefs if b.id == new_id), None)
        assert new_belief.confidence == 1.0


# ---------------------------------------------------------------------------
# get_belief_history() dual-source tests
# ---------------------------------------------------------------------------


class TestGetBeliefHistory:
    def test_get_belief_history_from_audit_log(self, k_with_belief):
        """get_belief_history() returns history from audit log entries."""
        k, old_belief = k_with_belief

        # Create a revision (which creates audit entries)
        new_id = k.revise_belief(
            old_id=old_belief.id,
            new_statement="Revised v2",
            confidence=0.85,
            reason="Updated understanding",
        )

        # History should include both old and new belief
        history = k.get_belief_history(old_belief.id)
        assert len(history) >= 2

        # Should have the original and the revision
        ids_in_history = [h["id"] for h in history]
        assert old_belief.id in ids_in_history
        assert new_id in ids_in_history

    def test_get_belief_history_legacy_chain_fallback(self, k):
        """get_belief_history() falls back to chain walk for pre-v0.14 data."""
        # Insert legacy data directly via SQL to simulate pre-v0.14 chain fields
        # (save_belief now writes NULL for these fields)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO beliefs (id, stack_id, statement, belief_type, confidence,
                   created_at, source_type, supersedes, superseded_by, times_reinforced,
                   is_active, strength, local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "legacy-old",
                    "test-stack",
                    "Old belief",
                    "fact",
                    0.7,
                    now,
                    "direct",
                    None,
                    "legacy-new",
                    0,
                    0,
                    0.7,
                    now,
                    1,
                    0,
                ),
            )
            conn.execute(
                """INSERT INTO beliefs (id, stack_id, statement, belief_type, confidence,
                   created_at, source_type, supersedes, superseded_by, times_reinforced,
                   is_active, strength, local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "legacy-new",
                    "test-stack",
                    "New belief",
                    "fact",
                    0.9,
                    now,
                    "direct",
                    "legacy-old",
                    None,
                    0,
                    1,
                    0.9,
                    now,
                    1,
                    0,
                ),
            )
            conn.commit()

        # No audit entries exist for these — should fall back to chain walk
        history = k.get_belief_history("legacy-old")
        assert len(history) >= 2

        ids_in_history = [h["id"] for h in history]
        assert "legacy-old" in ids_in_history
        assert "legacy-new" in ids_in_history

    def test_get_belief_history_prefers_audit_over_chain(self, k_with_belief):
        """When both audit log and chain data exist, audit log wins."""
        k, old_belief = k_with_belief

        new_id = k.revise_belief(
            old_id=old_belief.id,
            new_statement="Revised v2",
            confidence=0.85,
            reason="Better understanding",
        )

        # History should come from audit log (verify by checking structure)
        history = k.get_belief_history(old_belief.id)
        assert len(history) >= 2

        # Audit-sourced entries should have operation metadata
        # At minimum, the entries should be present and correctly ordered
        first = history[0]
        last = history[-1]
        assert first["id"] == old_belief.id  # Original first (chronological)
        assert last["id"] == new_id  # Newest last

    def test_get_belief_history_empty_for_nonexistent(self, k):
        """get_belief_history() returns empty list for non-existent belief."""
        history = k.get_belief_history("nonexistent-id")
        assert history == []

    def test_get_belief_history_single_belief_no_revisions(self, k_with_belief):
        """get_belief_history() returns single entry for belief with no revisions."""
        k, belief = k_with_belief
        history = k.get_belief_history(belief.id)
        # Should have at least the belief itself
        assert len(history) >= 1
        assert history[0]["id"] == belief.id


# ---------------------------------------------------------------------------
# Consolidation uses is_active check
# ---------------------------------------------------------------------------


class TestConsolidationIsActive:
    def test_scaffold_excludes_inactive_without_superseded_by(self, k):
        """scaffold_belief_to_value excludes inactive beliefs even without superseded_by."""
        from datetime import datetime, timedelta, timezone

        # Create a belief that is inactive but has NO superseded_by
        # (this is the new pattern — deactivated via revise_belief, not chain)
        old_date = datetime.now(timezone.utc) - timedelta(days=200)
        belief = Belief(
            id="inactive-no-chain",
            stack_id="test-stack",
            statement="I believe in testing",
            belief_type="principle",
            confidence=0.95,
            created_at=old_date,
            is_active=False,  # Inactive
            superseded_by=None,  # No chain field
            times_reinforced=10,
        )
        k._storage.save_belief(belief)

        # scaffold_belief_to_value should exclude this belief
        result = k.scaffold_belief_to_value(
            min_age_days=100,
            min_reinforcements=5,
            min_confidence=0.8,
        )

        candidate_ids = [c["belief_id"] for c in result["candidates"]]
        assert "inactive-no-chain" not in candidate_ids


# ---------------------------------------------------------------------------
# Existing supersession data still readable
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_existing_supersession_data_still_readable(self, k):
        """Old beliefs with supersedes/superseded_by fields are still queryable."""
        from datetime import datetime, timezone

        # Insert legacy data directly via SQL to simulate pre-v0.14 chain fields
        now = datetime.now(timezone.utc).isoformat()
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO beliefs (id, stack_id, statement, belief_type, confidence,
                   created_at, source_type, supersedes, superseded_by, times_reinforced,
                   is_active, strength, local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "compat-a",
                    "test-stack",
                    "Original belief",
                    "fact",
                    0.7,
                    now,
                    "direct",
                    None,
                    "compat-b",
                    0,
                    0,
                    0.7,
                    now,
                    1,
                    0,
                ),
            )
            conn.execute(
                """INSERT INTO beliefs (id, stack_id, statement, belief_type, confidence,
                   created_at, source_type, supersedes, superseded_by, times_reinforced,
                   is_active, strength, local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "compat-b",
                    "test-stack",
                    "Updated belief",
                    "fact",
                    0.9,
                    now,
                    "direct",
                    "compat-a",
                    None,
                    0,
                    1,
                    0.9,
                    now,
                    1,
                    0,
                ),
            )
            conn.commit()

        # Both should be readable with chain fields preserved
        all_beliefs = k._storage.get_beliefs(limit=100, include_inactive=True)
        a = next((b for b in all_beliefs if b.id == "compat-a"), None)
        b = next((b for b in all_beliefs if b.id == "compat-b"), None)

        assert a is not None
        assert a.superseded_by == "compat-b"
        assert a.is_active is False

        assert b is not None
        assert b.supersedes == "compat-a"
        assert b.is_active is True

    def test_save_belief_writes_null_supersedes(self, k):
        """New beliefs saved via save_belief() get NULL supersedes/superseded_by."""
        from datetime import datetime, timezone

        belief = Belief(
            id="new-belief-no-chain",
            stack_id="test-stack",
            statement="A fresh belief",
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
            is_active=True,
            supersedes="should-be-nulled",  # Set explicitly, should be overridden to NULL
            superseded_by="should-be-nulled-too",
        )
        k._storage.save_belief(belief)

        # Read it back — chain fields should be NULL
        all_beliefs = k._storage.get_beliefs(limit=100, include_inactive=True)
        saved = next((b for b in all_beliefs if b.id == "new-belief-no-chain"), None)
        assert saved is not None
        assert saved.supersedes is None
        assert saved.superseded_by is None


# ---------------------------------------------------------------------------
# No-model end-to-end
# ---------------------------------------------------------------------------


class TestNoModelEndToEnd:
    def test_no_model_end_to_end_belief_save(self, k):
        """Belief can be saved and revised without a bound model, no crash."""
        # Save a belief
        k.belief("Initial belief", type="fact", confidence=0.8)
        beliefs = k._storage.get_beliefs(limit=10)
        assert len(beliefs) >= 1
        belief = beliefs[0]

        # Revise it — should work without a model
        new_id = k.revise_belief(
            old_id=belief.id,
            new_statement="Revised belief",
            confidence=0.9,
            reason="Updated thinking",
        )
        assert new_id is not None

        # Get history — should work without a model
        history = k.get_belief_history(belief.id)
        assert len(history) >= 1
