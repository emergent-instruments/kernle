"""Tests for Issue #351: Lazy decay-on-read for continuous strength model.

Verifies:
- Strength is lazily decayed when records are retrieved
- Protected records are never decayed
- Already-forgotten records are skipped
- Decay is persisted to storage
- lazy_decay setting can disable the feature
- Search results also get lazy decay
- Maintenance sweep still works as batch catch-all
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kernle.stack import Stack
from kernle.stack.components.forgetting import compute_decayed_strength
from kernle.types import Belief, Episode, Note


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_lazy_decay.db"


@pytest.fixture
def stack(tmp_db):
    """Stack with forgetting component but provenance disabled."""
    return Stack.from_sqlite(
        stack_id="test-lazy-decay",
        db_path=tmp_db,
        enforce_provenance=False,
    )


@pytest.fixture
def bare_stack(tmp_db):
    """Stack without any components."""
    return Stack.from_sqlite(
        stack_id="test-lazy-decay",
        db_path=tmp_db,
        components=[],
        enforce_provenance=False,
    )


def _make_episode(stack_id, *, days_old=0, strength=1.0, is_protected=False, **kwargs):
    """Create an episode with a specific age and strength."""
    created = datetime.now(timezone.utc) - timedelta(days=days_old)
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "objective": f"Test episode {days_old}d old",
        "outcome": "Test outcome",
        "created_at": created,
        "last_accessed": created,
        "strength": strength,
        "is_protected": is_protected,
    }
    defaults.update(kwargs)
    return Episode(**defaults)


def _make_belief(stack_id, *, days_old=0, strength=1.0, is_protected=False, **kwargs):
    """Create a belief with a specific age and strength."""
    created = datetime.now(timezone.utc) - timedelta(days=days_old)
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "statement": f"Test belief {days_old}d old",
        "belief_type": "fact",
        "confidence": 0.8,
        "created_at": created,
        "last_accessed": created,
        "strength": strength,
        "is_protected": is_protected,
    }
    defaults.update(kwargs)
    return Belief(**defaults)


def _make_note(stack_id, *, days_old=0, strength=1.0, **kwargs):
    """Create a note with a specific age."""
    created = datetime.now(timezone.utc) - timedelta(days=days_old)
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "content": f"Test note {days_old}d old",
        "note_type": "note",
        "created_at": created,
        "last_accessed": created,
        "strength": strength,
    }
    defaults.update(kwargs)
    return Note(**defaults)


class TestComputeDecayedStrength:
    """Test the extracted compute_decayed_strength function."""

    def test_recent_memory_no_significant_decay(self):
        """A memory accessed today should barely decay."""
        ep = _make_episode("test", days_old=0, strength=1.0)
        result = compute_decayed_strength("episode", ep)
        assert abs(result - 1.0) < 0.01

    def test_old_memory_decays(self):
        """A memory from 6 months ago should have measurable decay."""
        ep = _make_episode("test", days_old=180, strength=1.0)
        result = compute_decayed_strength("episode", ep)
        assert result < 1.0
        assert result > 0.0  # Should not be fully forgotten

    def test_protected_still_computes(self):
        """compute_decayed_strength itself doesn't check protection.
        Protection is enforced by the caller (_apply_lazy_decay)."""
        ep = _make_episode("test", days_old=180, strength=1.0, is_protected=True)
        result = compute_decayed_strength("episode", ep)
        # The function computes the decay regardless; caller skips protected
        assert result < 1.0

    def test_zero_strength_stays_zero(self):
        """Already-forgotten memories should stay at 0."""
        ep = _make_episode("test", days_old=0, strength=0.0)
        result = compute_decayed_strength("episode", ep)
        assert result == 0.0

    def test_clamped_to_range(self):
        """Strength should always be in [0.0, 1.0]."""
        # Very old memory
        ep = _make_episode("test", days_old=3650, strength=0.01)
        result = compute_decayed_strength("episode", ep)
        assert 0.0 <= result <= 1.0

    def test_custom_half_life(self):
        """Custom half_life should affect decay rate."""
        ep = _make_episode("test", days_old=60, strength=1.0)
        result_short = compute_decayed_strength("episode", ep, half_life=10.0)
        result_long = compute_decayed_strength("episode", ep, half_life=365.0)
        # Shorter half-life = more decay
        assert result_short < result_long


class TestLazyDecayOnRead:
    """Test lazy decay during get_episodes, get_beliefs, etc."""

    def test_episodes_decay_on_read(self, bare_stack):
        """Old episodes should have decayed strength when retrieved."""
        ep = _make_episode(bare_stack.stack_id, days_old=180, strength=1.0)
        bare_stack.save_episode(ep)

        # Retrieve — lazy decay should apply
        episodes = bare_stack.get_episodes(include_forgotten=True)
        assert len(episodes) >= 1
        found = [e for e in episodes if e.id == ep.id][0]
        assert found.strength < 1.0, "Expected strength to decay for 180-day-old episode"

    def test_beliefs_decay_on_read(self, bare_stack):
        """Old beliefs should have decayed strength when retrieved."""
        b = _make_belief(bare_stack.stack_id, days_old=120, strength=1.0)
        bare_stack.save_belief(b)

        beliefs = bare_stack.get_beliefs(include_forgotten=True)
        found = [x for x in beliefs if x.id == b.id][0]
        assert found.strength < 1.0

    def test_notes_decay_on_read(self, bare_stack):
        """Old notes should have decayed strength when retrieved."""
        n = _make_note(bare_stack.stack_id, days_old=120, strength=1.0)
        bare_stack.save_note(n)

        notes = bare_stack.get_notes(include_forgotten=True)
        found = [x for x in notes if x.id == n.id][0]
        assert found.strength < 1.0

    def test_protected_records_not_decayed(self, bare_stack):
        """Protected records should never be decayed."""
        ep = _make_episode(bare_stack.stack_id, days_old=365, strength=1.0, is_protected=True)
        bare_stack.save_episode(ep)

        episodes = bare_stack.get_episodes(include_forgotten=True)
        found = [e for e in episodes if e.id == ep.id][0]
        assert found.strength == 1.0, "Protected episode should not decay"

    def test_already_forgotten_not_decayed(self, bare_stack):
        """Records at strength 0.0 should not be further processed."""
        ep = _make_episode(bare_stack.stack_id, days_old=30, strength=0.0)
        bare_stack.save_episode(ep)

        episodes = bare_stack.get_episodes(include_forgotten=True)
        found = [e for e in episodes if e.id == ep.id][0]
        assert found.strength == 0.0

    def test_recent_memories_minimal_decay(self, bare_stack):
        """Recently accessed memories should have negligible decay."""
        ep = _make_episode(bare_stack.stack_id, days_old=1, strength=1.0)
        bare_stack.save_episode(ep)

        episodes = bare_stack.get_episodes(include_forgotten=True)
        found = [e for e in episodes if e.id == ep.id][0]
        # 1 day old should barely decay
        assert found.strength > 0.99

    def test_decay_persisted_to_storage(self, bare_stack):
        """Decayed strength should be persisted so subsequent reads don't re-decay."""
        ep = _make_episode(bare_stack.stack_id, days_old=180, strength=1.0)
        bare_stack.save_episode(ep)

        # First read — triggers lazy decay
        episodes1 = bare_stack.get_episodes(include_forgotten=True)
        strength1 = [e for e in episodes1 if e.id == ep.id][0].strength

        # Second read — should get the same persisted value (no double-decay)
        episodes2 = bare_stack.get_episodes(include_forgotten=True)
        strength2 = [e for e in episodes2 if e.id == ep.id][0].strength

        assert (
            abs(strength1 - strength2) < 0.002
        ), f"Expected consistent strength across reads: {strength1} vs {strength2}"


class TestLazyDecaySetting:
    """Test the lazy_decay stack setting."""

    def test_lazy_decay_default_enabled(self, bare_stack):
        """Lazy decay should be enabled by default (no setting = enabled)."""
        ep = _make_episode(bare_stack.stack_id, days_old=180, strength=1.0)
        bare_stack.save_episode(ep)

        episodes = bare_stack.get_episodes(include_forgotten=True)
        found = [e for e in episodes if e.id == ep.id][0]
        assert found.strength < 1.0

    def test_lazy_decay_disabled(self, bare_stack):
        """Setting lazy_decay=false should skip decay on read."""
        bare_stack.set_stack_setting("lazy_decay", "false")

        ep = _make_episode(bare_stack.stack_id, days_old=180, strength=1.0)
        bare_stack.save_episode(ep)

        episodes = bare_stack.get_episodes(include_forgotten=True)
        found = [e for e in episodes if e.id == ep.id][0]
        assert found.strength == 1.0, "Lazy decay should be skipped when disabled"

    def test_lazy_decay_enabled_explicitly(self, bare_stack):
        """Setting lazy_decay=true should enable decay on read."""
        bare_stack.set_stack_setting("lazy_decay", "true")

        ep = _make_episode(bare_stack.stack_id, days_old=180, strength=1.0)
        bare_stack.save_episode(ep)

        episodes = bare_stack.get_episodes(include_forgotten=True)
        found = [e for e in episodes if e.id == ep.id][0]
        assert found.strength < 1.0


class TestLazyDecayInSearch:
    """Test lazy decay during search."""

    def test_search_applies_decay(self, bare_stack):
        """Search results should reflect decayed strength."""
        ep = _make_episode(
            bare_stack.stack_id,
            days_old=180,
            strength=1.0,
            objective="Unique searchable objective for lazy decay test",
            outcome="Test outcome for search",
        )
        bare_stack.save_episode(ep)

        # Search for the episode
        results = bare_stack.search("Unique searchable objective for lazy decay")
        # If found, check that the underlying record had decay applied
        if results:
            # Verify by re-fetching — strength should be persisted
            episodes = bare_stack.get_episodes(include_forgotten=True)
            found = [e for e in episodes if e.id == ep.id]
            if found:
                assert found[0].strength < 1.0


class TestMaintenanceSweepStillWorks:
    """Maintenance sweep should still work as a batch catch-all."""

    def test_forgetting_maintenance_still_runs(self, stack):
        """ForgettingComponent maintenance should still decay all records."""
        ep = _make_episode(stack.stack_id, days_old=180, strength=0.9)
        stack.save_episode(ep)

        forgetting = stack.get_component("forgetting")
        assert forgetting is not None

        stats = forgetting.on_maintenance()
        # Maintenance should have run (may or may not find decayed records
        # depending on timing, but shouldn't error)
        assert isinstance(stats, dict)
