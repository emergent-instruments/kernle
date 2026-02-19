"""Tests for continuous memory strength mechanics (v0.9.0 PR 4).

Tests cover:
- update_strength / update_strength_batch in SQLiteStorage
- get_all_active_memories in SQLiteStorage
- record_access strength boost in SQLiteStorage
- Strength-based filtering in Stack (get_*, search)
- ForgettingComponent on_maintenance strength decay + persistence
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from kernle.stack.components.forgetting import ForgettingComponent
from kernle.stack.sqlite_stack import Stack
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Belief, Episode, Goal, Note, Value

# ---- Fixtures ----

STACK_ID = "test-stack"


@pytest.fixture
def storage(tmp_path):
    """Create a fresh SQLiteStorage instance."""
    db_path = tmp_path / "test.db"
    s = SQLiteStorage(STACK_ID, db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def stack(tmp_path):
    """Create a bare Stack (no components) for testing."""
    db_path = tmp_path / "test.db"
    return Stack.from_sqlite(STACK_ID, db_path=db_path, components=[], enforce_provenance=False)


# ---- Helpers ----


def _ep(objective="Test episode", outcome="It happened"):
    return Episode(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        objective=objective,
        outcome=outcome,
        source_type="observation",
        source_entity="test",
    )


def _belief(statement="Test belief"):
    return Belief(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        statement=statement,
        belief_type="factual",
        confidence=0.8,
        source_type="inference",
        source_entity="test",
    )


def _note(content="Test note"):
    return Note(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        content=content,
        note_type="observation",
    )


def _goal(title="Test goal"):
    return Goal(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        title=title,
        description="A test goal",
        goal_type="task",
        priority="medium",
        source_type="inference",
    )


def _save_episode(storage, **kwargs):
    ep = _ep(**kwargs)
    storage.save_episode(ep)
    return ep.id


def _save_belief(storage, **kwargs):
    b = _belief(**kwargs)
    storage.save_belief(b)
    return b.id


def _save_note(storage, **kwargs):
    n = _note(**kwargs)
    storage.save_note(n)
    return n.id


def _save_goal(storage, **kwargs):
    g = _goal(**kwargs)
    storage.save_goal(g)
    return g.id


# ==============================================================================
# SQLiteStorage.update_strength
# ==============================================================================


class TestUpdateStrength:
    """Tests for SQLiteStorage.update_strength()."""

    def test_update_episode_strength(self, storage):
        eid = _save_episode(storage)
        assert storage.update_strength("episode", eid, 0.5)
        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(0.5)

    def test_update_belief_strength(self, storage):
        bid = _save_belief(storage)
        assert storage.update_strength("belief", bid, 0.3)
        b = storage.get_belief(bid)
        assert b.strength == pytest.approx(0.3)

    def test_update_note_strength(self, storage):
        nid = _save_note(storage)
        assert storage.update_strength("note", nid, 0.7)
        notes = storage.get_notes(limit=100)
        note = next(n for n in notes if n.id == nid)
        assert note.strength == pytest.approx(0.7)

    def test_update_goal_strength(self, storage):
        gid = _save_goal(storage)
        assert storage.update_strength("goal", gid, 0.1)
        goals = storage.get_goals(limit=100)
        goal = next(g for g in goals if g.id == gid)
        assert goal.strength == pytest.approx(0.1)

    def test_clamps_to_zero(self, storage):
        eid = _save_episode(storage)
        assert storage.update_strength("episode", eid, -0.5)
        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(0.0)

    def test_clamps_to_one(self, storage):
        eid = _save_episode(storage)
        assert storage.update_strength("episode", eid, 1.5)
        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(1.0)

    def test_nonexistent_memory_returns_false(self, storage):
        assert not storage.update_strength("episode", "no-such-id", 0.5)

    def test_invalid_type_returns_false(self, storage):
        assert not storage.update_strength("invalid_type", "some-id", 0.5)


# ==============================================================================
# SQLiteStorage.update_strength_batch
# ==============================================================================


class TestUpdateStrengthBatch:
    """Tests for SQLiteStorage.update_strength_batch()."""

    def test_batch_update_multiple_types(self, storage):
        eid = _save_episode(storage)
        bid = _save_belief(storage)
        nid = _save_note(storage)

        count = storage.update_strength_batch(
            [
                ("episode", eid, 0.5),
                ("belief", bid, 0.3),
                ("note", nid, 0.7),
            ]
        )
        assert count == 3

        assert storage.get_episode(eid).strength == pytest.approx(0.5)
        assert storage.get_belief(bid).strength == pytest.approx(0.3)
        notes = storage.get_notes(limit=100)
        note = next(n for n in notes if n.id == nid)
        assert note.strength == pytest.approx(0.7)

    def test_empty_batch_returns_zero(self, storage):
        assert storage.update_strength_batch([]) == 0

    def test_batch_clamps_values(self, storage):
        eid = _save_episode(storage)
        bid = _save_belief(storage)

        storage.update_strength_batch(
            [
                ("episode", eid, -1.0),
                ("belief", bid, 2.0),
            ]
        )

        assert storage.get_episode(eid).strength == pytest.approx(0.0)
        assert storage.get_belief(bid).strength == pytest.approx(1.0)

    def test_batch_skips_invalid_types(self, storage):
        eid = _save_episode(storage)

        count = storage.update_strength_batch(
            [
                ("episode", eid, 0.5),
                ("invalid_type", "some-id", 0.5),
            ]
        )
        assert count == 1


# ==============================================================================
# SQLiteStorage.get_all_active_memories
# ==============================================================================


class TestGetAllActiveMemories:
    """Tests for SQLiteStorage.get_all_active_memories()."""

    def test_returns_active_memories(self, storage):
        eid = _save_episode(storage)
        bid = _save_belief(storage)

        results = storage.get_all_active_memories()
        types_found = {mtype for mtype, _ in results}
        ids_found = {r.id for _, r in results}

        assert "episode" in types_found
        assert "belief" in types_found
        assert eid in ids_found
        assert bid in ids_found

    def test_excludes_forgotten(self, storage):
        eid = _save_episode(storage)
        storage.update_strength("episode", eid, 0.0)

        results = storage.get_all_active_memories()
        ids_found = {r.id for _, r in results}
        assert eid not in ids_found

    def test_excludes_protected(self, storage):
        """Protected memories should be excluded from decay sweep."""
        eid = _save_episode(storage)
        with storage._connect() as conn:
            conn.execute("UPDATE episodes SET is_protected = 1 WHERE id = ?", (eid,))
            conn.commit()

        results = storage.get_all_active_memories()
        ids_found = {r.id for _, r in results}
        assert eid not in ids_found

    def test_filter_by_type(self, storage):
        _save_episode(storage)
        _save_belief(storage)

        results = storage.get_all_active_memories(memory_types=["episode"])
        types_found = {mtype for mtype, _ in results}
        assert types_found == {"episode"}

    def test_empty_when_no_memories(self, storage):
        results = storage.get_all_active_memories()
        assert results == []


# ==============================================================================
# SQLiteStorage.record_access strength boost
# ==============================================================================


class TestRecordAccessStrengthBoost:
    """Tests for record_access() strength boost on access."""

    def test_access_boosts_strength(self, storage):
        eid = _save_episode(storage)
        storage.update_strength("episode", eid, 0.5)

        storage.record_access("episode", eid)

        ep = storage.get_episode(eid)
        assert ep.strength > 0.5
        assert ep.times_accessed == 1

    def test_access_boost_diminishes(self, storage):
        eid = _save_episode(storage)
        storage.update_strength("episode", eid, 0.5)

        # First access: times_accessed goes from 0→1, boost = 0.02 / (1 + 0/10) = 0.02
        storage.record_access("episode", eid)
        ep1 = storage.get_episode(eid)
        boost1 = ep1.strength - 0.5

        # Reset strength, access again: times_accessed now 1→2, boost = 0.02 / (1 + 1/10) ≈ 0.0182
        storage.update_strength("episode", eid, 0.5)
        storage.record_access("episode", eid)
        ep2 = storage.get_episode(eid)
        boost2 = ep2.strength - 0.5

        assert boost2 < boost1

    def test_access_caps_at_one(self, storage):
        eid = _save_episode(storage)
        # Default strength is 1.0
        storage.record_access("episode", eid)
        ep = storage.get_episode(eid)
        assert ep.strength <= 1.0


# ==============================================================================
# Stack strength-based filtering in get_* methods
# ==============================================================================


class TestStackStrengthFiltering:
    """Tests for strength-based filtering in Stack get_* methods."""

    def test_get_episodes_excludes_forgotten(self, stack):
        ep = _ep()
        eid = stack.save_episode(ep)
        stack._backend.update_strength("episode", eid, 0.0)

        episodes = stack.get_episodes()
        assert len(episodes) == 0

    def test_get_episodes_includes_forgotten_when_asked(self, stack):
        ep = _ep()
        eid = stack.save_episode(ep)
        stack._backend.update_strength("episode", eid, 0.0)

        episodes = stack.get_episodes(include_forgotten=True)
        assert len(episodes) == 1

    def test_get_beliefs_excludes_forgotten(self, stack):
        b = _belief()
        bid = stack.save_belief(b)
        stack._backend.update_strength("belief", bid, 0.0)

        beliefs = stack.get_beliefs()
        assert len(beliefs) == 0

    def test_get_beliefs_includes_forgotten_when_asked(self, stack):
        b = _belief()
        bid = stack.save_belief(b)
        stack._backend.update_strength("belief", bid, 0.0)

        beliefs = stack.get_beliefs(include_forgotten=True)
        assert len(beliefs) == 1

    def test_get_values_excludes_forgotten(self, stack):
        v = Value(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            name="test-val",
            statement="Test value",
            source_type="inference",
        )
        vid = stack.save_value(v)
        stack._backend.update_strength("value", vid, 0.0)

        values = stack.get_values()
        assert len(values) == 0

    def test_get_goals_excludes_forgotten(self, stack):
        g = _goal()
        gid = stack.save_goal(g)
        stack._backend.update_strength("goal", gid, 0.0)

        goals = stack.get_goals()
        assert len(goals) == 0

    def test_get_notes_excludes_forgotten(self, stack):
        n = _note()
        nid = stack.save_note(n)
        stack._backend.update_strength("note", nid, 0.0)

        notes = stack.get_notes()
        assert len(notes) == 0

    def test_low_strength_still_included(self, stack):
        """Memories with low (but non-zero) strength should still appear with include_forgotten."""
        ep = _ep()
        eid = stack.save_episode(ep)
        stack._backend.update_strength("episode", eid, 0.01)

        episodes = stack.get_episodes(include_forgotten=True)
        assert len(episodes) == 1


# ==============================================================================
# Stack search() strength filtering
# ==============================================================================


class TestSearchStrengthFiltering:
    """Tests for search() excluding forgotten/dormant memories."""

    def test_search_excludes_zero_strength(self, stack):
        ep = _ep(objective="unique findme test")
        eid = stack.save_episode(ep)
        stack._backend.update_strength("episode", eid, 0.0)

        results = stack.search("unique findme test")
        ids_found = {r.memory_id for r in results}
        assert eid not in ids_found

    def test_search_includes_positive_strength(self, stack):
        ep = _ep(objective="unique searchtest phrase")
        eid = stack.save_episode(ep)
        stack._backend.update_strength("episode", eid, 0.5)

        results = stack.search("unique searchtest phrase")
        ids_found = {r.memory_id for r in results}
        assert eid in ids_found


# ==============================================================================
# ForgettingComponent on_maintenance strength decay
# ==============================================================================


class TestForgettingMaintenanceDecay:
    """Tests for ForgettingComponent.on_maintenance() strength decay."""

    def _make_component_with_storage(self, storage):
        comp = ForgettingComponent()
        comp.attach("test-stack")
        comp.set_storage(storage)
        return comp

    def test_maintenance_returns_decay_stats(self, storage):
        _save_episode(storage)
        comp = self._make_component_with_storage(storage)

        result = comp.on_maintenance()
        assert "decayed" in result
        assert "candidates_found" in result
        assert "forgotten" in result
        assert "protected" in result

    def test_maintenance_no_storage_skips(self):
        comp = ForgettingComponent()
        comp.attach("test-stack")
        result = comp.on_maintenance()
        assert result["skipped"] is True

    def test_maintenance_decays_old_memories(self, storage):
        eid = _save_episode(storage)

        sixty_days_ago = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET created_at = ?, last_accessed = ? WHERE id = ?",
                (sixty_days_ago, sixty_days_ago, eid),
            )
            conn.commit()

        comp = self._make_component_with_storage(storage)
        result = comp.on_maintenance()

        assert result["decayed"] >= 1
        ep = storage.get_episode(eid)
        assert ep.strength < 1.0

    def test_maintenance_does_not_decay_recent_memories(self, storage):
        eid = _save_episode(storage)

        comp = self._make_component_with_storage(storage)
        comp.on_maintenance()

        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(1.0, abs=0.01)

    def test_maintenance_forgets_very_low_strength(self, storage):
        eid = _save_episode(storage)
        storage.update_strength("episode", eid, 0.1)

        comp = self._make_component_with_storage(storage)
        result = comp.on_maintenance()

        assert result["candidates_found"] >= 1

    def test_compute_decayed_strength_formula(self):
        """Verify the decay formula produces expected values."""
        comp = ForgettingComponent()
        comp.attach("test-stack")

        record = MagicMock()
        record.strength = 1.0
        record.times_accessed = 0
        record.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)
        record.created_at = datetime.now(timezone.utc) - timedelta(days=60)

        result = comp._compute_decayed_strength("episode", record)

        # days_since = 30 (using last_accessed)
        # half_life = 30 (default)
        # decay = 30/30 = 1.0
        # reinforcement = log(0+1) * 0.1 = 0
        # new_strength = 1.0 - (1.0 * 0.01) + 0 = 0.99
        assert result == pytest.approx(0.99, abs=0.01)

    def test_compute_decayed_strength_with_accesses(self):
        """Verify reinforcement from access count."""
        comp = ForgettingComponent()
        comp.attach("test-stack")

        record = MagicMock()
        record.strength = 0.5
        record.times_accessed = 100
        record.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)
        record.created_at = datetime.now(timezone.utc) - timedelta(days=60)

        result = comp._compute_decayed_strength("episode", record)

        expected = 0.5 - 0.01 + math.log(101) * 0.1 * 0.01
        assert result == pytest.approx(expected, abs=0.001)

    def test_compute_decayed_strength_goal_half_life(self):
        """Goal type overrides half-life."""
        comp = ForgettingComponent()
        comp.attach("test-stack")

        record = MagicMock()
        record.strength = 1.0
        record.times_accessed = 0
        record.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)
        record.created_at = None
        record.goal_type = "commitment"

        result = comp._compute_decayed_strength("goal", record)

        # half_life = 365 for commitment
        assert result > 0.998  # Very slow decay

    def test_compute_decayed_strength_clamps_to_zero(self):
        comp = ForgettingComponent()
        comp.attach("test-stack")

        record = MagicMock()
        record.strength = 0.01
        record.times_accessed = 0
        record.last_accessed = datetime.now(timezone.utc) - timedelta(days=365)
        record.created_at = None

        result = comp._compute_decayed_strength("episode", record)
        assert result >= 0.0

    def test_compute_decayed_strength_clamps_to_one(self):
        comp = ForgettingComponent()
        comp.attach("test-stack")

        record = MagicMock()
        record.strength = 0.999
        record.times_accessed = 10000
        record.last_accessed = datetime.now(timezone.utc)
        record.created_at = None

        result = comp._compute_decayed_strength("episode", record)
        assert result <= 1.0
