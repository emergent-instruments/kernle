"""Storage-level contract tests.

Verifies that a Storage backend correctly implements CRUD roundtrips,
settings, atomic updates, lineage queries, processing state, batch ops,
forgetting, and stats — all at the raw storage layer (no Stack).

These tests use the parameterized `storage` fixture from conftest.py.
External packages (e.g. kernle-supabase) can register their factory
in STORAGE_FACTORIES to run this entire suite against their backend.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.types import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    RawEntry,
    Relationship,
    Value,
)
from tests.contracts.conftest import CONTRACT_STACK_ID

STACK_ID = CONTRACT_STACK_ID


# ============================================================================
# Helpers
# ============================================================================


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_episode(**kw) -> Episode:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        objective="Learn Python",
        outcome="Built a web app",
        created_at=_now(),
        lessons=["Practice matters"],
    )
    defaults.update(kw)
    return Episode(**defaults)


def _make_belief(**kw) -> Belief:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        statement="Testing improves quality",
        belief_type="fact",
        confidence=0.85,
        created_at=_now(),
    )
    defaults.update(kw)
    return Belief(**defaults)


def _make_value(**kw) -> Value:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        name="Reliability",
        statement="Systems should be dependable",
        priority=75,
        created_at=_now(),
    )
    defaults.update(kw)
    return Value(**defaults)


def _make_goal(**kw) -> Goal:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        title="Ship v1.0",
        description="Release the first version",
        goal_type="task",
        priority="high",
        status="active",
        created_at=_now(),
    )
    defaults.update(kw)
    return Goal(**defaults)


def _make_note(**kw) -> Note:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        content="Important observation",
        note_type="insight",
        tags=["meta"],
        created_at=_now(),
    )
    defaults.update(kw)
    return Note(**defaults)


def _make_drive(**kw) -> Drive:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        drive_type="curiosity",
        intensity=0.7,
        focus_areas=["learning"],
        created_at=_now(),
    )
    defaults.update(kw)
    return Drive(**defaults)


def _make_relationship(**kw) -> Relationship:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        entity_name="alice",
        entity_type="human",
        relationship_type="collaborator",
        notes="Good partner",
        sentiment=0.6,
        created_at=_now(),
    )
    defaults.update(kw)
    return Relationship(**defaults)


def _make_raw(**kw) -> RawEntry:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        blob="Some raw content",
        captured_at=_now(),
        source="test",
    )
    defaults.update(kw)
    return RawEntry(**defaults)


# ============================================================================
# 1. CRUD Roundtrip Tests — each core memory type
# ============================================================================


class TestEpisodeCRUD:
    def test_save_and_retrieve(self, storage):
        ep = _make_episode()
        returned_id = storage.save_episode(ep)
        assert returned_id == ep.id

        episodes = storage.get_episodes(limit=10)
        found = [e for e in episodes if e.id == ep.id]
        assert len(found) == 1
        assert found[0].objective == "Learn Python"

    def test_get_by_id(self, storage):
        ep = _make_episode()
        storage.save_episode(ep)
        retrieved = storage.get_episode(ep.id)
        assert retrieved is not None
        assert retrieved.id == ep.id

    def test_get_episodes_empty(self, storage):
        episodes = storage.get_episodes(limit=10)
        assert episodes == []

    def test_filter_processed(self, storage):
        ep = _make_episode()
        storage.save_episode(ep)
        # Fresh episode is unprocessed
        unprocessed = storage.get_episodes(processed=False)
        assert any(e.id == ep.id for e in unprocessed)


class TestBeliefCRUD:
    def test_save_and_retrieve(self, storage):
        b = _make_belief()
        returned_id = storage.save_belief(b)
        assert returned_id == b.id

        beliefs = storage.get_beliefs(limit=10)
        found = [x for x in beliefs if x.id == b.id]
        assert len(found) == 1
        assert found[0].statement == "Testing improves quality"
        assert found[0].confidence == 0.85

    def test_find_belief_by_statement(self, storage):
        b = _make_belief(statement="Unique test statement XYZ")
        storage.save_belief(b)
        found = storage.find_belief("Unique test statement XYZ")
        assert found is not None
        assert found.id == b.id

    def test_find_belief_not_found(self, storage):
        found = storage.find_belief("Nonexistent statement")
        assert found is None


class TestValueCRUD:
    def test_save_and_retrieve(self, storage):
        v = _make_value()
        returned_id = storage.save_value(v)
        assert returned_id == v.id

        values = storage.get_values(limit=10)
        found = [x for x in values if x.id == v.id]
        assert len(found) == 1
        assert found[0].name == "Reliability"


class TestGoalCRUD:
    def test_save_and_retrieve(self, storage):
        g = _make_goal()
        returned_id = storage.save_goal(g)
        assert returned_id == g.id

        goals = storage.get_goals(limit=10)
        found = [x for x in goals if x.id == g.id]
        assert len(found) == 1
        assert found[0].title == "Ship v1.0"

    def test_filter_by_status(self, storage):
        g1 = _make_goal(status="active")
        g2 = _make_goal(status="completed")
        storage.save_goal(g1)
        storage.save_goal(g2)

        active = storage.get_goals(status="active")
        assert all(g.status == "active" for g in active)


class TestNoteCRUD:
    def test_save_and_retrieve(self, storage):
        n = _make_note()
        returned_id = storage.save_note(n)
        assert returned_id == n.id

        notes = storage.get_notes(limit=10)
        found = [x for x in notes if x.id == n.id]
        assert len(found) == 1
        assert found[0].content == "Important observation"

    def test_filter_by_type(self, storage):
        n1 = _make_note(note_type="insight")
        n2 = _make_note(note_type="decision")
        storage.save_note(n1)
        storage.save_note(n2)

        insights = storage.get_notes(note_type="insight")
        assert all(n.note_type == "insight" for n in insights)


class TestDriveCRUD:
    def test_save_and_retrieve(self, storage):
        d = _make_drive()
        returned_id = storage.save_drive(d)
        assert returned_id == d.id

        drives = storage.get_drives()
        found = [x for x in drives if x.id == d.id]
        assert len(found) == 1
        assert found[0].drive_type == "curiosity"

    def test_get_drive_by_type(self, storage):
        d = _make_drive(drive_type="exploration")
        storage.save_drive(d)
        retrieved = storage.get_drive("exploration")
        assert retrieved is not None
        assert retrieved.drive_type == "exploration"


class TestRelationshipCRUD:
    def test_save_and_retrieve(self, storage):
        r = _make_relationship()
        returned_id = storage.save_relationship(r)
        assert returned_id == r.id

        rels = storage.get_relationships()
        found = [x for x in rels if x.id == r.id]
        assert len(found) == 1
        assert found[0].entity_name == "alice"

    def test_get_by_entity_name(self, storage):
        r = _make_relationship(entity_name="bob")
        storage.save_relationship(r)
        retrieved = storage.get_relationship("bob")
        assert retrieved is not None
        assert retrieved.entity_name == "bob"

    def test_filter_by_entity_type(self, storage):
        r1 = _make_relationship(entity_name="a1", entity_type="human")
        r2 = _make_relationship(entity_name="a2", entity_type="agent")
        storage.save_relationship(r1)
        storage.save_relationship(r2)

        humans = storage.get_relationships(entity_type="human")
        assert all(r.entity_type == "human" for r in humans)


class TestRawCRUD:
    def test_save_and_list(self, storage):
        r = _make_raw()
        returned_id = storage.save_raw(blob=r.blob, source=r.source)
        assert isinstance(returned_id, str)
        assert len(returned_id) > 0

        raw_entries = storage.list_raw(limit=10)
        assert len(raw_entries) >= 1

    def test_get_by_id(self, storage):
        r = _make_raw()
        rid = storage.save_raw(blob=r.blob, source=r.source)
        retrieved = storage.get_raw(rid)
        assert retrieved is not None
        assert retrieved.blob == r.blob

    def test_mark_processed(self, storage):
        r = _make_raw()
        rid = storage.save_raw(blob=r.blob, source=r.source)
        result = storage.mark_raw_processed(rid, processed_into=["episode:ep-1"])
        assert result is True

        # After marking, it should be filtered out of unprocessed
        unprocessed = storage.list_raw(processed=False)
        assert not any(entry.id == rid for entry in unprocessed)


# ============================================================================
# 2. Settings Operations
# ============================================================================


class TestSettingsOperations:
    def test_set_and_get(self, storage):
        storage.set_stack_setting("test_key", "test_value")
        result = storage.get_stack_setting("test_key")
        assert result == "test_value"

    def test_get_nonexistent_returns_none(self, storage):
        result = storage.get_stack_setting("nonexistent_key")
        assert result is None

    def test_set_overwrites(self, storage):
        storage.set_stack_setting("key", "value1")
        storage.set_stack_setting("key", "value2")
        assert storage.get_stack_setting("key") == "value2"

    def test_get_all(self, storage):
        storage.set_stack_setting("k1", "v1")
        storage.set_stack_setting("k2", "v2")
        all_settings = storage.get_all_stack_settings()
        assert all_settings["k1"] == "v1"
        assert all_settings["k2"] == "v2"


# ============================================================================
# 3. Atomic Updates
# ============================================================================


class TestAtomicUpdates:
    def test_update_belief_atomic(self, storage):
        b = _make_belief(confidence=0.5)
        storage.save_belief(b)
        b.confidence = 0.99
        result = storage.update_belief_atomic(b)
        assert result is True

        updated = storage.find_belief(b.statement)
        assert updated is not None
        assert updated.confidence == pytest.approx(0.99, abs=0.01)

    def test_update_goal_atomic(self, storage):
        g = _make_goal(status="active")
        storage.save_goal(g)
        g.status = "completed"
        result = storage.update_goal_atomic(g)
        assert result is True

        goals = storage.get_goals(status="completed")
        assert any(x.id == g.id for x in goals)

    def test_update_drive_atomic(self, storage):
        d = _make_drive(intensity=0.5)
        storage.save_drive(d)
        d.intensity = 0.9
        result = storage.update_drive_atomic(d)
        assert result is True

        retrieved = storage.get_drive(d.drive_type)
        assert retrieved.intensity == pytest.approx(0.9, abs=0.01)

    def test_update_relationship_atomic(self, storage):
        r = _make_relationship(sentiment=0.3)
        storage.save_relationship(r)
        r.sentiment = 0.8
        result = storage.update_relationship_atomic(r)
        assert result is True

        retrieved = storage.get_relationship(r.entity_name)
        assert retrieved.sentiment == pytest.approx(0.8, abs=0.01)

    def test_update_episode_atomic(self, storage):
        ep = _make_episode(outcome="Original outcome")
        storage.save_episode(ep)
        ep.outcome = "Updated outcome"
        result = storage.update_episode_atomic(ep)
        assert result is True

        retrieved = storage.get_episode(ep.id)
        assert retrieved.outcome == "Updated outcome"


# ============================================================================
# 4. Processing State
# ============================================================================


class TestProcessingState:
    def test_mark_episode_processed(self, storage):
        ep = _make_episode()
        storage.save_episode(ep)
        result = storage.mark_episode_processed(ep.id)
        assert result is True

        # Verify it shows as processed
        processed = storage.get_episodes(processed=True)
        assert any(e.id == ep.id for e in processed)

    def test_mark_note_processed(self, storage):
        n = _make_note()
        storage.save_note(n)
        result = storage.mark_note_processed(n.id)
        assert result is True

    def test_mark_belief_processed(self, storage):
        b = _make_belief()
        storage.save_belief(b)
        result = storage.mark_belief_processed(b.id)
        assert result is True

    def test_mark_nonexistent_returns_false(self, storage):
        result = storage.mark_episode_processed("nonexistent-id")
        assert result is False


# ============================================================================
# 5. Lineage Queries
# ============================================================================


class TestLineageQueries:
    def test_get_memories_derived_from_empty(self, storage):
        result = storage.get_memories_derived_from("episode", "ep-nonexistent")
        assert result == []

    def test_get_memories_derived_from_with_data(self, storage):
        ep = _make_episode()
        storage.save_episode(ep)
        b = _make_belief(derived_from=[f"episode:{ep.id}"])
        storage.save_belief(b)

        result = storage.get_memories_derived_from("episode", ep.id)
        assert len(result) >= 1
        # Result is list of (child_type, child_id) tuples
        assert any(t[0] == "belief" and t[1] == b.id for t in result)

    def test_boost_memory_strength(self, storage):
        b = _make_belief(strength=0.5)
        storage.save_belief(b)
        result = storage.boost_memory_strength("belief", b.id, 0.2)
        assert result is True

        # Verify strength increased
        retrieved = storage.get_memory("belief", b.id)
        assert retrieved.strength == pytest.approx(0.7, abs=0.01)

    def test_boost_nonexistent_returns_false(self, storage):
        result = storage.boost_memory_strength("belief", "nonexistent-id", 0.1)
        assert result is False

    def test_log_belief_revision(self, storage):
        b_old = _make_belief(statement="Old belief")
        b_new = _make_belief(statement="New belief")
        storage.save_belief(b_old)
        storage.save_belief(b_new)

        result = storage.log_belief_revision(b_old.id, b_new.id, reason="updated evidence")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_ungrounded_memories(self, storage):
        result = storage.get_ungrounded_memories(storage.stack_id)
        assert isinstance(result, list)


# ============================================================================
# 6. Batch Operations
# ============================================================================


class TestBatchOperations:
    def test_save_episodes_batch(self, storage):
        eps = [_make_episode(objective=f"Task {i}") for i in range(3)]
        ids = storage.save_episodes_batch(eps)
        assert len(ids) == 3
        all_eps = storage.get_episodes(limit=10)
        assert len(all_eps) >= 3

    def test_save_beliefs_batch(self, storage):
        beliefs = [_make_belief(statement=f"Belief {i}") for i in range(3)]
        ids = storage.save_beliefs_batch(beliefs)
        assert len(ids) == 3

    def test_save_notes_batch(self, storage):
        notes = [_make_note(content=f"Note {i}") for i in range(3)]
        ids = storage.save_notes_batch(notes)
        assert len(ids) == 3


# ============================================================================
# 7. Stats
# ============================================================================


class TestStats:
    def test_empty_stats(self, storage):
        stats = storage.get_stats()
        assert isinstance(stats, dict)

    def test_stats_reflect_saved_data(self, storage):
        storage.save_episode(_make_episode())
        storage.save_belief(_make_belief())
        storage.save_value(_make_value())

        stats = storage.get_stats()
        assert stats.get("episodes", 0) >= 1
        assert stats.get("beliefs", 0) >= 1
        assert stats.get("values", 0) >= 1


# ============================================================================
# 8. Forgetting & Meta-Memory
# ============================================================================


class TestForgettingOperations:
    def test_forget_and_recover(self, storage):
        ep = _make_episode()
        storage.save_episode(ep)

        result = storage.forget_memory("episode", ep.id, reason="no longer needed")
        assert result is True

        # Recovered
        recovered = storage.recover_memory("episode", ep.id)
        assert recovered is True

    def test_protect_memory(self, storage):
        b = _make_belief()
        storage.save_belief(b)

        result = storage.protect_memory("belief", b.id, True)
        assert result is True

        # Unprotect
        result2 = storage.protect_memory("belief", b.id, False)
        assert result2 is True

    def test_record_access(self, storage):
        ep = _make_episode()
        storage.save_episode(ep)
        result = storage.record_access("episode", ep.id)
        assert result is True

    def test_update_strength(self, storage):
        b = _make_belief()
        storage.save_belief(b)
        result = storage.update_strength("belief", b.id, 0.3)
        assert result is True

        retrieved = storage.get_memory("belief", b.id)
        assert retrieved.strength == pytest.approx(0.3, abs=0.01)

    def test_update_strength_batch(self, storage):
        b1 = _make_belief(statement="Batch 1")
        b2 = _make_belief(statement="Batch 2")
        storage.save_belief(b1)
        storage.save_belief(b2)

        count = storage.update_strength_batch(
            [
                ("belief", b1.id, 0.2),
                ("belief", b2.id, 0.8),
            ]
        )
        assert count == 2


# ============================================================================
# 9. Search
# ============================================================================


class TestSearch:
    def test_search_empty(self, storage):
        results = storage.search("anything")
        assert results == []

    def test_search_finds_saved_content(self, storage):
        storage.save_episode(_make_episode(objective="Learn Rust programming"))
        results = storage.search("Rust")
        assert isinstance(results, list)
        # Text search should find it
        assert len(results) >= 1

    def test_search_with_limit(self, storage):
        for i in range(5):
            storage.save_note(_make_note(content=f"Topic alpha {i}"))
        results = storage.search("alpha", limit=2)
        assert len(results) <= 2


# ============================================================================
# 10. Meta-Memory Operations
# ============================================================================


class TestMetaMemoryOperations:
    def test_get_memory_by_type_and_id(self, storage):
        ep = _make_episode()
        storage.save_episode(ep)
        retrieved = storage.get_memory("episode", ep.id)
        assert retrieved is not None
        assert retrieved.id == ep.id

    def test_memory_exists(self, storage):
        ep = _make_episode()
        storage.save_episode(ep)
        assert storage.memory_exists("episode", ep.id) is True
        assert storage.memory_exists("episode", "nonexistent") is False

    def test_update_memory_meta_confidence(self, storage):
        b = _make_belief(confidence=0.5)
        storage.save_belief(b)
        result = storage.update_memory_meta("belief", b.id, confidence=0.95)
        assert result is True

        retrieved = storage.get_memory("belief", b.id)
        assert retrieved.confidence == pytest.approx(0.95, abs=0.01)


# ============================================================================
# 11. Stack Isolation
# ============================================================================


class TestStackIsolation:
    """Verify that storage respects stack_id boundaries."""

    def test_episodes_isolated_by_stack_id(self, storage, tmp_path):
        # Save an episode to the default stack
        ep = _make_episode()
        storage.save_episode(ep)

        # Verify the default stack can see it
        episodes = storage.get_episodes(limit=10)
        assert any(e.id == ep.id for e in episodes)

        # Create a second storage instance with a different stack_id
        # pointing at the same database file
        from kernle.storage.sqlite import SQLiteStorage

        other_storage = SQLiteStorage(
            stack_id="other-stack-isolation-test",
            db_path=tmp_path / "contract_test.db",
        )

        # The other stack must NOT see the episode
        other_episodes = other_storage.get_episodes(limit=10)
        assert not any(e.id == ep.id for e in other_episodes)

    def test_beliefs_isolated_by_stack_id(self, storage, tmp_path):
        b = _make_belief(statement="Isolation test belief")
        storage.save_belief(b)

        from kernle.storage.sqlite import SQLiteStorage

        other_storage = SQLiteStorage(
            stack_id="other-stack-isolation-test",
            db_path=tmp_path / "contract_test.db",
        )

        # Other stack should not find the belief
        found = other_storage.find_belief("Isolation test belief")
        assert found is None
