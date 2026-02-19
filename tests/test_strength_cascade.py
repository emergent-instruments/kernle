"""Tests for strength cascade through the provenance chain.

v0.10.0 introduces cascade behavior: changes to a memory's strength
can flag or affect memories derived from it.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.entity import Entity
from kernle.stack.sqlite_stack import Stack
from kernle.types import Belief, Episode, Goal, Note, RawEntry, Value


def _id():
    return str(uuid.uuid4())


def _make_episode(stack_id, derived_from=None, strength=1.0):
    return Episode(
        id=_id(),
        stack_id=stack_id,
        objective="test objective",
        outcome="test outcome",
        created_at=datetime.now(timezone.utc),
        source_type="direct_experience",
        derived_from=derived_from,
        strength=strength,
    )


def _make_belief(stack_id, derived_from=None, strength=1.0):
    return Belief(
        id=_id(),
        stack_id=stack_id,
        statement="test belief",
        belief_type="fact",
        confidence=0.8,
        created_at=datetime.now(timezone.utc),
        source_type="direct_experience",
        derived_from=derived_from,
        strength=strength,
    )


def _make_note(stack_id, derived_from=None, strength=1.0):
    return Note(
        id=_id(),
        stack_id=stack_id,
        content="test note",
        note_type="note",
        created_at=datetime.now(timezone.utc),
        source_type="direct_experience",
        derived_from=derived_from,
        strength=strength,
    )


def _make_value(stack_id, derived_from=None, strength=1.0):
    return Value(
        id=_id(),
        stack_id=stack_id,
        name="test value",
        statement="test statement",
        priority=50,
        created_at=datetime.now(timezone.utc),
        source_type="direct_experience",
        derived_from=derived_from,
        strength=strength,
    )


def _make_goal(stack_id, derived_from=None, strength=1.0):
    return Goal(
        id=_id(),
        stack_id=stack_id,
        title="test goal",
        goal_type="task",
        priority="medium",
        created_at=datetime.now(timezone.utc),
        source_type="direct_experience",
        derived_from=derived_from,
        strength=strength,
    )


@pytest.fixture
def stack(tmp_path):
    """Create an Stack with bare components for testing."""
    return Stack.from_sqlite(
        stack_id="test-cascade",
        db_path=tmp_path / "test.db",
        components=[],
    )


@pytest.fixture
def entity_with_stack(tmp_path):
    """Create an Entity with an attached Stack."""

    entity = Entity(core_id="test-entity", data_dir=tmp_path)
    stack = Stack.from_sqlite(
        stack_id="test-cascade",
        db_path=tmp_path / "test.db",
        components=[],
        enforce_provenance=False,
    )
    entity.attach_stack(stack, alias="default")
    return entity, stack


class TestGetMemoriesDerivedFrom:
    """Test get_memories_derived_from finds correct children."""

    def test_finds_belief_derived_from_episode(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        children = stack.get_memories_derived_from("episode", ep_id)
        assert len(children) == 1
        assert children[0] == ("belief", belief_id)

    def test_finds_multiple_children(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        b1 = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        b1_id = stack.save_belief(b1)

        b2 = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        b2_id = stack.save_belief(b2)

        goal = _make_goal(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        goal_id = stack.save_goal(goal)

        children = stack.get_memories_derived_from("episode", ep_id)
        child_set = {(t, i) for t, i in children}
        assert ("belief", b1_id) in child_set
        assert ("belief", b2_id) in child_set
        assert ("goal", goal_id) in child_set
        assert len(children) == 3

    def test_does_not_find_unrelated_memories(self, stack):
        ep1 = _make_episode(stack.stack_id)
        ep1_id = stack.save_episode(ep1)

        ep2 = _make_episode(stack.stack_id)
        ep2_id = stack.save_episode(ep2)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep2_id}"])
        stack.save_belief(belief)

        children = stack.get_memories_derived_from("episode", ep1_id)
        assert len(children) == 0

    def test_finds_across_memory_types(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        note = _make_note(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        stack.save_note(note)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        stack.save_belief(belief)

        children = stack.get_memories_derived_from("episode", ep_id)
        child_types = {t for t, _ in children}
        assert "note" in child_types
        assert "belief" in child_types

    def test_skips_annotation_refs(self, stack):
        """Memories with only context: or kernle: refs should not be found."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        # This belief cites context:cli, not ep_id
        belief = _make_belief(stack.stack_id, derived_from=["context:cli"])
        stack.save_belief(belief)

        children = stack.get_memories_derived_from("episode", ep_id)
        assert len(children) == 0

    def test_returns_empty_for_unknown_type(self, stack):
        children = stack.get_memories_derived_from("unknown_type", "some-id")
        assert children == []


class TestForgetCascade:
    """Test forget creates cascade audit entries for children."""

    def test_forget_creates_cascade_audit_entries(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        stack.forget_memory("episode", ep_id, "testing cascade")

        audits = stack.get_audit_log(memory_id=belief_id, operation="cascade_flag")
        assert len(audits) == 1
        assert audits[0]["details"]["cascade_source"] == f"episode:{ep_id}"
        assert audits[0]["details"]["reason"] == "source_forgotten"

    def test_forget_flags_multiple_children(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        b1 = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        b1_id = stack.save_belief(b1)

        b2 = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        b2_id = stack.save_belief(b2)

        stack.forget_memory("episode", ep_id, "testing cascade")

        audits1 = stack.get_audit_log(memory_id=b1_id, operation="cascade_flag")
        audits2 = stack.get_audit_log(memory_id=b2_id, operation="cascade_flag")
        assert len(audits1) == 1
        assert len(audits2) == 1

    def test_forget_does_not_modify_child_strength(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        stack.forget_memory("episode", ep_id, "testing cascade")

        # Child strength should remain unchanged (cascade is advisory)
        child = stack.get_memory("belief", belief_id)
        assert child.strength == 1.0

    def test_forget_no_cascade_when_no_children(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        stack.forget_memory("episode", ep_id, "testing cascade")

        audits = stack.get_audit_log(operation="cascade_flag")
        assert len(audits) == 0

    def test_forget_no_cascade_when_forget_fails(self, stack):
        """If memory doesn't exist, no cascade should happen."""
        result = stack.forget_memory("episode", "nonexistent-id", "testing")
        assert result is False

        audits = stack.get_audit_log(operation="cascade_flag")
        assert len(audits) == 0


class TestWeakenCascade:
    """Test weaken cascade behavior around the 0.2 dormant threshold."""

    def test_weaken_below_threshold_creates_cascade(self, stack):
        """Weakening from 0.5 to 0.1 (below 0.2) should cascade."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)
        # Set initial strength to 0.5
        stack._backend.weaken_memory("episode", ep_id, 0.5)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        # Weaken by 0.4: 0.5 -> 0.1 (below 0.2 threshold)
        stack.weaken_memory("episode", ep_id, 0.4)

        audits = stack.get_audit_log(memory_id=belief_id, operation="cascade_flag")
        assert len(audits) == 1
        assert audits[0]["details"]["reason"] == "source_dormant"

    def test_weaken_above_threshold_no_cascade(self, stack):
        """Weakening from 1.0 to 0.5 (above 0.2) should NOT cascade."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        # Weaken by 0.5: 1.0 -> 0.5 (above 0.2 threshold)
        stack.weaken_memory("episode", ep_id, 0.5)

        audits = stack.get_audit_log(memory_id=belief_id, operation="cascade_flag")
        assert len(audits) == 0

    def test_weaken_already_below_threshold_no_cascade(self, stack):
        """Weakening from 0.1 to 0.05 should NOT cascade (already below 0.2)."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)
        # Set to 0.1 first
        stack._backend.weaken_memory("episode", ep_id, 0.9)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        stack.save_belief(belief)

        # Weaken by 0.05: 0.1 -> 0.05 (already below threshold)
        stack.weaken_memory("episode", ep_id, 0.05)

        audits = stack.get_audit_log(operation="cascade_flag")
        assert len(audits) == 0

    def test_weaken_does_not_modify_child_strength(self, stack):
        """Cascade is advisory only — child strength should not change."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)
        stack._backend.weaken_memory("episode", ep_id, 0.5)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        stack.weaken_memory("episode", ep_id, 0.4)

        child = stack.get_memory("belief", belief_id)
        assert child.strength == 1.0


class TestVerifyBoost:
    """Test verify boosts source memory strength."""

    def test_verify_boosts_source_strength(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)
        # Set episode to 0.5 strength
        stack._backend.weaken_memory("episode", ep_id, 0.5)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        # Verify the belief
        stack.verify_memory("belief", belief_id)

        # Source episode should get +0.02 boost: 0.5 -> 0.52
        source = stack.get_memory("episode", ep_id)
        assert abs(source.strength - 0.52) < 0.001

    def test_verify_boosts_multiple_sources(self, stack):
        ep1 = _make_episode(stack.stack_id)
        ep1_id = stack.save_episode(ep1)
        stack._backend.weaken_memory("episode", ep1_id, 0.5)

        ep2 = _make_episode(stack.stack_id)
        ep2_id = stack.save_episode(ep2)
        stack._backend.weaken_memory("episode", ep2_id, 0.3)

        belief = _make_belief(
            stack.stack_id,
            derived_from=[f"episode:{ep1_id}", f"episode:{ep2_id}"],
        )
        belief_id = stack.save_belief(belief)

        stack.verify_memory("belief", belief_id)

        src1 = stack.get_memory("episode", ep1_id)
        src2 = stack.get_memory("episode", ep2_id)
        assert abs(src1.strength - 0.52) < 0.001
        assert abs(src2.strength - 0.72) < 0.001

    def test_verify_skips_annotation_refs(self, stack):
        """Annotation refs (context:, kernle:) should not be boosted."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)
        stack._backend.weaken_memory("episode", ep_id, 0.5)

        belief = _make_belief(
            stack.stack_id,
            derived_from=[f"episode:{ep_id}", "context:cli", "kernle:system"],
        )
        belief_id = stack.save_belief(belief)

        stack.verify_memory("belief", belief_id)

        # Only the episode should get boosted
        source = stack.get_memory("episode", ep_id)
        assert abs(source.strength - 0.52) < 0.001

    def test_verify_caps_at_1_0(self, stack):
        """Source strength should not exceed 1.0."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)
        # Episode at full strength (1.0)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        stack.verify_memory("belief", belief_id)

        source = stack.get_memory("episode", ep_id)
        assert source.strength <= 1.0

    def test_verify_no_boost_without_derived_from(self, stack):
        """Verifying a memory with no derived_from should not error."""
        belief = _make_belief(stack.stack_id, derived_from=None)
        belief_id = stack.save_belief(belief)

        result = stack.verify_memory("belief", belief_id)
        assert result is True


class TestGetUngroundedMemories:
    """Test get_ungrounded_memories returns correct results."""

    def test_finds_ungrounded_after_source_forgotten(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        # Forget the source episode
        stack.forget_memory("episode", ep_id, "testing ungrounded")

        ungrounded = stack.get_ungrounded_memories()
        assert len(ungrounded) == 1
        assert ungrounded[0][0] == "belief"
        assert ungrounded[0][1] == belief_id
        assert f"episode:{ep_id}" in ungrounded[0][2]

    def test_not_ungrounded_if_source_alive(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        stack.save_belief(belief)

        ungrounded = stack.get_ungrounded_memories()
        assert len(ungrounded) == 0

    def test_not_ungrounded_if_any_source_alive(self, stack):
        """If ANY source is alive, memory is grounded."""
        ep1 = _make_episode(stack.stack_id)
        ep1_id = stack.save_episode(ep1)

        ep2 = _make_episode(stack.stack_id)
        ep2_id = stack.save_episode(ep2)

        belief = _make_belief(
            stack.stack_id,
            derived_from=[f"episode:{ep1_id}", f"episode:{ep2_id}"],
        )
        stack.save_belief(belief)

        # Forget only one source
        stack.forget_memory("episode", ep1_id, "partial forget")

        ungrounded = stack.get_ungrounded_memories()
        assert len(ungrounded) == 0

    def test_ungrounded_when_all_sources_dead(self, stack):
        ep1 = _make_episode(stack.stack_id)
        ep1_id = stack.save_episode(ep1)

        ep2 = _make_episode(stack.stack_id)
        ep2_id = stack.save_episode(ep2)

        belief = _make_belief(
            stack.stack_id,
            derived_from=[f"episode:{ep1_id}", f"episode:{ep2_id}"],
        )
        belief_id = stack.save_belief(belief)

        # Forget both sources
        stack.forget_memory("episode", ep1_id, "forget 1")
        stack.forget_memory("episode", ep2_id, "forget 2")

        ungrounded = stack.get_ungrounded_memories()
        assert len(ungrounded) == 1
        assert ungrounded[0][1] == belief_id

    def test_ignores_annotation_refs(self, stack):
        """Memories with only annotation refs should not be flagged."""
        belief = _make_belief(
            stack.stack_id,
            derived_from=["context:cli", "kernle:system"],
        )
        stack.save_belief(belief)

        ungrounded = stack.get_ungrounded_memories()
        assert len(ungrounded) == 0

    def test_ungrounded_when_source_missing(self, stack):
        """A memory referencing a non-existent source is ungrounded."""
        belief = _make_belief(
            stack.stack_id,
            derived_from=["episode:nonexistent-id"],
        )
        belief_id = stack.save_belief(belief)

        ungrounded = stack.get_ungrounded_memories()
        assert len(ungrounded) == 1
        assert ungrounded[0][1] == belief_id

    def test_not_ungrounded_if_raw_source_exists(self, stack):
        """Episodes derived from existing raw entries are NOT ungrounded."""
        raw = RawEntry(
            id=_id(),
            stack_id=stack.stack_id,
            blob="test raw entry",
            source="test",
        )
        raw_id = stack.save_raw(raw)

        ep = _make_episode(stack.stack_id, derived_from=[f"raw:{raw_id}"])
        stack.save_episode(ep)

        ungrounded = stack.get_ungrounded_memories()
        assert len(ungrounded) == 0

    def test_ungrounded_if_raw_source_deleted(self, stack):
        """Episodes derived from deleted raw entries ARE ungrounded."""
        raw = RawEntry(
            id=_id(),
            stack_id=stack.stack_id,
            blob="test raw entry",
            source="test",
        )
        raw_id = stack.save_raw(raw)

        ep = _make_episode(stack.stack_id, derived_from=[f"raw:{raw_id}"])
        ep_id = stack.save_episode(ep)

        # Delete the raw entry
        stack._backend.delete_raw(raw_id)

        ungrounded = stack.get_ungrounded_memories()
        assert len(ungrounded) == 1
        assert ungrounded[0][1] == ep_id


class TestCascadeDepth:
    """Test that cascade depth is 1 (no recursive cascade)."""

    def test_cascade_does_not_recurse(self, stack):
        """Forgetting a grandparent should only flag direct children."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        value = _make_value(stack.stack_id, derived_from=[f"belief:{belief_id}"])
        value_id = stack.save_value(value)

        # Forget the episode (grandparent of value)
        stack.forget_memory("episode", ep_id, "testing depth")

        # Belief (direct child) should be flagged
        belief_audits = stack.get_audit_log(memory_id=belief_id, operation="cascade_flag")
        assert len(belief_audits) == 1

        # Value (grandchild) should NOT be flagged
        value_audits = stack.get_audit_log(memory_id=value_id, operation="cascade_flag")
        assert len(value_audits) == 0


class TestEntityCascadeWiring:
    """Test cascade methods are wired through Entity."""

    def test_entity_get_memories_derived_from(self, entity_with_stack):
        entity, stack = entity_with_stack

        ep_id = entity.episode("test", "outcome")
        belief_id = entity.belief("test belief", derived_from=[f"episode:{ep_id}"])

        children = entity.get_memories_derived_from("episode", ep_id)
        assert len(children) == 1
        assert children[0] == ("belief", belief_id)

    def test_entity_get_ungrounded_memories(self, entity_with_stack):
        entity, stack = entity_with_stack

        ep_id = entity.episode("test", "outcome")
        belief_id = entity.belief("test belief", derived_from=[f"episode:{ep_id}"])

        entity.forget("episode", ep_id, "testing ungrounded")

        ungrounded = entity.get_ungrounded_memories()
        assert len(ungrounded) == 1
        assert ungrounded[0][1] == belief_id

    def test_entity_forget_cascades(self, entity_with_stack):
        entity, stack = entity_with_stack

        ep_id = entity.episode("test", "outcome")
        belief_id = entity.belief("test belief", derived_from=[f"episode:{ep_id}"])

        entity.forget("episode", ep_id, "testing cascade")

        audits = stack.get_audit_log(memory_id=belief_id, operation="cascade_flag")
        assert len(audits) == 1

    def test_entity_verify_boosts_source(self, entity_with_stack):
        entity, stack = entity_with_stack

        ep_id = entity.episode("test", "outcome")
        # Weaken episode first
        stack.weaken_memory("episode", ep_id, 0.5)

        belief_id = entity.belief("test belief", derived_from=[f"episode:{ep_id}"])

        entity.verify("belief", belief_id)

        source = stack.get_memory("episode", ep_id)
        assert abs(source.strength - 0.52) < 0.001


class TestBoostMemoryStrength:
    """Test the boost_memory_strength storage method."""

    def test_boost_increases_strength(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)
        stack._backend.weaken_memory("episode", ep_id, 0.5)

        result = stack._backend.boost_memory_strength("episode", ep_id, 0.1)
        assert result is True

        memory = stack.get_memory("episode", ep_id)
        assert abs(memory.strength - 0.6) < 0.001

    def test_boost_caps_at_1_0(self, stack):
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        result = stack._backend.boost_memory_strength("episode", ep_id, 0.5)
        assert result is True

        memory = stack.get_memory("episode", ep_id)
        assert memory.strength <= 1.0

    def test_boost_returns_false_for_missing(self, stack):
        result = stack._backend.boost_memory_strength("episode", "nonexistent", 0.1)
        assert result is False

    def test_boost_returns_false_for_unknown_type(self, stack):
        result = stack._backend.boost_memory_strength("unknown_type", "some-id", 0.1)
        assert result is False
