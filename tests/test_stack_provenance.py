"""Tests for stack lifecycle states and provenance validation.

Tests cover:
- Stack lifecycle state transitions (INITIALIZING -> ACTIVE -> MAINTENANCE)
- Provenance enforcement when enforce_provenance=True (default)
- Provenance bypass when enforce_provenance=False
- Hierarchy rules: each type must cite allowed source types
- ProvenanceError for missing, malformed, or invalid references
- MaintenanceModeError during maintenance mode
- memory_exists() in SQLiteStorage
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.protocols import (
    MaintenanceModeError,
    ProvenanceError,
    StackState,
)
from kernle.stack import SQLiteStack
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


def _uid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_provenance.db"


@pytest.fixture
def stack(tmp_db):
    """Stack with enforce_provenance=False, bare components."""
    return SQLiteStack(
        stack_id="test-stack", db_path=tmp_db, components=[], enforce_provenance=False
    )


@pytest.fixture
def enforced_stack(tmp_db):
    """Stack with enforce_provenance=True, bare components."""
    return SQLiteStack(
        stack_id="test-stack",
        db_path=tmp_db,
        components=[],
        enforce_provenance=True,
    )


# ---- Stack Lifecycle State Tests ----


class TestStackLifecycle:
    def test_initial_state_is_initializing(self, stack):
        assert stack.state == StackState.INITIALIZING

    def test_on_attach_transitions_to_active(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        assert stack.state == StackState.ACTIVE

    def test_enter_maintenance_from_active(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        assert stack.state == StackState.MAINTENANCE

    def test_exit_maintenance_returns_to_active(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        stack.exit_maintenance()
        assert stack.state == StackState.ACTIVE

    def test_enter_maintenance_idempotent(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        stack.enter_maintenance()  # Should not raise
        assert stack.state == StackState.MAINTENANCE

    def test_exit_maintenance_noop_when_not_in_maintenance(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        stack.exit_maintenance()  # Should not raise or change state
        assert stack.state == StackState.ACTIVE

    def test_writes_allowed_in_initializing(self, enforced_stack):
        """Even with provenance enforcement, writes are allowed in INITIALIZING."""
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Seed objective",
            outcome="Seed outcome",
            source_type="seed",
            created_at=_now(),
        )
        ep_id = enforced_stack.save_episode(ep)
        assert ep_id

    def test_maintenance_mode_blocks_writes(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        enforced_stack.enter_maintenance()

        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Should fail",
            outcome="Never saved",
            source_type="observation",
            created_at=_now(),
        )
        with pytest.raises(MaintenanceModeError):
            enforced_stack.save_episode(ep)


# ---- Provenance Enforcement OFF (Default) ----


class TestProvenanceOff:
    """With enforce_provenance=False, provenance is not checked."""

    def test_episode_without_derived_from_allowed(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="No provenance needed",
            outcome="Still works",
            source_type="observation",
            created_at=_now(),
        )
        ep_id = stack.save_episode(ep)
        assert ep_id

    def test_belief_without_derived_from_allowed(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Test belief",
            source_type="inference",
            created_at=_now(),
        )
        belief_id = stack.save_belief(belief)
        assert belief_id


# ---- Provenance Enforcement ON ----


class TestProvenanceEnforced:
    """With enforce_provenance=True in ACTIVE state, provenance is enforced."""

    def _save_raw(self, stack):
        raw = RawEntry(
            id=_uid(),
            stack_id="test-stack",
            blob="Test raw content",
            source="test",
            captured_at=_now(),
        )
        return stack.save_raw(raw)

    def _save_episode(self, stack, raw_id):
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Test objective",
            outcome="Test outcome",
            source_type="observation",
            created_at=_now(),
            derived_from=[f"raw:{raw_id}"],
        )
        return stack.save_episode(ep)

    def _save_note(self, stack, raw_id):
        note = Note(
            id=_uid(),
            stack_id="test-stack",
            content="Test note",
            created_at=_now(),
            derived_from=[f"raw:{raw_id}"],
        )
        return stack.save_note(note)

    def _save_belief(self, stack, episode_id):
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Test belief",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"episode:{episode_id}"],
        )
        return stack.save_belief(belief)

    # -- Raw entries need no provenance --

    def test_raw_no_provenance_needed(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        assert raw_id

    # -- Episode requires raw --

    def test_episode_with_valid_provenance(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        assert ep_id

    def test_episode_without_derived_from_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="No provenance",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="derived_from must cite"):
            enforced_stack.save_episode(ep)

    def test_episode_with_wrong_source_type_raises(self, enforced_stack):
        """Episode must cite raw, not belief."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Wrong source",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
            derived_from=["belief:fake-id"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_episode(ep)

    def test_episode_with_nonexistent_raw_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Cites ghost",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
            derived_from=["raw:nonexistent-id"],
        )
        with pytest.raises(ProvenanceError, match="does not exist"):
            enforced_stack.save_episode(ep)

    # -- Note requires raw --

    def test_note_with_valid_provenance(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        note_id = self._save_note(enforced_stack, raw_id)
        assert note_id

    def test_note_without_provenance_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        note = Note(
            id=_uid(),
            stack_id="test-stack",
            content="No provenance",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="derived_from must cite"):
            enforced_stack.save_note(note)

    # -- Belief requires episode or note --

    def test_belief_from_episode(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)
        assert belief_id

    def test_belief_from_note(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        note_id = self._save_note(enforced_stack, raw_id)

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="From note",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"note:{note_id}"],
        )
        belief_id = enforced_stack.save_belief(belief)
        assert belief_id

    def test_belief_from_raw_raises(self, enforced_stack):
        """Belief cannot cite raw directly."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Skip a layer",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"raw:{raw_id}"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_belief(belief)

    def test_belief_without_provenance_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="No provenance",
            source_type="inference",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="derived_from must cite"):
            enforced_stack.save_belief(belief)

    # -- Value requires belief --

    def test_value_from_belief(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)

        value = Value(
            id=_uid(),
            stack_id="test-stack",
            name="honesty",
            statement="Value honesty",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"belief:{belief_id}"],
        )
        value_id = enforced_stack.save_value(value)
        assert value_id

    def test_value_from_episode_raises(self, enforced_stack):
        """Value must cite belief, not episode."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        value = Value(
            id=_uid(),
            stack_id="test-stack",
            name="honesty",
            statement="Value honesty",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_value(value)

    # -- Goal requires episode or belief --

    def test_goal_from_episode(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)

        goal = Goal(
            id=_uid(),
            stack_id="test-stack",
            title="Test goal",
            source_type="direct_experience",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}"],
        )
        goal_id = enforced_stack.save_goal(goal)
        assert goal_id

    def test_goal_from_belief(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)

        goal = Goal(
            id=_uid(),
            stack_id="test-stack",
            title="Test goal",
            source_type="direct_experience",
            created_at=_now(),
            derived_from=[f"belief:{belief_id}"],
        )
        goal_id = enforced_stack.save_goal(goal)
        assert goal_id

    # -- Relationship requires episode --

    def test_relationship_from_episode(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)

        rel = Relationship(
            id=_uid(),
            stack_id="test-stack",
            entity_name="Alice",
            entity_type="human",
            relationship_type="colleague",
            source_type="observation",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}"],
        )
        rel_id = enforced_stack.save_relationship(rel)
        assert rel_id

    def test_relationship_from_belief_raises(self, enforced_stack):
        """Relationship must cite episode, not belief."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)

        rel = Relationship(
            id=_uid(),
            stack_id="test-stack",
            entity_name="Bob",
            entity_type="human",
            relationship_type="colleague",
            source_type="observation",
            created_at=_now(),
            derived_from=[f"belief:{belief_id}"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_relationship(rel)

    # -- Drive requires episode or belief --

    def test_drive_from_episode(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)

        drive = Drive(
            id=_uid(),
            stack_id="test-stack",
            drive_type="curiosity",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}"],
        )
        drive_id = enforced_stack.save_drive(drive)
        assert drive_id

    def test_drive_from_belief(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)

        drive = Drive(
            id=_uid(),
            stack_id="test-stack",
            drive_type="curiosity",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"belief:{belief_id}"],
        )
        drive_id = enforced_stack.save_drive(drive)
        assert drive_id

    # -- Malformed provenance references --

    def test_malformed_reference_no_colon_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Bad ref",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
            derived_from=["no-colon-here"],
        )
        with pytest.raises(ProvenanceError, match="Expected format"):
            enforced_stack.save_episode(ep)

    def test_empty_derived_from_list_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Empty list",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
            derived_from=[],
        )
        with pytest.raises(ProvenanceError, match="derived_from must cite"):
            enforced_stack.save_episode(ep)

    # -- Multiple valid sources --

    def test_multiple_valid_sources(self, enforced_stack):
        """A belief can cite multiple episodes."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id1 = self._save_episode(enforced_stack, raw_id)
        ep_id2 = self._save_episode(enforced_stack, raw_id)

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="From two episodes",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"episode:{ep_id1}", f"episode:{ep_id2}"],
        )
        belief_id = enforced_stack.save_belief(belief)
        assert belief_id

    def test_mixed_valid_sources(self, enforced_stack):
        """A belief can cite both an episode and a note."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        note_id = self._save_note(enforced_stack, raw_id)

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="From episode and note",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}", f"note:{note_id}"],
        )
        belief_id = enforced_stack.save_belief(belief)
        assert belief_id

    def test_one_invalid_in_list_raises(self, enforced_stack):
        """If any reference is invalid, the whole save fails."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="One bad ref",
            source_type="inference",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}", "raw:some-raw"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_belief(belief)


# ---- memory_exists Tests ----


class TestMemoryExists:
    def test_existing_raw_found(self, stack):
        raw = RawEntry(
            id=_uid(),
            stack_id="test-stack",
            blob="Test",
            source="test",
            captured_at=_now(),
        )
        raw_id = stack.save_raw(raw)
        assert stack._backend.memory_exists("raw", raw_id)

    def test_nonexistent_raw_not_found(self, stack):
        assert not stack._backend.memory_exists("raw", "nonexistent")

    def test_existing_episode_found(self, stack):
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Test",
            outcome="Test",
            source_type="observation",
            created_at=_now(),
        )
        ep_id = stack.save_episode(ep)
        assert stack._backend.memory_exists("episode", ep_id)

    def test_unknown_type_returns_false(self, stack):
        assert not stack._backend.memory_exists("unknown_type", "some-id")

    def test_existing_belief_found(self, stack):
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Test",
            source_type="inference",
            created_at=_now(),
        )
        belief_id = stack.save_belief(belief)
        assert stack._backend.memory_exists("belief", belief_id)


# ---- INITIALIZING State Allows Seed Writes ----


class TestSeedWrites:
    def test_seed_writes_bypass_provenance(self, enforced_stack):
        """In INITIALIZING state, all writes are allowed without provenance."""
        assert enforced_stack.state == StackState.INITIALIZING

        # Save a belief directly -- no episode needed
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Seed belief",
            source_type="seed",
            created_at=_now(),
        )
        belief_id = enforced_stack.save_belief(belief)
        assert belief_id

        # Save a value directly -- no belief needed
        value = Value(
            id=_uid(),
            stack_id="test-stack",
            name="core-value",
            statement="Seed value",
            source_type="seed",
            created_at=_now(),
        )
        value_id = enforced_stack.save_value(value)
        assert value_id

    def test_after_attach_provenance_enforced(self, enforced_stack):
        """After on_attach, provenance must be provided."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        assert enforced_stack.state == StackState.ACTIVE

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Post-seed belief",
            source_type="inference",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError):
            enforced_stack.save_belief(belief)


# ==============================================================================
# Annotation ref compatibility (context:, kernle:)
# ==============================================================================


class TestAnnotationRefs:
    """Test that annotation refs (context:, kernle:) are accepted in derived_from."""

    @pytest.fixture
    def active_stack(self, tmp_path):
        """Stack in ACTIVE state with enforce_provenance=True."""
        stack = SQLiteStack(
            "test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )
        # Save a raw entry and episode for provenance chains
        raw = RawEntry(id=_uid(), stack_id="test-stack", blob="seed raw", source="test")
        raw_id = stack.save_raw(raw)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Seed episode",
            outcome="Done",
            source_type="seed",
            derived_from=[f"raw:{raw_id}"],
            created_at=_now(),
        )
        ep_id = stack.save_episode(ep)
        # Transition to ACTIVE
        stack.on_attach(core_id="core-1", inference=None)
        # Store IDs for tests
        stack._test_raw_id = raw_id
        stack._test_ep_id = ep_id
        return stack

    def test_context_ref_alongside_real_ref(self, active_stack):
        """context: annotation alongside a real episode ref should pass."""
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Belief with context annotation",
            source_type="inference",
            derived_from=[
                f"episode:{active_stack._test_ep_id}",
                "context:cli",
            ],
            created_at=_now(),
        )
        bid = active_stack.save_belief(belief)
        assert bid

    def test_kernle_ref_alongside_real_ref(self, active_stack):
        """kernle: annotation alongside a real episode ref should pass."""
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Belief with kernle annotation",
            source_type="inference",
            derived_from=[
                f"episode:{active_stack._test_ep_id}",
                "kernle:system",
            ],
            created_at=_now(),
        )
        bid = active_stack.save_belief(belief)
        assert bid

    def test_only_annotation_refs_rejected(self, active_stack):
        """Only annotation refs with no real provenance should be rejected."""
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Only annotations",
            source_type="inference",
            derived_from=["context:cli", "kernle:system"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="only annotation refs"):
            active_stack.save_belief(belief)

    def test_context_ref_with_complex_value(self, active_stack):
        """context: with colons in the value should work."""
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Context ref with colons",
            source_type="inference",
            derived_from=[
                f"episode:{active_stack._test_ep_id}",
                "context:kernle_seed_v2:batch_1",
            ],
            created_at=_now(),
        )
        bid = active_stack.save_belief(belief)
        assert bid

    def test_annotation_refs_in_episode(self, active_stack):
        """Annotation refs work for episodes too."""
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Episode with annotation",
            outcome="Done",
            source_type="processing",
            derived_from=[
                f"raw:{active_stack._test_raw_id}",
                "context:consolidation",
            ],
            created_at=_now(),
        )
        eid = active_stack.save_episode(ep)
        assert eid


# ==============================================================================
# Stack Settings
# ==============================================================================


class TestStackSettings:
    """Test per-stack settings (feature flags)."""

    def test_get_nonexistent_setting(self, tmp_path):
        stack = SQLiteStack("test", db_path=tmp_path / "test.db", components=[])
        assert stack.get_stack_setting("nonexistent") is None

    def test_set_and_get_setting(self, tmp_path):
        stack = SQLiteStack("test", db_path=tmp_path / "test.db", components=[])
        stack.set_stack_setting("enforce_provenance", "true")
        assert stack.get_stack_setting("enforce_provenance") == "true"

    def test_upsert_setting(self, tmp_path):
        stack = SQLiteStack("test", db_path=tmp_path / "test.db", components=[])
        stack.set_stack_setting("key", "v1")
        stack.set_stack_setting("key", "v2")
        assert stack.get_stack_setting("key") == "v2"

    def test_get_all_settings(self, tmp_path):
        stack = SQLiteStack("test", db_path=tmp_path / "test.db", components=[])
        stack.set_stack_setting("a", "1")
        stack.set_stack_setting("b", "2")
        settings = stack.get_all_stack_settings()
        assert settings["a"] == "1"
        assert settings["b"] == "2"

    def test_get_all_empty(self, tmp_path):
        stack = SQLiteStack("test", db_path=tmp_path / "test.db", components=[])
        settings = stack.get_all_stack_settings()
        # New stacks auto-set use_legacy_heuristics=false
        assert settings.get("use_legacy_heuristics") == "false"


# ==============================================================================
# Stack State Persistence
# ==============================================================================


class TestStackStatePersistence:
    """Test that stack lifecycle state survives across instances."""

    def test_new_stack_starts_initializing(self, tmp_path):
        stack = SQLiteStack("test", db_path=tmp_path / "test.db", components=[])
        assert stack.state == StackState.INITIALIZING

    def test_on_attach_persists_active_state(self, tmp_path):
        db_path = tmp_path / "test.db"
        stack = SQLiteStack("test", db_path=db_path, components=[])
        stack.on_attach(core_id="core-1", inference=None)
        assert stack.state == StackState.ACTIVE

        # New instance should load persisted ACTIVE state
        stack2 = SQLiteStack("test", db_path=db_path, components=[])
        assert stack2.state == StackState.ACTIVE

    def test_maintenance_state_persisted(self, tmp_path):
        db_path = tmp_path / "test.db"
        stack = SQLiteStack("test", db_path=db_path, components=[])
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        assert stack.state == StackState.MAINTENANCE

        stack2 = SQLiteStack("test", db_path=db_path, components=[])
        assert stack2.state == StackState.MAINTENANCE

    def test_exit_maintenance_persists(self, tmp_path):
        db_path = tmp_path / "test.db"
        stack = SQLiteStack("test", db_path=db_path, components=[])
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        stack.exit_maintenance()
        assert stack.state == StackState.ACTIVE

        stack2 = SQLiteStack("test", db_path=db_path, components=[])
        assert stack2.state == StackState.ACTIVE

    def test_enforce_provenance_persisted(self, tmp_path):
        """enforce_provenance can be loaded from stack settings."""
        db_path = tmp_path / "test.db"
        stack = SQLiteStack("test", db_path=db_path, components=[])
        stack.set_stack_setting("enforce_provenance", "true")

        # New instance without explicit enforce_provenance should pick it up
        stack2 = SQLiteStack("test", db_path=db_path, components=[])
        assert stack2._enforce_provenance is True

    def test_explicit_enforce_provenance_overrides(self, tmp_path):
        """Explicit enforce_provenance=True always wins."""
        db_path = tmp_path / "test.db"
        stack = SQLiteStack("test", db_path=db_path, components=[], enforce_provenance=True)
        assert stack._enforce_provenance is True


# ==============================================================================
# Plugin-sourced write bypass
# ==============================================================================


class TestPluginProvenanceBypass:
    """Test that plugin-sourced writes bypass provenance validation.

    Only plugins registered via register_plugin() are trusted.
    """

    @pytest.fixture
    def active_enforced_stack(self, tmp_path):
        stack = SQLiteStack(
            "test",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )
        # Save a raw + episode for non-plugin tests
        raw = RawEntry(id=_uid(), stack_id="test", blob="raw", source="test")
        raw_id = stack.save_raw(raw)
        ep = Episode(
            id=_uid(),
            stack_id="test",
            objective="Ep",
            outcome="Done",
            source_type="seed",
            derived_from=[f"raw:{raw_id}"],
            created_at=_now(),
        )
        ep_id = stack.save_episode(ep)
        # Register known plugins and transition to ACTIVE
        stack.register_plugin("chainbased")
        stack.register_plugin("fatline")
        stack.on_attach(core_id="core-1", inference=None)
        stack._test_raw_id = raw_id
        stack._test_ep_id = ep_id
        return stack

    def test_registered_plugin_bypasses_provenance(self, active_enforced_stack):
        """A belief from a registered plugin bypasses provenance."""
        belief = Belief(
            id=_uid(),
            stack_id="test",
            statement="Plugin belief",
            source_type="inference",
            derived_from=None,
            created_at=_now(),
        )
        belief.source_entity = "plugin:chainbased"
        bid = active_enforced_stack.save_belief(belief)
        assert bid

    def test_unregistered_plugin_rejected(self, active_enforced_stack):
        """A belief claiming plugin:unknown (not registered) is rejected."""
        belief = Belief(
            id=_uid(),
            stack_id="test",
            statement="Spoofed plugin belief",
            source_type="inference",
            derived_from=None,
            created_at=_now(),
        )
        belief.source_entity = "plugin:unknown_plugin"
        with pytest.raises(ProvenanceError):
            active_enforced_stack.save_belief(belief)

    def test_core_sourced_belief_requires_provenance(self, active_enforced_stack):
        """A belief with source_entity='core:...' still requires provenance."""
        belief = Belief(
            id=_uid(),
            stack_id="test",
            statement="Core belief",
            source_type="inference",
            derived_from=None,
            created_at=_now(),
        )
        belief.source_entity = "core:core-1"
        with pytest.raises(ProvenanceError):
            active_enforced_stack.save_belief(belief)

    def test_registered_plugin_goal_bypasses(self, active_enforced_stack):
        """Registered plugin goals bypass provenance."""
        goal = Goal(
            id=_uid(),
            stack_id="test",
            title="Plugin goal",
            goal_type="task",
            priority="medium",
            source_type="inference",
            derived_from=None,
            created_at=_now(),
        )
        goal.source_entity = "plugin:fatline"
        gid = active_enforced_stack.save_goal(goal)
        assert gid

    def test_unregister_plugin_revokes_bypass(self, active_enforced_stack):
        """After unregister_plugin(), that plugin's writes are rejected."""
        active_enforced_stack.unregister_plugin("chainbased")
        belief = Belief(
            id=_uid(),
            stack_id="test",
            statement="Revoked plugin",
            source_type="inference",
            derived_from=None,
            created_at=_now(),
        )
        belief.source_entity = "plugin:chainbased"
        with pytest.raises(ProvenanceError):
            active_enforced_stack.save_belief(belief)

    def test_plugin_blocked_in_maintenance(self, active_enforced_stack):
        """Plugin writes are still blocked in maintenance mode."""
        active_enforced_stack.enter_maintenance()
        belief = Belief(
            id=_uid(),
            stack_id="test",
            statement="Plugin belief in maintenance",
            source_type="inference",
            derived_from=None,
            created_at=_now(),
        )
        belief.source_entity = "plugin:chainbased"
        with pytest.raises(MaintenanceModeError):
            active_enforced_stack.save_belief(belief)


# ==============================================================================
# Maintenance mode independence from provenance flag
# ==============================================================================


class TestMaintenanceModeIndependence:
    """Maintenance mode blocks writes regardless of enforce_provenance."""

    def test_maintenance_blocks_when_provenance_off(self, tmp_path):
        """Maintenance blocks writes even when enforce_provenance=False."""
        stack = SQLiteStack(
            "test",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        belief = Belief(
            id=_uid(),
            stack_id="test",
            statement="Should be blocked",
            source_type="inference",
            created_at=_now(),
        )
        with pytest.raises(MaintenanceModeError):
            stack.save_belief(belief)

    def test_maintenance_blocks_raw_when_provenance_off(self, tmp_path):
        """Even raw writes blocked in maintenance mode."""
        stack = SQLiteStack(
            "test",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        raw = RawEntry(id=_uid(), stack_id="test", blob="test", source="test")
        with pytest.raises(MaintenanceModeError):
            stack.save_raw(raw)

    def test_exit_maintenance_allows_writes_again(self, tmp_path):
        stack = SQLiteStack(
            "test",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        stack.exit_maintenance()
        raw = RawEntry(id=_uid(), stack_id="test", blob="test", source="test")
        rid = stack.save_raw(raw)
        assert rid


# ==============================================================================
# Batch write provenance enforcement
# ==============================================================================


class TestBatchWriteProvenance:
    """Batch writes must validate provenance same as single writes."""

    @pytest.fixture
    def active_enforced_stack(self, tmp_path):
        stack = SQLiteStack(
            "test",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )
        raw = RawEntry(id=_uid(), stack_id="test", blob="raw", source="test")
        raw_id = stack.save_raw(raw)
        ep = Episode(
            id=_uid(),
            stack_id="test",
            objective="Ep",
            outcome="Done",
            source_type="seed",
            derived_from=[f"raw:{raw_id}"],
            created_at=_now(),
        )
        ep_id = stack.save_episode(ep)
        stack.on_attach(core_id="core-1", inference=None)
        stack._test_raw_id = raw_id
        stack._test_ep_id = ep_id
        return stack

    def test_batch_beliefs_without_provenance_rejected(self, active_enforced_stack):
        """Batch belief save with no derived_from should raise ProvenanceError."""
        beliefs = [
            Belief(
                id=_uid(),
                stack_id="test",
                statement="No provenance",
                source_type="inference",
                derived_from=None,
                created_at=_now(),
            )
        ]
        with pytest.raises(ProvenanceError):
            active_enforced_stack.save_beliefs_batch(beliefs)

    def test_batch_beliefs_with_provenance_accepted(self, active_enforced_stack):
        """Batch belief save with valid provenance should succeed."""
        beliefs = [
            Belief(
                id=_uid(),
                stack_id="test",
                statement="Valid provenance",
                source_type="inference",
                derived_from=[f"episode:{active_enforced_stack._test_ep_id}"],
                created_at=_now(),
            )
        ]
        ids = active_enforced_stack.save_beliefs_batch(beliefs)
        assert len(ids) == 1

    def test_batch_episodes_without_provenance_rejected(self, active_enforced_stack):
        episodes = [
            Episode(
                id=_uid(),
                stack_id="test",
                objective="No provenance",
                outcome="Done",
                source_type="processing",
                derived_from=None,
                created_at=_now(),
            )
        ]
        with pytest.raises(ProvenanceError):
            active_enforced_stack.save_episodes_batch(episodes)

    def test_batch_notes_without_provenance_rejected(self, active_enforced_stack):
        notes = [
            Note(
                id=_uid(),
                stack_id="test",
                content="No provenance",
                note_type="note",
                source_type="processing",
                derived_from=None,
                created_at=_now(),
            )
        ]
        with pytest.raises(ProvenanceError):
            active_enforced_stack.save_notes_batch(notes)

    def test_batch_blocked_in_maintenance(self, active_enforced_stack):
        """Batch writes blocked in maintenance mode."""
        active_enforced_stack.enter_maintenance()
        beliefs = [
            Belief(
                id=_uid(),
                stack_id="test",
                statement="Maintenance",
                source_type="inference",
                derived_from=[f"episode:{active_enforced_stack._test_ep_id}"],
                created_at=_now(),
            )
        ]
        with pytest.raises(MaintenanceModeError):
            active_enforced_stack.save_beliefs_batch(beliefs)


# ==============================================================================
# Stack-scoped settings
# ==============================================================================


class TestStackScopedSettings:
    """Settings are scoped per stack_id, not global to the DB."""

    def test_different_stacks_isolated(self, tmp_path):
        """Two stacks sharing a DB have independent settings."""
        db_path = tmp_path / "shared.db"
        stack_a = SQLiteStack("stack-a", db_path=db_path, components=[])
        stack_b = SQLiteStack("stack-b", db_path=db_path, components=[])

        stack_a.set_stack_setting("enforce_provenance", "true")

        assert stack_a.get_stack_setting("enforce_provenance") == "true"
        assert stack_b.get_stack_setting("enforce_provenance") is None

    def test_get_all_scoped(self, tmp_path):
        db_path = tmp_path / "shared.db"
        stack_a = SQLiteStack("stack-a", db_path=db_path, components=[])
        stack_b = SQLiteStack("stack-b", db_path=db_path, components=[])

        stack_a.set_stack_setting("a", "1")
        stack_b.set_stack_setting("b", "2")

        settings_a = stack_a.get_all_stack_settings()
        settings_b = stack_b.get_all_stack_settings()
        assert settings_a["a"] == "1"
        assert "b" not in settings_a
        assert settings_b["b"] == "2"
        assert "a" not in settings_b

    def test_set_enforce_provenance_live(self, tmp_path):
        """set_stack_setting('enforce_provenance', 'true') takes effect immediately."""
        db_path = tmp_path / "live.db"
        stack = SQLiteStack("test", db_path=db_path, components=[], enforce_provenance=False)
        raw = RawEntry(id=_uid(), stack_id="test", blob="raw", source="test")
        raw_id = stack.save_raw(raw)
        ep = Episode(
            id=_uid(),
            stack_id="test",
            objective="Ep",
            outcome="Done",
            source_type="seed",
            derived_from=[f"raw:{raw_id}"],
            created_at=_now(),
        )
        stack.save_episode(ep)
        stack.on_attach("core-1")

        # Provenance not enforced yet — belief without derived_from succeeds
        b1 = Belief(
            id=_uid(),
            stack_id="test",
            statement="No provenance",
            source_type="inference",
            created_at=_now(),
        )
        assert stack.save_belief(b1)

        # Enable provenance — should take effect immediately
        stack.set_stack_setting("enforce_provenance", "true")
        assert stack._enforce_provenance is True

        b2 = Belief(
            id=_uid(),
            stack_id="test",
            statement="No provenance after enable",
            source_type="inference",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError):
            stack.save_belief(b2)

        # Disable again — immediately permissive
        stack.set_stack_setting("enforce_provenance", "false")
        assert stack._enforce_provenance is False
        b3 = Belief(
            id=_uid(),
            stack_id="test",
            statement="Allowed again",
            source_type="inference",
            created_at=_now(),
        )
        assert stack.save_belief(b3)


# ==============================================================================
# Kernle strict-mode tests
# ==============================================================================


class TestKernleStrictMode:
    """Kernle(strict=True) routes writes through stack enforcement."""

    @pytest.fixture
    def strict_kernle(self, tmp_path):
        from kernle.core import Kernle
        from kernle.storage import SQLiteStorage

        db = tmp_path / "strict.db"
        storage = SQLiteStorage(stack_id="strict_agent", db_path=db)
        k = Kernle(
            stack_id="strict_agent",
            storage=storage,
            checkpoint_dir=tmp_path / "cp",
            strict=True,
        )
        yield k
        storage.close()

    @pytest.fixture
    def legacy_kernle(self, tmp_path):
        from kernle.core import Kernle
        from kernle.storage import SQLiteStorage

        db = tmp_path / "legacy.db"
        storage = SQLiteStorage(stack_id="legacy_agent", db_path=db)
        k = Kernle(
            stack_id="legacy_agent",
            storage=storage,
            checkpoint_dir=tmp_path / "cp",
            strict=False,
        )
        yield k
        storage.close()

    def test_strict_flag_stored(self, strict_kernle, legacy_kernle):
        assert strict_kernle._strict is True
        assert legacy_kernle._strict is False

    def test_write_backend_returns_stack_when_strict(self, strict_kernle):
        backend = strict_kernle._write_backend
        assert backend is strict_kernle.stack

    def test_write_backend_returns_storage_when_legacy(self, legacy_kernle):
        backend = legacy_kernle._write_backend
        assert backend is legacy_kernle._storage

    def test_strict_maintenance_blocks_episode(self, strict_kernle):
        """In strict mode, maintenance state blocks writes."""
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.episode(
                objective="Test",
                outcome="Should fail",
            )

    def test_strict_maintenance_blocks_episode_with_emotion(self, strict_kernle):
        """episode_with_emotion must use strict write path and honor maintenance mode."""
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.episode_with_emotion(
                objective="Test emotional",
                outcome="Should fail",
                valence=0.5,
                arousal=0.4,
                auto_detect=False,
            )

    def test_strict_maintenance_blocks_belief(self, strict_kernle):
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.belief(statement="Test belief")

    def test_strict_maintenance_blocks_value(self, strict_kernle):
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.value(name="Test", statement="Test value")

    def test_strict_maintenance_blocks_goal(self, strict_kernle):
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.goal(title="Test goal")

    def test_strict_maintenance_blocks_raw(self, strict_kernle):
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.raw(blob="Test blob")

    def test_strict_maintenance_blocks_note(self, strict_kernle):
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.note(content="Test note")

    def test_strict_maintenance_blocks_drive(self, strict_kernle):
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.drive(drive_type="curiosity")

    def test_strict_maintenance_blocks_relationship(self, strict_kernle):
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.relationship(other_stack_id="other")

    def test_strict_maintenance_blocks_batches(self, strict_kernle):
        stack = strict_kernle.stack
        stack.on_attach("strict_agent")
        stack.enter_maintenance()

        with pytest.raises(MaintenanceModeError):
            strict_kernle.episodes_batch([{"objective": "X", "outcome": "Y"}])

        with pytest.raises(MaintenanceModeError):
            strict_kernle.beliefs_batch([{"statement": "X"}])

        with pytest.raises(MaintenanceModeError):
            strict_kernle.notes_batch([{"content": "X"}])

    def test_legacy_mode_ignores_maintenance(self, legacy_kernle):
        """In legacy mode, writes go to storage directly — no enforcement."""
        # Force the stack into maintenance, but legacy mode bypasses it
        stack = legacy_kernle.stack
        stack.on_attach("legacy_agent")
        stack.enter_maintenance()

        # Legacy write goes directly to storage — no error
        raw_id = legacy_kernle.raw(blob="Should succeed in legacy mode")
        assert raw_id is not None

    def test_strict_auto_attaches_to_active(self, strict_kernle):
        """Strict mode auto-attaches the stack, transitioning to ACTIVE."""
        from kernle.protocols import StackState

        stack = strict_kernle.stack
        assert stack._state == StackState.ACTIVE

    def test_strict_active_allows_raw(self, strict_kernle):
        """In strict mode, raw writes succeed (raws need no provenance)."""
        raw_id = strict_kernle.raw(blob="Test raw in strict mode")
        assert raw_id is not None

    def test_strict_active_allows_episode_without_provenance(self, tmp_path):
        """Episodes without provenance succeed when provenance not enforced."""
        from kernle.core import Kernle
        from kernle.storage import SQLiteStorage

        db = tmp_path / "strict_no_prov.db"
        storage = SQLiteStorage(stack_id="strict_np", db_path=db)
        k = Kernle(
            stack_id="strict_np",
            storage=storage,
            checkpoint_dir=tmp_path / "cp",
            strict=True,
        )
        # Explicitly disable provenance enforcement on the stack
        k.stack._enforce_provenance = False
        ep_id = k.episode(
            objective="Test episode",
            outcome="Success",
        )
        assert ep_id is not None
        storage.close()

    def test_strict_enforced_rejects_belief_without_provenance(self, tmp_path):
        """strict + enforce_provenance rejects beliefs without derived_from."""
        from kernle.core import Kernle
        from kernle.storage import SQLiteStorage

        db = tmp_path / "strict_enforced.db"
        storage = SQLiteStorage(stack_id="strict_enf", db_path=db)
        k = Kernle(
            stack_id="strict_enf",
            storage=storage,
            checkpoint_dir=tmp_path / "cp",
            strict=True,
        )
        # Enable provenance enforcement
        k.stack.set_stack_setting("enforce_provenance", "true")
        k.stack._enforce_provenance = True

        with pytest.raises(ProvenanceError):
            k.belief(statement="No provenance belief")
        storage.close()

    def test_strict_requires_sqlite(self, tmp_path):
        """strict=True with non-SQLite storage raises ValueError."""
        from unittest.mock import MagicMock

        from kernle.core import Kernle

        mock_storage = MagicMock()
        mock_storage.is_online.return_value = False
        mock_storage.get_pending_sync_count.return_value = 0

        k = Kernle(
            stack_id="test",
            storage=mock_storage,
            checkpoint_dir=tmp_path / "cp",
            strict=True,
        )
        # stack property returns None for non-SQLite
        with pytest.raises(ValueError, match="strict=True requires SQLite"):
            k._write_backend
