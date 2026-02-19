"""Tests for canonical SourceType taxonomy and validation.

Covers:
- SourceType enum has all canonical values
- VALID_SOURCE_TYPE_VALUES is derived from the enum
- VALID_SOURCE_TYPES in tool_definitions matches the enum
- Strict mode rejects unknown source_type values
- Strict mode accepts all canonical source_type values
- Processing uses canonical "processing" (not "processed")
- Migration handles legacy "processed" entries
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.protocols import ProvenanceError, StackState
from kernle.stack.sqlite_stack import Stack
from kernle.types import (
    VALID_SOURCE_TYPE_VALUES,
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    RawEntry,
    Relationship,
    SourceType,
    Value,
)


def _uid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


# ---- SourceType Enum Completeness ----


class TestSourceTypeEnum:
    def test_all_canonical_values_present(self):
        """SourceType enum contains all canonical values."""
        expected = {
            "direct_experience",
            "inference",
            "external",
            "consolidation",
            "processing",
            "seed",
            "observation",
            "imported",
            "unknown",
        }
        actual = {st.value for st in SourceType}
        assert actual == expected

    def test_valid_source_type_values_matches_enum(self):
        """VALID_SOURCE_TYPE_VALUES is derived from SourceType enum."""
        enum_values = frozenset(st.value for st in SourceType)
        assert VALID_SOURCE_TYPE_VALUES == enum_values

    def test_valid_source_type_values_is_frozenset(self):
        """VALID_SOURCE_TYPE_VALUES is immutable."""
        assert isinstance(VALID_SOURCE_TYPE_VALUES, frozenset)

    def test_processed_not_in_enum(self):
        """The legacy 'processed' value is NOT in the canonical enum."""
        assert "processed" not in VALID_SOURCE_TYPE_VALUES


class TestToolDefinitionsAlignment:
    def test_valid_source_types_matches_enum(self):
        """MCP VALID_SOURCE_TYPES list matches the canonical enum."""
        from kernle.mcp.tool_definitions import VALID_SOURCE_TYPES

        assert set(VALID_SOURCE_TYPES) == VALID_SOURCE_TYPE_VALUES

    def test_valid_source_types_is_sorted(self):
        """MCP VALID_SOURCE_TYPES list is sorted for deterministic schema."""
        from kernle.mcp.tool_definitions import VALID_SOURCE_TYPES

        assert VALID_SOURCE_TYPES == sorted(VALID_SOURCE_TYPES)


# ---- Strict Mode Validation ----


@pytest.fixture
def active_enforced_stack(tmp_path):
    """Stack in ACTIVE state with provenance enforcement enabled."""
    stack = Stack.from_sqlite(
        stack_id="test-strict",
        db_path=tmp_path / "test.db",
        components=[],
        enforce_provenance=True,
    )
    # Create a raw entry for provenance (during INITIALIZING — no validation)
    raw = RawEntry(id=_uid(), stack_id="test-strict", blob="test input", source="test")
    raw_id = stack.save_raw(raw)
    # Transition to ACTIVE via on_attach (triggers provenance enforcement)
    stack.on_attach(core_id="test-core", inference=None)
    stack._test_raw_id = raw_id
    return stack


class TestSourceTypeValidation:
    """Strict mode rejects unknown source_type values."""

    def test_reject_unknown_source_type_episode(self, active_enforced_stack):
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="bogus_type",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="Unknown source_type 'bogus_type'"):
            active_enforced_stack.save_episode(ep)

    def test_reject_legacy_processed_source_type(self, active_enforced_stack):
        """The legacy 'processed' value is rejected in strict mode."""
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="processed",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="Unknown source_type 'processed'"):
            active_enforced_stack.save_episode(ep)

    def test_reject_unknown_source_type_belief(self, active_enforced_stack):
        # Need an episode for belief provenance
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="direct_experience",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        eid = active_enforced_stack.save_episode(ep)

        belief = Belief(
            id=_uid(),
            stack_id="test-strict",
            statement="Test belief",
            source_type="invented_type",
            derived_from=[f"episode:{eid}"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="Unknown source_type 'invented_type'"):
            active_enforced_stack.save_belief(belief)

    def test_reject_unknown_source_type_note(self, active_enforced_stack):
        note = Note(
            id=_uid(),
            stack_id="test-strict",
            content="Test note",
            source_type="made_up",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="Unknown source_type 'made_up'"):
            active_enforced_stack.save_note(note)

    def test_reject_unknown_source_type_value(self, active_enforced_stack):
        # Build provenance chain: raw -> episode -> belief -> value
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="direct_experience",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        eid = active_enforced_stack.save_episode(ep)
        belief = Belief(
            id=_uid(),
            stack_id="test-strict",
            statement="This is a sufficiently long test belief statement for lint",
            source_type="inference",
            derived_from=[f"episode:{eid}"],
            created_at=_now(),
        )
        bid = active_enforced_stack.save_belief(belief)

        value = Value(
            id=_uid(),
            stack_id="test-strict",
            name="Test Value Name",
            statement="This is a sufficiently long test value statement for lint",
            source_type="wrong",
            derived_from=[f"belief:{bid}"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="Unknown source_type 'wrong'"):
            active_enforced_stack.save_value(value)

    def test_reject_unknown_source_type_goal(self, active_enforced_stack):
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="direct_experience",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        eid = active_enforced_stack.save_episode(ep)

        goal = Goal(
            id=_uid(),
            stack_id="test-strict",
            title="Test goal",
            source_type="nope",
            derived_from=[f"episode:{eid}"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="Unknown source_type 'nope'"):
            active_enforced_stack.save_goal(goal)

    def test_reject_unknown_source_type_drive(self, active_enforced_stack):
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="direct_experience",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        eid = active_enforced_stack.save_episode(ep)

        drive = Drive(
            id=_uid(),
            stack_id="test-strict",
            drive_type="curiosity",
            source_type="invalid",
            derived_from=[f"episode:{eid}"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="Unknown source_type 'invalid'"):
            active_enforced_stack.save_drive(drive)

    def test_reject_unknown_source_type_relationship(self, active_enforced_stack):
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="direct_experience",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        eid = active_enforced_stack.save_episode(ep)

        rel = Relationship(
            id=_uid(),
            stack_id="test-strict",
            entity_name="Alice",
            entity_type="person",
            relationship_type="collaborator",
            source_type="fake",
            derived_from=[f"episode:{eid}"],
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="Unknown source_type 'fake'"):
            active_enforced_stack.save_relationship(rel)


class TestSourceTypeAcceptance:
    """Strict mode accepts all canonical source_type values."""

    @pytest.mark.parametrize(
        "source_type",
        [
            "direct_experience",
            "inference",
            "external",
            "consolidation",
            "processing",
            "seed",
            "observation",
            "unknown",
        ],
    )
    def test_accept_canonical_source_type_episode(self, active_enforced_stack, source_type):
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type=source_type,
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        eid = active_enforced_stack.save_episode(ep)
        assert eid

    @pytest.mark.parametrize(
        "source_type",
        [
            "direct_experience",
            "inference",
            "external",
            "consolidation",
            "processing",
            "seed",
            "observation",
            "unknown",
        ],
    )
    def test_accept_canonical_source_type_belief(self, active_enforced_stack, source_type):
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="direct_experience",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        eid = active_enforced_stack.save_episode(ep)
        belief = Belief(
            id=_uid(),
            stack_id="test-strict",
            statement="Test",
            source_type=source_type,
            derived_from=[f"episode:{eid}"],
            created_at=_now(),
        )
        bid = active_enforced_stack.save_belief(belief)
        assert bid


class TestSourceTypeNoEnforcement:
    """Without enforcement, unknown source_type values are allowed."""

    def test_allow_unknown_without_enforcement(self, tmp_path):
        stack = Stack.from_sqlite(
            stack_id="test-relaxed",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        stack.on_attach(core_id="test-core", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-relaxed",
            objective="Test",
            outcome="Done",
            source_type="completely_made_up",
            created_at=_now(),
        )
        eid = stack.save_episode(ep)
        assert eid

    def test_allow_unknown_during_initializing(self, tmp_path):
        stack = Stack.from_sqlite(
            stack_id="test-init",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=True,
        )
        # Stack starts in INITIALIZING — unknown source_type should be allowed
        assert stack._state == StackState.INITIALIZING
        ep = Episode(
            id=_uid(),
            stack_id="test-init",
            objective="Seed",
            outcome="Done",
            source_type="anything_goes_during_init",
            created_at=_now(),
        )
        eid = stack.save_episode(ep)
        assert eid


class TestSourceTypeBatchValidation:
    """Batch writes also validate source_type in strict mode."""

    def test_reject_unknown_source_type_batch_episodes(self, active_enforced_stack):
        episodes = [
            Episode(
                id=_uid(),
                stack_id="test-strict",
                objective="Test",
                outcome="Done",
                source_type="bad_value",
                derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
                created_at=_now(),
            )
        ]
        with pytest.raises(ProvenanceError, match="Unknown source_type 'bad_value'"):
            active_enforced_stack.save_episodes_batch(episodes)

    def test_reject_unknown_source_type_batch_beliefs(self, active_enforced_stack):
        ep = Episode(
            id=_uid(),
            stack_id="test-strict",
            objective="Test",
            outcome="Done",
            source_type="direct_experience",
            derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
            created_at=_now(),
        )
        eid = active_enforced_stack.save_episode(ep)

        beliefs = [
            Belief(
                id=_uid(),
                stack_id="test-strict",
                statement="Test",
                source_type="nonsense",
                derived_from=[f"episode:{eid}"],
                created_at=_now(),
            )
        ]
        with pytest.raises(ProvenanceError, match="Unknown source_type 'nonsense'"):
            active_enforced_stack.save_beliefs_batch(beliefs)

    def test_reject_unknown_source_type_batch_notes(self, active_enforced_stack):
        notes = [
            Note(
                id=_uid(),
                stack_id="test-strict",
                content="Test",
                source_type="garbage",
                derived_from=[f"raw:{active_enforced_stack._test_raw_id}"],
                created_at=_now(),
            )
        ]
        with pytest.raises(ProvenanceError, match="Unknown source_type 'garbage'"):
            active_enforced_stack.save_notes_batch(notes)


# ---- Processing Uses Canonical Values ----


class TestProcessingSourceType:
    """Processing module uses canonical 'processing' value."""

    def test_processing_write_memories_uses_processing(self):
        """_write_memories sets source_type='processing', not 'processed'."""
        from unittest.mock import MagicMock

        from kernle.processing import MemoryProcessor

        mock_stack = MagicMock()
        mock_stack.stack_id = "test"
        mock_stack.save_episode.return_value = "ep-1"
        mock_inference = MagicMock()

        processor = MemoryProcessor(
            stack=mock_stack,
            inference=mock_inference,
            core_id="test-core",
        )

        parsed = [
            {
                "objective": "test",
                "outcome": "done",
                "outcome_type": "success",
                "source_raw_ids": ["r1"],
            }
        ]
        processor._write_memories("raw_to_episode", parsed, [])
        saved_ep = mock_stack.save_episode.call_args[0][0]
        assert saved_ep.source_type == "processing"
        assert saved_ep.source_type != "processed"

    def test_processing_all_transitions_use_processing(self):
        """All transitions in _write_memories use 'processing' source_type."""
        from unittest.mock import MagicMock

        from kernle.processing import MemoryProcessor, PromotionGateConfig

        _no_gates = PromotionGateConfig(
            belief_min_evidence=0,
            belief_min_confidence=0.0,
            value_min_evidence=0,
            value_requires_protection=False,
        )

        mock_stack = MagicMock()
        mock_stack.stack_id = "test"
        mock_stack.save_episode.return_value = "ep-1"
        mock_stack.save_note.return_value = "n-1"
        mock_stack.save_belief.return_value = "b-1"
        mock_stack.save_goal.return_value = "g-1"
        mock_stack.save_relationship.return_value = "r-1"
        mock_stack.save_value.return_value = "v-1"
        mock_stack.save_drive.return_value = "d-1"
        mock_inference = MagicMock()

        processor = MemoryProcessor(
            stack=mock_stack,
            inference=mock_inference,
            core_id="test-core",
            promotion_gates=_no_gates,
        )

        transitions_and_data = [
            ("raw_to_episode", {"objective": "t", "outcome": "d", "source_raw_ids": ["r1"]}),
            ("raw_to_note", {"content": "note", "source_raw_ids": ["r1"]}),
            ("episode_to_belief", {"statement": "b", "source_episode_ids": ["e1"]}),
            ("episode_to_goal", {"title": "g", "source_episode_ids": ["e1"]}),
            (
                "episode_to_relationship",
                {"entity_name": "Alice", "source_episode_ids": ["e1"]},
            ),
            ("belief_to_value", {"name": "v", "source_belief_ids": ["b1"]}),
            ("episode_to_drive", {"drive_type": "curiosity", "source_episode_ids": ["e1"]}),
        ]

        save_methods = {
            "raw_to_episode": mock_stack.save_episode,
            "raw_to_note": mock_stack.save_note,
            "episode_to_belief": mock_stack.save_belief,
            "episode_to_goal": mock_stack.save_goal,
            "episode_to_relationship": mock_stack.save_relationship,
            "belief_to_value": mock_stack.save_value,
            "episode_to_drive": mock_stack.save_drive,
        }

        for transition, data in transitions_and_data:
            processor._write_memories(transition, [data], [])
            method = save_methods[transition]
            saved = method.call_args[0][0]
            assert saved.source_type == "processing", (
                f"Transition {transition} used source_type='{saved.source_type}' "
                f"instead of 'processing'"
            )
