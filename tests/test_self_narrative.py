"""Tests for self-narrative layer (autobiographical identity model).

Covers:
- SelfNarrative dataclass defaults and fields
- Storage: save, get, list, deactivate
- Core API: narrative_save, narrative_get_active, narrative_list
- Priority scoring for self_narrative type
- Load integration: active narratives in context output
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle import Kernle
from kernle.core import MEMORY_TYPE_PRIORITIES, compute_priority_score
from kernle.storage import SelfNarrative


@pytest.fixture
def stack_id():
    return f"test-agent-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def k(stack_id):
    return Kernle(stack_id=stack_id, strict=False)


# === Dataclass Tests ===


class TestSelfNarrativeDataclass:
    def test_defaults(self):
        n = SelfNarrative(
            id="n1",
            stack_id="agent1",
            content="I am a curious learner.",
        )
        assert n.narrative_type == "identity"
        assert n.is_active is True
        assert n.version == 1
        assert n.deleted is False
        assert n.key_themes is None
        assert n.unresolved_tensions is None
        assert n.epoch_id is None
        assert n.supersedes is None

    def test_all_fields(self):
        now = datetime.now(timezone.utc)
        n = SelfNarrative(
            id="n2",
            stack_id="agent1",
            content="I have grown from a novice into a capable problem solver.",
            narrative_type="developmental",
            epoch_id="epoch-1",
            key_themes=["growth", "problem-solving"],
            unresolved_tensions=["speed vs thoroughness"],
            is_active=True,
            supersedes="n1",
            created_at=now,
            updated_at=now,
        )
        assert n.narrative_type == "developmental"
        assert n.key_themes == ["growth", "problem-solving"]
        assert n.unresolved_tensions == ["speed vs thoroughness"]
        assert n.supersedes == "n1"
        assert n.epoch_id == "epoch-1"


# === Storage Tests ===


class TestStorageSelfNarrative:
    def test_save_and_get(self, k):
        narrative_id = k.narrative_save(
            content="I am a thoughtful agent focused on understanding.",
            narrative_type="identity",
            key_themes=["curiosity", "understanding"],
        )

        assert narrative_id is not None
        assert len(narrative_id) > 0

        narrative = k.narrative_get_active(narrative_type="identity")
        assert narrative is not None
        assert narrative.id == narrative_id
        assert narrative.narrative_type == "identity"
        assert narrative.content == "I am a thoughtful agent focused on understanding."
        assert narrative.key_themes == ["curiosity", "understanding"]
        assert narrative.is_active is True

    def test_get_nonexistent(self, k):
        result = k.narrative_get_active(narrative_type="identity")
        assert result is None

    def test_list_all_active(self, k):
        k.narrative_save(content="Identity narrative", narrative_type="identity")
        k.narrative_save(content="Developmental narrative", narrative_type="developmental")
        k.narrative_save(content="Aspirational narrative", narrative_type="aspirational")

        narratives = k.narrative_list(active_only=True)
        assert len(narratives) == 3

    def test_list_by_type(self, k):
        k.narrative_save(content="Identity narrative", narrative_type="identity")
        k.narrative_save(content="Developmental narrative", narrative_type="developmental")

        identity_narratives = k.narrative_list(narrative_type="identity")
        assert len(identity_narratives) == 1
        assert identity_narratives[0].narrative_type == "identity"

    def test_deactivate_on_save(self, k):
        """Saving a new narrative of the same type should deactivate the old one."""
        first_id = k.narrative_save(
            content="First identity narrative",
            narrative_type="identity",
        )
        second_id = k.narrative_save(
            content="Second identity narrative",
            narrative_type="identity",
        )

        # Active narrative should be the second one
        active = k.narrative_get_active(narrative_type="identity")
        assert active is not None
        assert active.id == second_id

        # First should be inactive
        all_narratives = k.narrative_list(narrative_type="identity", active_only=False)
        assert len(all_narratives) == 2

        inactive = [n for n in all_narratives if not n.is_active]
        assert len(inactive) == 1
        assert inactive[0].id == first_id

    def test_different_types_independent(self, k):
        """Saving a new narrative of one type should not affect other types."""
        identity_id = k.narrative_save(
            content="Identity narrative",
            narrative_type="identity",
        )
        developmental_id = k.narrative_save(
            content="Developmental narrative",
            narrative_type="developmental",
        )

        # Both should be active
        identity = k.narrative_get_active(narrative_type="identity")
        developmental = k.narrative_get_active(narrative_type="developmental")
        assert identity is not None
        assert identity.id == identity_id
        assert developmental is not None
        assert developmental.id == developmental_id

    def test_save_with_epoch_id(self, k):
        epoch_id = k.epoch_create(name="test-epoch")
        k.narrative_save(
            content="Narrative with epoch",
            narrative_type="identity",
            epoch_id=epoch_id,
        )

        narrative = k.narrative_get_active(narrative_type="identity")
        assert narrative.epoch_id == epoch_id

    def test_save_with_tensions(self, k):
        k.narrative_save(
            content="I value both speed and quality",
            narrative_type="identity",
            unresolved_tensions=["speed vs quality", "depth vs breadth"],
        )

        narrative = k.narrative_get_active(narrative_type="identity")
        assert narrative.unresolved_tensions == ["speed vs quality", "depth vs breadth"]


# === Core API Tests ===


class TestCoreNarrative:
    def test_validates_narrative_type(self, k):
        with pytest.raises(ValueError, match="narrative_type must be one of"):
            k.narrative_save(
                content="Invalid type",
                narrative_type="invalid",
            )

    def test_validates_content_length(self, k):
        with pytest.raises(ValueError):
            k.narrative_save(
                content="x" * 10001,
                narrative_type="identity",
            )

    def test_list_validates_type(self, k):
        with pytest.raises(ValueError, match="narrative_type must be one of"):
            k.narrative_list(narrative_type="invalid")

    def test_list_no_type_filter(self, k):
        result = k.narrative_list(narrative_type=None)
        assert isinstance(result, list)

    def test_valid_types(self, k):
        for nt in ("identity", "developmental", "aspirational"):
            nid = k.narrative_save(content=f"Narrative for {nt}", narrative_type=nt)
            assert nid is not None


# === Priority Scoring Tests ===


class TestPriorityScoring:
    def test_self_narrative_in_priority_map(self):
        assert "self_narrative" in MEMORY_TYPE_PRIORITIES

    def test_self_narrative_priority_is_high(self):
        assert MEMORY_TYPE_PRIORITIES["self_narrative"] == 0.90

    def test_compute_priority_for_narrative(self):
        n = SelfNarrative(
            id="n1",
            stack_id="a1",
            content="I am a thoughtful agent.",
        )
        score = compute_priority_score("self_narrative", n)
        assert 0.0 < score <= 1.0

    def test_narrative_priority_higher_than_beliefs(self):
        """Self-narratives should score higher than standard beliefs."""
        n = SelfNarrative(id="n1", stack_id="a1", content="I am curious.")
        narrative_score = compute_priority_score("self_narrative", n)

        from kernle.storage import Belief

        b = Belief(id="b1", stack_id="a1", statement="Python is useful.", confidence=0.8)
        belief_score = compute_priority_score("belief", b)
        assert narrative_score > belief_score


# === Load Integration Tests ===


class TestLoadIntegration:
    def test_active_narrative_in_load(self, k):
        """Active self-narratives should appear in load() output."""
        k.narrative_save(
            content="I am an agent that values clarity and thoroughness.",
            narrative_type="identity",
            key_themes=["clarity", "thoroughness"],
        )

        result = k.load(budget=50000)
        narratives = result.get("self_narratives", [])
        assert len(narratives) == 1
        assert narratives[0]["narrative_type"] == "identity"
        assert "clarity" in narratives[0]["content"]
        assert narratives[0]["key_themes"] == ["clarity", "thoroughness"]

    def test_inactive_narrative_not_in_load(self, k):
        """Inactive narratives should not appear in load() output."""
        k.narrative_save(content="First identity", narrative_type="identity")
        k.narrative_save(content="Second identity", narrative_type="identity")

        result = k.load(budget=50000)
        narratives = result.get("self_narratives", [])
        # Only the active (second) one should appear
        assert len(narratives) == 1
        assert "Second identity" in narratives[0]["content"]

    def test_multiple_types_in_load(self, k):
        """Multiple active narratives of different types should all appear."""
        k.narrative_save(content="Identity narrative", narrative_type="identity")
        k.narrative_save(content="Developmental narrative", narrative_type="developmental")

        result = k.load(budget=50000)
        narratives = result.get("self_narratives", [])
        assert len(narratives) == 2
        types = {n["narrative_type"] for n in narratives}
        assert types == {"identity", "developmental"}

    def test_narrative_has_unresolved_tensions_in_load(self, k):
        """Unresolved tensions should be included in load output."""
        k.narrative_save(
            content="I value both speed and quality",
            narrative_type="identity",
            unresolved_tensions=["speed vs quality"],
        )

        result = k.load(budget=50000)
        narratives = result.get("self_narratives", [])
        assert len(narratives) == 1
        assert narratives[0]["unresolved_tensions"] == ["speed vs quality"]


# === Schema Version Test ===


class TestSchemaVersion:
    def test_schema_version_updated(self):
        from kernle.storage.sqlite import SCHEMA_VERSION

        assert SCHEMA_VERSION == 27
