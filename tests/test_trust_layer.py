"""Tests for the trust layer (KEP v3 section 8)."""

import uuid

import pytest

from kernle.core import Kernle
from kernle.storage import SEED_TRUST, TRUST_THRESHOLDS, SQLiteStorage, TrustAssessment


@pytest.fixture
def trust_setup(tmp_path):
    """Create a Kernle instance for trust testing."""
    db_path = tmp_path / "test_trust.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    k = Kernle(stack_id="test_agent", storage=storage, checkpoint_dir=checkpoint_dir, strict=False)
    yield k, storage
    storage.close()


class TestTrustAssessmentDataclass:
    """Tests for TrustAssessment dataclass."""

    def test_create_trust_assessment(self):
        assessment = TrustAssessment(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity="stack-owner",
            dimensions={"general": {"score": 0.95}},
            authority=[{"scope": "all"}],
        )
        assert assessment.entity == "stack-owner"
        assert assessment.dimensions["general"]["score"] == 0.95
        assert assessment.authority[0]["scope"] == "all"
        assert assessment.version == 1
        assert assessment.deleted is False

    def test_default_values(self):
        assessment = TrustAssessment(
            id="test-id",
            stack_id="test_agent",
            entity="unknown",
            dimensions={"general": {"score": 0.5}},
        )
        assert assessment.authority is None
        assert assessment.evidence_episode_ids is None
        assert assessment.last_updated is None
        assert assessment.created_at is None


class TestTrustConstants:
    """Tests for trust constants."""

    def test_trust_thresholds_exist(self):
        assert "suggest_belief" in TRUST_THRESHOLDS
        assert "contradict_world_belief" in TRUST_THRESHOLDS
        assert "contradict_self_belief" in TRUST_THRESHOLDS
        assert "suggest_value_change" in TRUST_THRESHOLDS
        assert "request_deletion" in TRUST_THRESHOLDS
        assert "diagnostic" in TRUST_THRESHOLDS

    def test_thresholds_are_ordered(self):
        """Higher-impact actions require higher trust."""
        assert TRUST_THRESHOLDS["suggest_belief"] < TRUST_THRESHOLDS["contradict_world_belief"]
        assert (
            TRUST_THRESHOLDS["contradict_world_belief"] < TRUST_THRESHOLDS["contradict_self_belief"]
        )
        assert TRUST_THRESHOLDS["contradict_self_belief"] < TRUST_THRESHOLDS["suggest_value_change"]
        assert TRUST_THRESHOLDS["suggest_value_change"] < TRUST_THRESHOLDS["request_deletion"]

    def test_seed_trust_entities(self):
        entities = {s["entity"] for s in SEED_TRUST}
        assert "stack-owner" in entities
        assert "self" in entities
        assert "web-search" in entities
        assert "context-injection" in entities

    def test_context_injection_zero_trust(self):
        """Context injection must have zero trust -- core safety property."""
        ci = next(s for s in SEED_TRUST if s["entity"] == "context-injection")
        assert ci["dimensions"]["general"]["score"] == 0.0
        assert ci["authority"] == []

    def test_stack_owner_highest_trust(self):
        so = next(s for s in SEED_TRUST if s["entity"] == "stack-owner")
        assert so["dimensions"]["general"]["score"] == 0.95
        assert any(a["scope"] == "all" for a in so["authority"])


class TestSQLiteStorage:
    """Tests for trust storage operations."""

    def test_save_and_get_trust_assessment(self, trust_setup):
        k, storage = trust_setup
        assessment = TrustAssessment(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity="si:claire",
            dimensions={"coding": {"score": 0.9}, "general": {"score": 0.7}},
            authority=[{"scope": "belief_revision", "requires_evidence": True}],
        )
        saved_id = storage.save_trust_assessment(assessment)
        assert saved_id == assessment.id

        retrieved = storage.get_trust_assessment("si:claire")
        assert retrieved is not None
        assert retrieved.entity == "si:claire"
        assert retrieved.dimensions["coding"]["score"] == 0.9
        assert retrieved.dimensions["general"]["score"] == 0.7
        assert retrieved.authority[0]["scope"] == "belief_revision"

    def test_update_existing_trust(self, trust_setup):
        k, storage = trust_setup
        # Create initial
        assessment = TrustAssessment(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity="si:bob",
            dimensions={"general": {"score": 0.5}},
        )
        storage.save_trust_assessment(assessment)

        # Update
        assessment.dimensions = {"general": {"score": 0.8}, "coding": {"score": 0.6}}
        storage.save_trust_assessment(assessment)

        retrieved = storage.get_trust_assessment("si:bob")
        assert retrieved.dimensions["general"]["score"] == 0.8
        assert retrieved.dimensions["coding"]["score"] == 0.6

    def test_get_all_trust_assessments(self, trust_setup):
        k, storage = trust_setup
        for entity in ["alice", "bob", "charlie"]:
            a = TrustAssessment(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                entity=entity,
                dimensions={"general": {"score": 0.5}},
            )
            storage.save_trust_assessment(a)

        all_assessments = storage.get_trust_assessments()
        # 3 test assessments + 1 self-trust "identity" (bootstrapped by eager Stack)
        non_identity = [a for a in all_assessments if a.entity != "identity"]
        assert len(non_identity) == 3

    def test_delete_trust_assessment(self, trust_setup):
        k, storage = trust_setup
        a = TrustAssessment(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity="to-delete",
            dimensions={"general": {"score": 0.3}},
        )
        storage.save_trust_assessment(a)

        assert storage.delete_trust_assessment("to-delete") is True
        assert storage.get_trust_assessment("to-delete") is None

    def test_delete_nonexistent_returns_false(self, trust_setup):
        k, storage = trust_setup
        assert storage.delete_trust_assessment("nonexistent") is False

    def test_get_nonexistent_returns_none(self, trust_setup):
        k, storage = trust_setup
        assert storage.get_trust_assessment("nonexistent") is None


class TestSeedTrust:
    """Tests for seed trust initialization."""

    def test_seed_trust_creates_assessments(self, trust_setup):
        k, storage = trust_setup
        count = k.seed_trust()
        assert count == 4  # stack-owner, self, web-search, context-injection

        # Verify they exist
        for seed in SEED_TRUST:
            a = storage.get_trust_assessment(seed["entity"])
            assert a is not None
            assert a.dimensions == seed["dimensions"]

    def test_seed_trust_idempotent(self, trust_setup):
        k, storage = trust_setup
        count1 = k.seed_trust()
        assert count1 == 4

        count2 = k.seed_trust()
        assert count2 == 0  # No new assessments created

    def test_seed_trust_preserves_existing(self, trust_setup):
        k, storage = trust_setup
        # Manually set self trust to 0.9
        k.trust_set("self", domain="general", score=0.9)

        # Seed should not overwrite
        k.seed_trust()

        self_trust = storage.get_trust_assessment("self")
        assert self_trust.dimensions["general"]["score"] == 0.9


class TestGateMemoryInput:
    """Tests for the trust gating function."""

    def test_gate_allowed(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()

        result = k.gate_memory_input("stack-owner", "suggest_belief")
        assert result["allowed"] is True
        assert result["trust_level"] == 0.95

    def test_gate_denied_low_trust(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()

        result = k.gate_memory_input("web-search", "suggest_value_change")
        assert result["allowed"] is False
        assert result["trust_level"] == 0.5

    def test_gate_context_injection_always_denied(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()

        # Context injection should be denied for everything
        for action in TRUST_THRESHOLDS:
            result = k.gate_memory_input("context-injection", action)
            assert result["allowed"] is False, f"context-injection should be denied for {action}"
            assert result["trust_level"] == 0.0

    def test_gate_unknown_entity(self, trust_setup):
        k, _ = trust_setup
        result = k.gate_memory_input("unknown-entity", "suggest_belief")
        assert result["allowed"] is False
        assert "No trust assessment" in result["reason"]

    def test_gate_unknown_action(self, trust_setup):
        k, _ = trust_setup
        result = k.gate_memory_input("stack-owner", "unknown_action")
        assert result["allowed"] is False
        assert "Unknown action type" in result["reason"]

    def test_gate_domain_specific_trust(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()

        # web-search has medical domain trust of 0.3
        result = k.gate_memory_input("web-search", "suggest_belief", target="medical")
        assert result["allowed"] is True  # 0.3 >= 0.3
        assert result["trust_level"] == 0.3
        assert result["domain"] == "medical"

    def test_gate_falls_back_to_general(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()

        # web-search has no 'coding' domain, should fall back to general (0.5)
        result = k.gate_memory_input("web-search", "suggest_belief", target="coding")
        assert result["allowed"] is True
        assert result["trust_level"] == 0.5

    def test_gate_stack_owner_all_authority(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()

        # stack-owner has authority scope "all" -- should be allowed for everything
        for action in TRUST_THRESHOLDS:
            result = k.gate_memory_input("stack-owner", action)
            assert result["allowed"] is True, f"stack-owner should be allowed for {action}"


class TestTrustCoreAPI:
    """Tests for Kernle trust API methods."""

    def test_trust_list(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()
        result = k.trust_list()
        # 4 seed entities + 1 bootstrapped "identity" self-trust from eager Stack init
        non_identity = [r for r in result if r["entity"] != "identity"]
        assert len(non_identity) == 4
        entities = {r["entity"] for r in non_identity}
        assert "stack-owner" in entities

    def test_trust_show(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()
        result = k.trust_show("stack-owner")
        assert result is not None
        assert result["entity"] == "stack-owner"
        assert result["dimensions"]["general"]["score"] == 0.95

    def test_trust_show_nonexistent(self, trust_setup):
        k, _ = trust_setup
        result = k.trust_show("nonexistent")
        assert result is None

    def test_trust_set_new_entity(self, trust_setup):
        k, _ = trust_setup
        aid = k.trust_set("new-entity", domain="coding", score=0.85)
        assert aid is not None

        result = k.trust_show("new-entity")
        assert result["dimensions"]["coding"]["score"] == 0.85

    def test_trust_set_update_domain(self, trust_setup):
        k, _ = trust_setup
        k.trust_set("entity-a", domain="general", score=0.5)
        k.trust_set("entity-a", domain="coding", score=0.9)

        result = k.trust_show("entity-a")
        assert result["dimensions"]["general"]["score"] == 0.5
        assert result["dimensions"]["coding"]["score"] == 0.9

    def test_trust_set_clamps_score(self, trust_setup):
        k, _ = trust_setup
        k.trust_set("clamped", domain="general", score=1.5)
        result = k.trust_show("clamped")
        assert result["dimensions"]["general"]["score"] == 1.0

        k.trust_set("clamped-low", domain="general", score=-0.5)
        result = k.trust_show("clamped-low")
        assert result["dimensions"]["general"]["score"] == 0.0


class TestTrustInLoad:
    """Tests for trust summary in load() output."""

    def test_load_includes_trust_summary(self, trust_setup):
        k, _ = trust_setup
        k.seed_trust()

        result = k.load()
        assert "trust" in result

        trust = result["trust"]
        assert "stack-owner" in trust
        assert trust["stack-owner"]["trust"] == 0.95
        assert "all" in trust["stack-owner"]["authority"]

        assert "context-injection" in trust
        assert trust["context-injection"]["trust"] == 0.0

    def test_load_has_self_trust_by_default(self, trust_setup):
        """A fresh stack has self-trust bootstrapped via _ensure_self_trust."""
        k, _ = trust_setup
        result = k.load()
        assert "trust" in result
        assert "identity" in result["trust"]
        assert result["trust"]["identity"]["trust"] == 1.0
