"""Tests for kernle doctor structural command."""

import argparse
import json
import uuid
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import patch

import pytest

from kernle.cli.commands.doctor import cmd_doctor_structural
from kernle.storage import Belief, Episode, Goal, Note, Relationship
from kernle.structural import (
    StructuralFinding,
    check_belief_contradictions,
    check_low_confidence_beliefs,
    check_orphaned_references,
    check_stale_goals,
    check_stale_relationships,
    run_structural_checks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kernle_instance(tmp_path):
    """Kernle instance with SQLite storage for testing."""
    from kernle.core import Kernle
    from kernle.storage import SQLiteStorage

    db_path = tmp_path / "test_structural.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    k = Kernle(stack_id="test_agent", storage=storage, checkpoint_dir=checkpoint_dir, strict=False)
    yield k
    storage.close()


# ---------------------------------------------------------------------------
# StructuralFinding
# ---------------------------------------------------------------------------


class TestStructuralFinding:
    def test_to_dict(self):
        f = StructuralFinding(
            check="orphaned_reference",
            severity="error",
            memory_type="belief",
            memory_id="abc123",
            message="test message",
        )
        d = f.to_dict()
        assert d["check"] == "orphaned_reference"
        assert d["severity"] == "error"
        assert d["memory_type"] == "belief"
        assert d["memory_id"] == "abc123"
        assert d["message"] == "test message"


# ---------------------------------------------------------------------------
# Orphaned References
# ---------------------------------------------------------------------------


class TestOrphanedReferences:
    def test_no_orphans_when_empty(self, kernle_instance):
        findings = check_orphaned_references(kernle_instance)
        assert findings == []

    def test_no_orphans_with_valid_refs(self, kernle_instance):
        k = kernle_instance
        ep_id = str(uuid.uuid4())
        ep = Episode(
            id=ep_id,
            stack_id="test_agent",
            objective="test",
            outcome="ok",
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_episode(ep)

        belief = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="derived fact",
            confidence=0.8,
            derived_from=[f"episode:{ep_id}"],
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(belief)

        findings = check_orphaned_references(k)
        assert findings == []

    def test_detects_orphaned_derived_from(self, kernle_instance):
        k = kernle_instance
        belief = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="orphan ref test",
            confidence=0.8,
            derived_from=["episode:nonexistent-id"],
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(belief)

        findings = check_orphaned_references(k)
        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert findings[0].check == "orphaned_reference"
        assert "nonexistent-id" in findings[0].message

    def test_detects_orphaned_source_episodes(self, kernle_instance):
        k = kernle_instance
        note = Note(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            content="test note",
            source_episodes=["episode:does-not-exist"],
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_note(note)

        findings = check_orphaned_references(k)
        assert len(findings) == 1
        assert findings[0].check == "orphaned_reference"


# ---------------------------------------------------------------------------
# Low Confidence Beliefs
# ---------------------------------------------------------------------------


class TestLowConfidenceBeliefs:
    def test_no_findings_when_empty(self, kernle_instance):
        findings = check_low_confidence_beliefs(kernle_instance)
        assert findings == []

    def test_no_findings_above_threshold(self, kernle_instance):
        k = kernle_instance
        belief = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="confident belief",
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(belief)

        findings = check_low_confidence_beliefs(k)
        assert findings == []

    def test_detects_low_confidence(self, kernle_instance):
        k = kernle_instance
        belief = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="uncertain belief",
            confidence=0.1,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(belief)

        findings = check_low_confidence_beliefs(k)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert findings[0].check == "low_confidence_belief"
        assert "0.10" in findings[0].message

    def test_respects_custom_threshold(self, kernle_instance):
        k = kernle_instance
        belief = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="moderate belief",
            confidence=0.4,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(belief)

        # Default threshold 0.3 -- this should pass
        findings = check_low_confidence_beliefs(k, threshold=0.3)
        assert findings == []

        # Raise threshold -- this should now be flagged
        findings = check_low_confidence_beliefs(k, threshold=0.5)
        assert len(findings) == 1

    def test_includes_last_verified_info(self, kernle_instance):
        k = kernle_instance
        verified_date = datetime.now(timezone.utc) - timedelta(days=30)
        belief = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="low and verified",
            confidence=0.2,
            last_verified=verified_date,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(belief)

        findings = check_low_confidence_beliefs(k)
        assert len(findings) == 1
        assert "30d ago" in findings[0].message

    def test_never_verified_message(self, kernle_instance):
        k = kernle_instance
        belief = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="never verified belief",
            confidence=0.1,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(belief)

        findings = check_low_confidence_beliefs(k)
        assert len(findings) == 1
        assert "never verified" in findings[0].message


# ---------------------------------------------------------------------------
# Stale Relationships
# ---------------------------------------------------------------------------


class TestStaleRelationships:
    def test_no_findings_when_empty(self, kernle_instance):
        findings = check_stale_relationships(kernle_instance)
        assert findings == []

    def test_detects_zero_interactions(self, kernle_instance):
        k = kernle_instance
        rel = Relationship(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity_name="Alice",
            entity_type="human",
            relationship_type="collaborator",
            interaction_count=0,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_relationship(rel)

        findings = check_stale_relationships(k)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "zero interactions" in findings[0].message

    def test_detects_stale_last_interaction(self, kernle_instance):
        k = kernle_instance
        old_date = datetime.now(timezone.utc) - timedelta(days=120)
        rel = Relationship(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity_name="Bob",
            entity_type="human",
            relationship_type="mentor",
            interaction_count=5,
            last_interaction=old_date,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_relationship(rel)

        findings = check_stale_relationships(k)
        assert len(findings) == 1
        assert findings[0].severity == "info"
        assert "120d ago" in findings[0].message

    def test_no_findings_for_recent_interaction(self, kernle_instance):
        k = kernle_instance
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)
        rel = Relationship(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity_name="Carol",
            entity_type="human",
            relationship_type="colleague",
            interaction_count=3,
            last_interaction=recent_date,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_relationship(rel)

        findings = check_stale_relationships(k)
        assert findings == []


# ---------------------------------------------------------------------------
# Belief Contradictions
# ---------------------------------------------------------------------------


class TestBeliefContradictions:
    def test_no_findings_when_empty(self, kernle_instance):
        findings = check_belief_contradictions(kernle_instance)
        assert findings == []

    def test_detects_contradiction(self, kernle_instance):
        k = kernle_instance
        b1 = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="Testing should always be comprehensive and thorough",
            confidence=0.9,
            created_at=datetime.now(timezone.utc),
        )
        b2 = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="Testing should never be comprehensive or thorough",
            confidence=0.7,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(b1)
        k._storage.save_belief(b2)

        findings = check_belief_contradictions(k)
        assert len(findings) == 1
        assert findings[0].check == "belief_contradiction"
        assert findings[0].severity == "warning"

    def test_no_contradiction_for_unrelated_beliefs(self, kernle_instance):
        k = kernle_instance
        b1 = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="Python is a great programming language",
            confidence=0.9,
            created_at=datetime.now(timezone.utc),
        )
        b2 = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="Coffee helps with productivity",
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
        )
        k._storage.save_belief(b1)
        k._storage.save_belief(b2)

        findings = check_belief_contradictions(k)
        assert findings == []


# ---------------------------------------------------------------------------
# Stale Goals
# ---------------------------------------------------------------------------


class TestStaleGoals:
    def test_no_findings_when_empty(self, kernle_instance):
        findings = check_stale_goals(kernle_instance)
        assert findings == []

    def test_no_findings_for_recent_goal(self, kernle_instance):
        k = kernle_instance
        goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Recent goal",
            status="active",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        k._storage.save_goal(goal)

        findings = check_stale_goals(k)
        assert findings == []

    def test_detects_stale_active_goal(self, kernle_instance):
        k = kernle_instance
        goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Old goal",
            status="active",
            created_at=datetime.now(timezone.utc) - timedelta(days=90),
        )
        k._storage.save_goal(goal)

        findings = check_stale_goals(k)
        assert len(findings) == 1
        assert findings[0].severity == "info"
        assert findings[0].check == "stale_goal"
        assert "90d old" in findings[0].message

    def test_no_findings_for_completed_goal(self, kernle_instance):
        k = kernle_instance
        goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Completed goal",
            status="completed",
            created_at=datetime.now(timezone.utc) - timedelta(days=90),
        )
        k._storage.save_goal(goal)

        # completed goals are not returned by get_goals(status="active")
        findings = check_stale_goals(k)
        assert findings == []


# ---------------------------------------------------------------------------
# run_structural_checks (integration)
# ---------------------------------------------------------------------------


class TestRunStructuralChecks:
    def test_returns_empty_for_clean_db(self, kernle_instance):
        findings = run_structural_checks(kernle_instance)
        assert findings == []

    def test_aggregates_multiple_findings(self, kernle_instance):
        k = kernle_instance

        # Add a low-confidence belief
        k._storage.save_belief(
            Belief(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                statement="uncertain thing",
                confidence=0.1,
                created_at=datetime.now(timezone.utc),
            )
        )

        # Add a zero-interaction relationship
        k._storage.save_relationship(
            Relationship(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                entity_name="Ghost",
                entity_type="human",
                relationship_type="acquaintance",
                interaction_count=0,
                created_at=datetime.now(timezone.utc),
            )
        )

        findings = run_structural_checks(k)
        checks_found = {f.check for f in findings}
        assert "low_confidence_belief" in checks_found
        assert "stale_relationship" in checks_found


# ---------------------------------------------------------------------------
# cmd_doctor_structural (CLI output)
# ---------------------------------------------------------------------------


class TestCmdDoctorStructural:
    def test_clean_output(self, kernle_instance):
        args = argparse.Namespace(json=False, save_note=False)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor_structural(args, kernle_instance)
        output = fake_out.getvalue()
        assert "Structural Health Check" in output
        assert "healthy" in output.lower() or "passed" in output.lower()

    def test_json_output(self, kernle_instance):
        k = kernle_instance
        k._storage.save_belief(
            Belief(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                statement="low conf",
                confidence=0.1,
                created_at=datetime.now(timezone.utc),
            )
        )
        args = argparse.Namespace(json=True, save_note=False)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor_structural(args, k)
        output = json.loads(fake_out.getvalue())
        assert "summary" in output
        assert "findings" in output
        assert output["summary"]["warnings"] >= 1

    def test_text_output_with_findings(self, kernle_instance):
        k = kernle_instance
        k._storage.save_relationship(
            Relationship(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                entity_name="Phantom",
                entity_type="bot",
                relationship_type="peer",
                interaction_count=0,
                created_at=datetime.now(timezone.utc),
            )
        )
        args = argparse.Namespace(json=False, save_note=False)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor_structural(args, k)
        output = fake_out.getvalue()
        assert "WARNINGS" in output
        assert "zero interactions" in output

    def test_save_note_flag(self, kernle_instance):
        k = kernle_instance
        k._storage.save_belief(
            Belief(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                statement="flagged",
                confidence=0.05,
                created_at=datetime.now(timezone.utc),
            )
        )
        args = argparse.Namespace(json=False, save_note=True)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor_structural(args, k)
        output = fake_out.getvalue()
        assert "diagnostic note" in output.lower()

        # Verify note was actually saved
        notes = k._storage.get_notes(limit=10, note_type="diagnostic")
        assert len(notes) >= 1
        assert "Structural health check" in notes[0].content

        # Verify provenance fields
        assert notes[0].source_type == "observation"
        assert notes[0].source_entity == "kernle:doctor"

    def test_save_note_not_saved_when_no_findings(self, kernle_instance):
        args = argparse.Namespace(json=False, save_note=True)
        with patch("sys.stdout", new=StringIO()):
            cmd_doctor_structural(args, kernle_instance)

        notes = kernle_instance._storage.get_notes(limit=10, note_type="diagnostic")
        assert len(notes) == 0

    def test_privacy_no_content_in_output(self, kernle_instance):
        """Verify that actual belief statement content is NOT in the output."""
        k = kernle_instance
        secret_statement = "My secret inner thought about the meaning of life"
        k._storage.save_belief(
            Belief(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                statement=secret_statement,
                confidence=0.1,
                created_at=datetime.now(timezone.utc),
            )
        )
        args = argparse.Namespace(json=True, save_note=False)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor_structural(args, k)
        output = fake_out.getvalue()
        assert secret_statement not in output
