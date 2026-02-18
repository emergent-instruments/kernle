"""
Integration tests for the Kernle CLI.

These tests use real SQLite storage and real Kernle instances,
exercising actual database operations instead of mocks.
"""

import argparse
import json
from io import StringIO
from unittest.mock import patch

import pytest

from kernle.cli.__main__ import (
    cmd_checkpoint,
    cmd_episode,
    cmd_load,
    cmd_note,
    cmd_status,
)
from kernle.cli.commands import (
    cmd_anxiety,
    cmd_belief,
    cmd_doctor,
    cmd_doctor_structural,
    cmd_epoch,
    cmd_identity,
    cmd_narrative,
    cmd_summary,
)
from kernle.cli.commands.trust import cmd_trust
from kernle.core import Kernle
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def cli_kernle(tmp_path):
    """Create a real Kernle instance with SQLite storage for CLI testing."""
    db_path = tmp_path / "cli_test.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    storage = SQLiteStorage(stack_id="cli_test_agent", db_path=db_path)
    k = Kernle(
        stack_id="cli_test_agent", storage=storage, checkpoint_dir=checkpoint_dir, strict=False
    )
    bind_noop_model(k)

    yield k, storage
    storage.close()


class TestIdentityCommands:
    """Integration tests for identity commands with real storage."""

    def test_identity_show(self, cli_kernle):
        """Test identity show synthesizes from real data."""
        k, storage = cli_kernle

        # Seed some data so identity synthesis has material
        k.belief("Testing is important", confidence=0.9)
        k.episode("Built the auth system", "success", lessons=["Always hash passwords"])

        args = argparse.Namespace(identity_action="show", json=False)

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_identity(args, k)

        output = out.getvalue()
        assert "Identity Synthesis" in output
        assert "cli_test_agent" in output

    def test_identity_confidence(self, cli_kernle):
        """Test identity confidence returns a real score."""
        k, storage = cli_kernle

        k.belief("I value quality", confidence=0.85)

        args = argparse.Namespace(identity_action="confidence", json=True)

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_identity(args, k)

        result = json.loads(out.getvalue())
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))
        assert 0.0 <= result["confidence"] <= 1.0


class TestBeliefCommands:
    """Integration tests for belief commands with real storage."""

    def test_belief_list(self, cli_kernle):
        """Test listing beliefs after adding via Kernle API."""
        k, storage = cli_kernle

        # Add beliefs via the Kernle API
        k.belief("Testing leads to quality software", confidence=0.9)
        k.belief("Code review improves reliability", confidence=0.8)

        args = argparse.Namespace(
            belief_action="list",
            all=False,
            limit=20,
            json=False,
            scope=None,
            domain=None,
            abstraction_level=None,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_belief(args, k)

        output = out.getvalue()
        assert "Beliefs" in output
        assert "Testing leads to quality software" in output
        assert "Code review improves reliability" in output

    def test_belief_list_json(self, cli_kernle):
        """Test belief list with JSON output returns valid JSON."""
        k, storage = cli_kernle

        k.belief("Integration testing matters", confidence=0.75)

        args = argparse.Namespace(
            belief_action="list",
            all=False,
            limit=20,
            json=True,
            scope=None,
            domain=None,
            abstraction_level=None,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_belief(args, k)

        data = json.loads(out.getvalue())
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any("Integration testing matters" in b["statement"] for b in data)

    def test_belief_reinforce(self, cli_kernle):
        """Test reinforcing a belief updates its state."""
        k, storage = cli_kernle

        belief_id = k.belief("Persistence pays off", confidence=0.7)

        args = argparse.Namespace(
            belief_action="reinforce",
            id=belief_id,
            evidence=None,
            reason=None,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_belief(args, k)

        output = out.getvalue()
        assert "reinforced" in output

        # Verify in storage
        beliefs = storage.get_beliefs(limit=100)
        matching = [b for b in beliefs if b.id == belief_id]
        assert len(matching) == 1
        assert matching[0].times_reinforced >= 1


class TestEpisodeCommand:
    """Integration tests for episode recording."""

    def test_episode_record(self, cli_kernle):
        """Test recording an episode stores it in the database."""
        k, storage = cli_kernle

        args = argparse.Namespace(
            objective="Implemented user auth",
            outcome="success",
            lesson=["Always validate tokens", "Use bcrypt for hashing"],
            tag=["security", "backend"],
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
            valence=None,
            arousal=None,
            emotion=None,
            auto_emotion=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_episode(args, k)

        output = out.getvalue()
        assert "Episode saved:" in output
        assert "Lessons: 2" in output

        # Verify in storage
        episodes = storage.get_episodes()
        matching = [e for e in episodes if e.objective == "Implemented user auth"]
        assert len(matching) == 1
        assert matching[0].outcome_type == "success"
        assert "Always validate tokens" in matching[0].lessons


class TestNoteCommand:
    """Integration tests for note capture."""

    def test_note_add(self, cli_kernle):
        """Test adding a note stores it in the database."""
        k, storage = cli_kernle

        args = argparse.Namespace(
            content="Use React for frontend framework",
            type="decision",
            speaker=None,
            reason="Better ecosystem",
            tag=["frontend", "architecture"],
            protect=False,
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_note(args, k)

        output = out.getvalue()
        assert "Note saved:" in output

        # Verify in storage
        notes = storage.get_notes()
        matching = [n for n in notes if "React" in n.content]
        assert len(matching) == 1
        assert matching[0].note_type == "decision"


class TestLoadCommand:
    """Integration tests for the load command."""

    def test_load_with_real_data(self, cli_kernle):
        """Test load outputs formatted memory with real data."""
        k, storage = cli_kernle

        k.episode("Load test episode", "success", lessons=["Important lesson"])
        k.note("Load test decision", type="decision", reason="Testing")

        args = argparse.Namespace(
            json=False, sync=False, no_sync=False, budget=8000, no_truncate=False
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_load(args, k)

        output = out.getvalue()
        assert "Working Memory" in output or "Memory" in output

    def test_load_json(self, cli_kernle):
        """Test load with JSON output returns valid JSON."""
        k, storage = cli_kernle

        k.episode("JSON test episode", "success")

        args = argparse.Namespace(
            json=True, sync=False, no_sync=False, budget=8000, no_truncate=False
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_load(args, k)

        data = json.loads(out.getvalue())
        assert isinstance(data, dict)


class TestAnxietyCommand:
    """Integration tests for the anxiety command."""

    def test_anxiety_brief(self, cli_kernle):
        """Test brief anxiety check with real storage."""
        k, storage = cli_kernle

        args = argparse.Namespace(
            brief=True,
            detailed=False,
            actions=False,
            auto=False,
            context=None,
            limit=200000,
            emergency=False,
            summary=None,
            json=False,
            source="cli",
            triggered_by="manual",
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_anxiety(args, k)

        output = out.getvalue()
        # Brief mode outputs a single line with status
        assert "OK" in output or "WARN" in output or "CRITICAL" in output

    def test_anxiety_json(self, cli_kernle):
        """Test anxiety report JSON output with real storage."""
        k, storage = cli_kernle

        args = argparse.Namespace(
            brief=False,
            detailed=False,
            actions=False,
            auto=False,
            context=None,
            limit=200000,
            emergency=False,
            summary=None,
            json=True,
            source="cli",
            triggered_by="manual",
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_anxiety(args, k)

        report = json.loads(out.getvalue())
        assert "overall_score" in report
        assert "dimensions" in report
        assert isinstance(report["overall_score"], (int, float))


class TestEpochCommands:
    """Integration tests for epoch commands."""

    def test_epoch_create_and_list(self, cli_kernle):
        """Test creating and listing epochs with real storage."""
        k, storage = cli_kernle

        # Create an epoch
        create_args = argparse.Namespace(
            epoch_action="create",
            name="Integration Test Era",
            trigger="declared",
            trigger_description="Starting integration tests",
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(create_args, k)

        output = out.getvalue()
        assert "Epoch created" in output
        assert "Integration Test Era" in output

        # List epochs
        list_args = argparse.Namespace(
            epoch_action="list",
            limit=20,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(list_args, k)

        output = out.getvalue()
        assert "Integration Test Era" in output
        assert "ACTIVE" in output

    def test_epoch_current(self, cli_kernle):
        """Test getting the current active epoch."""
        k, storage = cli_kernle

        k.epoch_create(name="Current Epoch Test", trigger_type="declared")

        args = argparse.Namespace(
            epoch_action="current",
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        assert data is not None
        assert data["name"] == "Current Epoch Test"


class TestTrustCommands:
    """Integration tests for trust commands."""

    def test_trust_set_and_list(self, cli_kernle):
        """Test setting and listing trust with real storage."""
        k, storage = cli_kernle

        # Set trust
        set_args = argparse.Namespace(
            trust_action="set",
            entity="operator",
            score=0.85,
            domain="general",
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(set_args, k)

        output = out.getvalue()
        assert "Trust set" in output
        assert "operator" in output

        # List trust
        list_args = argparse.Namespace(trust_action="list")

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(list_args, k)

        output = out.getvalue()
        assert "operator" in output

    def test_trust_seed(self, cli_kernle):
        """Test seeding trust assessments."""
        k, storage = cli_kernle

        args = argparse.Namespace(trust_action="seed")

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "Seeded" in output or "already exist" in output

    def test_trust_seed_idempotent(self, cli_kernle):
        """Second seed reports already exist."""
        k, storage = cli_kernle

        args = argparse.Namespace(trust_action="seed")
        cmd_trust(args, k)  # first

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)  # second

        output = out.getvalue()
        assert "already exist" in output

    def test_trust_list_default(self, cli_kernle):
        """List with only bootstrap identity assessment shows it."""
        k, storage = cli_kernle

        args = argparse.Namespace(trust_action="list")

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "identity" in output

    def test_trust_list_shows_domain_scores(self, cli_kernle):
        """List shows domain-specific scores when present."""
        k, storage = cli_kernle

        k.trust_set("operator", domain="general", score=0.8)
        k.trust_set("operator", domain="memory", score=0.6)

        args = argparse.Namespace(trust_action="list")

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "operator" in output
        assert "80%" in output or "General" in output

    def test_trust_show(self, cli_kernle):
        """Show displays detailed trust info for an entity."""
        k, storage = cli_kernle

        k.trust_set("collaborator", domain="general", score=0.75)

        args = argparse.Namespace(trust_action="show", entity="collaborator")

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "collaborator" in output
        assert "Dimensions" in output
        assert "75%" in output or "general" in output

    def test_trust_show_nonexistent(self, cli_kernle):
        """Show for unknown entity reports not found."""
        k, storage = cli_kernle

        args = argparse.Namespace(trust_action="show", entity="nobody")

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "No trust assessment" in output

    def test_trust_set_invalid_score(self, cli_kernle):
        """Score outside 0.0-1.0 is rejected."""
        k, storage = cli_kernle

        args = argparse.Namespace(trust_action="set", entity="test", score=1.5, domain="general")

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "must be between" in output

    def test_trust_gate(self, cli_kernle):
        """Gate checks trust-based access control."""
        k, storage = cli_kernle

        k.trust_set("operator", domain="general", score=0.9)

        args = argparse.Namespace(
            trust_action="gate", source="operator", gate_action="write", domain=None
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "ALLOWED" in output or "DENIED" in output

    def test_trust_compute(self, cli_kernle):
        """Compute shows default trust for entity with no history."""
        k, storage = cli_kernle

        args = argparse.Namespace(trust_action="compute", entity="stranger", domain="general")
        setattr(args, "apply", False)

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "Default trust" in output or "Computed trust" in output

    def test_trust_compute_with_apply(self, cli_kernle):
        """Compute --apply stores the computed score."""
        k, storage = cli_kernle

        # Create some episode history
        k.episode("Collaborated with partner on task", "success")

        args = argparse.Namespace(trust_action="compute", entity="partner", domain="general")
        setattr(args, "apply", True)

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "Default trust" in output or "Applied" in output

    def test_trust_chain(self, cli_kernle):
        """Chain computes transitive trust through intermediaries."""
        k, storage = cli_kernle

        k.trust_set("alice", domain="general", score=0.9)
        k.trust_set("bob", domain="general", score=0.8)

        args = argparse.Namespace(
            trust_action="chain", target="bob", chain=["alice"], domain="general"
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "Transitive trust" in output
        assert "bob" in output

    def test_trust_decay(self, cli_kernle):
        """Decay reduces trust over time without interaction."""
        k, storage = cli_kernle

        k.trust_set("old-friend", domain="general", score=0.9)

        args = argparse.Namespace(trust_action="decay", entity="old-friend", days=30)

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "Applied trust decay for: old-friend" in output
        assert "Decay factor: 0.3000" in output
        assert "general: 78%" in output.lower()

        updated = k.trust_show("old-friend")
        assert updated is not None
        assert updated["dimensions"]["general"]["score"] == pytest.approx(0.78, abs=1e-4)

    def test_trust_unknown_action(self, cli_kernle):
        """Unknown action shows usage help."""
        k, storage = cli_kernle

        args = argparse.Namespace(trust_action=None)

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_trust(args, k)

        output = out.getvalue()
        assert "Usage" in output


class TestDoctorCommands:
    """Integration tests for doctor diagnostic commands."""

    def test_doctor_basic(self, cli_kernle):
        """Test basic doctor check with real storage."""
        k, storage = cli_kernle

        args = argparse.Namespace(
            json=False,
            verbose=False,
            fix=False,
            full=False,
            doctor_action=None,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_doctor(args, k)

        output = out.getvalue()
        assert "Kernle Doctor" in output
        assert "Stack:" in output

    def test_doctor_structural(self, cli_kernle):
        """Test structural health check with real storage."""
        k, storage = cli_kernle

        # Add some data so the check has something to analyze
        k.episode("Doctor test episode", "success")
        k.belief("Doctor test belief", confidence=0.8)

        args = argparse.Namespace(
            json=False,
            save_note=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_doctor_structural(args, k)

        output = out.getvalue()
        assert "Structural Health Check" in output

    def test_doctor_structural_json(self, cli_kernle):
        """Test structural health check JSON output."""
        k, storage = cli_kernle

        args = argparse.Namespace(
            json=True,
            save_note=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_doctor_structural(args, k)

        data = json.loads(out.getvalue())
        assert "summary" in data
        assert "findings" in data


class TestNarrativeCommands:
    """Integration tests for self-narrative commands."""

    def test_narrative_update_and_show(self, cli_kernle):
        """Test saving and retrieving a narrative with real storage."""
        k, storage = cli_kernle

        # Update (save) a narrative
        update_args = argparse.Namespace(
            narrative_action="update",
            type="identity",
            content="I am an integration testing agent. I value thoroughness.",
            theme=["testing", "quality"],
            tension=["speed vs thoroughness"],
            epoch_id=None,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_narrative(update_args, k)

        output = out.getvalue()
        assert "Narrative updated" in output

        # Show the narrative
        show_args = argparse.Namespace(
            narrative_action="show",
            type="identity",
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_narrative(show_args, k)

        output = out.getvalue()
        assert "integration testing agent" in output
        assert "testing" in output

    def test_narrative_history(self, cli_kernle):
        """Test narrative history with real data."""
        k, storage = cli_kernle

        k.narrative_save(
            content="First narrative version",
            narrative_type="identity",
        )
        k.narrative_save(
            content="Second narrative version",
            narrative_type="identity",
        )

        args = argparse.Namespace(
            narrative_action="history",
            type=None,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_narrative(args, k)

        output = out.getvalue()
        assert "Self-Narrative History" in output


class TestSummaryCommands:
    """Integration tests for summary commands."""

    def test_summary_write_and_list(self, cli_kernle):
        """Test writing and listing summaries with real storage."""
        k, storage = cli_kernle

        # Write a summary
        write_args = argparse.Namespace(
            summary_action="write",
            scope="month",
            content="Completed integration testing setup and ran all tests.",
            period_start="2025-01-01",
            period_end="2025-01-31",
            theme=["testing", "setup"],
            epoch_id=None,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_summary(write_args, k)

        output = out.getvalue()
        assert "Summary created" in output
        assert "month" in output

        # List summaries
        list_args = argparse.Namespace(
            summary_action="list",
            scope=None,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_summary(list_args, k)

        output = out.getvalue()
        assert "Summaries" in output
        assert "month" in output

    def test_summary_show(self, cli_kernle):
        """Test showing a specific summary."""
        k, storage = cli_kernle

        summary_id = k.summary_save(
            content="Detailed summary of testing work.",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
        )

        args = argparse.Namespace(
            summary_action="show",
            id=summary_id,
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_summary(args, k)

        output = out.getvalue()
        assert "Detailed summary of testing work" in output


class TestStatusCommand:
    """Integration tests for the status command."""

    def test_status_with_data(self, cli_kernle):
        """Test status shows real counts from storage."""
        k, storage = cli_kernle

        k.episode("Status test episode", "success")
        k.note("Status test note")
        k.belief("Status test belief", confidence=0.8)

        args = argparse.Namespace()

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_status(args, k)

        output = out.getvalue()
        assert "Memory Status" in output or "Status" in output
        assert "cli_test_agent" in output


class TestCheckpointIntegration:
    """Integration tests for checkpoint save/load/clear."""

    def test_checkpoint_roundtrip(self, cli_kernle):
        """Test saving, loading, and clearing a checkpoint."""
        k, storage = cli_kernle

        # Save
        save_args = argparse.Namespace(
            checkpoint_action="save",
            task="Integration test checkpoint",
            pending=["verify roundtrip", "clean up"],
            context="Testing checkpoint lifecycle",
            progress=None,
            next=None,
            blocker=None,
            sync=False,
            no_sync=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_checkpoint(save_args, k)

        assert "Checkpoint saved" in out.getvalue()

        # Load
        load_args = argparse.Namespace(
            checkpoint_action="load",
            json=False,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_checkpoint(load_args, k)

        output = out.getvalue()
        assert "Integration test checkpoint" in output

        # Clear
        clear_args = argparse.Namespace(checkpoint_action="clear")

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_checkpoint(clear_args, k)

        assert "Checkpoint cleared" in out.getvalue()

        # Verify cleared
        with patch("sys.stdout", new=StringIO()) as out:
            cmd_checkpoint(load_args, k)

        assert "No checkpoint found" in out.getvalue()
