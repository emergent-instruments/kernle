"""Tests for core/ package submodules — covers edge cases in validation,
checkpoint, and writers to maintain 80%+ coverage after fragmentation."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kernle.core import Kernle
from kernle.types import SourceType

# =========================================================================
# ValidationMixin edge cases
# =========================================================================


class TestValidationEdgeCases:
    """Cover uncovered validation paths."""

    def test_stack_id_with_path_separator_slash(self):
        with pytest.raises(ValueError, match="path separators"):
            Kernle(stack_id="foo/bar")

    def test_stack_id_with_backslash(self):
        with pytest.raises(ValueError, match="path separators"):
            Kernle(stack_id="foo\\bar")

    def test_stack_id_dot_dot(self):
        with pytest.raises(ValueError, match="relative path component"):
            Kernle(stack_id="..")

    def test_stack_id_single_dot(self):
        with pytest.raises(ValueError, match="relative path component"):
            Kernle(stack_id=".")

    def test_stack_id_only_special_chars(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            Kernle(stack_id="@#$%^&*()")

    def test_stack_id_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            Kernle(stack_id="a" * 101)

    def test_checkpoint_dir_outside_safe_paths(self):
        with pytest.raises(ValueError, match="within user home or temp"):
            Kernle(stack_id="test", checkpoint_dir=Path("/etc/kernle"))

    def test_validate_string_input_not_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            k = Kernle(stack_id="test", checkpoint_dir=Path(tmp) / "cp", strict=False)
            with pytest.raises(TypeError, match="must be a string"):
                k._validate_string_input(123, "field")

    def test_validate_string_input_too_long(self):
        with tempfile.TemporaryDirectory() as tmp:
            k = Kernle(stack_id="test", checkpoint_dir=Path(tmp) / "cp", strict=False)
            with pytest.raises(ValueError, match="too long"):
                k._validate_string_input("a" * 1001, "field", max_length=1000)

    def test_validate_string_input_null_byte_sanitization(self):
        with tempfile.TemporaryDirectory() as tmp:
            k = Kernle(stack_id="test", checkpoint_dir=Path(tmp) / "cp", strict=False)
            result = k._validate_string_input("hello\x00world", "field")
            assert result == "helloworld"

    def test_validate_derived_from_empty(self):
        assert Kernle._validate_derived_from([]) is None
        assert Kernle._validate_derived_from(None) is None

    def test_validate_derived_from_malformed_refs(self):
        # No colon = skipped
        assert Kernle._validate_derived_from(["nocolon"]) is None

    def test_validate_derived_from_unknown_type(self):
        # Unknown type prefix = skipped
        assert Kernle._validate_derived_from(["unknown:123"]) is None

    def test_validate_derived_from_valid_refs(self):
        result = Kernle._validate_derived_from(["episode:abc", "context:test", "belief:xyz"])
        assert result == ["episode:abc", "context:test", "belief:xyz"]

    def test_validate_derived_from_filters_empty_strings(self):
        result = Kernle._validate_derived_from(["", "episode:abc", None, "belief:xyz"])
        assert result == ["episode:abc", "belief:xyz"]


# =========================================================================
# CheckpointMixin edge cases
# =========================================================================


class TestCheckpointEdgeCases:
    """Cover uncovered checkpoint error handling paths."""

    def test_checkpoint_dir_creation_failure(self):
        """Checkpoint raises when directory creation fails."""
        with tempfile.TemporaryDirectory() as tmp:
            k = Kernle(stack_id="test", checkpoint_dir=Path(tmp) / "cp", strict=False)
            # Make the checkpoint dir unwritable
            with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
                with pytest.raises(ValueError, match="Cannot create checkpoint directory"):
                    k.checkpoint("test task")

    def test_checkpoint_corrupted_json(self):
        """Checkpoint handles corrupted JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            cp_dir = Path(tmp) / "cp"
            cp_dir.mkdir(parents=True)
            # Write corrupted JSON
            cp_file = cp_dir / "test.json"
            cp_file.write_text("not valid json{{{")

            k = Kernle(stack_id="test", checkpoint_dir=cp_dir, strict=False)
            # Should still save successfully (old data lost, new data created)
            result = k.checkpoint("test task")
            assert result["current_task"] == "test task"

    def test_checkpoint_save_write_failure(self):
        """Checkpoint raises when file write fails."""
        with tempfile.TemporaryDirectory() as tmp:
            cp_dir = Path(tmp) / "cp"
            cp_dir.mkdir(parents=True)
            k = Kernle(stack_id="test", checkpoint_dir=cp_dir, strict=False)

            # Mock open to fail on write
            original_open = open

            def mock_open_fail(path, mode="r", **kwargs):
                if "w" in mode and "test.json" in str(path):
                    raise OSError("Disk full")
                return original_open(path, mode, **kwargs)

            with patch("builtins.open", side_effect=mock_open_fail):
                with pytest.raises(ValueError, match="Cannot save checkpoint"):
                    k.checkpoint("test task")

    def test_load_checkpoint_dict_format(self):
        """load_checkpoint handles legacy dict (non-list) format."""
        with tempfile.TemporaryDirectory() as tmp:
            cp_dir = Path(tmp) / "cp"
            cp_dir.mkdir(parents=True)
            cp_file = cp_dir / "test.json"
            cp_file.write_text(json.dumps({"current_task": "legacy"}))

            k = Kernle(stack_id="test", checkpoint_dir=cp_dir, strict=False)
            result = k.load_checkpoint()
            assert result["current_task"] == "legacy"

    def test_load_checkpoint_too_large(self):
        """load_checkpoint rejects files exceeding size limit."""
        with tempfile.TemporaryDirectory() as tmp:
            cp_dir = Path(tmp) / "cp"
            cp_dir.mkdir(parents=True)
            cp_file = cp_dir / "test.json"
            cp_file.write_text("[]")  # Valid but small

            k = Kernle(stack_id="test", checkpoint_dir=cp_dir, strict=False)
            # Patch stat to report huge size
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=20 * 1024 * 1024)
                with pytest.raises(ValueError, match="too large"):
                    k.load_checkpoint()

    def test_checkpoint_episode_save_failure(self):
        """Checkpoint continues when episode save fails."""
        with tempfile.TemporaryDirectory() as tmp:
            cp_dir = Path(tmp) / "cp"
            k = Kernle(stack_id="test", checkpoint_dir=cp_dir, strict=False)
            # Mock _write_backend.save_episode to fail
            k._storage.save_episode = MagicMock(side_effect=sqlite3.OperationalError("DB error"))
            result = k.checkpoint("test task")
            # Should still succeed (episode save is best-effort)
            assert result["current_task"] == "test task"

    def test_checkpoint_boot_export_failure(self):
        """Checkpoint continues when boot file export fails."""
        with tempfile.TemporaryDirectory() as tmp:
            cp_dir = Path(tmp) / "cp"
            k = Kernle(stack_id="test", checkpoint_dir=cp_dir, strict=False)
            # Save at least once first to ensure the checkpoint file exists
            k.checkpoint("setup")

            # Now mock _export_boot_file to fail
            with patch.object(k, "_export_boot_file", side_effect=OSError("IO error")):
                result = k.checkpoint("test task 2")
                assert result["current_task"] == "test task 2"


# =========================================================================
# WritersMixin edge cases
# =========================================================================


class TestWritersEdgeCases:
    """Cover uncovered writer paths."""

    def _make_kernle(self):
        """Create a non-strict Kernle for testing."""
        tmp = tempfile.mkdtemp()
        return Kernle(stack_id="test-writers", checkpoint_dir=Path(tmp) / "cp", strict=False)

    def test_episode_with_repeat_and_avoid(self):
        """Episode correctly validates repeat and avoid params."""
        k = self._make_kernle()
        ep_id = k.episode(
            objective="Test",
            outcome="Done successfully",
            repeat=["good pattern"],
            avoid=["bad pattern"],
        )
        assert ep_id is not None
        # Verify stored data
        ep = k._storage.get_episode(ep_id)
        assert ep.objective == "Test"
        assert ep.outcome == "Done successfully"
        assert ep.outcome_type == "success"
        assert ep.repeat == ["good pattern"]
        assert ep.avoid == ["bad pattern"]

    def test_episode_failure_outcome_type(self):
        """Episode detects failure outcome type."""
        k = self._make_kernle()
        ep_id = k.episode(objective="Test", outcome="This failed badly")
        # Verify via storage
        ep = k._storage.get_episode(ep_id)
        assert ep.outcome_type == "failure"

    def test_episode_source_type_external(self):
        """Episode uses explicit external source_type."""
        k = self._make_kernle()
        ep_id = k.episode(
            objective="Test",
            outcome="Completed",
            source="told by Alice",
            source_type=SourceType.EXTERNAL,
        )
        ep = k._storage.get_episode(ep_id)
        assert ep.source_type == "external"

    def test_episode_source_type_inference(self):
        """Episode uses explicit inference source_type."""
        k = self._make_kernle()
        ep_id = k.episode(
            objective="Test",
            outcome="Completed",
            source="inferred from data",
            source_type=SourceType.INFERENCE,
        )
        ep = k._storage.get_episode(ep_id)
        assert ep.source_type == "inference"

    def test_episode_explicit_source_type(self):
        """Episode uses explicit source_type override."""
        k = self._make_kernle()
        ep_id = k.episode(
            objective="Test",
            outcome="Completed",
            source_type="seed",
        )
        ep = k._storage.get_episode(ep_id)
        assert ep.source_type == "seed"

    def test_update_episode_failure_outcome(self):
        """update_episode recalculates failure outcome_type."""
        k = self._make_kernle()
        ep_id = k.episode(objective="Test", outcome="Completed")
        result = k.update_episode(ep_id, outcome="This failed completely")
        assert result is True
        ep = k._storage.get_episode(ep_id)
        assert ep.outcome_type == "failure"

    def test_update_episode_partial_outcome(self):
        """update_episode recalculates partial outcome_type."""
        k = self._make_kernle()
        ep_id = k.episode(objective="Test", outcome="Completed successfully")
        result = k.update_episode(ep_id, outcome="Still working on it")
        assert result is True
        ep = k._storage.get_episode(ep_id)
        assert ep.outcome_type == "partial"

    def test_update_episode_not_found(self):
        """update_episode returns False for nonexistent ID."""
        k = self._make_kernle()
        result = k.update_episode("nonexistent-id")
        assert result is False

    def test_note_quote_type(self):
        """Note formats quote type correctly."""
        k = self._make_kernle()
        note_id = k.note(
            content="The truth is out there",
            type="quote",
            speaker="Mulder",
        )
        assert note_id is not None
        # Verify quote formatting in stored content
        notes = k._storage.get_notes(limit=100)
        matched = [n for n in notes if n.id == note_id]
        assert len(matched) == 1
        assert '> "The truth is out there"' in matched[0].content
        assert "Mulder" in matched[0].content
        assert matched[0].note_type == "quote"

    def test_note_decision_type(self):
        """Note formats decision type correctly."""
        k = self._make_kernle()
        note_id = k.note(
            content="Use Python for backend",
            type="decision",
            reason="Team expertise",
        )
        assert note_id is not None
        # Verify decision formatting in stored content
        notes = k._storage.get_notes(limit=100)
        matched = [n for n in notes if n.id == note_id]
        assert len(matched) == 1
        assert "**Decision**: Use Python for backend" in matched[0].content
        assert "**Reason**: Team expertise" in matched[0].content
        assert matched[0].note_type == "decision"

    def test_note_insight_type(self):
        """Note formats insight type correctly."""
        k = self._make_kernle()
        note_id = k.note(content="Users prefer simplicity", type="insight")
        assert note_id is not None
        # Verify insight formatting in stored content
        notes = k._storage.get_notes(limit=100)
        matched = [n for n in notes if n.id == note_id]
        assert len(matched) == 1
        assert "**Insight**: Users prefer simplicity" in matched[0].content
        assert matched[0].note_type == "insight"

    def test_note_invalid_type(self):
        """Note rejects invalid type."""
        k = self._make_kernle()
        with pytest.raises(ValueError, match="Invalid note type"):
            k.note(content="test", type="invalid")

    def test_note_source_type_quote_with_explicit_external(self):
        """Quote note with explicit source_type=external is stored correctly."""
        k = self._make_kernle()
        note_id = k.note(
            content="Quote text",
            type="quote",
            source="from a book",
            source_type="external",
        )
        # Verify source_type is external when explicitly set
        notes = k._storage.get_notes(limit=100)
        matched = [n for n in notes if n.id == note_id]
        assert len(matched) == 1
        assert matched[0].source_type == "external"

    def test_belief_source_consolidation(self):
        """Belief uses explicit consolidation source_type."""
        k = self._make_kernle()
        belief_id = k.belief(
            statement="Test belief",
            source="consolidation pass",
            source_type=SourceType.CONSOLIDATION,
        )
        # Via storage
        beliefs = k._storage.get_beliefs(limit=100)
        matched = [b for b in beliefs if b.id == belief_id]
        assert matched[0].source_type == "consolidation"

    def test_belief_source_seed(self):
        """Belief uses explicit seed source_type."""
        k = self._make_kernle()
        belief_id = k.belief(
            statement="Test belief",
            source="seed initialization",
            source_type=SourceType.SEED,
        )
        beliefs = k._storage.get_beliefs(limit=100)
        matched = [b for b in beliefs if b.id == belief_id]
        assert matched[0].source_type == "seed"

    def test_goal_invalid_type(self):
        """Goal rejects invalid goal_type."""
        k = self._make_kernle()
        with pytest.raises(ValueError, match="Invalid goal_type"):
            k.goal(title="test", goal_type="invalid")

    def test_goal_aspiration_protected(self):
        """Aspiration goals are auto-protected."""
        k = self._make_kernle()
        goal_id = k.goal(title="Become wise", goal_type="aspiration")
        assert goal_id is not None
        # Verify goal is stored as protected
        goals = k._storage.get_goals(status="active", limit=100)
        matched = [g for g in goals if g.id == goal_id]
        assert len(matched) == 1
        assert matched[0].is_protected is True
        assert matched[0].goal_type == "aspiration"
        assert matched[0].title == "Become wise"

    def test_goal_commitment_protected(self):
        """Commitment goals are auto-protected."""
        k = self._make_kernle()
        goal_id = k.goal(title="Always help", goal_type="commitment")
        assert goal_id is not None
        # Verify goal is stored as protected
        goals = k._storage.get_goals(status="active", limit=100)
        matched = [g for g in goals if g.id == goal_id]
        assert len(matched) == 1
        assert matched[0].is_protected is True
        assert matched[0].goal_type == "commitment"
        assert matched[0].title == "Always help"

    def test_update_goal_invalid_status(self):
        """update_goal rejects invalid status."""
        k = self._make_kernle()
        goal_id = k.goal(title="Test goal")
        with pytest.raises(ValueError, match="Invalid status"):
            k.update_goal(goal_id, status="invalid")

    def test_update_goal_invalid_priority(self):
        """update_goal rejects invalid priority."""
        k = self._make_kernle()
        goal_id = k.goal(title="Test goal")
        with pytest.raises(ValueError, match="Invalid priority"):
            k.update_goal(goal_id, priority="critical")

    def test_update_goal_not_found(self):
        """update_goal returns False for nonexistent ID."""
        k = self._make_kernle()
        result = k.update_goal("nonexistent")
        assert result is False

    def test_drive_invalid_type(self):
        """drive() rejects invalid drive type."""
        k = self._make_kernle()
        with pytest.raises(ValueError, match="Invalid drive type"):
            k.drive(drive_type="invalid")

    def test_drive_update_existing(self):
        """drive() updates existing drive."""
        k = self._make_kernle()
        drive_id1 = k.drive(drive_type="curiosity", intensity=0.5)
        drive_id2 = k.drive(drive_type="curiosity", intensity=0.8)
        assert drive_id1 == drive_id2  # Same drive updated

    def test_drive_source_type_consolidation(self):
        """Drive uses explicit consolidation source_type."""
        k = self._make_kernle()
        k.drive(
            drive_type="growth",
            intensity=0.5,
            source="consolidation",
            source_type=SourceType.CONSOLIDATION,
        )
        drive = k._storage.get_drive("growth")
        assert drive.source_type == "consolidation"

    def test_satisfy_drive_not_found(self):
        """satisfy_drive returns False for missing drive."""
        k = self._make_kernle()
        result = k.satisfy_drive("nonexistent")
        assert result is False

    def test_raw_strict_mode(self):
        """raw() in strict mode creates RawEntry object."""
        with tempfile.TemporaryDirectory() as tmp:
            k = Kernle(stack_id="test-raw", checkpoint_dir=Path(tmp) / "cp", strict=True)
            raw_id = k.raw("test blob", source="test")
            assert raw_id is not None

    def test_process_raw_not_found(self):
        """process_raw raises for missing entry."""
        k = self._make_kernle()
        with pytest.raises(ValueError, match="not found"):
            k.process_raw("nonexistent", "episode")

    def test_process_raw_invalid_type(self):
        """process_raw raises for invalid as_type."""
        k = self._make_kernle()
        raw_id = k.raw("test content")
        with pytest.raises(ValueError, match="Invalid as_type"):
            k.process_raw(raw_id, "invalid_type")

    def test_relationship_new(self):
        """relationship() creates new relationship."""
        k = self._make_kernle()
        rel_id = k.relationship(
            other_stack_id="Alice",
            trust_level=0.8,
            notes="A colleague",
            entity_type="person",
        )
        assert rel_id is not None
        # Verify stored relationship data
        rel = k._storage.get_relationship("Alice")
        assert rel is not None
        assert rel.id == rel_id
        assert rel.entity_name == "Alice"
        assert rel.entity_type == "person"
        assert rel.notes == "A colleague"
        # trust_level 0.8 -> sentiment (0.8 * 2) - 1 = 0.6
        assert abs(rel.sentiment - 0.6) < 0.01

    def test_relationship_update_existing(self):
        """relationship() updates existing relationship."""
        k = self._make_kernle()
        rel_id1 = k.relationship(other_stack_id="Bob", trust_level=0.5)
        rel_id2 = k.relationship(other_stack_id="Bob", trust_level=0.9, notes="Updated")
        assert rel_id1 == rel_id2
