"""Tests for CLI provenance (derived_from) wiring.

Verifies that derived_from is properly passed through CLI commands
to the underlying Kernle methods.
"""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

from kernle.cli.__main__ import cmd_relation
from kernle.cli.commands.import_cmd import _import_item, _item_signature


class TestRelationDerivedFrom:
    """Test that relation add/update passes derived_from."""

    def test_relation_add_with_derived_from(self, capsys):
        """relation add passes derived_from to k.relationship()."""
        k = MagicMock()
        k.relationship.return_value = "rel-123"

        args = Namespace(
            relation_action="add",
            name="Alice",
            type="person",
            trust=0.8,
            notes="Met at conference",
            derived_from=["episode:ep-abc", "note:note-def"],
            json=False,
        )

        cmd_relation(args, k)

        k.relationship.assert_called_once_with(
            "Alice",
            trust_level=0.8,
            notes="Met at conference",
            entity_type="person",
            derived_from=["episode:ep-abc", "note:note-def"],
        )
        captured = capsys.readouterr()
        assert "Relationship added: Alice" in captured.out
        assert "Derived from: 2 memories" in captured.out

    def test_relation_add_without_derived_from(self, capsys):
        """relation add works without derived_from."""
        k = MagicMock()
        k.relationship.return_value = "rel-123"

        args = Namespace(
            relation_action="add",
            name="Bob",
            type="person",
            trust=None,
            notes=None,
            json=False,
        )

        cmd_relation(args, k)

        k.relationship.assert_called_once_with(
            "Bob",
            trust_level=0.5,
            notes=None,
            entity_type="person",
            derived_from=None,
        )
        captured = capsys.readouterr()
        assert "Derived from" not in captured.out

    def test_relation_update_with_derived_from(self, capsys):
        """relation update passes derived_from to k.relationship()."""
        k = MagicMock()
        k.relationship.return_value = "rel-123"

        args = Namespace(
            relation_action="update",
            name="Alice",
            trust=0.9,
            notes=None,
            type=None,
            derived_from=["episode:ep-xyz"],
            json=False,
        )

        cmd_relation(args, k)

        k.relationship.assert_called_once_with(
            "Alice",
            trust_level=0.9,
            notes=None,
            entity_type=None,
            derived_from=["episode:ep-xyz"],
        )
        captured = capsys.readouterr()
        assert "Relationship updated: Alice" in captured.out
        assert "Derived from: 1 memories" in captured.out

    def test_relation_update_derived_from_only(self, capsys):
        """relation update with only derived_from should not require other fields."""
        k = MagicMock()
        k.relationship.return_value = "rel-123"

        args = Namespace(
            relation_action="update",
            name="Alice",
            trust=None,
            notes=None,
            type=None,
            derived_from=["context:imported_from_backup"],
            json=False,
        )

        cmd_relation(args, k)

        # Should not print the "Provide --trust..." error
        k.relationship.assert_called_once()
        captured = capsys.readouterr()
        assert "Relationship updated" in captured.out

    def test_relation_update_nothing_provided(self, capsys):
        """relation update with no changes should show error."""
        k = MagicMock()

        args = Namespace(
            relation_action="update",
            name="Alice",
            trust=None,
            notes=None,
            type=None,
            derived_from=None,
            json=False,
        )

        cmd_relation(args, k)

        k.relationship.assert_not_called()
        captured = capsys.readouterr()
        assert "Provide --trust" in captured.out


class TestBeliefSupersedeDerivedFrom:
    """Test that belief supersede already handles derived_from internally."""

    def test_supersede_sets_derived_from_in_core(self):
        """The core.py supersede_belief sets derived_from=[f'belief:{old_id}']."""
        # This tests that the core method itself sets derived_from,
        # so the CLI doesn't need to pass it explicitly.
        from kernle.core import Kernle

        k = MagicMock(spec=Kernle)
        # The important assertion is that core.py line ~4236 sets:
        # derived_from=[f"belief:{old_id}"]
        # This is tested by the core tests, not CLI tests.
        # Here we just verify the CLI calls revise_belief correctly.
        k.revise_belief.return_value = "new-belief-id"

        from kernle.cli.commands.belief import cmd_belief

        args = Namespace(
            belief_action="supersede",
            old_id="old-belief-123",
            new_statement="Updated belief statement",
            confidence=0.9,
            reason="New evidence found",
            json=False,
        )

        cmd_belief(args, k)

        k.revise_belief.assert_called_once_with(
            old_id="old-belief-123",
            new_statement="Updated belief statement",
            confidence=0.9,
            reason="New evidence found",
        )


class TestImportItemDerivedFrom:
    """Test that _import_item passes derived_from to Kernle methods."""

    def test_import_episode_with_derived_from(self):
        """Import episode passes derived_from with import fingerprint appended."""
        k = MagicMock()
        k.episode.return_value = "ep-123"

        item = {
            "type": "episode",
            "objective": "Test objective",
            "outcome": "Test outcome",
        }

        _import_item(item, k, derived_from=["context:imported_from_backup"])

        k.episode.assert_called_once()
        call_kwargs = k.episode.call_args[1]
        # Original derived_from preserved, fingerprint appended
        assert "context:imported_from_backup" in call_kwargs["derived_from"]
        fingerprint = _item_signature(item)
        if fingerprint:
            assert fingerprint in call_kwargs["derived_from"]

    def test_import_note_with_derived_from(self):
        """Import note passes derived_from with import fingerprint appended."""
        k = MagicMock()
        k.note.return_value = "note-123"

        item = {
            "type": "note",
            "content": "Test note",
        }

        _import_item(item, k, derived_from=["context:csv_import"])

        k.note.assert_called_once()
        call_kwargs = k.note.call_args[1]
        assert "context:csv_import" in call_kwargs["derived_from"]
        fingerprint = _item_signature(item)
        if fingerprint:
            assert fingerprint in call_kwargs["derived_from"]

    def test_import_belief_with_derived_from(self):
        """Import belief passes derived_from with import fingerprint appended."""
        k = MagicMock()
        k.belief.return_value = "belief-123"

        item = {
            "type": "belief",
            "statement": "Test belief",
            "confidence": 0.8,
        }

        _import_item(item, k, derived_from=["context:json_import"])

        k.belief.assert_called_once()
        call_kwargs = k.belief.call_args[1]
        assert "context:json_import" in call_kwargs["derived_from"]
        fingerprint = _item_signature(item)
        if fingerprint:
            assert fingerprint in call_kwargs["derived_from"]

    def test_import_value_with_derived_from(self):
        """Import value passes derived_from with import fingerprint appended."""
        k = MagicMock()
        k.value.return_value = "value-123"

        item = {
            "type": "value",
            "name": "Quality",
            "description": "High quality work",
        }

        _import_item(item, k, derived_from=["context:migration"])

        k.value.assert_called_once()
        call_kwargs = k.value.call_args[1]
        assert "context:migration" in call_kwargs["derived_from"]
        fingerprint = _item_signature(item)
        if fingerprint:
            assert fingerprint in call_kwargs["derived_from"]

    def test_import_goal_with_derived_from(self):
        """Import goal passes derived_from with import fingerprint appended."""
        k = MagicMock()
        k.goal.return_value = "goal-123"

        item = {
            "type": "goal",
            "description": "Test goal",
        }

        _import_item(item, k, derived_from=["context:backup_restore"])

        k.goal.assert_called_once()
        call_kwargs = k.goal.call_args[1]
        assert "context:backup_restore" in call_kwargs["derived_from"]
        fingerprint = _item_signature(item)
        if fingerprint:
            assert fingerprint in call_kwargs["derived_from"]

    def test_import_without_derived_from(self):
        """Import without derived_from gets just the import fingerprint."""
        k = MagicMock()
        k.episode.return_value = "ep-123"

        item = {
            "type": "episode",
            "objective": "Test",
            "outcome": "Done",
        }

        _import_item(item, k)

        k.episode.assert_called_once()
        call_kwargs = k.episode.call_args[1]
        fingerprint = _item_signature(item)
        if fingerprint:
            # Should have the import fingerprint
            assert call_kwargs["derived_from"] == [fingerprint]
        else:
            assert call_kwargs["derived_from"] is None

    def test_import_raw_ignores_derived_from(self):
        """Import raw does not pass derived_from (raw entries don't support it)."""
        k = MagicMock()
        k.raw.return_value = "raw-123"

        item = {
            "type": "raw",
            "content": "Raw thought",
            "source": "import",
        }

        _import_item(item, k, derived_from=["context:migration"])

        k.raw.assert_called_once()
        # raw() call should NOT have derived_from
        call_kwargs = k.raw.call_args[1]
        assert "derived_from" not in call_kwargs


class TestCoreDerivedFrom:
    """Test that core.py value(), goal(), and relationship() accept derived_from."""

    def _make_kernle(self):
        """Create a Kernle instance with real SQLite storage."""
        import tempfile

        from kernle.storage.sqlite import SQLiteStorage

        self._tmpdir = tempfile.mkdtemp()
        db_path = Path(self._tmpdir) / "test.db"
        storage = SQLiteStorage(stack_id="test-agent", db_path=db_path)

        from kernle import Kernle

        k = Kernle(
            stack_id="test-agent",
            storage=storage,
            checkpoint_dir=Path(self._tmpdir) / "cp",
            strict=False,
        )
        # Bind noop model to satisfy inference gate
        from tests.conftest import bind_noop_model

        bind_noop_model(k)
        return k, storage

    def test_value_with_derived_from(self):
        """Kernle.value() passes derived_from to Value dataclass."""
        k, storage = self._make_kernle()

        value_id = k.value(
            name="Integrity",
            statement="Act with integrity",
            derived_from=["episode:ep-abc"],
        )

        assert value_id is not None
        values = storage.get_values(limit=10)
        saved = [v for v in values if v.id == value_id]
        assert len(saved) == 1
        assert saved[0].derived_from == ["episode:ep-abc"]

    def test_value_without_derived_from(self):
        """Kernle.value() works without derived_from."""
        k, storage = self._make_kernle()

        value_id = k.value(name="Quality", statement="High quality")

        assert value_id is not None
        values = storage.get_values(limit=10)
        saved = [v for v in values if v.id == value_id]
        assert len(saved) == 1
        assert saved[0].derived_from is None

    def test_goal_with_derived_from(self):
        """Kernle.goal() passes derived_from to Goal dataclass."""
        k, storage = self._make_kernle()

        goal_id = k.goal(
            title="Complete migration",
            description="Finish the data migration",
            derived_from=["context:planning_session"],
        )

        assert goal_id is not None
        goals = storage.get_goals(limit=10)
        saved = [g for g in goals if g.id == goal_id]
        assert len(saved) == 1
        assert saved[0].derived_from == ["context:planning_session"]

    def test_goal_without_derived_from(self):
        """Kernle.goal() works without derived_from."""
        k, storage = self._make_kernle()

        goal_id = k.goal(title="Do something")

        assert goal_id is not None
        goals = storage.get_goals(limit=10)
        saved = [g for g in goals if g.id == goal_id]
        assert len(saved) == 1
        assert saved[0].derived_from is None

    def test_relationship_new_with_derived_from(self):
        """Kernle.relationship() passes derived_from when creating new."""
        k, storage = self._make_kernle()

        rel_id = k.relationship(
            "Alice",
            trust_level=0.8,
            entity_type="person",
            derived_from=["episode:ep-meeting"],
        )

        assert rel_id is not None
        rels = storage.get_relationships()
        saved = [r for r in rels if r.id == rel_id]
        assert len(saved) == 1
        assert saved[0].derived_from == ["episode:ep-meeting"]
        assert saved[0].entity_name == "Alice"

    def test_relationship_update_with_derived_from(self):
        """Kernle.relationship() sets derived_from on update."""
        k, storage = self._make_kernle()

        # Create initial relationship
        k.relationship(
            "Bob",
            trust_level=0.5,
            entity_type="person",
        )

        # Update with derived_from
        k.relationship(
            "Bob",
            trust_level=0.7,
            derived_from=["episode:ep-collaboration"],
        )

        rels = storage.get_relationships()
        bob_rels = [r for r in rels if r.entity_name == "Bob"]
        assert len(bob_rels) == 1
        assert bob_rels[0].derived_from == ["episode:ep-collaboration"]
