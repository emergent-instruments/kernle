"""Tests for provenance migration (#352).

Tests cover:
- backfill-provenance adding kernle:pre-v0.9-migration to episodes/notes without derived_from
- link-raw matching episodes to raw entries by timestamp and content
- get_pre_v09_memories finding annotated memories
- get_ungrounded_memories correctly skipping pre-v0.9 annotated memories
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from kernle.stack.sqlite_stack import Stack
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Drive, Episode, Goal, Note, Relationship, Value

STACK_ID = "test-migration"


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test.db"
    s = SQLiteStorage(STACK_ID, db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def stack(tmp_path):
    db_path = tmp_path / "test.db"
    return Stack.from_sqlite(STACK_ID, db_path=db_path, components=[], enforce_provenance=False)


# ==============================================================================
# Helpers
# ==============================================================================


def _save_raw(storage, blob="test raw entry", captured_at=None):
    """Save a raw entry and return its ID."""
    raw_id = storage.save_raw(blob, source="cli")
    if captured_at:
        with storage._connect() as conn:
            conn.execute(
                "UPDATE raw_entries SET captured_at = ? WHERE id = ?",
                (captured_at.isoformat(), raw_id),
            )
            conn.commit()
    return raw_id


def _ep(objective="Test episode", outcome="It happened", created_at=None):
    return Episode(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        objective=objective,
        outcome=outcome,
        source_type="direct_experience",
        created_at=created_at or datetime.now(timezone.utc),
    )


def _note(content="Test note", created_at=None):
    return Note(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        content=content,
        note_type="observation",
        created_at=created_at or datetime.now(timezone.utc),
    )


# ==============================================================================
# get_pre_v09_memories
# ==============================================================================


class TestGetPreV09Memories:
    """Tests for SQLiteStorage.get_pre_v09_memories()."""

    def test_finds_annotated_episodes(self, storage):
        ep = _ep()
        storage.save_episode(ep)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (json.dumps(["kernle:pre-v0.9-migration"]), ep.id),
            )
            conn.commit()

        results = storage.get_pre_v09_memories(STACK_ID)
        assert len(results) == 1
        assert results[0][0] == "episode"
        assert results[0][1] == ep.id
        assert results[0][2] is False  # no auto-link

    def test_detects_auto_linked(self, storage):
        ep = _ep()
        raw_id = _save_raw(storage)
        storage.save_episode(ep)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (
                    json.dumps(
                        [f"raw:{raw_id}", "kernle:auto-linked", "kernle:pre-v0.9-migration"]
                    ),
                    ep.id,
                ),
            )
            conn.commit()

        results = storage.get_pre_v09_memories(STACK_ID)
        assert len(results) == 1
        assert results[0][2] is True  # has auto-link

    def test_ignores_non_annotated(self, storage):
        ep = _ep()
        raw_id = _save_raw(storage)
        storage.save_episode(ep)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (json.dumps([f"raw:{raw_id}"]), ep.id),
            )
            conn.commit()

        results = storage.get_pre_v09_memories(STACK_ID)
        assert len(results) == 0

    def test_empty_stack(self, storage):
        results = storage.get_pre_v09_memories(STACK_ID)
        assert results == []


# ==============================================================================
# get_ungrounded_memories skips pre-v0.9 annotations
# ==============================================================================


class TestUngroundedSkipsPreV09:
    """Verify get_ungrounded_memories doesn't flag pre-v0.9 annotated memories."""

    def test_annotation_only_not_ungrounded(self, storage):
        ep = _ep()
        storage.save_episode(ep)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (json.dumps(["kernle:pre-v0.9-migration"]), ep.id),
            )
            conn.commit()

        results = storage.get_ungrounded_memories(STACK_ID)
        ids = [r[1] for r in results]
        assert ep.id not in ids

    def test_no_derived_from_not_ungrounded(self, storage):
        ep = _ep()
        storage.save_episode(ep)

        results = storage.get_ungrounded_memories(STACK_ID)
        ids = [r[1] for r in results]
        assert ep.id not in ids


# ==============================================================================
# backfill-provenance: adds pre-v0.9 annotation
# ==============================================================================


class TestBackfillProvenance:
    """Tests for _migrate_backfill_provenance adding pre-v0.9 annotations."""

    def test_episodes_get_annotation(self, stack):
        ep = _ep()
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        ep_calls = [c for c in calls if c[0][1] == ep.id]
        assert len(ep_calls) == 1
        # set_memory_source(type, id, source_type, derived_from=[...])
        call_kwargs = ep_calls[0].kwargs
        derived_from = call_kwargs.get("derived_from")
        assert derived_from is not None
        assert "kernle:pre-v0.9-migration" in derived_from

    def test_notes_get_annotation(self, stack):
        note = _note()
        stack.save_note(note)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        note_calls = [c for c in calls if c[0][1] == note.id]
        assert len(note_calls) == 1

    def test_episodes_with_provenance_untouched(self, stack):
        raw_id = _save_raw(stack._backend)
        ep = _ep()
        ep.derived_from = [f"raw:{raw_id}"]
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        ep_calls = [c for c in calls if c[0][1] == ep.id]
        assert len(ep_calls) == 0

    def test_dry_run_no_changes(self, stack):
        ep = _ep()
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = True
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        mock_k.set_memory_source.assert_not_called()

    def test_backfill_json_is_deterministic(self, stack, capsys):
        now = datetime.now(timezone.utc)
        ep_a = _ep(objective="first", outcome="outcome", created_at=now)
        ep_a.id = "ep-b"
        ep_b = _ep(objective="second", outcome="outcome", created_at=now)
        ep_b.id = "ep-a"

        stack.save_episode(ep_a)
        stack.save_episode(ep_b)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = True
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        output = json.loads(capsys.readouterr().out)
        assert output["status_snapshot"]["total_updates"] == 2
        assert output["status_snapshot"]["ids"] == sorted(output["status_snapshot"]["ids"])
        assert output["updates"][0]["id"] < output["updates"][1]["id"]
        assert output["status_snapshot_sha256"] == _snapshot_sha256(output["status_snapshot"])


def _snapshot_sha256(payload):
    return (
        __import__("hashlib")
        .sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        )
        .hexdigest()
    )


# ==============================================================================
# link-raw: matches episodes to raw entries
# ==============================================================================


class TestLinkRaw:
    """Tests for _migrate_link_raw matching by timestamp and content."""

    def test_links_by_timestamp_and_content(self, stack):
        now = datetime.now(timezone.utc)
        raw_id = _save_raw(
            stack._backend,
            blob="deployed api service to production",
            captured_at=now - timedelta(minutes=5),
        )

        ep = _ep(
            objective="deployed api service to production",
            outcome="success",
            created_at=now,
        )
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1
        derived_from = calls[0].kwargs.get("derived_from")
        assert f"raw:{raw_id}" in derived_from
        assert "kernle:auto-linked" in derived_from

    def test_no_link_outside_window_no_content(self, stack):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="completely unrelated cooking entry",
            captured_at=now - timedelta(hours=2),
        )

        ep = _ep(objective="deployed api service", outcome="success", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        mock_k.set_memory_source.assert_not_called()

    def test_skips_already_linked(self, stack):
        now = datetime.now(timezone.utc)
        raw_id = _save_raw(stack._backend, blob="test content", captured_at=now)

        ep = _ep(objective="test content episode", created_at=now)
        ep.derived_from = [f"raw:{raw_id}"]
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        mock_k.set_memory_source.assert_not_called()

    def test_dry_run_no_changes(self, stack):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="matching content dry run",
            captured_at=now,
        )

        ep = _ep(objective="matching content dry run", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = True
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        mock_k.set_memory_source.assert_not_called()

    def test_links_notes_too(self, stack):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="insight about testing prevents bugs from production",
            captured_at=now,
        )

        note = _note(content="insight about testing prevents bugs from production", created_at=now)
        stack.save_note(note)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1

    def test_content_match_picks_best(self, stack):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="cooking recipes for dinner tonight",
            captured_at=now - timedelta(minutes=10),
        )
        raw_matching_id = _save_raw(
            stack._backend,
            blob="deployed api service to staging environment successfully",
            captured_at=now - timedelta(minutes=10),
        )

        ep = _ep(
            objective="deployed api service to staging environment",
            outcome="it worked",
            created_at=now,
        )
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1
        derived_from = calls[0].kwargs.get("derived_from")
        assert f"raw:{raw_matching_id}" in derived_from

    def test_no_raw_entries(self, stack):
        ep = _ep()
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)  # Should not error

    def test_preserves_existing_annotations(self, stack):
        now = datetime.now(timezone.utc)
        raw_id = _save_raw(
            stack._backend,
            blob="test preserved annotation content here",
            captured_at=now,
        )

        ep = _ep(objective="test preserved annotation content here", created_at=now)
        stack.save_episode(ep)
        # Manually add pre-v0.9 annotation
        with stack._backend._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (json.dumps(["kernle:pre-v0.9-migration"]), ep.id),
            )
            conn.commit()

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1
        derived_from = calls[0].kwargs.get("derived_from")
        assert f"raw:{raw_id}" in derived_from
        assert "kernle:auto-linked" in derived_from
        assert "kernle:pre-v0.9-migration" in derived_from

    def test_all_flag_creates_synthetic_raw(self, stack):
        """--all creates synthetic raw entries for unmatched memories."""
        now = datetime.now(timezone.utc)
        # Raw entry is far away in time and has no content overlap
        _save_raw(
            stack._backend,
            blob="completely unrelated entry about cooking",
            captured_at=now - timedelta(hours=5),
        )

        ep = _ep(objective="deployed api service", outcome="success", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = True

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)
        # save_raw returns a synthetic raw id
        synthetic_id = str(uuid.uuid4())
        mock_k._storage.save_raw = MagicMock(return_value=synthetic_id)

        _migrate_link_raw(args, mock_k)

        # Synthetic raw should have been created
        mock_k._storage.save_raw.assert_called_once()
        call_args = mock_k._storage.save_raw.call_args
        assert "[migrated episode]" in call_args[0][0]
        assert "deployed api service" in call_args[0][0]

        # Memory should be linked to the synthetic raw
        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1
        derived_from = calls[0].kwargs.get("derived_from")
        assert f"raw:{synthetic_id}" in derived_from
        assert "kernle:auto-linked" in derived_from
        assert "kernle:synthetic-raw" in derived_from

    def test_all_flag_dry_run_no_synthetic(self, stack):
        """--all --dry-run shows synthetic links but doesn't create them."""
        now = datetime.now(timezone.utc)
        # Need at least one raw entry so the function doesn't bail early
        _save_raw(
            stack._backend,
            blob="unrelated raw entry",
            captured_at=now - timedelta(hours=5),
        )

        ep = _ep(objective="unmatched task", outcome="done", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        args = MagicMock()
        args.dry_run = True
        args.json = True
        args.window = 30
        args.link_all = True

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID

        _migrate_link_raw(args, mock_k)

        # Nothing should have been applied (dry run)
        mock_k.set_memory_source.assert_not_called()

    def test_all_flag_not_set_skips_unmatched(self, stack):
        """Without --all, unmatched memories are left alone."""
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="unrelated content",
            captured_at=now - timedelta(hours=5),
        )

        ep = _ep(objective="deployed api service", outcome="success", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        mock_k.set_memory_source.assert_not_called()

    def test_link_json_is_deterministic(self, stack, capsys):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="alpha raw match",
            captured_at=now,
        )
        _save_raw(
            stack._backend,
            blob="beta raw match",
            captured_at=now,
        )

        ep = Episode(
            id="z-episode",
            stack_id=STACK_ID,
            objective="alpha raw match",
            outcome="good",
            created_at=now,
        )
        note = Note(
            id="a-note",
            stack_id=STACK_ID,
            content="beta raw match",
            note_type="observation",
            created_at=now,
        )
        stack.save_episode(ep)
        stack.save_note(note)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        output = json.loads(capsys.readouterr().out)
        assert output["status_snapshot"]["matched_links"] == 2
        assert output["links"][0]["type"] == "episode"
        assert output["links"][1]["type"] == "note"
        assert output["status_snapshot"]["ids"][0] < output["status_snapshot"]["ids"][1]
        assert output["status_snapshot_sha256"] == _snapshot_sha256(output["status_snapshot"])


class TestPreV09ProvenanceBypass:
    """Test that pre-v0.9 migrated memories bypass provenance validation."""

    def test_pre_v09_annotation_bypasses_provenance(self, tmp_path):
        """Memories with kernle:pre-v0.9-migration pass provenance validation."""
        stack = Stack.from_sqlite(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=True
        )
        stack._state = stack._state.__class__["ACTIVE"]

        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="legacy episode",
            outcome="success",
            derived_from=["kernle:pre-v0.9-migration"],
            created_at=datetime.now(timezone.utc),
        )
        # Should not raise ProvenanceError
        stack.save_episode(ep)

    def test_annotation_only_still_fails_without_pre_v09(self, tmp_path):
        """Other annotation-only refs still fail provenance validation."""
        from kernle.stack.sqlite_stack import ProvenanceError

        stack = Stack.from_sqlite(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=True
        )
        stack._state = stack._state.__class__["ACTIVE"]

        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="episode with only context annotation",
            outcome="result",
            derived_from=["context:cli"],
            created_at=datetime.now(timezone.utc),
        )
        with pytest.raises(ProvenanceError):
            stack.save_episode(ep)


# ==============================================================================
# Helpers for additional memory types
# ==============================================================================


def _value(name="Test value", statement="Test statement"):
    return Value(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        name=name,
        statement=statement,
        created_at=datetime.now(timezone.utc),
    )


def _goal(title="Test goal", description="Test description"):
    return Goal(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        title=title,
        description=description,
        created_at=datetime.now(timezone.utc),
    )


def _drive(drive_type="curiosity"):
    return Drive(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        drive_type=drive_type,
        created_at=datetime.now(timezone.utc),
    )


def _relationship(entity_name="Alice", entity_type="human", relationship_type="collaborator"):
    return Relationship(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        entity_name=entity_name,
        entity_type=entity_type,
        relationship_type=relationship_type,
        created_at=datetime.now(timezone.utc),
    )


# ==============================================================================
# backfill-provenance: values, goals, drives, relationships
# ==============================================================================


class TestBackfillProvenanceAllTypes:
    """Tests for _migrate_backfill_provenance covering values, goals, drives, relationships."""

    def _run_backfill(self, stack, dry_run=False):
        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = dry_run
        args.json = True

        _migrate_backfill_provenance(args, mock_k)
        return mock_k

    def test_values_get_annotation(self, stack):
        val = _value()
        stack.save_value(val)

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        val_calls = [c for c in calls if c[0][1] == val.id]
        assert len(val_calls) == 1
        derived_from = val_calls[0].kwargs.get("derived_from")
        assert derived_from is not None
        assert "kernle:pre-v0.9-migration" in derived_from

    def test_goals_get_annotation(self, stack):
        goal = _goal()
        stack.save_goal(goal)

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        goal_calls = [c for c in calls if c[0][1] == goal.id]
        assert len(goal_calls) == 1
        derived_from = goal_calls[0].kwargs.get("derived_from")
        assert derived_from is not None
        assert "kernle:pre-v0.9-migration" in derived_from

    def test_drives_get_annotation(self, stack):
        drive = _drive()
        stack.save_drive(drive)

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        drive_calls = [c for c in calls if c[0][1] == drive.id]
        assert len(drive_calls) == 1
        derived_from = drive_calls[0].kwargs.get("derived_from")
        assert derived_from is not None
        assert "kernle:pre-v0.9-migration" in derived_from

    def test_relationships_get_annotation(self, stack):
        rel = _relationship()
        stack.save_relationship(rel)

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        rel_calls = [c for c in calls if c[0][1] == rel.id]
        assert len(rel_calls) == 1
        derived_from = rel_calls[0].kwargs.get("derived_from")
        assert derived_from is not None
        assert "kernle:pre-v0.9-migration" in derived_from

    def test_value_processed_migrated_to_processing(self, stack):
        val = _value()
        stack.save_value(val)
        # Manually set source_type to legacy "processed"
        with stack._backend._connect() as conn:
            conn.execute(
                "UPDATE agent_values SET source_type = ? WHERE id = ?",
                ("processed", val.id),
            )
            conn.commit()

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        val_calls = [c for c in calls if c[0][1] == val.id]
        assert len(val_calls) == 1
        # source_type should be "processing" (the canonical value)
        assert val_calls[0][0][2] == "processing"

    def test_goal_processed_migrated_to_processing(self, stack):
        goal = _goal()
        stack.save_goal(goal)
        with stack._backend._connect() as conn:
            conn.execute(
                "UPDATE goals SET source_type = ? WHERE id = ?",
                ("processed", goal.id),
            )
            conn.commit()

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        goal_calls = [c for c in calls if c[0][1] == goal.id]
        assert len(goal_calls) == 1
        assert goal_calls[0][0][2] == "processing"

    def test_drive_processed_migrated_to_processing(self, stack):
        drive = _drive()
        stack.save_drive(drive)
        with stack._backend._connect() as conn:
            conn.execute(
                "UPDATE drives SET source_type = ? WHERE id = ?",
                ("processed", drive.id),
            )
            conn.commit()

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        drive_calls = [c for c in calls if c[0][1] == drive.id]
        assert len(drive_calls) == 1
        assert drive_calls[0][0][2] == "processing"

    def test_relationship_processed_migrated_to_processing(self, stack):
        rel = _relationship()
        stack.save_relationship(rel)
        with stack._backend._connect() as conn:
            conn.execute(
                "UPDATE relationships SET source_type = ? WHERE id = ?",
                ("processed", rel.id),
            )
            conn.commit()

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        rel_calls = [c for c in calls if c[0][1] == rel.id]
        assert len(rel_calls) == 1
        assert rel_calls[0][0][2] == "processing"

    def test_value_with_provenance_untouched(self, stack):
        raw_id = _save_raw(stack._backend)
        val = _value()
        val.derived_from = [f"raw:{raw_id}"]
        val.source_type = "direct_experience"
        stack.save_value(val)

        mock_k = self._run_backfill(stack)

        calls = mock_k.set_memory_source.call_args_list
        val_calls = [c for c in calls if c[0][1] == val.id]
        assert len(val_calls) == 0

    def test_dry_run_no_changes_for_new_types(self, stack):
        val = _value()
        goal = _goal()
        drive = _drive()
        rel = _relationship()
        stack.save_value(val)
        stack.save_goal(goal)
        stack.save_drive(drive)
        stack.save_relationship(rel)

        mock_k = self._run_backfill(stack, dry_run=True)

        mock_k.set_memory_source.assert_not_called()
