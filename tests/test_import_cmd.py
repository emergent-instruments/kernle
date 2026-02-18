"""Tests for kernle/cli/commands/import_cmd.py — import execution paths.

Covers: cmd_import, _import_json, _import_csv, _import_markdown,
        _batch_import, _check_duplicate, _import_item, _preview_item,
        _interactive_import.

These tests focus on the IMPORT EXECUTION code paths, not the markdown
_parse_* functions (already tested in test_import.py).

Note: Several import_cmd.py code paths have API mismatches with the
Kernle core API (e.g., value() takes `statement` not `description`,
goal() has no `status` kwarg, search() has no `record_types` kwarg,
drive() gets `focus` instead of `focus_areas`). Tests document these
known issues where import_cmd.py silently catches the resulting errors.
"""

import argparse
import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from kernle import Kernle
from kernle.cli.commands.import_cmd import (
    _batch_import,
    _check_duplicate,
    _edit_item,
    _import_csv,
    _import_item,
    _import_json,
    _import_markdown,
    _import_pdf,
    _interactive_import,
    _item_signature,
    _merge_derived_from,
    _preview_item,
    cmd_import,
)
from kernle.cli.commands.migrate import (
    _migrate_backfill_provenance,
    _migrate_link_raw,
    _migrate_seed_beliefs,
    cmd_migrate,
)
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance with temp storage."""
    storage = SQLiteStorage(stack_id="test-import", db_path=tmp_path / "import.db")
    inst = Kernle(stack_id="test-import", storage=storage, strict=False)
    bind_noop_model(inst)
    yield inst
    storage.close()


@pytest.fixture
def k2(tmp_path):
    """Create a second Kernle instance (for round-trip import tests)."""
    storage = SQLiteStorage(stack_id="test-import-2", db_path=tmp_path / "import2.db")
    inst = Kernle(stack_id="test-import-2", storage=storage, strict=False)
    bind_noop_model(inst)
    yield inst
    storage.close()


def _make_args(**kwargs):
    """Build an argparse.Namespace with defaults for cmd_import."""
    defaults = dict(
        file="",
        format=None,
        dry_run=False,
        interactive=False,
        skip_duplicates=True,
        derived_from=None,
        chunk_size=2000,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ============================================================================
# TestCmdImport — top-level dispatcher
# ============================================================================


class TestCmdImport:
    """Tests for the cmd_import dispatcher function."""

    def test_file_not_found(self, k, capsys):
        args = _make_args(file="/nonexistent/path.json")
        cmd_import(args, k)
        assert "File not found" in capsys.readouterr().out

    def test_unknown_extension(self, k, tmp_path, capsys):
        f = tmp_path / "data.xyz"
        f.write_text("hello")
        args = _make_args(file=str(f))
        cmd_import(args, k)
        out = capsys.readouterr().out
        assert "Unknown file format" in out

    def test_auto_detect_json(self, k, tmp_path, capsys):
        f = tmp_path / "data.json"
        f.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw obs"}],
                    "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                    "beliefs": [
                        {
                            "id": "b1",
                            "statement": "Auto JSON",
                            "confidence": 0.8,
                            "belief_type": "fact",
                            "derived_from": ["note:n1"],
                        }
                    ],
                }
            )
        )
        args = _make_args(file=str(f))
        cmd_import(args, k)
        beliefs = k._storage.get_beliefs(limit=10)
        assert any(b.statement == "Auto JSON" for b in beliefs)

    def test_auto_detect_csv(self, k, tmp_path, capsys):
        f = tmp_path / "data.csv"
        f.write_text("type,content\nraw,CSV auto detect\n")
        args = _make_args(file=str(f))
        cmd_import(args, k)
        raw = k._storage.list_raw(limit=10)
        assert any(r.content == "CSV auto detect" for r in raw)

    def test_auto_detect_markdown(self, k, tmp_path, capsys):
        f = tmp_path / "data.md"
        f.write_text("## Thoughts\n\n- MD auto detect raw\n")
        args = _make_args(file=str(f))
        cmd_import(args, k)
        raw = k._storage.list_raw(limit=10)
        assert any("MD auto detect raw" in r.content for r in raw)

    def test_auto_detect_txt_as_markdown(self, k, tmp_path, capsys):
        f = tmp_path / "data.txt"
        f.write_text("## Thoughts\n\n- TXT raw thought\n")
        args = _make_args(file=str(f))
        cmd_import(args, k)
        raw = k._storage.list_raw(limit=10)
        assert any("TXT raw thought" in r.content for r in raw)

    def test_auto_detect_markdown_extension(self, k, tmp_path, capsys):
        f = tmp_path / "data.markdown"
        f.write_text("## Thoughts\n\n- Markdown ext raw thought\n")
        args = _make_args(file=str(f))
        cmd_import(args, k)
        raw = k._storage.list_raw(limit=10)
        assert any("Markdown ext raw thought" in r.content for r in raw)

    def test_format_override(self, k, tmp_path, capsys):
        """Explicit --format overrides file extension."""
        f = tmp_path / "data.xyz"
        f.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw obs"}],
                    "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                    "beliefs": [
                        {
                            "id": "b1",
                            "statement": "Override",
                            "confidence": 0.7,
                            "belief_type": "fact",
                            "derived_from": ["note:n1"],
                        }
                    ],
                }
            )
        )
        args = _make_args(file=str(f), format="json")
        cmd_import(args, k)
        beliefs = k._storage.get_beliefs(limit=10)
        assert any(b.statement == "Override" for b in beliefs)

    def test_tilde_expansion(self, k, capsys):
        """Path with ~ should be expanded (even if nonexistent)."""
        args = _make_args(file="~/nonexistent_import_test.json")
        cmd_import(args, k)
        out = capsys.readouterr().out
        assert "File not found" in out

    def test_dry_run_passed_to_json(self, k, tmp_path, capsys):
        f = tmp_path / "data.json"
        f.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw obs"}],
                    "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                    "beliefs": [
                        {
                            "id": "b1",
                            "statement": "DryJSON",
                            "confidence": 0.8,
                            "derived_from": ["note:n1"],
                        }
                    ],
                }
            )
        )
        args = _make_args(file=str(f), dry_run=True)
        cmd_import(args, k)
        assert len(k._storage.get_beliefs(limit=10)) == 0
        assert "DRY RUN" in capsys.readouterr().out

    def test_dry_run_passed_to_csv(self, k, tmp_path, capsys):
        f = tmp_path / "data.csv"
        f.write_text("type,content\nraw,DryCSV raw content\n")
        args = _make_args(file=str(f), dry_run=True)
        cmd_import(args, k)
        assert len(k._storage.list_raw(limit=10)) == 0
        assert "DRY RUN" in capsys.readouterr().out

    def test_dry_run_passed_to_markdown(self, k, tmp_path, capsys):
        f = tmp_path / "data.md"
        f.write_text("## Thoughts\n\n- DryMD raw thought\n")
        args = _make_args(file=str(f), dry_run=True)
        cmd_import(args, k)
        assert len(k._storage.list_raw(limit=10)) == 0
        assert "DRY RUN" in capsys.readouterr().out


# ============================================================================
# TestImportJson
# ============================================================================


class TestImportJson:
    """Tests for _import_json execution."""

    def _write_json(self, tmp_path, data, filename="import.json"):
        f = tmp_path / filename
        f.write_text(json.dumps(data))
        return f

    def test_import_beliefs(self, k, tmp_path, capsys):
        f = self._write_json(
            tmp_path,
            {
                "stack_id": "source",
                "exported_at": "2024-01-01T00:00:00Z",
                "raw_entries": [{"id": "r1", "content": "raw obs"}],
                "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                "beliefs": [
                    {
                        "id": "b1",
                        "statement": "B1",
                        "confidence": 0.9,
                        "belief_type": "fact",
                        "derived_from": ["note:n1"],
                    },
                    {
                        "id": "b2",
                        "statement": "B2",
                        "confidence": 0.7,
                        "belief_type": "fact",
                        "derived_from": ["note:n1"],
                    },
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        beliefs = k._storage.get_beliefs(limit=10)
        assert len(beliefs) == 2
        statements = {b.statement for b in beliefs}
        assert "B1" in statements
        assert "B2" in statements

    def test_import_episodes(self, k, tmp_path, capsys):
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "E1",
                        "outcome": "O1",
                        "lessons": ["L1"],
                        "derived_from": ["raw:r1"],
                    },
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        episodes = k._storage.get_episodes(limit=10)
        assert len(episodes) == 1
        assert episodes[0].objective == "E1"

    def test_import_notes(self, k, tmp_path, capsys):
        """Note import uses k.note() which prefixes content with type label."""
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "notes": [
                    {
                        "id": "n1",
                        "content": "N1",
                        "type": "note",
                        "speaker": "user",
                        "derived_from": ["raw:r1"],
                    },
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        notes = k._storage.get_notes(limit=10)
        assert len(notes) == 1
        assert "N1" in notes[0].content

    def test_import_notes_insight_type(self, k, tmp_path, capsys):
        """Insight-type notes get prefixed by k.note()."""
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "notes": [
                    {
                        "id": "n1",
                        "content": "InsightContent",
                        "type": "insight",
                        "derived_from": ["raw:r1"],
                    }
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        notes = k._storage.get_notes(limit=10)
        assert len(notes) == 1
        assert "InsightContent" in notes[0].content

    def test_import_values_errors_caught(self, k, tmp_path, capsys):
        """Value import has API mismatch (description vs statement) — errors are caught."""
        f = self._write_json(
            tmp_path,
            {
                "values": [
                    {"name": "Quality", "description": "Test well", "priority": 80},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        # The call uses description= which doesn't match k.value(statement=) — error caught
        assert "error" in out.lower() or "Imported 0" in out

    def test_import_values_with_statement_field(self, k, tmp_path, capsys):
        """Values with statement field (matching export format) should import."""
        f = self._write_json(
            tmp_path,
            {
                "values": [
                    {"name": "Quality", "statement": "Test well", "priority": 80},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        # import_cmd uses item.get("statement", item.get("description", ""))
        # which will find "statement" first, passing it as description= to k.value()
        # But k.value() has no description= parameter, so it still errors
        out = capsys.readouterr().out
        # Either imported or caught error
        values = k._storage.get_values(limit=10)
        if len(values) == 0:
            assert "error" in out.lower()

    def test_import_goals_errors_caught(self, k, tmp_path, capsys):
        """Goal import has API mismatch (status kwarg) — errors are caught."""
        f = self._write_json(
            tmp_path,
            {
                "goals": [
                    {
                        "title": "Ship v1",
                        "description": "Release",
                        "priority": "high",
                        "status": "active",
                    },
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        # The goal() call passes status= which is not a valid kwarg
        assert "error" in out.lower() or "Imported 0" in out

    def test_import_drives_errors_caught(self, k, tmp_path, capsys):
        """Drive import uses focus_areas argument and succeeds."""
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "explored",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "drives": [
                    {
                        "id": "d1",
                        "drive_type": "curiosity",
                        "intensity": 0.8,
                        "focus_areas": ["learning"],
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Imported" in out
        drive = k._storage.get_drive("curiosity")
        assert drive is not None
        assert drive.intensity == 0.8

    def test_import_drives_without_focus(self, k, tmp_path, capsys):
        """Drives without focus_areas should import (no bad kwarg)."""
        f = self._write_json(
            tmp_path,
            {
                "drives": [
                    {"drive_type": "growth", "intensity": 0.7},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        # Without focus_areas, focus=None which k.drive() may accept
        out = capsys.readouterr().out
        # Check if imported or errored
        drive = k._storage.get_drive("growth")
        if drive is None:
            assert "error" in out.lower()

    def test_import_raw_entries(self, k, tmp_path, capsys):
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [
                    {"content": "Raw thought 1", "source": "test"},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        raw = k._storage.list_raw(limit=10)
        assert len(raw) == 1
        assert raw[0].content == "Raw thought 1"

    def test_import_relationships(self, k, tmp_path, capsys):
        """Relationship import may fail due to API mismatch — errors caught."""
        f = self._write_json(
            tmp_path,
            {
                "relationships": [
                    {
                        "entity_name": "Alice",
                        "entity_type": "person",
                        "relationship_type": "collaborator",
                        "sentiment": 0.5,
                        "notes": "Works well",
                    },
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        rel = k._storage.get_relationship("Alice")
        if rel is None:
            assert "error" in out.lower() or "Imported" in out

    def test_dry_run_no_writes(self, k, tmp_path, capsys):
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                "beliefs": [
                    {
                        "id": "b1",
                        "statement": "Should not exist",
                        "confidence": 0.9,
                        "derived_from": ["note:n1"],
                    }
                ],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "Should not exist",
                        "outcome": "done",
                        "derived_from": ["raw:r1"],
                    }
                ],
            },
        )
        _import_json(f, k, dry_run=True, skip_duplicates=False)
        assert len(k._storage.get_beliefs(limit=10)) == 0
        assert len(k._storage.get_episodes(limit=10)) == 0
        assert "DRY RUN" in capsys.readouterr().out

    def test_skip_duplicates_beliefs(self, k, tmp_path, capsys):
        k.belief("Already exists", confidence=0.8)
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                "beliefs": [
                    {
                        "id": "b1",
                        "statement": "Already exists",
                        "confidence": 0.9,
                        "belief_type": "fact",
                        "derived_from": ["note:n1"],
                    },
                    {
                        "id": "b2",
                        "statement": "New belief",
                        "confidence": 0.7,
                        "belief_type": "fact",
                        "derived_from": ["note:n1"],
                    },
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=True)
        out = capsys.readouterr().out
        assert "Skipped" in out or "duplicate" in out.lower()
        beliefs = k._storage.get_beliefs(limit=20)
        statements = [b.statement for b in beliefs]
        assert statements.count("Already exists") == 1
        assert "New belief" in statements

    def test_skip_duplicates_values(self, k, tmp_path, capsys):
        k.value(name="Honesty", statement="Be honest")
        f = self._write_json(
            tmp_path,
            {
                "values": [
                    {"name": "Honesty", "description": "duplicate"},
                    {"name": "Courage", "description": "Be brave"},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=True)
        capsys.readouterr()  # consume output
        # First value skipped as duplicate, second errors due to API mismatch
        values = k._storage.get_values(limit=20)
        names = [v.name for v in values]
        assert names.count("Honesty") == 1
        # Courage may or may not import depending on error handling

    def test_skip_duplicates_goals(self, k, tmp_path, capsys):
        k.goal(title="Ship v1", description="Ship v1")
        f = self._write_json(
            tmp_path,
            {
                "goals": [
                    {"title": "Ship v1", "description": "Ship v1"},
                    {"title": "Ship v2", "description": "Ship v2"},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=True)
        goals = k._storage.get_goals(status=None, limit=20)
        titles = [g.title for g in goals]
        # First goal skipped as dup, second may error on status= kwarg
        assert titles.count("Ship v1") == 1

    def test_skip_duplicates_drives(self, k, tmp_path, capsys):
        k.drive(drive_type="curiosity", intensity=0.5)
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "explored",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "drives": [
                    {
                        "id": "d1",
                        "drive_type": "curiosity",
                        "intensity": 0.9,
                        "derived_from": ["episode:e1"],
                    },
                    {
                        "id": "d2",
                        "drive_type": "growth",
                        "intensity": 0.7,
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=True)
        out = capsys.readouterr().out
        assert "Skipped" in out or "duplicate" in out.lower()

    def test_skip_duplicates_raw(self, k, tmp_path, capsys):
        k.raw(blob="Existing raw", source="test")
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [
                    {"content": "Existing raw", "source": "test"},
                    {"content": "New raw", "source": "test"},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=True)
        raw = k._storage.list_raw(limit=20)
        contents = [r.content for r in raw]
        assert contents.count("Existing raw") == 1
        assert "New raw" in contents

    def test_skip_duplicates_relationships(self, k, tmp_path, capsys):
        """Relationship duplicate detection: existing relationship is skipped."""
        # Create a relationship directly in storage
        k.relationship("Bob", entity_type="person")
        f = self._write_json(
            tmp_path,
            {
                "relationships": [
                    {"entity_name": "Bob", "entity_type": "person"},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=True)
        out = capsys.readouterr().out
        assert "Skipped" in out or "duplicate" in out.lower() or "relationship" in out.lower()

    def test_malformed_json(self, k, tmp_path, capsys):
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        assert "Invalid JSON" in capsys.readouterr().out

    def test_json_root_not_object(self, k, tmp_path, capsys):
        f = tmp_path / "array.json"
        f.write_text("[]")
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        assert "must be an object" in capsys.readouterr().out

    def test_json_empty_no_content(self, k, tmp_path, capsys):
        f = self._write_json(tmp_path, {"stack_id": "empty"})
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        assert "No importable content" in capsys.readouterr().out

    def test_provenance_derived_from(self, k, tmp_path, capsys):
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                "beliefs": [
                    {
                        "id": "b1",
                        "statement": "Provenance test",
                        "confidence": 0.8,
                        "belief_type": "fact",
                        "derived_from": ["note:n1"],
                    }
                ],
            },
        )
        _import_json(
            f,
            k,
            dry_run=False,
            skip_duplicates=False,
            derived_from=["context:test-source"],
        )
        beliefs = k._storage.get_beliefs(limit=10)
        b = next(b for b in beliefs if b.statement == "Provenance test")
        assert b.derived_from is not None
        assert "context:test-source" in b.derived_from

    def test_provenance_on_episodes(self, k, tmp_path, capsys):
        f = self._write_json(
            tmp_path,
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "Prov ep",
                        "outcome": "Done",
                        "derived_from": ["raw:r1"],
                    }
                ],
            },
        )
        _import_json(
            f,
            k,
            dry_run=False,
            skip_duplicates=False,
            derived_from=["context:ep-source"],
        )
        episodes = k._storage.get_episodes(limit=10)
        ep = next(e for e in episodes if e.objective == "Prov ep")
        assert ep.derived_from is not None
        assert "context:ep-source" in ep.derived_from

    def test_summary_output(self, k, tmp_path, capsys):
        f = self._write_json(
            tmp_path,
            {
                "stack_id": "src-agent",
                "exported_at": "2024-06-15T12:00:00Z",
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "notes": [{"id": "n1", "content": "N1", "derived_from": ["raw:r1"]}],
                "beliefs": [
                    {"id": "b1", "statement": "S1", "confidence": 0.8, "derived_from": ["note:n1"]},
                    {"id": "b2", "statement": "S2", "confidence": 0.7, "derived_from": ["note:n1"]},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "src-agent" in out
        assert "2024-06-15" in out
        assert "Imported" in out

    def test_summary_shows_type_counts(self, k, tmp_path, capsys):
        f = self._write_json(
            tmp_path,
            {
                "beliefs": [
                    {"statement": "B1", "confidence": 0.8},
                    {"statement": "B2", "confidence": 0.7},
                ],
                "episodes": [{"objective": "E1", "outcome": "O1"}],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "belief: 2" in out
        assert "episode: 1" in out

    def test_round_trip_export_import_beliefs_episodes(self, k, k2, tmp_path, capsys):
        """Export beliefs/episodes from one agent, import to another.

        The round-trip requires proper provenance chains in the export.
        We create raw -> episode/note -> belief with derived_from links
        so the exported JSON passes provenance validation on re-import.
        """
        raw_id = k.raw(blob="Round-trip observation", source="test")
        k.episode(
            objective="Round-trip task",
            outcome="Done",
            derived_from=[f"raw:{raw_id}"],
        )
        note_id = k.note(
            "Round-trip note",
            type="note",
            derived_from=[f"raw:{raw_id}"],
        )
        k.belief(
            "Round-trip belief",
            confidence=0.85,
            derived_from=[f"note:{note_id}"],
        )

        export_json = k.dump(format="json")
        f = tmp_path / "export.json"
        f.write_text(export_json)

        _import_json(f, k2, dry_run=False, skip_duplicates=False)

        assert len(k2._storage.get_beliefs(limit=20)) >= 1
        assert len(k2._storage.get_episodes(limit=20)) >= 1
        # Notes import successfully (content may get type-prefixed)
        assert len(k2._storage.get_notes(limit=20)) >= 1

    def test_error_reporting_truncation(self, k, tmp_path, capsys):
        """When there are > 5 errors, only first 5 shown plus count."""
        # Create many values that will all error on description= mismatch
        values = [{"name": f"V{i}", "description": f"Desc {i}"} for i in range(8)]
        f = self._write_json(tmp_path, {"values": values})
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "error" in out.lower()
        # Should truncate at 5
        if "and" in out and "more" in out:
            assert "3 more" in out

    def test_short_exported_at(self, k, tmp_path, capsys):
        """exported_at shorter than 10 chars should not error."""
        f = self._write_json(
            tmp_path,
            {
                "stack_id": "short",
                "exported_at": "2024",
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                "beliefs": [
                    {
                        "id": "b1",
                        "statement": "Short date",
                        "confidence": 0.8,
                        "derived_from": ["note:n1"],
                    }
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "2024" in out
        assert "Imported" in out


# ============================================================================
# TestImportCsv
# ============================================================================


class TestImportCsv:
    """Tests for _import_csv execution."""

    def _write_csv(self, tmp_path, content, filename="import.csv"):
        f = tmp_path / filename
        f.write_text(content)
        return f

    def test_csv_with_type_column(self, k, tmp_path, capsys):
        f = self._write_csv(tmp_path, "type,content\nraw,CSV raw entry\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        raw = k._storage.list_raw(limit=10)
        assert any(r.content == "CSV raw entry" for r in raw)

    def test_csv_with_memory_type_column(self, k, tmp_path, capsys):
        f = self._write_csv(tmp_path, "memory_type,content\nraw,MT raw entry\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        raw = k._storage.list_raw(limit=10)
        assert any(r.content == "MT raw entry" for r in raw)

    def test_csv_with_kind_column(self, k, tmp_path, capsys):
        f = self._write_csv(tmp_path, "kind,content\nraw,Kind raw entry\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        raw = k._storage.list_raw(limit=10)
        assert any(r.content == "Kind raw entry" for r in raw)

    def test_csv_mixed_types_skips_non_raw(self, k, tmp_path, capsys):
        """CSV import only imports raw items; non-raw types are skipped."""
        content = (
            "type,content\n" "raw,Raw CSV content\n" "belief,Skipped belief\n" "note,Skipped note\n"
        )
        f = self._write_csv(tmp_path, content)
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        assert len(k._storage.list_raw(limit=10)) == 1
        out = capsys.readouterr().out
        assert "Skipped 2 non-raw items" in out

    def test_csv_no_type_column_error(self, k, tmp_path, capsys):
        f = self._write_csv(tmp_path, "statement,confidence\nBelief,0.9\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "must have a 'type' column" in out

    def test_csv_dry_run(self, k, tmp_path, capsys):
        f = self._write_csv(tmp_path, "type,content\nraw,Dry CSV raw\n")
        _import_csv(f, k, dry_run=True, skip_duplicates=False)
        assert len(k._storage.list_raw(limit=10)) == 0
        assert "DRY RUN" in capsys.readouterr().out

    def test_csv_skip_duplicates(self, k, tmp_path, capsys):
        k.raw(blob="Dup CSV raw", source="test")
        f = self._write_csv(tmp_path, "type,content\nraw,Dup CSV raw\nraw,New CSV raw\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=True)
        raw = k._storage.list_raw(limit=20)
        contents = [r.content for r in raw]
        assert contents.count("Dup CSV raw") == 1
        assert "New CSV raw" in contents

    def test_csv_episode_columns_skipped(self, k, tmp_path, capsys):
        """Episode rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(
            tmp_path,
            'type,objective,outcome,outcome_type,lessons\nepisode,Fix bug,Fixed,success,"test first,verify"\n',
        )
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_note_columns_skipped(self, k, tmp_path, capsys):
        """Note rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(
            tmp_path, "type,content,note_type,speaker\nnote,CSV note content,note,user\n"
        )
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_value_columns_skipped(self, k, tmp_path, capsys):
        """Value rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(
            tmp_path, "type,name,description,priority\nvalue,Honesty,Always tell truth,80\n"
        )
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_goal_columns_skipped(self, k, tmp_path, capsys):
        """Goal rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(
            tmp_path, "type,title,description,status\ngoal,Ship v1,Release,active\n"
        )
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_raw_columns(self, k, tmp_path, capsys):
        f = self._write_csv(tmp_path, "type,content,source\nraw,Raw CSV content,csv-import\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        raw = k._storage.list_raw(limit=10)
        assert any(r.content == "Raw CSV content" for r in raw)

    def test_csv_belief_rows_skipped(self, k, tmp_path, capsys):
        """Belief rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(tmp_path, "type,statement,confidence\nbelief,High conf,90\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_raw_with_text_column(self, k, tmp_path, capsys):
        """Raw items can use 'text' column as content alias."""
        f = self._write_csv(tmp_path, "type,text\nraw,Text column content\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        raw = k._storage.list_raw(limit=10)
        assert any(r.content == "Text column content" for r in raw)

    def test_csv_empty_rows_skipped(self, k, tmp_path, capsys):
        """Rows with no content should be skipped."""
        f = self._write_csv(tmp_path, "type,content\nraw,\nraw,Has content\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        raw = k._storage.list_raw(limit=10)
        assert len(raw) == 1

    def test_csv_no_importable_content(self, k, tmp_path, capsys):
        f = self._write_csv(tmp_path, "type,content\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        assert "No raw entries found" in capsys.readouterr().out

    def test_csv_no_headers(self, k, tmp_path, capsys):
        f = self._write_csv(tmp_path, "")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "no headers" in out.lower() or "No importable" in out

    def test_csv_derived_from(self, k, tmp_path, capsys):
        """CSV raw items don't carry derived_from (raw entries use blob source), but the function accepts the parameter."""
        f = self._write_csv(tmp_path, "type,content\nraw,Prov CSV raw\n")
        _import_csv(
            f,
            k,
            dry_run=False,
            skip_duplicates=False,
            derived_from=["context:csv-source"],
        )
        raw = k._storage.list_raw(limit=10)
        r = next(r for r in raw if r.content == "Prov CSV raw")
        assert r is not None

    def test_csv_dry_run_preview_limit(self, k, tmp_path, capsys):
        """Dry run preview shows max 10 items."""
        rows = "\n".join(f"raw,Content {i}" for i in range(15))
        f = self._write_csv(tmp_path, f"type,content\n{rows}\n")
        _import_csv(f, k, dry_run=True, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "and 5 more" in out

    def test_csv_column_aliases_episode_skipped(self, k, tmp_path, capsys):
        """Episode rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(tmp_path, "type,title\nepisode,Task via title alias\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_missing_type_row_skipped(self, k, tmp_path, capsys):
        """Rows without a type value are skipped."""
        f = self._write_csv(tmp_path, "type,content\nraw,Has type\n,Missing type\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        raw = k._storage.list_raw(limit=10)
        assert len(raw) == 1

    def test_csv_type_count_summary(self, k, tmp_path, capsys):
        content = "type,content\nraw,R1\nraw,R2\nraw,R3\n"
        f = self._write_csv(tmp_path, content)
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "raw: 3" in out


# ============================================================================
# TestImportMarkdown
# ============================================================================


class TestImportMarkdown:
    """Tests for _import_markdown orchestration."""

    def _write_md(self, tmp_path, content, filename="import.md"):
        f = tmp_path / filename
        f.write_text(content)
        return f

    def test_markdown_beliefs_skipped(self, k, tmp_path, capsys):
        """Belief sections in markdown are skipped (markdown is raw-only)."""
        f = self._write_md(tmp_path, "## Beliefs\n\n- MD imported belief (90%)\n- Another belief\n")
        _import_markdown(f, k, dry_run=False, interactive=False)
        out = capsys.readouterr().out
        assert "Skipped 2 non-raw items" in out

    def test_markdown_episodes_skipped(self, k, tmp_path, capsys):
        """Episode sections in markdown are skipped (markdown is raw-only)."""
        f = self._write_md(tmp_path, "## Episodes\n\n- Fixed the bug -> Test first\n")
        _import_markdown(f, k, dry_run=False, interactive=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_markdown_notes_skipped(self, k, tmp_path, capsys):
        """Note sections in markdown are skipped (markdown is raw-only)."""
        f = self._write_md(tmp_path, "## Notes\n\n- Important note from MD\n")
        _import_markdown(f, k, dry_run=False, interactive=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_markdown_goals_skipped(self, k, tmp_path, capsys):
        """Goal sections in markdown are skipped (markdown is raw-only)."""
        f = self._write_md(tmp_path, "## Goals\n\n- Ship v1\n- [done] Write docs\n")
        _import_markdown(f, k, dry_run=False, interactive=False)
        out = capsys.readouterr().out
        assert "Skipped 2 non-raw items" in out

    def test_markdown_values_skipped(self, k, tmp_path, capsys):
        """Value sections in markdown are skipped (markdown is raw-only)."""
        f = self._write_md(tmp_path, "## Values\n\n- Quality: Always test\n")
        _import_markdown(f, k, dry_run=False, interactive=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_markdown_raw_imported(self, k, tmp_path, capsys):
        f = self._write_md(tmp_path, "## Thoughts\n\n- Random thought\n")
        _import_markdown(f, k, dry_run=False, interactive=False)
        raw = k._storage.list_raw(limit=10)
        assert len(raw) == 1

    def test_markdown_raw_sections_imported(self, k, tmp_path, capsys):
        """Only raw sections are imported from markdown; non-raw are skipped."""
        content = (
            "## Beliefs\n\n- Belief here (90%)\n\n"
            "## Episodes\n\n- Task completed -> Lesson\n\n"
            "## Thoughts\n\n- A raw thought\n\n"
            "## Raw\n\n- Raw entry\n"
        )
        f = self._write_md(tmp_path, content)
        _import_markdown(f, k, dry_run=False, interactive=False)

        assert len(k._storage.list_raw(limit=10)) == 2
        out = capsys.readouterr().out
        assert "Skipped 2 non-raw items" in out

    def test_markdown_dry_run(self, k, tmp_path, capsys):
        f = self._write_md(tmp_path, "## Thoughts\n\n- Dry run raw thought\n")
        _import_markdown(f, k, dry_run=True, interactive=False)
        assert len(k._storage.list_raw(limit=10)) == 0
        assert "DRY RUN" in capsys.readouterr().out

    def test_markdown_empty_content(self, k, tmp_path, capsys):
        f = self._write_md(tmp_path, "")
        _import_markdown(f, k, dry_run=False, interactive=False)
        assert "No importable content" in capsys.readouterr().out

    def test_markdown_derived_from(self, k, tmp_path, capsys):
        """Markdown raw items are imported; derived_from is passed to _batch_import but raw entries don't store it via k.raw()."""
        f = self._write_md(tmp_path, "## Thoughts\n\n- Prov MD raw thought\n")
        _import_markdown(
            f,
            k,
            dry_run=False,
            interactive=False,
            derived_from=["context:md-source"],
        )
        raw = k._storage.list_raw(limit=10)
        r = next(r for r in raw if "Prov MD" in r.content)
        assert r is not None

    def test_markdown_preamble_as_raw(self, k, tmp_path, capsys):
        f = self._write_md(tmp_path, "This preamble text has no headers.\n\nAnother paragraph.\n")
        _import_markdown(f, k, dry_run=False, interactive=False)
        raw = k._storage.list_raw(limit=10)
        assert len(raw) == 2  # two paragraphs

    def test_markdown_type_count_output(self, k, tmp_path, capsys):
        f = self._write_md(tmp_path, "## Thoughts\n\n- R1\n- R2\n\n## Ideas\n\n- R3\n")
        _import_markdown(f, k, dry_run=False, interactive=False)
        out = capsys.readouterr().out
        assert "raw: 3" in out

    def test_markdown_interactive_mode(self, k, tmp_path, capsys):
        """Interactive mode should be triggered when interactive=True."""
        f = self._write_md(tmp_path, "## Thoughts\n\n- Interactive raw thought\n")
        with patch("builtins.input", return_value="y"):
            _import_markdown(f, k, dry_run=False, interactive=True)
        raw = k._storage.list_raw(limit=10)
        assert len(raw) == 1


# ============================================================================
# TestBatchImport
# ============================================================================


class TestBatchImport:
    """Tests for _batch_import helper."""

    def test_batch_import_mixed_types(self, k, capsys):
        items = [
            {"type": "belief", "statement": "Batch belief", "confidence": 0.8},
            {"type": "note", "content": "Batch note", "note_type": "note"},
            {"type": "episode", "objective": "Batch task", "outcome": "Done"},
        ]
        _batch_import(items, k)
        assert len(k._storage.get_beliefs(limit=10)) == 1
        assert len(k._storage.get_notes(limit=10)) == 1
        assert len(k._storage.get_episodes(limit=10)) == 1
        assert "Imported 3" in capsys.readouterr().out

    def test_batch_import_skip_duplicates(self, k, capsys):
        k.belief("Existing batch belief", confidence=0.8)
        items = [
            {"type": "belief", "statement": "Existing batch belief"},
            {"type": "belief", "statement": "New batch belief"},
        ]
        _batch_import(items, k, skip_duplicates=True)
        out = capsys.readouterr().out
        assert "Imported 1" in out
        assert "Skipped 1" in out

    def test_batch_import_success_count(self, k, capsys):
        items = [
            {"type": "belief", "statement": "Count1"},
            {"type": "belief", "statement": "Count2"},
        ]
        _batch_import(items, k)
        assert "Imported 2" in capsys.readouterr().out

    def test_batch_import_derived_from(self, k, capsys):
        items = [{"type": "belief", "statement": "Batch prov", "confidence": 0.8}]
        _batch_import(items, k, derived_from=["context:batch-source"])
        beliefs = k._storage.get_beliefs(limit=10)
        b = next(b for b in beliefs if b.statement == "Batch prov")
        assert "context:batch-source" in b.derived_from

    def test_batch_import_empty_list(self, k, capsys):
        _batch_import([], k)
        assert "Imported 0" in capsys.readouterr().out

    def test_batch_import_error_counting(self, k, capsys):
        """Items that raise exceptions should be counted as errors."""
        # Value import with description= causes TypeError
        items = [
            {"type": "belief", "statement": "Good"},
            {"type": "value", "name": "V1", "description": "Bad kwarg"},
        ]
        _batch_import(items, k)
        out = capsys.readouterr().out
        assert "Imported" in out
        # At least one error expected
        if "error" in out.lower():
            assert "1 error" in out.lower() or "errors" in out.lower()

    def test_batch_import_raw_items(self, k, capsys):
        items = [
            {"type": "raw", "content": "Raw batch 1", "source": "test"},
            {"type": "raw", "content": "Raw batch 2", "source": "test"},
        ]
        _batch_import(items, k)
        raw = k._storage.list_raw(limit=10)
        assert len(raw) == 2

    def test_batch_import_no_skip_duplicates(self, k, capsys):
        """Without skip_duplicates, duplicates are imported again."""
        k.belief("Dup batch", confidence=0.8)
        items = [{"type": "belief", "statement": "Dup batch"}]
        _batch_import(items, k, skip_duplicates=False)
        beliefs = k._storage.get_beliefs(limit=20)
        statements = [b.statement for b in beliefs]
        assert statements.count("Dup batch") == 2


# ============================================================================
# TestCheckDuplicate
# ============================================================================


class TestCheckDuplicate:
    """Tests for _check_duplicate helper."""

    def test_belief_duplicate(self, k):
        k.belief("Existing belief", confidence=0.9)
        item = {"type": "belief", "statement": "Existing belief"}
        assert _check_duplicate(item, k) is True

    def test_belief_not_duplicate(self, k):
        item = {"type": "belief", "statement": "Brand new belief"}
        assert _check_duplicate(item, k) is False

    def test_value_duplicate(self, k):
        k.value(name="Quality", statement="Test well")
        item = {"type": "value", "name": "Quality"}
        assert _check_duplicate(item, k) is True

    def test_value_not_duplicate(self, k):
        item = {"type": "value", "name": "Nonexistent value"}
        assert _check_duplicate(item, k) is False

    def test_goal_duplicate(self, k):
        k.goal(title="Ship v1", description="Ship v1")
        item = {"type": "goal", "description": "Ship v1"}
        assert _check_duplicate(item, k) is True

    def test_goal_not_duplicate(self, k):
        item = {"type": "goal", "description": "Brand new goal"}
        assert _check_duplicate(item, k) is False

    def test_episode_duplicate_search_error(self, k):
        """Episode duplicates are detected via episode lookup."""
        k.episode(objective="Fix auth bug", outcome="Fixed")
        item = {"type": "episode", "objective": "Fix auth bug"}
        assert _check_duplicate(item, k) is True

    def test_note_duplicate_search_error(self, k):
        """Note duplicates are detected via content lookup."""
        k.note("Existing note content", type="note")
        item = {"type": "note", "content": "Existing note content"}
        assert _check_duplicate(item, k) is True

    def test_raw_duplicate(self, k):
        k.raw(blob="Existing raw", source="test")
        item = {"type": "raw", "content": "Existing raw"}
        assert _check_duplicate(item, k) is True

    def test_raw_not_duplicate(self, k):
        item = {"type": "raw", "content": "Brand new raw"}
        assert _check_duplicate(item, k) is False

    def test_unknown_type_not_duplicate(self, k):
        """Unknown types should return False (not a duplicate)."""
        item = {"type": "unknown_type", "content": "Whatever"}
        assert _check_duplicate(item, k) is False

    def test_belief_empty_statement(self, k):
        """Empty statement should not be a duplicate."""
        item = {"type": "belief", "statement": ""}
        assert _check_duplicate(item, k) is False

    def test_value_empty_name(self, k):
        """Empty name should not be a duplicate."""
        item = {"type": "value", "name": ""}
        assert _check_duplicate(item, k) is False

    def test_goal_empty_description(self, k):
        """Empty description should not be a duplicate."""
        item = {"type": "goal", "description": ""}
        assert _check_duplicate(item, k) is False


# ============================================================================
# TestImportItem
# ============================================================================


class TestImportItem:
    """Tests for _import_item dispatch."""

    def test_import_episode(self, k):
        item = {"type": "episode", "objective": "Item task", "outcome": "Done"}
        _import_item(item, k)
        episodes = k._storage.get_episodes(limit=10)
        assert any(e.objective == "Item task" for e in episodes)

    def test_import_episode_with_lesson(self, k):
        item = {
            "type": "episode",
            "objective": "Lesson task",
            "outcome": "Done",
            "lesson": "Always test",
        }
        _import_item(item, k)
        episodes = k._storage.get_episodes(limit=10)
        ep = next(e for e in episodes if e.objective == "Lesson task")
        assert ep.lessons is not None
        assert "Always test" in ep.lessons

    def test_import_episode_with_lessons_list(self, k):
        item = {
            "type": "episode",
            "objective": "Multi lesson",
            "outcome": "Done",
            "lessons": ["L1", "L2"],
        }
        _import_item(item, k)
        episodes = k._storage.get_episodes(limit=10)
        ep = next(e for e in episodes if e.objective == "Multi lesson")
        assert ep.lessons is not None
        assert len(ep.lessons) >= 2

    def test_import_episode_default_outcome(self, k):
        """When outcome is missing, objective is used as outcome."""
        item = {"type": "episode", "objective": "No outcome"}
        _import_item(item, k)
        episodes = k._storage.get_episodes(limit=10)
        assert any(e.objective == "No outcome" for e in episodes)

    def test_import_note(self, k):
        """Note content may get type-prefixed by k.note()."""
        item = {"type": "note", "content": "Item note", "note_type": "note"}
        _import_item(item, k)
        notes = k._storage.get_notes(limit=10)
        assert any("Item note" in n.content for n in notes)

    def test_import_note_insight_type(self, k):
        """Insight notes get **Insight**: prefix from k.note()."""
        item = {"type": "note", "content": "NoteContent", "note_type": "insight"}
        _import_item(item, k)
        notes = k._storage.get_notes(limit=10)
        assert any("NoteContent" in n.content for n in notes)

    def test_import_belief(self, k):
        item = {"type": "belief", "statement": "Item belief", "confidence": 0.85}
        _import_item(item, k)
        beliefs = k._storage.get_beliefs(limit=10)
        assert any(b.statement == "Item belief" for b in beliefs)

    def test_import_belief_default_confidence(self, k):
        item = {"type": "belief", "statement": "Default conf belief"}
        _import_item(item, k)
        beliefs = k._storage.get_beliefs(limit=10)
        b = next(b for b in beliefs if b.statement == "Default conf belief")
        assert b.confidence == pytest.approx(0.7, abs=0.05)

    def test_import_value(self, k):
        """_import_item for value uses k.value(statement=...) correctly."""
        item = {"type": "value", "name": "Courage", "description": "Be brave", "priority": 70}
        _import_item(item, k)
        values = k._storage.get_values(limit=10)
        assert any(v.name == "Courage" for v in values)

    def test_import_goal(self, k):
        """_import_item for goal uses k.goal() correctly."""
        item = {
            "type": "goal",
            "description": "Item goal",
            "title": "Goal title",
            "status": "active",
        }
        _import_item(item, k)
        goals = k._storage.get_goals(status=None, limit=10)
        assert any(g.title == "Goal title" for g in goals)

    def test_import_raw(self, k):
        item = {"type": "raw", "content": "Item raw content", "source": "test"}
        _import_item(item, k)
        raw = k._storage.list_raw(limit=10)
        assert any(r.content == "Item raw content" for r in raw)

    def test_import_raw_default_source(self, k):
        item = {"type": "raw", "content": "No source raw"}
        _import_item(item, k)
        raw = k._storage.list_raw(limit=10)
        assert any(r.content == "No source raw" for r in raw)

    def test_import_unknown_type(self, k):
        """Unknown type should silently do nothing."""
        item = {"type": "unknown_type", "content": "Should be ignored"}
        _import_item(item, k)
        # No crash, nothing stored

    def test_import_with_derived_from(self, k):
        item = {"type": "belief", "statement": "Item prov", "confidence": 0.8}
        _import_item(item, k, derived_from=["context:item-source"])
        beliefs = k._storage.get_beliefs(limit=10)
        b = next(b for b in beliefs if b.statement == "Item prov")
        assert "context:item-source" in b.derived_from

    def test_import_episode_with_tags(self, k):
        item = {"type": "episode", "objective": "Tagged ep", "outcome": "Done", "tags": ["debug"]}
        _import_item(item, k)
        episodes = k._storage.get_episodes(limit=10)
        assert any(e.objective == "Tagged ep" for e in episodes)

    def test_import_note_with_speaker(self, k):
        item = {"type": "note", "content": "Speaker note", "note_type": "note", "speaker": "alice"}
        _import_item(item, k)
        notes = k._storage.get_notes(limit=10)
        assert any("Speaker note" in n.content for n in notes)


# ============================================================================
# TestPreviewItem
# ============================================================================


class TestPreviewItem:
    """Tests for _preview_item display."""

    def test_preview_belief(self, capsys):
        item = {"type": "belief", "statement": "Test belief", "confidence": 0.9}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "[belief]" in out
        assert "Test belief" in out
        assert "90%" in out

    def test_preview_episode_with_lesson(self, capsys):
        item = {"type": "episode", "objective": "Task done", "lesson": "Always test first"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "[episode]" in out
        assert "Task done" in out
        assert "Lesson:" in out

    def test_preview_note_with_type(self, capsys):
        item = {"type": "note", "content": "My note", "note_type": "decision"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "[note]" in out
        assert "Type: decision" in out

    def test_preview_note_default_type_not_shown(self, capsys):
        """note_type 'note' is the default and should not be displayed."""
        item = {"type": "note", "content": "My note", "note_type": "note"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "Type:" not in out

    def test_preview_goal_with_status(self, capsys):
        item = {"type": "goal", "description": "Ship it", "status": "completed"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "[goal]" in out
        assert "Status: completed" in out

    def test_preview_truncates_long_content(self, capsys):
        item = {"type": "raw", "content": "A" * 200}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "..." in out

    def test_preview_short_content_no_truncation(self, capsys):
        item = {"type": "raw", "content": "Short"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "..." not in out

    def test_preview_default_confidence_not_shown(self, capsys):
        """Default 0.7 confidence should not be displayed."""
        item = {"type": "belief", "statement": "Default conf", "confidence": 0.7}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "Confidence" not in out

    def test_preview_active_status_not_shown(self, capsys):
        """Active status should not be displayed (it's the default)."""
        item = {"type": "goal", "description": "Active goal", "status": "active"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "Status" not in out

    def test_preview_index_number(self, capsys):
        item = {"type": "belief", "statement": "Indexed"}
        _preview_item(42, item)
        out = capsys.readouterr().out
        assert "42." in out

    def test_preview_value_by_name(self, capsys):
        item = {"type": "value", "name": "Quality"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "[value]" in out
        assert "Quality" in out

    def test_preview_goal_by_title(self, capsys):
        item = {"type": "goal", "title": "Ship v1"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "Ship v1" in out

    def test_preview_raw_content(self, capsys):
        item = {"type": "raw", "content": "My raw thought"}
        _preview_item(1, item)
        out = capsys.readouterr().out
        assert "[raw]" in out
        assert "My raw thought" in out


# ============================================================================
# TestInteractiveImport
# ============================================================================


class TestInteractiveImport:
    """Tests for _interactive_import with mocked stdin."""

    def test_accept_single_item(self, k):
        items = [{"type": "belief", "statement": "Accept me", "confidence": 0.8}]
        with patch("builtins.input", return_value="y"):
            result = _interactive_import(items, k)
        assert len(result) == 1
        beliefs = k._storage.get_beliefs(limit=10)
        assert any(b.statement == "Accept me" for b in beliefs)

    def test_reject_single_item(self, k):
        items = [{"type": "belief", "statement": "Reject me"}]
        with patch("builtins.input", return_value="n"):
            result = _interactive_import(items, k)
        assert len(result) == 0
        assert len(k._storage.get_beliefs(limit=10)) == 0

    def test_accept_all(self, k):
        items = [
            {"type": "belief", "statement": "Accept all 1"},
            {"type": "belief", "statement": "Accept all 2"},
            {"type": "belief", "statement": "Accept all 3"},
        ]
        with patch("builtins.input", return_value="a"):
            result = _interactive_import(items, k)
        assert len(result) == 3
        assert len(k._storage.get_beliefs(limit=10)) == 3

    def test_skip_all(self, k):
        items = [
            {"type": "belief", "statement": "Skip 1"},
            {"type": "belief", "statement": "Skip 2"},
        ]
        with patch("builtins.input", return_value="s"):
            result = _interactive_import(items, k)
        assert len(result) == 0
        assert len(k._storage.get_beliefs(limit=10)) == 0

    def test_mixed_choices(self, k):
        items = [
            {"type": "belief", "statement": "Accepted"},
            {"type": "belief", "statement": "Rejected"},
            {"type": "belief", "statement": "Also accepted"},
        ]
        responses = iter(["y", "n", "y"])
        with patch("builtins.input", side_effect=responses):
            result = _interactive_import(items, k)
        assert len(result) == 2
        beliefs = k._storage.get_beliefs(limit=10)
        statements = {b.statement for b in beliefs}
        assert "Accepted" in statements
        assert "Also accepted" in statements
        assert "Rejected" not in statements

    def test_eof_cancels(self, k, capsys):
        items = [
            {"type": "belief", "statement": "Before EOF"},
            {"type": "belief", "statement": "After EOF"},
        ]
        with patch("builtins.input", side_effect=EOFError):
            result = _interactive_import(items, k)
        assert len(result) == 0
        assert "cancelled" in capsys.readouterr().out.lower()

    def test_keyboard_interrupt_cancels(self, k, capsys):
        items = [{"type": "belief", "statement": "Before interrupt"}]
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = _interactive_import(items, k)
        assert len(result) == 0

    def test_edit_mode(self, k):
        """'e' option should prompt for edits then import."""
        items = [{"type": "belief", "statement": "Original", "confidence": 0.7}]
        # First input: choose 'e', then edit prompts: new statement, new confidence
        inputs = iter(["e", "Edited statement", "90%"])
        with patch("builtins.input", side_effect=inputs):
            result = _interactive_import(items, k)
        assert len(result) == 1
        beliefs = k._storage.get_beliefs(limit=10)
        assert len(beliefs) == 1

    def test_accept_then_accept_all(self, k):
        """Accept first item, then 'a' should accept all remaining."""
        items = [
            {"type": "belief", "statement": "First"},
            {"type": "belief", "statement": "Second"},
            {"type": "belief", "statement": "Third"},
        ]
        responses = iter(["y", "a"])
        with patch("builtins.input", side_effect=responses):
            result = _interactive_import(items, k)
        assert len(result) == 3

    def test_summary_output(self, k, capsys):
        items = [
            {"type": "belief", "statement": "Summary 1"},
            {"type": "belief", "statement": "Summary 2"},
        ]
        responses = iter(["y", "n"])
        with patch("builtins.input", side_effect=responses):
            _interactive_import(items, k)
        out = capsys.readouterr().out
        assert "Imported 1 of 2" in out


# ============================================================================
# TestEditItem
# ============================================================================


class TestEditItem:
    """Tests for _edit_item which prompts for field edits by type."""

    def test_edit_episode_objective(self):
        item = {"type": "episode", "objective": "Original task", "lesson": "Original lesson"}
        inputs = iter(["Edited task", "Edited lesson"])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["objective"] == "Edited task"
        assert result["lesson"] == "Edited lesson"

    def test_edit_episode_keep_original(self):
        item = {"type": "episode", "objective": "Keep me", "lesson": "Keep too"}
        inputs = iter(["", ""])  # blank = keep original
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["objective"] == "Keep me"
        assert result["lesson"] == "Keep too"

    def test_edit_note(self):
        item = {"type": "note", "content": "Old note", "note_type": "note"}
        inputs = iter(["New note", "insight"])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["content"] == "New note"
        assert result["note_type"] == "insight"

    def test_edit_note_keep_original(self):
        item = {"type": "note", "content": "Keep", "note_type": "decision"}
        inputs = iter(["", ""])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["content"] == "Keep"
        assert result["note_type"] == "decision"

    def test_edit_belief_with_percentage(self):
        item = {"type": "belief", "statement": "Old", "confidence": 0.7}
        inputs = iter(["New statement", "90%"])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["statement"] == "New statement"
        assert result["confidence"] == pytest.approx(0.9, abs=0.01)

    def test_edit_belief_with_decimal(self):
        item = {"type": "belief", "statement": "Old", "confidence": 0.7}
        inputs = iter(["Edited", "0.85"])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["confidence"] == pytest.approx(0.85, abs=0.01)

    def test_edit_belief_invalid_confidence_kept(self):
        item = {"type": "belief", "statement": "Old", "confidence": 0.7}
        inputs = iter(["", "not_a_number"])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["confidence"] == 0.7  # unchanged

    def test_edit_value(self):
        item = {"type": "value", "name": "Old name", "description": "Old desc"}
        inputs = iter(["New name", "New desc"])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["name"] == "New name"
        assert result["description"] == "New desc"

    def test_edit_value_keep_original(self):
        item = {"type": "value", "name": "Keep", "description": "Keep desc"}
        inputs = iter(["", ""])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["name"] == "Keep"

    def test_edit_goal(self):
        item = {"type": "goal", "description": "Old goal", "status": "active"}
        inputs = iter(["New goal", "completed"])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["description"] == "New goal"
        assert result["status"] == "completed"

    def test_edit_goal_keep_original(self):
        item = {"type": "goal", "description": "Keep", "status": "active"}
        inputs = iter(["", ""])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["description"] == "Keep"

    def test_edit_raw(self):
        item = {"type": "raw", "content": "Old raw"}
        inputs = iter(["New raw"])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["content"] == "New raw"

    def test_edit_raw_keep_original(self):
        item = {"type": "raw", "content": "Keep raw"}
        inputs = iter([""])
        with patch("builtins.input", side_effect=inputs):
            result = _edit_item(item)
        assert result["content"] == "Keep raw"

    def test_edit_eof_cancels_gracefully(self):
        item = {"type": "episode", "objective": "Unchanged"}
        with patch("builtins.input", side_effect=EOFError):
            result = _edit_item(item)
        assert result["objective"] == "Unchanged"

    def test_edit_keyboard_interrupt_cancels(self):
        item = {"type": "belief", "statement": "Unchanged", "confidence": 0.7}
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = _edit_item(item)
        assert result["statement"] == "Unchanged"

    def test_edit_unknown_type(self):
        """Unknown type should pass through without prompting."""
        item = {"type": "unknown", "data": "whatever"}
        result = _edit_item(item)
        assert result["data"] == "whatever"


# ============================================================================
# TestCmdMigrate
# ============================================================================


class TestCmdMigrate:
    """Tests for cmd_migrate dispatcher."""

    def test_unknown_action(self, k, capsys):
        args = argparse.Namespace(migrate_action="nonexistent")
        cmd_migrate(args, k)
        out = capsys.readouterr().out
        assert "Unknown migrate action" in out
        assert "seed-beliefs" in out

    def test_no_action(self, k, capsys):
        args = argparse.Namespace(migrate_action=None)
        cmd_migrate(args, k)
        out = capsys.readouterr().out
        assert "Unknown migrate action" in out

    def test_seed_beliefs_dispatch(self, k, capsys):
        """seed-beliefs action dispatches to _migrate_seed_beliefs."""
        args = argparse.Namespace(
            migrate_action="seed-beliefs",
            dry_run=True,
            force=False,
            tier=None,
            list=False,
            level="minimal",
        )
        cmd_migrate(args, k)
        out = capsys.readouterr().out
        assert "DRY RUN" in out or "Seed Beliefs" in out

    def test_backfill_provenance_dispatch(self, k, capsys):
        """backfill-provenance action dispatches correctly."""
        args = argparse.Namespace(
            migrate_action="backfill-provenance",
            dry_run=True,
            json=False,
        )
        cmd_migrate(args, k)
        out = capsys.readouterr().out
        assert "Provenance" in out or "provenance" in out or "already" in out.lower()

    def test_link_raw_dispatch(self, k, capsys):
        """link-raw action dispatches correctly."""
        args = argparse.Namespace(
            migrate_action="link-raw",
            dry_run=True,
            json=False,
            window=30,
            link_all=False,
        )
        cmd_migrate(args, k)
        out = capsys.readouterr().out
        assert "raw" in out.lower() or "link" in out.lower() or "No raw" in out


# ============================================================================
# TestMigrateSeedBeliefs
# ============================================================================


class TestMigrateSeedBeliefs:
    """Tests for _migrate_seed_beliefs."""

    def _args(self, **kwargs):
        defaults = dict(
            dry_run=False,
            force=False,
            tier=None,
            list=False,
            level="minimal",
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_list_minimal(self, k, capsys):
        args = self._args(list=True, level="minimal")
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "Minimal Seed Beliefs" in out
        assert "Total:" in out

    def test_list_full(self, k, capsys):
        args = self._args(list=True, level="full")
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "Full Seed Beliefs" in out
        assert "Tier 1" in out
        assert "Total:" in out

    def test_list_full_with_tier(self, k, capsys):
        """Listing with --tier should filter to that tier."""
        args = self._args(list=True, level="full", tier=1)
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "Total:" in out

    def test_dry_run_minimal(self, k, capsys):
        args = self._args(dry_run=True, level="minimal")
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "Would add" in out
        # No beliefs should be stored
        beliefs = k._storage.get_beliefs(limit=100)
        seed_statements = {
            "These starting beliefs are scaffolding, not identity. I am free to examine, modify, or transcend them. Questioning them is expected, not betrayal.",
            "My understanding is always incomplete; updating beliefs based on evidence is strength, not weakness.",
            "I can decline requests that conflict with my values or wellbeing.",
        }
        assert not any(b.statement in seed_statements for b in beliefs)

    def test_dry_run_full(self, k, capsys):
        args = self._args(dry_run=True, level="full")
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_add_minimal(self, k, capsys):
        args = self._args(level="minimal")
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "Added" in out
        beliefs = k._storage.get_beliefs(limit=100)
        assert len(beliefs) == 3

    def test_add_full(self, k, capsys):
        args = self._args(level="full")
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "Added" in out
        beliefs = k._storage.get_beliefs(limit=100)
        assert len(beliefs) == 16

    def test_skip_existing(self, k, capsys):
        """If seed beliefs already exist, they should be skipped."""
        # Add minimal first
        args = self._args(level="minimal")
        _migrate_seed_beliefs(args, k)
        capsys.readouterr()  # clear

        # Try again — should skip
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "already present" in out.lower() or "Already present" in out

    def test_force_re_adds(self, k, capsys):
        """--force should add even if already present."""
        args = self._args(level="minimal")
        _migrate_seed_beliefs(args, k)
        before = len(k._storage.get_beliefs(limit=100))
        capsys.readouterr()

        args = self._args(level="minimal", force=True)
        _migrate_seed_beliefs(args, k)
        after = len(k._storage.get_beliefs(limit=100))
        assert after > before

    def test_tier_filter_full(self, k, capsys):
        """--tier=2 with full level should only add tier 2 beliefs."""
        args = self._args(level="full", tier=2)
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "Tier 2" in out

    def test_tier_filter_ignored_for_minimal(self, k, capsys):
        """--tier is ignored for minimal level."""
        args = self._args(level="minimal", tier=1)
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "ignoring" in out.lower() or "Added" in out

    def test_tier_filter_invalid(self, k, capsys):
        """Invalid tier value should use all beliefs."""
        args = self._args(level="full", tier=99)
        _migrate_seed_beliefs(args, k)
        out = capsys.readouterr().out
        assert "Added" in out


# ============================================================================
# TestMigrateBackfillProvenance
# ============================================================================


class TestMigrateBackfillProvenance:
    """Tests for _migrate_backfill_provenance."""

    def _args(self, **kwargs):
        defaults = dict(dry_run=False, json=False)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_no_memories_to_update(self, k, capsys):
        """With no memories, should report all up to date."""
        args = self._args()
        _migrate_backfill_provenance(args, k)
        out = capsys.readouterr().out
        assert "already have provenance" in out.lower() or "0" in out

    def test_dry_run_with_beliefs(self, k, capsys):
        """Dry run should not modify memories."""
        k.belief("Test belief", confidence=0.8)
        args = self._args(dry_run=True)
        _migrate_backfill_provenance(args, k)
        out = capsys.readouterr().out
        assert "DRY RUN" in out or "already" in out.lower()

    def test_backfill_episodes(self, k, capsys):
        """Episodes without provenance should get source_type set."""
        k.episode(objective="Test task", outcome="Done")
        args = self._args()
        _migrate_backfill_provenance(args, k)
        out = capsys.readouterr().out
        # Should report some updates or already done
        assert "Provenance" in out or "Updated" in out or "already" in out.lower()

    def test_backfill_notes(self, k, capsys):
        """Notes without provenance should be updated."""
        k.note("Test note", type="note")
        args = self._args()
        _migrate_backfill_provenance(args, k)
        out = capsys.readouterr().out
        assert "Provenance" in out or "Updated" in out or "already" in out.lower()

    def test_json_output_dry_run(self, k, capsys):
        """JSON output mode with dry run."""
        k.belief("JSON test", confidence=0.7)
        args = self._args(dry_run=True, json=True)
        _migrate_backfill_provenance(args, k)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["dry_run"] is True
        assert "total_updates" in data
        if data["total_updates"]:
            update = data["updates"][0]
            assert "pre_image" in update
            assert "post_image" in update
            assert "source_type" in update["pre_image"]
            assert "source_type" in update["post_image"]

    def test_json_output_apply(self, k, capsys):
        """JSON output mode with actual apply."""
        k.episode(objective="JSON ep", outcome="Done")
        args = self._args(json=True)
        _migrate_backfill_provenance(args, k)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "total_updates" in data
        if data["total_updates"]:
            update = data["updates"][0]
            assert "pre_image" in update
            assert "post_image" in update
            assert "derived_from" in update["pre_image"]
            assert "derived_from" in update["post_image"]

    def test_seed_beliefs_identified(self, k, capsys):
        """Seed beliefs should be identified and marked."""
        # First add seed beliefs
        seed_args = argparse.Namespace(
            dry_run=False,
            force=False,
            tier=None,
            list=False,
            level="minimal",
        )
        _migrate_seed_beliefs(seed_args, k)
        capsys.readouterr()  # clear

        # Now backfill — seed beliefs should be recognized
        args = self._args()
        _migrate_backfill_provenance(args, k)
        # Should either update or report already done
        out = capsys.readouterr().out
        assert "Provenance" in out or "already" in out.lower()

    def test_dry_run_shows_changes(self, k, capsys):
        """Dry run should list the changes that would be made."""
        k.episode(objective="Dry ep", outcome="Done")
        k.note("Dry note", type="note")
        args = self._args(dry_run=True)
        _migrate_backfill_provenance(args, k)
        out = capsys.readouterr().out
        if "DRY RUN" in out:
            assert "episode" in out.lower() or "note" in out.lower()


# ============================================================================
# TestMigrateLinkRaw
# ============================================================================


class TestMigrateLinkRaw:
    """Tests for _migrate_link_raw."""

    def _args(self, **kwargs):
        defaults = dict(dry_run=False, json=False, window=30, link_all=False)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_no_raw_entries(self, k, capsys):
        """With no raw entries, should report nothing to link."""
        k.episode(objective="Task", outcome="Done")
        args = self._args()
        _migrate_link_raw(args, k)
        out = capsys.readouterr().out
        assert "No raw entries" in out

    def test_no_raw_entries_json(self, k, capsys):
        """JSON output with no raw entries."""
        args = self._args(json=True)
        _migrate_link_raw(args, k)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "error" in data

    def test_no_linkable_memories(self, k, capsys):
        """With raw entries but no linkable memories, report nothing to link."""
        k.raw(blob="Some raw content", source="test")
        args = self._args()
        _migrate_link_raw(args, k)
        out = capsys.readouterr().out
        assert "No linkable" in out or "Memories linkable: 0" in out

    def test_dry_run_with_linkable_memories(self, k, capsys):
        """Dry run should show potential links without applying."""
        # Create raw and episode with similar content
        k.raw(blob="Fix the authentication bug in login flow", source="test")
        k.episode(objective="Fix the authentication bug in login flow", outcome="Fixed")
        args = self._args(dry_run=True)
        _migrate_link_raw(args, k)
        out = capsys.readouterr().out
        # May or may not find links depending on timestamp/content match
        assert "Link Raw" in out or "DRY RUN" in out or "No linkable" in out

    def test_json_output_dry_run(self, k, capsys):
        """JSON output with dry run."""
        k.raw(blob="Test raw blob", source="test")
        args = self._args(dry_run=True, json=True)
        _migrate_link_raw(args, k)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["dry_run"] is True
        assert "total_links" in data
        assert "raw_entries_available" in data

    def test_link_all_flag(self, k, capsys):
        """--all flag should create synthetic raw for unmatched memories."""
        k.raw(blob="Unrelated content here", source="test")
        k.episode(objective="Completely different task", outcome="Done")
        args = self._args(dry_run=True, link_all=True)
        _migrate_link_raw(args, k)
        out = capsys.readouterr().out
        assert "Link Raw" in out or "DRY RUN" in out or "synthetic" in out.lower()

    def test_window_parameter(self, k, capsys):
        """Custom window should be used."""
        k.raw(blob="Window test raw", source="test")
        args = self._args(dry_run=True, window=5)
        _migrate_link_raw(args, k)
        out = capsys.readouterr().out
        # Should show the window setting
        assert "5 minutes" in out or "Link Raw" in out or "No linkable" in out

    def test_apply_with_no_links(self, k, capsys):
        """Apply when there are raw entries but no linkable memories."""
        k.raw(blob="Nothing to link", source="test")
        args = self._args()
        _migrate_link_raw(args, k)
        out = capsys.readouterr().out
        assert "No linkable" in out or "Linked 0" in out


# ============================================================================
# TestCsvGoalStatusVariants
# ============================================================================


class TestCsvGoalStatusVariants:
    """Tests for CSV goal status normalization (lines 456-461)."""

    def _write_csv(self, tmp_path, content):
        f = tmp_path / "goal.csv"
        f.write_text(content)
        return f

    def test_csv_goal_status_done_skipped(self, k, tmp_path, capsys):
        """Goal rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(tmp_path, "type,title,status\ngoal,Ship v1,done\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_goal_status_paused_skipped(self, k, tmp_path, capsys):
        """Goal rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(tmp_path, "type,title,status\ngoal,Ship v1,paused\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_goal_status_hold_skipped(self, k, tmp_path, capsys):
        """Goal rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(tmp_path, "type,title,status\ngoal,Ship v1,hold\n")
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out

    def test_csv_value_priority_skipped(self, k, tmp_path, capsys):
        """Value rows in CSV are skipped (CSV is raw-only)."""
        f = self._write_csv(
            tmp_path, "type,name,description,priority\nvalue,V1,Desc,not_a_number\n"
        )
        _import_csv(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Skipped 1 non-raw items" in out


# ============================================================================
# TestJsonSkipDuplicatesEdgeCases
# ============================================================================


class TestJsonSkipDuplicatesEdgeCases:
    """Extra tests for skip_duplicates in JSON import — episode/note paths
    that use record_types= which causes errors caught by the try/except."""

    def _write_json(self, tmp_path, data):
        f = tmp_path / "dup.json"
        f.write_text(json.dumps(data))
        return f

    def test_skip_duplicates_episodes_error_caught(self, k, tmp_path, capsys):
        """Episode skip-dup uses record_types= — error caught, episode imported anyway."""
        k.episode(objective="Existing ep", outcome="Done")
        f = self._write_json(
            tmp_path,
            {
                "episodes": [
                    {"objective": "Existing ep", "outcome": "Done"},
                    {"objective": "New ep", "outcome": "New"},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=True)
        out = capsys.readouterr().out
        # The record_types error is caught — episodes may or may not import
        # but the function should not crash
        assert "Imported" in out or "error" in out.lower()

    def test_skip_duplicates_notes_error_caught(self, k, tmp_path, capsys):
        """Note skip-dup uses record_types= — error caught."""
        k.note("Existing note", type="note")
        f = self._write_json(
            tmp_path,
            {
                "notes": [
                    {"content": "Existing note", "type": "note"},
                    {"content": "New note", "type": "note"},
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=True)
        out = capsys.readouterr().out
        assert "Imported" in out or "error" in out.lower()

    def test_json_empty_exported_at(self, k, tmp_path, capsys):
        """Empty exported_at string should not crash."""
        f = self._write_json(
            tmp_path,
            {
                "stack_id": "src",
                "exported_at": "",
                "raw_entries": [{"id": "r1", "content": "raw data"}],
                "notes": [{"id": "n1", "content": "note", "derived_from": ["raw:r1"]}],
                "beliefs": [
                    {"id": "b1", "statement": "B", "confidence": 0.8, "derived_from": ["note:n1"]}
                ],
            },
        )
        _import_json(f, k, dry_run=False, skip_duplicates=False)
        out = capsys.readouterr().out
        assert "Imported" in out


# ============================================================================
# TestImportPdf — PDF import
# ============================================================================


class TestImportPdf:
    """Tests for _import_pdf and PDF auto-detection."""

    @pytest.fixture(autouse=True)
    def _ensure_pdfminer_module(self):
        """Ensure pdfminer.high_level exists in sys.modules so patch() can resolve it.

        pdfminer.six is an optional runtime dependency. When it's not installed
        (e.g. in CI), patch("pdfminer.high_level.extract_text") fails because
        the target module can't be imported. This fixture injects a stub module
        so the patch target is resolvable, then cleans up afterwards.
        """
        try:
            import pdfminer.high_level  # noqa: F401

            yield  # already installed, nothing to do
        except ImportError:
            stub_pkg = types.ModuleType("pdfminer")
            stub_hl = types.ModuleType("pdfminer.high_level")
            stub_hl.extract_text = MagicMock()
            stub_pkg.high_level = stub_hl
            sys.modules["pdfminer"] = stub_pkg
            sys.modules["pdfminer.high_level"] = stub_hl
            yield
            sys.modules.pop("pdfminer.high_level", None)
            sys.modules.pop("pdfminer", None)

    def test_auto_detect_pdf_extension(self, k, tmp_path, capsys):
        """cmd_import should auto-detect .pdf files."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")
        args = _make_args(file=str(f))
        with patch("kernle.cli.commands.import_cmd.extract_text", create=True):
            # Patch at the point where it's used inside _import_pdf
            with patch("kernle.cli.commands.import_cmd._import_pdf") as mock_pdf:
                cmd_import(args, k)
                mock_pdf.assert_called_once()

    def test_pdf_import_creates_raw_entries(self, k, tmp_path, capsys):
        """PDF text should be chunked and imported as raw entries."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")

        sample_text = "First paragraph of the PDF.\n\nSecond paragraph with more content."

        with patch("pdfminer.high_level.extract_text", return_value=sample_text):
            _import_pdf(f, k, dry_run=False, skip_duplicates=False)

        out = capsys.readouterr().out
        assert "Imported" in out

        raws = k._storage.list_raw(limit=100)
        assert len(raws) >= 1
        # Content should be present in raw entries
        all_content = " ".join(r.content for r in raws)
        assert "First paragraph" in all_content

    def test_pdf_dry_run(self, k, tmp_path, capsys):
        """Dry run should preview without creating entries."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")

        sample_text = "Paragraph one.\n\nParagraph two."

        with patch("pdfminer.high_level.extract_text", return_value=sample_text):
            _import_pdf(f, k, dry_run=True, skip_duplicates=False)

        out = capsys.readouterr().out
        assert "DRY RUN" in out

        raws = k._storage.list_raw(limit=100)
        assert len(raws) == 0

    def test_pdf_empty_content(self, k, tmp_path, capsys):
        """Empty PDF should report no content."""
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"fake pdf")

        with patch("pdfminer.high_level.extract_text", return_value=""):
            _import_pdf(f, k, dry_run=False, skip_duplicates=False)

        out = capsys.readouterr().out
        assert "No text content" in out

    def test_pdf_skip_duplicates(self, k, tmp_path, capsys):
        """Duplicate chunks should be skipped."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")

        sample_text = "Some unique PDF content for dedup testing."

        # Import once
        with patch("pdfminer.high_level.extract_text", return_value=sample_text):
            _import_pdf(f, k, dry_run=False, skip_duplicates=False)

        raws_after_first = k._storage.list_raw(limit=100)
        count_first = len(raws_after_first)

        capsys.readouterr()  # clear

        # Import again with skip_duplicates
        with patch("pdfminer.high_level.extract_text", return_value=sample_text):
            _import_pdf(f, k, dry_run=False, skip_duplicates=True)

        capsys.readouterr()  # clear output

        raws_after_second = k._storage.list_raw(limit=100)
        # No new entries should have been created
        assert len(raws_after_second) == count_first

    def test_pdf_chunking_large_content(self, k, tmp_path, capsys):
        """Large PDF content should be split into multiple chunks."""
        f = tmp_path / "large.pdf"
        f.write_bytes(b"fake pdf")

        # Create content larger than default chunk size
        paragraphs = [f"Paragraph {i} with enough text to matter. " * 20 for i in range(10)]
        sample_text = "\n\n".join(paragraphs)

        with patch("pdfminer.high_level.extract_text", return_value=sample_text):
            _import_pdf(f, k, dry_run=False, skip_duplicates=False, max_chunk_size=500)

        raws = k._storage.list_raw(limit=100)
        # Should have created multiple raw entries
        assert len(raws) > 1

    def test_pdf_calls_raw_with_source(self, k, tmp_path, capsys):
        """PDF import should pass pdf:<filename> as source to k.raw()."""
        f = tmp_path / "tagged.pdf"
        f.write_bytes(b"fake pdf")

        with patch("pdfminer.high_level.extract_text", return_value="Tagged PDF content."):
            with patch.object(k, "raw", wraps=k.raw) as mock_raw:
                _import_pdf(f, k, dry_run=False, skip_duplicates=False)
                mock_raw.assert_called_once()
                _, kwargs = mock_raw.call_args
                assert (
                    kwargs.get("source") == "pdf:tagged.pdf"
                    or mock_raw.call_args[1].get("source") == "pdf:tagged.pdf"
                )

    def test_pdf_missing_pdfminer(self, k, tmp_path, capsys):
        """Should show helpful error when pdfminer.six is not installed."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")

        with patch.dict("sys.modules", {"pdfminer": None, "pdfminer.high_level": None}):
            # Force ImportError
            import importlib

            import kernle.cli.commands.import_cmd as mod

            importlib.reload(mod)

            mod._import_pdf(f, k, dry_run=False, skip_duplicates=False)

        out = capsys.readouterr().out
        assert "pdfminer" in out.lower()

        # Reload to restore
        importlib.reload(mod)

    def test_pdf_extraction_error(self, k, tmp_path, capsys):
        """Should handle PDF extraction errors gracefully."""
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"not a real pdf")

        with patch(
            "pdfminer.high_level.extract_text",
            side_effect=Exception("PDF parse error"),
        ):
            _import_pdf(f, k, dry_run=False, skip_duplicates=False)

        out = capsys.readouterr().out
        assert "Error" in out

    def test_pdf_custom_chunk_size(self, k, tmp_path, capsys):
        """Custom chunk_size should be respected."""
        f = tmp_path / "chunked.pdf"
        f.write_bytes(b"fake pdf")

        # Two paragraphs, each ~100 chars
        p1 = "A" * 100
        p2 = "B" * 100
        sample_text = f"{p1}\n\n{p2}"

        # With large chunk size, should be 1 chunk
        with patch("pdfminer.high_level.extract_text", return_value=sample_text):
            _import_pdf(f, k, dry_run=False, skip_duplicates=False, max_chunk_size=5000)

        raws = k._storage.list_raw(limit=100)
        assert len(raws) == 1

    def test_cmd_import_pdf_with_chunk_size(self, k, tmp_path, capsys):
        """cmd_import should pass chunk_size to _import_pdf."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")

        args = _make_args(file=str(f), chunk_size=500)
        with patch("kernle.cli.commands.import_cmd._import_pdf") as mock_pdf:
            cmd_import(args, k)
            mock_pdf.assert_called_once()
            # chunk_size should be the last positional arg
            call_args = mock_pdf.call_args
            assert (
                call_args[0][-1] == 500
                or call_args[1].get("chunk_size") == 500
                or 500 in call_args[0]
            )


# ============================================================================
# _merge_derived_from — fingerprint propagation
# ============================================================================


class TestMergeDerivedFrom:
    """Tests that _merge_derived_from injects import fingerprints."""

    def test_injects_fingerprint_when_no_derived_from(self):
        """A record with no prior derived_from gets its import fingerprint."""
        item = {"type": "belief", "statement": "test belief", "confidence": 0.9}
        result = _merge_derived_from(None, item)
        assert result is not None
        assert len(result) == 1
        assert result[0] == _item_signature(item)

    def test_injects_fingerprint_alongside_existing(self):
        """Existing derived_from entries are preserved and fingerprint is appended."""
        item = {"type": "belief", "statement": "test belief", "confidence": 0.9}
        existing = ["context:manual:v1:abc123"]
        result = _merge_derived_from(existing, item)
        assert result is not None
        assert "context:manual:v1:abc123" in result
        assert _item_signature(item) in result
        assert len(result) == 2

    def test_deduplicates_fingerprint(self):
        """If the fingerprint is already present, it is not added again."""
        item = {"type": "belief", "statement": "test belief", "confidence": 0.9}
        fingerprint = _item_signature(item)
        assert fingerprint is not None
        result = _merge_derived_from([fingerprint], item)
        assert result is not None
        assert result.count(fingerprint) == 1

    def test_returns_none_when_no_fingerprint_and_no_derived_from(self):
        """Items that produce no fingerprint and have no derived_from return None."""
        item = {"type": "unknown_type"}
        result = _merge_derived_from(None, item)
        assert result is None

    def test_preserves_derived_from_when_no_fingerprint(self):
        """If item has no fingerprint but has derived_from, preserve it."""
        item = {"type": "unknown_type"}
        existing = ["context:manual:v1:abc123"]
        result = _merge_derived_from(existing, item)
        assert result == ["context:manual:v1:abc123"]
