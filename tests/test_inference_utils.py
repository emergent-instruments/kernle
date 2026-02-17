"""Tests for Phase 2: Shared inference contract (inference_utils).

TDD tests for:
- parse_inference_json() parsing and validation
- InferenceResult dataclass
- use_legacy_heuristics flag read/write/defaults
"""

import json
import logging

import pytest

from kernle.core.inference_utils import InferenceResult, parse_inference_json

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logger():
    return logging.getLogger("test_inference_utils")


# ---------------------------------------------------------------------------
# parse_inference_json() tests
# ---------------------------------------------------------------------------


class TestParseInferenceJson:
    def test_parse_valid_json(self, logger):
        """Well-formed response with required fields parsed correctly."""
        raw = json.dumps({"valence": 0.8, "arousal": 0.6, "tags": ["joy"]})
        result = parse_inference_json(
            raw,
            required_fields=["valence", "arousal"],
            fallback={"valence": 0.0, "arousal": 0.0, "tags": []},
            logger=logger,
        )
        assert isinstance(result, InferenceResult)
        assert result.data["valence"] == 0.8
        assert result.data["arousal"] == 0.6
        assert result.data["tags"] == ["joy"]
        assert result.raw == raw
        assert result.fallback_used is False

    def test_parse_malformed_json_returns_fallback(self, logger):
        """Garbage input returns fallback with flag set."""
        fallback = {"valence": 0.0, "arousal": 0.0, "tags": []}
        result = parse_inference_json(
            "not valid json {{{",
            required_fields=["valence"],
            fallback=fallback,
            logger=logger,
        )
        assert result.data == fallback
        assert result.fallback_used is True

    def test_parse_missing_required_fields_returns_fallback(self, logger):
        """Partial JSON missing required fields returns fallback."""
        raw = json.dumps({"valence": 0.8})  # missing "arousal"
        fallback = {"valence": 0.0, "arousal": 0.0}
        result = parse_inference_json(
            raw,
            required_fields=["valence", "arousal"],
            fallback=fallback,
            logger=logger,
        )
        assert result.data == fallback
        assert result.fallback_used is True

    def test_parse_extra_fields_accepted(self, logger):
        """Extra fields beyond required don't cause failure."""
        raw = json.dumps(
            {"valence": 0.5, "arousal": 0.3, "extra_field": "bonus", "tags": ["neutral"]}
        )
        result = parse_inference_json(
            raw,
            required_fields=["valence", "arousal"],
            fallback={"valence": 0.0, "arousal": 0.0},
            logger=logger,
        )
        assert result.data["valence"] == 0.5
        assert result.data["extra_field"] == "bonus"
        assert result.fallback_used is False

    def test_parse_empty_string_returns_fallback(self, logger):
        """Empty string returns fallback."""
        fallback = {"valence": 0.0}
        result = parse_inference_json(
            "", required_fields=["valence"], fallback=fallback, logger=logger
        )
        assert result.data == fallback
        assert result.fallback_used is True

    def test_parse_json_array_returns_fallback(self, logger):
        """JSON array (not object) returns fallback."""
        raw = json.dumps([{"valence": 0.8}])
        fallback = {"valence": 0.0}
        result = parse_inference_json(
            raw, required_fields=["valence"], fallback=fallback, logger=logger
        )
        assert result.data == fallback
        assert result.fallback_used is True

    def test_parse_no_required_fields(self, logger):
        """With empty required_fields, any valid JSON object succeeds."""
        raw = json.dumps({"anything": "works"})
        result = parse_inference_json(raw, required_fields=[], fallback={}, logger=logger)
        assert result.data["anything"] == "works"
        assert result.fallback_used is False

    def test_parse_json_with_markdown_fencing(self, logger):
        """JSON wrapped in markdown code fences is extracted and parsed."""
        raw = '```json\n{"valence": 0.7, "arousal": 0.4}\n```'
        result = parse_inference_json(
            raw,
            required_fields=["valence", "arousal"],
            fallback={"valence": 0.0, "arousal": 0.0},
            logger=logger,
        )
        assert result.data["valence"] == 0.7
        assert result.fallback_used is False


# ---------------------------------------------------------------------------
# use_legacy_heuristics flag tests
# ---------------------------------------------------------------------------


class TestLegacyHeuristicsFlag:
    def test_legacy_heuristics_flag_true(self, tmp_path):
        """Setting 'true' returns True."""
        from kernle.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(stack_id="test-stack", db_path=tmp_path / "test.db")
        try:
            storage.set_stack_setting("use_legacy_heuristics", "true")
            assert storage.get_stack_setting("use_legacy_heuristics") == "true"
        finally:
            storage.close()

    def test_legacy_heuristics_flag_false(self, tmp_path):
        """Setting 'false' returns 'false'."""
        from kernle.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(stack_id="test-stack", db_path=tmp_path / "test.db")
        try:
            storage.set_stack_setting("use_legacy_heuristics", "false")
            assert storage.get_stack_setting("use_legacy_heuristics") == "false"
        finally:
            storage.close()

    def test_legacy_heuristics_flag_missing_defaults_true(self, tmp_path):
        """No setting returns None (callers should default to True/legacy)."""
        from kernle.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(stack_id="test-stack", db_path=tmp_path / "test.db")
        try:
            result = storage.get_stack_setting("use_legacy_heuristics")
            assert result is None  # Callers interpret None as legacy=True
        finally:
            storage.close()

    def test_new_stack_gets_legacy_false(self, tmp_path):
        """Fresh stack creation sets use_legacy_heuristics=false."""
        from kernle.stack.sqlite_stack import SQLiteStack

        stack = SQLiteStack(stack_id="new-stack", db_path=tmp_path / "new.db")
        try:
            setting = stack._backend.get_stack_setting("use_legacy_heuristics")
            assert setting == "false"
        finally:
            stack._backend.close()

    def test_migrated_stack_gets_legacy_true(self, tmp_path):
        """Migration from v26 sets use_legacy_heuristics=true for existing stacks."""
        import sqlite3

        from kernle.storage.schema import migrate_schema

        db_path = tmp_path / "migrate.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Minimal v26 schema
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER);
            INSERT INTO schema_version (version) VALUES (26);

            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                deleted INTEGER DEFAULT 0
            );

            CREATE TABLE memory_audit (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                details TEXT,
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE stack_settings (
                stack_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (stack_id, key)
            );
        """)
        # Insert a setting to indicate this is an existing stack
        conn.execute(
            "INSERT INTO stack_settings VALUES (?, ?, ?, ?)",
            ("test-stack", "some_key", "some_value", "2025-01-01T00:00:00Z"),
        )
        conn.commit()

        migrate_schema(conn, "test-stack")

        row = conn.execute(
            "SELECT value FROM stack_settings WHERE stack_id = ? AND key = ?",
            ("test-stack", "use_legacy_heuristics"),
        ).fetchone()
        assert row is not None
        assert row[0] == "true"
        conn.close()

    def test_existing_stack_with_notes_only_gets_legacy_true(self, tmp_path):
        """Stack with notes but no episodes should default to legacy=true."""
        import sqlite3

        from kernle.stack.sqlite_stack import SQLiteStack
        from kernle.types import Note

        # Create a stack and add a note
        stack = SQLiteStack(
            stack_id="notes-only",
            db_path=tmp_path / "notes.db",
            components=[],
        )
        note = Note(
            id="n-1",
            stack_id="notes-only",
            content="an important note",
            note_type="observation",
        )
        stack._backend.save_note(note)
        stack._backend.close()

        # Clear the setting via raw sqlite to simulate missing migration
        conn = sqlite3.connect(str(tmp_path / "notes.db"))
        conn.execute("DELETE FROM stack_settings WHERE key = 'use_legacy_heuristics'")
        conn.commit()
        conn.close()

        # Re-open — init should detect existing data and set "true"
        stack2 = SQLiteStack(
            stack_id="notes-only",
            db_path=tmp_path / "notes.db",
            components=[],
        )
        try:
            setting = stack2._backend.get_stack_setting("use_legacy_heuristics")
            assert setting == "true"
        finally:
            stack2._backend.close()
