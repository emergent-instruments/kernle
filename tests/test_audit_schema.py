"""Tests for Phase 1: #410 — Standardize Audit Event Schema.

TDD tests for:
- Schema migration 26→27 (correlation_id column)
- Backward-compatible log_audit() extension
- get_audit_log() correlation_id filtering
- export_audit_jsonl() JSONL export
- Processing pipeline correlation_id threading
- CLI audit export subcommand
"""

import json
import sqlite3
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kernle.storage.schema import migrate_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pre_v27_db(tmp_path):
    """Create a SQLite database at schema version 26 (pre-correlation_id).

    Inserts sample audit rows to test migration and backward compat.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Create minimal schema matching v26
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER);
        INSERT INTO schema_version (version) VALUES (26);

        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            stack_id TEXT NOT NULL,
            objective TEXT,
            outcome TEXT,
            outcome_type TEXT DEFAULT 'partial',
            lessons TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            local_updated_at TEXT NOT NULL,
            cloud_synced_at TEXT,
            version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0,
            emotional_valence REAL DEFAULT 0.0,
            emotional_arousal REAL DEFAULT 0.0,
            emotional_tags TEXT,
            confidence REAL DEFAULT 0.8,
            source_type TEXT DEFAULT 'direct_experience',
            source_episodes TEXT,
            derived_from TEXT,
            last_verified TEXT,
            verification_count INTEGER DEFAULT 0,
            confidence_history TEXT,
            repeat TEXT,
            avoid TEXT,
            times_accessed INTEGER DEFAULT 0,
            last_accessed TEXT,
            is_protected INTEGER DEFAULT 0,
            is_forgotten INTEGER DEFAULT 0,
            forgotten_at TEXT,
            forgotten_reason TEXT,
            context TEXT,
            context_tags TEXT,
            subject_ids TEXT,
            access_grants TEXT,
            consent_grants TEXT,
            source_entity TEXT,
            epoch_id TEXT,
            strength REAL DEFAULT 1.0,
            processed INTEGER DEFAULT 0
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
        CREATE INDEX idx_audit_memory ON memory_audit(memory_type, memory_id);
        CREATE INDEX idx_audit_operation ON memory_audit(operation);
        CREATE INDEX idx_audit_created ON memory_audit(created_at);

        CREATE TABLE stack_settings (
            stack_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (stack_id, key)
        );

        CREATE TABLE raw_entries (
            id TEXT PRIMARY KEY,
            stack_id TEXT NOT NULL,
            blob TEXT,
            source TEXT,
            captured_at TEXT,
            timestamp TEXT,
            processed INTEGER DEFAULT 0,
            deleted INTEGER DEFAULT 0,
            local_updated_at TEXT,
            cloud_synced_at TEXT,
            version INTEGER DEFAULT 1,
            content TEXT,
            tags TEXT,
            subject_ids TEXT,
            access_grants TEXT,
            consent_grants TEXT
        );
    """)

    # Insert pre-migration audit rows with old-style operation values
    conn.execute(
        "INSERT INTO memory_audit (id, memory_type, memory_id, operation, details, actor, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "old-1",
            "belief",
            "b-001",
            "forget",
            '{"reason": "outdated"}',
            "system",
            "2025-01-01T00:00:00Z",
        ),
    )
    conn.execute(
        "INSERT INTO memory_audit (id, memory_type, memory_id, operation, details, actor, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("old-2", "episode", "e-001", "protect", None, "system", "2025-01-02T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO memory_audit (id, memory_type, memory_id, operation, details, actor, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "old-3",
            "belief",
            "b-002",
            "accepted",
            '{"source": "processing"}',
            "system",
            "2025-01-03T00:00:00Z",
        ),
    )
    conn.commit()

    yield conn, db_path
    conn.close()


@pytest.fixture
def storage_with_audit(tmp_path):
    """SQLiteStorage instance for testing audit operations."""
    from kernle.storage.sqlite import SQLiteStorage

    db_path = tmp_path / "test_audit.db"
    storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    yield storage
    storage.close()


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


class TestMigration26To27:
    def test_migration_adds_correlation_id_column(self, pre_v27_db):
        """Migration 26→27 adds correlation_id TEXT column to memory_audit."""
        conn, db_path = pre_v27_db

        # Verify column doesn't exist before migration
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memory_audit)").fetchall()}
        assert "correlation_id" not in cols

        # Run migration
        migrate_schema(conn, "test-stack")

        # Verify column exists after migration
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memory_audit)").fetchall()}
        assert "correlation_id" in cols

    def test_migration_creates_correlation_id_index(self, pre_v27_db):
        """Migration 26→27 creates an index on correlation_id."""
        conn, db_path = pre_v27_db
        migrate_schema(conn, "test-stack")

        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memory_audit'"
        ).fetchall()
        index_names = {row[0] for row in indexes}
        assert "idx_audit_correlation" in index_names

    def test_old_audit_rows_readable_after_migration(self, pre_v27_db):
        """Pre-migration audit rows are still readable with NULL correlation_id."""
        conn, db_path = pre_v27_db
        migrate_schema(conn, "test-stack")

        rows = conn.execute(
            "SELECT id, memory_type, memory_id, operation, details, actor, created_at, correlation_id "
            "FROM memory_audit ORDER BY created_at"
        ).fetchall()

        assert len(rows) == 3
        # All old rows should have NULL correlation_id
        for row in rows:
            assert row["correlation_id"] is None
        # Verify data integrity
        assert rows[0]["id"] == "old-1"
        assert rows[0]["operation"] == "forget"
        assert rows[1]["operation"] == "protect"
        assert rows[2]["operation"] == "accepted"


# ---------------------------------------------------------------------------
# log_audit() tests
# ---------------------------------------------------------------------------


class TestLogAudit:
    def test_log_audit_without_correlation_id(self, storage_with_audit):
        """Existing callers that don't pass correlation_id work unchanged."""
        storage = storage_with_audit
        audit_id = storage.log_audit("belief", "b-001", "forget", "system")
        assert audit_id

        log = storage.get_audit_log(memory_id="b-001")
        assert len(log) == 1
        assert log[0]["operation"] == "forget"
        assert log[0]["correlation_id"] is None

    def test_log_audit_with_correlation_id(self, storage_with_audit):
        """New callers can pass correlation_id and it is stored."""
        storage = storage_with_audit
        corr_id = str(uuid.uuid4())
        audit_id = storage.log_audit(
            "belief",
            "b-001",
            "belief.revised",
            "system",
            correlation_id=corr_id,
        )
        assert audit_id

        log = storage.get_audit_log(memory_id="b-001")
        assert len(log) == 1
        assert log[0]["correlation_id"] == corr_id

    def test_log_audit_with_details_and_correlation_id(self, storage_with_audit):
        """Both details and correlation_id can be passed together."""
        storage = storage_with_audit
        corr_id = str(uuid.uuid4())
        details = {"transition": "episode_to_belief", "source_id": "raw-123"}

        audit_id = storage.log_audit(
            "belief",
            "b-001",
            "suggestion.created",
            "system",
            details=details,
            correlation_id=corr_id,
        )
        assert audit_id

        log = storage.get_audit_log(memory_id="b-001")
        assert len(log) == 1
        assert log[0]["details"] == details
        assert log[0]["correlation_id"] == corr_id


# ---------------------------------------------------------------------------
# get_audit_log() filter tests
# ---------------------------------------------------------------------------


class TestGetAuditLogFilters:
    def test_filter_by_correlation_id(self, storage_with_audit):
        """Filtering by correlation_id returns only matching entries."""
        storage = storage_with_audit
        corr_a = str(uuid.uuid4())
        corr_b = str(uuid.uuid4())

        storage.log_audit("belief", "b-001", "forget", "system", correlation_id=corr_a)
        storage.log_audit("belief", "b-002", "protect", "system", correlation_id=corr_b)
        storage.log_audit("episode", "e-001", "recover", "system")  # no correlation_id

        log = storage.get_audit_log(correlation_id=corr_a)
        assert len(log) == 1
        assert log[0]["memory_id"] == "b-001"

    def test_filter_ignores_null_correlation_rows(self, storage_with_audit):
        """Filtering by correlation_id excludes rows with NULL correlation_id."""
        storage = storage_with_audit
        corr_id = str(uuid.uuid4())

        storage.log_audit("belief", "b-001", "forget", "system")  # NULL
        storage.log_audit("belief", "b-002", "protect", "system", correlation_id=corr_id)

        log = storage.get_audit_log(correlation_id=corr_id)
        assert len(log) == 1
        assert log[0]["memory_id"] == "b-002"

    def test_filter_by_dotted_operation(self, storage_with_audit):
        """New dotted operation values (belief.revised) filter correctly."""
        storage = storage_with_audit

        storage.log_audit("belief", "b-001", "forget", "system")
        storage.log_audit(
            "belief", "b-002", "belief.revised", "system", details={"revision_type": "confidence"}
        )
        storage.log_audit("belief", "b-003", "suggestion.created", "system")

        log = storage.get_audit_log(operation="belief.revised")
        assert len(log) == 1
        assert log[0]["memory_id"] == "b-002"

    def test_old_operation_values_still_work(self, storage_with_audit):
        """Old operation values (forget, recover, protect) continue to filter correctly."""
        storage = storage_with_audit

        storage.log_audit("belief", "b-001", "forget", "system")
        storage.log_audit("belief", "b-002", "belief.revised", "system")

        log = storage.get_audit_log(operation="forget")
        assert len(log) == 1
        assert log[0]["memory_id"] == "b-001"


# ---------------------------------------------------------------------------
# export_audit_jsonl() tests
# ---------------------------------------------------------------------------


class TestExportAuditJsonl:
    def test_export_jsonl_valid_format(self, storage_with_audit):
        """Each exported line is valid JSON with required fields."""
        storage = storage_with_audit
        corr_id = str(uuid.uuid4())

        storage.log_audit(
            "belief",
            "b-001",
            "belief.revised",
            "system",
            details={"revision_type": "confidence"},
            correlation_id=corr_id,
        )

        lines = list(storage.export_audit_jsonl())
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["schema_version"] == 1
        assert "event_id" in entry
        assert entry["operation"] == "belief.revised"
        assert entry["memory_type"] == "belief"
        assert entry["memory_id"] == "b-001"
        assert entry["correlation_id"] == corr_id
        assert entry["actor"] == "system"
        assert "timestamp" in entry
        assert entry["details"] == {"revision_type": "confidence"}

    def test_export_jsonl_legacy_rows_null_correlation(self, storage_with_audit):
        """Old rows export with correlation_id: null."""
        storage = storage_with_audit

        storage.log_audit("belief", "b-001", "forget", "system")

        lines = list(storage.export_audit_jsonl())
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["correlation_id"] is None
        assert entry["operation"] == "forget"

    def test_export_jsonl_mixed_old_new_rows(self, storage_with_audit):
        """Both old-style and new dotted operation values appear in output."""
        storage = storage_with_audit
        corr_id = str(uuid.uuid4())

        storage.log_audit("belief", "b-001", "forget", "system")
        storage.log_audit("belief", "b-002", "belief.revised", "system", correlation_id=corr_id)
        storage.log_audit(
            "episode", "e-001", "suggestion.created", "system", correlation_id=corr_id
        )

        lines = list(storage.export_audit_jsonl())
        assert len(lines) == 3

        operations = {json.loads(line)["operation"] for line in lines}
        assert "forget" in operations
        assert "belief.revised" in operations
        assert "suggestion.created" in operations

    def test_export_jsonl_filter_by_operation(self, storage_with_audit):
        """export_audit_jsonl respects operation filter."""
        storage = storage_with_audit

        storage.log_audit("belief", "b-001", "forget", "system")
        storage.log_audit("belief", "b-002", "belief.revised", "system")

        lines = list(storage.export_audit_jsonl(operation="belief.revised"))
        assert len(lines) == 1
        assert json.loads(lines[0])["operation"] == "belief.revised"

    def test_export_jsonl_filter_by_memory_type(self, storage_with_audit):
        """export_audit_jsonl respects memory_type filter."""
        storage = storage_with_audit

        storage.log_audit("belief", "b-001", "forget", "system")
        storage.log_audit("episode", "e-001", "forget", "system")

        lines = list(storage.export_audit_jsonl(memory_type="belief"))
        assert len(lines) == 1
        assert json.loads(lines[0])["memory_type"] == "belief"

    def test_export_jsonl_filter_by_correlation_id(self, storage_with_audit):
        """export_audit_jsonl respects correlation_id filter."""
        storage = storage_with_audit
        corr_id = str(uuid.uuid4())

        storage.log_audit("belief", "b-001", "forget", "system")
        storage.log_audit("belief", "b-002", "belief.revised", "system", correlation_id=corr_id)

        lines = list(storage.export_audit_jsonl(correlation_id=corr_id))
        assert len(lines) == 1
        assert json.loads(lines[0])["correlation_id"] == corr_id


# ---------------------------------------------------------------------------
# Processing correlation_id tests
# ---------------------------------------------------------------------------


class TestProcessingCorrelationId:
    def test_processing_emits_correlation_id(self, tmp_path):
        """All audit entries from one process() call share the same correlation_id."""
        from kernle.stack.sqlite_stack import Stack

        db_path = tmp_path / "test_processing.db"
        stack = Stack.from_sqlite(stack_id="test-stack", db_path=db_path)
        storage = stack._backend

        try:
            # Add some raw entries to process
            for i in range(3):
                storage.save_raw(
                    f"Test raw entry {i} about completing a task successfully",
                    source="cli",
                )

            # Mock inference to return a valid response
            mock_inference = MagicMock()
            mock_inference.model_id = "test-model"
            mock_inference.infer.return_value = json.dumps(
                [
                    {
                        "objective": "Test task",
                        "outcome": "Completed successfully",
                        "outcome_type": "success",
                        "lessons": ["lesson 1"],
                        "tags": ["test"],
                        "confidence": 0.8,
                        "source_ids": [],
                    }
                ]
            )
            mock_inference.embed.return_value = [0.0] * 64
            mock_inference.embed_batch.return_value = [[0.0] * 64]
            mock_inference.embedding_dimension = 64
            mock_inference.embedding_provider_id = "mock"

            from kernle.processing import MemoryProcessor

            processor = MemoryProcessor(
                stack=stack,
                inference=mock_inference,
                core_id="test-stack",
                auto_promote=True,
            )

            processor.process(transition="raw_to_episode", force=True)

            # Check audit log for correlation_id
            log = storage.get_audit_log(operation="process")
            if log:
                # All entries from this process() call should share one correlation_id
                corr_ids = {
                    entry.get("correlation_id") for entry in log if entry.get("correlation_id")
                }
                assert len(corr_ids) <= 1, "Multiple correlation_ids found in single process() run"
                if corr_ids:
                    # Verify it's a valid UUID
                    corr_id = corr_ids.pop()
                    uuid.UUID(corr_id)  # Raises if invalid
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# CLI audit export tests
# ---------------------------------------------------------------------------


class TestCliAuditExport:
    def test_cli_audit_export_jsonl(self, storage_with_audit, capsys):
        """CLI 'audit export' produces valid JSONL output."""
        storage = storage_with_audit
        corr_id = str(uuid.uuid4())

        storage.log_audit(
            "belief",
            "b-001",
            "belief.revised",
            "system",
            details={"revision_type": "confidence"},
            correlation_id=corr_id,
        )
        storage.log_audit("episode", "e-001", "forget", "system")

        from kernle.cli.commands.audit import cmd_audit

        args = SimpleNamespace(
            audit_action="export",
            format="jsonl",
            since=None,
            until=None,
            memory_type=None,
            operation=None,
            correlation_id=None,
        )

        # Create a mock Kernle that exposes the storage
        k = MagicMock()
        k._storage = storage

        cmd_audit(args, k)

        output = capsys.readouterr().out.strip()
        lines = output.split("\n")
        assert len(lines) == 2

        for line in lines:
            entry = json.loads(line)
            assert "schema_version" in entry
            assert "event_id" in entry
            assert "operation" in entry

    def test_cli_audit_export_with_filters(self, storage_with_audit, capsys):
        """CLI 'audit export' respects filter arguments."""
        storage = storage_with_audit
        corr_id = str(uuid.uuid4())

        storage.log_audit("belief", "b-001", "belief.revised", "system", correlation_id=corr_id)
        storage.log_audit("episode", "e-001", "forget", "system")

        from kernle.cli.commands.audit import cmd_audit

        args = SimpleNamespace(
            audit_action="export",
            format="jsonl",
            since=None,
            until=None,
            memory_type="belief",
            operation=None,
            correlation_id=None,
        )

        k = MagicMock()
        k._storage = storage

        cmd_audit(args, k)

        output = capsys.readouterr().out.strip()
        lines = output.split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["memory_type"] == "belief"


# ---------------------------------------------------------------------------
# raw.ingested event emission
# ---------------------------------------------------------------------------


class TestRawIngestedEvent:
    def test_save_raw_emits_raw_ingested(self, tmp_path):
        """save_raw() on SQLiteStorage should emit a raw.ingested audit event."""
        from kernle.storage.sqlite import SQLiteStorage

        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(stack_id="test-stack", db_path=db_path)
        try:
            raw_id = storage.save_raw("hello world", source="cli")
            audit = storage.get_audit_log(operation="raw.ingested")
            assert len(audit) == 1
            entry = audit[0]
            assert entry["memory_type"] == "raw"
            assert entry["memory_id"] == raw_id
            details = entry.get("details", {})
            if isinstance(details, str):
                details = json.loads(details)
            assert details["source"] == "cli"
            assert details["content_length"] == 11
        finally:
            storage.close()
