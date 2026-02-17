"""Tests for kernle.storage.schema module.

Tests the extracted schema, migration, and FTS5 functions.
"""

import sqlite3

import pytest

from kernle.storage.schema import (
    ALLOWED_TABLES,
    SCHEMA,
    SCHEMA_VERSION,
    VECTOR_SCHEMA,
    ensure_raw_fts5,
    init_db,
    migrate_schema,
    validate_table_name,
)


class TestValidateTableName:
    """Tests for validate_table_name()."""

    def test_valid_table_names(self):
        for table in ALLOWED_TABLES:
            assert validate_table_name(table) == table

    def test_invalid_table_name_raises(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            validate_table_name("not_a_table")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_table_name("")

    def test_sql_injection_attempt(self):
        with pytest.raises(ValueError):
            validate_table_name("episodes; DROP TABLE episodes")


class TestSchemaConstants:
    """Tests for schema constants."""

    def test_schema_version_is_current(self):
        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION == 27

    def test_allowed_tables_has_core_tables(self):
        core_tables = {
            "episodes",
            "beliefs",
            "agent_values",
            "goals",
            "notes",
            "drives",
            "relationships",
            "raw_entries",
        }
        assert core_tables.issubset(ALLOWED_TABLES)

    def test_schema_is_string(self):
        assert isinstance(SCHEMA, str)
        assert "CREATE TABLE" in SCHEMA

    def test_vector_schema_has_placeholder(self):
        assert "{dim}" in VECTOR_SCHEMA

    def test_vector_schema_can_be_formatted(self):
        formatted = VECTOR_SCHEMA.format(dim=128)
        assert "FLOAT[128]" in formatted


class TestEnsureRawFts5:
    """Tests for ensure_raw_fts5()."""

    def test_creates_fts5_table(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Need raw_entries table first
        conn.executescript(SCHEMA)
        ensure_raw_fts5(conn)
        # Check table exists (FTS5 may or may not be available depending on SQLite build)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_fts'"
        ).fetchone()
        # FTS5 availability depends on SQLite build; assert result if table was created
        if row is not None:
            assert row["name"] == "raw_fts"
        conn.close()

    def test_idempotent_call(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        # Call twice - should not error
        ensure_raw_fts5(conn)
        ensure_raw_fts5(conn)
        # Verify table still works after second call
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_fts'"
        ).fetchone()
        if row is not None:
            assert row["name"] == "raw_fts"
        conn.close()


class TestMigrateSchema:
    """Tests for migrate_schema()."""

    def test_fresh_database_no_migration(self):
        """Fresh database should not need migration."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # No tables exist yet
        migrate_schema(conn, "test-stack")
        # Should return without error
        conn.close()

    def test_migration_on_existing_schema(self):
        """Migration should handle existing tables gracefully."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create schema first
        conn.executescript(SCHEMA)
        # Run migration - should be idempotent
        migrate_schema(conn, "test-stack")
        conn.close()

    def test_migration_adds_missing_columns(self):
        """Test migration adds columns to a minimal episodes table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create a minimal episodes table (like an old schema)
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()

        # Run migration
        migrate_schema(conn, "test-stack")

        # Check that new columns were added
        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
        assert "emotional_valence" in cols
        assert "emotional_arousal" in cols
        assert "confidence" in cols
        assert "source_type" in cols
        assert "strength" in cols
        assert "context" in cols
        assert "epoch_id" in cols
        conn.close()

    def test_migration_adds_sync_conflict_metadata_columns(self):
        """Add sync_conflict metadata columns when upgrading older schemas."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
            """)
        conn.execute("""
            CREATE TABLE sync_conflicts (
                id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                local_version TEXT NOT NULL,
                cloud_version TEXT NOT NULL,
                resolution TEXT NOT NULL,
                resolved_at TEXT NOT NULL,
                local_summary TEXT,
                cloud_summary TEXT
            )
            """)
        conn.commit()

        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(sync_conflicts)").fetchall()}
        assert "source" in cols
        assert "diff_hash" in cols
        assert "policy_decision" in cols
        conn.close()

    def test_migration_creates_health_check_table(self):
        """Migration creates health_check_events when missing."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create a minimal episodes table to trigger migration
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "health_check_events" in tables
        conn.close()

    def test_migration_creates_boot_config(self):
        """Migration creates boot_config table when missing."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "boot_config" in tables
        conn.close()

    def test_migration_creates_trust_assessments(self):
        """Migration creates trust_assessments table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "trust_assessments" in tables
        conn.close()

    def test_migration_creates_diagnostic_tables(self):
        """Migration creates diagnostic tables."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "diagnostic_sessions" in tables
        assert "diagnostic_reports" in tables
        conn.close()

    def test_migration_creates_summaries_and_epochs(self):
        """Migration creates summaries and epochs tables."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "summaries" in tables
        assert "epochs" in tables
        conn.close()

    def test_migration_creates_self_narratives(self):
        """Migration creates self_narratives table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "self_narratives" in tables
        conn.close()

    def test_migration_creates_memory_audit(self):
        """Migration creates memory_audit table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "memory_audit" in tables
        conn.close()

    def test_migration_creates_processing_config(self):
        """Migration creates processing_config table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "processing_config" in tables
        conn.close()

    def test_migration_creates_stack_settings(self):
        """Migration creates stack_settings table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "stack_settings" in tables
        conn.close()

    def test_migration_forgetting_fields(self):
        """Migration adds forgetting fields to all memory tables."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
        assert "times_accessed" in cols
        assert "last_accessed" in cols
        assert "is_protected" in cols
        assert "strength" in cols
        conn.close()

    def test_migration_privacy_fields(self):
        """Migration adds privacy fields to tables."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
        assert "subject_ids" in cols
        assert "access_grants" in cols
        assert "consent_grants" in cols
        conn.close()

    def test_migration_source_entity_field(self):
        """Migration adds source_entity field."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
        assert "source_entity" in cols
        conn.close()

    def test_migration_belief_scope_fields(self):
        """Migration adds belief scope fields."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Need beliefs table
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY, stack_id TEXT, objective TEXT, outcome TEXT,
                created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE beliefs (
                id TEXT PRIMARY KEY, stack_id TEXT, statement TEXT,
                confidence REAL DEFAULT 0.8, created_at TEXT,
                local_updated_at TEXT, version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(beliefs)").fetchall()}
        assert "belief_scope" in cols
        assert "source_domain" in cols
        assert "abstraction_level" in cols
        conn.close()


class TestInitDb:
    """Tests for init_db()."""

    def test_init_db_creates_tables(self, tmp_path):
        """init_db creates all required tables."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        def mock_load_vec(c):
            pass

        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()

        init_db(conn, "test-stack", False, 128, mock_load_vec, db_path, agent_dir)

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "episodes" in tables
        assert "beliefs" in tables
        assert "raw_entries" in tables
        assert "schema_version" in tables
        conn.close()

    def test_init_db_sets_schema_version(self, tmp_path):
        """init_db sets the schema version."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        def mock_load_vec(c):
            pass

        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()

        init_db(conn, "test-stack", False, 128, mock_load_vec, db_path, agent_dir)

        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert row is not None
        assert row["version"] == SCHEMA_VERSION
        conn.close()

    def test_init_db_idempotent(self, tmp_path):
        """init_db can be called multiple times safely."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        def mock_load_vec(c):
            pass

        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()

        init_db(conn, "test-stack", False, 128, mock_load_vec, db_path, agent_dir)
        init_db(conn, "test-stack", False, 128, mock_load_vec, db_path, agent_dir)
        conn.close()

    def test_init_db_with_vec(self, tmp_path):
        """init_db handles vec extension (mocked)."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        vec_loaded = []

        def mock_load_vec(c):
            vec_loaded.append(True)

        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()

        # has_vec=True triggers load_vec_fn call
        init_db(conn, "test-stack", True, 128, mock_load_vec, db_path, agent_dir)
        assert len(vec_loaded) == 1
        conn.close()


def _make_old_schema_conn():
    """Create an in-memory conn with minimal old schema tables for migration tests."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Create minimal tables like an old schema (v8 style)
    conn.execute("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY, stack_id TEXT NOT NULL, objective TEXT, outcome TEXT,
            created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE beliefs (
            id TEXT PRIMARY KEY, stack_id TEXT, statement TEXT,
            confidence REAL DEFAULT 0.8, created_at TEXT,
            local_updated_at TEXT, version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE agent_values (
            id TEXT PRIMARY KEY, stack_id TEXT, name TEXT, description TEXT,
            importance REAL DEFAULT 0.5, created_at TEXT,
            local_updated_at TEXT, version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE goals (
            id TEXT PRIMARY KEY, stack_id TEXT, description TEXT,
            status TEXT DEFAULT 'active', created_at TEXT,
            local_updated_at TEXT, version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE notes (
            id TEXT PRIMARY KEY, stack_id TEXT, content TEXT,
            created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE drives (
            id TEXT PRIMARY KEY, stack_id TEXT, name TEXT, description TEXT,
            intensity REAL DEFAULT 0.5, created_at TEXT,
            local_updated_at TEXT, version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE relationships (
            id TEXT PRIMARY KEY, stack_id TEXT, entity_name TEXT,
            relationship_type TEXT, trust_level REAL DEFAULT 0.5,
            created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE raw_entries (
            id TEXT PRIMARY KEY, stack_id TEXT, content TEXT,
            timestamp TEXT, source TEXT, processed INTEGER DEFAULT 0,
            processed_into TEXT, tags TEXT, confidence REAL DEFAULT 1.0,
            source_type TEXT DEFAULT 'direct_experience',
            local_updated_at TEXT, cloud_synced_at TEXT,
            version INTEGER DEFAULT 1, deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE sync_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT, record_id TEXT, operation TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


class TestMigrateSchemaPerTable:
    """Tests for per-table column migration paths in migrate_schema."""

    def test_values_table_gets_new_columns(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(agent_values)").fetchall()}
        assert "confidence" in cols
        assert "source_type" in cols
        assert "source_episodes" in cols
        assert "derived_from" in cols
        assert "last_verified" in cols
        assert "verification_count" in cols
        assert "confidence_history" in cols
        conn.close()

    def test_goals_table_gets_new_columns(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(goals)").fetchall()}
        assert "confidence" in cols
        assert "source_type" in cols
        assert "derived_from" in cols
        assert "goal_type" in cols
        conn.close()

    def test_notes_table_gets_new_columns(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
        assert "confidence" in cols
        assert "source_type" in cols
        assert "derived_from" in cols
        conn.close()

    def test_drives_table_gets_new_columns(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(drives)").fetchall()}
        assert "confidence" in cols
        assert "source_type" in cols
        assert "derived_from" in cols
        conn.close()

    def test_relationships_table_gets_new_columns(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(relationships)").fetchall()}
        assert "confidence" in cols
        assert "source_type" in cols
        assert "derived_from" in cols
        conn.close()

    def test_sync_queue_gets_new_columns(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(sync_queue)").fetchall()}
        assert "payload" in cols
        assert "data" in cols
        assert "local_updated_at" in cols
        assert "synced" in cols
        assert "retry_count" in cols
        assert "last_error" in cols
        assert "last_attempt_at" in cols
        conn.close()

    def test_raw_entries_gets_blob_column(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(raw_entries)").fetchall()}
        assert "blob" in cols
        assert "captured_at" in cols
        conn.close()

    def test_raw_blob_data_migration(self):
        """Old content data migrated to blob format."""
        conn = _make_old_schema_conn()
        # Insert old-style entry with content but no blob
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, content, timestamp, source, processed, "
            "local_updated_at, version, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "my content", "2026-01-01T10:00:00", "cli", 0, "2026-01-01", 1, 0),
        )
        conn.commit()

        migrate_schema(conn, "s")

        row = conn.execute("SELECT blob, captured_at FROM raw_entries WHERE id = 'r1'").fetchone()
        assert row["blob"] is not None
        assert "my content" in row["blob"]
        assert row["captured_at"] == "2026-01-01T10:00:00"
        conn.close()

    def test_raw_blob_migration_normalizes_source(self):
        """Raw migration normalizes old source values."""
        conn = _make_old_schema_conn()
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, content, timestamp, source, processed, "
            "local_updated_at, version, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "content", "2026-01-01", "manual", 0, "2026-01-01", 1, 0),
        )
        conn.commit()

        migrate_schema(conn, "s")

        row = conn.execute("SELECT source FROM raw_entries WHERE id = 'r1'").fetchone()
        assert row["source"] == "cli"  # manual -> cli
        conn.close()

    def test_forgetting_fields_on_all_tables(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        for table in [
            "episodes",
            "beliefs",
            "agent_values",
            "goals",
            "notes",
            "drives",
            "relationships",
        ]:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "times_accessed" in cols, f"{table} missing times_accessed"
            assert "last_accessed" in cols, f"{table} missing last_accessed"
            assert "is_protected" in cols, f"{table} missing is_protected"
            assert "strength" in cols, f"{table} missing strength"
        conn.close()

    def test_source_entity_on_all_tables(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        for table in [
            "episodes",
            "beliefs",
            "agent_values",
            "goals",
            "notes",
            "drives",
            "relationships",
        ]:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "source_entity" in cols, f"{table} missing source_entity"
        conn.close()

    def test_privacy_fields_on_tables(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        for table in [
            "episodes",
            "beliefs",
            "agent_values",
            "goals",
            "notes",
            "drives",
            "relationships",
            "raw_entries",
        ]:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "subject_ids" in cols, f"{table} missing subject_ids"
            assert "access_grants" in cols, f"{table} missing access_grants"
            assert "consent_grants" in cols, f"{table} missing consent_grants"
        conn.close()

    def test_epoch_id_on_all_tables(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        for table in [
            "episodes",
            "beliefs",
            "agent_values",
            "goals",
            "notes",
            "drives",
            "relationships",
        ]:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "epoch_id" in cols, f"{table} missing epoch_id"
        conn.close()

    def test_belief_scope_fields(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(beliefs)").fetchall()}
        assert "belief_scope" in cols
        assert "source_domain" in cols
        assert "abstraction_level" in cols
        conn.close()

    def test_processed_column_added(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        for table in ["episodes", "notes", "beliefs"]:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "processed" in cols, f"{table} missing processed column"
        conn.close()


class TestMigrateSchemaV23:
    """Tests for v23 agent_id -> stack_id rename migration."""

    def test_v23_renames_agent_id_to_stack_id(self):
        """Tables with agent_id get it renamed to stack_id."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, objective TEXT, outcome TEXT,
                created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()

        migrate_schema(conn, "test-stack")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
        assert "stack_id" in cols
        assert "agent_id" not in cols
        conn.close()

    def test_v23_renames_agent_prefixed_tables(self):
        """agent_trust_assessments -> trust_assessments etc."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create with old-style agent_id and agent-prefixed tables
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, objective TEXT, outcome TEXT,
                created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE agent_trust_assessments (
                id TEXT PRIMARY KEY, agent_id TEXT, entity_name TEXT,
                trust_score REAL, created_at TEXT
            )
        """)
        conn.commit()

        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "trust_assessments" in tables
        assert "agent_trust_assessments" not in tables
        conn.close()

    def test_v23_catchup_renames_remaining_agent_id(self):
        """Catchup migration handles any remaining agent_id columns."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create a table with agent_id that isn't in the explicit list
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, objective TEXT, outcome TEXT,
                created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE beliefs (
                id TEXT PRIMARY KEY, agent_id TEXT, statement TEXT,
                created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.commit()

        migrate_schema(conn, "test-stack")

        beliefs_cols = {row[1] for row in conn.execute("PRAGMA table_info(beliefs)").fetchall()}
        assert "stack_id" in beliefs_cols
        assert "agent_id" not in beliefs_cols
        conn.close()


class TestMigrateSchemaV24:
    """Tests for v24 strength migration (is_forgotten -> strength)."""

    def test_strength_column_added(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        for table in [
            "episodes",
            "beliefs",
            "agent_values",
            "goals",
            "notes",
            "drives",
            "relationships",
        ]:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "strength" in cols, f"{table} missing strength"
        conn.close()

    def test_is_forgotten_migrated_to_strength_zero(self):
        """Existing is_forgotten=1 records get strength=0.0."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY, stack_id TEXT, objective TEXT, outcome TEXT,
                created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0, is_forgotten INTEGER DEFAULT 0
            )
        """)
        conn.execute(
            "INSERT INTO episodes (id, stack_id, objective, outcome, created_at, "
            "local_updated_at, is_forgotten) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("e1", "s", "test", "done", "2026-01-01", "2026-01-01", 1),
        )
        conn.commit()

        migrate_schema(conn, "s")

        row = conn.execute("SELECT strength FROM episodes WHERE id = 'e1'").fetchone()
        assert row["strength"] == 0.0
        conn.close()


class TestMigrateStackSettings:
    """Tests for stack_settings migration."""

    def test_stack_settings_created_when_missing(self):
        conn = _make_old_schema_conn()
        migrate_schema(conn, "test-stack")

        tables = {
            t[0]
            for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "stack_settings" in tables
        conn.close()

    def test_stack_settings_migrated_when_missing_stack_id(self):
        """Old stack_settings without stack_id gets migrated."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY, stack_id TEXT, objective TEXT, outcome TEXT,
                created_at TEXT, local_updated_at TEXT, version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        # Old-style stack_settings without stack_id
        conn.execute("""
            CREATE TABLE stack_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO stack_settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("theme", "dark", "2026-01-01"),
        )
        conn.commit()

        migrate_schema(conn, "test-stack")

        # Should now have stack_id
        cols = {row[1] for row in conn.execute("PRAGMA table_info(stack_settings)").fetchall()}
        assert "stack_id" in cols

        # Old data should be preserved with the stack_id
        row = conn.execute("SELECT * FROM stack_settings WHERE key = 'theme'").fetchone()
        assert row["value"] == "dark"
        assert row["stack_id"] == "test-stack"
        conn.close()


class TestMigrateSchemaV25:
    """Tests for v25 migration: captured_at backfill + compound index."""

    def test_v25_backfills_captured_at(self):
        """Rows with captured_at IS NULL get backfilled from timestamp."""
        conn = _make_old_schema_conn()
        # Insert a raw entry with timestamp but no captured_at
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, content, timestamp, source, processed, "
            "local_updated_at, version, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "my content", "2026-01-15T12:00:00", "cli", 0, "2026-01-15", 1, 0),
        )
        conn.commit()

        migrate_schema(conn, "s")

        row = conn.execute("SELECT captured_at FROM raw_entries WHERE id = 'r1'").fetchone()
        assert row["captured_at"] == "2026-01-15T12:00:00"
        conn.close()

    def test_v25_does_not_overwrite_existing_captured_at(self):
        """Rows that already have captured_at should not be changed."""
        conn = _make_old_schema_conn()
        migrate_schema(conn, "s")  # This adds blob/captured_at columns

        # Insert with both captured_at and timestamp set
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, content, blob, timestamp, captured_at, "
            "source, processed, local_updated_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "r2",
                "s",
                "content",
                "blob",
                "2026-01-10T10:00:00",
                "2026-01-15T12:00:00",
                "cli",
                0,
                "2026-01-15",
                1,
                0,
            ),
        )
        conn.commit()

        # Run migration again
        migrate_schema(conn, "s")

        row = conn.execute("SELECT captured_at FROM raw_entries WHERE id = 'r2'").fetchone()
        assert row["captured_at"] == "2026-01-15T12:00:00"  # Unchanged
        conn.close()

    def test_v24_to_v25_replaces_index(self):
        """Migration should replace the old single-column index with compound (captured_at DESC, id DESC)."""
        conn = _make_old_schema_conn()

        # First run migration to get up to v24 (adds blob, captured_at columns)
        migrate_schema(conn, "test-stack")

        # Manually create the old-style single-column index to simulate a v24 database
        conn.execute("DROP INDEX IF EXISTS idx_raw_captured_at")
        conn.execute("CREATE INDEX idx_raw_captured_at ON raw_entries(stack_id, captured_at DESC)")
        conn.commit()

        # Verify the old index does NOT contain "id DESC"
        old_idx = conn.execute("""
            SELECT sql FROM sqlite_master
            WHERE type='index' AND name='idx_raw_captured_at'
        """).fetchone()
        assert old_idx is not None
        assert "id DESC" not in old_idx[0], "Pre-condition: old index should not have id DESC"

        # Run migration — should detect old index and replace it
        migrate_schema(conn, "test-stack")

        # Verify the new compound index
        new_idx = conn.execute("""
            SELECT sql FROM sqlite_master
            WHERE type='index' AND name='idx_raw_captured_at'
        """).fetchone()
        assert new_idx is not None
        assert "id DESC" in new_idx[0], "New index should contain id DESC"
        assert "captured_at DESC" in new_idx[0]
        conn.close()
