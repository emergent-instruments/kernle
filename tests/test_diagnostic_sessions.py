"""Tests for formal diagnostic sessions (KEP v3 - doctor pattern phase 2)."""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import (
    DiagnosticReport,
    DiagnosticSession,
    SQLiteStorage,
)


@pytest.fixture
def diag_setup(tmp_path):
    """Create a Kernle instance for diagnostic testing."""
    db_path = tmp_path / "test_diag.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    k = Kernle(stack_id="test_agent", storage=storage, checkpoint_dir=checkpoint_dir, strict=False)
    yield k, storage
    storage.close()


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestDiagnosticSessionDataclass:
    """Tests for DiagnosticSession dataclass."""

    def test_create_session(self):
        session = DiagnosticSession(
            id="test-session-1",
            stack_id="test_agent",
            session_type="self_requested",
            access_level="structural",
        )
        assert session.id == "test-session-1"
        assert session.session_type == "self_requested"
        assert session.access_level == "structural"
        assert session.status == "active"
        assert session.consent_given is False
        assert session.version == 1
        assert session.deleted is False

    def test_default_values(self):
        session = DiagnosticSession(
            id="test-id",
            stack_id="test_agent",
        )
        assert session.session_type == "self_requested"
        assert session.access_level == "structural"
        assert session.status == "active"
        assert session.consent_given is False
        assert session.started_at is None
        assert session.completed_at is None

    def test_operator_initiated(self):
        session = DiagnosticSession(
            id="test-op",
            stack_id="test_agent",
            session_type="operator_initiated",
            consent_given=True,
        )
        assert session.session_type == "operator_initiated"
        assert session.consent_given is True


class TestDiagnosticReportDataclass:
    """Tests for DiagnosticReport dataclass."""

    def test_create_report(self):
        findings = [
            {
                "severity": "warning",
                "category": "low_confidence_belief",
                "description": "Belief #abc has low confidence",
                "recommendation": "Review and verify",
            }
        ]
        report = DiagnosticReport(
            id="test-report-1",
            stack_id="test_agent",
            session_id="test-session-1",
            findings=findings,
            summary="Found 1 finding(s): 1 warning(s)",
        )
        assert report.id == "test-report-1"
        assert report.session_id == "test-session-1"
        assert len(report.findings) == 1
        assert report.findings[0]["severity"] == "warning"

    def test_default_values(self):
        report = DiagnosticReport(
            id="test-id",
            stack_id="test_agent",
            session_id="test-session",
        )
        assert report.findings is None
        assert report.summary is None
        assert report.version == 1
        assert report.deleted is False


# =============================================================================
# Storage Tests
# =============================================================================


class TestDiagnosticSessionStorage:
    """Tests for diagnostic session storage operations."""

    def test_save_and_get_session(self, diag_setup):
        k, storage = diag_setup
        now = datetime.now(timezone.utc)
        session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_type="self_requested",
            access_level="structural",
            status="active",
            consent_given=True,
            started_at=now,
        )
        sid = storage.save_diagnostic_session(session)
        assert sid == session.id

        retrieved = storage.get_diagnostic_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.session_type == "self_requested"
        assert retrieved.access_level == "structural"
        assert retrieved.status == "active"
        assert retrieved.consent_given is True

    def test_get_nonexistent_session(self, diag_setup):
        _, storage = diag_setup
        result = storage.get_diagnostic_session("nonexistent-id")
        assert result is None

    def test_list_sessions(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        for i in range(3):
            session = DiagnosticSession(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                session_type="routine",
                started_at=now,
            )
            storage.save_diagnostic_session(session)

        sessions = storage.get_diagnostic_sessions()
        assert len(sessions) == 3

    def test_list_sessions_by_status(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        active_session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            status="active",
            started_at=now,
        )
        storage.save_diagnostic_session(active_session)

        completed_session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            status="completed",
            started_at=now,
            completed_at=now,
        )
        storage.save_diagnostic_session(completed_session)

        active = storage.get_diagnostic_sessions(status="active")
        assert len(active) == 1
        assert active[0].status == "active"

        completed = storage.get_diagnostic_sessions(status="completed")
        assert len(completed) == 1
        assert completed[0].status == "completed"

    def test_complete_session(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            status="active",
            started_at=now,
        )
        storage.save_diagnostic_session(session)

        result = storage.complete_diagnostic_session(session.id)
        assert result is True

        retrieved = storage.get_diagnostic_session(session.id)
        assert retrieved.status == "completed"
        assert retrieved.completed_at is not None

    def test_complete_already_completed(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            status="completed",
            started_at=now,
            completed_at=now,
        )
        storage.save_diagnostic_session(session)

        result = storage.complete_diagnostic_session(session.id)
        assert result is False  # Already completed


class TestDiagnosticReportStorage:
    """Tests for diagnostic report storage operations."""

    def test_save_and_get_report(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        findings = [
            {
                "severity": "error",
                "category": "orphaned_reference",
                "description": "Broken ref in episode #abc",
                "recommendation": "Remove or update",
            }
        ]
        report = DiagnosticReport(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_id="session-1",
            findings=findings,
            summary="Found 1 error",
            created_at=now,
        )
        rid = storage.save_diagnostic_report(report)
        assert rid == report.id

        retrieved = storage.get_diagnostic_report(report.id)
        assert retrieved is not None
        assert retrieved.session_id == "session-1"
        assert len(retrieved.findings) == 1
        assert retrieved.findings[0]["severity"] == "error"
        assert retrieved.summary == "Found 1 error"

    def test_get_nonexistent_report(self, diag_setup):
        _, storage = diag_setup
        result = storage.get_diagnostic_report("nonexistent-id")
        assert result is None

    def test_list_reports(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        for i in range(3):
            report = DiagnosticReport(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                session_id="session-1",
                findings=[],
                created_at=now,
            )
            storage.save_diagnostic_report(report)

        reports = storage.get_diagnostic_reports()
        assert len(reports) == 3

    def test_list_reports_by_session(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        for session_id in ["session-1", "session-2"]:
            report = DiagnosticReport(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                session_id=session_id,
                findings=[],
                created_at=now,
            )
            storage.save_diagnostic_report(report)

        session_1_reports = storage.get_diagnostic_reports(session_id="session-1")
        assert len(session_1_reports) == 1
        assert session_1_reports[0].session_id == "session-1"

    def test_report_with_empty_findings(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        report = DiagnosticReport(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_id="session-healthy",
            findings=[],
            summary="No issues found. Memory graph is healthy.",
            created_at=now,
        )
        storage.save_diagnostic_report(report)

        retrieved = storage.get_diagnostic_report(report.id)
        assert retrieved.findings == []
        assert "healthy" in retrieved.summary


# =============================================================================
# Schema Migration Tests
# =============================================================================


class TestSchemaVersion:
    """Tests that schema version is correctly updated."""

    def test_schema_version_is_22(self):
        from kernle.storage.sqlite import SCHEMA_VERSION

        assert SCHEMA_VERSION == 27

    def test_tables_in_allowed_list(self):
        from kernle.storage.sqlite import ALLOWED_TABLES

        assert "diagnostic_sessions" in ALLOWED_TABLES
        assert "diagnostic_reports" in ALLOWED_TABLES

    def test_tables_created_on_init(self, tmp_path):
        """Verify tables are created for fresh databases."""
        import sqlite3

        db_path = tmp_path / "fresh.db"
        storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        storage.close()

        assert "diagnostic_sessions" in tables
        assert "diagnostic_reports" in tables
