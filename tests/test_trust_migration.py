"""Tests for self-trust bootstrapping after migration (#700)."""

import uuid

import pytest

from kernle.stack.sqlite_stack import Stack
from kernle.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    s = SQLiteStorage(stack_id="trust-test", db_path=tmp_path / "trust.db")
    yield s
    s.close()


class TestSelfTrustBootstrap:
    """Verify _ensure_self_trust creates identity assessment on first init."""

    def test_migration_initializes_self_trust(self, tmp_path):
        """A fresh Stack should have a self-trust assessment for 'identity'."""
        stack = Stack.from_sqlite(
            stack_id="trust-init",
            db_path=tmp_path / "init.db",
            components=[],
        )
        assessment = stack._backend.get_trust_assessment("identity")
        assert assessment is not None
        assert assessment.entity == "identity"
        assert assessment.dimensions["general"]["score"] == 1.0
        stack._backend.close()

    def test_self_trust_is_idempotent(self, tmp_path):
        """Calling _ensure_self_trust twice does not duplicate the assessment."""
        db = tmp_path / "idem.db"
        stack = Stack.from_sqlite(stack_id="trust-idem", db_path=db, components=[])
        first_id = stack._backend.get_trust_assessment("identity").id

        # Call again explicitly
        stack._ensure_self_trust()
        second_id = stack._backend.get_trust_assessment("identity").id

        # save_trust_assessment updates existing rather than inserting a duplicate
        assert first_id == second_id

        # Only one assessment with entity='identity' should exist
        all_assessments = stack._backend.get_trust_assessments()
        identity_assessments = [a for a in all_assessments if a.entity == "identity"]
        assert len(identity_assessments) == 1
        stack._backend.close()

    def test_existing_trust_not_overwritten(self, tmp_path):
        """If a custom trust assessment for identity already exists, don't clobber it."""
        from kernle.types import TrustAssessment

        db = tmp_path / "existing.db"
        backend = SQLiteStorage(stack_id="trust-exist", db_path=db)

        # Manually insert a custom trust assessment before stack init
        custom = TrustAssessment(
            id=str(uuid.uuid4()),
            stack_id="trust-exist",
            entity="identity",
            dimensions={"general": {"score": 0.75}},
        )
        backend.save_trust_assessment(custom)

        # Now create the stack -- _ensure_self_trust should see existing and not overwrite
        stack = Stack.from_sqlite(stack_id="trust-exist", db_path=db, components=[])
        assessment = stack._backend.get_trust_assessment("identity")
        assert assessment is not None
        # The score should still be 0.75, not reset to 1.0
        assert assessment.dimensions["general"]["score"] == 0.75
        stack._backend.close()
        backend.close()
