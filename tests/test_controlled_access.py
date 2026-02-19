"""Tests for controlled access model (v0.9.0 PR 5).

Tests cover:
- SQLiteStorage: weaken_memory, verify_memory, log_audit, get_audit_log
- SQLiteStorage: audit trail in forget_memory, recover_memory, protect_memory
- Stack: routing of new methods
- Entity: weaken, forget, recover, verify, protect with audit logging
"""

from __future__ import annotations

import uuid

import pytest

from kernle.entity import Entity
from kernle.stack.sqlite_stack import Stack
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Episode

STACK_ID = "test-stack"


@pytest.fixture
def storage(tmp_path):
    s = SQLiteStorage(STACK_ID, db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def stack(tmp_path):
    return Stack.from_sqlite(
        STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
    )


@pytest.fixture
def entity(tmp_path):
    ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
    st = Stack.from_sqlite(
        STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
    )
    ent.attach_stack(st)
    return ent


def _ep():
    return Episode(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        objective="Test episode",
        outcome="It happened",
        source_type="observation",
        source_entity="test",
    )


def _save_ep(storage):
    ep = _ep()
    storage.save_episode(ep)
    return ep.id


# ==============================================================================
# SQLiteStorage.weaken_memory
# ==============================================================================


class TestWeakenMemory:
    def test_weakens_by_amount(self, storage):
        eid = _save_ep(storage)
        assert storage.weaken_memory("episode", eid, 0.3)
        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(0.7)

    def test_clamps_to_zero(self, storage):
        eid = _save_ep(storage)
        assert storage.weaken_memory("episode", eid, 2.0)
        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(0.0)

    def test_skips_protected(self, storage):
        eid = _save_ep(storage)
        storage.protect_memory("episode", eid, True)
        assert not storage.weaken_memory("episode", eid, 0.3)
        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(1.0)

    def test_nonexistent_returns_false(self, storage):
        assert not storage.weaken_memory("episode", "no-such-id", 0.3)

    def test_invalid_type_returns_false(self, storage):
        assert not storage.weaken_memory("invalid", "some-id", 0.3)

    def test_negative_amount_treated_as_positive(self, storage):
        eid = _save_ep(storage)
        storage.weaken_memory("episode", eid, -0.3)
        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(0.7)


# ==============================================================================
# SQLiteStorage.verify_memory
# ==============================================================================


class TestVerifyMemory:
    def test_boosts_strength(self, storage):
        eid = _save_ep(storage)
        storage.update_strength("episode", eid, 0.5)
        assert storage.verify_memory("episode", eid)
        ep = storage.get_episode(eid)
        assert ep.strength == pytest.approx(0.6)

    def test_increments_verification_count(self, storage):
        eid = _save_ep(storage)
        storage.verify_memory("episode", eid)
        storage.verify_memory("episode", eid)
        ep = storage.get_episode(eid)
        assert ep.verification_count == 2

    def test_caps_at_one(self, storage):
        eid = _save_ep(storage)
        # Default strength is 1.0
        storage.verify_memory("episode", eid)
        ep = storage.get_episode(eid)
        assert ep.strength <= 1.0

    def test_sets_last_verified(self, storage):
        eid = _save_ep(storage)
        storage.verify_memory("episode", eid)
        ep = storage.get_episode(eid)
        assert ep.last_verified is not None

    def test_nonexistent_returns_false(self, storage):
        assert not storage.verify_memory("episode", "no-such-id")

    def test_invalid_type_returns_false(self, storage):
        assert not storage.verify_memory("invalid", "some-id")


# ==============================================================================
# SQLiteStorage.log_audit / get_audit_log
# ==============================================================================


class TestAuditLog:
    def test_log_and_retrieve(self, storage):
        audit_id = storage.log_audit(
            "episode",
            "test-id",
            "weaken",
            "core:test",
            {"amount": 0.3, "reason": "testing"},
        )
        assert audit_id

        log = storage.get_audit_log(memory_id="test-id")
        assert len(log) == 1
        assert log[0]["operation"] == "weaken"
        assert log[0]["actor"] == "core:test"
        assert log[0]["details"]["amount"] == 0.3

    def test_filter_by_operation(self, storage):
        storage.log_audit("episode", "id1", "weaken", "core:test")
        storage.log_audit("episode", "id2", "verify", "core:test")

        log = storage.get_audit_log(operation="verify")
        assert len(log) == 1
        assert log[0]["memory_id"] == "id2"

    def test_filter_by_memory_type(self, storage):
        storage.log_audit("episode", "id1", "weaken", "core:test")
        storage.log_audit("belief", "id2", "weaken", "core:test")

        log = storage.get_audit_log(memory_type="belief")
        assert len(log) == 1
        assert log[0]["memory_id"] == "id2"

    def test_limit(self, storage):
        for i in range(10):
            storage.log_audit("episode", f"id{i}", "weaken", "core:test")

        log = storage.get_audit_log(limit=3)
        assert len(log) == 3

    def test_no_details(self, storage):
        storage.log_audit("episode", "id1", "recover", "core:test")
        log = storage.get_audit_log(memory_id="id1")
        assert log[0]["details"] is None

    def test_empty_log(self, storage):
        log = storage.get_audit_log()
        assert log == []


# ==============================================================================
# Audit trail in existing methods
# ==============================================================================


class TestAuditInExistingMethods:
    def test_forget_creates_audit(self, storage):
        eid = _save_ep(storage)
        storage.forget_memory("episode", eid, "test reason")

        log = storage.get_audit_log(memory_id=eid, operation="forget")
        assert len(log) == 1
        assert log[0]["details"]["reason"] == "test reason"

    def test_recover_creates_audit(self, storage):
        eid = _save_ep(storage)
        storage.forget_memory("episode", eid, "forgot")
        storage.recover_memory("episode", eid)

        log = storage.get_audit_log(memory_id=eid, operation="recover")
        assert len(log) == 1

    def test_protect_creates_audit(self, storage):
        eid = _save_ep(storage)
        storage.protect_memory("episode", eid, True)

        log = storage.get_audit_log(memory_id=eid, operation="protect")
        assert len(log) == 1

    def test_unprotect_creates_audit(self, storage):
        eid = _save_ep(storage)
        storage.protect_memory("episode", eid, True)
        storage.protect_memory("episode", eid, False)

        log = storage.get_audit_log(memory_id=eid, operation="unprotect")
        assert len(log) == 1

    def test_failed_forget_no_audit(self, storage):
        """Protected memories should not create forget audit."""
        eid = _save_ep(storage)
        storage.protect_memory("episode", eid, True)
        storage.forget_memory("episode", eid, "should fail")

        log = storage.get_audit_log(memory_id=eid, operation="forget")
        assert len(log) == 0

    def test_forget_deleted_memory_noop(self, storage):
        eid = _save_ep(storage)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET deleted = 1 WHERE id = ? AND stack_id = ?", (eid, STACK_ID)
            )
            conn.commit()

        assert not storage.forget_memory("episode", eid, "should be ignored")
        log = storage.get_audit_log(memory_id=eid, operation="forget")
        assert len(log) == 0

    def test_recover_deleted_memory_noop(self, storage):
        eid = _save_ep(storage)
        storage.forget_memory("episode", eid, "forgot")
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET deleted = 1 WHERE id = ? AND stack_id = ?", (eid, STACK_ID)
            )
            conn.commit()

        assert not storage.recover_memory("episode", eid)
        log = storage.get_audit_log(memory_id=eid, operation="recover")
        assert len(log) == 0

    def test_protect_deleted_memory_noop(self, storage):
        eid = _save_ep(storage)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET deleted = 1 WHERE id = ? AND stack_id = ?", (eid, STACK_ID)
            )
            conn.commit()

        assert not storage.protect_memory("episode", eid, True)
        log = storage.get_audit_log(memory_id=eid, operation="protect")
        assert len(log) == 0


# ==============================================================================
# Stack routing
# ==============================================================================


class TestStackRouting:
    def test_weaken_routes(self, stack):
        ep = _ep()
        eid = stack.save_episode(ep)
        assert stack.weaken_memory("episode", eid, 0.3)
        ep = stack.get_episodes()[0]
        assert ep.strength == pytest.approx(0.7)

    def test_verify_routes(self, stack):
        ep = _ep()
        eid = stack.save_episode(ep)
        assert stack.verify_memory("episode", eid)

    def test_log_audit_routes(self, stack):
        audit_id = stack.log_audit("episode", "test-id", "test", actor="core:test")
        assert audit_id

    def test_get_audit_log_routes(self, stack):
        stack.log_audit("episode", "test-id", "test", actor="core:test")
        log = stack.get_audit_log(memory_id="test-id")
        assert len(log) == 1


# ==============================================================================
# Entity controlled access methods
# ==============================================================================


class TestEntityWeaken:
    def test_weakens_memory(self, entity):
        eid = entity.episode("Test", "Result")
        assert entity.weaken("episode", eid, 0.3, reason="too old")
        stack = entity.active_stack
        ep = stack.get_episodes()[0]
        assert ep.strength == pytest.approx(0.7)

    def test_weaken_creates_audit(self, entity):
        eid = entity.episode("Test", "Result")
        entity.weaken("episode", eid, 0.3, reason="too old")
        stack = entity.active_stack
        log = stack.get_audit_log(memory_id=eid, operation="weaken")
        assert len(log) == 1
        assert log[0]["actor"] == "core:test-core"
        assert log[0]["details"]["amount"] == 0.3
        assert log[0]["details"]["reason"] == "too old"


class TestEntityForget:
    def test_forgets_memory(self, entity):
        eid = entity.episode("Test", "Result")
        assert entity.forget("episode", eid, "no longer relevant")
        stack = entity.active_stack
        episodes = stack.get_episodes()
        assert len(episodes) == 0

    def test_forget_creates_audit(self, entity):
        eid = entity.episode("Test", "Result")
        entity.forget("episode", eid, "no longer relevant")
        stack = entity.active_stack
        log = stack.get_audit_log(memory_id=eid, operation="forget")
        assert len(log) == 1


class TestEntityRecover:
    def test_recovers_forgotten(self, entity):
        eid = entity.episode("Test", "Result")
        entity.forget("episode", eid, "forgot")
        assert entity.recover("episode", eid)
        stack = entity.active_stack
        episodes = stack.get_episodes(include_weak=True)
        assert len(episodes) == 1
        assert episodes[0].strength == pytest.approx(0.2)

    def test_recover_creates_audit(self, entity):
        eid = entity.episode("Test", "Result")
        entity.forget("episode", eid, "forgot")
        entity.recover("episode", eid)
        stack = entity.active_stack
        log = stack.get_audit_log(memory_id=eid, operation="recover")
        assert len(log) == 1


class TestEntityVerify:
    def test_verifies_memory(self, entity):
        eid = entity.episode("Test", "Result")
        assert entity.verify("episode", eid, evidence="confirmed by observation")
        stack = entity.active_stack
        ep = stack.get_episodes()[0]
        assert ep.verification_count == 1

    def test_verify_creates_audit(self, entity):
        eid = entity.episode("Test", "Result")
        entity.verify("episode", eid, evidence="confirmed")
        stack = entity.active_stack
        log = stack.get_audit_log(memory_id=eid, operation="verify")
        assert len(log) == 1
        assert log[0]["actor"] == "core:test-core"
        assert log[0]["details"]["evidence"] == "confirmed"


class TestEntityProtect:
    def test_protects_memory(self, entity):
        eid = entity.episode("Test", "Result")
        assert entity.protect("episode", eid)
        # Try to forget — should fail
        assert not entity.forget("episode", eid, "should fail")

    def test_unprotects_memory(self, entity):
        eid = entity.episode("Test", "Result")
        entity.protect("episode", eid, True)
        entity.protect("episode", eid, False)
        # Now forget should work
        assert entity.forget("episode", eid, "now it works")

    def test_protect_creates_audit(self, entity):
        eid = entity.episode("Test", "Result")
        entity.protect("episode", eid)
        stack = entity.active_stack
        log = stack.get_audit_log(memory_id=eid, operation="protect")
        assert len(log) == 1


class TestEntityNoStack:
    def test_weaken_requires_stack(self):
        ent = Entity(core_id="test-no-stack")
        with pytest.raises(Exception):
            ent.weaken("episode", "id", 0.3)

    def test_forget_requires_stack(self):
        ent = Entity(core_id="test-no-stack")
        with pytest.raises(Exception):
            ent.forget("episode", "id", "reason")

    def test_recover_requires_stack(self):
        ent = Entity(core_id="test-no-stack")
        with pytest.raises(Exception):
            ent.recover("episode", "id")

    def test_verify_requires_stack(self):
        ent = Entity(core_id="test-no-stack")
        with pytest.raises(Exception):
            ent.verify("episode", "id")

    def test_protect_requires_stack(self):
        ent = Entity(core_id="test-no-stack")
        with pytest.raises(Exception):
            ent.protect("episode", "id")
