"""Tests for atomic update methods on goals, drives, and relationships.

Verifies that update_goal_atomic, update_drive_atomic, and
update_relationship_atomic perform optimistic concurrency control
correctly, and that writers.py uses them instead of manual
version-increment + save patterns.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import Drive, Goal, Relationship, SQLiteStorage, VersionConflictError
from tests.conftest import bind_noop_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_goal(stack_id: str, **overrides) -> Goal:
    defaults = dict(
        id=str(uuid.uuid4()),
        stack_id=stack_id,
        title="Test goal",
        description="A test goal",
        goal_type="task",
        priority="medium",
        status="active",
        created_at=datetime.now(timezone.utc),
        version=1,
    )
    defaults.update(overrides)
    return Goal(**defaults)


def _make_drive(stack_id: str, **overrides) -> Drive:
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=str(uuid.uuid4()),
        stack_id=stack_id,
        drive_type="growth",
        intensity=0.5,
        focus_areas=["learning"],
        created_at=now,
        updated_at=now,
        version=1,
    )
    defaults.update(overrides)
    return Drive(**defaults)


def _make_relationship(stack_id: str, **overrides) -> Relationship:
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=str(uuid.uuid4()),
        stack_id=stack_id,
        entity_name="other-agent",
        entity_type="agent",
        relationship_type="collaboration",
        notes="Test relationship",
        sentiment=0.5,
        interaction_count=1,
        last_interaction=now,
        created_at=now,
        version=1,
    )
    defaults.update(overrides)
    return Relationship(**defaults)


# ---------------------------------------------------------------------------
# Goal atomic update tests
# ---------------------------------------------------------------------------


class TestUpdateGoalAtomic:

    def test_increments_version(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        goal = _make_goal("test_agent")
        storage.save_goal(goal)

        # Modify and atomic-update
        goal.description = "Updated description"
        result = storage.update_goal_atomic(goal)

        assert result is True
        goals = storage.get_goals(status=None, limit=100)
        updated = next(g for g in goals if g.id == goal.id)
        assert updated.version == 2
        assert updated.description == "Updated description"

    def test_version_conflict_raises(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        goal = _make_goal("test_agent")
        storage.save_goal(goal)

        # Simulate a concurrent modification: bump version in DB directly
        with storage._connect() as conn:
            conn.execute("UPDATE goals SET version = 2 WHERE id = ?", (goal.id,))
            conn.commit()

        # Now try atomic update with stale expected_version=1
        goal.description = "Stale update"
        with pytest.raises(VersionConflictError) as exc_info:
            storage.update_goal_atomic(goal, expected_version=1)

        assert exc_info.value.expected_version == 1
        assert exc_info.value.actual_version == 2

    def test_not_found_returns_false(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        goal = _make_goal("test_agent")
        # Do NOT save the goal
        result = storage.update_goal_atomic(goal)
        assert result is False

    def test_updates_all_fields(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        goal = _make_goal("test_agent")
        storage.save_goal(goal)

        goal.status = "completed"
        goal.priority = "high"
        goal.title = "New title"
        goal.context = "project:test"
        goal.context_tags = ["tag1", "tag2"]
        storage.update_goal_atomic(goal)

        updated = next(g for g in storage.get_goals(status=None, limit=100) if g.id == goal.id)
        assert updated.status == "completed"
        assert updated.priority == "high"
        assert updated.title == "New title"
        assert updated.context == "project:test"
        assert updated.context_tags == ["tag1", "tag2"]


# ---------------------------------------------------------------------------
# Drive atomic update tests
# ---------------------------------------------------------------------------


class TestUpdateDriveAtomic:

    def test_increments_version(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        drive = _make_drive("test_agent")
        storage.save_drive(drive)

        drive.intensity = 0.9
        result = storage.update_drive_atomic(drive)

        assert result is True
        updated = storage.get_drive(drive.drive_type)
        assert updated.version == 2
        assert updated.intensity == 0.9

    def test_version_conflict_raises(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        drive = _make_drive("test_agent")
        storage.save_drive(drive)

        with storage._connect() as conn:
            conn.execute("UPDATE drives SET version = 2 WHERE id = ?", (drive.id,))
            conn.commit()

        drive.intensity = 0.8
        with pytest.raises(VersionConflictError) as exc_info:
            storage.update_drive_atomic(drive, expected_version=1)

        assert exc_info.value.expected_version == 1
        assert exc_info.value.actual_version == 2

    def test_not_found_returns_false(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        drive = _make_drive("test_agent")
        result = storage.update_drive_atomic(drive)
        assert result is False

    def test_updates_fields(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        drive = _make_drive("test_agent")
        storage.save_drive(drive)

        drive.intensity = 0.3
        drive.focus_areas = ["new-area"]
        drive.context = "project:new"
        storage.update_drive_atomic(drive)

        updated = storage.get_drive(drive.drive_type)
        assert updated.intensity == 0.3
        assert updated.focus_areas == ["new-area"]
        assert updated.context == "project:new"


# ---------------------------------------------------------------------------
# Relationship atomic update tests
# ---------------------------------------------------------------------------


class TestUpdateRelationshipAtomic:

    def test_increments_version(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship("test_agent")
        storage.save_relationship(rel)

        rel.notes = "Updated notes"
        result = storage.update_relationship_atomic(rel)

        assert result is True
        updated = storage.get_relationship(rel.entity_name)
        assert updated.version == 2
        assert updated.notes == "Updated notes"

    def test_version_conflict_raises(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship("test_agent")
        storage.save_relationship(rel)

        with storage._connect() as conn:
            conn.execute("UPDATE relationships SET version = 2 WHERE id = ?", (rel.id,))
            conn.commit()

        rel.notes = "Stale update"
        with pytest.raises(VersionConflictError) as exc_info:
            storage.update_relationship_atomic(rel, expected_version=1)

        assert exc_info.value.expected_version == 1
        assert exc_info.value.actual_version == 2

    def test_not_found_returns_false(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship("test_agent")
        result = storage.update_relationship_atomic(rel)
        assert result is False

    def test_updates_fields(self, sqlite_storage_factory, tmp_path):
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship("test_agent")
        storage.save_relationship(rel)

        rel.sentiment = -0.5
        rel.interaction_count = 5
        rel.entity_type = "organization"
        rel.context = "project:collab"
        rel.context_tags = ["important"]
        storage.update_relationship_atomic(rel)

        updated = storage.get_relationship(rel.entity_name)
        assert updated.sentiment == -0.5
        assert updated.interaction_count == 5
        assert updated.entity_type == "organization"
        assert updated.context == "project:collab"
        assert updated.context_tags == ["important"]


# ---------------------------------------------------------------------------
# Writers integration tests
# ---------------------------------------------------------------------------


class TestWritersUseAtomicMethods:
    """Verify that the writers layer calls atomic update methods end-to-end."""

    def _make_kernle(self, tmp_path):
        db_path = tmp_path / "kernle.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
        k = Kernle(
            stack_id="test_agent",
            storage=storage,
            checkpoint_dir=checkpoint_dir,
            strict=False,
        )
        bind_noop_model(k)
        return k, storage

    def test_update_goal_uses_atomic(self, tmp_path):
        k, storage = self._make_kernle(tmp_path)

        # Create a goal through the writer
        goal_id = k.goal(
            title="Test goal",
            description="Original description",
            priority="medium",
        )

        # Verify initial version
        goals = storage.get_goals(status=None, limit=100)
        goal = next(g for g in goals if g.id == goal_id)
        assert goal.version == 1

        # Update via writer (should use atomic method)
        result = k.update_goal(goal_id, description="Updated description")
        assert result is True

        # Verify version was atomically incremented
        goals = storage.get_goals(status=None, limit=100)
        updated = next(g for g in goals if g.id == goal_id)
        assert updated.version == 2
        assert updated.description == "Updated description"

        storage.close()

    def test_drive_update_uses_atomic(self, tmp_path):
        k, storage = self._make_kernle(tmp_path)

        # Create a drive
        k.drive("growth", intensity=0.5, focus_areas=["learning"])

        drive = storage.get_drive("growth")
        assert drive.version == 1

        # Update via writer (should use atomic method)
        k.drive("growth", intensity=0.8, focus_areas=["new-focus"])

        updated = storage.get_drive("growth")
        assert updated.version == 2
        assert updated.intensity == 0.8
        assert updated.focus_areas == ["new-focus"]

        storage.close()

    def test_satisfy_drive_uses_atomic(self, tmp_path):
        k, storage = self._make_kernle(tmp_path)

        k.drive("curiosity", intensity=0.7)
        drive = storage.get_drive("curiosity")
        assert drive.version == 1

        result = k.satisfy_drive("curiosity", amount=0.2)
        assert result is True

        updated = storage.get_drive("curiosity")
        assert updated.version == 2
        assert updated.intensity == pytest.approx(0.5)

        storage.close()

    def test_relationship_update_uses_atomic(self, tmp_path):
        k, storage = self._make_kernle(tmp_path)

        # Create a relationship
        rel_id = k.relationship("partner-agent", trust_level=0.7, notes="Initial")

        rel = storage.get_relationship("partner-agent")
        assert rel.version == 1

        # Update via writer (should use atomic method)
        rel_id2 = k.relationship("partner-agent", trust_level=0.9, notes="Updated")
        assert rel_id == rel_id2

        updated = storage.get_relationship("partner-agent")
        assert updated.version == 2
        assert updated.notes == "Updated"

        storage.close()
