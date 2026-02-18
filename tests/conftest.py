"""
Pytest fixtures and test configuration for Kernle tests.

Updated to work with the storage abstraction layer.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import Belief, Drive, Episode, Goal, Note, SQLiteStorage, Value


def bind_noop_model(k: Kernle) -> None:
    """Bind a no-op model to a Kernle instance for testing.

    This satisfies the inference gate on higher-tier writers without
    requiring actual inference capability.
    """
    from kernle.importers.import_model import bind_import_model

    bind_import_model(k)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint files."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary SQLite database path."""
    return tmp_path / "test_memories.db"


@pytest.fixture
def sqlite_storage(temp_db_path):
    """SQLite storage instance for testing."""
    storage = SQLiteStorage(
        stack_id="test_agent",
        db_path=temp_db_path,
    )
    yield storage
    storage.close()


@pytest.fixture
def sqlite_storage_factory():
    """Factory for SQLiteStorage with centralized teardown.

    Useful in high-duplication test modules that need custom stack IDs
    or DB filenames while keeping fixture cleanup consistent.
    """
    storages = []

    def _create(*, stack_id: str, db_path):
        storage = SQLiteStorage(stack_id=stack_id, db_path=db_path)
        storages.append(storage)
        return storage

    yield _create

    for storage in storages:
        storage.close()


@pytest.fixture
def kernle_instance(temp_checkpoint_dir, temp_db_path):
    """Kernle instance with SQLite storage for testing."""
    storage = SQLiteStorage(
        stack_id="test_agent",
        db_path=temp_db_path,
    )

    kernle = Kernle(
        stack_id="test_agent", storage=storage, checkpoint_dir=temp_checkpoint_dir, strict=False
    )
    bind_noop_model(kernle)

    yield kernle, storage
    storage.close()


@pytest.fixture
def sample_episode():
    """Sample episode for testing."""
    return Episode(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        objective="Complete unit tests for Kernle",
        outcome="All tests passing with good coverage",
        outcome_type="success",
        lessons=["Always test edge cases", "Mock external dependencies"],
        tags=["testing", "development"],
        created_at=datetime.now(timezone.utc),
        confidence=0.9,
    )


@pytest.fixture
def sample_note():
    """Sample note for testing."""
    return Note(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        content="**Decision**: Use pytest for testing framework",
        note_type="decision",
        reason="Industry standard with good plugin ecosystem",
        tags=["testing"],
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_belief():
    """Sample belief for testing."""
    return Belief(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        statement="Comprehensive testing leads to more reliable software",
        belief_type="fact",
        confidence=0.9,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_value():
    """Sample value for testing."""
    return Value(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        name="Quality",
        statement="Software should be thoroughly tested and reliable",
        priority=80,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_goal():
    """Sample goal for testing."""
    return Goal(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        title="Achieve 80%+ test coverage",
        description="Write comprehensive tests for the entire Kernle system",
        priority="high",
        status="active",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_drive():
    """Sample drive for testing."""
    return Drive(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        drive_type="growth",
        intensity=0.7,
        focus_areas=["learning", "improvement"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def populated_storage(
    kernle_instance,
    sample_episode,
    sample_note,
    sample_belief,
    sample_value,
    sample_goal,
    sample_drive,
):
    """Populate the kernle_instance storage with sample data.

    This fixture depends on kernle_instance and populates its storage.
    Use both fixtures together: (kernle_instance, populated_storage)
    """
    kernle, storage = kernle_instance

    # Save sample data
    storage.save_episode(sample_episode)
    storage.save_note(sample_note)
    storage.save_belief(sample_belief)
    storage.save_value(sample_value)
    storage.save_goal(sample_goal)
    storage.save_drive(sample_drive)

    # Add some additional test data
    # Episode without lessons (not reflected)
    unreflected_episode = Episode(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        objective="Debug memory leak",
        outcome="Could not reproduce the issue",
        outcome_type="failure",
        lessons=["Need better monitoring tools"],
        tags=["debugging"],
        created_at=datetime.now(timezone.utc),
    )
    storage.save_episode(unreflected_episode)

    # Checkpoint episode (should be filtered from recent work)
    checkpoint_episode = Episode(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        objective="Implement caching",
        outcome="Basic caching implemented, optimization needed",
        outcome_type="partial",
        lessons=["Start simple, then optimize"],
        tags=["checkpoint"],
        created_at=datetime.now(timezone.utc),
    )
    storage.save_episode(checkpoint_episode)

    # Additional note
    insight_note = Note(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        content="**Insight**: Mocking is crucial for isolated testing",
        note_type="insight",
        tags=["testing"],
        created_at=datetime.now(timezone.utc),
    )
    storage.save_note(insight_note)

    return storage
