"""Tests for source_entity persistence in Relationship CRUD (#831).

The Relationship dataclass has source_entity: Optional[str] = None and the
schema has a source_entity TEXT column, but the CRUD layer silently drops
this field on INSERT, UPDATE, and READ.

Tests verify:
- _row_to_relationship reads source_entity from the database row
- save_relationship persists source_entity on INSERT
- save_relationship persists source_entity on UPDATE
- update_relationship_atomic persists source_entity
- update_relationship_atomic preserves source_entity when not explicitly set
- Round-trip through entity.relationship() with source param
"""

import uuid
from datetime import datetime, timezone

from kernle.storage import Relationship


def _make_relationship(stack_id: str, **overrides) -> Relationship:
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=str(uuid.uuid4()),
        stack_id=stack_id,
        entity_name="test-entity",
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


class TestSaveRelationshipSourceEntity:
    """Tests for source_entity persistence in save_relationship (INSERT path)."""

    def test_save_relationship_persists_source_entity(self, sqlite_storage_factory, tmp_path):
        """Round-trip: save with source_entity, load, verify field present."""
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship(
            "test_agent",
            entity_name="Alice",
            source_entity="cli-user",
        )
        storage.save_relationship(rel)

        loaded = storage.get_relationship("Alice")
        assert loaded is not None
        assert loaded.source_entity == "cli-user"

    def test_save_relationship_without_source_entity(self, sqlite_storage_factory, tmp_path):
        """Saving without source_entity stores None (default)."""
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship("test_agent", entity_name="Bob")
        storage.save_relationship(rel)

        loaded = storage.get_relationship("Bob")
        assert loaded is not None
        assert loaded.source_entity is None


class TestSaveRelationshipUpdateSourceEntity:
    """Tests for source_entity persistence in save_relationship (UPDATE path)."""

    def test_update_relationship_persists_source_entity(self, sqlite_storage_factory, tmp_path):
        """Updating an existing relationship with source_entity persists it."""
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship(
            "test_agent",
            entity_name="Carol",
            source_entity="original-caller",
        )
        storage.save_relationship(rel)

        # Load, modify source_entity, save again (triggers UPDATE path)
        updated = storage.get_relationship("Carol")
        updated.source_entity = "new-caller"
        storage.save_relationship(updated)

        reloaded = storage.get_relationship("Carol")
        assert reloaded is not None
        assert reloaded.source_entity == "new-caller"


class TestUpdateRelationshipAtomicSourceEntity:
    """Tests for source_entity persistence in update_relationship_atomic."""

    def test_update_relationship_atomic_persists_source_entity(
        self, sqlite_storage_factory, tmp_path
    ):
        """update_relationship_atomic with source_entity persists it."""
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship("test_agent", entity_name="Dave")
        storage.save_relationship(rel)

        rel.source_entity = "updated-source"
        storage.update_relationship_atomic(rel)

        loaded = storage.get_relationship("Dave")
        assert loaded is not None
        assert loaded.source_entity == "updated-source"

    def test_update_relationship_preserves_source_entity_when_not_set(
        self, sqlite_storage_factory, tmp_path
    ):
        """Updating without changing source_entity preserves existing value."""
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        rel = _make_relationship(
            "test_agent",
            entity_name="Eve",
            source_entity="original-source",
        )
        storage.save_relationship(rel)

        # Load, change something else, save via atomic update
        loaded = storage.get_relationship("Eve")
        loaded.notes = "Updated notes only"
        storage.update_relationship_atomic(loaded)

        reloaded = storage.get_relationship("Eve")
        assert reloaded is not None
        assert reloaded.source_entity == "original-source"
        assert reloaded.notes == "Updated notes only"


class TestRowToRelationshipSourceEntity:
    """Tests for _row_to_relationship reading source_entity from the row."""

    def test_row_to_relationship_reads_source_entity(self, sqlite_storage_factory, tmp_path):
        """Verify deserialization includes source_entity from row."""
        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        # Write directly to DB to prove the read path works
        rel_id = str(uuid.uuid4())
        with storage._connect() as conn:
            conn.execute(
                """
                INSERT INTO relationships
                (id, stack_id, entity_name, entity_type, relationship_type,
                 sentiment, interaction_count, created_at,
                 source_entity,
                 local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_id,
                    "test_agent",
                    "DirectInsert",
                    "person",
                    "test",
                    0.0,
                    0,
                    datetime.now(timezone.utc).isoformat(),
                    "direct-source",
                    datetime.now(timezone.utc).isoformat(),
                    1,
                    0,
                ),
            )
            conn.commit()

        loaded = storage.get_relationship("DirectInsert")
        assert loaded is not None
        assert loaded.source_entity == "direct-source"


class TestEntityRelationshipSourceEntityRoundTrip:
    """Tests for Entity.relationship() with source param round-trip."""

    def test_entity_relationship_source_entity_round_trip(self, sqlite_storage_factory, tmp_path):
        """Entity.relationship() with source param -> source_entity persists through write+read."""
        from kernle.core import Kernle

        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        k = Kernle(
            stack_id="test_agent",
            storage=storage,
            checkpoint_dir=checkpoint_dir,
            strict=False,
        )

        # Create relationship with source_entity
        r_id = k.relationship(
            "RoundTripEntity",
            entity_type="agent",
            source_entity="test-caller",
        )
        assert r_id is not None

        # Verify source_entity was persisted
        loaded = storage.get_relationship("RoundTripEntity")
        assert loaded is not None
        assert loaded.source_entity == "test-caller"

    def test_entity_relationship_update_source_entity_round_trip(
        self, sqlite_storage_factory, tmp_path
    ):
        """Updating via Entity.relationship() with source_entity persists the new value."""
        from kernle.core import Kernle

        storage = sqlite_storage_factory(stack_id="test_agent", db_path=tmp_path / "test.db")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        k = Kernle(
            stack_id="test_agent",
            storage=storage,
            checkpoint_dir=checkpoint_dir,
            strict=False,
        )

        # Create, then update with new source_entity
        k.relationship("UpdateEntity", entity_type="person")
        k.relationship("UpdateEntity", source_entity="updated-caller")

        loaded = storage.get_relationship("UpdateEntity")
        assert loaded is not None
        assert loaded.source_entity == "updated-caller"
