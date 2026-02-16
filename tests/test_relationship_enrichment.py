"""Tests for relationship enrichment: history tracking and entity models.

Tests cover:
- RelationshipHistoryEntry dataclass
- EntityModel dataclass
- Write-on-change history logging in save_relationship()
- Relationship history retrieval
- Entity model CRUD operations
- Core API methods (get_relationship_history, add_entity_model, etc.)
- CLI command handlers
- Relationship source_type and source_entity provenance
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import EntityModel, Relationship, RelationshipHistoryEntry
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import SourceType

# === Fixtures ===


@pytest.fixture
def storage(tmp_path):
    """SQLite storage instance for testing."""
    s = SQLiteStorage(stack_id="test_agent", db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def kernle_with_storage(tmp_path):
    """Kernle instance with SQLite storage."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    s = SQLiteStorage(stack_id="test_agent", db_path=tmp_path / "test.db")
    k = Kernle(stack_id="test_agent", storage=s, checkpoint_dir=checkpoint_dir, strict=False)
    yield k, s
    s.close()


@pytest.fixture
def relationship(storage):
    """Create and save a test relationship, return it."""
    rel = Relationship(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        entity_name="Alice",
        entity_type="person",
        relationship_type="colleague",
        notes="Works on the same team",
        sentiment=0.5,
        interaction_count=3,
        last_interaction=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )
    storage.save_relationship(rel)
    return rel


# === Dataclass Tests ===


class TestRelationshipHistoryEntry:
    def test_create_entry(self):
        entry = RelationshipHistoryEntry(
            id="test-id",
            stack_id="agent-1",
            relationship_id="rel-1",
            entity_name="Alice",
            event_type="trust_change",
            old_value='{"sentiment": 0.0}',
            new_value='{"sentiment": 0.5}',
            created_at=datetime.now(timezone.utc),
        )
        assert entry.event_type == "trust_change"
        assert entry.entity_name == "Alice"
        assert entry.old_value == '{"sentiment": 0.0}'

    def test_defaults(self):
        entry = RelationshipHistoryEntry(
            id="test-id",
            stack_id="agent-1",
            relationship_id="rel-1",
            entity_name="Bob",
            event_type="interaction",
        )
        assert entry.old_value is None
        assert entry.new_value is None
        assert entry.episode_id is None
        assert entry.notes is None
        assert entry.version == 1
        assert entry.deleted is False


class TestEntityModel:
    def test_create_model(self):
        model = EntityModel(
            id="model-1",
            stack_id="agent-1",
            entity_name="Alice",
            model_type="behavioral",
            observation="Alice prefers detailed explanations",
            confidence=0.8,
        )
        assert model.model_type == "behavioral"
        assert model.observation == "Alice prefers detailed explanations"
        assert model.confidence == 0.8

    def test_defaults(self):
        model = EntityModel(
            id="model-1",
            stack_id="agent-1",
            entity_name="Bob",
            model_type="preference",
            observation="Prefers concise output",
        )
        assert model.confidence == 0.7
        assert model.source_episodes is None
        assert model.subject_ids is None
        assert model.version == 1
        assert model.deleted is False


# === Write-on-Change History Tests ===


class TestWriteOnChangeHistory:
    def test_trust_change_logged(self, storage, relationship):
        """Updating sentiment should log a trust_change event."""
        # Update the relationship with a new sentiment
        rel = storage.get_relationship("Alice")
        rel.sentiment = 0.8
        storage.save_relationship(rel)

        history = storage.get_relationship_history("Alice")
        trust_events = [e for e in history if e.event_type == "trust_change"]
        assert len(trust_events) >= 1

        event = trust_events[0]
        old = json.loads(event.old_value)
        new = json.loads(event.new_value)
        assert old["sentiment"] == 0.5
        assert new["sentiment"] == 0.8

    def test_type_change_logged(self, storage, relationship):
        """Changing relationship_type should log a type_change event."""
        rel = storage.get_relationship("Alice")
        rel.relationship_type = "friend"
        storage.save_relationship(rel)

        history = storage.get_relationship_history("Alice")
        type_events = [e for e in history if e.event_type == "type_change"]
        assert len(type_events) >= 1

        event = type_events[0]
        old = json.loads(event.old_value)
        new = json.loads(event.new_value)
        assert old["relationship_type"] == "colleague"
        assert new["relationship_type"] == "friend"

    def test_note_change_logged(self, storage, relationship):
        """Changing notes should log a note event."""
        rel = storage.get_relationship("Alice")
        rel.notes = "Now manages the project"
        storage.save_relationship(rel)

        history = storage.get_relationship_history("Alice")
        note_events = [e for e in history if e.event_type == "note"]
        assert len(note_events) >= 1

        event = note_events[0]
        new = json.loads(event.new_value)
        assert new["notes"] == "Now manages the project"

    def test_interaction_logged(self, storage, relationship):
        """Incrementing interaction_count should log an interaction event."""
        rel = storage.get_relationship("Alice")
        rel.interaction_count = 4
        storage.save_relationship(rel)

        history = storage.get_relationship_history("Alice")
        interaction_events = [e for e in history if e.event_type == "interaction"]
        assert len(interaction_events) >= 1

    def test_no_change_no_history(self, storage, relationship):
        """Saving a relationship without changes should not log history."""
        rel = storage.get_relationship("Alice")
        # Save without any changes
        storage.save_relationship(rel)

        history = storage.get_relationship_history("Alice")
        # Should be empty since nothing changed
        assert len(history) == 0

    def test_new_relationship_no_history(self, storage):
        """Creating a new relationship should not log history."""
        rel = Relationship(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity_name="Bob",
            entity_type="person",
            relationship_type="acquaintance",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_relationship(rel)

        history = storage.get_relationship_history("Bob")
        assert len(history) == 0

    def test_multiple_changes_logged_separately(self, storage, relationship):
        """Multiple field changes in one save should create separate events."""
        rel = storage.get_relationship("Alice")
        rel.sentiment = 0.9
        rel.relationship_type = "mentor"
        rel.notes = "Great mentor"
        rel.interaction_count = 5
        storage.save_relationship(rel)

        history = storage.get_relationship_history("Alice")
        event_types = {e.event_type for e in history}
        assert "trust_change" in event_types
        assert "type_change" in event_types
        assert "note" in event_types
        assert "interaction" in event_types


# === Relationship History Storage Tests ===


class TestRelationshipHistoryStorage:
    def test_save_and_get_history(self, storage, relationship):
        """Test direct save and retrieval of history entries."""
        entry = RelationshipHistoryEntry(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            relationship_id=relationship.id,
            entity_name="Alice",
            event_type="interaction",
            new_value=json.dumps({"interaction_count": 4}),
            created_at=datetime.now(timezone.utc),
        )
        storage.save_relationship_history(entry)

        history = storage.get_relationship_history("Alice")
        assert len(history) == 1
        assert history[0].event_type == "interaction"

    def test_filter_by_event_type(self, storage, relationship):
        """Test filtering history by event type."""
        for event_type in ["interaction", "trust_change", "interaction"]:
            entry = RelationshipHistoryEntry(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                relationship_id=relationship.id,
                entity_name="Alice",
                event_type=event_type,
                created_at=datetime.now(timezone.utc),
            )
            storage.save_relationship_history(entry)

        all_history = storage.get_relationship_history("Alice")
        assert len(all_history) == 3

        interactions = storage.get_relationship_history("Alice", event_type="interaction")
        assert len(interactions) == 2

        trust = storage.get_relationship_history("Alice", event_type="trust_change")
        assert len(trust) == 1

    def test_history_limit(self, storage, relationship):
        """Test history retrieval limit."""
        for i in range(10):
            entry = RelationshipHistoryEntry(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                relationship_id=relationship.id,
                entity_name="Alice",
                event_type="interaction",
                created_at=datetime.now(timezone.utc),
            )
            storage.save_relationship_history(entry)

        history = storage.get_relationship_history("Alice", limit=5)
        assert len(history) == 5

    def test_history_ordered_by_created_at_desc(self, storage, relationship):
        """Test that history is returned most recent first."""
        for i in range(3):
            entry = RelationshipHistoryEntry(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                relationship_id=relationship.id,
                entity_name="Alice",
                event_type="interaction",
                notes=f"event-{i}",
                created_at=datetime.now(timezone.utc),
            )
            storage.save_relationship_history(entry)

        history = storage.get_relationship_history("Alice")
        assert len(history) == 3
        # Most recent should have highest notes number
        assert history[0].notes == "event-2"

    def test_empty_history(self, storage):
        """Test getting history for non-existent entity."""
        history = storage.get_relationship_history("NonExistent")
        assert history == []


# === Entity Model Storage Tests ===


class TestEntityModelStorage:
    def test_save_and_get_model(self, storage):
        """Test saving and retrieving an entity model."""
        model = EntityModel(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity_name="Alice",
            model_type="behavioral",
            observation="Alice prefers structured communication",
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
        )
        model_id = storage.save_entity_model(model)

        retrieved = storage.get_entity_model(model_id)
        assert retrieved is not None
        assert retrieved.entity_name == "Alice"
        assert retrieved.model_type == "behavioral"
        assert retrieved.observation == "Alice prefers structured communication"
        assert retrieved.confidence == 0.8

    def test_auto_populate_subject_ids(self, storage):
        """Test that subject_ids is auto-populated from entity_name."""
        model = EntityModel(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            entity_name="Bob",
            model_type="preference",
            observation="Prefers short responses",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_entity_model(model)

        retrieved = storage.get_entity_model(model.id)
        assert retrieved.subject_ids == ["Bob"]

    def test_list_all_models(self, storage):
        """Test listing all entity models."""
        for name, mtype in [("Alice", "behavioral"), ("Bob", "preference")]:
            model = EntityModel(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                entity_name=name,
                model_type=mtype,
                observation=f"Observation about {name}",
                created_at=datetime.now(timezone.utc),
            )
            storage.save_entity_model(model)

        models = storage.get_entity_models()
        assert len(models) == 2

    def test_filter_by_entity_name(self, storage):
        """Test filtering models by entity name."""
        for name in ["Alice", "Alice", "Bob"]:
            model = EntityModel(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                entity_name=name,
                model_type="behavioral",
                observation=f"About {name}",
                created_at=datetime.now(timezone.utc),
            )
            storage.save_entity_model(model)

        alice_models = storage.get_entity_models(entity_name="Alice")
        assert len(alice_models) == 2

    def test_filter_by_model_type(self, storage):
        """Test filtering models by model type."""
        for mtype in ["behavioral", "preference", "behavioral"]:
            model = EntityModel(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                entity_name="Alice",
                model_type=mtype,
                observation=f"A {mtype} observation",
                created_at=datetime.now(timezone.utc),
            )
            storage.save_entity_model(model)

        behavioral = storage.get_entity_models(model_type="behavioral")
        assert len(behavioral) == 2

    def test_model_limit(self, storage):
        """Test model retrieval limit."""
        for i in range(10):
            model = EntityModel(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                entity_name="Alice",
                model_type="behavioral",
                observation=f"Observation {i}",
                created_at=datetime.now(timezone.utc),
            )
            storage.save_entity_model(model)

        models = storage.get_entity_models(limit=5)
        assert len(models) == 5

    def test_get_nonexistent_model(self, storage):
        """Test getting a non-existent model."""
        result = storage.get_entity_model("non-existent-id")
        assert result is None


# === Core API Tests ===


class TestCoreRelationshipHistory:
    def test_get_relationship_history(self, kernle_with_storage):
        """Test getting relationship history via Kernle API."""
        k, s = kernle_with_storage

        # Create and update a relationship
        k.relationship("Alice", trust_level=0.5, entity_type="person")
        k.relationship("Alice", trust_level=0.8)

        history = k.get_relationship_history("Alice")
        assert len(history) > 0

        # Should have a trust_change entry
        trust_changes = [e for e in history if e["event_type"] == "trust_change"]
        assert len(trust_changes) >= 1

    def test_get_history_with_event_type_filter(self, kernle_with_storage):
        """Test filtering history by event type."""
        k, s = kernle_with_storage

        k.relationship("Alice", trust_level=0.5, entity_type="person")
        k.relationship("Alice", trust_level=0.8)
        k.relationship("Alice", notes="Updated notes")

        trust_history = k.get_relationship_history("Alice", event_type="trust_change")
        note_history = k.get_relationship_history("Alice", event_type="note")

        assert all(e["event_type"] == "trust_change" for e in trust_history)
        assert all(e["event_type"] == "note" for e in note_history)

    def test_history_returns_dicts(self, kernle_with_storage):
        """Test that history entries are returned as dicts."""
        k, s = kernle_with_storage

        k.relationship("Alice", trust_level=0.5, entity_type="person")
        k.relationship("Alice", trust_level=0.8)

        history = k.get_relationship_history("Alice")
        assert len(history) > 0

        entry = history[0]
        assert isinstance(entry, dict)
        assert "id" in entry
        assert "event_type" in entry
        assert "created_at" in entry


class TestCoreEntityModels:
    def test_add_entity_model(self, kernle_with_storage):
        """Test adding an entity model via Kernle API."""
        k, s = kernle_with_storage

        model_id = k.add_entity_model(
            entity_name="Alice",
            model_type="behavioral",
            observation="Alice prefers morning meetings",
            confidence=0.8,
        )
        assert model_id is not None
        assert len(model_id) > 0

    def test_get_entity_models(self, kernle_with_storage):
        """Test listing entity models via Kernle API."""
        k, s = kernle_with_storage

        k.add_entity_model("Alice", "behavioral", "Prefers email")
        k.add_entity_model("Alice", "preference", "Likes dark mode")
        k.add_entity_model("Bob", "capability", "Expert in Python")

        all_models = k.get_entity_models()
        assert len(all_models) == 3

        alice_models = k.get_entity_models(entity_name="Alice")
        assert len(alice_models) == 2

        behavioral = k.get_entity_models(model_type="behavioral")
        assert len(behavioral) == 1

    def test_get_entity_model(self, kernle_with_storage):
        """Test getting a specific entity model."""
        k, s = kernle_with_storage

        model_id = k.add_entity_model("Alice", "behavioral", "Responds quickly to urgent items")

        model = k.get_entity_model(model_id)
        assert model is not None
        assert model["entity_name"] == "Alice"
        assert model["model_type"] == "behavioral"
        assert model["observation"] == "Responds quickly to urgent items"

    def test_get_nonexistent_model(self, kernle_with_storage):
        """Test getting a non-existent entity model."""
        k, s = kernle_with_storage
        result = k.get_entity_model("non-existent-id")
        assert result is None

    def test_invalid_model_type(self, kernle_with_storage):
        """Test that invalid model types raise ValueError."""
        k, s = kernle_with_storage

        with pytest.raises(ValueError, match="Invalid model_type"):
            k.add_entity_model("Alice", "invalid_type", "Some observation")

    def test_confidence_clamped(self, kernle_with_storage):
        """Test that confidence is clamped to 0.0-1.0."""
        k, s = kernle_with_storage

        model_id = k.add_entity_model("Alice", "behavioral", "Test", confidence=1.5)
        model = k.get_entity_model(model_id)
        assert model["confidence"] == 1.0

        model_id2 = k.add_entity_model("Alice", "behavioral", "Test2", confidence=-0.5)
        model2 = k.get_entity_model(model_id2)
        assert model2["confidence"] == 0.0

    def test_model_returns_dict(self, kernle_with_storage):
        """Test that entity models are returned as dicts."""
        k, s = kernle_with_storage

        model_id = k.add_entity_model("Alice", "behavioral", "Observation")
        model = k.get_entity_model(model_id)

        assert isinstance(model, dict)
        assert "id" in model
        assert "entity_name" in model
        assert "model_type" in model
        assert "observation" in model
        assert "confidence" in model
        assert "created_at" in model

    def test_source_episodes_stored(self, kernle_with_storage):
        """Test that source episodes are stored correctly."""
        k, s = kernle_with_storage

        ep_ids = ["ep-1", "ep-2"]
        model_id = k.add_entity_model(
            "Alice",
            "behavioral",
            "Observation",
            source_episodes=ep_ids,
        )
        model = k.get_entity_model(model_id)
        assert model["source_episodes"] == ep_ids


# === Relationship Source Provenance Tests ===


class TestRelationshipSourceProvenance:
    """Tests for source_type and source_entity on relationship writes."""

    def test_new_relationship_explicit_external_source_type(self, kernle_with_storage):
        """New relationship with explicit source_type=EXTERNAL stores 'external'."""
        k, s = kernle_with_storage

        k.relationship(
            "ExternalBot",
            entity_type="agent",
            source_type=SourceType.EXTERNAL,
        )
        rel = s.get_relationship("ExternalBot")
        assert rel is not None
        assert rel.source_type == "external"

    def test_new_relationship_explicit_source_entity_accepted(self, kernle_with_storage):
        """Relationship writer accepts source_entity parameter without error."""
        k, s = kernle_with_storage

        # Verify source_entity parameter is accepted
        r_id = k.relationship(
            "NamedCaller",
            entity_type="agent",
            source_entity="test-caller",
        )
        assert isinstance(r_id, str)
        assert len(r_id) > 0

    def test_new_relationship_default_source_type(self, kernle_with_storage):
        """New relationship without explicit source_type defaults to 'direct_experience'."""
        k, s = kernle_with_storage

        k.relationship("DefaultPerson", entity_type="person")
        rel = s.get_relationship("DefaultPerson")
        assert rel is not None
        assert rel.source_type == "direct_experience"

    def test_updated_relationship_preserves_source_type_when_not_passed(self, kernle_with_storage):
        """Updating a relationship without source_type keeps the existing value."""
        k, s = kernle_with_storage

        k.relationship(
            "PreserveTest",
            entity_type="person",
            source_type=SourceType.EXTERNAL,
        )
        rel_before = s.get_relationship("PreserveTest")
        assert rel_before.source_type == "external"

        # Update without passing source_type
        k.relationship("PreserveTest", trust_level=0.9)

        rel_after = s.get_relationship("PreserveTest")
        assert rel_after.source_type == "external"

    def test_updated_relationship_changes_source_type_when_passed(self, kernle_with_storage):
        """Updating a relationship with explicit source_type changes the stored value."""
        k, s = kernle_with_storage

        k.relationship(
            "ChangeTest",
            entity_type="person",
            source_type=SourceType.DIRECT_EXPERIENCE,
        )
        rel_before = s.get_relationship("ChangeTest")
        assert rel_before.source_type == "direct_experience"

        # Update with new source_type
        k.relationship("ChangeTest", source_type=SourceType.EXTERNAL)

        rel_after = s.get_relationship("ChangeTest")
        assert rel_after.source_type == "external"

    def test_updated_relationship_source_entity_param_accepted(self, kernle_with_storage):
        """Updating a relationship with source_entity is accepted without error."""
        k, s = kernle_with_storage

        k.relationship(
            "EntityChangeTest",
            entity_type="person",
        )

        # Update with source_entity should not raise
        r_id = k.relationship("EntityChangeTest", source_entity="new-caller")
        assert isinstance(r_id, str)

    def test_new_relationship_string_source_type(self, kernle_with_storage):
        """New relationship with string source_type normalizes correctly."""
        k, s = kernle_with_storage

        k.relationship(
            "StringType",
            entity_type="person",
            source_type="observation",
        )
        rel = s.get_relationship("StringType")
        assert rel is not None
        assert rel.source_type == "observation"

    def test_new_relationship_invalid_source_type_raises(self, kernle_with_storage):
        """Invalid source_type raises ValueError."""
        k, s = kernle_with_storage

        with pytest.raises(ValueError, match="Invalid source_type"):
            k.relationship(
                "BadType",
                entity_type="person",
                source_type="completely_invalid",
            )
