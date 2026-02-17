"""Relationships CRUD operations extracted from SQLiteStorage.

Handles relationship management, relationship history tracking,
and entity models. All functions receive dependencies explicitly
(connection factory, serializers, sync callbacks) to avoid circular
imports and enable independent testing.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from .base import (
    EntityModel,
    Relationship,
    RelationshipHistoryEntry,
)
from .memory_crud import _row_to_entity_model as _mc_row_to_entity_model
from .memory_crud import _row_to_relationship as _mc_row_to_relationship
from .memory_crud import _row_to_relationship_history as _mc_row_to_relationship_history

logger = logging.getLogger(__name__)


def _log_relationship_changes(
    conn: Any,
    stack_id: str,
    existing_row: sqlite3.Row,
    new_rel: Relationship,
    now: str,
) -> None:
    """Detect changes between existing and new relationship, log history entries."""
    rel_id = existing_row["id"]
    entity_name = existing_row["entity_name"]

    # Check sentiment/trust change
    old_sentiment = existing_row["sentiment"]
    if new_rel.sentiment != old_sentiment:
        entry_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO relationship_history
            (id, stack_id, relationship_id, entity_name, event_type,
             old_value, new_value, notes, created_at,
             local_updated_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry_id,
                stack_id,
                rel_id,
                entity_name,
                "trust_change",
                json.dumps({"sentiment": old_sentiment}),
                json.dumps({"sentiment": new_rel.sentiment}),
                None,
                now,
                now,
                1,
                0,
            ),
        )

    # Check relationship_type change
    old_type = existing_row["relationship_type"]
    if new_rel.relationship_type != old_type:
        entry_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO relationship_history
            (id, stack_id, relationship_id, entity_name, event_type,
             old_value, new_value, notes, created_at,
             local_updated_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry_id,
                stack_id,
                rel_id,
                entity_name,
                "type_change",
                json.dumps({"relationship_type": old_type}),
                json.dumps({"relationship_type": new_rel.relationship_type}),
                None,
                now,
                now,
                1,
                0,
            ),
        )

    # Check notes change
    old_notes = existing_row["notes"]
    if new_rel.notes != old_notes and new_rel.notes is not None:
        entry_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO relationship_history
            (id, stack_id, relationship_id, entity_name, event_type,
             old_value, new_value, notes, created_at,
             local_updated_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry_id,
                stack_id,
                rel_id,
                entity_name,
                "note",
                json.dumps({"notes": old_notes}) if old_notes else None,
                json.dumps({"notes": new_rel.notes}),
                None,
                now,
                now,
                1,
                0,
            ),
        )

    # Check interaction count change (log interaction event)
    old_count = existing_row["interaction_count"]
    if new_rel.interaction_count > old_count:
        entry_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO relationship_history
            (id, stack_id, relationship_id, entity_name, event_type,
             old_value, new_value, notes, created_at,
             local_updated_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry_id,
                stack_id,
                rel_id,
                entity_name,
                "interaction",
                json.dumps({"interaction_count": old_count}),
                json.dumps({"interaction_count": new_rel.interaction_count}),
                None,
                now,
                now,
                1,
                0,
            ),
        )


def save_relationship(
    connect_fn: Callable,
    stack_id: str,
    relationship: Relationship,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    sync_to_file: Callable,
    *,
    lineage_checker: Optional[Callable] = None,
) -> str:
    """Save or update a relationship. Logs history on changes.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        relationship: The Relationship to save.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation (conn, table, id, op, data).
        sync_to_file: Syncs relationships to flat file (no args).
        lineage_checker: Optional callable for derived-from cycle detection.
            Receives (storage, memory_type, id, derived_from).
    """
    if not relationship.id:
        relationship.id = str(uuid.uuid4())

    if relationship.derived_from and lineage_checker:
        lineage_checker("relationship", relationship.id, relationship.derived_from)

    now = now_fn()

    with connect_fn() as conn:
        # Check if exists - fetch full row for change detection
        existing = conn.execute(
            "SELECT * FROM relationships WHERE stack_id = ? AND entity_name = ?",
            (stack_id, relationship.entity_name),
        ).fetchone()

        if existing:
            relationship.id = existing["id"]

            # Detect changes and log history
            _log_relationship_changes(conn, stack_id, existing, relationship, now)

            conn.execute(
                """
                UPDATE relationships SET
                    entity_type = ?, relationship_type = ?, notes = ?,
                    sentiment = ?, interaction_count = ?, last_interaction = ?,
                    confidence = ?, source_type = ?, source_entity = ?,
                    source_episodes = ?,
                    derived_from = ?, last_verified = ?, verification_count = ?,
                    confidence_history = ?, context = ?, context_tags = ?,
                    subject_ids = ?, access_grants = ?, consent_grants = ?,
                    local_updated_at = ?, version = version + 1
                WHERE id = ?
            """,
                (
                    relationship.entity_type,
                    relationship.relationship_type,
                    relationship.notes,
                    relationship.sentiment,
                    relationship.interaction_count,
                    (
                        relationship.last_interaction.isoformat()
                        if relationship.last_interaction
                        else None
                    ),
                    relationship.confidence,
                    relationship.source_type,
                    relationship.source_entity,
                    to_json(relationship.source_episodes),
                    to_json(relationship.derived_from),
                    (
                        relationship.last_verified.isoformat()
                        if relationship.last_verified
                        else None
                    ),
                    relationship.verification_count,
                    to_json(relationship.confidence_history),
                    relationship.context,
                    to_json(relationship.context_tags),
                    to_json(getattr(relationship, "subject_ids", None)),
                    to_json(getattr(relationship, "access_grants", None)),
                    to_json(getattr(relationship, "consent_grants", None)),
                    now,
                    relationship.id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO relationships
                (id, stack_id, entity_name, entity_type, relationship_type, notes,
                 sentiment, interaction_count, last_interaction, created_at,
                 confidence, source_type, source_entity, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 strength,
                 context, context_tags,
                 subject_ids, access_grants, consent_grants,
                 epoch_id,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    relationship.id,
                    stack_id,
                    relationship.entity_name,
                    relationship.entity_type,
                    relationship.relationship_type,
                    relationship.notes,
                    relationship.sentiment,
                    relationship.interaction_count,
                    (
                        relationship.last_interaction.isoformat()
                        if relationship.last_interaction
                        else None
                    ),
                    now,
                    relationship.confidence,
                    relationship.source_type,
                    relationship.source_entity,
                    to_json(relationship.source_episodes),
                    to_json(relationship.derived_from),
                    (
                        relationship.last_verified.isoformat()
                        if relationship.last_verified
                        else None
                    ),
                    relationship.verification_count,
                    to_json(relationship.confidence_history),
                    relationship.strength,
                    relationship.context,
                    to_json(relationship.context_tags),
                    to_json(getattr(relationship, "subject_ids", None)),
                    to_json(getattr(relationship, "access_grants", None)),
                    to_json(getattr(relationship, "consent_grants", None)),
                    relationship.epoch_id,
                    now,
                    None,
                    1,
                    0,
                ),
            )

        # Queue for sync with record data
        relationship_data = to_json(record_to_dict(relationship))
        queue_sync(conn, "relationships", relationship.id, "upsert", data=relationship_data)
        conn.commit()

    # Sync to flat file
    sync_to_file()

    return relationship.id


def get_relationships(
    connect_fn: Callable,
    stack_id: str,
    build_access_filter: Callable,
    entity_type: Optional[str] = None,
    requesting_entity: Optional[str] = None,
) -> List[Relationship]:
    """Get relationships, optionally filtered by entity_type."""
    query = "SELECT * FROM relationships WHERE stack_id = ? AND deleted = 0"
    params: List[Any] = [stack_id]

    if entity_type:
        query += " AND entity_type = ?"
        params.append(entity_type)

    access_filter, access_params = build_access_filter(requesting_entity)
    query += access_filter
    params.extend(access_params)

    with connect_fn() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_mc_row_to_relationship(row) for row in rows]


def get_relationship(
    connect_fn: Callable,
    stack_id: str,
    entity_name: str,
) -> Optional[Relationship]:
    """Get a specific relationship by entity_name."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM relationships WHERE stack_id = ? AND entity_name = ? AND deleted = 0",
            (stack_id, entity_name),
        ).fetchone()

    return _mc_row_to_relationship(row) if row else None


def save_relationship_history(
    connect_fn: Callable,
    stack_id: str,
    entry: RelationshipHistoryEntry,
    now_fn: Callable[[], str],
) -> str:
    """Save a relationship history entry."""
    if not entry.id:
        entry.id = str(uuid.uuid4())

    now = now_fn()
    if not entry.created_at:
        entry.created_at = datetime.now(timezone.utc)

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT INTO relationship_history
            (id, stack_id, relationship_id, entity_name, event_type,
             old_value, new_value, episode_id, notes, created_at,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry.id,
                stack_id,
                entry.relationship_id,
                entry.entity_name,
                entry.event_type,
                entry.old_value,
                entry.new_value,
                entry.episode_id,
                entry.notes,
                entry.created_at.isoformat() if entry.created_at else now,
                now,
                None,
                1,
                0,
            ),
        )
        conn.commit()

    return entry.id


def get_relationship_history(
    connect_fn: Callable,
    stack_id: str,
    entity_name: str,
    event_type: Optional[str] = None,
    limit: int = 50,
) -> List[RelationshipHistoryEntry]:
    """Get history entries for a relationship."""
    query = (
        "SELECT * FROM relationship_history "
        "WHERE stack_id = ? AND entity_name = ? AND deleted = 0"
    )
    params: List[Any] = [stack_id, entity_name]

    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with connect_fn() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_mc_row_to_relationship_history(row) for row in rows]


def save_entity_model(
    connect_fn: Callable,
    stack_id: str,
    model: EntityModel,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
) -> str:
    """Save an entity model."""
    if not model.id:
        model.id = str(uuid.uuid4())

    now = now_fn()

    # Auto-populate subject_ids from entity_name
    if not model.subject_ids:
        model.subject_ids = [model.entity_name]

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO entity_models
            (id, stack_id, entity_name, model_type, observation, confidence,
             source_episodes, created_at, updated_at,
             subject_ids, access_grants, consent_grants,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                model.id,
                stack_id,
                model.entity_name,
                model.model_type,
                model.observation,
                model.confidence,
                to_json(model.source_episodes),
                (model.created_at.isoformat() if model.created_at else now),
                now,
                to_json(model.subject_ids),
                to_json(model.access_grants),
                to_json(model.consent_grants),
                now,
                None,
                model.version,
                0,
            ),
        )
        conn.commit()

    return model.id


def get_entity_models(
    connect_fn: Callable,
    stack_id: str,
    entity_name: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = 100,
) -> List[EntityModel]:
    """Get entity models, optionally filtered."""
    query = "SELECT * FROM entity_models WHERE stack_id = ? AND deleted = 0"
    params: List[Any] = [stack_id]

    if entity_name:
        query += " AND entity_name = ?"
        params.append(entity_name)
    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with connect_fn() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_mc_row_to_entity_model(row) for row in rows]


def get_entity_model(
    connect_fn: Callable,
    stack_id: str,
    model_id: str,
) -> Optional[EntityModel]:
    """Get a specific entity model by ID."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM entity_models WHERE id = ? AND stack_id = ? AND deleted = 0",
            (model_id, stack_id),
        ).fetchone()

    return _mc_row_to_entity_model(row) if row else None


def row_to_relationship(row: sqlite3.Row) -> Relationship:
    """Convert a database row to a Relationship dataclass."""
    return _mc_row_to_relationship(row)


def row_to_relationship_history(row: sqlite3.Row) -> RelationshipHistoryEntry:
    """Convert a database row to a RelationshipHistoryEntry dataclass."""
    return _mc_row_to_relationship_history(row)


def row_to_entity_model(row: sqlite3.Row) -> EntityModel:
    """Convert a database row to an EntityModel dataclass."""
    return _mc_row_to_entity_model(row)
