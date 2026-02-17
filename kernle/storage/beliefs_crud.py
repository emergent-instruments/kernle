"""Beliefs CRUD operations extracted from SQLiteStorage.

Pilot extraction (#732) — demonstrates the pattern for decomposing
SQLiteStorage's god-object into focused modules.

All functions receive dependencies explicitly (connection factory,
serializers, sync callbacks) to avoid circular imports and enable
independent testing.
"""

import logging
import sqlite3
import uuid
from typing import Any, Callable, List, Optional

from .base import Belief, VersionConflictError
from .memory_crud import _row_to_belief as _mc_row_to_belief

logger = logging.getLogger(__name__)


def save_belief(
    connect_fn: Callable,
    stack_id: str,
    belief: Belief,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    sync_to_file: Callable,
    *,
    lineage_checker: Optional[Callable] = None,
) -> str:
    """Save a belief.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        belief: The Belief to save.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation (conn, table, id, op, data).
        save_embedding: Saves embedding for search (conn, table, id, content).
        sync_to_file: Syncs beliefs to flat file (no args).
        lineage_checker: Optional callable for derived-from cycle detection.
            Receives (storage, memory_type, id, derived_from).
    """
    if not belief.id:
        belief.id = str(uuid.uuid4())

    if belief.derived_from and lineage_checker:
        lineage_checker("belief", belief.id, belief.derived_from)

    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO beliefs
            (id, stack_id, statement, belief_type, confidence, created_at,
             source_type, source_episodes, derived_from,
             last_verified, verification_count, confidence_history,
             supersedes, superseded_by, times_reinforced, is_active,
             strength,
             context, context_tags, source_entity, subject_ids, access_grants, consent_grants,
             processed,
             belief_scope, source_domain, cross_domain_applications, abstraction_level,
             epoch_id,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                belief.id,
                stack_id,
                belief.statement,
                belief.belief_type,
                belief.confidence,
                belief.created_at.isoformat() if belief.created_at else now,
                belief.source_type,
                to_json(belief.source_episodes),
                to_json(belief.derived_from),
                belief.last_verified.isoformat() if belief.last_verified else None,
                belief.verification_count,
                to_json(belief.confidence_history),
                None,  # supersedes — deprecated, always NULL (v0.14+)
                None,  # superseded_by — deprecated, always NULL (v0.14+)
                belief.times_reinforced,
                1 if belief.is_active else 0,
                belief.strength,
                belief.context,
                to_json(belief.context_tags),
                getattr(belief, "source_entity", None),
                to_json(getattr(belief, "subject_ids", None)),
                to_json(getattr(belief, "access_grants", None)),
                to_json(getattr(belief, "consent_grants", None)),
                1 if belief.processed else 0,
                getattr(belief, "belief_scope", "world"),
                getattr(belief, "source_domain", None),
                to_json(getattr(belief, "cross_domain_applications", None)),
                getattr(belief, "abstraction_level", "specific"),
                belief.epoch_id,
                now,
                belief.cloud_synced_at.isoformat() if belief.cloud_synced_at else None,
                belief.version,
                1 if belief.deleted else 0,
            ),
        )
        belief_data = to_json(record_to_dict(belief))
        queue_sync(conn, "beliefs", belief.id, "upsert", data=belief_data)
        save_embedding(conn, "beliefs", belief.id, belief.statement)
        conn.commit()

    sync_to_file()
    return belief.id


def update_belief_atomic(
    connect_fn: Callable,
    stack_id: str,
    belief: Belief,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    sync_to_file: Callable,
    expected_version: Optional[int] = None,
) -> bool:
    """Update a belief with optimistic concurrency control.

    Returns True if update succeeded.
    Raises VersionConflictError if the record's version doesn't match expected.
    """
    if expected_version is None:
        expected_version = belief.version

    now = now_fn()

    with connect_fn() as conn:
        current = conn.execute(
            "SELECT version FROM beliefs WHERE id = ? AND stack_id = ?",
            (belief.id, stack_id),
        ).fetchone()

        if not current:
            return False

        current_version = current["version"]
        if current_version != expected_version:
            raise VersionConflictError("beliefs", belief.id, expected_version, current_version)

        cursor = conn.execute(
            """
            UPDATE beliefs SET
                statement = ?,
                belief_type = ?,
                confidence = ?,
                source_type = ?,
                source_episodes = ?,
                derived_from = ?,
                last_verified = ?,
                verification_count = ?,
                confidence_history = ?,
                supersedes = ?,
                superseded_by = ?,
                times_reinforced = ?,
                is_active = ?,
                context = ?,
                context_tags = ?,
                belief_scope = ?,
                source_domain = ?,
                cross_domain_applications = ?,
                abstraction_level = ?,
                local_updated_at = ?,
                deleted = ?,
                version = version + 1
            WHERE id = ? AND stack_id = ? AND version = ?
            """,
            (
                belief.statement,
                belief.belief_type,
                belief.confidence,
                belief.source_type,
                to_json(belief.source_episodes),
                to_json(belief.derived_from),
                belief.last_verified.isoformat() if belief.last_verified else None,
                belief.verification_count,
                to_json(belief.confidence_history),
                None,  # supersedes — deprecated, always NULL (v0.14+)
                None,  # superseded_by — deprecated, always NULL (v0.14+)
                belief.times_reinforced,
                1 if belief.is_active else 0,
                belief.context,
                to_json(belief.context_tags),
                getattr(belief, "belief_scope", "world"),
                getattr(belief, "source_domain", None),
                to_json(getattr(belief, "cross_domain_applications", None)),
                getattr(belief, "abstraction_level", "specific"),
                now,
                1 if belief.deleted else 0,
                belief.id,
                stack_id,
                expected_version,
            ),
        )

        if cursor.rowcount == 0:
            conn.rollback()
            new_current = conn.execute(
                "SELECT version FROM beliefs WHERE id = ? AND stack_id = ?",
                (belief.id, stack_id),
            ).fetchone()
            actual = new_current["version"] if new_current else -1
            raise VersionConflictError("beliefs", belief.id, expected_version, actual)

        belief.version = expected_version + 1
        belief_data = to_json(record_to_dict(belief))
        queue_sync(conn, "beliefs", belief.id, "upsert", data=belief_data)
        save_embedding(conn, "beliefs", belief.id, belief.statement)
        conn.commit()

    sync_to_file()
    return True


def save_beliefs_batch(
    connect_fn: Callable,
    stack_id: str,
    beliefs: List[Belief],
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    sync_to_file: Callable,
) -> List[str]:
    """Save multiple beliefs in a single transaction."""
    if not beliefs:
        return []
    now = now_fn()
    ids = []
    with connect_fn() as conn:
        for belief in beliefs:
            if not belief.id:
                belief.id = str(uuid.uuid4())
            ids.append(belief.id)
            conn.execute(
                """
                INSERT OR REPLACE INTO beliefs
                (id, stack_id, statement, belief_type, confidence, created_at,
                 source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 supersedes, superseded_by, times_reinforced, is_active,
                 times_accessed, last_accessed, is_protected, strength,
                 context, context_tags, processed,
                 belief_scope, source_domain, cross_domain_applications, abstraction_level,
                 epoch_id,
                 local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    belief.id,
                    stack_id,
                    belief.statement,
                    belief.belief_type,
                    belief.confidence,
                    belief.created_at.isoformat() if belief.created_at else now,
                    belief.source_type,
                    to_json(belief.source_episodes),
                    to_json(belief.derived_from),
                    belief.last_verified.isoformat() if belief.last_verified else None,
                    belief.verification_count,
                    to_json(belief.confidence_history),
                    None,  # supersedes — deprecated, always NULL (v0.14+)
                    None,  # superseded_by — deprecated, always NULL (v0.14+)
                    belief.times_reinforced,
                    1 if belief.is_active else 0,
                    belief.times_accessed,
                    belief.last_accessed.isoformat() if belief.last_accessed else None,
                    1 if belief.is_protected else 0,
                    belief.strength,
                    belief.context,
                    to_json(belief.context_tags),
                    1 if belief.processed else 0,
                    getattr(belief, "belief_scope", "world"),
                    getattr(belief, "source_domain", None),
                    to_json(getattr(belief, "cross_domain_applications", None)),
                    getattr(belief, "abstraction_level", "specific"),
                    belief.epoch_id,
                    now,
                    belief.cloud_synced_at.isoformat() if belief.cloud_synced_at else None,
                    belief.version,
                    1 if belief.deleted else 0,
                ),
            )
            belief_data = to_json(record_to_dict(belief))
            queue_sync(conn, "beliefs", belief.id, "upsert", data=belief_data)
            save_embedding(conn, "beliefs", belief.id, belief.statement)
        conn.commit()
    sync_to_file()
    return ids


def get_beliefs(
    connect_fn: Callable,
    stack_id: str,
    build_access_filter: Callable,
    limit: int = 100,
    include_inactive: bool = False,
    requesting_entity: Optional[str] = None,
    processed: Optional[bool] = None,
) -> List[Belief]:
    """Get beliefs with optional privacy/processing filters."""
    access_filter, access_params = build_access_filter(requesting_entity)
    processed_filter = ""
    processed_params: List[Any] = []
    if processed is not None:
        processed_filter = " AND processed = ?"
        processed_params = [1 if processed else 0]
    with connect_fn() as conn:
        if include_inactive:
            rows = conn.execute(
                f"SELECT * FROM beliefs WHERE stack_id = ? AND deleted = 0{access_filter}{processed_filter} ORDER BY created_at DESC LIMIT ?",
                [stack_id] + access_params + processed_params + [limit],
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM beliefs WHERE stack_id = ? AND deleted = 0 AND (is_active = 1 OR is_active IS NULL){access_filter}{processed_filter} ORDER BY created_at DESC LIMIT ?",
                [stack_id] + access_params + processed_params + [limit],
            ).fetchall()

    return [_mc_row_to_belief(row) for row in rows]


def find_belief(
    connect_fn: Callable,
    stack_id: str,
    statement: str,
) -> Optional[Belief]:
    """Find a belief by exact statement text."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM beliefs WHERE stack_id = ? AND statement = ? AND deleted = 0",
            (stack_id, statement),
        ).fetchone()

    return _mc_row_to_belief(row) if row else None


def get_belief(
    connect_fn: Callable,
    stack_id: str,
    belief_id: str,
    build_access_filter: Callable,
    requesting_entity: Optional[str] = None,
) -> Optional[Belief]:
    """Get a specific belief by ID with optional privacy filter."""
    query = "SELECT * FROM beliefs WHERE id = ? AND stack_id = ?"
    params: List[Any] = [belief_id, stack_id]

    access_filter, access_params = build_access_filter(requesting_entity)
    query += access_filter
    params.extend(access_params)

    with connect_fn() as conn:
        row = conn.execute(query, params).fetchone()

    return _mc_row_to_belief(row) if row else None


def get_belief_by_id(
    connect_fn: Callable,
    stack_id: str,
    belief_id: str,
) -> Optional[Belief]:
    """Get a belief by ID (internal, no privacy filter)."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM beliefs WHERE id = ? AND stack_id = ? AND deleted = 0",
            (belief_id, stack_id),
        ).fetchone()
    return _mc_row_to_belief(row) if row else None


def row_to_belief(row: sqlite3.Row) -> Belief:
    """Convert a database row to a Belief dataclass."""
    return _mc_row_to_belief(row)
