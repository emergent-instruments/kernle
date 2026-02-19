"""Meta-memory operations extracted from SQLiteStorage.

Cross-type memory queries: get_memory, memory_exists, update_strength,
update_strength_batch, update_memory_meta, get_memories_by_confidence,
get_memories_by_source. These operate across all memory types using
generic SQL queries parameterized by table name.

All functions receive dependencies explicitly (connection factory,
converters, serializers) to avoid circular imports and enable
independent testing.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .base import SearchResult
from .schema import validate_table_name

logger = logging.getLogger(__name__)

# Canonical mapping from logical memory type to DB table name
MEMORY_TYPE_TABLE_MAP = {
    "episode": "episodes",
    "belief": "beliefs",
    "value": "agent_values",
    "goal": "goals",
    "note": "notes",
    "drive": "drives",
    "relationship": "relationships",
}

# Extended map that includes raw entries and playbooks (for memory_exists)
MEMORY_TYPE_TABLE_MAP_EXTENDED = {
    **MEMORY_TYPE_TABLE_MAP,
    "raw": "raw_entries",
    "playbook": "playbooks",
}


def memory_exists(
    conn: sqlite3.Connection,
    stack_id: str,
    memory_type: str,
    memory_id: str,
) -> bool:
    """Check if a memory record exists in the stack.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        memory_type: Type of memory (episode, belief, note, raw, etc.)
        memory_id: ID of the memory record

    Returns:
        True if the record exists and is not deleted
    """
    table = MEMORY_TYPE_TABLE_MAP_EXTENDED.get(memory_type)
    if not table:
        return False
    row = conn.execute(
        f"SELECT 1 FROM {table} WHERE id = ? AND stack_id = ? AND deleted = 0",
        (memory_id, stack_id),
    ).fetchone()
    return row is not None


def get_memory(
    conn: sqlite3.Connection,
    stack_id: str,
    memory_type: str,
    memory_id: str,
    row_converters: Dict[str, Callable],
) -> Optional[Any]:
    """Get a memory by type and ID.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        memory_type: Type of memory (episode, belief, value, goal, note, drive, relationship)
        memory_id: ID of the memory
        row_converters: Dict mapping memory_type to (table_name, row_converter) tuples.

    Returns:
        The memory record or None if not found
    """
    entry = row_converters.get(memory_type)
    if not entry:
        return None

    table, converter = entry
    row = conn.execute(
        f"SELECT * FROM {table} WHERE id = ? AND stack_id = ? AND deleted = 0",
        (memory_id, stack_id),
    ).fetchone()
    return converter(row) if row else None


def update_strength(
    conn: sqlite3.Connection,
    stack_id: str,
    memory_type: str,
    memory_id: str,
    strength: float,
    now_fn: Callable[[], str],
) -> bool:
    """Update the strength field of a memory.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        memory_type: Type of memory
        memory_id: ID of the memory
        strength: New strength value (clamped to 0.0-1.0)
        now_fn: Callable returning current UTC timestamp as ISO string.

    Returns:
        True if updated, False if memory not found
    """
    table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
    if not table:
        return False

    strength = max(0.0, min(1.0, strength))
    now = now_fn()

    cursor = conn.execute(
        f"""UPDATE {table}
           SET strength = ?,
               local_updated_at = ?
           WHERE id = ? AND stack_id = ? AND deleted = 0""",
        (strength, now, memory_id, stack_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def update_strength_batch(
    conn: sqlite3.Connection,
    stack_id: str,
    updates: list,
    now_fn: Callable[[], str],
) -> int:
    """Update strength for multiple memories in a single transaction.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        updates: List of (memory_type, memory_id, new_strength) tuples
        now_fn: Callable returning current UTC timestamp as ISO string.

    Returns:
        Number of memories successfully updated
    """
    if not updates:
        return 0

    now = now_fn()
    total_updated = 0

    for memory_type, memory_id, strength in updates:
        table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
        if not table:
            continue
        strength = max(0.0, min(1.0, strength))
        cursor = conn.execute(
            f"""UPDATE {table}
               SET strength = ?,
                   last_accessed = ?,
                   local_updated_at = ?
               WHERE id = ? AND stack_id = ? AND deleted = 0""",
            (strength, now, now, memory_id, stack_id),
        )
        total_updated += cursor.rowcount
    conn.commit()

    return total_updated


def update_memory_meta(
    conn: sqlite3.Connection,
    stack_id: str,
    memory_type: str,
    memory_id: str,
    to_json: Callable[[Any], Optional[str]],
    now_fn: Callable[[], str],
    queue_sync: Callable,
    normalize_source_type: Callable,
    lineage_checker: Callable,
    confidence: Optional[float] = None,
    source_type: Optional[str] = None,
    source_episodes: Optional[List[str]] = None,
    derived_from: Optional[List[str]] = None,
    last_verified: Optional[datetime] = None,
    verification_count: Optional[int] = None,
    confidence_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Update meta-memory fields for a memory.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        memory_type: Type of memory
        memory_id: ID of the memory
        to_json: Serializes objects to JSON strings.
        now_fn: Callable returning current UTC timestamp as ISO string.
        queue_sync: Callable(conn, table, record_id, op) for sync queueing.
        normalize_source_type: Callable to normalize source_type values.
        lineage_checker: Callable(storage, memory_type, memory_id, derived_from)
            for cycle detection.
        confidence: New confidence value
        source_type: New source type
        source_episodes: New source episodes list
        derived_from: New derived_from list
        last_verified: New verification timestamp
        verification_count: New verification count
        confidence_history: New confidence history

    Returns:
        True if updated, False if memory not found
    """
    table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
    if not table:
        return False
    validate_table_name(table)

    # Build update query dynamically
    updates = []
    params = []

    if confidence is not None:
        updates.append("confidence = ?")
        params.append(confidence)
    if source_type is not None:
        source_type = normalize_source_type(source_type)
        updates.append("source_type = ?")
        params.append(source_type)
    if source_episodes is not None:
        updates.append("source_episodes = ?")
        params.append(to_json(source_episodes))
    if derived_from is not None:
        lineage_checker(memory_type, memory_id, derived_from)
        updates.append("derived_from = ?")
        params.append(to_json(derived_from))
    if last_verified is not None:
        updates.append("last_verified = ?")
        params.append(last_verified.isoformat())
    if verification_count is not None:
        updates.append("verification_count = ?")
        params.append(verification_count)
    if confidence_history is not None:
        # Cap confidence_history to prevent unbounded growth
        max_confidence_history = 100
        if len(confidence_history) > max_confidence_history:
            confidence_history = confidence_history[-max_confidence_history:]
        updates.append("confidence_history = ?")
        params.append(to_json(confidence_history))

    if not updates:
        return False

    # Also update local_updated_at
    updates.append("local_updated_at = ?")
    params.append(now_fn())

    # Add version increment
    updates.append("version = version + 1")

    # Add WHERE clause params
    params.extend([memory_id, stack_id])

    query = f"UPDATE {table} SET {', '.join(updates)} WHERE id = ? AND stack_id = ? AND deleted = 0"

    cursor = conn.execute(query, params)
    if cursor.rowcount > 0:
        queue_sync(conn, table, memory_id, "upsert")
        conn.commit()
        return True
    return False


def get_memories_by_confidence(
    conn: sqlite3.Connection,
    stack_id: str,
    threshold: float,
    row_converters: Dict[str, Callable],
    safe_get: Callable,
    below: bool = True,
    memory_types: Optional[List[str]] = None,
    limit: int = 100,
) -> List[SearchResult]:
    """Get memories filtered by confidence threshold.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        threshold: Confidence threshold
        row_converters: Dict mapping memory_type to (table_name, row_converter) tuples.
        safe_get: Callable(row, key, default) for safe row access.
        below: If True, get memories below threshold; if False, above
        memory_types: Filter by type (episode, belief, etc.)
        limit: Maximum results

    Returns:
        List of matching memories with their types
    """
    results = []
    op = "<" if below else ">="
    types = memory_types or [
        "episode",
        "belief",
        "value",
        "goal",
        "note",
        "drive",
        "relationship",
    ]

    for memory_type in types:
        entry = row_converters.get(memory_type)
        if not entry:
            continue

        table, converter = entry
        validate_table_name(table)  # Security: validate before SQL use
        query = f"""
            SELECT * FROM {table}
            WHERE stack_id = ? AND deleted = 0
            AND confidence {op} ?
            ORDER BY confidence {"ASC" if below else "DESC"}
            LIMIT ?
        """

        try:
            rows = conn.execute(query, (stack_id, threshold, limit)).fetchall()
            for row in rows:
                results.append(
                    SearchResult(
                        record=converter(row),
                        record_type=memory_type,
                        score=safe_get(row, "confidence", 0.8),
                    )
                )
        except Exception as e:
            # Column might not exist in old schema
            logger.debug(f"Could not query {table} by confidence: {e}", exc_info=True)

    # Sort by confidence
    results.sort(key=lambda x: x.score, reverse=not below)
    return results[:limit]


def get_memories_by_source(
    conn: sqlite3.Connection,
    stack_id: str,
    source_type: str,
    row_converters: Dict[str, Callable],
    safe_get: Callable,
    memory_types: Optional[List[str]] = None,
    limit: int = 100,
) -> List[SearchResult]:
    """Get memories filtered by source type.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        source_type: Source type to filter by
        row_converters: Dict mapping memory_type to (table_name, row_converter) tuples.
        safe_get: Callable(row, key, default) for safe row access.
        memory_types: Filter by memory type
        limit: Maximum results

    Returns:
        List of matching memories
    """
    results = []
    types = memory_types or [
        "episode",
        "belief",
        "value",
        "goal",
        "note",
        "drive",
        "relationship",
    ]

    for memory_type in types:
        entry = row_converters.get(memory_type)
        if not entry:
            continue

        table, converter = entry
        validate_table_name(table)  # Security: validate before SQL use
        query = f"""
            SELECT * FROM {table}
            WHERE stack_id = ? AND deleted = 0
            AND source_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """

        try:
            rows = conn.execute(query, (stack_id, source_type, limit)).fetchall()
            for row in rows:
                results.append(
                    SearchResult(
                        record=converter(row),
                        record_type=memory_type,
                        score=safe_get(row, "confidence", 0.8),
                    )
                )
        except Exception as e:
            # Column might not exist in old schema
            logger.debug(f"Could not query {table} by source_type: {e}", exc_info=True)

    return results[:limit]
