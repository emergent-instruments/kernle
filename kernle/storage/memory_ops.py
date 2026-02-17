"""Memory lifecycle operations for Kernle storage.

MemoryOps class handles forget/recover/protect/weaken/verify/boost/audit
operations and derived-from lineage queries. Receives dependencies
explicitly to avoid circular imports.
"""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from .base import SearchResult

logger = logging.getLogger(__name__)


def _exact_derived_from_match(
    derived_from_json: Optional[str], memory_type: str, memory_id: str
) -> bool:
    """Check if a derived_from JSON array contains an exact 'type:id' entry.

    The LIKE query used as a prefilter can produce false positives when one
    ID is a prefix of another (e.g., "belief:abc" substring-matches
    "belief:abcdef"). This helper parses the JSON and checks for exact
    membership.

    Args:
        derived_from_json: Raw JSON string from the derived_from column.
        memory_type: The memory type prefix (e.g., "belief").
        memory_id: The exact memory ID to match.

    Returns:
        True if the exact 'type:id' string is in the parsed list, False otherwise.
    """
    if not derived_from_json:
        return False
    try:
        derived_from = json.loads(derived_from_json)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(derived_from, list):
        return False

    target = f"{memory_type}:{memory_id}"
    return target in derived_from


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


class MemoryOps:
    """Memory lifecycle operations: forget, recover, protect, weaken, verify, boost, audit.

    Args:
        connect_fn: Callable returning a DB connection context manager.
        stack_id: The stack identifier.
        now_fn: Callable returning current timestamp string.
        safe_get_fn: Callable(row, key, default) for safe row access.
        queue_sync_fn: Callable(conn, table, record_id, op) for sync queueing.
        validate_table_name_fn: Callable(table) for SQL injection prevention.
    """

    def __init__(
        self,
        connect_fn: Callable,
        stack_id: str,
        now_fn: Callable[[], str],
        safe_get_fn: Callable,
        queue_sync_fn: Callable,
        validate_table_name_fn: Callable,
    ):
        self._connect = connect_fn
        self.stack_id = stack_id
        self._now = now_fn
        self._safe_get = safe_get_fn
        self._queue_sync = queue_sync_fn
        self._validate_table_name = validate_table_name_fn

    def forget_memory(
        self,
        memory_type: str,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Tombstone a memory (mark as forgotten, don't delete).

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            reason: Optional reason for forgetting

        Returns:
            True if forgotten, False if not found, already forgotten, or protected
        """
        table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
        if not table:
            return False
        self._validate_table_name(table)

        now = self._now()

        with self._connect() as conn:
            row = conn.execute(
                f"SELECT is_protected, strength, deleted FROM {table} WHERE id = ? AND stack_id = ? AND deleted = 0",
                (memory_id, self.stack_id),
            ).fetchone()

            if not row:
                return False

            if self._safe_get(row, "is_protected", 0):
                logger.debug(f"Cannot forget protected memory {memory_type}:{memory_id}")
                return False

            if float(self._safe_get(row, "strength", 1.0)) == 0.0:
                return False

            cursor = conn.execute(
                f"""UPDATE {table}
                   SET strength = 0.0,
                       local_updated_at = ?
                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                (now, memory_id, self.stack_id),
            )
            if cursor.rowcount > 0:
                conn.execute(
                    """INSERT INTO memory_audit (id, memory_type, memory_id, operation, details, actor, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        memory_type,
                        memory_id,
                        "forget",
                        json.dumps({"reason": reason}) if reason else None,
                        "system",
                        now,
                    ),
                )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if recovered, False if not found or not forgotten
        """
        table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
        if not table:
            return False
        self._validate_table_name(table)

        now = self._now()

        with self._connect() as conn:
            row = conn.execute(
                f"SELECT strength FROM {table} WHERE id = ? AND stack_id = ? AND deleted = 0",
                (memory_id, self.stack_id),
            ).fetchone()

            if not row or float(self._safe_get(row, "strength", 1.0)) > 0.0:
                return False

            cursor = conn.execute(
                f"""UPDATE {table}
                   SET strength = 0.2,
                       local_updated_at = ?
                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                (now, memory_id, self.stack_id),
            )
            if cursor.rowcount > 0:
                conn.execute(
                    """INSERT INTO memory_audit (id, memory_type, memory_id, operation, details, actor, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), memory_type, memory_id, "recover", None, "system", now),
                )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def protect_memory(self, memory_type: str, memory_id: str, protected: bool = True) -> bool:
        """Mark a memory as protected from forgetting.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            protected: True to protect, False to unprotect

        Returns:
            True if updated, False if memory not found
        """
        table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
        if not table:
            return False
        self._validate_table_name(table)

        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET is_protected = ?,
                       local_updated_at = ?
                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                (1 if protected else 0, now, memory_id, self.stack_id),
            )
            if cursor.rowcount > 0:
                conn.execute(
                    """INSERT INTO memory_audit (id, memory_type, memory_id, operation, details, actor, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        memory_type,
                        memory_id,
                        "protect" if protected else "unprotect",
                        None,
                        "system",
                        now,
                    ),
                )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def log_audit(
        self,
        memory_type: str,
        memory_id: str,
        operation: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Log an audit entry for a memory operation.

        Args:
            memory_type: Type of memory affected
            memory_id: ID of the memory affected
            operation: Operation name (forget, recover, protect, weaken, verify,
                or dotted format like belief.revised, suggestion.created)
            actor: Who performed the operation (e.g. 'core:{id}', 'plugin:{name}')
            details: Optional JSON-serializable details about the operation
            correlation_id: Optional ID linking related audit entries (e.g. from
                a single process() run)

        Returns:
            The audit entry ID
        """
        audit_id = str(uuid.uuid4())
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO memory_audit (id, memory_type, memory_id, operation, details, actor, created_at, correlation_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    audit_id,
                    memory_type,
                    memory_id,
                    operation,
                    json.dumps(details) if details else None,
                    actor,
                    now,
                    correlation_id,
                ),
            )
            conn.commit()
        return audit_id

    def get_audit_log(
        self,
        *,
        memory_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries.

        Args:
            memory_type: Filter by memory type
            memory_id: Filter by memory ID
            operation: Filter by operation type
            correlation_id: Filter by correlation ID (links related audit entries)
            limit: Max entries to return

        Returns:
            List of audit entry dicts
        """
        conditions = ["1=1"]
        params: list = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if memory_id:
            conditions.append("memory_id = ?")
            params.append(memory_id)
        if operation:
            conditions.append("operation = ?")
            params.append(operation)
        if correlation_id:
            conditions.append("correlation_id = ?")
            params.append(correlation_id)

        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT id, memory_type, memory_id, operation, details, actor, created_at, correlation_id
                   FROM memory_audit
                   WHERE {' AND '.join(conditions)}
                   ORDER BY created_at DESC
                   LIMIT ?""",
                params,
            ).fetchall()

        results = []
        for row in rows:
            entry = {
                "id": row["id"],
                "memory_type": row["memory_type"],
                "memory_id": row["memory_id"],
                "operation": row["operation"],
                "details": json.loads(row["details"]) if row["details"] else None,
                "actor": row["actor"],
                "created_at": row["created_at"],
                "correlation_id": row["correlation_id"],
            }
            results.append(entry)
        return results

    def export_audit_jsonl(
        self,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        memory_type: Optional[str] = None,
        operation: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """Export audit log entries as JSONL lines.

        Each yielded line is a JSON string with schema_version 1 format:
        {schema_version, event_id, operation, timestamp, memory_type,
         memory_id, correlation_id, actor, details}

        Args:
            since: Filter entries created at or after this ISO timestamp
            until: Filter entries created before this ISO timestamp
            memory_type: Filter by memory type
            operation: Filter by operation type
            correlation_id: Filter by correlation ID

        Yields:
            JSON string per audit entry (no trailing newline)
        """
        conditions = ["1=1"]
        params: list = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if operation:
            conditions.append("operation = ?")
            params.append(operation)
        if correlation_id:
            conditions.append("correlation_id = ?")
            params.append(correlation_id)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        if until:
            conditions.append("created_at < ?")
            params.append(until)

        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT id, memory_type, memory_id, operation, details, actor, created_at, correlation_id
                   FROM memory_audit
                   WHERE {' AND '.join(conditions)}
                   ORDER BY created_at ASC""",
                params,
            ).fetchall()

        for row in rows:
            entry = {
                "schema_version": 1,
                "event_id": row["id"],
                "operation": row["operation"],
                "timestamp": row["created_at"],
                "memory_type": row["memory_type"],
                "memory_id": row["memory_id"],
                "correlation_id": row["correlation_id"],
                "actor": row["actor"],
                "details": json.loads(row["details"]) if row["details"] else None,
            }
            yield json.dumps(entry, default=str)

    def weaken_memory(
        self,
        memory_type: str,
        memory_id: str,
        amount: float,
    ) -> bool:
        """Reduce a memory's strength by a given amount.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            amount: Amount to reduce strength by (positive value)

        Returns:
            True if updated, False if memory not found or protected
        """
        table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
        if not table:
            return False

        amount = abs(amount)
        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET strength = MAX(0.0, COALESCE(strength, 1.0) - ?),
                       local_updated_at = ?
                   WHERE id = ? AND stack_id = ? AND deleted = 0
                     AND COALESCE(is_protected, 0) = 0""",
                (amount, now, memory_id, self.stack_id),
            )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def verify_memory(
        self,
        memory_type: str,
        memory_id: str,
    ) -> bool:
        """Verify a memory: boost strength and increment verification count.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if updated, False if memory not found
        """
        table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
        if not table:
            return False

        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET strength = MIN(1.0, COALESCE(strength, 1.0) + 0.1),
                       verification_count = COALESCE(verification_count, 0) + 1,
                       last_verified = ?,
                       local_updated_at = ?
                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                (now, now, memory_id, self.stack_id),
            )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def boost_memory_strength(
        self,
        memory_type: str,
        memory_id: str,
        amount: float,
    ) -> bool:
        """Boost a memory's strength by a given amount (capped at 1.0).

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            amount: Amount to increase strength by (positive value)

        Returns:
            True if updated, False if memory not found
        """
        table = MEMORY_TYPE_TABLE_MAP.get(memory_type)
        if not table:
            return False
        self._validate_table_name(table)

        amount = abs(amount)
        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET strength = MIN(1.0, COALESCE(strength, 1.0) + ?),
                       local_updated_at = ?
                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                (amount, now, memory_id, self.stack_id),
            )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def get_memories_derived_from(self, memory_type: str, memory_id: str) -> List[tuple]:
        """Find all memories that cite 'type:id' in their derived_from.

        Args:
            memory_type: Type of the source memory (e.g. 'episode')
            memory_id: ID of the source memory

        Returns:
            List of (child_memory_type, child_memory_id) tuples
        """
        search_pattern = f'%"{memory_type}:{memory_id}"%'
        results: List[tuple] = []

        with self._connect() as conn:
            for mem_type, table in MEMORY_TYPE_TABLE_MAP.items():
                rows = conn.execute(
                    f"""SELECT id, derived_from FROM {table}
                       WHERE stack_id = ? AND deleted = 0
                         AND derived_from LIKE ?""",
                    (self.stack_id, search_pattern),
                ).fetchall()
                for row in rows:
                    # Post-filter: LIKE is a cheap prefilter but can match
                    # prefix substrings (e.g., "belief:abc" inside "belief:abcdef").
                    # Verify exact membership in the parsed JSON array.
                    if _exact_derived_from_match(row["derived_from"], memory_type, memory_id):
                        results.append((mem_type, row["id"]))

        return results

    def get_ungrounded_memories(self, stack_id: str) -> List[tuple]:
        """Find memories where ALL source refs have strength 0.0 or don't exist.

        Only considers refs in type:id format (skips context: and kernle: prefixes).
        Returns memories that have derived_from entries but every referenced
        source is either forgotten (strength 0.0) or missing.

        Args:
            stack_id: Stack ID to search in

        Returns:
            List of (memory_type, memory_id, [source_refs]) tuples
        """
        # Annotation ref types to skip
        skip_prefixes = {"context", "kernle"}

        # Lookup tables for checking strength
        strength_table_map = {
            "raw": "raw_entries",
            **MEMORY_TYPE_TABLE_MAP,
        }

        results: List[tuple] = []

        with self._connect() as conn:
            for mem_type, table in MEMORY_TYPE_TABLE_MAP.items():
                rows = conn.execute(
                    f"""SELECT id, derived_from FROM {table}
                       WHERE stack_id = ? AND deleted = 0
                         AND derived_from IS NOT NULL AND derived_from != '[]' AND derived_from != 'null'""",
                    (stack_id,),
                ).fetchall()

                for row in rows:
                    derived_from_raw = row["derived_from"]
                    if not derived_from_raw:
                        continue
                    try:
                        derived_from = json.loads(derived_from_raw)
                    except (json.JSONDecodeError, TypeError):
                        continue  # Malformed derived_from JSON — skip entry
                    if not isinstance(derived_from, list) or not derived_from:
                        continue

                    # Filter to real source refs only
                    source_refs = []
                    for ref in derived_from:
                        if not ref or ":" not in ref:
                            continue
                        ref_type = ref.split(":", 1)[0]
                        if ref_type in skip_prefixes:
                            continue
                        source_refs.append(ref)

                    if not source_refs:
                        continue

                    # Check if ALL source refs are dead (strength 0.0 or missing)
                    all_dead = True
                    for ref in source_refs:
                        ref_type, ref_id = ref.split(":", 1)
                        ref_table = strength_table_map.get(ref_type)
                        if not ref_table:
                            all_dead = False
                            break
                        if ref_type == "raw":
                            ref_row = conn.execute(
                                f"""SELECT id FROM {ref_table}
                                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                                (ref_id, stack_id),
                            ).fetchone()
                            if ref_row is not None:
                                all_dead = False
                                break
                        else:
                            ref_row = conn.execute(
                                f"""SELECT strength FROM {ref_table}
                                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                                (ref_id, stack_id),
                            ).fetchone()
                            if ref_row is not None:
                                strength = float(self._safe_get(ref_row, "strength", 1.0))
                                if strength > 0.0:
                                    all_dead = False
                                    break

                    if all_dead:
                        results.append((mem_type, row["id"], source_refs))

        return results

    def get_pre_v09_memories(self, stack_id: str) -> List[tuple]:
        """Find memories annotated with kernle:pre-v0.9-migration.

        These are memories that existed before provenance enforcement was
        introduced in v0.9. They have a migration annotation but no real
        provenance chain (no raw, episode, or note refs).

        Returns:
            List of (memory_type, memory_id, has_auto_link) tuples.
            has_auto_link is True if the memory was also linked to a raw
            entry via migrate link-raw.
        """
        results: List[tuple] = []

        with self._connect() as conn:
            for mem_type, table in MEMORY_TYPE_TABLE_MAP.items():
                rows = conn.execute(
                    f"""SELECT id, derived_from FROM {table}
                       WHERE stack_id = ? AND deleted = 0
                         AND derived_from LIKE '%pre-v0.9-migration%'""",
                    (stack_id,),
                ).fetchall()

                for row in rows:
                    derived_from_raw = row["derived_from"]
                    if not derived_from_raw:
                        continue
                    try:
                        derived_from = json.loads(derived_from_raw)
                    except (json.JSONDecodeError, TypeError):
                        continue  # Malformed derived_from JSON — skip entry
                    if not isinstance(derived_from, list):
                        continue

                    has_auto_link = any(
                        ref.startswith("raw:") for ref in derived_from if ref and ":" in ref
                    )
                    results.append((mem_type, row["id"], has_auto_link))

        return results

    def get_forgetting_candidates(
        self,
        row_converters: Dict[str, Callable],
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
        threshold: float = 0.5,
    ) -> List[SearchResult]:
        """Get memories that are candidates for forgetting.

        Args:
            row_converters: Dict mapping memory_type to row converter callable.
            memory_types: Filter by memory type
            limit: Maximum results
            threshold: Strength threshold (memories below this are candidates)

        Returns:
            List of candidate memories with strength as score
        """
        results = []
        types = memory_types or ["episode", "belief", "goal", "note", "relationship"]

        with self._connect() as conn:
            for memory_type in types:
                if memory_type not in MEMORY_TYPE_TABLE_MAP:
                    continue
                converter = row_converters.get(memory_type)
                if not converter:
                    continue

                table = MEMORY_TYPE_TABLE_MAP[memory_type]
                foundational_filter = ""
                if memory_type == "belief":
                    foundational_filter = "AND COALESCE(is_foundational, 0) = 0"

                query = f"""
                    SELECT * FROM {table}
                    WHERE stack_id = ?
                    AND deleted = 0
                    AND COALESCE(is_protected, 0) = 0
                    AND COALESCE(strength, 1.0) > 0.0
                    AND COALESCE(strength, 1.0) < ?
                    {foundational_filter}
                    ORDER BY strength ASC
                    LIMIT ?
                """

                try:
                    rows = conn.execute(query, (self.stack_id, threshold, limit * 2)).fetchall()
                    for row in rows:
                        record = converter(row)
                        strength = float(self._safe_get(row, "strength", 1.0))

                        results.append(
                            SearchResult(record=record, record_type=memory_type, score=strength)
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not get forgetting candidates from {table}: {e}", exc_info=True
                    )

        results.sort(key=lambda x: x.score)
        return results[:limit]

    def get_forgotten_memories(
        self,
        row_converters: Dict[str, Callable],
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get all forgotten (tombstoned) memories.

        Args:
            row_converters: Dict mapping memory_type to row converter callable.
            memory_types: Filter by memory type
            limit: Maximum results

        Returns:
            List of forgotten memories
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

        with self._connect() as conn:
            for memory_type in types:
                if memory_type not in MEMORY_TYPE_TABLE_MAP:
                    continue
                converter = row_converters.get(memory_type)
                if not converter:
                    continue

                table = MEMORY_TYPE_TABLE_MAP[memory_type]
                query = f"""
                    SELECT * FROM {table}
                    WHERE stack_id = ?
                    AND deleted = 0
                    AND COALESCE(strength, 1.0) = 0.0
                    ORDER BY created_at DESC
                    LIMIT ?
                """

                try:
                    rows = conn.execute(query, (self.stack_id, limit)).fetchall()
                    for row in rows:
                        results.append(
                            SearchResult(
                                record=converter(row),
                                record_type=memory_type,
                                score=0.0,
                            )
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not get forgotten memories from {table}: {e}", exc_info=True
                    )

        return results[:limit]
