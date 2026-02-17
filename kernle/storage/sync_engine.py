"""Sync engine for Kernle storage.

SyncEngine class handles sync queue management, push/pull operations,
merge logic, and conflict resolution. Receives the host SQLiteStorage
instance to access DB connection and record operations.
"""

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from kernle.types import SYNC_COMPLETED, SYNC_DEAD_LETTER, SYNC_PENDING

from .base import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    Playbook,
    QueuedChange,
    Relationship,
    SyncConflict,
    SyncResult,
    Value,
)

logger = logging.getLogger(__name__)

# Maximum size for merged arrays during sync to prevent resource exhaustion
MAX_SYNC_ARRAY_SIZE = 500

# Tables that are intentionally kept local and are not pushed to cloud storage.
LOCAL_ONLY_SYNC_TABLES = frozenset({"memory_suggestions"})

# Array fields that should be merged (set union) during sync rather than overwritten
SYNC_ARRAY_FIELDS: Dict[str, List[str]] = {
    "episodes": [
        "lessons",
        "tags",
        "emotional_tags",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "beliefs": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "notes": [
        "tags",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "drives": [
        "focus_areas",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "agent_values": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "goals": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "relationships": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "playbooks": [
        "trigger_conditions",
        "failure_modes",
        "recovery_steps",
        "source_episodes",
        "tags",
    ],
}


class SyncEngine:
    """Sync engine handling queue management, push/pull, merge, and conflict resolution.

    Args:
        host: The SQLiteStorage instance providing DB access and record operations.
        validate_table_name_fn: Callable to validate table names against allowlist.
    """

    def __init__(self, host, validate_table_name_fn: Callable):
        self._host = host
        self._validate_table_name = validate_table_name_fn

    # === Queue Operations ===

    def queue_sync_operation(
        self, operation: str, table: str, record_id: str, data: Optional[Dict[str, Any]] = None
    ) -> int:
        """Queue a sync operation for later synchronization."""
        if table in LOCAL_ONLY_SYNC_TABLES:
            return 0

        now = self._host._now()
        data_json = self._host._to_json(data) if data else None

        with self._host._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO sync_queue
                   (table_name, record_id, operation, data, local_updated_at, synced, queued_at)
                   VALUES (?, ?, ?, ?, ?, 0, ?)
                   ON CONFLICT(table_name, record_id) WHERE synced = 0
                   DO UPDATE SET
                       operation = excluded.operation,
                       data = excluded.data,
                       local_updated_at = excluded.local_updated_at,
                       queued_at = excluded.queued_at""",
                (table, record_id, operation, data_json, now, now),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_pending_sync_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all unsynced operations from the queue."""
        with self._host._connect() as conn:
            rows = conn.execute(
                """SELECT id, operation, table_name, record_id, data, local_updated_at
                   FROM sync_queue
                   WHERE synced = 0
                   ORDER BY id
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "operation": row["operation"],
                "table_name": row["table_name"],
                "record_id": row["record_id"],
                "data": self._host._from_json(row["data"]) if row["data"] else None,
                "local_updated_at": self._host._parse_datetime(row["local_updated_at"]),
            }
            for row in rows
        ]

    def mark_synced(self, ids: List[int]) -> int:
        """Mark sync queue entries as synced."""
        if not ids:
            return 0

        with self._host._connect() as conn:
            placeholders = ",".join("?" * len(ids))
            cursor = conn.execute(
                f"UPDATE sync_queue SET synced = 1 WHERE id IN ({placeholders})", ids
            )
            conn.commit()
            return cursor.rowcount

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync queue status with counts."""
        with self._host._connect() as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE synced = ?", (SYNC_PENDING,)
            ).fetchone()[0]
            synced = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE synced = ?", (SYNC_COMPLETED,)
            ).fetchone()[0]
            dead_letter = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE synced = ?", (SYNC_DEAD_LETTER,)
            ).fetchone()[0]

            table_rows = conn.execute(
                """SELECT table_name, COUNT(*) as count
                   FROM sync_queue WHERE synced = ?
                   GROUP BY table_name""",
                (SYNC_PENDING,),
            ).fetchall()
            by_table = {row["table_name"]: row["count"] for row in table_rows}

            op_rows = conn.execute(
                """SELECT operation, COUNT(*) as count
                   FROM sync_queue WHERE synced = ?
                   GROUP BY operation""",
                (SYNC_PENDING,),
            ).fetchall()
            by_operation = {row["operation"]: row["count"] for row in op_rows}

        return {
            "pending": pending,
            "synced": synced,
            "dead_letter": dead_letter,
            "total": pending + synced + dead_letter,
            "by_table": by_table,
            "by_operation": by_operation,
        }

    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        with self._host._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sync_queue WHERE synced = 0").fetchone()[0]
        return count

    def get_queued_changes(self, limit: int = 100, max_retries: int = 5) -> List[QueuedChange]:
        """Get queued changes for sync."""
        with self._host._connect() as conn:
            rows = conn.execute(
                """SELECT id, table_name, record_id, operation,
                          COALESCE(payload, data) as payload, queued_at,
                          COALESCE(retry_count, 0) as retry_count,
                          last_error, last_attempt_at
                   FROM sync_queue
                   WHERE synced = 0 AND COALESCE(retry_count, 0) < ?
                   ORDER BY id
                   LIMIT ?""",
                (max_retries, limit),
            ).fetchall()

        return [
            QueuedChange(
                id=row["id"],
                table_name=row["table_name"],
                record_id=row["record_id"],
                operation=row["operation"],
                payload=row["payload"],
                queued_at=self._host._parse_datetime(row["queued_at"]),
                retry_count=row["retry_count"] or 0,
                last_error=row["last_error"],
                last_attempt_at=self._host._parse_datetime(row["last_attempt_at"]),
            )
            for row in rows
        ]

    def _clear_queued_change(self, conn: sqlite3.Connection, queue_id: int):
        """Mark a change as synced."""
        conn.execute("UPDATE sync_queue SET synced = 1 WHERE id = ?", (queue_id,))

    def _record_sync_failure(self, conn: sqlite3.Connection, queue_id: int, error: str) -> int:
        """Record a sync failure and increment retry count."""
        now = self._host._now()
        conn.execute(
            """UPDATE sync_queue
               SET retry_count = COALESCE(retry_count, 0) + 1,
                   last_error = ?,
                   last_attempt_at = ?
               WHERE id = ?""",
            (error[:500], now, queue_id),
        )
        row = conn.execute(
            "SELECT retry_count FROM sync_queue WHERE id = ?", (queue_id,)
        ).fetchone()
        return row["retry_count"] if row else 0

    def get_failed_sync_records(self, min_retries: int = 5) -> List[QueuedChange]:
        """Get sync records that have exceeded max retries."""
        with self._host._connect() as conn:
            rows = conn.execute(
                """SELECT id, table_name, record_id, operation,
                          COALESCE(payload, data) as payload, queued_at,
                          COALESCE(retry_count, 0) as retry_count,
                          last_error, last_attempt_at
                   FROM sync_queue
                   WHERE synced = 0 AND COALESCE(retry_count, 0) >= ?
                   ORDER BY last_attempt_at DESC
                   LIMIT 100""",
                (min_retries,),
            ).fetchall()

        return [
            QueuedChange(
                id=row["id"],
                table_name=row["table_name"],
                record_id=row["record_id"],
                operation=row["operation"],
                payload=row["payload"],
                queued_at=self._host._parse_datetime(row["queued_at"]),
                retry_count=row["retry_count"] or 0,
                last_error=row["last_error"],
                last_attempt_at=self._host._parse_datetime(row["last_attempt_at"]),
            )
            for row in rows
        ]

    def clear_failed_sync_records(self, older_than_days: int = 7) -> int:
        """Move failed sync records older than the specified days to dead-letter state.

        Dead-lettered entries use synced=SYNC_DEAD_LETTER (2) so they remain
        distinguishable from successfully synced records (synced=1).
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cutoff_str = cutoff.isoformat()

        with self._host._connect() as conn:
            cursor = conn.execute(
                """UPDATE sync_queue
                   SET synced = ?
                   WHERE synced = ?
                     AND COALESCE(retry_count, 0) >= 5
                     AND last_attempt_at < ?""",
                (SYNC_DEAD_LETTER, SYNC_PENDING, cutoff_str),
            )
            count = cursor.rowcount
            conn.commit()

        return count

    def get_dead_letter_count(self) -> int:
        """Get count of dead-lettered sync records."""
        with self._host._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE synced = ?",
                (SYNC_DEAD_LETTER,),
            ).fetchone()[0]
        return count

    def requeue_dead_letters(self, record_ids: list[int] | None = None) -> int:
        """Re-enqueue dead-lettered entries for retry.

        Args:
            record_ids: Specific IDs to requeue, or None for all.
        Returns:
            Number of entries requeued.
        """
        with self._host._connect() as conn:
            if record_ids:
                placeholders = ",".join("?" for _ in record_ids)
                cursor = conn.execute(
                    f"UPDATE sync_queue SET synced = {SYNC_PENDING}, retry_count = 0, "
                    f"last_error = NULL WHERE synced = {SYNC_DEAD_LETTER} "
                    f"AND id IN ({placeholders})",
                    record_ids,
                )
            else:
                cursor = conn.execute(
                    "UPDATE sync_queue SET synced = ?, retry_count = 0, last_error = NULL "
                    "WHERE synced = ?",
                    (SYNC_PENDING, SYNC_DEAD_LETTER),
                )
            conn.commit()
            return cursor.rowcount

    # === Sync Metadata ===

    def _get_sync_meta(self, key: str) -> Optional[str]:
        """Get a sync metadata value."""
        with self._host._connect() as conn:
            row = conn.execute("SELECT value FROM sync_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_sync_meta(self, key: str, value: str):
        """Set a sync metadata value."""
        with self._host._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sync_meta (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, self._host._now()),
            )
            conn.commit()

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the timestamp of the last successful sync."""
        value = self._get_sync_meta("last_sync_time")
        return self._host._parse_datetime(value) if value else None

    # === Conflict Management ===

    def get_sync_conflicts(self, limit: int = 100) -> List[SyncConflict]:
        """Get recent sync conflict history."""
        with self._host._connect() as conn:
            rows = conn.execute(
                """SELECT id, table_name, record_id, local_version, cloud_version,
                          resolution, resolved_at, local_summary, cloud_summary,
                          source, diff_hash, policy_decision
                   FROM sync_conflicts
                   ORDER BY resolved_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            SyncConflict(
                id=row["id"],
                table=row["table_name"],
                record_id=row["record_id"],
                local_version=json.loads(row["local_version"]),
                cloud_version=json.loads(row["cloud_version"]),
                resolution=row["resolution"],
                resolved_at=self._host._parse_datetime(row["resolved_at"])
                or datetime.now(timezone.utc),
                local_summary=row["local_summary"],
                cloud_summary=row["cloud_summary"],
                source=row["source"] if "source" in row.keys() else None,
                diff_hash=row["diff_hash"] if "diff_hash" in row.keys() else None,
                policy_decision=row["policy_decision"] if "policy_decision" in row.keys() else None,
            )
            for row in rows
        ]

    def save_sync_conflict(self, conflict: SyncConflict) -> str:
        """Save a sync conflict record. Deduplicates by diff_hash when available."""
        with self._host._connect() as conn:
            # Deduplicate by diff_hash if available
            if conflict.diff_hash:
                existing = conn.execute(
                    "SELECT id FROM sync_conflicts WHERE diff_hash = ?",
                    (conflict.diff_hash,),
                ).fetchone()
                if existing:
                    return existing["id"]  # Already recorded

            conn.execute(
                """INSERT INTO sync_conflicts
                   (id, table_name, record_id, local_version, cloud_version,
                    resolution, resolved_at, local_summary, cloud_summary,
                    source, diff_hash, policy_decision)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conflict.id,
                    conflict.table,
                    conflict.record_id,
                    json.dumps(conflict.local_version),
                    json.dumps(conflict.cloud_version),
                    conflict.resolution,
                    (
                        conflict.resolved_at.isoformat()
                        if isinstance(conflict.resolved_at, datetime)
                        else conflict.resolved_at
                    ),
                    conflict.local_summary,
                    conflict.cloud_summary,
                    conflict.source,
                    conflict.diff_hash,
                    conflict.policy_decision,
                ),
            )
            conn.commit()
        return conflict.id

    def clear_sync_conflicts(self, before: Optional[datetime] = None) -> int:
        """Clear sync conflict history."""
        with self._host._connect() as conn:
            if before:
                cursor = conn.execute(
                    "DELETE FROM sync_conflicts WHERE resolved_at < ?", (before.isoformat(),)
                )
            else:
                cursor = conn.execute("DELETE FROM sync_conflicts")
            conn.commit()
            return cursor.rowcount

    # === Connectivity ===

    def is_online(self) -> bool:
        """Check if cloud storage is reachable."""
        if not self._host.cloud_storage:
            return False

        now = datetime.now(timezone.utc)
        if self._host._last_connectivity_check:
            elapsed = (now - self._host._last_connectivity_check).total_seconds()
            if elapsed < self._host._connectivity_cache_ttl:
                return self._host._is_online_cached

        try:
            import socket

            old_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(self._host.CONNECTIVITY_TIMEOUT)
                self._host.cloud_storage.get_stats()
                self._host._is_online_cached = True
            except Exception as e:
                logger.debug(f"Connectivity check failed: {e}", exc_info=True)
                self._host._is_online_cached = False
            finally:
                socket.setdefaulttimeout(old_timeout)
        except Exception as e:
            logger.debug(f"Connectivity check error: {e}", exc_info=True)
            self._host._is_online_cached = False

        self._host._last_connectivity_check = now
        return self._host._is_online_cached

    # === Core Push/Pull ===

    def _mark_synced(self, conn: sqlite3.Connection, table: str, record_id: str):
        """Mark a record as synced with the cloud."""
        self._validate_table_name(table)
        now = self._host._now()
        conn.execute(
            f"UPDATE {table} SET cloud_synced_at = ? WHERE id = ? AND stack_id = ?",
            (now, record_id, self._host.stack_id),
        )

    def _get_record_for_push(self, table: str, record_id: str) -> Optional[Any]:
        """Get a record by table and ID for pushing to cloud."""
        self._validate_table_name(table)
        with self._host._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND stack_id = ?",
                (record_id, self._host.stack_id),
            ).fetchone()

        if not row:
            return None

        converters = {
            "episodes": self._host._row_to_episode,
            "notes": self._host._row_to_note,
            "beliefs": self._host._row_to_belief,
            "agent_values": self._host._row_to_value,
            "goals": self._host._row_to_goal,
            "drives": self._host._row_to_drive,
            "relationships": self._host._row_to_relationship,
            "playbooks": self._host._row_to_playbook,
            "raw_entries": self._host._row_to_raw_entry,
        }

        converter = converters.get(table)
        return converter(row) if converter else None

    def _push_record(self, table: str, record: Any) -> bool:
        """Push a single record to cloud storage."""
        if not self._host.cloud_storage:
            return False

        try:
            if table == "episodes":
                self._host.cloud_storage.save_episode(record)
            elif table == "notes":
                self._host.cloud_storage.save_note(record)
            elif table == "beliefs":
                self._host.cloud_storage.save_belief(record)
            elif table == "agent_values":
                self._host.cloud_storage.save_value(record)
            elif table == "goals":
                self._host.cloud_storage.save_goal(record)
            elif table == "drives":
                self._host.cloud_storage.save_drive(record)
            elif table == "relationships":
                self._host.cloud_storage.save_relationship(record)
            elif table == "playbooks":
                self._host.cloud_storage.save_playbook(record)
            else:
                logger.warning(f"Unknown table for push: {table}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to push record {table}:{record.id}: {e}", exc_info=True)
            return False

    def sync(self) -> SyncResult:
        """Sync with cloud storage."""
        result = SyncResult()

        if not self._host.cloud_storage:
            logger.debug("No cloud storage configured, skipping sync")
            return result

        if not self.is_online():
            logger.info("Offline - sync skipped, changes queued")
            result.errors.append("Offline - cannot reach cloud storage")
            return result

        # Phase 1: Push queued changes
        queued = self.get_queued_changes(limit=100, max_retries=5)
        failed_count = len(self.get_failed_sync_records(min_retries=5))
        if failed_count > 0:
            logger.info(f"Skipping {failed_count} records that exceeded max retries")

        logger.debug(f"Pushing {len(queued)} queued changes")

        with self._host._connect() as conn:
            for change in queued:
                try:
                    if change.table_name in LOCAL_ONLY_SYNC_TABLES:
                        retry_count = self._record_sync_failure(
                            conn,
                            change.id,
                            f"Table {change.table_name} is local-only and cannot be pushed to cloud",
                        )
                        result.errors.append(
                            f"Skipped push for local-only table {change.table_name}:{change.record_id} "
                            f"(retry {retry_count}/5)"
                        )
                        continue

                    record = self._get_record_for_push(change.table_name, change.record_id)

                    if record is None:
                        if change.operation == "delete":
                            self._clear_queued_change(conn, change.id)
                            result.pushed += 1
                        else:
                            self._clear_queued_change(conn, change.id)
                        continue

                    if self._push_record(change.table_name, record):
                        self._mark_synced(conn, change.table_name, change.record_id)
                        self._clear_queued_change(conn, change.id)
                        result.pushed += 1
                    else:
                        retry_count = self._record_sync_failure(
                            conn, change.id, "Push failed - cloud returned error"
                        )
                        if retry_count >= 5:
                            logger.warning(
                                f"Record {change.table_name}:{change.record_id} "
                                f"exceeded max retries, moving to dead letter queue"
                            )
                        result.errors.append(
                            f"Failed to push {change.table_name}:{change.record_id} "
                            f"(retry {retry_count}/5)"
                        )

                except Exception as e:
                    error_msg = str(e)[:500]
                    retry_count = self._record_sync_failure(conn, change.id, error_msg)
                    logger.error(
                        f"Error pushing {change.table_name}:{change.record_id}: {e} "
                        f"(retry {retry_count}/5)",
                        exc_info=True,
                    )
                    result.errors.append(
                        f"Error pushing {change.table_name}:{change.record_id}: {error_msg}"
                    )

            conn.commit()

        # Phase 2: Pull remote changes
        pull_result = self.pull_changes()
        result.pulled = pull_result.pulled
        result.conflicts = pull_result.conflicts
        result.errors.extend(pull_result.errors)

        if result.success or (result.pushed > 0 or result.pulled > 0):
            self._set_sync_meta("last_sync_time", self._host._now())

        logger.info(
            f"Sync complete: pushed={result.pushed}, pulled={result.pulled}, conflicts={result.conflict_count}"
        )
        return result

    def pull_changes(self, since: Optional[datetime] = None) -> SyncResult:
        """Pull changes from cloud since the given timestamp."""
        result = SyncResult()

        if not self._host.cloud_storage:
            return result

        if since is None:
            since = self.get_last_sync_time()

        tables_and_getters = [
            ("episodes", self._host.cloud_storage.get_episodes, self._merge_episode),
            ("notes", self._host.cloud_storage.get_notes, self._merge_note),
            ("beliefs", self._host.cloud_storage.get_beliefs, self._merge_belief),
            ("agent_values", self._host.cloud_storage.get_values, self._merge_value),
            ("goals", self._host.cloud_storage.get_goals, self._merge_goal),
            ("drives", self._host.cloud_storage.get_drives, self._merge_drive),
            ("relationships", self._host.cloud_storage.get_relationships, self._merge_relationship),
            ("playbooks", self._host.cloud_storage.list_playbooks, self._merge_playbook),
        ]

        for table, getter, merger in tables_and_getters:
            try:
                if table == "episodes":
                    cloud_records = getter(limit=1000, since=since)
                elif table == "notes":
                    cloud_records = getter(limit=1000, since=since)
                elif table == "goals":
                    cloud_records = getter(status=None, limit=1000)
                else:
                    cloud_records = getter(limit=1000) if callable(getter) else getter()

                for cloud_record in cloud_records:
                    pull_count, conflict = merger(cloud_record)
                    result.pulled += pull_count
                    if conflict:
                        result.conflicts.append(conflict)

            except Exception as e:
                logger.error(f"Failed to pull from {table}: {e}", exc_info=True)
                result.errors.append(f"Failed to pull {table}: {str(e)}")

        return result

    # === Merge Methods ===

    def _merge_episode(self, cloud_record: Episode) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud episode with local."""
        return self._merge_record("episodes", cloud_record, self._host.get_episode)

    def _merge_note(self, cloud_record: Note) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud note with local."""
        local = None
        with self._host._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()
            if row:
                local = self._host._row_to_note(row)
        return self._merge_generic(
            "notes", cloud_record, local, lambda: self._host.save_note(cloud_record)
        )

    def _merge_belief(self, cloud_record: Belief) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud belief with local."""
        local = None
        with self._host._connect() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()
            if row:
                local = self._host._row_to_belief(row)
        return self._merge_generic(
            "beliefs", cloud_record, local, lambda: self._host.save_belief(cloud_record)
        )

    def _merge_value(self, cloud_record: Value) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud value with local."""
        local = None
        with self._host._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_values WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()
            if row:
                local = self._host._row_to_value(row)
        return self._merge_generic(
            "agent_values", cloud_record, local, lambda: self._host.save_value(cloud_record)
        )

    def _merge_goal(self, cloud_record: Goal) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud goal with local."""
        local = None
        with self._host._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()
            if row:
                local = self._host._row_to_goal(row)
        return self._merge_generic(
            "goals", cloud_record, local, lambda: self._host.save_goal(cloud_record)
        )

    def _merge_drive(self, cloud_record: Drive) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud drive with local."""
        local = self._host.get_drive(cloud_record.drive_type)
        return self._merge_generic(
            "drives", cloud_record, local, lambda: self._host.save_drive(cloud_record)
        )

    def _merge_relationship(self, cloud_record: Relationship) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud relationship with local."""
        local = self._host.get_relationship(cloud_record.entity_name)
        return self._merge_generic(
            "relationships", cloud_record, local, lambda: self._host.save_relationship(cloud_record)
        )

    def _merge_playbook(self, cloud_record: Playbook) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud playbook with local."""
        local = self._host.get_playbook(cloud_record.id)
        return self._merge_generic(
            "playbooks", cloud_record, local, lambda: self._host.save_playbook(cloud_record)
        )

    def _merge_record(
        self, table: str, cloud_record: Any, get_local
    ) -> tuple[int, Optional[SyncConflict]]:
        """Generic merge for records with an ID-based getter."""
        local = get_local(cloud_record.id)
        return self._merge_generic(
            table, cloud_record, local, lambda: self._save_from_cloud(table, cloud_record)
        )

    def _merge_array_fields(self, table: str, winner: Any, loser: Any) -> Any:
        """Merge array fields from loser into winner using set union."""
        from dataclasses import replace

        array_fields = SYNC_ARRAY_FIELDS.get(table, [])
        if not array_fields:
            return winner

        updates = {}
        for field_name in array_fields:
            if not hasattr(winner, field_name) or not hasattr(loser, field_name):
                continue

            winner_val = getattr(winner, field_name)
            loser_val = getattr(loser, field_name)

            if not loser_val:
                continue
            if not winner_val:
                updates[field_name] = list(loser_val)
                continue

            try:
                if winner_val and isinstance(winner_val[0], dict):
                    seen = set()
                    merged = []
                    for item in winner_val:
                        key = json.dumps(item, sort_keys=True)
                        if key not in seen:
                            seen.add(key)
                            merged.append(item)
                    for item in loser_val:
                        key = json.dumps(item, sort_keys=True)
                        if key not in seen:
                            seen.add(key)
                            merged.append(item)
                    if len(merged) > MAX_SYNC_ARRAY_SIZE:
                        logger.warning(
                            f"Array field {field_name} exceeded max size ({len(merged)} > {MAX_SYNC_ARRAY_SIZE}), truncating"
                        )
                        merged = merged[:MAX_SYNC_ARRAY_SIZE]
                    updates[field_name] = merged
                else:
                    merged = list(set(winner_val) | set(loser_val))
                    if len(merged) > MAX_SYNC_ARRAY_SIZE:
                        logger.warning(
                            f"Array field {field_name} exceeded max size ({len(merged)} > {MAX_SYNC_ARRAY_SIZE}), truncating"
                        )
                        merged = merged[:MAX_SYNC_ARRAY_SIZE]
                    updates[field_name] = merged
            except (TypeError, KeyError):
                logger.warning(
                    f"Failed to merge array field {field_name}, keeping winner's value",
                    exc_info=True,
                )
                continue

        if updates:
            return replace(winner, **updates)
        return winner

    def _mark_synced_and_cleanup_queue(self, table: str, record_id: str):
        """Atomically mark a record as synced and remove its sync queue entry.

        This consolidates the mark-synced + queue-delete into a single transaction
        to close the race window where a crash between save and cleanup could leave
        orphaned queue entries that cause duplicate syncs on recovery.
        """
        with self._host._connect() as conn:
            self._mark_synced(conn, table, record_id)
            conn.execute(
                "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                (table, record_id),
            )
            conn.commit()

    def _record_already_applied(self, table: str, cloud_record: Any) -> bool:
        """Check if a cloud record has already been applied locally.

        Detects the recovery scenario where save_fn() completed but queue cleanup
        failed (crash between save and cleanup). If the local record exists with
        matching version data, the save can be skipped to prevent re-queuing.

        Returns True if the record exists locally with matching content, meaning
        the previous sync application succeeded and only cleanup is needed.
        """
        self._validate_table_name(table)
        with self._host._connect() as conn:
            row = conn.execute(
                f"SELECT version, cloud_synced_at FROM {table} WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()

        if not row:
            return False

        local_version = row["version"]
        cloud_version = getattr(cloud_record, "version", None)

        # If the local record has the same version as the cloud record,
        # the save was already applied (recovery scenario).
        if cloud_version is not None and local_version == cloud_version:
            # Also check cloud_synced_at is set, which confirms it was a sync save
            if row["cloud_synced_at"] is not None:
                return True

        return False

    def _merge_generic(
        self, table: str, cloud_record: Any, local_record: Optional[Any], save_fn
    ) -> tuple[int, Optional[SyncConflict]]:
        """Generic merge logic with last-write-wins for scalar fields, set union for arrays.

        Race window mitigation: save_fn() and queue cleanup cannot share a single
        transaction because save_fn is an opaque callable that manages its own
        connection. To handle the crash-between-save-and-cleanup scenario:

        1. Before calling save_fn(), check if the record was already applied
           (recovery from a previous interrupted sync). If so, skip save_fn()
           and proceed directly to queue cleanup.

        2. After save_fn() completes, atomically mark synced and clean the queue
           in a single transaction via _mark_synced_and_cleanup_queue().

        This ensures that:
        - Duplicate saves are detected and skipped (idempotent replay)
        - Queue cleanup is atomic with the synced-at marker update
        - No orphaned queue entries survive across crash recovery
        """
        if local_record is None:
            # Check if this is a recovery scenario: record was saved previously
            # but queue cleanup failed. If so, skip save to avoid re-queuing.
            if self._record_already_applied(table, cloud_record):
                logger.info(
                    f"Duplicate sync detected for {table}:{cloud_record.id} "
                    f"(v{getattr(cloud_record, 'version', '?')}), "
                    f"skipping save — cleaning up queue only"
                )
            else:
                save_fn()
            self._mark_synced_and_cleanup_queue(table, cloud_record.id)
            return (1, None)

        cloud_time = cloud_record.cloud_synced_at or cloud_record.local_updated_at
        local_time = local_record.local_updated_at

        if cloud_time and local_time:
            if cloud_time > local_time:
                merged_record = self._merge_array_fields(table, cloud_record, local_record)
                conflict = self._create_conflict(
                    table,
                    cloud_record.id,
                    local_record,
                    cloud_record,
                    "cloud_wins_arrays_merged",
                    policy_decision="newer_cloud_timestamp",
                )
                if not self._record_already_applied(table, cloud_record):
                    self._save_from_cloud(table, merged_record)
                self._mark_synced_and_cleanup_queue(table, cloud_record.id)
                self.save_sync_conflict(conflict)
                return (1, conflict)

            if local_time > cloud_time:
                merged_record = self._merge_array_fields(table, local_record, cloud_record)
                if merged_record is not local_record and not self._record_already_applied(
                    table, cloud_record
                ):
                    self._save_from_cloud(table, merged_record)
                conflict = self._create_conflict(
                    table,
                    cloud_record.id,
                    local_record,
                    cloud_record,
                    "local_wins_arrays_merged",
                    policy_decision="older_local_timestamp",
                )
                self._mark_synced_and_cleanup_queue(table, cloud_record.id)
                self.save_sync_conflict(conflict)
                return (0, conflict)

            # Equal timestamps: deterministic tie-break for replay safety.
            if self._build_record_snapshot(cloud_record) == self._build_record_snapshot(
                local_record
            ):
                self._mark_synced_and_cleanup_queue(table, cloud_record.id)
                return (0, None)

            policy_decision = self._choose_tie_break_policy(local_record, cloud_record)
            if policy_decision == "local_wins_tie_hash":
                merged_record = self._merge_array_fields(table, local_record, cloud_record)
                if merged_record is not local_record and not self._record_already_applied(
                    table, cloud_record
                ):
                    self._save_from_cloud(table, merged_record)
                resolution = "local_wins_arrays_merged"
            else:
                merged_record = self._merge_array_fields(table, cloud_record, local_record)
                if merged_record is not cloud_record and not self._record_already_applied(
                    table, cloud_record
                ):
                    self._save_from_cloud(table, merged_record)
                resolution = "cloud_wins_arrays_merged"

            conflict = self._create_conflict(
                table,
                cloud_record.id,
                local_record,
                cloud_record,
                resolution,
                policy_decision=policy_decision,
            )
            self._mark_synced_and_cleanup_queue(table, cloud_record.id)
            self.save_sync_conflict(conflict)
            return (0, conflict)

        if cloud_time:
            # Check for recovery scenario before calling save_fn
            if self._record_already_applied(table, cloud_record):
                logger.info(
                    f"Duplicate sync detected for {table}:{cloud_record.id} "
                    f"(v{getattr(cloud_record, 'version', '?')}), "
                    f"skipping save — cleaning up queue only"
                )
            else:
                save_fn()
            self._mark_synced_and_cleanup_queue(table, cloud_record.id)
            return (1, None)

        # Fallback: both timestamps missing — apply cloud record with warning
        logger.warning(
            "Sync merge for %s:%s has no timestamps — applying cloud record as fallback",
            table,
            cloud_record.id,
        )
        if self._record_already_applied(table, cloud_record):
            logger.info(
                "Duplicate sync detected for %s:%s (no-timestamp fallback), "
                "skipping save — cleaning up queue only",
                table,
                cloud_record.id,
            )
        elif local_record is not None:
            # Both sides exist but lack timestamps — merge arrays and record conflict
            merged_record = self._merge_array_fields(table, cloud_record, local_record)
            if not self._record_already_applied(table, cloud_record):
                self._save_from_cloud(table, merged_record)
            conflict = self._create_conflict(
                table,
                cloud_record.id,
                local_record,
                cloud_record,
                "cloud_wins_arrays_merged",
                policy_decision="no_timestamps_fallback",
            )
            self._mark_synced_and_cleanup_queue(table, cloud_record.id)
            self.save_sync_conflict(conflict)
            return (1, conflict)
        else:
            save_fn()
        self._mark_synced_and_cleanup_queue(table, cloud_record.id)
        return (1, None)

    def _create_conflict(
        self,
        table: str,
        record_id: str,
        local_record: Any,
        cloud_record: Any,
        resolution: str,
        *,
        policy_decision: Optional[str] = None,
    ) -> SyncConflict:
        """Create a SyncConflict record with human-readable summaries."""
        local_summary = self._get_record_summary(table, local_record)
        cloud_summary = self._get_record_summary(table, cloud_record)
        local_dict = self._record_to_dict(local_record)
        cloud_dict = self._record_to_dict(cloud_record)
        return SyncConflict(
            id=str(uuid.uuid4()),
            table=table,
            record_id=record_id,
            local_version=local_dict,
            cloud_version=cloud_dict,
            resolution=resolution,
            resolved_at=datetime.now(timezone.utc),
            local_summary=local_summary,
            cloud_summary=cloud_summary,
            source="sync_engine",
            diff_hash=self._build_conflict_hash(local_dict, cloud_dict),
            policy_decision=policy_decision,
        )

    def _build_record_snapshot(self, record: Any) -> Dict[str, Any]:
        """Build a deterministic snapshot for timestamp-equality comparison."""
        snapshot = self._record_to_dict(record)
        if not isinstance(snapshot, dict):
            return {}

        # Ignore mutable sync metadata when comparing semantically-equivalent records.
        # The cloud metadata timestamp can differ while payload/content remains identical,
        # and should not trigger a conflict in that case.
        snapshot.pop("cloud_synced_at", None)
        return snapshot

    def _build_record_hash(self, record: Dict[str, Any]) -> Optional[str]:
        """Build a deterministic hash for a record snapshot."""
        try:
            return hashlib.sha256(
                json.dumps(record, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        except Exception as exc:
            logger.debug(
                "Swallowed %s in _build_record_hash: %s", type(exc).__name__, exc, exc_info=True
            )
            return None

    def _build_conflict_hash(
        self, local_version: Dict[str, Any], cloud_version: Dict[str, Any]
    ) -> Optional[str]:
        """Build a deterministic hash for conflict comparisons and auditing."""
        try:
            payload = {"cloud": cloud_version, "local": local_version}
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        except Exception as exc:
            logger.debug(
                "Swallowed %s in _build_conflict_hash: %s", type(exc).__name__, exc, exc_info=True
            )
            return None

    def _choose_tie_break_policy(self, local_record: Any, cloud_record: Any) -> str:
        """Deterministically choose conflict winner when timestamps are equal."""
        local_snapshot = self._build_record_snapshot(local_record)
        cloud_snapshot = self._build_record_snapshot(cloud_record)

        local_hash = self._build_record_hash(local_snapshot)
        cloud_hash = self._build_record_hash(cloud_snapshot)

        if local_hash is None or cloud_hash is None:
            return "local_wins_tie_hash"

        # Deterministic, stable tie-breaker. Smallest hash wins.
        return "cloud_wins_tie_hash" if cloud_hash < local_hash else "local_wins_tie_hash"

    def _get_record_summary(self, table: str, record: Any) -> str:
        """Get a human-readable summary of a record for conflict display."""
        if table == "episodes":
            return record.objective[:50] + "..." if len(record.objective) > 50 else record.objective
        elif table == "notes":
            return record.content[:50] + "..." if len(record.content) > 50 else record.content
        elif table == "beliefs":
            return record.statement[:50] + "..." if len(record.statement) > 50 else record.statement
        elif table == "agent_values":
            stmt = record.statement[:40] + "..." if len(record.statement) > 40 else record.statement
            return f"{record.name}: {stmt}"
        elif table == "goals":
            return record.title[:50] + "..." if len(record.title) > 50 else record.title
        elif table == "drives":
            return f"{record.drive_type} (intensity: {record.intensity})"
        elif table == "relationships":
            return f"{record.entity_name} ({record.relationship_type})"
        elif table == "playbooks":
            if record.name:
                return f"{record.name} :: {record.description[:60] + '...' if len(record.description) > 60 else record.description}"
            return f"{table}:{record.id}"
        return f"{table}:{record.id}"

    def _record_to_dict(self, record: Any) -> Dict[str, Any]:
        """Convert a dataclass record to a dict for JSON storage."""
        from dataclasses import asdict

        try:
            d = asdict(record)
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            return d
        except Exception as e:
            logger.debug(f"Failed to serialize record, using fallback: {e}", exc_info=True)
            return {"id": getattr(record, "id", "unknown")}

    # === Pull Apply (CLI delegation target) ===

    # Table name aliases: HTTP pull payloads may use "values" instead of "agent_values".
    _TABLE_ALIASES = {"values": "agent_values"}

    # Dispatch table: canonical table name -> merge method name
    _MERGE_DISPATCH = {
        "episodes": "_merge_episode",
        "notes": "_merge_note",
        "beliefs": "_merge_belief",
        "agent_values": "_merge_value",
        "goals": "_merge_goal",
        "drives": "_merge_drive",
        "relationships": "_merge_relationship",
        "playbooks": "_merge_playbook",
    }

    def apply_pull_operation(self, op: dict) -> tuple[bool, int, Optional[str]]:
        """Apply a single HTTP pull operation dict to local storage.

        Args:
            op: Dict with keys "table", "record_id", "operation", "data".

        Returns:
            (success, pull_count, error_message):
              - (True, 1, None)  = applied
              - (True, 0, None)  = no-op (identical record)
              - (False, 0, msg)  = failure
        """
        table = op.get("table", "")
        record_id = op.get("record_id", "")
        operation = op.get("operation", "")
        data = op.get("data", {})

        # Normalize table aliases
        table = self._TABLE_ALIASES.get(table, table)

        # Validate operation type
        if operation == "delete":
            return (False, 0, "delete operations not supported")
        if operation not in ("upsert", "insert", "update"):
            return (False, 0, f"unsupported operation type: {operation}")

        # Validate table
        merge_method_name = self._MERGE_DISPATCH.get(table)
        if not merge_method_name:
            return (False, 0, f"unsupported table: {table}")

        try:
            record = self._record_from_pull_data(table, record_id, data)
        except (ValueError, KeyError, TypeError) as e:
            return (False, 0, str(e))

        merge_method = getattr(self, merge_method_name)
        try:
            pull_count, _conflict = merge_method(record)
        except Exception as e:
            return (False, 0, f"failed to merge record: {e}")

        return (True, pull_count, None)

    def _record_from_pull_data(self, table: str, record_id: str, data: dict) -> Any:
        """Convert HTTP pull data dict to a typed dataclass record.

        Sets provenance fields: source_type='external', source_entity='kernle:sync',
        stack_id=self._host.stack_id. Parses sync timestamps from ISO strings.

        Raises:
            ValueError: If required fields are missing for the table type.
        """
        from kernle.types import parse_datetime

        # Parse sync metadata timestamps
        local_updated_at = None
        cloud_synced_at = None
        version = data.get("version", 1)

        raw_lua = data.get("local_updated_at")
        if raw_lua:
            parsed = parse_datetime(raw_lua)
            if isinstance(parsed, datetime):
                local_updated_at = parsed

        raw_csa = data.get("cloud_synced_at")
        if raw_csa:
            parsed = parse_datetime(raw_csa)
            if isinstance(parsed, datetime):
                cloud_synced_at = parsed

        # Common sync metadata (all tables have these)
        common = {
            "id": record_id,
            "stack_id": self._host.stack_id,
            "local_updated_at": local_updated_at,
            "cloud_synced_at": cloud_synced_at,
            "version": version,
        }
        # Provenance fields (only tables with source_type/source_entity fields)
        provenance = {
            "source_type": "external",
            "source_entity": "kernle:sync",
        }

        if table == "episodes":
            return Episode(
                **common,
                **provenance,
                objective=data.get("objective", ""),
                outcome=data.get("outcome", data.get("outcome_description", "")),
                outcome_type=data.get("outcome_type", "neutral"),
                lessons=data.get("lessons", data.get("lessons_learned", [])),
                tags=data.get("tags", []),
            )

        if table == "notes":
            return Note(
                **common,
                **provenance,
                content=data.get("content", ""),
                note_type=data.get("note_type", "note"),
                tags=data.get("tags", []),
            )

        if table == "beliefs":
            return Belief(
                **common,
                **provenance,
                statement=data.get("statement", ""),
                belief_type=data.get("belief_type", "fact"),
                confidence=data.get("confidence", 0.8),
            )

        if table == "agent_values":
            return Value(
                **common,
                **provenance,
                name=data.get("name", ""),
                statement=data.get("statement", ""),
                priority=data.get("priority", 50),
            )

        if table == "goals":
            return Goal(
                **common,
                **provenance,
                title=data.get("title", ""),
                goal_type=data.get("goal_type", "task"),
                priority=data.get("priority", "medium"),
                status=data.get("status", "active"),
            )

        if table == "drives":
            drive_type = data.get("drive_type")
            if not drive_type:
                raise ValueError("missing required field: drive_type")
            return Drive(
                **common,
                **provenance,
                drive_type=drive_type,
                intensity=data.get("intensity", 0.5),
                focus_areas=data.get("focus_areas"),
            )

        if table == "relationships":
            entity_name = data.get("entity_name")
            if not entity_name:
                raise ValueError("missing required field: entity_name")
            return Relationship(
                **common,
                **provenance,
                entity_name=entity_name,
                entity_type=data.get("entity_type", "unknown"),
                relationship_type=data.get("relationship_type", "interaction"),
                sentiment=data.get("sentiment", 0.0),
                interaction_count=data.get("interaction_count", 0),
            )

        if table == "playbooks":
            name = data.get("name")
            if not name:
                raise ValueError("missing required field: name")
            # Playbook has no source_type/source_entity fields
            return Playbook(
                **common,
                name=name,
                description=data.get("description", ""),
                trigger_conditions=data.get("trigger_conditions", []),
                steps=data.get("steps", []),
                failure_modes=data.get("failure_modes", []),
                recovery_steps=data.get("recovery_steps"),
                tags=data.get("tags"),
            )

        raise ValueError(f"unsupported table: {table}")

    def _save_from_cloud(self, table: str, record: Any):
        """Save a record that came from cloud."""
        if table == "episodes":
            self._host.save_episode(record)
        elif table == "notes":
            self._host.save_note(record)
        elif table == "beliefs":
            self._host.save_belief(record)
        elif table == "agent_values":
            self._host.save_value(record)
        elif table == "goals":
            self._host.save_goal(record)
        elif table == "drives":
            self._host.save_drive(record)
        elif table == "relationships":
            self._host.save_relationship(record)
        elif table == "playbooks":
            self._host.save_playbook(record)
