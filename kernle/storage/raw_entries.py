"""Raw entry CRUD and FTS operations for Kernle storage.

Extracted from sqlite.py. Contains module-level functions for:
- save_raw, get_raw, list_raw, delete_raw
- FTS5 search and index updates
- Flat file sync (append, sync_from_files, import)
- Processing state (mark_raw_processed, mark_episode/note/belief_processed)
- Processing config and stack settings
- Row deserialization (_row_to_raw_entry)

SQLiteStorage keeps thin wrapper methods that delegate here.
"""

import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import RawEntry, parse_datetime

logger = logging.getLogger(__name__)
_MARKABLE_RECORD_TABLES = frozenset({"episodes", "notes", "beliefs"})


def _to_json(data: Any) -> Optional[str]:
    if data is None:
        return None
    return json.dumps(data)


def _from_json(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _safe_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        value = row[key]
        return value if value is not None else default
    except (IndexError, KeyError):
        return default


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    return parse_datetime(s)


def _utc_now() -> str:
    from .base import utc_now

    return utc_now()


def save_raw(
    conn: sqlite3.Connection,
    stack_id: str,
    blob: str,
    source: str,
    raw_dir: Path,
    queue_sync_fn: Callable,
    save_embedding_fn: Callable,
    should_sync_raw_fn: Callable,
) -> str:
    """Save a raw entry for later processing.

    Args:
        conn: Database connection.
        stack_id: Stack identifier.
        blob: The raw brain dump content.
        source: Source identifier (cli|mcp|sdk|import|unknown).
        raw_dir: Path to raw flat files directory.
        queue_sync_fn: Function to queue sync operations.
        save_embedding_fn: Function to save embeddings.
        should_sync_raw_fn: Function to check if raw sync is enabled.

    Returns:
        The raw entry ID.
    """
    # Normalize source to valid enum values
    valid_sources = {"cli", "mcp", "sdk", "import", "unknown"}
    if source == "manual":
        source = "cli"
    elif source not in valid_sources:
        if "auto" in source.lower():
            source = "sdk"
        else:
            source = "unknown"

    # Size warnings (don't reject, let anxiety system handle)
    blob_size = len(blob.encode("utf-8"))
    if blob_size > 50 * 1024 * 1024:  # 50MB - reject
        raise ValueError(
            f"Raw entry too large ({blob_size / 1024 / 1024:.1f}MB). "
            "Consider breaking into smaller chunks or processing immediately."
        )
    elif blob_size > 10 * 1024 * 1024:  # 10MB
        logger.warning(f"Extremely large raw entry ({blob_size / 1024 / 1024:.1f}MB)")
    elif blob_size > 1 * 1024 * 1024:  # 1MB
        logger.warning(f"Very large raw entry ({blob_size / 1024:.0f}KB) - consider processing")
    elif blob_size > 100 * 1024:  # 100KB
        logger.info(f"Large raw entry ({blob_size / 1024:.0f}KB)")

    raw_id = str(uuid.uuid4())
    now = _utc_now()

    # 1. Write to flat file (blob acts as flat file content)
    append_raw_to_file(raw_dir, raw_id, blob, now, source, None)

    # 2. Index in SQLite with both blob and legacy columns for compatibility
    conn.execute(
        """
        INSERT INTO raw_entries
        (id, stack_id, blob, captured_at, source, processed, processed_into,
         content, timestamp, tags, confidence, source_type,
         local_updated_at, cloud_synced_at, version, deleted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            raw_id,
            stack_id,
            blob,  # Primary blob field
            now,  # captured_at
            source,
            0,  # processed = False
            None,  # processed_into
            blob,  # Legacy content field (same as blob for new entries)
            now,  # Legacy timestamp field (same as captured_at)
            None,  # Legacy tags field (removed)
            1.0,  # confidence (deprecated)
            "direct_experience",  # source_type (deprecated)
            now,
            None,
            1,
            0,
        ),
    )

    # Update FTS index for keyword search
    update_raw_fts(conn, raw_id, blob)

    # Queue for sync (if raw sync is enabled - off by default)
    if should_sync_raw_fn():
        raw_data = _to_json(
            {
                "id": raw_id,
                "stack_id": stack_id,
                "blob": blob,
                "captured_at": now,
                "source": source,
                "processed": False,
            }
        )
        queue_sync_fn(conn, "raw_entries", raw_id, "upsert", data=raw_data)

    # Save embedding for search (on blob content)
    save_embedding_fn(conn, "raw_entries", raw_id, blob)

    conn.commit()
    return raw_id


def should_sync_raw() -> bool:
    """Check if raw entries should be synced to cloud.

    Raw sync is OFF by default for security (raw blobs often contain
    accidental secrets). Users must explicitly enable it.
    """
    from kernle.utils import get_kernle_home

    # Check environment variable
    raw_sync_env = os.environ.get("KERNLE_RAW_SYNC", "").lower()
    if raw_sync_env in ("true", "1", "yes", "on"):
        return True
    if raw_sync_env in ("false", "0", "no", "off"):
        return False

    # Check config file
    config_path = get_kernle_home() / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get("sync", {}).get("raw", False)
        except (json.JSONDecodeError, OSError):
            pass  # Sync config missing or malformed — use defaults

    # Default: OFF for security
    return False


def update_raw_fts(conn: sqlite3.Connection, raw_id: str, blob: str) -> None:
    """Update FTS5 index for a raw entry."""
    try:
        # Get rowid for the entry
        result = conn.execute("SELECT rowid FROM raw_entries WHERE id = ?", (raw_id,)).fetchone()
        if result:
            rowid = result[0]
            # Insert into FTS index
            conn.execute("INSERT INTO raw_fts(rowid, blob) VALUES (?, ?)", (rowid, blob))
    except sqlite3.OperationalError as e:
        # FTS5 might not be available
        if "no such table" not in str(e).lower():
            logger.debug(f"FTS update failed: {e}", exc_info=True)


def append_raw_to_file(
    raw_dir: Path,
    raw_id: str,
    content: str,
    timestamp: str,
    source: str,
    tags: Optional[List[str]] = None,
) -> None:
    """Append a raw entry to the daily flat file.

    File format (greppable, human-readable):
    ```
    ## HH:MM:SS [id_prefix] source
    Content goes here
    Tags: tag1, tag2

    ```
    """
    try:
        # Parse date from timestamp for filename
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")

        daily_file = raw_dir / f"{date_str}.md"

        # Build entry
        lines = []
        lines.append(f"## {time_str} [{raw_id[:8]}] {source}")
        lines.append(content)
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        lines.append("")  # Blank line separator

        # Append to file
        with open(daily_file, "a", encoding="utf-8") as f:
            # Add header if new file
            if daily_file.stat().st_size == 0 if daily_file.exists() else True:
                f.write(f"# Raw Captures - {date_str}\n\n")
            f.write("\n".join(lines) + "\n")

    except Exception as e:
        logger.warning(f"Failed to write raw entry to flat file: {e}", exc_info=True)
        # Don't fail - SQLite is the backup


def get_raw_dir(raw_dir: Path) -> Path:
    """Get the path to the raw flat files directory."""
    return raw_dir


def get_raw_files(raw_dir: Path) -> List[Path]:
    """Get list of raw flat files, sorted by date descending."""
    if not raw_dir.exists():
        return []
    files = sorted(raw_dir.glob("*.md"), reverse=True)
    return files


def sync_raw_from_files(
    conn: sqlite3.Connection,
    stack_id: str,
    raw_dir: Path,
    save_embedding_fn: Callable,
) -> Dict[str, Any]:
    """Sync raw entries from flat files into SQLite.

    Parses flat files and imports any entries not already in SQLite.
    This enables bidirectional editing - add entries via vim, then sync.

    Returns:
        Dict with imported_count, skipped_count, errors
    """
    result = {
        "imported": 0,
        "skipped": 0,
        "errors": [],
        "files_processed": 0,
    }

    files = get_raw_files(raw_dir)

    # Get existing IDs for quick lookup
    rows = conn.execute("SELECT id FROM raw_entries WHERE stack_id = ?", (stack_id,)).fetchall()
    existing_ids = {row["id"] for row in rows}

    # Pattern to match entry headers: ## HH:MM:SS [id_prefix] source
    # ID can be alphanumeric (for manually created entries)
    header_pattern = re.compile(r"^## (\d{2}:\d{2}:\d{2}) \[([a-zA-Z0-9]+)\] ([\w-]+)$")

    for file_path in files:
        result["files_processed"] += 1
        try:
            # Extract date from filename (2026-01-28.md)
            date_str = file_path.stem  # "2026-01-28"

            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Split into entries
            lines = file_content.split("\n")
            current_entry = None

            for line in lines:
                header_match = header_pattern.match(line)

                if header_match:
                    # Save previous entry if exists
                    if current_entry and current_entry.get("content_lines"):
                        current_entry["content"] = "\n".join(current_entry["content_lines"])
                        import_raw_entry(
                            conn,
                            stack_id,
                            current_entry,
                            existing_ids,
                            result,
                            save_embedding_fn,
                        )

                    # Start new entry
                    time_str, id_prefix, source = header_match.groups()
                    current_entry = {
                        "id_prefix": id_prefix,
                        "timestamp": f"{date_str}T{time_str}",
                        "source": source,
                        "content_lines": [],
                        "tags": None,
                    }
                elif current_entry is not None:
                    # Check for tags line
                    if line.startswith("Tags: "):
                        current_entry["tags"] = [t.strip() for t in line[6:].split(",")]
                    elif line.startswith("# Raw Captures"):
                        pass  # Skip header
                    elif line.strip():  # Non-empty content
                        current_entry["content_lines"].append(line)

            # Don't forget last entry
            if current_entry and current_entry.get("content_lines"):
                current_entry["content"] = "\n".join(current_entry["content_lines"])
                import_raw_entry(
                    conn,
                    stack_id,
                    current_entry,
                    existing_ids,
                    result,
                    save_embedding_fn,
                )

        except Exception as e:
            logger.debug(
                "Raw entry file import failed for %s: %s", file_path.name, e, exc_info=True
            )
            result["errors"].append(f"{file_path.name}: {str(e)}")

    return result


def import_raw_entry(
    conn: sqlite3.Connection,
    stack_id: str,
    entry: Dict[str, Any],
    existing_ids: set,
    result: Dict[str, Any],
    save_embedding_fn: Callable,
) -> None:
    """Import a single raw entry if not already in SQLite."""
    # Check if any existing ID starts with this prefix
    id_prefix = entry.get("id_prefix", "")
    matching_ids = [eid for eid in existing_ids if eid.startswith(id_prefix)]

    if matching_ids:
        result["skipped"] += 1
        return

    # Generate new ID (use prefix + random suffix to maintain traceability)
    new_id = id_prefix + str(uuid.uuid4())[8:]  # Keep prefix, new suffix
    content = "\n".join(entry.get("content_lines", []))

    if not content.strip():
        result["skipped"] += 1
        return

    try:
        now = _utc_now()
        timestamp = entry.get("timestamp", now)

        conn.execute(
            """
            INSERT INTO raw_entries
            (id, stack_id, content, timestamp, source, processed, processed_into, tags,
             confidence, source_type, local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                new_id,
                stack_id,
                content,
                timestamp,
                entry.get("source", "file-sync"),
                0,
                None,
                _to_json(entry.get("tags")),
                1.0,
                "file_import",
                now,
                None,
                1,
                0,
            ),
        )
        save_embedding_fn(conn, "raw_entries", new_id, content)
        conn.commit()

        existing_ids.add(new_id)
        result["imported"] += 1

    except Exception as e:
        logger.debug("Raw entry import failed for %s: %s", id_prefix, e, exc_info=True)
        result["errors"].append(f"Entry {id_prefix}: {str(e)}")


def get_raw(
    conn: sqlite3.Connection,
    stack_id: str,
    raw_id: str,
) -> Optional[RawEntry]:
    """Get a specific raw entry by ID."""
    row = conn.execute(
        "SELECT * FROM raw_entries WHERE id = ? AND stack_id = ? AND deleted = 0",
        (raw_id, stack_id),
    ).fetchone()

    return row_to_raw_entry(row) if row else None


def list_raw(
    conn: sqlite3.Connection,
    stack_id: str,
    processed: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[RawEntry]:
    """Get raw entries, optionally filtered by processed state.

    Args:
        conn: Database connection.
        stack_id: Stack identifier.
        processed: Filter by processed state (None = all).
        limit: Maximum entries to return. Must be positive.
        offset: Number of entries to skip. Must be non-negative.

    Raises:
        ValueError: If limit <= 0 or offset < 0.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")

    query = "SELECT * FROM raw_entries WHERE stack_id = ? AND deleted = 0"
    params: List[Any] = [stack_id]

    if processed is not None:
        query += " AND processed = ?"
        params.append(1 if processed else 0)

    query += " ORDER BY captured_at DESC, id DESC LIMIT ? OFFSET ?"
    params.append(limit)
    params.append(offset)

    rows = conn.execute(query, params).fetchall()
    return [row_to_raw_entry(row) for row in rows]


def find_by_id_prefix(
    conn: sqlite3.Connection,
    stack_id: str,
    prefix: str,
    limit: int = 10,
) -> List[RawEntry]:
    """Find raw entries whose ID starts with the given prefix.

    Uses database-level LIKE query for efficiency. Handles LIKE-special
    characters (%, _) by escaping them.

    Args:
        conn: Database connection.
        stack_id: Stack identifier.
        prefix: ID prefix to search for.
        limit: Maximum matches to return.

    Returns:
        List of matching RawEntry objects.
    """
    # Escape LIKE-special characters in the prefix
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"{escaped}%"

    rows = conn.execute(
        "SELECT * FROM raw_entries WHERE id LIKE ? ESCAPE '\\' AND stack_id = ? AND deleted = 0 "
        "ORDER BY captured_at DESC LIMIT ?",
        (pattern, stack_id, limit),
    ).fetchall()
    return [row_to_raw_entry(row) for row in rows]


def search_raw_fts(
    conn: sqlite3.Connection,
    stack_id: str,
    query: str,
    limit: int = 50,
) -> List[RawEntry]:
    """Search raw entries using FTS5 keyword search.

    This is a safety net for when backlogs accumulate.
    """
    try:
        # FTS5 MATCH query with relevance ranking
        rows = conn.execute(
            """
            SELECT r.* FROM raw_entries r
            JOIN raw_fts f ON r.rowid = f.rowid
            WHERE r.stack_id = ? AND r.deleted = 0
            AND raw_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (stack_id, query, limit),
        ).fetchall()
        return [row_to_raw_entry(row) for row in rows]
    except sqlite3.OperationalError as e:
        # FTS5 not available, fall back to LIKE search
        if "no such table" in str(e).lower() or "fts5" in str(e).lower():
            logger.debug("FTS5 not available, using LIKE fallback", exc_info=True)
            escaped_query = escape_like_pattern(query)
            rows = conn.execute(
                """
                SELECT * FROM raw_entries
                WHERE stack_id = ? AND deleted = 0
                AND (COALESCE(blob, content, '') LIKE ? ESCAPE '\\')
                ORDER BY COALESCE(captured_at, timestamp) DESC
                LIMIT ?
                """,
                (stack_id, f"%{escaped_query}%", limit),
            ).fetchall()
            return [row_to_raw_entry(row) for row in rows]
        raise


def escape_like_pattern(pattern: str) -> str:
    """Escape LIKE pattern special characters to prevent pattern injection."""
    return pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def mark_raw_processed(
    conn: sqlite3.Connection,
    stack_id: str,
    raw_id: str,
    processed_into: List[str],
    queue_sync_fn: Callable,
) -> bool:
    """Mark a raw entry as processed into other memories."""
    now = _utc_now()

    cursor = conn.execute(
        """
        UPDATE raw_entries SET
            processed = 1,
            processed_into = ?,
            local_updated_at = ?,
            version = version + 1
        WHERE id = ? AND stack_id = ? AND deleted = 0
    """,
        (_to_json(processed_into), now, raw_id, stack_id),
    )
    if cursor.rowcount > 0:
        queue_sync_fn(conn, "raw_entries", raw_id, "upsert")
        conn.commit()
        return True
    return False


def mark_processed(
    conn: sqlite3.Connection,
    stack_id: str,
    table: str,
    record_id: str,
    queue_sync_fn: Callable,
) -> bool:
    """Mark a record as processed for promotion.

    Works for episodes, notes, and beliefs tables.
    """
    if table not in _MARKABLE_RECORD_TABLES:
        logger.warning("Attempted mark_processed with non-whitelisted table: %s", table)
        return False

    now = _utc_now()
    cursor = conn.execute(
        f"""
        UPDATE {table} SET
            processed = 1,
            local_updated_at = ?,
            version = version + 1
        WHERE id = ? AND stack_id = ? AND deleted = 0
    """,
        (now, record_id, stack_id),
    )
    if cursor.rowcount > 0:
        queue_sync_fn(conn, table, record_id, "upsert")
        conn.commit()
        return True
    return False


def get_processing_config(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get all processing configuration entries."""
    rows = conn.execute("SELECT * FROM processing_config ORDER BY layer_transition").fetchall()
    result = []
    for row in rows:
        result.append(
            {
                "layer_transition": row["layer_transition"],
                "enabled": bool(row["enabled"]),
                "model_id": row["model_id"],
                "quantity_threshold": row["quantity_threshold"],
                "valence_threshold": row["valence_threshold"],
                "time_threshold_hours": row["time_threshold_hours"],
                "batch_size": row["batch_size"],
                "max_sessions_per_day": row["max_sessions_per_day"],
                "updated_at": row["updated_at"],
            }
        )
    return result


def set_processing_config(
    conn: sqlite3.Connection,
    layer_transition: str,
    *,
    enabled: Optional[bool] = None,
    model_id: Optional[str] = None,
    quantity_threshold: Optional[int] = None,
    valence_threshold: Optional[float] = None,
    time_threshold_hours: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_sessions_per_day: Optional[int] = None,
) -> bool:
    """Upsert a processing configuration entry."""
    now = _utc_now()
    # Check if exists
    existing = conn.execute(
        "SELECT * FROM processing_config WHERE layer_transition = ?",
        (layer_transition,),
    ).fetchone()

    if existing:
        # Build SET clause from non-None values
        updates: Dict[str, Any] = {"updated_at": now}
        if enabled is not None:
            updates["enabled"] = 1 if enabled else 0
        if model_id is not None:
            updates["model_id"] = model_id
        if quantity_threshold is not None:
            updates["quantity_threshold"] = quantity_threshold
        if valence_threshold is not None:
            updates["valence_threshold"] = valence_threshold
        if time_threshold_hours is not None:
            updates["time_threshold_hours"] = time_threshold_hours
        if batch_size is not None:
            updates["batch_size"] = batch_size
        if max_sessions_per_day is not None:
            updates["max_sessions_per_day"] = max_sessions_per_day

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [layer_transition]
        conn.execute(
            f"UPDATE processing_config SET {set_clause} WHERE layer_transition = ?",
            values,
        )
    else:
        conn.execute(
            """
            INSERT INTO processing_config
                (layer_transition, enabled, model_id, quantity_threshold,
                 valence_threshold, time_threshold_hours, batch_size,
                 max_sessions_per_day, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                layer_transition,
                1 if (enabled is None or enabled) else 0,
                model_id,
                quantity_threshold,
                valence_threshold,
                time_threshold_hours,
                batch_size or 10,
                max_sessions_per_day,
                now,
            ),
        )
    conn.commit()
    return True


def get_stack_setting(
    conn: sqlite3.Connection,
    stack_id: str,
    key: str,
) -> Optional[str]:
    """Get a stack setting value by key (scoped to this stack)."""
    row = conn.execute(
        "SELECT value FROM stack_settings WHERE stack_id = ? AND key = ?",
        (stack_id, key),
    ).fetchone()
    return row["value"] if row else None


def set_stack_setting(
    conn: sqlite3.Connection,
    stack_id: str,
    key: str,
    value: str,
) -> None:
    """Set a stack setting (upsert, scoped to this stack)."""
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO stack_settings (stack_id, key, value, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(stack_id, key) DO UPDATE SET value = ?, updated_at = ?
        """,
        (stack_id, key, value, now, value, now),
    )
    conn.commit()


def get_all_stack_settings(
    conn: sqlite3.Connection,
    stack_id: str,
) -> Dict[str, str]:
    """Get all stack settings as a dict (scoped to this stack)."""
    rows = conn.execute(
        "SELECT key, value FROM stack_settings WHERE stack_id = ? ORDER BY key",
        (stack_id,),
    ).fetchall()
    return {row["key"]: row["value"] for row in rows}


def delete_raw(
    conn: sqlite3.Connection,
    stack_id: str,
    raw_id: str,
    queue_sync_fn: Callable,
) -> bool:
    """Delete a raw entry (soft delete by marking deleted=1)."""
    now = _utc_now()

    cursor = conn.execute(
        """
        UPDATE raw_entries SET
            deleted = 1,
            local_updated_at = ?,
            version = version + 1
        WHERE id = ? AND stack_id = ? AND deleted = 0
    """,
        (now, raw_id, stack_id),
    )
    if cursor.rowcount > 0:
        queue_sync_fn(conn, "raw_entries", raw_id, "delete")
        conn.commit()
        return True
    return False


def row_to_raw_entry(row: sqlite3.Row) -> RawEntry:
    """Convert a row to a RawEntry.

    Handles both new (blob/captured_at) and legacy (content/timestamp) schemas.
    """
    # Get blob - prefer blob field, fall back to content for legacy data
    blob = _safe_get(row, "blob", None) or _safe_get(row, "content", "")

    # Get captured_at - prefer captured_at, fall back to timestamp for legacy data
    captured_at_str = _safe_get(row, "captured_at", None) or _safe_get(row, "timestamp", None)
    captured_at = _parse_dt(captured_at_str)

    return RawEntry(
        id=row["id"],
        stack_id=row["stack_id"],
        blob=blob,
        captured_at=captured_at,
        source=row["source"],
        processed=bool(row["processed"]),
        processed_into=_from_json(row["processed_into"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        # Legacy fields (deprecated)
        content=_safe_get(row, "content", None),
        timestamp=_parse_dt(_safe_get(row, "timestamp", None)),
        tags=_from_json(_safe_get(row, "tags", None)),
        confidence=_safe_get(row, "confidence", 1.0),
        source_type=_safe_get(row, "source_type", "direct_experience"),
    )
