"""SQLite storage backend for Kernle.

Local-first storage with:
- SQLite for structured data
- sqlite-vec for vector search (semantic search)
- Sync metadata for cloud synchronization
"""

import contextlib
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kernle.protocols import ModelStatus
from kernle.types import VALID_SOURCE_TYPE_VALUES, SourceType
from kernle.utils import get_kernle_home

from . import beliefs_crud as _beliefs_crud
from . import diagnostic_crud as _diagnostic_crud
from . import drives_crud as _drives_crud
from . import episodes_crud as _episodes_crud
from . import epoch_crud as _epoch_crud
from . import goals_crud as _goals_crud
from . import meta_memory_ops as _meta_memory_ops
from . import narrative_summary_crud as _narrative_summary_crud
from . import notes_crud as _notes_crud
from . import playbooks_crud as _playbooks_crud
from . import relationships_crud as _relationships_crud
from . import search_impl as _search_impl
from . import stats_ops as _stats_ops
from . import suggestions_crud as _suggestions_crud
from . import trust_crud as _trust_crud
from . import values_crud as _values_crud
from .base import (
    Belief,
    DiagnosticReport,
    DiagnosticSession,
    Drive,
    EntityModel,
    Episode,
    Epoch,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    Relationship,
    RelationshipHistoryEntry,
    SearchResult,
    SelfNarrative,
    Summary,
    TrustAssessment,
    Value,
    VersionConflictError,
    parse_datetime,
    utc_now,
)
from .cloud import CloudClient
from .embeddings import (
    EmbeddingProvider,
    HashEmbedder,
    pack_embedding,
)
from .flat_files import (
    init_flat_files,
    sync_beliefs_to_file,
    sync_goals_to_file,
    sync_relationships_to_file,
    sync_values_to_file,
)
from .health import get_health_check_stats as _get_health_check_stats
from .health import log_health_check as _log_health_check
from .lineage import check_derived_from_cycle
from .memory_crud import (
    _row_to_belief as _mc_row_to_belief,
)
from .memory_crud import (
    _row_to_diagnostic_report as _mc_row_to_diagnostic_report,
)
from .memory_crud import (
    _row_to_diagnostic_session as _mc_row_to_diagnostic_session,
)
from .memory_crud import (
    _row_to_drive as _mc_row_to_drive,
)
from .memory_crud import (
    _row_to_entity_model as _mc_row_to_entity_model,
)
from .memory_crud import (
    _row_to_episode as _mc_row_to_episode,
)
from .memory_crud import (
    _row_to_epoch as _mc_row_to_epoch,
)
from .memory_crud import (
    _row_to_goal as _mc_row_to_goal,
)
from .memory_crud import (
    _row_to_note as _mc_row_to_note,
)
from .memory_crud import (
    _row_to_playbook as _mc_row_to_playbook,
)
from .memory_crud import (
    _row_to_relationship as _mc_row_to_relationship,
)
from .memory_crud import (
    _row_to_relationship_history as _mc_row_to_relationship_history,
)
from .memory_crud import (
    _row_to_self_narrative as _mc_row_to_self_narrative,
)
from .memory_crud import (
    _row_to_suggestion as _mc_row_to_suggestion,
)
from .memory_crud import (
    _row_to_summary as _mc_row_to_summary,
)
from .memory_crud import (
    _row_to_value as _mc_row_to_value,
)
from .memory_ops import MemoryOps
from .raw_entries import (
    append_raw_to_file,
    escape_like_pattern,
    row_to_raw_entry,
    should_sync_raw,
    update_raw_fts,
)
from .raw_entries import (
    delete_raw as _delete_raw,
)
from .raw_entries import (
    find_by_id_prefix as _find_by_id_prefix,
)
from .raw_entries import (
    get_all_stack_settings as _get_all_stack_settings,
)
from .raw_entries import (
    get_processing_config as _get_processing_config,
)
from .raw_entries import (
    get_raw as _get_raw,
)
from .raw_entries import (
    get_raw_files as _get_raw_files,
)
from .raw_entries import (
    get_stack_setting as _get_stack_setting,
)
from .raw_entries import (
    list_raw as _list_raw,
)
from .raw_entries import (
    mark_processed as _mark_processed,
)
from .raw_entries import (
    mark_raw_processed as _mark_raw_processed,
)
from .raw_entries import (
    save_raw as _save_raw,
)
from .raw_entries import (
    search_raw_fts as _search_raw_fts,
)
from .raw_entries import (
    set_processing_config as _set_processing_config,
)
from .raw_entries import (
    set_stack_setting as _set_stack_setting,
)
from .raw_entries import (
    sync_raw_from_files as _sync_raw_from_files,
)
from .schema import ALLOWED_TABLES as ALLOWED_TABLES  # noqa: F401 — re-export
from .schema import SCHEMA_VERSION as SCHEMA_VERSION  # noqa: F401 — re-export
from .schema import (
    ensure_raw_fts5,
    init_db,
    migrate_schema,
    validate_table_name,
)
from .sync_engine import SyncEngine

if TYPE_CHECKING:
    from .base import Storage as StorageProtocol

logger = logging.getLogger(__name__)


def _normalize_source_type(source_type: Any):
    """Normalize source_type for metadata writes."""
    if isinstance(source_type, SourceType):
        return source_type.value

    if not isinstance(source_type, str):
        raise ValueError("source_type must be a string or SourceType")

    normalized = source_type.strip().lower()
    if not normalized:
        raise ValueError("source_type cannot be empty")

    if normalized not in VALID_SOURCE_TYPE_VALUES:
        raise ValueError(
            f"Invalid source_type: '{source_type}'. "
            f"Valid values: {sorted(VALID_SOURCE_TYPE_VALUES)}"
        )

    return normalized


# Tables that are intentionally local-only and should never be enqueued for cloud sync.
LOCAL_ONLY_SYNC_TABLES = frozenset({"memory_suggestions"})


# NOTE: SCHEMA_VERSION, ALLOWED_TABLES, validate_table_name, SCHEMA, and VECTOR_SCHEMA
# have been moved to storage/schema.py and are imported at the top of this file.


class SQLiteStorage:
    """SQLite-based local storage for Kernle.

    Features:
    - Zero-config local storage
    - Semantic search with sqlite-vec (when available)
    - Sync metadata for cloud synchronization
    - Offline-first with automatic queue when disconnected
    """

    # Connectivity check timeout (seconds)
    CONNECTIVITY_TIMEOUT = 5.0
    # Cloud search timeout (seconds)
    CLOUD_SEARCH_TIMEOUT = 3.0
    # Retry window for transient embedding backend failures (seconds)
    EMBEDDER_RETRY_SECONDS = 60

    def __init__(
        self,
        stack_id: str,
        db_path: Optional[Path] = None,
        cloud_storage: Optional["StorageProtocol"] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        # Defense-in-depth: reject path traversal in stack_id before using in paths
        if not stack_id or not stack_id.strip():
            raise ValueError("Stack ID cannot be empty")
        if "/" in stack_id or "\\" in stack_id:
            raise ValueError("Stack ID must not contain path separators")
        if stack_id.strip() in (".", ".."):
            raise ValueError("Stack ID must not be a relative path component")
        if ".." in stack_id.split("."):
            raise ValueError("Stack ID must not contain path traversal sequences")

        self.stack_id = stack_id
        self.db_path = self._resolve_db_path(db_path)
        self.cloud_storage = cloud_storage  # For sync

        # Connectivity cache
        self._last_connectivity_check: Optional[datetime] = None
        self._is_online_cached: bool = False
        self._connectivity_cache_ttl = 30  # seconds

        # Cloud search client
        self._cloud = CloudClient(stack_id, self.CLOUD_SEARCH_TIMEOUT)

        # Memory lifecycle operations
        self._memory_ops = MemoryOps(
            connect_fn=self._connect,
            stack_id=stack_id,
            now_fn=self._now,
            safe_get_fn=self._safe_get,
            queue_sync_fn=self._queue_sync,
            validate_table_name_fn=validate_table_name,
        )

        # Sync engine
        self._sync_engine = SyncEngine(self, validate_table_name)

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for sqlite-vec first
        self._has_vec = self._check_sqlite_vec()

        # Initialize embedder
        self._preferred_embedder = embedder or HashEmbedder()
        self._embedder = self._preferred_embedder
        self._embedder_fallback = self._make_fallback_embedder(self._preferred_embedder)
        self._embedder_retry_at: Optional[datetime] = None

        # Initialize flat file directories for all memory layers
        self._agent_dir = self.db_path.parent / stack_id
        self._agent_dir.mkdir(parents=True, exist_ok=True)

        self._raw_dir = self._agent_dir / "raw"
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        # Identity files (single files, always complete)
        self._beliefs_file = self._agent_dir / "beliefs.md"
        self._values_file = self._agent_dir / "values.md"
        self._relationships_file = self._agent_dir / "relationships.md"
        self._goals_file = self._agent_dir / "goals.md"

        # Initialize database
        self._init_db()

        # Initialize flat files from existing data
        self._init_flat_files()

        if not self._has_vec:
            logger.info("sqlite-vec not available, semantic search will use text matching")

    @staticmethod
    def _make_fallback_embedder(embedder: EmbeddingProvider) -> Optional[EmbeddingProvider]:
        if isinstance(embedder, HashEmbedder):
            return None
        return HashEmbedder(dim=embedder.dimension)

    def _maybe_restore_preferred_embedder(self) -> None:
        if self._embedder_fallback is None:
            return
        if self._embedder is not self._embedder_fallback:
            return
        if self._embedder_retry_at is None:
            return
        if datetime.now(timezone.utc) < self._embedder_retry_at:
            return

        self._embedder = self._preferred_embedder
        logger.info(
            "Attempting to restore preferred embedding provider at %s",
            datetime.now(timezone.utc).isoformat(),
        )

    def _embed_text(self, text: str, *, context: str) -> Optional[list[float]]:
        self._maybe_restore_preferred_embedder()

        try:
            return self._embedder.embed(text)
        except Exception as exc:
            if isinstance(self._embedder, HashEmbedder):
                logger.warning("Hash embedder failed for %s: %s", context, exc, exc_info=True)
                return None

            error_class = getattr(exc, "error_class", "unknown")
            provider_name = type(self._embedder).__name__
            if self._embedder_fallback is None:
                status = ModelStatus(
                    provider=provider_name,
                    available=False,
                    error_class=error_class,
                    error_message=str(exc),
                    degraded=False,
                )
                logger.warning(
                    "Embedding provider failed for %s and no fallback is configured: %s",
                    context,
                    exc,
                    extra={
                        "model_status": status,
                        "error_class": error_class,
                        "provider": provider_name,
                    },
                    exc_info=True,
                )
                return None

            status = ModelStatus(
                provider=provider_name,
                available=False,
                error_class=error_class,
                error_message=str(exc),
                degraded=True,
            )
            logger.warning(
                "Preferred embedding provider failed for %s; falling back to hash embeddings until retry: %s",
                context,
                exc,
                extra={
                    "model_status": status,
                    "error_class": error_class,
                    "provider": provider_name,
                    "degraded": True,
                },
                exc_info=True,
            )
            self._embedder_retry_at = datetime.now(timezone.utc) + timedelta(
                seconds=self.EMBEDDER_RETRY_SECONDS
            )
            self._embedder = self._embedder_fallback
            try:
                return self._embedder_fallback.embed(text)
            except Exception as fallback_exc:
                logger.warning(
                    "Fallback embedding failed for %s: %s", context, fallback_exc, exc_info=True
                )
                return None

    def _init_flat_files(self) -> None:
        """Initialize flat files from existing database data."""
        init_flat_files(
            self._beliefs_file,
            self._values_file,
            self._relationships_file,
            self._goals_file,
            self._sync_beliefs_to_file,
            self._sync_values_to_file,
            self._sync_goals_to_file,
            self._sync_relationships_to_file,
        )

    def _resolve_db_path(self, db_path: Optional[Path]) -> Path:
        """Resolve the database path, falling back to temp dir if home is not writable."""
        import tempfile

        if db_path is not None:
            return self._validate_db_path(db_path)

        default_path = get_kernle_home() / "memories.db"
        try:
            default_path.parent.mkdir(parents=True, exist_ok=True)
            return self._validate_db_path(default_path)
        except (OSError, PermissionError) as e:
            # Home dir not writable (sandboxed/container/CI environment)
            fallback_dir = Path(tempfile.gettempdir()) / ".kernle"
            fallback_path = fallback_dir / "memories.db"
            logger.warning(
                f"Cannot write to {default_path.parent} ({e}), " f"falling back to {fallback_dir}",
                exc_info=True,
            )
            return self._validate_db_path(fallback_path)

    def _validate_db_path(self, db_path: Path) -> Path:
        """Validate database path to prevent path traversal attacks."""
        import tempfile

        try:
            # Resolve to absolute path to prevent directory traversal
            resolved_path = db_path.resolve()

            # Ensure it's within a safe directory (user's home, system temp, or /tmp)
            home_path = Path.home().resolve()
            tmp_path = Path("/tmp").resolve()
            system_temp = Path(tempfile.gettempdir()).resolve()

            # Use is_relative_to() for secure path validation (Python 3.9+)
            is_safe = (
                resolved_path.is_relative_to(home_path)
                or resolved_path.is_relative_to(tmp_path)
                or resolved_path.is_relative_to(system_temp)
            )

            # Also allow /var/folders on macOS (where tempfile creates dirs)
            if not is_safe:
                try:
                    var_folders = Path("/var/folders").resolve()
                    private_var_folders = Path("/private/var/folders").resolve()
                    is_safe = resolved_path.is_relative_to(
                        var_folders
                    ) or resolved_path.is_relative_to(private_var_folders)
                except (OSError, ValueError):
                    pass  # macOS realpath resolution failed — skip check

            if not is_safe:
                raise ValueError("Database path must be within user home or temp directory")

            return resolved_path

        except (OSError, ValueError) as e:
            logger.error(f"Invalid database path: {e}", exc_info=True)
            raise ValueError(f"Invalid database path: {e}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with sqlite-vec loaded if available.

        IMPORTANT: Callers should use this with contextlib.closing() or
        call conn.close() explicitly to avoid resource warnings:

            from contextlib import closing
            with closing(self._get_conn()) as conn:
                ...

        Or use the _connect() context manager which handles both
        commit/rollback and close.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        if self._has_vec:
            self._load_vec(conn)
        return conn

    @contextlib.contextmanager
    def _connect(self):
        """Context manager that handles transactions AND closes connection.

        Use this instead of `with self._connect() as conn:` to avoid
        unclosed connection warnings. This handles:
        - Transaction commit on success
        - Transaction rollback on exception
        - Connection close in all cases
        """
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            logger.debug(f"Transaction failed, rolling back: {e}", exc_info=True)
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self):
        """Close any resources.

        Since we create connections per-operation with context managers,
        this primarily exists for API compatibility and explicit cleanup.
        """
        pass  # No persistent connections to close

    # === Cloud Search Methods (delegated to CloudClient) ===

    def has_cloud_credentials(self) -> bool:
        """Check if cloud credentials are available."""
        return self._cloud.has_cloud_credentials()

    def cloud_health_check(self, timeout: float = 3.0) -> Dict[str, Any]:
        """Test cloud backend connectivity."""
        return self._cloud.cloud_health_check(timeout)

    def _cloud_search(
        self,
        query: str,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[List[SearchResult]]:
        """Search memories via cloud backend."""
        return self._cloud._cloud_search(query, limit, record_types, timeout)

    def _init_db(self):
        """Initialize the database schema. Delegates to schema.init_db()."""
        with self._connect() as conn:
            init_db(
                conn=conn,
                stack_id=self.stack_id,
                has_vec=self._has_vec,
                embedder_dimension=self._embedder.dimension,
                load_vec_fn=self._load_vec,
                db_path=self.db_path,
                agent_dir=self._agent_dir,
            )

    def _ensure_raw_fts5(self, conn: sqlite3.Connection):
        """Create FTS5 virtual table. Delegates to schema.ensure_raw_fts5()."""
        ensure_raw_fts5(conn)

    def _migrate_schema(self, conn: sqlite3.Connection):
        """Run schema migrations. Delegates to schema.migrate_schema()."""
        migrate_schema(conn, self.stack_id)

    def _check_sqlite_vec(self) -> bool:
        """Check if sqlite-vec extension is available."""
        try:
            import sqlite_vec

            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.close()
            return True
        except ImportError:
            logger.debug("sqlite-vec package not installed", exc_info=True)
            return False
        except Exception as e:
            logger.debug(f"sqlite-vec not available: {e}", exc_info=True)
            return False

    def _load_vec(self, conn: sqlite3.Connection):
        """Load sqlite-vec extension into connection."""
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception as e:
            logger.error(f"Could not load sqlite-vec: {e}", exc_info=True)

    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return utc_now()

    def _parse_datetime(self, s: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        return parse_datetime(s)

    def _to_json(self, data: Any) -> Optional[str]:
        """Convert to JSON string."""
        if data is None:
            return None
        return json.dumps(data)

    def _from_json(self, s: Optional[str]) -> Any:
        """Parse JSON string."""
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    def _safe_get(self, row: sqlite3.Row, key: str, default: Any = None) -> Any:
        """Safely get a value from a row, returning default if column missing.

        Useful for backwards compatibility when schema is migrated.
        """
        try:
            value = row[key]
            return value if value is not None else default
        except (IndexError, KeyError):
            return default

    def _record_to_dict(self, record: Any) -> Dict[str, Any]:
        """Serialize a record to a dictionary for sync queue data.

        Handles datetime conversion and nested objects.
        """
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

    def _build_access_filter(
        self, requesting_entity: Optional[str] = None
    ) -> tuple[str, List[Any]]:
        """Build SQL filter for privacy access control.

        Args:
            requesting_entity: Entity requesting access. None means self-access (see everything).

        Returns:
            Tuple of (where_clause, params) for SQL query.

        Logic:
            - If requesting_entity is None → no filter (self-access, see everything)
            - If requesting_entity is set → filter records where:
              - access_grants IS NULL (private to self only), OR
              - access_grants = '[]' (private to self only), OR
              - access_grants contains requesting_entity
        """
        if requesting_entity is None:
            # Self-access: see everything
            return ("", [])

        # External access: only show records where requesting_entity is in access_grants
        # NULL or empty access_grants = private to self only
        # Escape LIKE metacharacters to prevent pattern injection (%, _, \)
        escaped_entity = escape_like_pattern(requesting_entity)
        where_clause = """
            AND (access_grants IS NOT NULL
                 AND access_grants != '[]'
                 AND access_grants LIKE ? ESCAPE '\\')
        """
        params = [f'%"{escaped_entity}"%']

        return (where_clause, params)

    def _queue_sync(
        self,
        conn: sqlite3.Connection,
        table: str,
        record_id: str,
        operation: str,
        payload: Optional[str] = None,
        data: Optional[str] = None,
    ):
        """Queue a change for sync.

        Deduplicates by (table, record_id) - only keeps latest operation.
        Uses UPSERT (INSERT ... ON CONFLICT) for atomic operation to prevent
        race conditions between concurrent writes.

        Stores data in both `data` and `payload` columns for consistency.
        The `payload` column is the canonical source; `data` is kept for
        backward compatibility.
        """
        if table in LOCAL_ONLY_SYNC_TABLES:
            return

        now = self._now()

        # Normalize: use whichever is provided, store in both columns
        effective_payload = payload or data
        effective_data = data or payload

        # Use atomic UPSERT to prevent race condition between SELECT and UPDATE/INSERT
        # This requires a unique index on (table_name, record_id) where synced = 0
        # We use INSERT ... ON CONFLICT DO UPDATE for atomicity
        conn.execute(
            """INSERT INTO sync_queue
               (table_name, record_id, operation, data, local_updated_at, synced, payload, queued_at)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?)
               ON CONFLICT(table_name, record_id) WHERE synced = 0
               DO UPDATE SET
                   operation = excluded.operation,
                   data = excluded.data,
                   local_updated_at = excluded.local_updated_at,
                   payload = excluded.payload,
                   queued_at = excluded.queued_at""",
            (table, record_id, operation, effective_data, now, effective_payload, now),
        )

    def _content_hash(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _save_embedding(self, conn: sqlite3.Connection, table: str, record_id: str, content: str):
        """Save embedding for a record."""
        if not self._has_vec:
            return

        content_hash = self._content_hash(content)
        # Include stack_id in vec_id for isolation (security: prevents cross-agent timing leaks)
        vec_id = f"{self.stack_id}:{table}:{record_id}"

        # Check if embedding exists and is current
        existing = conn.execute(
            "SELECT content_hash FROM embedding_meta WHERE id = ?", (vec_id,)
        ).fetchone()

        if existing and existing["content_hash"] == content_hash:
            return  # Already up to date

        embedding = self._embed_text(content, context=f"vector-save:{table}:{record_id}")
        if not embedding:
            return

        # Re-check after _embed_text (which may have switched to fallback)
        is_fallback = (
            self._embedder is self._embedder_fallback and self._embedder_fallback is not None
        )
        provider_name = type(self._embedder).__name__

        packed = pack_embedding(embedding)

        try:
            # Upsert into vector table
            conn.execute(
                "INSERT OR REPLACE INTO vec_embeddings (id, embedding) VALUES (?, ?)",
                (vec_id, packed),
            )

            # Update metadata with provider observability
            conn.execute(
                """INSERT OR REPLACE INTO embedding_meta
                   (id, table_name, record_id, content_hash, created_at,
                    embedding_provider, fallback_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    vec_id,
                    table,
                    record_id,
                    content_hash,
                    self._now(),
                    provider_name,
                    1 if is_fallback else 0,
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to save embedding for {vec_id}: {e}", exc_info=True)
            # Ensure stale embedding state never persists if writes partially fail.
            # This prevents old vectors/metadata from being used after transient
            # provider or storage failures.
            try:
                conn.execute("DELETE FROM vec_embeddings WHERE id = ?", (vec_id,))
                conn.execute("DELETE FROM embedding_meta WHERE id = ?", (vec_id,))
                logger.warning(
                    "Cleaned stale embedding cache entry for %s after save failure",
                    vec_id,
                    exc_info=True,
                )
            except sqlite3.OperationalError as cleanup_error:
                logger.error(
                    "Failed to clean stale embedding cache entry for %s: %s",
                    vec_id,
                    cleanup_error,
                    exc_info=True,
                )
                raise e from cleanup_error
            except Exception as cleanup_error:
                logger.error(
                    "Unexpected error cleaning stale embedding cache entry for %s: %s",
                    vec_id,
                    cleanup_error,
                    exc_info=True,
                )
                raise e from cleanup_error

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get embedding provider statistics for observability.

        Returns a dict with:
        - total: total embeddings stored
        - by_provider: counts per embedding_provider
        - fallback_count: number of embeddings generated via fallback
        - current_provider: name of the currently active embedding provider
        - is_degraded: whether the system is currently using the fallback
        """
        stats: dict[str, Any] = {
            "total": 0,
            "by_provider": {},
            "fallback_count": 0,
            "current_provider": type(self._embedder).__name__,
            "is_degraded": (
                self._embedder is self._embedder_fallback and self._embedder_fallback is not None
            ),
        }

        if not self._has_vec:
            return stats

        with self._connect() as conn:
            # Total count
            row = conn.execute("SELECT COUNT(*) FROM embedding_meta").fetchone()
            stats["total"] = row[0] if row else 0

            # Per-provider breakdown
            rows = conn.execute(
                "SELECT embedding_provider, COUNT(*) as cnt FROM embedding_meta "
                "GROUP BY embedding_provider"
            ).fetchall()
            for r in rows:
                provider = r["embedding_provider"] or "unknown"
                stats["by_provider"][provider] = r["cnt"]

            # Fallback count
            row = conn.execute(
                "SELECT COUNT(*) FROM embedding_meta WHERE fallback_used = 1"
            ).fetchone()
            stats["fallback_count"] = row[0] if row else 0

        return stats

    def _get_searchable_content(self, record_type: str, record: Any) -> str:
        """Get searchable text content from a record."""
        if record_type == "episode":
            parts = [record.objective, record.outcome]
            if record.lessons:
                parts.extend(record.lessons)
            return " ".join(filter(None, parts))
        elif record_type == "note":
            return record.content
        elif record_type == "belief":
            return record.statement
        elif record_type == "value":
            return f"{record.name}: {record.statement}"
        elif record_type == "goal":
            return f"{record.title} {record.description or ''}"
        return ""

    # === Episodes ===

    def save_episode(self, episode: Episode) -> str:
        """Save an episode. Delegates to episodes_crud."""
        return _episodes_crud.save_episode(
            self._connect,
            self.stack_id,
            episode,
            self._now,
            self._parse_datetime,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._get_searchable_content,
            lineage_checker=lambda t, i, d: check_derived_from_cycle(self, t, i, d),
        )

    def update_episode_atomic(
        self, episode: Episode, expected_version: Optional[int] = None
    ) -> bool:
        """Update an episode with optimistic concurrency control. Delegates to episodes_crud."""
        return _episodes_crud.update_episode_atomic(
            self._connect,
            self.stack_id,
            episode,
            self._now,
            self._to_json,
            self._from_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._get_searchable_content,
            expected_version=expected_version,
        )

    def save_episodes_batch(self, episodes: List[Episode]) -> List[str]:
        """Save multiple episodes in a single transaction. Delegates to episodes_crud."""
        return _episodes_crud.save_episodes_batch(
            self._connect,
            self.stack_id,
            episodes,
            self._now,
            self._parse_datetime,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._get_searchable_content,
        )

    def get_episodes(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        requesting_entity: Optional[str] = None,
        processed: Optional[bool] = None,
    ) -> List[Episode]:
        """Get episodes. Delegates to episodes_crud."""
        return _episodes_crud.get_episodes(
            self._connect,
            self.stack_id,
            self._build_access_filter,
            limit=limit,
            since=since,
            tags=tags,
            requesting_entity=requesting_entity,
            processed=processed,
        )

    def memory_exists(self, memory_type: str, memory_id: str) -> bool:
        """Check if a memory record exists in the stack. Delegates to meta_memory_ops."""
        with self._connect() as conn:
            return _meta_memory_ops.memory_exists(conn, self.stack_id, memory_type, memory_id)

    def get_episode(
        self, episode_id: str, requesting_entity: Optional[str] = None
    ) -> Optional[Episode]:
        """Get a specific episode. Delegates to episodes_crud."""
        return _episodes_crud.get_episode(
            self._connect,
            self.stack_id,
            episode_id,
            self._build_access_filter,
            requesting_entity=requesting_entity,
        )

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert row to Episode. Delegates to memory_crud._row_to_episode()."""
        return _mc_row_to_episode(row)

    def get_episodes_by_source_entity(self, source_entity: str, limit: int = 500) -> List[Episode]:
        """Get episodes associated with a source entity. Delegates to episodes_crud."""
        return _episodes_crud.get_episodes_by_source_entity(
            self._connect,
            self.stack_id,
            source_entity,
            limit,
        )

    def update_episode_emotion(
        self, episode_id: str, valence: float, arousal: float, tags: Optional[List[str]] = None
    ) -> bool:
        """Update emotional associations for an episode. Delegates to episodes_crud."""
        return _episodes_crud.update_episode_emotion(
            self._connect,
            self.stack_id,
            episode_id,
            valence,
            arousal,
            self._now,
            self._to_json,
            self._queue_sync,
            tags=tags,
        )

    def search_by_emotion(
        self,
        valence_range: Optional[tuple] = None,
        arousal_range: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """Find episodes matching emotional criteria. Delegates to episodes_crud."""
        return _episodes_crud.search_by_emotion(
            self._connect,
            self.stack_id,
            valence_range=valence_range,
            arousal_range=arousal_range,
            tags=tags,
            limit=limit,
        )

    def get_emotional_episodes(self, days: int = 7, limit: int = 100) -> List[Episode]:
        """Get episodes with emotional data. Delegates to episodes_crud."""
        return _episodes_crud.get_emotional_episodes(
            self._connect,
            self.stack_id,
            days=days,
            limit=limit,
        )

    # === Beliefs ===

    def save_belief(self, belief: Belief) -> str:
        """Save a belief. Delegates to beliefs_crud."""
        return _beliefs_crud.save_belief(
            self._connect,
            self.stack_id,
            belief,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._sync_beliefs_to_file,
            lineage_checker=lambda t, i, d: check_derived_from_cycle(self, t, i, d),
        )

    def update_belief_atomic(self, belief: Belief, expected_version: Optional[int] = None) -> bool:
        """Update a belief with optimistic concurrency control. Delegates to beliefs_crud."""
        return _beliefs_crud.update_belief_atomic(
            self._connect,
            self.stack_id,
            belief,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._sync_beliefs_to_file,
            expected_version=expected_version,
        )

    def save_beliefs_batch(self, beliefs: List[Belief]) -> List[str]:
        """Save multiple beliefs in a single transaction. Delegates to beliefs_crud."""
        return _beliefs_crud.save_beliefs_batch(
            self._connect,
            self.stack_id,
            beliefs,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._sync_beliefs_to_file,
        )

    def _sync_beliefs_to_file(self) -> None:
        """Write all active beliefs to flat file."""
        sync_beliefs_to_file(
            self._beliefs_file, self.get_beliefs(limit=500, include_inactive=False), self._now()
        )

    def get_beliefs(
        self,
        limit: int = 100,
        include_inactive: bool = False,
        requesting_entity: Optional[str] = None,
        processed: Optional[bool] = None,
    ) -> List[Belief]:
        """Get beliefs. Delegates to beliefs_crud."""
        return _beliefs_crud.get_beliefs(
            self._connect,
            self.stack_id,
            self._build_access_filter,
            limit=limit,
            include_inactive=include_inactive,
            requesting_entity=requesting_entity,
            processed=processed,
        )

    def find_belief(self, statement: str) -> Optional[Belief]:
        """Find a belief by statement. Delegates to beliefs_crud."""
        return _beliefs_crud.find_belief(self._connect, self.stack_id, statement)

    def get_belief(
        self, belief_id: str, requesting_entity: Optional[str] = None
    ) -> Optional[Belief]:
        """Get a specific belief by ID. Delegates to beliefs_crud."""
        return _beliefs_crud.get_belief(
            self._connect,
            self.stack_id,
            belief_id,
            self._build_access_filter,
            requesting_entity=requesting_entity,
        )

    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        """Convert row to Belief. Delegates to memory_crud._row_to_belief()."""
        return _mc_row_to_belief(row)

    # === Values ===

    def save_value(self, value: Value) -> str:
        """Save a value. Delegates to values_crud."""
        return _values_crud.save_value(
            self._connect,
            self.stack_id,
            value,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._sync_values_to_file,
            lineage_checker=lambda t, i, d: check_derived_from_cycle(self, t, i, d),
        )

    def _sync_values_to_file(self) -> None:
        """Write all values to flat file."""
        sync_values_to_file(self._values_file, self.get_values(limit=100), self._now())

    def get_values(self, limit: int = 100, requesting_entity: Optional[str] = None) -> List[Value]:
        """Get values ordered by priority. Delegates to values_crud."""
        return _values_crud.get_values(
            self._connect,
            self.stack_id,
            self._build_access_filter,
            limit=limit,
            requesting_entity=requesting_entity,
        )

    def _row_to_value(self, row: sqlite3.Row) -> Value:
        """Convert row to Value. Delegates to memory_crud._row_to_value()."""
        return _mc_row_to_value(row)

    # === Goals ===

    def save_goal(self, goal: Goal) -> str:
        """Save a goal. Delegates to goals_crud."""
        return _goals_crud.save_goal(
            self._connect,
            self.stack_id,
            goal,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._sync_goals_to_file,
            lineage_checker=lambda t, i, d: check_derived_from_cycle(self, t, i, d),
        )

    def update_goal_atomic(self, goal: Goal, expected_version: Optional[int] = None) -> bool:
        """Update a goal with optimistic concurrency control. Delegates to goals_crud."""
        return _goals_crud.update_goal_atomic(
            self._connect,
            self.stack_id,
            goal,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            self._sync_goals_to_file,
            expected_version=expected_version,
        )

    def _sync_goals_to_file(self) -> None:
        """Write all active goals to flat file."""
        sync_goals_to_file(self._goals_file, self.get_goals(status=None, limit=100), self._now())

    def get_goals(
        self,
        status: Optional[str] = "active",
        limit: int = 100,
        requesting_entity: Optional[str] = None,
    ) -> List[Goal]:
        """Get goals. Delegates to goals_crud."""
        return _goals_crud.get_goals(
            self._connect,
            self.stack_id,
            self._build_access_filter,
            status=status,
            limit=limit,
            requesting_entity=requesting_entity,
        )

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        """Convert row to Goal. Delegates to memory_crud._row_to_goal()."""
        return _mc_row_to_goal(row)

    # === Notes ===

    def save_note(self, note: Note) -> str:
        """Save a note. Delegates to notes_crud."""
        return _notes_crud.save_note(
            self._connect,
            self.stack_id,
            note,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
            lineage_checker=lambda t, i, d: check_derived_from_cycle(self, t, i, d),
        )

    def save_notes_batch(self, notes: List[Note]) -> List[str]:
        """Save multiple notes in a single transaction. Delegates to notes_crud."""
        return _notes_crud.save_notes_batch(
            self._connect,
            self.stack_id,
            notes,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
        )

    def get_notes(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        note_type: Optional[str] = None,
        requesting_entity: Optional[str] = None,
    ) -> List[Note]:
        """Get notes. Delegates to notes_crud."""
        return _notes_crud.get_notes(
            self._connect,
            self.stack_id,
            self._build_access_filter,
            limit=limit,
            since=since,
            note_type=note_type,
            requesting_entity=requesting_entity,
        )

    def _row_to_note(self, row: sqlite3.Row) -> Note:
        """Convert row to Note. Delegates to memory_crud._row_to_note()."""
        return _mc_row_to_note(row)

    # === Drives ===

    def save_drive(self, drive: Drive) -> str:
        """Save or update a drive. Delegates to drives_crud."""
        return _drives_crud.save_drive(
            self._connect,
            self.stack_id,
            drive,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            lineage_checker=lambda t, i, d: check_derived_from_cycle(self, t, i, d),
        )

    def update_drive_atomic(self, drive: Drive, expected_version: Optional[int] = None) -> bool:
        """Update a drive with optimistic concurrency control. Delegates to drives_crud."""
        return _drives_crud.update_drive_atomic(
            self._connect,
            self.stack_id,
            drive,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            expected_version=expected_version,
        )

    def get_drives(self, requesting_entity: Optional[str] = None) -> List[Drive]:
        """Get all drives. Delegates to drives_crud."""
        return _drives_crud.get_drives(
            self._connect,
            self.stack_id,
            self._build_access_filter,
            requesting_entity=requesting_entity,
        )

    def get_drive(self, drive_type: str) -> Optional[Drive]:
        """Get a specific drive. Delegates to drives_crud."""
        return _drives_crud.get_drive(self._connect, self.stack_id, drive_type)

    def _row_to_drive(self, row: sqlite3.Row) -> Drive:
        """Convert row to Drive. Delegates to memory_crud._row_to_drive()."""
        return _mc_row_to_drive(row)

    # === Relationships ===

    def save_relationship(self, relationship: Relationship) -> str:
        """Save or update a relationship. Delegates to relationships_crud."""
        return _relationships_crud.save_relationship(
            self._connect,
            self.stack_id,
            relationship,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._sync_relationships_to_file,
            lineage_checker=lambda t, i, d: check_derived_from_cycle(self, t, i, d),
        )

    def update_relationship_atomic(
        self, relationship: Relationship, expected_version: Optional[int] = None
    ) -> bool:
        """Update a relationship with optimistic concurrency control.

        Args:
            relationship: The relationship with updated fields
            expected_version: The version we expect the record to have.
                             If None, uses relationship.version.

        Returns:
            True if update succeeded

        Raises:
            VersionConflictError: If the record's version doesn't match expected
        """
        if expected_version is None:
            expected_version = relationship.version

        now = self._now()

        with self._connect() as conn:
            # Fetch full row for version check and history tracking
            current = conn.execute(
                "SELECT * FROM relationships WHERE id = ? AND stack_id = ?",
                (relationship.id, self.stack_id),
            ).fetchone()

            if not current:
                return False

            current_version = current["version"]
            if current_version != expected_version:
                raise VersionConflictError(
                    "relationships",
                    relationship.id,
                    expected_version,
                    current_version,
                )

            # Track relationship changes (history entries)
            self._log_relationship_changes(conn, current, relationship, now)

            # Atomic update with version increment
            cursor = conn.execute(
                """
                UPDATE relationships SET
                    entity_type = ?,
                    relationship_type = ?,
                    notes = ?,
                    sentiment = ?,
                    interaction_count = ?,
                    last_interaction = ?,
                    confidence = ?,
                    source_type = ?,
                    source_entity = COALESCE(?, source_entity),
                    source_episodes = ?,
                    derived_from = ?,
                    last_verified = ?,
                    verification_count = ?,
                    confidence_history = ?,
                    context = ?,
                    context_tags = ?,
                    local_updated_at = ?,
                    version = version + 1
                WHERE id = ? AND stack_id = ? AND version = ?
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
                    self._to_json(relationship.source_episodes),
                    self._to_json(relationship.derived_from),
                    (
                        relationship.last_verified.isoformat()
                        if relationship.last_verified
                        else None
                    ),
                    relationship.verification_count,
                    self._to_json(relationship.confidence_history),
                    relationship.context,
                    self._to_json(relationship.context_tags),
                    now,
                    relationship.id,
                    self.stack_id,
                    expected_version,
                ),
            )

            if cursor.rowcount == 0:
                conn.rollback()
                new_current = conn.execute(
                    "SELECT version FROM relationships WHERE id = ? AND stack_id = ?",
                    (relationship.id, self.stack_id),
                ).fetchone()
                actual = new_current["version"] if new_current else -1
                raise VersionConflictError(
                    "relationships",
                    relationship.id,
                    expected_version,
                    actual,
                )

            # Queue for sync
            relationship.version = expected_version + 1
            relationship_data = self._to_json(self._record_to_dict(relationship))
            self._queue_sync(
                conn, "relationships", relationship.id, "upsert", data=relationship_data
            )

            conn.commit()

        # Sync to flat file
        self._sync_relationships_to_file()

        return True

    def _sync_relationships_to_file(self) -> None:
        """Write all relationships to flat file."""
        sync_relationships_to_file(self._relationships_file, self.get_relationships(), self._now())

    def get_relationships(
        self, entity_type: Optional[str] = None, requesting_entity: Optional[str] = None
    ) -> List[Relationship]:
        """Get relationships. Delegates to relationships_crud."""
        return _relationships_crud.get_relationships(
            self._connect,
            self.stack_id,
            self._build_access_filter,
            entity_type=entity_type,
            requesting_entity=requesting_entity,
        )

    def get_relationship(self, entity_name: str) -> Optional[Relationship]:
        """Get a specific relationship. Delegates to relationships_crud."""
        return _relationships_crud.get_relationship(self._connect, self.stack_id, entity_name)

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        """Convert row to Relationship. Delegates to memory_crud._row_to_relationship()."""
        return _mc_row_to_relationship(row)

    # === Epochs (KEP v3 temporal eras) ===

    def save_epoch(self, epoch: Epoch) -> str:
        """Save an epoch. Delegates to epoch_crud."""
        return _epoch_crud.save_epoch(self._connect, self.stack_id, epoch, self._now, self._to_json)

    def get_epoch(self, epoch_id: str) -> Optional[Epoch]:
        """Get a specific epoch by ID. Delegates to epoch_crud."""
        return _epoch_crud.get_epoch(self._connect, self.stack_id, epoch_id)

    def get_epochs(self, limit: int = 100) -> List[Epoch]:
        """Get all epochs. Delegates to epoch_crud."""
        return _epoch_crud.get_epochs(self._connect, self.stack_id, limit=limit)

    def get_current_epoch(self) -> Optional[Epoch]:
        """Get the currently active epoch. Delegates to epoch_crud."""
        return _epoch_crud.get_current_epoch(self._connect, self.stack_id)

    def close_epoch(self, epoch_id: str, summary: Optional[str] = None) -> bool:
        """Close an epoch. Delegates to epoch_crud."""
        return _epoch_crud.close_epoch(
            self._connect, self.stack_id, epoch_id, self._now, summary=summary
        )

    def _row_to_epoch(self, row: sqlite3.Row) -> Epoch:
        """Convert row to Epoch. Delegates to memory_crud._row_to_epoch()."""
        return _mc_row_to_epoch(row)

    # === Summaries (Fractal Summarization) ===

    def save_summary(self, summary: Summary) -> str:
        """Save a summary. Delegates to narrative_summary_crud."""
        return _narrative_summary_crud.save_summary(
            self._connect, self.stack_id, summary, self._now, self._to_json
        )

    def get_summary(self, summary_id: str) -> Optional[Summary]:
        """Get a specific summary by ID. Delegates to narrative_summary_crud."""
        return _narrative_summary_crud.get_summary(self._connect, self.stack_id, summary_id)

    def list_summaries(self, stack_id: str, scope: Optional[str] = None) -> List[Summary]:
        """Get summaries. Delegates to narrative_summary_crud."""
        return _narrative_summary_crud.list_summaries(self._connect, self.stack_id, scope=scope)

    def _row_to_summary(self, row: sqlite3.Row) -> Summary:
        """Convert row to Summary. Delegates to memory_crud._row_to_summary()."""
        return _mc_row_to_summary(row)

    # === Self-Narratives (KEP v3) ===

    def save_self_narrative(self, narrative: SelfNarrative) -> str:
        """Save a self-narrative. Delegates to narrative_summary_crud."""
        return _narrative_summary_crud.save_self_narrative(
            self._connect, self.stack_id, narrative, self._now, self._to_json
        )

    def get_self_narrative(self, narrative_id: str) -> Optional[SelfNarrative]:
        """Get a specific self-narrative. Delegates to narrative_summary_crud."""
        return _narrative_summary_crud.get_self_narrative(
            self._connect, self.stack_id, narrative_id
        )

    def list_self_narratives(
        self,
        stack_id: str,
        narrative_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[SelfNarrative]:
        """Get self-narratives. Delegates to narrative_summary_crud."""
        return _narrative_summary_crud.list_self_narratives(
            self._connect, self.stack_id, narrative_type=narrative_type, active_only=active_only
        )

    def deactivate_self_narratives(self, stack_id: str, narrative_type: str) -> int:
        """Deactivate all active narratives of a given type. Delegates to narrative_summary_crud."""
        return _narrative_summary_crud.deactivate_self_narratives(
            self._connect, self.stack_id, narrative_type, self._now
        )

    def _row_to_self_narrative(self, row: sqlite3.Row) -> SelfNarrative:
        """Convert row to SelfNarrative. Delegates to memory_crud._row_to_self_narrative()."""
        return _mc_row_to_self_narrative(row)

    # === Trust Assessments (KEP v3) ===

    def save_trust_assessment(self, assessment: TrustAssessment) -> str:
        """Save or update a trust assessment. Delegates to trust_crud."""
        return _trust_crud.save_trust_assessment(
            self._connect, self.stack_id, assessment, self._now
        )

    def get_trust_assessment(self, entity: str) -> Optional[TrustAssessment]:
        """Get a trust assessment. Delegates to trust_crud."""
        return _trust_crud.get_trust_assessment(self._connect, self.stack_id, entity)

    def get_trust_assessments(self) -> List[TrustAssessment]:
        """Get all trust assessments. Delegates to trust_crud."""
        return _trust_crud.get_trust_assessments(self._connect, self.stack_id)

    def delete_trust_assessment(self, entity: str) -> bool:
        """Delete a trust assessment. Delegates to trust_crud."""
        return _trust_crud.delete_trust_assessment(self._connect, self.stack_id, entity, self._now)

    # === Diagnostic Sessions & Reports ===

    def save_diagnostic_session(self, session: DiagnosticSession) -> str:
        """Save a diagnostic session. Delegates to diagnostic_crud."""
        return _diagnostic_crud.save_diagnostic_session(
            self._connect, self.stack_id, session, self._now
        )

    def get_diagnostic_session(self, session_id: str) -> Optional[DiagnosticSession]:
        """Get a specific diagnostic session. Delegates to diagnostic_crud."""
        return _diagnostic_crud.get_diagnostic_session(self._connect, self.stack_id, session_id)

    def get_diagnostic_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[DiagnosticSession]:
        """Get diagnostic sessions. Delegates to diagnostic_crud."""
        return _diagnostic_crud.get_diagnostic_sessions(
            self._connect, self.stack_id, status=status, limit=limit
        )

    def complete_diagnostic_session(self, session_id: str) -> bool:
        """Mark a diagnostic session as completed. Delegates to diagnostic_crud."""
        return _diagnostic_crud.complete_diagnostic_session(
            self._connect, self.stack_id, session_id, self._now
        )

    def save_diagnostic_report(self, report: DiagnosticReport) -> str:
        """Save a diagnostic report. Delegates to diagnostic_crud."""
        return _diagnostic_crud.save_diagnostic_report(
            self._connect, self.stack_id, report, self._now
        )

    def get_diagnostic_report(self, report_id: str) -> Optional[DiagnosticReport]:
        """Get a specific diagnostic report. Delegates to diagnostic_crud."""
        return _diagnostic_crud.get_diagnostic_report(self._connect, self.stack_id, report_id)

    def get_diagnostic_reports(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[DiagnosticReport]:
        """Get diagnostic reports. Delegates to diagnostic_crud."""
        return _diagnostic_crud.get_diagnostic_reports(
            self._connect, self.stack_id, session_id=session_id, limit=limit
        )

    def _row_to_diagnostic_session(self, row: sqlite3.Row) -> DiagnosticSession:
        """Convert row to DiagnosticSession. Delegates to memory_crud._row_to_diagnostic_session()."""
        return _mc_row_to_diagnostic_session(row)

    def _row_to_diagnostic_report(self, row: sqlite3.Row) -> DiagnosticReport:
        """Convert row to DiagnosticReport. Delegates to memory_crud._row_to_diagnostic_report()."""
        return _mc_row_to_diagnostic_report(row)

    # === Relationship History & Entity Models ===

    def _log_relationship_changes(
        self,
        conn: Any,
        existing_row: sqlite3.Row,
        new_rel: Relationship,
        now: str,
    ) -> None:
        """Detect changes between existing and new relationship. Delegates to relationships_crud."""
        _relationships_crud._log_relationship_changes(
            conn, self.stack_id, existing_row, new_rel, now
        )

    # === Relationship History ===

    def save_relationship_history(self, entry: RelationshipHistoryEntry) -> str:
        """Save a relationship history entry. Delegates to relationships_crud."""
        return _relationships_crud.save_relationship_history(
            self._connect, self.stack_id, entry, self._now
        )

    def get_relationship_history(
        self,
        entity_name: str,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[RelationshipHistoryEntry]:
        """Get history entries for a relationship. Delegates to relationships_crud."""
        return _relationships_crud.get_relationship_history(
            self._connect, self.stack_id, entity_name, event_type=event_type, limit=limit
        )

    def _row_to_relationship_history(self, row: sqlite3.Row) -> RelationshipHistoryEntry:
        """Convert row to RelationshipHistoryEntry. Delegates to memory_crud._row_to_relationship_history()."""
        return _mc_row_to_relationship_history(row)

    # === Entity Models ===

    def save_entity_model(self, model: EntityModel) -> str:
        """Save an entity model. Delegates to relationships_crud."""
        return _relationships_crud.save_entity_model(
            self._connect, self.stack_id, model, self._now, self._to_json
        )

    def get_entity_models(
        self,
        entity_name: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[EntityModel]:
        """Get entity models, optionally filtered. Delegates to relationships_crud."""
        return _relationships_crud.get_entity_models(
            self._connect,
            self.stack_id,
            entity_name=entity_name,
            model_type=model_type,
            limit=limit,
        )

    def get_entity_model(self, model_id: str) -> Optional[EntityModel]:
        """Get a specific entity model by ID. Delegates to relationships_crud."""
        return _relationships_crud.get_entity_model(self._connect, self.stack_id, model_id)

    def _row_to_entity_model(self, row: sqlite3.Row) -> EntityModel:
        """Convert row to EntityModel. Delegates to memory_crud._row_to_entity_model()."""
        return _mc_row_to_entity_model(row)

    # === Playbooks (Procedural Memory) ===

    def save_playbook(self, playbook: Playbook) -> str:
        """Save a playbook. Delegates to playbooks_crud."""
        return _playbooks_crud.save_playbook(
            self._connect,
            self.stack_id,
            playbook,
            self._now,
            self._to_json,
            self._record_to_dict,
            self._queue_sync,
            self._save_embedding,
        )

    def get_playbook(self, playbook_id: str) -> Optional[Playbook]:
        """Get a specific playbook by ID. Delegates to playbooks_crud."""
        return _playbooks_crud.get_playbook(self._connect, self.stack_id, playbook_id)

    def list_playbooks(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        requesting_entity: Optional[str] = None,
    ) -> List[Playbook]:
        """Get playbooks. Delegates to playbooks_crud."""
        return _playbooks_crud.list_playbooks(
            self._connect,
            self.stack_id,
            self._build_access_filter,
            tags=tags,
            limit=limit,
            requesting_entity=requesting_entity,
        )

    def search_playbooks(self, query: str, limit: int = 10) -> List[Playbook]:
        """Search playbooks. Delegates to playbooks_crud."""
        return _playbooks_crud.search_playbooks(
            self._connect,
            self.stack_id,
            query,
            limit=limit,
            has_vec=self._has_vec,
            embed_text=self._embed_text,
            pack_embedding_fn=pack_embedding,
            get_playbook_fn=self.get_playbook,
            tokenize_query=self._tokenize_query,
            build_token_filter=self._build_token_filter,
            token_match_score=self._token_match_score,
            escape_like_pattern=escape_like_pattern,
        )

    def update_playbook_usage(self, playbook_id: str, success: bool) -> bool:
        """Update playbook usage statistics. Delegates to playbooks_crud."""
        return _playbooks_crud.update_playbook_usage(
            self._connect,
            self.stack_id,
            playbook_id,
            success,
            self._now,
            self._queue_sync,
            self.get_playbook,
        )

    def _row_to_playbook(self, row: sqlite3.Row) -> Playbook:
        """Convert row to Playbook. Delegates to memory_crud._row_to_playbook()."""
        return _mc_row_to_playbook(row)

    # === Boot Config ===

    def boot_set(self, key: str, value: str) -> None:
        """Set a boot config value. Creates or updates."""
        if not key or not isinstance(key, str):
            raise ValueError("Boot config key must be a non-empty string")
        if not isinstance(value, str):
            raise ValueError("Boot config value must be a string")
        # Strip whitespace from key, preserve value
        key = key.strip()
        if not key:
            raise ValueError("Boot config key must be a non-empty string")

        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO boot_config (id, stack_id, key, value, created_at, updated_at)
                VALUES (lower(hex(randomblob(4))), ?, ?, ?, ?, ?)
                ON CONFLICT(stack_id, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (self.stack_id, key, value, now, now),
            )

    def boot_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a boot config value. Returns default if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM boot_config WHERE stack_id = ? AND key = ?",
                (self.stack_id, key),
            ).fetchone()
        return row["value"] if row else default

    def boot_list(self) -> Dict[str, str]:
        """List all boot config values as a dict."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM boot_config WHERE stack_id = ? ORDER BY key",
                (self.stack_id,),
            ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def boot_delete(self, key: str) -> bool:
        """Delete a boot config value. Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM boot_config WHERE stack_id = ? AND key = ?",
                (self.stack_id, key),
            )
        return cursor.rowcount > 0

    def boot_clear(self) -> int:
        """Clear all boot config for this agent. Returns count deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM boot_config WHERE stack_id = ?",
                (self.stack_id,),
            )
        return cursor.rowcount

    # === Raw Entries ===
    # Delegated to storage/raw_entries.py

    def save_raw(self, blob: str, source: str = "unknown") -> str:
        """Save a raw entry. Delegates to raw_entries.save_raw()."""
        with self._connect() as conn:
            raw_id = _save_raw(
                conn=conn,
                stack_id=self.stack_id,
                blob=blob,
                source=source,
                raw_dir=self._raw_dir,
                queue_sync_fn=self._queue_sync,
                save_embedding_fn=self._save_embedding,
                should_sync_raw_fn=self._should_sync_raw,
            )
        self.log_audit(
            "raw",
            raw_id,
            "raw.ingested",
            actor="system",
            details={"source": source, "content_length": len(blob)},
        )
        return raw_id

    def _should_sync_raw(self) -> bool:
        """Check if raw sync is enabled. Delegates to raw_entries.should_sync_raw()."""
        return should_sync_raw()

    def _update_raw_fts(self, conn, raw_id, blob):
        """Update FTS5 index. Delegates to raw_entries.update_raw_fts()."""
        update_raw_fts(conn, raw_id, blob)

    def _append_raw_to_file(self, raw_id, content, timestamp, source, tags=None):
        """Append to flat file. Delegates to raw_entries.append_raw_to_file()."""
        append_raw_to_file(self._raw_dir, raw_id, content, timestamp, source, tags)

    def get_raw_dir(self):
        """Get raw flat files directory path."""
        return self._raw_dir

    def get_raw_files(self):
        """Get list of raw flat files. Delegates to raw_entries.get_raw_files()."""
        return _get_raw_files(self._raw_dir)

    def sync_raw_from_files(self):
        """Sync raw entries from flat files. Delegates to raw_entries.sync_raw_from_files()."""
        with self._connect() as conn:
            return _sync_raw_from_files(
                conn=conn,
                stack_id=self.stack_id,
                raw_dir=self._raw_dir,
                save_embedding_fn=self._save_embedding,
            )

    def _import_raw_entry(self, entry, existing_ids, result):
        """Import a raw entry. Delegates to raw_entries.import_raw_entry()."""
        from .raw_entries import import_raw_entry

        with self._connect() as conn:
            import_raw_entry(
                conn=conn,
                stack_id=self.stack_id,
                entry=entry,
                existing_ids=existing_ids,
                result=result,
                save_embedding_fn=self._save_embedding,
            )

    def get_raw(self, raw_id):
        """Get a raw entry by ID. Delegates to raw_entries.get_raw()."""
        with self._connect() as conn:
            return _get_raw(conn, self.stack_id, raw_id)

    def list_raw(self, processed=None, limit=100, offset=0):
        """List raw entries. Delegates to raw_entries.list_raw()."""
        with self._connect() as conn:
            return _list_raw(conn, self.stack_id, processed, limit, offset)

    def search_raw_fts(self, query, limit=50):
        """Search raw entries via FTS5. Delegates to raw_entries.search_raw_fts()."""
        with self._connect() as conn:
            return _search_raw_fts(conn, self.stack_id, query, limit)

    def find_raw_by_prefix(self, prefix, limit=10):
        """Find raw entries by ID prefix. Delegates to raw_entries.find_by_id_prefix()."""
        with self._connect() as conn:
            return _find_by_id_prefix(conn, self.stack_id, prefix, limit)

    def _escape_like_pattern(self, pattern):
        """Escape LIKE pattern. Delegates to raw_entries.escape_like_pattern()."""
        return escape_like_pattern(pattern)

    def mark_raw_processed(self, raw_id, processed_into):
        """Mark raw entry as processed. Delegates to raw_entries.mark_raw_processed()."""
        with self._connect() as conn:
            return _mark_raw_processed(
                conn, self.stack_id, raw_id, processed_into, self._queue_sync
            )

    def mark_episode_processed(self, episode_id):
        """Mark episode as processed. Delegates to raw_entries.mark_processed()."""
        with self._connect() as conn:
            return _mark_processed(conn, self.stack_id, "episodes", episode_id, self._queue_sync)

    def mark_note_processed(self, note_id):
        """Mark note as processed. Delegates to raw_entries.mark_processed()."""
        with self._connect() as conn:
            return _mark_processed(conn, self.stack_id, "notes", note_id, self._queue_sync)

    def mark_belief_processed(self, belief_id):
        """Mark belief as processed. Delegates to raw_entries.mark_processed()."""
        with self._connect() as conn:
            return _mark_processed(conn, self.stack_id, "beliefs", belief_id, self._queue_sync)

    def get_processing_config(self):
        """Get processing config. Delegates to raw_entries.get_processing_config()."""
        with self._connect() as conn:
            return _get_processing_config(conn)

    def set_processing_config(
        self,
        layer_transition,
        *,
        enabled=None,
        model_id=None,
        quantity_threshold=None,
        valence_threshold=None,
        time_threshold_hours=None,
        batch_size=None,
        max_sessions_per_day=None,
    ):
        """Set processing config. Delegates to raw_entries.set_processing_config()."""
        with self._connect() as conn:
            return _set_processing_config(
                conn,
                layer_transition,
                enabled=enabled,
                model_id=model_id,
                quantity_threshold=quantity_threshold,
                valence_threshold=valence_threshold,
                time_threshold_hours=time_threshold_hours,
                batch_size=batch_size,
                max_sessions_per_day=max_sessions_per_day,
            )

    def get_stack_setting(self, key):
        """Get a stack setting. Delegates to raw_entries.get_stack_setting()."""
        with self._connect() as conn:
            return _get_stack_setting(conn, self.stack_id, key)

    def set_stack_setting(self, key, value):
        """Set a stack setting. Delegates to raw_entries.set_stack_setting()."""
        with self._connect() as conn:
            _set_stack_setting(conn, self.stack_id, key, value)

    def get_all_stack_settings(self):
        """Get all stack settings. Delegates to raw_entries.get_all_stack_settings()."""
        with self._connect() as conn:
            return _get_all_stack_settings(conn, self.stack_id)

    def delete_raw(self, raw_id):
        """Delete a raw entry. Delegates to raw_entries.delete_raw()."""
        with self._connect() as conn:
            return _delete_raw(conn, self.stack_id, raw_id, self._queue_sync)

    def _row_to_raw_entry(self, row):
        """Convert row to RawEntry. Delegates to raw_entries.row_to_raw_entry()."""
        return row_to_raw_entry(row)

    # === Memory Suggestions ===

    def save_suggestion(self, suggestion: MemorySuggestion) -> str:
        """Save a memory suggestion. Delegates to suggestions_crud."""
        return _suggestions_crud.save_suggestion(
            self._connect, self.stack_id, suggestion, self._now, self._to_json, self._queue_sync
        )

    def get_suggestion(self, suggestion_id: str) -> Optional[MemorySuggestion]:
        """Get a specific suggestion. Delegates to suggestions_crud."""
        return _suggestions_crud.get_suggestion(self._connect, self.stack_id, suggestion_id)

    def get_suggestions(
        self,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
        min_confidence: Optional[float] = None,
        max_age_hours: Optional[float] = None,
        source_raw_id: Optional[str] = None,
    ) -> List[MemorySuggestion]:
        """Get suggestions. Delegates to suggestions_crud."""
        return _suggestions_crud.get_suggestions(
            self._connect,
            self.stack_id,
            status=status,
            memory_type=memory_type,
            limit=limit,
            min_confidence=min_confidence,
            max_age_hours=max_age_hours,
            source_raw_id=source_raw_id,
        )

    def expire_suggestions(
        self,
        max_age_hours: float = 168.0,
    ) -> List[str]:
        """Auto-dismiss pending suggestions older than max_age_hours.

        Args:
            max_age_hours: Age threshold in hours (default: 168 = 7 days)

        Returns:
            List of expired suggestion IDs
        """
        now = self._now()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

        with self._connect() as conn:
            # Find pending suggestions older than cutoff
            rows = conn.execute(
                """SELECT id FROM memory_suggestions
                   WHERE stack_id = ? AND deleted = 0
                   AND status = 'pending' AND created_at < ?""",
                (self.stack_id, cutoff),
            ).fetchall()

            expired_ids = [row["id"] for row in rows]

            if expired_ids:
                placeholders = ",".join("?" for _ in expired_ids)
                conn.execute(
                    f"""UPDATE memory_suggestions SET
                        status = 'expired',
                        resolved_at = ?,
                        resolution_reason = 'auto-expired after {max_age_hours:.0f}h',
                        local_updated_at = ?,
                        version = version + 1
                    WHERE id IN ({placeholders}) AND stack_id = ? AND deleted = 0""",
                    (now, now, *expired_ids, self.stack_id),
                )
                for sid in expired_ids:
                    self._queue_sync(conn, "memory_suggestions", sid, "upsert")
                conn.commit()

        return expired_ids

    def update_suggestion_status(
        self,
        suggestion_id: str,
        status: str,
        resolution_reason: Optional[str] = None,
        promoted_to: Optional[str] = None,
    ) -> bool:
        """Update the status of a suggestion. Delegates to suggestions_crud."""
        return _suggestions_crud.update_suggestion_status(
            self._connect,
            self.stack_id,
            suggestion_id,
            status,
            self._now,
            self._queue_sync,
            resolution_reason=resolution_reason,
            promoted_to=promoted_to,
        )

    def delete_suggestion(self, suggestion_id: str) -> bool:
        """Delete a suggestion. Delegates to suggestions_crud."""
        return _suggestions_crud.delete_suggestion(
            self._connect, self.stack_id, suggestion_id, self._now, self._queue_sync
        )

    def _row_to_suggestion(self, row: sqlite3.Row) -> MemorySuggestion:
        """Convert row to MemorySuggestion. Delegates to memory_crud._row_to_suggestion()."""
        return _mc_row_to_suggestion(row)

    # === Search ===

    def search(
        self,
        query: str,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        prefer_cloud: bool = True,
        requesting_entity: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search across memories using hybrid cloud/local strategy.

        Strategy:
        1. If cloud credentials are configured and prefer_cloud=True,
           try cloud search first with 3s timeout
        2. On cloud failure or no credentials, fall back to local search
        3. Local search uses sqlite-vec (if available) or text matching

        Args:
            query: Search query string
            limit: Maximum results to return
            record_types: Filter by memory type (episode, note, belief, value, goal)
            prefer_cloud: If True, try cloud search first (default True)

        Returns:
            List of SearchResult objects
        """
        types = record_types or ["episode", "note", "belief", "value", "goal"]

        # Try cloud search first if configured and preferred
        if prefer_cloud and self.has_cloud_credentials():
            cloud_results = self._cloud_search(query, limit, types)
            if cloud_results is not None:
                logger.debug(f"Cloud search returned {len(cloud_results)} results")
                return cloud_results
            logger.debug("Cloud search failed, falling back to local search")

        # Fall back to local search
        return self._local_search(query, limit, types, requesting_entity=requesting_entity)

    def _local_search(
        self, query: str, limit: int, types: List[str], requesting_entity: Optional[str] = None
    ) -> List[SearchResult]:
        """Local search using sqlite-vec or text matching.

        Args:
            query: Search query
            limit: Maximum results
            types: Memory types to search
            requesting_entity: If provided, filter by access_grants.

        Returns:
            List of SearchResult
        """
        if self._has_vec:
            results = self._vector_search(query, limit, types)
        else:
            results = self._text_search(query, limit, types, requesting_entity=requesting_entity)

        # Apply privacy filtering for external entity access
        if requesting_entity is not None:
            filtered = []
            for r in results:
                grants = getattr(r.record, "access_grants", None)
                if grants and requesting_entity in grants:
                    filtered.append(r)
                # NULL or empty grants = private, skip for external access
            return filtered[:limit]
        return results

    def _vector_search(self, query: str, limit: int, types: List[str]) -> List[SearchResult]:
        """Semantic search using sqlite-vec."""
        results = []

        # Embed query
        query_embedding = self._embed_text(query, context="vector-search")
        if not query_embedding:
            return self._text_search(query, limit, types)
        query_packed = pack_embedding(query_embedding)

        # Map types to table names
        table_map = {
            "episode": "episodes",
            "note": "notes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
        }

        with self._connect() as conn:
            # Build table prefix filter
            table_prefixes = [table_map[t] for t in types if t in table_map]

            # Query vector table for nearest neighbors
            # Use KNN search with sqlite-vec
            try:
                rows = conn.execute(
                    """SELECT id, distance
                       FROM vec_embeddings
                       WHERE embedding MATCH ?
                       ORDER BY distance
                       LIMIT ?""",
                    (query_packed, limit * 2),  # Get more to filter by type
                ).fetchall()
            except Exception as e:
                logger.warning(
                    f"Vector search failed: {e}, falling back to text search", exc_info=True
                )
                return self._text_search(query, limit, types)

            # Fetch actual records
            # Security: filter by stack_id prefix first to prevent timing side-channel
            agent_prefix = f"{self.stack_id}:"
            for row in rows:
                vec_id = row["id"]
                distance = row["distance"]

                # Parse vec_id - supports both new and legacy formats
                # New format: stack_id:table:record_id
                # Legacy format: table:record_id
                if vec_id.startswith(agent_prefix):
                    # New format - stack_id verified
                    parts = vec_id.split(":", 2)
                    if len(parts) != 3:
                        continue
                    _, table_name, record_id = parts
                else:
                    # Legacy format - stack_id will be verified in _fetch_record
                    parts = vec_id.split(":", 1)
                    if len(parts) != 2:
                        continue
                    table_name, record_id = parts

                # Filter by requested types
                if table_name not in table_prefixes:
                    continue

                # Convert distance to similarity score (lower distance = higher score)
                # For cosine distance, range is [0, 2], so we normalize
                score = max(0.0, 1.0 - distance / 2.0)

                # Fetch the actual record
                record, record_type = self._fetch_record(conn, table_name, record_id)
                if record:
                    results.append(
                        SearchResult(record=record, record_type=record_type, score=score)
                    )

                if len(results) >= limit:
                    break

        return results

    def _fetch_record(self, conn: sqlite3.Connection, table: str, record_id: str) -> tuple:
        """Fetch a record by table and ID."""
        type_map = {
            "episodes": ("episode", self._row_to_episode),
            "notes": ("note", self._row_to_note),
            "beliefs": ("belief", self._row_to_belief),
            "agent_values": ("value", self._row_to_value),
            "goals": ("goal", self._row_to_goal),
        }

        if table not in type_map:
            return None, None

        record_type, converter = type_map[table]
        validate_table_name(table)  # Security: validate before SQL use

        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0",
            (record_id, self.stack_id),
        ).fetchone()

        if row:
            return converter(row), record_type
        return None, None

    @staticmethod
    def _tokenize_query(query: str) -> List[str]:
        """Split a search query into meaningful tokens. Delegates to search_impl."""
        return _search_impl.tokenize_query(query)

    @staticmethod
    def _build_token_filter(tokens: List[str], columns: List[str]) -> tuple:
        """Build a tokenized OR filter. Delegates to search_impl."""
        return _search_impl.build_token_filter(tokens, columns)

    @staticmethod
    def _token_match_score(text: str, tokens: List[str]) -> float:
        """Score text by token match. Delegates to search_impl."""
        return _search_impl.token_match_score(text, tokens)

    def _text_search(
        self, query: str, limit: int, types: List[str], requesting_entity: Optional[str] = None
    ) -> List[SearchResult]:
        """Fallback text-based search. Delegates to search_impl."""
        row_converters = {
            "episode": self._row_to_episode,
            "note": self._row_to_note,
            "belief": self._row_to_belief,
            "value": self._row_to_value,
            "goal": self._row_to_goal,
        }
        with self._connect() as conn:
            return _search_impl.text_search(
                conn,
                self.stack_id,
                query,
                limit,
                types,
                row_converters,
                self._build_access_filter,
                requesting_entity=requesting_entity,
            )

    # === Stats ===

    def get_stats(self) -> Dict[str, int]:
        """Get counts of each record type. Delegates to stats_ops."""
        with self._connect() as conn:
            return _stats_ops.get_stats(conn, self.stack_id)

    # === Batch Loading ===

    def load_all(
        self,
        values_limit: Optional[int] = 10,
        beliefs_limit: Optional[int] = 20,
        goals_limit: Optional[int] = 10,
        goals_status: str = "active",
        episodes_limit: Optional[int] = 20,
        notes_limit: Optional[int] = 5,
        drives_limit: Optional[int] = None,
        relationships_limit: Optional[int] = None,
        epoch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load all memory types in a single database connection. Delegates to stats_ops."""
        row_converters = {
            "value": self._row_to_value,
            "belief": self._row_to_belief,
            "goal": self._row_to_goal,
            "drive": self._row_to_drive,
            "episode": self._row_to_episode,
            "note": self._row_to_note,
            "relationship": self._row_to_relationship,
        }
        with self._connect() as conn:
            return _stats_ops.load_all(
                conn,
                self.stack_id,
                row_converters,
                values_limit=values_limit,
                beliefs_limit=beliefs_limit,
                goals_limit=goals_limit,
                goals_status=goals_status,
                episodes_limit=episodes_limit,
                notes_limit=notes_limit,
                drives_limit=drives_limit,
                relationships_limit=relationships_limit,
                epoch_id=epoch_id,
            )

    # === Meta-Memory ===

    def get_memory(self, memory_type: str, memory_id: str) -> Optional[Any]:
        """Get a memory by type and ID. Delegates to meta_memory_ops."""
        row_converters = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "drive": ("drives", self._row_to_drive),
            "relationship": ("relationships", self._row_to_relationship),
        }
        with self._connect() as conn:
            return _meta_memory_ops.get_memory(
                conn, self.stack_id, memory_type, memory_id, row_converters
            )

    def _get_belief_by_id(self, belief_id: str) -> Optional[Belief]:
        """Get a belief by ID. Delegates to beliefs_crud."""
        return _beliefs_crud.get_belief_by_id(self._connect, self.stack_id, belief_id)

    def _get_value_by_id(self, value_id: str) -> Optional[Value]:
        """Get a value by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_values WHERE id = ? AND stack_id = ? AND deleted = 0",
                (value_id, self.stack_id),
            ).fetchone()
        return self._row_to_value(row) if row else None

    def _get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ? AND stack_id = ? AND deleted = 0",
                (goal_id, self.stack_id),
            ).fetchone()
        return self._row_to_goal(row) if row else None

    def _get_note_by_id(self, note_id: str) -> Optional[Note]:
        """Get a note by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ? AND stack_id = ? AND deleted = 0",
                (note_id, self.stack_id),
            ).fetchone()
        return self._row_to_note(row) if row else None

    def _get_drive_by_id(self, drive_id: str) -> Optional[Drive]:
        """Get a drive by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM drives WHERE id = ? AND stack_id = ? AND deleted = 0",
                (drive_id, self.stack_id),
            ).fetchone()
        return self._row_to_drive(row) if row else None

    def _get_relationship_by_id(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relationships WHERE id = ? AND stack_id = ? AND deleted = 0",
                (relationship_id, self.stack_id),
            ).fetchone()
        return self._row_to_relationship(row) if row else None

    def update_memory_meta(
        self,
        memory_type: str,
        memory_id: str,
        confidence: Optional[float] = None,
        source_type: Optional[str] = None,
        source_episodes: Optional[List[str]] = None,
        derived_from: Optional[List[str]] = None,
        last_verified: Optional[datetime] = None,
        verification_count: Optional[int] = None,
        confidence_history: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Update meta-memory fields. Delegates to meta_memory_ops."""
        with self._connect() as conn:
            return _meta_memory_ops.update_memory_meta(
                conn,
                self.stack_id,
                memory_type,
                memory_id,
                self._to_json,
                self._now,
                self._queue_sync,
                _normalize_source_type,
                lambda t, i, d: check_derived_from_cycle(self, t, i, d),
                confidence=confidence,
                source_type=source_type,
                source_episodes=source_episodes,
                derived_from=derived_from,
                last_verified=last_verified,
                verification_count=verification_count,
                confidence_history=confidence_history,
            )

    def get_memories_by_confidence(
        self,
        threshold: float,
        below: bool = True,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories filtered by confidence. Delegates to meta_memory_ops."""
        row_converters = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "drive": ("drives", self._row_to_drive),
            "relationship": ("relationships", self._row_to_relationship),
        }
        with self._connect() as conn:
            return _meta_memory_ops.get_memories_by_confidence(
                conn,
                self.stack_id,
                threshold,
                row_converters,
                self._safe_get,
                below=below,
                memory_types=memory_types,
                limit=limit,
            )

    def get_memories_by_source(
        self,
        source_type: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories filtered by source type. Delegates to meta_memory_ops."""
        row_converters = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "drive": ("drives", self._row_to_drive),
            "relationship": ("relationships", self._row_to_relationship),
        }
        with self._connect() as conn:
            return _meta_memory_ops.get_memories_by_source(
                conn,
                self.stack_id,
                source_type,
                row_converters,
                self._safe_get,
                memory_types=memory_types,
                limit=limit,
            )

    # === Forgetting ===

    def record_access(self, memory_type: str, memory_id: str) -> bool:
        """Record that a memory was accessed (for salience tracking).

        Increments times_accessed and updates last_accessed timestamp.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if updated, False if memory not found
        """
        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        now = self._now()

        # Boost strength slightly on access (diminishing returns)
        # boost = 0.02 / (1 + times_accessed / 10) — starts at 0.02, falls to ~0.002 at 100 accesses
        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET times_accessed = COALESCE(times_accessed, 0) + 1,
                       last_accessed = ?,
                       local_updated_at = ?,
                       strength = MIN(1.0,
                           COALESCE(strength, 1.0) + 0.02 / (1.0 + COALESCE(times_accessed, 0) / 10.0))
                   WHERE id = ? AND stack_id = ?""",
                (now, now, memory_id, self.stack_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def record_access_batch(self, accesses: List[tuple[str, str]]) -> int:
        """Record multiple memory accesses in a single transaction.

        This is more efficient than calling record_access() for each item
        because it uses a single database connection and transaction.

        Args:
            accesses: List of (memory_type, memory_id) tuples

        Returns:
            Number of memories successfully updated
        """
        if not accesses:
            return 0

        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }

        # Group by table for efficient batch updates
        by_table: Dict[str, List[str]] = {}
        for memory_type, memory_id in accesses:
            table = table_map.get(memory_type)
            if table:
                if table not in by_table:
                    by_table[table] = []
                by_table[table].append(memory_id)

        if not by_table:
            return 0

        now = self._now()
        total_updated = 0

        with self._connect() as conn:
            for table, ids in by_table.items():
                # Validate table name for safety
                validate_table_name(table)

                # Update all IDs in this table at once
                placeholders = ",".join("?" * len(ids))
                cursor = conn.execute(
                    f"""UPDATE {table}
                       SET times_accessed = COALESCE(times_accessed, 0) + 1,
                           last_accessed = ?,
                           local_updated_at = ?
                       WHERE id IN ({placeholders}) AND stack_id = ?""",
                    (now, now, *ids, self.stack_id),
                )
                total_updated += cursor.rowcount

            conn.commit()

        return total_updated

    def update_strength(self, memory_type: str, memory_id: str, strength: float) -> bool:
        """Update the strength field of a memory. Delegates to meta_memory_ops."""
        with self._connect() as conn:
            return _meta_memory_ops.update_strength(
                conn, self.stack_id, memory_type, memory_id, strength, self._now
            )

    def update_strength_batch(self, updates: list[tuple[str, str, float]]) -> int:
        """Update strength for multiple memories. Delegates to meta_memory_ops."""
        with self._connect() as conn:
            return _meta_memory_ops.update_strength_batch(conn, self.stack_id, updates, self._now)

    def update_memory_metadata(
        self,
        memory_type: str,
        memory_id: str,
        **kwargs: Any,
    ) -> bool:
        """Update arbitrary metadata fields on a memory record.

        Only updates fields that exist in the schema for the given table.
        Used by Stack for persisting component metadata (e.g. emotional tags).

        Args:
            memory_type: Type of memory (episode, belief, note, etc.)
            memory_id: ID of the memory
            **kwargs: Field name/value pairs to update

        Returns:
            True if any fields were updated, False if not found or no valid fields
        """
        import json as _json

        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }
        table = table_map.get(memory_type)
        if not table:
            return False

        # Allowed schema fields per table
        allowed_fields = {
            "episodes": {"emotional_valence", "emotional_arousal", "emotional_tags"},
        }
        fields = allowed_fields.get(table, set())
        if not fields:
            return False

        updates = []
        params = []
        for key, value in kwargs.items():
            if key not in fields:
                continue
            updates.append(f"{key} = ?")
            if isinstance(value, (list, dict)):
                params.append(_json.dumps(value))
            else:
                params.append(value)

        if not updates:
            return False

        params.extend([memory_id, self.stack_id])
        sql = f"UPDATE {table} SET {', '.join(updates)} WHERE id = ? AND stack_id = ?"
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0

    def get_all_active_memories(
        self, memory_types: Optional[list[str]] = None
    ) -> list[tuple[str, Any]]:
        """Get all active (non-deleted, non-forgotten) memories for strength decay.

        Args:
            memory_types: Types to include (default: all except raw)

        Returns:
            List of (memory_type, record) tuples
        """
        if memory_types is None:
            memory_types = ["episode", "belief", "goal", "note", "relationship"]

        table_map = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "relationship": ("relationships", self._row_to_relationship),
        }

        results = []
        with self._connect() as conn:
            for mtype in memory_types:
                entry = table_map.get(mtype)
                if not entry:
                    continue
                table, converter = entry
                rows = conn.execute(
                    f"""SELECT * FROM {table}
                       WHERE stack_id = ?
                         AND deleted = 0
                         AND COALESCE(is_protected, 0) = 0
                         AND COALESCE(strength, 1.0) > 0.0
                       ORDER BY strength ASC
                       LIMIT 500""",
                    (self.stack_id,),
                ).fetchall()
                for row in rows:
                    record = converter(dict(row))
                    results.append((mtype, record))

        return results

    # === Memory Lifecycle Operations (delegated to MemoryOps) ===

    def forget_memory(self, memory_type: str, memory_id: str, reason: Optional[str] = None) -> bool:
        """Tombstone a memory (mark as forgotten, don't delete)."""
        return self._memory_ops.forget_memory(memory_type, memory_id, reason)

    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory."""
        return self._memory_ops.recover_memory(memory_type, memory_id)

    def protect_memory(self, memory_type: str, memory_id: str, protected: bool = True) -> bool:
        """Mark a memory as protected from forgetting."""
        return self._memory_ops.protect_memory(memory_type, memory_id, protected)

    def log_audit(
        self,
        memory_type: str,
        memory_id: str,
        operation: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Log an audit entry for a memory operation."""
        return self._memory_ops.log_audit(
            memory_type, memory_id, operation, actor, details, correlation_id
        )

    def get_audit_log(
        self,
        *,
        memory_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return self._memory_ops.get_audit_log(
            memory_type=memory_type,
            memory_id=memory_id,
            operation=operation,
            correlation_id=correlation_id,
            limit=limit,
        )

    def export_audit_jsonl(self, **kwargs):
        """Export audit log entries as JSONL lines."""
        return self._memory_ops.export_audit_jsonl(**kwargs)

    def log_belief_revision(
        self,
        old_id: str,
        new_id: str,
        reason: str | None = None,
        actor: str = "system",
        correlation_id: str | None = None,
    ) -> tuple[str, str]:
        """Log audit entries for a belief revision."""
        return self._memory_ops.log_belief_revision(old_id, new_id, reason, actor, correlation_id)

    def weaken_memory(self, memory_type: str, memory_id: str, amount: float) -> bool:
        """Reduce a memory's strength by a given amount."""
        return self._memory_ops.weaken_memory(memory_type, memory_id, amount)

    def verify_memory(self, memory_type: str, memory_id: str) -> bool:
        """Verify a memory: boost strength and increment verification count."""
        return self._memory_ops.verify_memory(memory_type, memory_id)

    def boost_memory_strength(self, memory_type: str, memory_id: str, amount: float) -> bool:
        """Boost a memory's strength by a given amount (capped at 1.0)."""
        return self._memory_ops.boost_memory_strength(memory_type, memory_id, amount)

    def get_memories_derived_from(self, memory_type: str, memory_id: str) -> List[tuple]:
        """Find all memories that cite 'type:id' in their derived_from."""
        return self._memory_ops.get_memories_derived_from(memory_type, memory_id)

    def get_ungrounded_memories(self, stack_id: str) -> List[tuple]:
        """Find memories where ALL source refs have strength 0.0 or don't exist."""
        return self._memory_ops.get_ungrounded_memories(stack_id)

    def get_pre_v09_memories(self, stack_id: str) -> List[tuple]:
        """Find memories annotated with kernle:pre-v0.9-migration."""
        return self._memory_ops.get_pre_v09_memories(stack_id)

    def get_forgetting_candidates(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
        threshold: float = 0.5,
    ) -> List[SearchResult]:
        """Get memories that are candidates for forgetting."""
        return self._memory_ops.get_forgetting_candidates(
            self._row_converters(), memory_types, limit, threshold
        )

    def get_forgotten_memories(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get all forgotten (tombstoned) memories."""
        return self._memory_ops.get_forgotten_memories(self._row_converters(), memory_types, limit)

    def _row_converters(self) -> Dict[str, Any]:
        """Return a dict of memory_type -> row converter callable."""
        return {
            "episode": self._row_to_episode,
            "belief": self._row_to_belief,
            "value": self._row_to_value,
            "goal": self._row_to_goal,
            "note": self._row_to_note,
            "drive": self._row_to_drive,
            "relationship": self._row_to_relationship,
        }

    # === Sync Engine (delegated to SyncEngine) ===

    def queue_sync_operation(self, operation: str, table: str, record_id: str, data=None) -> int:
        """Queue a sync operation for later synchronization."""
        return self._sync_engine.queue_sync_operation(operation, table, record_id, data)

    def get_pending_sync_operations(self, limit: int = 100):
        """Get all unsynced operations from the queue."""
        return self._sync_engine.get_pending_sync_operations(limit)

    def mark_synced(self, ids) -> int:
        """Mark sync queue entries as synced."""
        return self._sync_engine.mark_synced(ids)

    def get_sync_status(self):
        """Get sync queue status with counts."""
        return self._sync_engine.get_sync_status()

    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        return self._sync_engine.get_pending_sync_count()

    def get_queued_changes(self, limit: int = 100, max_retries: int = 5):
        """Get queued changes for sync."""
        return self._sync_engine.get_queued_changes(limit, max_retries)

    def _clear_queued_change(self, conn, queue_id: int):
        """Mark a change as synced."""
        self._sync_engine._clear_queued_change(conn, queue_id)

    def _record_sync_failure(self, conn, queue_id: int, error: str) -> int:
        """Record a sync failure and increment retry count."""
        return self._sync_engine._record_sync_failure(conn, queue_id, error)

    def get_failed_sync_records(self, min_retries: int = 5):
        """Get sync records that have exceeded max retries."""
        return self._sync_engine.get_failed_sync_records(min_retries)

    def clear_failed_sync_records(self, older_than_days: int = 7) -> int:
        """Clear failed sync records older than the specified days."""
        return self._sync_engine.clear_failed_sync_records(older_than_days)

    def get_dead_letter_count(self) -> int:
        """Get count of dead-lettered sync records."""
        return self._sync_engine.get_dead_letter_count()

    def requeue_dead_letters(self, record_ids: list[int] | None = None) -> int:
        """Re-enqueue dead-lettered entries for retry."""
        return self._sync_engine.requeue_dead_letters(record_ids)

    def _get_sync_meta(self, key: str):
        """Get a sync metadata value."""
        return self._sync_engine._get_sync_meta(key)

    def _set_sync_meta(self, key: str, value: str):
        """Set a sync metadata value."""
        self._sync_engine._set_sync_meta(key, value)

    def get_last_sync_time(self):
        """Get the timestamp of the last successful sync."""
        return self._sync_engine.get_last_sync_time()

    def get_sync_conflicts(self, limit: int = 100):
        """Get recent sync conflict history."""
        return self._sync_engine.get_sync_conflicts(limit)

    def save_sync_conflict(self, conflict) -> str:
        """Save a sync conflict record."""
        return self._sync_engine.save_sync_conflict(conflict)

    def clear_sync_conflicts(self, before=None) -> int:
        """Clear sync conflict history."""
        return self._sync_engine.clear_sync_conflicts(before)

    def is_online(self) -> bool:
        """Check if cloud storage is reachable."""
        return self._sync_engine.is_online()

    def _mark_synced(self, conn, table: str, record_id: str):
        """Mark a record as synced with the cloud."""
        self._sync_engine._mark_synced(conn, table, record_id)

    def _get_record_for_push(self, table: str, record_id: str):
        """Get a record by table and ID for pushing to cloud."""
        return self._sync_engine._get_record_for_push(table, record_id)

    def _push_record(self, table: str, record) -> bool:
        """Push a single record to cloud storage."""
        return self._sync_engine._push_record(table, record)

    def sync(self):
        """Sync with cloud storage."""
        return self._sync_engine.sync()

    def pull_changes(self, since=None):
        """Pull changes from cloud since the given timestamp."""
        return self._sync_engine.pull_changes(since)

    def apply_pull_operation(self, op: dict):
        """Apply a single HTTP pull operation via SyncEngine."""
        return self._sync_engine.apply_pull_operation(op)

    def _record_from_pull_data(self, table: str, record_id: str, data: dict):
        """Convert HTTP pull data dict to typed record via SyncEngine."""
        return self._sync_engine._record_from_pull_data(table, record_id, data)

    def _merge_array_fields(self, table: str, winner, loser):
        """Merge array fields from loser into winner using set union."""
        return self._sync_engine._merge_array_fields(table, winner, loser)

    # === Health Check Compliance Tracking ===

    def log_health_check(
        self, anxiety_score: Optional[int] = None, source: str = "cli", triggered_by: str = "manual"
    ) -> str:
        """Log a health check event for compliance tracking."""
        return _log_health_check(
            self._connect, self.stack_id, self._now(), anxiety_score, source, triggered_by
        )

    def get_health_check_stats(self) -> Dict[str, Any]:
        """Get health check compliance statistics."""
        return _get_health_check_stats(self._connect, self.stack_id)

    # === Admin Methods (CLI stack management) ===

    def list_stack_ids(self) -> list[str]:
        """List all distinct stack IDs across all tables."""
        tables = [
            "episodes",
            "notes",
            "beliefs",
            "goals",
            "agent_values",
            "drives",
            "relationships",
            "raw_entries",
        ]
        with self._connect() as conn:
            parts = []
            for table in tables:
                try:
                    parts.append(f"SELECT DISTINCT stack_id FROM {validate_table_name(table)}")
                except ValueError:
                    continue  # Invalid table name — skip
            if not parts:
                return []
            query = " UNION ".join(parts)
            try:
                rows = conn.execute(query).fetchall()
                return sorted(row[0] for row in rows)
            except sqlite3.Error:
                return []  # Query failed — degrade to empty list

    def get_stack_counts(self, stack_id: str) -> dict[str, int]:
        """Get record counts per table for a stack.

        Covers all tables that delete_stack_data() touches so callers
        get a complete picture before deletion.
        """
        # Only tables with a stack_id column. checkpoints and sync_queue
        # are excluded because they lack stack_id.
        table_map = {
            "episodes": "episodes",
            "notes": "notes",
            "beliefs": "beliefs",
            "goals": "goals",
            "values": "agent_values",
            "drives": "drives",
            "relationships": "relationships",
            "playbooks": "playbooks",
            "raw_entries": "raw_entries",
        }
        import sqlite3

        counts: dict[str, int] = {}
        with self._connect() as conn:
            for label, table in table_map.items():
                try:
                    validate_table_name(table)
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE stack_id = ?",
                        (stack_id,),
                    ).fetchone()
                    counts[label] = row[0] if row else 0
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e):
                        counts[label] = 0
                    else:
                        raise
        return counts

    def delete_stack_data(self, stack_id: str) -> dict[str, int]:
        """Delete all data for a stack across all tables in a single transaction.

        Tables that exist in the database are deleted atomically — if any
        DELETE statement fails (other than "no such table"), the entire
        transaction is rolled back and the error propagates.  Missing tables
        are skipped silently since schema versions vary across deployments.

        Returns:
            Dict mapping table name to number of deleted rows (only non-zero entries).

        Raises:
            sqlite3.OperationalError: If a DELETE fails for reasons other than
                a missing table.
        """
        import sqlite3

        # Only tables with a stack_id column. checkpoints and sync_queue
        # are excluded because they lack stack_id.
        tables = [
            "episodes",
            "notes",
            "beliefs",
            "goals",
            "agent_values",
            "drives",
            "relationships",
            "playbooks",
            "raw_entries",
        ]
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            for table in tables:
                validate_table_name(table)
                try:
                    cursor = conn.execute(f"DELETE FROM {table} WHERE stack_id = ?", (stack_id,))
                    if cursor.rowcount > 0:
                        deleted[table] = cursor.rowcount
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e):
                        continue
                    raise

            # Vec tables use LIKE pattern (id format: {stack_id}:...)
            escaped = escape_like_pattern(stack_id)
            like_pattern = f"{escaped}:%"
            for vec_table in ("vec_embeddings", "embedding_meta"):
                try:
                    cursor = conn.execute(
                        f"DELETE FROM {vec_table} WHERE id LIKE ? ESCAPE '\\'",
                        (like_pattern,),
                    )
                    if cursor.rowcount > 0:
                        deleted[vec_table] = cursor.rowcount
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e):
                        continue
                    raise

        return deleted
