"""Sync commands for Kernle CLI — local-to-cloud synchronization."""

import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from kernle.core.validation import validate_backend_url as _validate_backend_url
from kernle.utils import get_kernle_home

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)

PULL_POISON_META_KEY = "pull_poison_operations"


def cmd_sync(args, k: "Kernle"):
    """Handle sync subcommands for local-to-cloud synchronization."""
    # Load credentials with priority:
    # 1. ~/.kernle/credentials.json (preferred)
    # 2. Environment variables (fallback)
    # 3. ~/.kernle/config.json (legacy fallback)

    backend_url = None
    auth_token = None
    user_id = None

    # Try credentials.json first (preferred)
    credentials_path = get_kernle_home() / "credentials.json"
    if credentials_path.exists():
        try:
            import json as json_module

            with open(credentials_path) as f:
                creds = json_module.load(f)
                backend_url = creds.get("backend_url")
                # Support multiple auth token field names
                auth_token = creds.get("auth_token") or creds.get("token") or creds.get("api_key")
                user_id = creds.get("user_id")
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.debug(f"Failed to load credentials file: {e}", exc_info=True)
            # Fall through to env vars

    # Fall back to environment variables
    if not backend_url:
        backend_url = os.environ.get("KERNLE_BACKEND_URL")
    if not auth_token:
        auth_token = os.environ.get("KERNLE_AUTH_TOKEN")
    if not user_id:
        user_id = os.environ.get("KERNLE_USER_ID")

    # Legacy fallback: check config.json
    config_path = get_kernle_home() / "config.json"
    if config_path.exists() and (not backend_url or not auth_token):
        try:
            import json as json_module

            with open(config_path) as f:
                config = json_module.load(f)
                backend_url = backend_url or config.get("backend_url")
                auth_token = auth_token or config.get("auth_token")
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.debug(f"Failed to load legacy config file: {e}", exc_info=True)

    # Validate backend URL security
    if backend_url:
        backend_url = _validate_backend_url(backend_url)

    def get_local_project_name():
        """Extract the local project name from stack_id (without namespace)."""
        # k.stack_id might be "roundtable" or "user123/roundtable"
        # We want just "roundtable"
        stack_id = k.stack_id
        if "/" in stack_id:
            return stack_id.split("/")[-1]
        return stack_id

    def get_namespaced_stack_id():
        """Get the full namespaced agent ID (user_id/project_name)."""
        project_name = get_local_project_name()
        if user_id:
            return f"{user_id}/{project_name}"
        return project_name

    def get_http_client():
        """Get an HTTP client for backend requests."""
        try:
            import httpx

            return httpx
        except ImportError:
            print("✗ httpx not installed. Run: pip install httpx")
            sys.exit(1)

    def check_backend_connection(httpx_client):
        """Check if backend is reachable and authenticated."""
        if not backend_url:
            return False, "No backend URL configured"
        if not auth_token:
            return False, "Not authenticated (run `kernle auth login`)"

        try:
            response = httpx_client.get(
                f"{backend_url.rstrip('/')}/health",
                timeout=5.0,
            )
            if response.status_code == 200:
                return True, "Connected"
            return False, f"Backend returned status {response.status_code}"
        except Exception as e:
            logger.debug("Sync connectivity check failed: %s", e, exc_info=True)
            return False, f"Connection failed: {e}"

    def get_headers():
        """Get authorization headers for backend requests."""
        return {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

    def format_datetime(dt):
        """Format datetime for API requests."""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    # Map local table names to backend table names
    table_name_map = {
        "values": "values",
        "agent_beliefs": "beliefs",
        "agent_episodes": "episodes",
        "agent_notes": "notes",
        "agent_goals": "goals",
        "agent_drives": "drives",
        "agent_relationships": "relationships",
        "agent_playbooks": "playbooks",
        "agent_raw": "raw_captures",
        "raw_entries": "raw_captures",  # actual local table name
    }

    def _operation_identity(table, record_id):
        """Build a stable identity tuple for sync operation matching."""
        if not record_id:
            return None
        return (table or "", str(record_id))

    def _extract_identities(items):
        """Extract operation identities from response/conflict structures."""
        identities = set()
        if not isinstance(items, list):
            return identities

        for item in items:
            if isinstance(item, dict):
                table = item.get("table") or item.get("table_name")
                record_id = item.get("record_id") or item.get("id")
                identity = _operation_identity(table, record_id)
                if identity:
                    identities.add(identity)
            elif isinstance(item, str):
                # Support record_id-only formats
                identities.add(("", item))
        return identities

    def _identity_matches(identity, identities):
        """Match full identity first, then record_id-only fallback."""
        if not identity:
            return False
        return identity in identities or ("", identity[1]) in identities

    def _resolve_acked_changes(result, operations, op_identity_to_change):
        """Resolve acknowledged queue items using operation identity, not position."""
        if not operations:
            return []

        explicit_ack_keys = (
            "acknowledged_operations",
            "acknowledged",
            "acked_operations",
            "synced_operations",
            "applied_operations",
            "applied",
            "synced_records",
            "record_ids",
        )

        ack_identities = set()
        for key in explicit_ack_keys:
            value = result.get(key)
            if isinstance(value, list):
                ack_identities = _extract_identities(value)
                if ack_identities:
                    break

        if ack_identities:
            acknowledged = []
            seen = set()
            for identity, change in op_identity_to_change.items():
                if identity in seen:
                    continue
                if _identity_matches(identity, ack_identities):
                    acknowledged.append(change)
                    seen.add(identity)
            return acknowledged

        # Fallback inference: treat conflict identities as failed and acknowledge the rest.
        conflict_identities = _extract_identities(result.get("conflicts", []))
        inferred_identities = []
        for op in operations:
            identity = _operation_identity(op.get("table"), op.get("record_id"))
            if identity and not _identity_matches(identity, conflict_identities):
                inferred_identities.append(identity)

        # Safety guard: if server count disagrees with identity inference, do not ack blindly.
        synced_count = result.get("synced")
        if isinstance(synced_count, int) and synced_count != len(inferred_identities):
            logger.warning(
                "Push acknowledgement mismatch: synced=%s inferred=%s; refusing positional ack.",
                synced_count,
                len(inferred_identities),
            )
            return []

        acknowledged = []
        seen = set()
        for identity in inferred_identities:
            if identity in seen:
                continue
            change = op_identity_to_change.get(identity)
            if change:
                acknowledged.append(change)
                seen.add(identity)
        return acknowledged

    def _build_push_operations(queued_changes):
        """Build push payload operations and map them back to queue identities."""
        operations = []
        skipped_orphans = 0
        op_identity_to_change = {}

        for change in queued_changes:
            op_type = (
                "update" if change.operation in ("upsert", "insert", "update") else change.operation
            )

            # Map table name for backend
            backend_table = table_name_map.get(change.table_name, change.table_name)

            op_data = {
                "operation": op_type,
                "table": backend_table,
                "record_id": change.record_id,
                "local_updated_at": format_datetime(change.queued_at),
                "version": 1,
            }

            # Payload-first: use stored data, fall back to source table
            if op_type != "delete":
                record_dict = None

                if change.payload:
                    try:
                        record_dict = json.loads(change.payload)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if not record_dict:
                    record = k._storage._get_record_for_push(change.table_name, change.record_id)
                    if record:
                        # Extract dataclass fields directly to keep payload schema-aligned.
                        import dataclasses as _dc

                        record_dict = {}
                        for f in _dc.fields(record):
                            value = getattr(record, f.name)
                            if value is None:
                                continue
                            if hasattr(value, "isoformat"):
                                value = value.isoformat()
                            record_dict[f.name] = value

                if record_dict:
                    op_data["data"] = record_dict
                else:
                    skipped_orphans += 1
                    with k._storage._connect() as conn:
                        conn.execute(
                            "UPDATE sync_queue SET synced = 1 WHERE id = ?",
                            (change.id,),
                        )
                    continue

            operations.append(op_data)
            identity = _operation_identity(backend_table, change.record_id)
            if identity:
                op_identity_to_change[identity] = change

        return operations, skipped_orphans, op_identity_to_change

    def _serialize_json(value):
        """Serialize to stable JSON for metadata storage/fingerprinting."""
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        except Exception as exc:
            logger.debug(
                "Swallowed %s in _serialize_json, using fallback: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return json.dumps(str(value))

    def _operation_payload_hash(operation):
        """Build stable hash for operation payload for conflict comparison."""
        return hashlib.sha256(
            _serialize_json((operation or {}).get("data")).encode("utf-8")
        ).hexdigest()

    def _build_conflict_envelope(operation, error=None, resolution="unknown", stage="sync"):
        """Build a normalized conflict envelope for pull/push diagnostics."""
        operation = operation or {}
        table = str(operation.get("table") or operation.get("table_name") or "unknown")
        record_id = str(operation.get("record_id") or operation.get("id") or "unknown")
        op_name = str(operation.get("operation") or "unknown")
        payload = operation.get("data")
        return {
            "id": str(uuid4()),
            "stage": stage,
            "operation_identity": _operation_identity(table, record_id),
            "operation": op_name,
            "table": table,
            "record_id": record_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolution": resolution,
            "error": error or "unknown error",
            "payload_hash": _operation_payload_hash(operation),
            "payload_snapshot": _serialize_json(payload),
        }

    def _normalize_push_conflict(conflict, operations):
        """Normalize backend-sourced conflict entries into stable envelopes."""
        if not isinstance(conflict, dict):
            conflict = {"error": str(conflict)}
        table = str(conflict.get("table") or conflict.get("table_name") or "unknown")
        record_id = str(conflict.get("record_id") or "unknown")
        operation_name = str(conflict.get("operation") or "unknown")
        error = str(
            conflict.get("error")
            or conflict.get("message")
            or conflict.get("reason")
            or "version conflict"
        )

        matched = None
        if operations:
            for candidate in operations:
                if not isinstance(candidate, dict):
                    continue
                if str(candidate.get("record_id") or "") != record_id:
                    continue
                if table != "unknown" and str(candidate.get("table") or "unknown") != table:
                    continue
                if (
                    operation_name != "unknown"
                    and str(candidate.get("operation") or "unknown") != operation_name
                ):
                    continue
                matched = candidate
                break
            if matched is None and table == "unknown":
                for candidate in operations:
                    if str(candidate.get("record_id") or "") == record_id:
                        matched = candidate
                        break

        if not matched:
            matched = {"table": table, "record_id": record_id, "operation": operation_name}

        envelope = _build_conflict_envelope(
            matched,
            error=error,
            resolution="backend_rejected",
            stage="push",
        )
        envelope["backend_payload"] = {
            "raw": {k: v for k, v in conflict.items() if k != "raw"},
        }
        return envelope

    def _sort_conflicts(conflicts):
        """Sort conflict/envelope payloads deterministically."""
        return sorted(
            conflicts,
            key=lambda item: (
                str(item.get("table") or "unknown"),
                str(item.get("record_id") or item.get("operation", {}).get("record_id") or ""),
                str(item.get("operation") or item.get("operation_identity") or "unknown"),
            ),
        )

    def _conflict_snapshot(conflicts):
        """Build deterministic conflict summary for logs and JSON assertions."""
        normalized = _sort_conflicts(conflicts)
        return {
            "count": len(normalized),
            "records": [
                {
                    "table": item.get("table"),
                    "record_id": item.get("record_id"),
                    "operation": item.get("operation"),
                    "error": str(item.get("error") or ""),
                    "payload_hash": item.get("payload_hash", ""),
                }
                for item in normalized
            ],
        }

    def _pull_poison_key(op):
        """Build a stable key for a failed pull operation."""
        table = str(op.get("table") or "")
        record_id = str(op.get("record_id") or "")
        operation = str(op.get("operation") or "")
        if table and record_id and operation:
            return f"{table}:{record_id}:{operation}"
        digest = hashlib.sha256(_serialize_json(op).encode("utf-8")).hexdigest()[:16]
        return f"invalid:{digest}"

    def _load_pull_poison_records():
        """Load quarantined pull operations from sync metadata."""
        raw = k._storage._get_sync_meta(PULL_POISON_META_KEY)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Corrupt pull poison metadata; resetting.", exc_info=True)
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(key): value for key, value in parsed.items() if isinstance(value, dict)}

    def _save_pull_poison_records(records):
        """Persist quarantined pull operations."""
        k._storage._set_sync_meta(PULL_POISON_META_KEY, _serialize_json(records or {}))

    def _save_pull_apply_conflict(op, error, envelope=None):
        """Persist a failed pull apply as conflict history for observability."""
        from kernle.types import SyncConflict

        table = str(op.get("table") or "unknown")
        record_id = str(op.get("record_id") or f"unknown-{uuid4()}")
        envelope = envelope or _build_conflict_envelope(
            op, error=error, resolution="pull_apply_failed", stage="pull_apply"
        )
        summary = f"pull apply failed: {error}"[:200]
        local_version = {"payload": op}
        cloud_version = {"operation": op, "conflict_envelope": envelope}
        diff_payload = json.dumps(
            {"local": local_version, "cloud": cloud_version}, sort_keys=True, default=str
        )
        diff_hash = hashlib.sha256(diff_payload.encode("utf-8")).hexdigest()
        conflict = SyncConflict(
            id=str(uuid4()),
            table=table,
            record_id=record_id,
            local_version=local_version,
            cloud_version=cloud_version,
            resolution="pull_apply_failed",
            resolved_at=datetime.now(timezone.utc),
            local_summary="apply failed",
            cloud_summary=summary,
            diff_hash=diff_hash,
        )
        try:
            k._storage.save_sync_conflict(conflict)
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "Failed to persist pull apply conflict for %s:%s: %s",
                table,
                record_id,
                exc,
                extra={"table": table, "record_id": record_id, "error_type": type(exc).__name__},
                exc_info=True,
            )

    def _save_push_apply_conflict(envelope):
        """Persist a backend-rejected push conflict for auditability."""
        from kernle.types import SyncConflict

        table = str(envelope.get("table") or "unknown")
        record_id = str(envelope.get("record_id") or f"unknown-{uuid4()}")
        operation = str(envelope.get("operation") or "unknown")
        error = str(envelope.get("error") or "backend rejected")

        local_version = {
            "operation": {
                "table": table,
                "record_id": envelope.get("record_id"),
                "operation": operation,
            },
            "payload_hash": envelope.get("payload_hash"),
            "payload_snapshot": envelope.get("payload_snapshot"),
            "operation_identity": envelope.get("operation_identity"),
            "timestamp": envelope.get("timestamp"),
        }
        cloud_version = {
            "backend_payload": envelope.get("backend_payload"),
            "error": error,
        }
        diff_payload = json.dumps(
            {"local": local_version, "cloud": cloud_version}, sort_keys=True, default=str
        )
        diff_hash = hashlib.sha256(diff_payload.encode("utf-8")).hexdigest()

        conflict = SyncConflict(
            id=str(uuid4()),
            table=table,
            record_id=record_id,
            local_version=local_version,
            cloud_version=cloud_version,
            resolution="backend_rejected",
            resolved_at=datetime.now(timezone.utc),
            local_summary=f"{table}:{record_id} {operation}",
            cloud_summary=error,
            diff_hash=diff_hash,
        )

        try:
            k._storage.save_sync_conflict(conflict)
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "Failed to persist push conflict for %s:%s: %s",
                table,
                record_id,
                exc,
                extra={"table": table, "record_id": record_id, "error_type": type(exc).__name__},
                exc_info=True,
            )

    def _apply_single_pull_operation(op):
        """Apply a single pulled operation; return (applied, error_message)."""
        table = op.get("table")
        record_id = op.get("record_id")
        operation = op.get("operation")
        data = op.get("data", {})

        try:
            if not table or not record_id or not operation:
                return False, "invalid payload (missing table/record_id/operation)"

            if operation == "delete":
                # Delete merge behavior is not implemented in this CLI path yet.
                return False, "delete pull operations are not supported in CLI sync"

            if operation not in ("upsert", "insert", "update"):
                return False, f"unsupported operation type: {operation}"

            # Upsert known record types
            if table == "episodes" and data:
                from kernle.storage import Episode

                ep = Episode(
                    id=record_id,
                    stack_id=k.stack_id,
                    objective=data.get("objective", ""),
                    outcome_type=data.get("outcome_type", "neutral"),
                    outcome=data.get("outcome", data.get("outcome_description", "")),
                    lessons=data.get("lessons", data.get("lessons_learned", [])),
                    tags=data.get("tags", []),
                    source_type="external",
                    source_entity="kernle:sync",
                )
                k._storage.save_episode(ep)
                # Mark as synced (don't queue for push)
                with k._storage._connect() as conn:
                    k._storage._mark_synced(conn, table, record_id)
                    conn.execute(
                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                        (table, record_id),
                    )
                    conn.commit()
                return True, None

            if table == "notes" and data:
                from kernle.storage import Note

                note = Note(
                    id=record_id,
                    stack_id=k.stack_id,
                    content=data.get("content", ""),
                    note_type=data.get("note_type", "note"),
                    tags=data.get("tags", []),
                    source_type="external",
                    source_entity="kernle:sync",
                )
                k._storage.save_note(note)
                with k._storage._connect() as conn:
                    k._storage._mark_synced(conn, table, record_id)
                    conn.execute(
                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                        (table, record_id),
                    )
                    conn.commit()
                return True, None

            return False, f"unhandled pull operation for table={table}"

        except (sqlite3.Error, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            return False, f"{type(e).__name__}: {e}"

    def _retry_poisoned_pull_operations(poison_records):
        """Retry previously quarantined pull operations."""
        attempted = 0
        recovered = 0

        for key, entry in list(poison_records.items()):
            op = entry.get("operation_payload")
            if not isinstance(op, dict):
                continue

            attempted += 1
            applied, error = _apply_single_pull_operation(op)
            if applied:
                recovered += 1
                poison_records.pop(key, None)
                continue

            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry["last_seen_at"] = k._storage._now()
            entry["last_error"] = (error or "unknown error")[:500]
            poison_records[key] = entry

        return attempted, recovered

    def _apply_pull_operations(operations):
        """Apply pulled operations locally.

        Returns:
            tuple[int, list[dict], set[str]]: applied count, failed op details, applied keys.
        """
        applied = 0
        failed_ops = []
        applied_keys = set()

        for op in operations:
            key = _pull_poison_key(op)
            applied_ok, error = _apply_single_pull_operation(op)
            if applied_ok:
                applied += 1
                applied_keys.add(key)
                continue

            logger.debug(
                "Failed to apply pull operation for %s:%s (%s): %s",
                op.get("table"),
                op.get("record_id"),
                op.get("operation"),
                error,
            )
            failed_ops.append(
                {
                    "key": key,
                    "operation": op,
                    "error": (error or "unknown error")[:500],
                    "envelope": _build_conflict_envelope(
                        op,
                        error=error,
                        resolution="pull_apply_failed",
                        stage="pull_apply",
                    ),
                }
            )

        return applied, failed_ops, applied_keys

    def _quarantine_failed_pull_operations(failed_ops, poison_records):
        """Store failed pull ops in sync metadata and conflict history."""
        newly_poisoned = 0

        for failed in failed_ops:
            op = failed["operation"]
            key = failed["key"]
            error = failed["error"]
            envelope = failed.get("envelope") or _build_conflict_envelope(
                op, error=error, resolution="pull_apply_failed", stage="pull_apply"
            )
            digest = _serialize_json(op)

            existing = poison_records.get(key)
            first_seen = existing.get("first_seen_at") if existing else k._storage._now()
            attempts = int(existing.get("attempts", 0)) + 1 if existing else 1
            payload_changed = bool(existing and existing.get("payload_digest") != digest)

            poison_records[key] = {
                "table": op.get("table"),
                "record_id": op.get("record_id"),
                "operation": op.get("operation"),
                "attempts": attempts,
                "first_seen_at": first_seen,
                "last_seen_at": k._storage._now(),
                "last_error": error,
                "payload_digest": digest,
                "operation_payload": op,
                "conflict_envelope": envelope,
            }

            if existing is None or payload_changed:
                _save_pull_apply_conflict(op, error, envelope=envelope)
                newly_poisoned += 1

        return newly_poisoned

    if args.sync_action == "status":
        httpx = get_http_client()

        # Get local status from storage
        pending_count = k._storage.get_pending_sync_count()
        dead_letter_count = k._storage.get_dead_letter_count()
        last_sync = k._storage.get_last_sync_time()
        is_online = k._storage.is_online()
        pull_poison_pending = len(_load_pull_poison_records())

        # Check backend connection
        backend_connected, connection_msg = check_backend_connection(httpx)

        # Get namespaced agent ID for display
        local_project = get_local_project_name()
        namespaced_id = get_namespaced_stack_id()

        if args.json:
            status_data = {
                "version": 1,
                "local_stack_id": local_project,
                "namespaced_stack_id": namespaced_id if user_id else None,
                "user_id": user_id,
                "pending_operations": pending_count,
                "dead_letter_count": dead_letter_count,
                "pull_recovery_queue": pull_poison_pending,
                "last_sync_time": format_datetime(last_sync),
                "local_storage_online": is_online,
                "backend_url": backend_url or "(not configured)",
                "backend_connected": backend_connected,
                "connection_status": connection_msg,
                "authenticated": bool(auth_token),
            }
            print(json.dumps(status_data, indent=2, default=str))
        else:
            print("Sync Status")
            print("=" * 50)
            print()

            # Agent/Project info
            print(f"📦 Local project: {local_project}")
            if user_id and backend_connected:
                print(f"   Synced as: {namespaced_id}")
            elif user_id:
                print(f"   Will sync as: {namespaced_id}")
            print()

            # Connection status
            conn_icon = "🟢" if backend_connected else "🔴"
            print(f"{conn_icon} Backend: {connection_msg}")
            if backend_url:
                print(f"   URL: {backend_url}")
            if user_id:
                print(f"   User: {user_id}")
            print()

            # Pending operations
            pending_icon = "🟢" if pending_count == 0 else "🟡" if pending_count < 10 else "🟠"
            print(f"{pending_icon} Pending operations: {pending_count}")

            # Dead-lettered operations
            if dead_letter_count > 0:
                print(f"🔴 Dead-lettered: {dead_letter_count}")
                print("   These entries failed permanently and will be skipped during sync.")

            # Last sync time
            if last_sync:
                now = datetime.now(timezone.utc)
                if hasattr(last_sync, "tzinfo") and last_sync.tzinfo is None:
                    from datetime import timezone as tz

                    last_sync = last_sync.replace(tzinfo=tz.utc)
                elapsed = now - last_sync
                if elapsed.total_seconds() < 60:
                    elapsed_str = "just now"
                elif elapsed.total_seconds() < 3600:
                    elapsed_str = f"{int(elapsed.total_seconds() / 60)} minutes ago"
                elif elapsed.total_seconds() < 86400:
                    elapsed_str = f"{int(elapsed.total_seconds() / 3600)} hours ago"
                else:
                    elapsed_str = f"{int(elapsed.total_seconds() / 86400)} days ago"
                print(f"🕐 Last sync: {elapsed_str}")
                print(f"   ({last_sync.isoformat()[:19]})")
            else:
                print("🕐 Last sync: Never")

            # Suggestions
            print()
            if pending_count > 0 and backend_connected:
                print("💡 Run `kernle sync push` to upload pending changes")
            elif not backend_connected and not auth_token:
                print("💡 Run `kernle auth login` to authenticate")
            elif not backend_connected:
                print("💡 Check backend connection or run `kernle auth login`")

            if pull_poison_pending > 0:
                print(f"⚠️  Pull recovery queue: {pull_poison_pending} operation(s) pending retry")
                print("   Run `kernle sync pull` to recover pending operations")

    elif args.sync_action == "push":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        # Get pending changes from storage
        queued_changes = k._storage.get_queued_changes(limit=args.limit)

        if not queued_changes:
            print("✓ No pending changes to push")
            return

        print(f"Pushing {len(queued_changes)} changes to backend...")
        operations, skipped_orphans, op_identity_to_change = _build_push_operations(queued_changes)

        if skipped_orphans > 0:
            print(f"⚠️  Skipped {skipped_orphans} orphaned entries (source records deleted)")

        # Send to backend
        # Include stack_id as just the local project name
        # Backend will namespace it with the authenticated user_id
        try:
            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/push",
                headers=get_headers(),
                json={
                    "stack_id": local_project,  # Local name only, backend namespaces
                    "operations": operations,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                conflicts = result.get("conflicts", [])
                normalized_conflicts = [
                    _normalize_push_conflict(conflict, operations) for conflict in conflicts
                ]
                normalized_conflicts = _sort_conflicts(normalized_conflicts)
                acknowledged_changes = _resolve_acked_changes(
                    result, operations, op_identity_to_change
                )
                synced = len(acknowledged_changes)

                if conflicts:
                    for envelope in normalized_conflicts:
                        _save_push_apply_conflict(envelope)

                # Clear synced items from local queue
                with k._storage._connect() as conn:
                    for change in acknowledged_changes:
                        k._storage._clear_queued_change(conn, change.id)
                        k._storage._mark_synced(conn, change.table_name, change.record_id)
                    conn.commit()
                result["synced"] = synced

                if args.json:
                    result["version"] = 1
                    result["local_project"] = local_project
                    result["namespaced_id"] = get_namespaced_stack_id()
                    result["conflicts"] = normalized_conflicts
                    result["conflict_snapshot"] = _conflict_snapshot(normalized_conflicts)
                    print(json.dumps(result, indent=2, default=str))
                else:
                    namespaced = get_namespaced_stack_id()
                    print(f"✓ Pushed {synced} changes")
                    if user_id:
                        print(f"  Synced as: {namespaced}")
                if not args.json and conflicts:
                    print(f"⚠️  {len(conflicts)} conflicts:")
                    for c in normalized_conflicts[:5]:
                        print(
                            "   - {table}:{record_id} {op}: {error} "
                            "(payload_sha256={hash_prefix}...)".format(
                                table=c.get("table", "unknown"),
                                record_id=c.get("record_id", "unknown"),
                                op=c.get("operation", "unknown"),
                                error=c.get("error", "unknown error"),
                                hash_prefix=str(c.get("payload_hash", ""))[:16],
                            )
                        )
                    print("   ℹ️  Re-run sync after resolving server-side conflicts")
            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            else:
                print(f"✗ Push failed: {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except Exception as e:
            logger.warning("Push failed: %s", e, exc_info=True)
            print(f"✗ Push failed: {e}")
            print("  Tip: changes are queued locally and will be pushed on next `kernle sync push`")
            sys.exit(1)

    elif args.sync_action == "pull":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        # Get last sync time for incremental pull
        since = k._storage.get_last_sync_time() if not args.full else None

        print(f"Pulling changes from backend{' (full)' if args.full else ''}...")

        try:
            # Include stack_id - backend will namespace with user_id
            request_data = {
                "stack_id": local_project,  # Local name only, backend namespaces
            }
            if since and not args.full:
                request_data["since"] = format_datetime(since)

            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/pull",
                headers=get_headers(),
                json=request_data,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                operations = result.get("operations", [])
                has_more = result.get("has_more", False)

                poison_records = _load_pull_poison_records()
                _, recovered_from_poison = _retry_poisoned_pull_operations(poison_records)
                newly_poisoned = 0

                if operations:
                    # Apply fresh operations locally.
                    applied_now, failed_ops, applied_keys = _apply_pull_operations(operations)
                    for key in applied_keys:
                        poison_records.pop(key, None)
                    newly_poisoned = _quarantine_failed_pull_operations(failed_ops, poison_records)
                    k._storage._set_sync_meta("last_sync_time", k._storage._now())
                else:
                    applied_now = 0
                    failed_ops = []

                _save_pull_poison_records(poison_records)

                applied = applied_now + recovered_from_poison
                conflicts = len(failed_ops)
                poison_pending = len(poison_records)
                if conflicts or recovered_from_poison > 0:
                    logger.warning(
                        "sync pull partial completion: pulled=%s conflicts=%s recovered=%s pending=%s",
                        applied,
                        conflicts,
                        recovered_from_poison,
                        poison_pending,
                    )

                if not operations and applied == 0:
                    print("✓ Already up to date")
                    if poison_pending > 0:
                        print(f"⚠️  {poison_pending} pull operations remain quarantined for retry")
                        print("   Run `kernle sync pull` again to recover pending operations")
                    return

                if args.json:
                    serialized_failed_ops = []
                    for failed in failed_ops:
                        serialized_failed_ops.append(
                            {
                                "operation": failed.get("operation", {}),
                                "error": failed.get("error"),
                                "envelope": failed.get("envelope"),
                            }
                        )
                    serialized_failed_ops = _sort_conflicts(serialized_failed_ops)
                    conflict_envelopes = [f.get("envelope") for f in serialized_failed_ops]
                    print(
                        json.dumps(
                            {
                                "version": 1,
                                "pulled": applied,
                                "conflicts": conflicts,
                                "conflict_envelopes": conflict_envelopes,
                                "conflict_snapshot": _conflict_snapshot(conflict_envelopes),
                                "has_more": has_more,
                                "poisoned_new": newly_poisoned,
                                "poisoned_pending": poison_pending,
                                "recovered_from_quarantine": recovered_from_poison,
                                "recovery_state": (
                                    "partial" if (conflicts or recovered_from_poison > 0) else "ok"
                                ),
                                "failed_operations": serialized_failed_ops,
                                "local_project": local_project,
                                "namespaced_id": get_namespaced_stack_id(),
                            },
                            indent=2,
                        )
                    )
                else:
                    print(f"✓ Pulled {applied} changes")
                    if user_id:
                        print(f"  From: {get_namespaced_stack_id()}")
                    if recovered_from_poison:
                        print(f"   ✅ {recovered_from_poison} quarantined operations recovered")
                    if conflicts > 0:
                        print(f"⚠️  {conflicts} conflicts during apply")
                        for failed in _sort_conflicts(failed_ops)[:5]:
                            envelope = failed.get("envelope") or {}
                            hash_value = envelope.get("payload_hash", "")
                            print(
                                "  - {table}:{record_id} {op}: {error} "
                                "(payload_sha256={hash_prefix}...)".format(
                                    table=failed.get("operation", {}).get("table", "unknown"),
                                    record_id=failed.get("operation", {}).get(
                                        "record_id", "unknown"
                                    ),
                                    op=failed.get("operation", {}).get("operation", "unknown"),
                                    error=failed.get("error", "unknown error"),
                                    hash_prefix=str(hash_value)[:16],
                                )
                            )
                    if poison_pending > 0:
                        print(f"⚠️  {poison_pending} pull operations quarantined for retry")
                        print("   Run `kernle sync pull` again to retry recoverable operations")
                    elif conflicts > 0:
                        print("   ℹ️  Conflicts are recoverable and can be retried on next pull")
                    if has_more:
                        print("ℹ️  More changes available - run `kernle sync pull` again")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            else:
                print(f"✗ Pull failed: {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except Exception as e:
            logger.warning("Pull failed: %s", e, exc_info=True)
            print(f"✗ Pull failed: {e}")
            print("  Tip: check network connectivity, then retry with `kernle sync pull`")
            sys.exit(1)

    elif args.sync_action == "full":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        print("Running full bidirectional sync...")
        if user_id:
            print(f"  Syncing as: {get_namespaced_stack_id()}")
        print()

        # Step 1: Pull first (to get remote changes)
        print("Step 1: Pulling remote changes...")
        pulled = 0
        pull_conflicts = 0
        pulled_failed_ops = []
        pull_has_more = False
        pull_conflict_envelopes = []
        pull_newly_poisoned = 0
        pull_recovered_from_poison = 0
        pull_poison_pending = 0
        try:
            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/pull",
                headers=get_headers(),
                json={
                    "stack_id": local_project,
                    "since": format_datetime(k._storage.get_last_sync_time()),
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                pulled_ops = result.get("operations", [])
                pull_has_more = result.get("has_more", False)

                poison_records = _load_pull_poison_records()
                _, recovered_from_poison = _retry_poisoned_pull_operations(poison_records)
                pull_recovered_from_poison = recovered_from_poison
                pulled_now = 0
                failed_ops = []
                newly_poisoned = 0

                if pulled_ops:
                    pulled_now, failed_ops, applied_keys = _apply_pull_operations(pulled_ops)
                    for key in applied_keys:
                        poison_records.pop(key, None)
                    newly_poisoned = _quarantine_failed_pull_operations(failed_ops, poison_records)
                    k._storage._set_sync_meta("last_sync_time", k._storage._now())

                pulled = pulled_now + recovered_from_poison
                pull_conflicts = len(failed_ops)
                pulled_failed_ops = failed_ops
                pull_conflict_envelopes = [f.get("envelope") for f in failed_ops]
                pull_newly_poisoned = newly_poisoned
                pull_poison_pending = len(poison_records)
                _save_pull_poison_records(poison_records)
            else:
                print(f"  ⚠️  Pull returned status {response.status_code}")
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            print(f"  ⚠️  Pull failed: {e}")

        print(f"  ✓ Pulled {pulled} changes")
        if pull_conflicts > 0 or pull_recovered_from_poison > 0:
            logger.warning(
                "sync full pull phase partial completion: pulled=%s conflicts=%s recovered=%s pending=%s",
                pulled,
                pull_conflicts,
                pull_recovered_from_poison,
                pull_poison_pending,
            )
        if pull_newly_poisoned > 0:
            print(f"  ⚠️  {pull_newly_poisoned} new pull conflicts were quarantined")
        if pull_recovered_from_poison > 0:
            print(f"  ✅ {pull_recovered_from_poison} quarantined operations recovered this pass")
        if pull_conflicts > 0:
            print(f"  ⚠️  {pull_conflicts} conflicts during apply")
            for failed in _sort_conflicts(pulled_failed_ops)[:5]:
                envelope = failed.get("envelope") or {}
                print(
                    "  - {table}:{record_id} {op}: {error} "
                    "(payload_sha256={hash_prefix}...)".format(
                        table=failed.get("operation", {}).get("table", "unknown"),
                        record_id=failed.get("operation", {}).get("record_id", "unknown"),
                        op=failed.get("operation", {}).get("operation", "unknown"),
                        error=failed.get("error", "unknown error"),
                        hash_prefix=str(envelope.get("payload_hash", ""))[:16],
                    )
                )
        if pull_has_more:
            print("  ℹ️  More changes available - run `kernle sync pull` again")
        if pull_conflict_envelopes:
            print("  ℹ️  Re-run `kernle sync pull` to retry quarantined pull operations")
        if pull_poison_pending > 0:
            print(f"  ⚠️  {pull_poison_pending} pull operations still pending recovery")

        # Step 2: Push local changes
        print("Step 2: Pushing local changes...")
        queued_changes = k._storage.get_queued_changes(limit=1000)

        if not queued_changes:
            print("  ✓ No pending changes to push")
            print(
                f"  {pull_conflict_envelopes and '✅ Pull complete with conflicts' or '✅ Full sync complete'}"
            )
            print()
            print("✓ Full sync complete")
            remaining = k._storage.get_pending_sync_count()
            if remaining > 0:
                print(f"ℹ️  {remaining} operations still pending")
            return

        operations, skipped_orphans, op_identity_to_change = _build_push_operations(queued_changes)

        if skipped_orphans > 0:
            print(f"  ⚠️  Skipped {skipped_orphans} orphaned entries (source records deleted)")

        try:
            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/push",
                headers=get_headers(),
                json={
                    "stack_id": local_project,  # Local name only, backend namespaces
                    "operations": operations,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                conflicts = result.get("conflicts", [])
                normalized_conflicts = [
                    _normalize_push_conflict(conflict, operations) for conflict in conflicts
                ]
                normalized_conflicts = _sort_conflicts(normalized_conflicts)
                acknowledged_changes = _resolve_acked_changes(
                    result, operations, op_identity_to_change
                )
                synced = len(acknowledged_changes)
                if conflicts:
                    for envelope in normalized_conflicts:
                        _save_push_apply_conflict(envelope)
                # Clear synced items from local queue
                with k._storage._connect() as conn:
                    for change in acknowledged_changes:
                        k._storage._clear_queued_change(conn, change.id)
                        k._storage._mark_synced(conn, change.table_name, change.record_id)
                    conn.commit()
                print(f"  ✓ Pushed {synced} changes")
                if normalized_conflicts:
                    logger.warning(
                        "sync full push partial completion: synced=%s conflicts=%s",
                        synced,
                        len(normalized_conflicts),
                    )
                if normalized_conflicts:
                    print(f"  ⚠️  {len(normalized_conflicts)} conflicts:")
                    for c in normalized_conflicts[:5]:
                        print(
                            "   - {table}:{record_id} {op}: {error} "
                            "(payload_sha256={hash_prefix}...)".format(
                                table=c.get("table", "unknown"),
                                record_id=c.get("record_id", "unknown"),
                                op=c.get("operation", "unknown"),
                                error=c.get("error", "unknown error"),
                                hash_prefix=str(c.get("payload_hash", ""))[:16],
                            )
                        )
                    print("  ℹ️  Run `kernle sync full` again after resolving push conflicts")
            else:
                print(f"  ⚠️  Push returned status {response.status_code}")
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            print(f"  ⚠️  Push failed: {e}")

        # Show final status
        print()
        print("✓ Full sync complete")
        print(f"  pulled conflict envelopes: {len(pull_conflict_envelopes)}")
        if pull_conflict_envelopes:
            print(
                f"  pull conflict envelopes (deterministic): {len(_sort_conflicts(pull_conflict_envelopes))}"
            )
        if pull_poison_pending > 0:
            print(f"  ⚠️  {pull_poison_pending} pull operations pending recovery")
        if pull_newly_poisoned > 0:
            print(f"  ⚠️  {pull_newly_poisoned} pull conflicts were quarantined")
        if not pull_conflict_envelopes and not pull_newly_poisoned and not pull_poison_pending:
            print("  ✅ Pull phase had no quarantined conflicts")

        remaining = k._storage.get_pending_sync_count()
        if remaining > 0:
            print(f"ℹ️  {remaining} operations still pending")

    elif args.sync_action == "conflicts":
        # Get conflict history from storage
        if args.clear:
            cleared = k._storage.clear_sync_conflicts()
            if args.json:
                print(json.dumps({"version": 1, "cleared": cleared}))
            else:
                print(f"✓ Cleared {cleared} conflict records")
            return

        conflicts = k._storage.get_sync_conflicts(limit=args.limit)

        if args.json:
            conflict_data = []
            for c in conflicts:
                conflict_data.append(
                    {
                        "id": c.id,
                        "table": c.table,
                        "record_id": c.record_id,
                        "resolution": c.resolution,
                        "resolved_at": c.resolved_at.isoformat() if c.resolved_at else None,
                        "local_summary": c.local_summary,
                        "cloud_summary": c.cloud_summary,
                    }
                )
            print(
                json.dumps(
                    {"version": 1, "conflicts": conflict_data, "count": len(conflicts)}, indent=2
                )
            )
        else:
            if not conflicts:
                print("No sync conflicts in history")
                print("  Conflicts are recorded when local and cloud versions differ during sync")
                return

            print(f"Sync Conflict History ({len(conflicts)} conflicts)")
            print()

            for c in conflicts:
                if c.resolution == "cloud_wins":
                    resolution_icon = "↓"
                    resolution_text = "cloud wins"
                elif c.resolution == "local_wins":
                    resolution_icon = "↑"
                    resolution_text = "local wins"
                elif c.resolution == "pull_apply_failed":
                    resolution_icon = "!"
                    resolution_text = "pull apply failed"
                else:
                    resolution_icon = "?"
                    resolution_text = c.resolution
                when = c.resolved_at.strftime("%Y-%m-%d %H:%M") if c.resolved_at else "unknown"

                print(f"{resolution_icon} {c.table}:{c.record_id[:8]}... ({resolution_text})")
                print(f"  Resolved: {when}")
                if c.local_summary:
                    print(f'  Local:  "{c.local_summary}"')
                if c.cloud_summary:
                    print(f'  Cloud:  "{c.cloud_summary}"')
                print()

            print("💡 Use `kernle sync conflicts --clear` to clear history")
