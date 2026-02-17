"""Import command for migrating flat files to Kernle.

Supports importing from:
- Markdown files (.md, .markdown, .txt)
- JSON files (Kernle export format)
- CSV files (tabular format)
- PDF files (.pdf)
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from kernle.dedup import load_raw_content_hashes, strip_corpus_header
from kernle.processing import compute_content_hash

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle

logger = logging.getLogger(__name__)

_DUPLICATE_SCAN_LIMIT = 100_000
_IMPORT_FINGERPRINT_SCHEME = "context:import-fingerprint:v1"


def _validate_import_numeric(
    value: Any,
    min_val: float,
    max_val: float,
    default: float,
    *,
    strict: bool = False,
) -> Tuple[float, bool]:
    """Validate and clamp a numeric field for import.

    Returns:
        (validated_value, rejected) — rejected=True means skip in strict mode.
    """
    if value is None:
        return default, False
    if isinstance(value, bool):
        return default, strict
    try:
        fval = float(value)
    except (ValueError, TypeError):
        return default, strict
    if math.isnan(fval) or math.isinf(fval):
        return default, strict
    if fval < min_val or fval > max_val:
        if strict:
            return fval, True
        return max(min_val, min(max_val, fval)), False
    return fval, False


def _normalize_text(value: Any) -> str:
    """Normalize text for deterministic hashing."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_list(value: Any) -> List[str]:
    """Normalize list-like values for deterministic comparisons."""
    if not value:
        return []
    if isinstance(value, list):
        items = [str(item).strip().lower() for item in value]
    else:
        items = [str(value).strip().lower()]
    return sorted([item for item in items if item])


def _fingerprint_ref_prefix() -> str:
    return f"{_IMPORT_FINGERPRINT_SCHEME}:"


def _normalize_derived_from(derived_from: Optional[List[str]]) -> List[str]:
    """Normalize derived_from refs for stable storage and dedupe."""
    if not derived_from:
        return []

    cleaned: List[str] = []
    for ref in derived_from:
        if not isinstance(ref, str):
            continue
        ref = ref.strip()
        if not ref:
            continue
        if ref not in cleaned:
            cleaned.append(ref)
    return cleaned


def _build_import_fingerprint(item: Dict[str, Any]) -> Optional[str]:
    """Build a stable fingerprint for imported import payloads."""
    item_type = item.get("type")
    if not item_type:
        return None

    payload: Dict[str, Any] = {"type": item_type}
    if item_type == "episode":
        objective = _normalize_text(item.get("objective"))
        outcome = _normalize_text(item.get("outcome", item.get("objective")))
        lessons = _normalize_list(item.get("lesson") or item.get("lessons"))
        payload.update({"objective": objective, "outcome": outcome, "lessons": lessons})
    elif item_type == "note":
        content = _normalize_text(item.get("content"))
        note_type = _normalize_text(item.get("note_type") or item.get("type"))
        speaker = _normalize_text(item.get("speaker"))
        payload.update(
            {
                "content": content,
                "note_type": note_type,
                "speaker": speaker,
            }
        )
    elif item_type == "belief":
        statement = _normalize_text(item.get("statement"))
        payload.update({"statement": statement})
    elif item_type == "value":
        name = _normalize_text(item.get("name"))
        description = _normalize_text(item.get("description"))
        payload.update({"name": name, "description": description})
    elif item_type == "goal":
        title = _normalize_text(item.get("title"))
        description = _normalize_text(item.get("description", item.get("title")))
        status = _normalize_text(item.get("status"))
        payload.update(
            {
                "title": title,
                "description": description,
                "status": status,
            }
        )
    elif item_type == "raw":
        payload.update({"content": _normalize_text(item.get("content"))})
    elif item_type == "drive":
        drive_type = _normalize_text(item.get("drive_type"))
        intensity = item.get("intensity")
        focus = _normalize_list(item.get("focus_areas"))
        payload.update({"drive_type": drive_type, "intensity": intensity, "focus_areas": focus})
    elif item_type == "relationship":
        entity_name = _normalize_text(item.get("entity_name"))
        entity_type = _normalize_text(item.get("entity_type"))
        relationship_type = _normalize_text(item.get("relationship_type"))
        sentiment = item.get("sentiment")
        payload.update(
            {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "relationship_type": relationship_type,
                "sentiment": sentiment,
            }
        )
    else:
        return None

    if all(v in (None, "", [], {}) for v in payload.values()):
        return None

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{_fingerprint_ref_prefix()}{compute_content_hash(canonical)}"


def _extract_import_fingerprints(derived_from: Optional[List[str]]) -> List[str]:
    """Return all import fingerprint refs from a derived_from list."""
    prefix = _fingerprint_ref_prefix()
    result = []
    for ref in derived_from or []:
        if not isinstance(ref, str):
            continue
        if ref.startswith(prefix):
            result.append(ref)
    return result


def _build_import_fingerprint_index(k: "Kernle") -> Dict[str, set]:
    """Build a cache of existing import fingerprints for quick duplicate checks."""
    index: Dict[str, set] = {
        "episode": set(),
        "note": set(),
        "belief": set(),
        "value": set(),
        "goal": set(),
        "raw": set(),
        "drive": set(),
        "relationship": set(),
    }

    for belief in k._storage.get_beliefs(limit=_DUPLICATE_SCAN_LIMIT):
        index["belief"].update(_extract_import_fingerprints(getattr(belief, "derived_from", None)))
    for value in k._storage.get_values(limit=_DUPLICATE_SCAN_LIMIT):
        index["value"].update(_extract_import_fingerprints(getattr(value, "derived_from", None)))
    for goal in k._storage.get_goals(status=None, limit=_DUPLICATE_SCAN_LIMIT):
        index["goal"].update(_extract_import_fingerprints(getattr(goal, "derived_from", None)))
    for episode in k._storage.get_episodes(limit=_DUPLICATE_SCAN_LIMIT):
        index["episode"].update(
            _extract_import_fingerprints(getattr(episode, "derived_from", None))
        )
    for note in k._storage.get_notes(limit=_DUPLICATE_SCAN_LIMIT):
        index["note"].update(_extract_import_fingerprints(getattr(note, "derived_from", None)))
    for drive in k._storage.get_drives():
        index["drive"].update(_extract_import_fingerprints(getattr(drive, "derived_from", None)))
    for relationship in k._storage.get_relationships():
        index["relationship"].update(
            _extract_import_fingerprints(getattr(relationship, "derived_from", None))
        )

    raw_hashes = load_raw_content_hashes(k._storage, limit=_DUPLICATE_SCAN_LIMIT)
    index["raw"].update(raw_hashes.hashes)

    return index


def _seen_signature_buckets() -> Dict[str, set]:
    return {
        key: set()
        for key in ("episode", "note", "belief", "value", "goal", "raw", "drive", "relationship")
    }


def _item_signature(item: Dict[str, Any]) -> Optional[str]:
    if item.get("type") == "raw":
        content = strip_corpus_header(item.get("content", ""))
        content = _normalize_text(content)
        return compute_content_hash(content) if content else None
    return _build_import_fingerprint(item)


def _register_seen_signature(item: Dict[str, Any], seen: Dict[str, set]) -> None:
    signature = _item_signature(item)
    if not signature:
        return
    item_type = item.get("type")
    if item_type in seen:
        seen[item_type].add(signature)


def _merge_derived_from(
    derived_from: Optional[List[str]],
    item: Dict[str, Any],
) -> Optional[List[str]]:
    """Merge CLI-provided derived_from with import fingerprint.

    Appends the item's import fingerprint to the derived_from list so every
    imported record carries a ``context:import-fingerprint:v1:*`` provenance
    reference.  This enables duplicate suppression on repeated imports and
    preserves the signal that a record is import-derived.
    """
    signatures = _normalize_derived_from(derived_from)
    fingerprint = _item_signature(item)
    if not fingerprint:
        return signatures or None
    if fingerprint not in signatures:
        signatures.append(fingerprint)
    return signatures or None


def _check_duplicate(
    item: Dict[str, Any],
    k: "Kernle",
    *,
    seen_signatures: Optional[Dict[str, set]] = None,
    existing_fingerprints: Optional[Dict[str, set]] = None,
) -> bool:
    """Check if an item already exists.

    We first use import fingerprints when available, then fall back to legacy
    heuristics for records that predate fingerprint tracking.
    """
    item_type = item.get("type")
    if not item_type:
        return False

    signature = _item_signature(item)
    has_existing_fingerprints = existing_fingerprints is not None
    seen_signatures = seen_signatures or {}
    existing_fingerprints = existing_fingerprints or {}

    if item_type in ("episode", "note", "belief", "value", "goal", "raw", "drive", "relationship"):
        if signature:
            if signature in seen_signatures.get(item_type, set()):
                return True
            if signature in existing_fingerprints.get(item_type, set()):
                return True

        # Fallback legacy duplicate checks for non-fingerprint entries.
        if item_type == "belief":
            statement = item.get("statement", "")
            if k._storage.find_belief(statement):
                return True
        elif item_type == "value":
            name = item.get("name", "")
            existing = k._storage.get_values(limit=_DUPLICATE_SCAN_LIMIT)
            if any(v.name == name for v in existing):
                return True
        elif item_type == "goal":
            desc = item.get("description", "")
            title = item.get("title", "")
            existing = k._storage.get_goals(status=None, limit=_DUPLICATE_SCAN_LIMIT)
            if any(g.title == title or g.description == desc for g in existing):
                return True
        elif item_type == "episode":
            objective = item.get("objective", "")
            existing = k._storage.get_episodes(limit=_DUPLICATE_SCAN_LIMIT)
            if any(getattr(ep, "objective", None) == objective for ep in existing):
                return True
        elif item_type == "note":
            content = item.get("content", "")
            existing = k._storage.get_notes(limit=_DUPLICATE_SCAN_LIMIT)
            if any(getattr(n, "content", None) == content for n in existing):
                return True
        elif item_type == "raw":
            content = item.get("content", "")
            if not content:
                return False
            h = _item_signature({"type": "raw", "content": content})
            if not h:
                return False
            if h in existing_fingerprints.get("raw", set()):
                return True
            if not has_existing_fingerprints:
                raw_hashes = load_raw_content_hashes(k._storage, limit=_DUPLICATE_SCAN_LIMIT)
                if h in raw_hashes.hashes:
                    return True
        elif item_type == "drive":
            drive_type = item.get("drive_type", "")
            if k._storage.get_drive(drive_type):
                return True
        elif item_type == "relationship":
            entity_name = item.get("entity_name", "")
            if k._storage.get_relationship(entity_name):
                return True

    return False


def cmd_import(args: "argparse.Namespace", k: "Kernle") -> None:
    """Import memories from external files.

    Supports markdown, JSON (Kernle export format), and CSV files.
    Auto-detects format from file extension, or use --format to specify.
    """
    file_path = Path(args.file).expanduser()

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    # Determine format
    file_format = getattr(args, "format", None)
    if not file_format:
        # Auto-detect from extension
        suffix = file_path.suffix.lower()
        if suffix in (".md", ".markdown", ".txt"):
            file_format = "markdown"
        elif suffix == ".json":
            file_format = "json"
        elif suffix == ".csv":
            file_format = "csv"
        elif suffix == ".pdf":
            file_format = "pdf"
        else:
            print(f"Error: Unknown file format: {suffix}")
            print("Use --format to specify: markdown, json, csv, or pdf")
            return

    dry_run = getattr(args, "dry_run", False)
    interactive = getattr(args, "interactive", False)
    target_layer = getattr(args, "layer", None)
    skip_duplicates = getattr(args, "skip_duplicates", True)
    derived_from = getattr(args, "derived_from", None)
    strict = getattr(args, "strict", False)

    chunk_size = getattr(args, "chunk_size", 2000)

    if file_format == "markdown":
        _import_markdown(file_path, k, dry_run, interactive, target_layer, derived_from)
    elif file_format == "json":
        _import_json(file_path, k, dry_run, skip_duplicates, derived_from, strict=strict)
    elif file_format == "csv":
        _import_csv(
            file_path, k, dry_run, target_layer, skip_duplicates, derived_from, strict=strict
        )
    elif file_format == "pdf":
        _import_pdf(file_path, k, dry_run, skip_duplicates, derived_from, chunk_size)


def _import_markdown(
    file_path: Path,
    k: "Kernle",
    dry_run: bool,
    interactive: bool,
    target_layer: Optional[str],
    derived_from: Optional[List[str]] = None,
) -> None:
    """Import from a markdown file."""
    content = file_path.read_text(encoding="utf-8")

    # Parse the content
    items = _parse_markdown(content)

    if not items:
        print("No importable content found in file")
        print("\nExpected formats:")
        print("  ## Episodes / ## Lessons - for episode entries")
        print("  ## Decisions / ## Notes - for note entries")
        print("  ## Beliefs - for belief entries")
        print("  ## Values / ## Principles - for value entries")
        print("  ## Goals / ## Tasks - for goal entries")
        print("  ## Raw / ## Thoughts - for raw entries")
        print("  Freeform paragraphs - imported as raw entries")
        return

    # If layer specified, override detected types
    if target_layer:
        for item in items:
            item["type"] = target_layer

    # Show what we found
    type_counts: Dict[str, int] = {}
    for item in items:
        t = item["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"Found {len(items)} items to import:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()

    if dry_run:
        print("=== DRY RUN (no changes made) ===\n")
        for i, item in enumerate(items, 1):
            _preview_item(i, item)
        return

    if interactive:
        _interactive_import(items, k, derived_from)
    else:
        _batch_import(items, k, derived_from=derived_from)


def _import_json(
    file_path: Path,
    k: "Kernle",
    dry_run: bool,
    skip_duplicates: bool,
    derived_from: Optional[List[str]] = None,
    *,
    strict: bool = False,
) -> None:
    """Import from a Kernle JSON export file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return

    # Validate it looks like a Kernle export
    if not isinstance(data, dict):
        print("Error: JSON must be an object at the root level")
        return

    source_agent = data.get("stack_id", "unknown")
    exported_at = data.get("exported_at", "unknown")

    print(
        f"Importing from agent '{source_agent}' (exported {exported_at[:10] if len(exported_at) > 10 else exported_at})"
    )
    print()

    # Count items by type
    type_counts: Dict[str, int] = {}
    for memory_type in [
        "values",
        "beliefs",
        "goals",
        "episodes",
        "notes",
        "drives",
        "relationships",
        "raw_entries",
    ]:
        items = data.get(memory_type, [])
        if items:
            # Normalize type name
            normalized = memory_type.rstrip("s") if memory_type != "raw_entries" else "raw"
            if normalized == "raw_entrie":
                normalized = "raw"
            type_counts[normalized] = len(items)

    if not type_counts:
        print("No importable content found in JSON file")
        return

    print(f"Found {sum(type_counts.values())} items to import:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()

    if dry_run:
        print("=== DRY RUN (no changes made) ===")
        return

    # Import each type
    imported: Dict[str, int] = {}
    skipped: Dict[str, int] = {}
    errors: List[str] = []
    existing_fingerprints = _build_import_fingerprint_index(k) if skip_duplicates else None
    seen_signatures = _seen_signature_buckets()

    # Values
    for item in data.get("values", []):
        priority_val, rejected = _validate_import_numeric(
            item.get("priority", 50), 0, 100, 50, strict=strict
        )
        if rejected:
            errors.append(f"value: priority out of range: {item.get('priority')}")
            continue
        import_item = {
            "type": "value",
            "name": item.get("name", ""),
            "description": item.get("statement", item.get("description", "")),
            "priority": int(priority_val),
        }
        import_derived_from = _merge_derived_from(derived_from, import_item)
        try:
            if skip_duplicates and _check_duplicate(
                import_item,
                k,
                seen_signatures=seen_signatures,
                existing_fingerprints=existing_fingerprints,
            ):
                skipped["value"] = skipped.get("value", 0) + 1
                continue

            k.value(
                name=import_item["name"],
                description=import_item["description"],
                priority=import_item["priority"],
                derived_from=import_derived_from,
            )
            imported["value"] = imported.get("value", 0) + 1
            if skip_duplicates:
                _register_seen_signature(import_item, seen_signatures)
        except Exception as e:
            logger.debug("Import value failed: %s", e, exc_info=True)
            errors.append(f"value: {str(e)[:50]}")

    # Beliefs
    for item in data.get("beliefs", []):
        conf_val, rejected = _validate_import_numeric(
            item.get("confidence", 0.8), 0.0, 1.0, 0.8, strict=strict
        )
        if rejected:
            errors.append(f"belief: confidence out of range: {item.get('confidence')}")
            continue
        import_item = {
            "type": "belief",
            "statement": item.get("statement", ""),
            "belief_type": item.get("type", "fact"),
            "confidence": conf_val,
        }
        import_derived_from = _merge_derived_from(derived_from, import_item)
        try:
            if skip_duplicates and _check_duplicate(
                import_item,
                k,
                seen_signatures=seen_signatures,
                existing_fingerprints=existing_fingerprints,
            ):
                skipped["belief"] = skipped.get("belief", 0) + 1
                continue

            k.belief(
                statement=import_item["statement"],
                type=import_item["belief_type"],
                confidence=import_item["confidence"],
                derived_from=import_derived_from,
            )
            imported["belief"] = imported.get("belief", 0) + 1
            if skip_duplicates:
                _register_seen_signature(import_item, seen_signatures)
        except Exception as e:
            logger.debug("Import belief failed: %s", e, exc_info=True)
            errors.append(f"belief: {str(e)[:50]}")

    # Goals
    for item in data.get("goals", []):
        title = item.get("title", "")
        description = item.get("description", title)
        import_item = {
            "type": "goal",
            "title": title,
            "description": description,
            "priority": item.get("priority", "medium"),
            "status": item.get("status", "active"),
        }
        import_derived_from = _merge_derived_from(derived_from, import_item)
        try:
            if skip_duplicates:
                if _check_duplicate(
                    import_item,
                    k,
                    seen_signatures=seen_signatures,
                    existing_fingerprints=existing_fingerprints,
                ):
                    skipped["goal"] = skipped.get("goal", 0) + 1
                    continue

            k.goal(
                description=import_item["description"],
                title=import_item["title"],
                priority=import_item["priority"],
                status=import_item["status"],
                derived_from=import_derived_from,
            )
            imported["goal"] = imported.get("goal", 0) + 1
            if skip_duplicates:
                _register_seen_signature(import_item, seen_signatures)
        except Exception as e:
            logger.debug("Import goal failed: %s", e, exc_info=True)
            errors.append(f"goal: {str(e)[:50]}")

    # Episodes
    for item in data.get("episodes", []):
        import_item = {
            "type": "episode",
            "objective": item.get("objective", ""),
            "outcome": item.get("outcome", item.get("objective", "")),
            "lessons": item.get("lessons"),
            "tags": item.get("tags"),
        }
        import_derived_from = _merge_derived_from(derived_from, import_item)
        try:
            if skip_duplicates and _check_duplicate(
                import_item,
                k,
                seen_signatures=seen_signatures,
                existing_fingerprints=existing_fingerprints,
            ):
                skipped["episode"] = skipped.get("episode", 0) + 1
                continue

            k.episode(
                objective=import_item["objective"],
                outcome=import_item["outcome"],
                lessons=import_item["lessons"],
                tags=import_item["tags"],
                derived_from=import_derived_from,
            )
            imported["episode"] = imported.get("episode", 0) + 1
            if skip_duplicates:
                _register_seen_signature(import_item, seen_signatures)
        except Exception as e:
            logger.debug("Import episode failed: %s", e, exc_info=True)
            errors.append(f"episode: {str(e)[:50]}")

    # Notes
    for item in data.get("notes", []):
        note_type = item.get("type", "note")
        import_item = {
            "type": "note",
            "content": item.get("content", ""),
            "note_type": note_type,
            "speaker": item.get("speaker"),
            "reason": item.get("reason"),
            "tags": item.get("tags"),
        }
        import_derived_from = _merge_derived_from(derived_from, import_item)
        try:
            if skip_duplicates and _check_duplicate(
                import_item,
                k,
                seen_signatures=seen_signatures,
                existing_fingerprints=existing_fingerprints,
            ):
                skipped["note"] = skipped.get("note", 0) + 1
                continue

            k.note(
                content=import_item["content"],
                type=import_item["note_type"],
                speaker=import_item["speaker"],
                reason=import_item["reason"],
                tags=import_item["tags"],
                derived_from=import_derived_from,
            )
            imported["note"] = imported.get("note", 0) + 1
            if skip_duplicates:
                _register_seen_signature(import_item, seen_signatures)
        except Exception as e:
            logger.debug("Import note failed: %s", e, exc_info=True)
            errors.append(f"note: {str(e)[:50]}")

    # Drives
    for item in data.get("drives", []):
        intensity_val, rejected = _validate_import_numeric(
            item.get("intensity", 0.5), 0.0, 1.0, 0.5, strict=strict
        )
        if rejected:
            errors.append(f"drive: intensity out of range: {item.get('intensity')}")
            continue
        import_item = {
            "type": "drive",
            "drive_type": item.get("drive_type", ""),
            "intensity": intensity_val,
            "focus_areas": item.get("focus_areas"),
        }
        import_derived_from = _merge_derived_from(derived_from, import_item)
        try:
            if skip_duplicates and _check_duplicate(
                import_item,
                k,
                seen_signatures=seen_signatures,
                existing_fingerprints=existing_fingerprints,
            ):
                skipped["drive"] = skipped.get("drive", 0) + 1
                continue

            k.drive(
                drive_type=import_item["drive_type"],
                intensity=import_item["intensity"],
                focus_areas=import_item["focus_areas"],
                derived_from=import_derived_from,
            )
            imported["drive"] = imported.get("drive", 0) + 1
            if skip_duplicates:
                _register_seen_signature(import_item, seen_signatures)
        except Exception as e:
            logger.debug("Import drive failed: %s", e, exc_info=True)
            errors.append(f"drive: {str(e)[:50]}")

    # Relationships
    for item in data.get("relationships", []):
        sentiment_val, rejected = _validate_import_numeric(
            item.get("sentiment", 0.0), -1.0, 1.0, 0.0, strict=strict
        )
        if rejected:
            errors.append(f"relationship: sentiment out of range: {item.get('sentiment')}")
            continue
        import_item = {
            "type": "relationship",
            "entity_name": item.get("entity_name", ""),
            "entity_type": item.get("entity_type", "unknown"),
            "relationship_type": item.get("relationship_type", "knows"),
            "sentiment": sentiment_val,
            "notes": item.get("notes"),
        }
        import_derived_from = _merge_derived_from(derived_from, import_item)
        try:
            if skip_duplicates and _check_duplicate(
                import_item,
                k,
                seen_signatures=seen_signatures,
                existing_fingerprints=existing_fingerprints,
            ):
                skipped["relationship"] = skipped.get("relationship", 0) + 1
                continue

            k.relationship(
                entity_name=import_item["entity_name"],
                entity_type=import_item["entity_type"],
                relationship_type=import_item["relationship_type"],
                sentiment=import_item["sentiment"],
                notes=import_item["notes"],
                derived_from=import_derived_from,
            )
            imported["relationship"] = imported.get("relationship", 0) + 1
            if skip_duplicates:
                _register_seen_signature(import_item, seen_signatures)
        except Exception as e:
            logger.debug("Import relationship failed: %s", e, exc_info=True)
            errors.append(f"relationship: {str(e)[:50]}")

    # Raw entries
    for item in data.get("raw_entries", []):
        import_item = {
            "type": "raw",
            "content": item.get("content", ""),
            "source": item.get("source", "import"),
        }
        try:
            if skip_duplicates and _check_duplicate(
                import_item,
                k,
                seen_signatures=seen_signatures,
                existing_fingerprints=existing_fingerprints,
            ):
                skipped["raw"] = skipped.get("raw", 0) + 1
                continue

            k.raw(blob=import_item["content"], source=import_item["source"])
            imported["raw"] = imported.get("raw", 0) + 1
            if skip_duplicates:
                _register_seen_signature(import_item, seen_signatures)
        except Exception as e:
            logger.debug("Import raw entry failed: %s", e, exc_info=True)
            errors.append(f"raw: {str(e)[:50]}")

    # Summary
    total_imported = sum(imported.values())
    total_skipped = sum(skipped.values())

    print(f"Imported {total_imported} items")
    if imported:
        for t, count in sorted(imported.items()):
            print(f"  {t}: {count}")

    if total_skipped > 0:
        print(f"\nSkipped {total_skipped} duplicates")
        for t, count in sorted(skipped.items()):
            print(f"  {t}: {count}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def _import_csv(
    file_path: Path,
    k: "Kernle",
    dry_run: bool,
    target_layer: Optional[str],
    skip_duplicates: bool,
    derived_from: Optional[List[str]] = None,
    *,
    strict: bool = False,
) -> None:
    """Import from a CSV file."""
    import csv
    import io

    content = file_path.read_text(encoding="utf-8")

    try:
        reader = csv.DictReader(io.StringIO(content))
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]
    except Exception as e:
        logger.debug("CSV parsing failed: %s", e, exc_info=True)
        print(f"Error: Invalid CSV: {e}")
        return

    if not headers:
        print("Error: CSV file has no headers")
        return

    # Check if 'type' column exists
    has_type_column = any(h in ["type", "memory_type", "kind"] for h in headers)

    if not has_type_column and not target_layer:
        print("Error: CSV must have a 'type' column or use --layer to specify memory type")
        print("Valid types: episode, note, belief, value, goal, raw")
        return

    # Parse items
    items: List[Dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(content))  # Re-read

    for row in reader:
        # Normalize keys
        row = {k.lower().strip(): v.strip() if v else "" for k, v in row.items()}

        # Determine type
        if target_layer:
            item_type = target_layer
        else:
            item_type = row.get("type") or row.get("memory_type") or row.get("kind")
            if not item_type:
                continue

        item_type = item_type.lower().strip()

        # Build item based on type
        item = {"type": item_type}

        if item_type == "episode":
            item["objective"] = row.get("objective") or row.get("title") or row.get("task", "")
            item["outcome"] = row.get("outcome") or row.get("result") or item["objective"]
            item["outcome_type"] = row.get("outcome_type") or row.get("status")
            lessons = row.get("lessons") or row.get("lesson", "")
            item["lessons"] = (
                [lesson.strip() for lesson in lessons.split(",") if lesson.strip()]
                if lessons
                else None
            )
        elif item_type == "note":
            item["content"] = row.get("content") or row.get("text") or row.get("note", "")
            item["note_type"] = row.get("note_type") or row.get("category") or "note"
            item["speaker"] = row.get("speaker") or row.get("author")
        elif item_type == "belief":
            item["statement"] = row.get("statement") or row.get("belief") or row.get("content", "")
            conf = row.get("confidence") or row.get("conf") or "0.7"
            try:
                fval = float(conf)
                if fval > 1:
                    fval = fval / 100
            except (ValueError, TypeError):
                if strict:
                    continue
                fval = 0.7
            conf_val, rejected = _validate_import_numeric(fval, 0.0, 1.0, 0.7, strict=strict)
            if rejected:
                continue
            item["confidence"] = conf_val
        elif item_type == "value":
            item["name"] = row.get("name") or row.get("value") or row.get("title", "")
            item["description"] = row.get("description") or row.get("statement") or item["name"]
            prio_val, rejected = _validate_import_numeric(
                row.get("priority", "50"), 0, 100, 50, strict=strict
            )
            if rejected:
                continue
            item["priority"] = int(prio_val)
        elif item_type == "goal":
            item["title"] = row.get("title") or row.get("goal") or row.get("name", "")
            item["description"] = row.get("description") or item["title"]
            status = row.get("status", "active").lower()
            if status in ("done", "complete", "completed", "true", "1", "yes"):
                item["status"] = "completed"
            elif status in ("paused", "hold"):
                item["status"] = "paused"
            else:
                item["status"] = "active"
        elif item_type == "raw":
            item["content"] = row.get("content") or row.get("text") or row.get("raw", "")
            item["source"] = row.get("source", "csv-import")

        # Skip empty items
        content_field = (
            item.get("content")
            or item.get("objective")
            or item.get("statement")
            or item.get("name")
            or item.get("title")
        )
        if content_field:
            items.append(item)

    if not items:
        print("No importable content found in CSV file")
        return

    # Show what we found
    type_counts: Dict[str, int] = {}
    for item in items:
        t = item["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"Found {len(items)} items to import:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()

    if dry_run:
        print("=== DRY RUN (no changes made) ===\n")
        for i, item in enumerate(items[:10], 1):
            _preview_item(i, item)
        if len(items) > 10:
            print(f"... and {len(items) - 10} more items")
        return

    # Import
    _batch_import(items, k, skip_duplicates, derived_from=derived_from)


def _import_pdf(
    file_path: Path,
    k: "Kernle",
    dry_run: bool,
    skip_duplicates: bool,
    derived_from: Optional[List[str]] = None,
    max_chunk_size: int = 2000,
) -> None:
    """Import from a PDF file by extracting text and chunking into raw entries."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        print("Error: PDF support requires pdfminer.six")
        print("Install with: pip install pdfminer.six")
        return

    try:
        text = extract_text(str(file_path))
    except Exception as e:
        logger.debug("PDF text extraction failed: %s", e, exc_info=True)
        print(f"Error extracting text from PDF: {e}")
        return

    if not text or not text.strip():
        print("No text content found in PDF")
        return

    # Chunk the extracted text using corpus chunking logic
    from kernle.corpus import chunk_generic

    chunks = chunk_generic(text, str(file_path), max_chunk_size)

    if not chunks:
        print("No chunks created from PDF content")
        return

    print(f"Extracted {len(chunks)} chunks from {file_path.name}")

    if dry_run:
        print("=== DRY RUN (no changes made) ===\n")
        for i, chunk in enumerate(chunks[:10], 1):
            preview = chunk["content"][:80].replace("\n", " ")
            print(f"{i}. [{chunk['chunk_name']}] {preview}...")
        if len(chunks) > 10:
            print(f"... and {len(chunks) - 10} more chunks")
        return

    # Import as raw entries with file metadata
    imported = 0
    skipped = 0
    errors: List[str] = []
    seen_hashes: set = set()

    # Pre-load existing content hashes for dedup (strips corpus headers
    # so PDF chunks match against the actual content of corpus entries)
    existing_hashes: set = set()
    if skip_duplicates:
        result = load_raw_content_hashes(k._storage)
        existing_hashes = result.hashes

    for chunk in chunks:
        content = chunk["content"]
        content_hash = compute_content_hash(content)

        # Dedup within this import and against existing
        if content_hash in seen_hashes or content_hash in existing_hashes:
            skipped += 1
            continue
        seen_hashes.add(content_hash)

        try:
            source = f"pdf:{file_path.name}"
            k.raw(blob=content, source=source)
            imported += 1
        except Exception as e:
            logger.debug("Import PDF chunk failed: %s", e, exc_info=True)
            errors.append(f"chunk {chunk['chunk_name']}: {str(e)[:50]}")

    print(f"Imported {imported} raw entries from PDF")
    if skipped > 0:
        print(f"Skipped {skipped} duplicates")
    if errors:
        print(f"{len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


# ============================================================================
# Markdown parsing functions (kept for backwards compatibility)
# ============================================================================


def _parse_markdown(content: str) -> List[Dict[str, Any]]:
    """Parse markdown content into importable items.

    Detects sections like:
    - ## Episodes, ## Lessons -> episode
    - ## Decisions, ## Notes, ## Insights -> note
    - ## Beliefs -> belief
    - ## Raw, ## Thoughts, ## Scratch -> raw
    - Unstructured text -> raw
    """
    items: List[Dict[str, Any]] = []

    # Split into sections by ## headers
    sections = re.split(r"^## (.+)$", content, flags=re.MULTILINE)

    # First section (before any ##) is preamble
    if sections[0].strip():
        # Check if it has bullet points or paragraphs
        preamble = sections[0].strip()
        for para in _split_paragraphs(preamble):
            if para.strip():
                items.append({"type": "raw", "content": para.strip(), "source": "preamble"})

    # Process header sections
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break

        header = sections[i].strip().lower()
        section_content = sections[i + 1].strip()

        if not section_content:
            continue

        # Determine type from header
        if any(h in header for h in ["episode", "lesson", "experience", "event"]):
            items.extend(_parse_episodes(section_content))
        elif any(h in header for h in ["decision", "note", "insight", "observation"]):
            items.extend(_parse_notes(section_content, header))
        elif "belief" in header:
            items.extend(_parse_beliefs(section_content))
        elif any(h in header for h in ["value", "principle"]):
            items.extend(_parse_values(section_content))
        elif any(h in header for h in ["goal", "objective", "todo", "task"]):
            items.extend(_parse_goals(section_content))
        elif any(h in header for h in ["raw", "thought", "scratch", "draft", "idea"]):
            items.extend(_parse_raw(section_content))
        else:
            # Unknown section - treat as raw
            items.extend(_parse_raw(section_content))

    return items


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _parse_episodes(content: str) -> List[Dict[str, Any]]:
    """Parse episode entries from section content."""
    items = []

    # Look for bullet points or numbered items
    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Try to extract lesson (after -> or "Lesson:")
        lesson = None
        if "->" in entry:
            parts = entry.split("->", 1)
            entry = parts[0].strip()
            lesson = parts[1].strip()
        elif "lesson:" in entry.lower():
            # Match (Lesson: X) or just Lesson: X
            match = re.search(r"\(lesson:\s*([^)]+)\)", entry, re.IGNORECASE)
            if match:
                lesson = match.group(1).strip()
                entry = re.sub(r"\(lesson:\s*[^)]+\)", "", entry, flags=re.IGNORECASE).strip()
            else:
                # No parentheses version
                match = re.search(r"lesson:\s*(.+)", entry, re.IGNORECASE)
                if match:
                    lesson = match.group(1).strip()
                    entry = re.sub(r"lesson:\s*.+", "", entry, flags=re.IGNORECASE).strip()

        items.append(
            {
                "type": "episode",
                "objective": entry[:200] if len(entry) > 200 else entry,
                "outcome": entry,
                "lesson": lesson,
                "source": "episodes section",
            }
        )

    return items


def _parse_notes(content: str, header: str) -> List[Dict[str, Any]]:
    """Parse note entries from section content."""
    items = []

    # Determine note type from header
    if "decision" in header:
        note_type = "decision"
    elif "insight" in header:
        note_type = "insight"
    elif "observation" in header:
        note_type = "observation"
    else:
        note_type = "note"

    # Split by bullets or paragraphs
    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        items.append(
            {
                "type": "note",
                "content": entry,
                "note_type": note_type,
                "source": f"{header} section",
            }
        )

    return items


def _parse_beliefs(content: str) -> List[Dict[str, Any]]:
    """Parse belief entries from section content."""
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Try to extract confidence (e.g., "(80%)" or "[0.8]" or "(confidence: 0.9)")
        confidence = 0.7  # default
        conf_match = re.search(r"\((\d+)%\)|\[(\d*\.?\d+)\]", entry)
        if conf_match:
            if conf_match.group(1):
                confidence = int(conf_match.group(1)) / 100
            elif conf_match.group(2):
                confidence = float(conf_match.group(2))
            entry = re.sub(r"\(\d+%\)|\[\d*\.?\d+\]", "", entry).strip()
        else:
            # Try (confidence: N) format
            conf_match = re.search(r"\(confidence:\s*(\d*\.?\d+)\)", entry, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1))
                if confidence > 1:
                    confidence = confidence / 100
                entry = re.sub(
                    r"\(confidence:\s*\d*\.?\d+\)", "", entry, flags=re.IGNORECASE
                ).strip()

        items.append(
            {
                "type": "belief",
                "statement": entry,
                "confidence": min(1.0, max(0.0, confidence)),
                "source": "beliefs section",
            }
        )

    return items


def _parse_values(content: str) -> List[Dict[str, Any]]:
    """Parse value entries from section content."""
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Check for name: description format
        if ":" in entry:
            name, desc = entry.split(":", 1)
            name = name.strip()
            desc = desc.strip()
        else:
            name = entry[:50]
            desc = entry

        items.append(
            {"type": "value", "name": name, "description": desc, "source": "values section"}
        )

    return items


def _parse_goals(content: str) -> List[Dict[str, Any]]:
    """Parse goal entries from section content."""
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Check for [done] or [x] markers
        status = "active"
        if re.search(r"\[x\]|\[done\]|\[complete\]", entry, re.IGNORECASE):
            status = "completed"
            entry = re.sub(r"\[x\]|\[done\]|\[complete\]", "", entry, flags=re.IGNORECASE).strip()

        items.append(
            {"type": "goal", "description": entry, "status": status, "source": "goals section"}
        )

    return items


def _parse_raw(content: str) -> List[Dict[str, Any]]:
    """Parse raw entries from section content."""
    items = []

    # Check for bullet points first
    if re.search(r"^[-*]\s+", content, flags=re.MULTILINE):
        entries = re.split(r"^[-*]\s+", content, flags=re.MULTILINE)
    else:
        entries = _split_paragraphs(content)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        items.append({"type": "raw", "content": entry, "source": "raw section"})

    return items


# ============================================================================
# Import helpers
# ============================================================================


def _preview_item(index: int, item: Dict[str, Any]) -> None:
    """Print preview of an item."""
    t = item["type"]
    content = (
        item.get("content")
        or item.get("objective")
        or item.get("statement")
        or item.get("description")
        or item.get("name")
        or item.get("title", "")
    )
    preview = content[:80] + "..." if len(content) > 80 else content

    print(f"{index}. [{t}] {preview}")

    if item.get("lesson"):
        print(f"   -> Lesson: {item['lesson'][:60]}")
    if item.get("note_type") and item.get("note_type") != "note":
        print(f"   Type: {item['note_type']}")
    if item.get("confidence") and item["confidence"] != 0.7:
        print(f"   Confidence: {item['confidence']:.0%}")
    if item.get("status") and item["status"] != "active":
        print(f"   Status: {item['status']}")


def _interactive_import(
    items: List[Dict[str, Any]],
    k: "Kernle",
    derived_from: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Interactive import with user confirmation for each item."""
    imported = []

    print("Interactive mode: [y]es / [n]o / [e]dit / [s]kip all / [a]ccept all\n")

    accept_all = False

    for i, item in enumerate(items, 1):
        if accept_all:
            _import_item(item, k, derived_from)
            imported.append(item)
            continue

        _preview_item(i, item)

        try:
            choice = input("Import? [y/n/e/s/a]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nImport cancelled")
            break

        if choice == "a":
            accept_all = True
            _import_item(item, k, derived_from)
            imported.append(item)
        elif choice == "y":
            _import_item(item, k, derived_from)
            imported.append(item)
        elif choice == "s":
            print(f"Skipping remaining {len(items) - i + 1} items")
            break
        elif choice == "e":
            item = _edit_item(item)
            _import_item(item, k, derived_from)
            imported.append(item)
        else:
            print("  Skipped")

        print()

    print(f"\nImported {len(imported)} of {len(items)} items")
    return imported


def _edit_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Allow user to edit an item before import."""
    t = item["type"]

    try:
        if t == "episode":
            new = input(f"  Objective [{item.get('objective', '')[:50]}]: ").strip()
            if new:
                item["objective"] = new
            new = input(f"  Lesson [{item.get('lesson', '')}]: ").strip()
            if new:
                item["lesson"] = new
        elif t == "note":
            new = input(f"  Content [{item.get('content', '')[:50]}]: ").strip()
            if new:
                item["content"] = new
            new = input(f"  Type [{item.get('note_type', 'note')}]: ").strip()
            if new:
                item["note_type"] = new
        elif t == "belief":
            new = input(f"  Statement [{item.get('statement', '')[:50]}]: ").strip()
            if new:
                item["statement"] = new
            new = input(f"  Confidence [{item.get('confidence', 0.7):.0%}]: ").strip()
            if new:
                try:
                    item["confidence"] = (
                        float(new.replace("%", "")) / 100 if "%" in new else float(new)
                    )
                except ValueError:
                    pass  # Non-numeric value — keep existing
        elif t == "value":
            new = input(f"  Name [{item.get('name', '')[:50]}]: ").strip()
            if new:
                item["name"] = new
            new = input(f"  Description [{item.get('description', '')[:50]}]: ").strip()
            if new:
                item["description"] = new
        elif t == "goal":
            new = input(f"  Description [{item.get('description', '')[:50]}]: ").strip()
            if new:
                item["description"] = new
            new = input(f"  Status [{item.get('status', 'active')}]: ").strip()
            if new:
                item["status"] = new
        elif t == "raw":
            new = input(f"  Content [{item.get('content', '')[:50]}]: ").strip()
            if new:
                item["content"] = new
    except (EOFError, KeyboardInterrupt):
        pass  # User cancelled input — exit gracefully

    return item


def _batch_import(
    items: List[Dict[str, Any]],
    k: "Kernle",
    skip_duplicates: bool = False,
    derived_from: Optional[List[str]] = None,
) -> None:
    """Batch import all items."""
    success = 0
    skipped = 0
    errors = []
    seen_signatures = _seen_signature_buckets()
    existing_fingerprints = _build_import_fingerprint_index(k) if skip_duplicates else None

    for item in items:
        try:
            # Check for duplicates if requested
            if skip_duplicates:
                is_dup = _check_duplicate(
                    item,
                    k,
                    seen_signatures=seen_signatures,
                    existing_fingerprints=existing_fingerprints,
                )
                if is_dup:
                    skipped += 1
                    continue

            _import_item(item, k, derived_from)
            if skip_duplicates:
                _register_seen_signature(item, seen_signatures)
            success += 1
        except Exception as e:
            logger.debug(
                "Import item %s failed: %s",
                item.get("type", "unknown"),
                e,
                exc_info=True,
            )
            errors.append(f"{item['type']}: {str(e)[:50]}")

    print(f"Imported {success} items")
    if skipped > 0:
        print(f"Skipped {skipped} duplicates")
    if errors:
        print(f"{len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def _import_item(
    item: Dict[str, Any],
    k: "Kernle",
    derived_from: Optional[List[str]] = None,
) -> None:
    """Import a single item into Kernle."""
    t = item["type"]
    merged_derived_from = _merge_derived_from(derived_from, item)

    if t == "episode":
        lessons = [item["lesson"]] if item.get("lesson") else item.get("lessons")
        k.episode(
            objective=item["objective"],
            outcome=item.get("outcome", item["objective"]),
            lessons=lessons,
            tags=item.get("tags"),
            derived_from=merged_derived_from,
        )
    elif t == "note":
        k.note(
            content=item["content"],
            type=item.get("note_type", "note"),
            speaker=item.get("speaker"),
            reason=item.get("reason"),
            tags=item.get("tags"),
            derived_from=merged_derived_from,
        )
    elif t == "belief":
        k.belief(
            statement=item["statement"],
            confidence=item.get("confidence", 0.7),
            type=item.get("belief_type", "fact"),
            derived_from=merged_derived_from,
        )
    elif t == "value":
        k.value(
            name=item["name"],
            description=item.get("description", item["name"]),
            priority=item.get("priority", 50),
            derived_from=merged_derived_from,
        )
    elif t == "goal":
        goal_title = item.get("title") or item.get("description") or ""
        k.goal(
            description=item.get("description", item.get("title", "")),
            title=goal_title,
            status=item.get("status", "active"),
            priority=item.get("priority", "medium"),
            derived_from=merged_derived_from,
        )
    elif t == "raw":
        k.raw(blob=item["content"], source=item.get("source", "import"))
