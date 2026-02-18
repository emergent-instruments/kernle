"""JSON importer for Kernle.

Imports from the JSON format exported by `kernle export --format json`.
This preserves metadata like confidence, timestamps, and relationships.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)


@dataclass
class JsonImportItem:
    """A parsed JSON item ready for import."""

    type: str  # episode, note, belief, value, goal, drive, relationship, raw
    data: Dict[str, Any] = field(default_factory=dict)


def _validate_range(
    value: float,
    min_val: float,
    max_val: float,
    field_name: str,
    *,
    strict: bool = False,
    coercion_warnings: Optional[List[Dict[str, Any]]] = None,
) -> tuple[float, bool]:
    """Validate and optionally clamp a numeric value to a range.

    Args:
        value: The value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        field_name: Name of the field for warning messages
        strict: If True, reject out-of-range values instead of clamping
        coercion_warnings: List to append warnings to when clamping

    Returns:
        Tuple of (clamped_value, rejected). If rejected=True, the item should be skipped.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value, True  # Always reject non-finite values

    if min_val <= value <= max_val:
        return value, False

    if strict:
        return value, True

    original = value
    clamped = max(min_val, min(max_val, value))

    if coercion_warnings is not None:
        coercion_warnings.append(
            {
                "field": field_name,
                "original": original,
                "coerced_to": clamped,
                "reason": f"{field_name} clamped from {original} to {clamped} (valid range: [{min_val}, {max_val}])",
            }
        )
    else:
        logger.warning(
            "%s value %s out of range [%s, %s], clamped to %s",
            field_name,
            original,
            min_val,
            max_val,
            clamped,
        )

    return clamped, False


class JsonImporter:
    """Import memories from Kernle JSON export files.

    Supports the JSON format from `kernle export --format json` and
    `kernle dump --format json`.
    """

    def __init__(self, file_path: str, strict: bool = False):
        """Initialize with path to JSON file.

        Args:
            file_path: Path to the JSON file to import
            strict: If True, reject items with out-of-range values
        """
        self.file_path = Path(file_path).expanduser()
        self.items: List[JsonImportItem] = []
        self.source_stack_id: Optional[str] = None
        self.strict = strict

    def parse(self) -> List[JsonImportItem]:
        """Parse the JSON file and return importable items.

        Returns:
            List of JsonImportItem objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        content = self.file_path.read_text(encoding="utf-8")
        self.items, self.source_stack_id = parse_kernle_json(content)
        return self.items

    def import_to(
        self, k: "Kernle", dry_run: bool = False, skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Import parsed items into a Kernle instance.

        Args:
            k: Kernle instance to import into
            dry_run: If True, don't actually import, just return counts
            skip_duplicates: If True, skip items that already exist (by content)

        Returns:
            Dict with counts of items imported by type and any errors
        """
        if not self.items:
            self.parse()

        counts: Dict[str, int] = {}
        skipped: Dict[str, int] = {}
        errors: List[str] = []
        coercion_warnings: List[Dict[str, Any]] = []

        for item in self.items:
            try:
                if not dry_run:
                    imported = _import_json_item(
                        item,
                        k,
                        skip_duplicates,
                        strict=self.strict,
                        coercion_warnings=coercion_warnings,
                    )
                    if imported:
                        counts[item.type] = counts.get(item.type, 0) + 1
                    else:
                        skipped[item.type] = skipped.get(item.type, 0) + 1
                else:
                    counts[item.type] = counts.get(item.type, 0) + 1
            except Exception as e:
                logger.debug("JSON import item %s failed: %s", item.type, e, exc_info=True)
                errors.append(f"{item.type}: {str(e)[:50]}")

        return {
            "imported": counts,
            "skipped": skipped,
            "errors": errors,
            "source_stack_id": self.source_stack_id,
            "coercion_warnings": coercion_warnings,
        }


def parse_kernle_json(content: str) -> tuple[List[JsonImportItem], Optional[str]]:
    """Parse Kernle JSON export format.

    Expected format:
    {
        "stack_id": "...",
        "exported_at": "...",
        "values": [...],
        "beliefs": [...],
        "goals": [...],
        "episodes": [...],
        "notes": [...],
        "drives": [...],
        "relationships": [...],
        "raw_entries": [...]  # optional
    }

    Args:
        content: JSON content string

    Returns:
        Tuple of (list of JsonImportItem, source stack_id)

    Raises:
        json.JSONDecodeError: If content is not valid JSON
        ValueError: If the format doesn't match expected structure
    """
    data = json.loads(content)

    # Validate it looks like a Kernle export
    if not isinstance(data, dict):
        raise ValueError("JSON must be an object at the root level")

    stack_id = data.get("stack_id")
    items: List[JsonImportItem] = []

    # Parse each memory type
    for value_data in data.get("values", []):
        items.append(JsonImportItem(type="value", data=value_data))

    for belief_data in data.get("beliefs", []):
        items.append(JsonImportItem(type="belief", data=belief_data))

    for goal_data in data.get("goals", []):
        items.append(JsonImportItem(type="goal", data=goal_data))

    for episode_data in data.get("episodes", []):
        items.append(JsonImportItem(type="episode", data=episode_data))

    for note_data in data.get("notes", []):
        items.append(JsonImportItem(type="note", data=note_data))

    for drive_data in data.get("drives", []):
        items.append(JsonImportItem(type="drive", data=drive_data))

    for rel_data in data.get("relationships", []):
        items.append(JsonImportItem(type="relationship", data=rel_data))

    for raw_data in data.get("raw_entries", []):
        items.append(JsonImportItem(type="raw", data=raw_data))

    return items, stack_id


def _import_json_item(
    item: JsonImportItem,
    k: "Kernle",
    skip_duplicates: bool = True,
    *,
    strict: bool = False,
    coercion_warnings: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Import a single JSON item into Kernle.

    Args:
        item: The JsonImportItem to import
        k: Kernle instance
        skip_duplicates: If True, skip items that already exist
        strict: If True, reject items with out-of-range values
        coercion_warnings: List to append coercion warnings to

    Returns:
        True if imported, False if skipped/rejected
    """
    t = item.type
    data = item.data

    if t == "episode":
        # Check for duplicate by objective + outcome
        if skip_duplicates:
            existing = k._storage.get_episodes(limit=100)
            for ep in existing:
                if ep.objective == data.get("objective") and ep.outcome == data.get("outcome"):
                    return False

        # Build tags, folding in outcome_type if present
        tags = data.get("tags") or []
        outcome_type = data.get("outcome_type")
        if outcome_type:
            tags = list(tags) + [f"outcome:{outcome_type}"]

        k.episode(
            objective=data.get("objective", ""),
            outcome=data.get("outcome", ""),
            lessons=data.get("lessons"),
            tags=tags or None,
        )
        return True

    elif t == "note":
        if skip_duplicates:
            existing = k._storage.get_notes(limit=100)
            for n in existing:
                if n.content == data.get("content"):
                    return False

        k.note(
            content=data.get("content", ""),
            type=data.get("type", "note"),
            speaker=data.get("speaker"),
            reason=data.get("reason"),
            tags=data.get("tags"),
        )
        return True

    elif t == "belief":
        statement = data.get("statement", "")
        if skip_duplicates:
            existing = k._storage.find_belief(statement)
            if existing:
                return False

        # Validate confidence
        confidence = data.get("confidence", 0.8)
        if isinstance(confidence, bool):
            if strict:
                return False
            confidence = 0.8
        elif isinstance(confidence, (int, float)):
            confidence, rejected = _validate_range(
                float(confidence),
                0.0,
                1.0,
                "confidence",
                strict=strict,
                coercion_warnings=coercion_warnings,
            )
            if rejected:
                return False

        k.belief(
            statement=statement,
            type=data.get("type", "fact"),
            confidence=confidence,
            source_type="imported",
        )
        return True

    elif t == "value":
        name = data.get("name", "")
        if skip_duplicates:
            existing = k._storage.get_values(limit=100)
            for v in existing:
                if v.name == name:
                    return False

        # Validate priority if it's numeric
        priority = data.get("priority", 50)
        if isinstance(priority, bool):
            if strict:
                return False
            priority = 50
        elif isinstance(priority, (int, float)):
            # Check NaN/Inf before int() conversion (int(float('nan')) raises)
            if isinstance(priority, float) and (math.isnan(priority) or math.isinf(priority)):
                if strict:
                    return False
                priority = 50
            else:
                int_priority = int(priority)
                int_priority, rejected = _validate_range(
                    float(int_priority),
                    0,
                    100,
                    "priority",
                    strict=strict,
                    coercion_warnings=coercion_warnings,
                )
                if rejected:
                    return False
                priority = int(int_priority)

        k.value(
            name=name,
            statement=data.get("statement", data.get("description", name)),
            priority=priority,
            source_type="imported",
        )
        return True

    elif t == "goal":
        description = data.get("description") or data.get("title", "")
        if skip_duplicates:
            existing = k._storage.get_goals(status=None, limit=100)
            for g in existing:
                if g.title == data.get("title") or g.description == description:
                    return False

        k.goal(
            title=data.get("title") or description,
            description=description,
            priority=data.get("priority", "medium"),
            source_type="imported",
        )
        return True

    elif t == "drive":
        drive_type = data.get("drive_type", "")
        if skip_duplicates:
            existing = k._storage.get_drive(drive_type)
            if existing:
                return False

        # Validate intensity
        intensity = data.get("intensity", 0.5)
        if isinstance(intensity, bool):
            if strict:
                return False
            intensity = 0.5
        elif isinstance(intensity, (int, float)):
            intensity, rejected = _validate_range(
                float(intensity),
                0.0,
                1.0,
                "intensity",
                strict=strict,
                coercion_warnings=coercion_warnings,
            )
            if rejected:
                return False

        k.drive(
            drive_type=drive_type,
            intensity=intensity,
            focus_areas=data.get("focus_areas"),
            source_type="imported",
        )
        return True

    elif t == "relationship":
        entity_name = data.get("entity_name", "")
        if skip_duplicates:
            existing = k._storage.get_relationship(entity_name)
            if existing:
                return False

        # Map JSON fields to API parameters:
        # entity_name -> other_stack_id, sentiment -> trust_level (rescale -1..1 to 0..1),
        # relationship_type -> interaction_type
        sentiment = data.get("sentiment", 0.0)

        # Validate sentiment
        if isinstance(sentiment, bool):
            if strict:
                return False
            sentiment = 0.0
        elif isinstance(sentiment, (int, float)):
            sentiment, rejected = _validate_range(
                float(sentiment),
                -1.0,
                1.0,
                "sentiment",
                strict=strict,
                coercion_warnings=coercion_warnings,
            )
            if rejected:
                return False

        trust_level = (sentiment + 1.0) / 2.0  # Convert sentiment (-1..1) to trust (0..1)

        k.relationship(
            other_stack_id=entity_name,
            entity_type=data.get("entity_type", "unknown"),
            interaction_type=data.get("relationship_type", "knows"),
            trust_level=trust_level,
            notes=data.get("notes"),
            source_type="imported",
        )
        return True

    elif t == "raw":
        content = data.get("content", "")
        if skip_duplicates:
            existing = k._storage.list_raw(limit=100)
            for r in existing:
                if r.content == content:
                    return False

        k.raw(
            blob=content,
            source=data.get("source", "import"),
        )
        return True

    return False
