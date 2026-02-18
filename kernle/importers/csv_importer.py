"""CSV importer for Kernle.

Imports from CSV files with columns for memory fields.
Supports bulk import of memories in tabular format.
"""

import csv
import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, overload

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)


# Column name mappings for each memory type
COLUMN_MAPPINGS = {
    "episode": {
        "objective": ["objective", "title", "task", "name"],
        "outcome": ["outcome", "result", "description", "desc"],
        "outcome_type": ["outcome_type", "type", "status", "result_type"],
        "lessons": ["lessons", "lesson", "learnings", "learning"],
        "tags": ["tags", "tag", "labels", "label"],
    },
    "note": {
        "content": ["content", "text", "note", "body", "description"],
        "type": ["type", "note_type", "category", "kind"],
        "speaker": ["speaker", "author", "from", "source"],
        "reason": ["reason", "why", "context"],
        "tags": ["tags", "tag", "labels", "label"],
    },
    "belief": {
        "statement": ["statement", "belief", "content", "text", "description"],
        "confidence": ["confidence", "conf", "certainty", "probability"],
        "type": ["type", "belief_type", "category", "kind"],
    },
    "value": {
        "name": ["name", "value", "title"],
        "description": ["description", "desc", "statement", "content", "text"],
        "priority": ["priority", "importance", "weight", "rank"],
    },
    "goal": {
        "title": ["title", "goal", "name", "objective"],
        "description": ["description", "desc", "details", "content"],
        "priority": ["priority", "importance", "urgency"],
        "status": ["status", "state", "complete", "completed"],
    },
    "raw": {
        "content": ["content", "text", "raw", "body", "data"],
        "source": ["source", "from", "origin"],
        "tags": ["tags", "tag", "labels", "label"],
    },
}


@dataclass
class CsvImportItem:
    """A parsed CSV row ready for import."""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)


class CsvImporter:
    """Import memories from CSV files.

    The CSV file must have a 'type' column indicating the memory type
    (episode, note, belief, value, goal, raw) and columns matching
    the expected fields for each type.

    Example CSV:
    ```csv
    type,content,confidence
    belief,"Testing is important",0.9
    belief,"Code should be readable",0.85
    note,"Review the authentication system",
    ```
    """

    def __init__(
        self,
        file_path: str,
        memory_type: Optional[str] = None,
        strict: bool = False,
    ):
        """Initialize with path to CSV file.

        Args:
            file_path: Path to the CSV file to import
            memory_type: If set, treat all rows as this type (overrides 'type' column)
            strict: If True, reject rows with malformed values instead of coercing
        """
        self.file_path = Path(file_path).expanduser()
        self.items: List[CsvImportItem] = []
        self.memory_type = memory_type
        self.strict = strict
        self.coercion_warnings: List[Dict[str, Any]] = []
        self.rejections: List[Dict[str, Any]] = []

    def parse(self) -> List[CsvImportItem]:
        """Parse the CSV file and return importable items.

        Returns:
            List of CsvImportItem objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        content = self.file_path.read_text(encoding="utf-8")
        self.items, self.coercion_warnings, self.rejections = parse_csv(
            content,
            self.memory_type,
            strict=self.strict,
            return_warnings=True,
            return_rejections=True,
        )
        return self.items

    def import_to(
        self, k: "Kernle", dry_run: bool = False, skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Import parsed items into a Kernle instance.

        Args:
            k: Kernle instance to import into
            dry_run: If True, don't actually import, just return counts
            skip_duplicates: If True, skip items that already exist

        Returns:
            Dict with counts of items imported by type and any errors
        """
        if not self.items:
            self.parse()

        counts: Dict[str, int] = {}
        skipped: Dict[str, int] = {}
        errors: List[str] = []
        rejections: List[Dict[str, Any]] = list(self.rejections)

        for i, item in enumerate(self.items):
            try:
                if not dry_run:
                    imported = _import_csv_item(item, k, skip_duplicates)
                    if imported:
                        counts[item.type] = counts.get(item.type, 0) + 1
                    else:
                        skipped[item.type] = skipped.get(item.type, 0) + 1
                        rejections.append(
                            {
                                "row": i + 2,
                                "type": item.type,
                                "reason": "missing required field or duplicate",
                            }
                        )
                else:
                    counts[item.type] = counts.get(item.type, 0) + 1
            except Exception as e:
                logger.debug(
                    "CSV import row %d (%s) failed: %s", i + 2, item.type, e, exc_info=True
                )
                errors.append(f"Row {i + 2}: {item.type}: {str(e)[:50]}")

        return {
            "imported": counts,
            "skipped": skipped,
            "errors": errors,
            "coercion_warnings": list(self.coercion_warnings),
            "rejections": rejections,
        }


@overload
def parse_csv(
    content: str,
    memory_type: Optional[str] = None,
    *,
    strict: bool = False,
    return_warnings: bool = False,
    return_rejections: bool = False,
) -> List[CsvImportItem]: ...


@overload
def parse_csv(
    content: str,
    memory_type: Optional[str] = None,
    *,
    strict: bool = False,
    return_warnings: bool = True,
    return_rejections: bool = False,
) -> Tuple[List[CsvImportItem], List[Dict[str, Any]]]: ...


@overload
def parse_csv(
    content: str,
    memory_type: Optional[str] = None,
    *,
    strict: bool = False,
    return_warnings: bool = True,
    return_rejections: bool = True,
) -> Tuple[List[CsvImportItem], List[Dict[str, Any]], List[Dict[str, Any]]]: ...


def parse_csv(
    content: str,
    memory_type: Optional[str] = None,
    *,
    strict: bool = False,
    return_warnings: bool = False,
    return_rejections: bool = False,
) -> Union[
    List[CsvImportItem],
    Tuple[List[CsvImportItem], List[Dict[str, Any]]],
    Tuple[List[CsvImportItem], List[Dict[str, Any]], List[Dict[str, Any]]],
]:
    """Parse CSV content into importable items.

    Args:
        content: CSV content string
        memory_type: If set, treat all rows as this type
        strict: If True, reject rows with malformed values instead of coercing
        return_warnings: If True, return (items, warnings) tuple
        return_rejections: If True, return (items, warnings, rejections) tuple
            (requires return_warnings=True)

    Returns:
        List of CsvImportItem objects, or tuple with warnings/rejections

    Raises:
        ValueError: If the CSV format is invalid
    """
    items: List[CsvImportItem] = []
    warnings: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []

    reader = csv.DictReader(io.StringIO(content))
    headers = [h.lower().strip() for h in (reader.fieldnames or [])]

    if not headers:
        raise ValueError("CSV file has no headers")

    # Check if 'type' column exists
    has_type_column = any(h in ["type", "memory_type", "kind"] for h in headers)

    if not has_type_column and not memory_type:
        raise ValueError(
            "CSV must have a 'type' column or specify --type when importing. "
            "Valid types: episode, note, belief, value, goal, raw"
        )

    for row_num, row in enumerate(reader, start=2):
        # Normalize keys to lowercase
        row = {k.lower().strip(): v.strip() if v else "" for k, v in row.items()}

        # Determine memory type
        if memory_type:
            item_type = memory_type
        else:
            item_type = row.get("type") or row.get("memory_type") or row.get("kind")
            if not item_type:
                continue  # Skip rows without type

        item_type = item_type.lower().strip()

        # Validate type
        if item_type not in COLUMN_MAPPINGS:
            continue  # Skip unknown types

        # Map columns to expected field names with validation
        data, row_warnings, row_rejected = _map_columns(
            row, item_type, strict=strict, row_num=row_num
        )

        warnings.extend(row_warnings)

        if row_rejected:
            rejections.extend(row_rejected)
            continue  # Skip this row in strict mode

        # Skip empty rows
        if not any(data.values()):
            continue

        items.append(CsvImportItem(type=item_type, data=data))

    if return_warnings and return_rejections:
        return items, warnings, rejections
    elif return_warnings:
        return items, warnings
    return items


def _map_columns(
    row: Dict[str, str],
    memory_type: str,
    *,
    strict: bool = False,
    row_num: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Map CSV columns to memory field names with validation.

    Args:
        row: The CSV row dict
        memory_type: The type of memory
        strict: If True, reject rows with malformed values
        row_num: Row number for warning/rejection reporting

    Returns:
        Tuple of (data dict, coercion warnings, rejections)
    """
    mappings = COLUMN_MAPPINGS.get(memory_type, {})
    result: Dict[str, Any] = {}
    warnings: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []

    for field_name, aliases in mappings.items():
        for alias in aliases:
            if alias in row and row[alias]:
                value = row[alias]

                # Type conversions with bounds validation
                if field_name == "confidence":
                    try:
                        fval = float(value)
                    except ValueError:
                        if strict:
                            rejections.append(
                                {
                                    "row": row_num,
                                    "field": "confidence",
                                    "value": value,
                                    "reason": f"non-numeric confidence value: {value!r}",
                                }
                            )
                            break
                        else:
                            warnings.append(
                                {
                                    "row": row_num,
                                    "field": "confidence",
                                    "original": value,
                                    "coerced_to": 0.7,
                                    "reason": f"non-numeric confidence value: {value!r}, defaulting to 0.7",
                                }
                            )
                            value = 0.7
                            result[field_name] = value
                            break

                    # Reject non-finite values
                    if math.isnan(fval) or math.isinf(fval):
                        rejections.append(
                            {
                                "row": row_num,
                                "field": "confidence",
                                "value": value,
                                "reason": f"non-finite confidence value: {value!r}",
                            }
                        )
                        break

                    # Auto-scale percentage values (1 < x <= 100)
                    if fval > 1:
                        fval = fval / 100

                    # Validate bounds after scaling
                    if fval < 0 or fval > 1:
                        if strict:
                            rejections.append(
                                {
                                    "row": row_num,
                                    "field": "confidence",
                                    "value": value,
                                    "reason": f"confidence out of range [0.0, 1.0] after scaling: {fval}",
                                }
                            )
                            break
                        else:
                            original = fval
                            fval = max(0.0, min(1.0, fval))
                            warnings.append(
                                {
                                    "row": row_num,
                                    "field": "confidence",
                                    "original": value,
                                    "coerced_to": fval,
                                    "reason": f"confidence clamped from {original} to {fval}",
                                }
                            )

                    value = fval

                elif field_name == "priority" and memory_type == "value":
                    try:
                        ival = int(value)
                    except ValueError:
                        if strict:
                            rejections.append(
                                {
                                    "row": row_num,
                                    "field": "priority",
                                    "value": value,
                                    "reason": f"non-numeric priority value: {value!r}",
                                }
                            )
                            break
                        else:
                            warnings.append(
                                {
                                    "row": row_num,
                                    "field": "priority",
                                    "original": value,
                                    "coerced_to": 50,
                                    "reason": f"non-numeric priority value: {value!r}, defaulting to 50",
                                }
                            )
                            value = 50
                            result[field_name] = value
                            break

                    # Clamp to valid range
                    if ival < 0 or ival > 100:
                        if strict:
                            rejections.append(
                                {
                                    "row": row_num,
                                    "field": "priority",
                                    "value": value,
                                    "reason": f"priority out of range [0, 100]: {ival}",
                                }
                            )
                            break
                        else:
                            original = ival
                            ival = max(0, min(100, ival))
                            warnings.append(
                                {
                                    "row": row_num,
                                    "field": "priority",
                                    "original": value,
                                    "coerced_to": ival,
                                    "reason": f"priority clamped from {original} to {ival}",
                                }
                            )

                    value = ival

                elif field_name == "intensity":
                    try:
                        fval = float(value)
                    except ValueError:
                        if strict:
                            rejections.append(
                                {
                                    "row": row_num,
                                    "field": "intensity",
                                    "value": value,
                                    "reason": f"non-numeric intensity value: {value!r}",
                                }
                            )
                            break
                        else:
                            warnings.append(
                                {
                                    "row": row_num,
                                    "field": "intensity",
                                    "original": value,
                                    "coerced_to": 0.5,
                                    "reason": f"non-numeric intensity value: {value!r}, defaulting to 0.5",
                                }
                            )
                            value = 0.5
                            result[field_name] = value
                            break

                    if fval < 0 or fval > 1:
                        if strict:
                            rejections.append(
                                {
                                    "row": row_num,
                                    "field": "intensity",
                                    "value": value,
                                    "reason": f"intensity out of range [0.0, 1.0]: {fval}",
                                }
                            )
                            break
                        else:
                            original = fval
                            fval = max(0.0, min(1.0, fval))
                            warnings.append(
                                {
                                    "row": row_num,
                                    "field": "intensity",
                                    "original": value,
                                    "coerced_to": fval,
                                    "reason": f"intensity clamped from {original} to {fval}",
                                }
                            )

                    value = fval

                elif field_name in ("tags", "lessons"):
                    # Split comma-separated values
                    value = [v.strip() for v in value.split(",") if v.strip()]

                result[field_name] = value
                break

    return result, warnings, rejections


def _import_csv_item(item: CsvImportItem, k: "Kernle", skip_duplicates: bool = True) -> bool:
    """Import a single CSV item into Kernle.

    Args:
        item: The CsvImportItem to import
        k: Kernle instance
        skip_duplicates: If True, skip items that already exist

    Returns:
        True if imported, False if skipped
    """
    t = item.type
    data = item.data

    if t == "episode":
        objective = data.get("objective", "")
        if not objective:
            return False

        if skip_duplicates:
            existing_episodes = k._storage.get_episodes(limit=500)
            for episode in existing_episodes:
                if episode.objective == objective:
                    return False

        # Build tags, folding in outcome_type if present
        tags = data.get("tags") or []
        outcome_type = data.get("outcome_type")
        if outcome_type:
            tags = list(tags) + [f"outcome:{outcome_type}"]

        k.episode(
            objective=objective,
            outcome=data.get("outcome", objective),
            lessons=data.get("lessons"),
            tags=tags or None,
        )
        return True

    elif t == "note":
        content = data.get("content", "")
        if not content:
            return False

        if skip_duplicates:
            existing_notes = k._storage.get_notes(limit=500)
            for note in existing_notes:
                if note.content == content:
                    return False

        k.note(
            content=content,
            type=data.get("type", "note"),
            speaker=data.get("speaker"),
            reason=data.get("reason"),
            tags=data.get("tags"),
        )
        return True

    elif t == "belief":
        statement = data.get("statement", "")
        if not statement:
            return False

        if skip_duplicates:
            existing = k._storage.find_belief(statement)
            if existing:
                return False

        belief_type = data.get("belief_type", data.get("type", "fact"))
        if belief_type == "belief":
            belief_type = "fact"

        k.belief(
            statement=statement,
            type=belief_type,
            confidence=data.get("confidence", 0.8),
            source_type="imported",
        )
        return True

    elif t == "value":
        name = data.get("name", "")
        if not name:
            return False

        if skip_duplicates:
            existing = k._storage.get_values(limit=100)
            for v in existing:
                if v.name == name:
                    return False

        k.value(
            name=name,
            statement=data.get("description", name),
            priority=data.get("priority", 50),
            source_type="imported",
        )
        return True

    elif t == "goal":
        title = data.get("title", "")
        description = data.get("description", title)
        if not title and not description:
            return False

        if skip_duplicates:
            existing = k._storage.get_goals(status=None, limit=100)
            for g in existing:
                if g.title == title or g.description == description:
                    return False

        # Map status values
        status = data.get("status", "active")
        if status.lower() in ("done", "complete", "completed", "true", "1", "yes"):
            status = "completed"
        elif status.lower() in ("paused", "hold", "on hold"):
            status = "paused"
        else:
            status = "active"

        goal_id = k.goal(
            title=title or description,
            description=description,
            priority=data.get("priority", "medium"),
            source_type="imported",
        )

        # goal() always creates with status="active"; update if needed
        if status != "active":
            k.update_goal(goal_id, status=status)

        return True

    elif t == "raw":
        content = data.get("content", "")
        if not content:
            return False

        if skip_duplicates:
            existing = k._storage.list_raw(limit=100)
            for r in existing:
                if r.content == content:
                    return False

        k.raw(
            blob=content,
            source=data.get("source", "csv-import"),
        )
        return True

    return False
