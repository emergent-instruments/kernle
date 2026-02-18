"""Provenance chain validation for Kernle import.

Hard validation rules for JSON import. Returns errors only; caller decides
fail/raise. Every non-raw imported item must have valid derived_from references
pointing to items within the same import payload, all the way down to raw.
"""

from typing import Any, Dict, List, Tuple

# What each type is allowed to derive from
IMPORT_PROVENANCE_RULES: Dict[str, List[str]] = {
    "episode": ["raw"],
    "note": ["raw"],
    "belief": ["episode", "note"],
    "goal": ["episode", "belief"],
    "relationship": ["episode"],
    "value": ["belief"],
    "drive": ["episode", "belief"],
}

# Import dependency order (lowest tier first)
IMPORT_ORDER: List[str] = [
    "raw",
    "episode",
    "note",
    "belief",
    "goal",
    "drive",
    "relationship",
    "value",
]

# Whitelisted annotation ref prefixes (not traversable provenance)
ANNOTATION_PREFIXES: Tuple[str, ...] = ("context:", "kernle:")


def _is_annotation_ref(ref: str) -> bool:
    """Check if a derived_from ref is an annotation (not provenance)."""
    return any(ref.startswith(prefix) for prefix in ANNOTATION_PREFIXES)


def _parse_ref(ref: str) -> Tuple[str, str]:
    """Parse a 'type:id' ref into (type, id). Returns ('', '') if malformed."""
    if ":" not in ref:
        return "", ""
    parts = ref.split(":", 1)
    return parts[0], parts[1]


def validate_provenance_chains(items: List[Dict[str, Any]]) -> List[str]:
    """Validate provenance chains within an import dataset.

    Rules (hard, no exceptions):
    1. Every non-raw item MUST have derived_from (non-empty after filtering annotations)
    2. Every provenance ref MUST exist in the same import payload
    3. Each ref type MUST match the rule table for the item's type
    4. Annotation refs (context:*, kernle:*) are allowed but don't count as provenance

    Args:
        items: List of dicts with at least 'type', 'id', and 'derived_from' keys.

    Returns:
        List of error strings (empty = valid).
    """
    errors: List[str] = []

    # Build index: "type:id" -> True for every item in the import
    index: set = set()
    for item in items:
        item_type = item.get("type", "")
        item_id = item.get("id", "")
        if item_type and item_id:
            index.add(f"{item_type}:{item_id}")

    for item in items:
        item_type = item.get("type", "")
        item_id = item.get("id", "")

        # Raw entries don't need provenance
        if item_type == "raw":
            continue

        # Skip types not in our rule table
        if item_type not in IMPORT_PROVENANCE_RULES:
            continue

        allowed_types = IMPORT_PROVENANCE_RULES[item_type]
        derived_from = item.get("derived_from") or []

        # Separate provenance refs from annotation refs
        provenance_refs: List[str] = []
        for ref in derived_from:
            if not isinstance(ref, str):
                continue
            if _is_annotation_ref(ref):
                continue
            provenance_refs.append(ref)

        # Rule 1: Must have at least one provenance ref
        if not provenance_refs:
            errors.append(
                f"{item_type} '{item_id}' has no provenance refs "
                f"(must reference: {', '.join(allowed_types)})"
            )
            continue

        # Validate each provenance ref
        for ref in provenance_refs:
            ref_type, ref_id = _parse_ref(ref)

            if not ref_type or not ref_id:
                errors.append(
                    f"{item_type} '{item_id}' has malformed ref '{ref}' "
                    f"(expected format: type:id)"
                )
                continue

            # Rule 2: Ref must exist in this import
            if ref not in index:
                errors.append(
                    f"{item_type} '{item_id}' references {ref} " f"which is not in this import"
                )
                continue

            # Rule 3: Ref type must be allowed for this item type
            if ref_type not in allowed_types:
                errors.append(
                    f"{item_type} '{item_id}' references {ref_type} "
                    f"but must derive from: {', '.join(allowed_types)}"
                )

    return errors
