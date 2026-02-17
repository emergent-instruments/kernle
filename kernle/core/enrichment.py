"""Pure enrichment functions for memory write paths.

Extracted from WritersMixin so that Entity methods, batch methods,
and WritersMixin itself all share the same enrichment logic.
All functions are stateless and side-effect-free.
"""

from typing import List, Optional, Union, get_args

from kernle.protocols import BeliefType, NoteType
from kernle.types import VALID_SOURCE_TYPE_VALUES, SourceType

# =========================================================================
# Constants
# =========================================================================

VALID_BELIEF_TYPES: frozenset[str] = frozenset(get_args(BeliefType))
VALID_NOTE_TYPES: frozenset[str] = frozenset(get_args(NoteType))
VALID_GOAL_TYPES = ("task", "aspiration", "commitment", "exploration")
DRIVE_TYPES = ["existence", "growth", "curiosity", "connection", "reproduction"]

# =========================================================================
# Outcome inference
# =========================================================================

_SUCCESS_WORDS = ("success", "done", "completed", "finished", "accomplished")
_FAILURE_WORDS = ("fail", "error", "broke", "unable", "couldn't")


def infer_outcome_type(outcome: str) -> str:
    """Classify outcome text as success/failure/partial using keyword matching.

    Args:
        outcome: The outcome description text.

    Returns:
        One of "success", "failure", or "partial".
    """
    outcome_lower = outcome.lower().strip()
    if any(word in outcome_lower for word in _SUCCESS_WORDS):
        return "success"
    if any(word in outcome_lower for word in _FAILURE_WORDS):
        return "failure"
    return "partial"


# =========================================================================
# Content formatting
# =========================================================================


def format_note_content(
    content: str,
    note_type: str,
    speaker: Optional[str] = None,
    reason: Optional[str] = None,
) -> str:
    """Format note content based on note_type.

    Args:
        content: Raw note content.
        note_type: The type of note (decision, quote, insight, etc.).
        speaker: Speaker name for quote-type notes.
        reason: Reason for decision-type notes.

    Returns:
        Formatted content string.
    """
    if note_type == "decision":
        formatted = f"**Decision**: {content}"
        if reason:
            formatted += f"\n**Reason**: {reason}"
        return formatted
    if note_type == "quote":
        speaker_name = speaker or "Unknown"
        return f'> "{content}"\n> — {speaker_name}'
    if note_type == "insight":
        return f"**Insight**: {content}"
    return content


# =========================================================================
# Derived-from building
# =========================================================================


def build_derived_from(
    derived_from: Optional[List[str]],
    source: Optional[str] = None,
) -> Optional[List[str]]:
    """Build derived_from list with optional source context marker.

    Args:
        derived_from: Existing list of memory references.
        source: Source context string; if provided, appends ``context:{source}``.

    Returns:
        The assembled list, or None if empty.
    """
    result = list(derived_from) if derived_from else []
    if source:
        result.append(f"context:{source}")
    return result if result else None


# =========================================================================
# Clamping helpers
# =========================================================================


def clamp_confidence(confidence: float) -> float:
    """Clamp confidence to the valid range [0.0, 1.0]."""
    return max(0.0, min(1.0, confidence))


def clamp_intensity(intensity: float) -> float:
    """Clamp intensity to the valid range [0.0, 1.0]."""
    return max(0.0, min(1.0, intensity))


# =========================================================================
# Type validation
# =========================================================================


def validate_goal_type(goal_type: str) -> str:
    """Validate goal_type against the allowed set.

    Args:
        goal_type: The goal type to validate.

    Returns:
        The validated goal_type (unchanged).

    Raises:
        ValueError: If goal_type is not in VALID_GOAL_TYPES.
    """
    if goal_type not in VALID_GOAL_TYPES:
        raise ValueError(f"Invalid goal_type. Must be one of: {', '.join(VALID_GOAL_TYPES)}")
    return goal_type


def validate_drive_type(drive_type: str) -> None:
    """Validate drive_type against the allowed set.

    Args:
        drive_type: The drive type to validate.

    Raises:
        ValueError: If drive_type is not in DRIVE_TYPES.
    """
    if drive_type not in DRIVE_TYPES:
        raise ValueError(f"Invalid drive type. Must be one of: {DRIVE_TYPES}")


# =========================================================================
# Type normalization
# =========================================================================


def normalize_source_type(source_type: Optional[Union[str, SourceType]]) -> SourceType:
    """Return a canonical ``SourceType`` and reject invalid values."""
    if source_type is None:
        return SourceType.DIRECT_EXPERIENCE
    if isinstance(source_type, SourceType):
        return source_type
    if not isinstance(source_type, str):
        raise ValueError("source_type must be a string or SourceType")

    normalized = source_type.strip().lower()
    if normalized in VALID_SOURCE_TYPE_VALUES:
        return SourceType(normalized)
    raise ValueError(
        f"Invalid source_type: '{source_type}'. "
        f"Valid values: {sorted(VALID_SOURCE_TYPE_VALUES)}"
    )


def normalize_belief_type(belief_type: Optional[str]) -> str:
    """Return a canonical belief type and reject invalid values."""
    if belief_type is None:
        return "fact"

    if not isinstance(belief_type, str):
        raise ValueError("belief_type must be a string")

    normalized = belief_type.strip().lower()
    if normalized in VALID_BELIEF_TYPES:
        return normalized

    raise ValueError(
        "Invalid belief type. Must be one of: " + ", ".join(sorted(VALID_BELIEF_TYPES))
    )


def normalize_note_type(note_type: Optional[str]) -> str:
    """Return a canonical note type and reject invalid values."""
    if note_type is None:
        return "note"

    if not isinstance(note_type, str):
        raise ValueError("note_type must be a string")

    normalized = note_type.strip().lower()
    if normalized in VALID_NOTE_TYPES:
        return normalized

    raise ValueError("Invalid note type. Must be one of: " + ", ".join(sorted(VALID_NOTE_TYPES)))
