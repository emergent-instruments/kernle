"""Shared anxiety scoring functions.

Pure functions for computing anxiety dimension scores, used by both
AnxietyMixin (7-dim, Kernle layer) and AnxietyComponent (5-dim, stack layer).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---- Anxiety Level Thresholds ----

ANXIETY_LEVELS = {
    (0, 30): ("calm", "Calm"),
    (31, 50): ("aware", "Aware"),
    (51, 70): ("elevated", "Elevated"),
    (71, 85): ("high", "High"),
    (86, 100): ("critical", "Critical"),
}

# Per-level hysteresis thresholds: (enter_threshold, exit_threshold)
# enter_threshold = score required to transition UP into this level
# exit_threshold  = score below which we drop DOWN out of this level
LEVEL_HYSTERESIS = {
    "aware": (35, 25),  # enter aware at 35, exit back to calm at 25
    "elevated": (55, 45),  # enter elevated at 55, exit back to aware at 45
    "high": (75, 65),  # enter high at 75, exit back to elevated at 65
    "critical": (90, 80),  # enter critical at 90, exit back to high at 80
}


def get_anxiety_level(score: int) -> Tuple[str, str]:
    """Get (key, label) for an anxiety score."""
    for (low, high), (key, label) in ANXIETY_LEVELS.items():
        if low <= score <= high:
            return key, label
    return "critical", "Critical"


# ---- Dimension Weights ----

# 7-dimension weights (Kernle layer — includes context_pressure + unsaved_work)
SEVEN_DIM_WEIGHTS = {
    "context_pressure": 0.25,
    "unsaved_work": 0.20,
    "consolidation_debt": 0.15,
    "raw_aging": 0.10,
    "identity_coherence": 0.10,
    "memory_uncertainty": 0.10,
    "epoch_staleness": 0.10,
}

# 5-dimension weights (stack layer — renormalized from 7-dim shared subset)
_shared_keys = (
    "consolidation_debt",
    "raw_aging",
    "identity_coherence",
    "memory_uncertainty",
    "epoch_staleness",
)
_shared_total = sum(SEVEN_DIM_WEIGHTS[k] for k in _shared_keys)
FIVE_DIM_WEIGHTS = {k: SEVEN_DIM_WEIGHTS[k] / _shared_total for k in _shared_keys}


# ---- Scoring Functions ----


def compute_consolidation_score(unreflected_count: int) -> int:
    """Score for unreflected episodes (0-100)."""
    if unreflected_count <= 3:
        return unreflected_count * 7
    elif unreflected_count <= 7:
        return int(21 + (unreflected_count - 3) * 10)
    elif unreflected_count <= 15:
        return int(61 + (unreflected_count - 7) * 4)
    else:
        return min(100, int(93 + (unreflected_count - 15) * 0.5))


def compute_identity_coherence_score(identity_confidence: float) -> int:
    """Score for identity coherence (inverted: high confidence = low anxiety)."""
    return int((1.0 - identity_confidence) * 100)


def compute_memory_uncertainty_score(low_conf_count: int) -> int:
    """Score for low-confidence beliefs (0-100)."""
    if low_conf_count <= 2:
        return low_conf_count * 15
    elif low_conf_count <= 5:
        return int(30 + (low_conf_count - 2) * 15)
    else:
        return min(100, int(75 + (low_conf_count - 5) * 5))


def compute_raw_aging_score(total_unprocessed: int, aging_count: int, oldest_hours: float) -> int:
    """Score for aging raw entries (0-100).

    Args:
        total_unprocessed: Total number of unprocessed raw entries.
        aging_count: Number of entries older than the age threshold.
        oldest_hours: Age of the oldest entry in hours.
    """
    if total_unprocessed == 0:
        return 0
    elif aging_count == 0:
        return min(30, total_unprocessed * 3)
    elif aging_count <= 3:
        return int(30 + aging_count * 15)
    elif aging_count <= 7:
        return int(60 + (aging_count - 3) * 8)
    else:
        return min(100, int(92 + (aging_count - 7) * 1))


def compute_raw_aging_score_weighted(entries: List[Any], age_threshold_hours: int = 24) -> int:
    """Score for aging raw entries weighted by content length (0-100).

    A companion to compute_raw_aging_score() that weights entries by their
    content length instead of counting them equally. Callers can opt into
    this for more nuanced scoring of short/ambiguous prompts.

    Weight tiers:
        < 20 chars:  0.3 (low information)
        20-200 chars: 0.7 (moderate)
        > 200 chars:  1.0 (full weight)

    Args:
        entries: List of raw entry objects with ``content``, ``captured_at``
            and/or ``timestamp`` attributes.
        age_threshold_hours: Hours after which an entry is considered aging.
    """
    if not entries:
        return 0

    now = datetime.now(timezone.utc)
    total_weight = 0.0
    aging_weight = 0.0
    oldest_hours = 0.0

    for entry in entries:
        # Determine content-based weight
        content = getattr(entry, "content", None) or ""
        length = len(content)
        if length < 20:
            w = 0.3
        elif length <= 200:
            w = 0.7
        else:
            w = 1.0

        total_weight += w

        # Determine age
        entry_time = getattr(entry, "captured_at", None) or getattr(entry, "timestamp", None)
        if entry_time:
            try:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                age = now - entry_time
                entry_hours = age.total_seconds() / 3600
                if entry_hours > age_threshold_hours:
                    aging_weight += w
                if entry_hours > oldest_hours:
                    oldest_hours = entry_hours
            except (ValueError, TypeError, AttributeError):
                continue  # Skip entries with unparseable age/weight fields

    # Convert weighted counts to integer-like values for the existing scorer
    effective_total = total_weight
    effective_aging = aging_weight

    return compute_raw_aging_score(
        int(round(effective_total)),
        int(round(effective_aging)),
        oldest_hours,
    )


def score_with_confidence(score: int, sample_count: int) -> Dict[str, Any]:
    """Return score with confidence based on sample size.

    Confidence reflects how much data backs the score:
        0 samples:  0.0 (no data)
        1-2 samples: 0.5 (very sparse)
        3-5 samples: 0.7 (moderate)
        6+ samples:  0.9 (well-supported)

    Args:
        score: The raw dimension score (0-100).
        sample_count: Number of data points that contributed to the score.
    """
    if sample_count == 0:
        confidence = 0.0
    elif sample_count <= 2:
        confidence = 0.5
    elif sample_count <= 5:
        confidence = 0.7
    else:
        confidence = 0.9
    return {"score": score, "confidence": confidence}


def apply_hysteresis(
    current_score: int,
    previous_level: Optional[str],
) -> str:
    """Apply hysteresis to prevent oscillation near thresholds.

    When a score hovers near a level boundary, rapid toggling between
    adjacent levels is distracting. Hysteresis uses separate enter and
    exit thresholds so that a higher score is needed to *enter* the next
    level than is needed to *stay* there.

    Per-level thresholds are defined in :data:`LEVEL_HYSTERESIS`. Each
    level boundary has its own enter/exit pair, so transitions between
    calm/aware, aware/elevated, elevated/high, and high/critical are all
    handled independently.

    Args:
        current_score: The current numeric anxiety score (0-100).
        previous_level: The last computed level key (e.g. "elevated").
            Pass ``None`` on first call or when no history is available.

    Returns:
        The level key string (e.g. "calm", "aware", "elevated", "high",
        "critical").
    """
    # Standard level from raw score (no hysteresis)
    standard_key, _ = get_anxiety_level(current_score)

    if previous_level is None:
        return standard_key

    # Ordered level keys for comparison
    level_order = ["calm", "aware", "elevated", "high", "critical"]

    prev_idx = level_order.index(previous_level) if previous_level in level_order else -1

    if prev_idx < 0:
        return standard_key

    std_idx = level_order.index(standard_key)

    # Same level -- no transition needed
    if std_idx == prev_idx:
        return previous_level

    # Trying to move UP: walk one level at a time from the previous level.
    # Each step requires the score to meet the enter_threshold for that level.
    if std_idx > prev_idx:
        result_idx = prev_idx
        for step in range(prev_idx + 1, len(level_order)):
            target_level = level_order[step]
            enter_thresh, _ = LEVEL_HYSTERESIS.get(target_level, (0, 0))
            if current_score >= enter_thresh:
                result_idx = step
            else:
                break
        return level_order[result_idx]

    # Trying to move DOWN: walk one level at a time from the previous level.
    # Each step requires the score to be below the exit_threshold for that level.
    if std_idx < prev_idx:
        result_idx = prev_idx
        for step in range(prev_idx, 0, -1):
            current_level = level_order[step]
            _, exit_thresh = LEVEL_HYSTERESIS.get(current_level, (0, 0))
            if current_score < exit_thresh:
                result_idx = step - 1
            else:
                break
        return level_order[result_idx]

    return previous_level


def compute_epoch_staleness_score(months: Optional[float]) -> int:
    """Score for epoch staleness (0-100)."""
    if months is None:
        return 0
    elif months < 6:
        return int(months * 5)
    elif months < 12:
        return int(30 + (months - 6) * 6.7)
    elif months < 18:
        return int(70 + (months - 12) * 4)
    else:
        return min(100, int(94 + (months - 18) * 1.0))


def compute_context_pressure_score(context_pressure_pct: int) -> int:
    """Score for context pressure (0-100). Kernle-only dimension."""
    if context_pressure_pct < 50:
        return int(context_pressure_pct * 0.6)
    elif context_pressure_pct < 70:
        return int(30 + (context_pressure_pct - 50) * 1.5)
    elif context_pressure_pct < 85:
        return int(60 + (context_pressure_pct - 70) * 2)
    else:
        return min(100, int(90 + (context_pressure_pct - 85) * 0.67))


def compute_unsaved_work_score(checkpoint_age_minutes: Optional[int]) -> int:
    """Score for unsaved work (0-100). Kernle-only dimension."""
    if checkpoint_age_minutes is None:
        return 50
    elif checkpoint_age_minutes < 15:
        return int(checkpoint_age_minutes * 2)
    elif checkpoint_age_minutes < 60:
        return int(30 + (checkpoint_age_minutes - 15) * 1.1)
    else:
        return min(100, int(80 + (checkpoint_age_minutes - 60) * 0.33))
