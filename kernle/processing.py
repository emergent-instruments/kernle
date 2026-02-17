"""Memory Processing Sessions — focused inference for memory promotion.

Runs targeted processing sessions that promote memories up the hierarchy:
  raw → episode/note → belief → value
  raw → episode → goal, relationship, drive

Each layer transition has its own prompts, triggers, and configuration.
Processing uses the bound model (via InferenceService) to reason about
unprocessed memories and create higher-layer memories with provenance.

Design: Option 2 (entity-level processing) — structured inference calls
through focused prompts. Designed to evolve toward Option 3 (full
recursive self-invocation) by swapping the process_layer implementation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Deduplication Utilities
# =============================================================================


def compute_provenance_hash(derived_from: List[str]) -> str:
    """Compute a stable hash from a sorted provenance set.

    Given a list of derived_from refs (e.g. ["raw:abc", "raw:def"]),
    produces a deterministic hex digest. Two memories with the same
    provenance set will always produce the same hash.
    """
    if not derived_from:
        return ""
    canonical = "|".join(sorted(derived_from))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute a hash from normalized content text.

    Normalizes whitespace and lowercases before hashing to catch
    near-identical content from different inference runs.
    """
    if not content:
        return ""
    normalized = " ".join(content.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _extract_content_text(transition: str, item: dict) -> str:
    """Extract the primary content text from a parsed item for hashing."""
    if transition in ("raw_to_episode", "episode_to_goal"):
        parts = [item.get("objective", ""), item.get("outcome", "")]
        if transition == "episode_to_goal":
            parts = [item.get("title", ""), item.get("description", "")]
        return " ".join(p for p in parts if p)
    elif transition == "raw_to_note":
        return item.get("content", "")
    elif transition == "episode_to_belief":
        return item.get("statement", "")
    elif transition == "episode_to_relationship":
        return item.get("entity_name", "")
    elif transition == "belief_to_value":
        parts = [item.get("name", ""), item.get("statement", "")]
        return " ".join(p for p in parts if p)
    elif transition == "episode_to_drive":
        return item.get("drive_type", "")
    return ""


def _parse_created_at(value: Any) -> Optional[datetime]:
    """Parse audit log created_at values into aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_derived_from(transition: str, item: dict) -> List[str]:
    """Extract the derived_from list from a parsed item."""
    if transition in ("raw_to_episode", "raw_to_note"):
        raw_ids = item.get("source_raw_ids", [])
        return [f"raw:{rid}" for rid in raw_ids]
    elif transition in (
        "episode_to_belief",
        "episode_to_goal",
        "episode_to_relationship",
        "episode_to_drive",
    ):
        ep_ids = item.get("source_episode_ids", [])
        return [f"episode:{eid}" for eid in ep_ids]
    elif transition == "belief_to_value":
        belief_ids = item.get("source_belief_ids", [])
        return [f"belief:{bid}" for bid in belief_ids]
    return []


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PromotionGateConfig:
    """Configurable promotion eligibility criteria.

    Identity layers should be "hard to write, easy to read." These gates
    ensure promotions are earned through repeated evidence, not created
    from thin evidence.
    """

    # Belief creation gates
    belief_min_evidence: int = 3  # Min episodes/notes required in derived_from
    belief_min_confidence: float = 0.6  # Min confidence score

    # Value creation gates
    value_min_evidence: int = 5  # Min beliefs/episodes in derived_from
    value_requires_protection: bool = True  # Source beliefs must be protected


# Singleton default config
DEFAULT_PROMOTION_GATES = PromotionGateConfig()


@dataclass
class PromotionGateResult:
    """Result of a promotion gate check for a single item."""

    passed: bool
    transition: str
    failures: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.passed:
            return "promotion gate passed"
        return "; ".join(self.failures)


@dataclass
class LayerConfig:
    """Configuration for a single layer transition."""

    layer_transition: str  # e.g., "raw_to_episode"
    enabled: bool = True
    model_id: Optional[str] = None  # Override model for this transition
    quantity_threshold: int = 10  # Min items to trigger
    valence_threshold: float = 3.0  # Emotional valence threshold
    time_threshold_hours: int = 24  # Max hours before forced processing
    batch_size: int = 10  # Max items per session
    max_sessions_per_day: int = 10  # Cost cap


# Default configs for each layer transition
DEFAULT_LAYER_CONFIGS: Dict[str, LayerConfig] = {
    "raw_to_episode": LayerConfig(
        layer_transition="raw_to_episode",
        quantity_threshold=10,
        batch_size=10,
    ),
    "raw_to_note": LayerConfig(
        layer_transition="raw_to_note",
        quantity_threshold=10,
        batch_size=10,
    ),
    "episode_to_belief": LayerConfig(
        layer_transition="episode_to_belief",
        quantity_threshold=5,
        batch_size=10,
    ),
    "episode_to_goal": LayerConfig(
        layer_transition="episode_to_goal",
        quantity_threshold=5,
        batch_size=10,
    ),
    "episode_to_relationship": LayerConfig(
        layer_transition="episode_to_relationship",
        quantity_threshold=3,
        batch_size=5,
    ),
    "belief_to_value": LayerConfig(
        layer_transition="belief_to_value",
        quantity_threshold=5,
        batch_size=10,
    ),
    "episode_to_drive": LayerConfig(
        layer_transition="episode_to_drive",
        quantity_threshold=5,
        batch_size=10,
    ),
}

# All valid layer transitions
# Deterministic execution order when iterating all transitions.
VALID_TRANSITION_ORDER = tuple(DEFAULT_LAYER_CONFIGS.keys())
VALID_TRANSITIONS = set(DEFAULT_LAYER_CONFIGS.keys())


# =============================================================================
# Results
# =============================================================================


@dataclass
class ProcessingResult:
    """Result of a single processing session."""

    layer_transition: str
    source_count: int  # How many source memories were processed
    created: List[Dict[str, str]] = field(
        default_factory=list
    )  # [{"type": "episode", "id": "..."}]
    suggestions: List[Dict[str, str]] = field(
        default_factory=list
    )  # [{"type": "episode", "id": "suggestion-id"}]
    errors: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    inference_blocked: bool = False  # True if blocked by no-inference safety
    auto_promote: bool = False  # Whether direct promotion was used
    deduplicated: int = 0  # Items skipped due to provenance/content match
    gate_blocked: int = 0  # Items blocked by promotion gates
    gate_details: List[str] = field(default_factory=list)  # Per-item gate failure messages
    no_output_processed: bool = False  # True when inference produced no parseable output


# =============================================================================
# Inference Safety Policy
# =============================================================================

# Identity-layer transitions that require inference to produce quality output.
# These write to layers that are "hard to change" — bad data here is costly.
IDENTITY_LAYER_TRANSITIONS = frozenset(
    {
        "episode_to_belief",
        "episode_to_goal",
        "episode_to_relationship",
        "episode_to_drive",
        "belief_to_value",
    }
)

# Transitions that are always blocked without inference — no override possible.
# Values are the highest identity layer; malformed values corrupt the entity.
NO_OVERRIDE_TRANSITIONS = frozenset(
    {
        "belief_to_value",
    }
)

# Transitions that can be overridden without inference if strict conditions are met.
# Beliefs require: force=True, explicit override flag, high confidence, evidence count.
OVERRIDE_TRANSITIONS = frozenset(
    {
        "episode_to_belief",
        "episode_to_goal",
        "episode_to_relationship",
        "episode_to_drive",
    }
)

# Minimum evidence count required for no-inference belief override
NO_INFERENCE_MIN_EVIDENCE = 3
# Minimum confidence required for no-inference belief override
NO_INFERENCE_MIN_CONFIDENCE = 0.9


# =============================================================================
# Prompts
# =============================================================================

LAYER_PROMPTS: Dict[str, Dict[str, str]] = {
    "raw_to_episode": {
        "system": (
            "You are a memory processing system. Your job is to structure raw "
            "memory captures into episodic memories. Each episode should have a clear "
            "objective (what was happening), outcome (what resulted), and optionally "
            "lessons learned. Group related raw entries into single episodes when "
            "they describe the same event or interaction."
        ),
        "template": (
            "Process these raw memory captures into structured episodes.\n\n"
            "RAW CAPTURES:\n{sources}\n\n"
            "EXISTING EPISODES (for deduplication):\n{context}\n\n"
            "For each episode, respond with a JSON array of objects:\n"
            '[{{"objective": "...", "outcome": "...", "outcome_type": "success|failure|neutral|mixed", '
            '"lessons": ["..."], "source_raw_ids": ["raw_id_1", "raw_id_2"]}}]\n\n'
            "Rules:\n"
            "- Each objective and outcome must be a self-contained description, "
            "meaningful without the original raw context\n"
            "- Group related raws into one episode\n"
            "- Skip raws that duplicate existing episodes\n"
            "- Each raw should appear in at most one episode\n"
            "- If no meaningful episodes can be formed, return an empty array []\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "raw_to_note": {
        "system": (
            "You are a memory processing system. Your job is to extract factual "
            "notes from raw memory captures. Notes are factual observations, "
            "references, or structured information — not experiences."
        ),
        "template": (
            "Extract factual notes from these raw memory captures.\n\n"
            "RAW CAPTURES:\n{sources}\n\n"
            "For each note, respond with a JSON array of objects:\n"
            '[{{"content": "...", "note_type": "observation|reference|procedure|fact", '
            '"source_raw_ids": ["raw_id_1"]}}]\n\n'
            "Rules:\n"
            "- Each note must be a self-contained statement, meaningful on its own "
            "without the original raw context\n"
            "- Only extract clear factual content, not experiences\n"
            "- Each raw can produce zero or more notes\n"
            "- If no clear facts are present, return an empty array []\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "episode_to_belief": {
        "system": (
            "You are a memory processing system. Your job is to identify beliefs "
            "that emerge from experiences. A belief is a general statement about "
            "how the world works, derived from specific episodes."
        ),
        "template": (
            "What beliefs emerge from these experiences?\n\n"
            "EPISODES:\n{sources}\n\n"
            "EXISTING BELIEFS (avoid duplicates):\n{context}\n\n"
            "For each belief, respond with a JSON array of objects:\n"
            '[{{"statement": "...", "belief_type": "causal|evaluative|procedural|factual", '
            '"confidence": 0.5-0.9, "source_episode_ids": ["ep_id_1"]}}]\n\n'
            "Rules:\n"
            "- Each statement must be a self-contained assertion that is meaningful "
            "without any surrounding context — not a response or fragment like 'yes' or 'it works'\n"
            "- Only create beliefs supported by multiple episodes or strong single episodes\n"
            "- Avoid duplicating existing beliefs\n"
            "- Confidence reflects how well-supported the belief is\n"
            "- If no substantial beliefs emerge, return an empty array []\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "episode_to_goal": {
        "system": (
            "You are a memory processing system. Your job is to identify goals "
            "that emerge from experiences. Goals are things the entity should pursue "
            "based on what it has learned."
        ),
        "template": (
            "What goals should be pursued based on these experiences?\n\n"
            "EPISODES:\n{sources}\n\n"
            "CURRENT GOALS:\n{context}\n\n"
            "For each goal, respond with a JSON array of objects:\n"
            '[{{"title": "...", "description": "...", "goal_type": "task|aspiration|commitment|exploration", '
            '"priority": "low|medium|high|critical", "source_episode_ids": ["ep_id_1"]}}]\n\n'
            "Rules:\n"
            "- Each title and description must be self-contained and meaningful "
            "without the original episode context\n"
            "- Only suggest actionable, concrete goals\n"
            "- Avoid duplicating existing goals\n"
            "- If no actionable goals emerge, return an empty array []\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "episode_to_relationship": {
        "system": (
            "You are a memory processing system. Your job is to identify or update "
            "relationships based on interactions recorded in episodes."
        ),
        "template": (
            "What relationships are revealed or updated by these interactions?\n\n"
            "EPISODES:\n{sources}\n\n"
            "EXISTING RELATIONSHIPS:\n{context}\n\n"
            "For each relationship, respond with a JSON array of objects:\n"
            '[{{"entity_name": "...", "entity_type": "person|org|system|other", '
            '"sentiment": -1.0 to 1.0, "context_note": "...", '
            '"source_episode_ids": ["ep_id_1"]}}]\n\n'
            "Rules:\n"
            "- Each entity_name and context_note must be self-contained and meaningful "
            "without the original episode context\n"
            "- Only create relationships with clearly identified entities\n"
            "- Update existing relationships rather than creating duplicates\n"
            "- If no clear relationships are present, return an empty array []\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "belief_to_value": {
        "system": (
            "You are a memory processing system. Your job is to identify core "
            "values that emerge from strongly-held beliefs. Values are fundamental "
            "principles that guide behavior."
        ),
        "template": (
            "Which of these beliefs represent core values?\n\n"
            "STRONG BELIEFS:\n{sources}\n\n"
            "EXISTING VALUES:\n{context}\n\n"
            "For each value, respond with a JSON array of objects:\n"
            '[{{"name": "...", "statement": "...", "priority": 1-100, '
            '"source_belief_ids": ["belief_id_1"]}}]\n\n'
            "Rules:\n"
            "- Each name and statement must be self-contained and meaningful "
            "without the original belief context\n"
            "- Only promote beliefs that are fundamental and enduring\n"
            "- Avoid duplicating existing values\n"
            "- Priority 1=minor, 100=core identity\n"
            "- If no beliefs rise to the level of core values, return an empty array []\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "episode_to_drive": {
        "system": (
            "You are a memory processing system. Your job is to identify "
            "motivational drives that emerge from experiences and beliefs."
        ),
        "template": (
            "What drives or motivations emerge from these experiences?\n\n"
            "EPISODES:\n{sources}\n\n"
            "CURRENT DRIVES:\n{context}\n\n"
            "For each drive, respond with a JSON array of objects:\n"
            '[{{"drive_type": "...", "intensity": 0.1-1.0, '
            '"source_episode_ids": ["ep_id_1"]}}]\n\n'
            "Rules:\n"
            "- Each drive_type must be a self-contained description, meaningful "
            "without the original episode context\n"
            "- Only identify clear motivational patterns\n"
            "- Avoid duplicating existing drives\n"
            "- If no clear drives emerge, return an empty array []\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
}


# =============================================================================
# Trigger Evaluation
# =============================================================================


def evaluate_triggers(
    transition: str,
    config: LayerConfig,
    unprocessed_count: int,
    cumulative_valence: float = 0.0,
    hours_since_last: Optional[float] = None,
) -> bool:
    """Check if processing should be triggered for a layer transition.

    Returns True if any trigger condition is met:
    - Quantity: unprocessed count >= quantity_threshold
    - Valence: cumulative emotional arousal >= valence_threshold
    - Time: hours since oldest unprocessed entry >= time_threshold_hours
    """
    if not config.enabled:
        return False

    # Quantity threshold
    if unprocessed_count >= config.quantity_threshold:
        return True

    # Emotional valence/arousal threshold
    if cumulative_valence >= config.valence_threshold:
        return True

    # Time threshold
    if hours_since_last is not None and config.time_threshold_hours > 0:
        if hours_since_last >= config.time_threshold_hours:
            return True

    return False


# =============================================================================
# Trigger Signal Helpers
# =============================================================================


def _extract_timestamp(obj: Any, attr: str) -> Optional[datetime]:
    """Safely extract a datetime attribute, returning None if not a real datetime."""
    ts = getattr(obj, attr, None)
    if isinstance(ts, datetime):
        return ts
    return None


def _oldest_timestamp(items: list, attr: str) -> Optional[datetime]:
    """Find the oldest datetime value for a given attribute across items."""
    oldest = None
    for item in items:
        ts = _extract_timestamp(item, attr)
        if ts is None:
            continue
        if oldest is None or ts < oldest:
            oldest = ts
    return oldest


def _hours_since(ts: Optional[datetime], now: datetime) -> Optional[float]:
    """Compute hours between a timestamp and now, handling naive datetimes."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = now - ts
    return delta.total_seconds() / 3600.0


def _hours_since_oldest_raw(entries: list, now: datetime) -> Optional[float]:
    """Compute hours since the oldest unprocessed raw entry was captured."""
    return _hours_since(_oldest_timestamp(entries, "captured_at"), now)


def _hours_since_oldest_episode(episodes: list, now: datetime) -> Optional[float]:
    """Compute hours since the oldest unprocessed episode was created."""
    return _hours_since(_oldest_timestamp(episodes, "created_at"), now)


def _hours_since_oldest_belief(beliefs: list, now: datetime) -> Optional[float]:
    """Compute hours since the oldest unprocessed belief was created."""
    return _hours_since(_oldest_timestamp(beliefs, "created_at"), now)


def _cumulative_arousal(episodes: list) -> float:
    """Sum emotional_arousal across unprocessed episodes.

    Episodes with high emotional arousal (e.g. a single intense event)
    can trigger consolidation even when the count is below threshold.
    """
    total = 0.0
    for ep in episodes:
        val = getattr(ep, "emotional_arousal", 0.0)
        if isinstance(val, (int, float)):
            total += val
    return total


# =============================================================================
# Memory Processor
# =============================================================================


class MemoryProcessor:
    """Runs focused inference sessions for memory processing.

    Owned by Entity. Uses InferenceService to run layer-specific
    processing sessions that promote memories up the hierarchy.

    Safety: When inference_available=False, identity-layer transitions
    are blocked. Values cannot be created. Beliefs and other identity
    layers require explicit override with evidence requirements.
    """

    def __init__(
        self,
        stack: Any,  # StackProtocol
        inference: Any,  # InferenceService
        core_id: str,
        configs: Optional[Dict[str, LayerConfig]] = None,
        inference_available: bool = True,
        auto_promote: bool = False,
        promotion_gates: Optional[PromotionGateConfig] = None,
    ) -> None:
        self._stack = stack
        self._inference = inference
        self._core_id = core_id
        self._configs = configs or dict(DEFAULT_LAYER_CONFIGS)
        self._inference_available = inference_available
        self._auto_promote = auto_promote
        self._promotion_gates = promotion_gates or DEFAULT_PROMOTION_GATES

    def update_config(self, transition: str, config: LayerConfig) -> None:
        """Update configuration for a layer transition."""
        self._configs[transition] = config

    def get_config(self, transition: str) -> Optional[LayerConfig]:
        """Get configuration for a layer transition."""
        return self._configs.get(transition)

    def check_triggers(self, transition: str) -> bool:
        """Check if processing should be triggered for a transition.

        Computes all three trigger signals from actual memory data:
        - Quantity: count of unprocessed source memories
        - Time: hours since the oldest unprocessed entry was captured/created
        - Valence: cumulative emotional arousal across unprocessed entries
        """
        config = self._configs.get(transition)
        if config is None or not config.enabled:
            return False

        backend = self._stack._backend if hasattr(self._stack, "_backend") else None
        if backend is None:
            return False

        now = datetime.now(timezone.utc)

        if transition in ("raw_to_episode", "raw_to_note"):
            unprocessed = backend.list_raw(processed=False, limit=config.quantity_threshold + 1)
            hours_since_last = _hours_since_oldest_raw(unprocessed, now)
            return evaluate_triggers(
                transition,
                config,
                len(unprocessed),
                hours_since_last=hours_since_last,
            )

        if transition in (
            "episode_to_belief",
            "episode_to_goal",
            "episode_to_relationship",
            "episode_to_drive",
        ):
            unprocessed = backend.get_episodes(limit=config.quantity_threshold + 1, processed=False)
            cumulative_valence = _cumulative_arousal(unprocessed)
            hours_since_last = _hours_since_oldest_episode(unprocessed, now)
            return evaluate_triggers(
                transition,
                config,
                len(unprocessed),
                cumulative_valence=cumulative_valence,
                hours_since_last=hours_since_last,
            )

        if transition == "belief_to_value":
            unprocessed = backend.get_beliefs(limit=config.quantity_threshold + 1, processed=False)
            hours_since_last = _hours_since_oldest_belief(unprocessed, now)
            return evaluate_triggers(
                transition,
                config,
                len(unprocessed),
                hours_since_last=hours_since_last,
            )

        return False

    def process(
        self,
        transition: Optional[str] = None,
        *,
        force: bool = False,
        allow_no_inference_override: bool = False,
        auto_promote: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> List[ProcessingResult]:
        """Run processing for one or all layer transitions.

        Args:
            transition: Specific transition to process (None = check all)
            force: Process even if triggers aren't met
            allow_no_inference_override: Allow identity-layer writes without
                inference (except values). Requires force=True and only works
                for transitions in OVERRIDE_TRANSITIONS.
            auto_promote: If True, directly write memories. If False (default),
                create suggestions for review. None uses instance default.
            batch_size: Override the per-transition batch size (None = use config).

        Returns:
            List of ProcessingResult for each transition that ran
        """
        results = []
        promote = auto_promote if auto_promote is not None else self._auto_promote

        # Generate a correlation_id for this entire process() run so all
        # audit entries from one invocation can be linked together.
        correlation_id = str(uuid.uuid4())

        transitions = [transition] if transition else list(VALID_TRANSITION_ORDER)

        for t in transitions:
            config = self._configs.get(t)
            if config is None or not config.enabled:
                continue

            # Inference safety gate — checked even when force=True
            blocked = self._check_inference_safety(t, force, allow_no_inference_override)
            if blocked is not None:
                results.append(blocked)
                continue

            if not force and not self.check_triggers(t):
                continue

            result = self._process_layer(
                t,
                config,
                auto_promote=promote,
                batch_size=batch_size,
                correlation_id=correlation_id,
            )
            results.append(result)

        return results

    def _check_inference_safety(
        self,
        transition: str,
        force: bool,
        allow_override: bool,
    ) -> Optional[ProcessingResult]:
        """Check if a transition is blocked by no-inference safety policy.

        Returns a ProcessingResult with inference_blocked=True if blocked,
        or None if the transition is allowed to proceed.
        """
        if self._inference_available:
            return None

        if transition not in IDENTITY_LAYER_TRANSITIONS:
            # Non-identity transitions (raw_to_episode, raw_to_note) require
            # inference to generate output. Block them at the policy level
            # with a clear message so callers (e.g. exhaust runner) can
            # detect inference-blocked state cleanly.
            return ProcessingResult(
                layer_transition=transition,
                source_count=0,
                skipped=True,
                skip_reason=(
                    "Blocked: inference unavailable. " "Bind a model to process raw entries."
                ),
                inference_blocked=True,
            )

        # Values are never allowed without inference
        if transition in NO_OVERRIDE_TRANSITIONS:
            return ProcessingResult(
                layer_transition=transition,
                source_count=0,
                skipped=True,
                skip_reason=(
                    "Blocked: inference unavailable. "
                    "Value creation requires inference — "
                    "cannot promote to identity layer without model."
                ),
                inference_blocked=True,
            )

        # Other identity-layer transitions can be overridden with explicit opt-in
        if transition in OVERRIDE_TRANSITIONS:
            if not (force and allow_override):
                return ProcessingResult(
                    layer_transition=transition,
                    source_count=0,
                    skipped=True,
                    skip_reason=(
                        "Blocked: inference unavailable. "
                        "Use force=True with allow_no_inference_override=True "
                        "to override for this transition."
                    ),
                    inference_blocked=True,
                )
            # Override allowed — log warning and proceed
            logger.warning("Processing %s without inference (override enabled)", transition)

        return None

    def _get_inference_model_id(self) -> Optional[str]:
        model_id = getattr(self._inference, "model_id", None)
        if not isinstance(model_id, str):
            return None
        value = model_id.strip()
        return value or None

    def _model_id_mismatch_reason(self, transition: str, model_id: Optional[str]) -> Optional[str]:
        if model_id is None:
            return None

        configured = str(model_id).strip()
        if not configured:
            return None

        current_model_id = self._get_inference_model_id()
        if current_model_id is None:
            return (
                f"Blocked: processing '{transition}' requires model '{configured}', "
                "but no model is bound."
            )

        if current_model_id != configured:
            return (
                f"Blocked: processing '{transition}' requires model '{configured}', "
                f"current model is '{current_model_id}'."
            )

        return None

    def _sessions_processed_today(self, transition: str) -> int:
        """Count today's processing sessions for this transition."""
        try:
            audit_entries = self._stack.get_audit_log(
                memory_type="processing",
                memory_id=transition,
                operation="process",
                limit=10_000,
            )
        except Exception as exc:
            logger.debug(
                "Unable to read processing audit log for %s: %s", transition, exc, exc_info=True
            )
            return 0

        if not isinstance(audit_entries, list):
            return 0

        today = datetime.now(timezone.utc).date()
        count = 0
        for entry in audit_entries:
            if not isinstance(entry, dict):
                continue
            created = _parse_created_at(entry.get("created_at"))
            if created is not None and created.date() == today:
                count += 1
        return count

    def _daily_session_cap_reason(
        self,
        transition: str,
        max_sessions_per_day: Optional[int],
    ) -> Optional[str]:
        if max_sessions_per_day is None:
            return None

        try:
            max_sessions = int(max_sessions_per_day)
        except (TypeError, ValueError):
            return None

        if max_sessions <= 0:
            return (
                f"Blocked: processing '{transition}' has max_sessions_per_day={max_sessions}. "
                "Set to a positive value to enable processing."
            )

        sessions_today = self._sessions_processed_today(transition)
        if sessions_today >= max_sessions:
            return (
                f"Blocked: max_sessions_per_day reached for '{transition}' "
                f"({sessions_today}/{max_sessions} for today)."
            )
        return None

    def _no_inference_override_reason(self, transition: str, sources: list) -> Optional[str]:
        if transition not in OVERRIDE_TRANSITIONS:
            return None

        if len(sources) < NO_INFERENCE_MIN_EVIDENCE:
            return (
                f"Blocked: no-inference override requires at least "
                f"{NO_INFERENCE_MIN_EVIDENCE} source memories for {transition}."
            )

        min_confidence = None
        for source in sources:
            confidence = _as_float(getattr(source, "confidence", None), default=1.0)
            if min_confidence is None or confidence < min_confidence:
                min_confidence = confidence

        if min_confidence is None or min_confidence < NO_INFERENCE_MIN_CONFIDENCE:
            current = 0.0 if min_confidence is None else min_confidence
            return (
                f"Blocked: no-inference override for {transition} requires minimum "
                f"confidence {NO_INFERENCE_MIN_CONFIDENCE:.2f}; minimum observed is "
                f"{current:.2f}."
            )

        return None

    def _check_promotion_gate(self, transition: str, item: dict) -> PromotionGateResult:
        """Check if a parsed item meets promotion gate criteria.

        Only applies to identity-layer transitions (episode_to_belief,
        belief_to_value). Other transitions always pass.

        Args:
            transition: The layer transition being processed.
            item: The parsed item from inference output.

        Returns:
            PromotionGateResult with pass/fail and failure details.
        """
        gates = self._promotion_gates
        failures: List[str] = []

        if transition == "episode_to_belief":
            # Evidence count check
            evidence = item.get("source_episode_ids", [])
            if len(evidence) < gates.belief_min_evidence:
                failures.append(
                    f"insufficient evidence: {len(evidence)} episodes "
                    f"(need >= {gates.belief_min_evidence})"
                )
            # Confidence floor check
            confidence = item.get("confidence", 0.7)
            if confidence < gates.belief_min_confidence:
                failures.append(
                    f"confidence too low: {confidence:.2f} "
                    f"(need >= {gates.belief_min_confidence:.2f})"
                )

        elif transition == "belief_to_value":
            # Evidence count check
            evidence = item.get("source_belief_ids", [])
            if len(evidence) < gates.value_min_evidence:
                failures.append(
                    f"insufficient evidence: {len(evidence)} beliefs "
                    f"(need >= {gates.value_min_evidence})"
                )
            # Protection flag check — verify source beliefs are protected
            if evidence:
                missing = []
                unprotected = []
                for bid in evidence:
                    belief = self._stack.get_memory("belief", bid)
                    if belief is None:
                        missing.append(bid)
                        continue
                    if belief and not getattr(belief, "is_protected", False):
                        unprotected.append(bid)
                if missing:
                    failures.append(
                        f"missing source beliefs: {', '.join(missing[:3])}"
                        + ("..." if len(missing) > 3 else "")
                    )
                if gates.value_requires_protection:
                    if unprotected:
                        failures.append(
                            f"unprotected source beliefs: {', '.join(unprotected[:3])}... "
                            f"(value promotion requires protected beliefs)"
                        )

        return PromotionGateResult(
            passed=len(failures) == 0,
            transition=transition,
            failures=failures,
        )

    def _process_layer(
        self,
        transition: str,
        config: LayerConfig,
        *,
        auto_promote: bool = False,
        batch_size: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> ProcessingResult:
        """Run one processing pass for a specific layer transition."""
        prompts = LAYER_PROMPTS.get(transition)
        if prompts is None:
            return ProcessingResult(
                layer_transition=transition,
                source_count=0,
                skipped=True,
                skip_reason=f"No prompts for transition: {transition}",
            )

        # 1. Gather unprocessed source memories
        effective_batch = batch_size if batch_size is not None else config.batch_size
        sources = self._gather_sources(transition, effective_batch)
        if not sources:
            return ProcessingResult(
                layer_transition=transition,
                source_count=0,
                skipped=True,
                skip_reason="No unprocessed sources",
            )

        model_mismatch = self._model_id_mismatch_reason(transition, config.model_id)
        if model_mismatch is not None:
            return ProcessingResult(
                layer_transition=transition,
                source_count=len(sources),
                skipped=True,
                skip_reason=model_mismatch,
            )

        session_limit_reason = self._daily_session_cap_reason(
            transition,
            config.max_sessions_per_day,
        )
        if session_limit_reason is not None:
            return ProcessingResult(
                layer_transition=transition,
                source_count=len(sources),
                skipped=True,
                skip_reason=session_limit_reason,
            )

        # 2. Load context (existing memories for dedup)
        if not self._inference_available and transition in OVERRIDE_TRANSITIONS:
            override_reason = self._no_inference_override_reason(transition, sources)
            if override_reason is not None:
                return ProcessingResult(
                    layer_transition=transition,
                    source_count=len(sources),
                    skipped=True,
                    skip_reason=override_reason,
                    inference_blocked=True,
                )

            return ProcessingResult(
                layer_transition=transition,
                source_count=len(sources),
                skipped=True,
                skip_reason=(
                    "Blocked: no inference available for identity-layer override transitions."
                ),
                inference_blocked=True,
            )

        context = self._gather_context(transition)

        # 3. Build prompt
        sources_text = self._format_sources(transition, sources)
        context_text = self._format_context(transition, context)
        prompt = prompts["template"].format(sources=sources_text, context=context_text)
        system = prompts["system"]

        # 4. Call inference
        try:
            response = self._inference.infer(prompt, system=system)
        except Exception as e:
            logger.error("Processing inference failed for %s: %s", transition, e, exc_info=True)
            return ProcessingResult(
                layer_transition=transition,
                source_count=len(sources),
                errors=[f"Inference failed: {e}"],
            )

        # 5. Parse response
        result = ProcessingResult(
            layer_transition=transition,
            source_count=len(sources),
            auto_promote=auto_promote,
        )

        try:
            parsed = self._parse_response(response)
        except Exception as e:
            logger.error(
                "Failed to parse processing response for %s: %s", transition, e, exc_info=True
            )
            result.errors.append(f"Parse failed: {e}")
            return result

        # 6. Build dedup index from target memories for idempotent transitions
        dedup_targets = self._gather_dedup_targets(transition)
        dedup_index = self._build_existing_index(transition, dedup_targets)

        # 7. Write memories or create suggestions (dedup-aware)
        if auto_promote:
            # Direct promotion (opt-in): write memories immediately
            created = self._write_memories(
                transition,
                parsed,
                sources,
                dedup_index,
                correlation_id=correlation_id,
            )
            result.created = created
        else:
            # Default: create suggestions for review
            suggestions = self._write_suggestions(
                transition,
                parsed,
                sources,
                correlation_id=correlation_id,
            )
            result.suggestions = suggestions
        result.errors.extend(getattr(self, "_last_write_errors", []))
        result.deduplicated = getattr(self, "_last_deduplicated", 0)
        result.gate_blocked = getattr(self, "_last_gate_blocked", 0)
        result.gate_details = getattr(self, "_last_gate_details", [])

        # 8. Mark sources as processed.
        # If items were gate-blocked (conditions may improve later), keep
        # sources unprocessed so they can be re-evaluated.
        created_or_suggested = result.created or result.suggestions
        result.no_output_processed = not bool(parsed)
        all_gate_blocked = result.gate_blocked > 0 and not created_or_suggested
        had_write_errors = bool(getattr(self, "_last_write_errors", []))
        if not all_gate_blocked and not had_write_errors and not result.no_output_processed:
            self._mark_processed(transition, sources, created_or_suggested or [])

        # 9. Log audit
        self._stack.log_audit(
            "processing",
            transition,
            "process",
            actor=f"core:{self._core_id}",
            details={
                "source_count": len(sources),
                "created_count": len(result.created),
                "suggestion_count": len(result.suggestions),
                "auto_promote": auto_promote,
                "deduplicated": result.deduplicated,
                "gate_blocked": result.gate_blocked,
                "gate_details": result.gate_details,
                "errors": result.errors,
            },
            correlation_id=correlation_id,
        )

        return result

    # ---- Source Gathering ----

    def _gather_sources(self, transition: str, batch_size: int) -> list:
        """Gather unprocessed source memories for a transition."""
        backend = self._stack._backend if hasattr(self._stack, "_backend") else None
        if backend is None:
            return []

        if transition in ("raw_to_episode", "raw_to_note"):
            return backend.list_raw(processed=False, limit=batch_size)

        if transition in (
            "episode_to_belief",
            "episode_to_goal",
            "episode_to_relationship",
            "episode_to_drive",
        ):
            return backend.get_episodes(limit=batch_size, processed=False)

        if transition == "belief_to_value":
            return backend.get_beliefs(limit=batch_size, processed=False)

        return []

    def _gather_context(self, transition: str) -> list:
        """Load existing memories for deduplication context."""
        if transition in ("raw_to_episode", "raw_to_note"):
            return self._stack.get_episodes(limit=20)

        if transition == "episode_to_belief":
            return self._stack.get_beliefs(limit=20)

        if transition == "episode_to_goal":
            return self._stack.get_goals(limit=20)

        if transition == "episode_to_relationship":
            return self._stack.get_relationships()

        if transition == "belief_to_value":
            return self._stack.get_values(limit=20)

        if transition == "episode_to_drive":
            return self._stack.get_drives()

        return []

    def _gather_dedup_targets(self, transition: str) -> list:
        """Load existing target-type memories for deduplication.

        Unlike _gather_context (which provides prompt context for the LLM),
        this queries the actual target memory type that the transition produces.
        For example, raw_to_note produces notes, so we query notes here.
        """
        if transition == "raw_to_episode":
            return self._stack.get_episodes(limit=100)

        if transition == "raw_to_note":
            return self._stack.get_notes(limit=100)

        if transition == "episode_to_belief":
            return self._stack.get_beliefs(limit=100)

        if transition == "episode_to_goal":
            return self._stack.get_goals(limit=100)

        if transition == "episode_to_relationship":
            return self._stack.get_relationships()

        if transition == "belief_to_value":
            return self._stack.get_values(limit=100)

        if transition == "episode_to_drive":
            return self._stack.get_drives()

        return []

    # ---- Formatting ----

    def _format_sources(self, transition: str, sources: list) -> str:
        """Format source memories for the prompt."""
        lines = []
        for s in sources:
            if transition in ("raw_to_episode", "raw_to_note"):
                blob = getattr(s, "blob", None) or getattr(s, "content", "") or ""
                lines.append(f"[{s.id}] {blob[:500]}")
            elif hasattr(s, "objective"):  # Episode
                lines.append(f"[{s.id}] {s.objective}: {s.outcome}")
            elif hasattr(s, "statement"):  # Belief
                lines.append(f"[{s.id}] {s.statement} (confidence: {s.confidence})")
            else:
                lines.append(f"[{s.id}] {str(s)[:200]}")
        return "\n".join(lines) if lines else "(none)"

    def _format_context(self, transition: str, context: list) -> str:
        """Format context memories for deduplication."""
        if not context:
            return "(none)"
        lines = []
        for c in context:
            if hasattr(c, "objective"):  # Episode
                lines.append(f"- {c.objective}: {c.outcome}")
            elif hasattr(c, "statement") and hasattr(c, "belief_type"):  # Belief
                lines.append(f"- {c.statement}")
            elif hasattr(c, "statement") and hasattr(c, "name"):  # Value
                lines.append(f"- {c.name}: {c.statement}")
            elif hasattr(c, "title"):  # Goal
                desc = c.title
                if c.description:
                    desc += f": {c.description}"
                lines.append(f"- {desc}")
            elif hasattr(c, "drive_type"):  # Drive
                lines.append(f"- {c.drive_type} (intensity: {c.intensity})")
            elif hasattr(c, "entity_name"):  # Relationship
                lines.append(f"- {c.entity_name} ({c.entity_type})")
            else:
                lines.append(f"- {str(c)[:100]}")
        return "\n".join(lines[:10])

    # ---- Response Parsing ----

    def _parse_response(self, response: str) -> list:
        """Parse the model's JSON array response."""
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)

    # ---- Deduplication ----

    def _build_existing_index(self, transition: str, context: list) -> Dict[str, Any]:
        """Build an index of existing memories for deduplication.

        Returns a dict with content hashes for content-based dedup only.
        Provenance-based dedup is intentionally absent — the same source
        material should be allowed to produce multiple memories (e.g. an
        episode AND a note from the same raw entry, or different insights
        from re-processing). Near-duplicate cleanup is left to the
        consolidation component as a higher-level cognitive activity.

        The index covers memories already present in the stack that could
        be exact duplicates of what this processing pass would create.
        """
        content_index: Dict[str, Any] = {}  # content_hash -> record

        for record in context:
            # Build content hash based on what type of record this is
            content_text = ""
            if hasattr(record, "objective"):
                content_text = f"{record.objective} {record.outcome}"
            elif hasattr(record, "statement") and hasattr(record, "belief_type"):
                content_text = record.statement
            elif hasattr(record, "content") and hasattr(record, "note_type"):
                content_text = record.content
            elif hasattr(record, "title"):
                content_text = f"{record.title} {getattr(record, 'description', '') or ''}"
            elif hasattr(record, "drive_type"):
                content_text = record.drive_type
            elif hasattr(record, "entity_name"):
                content_text = record.entity_name
            elif hasattr(record, "name") and hasattr(record, "statement"):
                content_text = f"{record.name} {record.statement}"

            if content_text:
                chash = compute_content_hash(content_text)
                if chash:
                    content_index[chash] = record

        return {"content": content_index}

    def _check_duplicate(
        self, transition: str, item: dict, dedup_index: Dict[str, Any]
    ) -> Optional[str]:
        """Check if a parsed item is an exact duplicate of an existing memory.

        Policy: only skip creation when content is identical. Provenance
        is not checked — the same source material may legitimately produce
        multiple memories (different layers, different insights, different
        framings). Near-duplicate consolidation is a separate cognitive
        activity handled by the consolidation component.

        Returns:
            None if not a duplicate.
            The existing memory ID if it's a duplicate (skip creation).
        """
        content_text = _extract_content_text(transition, item)

        # Check content hash match — only block exact duplicates
        if content_text:
            chash = compute_content_hash(content_text)
            if chash and chash in dedup_index["content"]:
                existing = dedup_index["content"][chash]
                existing_id = getattr(existing, "id", None)
                logger.info(
                    "Dedup: skipping %s item — content match with %s",
                    transition,
                    existing_id,
                )
                return existing_id

        return None

    def _suggestion_fingerprint(self, memory_type: str, item: dict, source_ids: List[str]) -> str:
        """Build a stable fingerprint for suggestion deduplication.

        Confidence and source-ID order are ignored so retries that produce
        equivalent suggestions map to the same fingerprint.
        """
        if not isinstance(item, dict):
            return ""

        canonical_content: Dict[str, Any] = {}
        for key, value in item.items():
            if key in {"confidence", "source_raw_ids", "source_episode_ids", "source_belief_ids"}:
                continue
            canonical_content[key] = value

        payload = {
            "memory_type": memory_type,
            "content": canonical_content,
            "source_ids": sorted(set(source_ids)),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _load_pending_suggestion_fingerprints(self, memory_type: str) -> set[str]:
        """Load fingerprints for existing pending suggestions of a memory type."""
        try:
            pending = self._stack.get_suggestions(
                status="pending",
                memory_type=memory_type,
                limit=10000,
            )
        except Exception as e:
            logger.debug("Failed to load pending suggestions for dedup: %s", e, exc_info=True)
            return set()

        if not isinstance(pending, list):
            return set()

        fingerprints: set[str] = set()
        for suggestion in pending:
            if isinstance(suggestion, dict):
                content = suggestion.get("content")
                source_ids = suggestion.get("source_raw_ids", [])
            else:
                content = getattr(suggestion, "content", None)
                source_ids = getattr(suggestion, "source_raw_ids", [])

            if not isinstance(content, dict):
                continue
            if not isinstance(source_ids, list):
                source_ids = []

            fp = self._suggestion_fingerprint(memory_type, content, source_ids)
            if fp:
                fingerprints.add(fp)

        return fingerprints

    # ---- Memory Writing ----

    def _write_memories(
        self,
        transition: str,
        parsed: list,
        sources: list,
        dedup_index: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Write parsed memories through the stack with provenance.

        Args:
            transition: The layer transition being processed.
            parsed: List of parsed items from inference.
            sources: Source memories that were processed.
            dedup_index: Optional dedup index from _build_existing_index.
                If provided, items are checked for duplicates before writing.

        Returns:
            List of {"type": ..., "id": ...} dicts for created memories.
        """
        from kernle.types import Belief, Drive, Episode, Goal, Note, Relationship, Value

        created = []
        self._last_deduplicated = 0
        self._last_gate_blocked = 0
        self._last_gate_details: List[str] = []
        self._last_write_errors: List[str] = []
        now = datetime.now(timezone.utc)

        for item in parsed:
            try:
                # Check for duplicates if dedup index is available
                if dedup_index is not None:
                    existing_id = self._check_duplicate(transition, item, dedup_index)
                    if existing_id is not None:
                        self._last_deduplicated += 1
                        continue

                # Check promotion gates for identity-layer transitions
                gate_result = self._check_promotion_gate(transition, item)
                if not gate_result.passed:
                    self._last_gate_blocked += 1
                    self._last_gate_details.append(gate_result.summary)
                    logger.info(
                        "Promotion gate blocked %s item: %s",
                        transition,
                        gate_result.summary,
                    )
                    continue

                if transition == "raw_to_episode":
                    raw_ids = item.get("source_raw_ids", [])
                    derived_from = [f"raw:{rid}" for rid in raw_ids]
                    ep = Episode(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        objective=item["objective"],
                        outcome=item["outcome"],
                        outcome_type=item.get("outcome_type", "neutral"),
                        lessons=item.get("lessons"),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    eid = self._stack.save_episode(ep)
                    created.append({"type": "episode", "id": eid})

                elif transition == "raw_to_note":
                    raw_ids = item.get("source_raw_ids", [])
                    derived_from = [f"raw:{rid}" for rid in raw_ids]
                    note = Note(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        content=item["content"],
                        note_type=item.get("note_type", "observation"),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    nid = self._stack.save_note(note)
                    created.append({"type": "note", "id": nid})

                elif transition == "episode_to_belief":
                    ep_ids = item.get("source_episode_ids", [])
                    derived_from = [f"episode:{eid}" for eid in ep_ids]
                    belief = Belief(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        statement=item["statement"],
                        belief_type=item.get("belief_type", "factual"),
                        confidence=item.get("confidence", 0.7),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    bid = self._stack.save_belief(belief)
                    created.append({"type": "belief", "id": bid})

                elif transition == "episode_to_goal":
                    ep_ids = item.get("source_episode_ids", [])
                    derived_from = [f"episode:{eid}" for eid in ep_ids]
                    goal = Goal(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        title=item.get("title", item.get("description", "")),
                        description=item.get("description"),
                        goal_type=item.get("goal_type", "task"),
                        priority=item.get("priority", "medium"),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    gid = self._stack.save_goal(goal)
                    created.append({"type": "goal", "id": gid})

                elif transition == "episode_to_relationship":
                    ep_ids = item.get("source_episode_ids", [])
                    derived_from = [f"episode:{eid}" for eid in ep_ids]
                    rel = Relationship(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        entity_name=item["entity_name"],
                        entity_type=item.get("entity_type", "person"),
                        relationship_type=item.get("relationship_type", "acquaintance"),
                        notes=item.get("context_note"),
                        sentiment=item.get("sentiment", 0.0),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    rid = self._stack.save_relationship(rel)
                    created.append({"type": "relationship", "id": rid})

                elif transition == "belief_to_value":
                    belief_ids = item.get("source_belief_ids", [])
                    derived_from = [f"belief:{bid}" for bid in belief_ids]
                    value = Value(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        name=item["name"],
                        statement=item.get("statement", item["name"]),
                        priority=item.get("priority", 50),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    vid = self._stack.save_value(value)
                    created.append({"type": "value", "id": vid})

                elif transition == "episode_to_drive":
                    ep_ids = item.get("source_episode_ids", [])
                    derived_from = [f"episode:{eid}" for eid in ep_ids]
                    drive = Drive(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        drive_type=item.get("drive_type") or "motivation",
                        intensity=item.get("intensity") or 0.5,
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    did = self._stack.save_drive(drive)
                    created.append({"type": "drive", "id": did})

                # Emit memory.promoted audit event for direct writes
                if created:
                    promoted = created[-1]
                    self._stack.log_audit(
                        promoted["type"],
                        promoted["id"],
                        "memory.promoted",
                        actor=f"core:{self._core_id}",
                        details={"transition": transition, "source_id": promoted["id"]},
                        correlation_id=correlation_id,
                    )

                # Update dedup index with newly created memory so subsequent
                # items in this batch are checked against it (intra-batch dedup).
                if dedup_index is not None and created:
                    new_entry = created[-1]
                    new_id = new_entry["id"]

                    # Create a lightweight record for the index
                    class _Ref:
                        pass

                    ref = _Ref()
                    ref.id = new_id

                    content_text = _extract_content_text(transition, item)
                    if content_text:
                        chash = compute_content_hash(content_text)
                        if chash:
                            dedup_index["content"][chash] = ref

            except Exception as e:
                logger.error("Failed to write %s memory: %s", transition, e, exc_info=True)
                self._last_write_errors.append(f"Write failed for {transition}: {e}")

        return created

    def _write_suggestions(
        self,
        transition: str,
        parsed: list,
        sources: list,
        correlation_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Create MemorySuggestions instead of directly writing memories.

        Returns list of dicts like [{"type": "episode", "id": "suggestion-id"}]
        where id is the suggestion ID (not a memory ID).
        """
        from kernle.types import SUGGESTION_MEMORY_TYPES, MemorySuggestion

        suggestions_created: List[Dict[str, str]] = []
        self._last_deduplicated = 0
        self._last_gate_blocked = 0
        self._last_gate_details: List[str] = []
        self._last_write_errors: List[str] = []
        now = datetime.now(timezone.utc)

        # Map transitions to their target memory type
        transition_to_type = {
            "raw_to_episode": "episode",
            "raw_to_note": "note",
            "episode_to_belief": "belief",
            "episode_to_goal": "goal",
            "episode_to_relationship": "relationship",
            "belief_to_value": "value",
            "episode_to_drive": "drive",
        }
        memory_type = transition_to_type.get(transition)
        if memory_type is None:
            return suggestions_created
        assert (
            memory_type in SUGGESTION_MEMORY_TYPES
        ), f"transition_to_type produced '{memory_type}' not in SUGGESTION_MEMORY_TYPES"
        seen_fingerprints = self._load_pending_suggestion_fingerprints(memory_type)

        for item in parsed:
            try:
                # Check promotion gates for identity-layer transitions
                gate_result = self._check_promotion_gate(transition, item)
                if not gate_result.passed:
                    self._last_gate_blocked += 1
                    self._last_gate_details.append(gate_result.summary)
                    logger.info(
                        "Promotion gate blocked %s suggestion: %s",
                        transition,
                        gate_result.summary,
                    )
                    continue

                # Extract source IDs for provenance
                source_ids = self._extract_source_ids(transition, item)
                fingerprint = self._suggestion_fingerprint(memory_type, item, source_ids)
                if fingerprint and fingerprint in seen_fingerprints:
                    self._last_deduplicated += 1
                    logger.info(
                        "Dedup: skipping %s suggestion — matching pending suggestion already exists",
                        transition,
                    )
                    continue

                suggestion = MemorySuggestion(
                    id=str(uuid.uuid4()),
                    stack_id=self._stack.stack_id,
                    memory_type=memory_type,
                    content=item,
                    confidence=item.get("confidence", 0.7),
                    source_raw_ids=source_ids,
                    status="pending",
                    created_at=now,
                )
                sid = self._stack.save_suggestion(suggestion)
                suggestions_created.append({"type": memory_type, "id": sid})
                self._stack.log_audit(
                    "suggestion",
                    sid,
                    "suggestion.created",
                    details={"transition": transition, "source_id": sid},
                    correlation_id=correlation_id,
                )
                if fingerprint:
                    seen_fingerprints.add(fingerprint)

            except Exception as e:
                logger.error("Failed to create %s suggestion: %s", transition, e, exc_info=True)
                self._last_write_errors.append(f"Suggestion write failed for {transition}: {e}")

        return suggestions_created

    def _extract_source_ids(self, transition: str, item: dict) -> List[str]:
        """Extract source memory IDs from a parsed item based on transition type."""
        if transition in ("raw_to_episode", "raw_to_note"):
            return item.get("source_raw_ids", [])
        if transition in (
            "episode_to_belief",
            "episode_to_goal",
            "episode_to_relationship",
            "episode_to_drive",
        ):
            return [f"episode:{eid}" for eid in item.get("source_episode_ids", [])]
        if transition == "belief_to_value":
            return [f"belief:{bid}" for bid in item.get("source_belief_ids", [])]
        return []

    # ---- Mark Processed ----

    def _mark_processed(
        self, transition: str, sources: list, created: List[Dict[str, str]]
    ) -> None:
        """Mark source memories as processed."""
        backend = self._stack._backend if hasattr(self._stack, "_backend") else None
        if backend is None:
            return

        created_refs = [f"{c['type']}:{c['id']}" for c in created]

        if transition in ("raw_to_episode", "raw_to_note"):
            for raw in sources:
                backend.mark_raw_processed(raw.id, created_refs)
        elif transition in (
            "episode_to_belief",
            "episode_to_goal",
            "episode_to_relationship",
            "episode_to_drive",
        ):
            for ep in sources:
                backend.mark_episode_processed(ep.id)
        elif transition == "belief_to_value":
            for belief in sources:
                backend.mark_belief_processed(belief.id)
