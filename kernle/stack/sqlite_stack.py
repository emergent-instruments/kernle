"""Stack - StackProtocol implementation with pluggable storage backend.

This is the default memory stack implementation. It accepts any Storage
protocol implementation and adds:
- StackProtocol interface conformance
- Feature mixins (consolidation, emotions, forgetting, knowledge,
  metamemory, suggestions)
- Composition hooks (on_attach, on_detach, on_model_changed)
- Component registry infrastructure (stack components)

The storage backend is injected via the constructor. Use Stack.from_sqlite()
for the common local-agent case with SQLite.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, get_args

from kernle.features import (
    ConsolidationMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    SuggestionsMixin,
)
from kernle.protocols import (
    DumpFormat,
    GoalStatus,
    InferenceService,
    MaintenanceModeError,
    ProcessingTransition,
    ProvenanceError,
    SearchRecordType,
    StackComponentProtocol,
    StackState,
    SuggestionMemoryType,
    SuggestionStatus,
)
from kernle.protocols import (
    SearchResult as ProtocolSearchResult,
)
from kernle.protocols import (
    SyncResult as ProtocolSyncResult,
)
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import (
    VALID_SOURCE_TYPE_VALUES,
    Belief,
    Drive,
    EntityModel,
    Episode,
    Epoch,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    RawEntry,
    Relationship,
    SelfNarrative,
    Summary,
    TrustAssessment,
    Value,
)

logger = logging.getLogger(__name__)

# Default token budget for memory loading
DEFAULT_TOKEN_BUDGET = 8000
MAX_TOKEN_BUDGET = 50000
MIN_TOKEN_BUDGET = 100
DEFAULT_MAX_ITEM_CHARS = 500
TOKEN_ESTIMATION_SAFETY_MARGIN = 1.3

# Priority scores for each memory type (higher = more important)
MEMORY_TYPE_PRIORITIES = {
    "value": 0.90,
    "self_narrative": 0.90,
    "summary_decade": 0.95,
    "summary_epoch": 0.85,
    "summary_year": 0.80,
    "belief": 0.70,
    "goal": 0.65,
    "drive": 0.60,
    "summary_quarter": 0.50,
    "episode": 0.40,
    "summary_month": 0.35,
    "note": 0.35,
    "relationship": 0.30,
}


# Strength tier thresholds (documented in memory-integrity.md)
STRENGTH_FORGOTTEN = 0.0  # Tombstoned — only via recover()
STRENGTH_DORMANT = 0.2  # Only via explicit include_dormant=True
STRENGTH_WEAK = 0.5  # Excluded from load(), still searchable
STRENGTH_FADING = 0.8  # Included but reduced priority

VALID_GOAL_STATUSES = frozenset(get_args(GoalStatus))
VALID_SUGGESTION_STATUSES = frozenset(get_args(SuggestionStatus))
VALID_SUGGESTION_MEMORY_TYPES = frozenset(get_args(SuggestionMemoryType))
VALID_SEARCH_RECORD_TYPES = frozenset(get_args(SearchRecordType))
VALID_PROCESSING_TRANSITIONS = frozenset(get_args(ProcessingTransition))
VALID_DUMP_FORMATS = frozenset(get_args(DumpFormat))


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars/token with safety margin)."""
    if not text:
        return 0
    return int(len(text) // 4 * TOKEN_ESTIMATION_SAFETY_MARGIN)


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary with ellipsis."""
    if not text or len(text) <= max_chars:
        return text
    target = max_chars - 3
    if target <= 0:
        return "..."
    truncated = text[:target]
    last_space = truncated.rfind(" ")
    if last_space > target // 2:
        truncated = truncated[:last_space]
    return truncated + "..."


def _compute_priority_score(memory_type: str, record: Any) -> float:
    """Compute priority score for budget-based selection."""
    base = MEMORY_TYPE_PRIORITIES.get(memory_type, 0.20)
    bonus = 0.0
    if memory_type == "value":
        bonus = getattr(record, "priority", 50) / 1000.0
    elif memory_type == "belief":
        bonus = getattr(record, "confidence", 0.8) / 10.0
    elif memory_type == "drive":
        bonus = getattr(record, "intensity", 0.5) / 10.0

    # Reduce priority for fading memories (strength 0.5-0.8)
    strength = getattr(record, "strength", 1.0)
    if strength < STRENGTH_FADING:
        return (base + bonus) * 0.5
    return base + bonus


def _normalize_suggestion_provenance_refs(source_refs: Optional[List[str]]) -> List[str]:
    """Normalize suggestion source refs to typed provenance refs.

    Backward compatibility:
    - Plain IDs are treated as raw IDs -> "raw:<id>"
    - Typed refs ("episode:<id>", "belief:<id>", etc.) are preserved
    """
    normalized: List[str] = []
    seen = set()
    for ref in source_refs or []:
        if not isinstance(ref, str):
            continue
        ref = ref.strip()
        if not ref:
            continue
        if ":" not in ref:
            ref = f"raw:{ref}"
        ref_type, ref_id = ref.split(":", 1)
        if not ref_type or not ref_id:
            continue
        canonical = f"{ref_type}:{ref_id}"
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def _sorted_literals(values: frozenset[str]) -> str:
    return ", ".join(sorted(values))


def _validate_literal_value(value: Optional[str], allowed: frozenset[str], field: str) -> None:
    if value is None:
        return
    if value not in allowed:
        raise ValueError(
            f"Invalid {field}: {value!r}. Valid values are: {_sorted_literals(allowed)}"
        )


def _validate_record_types(record_types: Optional[list[str] | tuple[str, ...]]) -> None:
    if record_types is None:
        return
    if not isinstance(record_types, (list, tuple)):
        raise ValueError("record_types must be a sequence of memory record type strings")
    for record_type in record_types:
        if not isinstance(record_type, str):
            raise ValueError("record_types values must be strings")
        _validate_literal_value(record_type, VALID_SEARCH_RECORD_TYPES, "search record type")


# Provenance hierarchy: which source types are allowed for each memory type
PROVENANCE_RULES: Dict[str, List[str]] = {
    "episode": ["raw"],
    "note": ["raw"],
    "belief": ["episode", "note"],
    "goal": ["episode", "belief"],
    "relationship": ["episode"],
    "value": ["belief"],
    "drive": ["episode", "belief"],
}

# Annotation ref types: valid in derived_from but not traversable provenance.
# These are metadata markers (e.g., "context:cli", "kernle:system") that
# indicate how/where a memory was created, not what it was derived from.
# Lineage.py also skips these during cycle detection.
ANNOTATION_REF_TYPES = {"context", "kernle"}


class Stack(
    ConsolidationMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    SuggestionsMixin,
):
    """Memory stack conforming to StackProtocol with pluggable storage.

    Accepts any Storage protocol implementation via the ``storage`` parameter.
    Use ``Stack.from_sqlite()`` for the common local-agent case.

    Works in detached mode (no core attached) for read/write/search.
    Feature mixins are applied the same way as on Kernle.
    """

    def __init__(
        self,
        stack_id: str,
        *,
        storage: Any,  # Storage protocol — Any to avoid circular import
        components: Optional[List[StackComponentProtocol]] = None,
        enforce_provenance: bool = True,
        lint_on_save: bool = True,
    ):
        self._backend = storage
        # Alias for mixin compatibility (mixins access self._storage)
        self._storage = self._backend

        # Component registry (v0.5.0 infrastructure)
        self._components: Dict[str, StackComponentProtocol] = {}

        # Composition state
        self._attached_core_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._enforce_provenance = enforce_provenance
        self._lint_on_save = lint_on_save
        self._registered_plugins: set = set()

        # Load persisted state or default to INITIALIZING
        persisted_state = self._backend.get_stack_setting("stack_state")
        if persisted_state and persisted_state in StackState.__members__:
            self._state: StackState = StackState[persisted_state]
        else:
            self._state = StackState.INITIALIZING

        # Load enforce_provenance from settings if not explicitly set
        if not enforce_provenance:
            persisted_provenance = self._backend.get_stack_setting("enforce_provenance")
            if persisted_provenance == "true":
                self._enforce_provenance = True

        # Auto-load components: None = defaults, [] = bare, list = explicit
        if components is None:
            from kernle.stack.components import get_default_components

            components = get_default_components()
        for component in components:
            self.add_component(component)

        # Bootstrap self-trust if missing (e.g. after migration creates the table)
        self._ensure_self_trust()

    @classmethod
    def from_sqlite(
        cls,
        stack_id: str,
        *,
        db_path: Optional[Path] = None,
        cloud_storage: Optional[Any] = None,
        embedder: Optional[Any] = None,
        components: Optional[List[StackComponentProtocol]] = None,
        enforce_provenance: bool = True,
    ) -> "Stack":
        """Create a Stack backed by SQLiteStorage (convenience factory).

        This is the common path for local agents (Claude Code, Codex, etc.).
        For other backends, construct Stack directly with ``storage=``.

        Args:
            stack_id: Unique identifier for the stack
            db_path: Path to SQLite database (auto-resolved if None)
            cloud_storage: Optional cloud storage for hybrid search
            embedder: Optional embedding provider
            components: Stack components (None = defaults, [] = bare)
            enforce_provenance: Whether to enforce provenance on writes
        """
        storage = SQLiteStorage(
            stack_id=stack_id,
            db_path=db_path,
            cloud_storage=cloud_storage,
            embedder=embedder,
        )
        return cls(
            stack_id=stack_id,
            storage=storage,
            components=components,
            enforce_provenance=enforce_provenance,
        )

    def _ensure_self_trust(self) -> None:
        """Bootstrap self-trust assessment if missing after migration."""
        existing = self._backend.get_trust_assessment("identity")
        if not existing:
            assessment = TrustAssessment(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                entity="identity",
                dimensions={"general": {"score": 1.0}},
                authority=[{"scope": "all"}],
                evidence_episode_ids=[],
                created_at=datetime.now(timezone.utc),
            )
            self._backend.save_trust_assessment(assessment)

    # ---- Properties ----

    @property
    def stack_id(self) -> str:
        return self._backend.stack_id

    @property
    def schema_version(self) -> int:
        from kernle.storage.sqlite import SCHEMA_VERSION

        return SCHEMA_VERSION

    # ---- Component Management ----

    @property
    def components(self) -> Dict[str, StackComponentProtocol]:
        return dict(self._components)

    def add_component(self, component: StackComponentProtocol) -> None:
        """Add a component to this stack.

        Components are maintained in priority order (lower priority runs first).
        """
        name = component.name
        if name in self._components:
            raise ValueError(f"Component '{name}' already registered")
        component.attach(self.stack_id, self._inference)
        if hasattr(component, "set_storage"):
            component.set_storage(self._storage)
        self._components[name] = component
        self._sort_components()

    def remove_component(self, name: str) -> None:
        """Remove a component by name."""
        if name not in self._components:
            raise ValueError(f"Component '{name}' not found")
        component = self._components[name]
        if component.required:
            raise ValueError(f"Cannot remove required component '{name}'")
        component.detach()
        del self._components[name]

    def get_component(self, name: str) -> Optional[StackComponentProtocol]:
        return self._components.get(name)

    def _sort_components(self) -> None:
        """Re-sort _components dict by priority (lower = runs first)."""

        def _get_priority(item):
            val = getattr(item[1], "priority", 200)
            return val if isinstance(val, (int, float)) else 200

        sorted_items = sorted(self._components.items(), key=_get_priority)
        self._components = dict(sorted_items)

    # ---- Plugin Registration ----

    def register_plugin(self, plugin_name: str) -> None:
        """Register a plugin name as trusted for provenance bypass."""
        was_registered = plugin_name in self._registered_plugins
        self._registered_plugins.add(plugin_name)
        self.log_audit(
            "provenance_policy",
            plugin_name,
            "register_plugin",
            actor=self._attached_core_id or "system",
            details={
                "plugin": plugin_name,
                "already_registered": was_registered,
                "stack_id": self.stack_id,
            },
        )
        if not was_registered:
            logger.info(
                "provenance policy update: plugin=%s action=register actor=%s at=%s",
                plugin_name,
                self._attached_core_id or "system",
                datetime.now(timezone.utc).isoformat(),
            )

    def unregister_plugin(self, plugin_name: str) -> None:
        """Remove a plugin from the trusted set."""
        was_registered = plugin_name in self._registered_plugins
        self._registered_plugins.discard(plugin_name)
        self.log_audit(
            "provenance_policy",
            plugin_name,
            "unregister_plugin",
            actor=self._attached_core_id or "system",
            details={
                "plugin": plugin_name,
                "was_registered": was_registered,
                "stack_id": self.stack_id,
            },
        )
        if was_registered:
            logger.info(
                "provenance policy update: plugin=%s action=unregister actor=%s at=%s",
                plugin_name,
                self._attached_core_id or "system",
                datetime.now(timezone.utc).isoformat(),
            )

    def _log_partial_failure(self, component_name: str, hook_name: str, exc: Exception) -> None:
        """Log a component hook failure with standardized structured fields.

        All partial-failure paths should use this helper so that log
        consumers can reliably filter on ``component``, ``hook``, and
        ``error_type`` extra fields.
        """
        logger.warning(
            "Component %s %s failed: %s — memory operation continues without component",
            component_name,
            hook_name,
            exc,
            extra={
                "component": component_name,
                "hook": hook_name,
                "error_type": type(exc).__name__,
            },
        )

    def _set_inference_for_components(self, inference: Optional[InferenceService]) -> None:
        for component in self._components.values():
            try:
                component.set_inference(inference)
            except Exception as exc:
                self._log_partial_failure(component.name, "set_inference", exc)

    def maintenance(self) -> Dict[str, Any]:
        """Run maintenance on all components."""
        results: Dict[str, Any] = {}
        for name, component in self._components.items():
            try:
                stats = component.on_maintenance()
                if stats:
                    results[name] = stats
            except Exception as e:
                self._log_partial_failure(name, "on_maintenance", e)
                results[name] = {"error": str(e)}
        return results

    # ---- Component Dispatch Helpers ----

    def _dispatch_on_save(self, memory_type: str, memory_id: str, memory: Any) -> None:
        """Notify components after a memory is saved, persisting any returned metadata."""
        for name, component in self._components.items():
            try:
                result = component.on_save(memory_type, memory_id, memory)
                if result and isinstance(result, dict):
                    self._persist_on_save_metadata(memory_type, memory_id, result)
            except Exception as e:
                self._log_partial_failure(name, "on_save", e)

    def _persist_on_save_metadata(self, memory_type: str, memory_id: str, metadata: dict) -> None:
        """Persist metadata returned by on_save components.

        Handles two categories of component output:
        1. Schema-mapped fields (e.g., emotional_valence on episodes) are
           persisted directly to the record's row.
        2. Advisory metadata (e.g., trust_warning, contradictions) is logged
           for observability but not written to storage since there are no
           corresponding schema columns.
        """
        # Advisory keys that components return for observability.
        # These are logged rather than persisted to schema columns.
        advisory_keys = {"trust_warning", "trust_level", "contradictions", "domain"}

        # Log advisory metadata so it's observable (not silently dropped)
        advisory = {k: v for k, v in metadata.items() if k in advisory_keys}
        if advisory:
            logger.info(
                "Component advisory on save (%s:%s): %s",
                memory_type,
                memory_id[:12],
                advisory,
            )

        # Persist schema-mapped fields via protocol method (no _connect() leak)
        schema_fields = {k: v for k, v in metadata.items() if k not in advisory_keys}
        if schema_fields:
            self._backend.update_memory_metadata(memory_type, memory_id, **schema_fields)

    def _dispatch_on_search(
        self, query: str, results: List[ProtocolSearchResult]
    ) -> List[ProtocolSearchResult]:
        """Let components modify search results."""
        for name, component in self._components.items():
            try:
                modified = component.on_search(query, results)
                if modified is not None:
                    results = modified
            except Exception as e:
                self._log_partial_failure(name, "on_search", e)
        return results

    def _dispatch_on_load(self, context: Dict[str, Any]) -> None:
        """Notify components when working memory is loaded."""
        for name, component in self._components.items():
            try:
                component.on_load(context)
            except Exception as e:
                self._log_partial_failure(name, "on_load", e)

    # ---- Strength Tier Filtering ----

    # ---- Lazy Decay-on-Read ----

    def _apply_lazy_decay(self, records: list, memory_type: str) -> list:
        """Apply time-based strength decay to retrieved records.

        Computes what strength WOULD be based on elapsed time since
        last_accessed, updates records in-place, and persists changes
        for records that differ meaningfully (>0.001) from stored value.

        Protected records are never decayed.

        Args:
            records: List of memory records to decay.
            memory_type: The memory type string (episode, belief, etc.).

        Returns:
            The same list with updated strength values.
        """
        # Check if lazy decay is enabled
        lazy_decay_setting = self.get_stack_setting("lazy_decay")
        if lazy_decay_setting == "false":
            return records

        from kernle.stack.components.forgetting import compute_decayed_strength

        strength_updates: list[tuple[str, str, float]] = []

        for record in records:
            # Skip protected records
            if getattr(record, "is_protected", False):
                continue

            current_strength = getattr(record, "strength", 1.0)
            # Skip already-forgotten records
            if current_strength <= 0.0:
                continue

            new_strength = compute_decayed_strength(memory_type, record)

            if abs(new_strength - current_strength) > 0.001:
                # Update the record object in-place
                if hasattr(record, "strength"):
                    object.__setattr__(record, "strength", new_strength)
                if hasattr(record, "last_accessed"):
                    object.__setattr__(record, "last_accessed", datetime.now(timezone.utc))
                strength_updates.append((memory_type, record.id, new_strength))

        # Persist all updates in a single batch, also updating last_accessed
        # so the next read doesn't re-decay from the same reference time.
        if strength_updates:
            try:
                self._persist_decay_updates(strength_updates)
            except Exception as e:
                self._log_partial_failure("lazy_decay", "persist_updates", e)

        return records

    def _persist_decay_updates(self, updates: list[tuple[str, str, float]]) -> None:
        """Persist lazy decay strength updates via protocol method.

        Delegates to the backend's update_strength_batch, which handles
        clamping, last_accessed bumps, and transactional writes.
        """
        if not updates:
            return
        self._backend.update_strength_batch(updates)

    @staticmethod
    def _filter_by_strength(
        memories: list,
        include_forgotten: bool = False,
        include_weak: bool = False,
    ) -> list:
        """Filter memories by strength tier.

        Tiers:
          0.8-1.0 Strong — always included
          0.5-0.8 Fading — always included (reduced priority in load)
          0.2-0.5 Weak   — excluded unless include_weak=True
          0.0-0.2 Dormant — excluded unless include_forgotten=True
          0.0     Forgotten — excluded unless include_forgotten=True
        """
        if include_forgotten:
            return memories
        if include_weak:
            min_strength = STRENGTH_DORMANT
        else:
            min_strength = STRENGTH_WEAK
        return [m for m in memories if getattr(m, "strength", 1.0) >= min_strength]

    # ---- State Management ----

    @property
    def state(self) -> StackState:
        """Current lifecycle state of the stack."""
        return self._state

    def enter_maintenance(self) -> None:
        """Enter maintenance mode. Only controlled admin operations allowed."""
        if self._state == StackState.MAINTENANCE:
            return
        self._state = StackState.MAINTENANCE
        self._backend.set_stack_setting("stack_state", StackState.MAINTENANCE.name)

    def exit_maintenance(self) -> None:
        """Exit maintenance mode, returning to ACTIVE state."""
        if self._state != StackState.MAINTENANCE:
            return
        self._state = StackState.ACTIVE
        self._backend.set_stack_setting("stack_state", StackState.ACTIVE.name)

    # ---- Provenance Validation ----

    def _check_maintenance(self, operation: str) -> None:
        """Block writes during maintenance mode.

        Lighter check than _validate_provenance — used for lifecycle/metadata
        writes (epochs, summaries, narratives, entity models) that don't need
        provenance validation but should respect maintenance mode.

        Raises:
            MaintenanceModeError: If stack is in maintenance mode.
        """
        if self._state == StackState.MAINTENANCE:
            raise MaintenanceModeError(
                f"Cannot save {operation} in maintenance mode. Use exit_maintenance() first."
            )

    def _validate_provenance(
        self, memory_type: str, derived_from: Optional[list], source_entity: Optional[str] = None
    ) -> None:
        """Validate provenance for a memory write.

        Only enforced when stack is ACTIVE. INITIALIZING allows any write
        (for seed data). MAINTENANCE always rejects writes (independent of
        provenance flag). Plugin-sourced writes have relaxed provenance
        requirements but are still blocked in maintenance mode.

        Raises:
            ProvenanceError: If provenance is missing or invalid
            MaintenanceModeError: If stack is in maintenance mode
        """
        if self._state == StackState.INITIALIZING:
            return  # Seed writes don't need provenance

        # Maintenance mode always blocks writes, regardless of provenance flag
        if self._state == StackState.MAINTENANCE:
            raise MaintenanceModeError(
                f"Cannot save {memory_type} in maintenance mode. " "Use exit_maintenance() first."
            )

        if not self._enforce_provenance:
            return  # Provenance enforcement disabled

        # System-generated writes (kernle:*) bypass provenance.
        # These are internal operations (checkpoints, maintenance, sync)
        # that produce valid memories without raw-entry derivation.
        if source_entity and source_entity.startswith("kernle:"):
            return

        # Pre-v0.9 migrated memories bypass provenance.
        # These existed before provenance enforcement and have been
        # annotated by `kernle migrate backfill-provenance`.
        if derived_from:
            for ref in derived_from:
                if ref and ref.startswith("kernle:pre-v0.9"):
                    return

        # Plugin-sourced writes have relaxed provenance requirements,
        # but only for plugins actually registered with this stack
        if source_entity and source_entity.startswith("plugin:"):
            plugin_name = source_entity[len("plugin:") :]
            if plugin_name in self._registered_plugins:
                return

        # Raw entries don't need provenance
        if memory_type not in PROVENANCE_RULES:
            return

        allowed_types = PROVENANCE_RULES[memory_type]

        if not derived_from:
            raise ProvenanceError(
                f"Cannot save {memory_type} without provenance. "
                f"derived_from must cite at least one: {', '.join(allowed_types)}"
            )

        has_real_ref = False
        for ref in derived_from:
            if ":" not in ref:
                raise ProvenanceError(
                    f"Invalid provenance reference '{ref}'. "
                    "Expected format 'type:id' (e.g., 'episode:abc123')"
                )
            ref_type, ref_id = ref.split(":", 1)
            # Annotation refs (context:, kernle:) are metadata markers,
            # not provenance sources. Skip hierarchy/existence checks.
            if ref_type in ANNOTATION_REF_TYPES:
                continue
            has_real_ref = True
            if ref_type not in allowed_types:
                raise ProvenanceError(
                    f"Invalid provenance for {memory_type}: '{ref_type}' is not an allowed source. "
                    f"Allowed sources: {', '.join(allowed_types)}"
                )
            # Verify the referenced memory exists
            if not self._backend.memory_exists(ref_type, ref_id):
                raise ProvenanceError(f"Referenced {ref_type}:{ref_id} does not exist in the stack")

        # Must have at least one real provenance ref (not just annotations)
        if not has_real_ref:
            raise ProvenanceError(
                f"Cannot save {memory_type} with only annotation refs. "
                f"derived_from must cite at least one: {', '.join(allowed_types)}"
            )

    def _validate_source_type(self, memory_type: str, source_type: str) -> None:
        """Reject unknown source_type values when provenance is enforced.

        Only validated during ACTIVE state with enforce_provenance=True.
        INITIALIZING allows any value (for seed/migration data).
        """
        if self._state != StackState.ACTIVE:
            return
        if not self._enforce_provenance:
            return
        if source_type and source_type not in VALID_SOURCE_TYPE_VALUES:
            raise ProvenanceError(
                f"Unknown source_type '{source_type}' on {memory_type}. "
                f"Valid values: {', '.join(sorted(VALID_SOURCE_TYPE_VALUES))}"
            )

    # ---- Write Operations ----

    def save_episode(self, episode: Episode) -> str:
        self._validate_provenance(
            "episode", episode.derived_from, getattr(episode, "source_entity", None)
        )
        self._validate_source_type("episode", episode.source_type)
        result_id = self._backend.save_episode(episode)
        self._dispatch_on_save("episode", result_id, episode)
        return result_id

    def save_belief(self, belief: Belief) -> str:
        self._validate_provenance(
            "belief", belief.derived_from, getattr(belief, "source_entity", None)
        )
        self._validate_source_type("belief", belief.source_type)
        # Lint check: reject malformed or low-signal beliefs (ACTIVE state + lint enabled)
        if self._state == StackState.ACTIVE and self._lint_on_save:
            lint_result = self._lint_belief(belief)
            if not lint_result.passed:
                return self._redirect_to_suggestion("belief", belief, lint_result)
        result_id = self._backend.save_belief(belief)
        self._dispatch_on_save("belief", result_id, belief)
        return result_id

    def save_value(self, value: Value) -> str:
        self._validate_provenance(
            "value", value.derived_from, getattr(value, "source_entity", None)
        )
        self._validate_source_type("value", value.source_type)
        # Lint check: reject malformed or low-signal values (ACTIVE state + lint enabled)
        if self._state == StackState.ACTIVE and self._lint_on_save:
            lint_result = self._lint_value(value)
            if not lint_result.passed:
                return self._redirect_to_suggestion("value", value, lint_result)
        result_id = self._backend.save_value(value)
        self._dispatch_on_save("value", result_id, value)
        return result_id

    def save_goal(self, goal: Goal) -> str:
        self._validate_provenance("goal", goal.derived_from, getattr(goal, "source_entity", None))
        self._validate_source_type("goal", goal.source_type)
        result_id = self._backend.save_goal(goal)
        self._dispatch_on_save("goal", result_id, goal)
        return result_id

    def save_note(self, note: Note) -> str:
        self._validate_provenance("note", note.derived_from, getattr(note, "source_entity", None))
        self._validate_source_type("note", note.source_type)
        result_id = self._backend.save_note(note)
        self._dispatch_on_save("note", result_id, note)
        return result_id

    def save_drive(self, drive: Drive) -> str:
        self._validate_provenance(
            "drive", drive.derived_from, getattr(drive, "source_entity", None)
        )
        self._validate_source_type("drive", drive.source_type)
        result_id = self._backend.save_drive(drive)
        self._dispatch_on_save("drive", result_id, drive)
        return result_id

    def save_relationship(self, relationship: Relationship) -> str:
        self._validate_provenance(
            "relationship", relationship.derived_from, getattr(relationship, "source_entity", None)
        )
        self._validate_source_type("relationship", relationship.source_type)
        result_id = self._backend.save_relationship(relationship)
        self._dispatch_on_save("relationship", result_id, relationship)
        return result_id

    def update_goal_atomic(self, goal: Goal):
        self._validate_provenance("goal", goal.derived_from, getattr(goal, "source_entity", None))
        self._validate_source_type("goal", goal.source_type)
        self._backend.update_goal_atomic(goal)
        self._dispatch_on_save("goal", goal.id, goal)

    def update_drive_atomic(self, drive: Drive, expected_version: Optional[int] = None) -> bool:
        self._validate_provenance(
            "drive", drive.derived_from, getattr(drive, "source_entity", None)
        )
        self._validate_source_type("drive", drive.source_type)
        result = self._backend.update_drive_atomic(drive, expected_version=expected_version)
        self._dispatch_on_save("drive", drive.id, drive)
        return result

    def get_drive(self, drive_type: str):
        """Delegate read to backend for consistent strict-mode path."""
        return self._backend.get_drive(drive_type)

    def get_relationship(self, entity_name: str):
        """Delegate read to backend for consistent strict-mode path."""
        return self._backend.get_relationship(entity_name)

    def update_relationship_atomic(
        self, relationship: Relationship, expected_version: Optional[int] = None
    ) -> bool:
        self._validate_provenance(
            "relationship", relationship.derived_from, getattr(relationship, "source_entity", None)
        )
        self._validate_source_type("relationship", relationship.source_type)
        result = self._backend.update_relationship_atomic(
            relationship, expected_version=expected_version
        )
        self._dispatch_on_save("relationship", relationship.id, relationship)
        return result

    def save_raw(self, raw: RawEntry) -> str:
        self._validate_provenance("raw", None)  # Raw entries need no provenance
        result_id = self._backend.save_raw(
            blob=raw.blob or "",
            source=raw.source,
        )
        self._dispatch_on_save("raw", result_id, raw)
        return result_id

    def save_playbook(self, playbook: Playbook) -> str:
        result_id = self._backend.save_playbook(playbook)
        self._dispatch_on_save("playbook", result_id, playbook)
        return result_id

    def save_epoch(self, epoch: Epoch) -> str:
        self._check_maintenance("epoch")
        result_id = self._backend.save_epoch(epoch)
        self._dispatch_on_save("epoch", result_id, epoch)
        return result_id

    def close_epoch(self, epoch_id: str, summary: Optional[str] = None) -> bool:
        self._check_maintenance("epoch")
        return self._backend.close_epoch(epoch_id, summary=summary)

    def save_summary(self, summary: Summary) -> str:
        self._check_maintenance("summary")
        result_id = self._backend.save_summary(summary)
        self._dispatch_on_save("summary", result_id, summary)
        return result_id

    def save_self_narrative(self, narrative: SelfNarrative) -> str:
        self._check_maintenance("narrative")
        result_id = self._backend.save_self_narrative(narrative)
        self._dispatch_on_save("self_narrative", result_id, narrative)
        return result_id

    def deactivate_self_narratives(self, stack_id: str, narrative_type: str) -> int:
        self._check_maintenance("narrative")
        return self._backend.deactivate_self_narratives(stack_id, narrative_type)

    def save_entity_model(self, model: EntityModel) -> str:
        self._check_maintenance("entity_model")
        return self._backend.save_entity_model(model)

    def save_suggestion(self, suggestion: MemorySuggestion) -> str:
        result_id = self._backend.save_suggestion(suggestion)
        self._dispatch_on_save("suggestion", result_id, suggestion)
        return result_id

    def get_suggestion(self, suggestion_id: str) -> Optional[MemorySuggestion]:
        return self._backend.get_suggestion(suggestion_id)

    def get_suggestions(
        self,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
        min_confidence: Optional[float] = None,
        max_age_hours: Optional[float] = None,
        source_raw_id: Optional[str] = None,
    ) -> List[MemorySuggestion]:
        _validate_literal_value(status, VALID_SUGGESTION_STATUSES, "suggestion status")
        _validate_literal_value(
            memory_type, VALID_SUGGESTION_MEMORY_TYPES, "suggestion memory_type"
        )
        return self._backend.get_suggestions(
            status=status,
            memory_type=memory_type,
            limit=limit,
            min_confidence=min_confidence,
            max_age_hours=max_age_hours,
            source_raw_id=source_raw_id,
        )

    def accept_suggestion(
        self,
        suggestion_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Accept a pending suggestion and promote it to a structured memory.

        Creates the target memory (episode/belief/note) with full provenance,
        sets promoted_to on the suggestion, and logs an audit event.

        Returns the ID of the created memory, or None if the suggestion
        was not found or not pending.
        """
        suggestion = self._backend.get_suggestion(suggestion_id)
        if not suggestion or suggestion.status != "pending":
            return None

        content = suggestion.content.copy()
        if modifications:
            content.update(modifications)

        memory_id = None
        memory_type = suggestion.memory_type

        # Preserve typed provenance when suggestions come from non-raw transitions.
        derived_from = _normalize_suggestion_provenance_refs(suggestion.source_raw_ids)

        if memory_type == "episode":
            ep = Episode(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                objective=content.get("objective", ""),
                outcome=content.get("outcome", ""),
                outcome_type=content.get("outcome_type", "unknown"),
                lessons=content.get("lessons", []),
                tags=["auto-suggested"],
                derived_from=derived_from,
                source_type="processing",
                source_entity="kernle:suggestion-promotion",
                created_at=datetime.now(timezone.utc),
            )
            memory_id = self.save_episode(ep)
        elif memory_type == "belief":
            belief = Belief(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                statement=content.get("statement", ""),
                belief_type=content.get("belief_type", "fact"),
                confidence=content.get("confidence", 0.7),
                derived_from=derived_from,
                source_type="processing",
                source_entity="kernle:suggestion-promotion",
                created_at=datetime.now(timezone.utc),
            )
            memory_id = self.save_belief(belief)
        elif memory_type == "note":
            note = Note(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                content=content.get("content", ""),
                note_type=content.get("note_type", "note"),
                speaker=content.get("speaker"),
                reason=content.get("reason"),
                tags=["auto-suggested"],
                derived_from=derived_from,
                source_type="processing",
                source_entity="kernle:suggestion-promotion",
                created_at=datetime.now(timezone.utc),
            )
            memory_id = self.save_note(note)
        elif memory_type == "goal":
            goal = Goal(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                title=content.get("title", ""),
                description=content.get("description"),
                goal_type=content.get("goal_type", "task"),
                priority=content.get("priority", "medium"),
                status=content.get("status", "active"),
                derived_from=derived_from,
                source_type="processing",
                source_entity="kernle:suggestion-promotion",
                created_at=datetime.now(timezone.utc),
            )
            memory_id = self.save_goal(goal)
        elif memory_type == "value":
            value = Value(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                name=content.get("name", ""),
                statement=content.get("statement", ""),
                priority=content.get("priority", 50),
                derived_from=derived_from,
                source_type="processing",
                source_entity="kernle:suggestion-promotion",
                created_at=datetime.now(timezone.utc),
            )
            memory_id = self.save_value(value)
        elif memory_type == "relationship":
            relationship = Relationship(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                entity_name=content.get("entity_name", ""),
                entity_type=content.get("entity_type", "unknown"),
                relationship_type=content.get("relationship_type", ""),
                notes=content.get("notes"),
                sentiment=content.get("sentiment", 0.0),
                derived_from=derived_from,
                source_type="processing",
                source_entity="kernle:suggestion-promotion",
                created_at=datetime.now(timezone.utc),
            )
            memory_id = self.save_relationship(relationship)
        elif memory_type == "drive":
            drive = Drive(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                drive_type=content.get("drive_type", ""),
                intensity=content.get("intensity", 0.5),
                focus_areas=content.get("focus_areas"),
                derived_from=derived_from,
                source_type="processing",
                source_entity="kernle:suggestion-promotion",
                created_at=datetime.now(timezone.utc),
            )
            memory_id = self.save_drive(drive)
        else:
            raise ValueError(f"Unsupported suggestion type: {memory_type}")

        # Check if save was lint-redirected (not actually created)
        if isinstance(memory_id, str) and memory_id.startswith("suggestion:"):
            # Lint rejected — original suggestion stays pending, new suggestion was created
            return None

        if memory_id:
            status = "modified" if modifications else "promoted"
            self._backend.update_suggestion_status(
                suggestion_id=suggestion_id,
                status=status,
                promoted_to=f"{memory_type}:{memory_id}",
            )
            # Mark only true raw refs as processed.
            for raw_ref in derived_from:
                ref_type, ref_id = raw_ref.split(":", 1)
                if ref_type != "raw":
                    continue
                self._backend.mark_raw_processed(
                    raw_id=ref_id,
                    processed_into=[f"{memory_type}:{memory_id}"],
                )
            # Audit
            self.log_audit(
                "suggestion",
                suggestion_id,
                "suggestion.resolved",
                details={
                    "resolution": "accepted",
                    "suggestion_id": suggestion_id,
                    "promoted_to": f"{memory_type}:{memory_id}",
                    "memory_type": memory_type,
                    "had_modifications": bool(modifications),
                },
            )

        return memory_id

    def dismiss_suggestion(
        self,
        suggestion_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Dismiss a pending suggestion (will not be promoted).

        Sets status to 'dismissed' with an optional reason. Logs an audit event.
        """
        suggestion = self._backend.get_suggestion(suggestion_id)
        if not suggestion or suggestion.status != "pending":
            return False

        result = self._backend.update_suggestion_status(
            suggestion_id=suggestion_id,
            status="dismissed",
            resolution_reason=reason,
        )
        if result:
            self.log_audit(
                "suggestion",
                suggestion_id,
                "suggestion.resolved",
                details={
                    "resolution": "dismissed",
                    "suggestion_id": suggestion_id,
                    "reason": reason,
                },
            )
        return result

    def expire_suggestions(self, max_age_hours: float = 168.0) -> List[str]:
        """Auto-dismiss pending suggestions older than max_age_hours.

        Logs an audit event for each expired suggestion.

        Args:
            max_age_hours: Age threshold in hours (default: 168 = 7 days)

        Returns:
            List of expired suggestion IDs
        """
        expired_ids = self._backend.expire_suggestions(max_age_hours=max_age_hours)
        for sid in expired_ids:
            self.log_audit(
                "suggestion",
                sid,
                "suggestion.resolved",
                details={
                    "resolution": "expired",
                    "suggestion_id": sid,
                    "max_age_hours": max_age_hours,
                },
            )
        return expired_ids

    # ---- Batch Write ----

    def save_episodes_batch(self, episodes: List[Episode]) -> List[str]:
        for ep in episodes:
            self._validate_provenance(
                "episode", ep.derived_from, getattr(ep, "source_entity", None)
            )
            self._validate_source_type("episode", ep.source_type)
        ids = self._backend.save_episodes_batch(episodes)
        for ep, eid in zip(episodes, ids):
            self._dispatch_on_save("episode", eid, ep)
        return ids

    def save_beliefs_batch(self, beliefs: List[Belief]) -> List[str]:
        for belief in beliefs:
            self._validate_provenance(
                "belief", belief.derived_from, getattr(belief, "source_entity", None)
            )
            self._validate_source_type("belief", belief.source_type)
        ids = self._backend.save_beliefs_batch(beliefs)
        for belief, bid in zip(beliefs, ids):
            self._dispatch_on_save("belief", bid, belief)
        return ids

    def save_notes_batch(self, notes: List[Note]) -> List[str]:
        for note in notes:
            self._validate_provenance(
                "note", note.derived_from, getattr(note, "source_entity", None)
            )
            self._validate_source_type("note", note.source_type)
        ids = self._backend.save_notes_batch(notes)
        for note, nid in zip(notes, ids):
            self._dispatch_on_save("note", nid, note)
        return ids

    # ---- Read Operations ----

    def get_episodes(
        self,
        *,
        limit: int = 50,
        tags: Optional[List[str]] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
        include_weak: bool = False,
    ) -> List[Episode]:
        episodes = self._backend.get_episodes(limit=limit, tags=tags)
        episodes = self._apply_lazy_decay(episodes, "episode")
        episodes = self._filter_by_strength(episodes, include_forgotten, include_weak)
        return episodes

    def get_beliefs(
        self,
        *,
        limit: int = 50,
        belief_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
        include_weak: bool = False,
    ) -> List[Belief]:
        beliefs = self._backend.get_beliefs(limit=limit)
        if belief_type:
            beliefs = [b for b in beliefs if b.belief_type == belief_type]
        if min_confidence is not None:
            beliefs = [b for b in beliefs if b.confidence >= min_confidence]
        beliefs = self._apply_lazy_decay(beliefs, "belief")
        beliefs = self._filter_by_strength(beliefs, include_forgotten, include_weak)
        return beliefs

    def get_values(
        self,
        *,
        limit: int = 50,
        context: Optional[str] = None,
        include_forgotten: bool = False,
        include_weak: bool = False,
    ) -> List[Value]:
        values = self._backend.get_values(limit=limit)
        values = self._apply_lazy_decay(values, "value")
        values = self._filter_by_strength(values, include_forgotten, include_weak)
        return values

    def get_goals(
        self,
        *,
        limit: int = 50,
        status: Optional[str] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
        include_weak: bool = False,
    ) -> List[Goal]:
        _validate_literal_value(status, VALID_GOAL_STATUSES, "goal status")
        goals = self._backend.get_goals(status=status, limit=limit)
        goals = self._apply_lazy_decay(goals, "goal")
        goals = self._filter_by_strength(goals, include_forgotten, include_weak)
        return goals

    def get_notes(
        self,
        *,
        limit: int = 50,
        note_type: Optional[str] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
        include_weak: bool = False,
    ) -> List[Note]:
        notes = self._backend.get_notes(limit=limit, note_type=note_type)
        notes = self._apply_lazy_decay(notes, "note")
        notes = self._filter_by_strength(notes, include_forgotten, include_weak)
        return notes

    def get_drives(self, *, include_expired: bool = False) -> List[Drive]:
        drives = self._backend.get_drives()
        if include_expired:
            return drives
        return [drive for drive in drives if getattr(drive, "strength", 0.0) > STRENGTH_DORMANT]

    def get_relationships(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_trust: Optional[float] = None,
    ) -> List[Relationship]:
        rels = self._backend.get_relationships(entity_type=entity_type)
        if entity_id:
            rels = [r for r in rels if r.entity_name == entity_id]
        if min_trust is not None:
            rels = [r for r in rels if (r.sentiment + 1) / 2 >= min_trust]
        return rels

    def get_raw(
        self,
        *,
        limit: int = 50,
        tags: Optional[List[str]] = None,
    ) -> List[RawEntry]:
        entries = self._backend.list_raw(limit=limit)
        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if e.tags and tag_set.intersection(e.tags)]
        return entries

    def get_memory(self, memory_type: str, memory_id: str) -> Optional[Any]:
        return self._backend.get_memory(memory_type, memory_id)

    # ---- Search ----

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        context: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[ProtocolSearchResult]:
        _validate_record_types(record_types)
        storage_results = self._backend.search(
            query=query,
            limit=limit,
            record_types=record_types,
        )

        # Apply lazy decay to search result records
        lazy_decay_setting = self.get_stack_setting("lazy_decay")
        if lazy_decay_setting != "false":
            from kernle.stack.components.forgetting import compute_decayed_strength

            decay_updates: list[tuple[str, str, float]] = []
            for sr in storage_results:
                record = sr.record
                if getattr(record, "is_protected", False):
                    continue
                current_strength = getattr(record, "strength", 1.0)
                if current_strength <= 0.0:
                    continue
                new_strength = compute_decayed_strength(sr.record_type, record)
                if abs(new_strength - current_strength) > 0.001:
                    if hasattr(record, "strength"):
                        object.__setattr__(record, "strength", new_strength)
                    if hasattr(record, "last_accessed"):
                        object.__setattr__(record, "last_accessed", datetime.now(timezone.utc))
                    decay_updates.append((sr.record_type, record.id, new_strength))
            if decay_updates:
                try:
                    self._persist_decay_updates(decay_updates)
                except Exception as e:
                    self._log_partial_failure("lazy_decay", "persist_search_updates", e)

        results = []
        for sr in storage_results:
            record = sr.record
            content = ""
            if sr.record_type == "episode":
                content = f"{record.objective}: {record.outcome}"
            elif sr.record_type == "belief":
                content = record.statement
            elif sr.record_type == "value":
                content = f"{record.name}: {record.statement}"
            elif sr.record_type == "goal":
                content = f"{record.title}: {record.description or ''}"
            elif sr.record_type == "note":
                content = record.content
            elif sr.record_type == "relationship":
                content = f"{record.entity_name}: {record.notes or ''}"
            else:
                content = str(record)[:200]

            # Exclude dormant/forgotten memories from search (weak OK)
            record_strength = getattr(record, "strength", 1.0)
            if record_strength < STRENGTH_DORMANT:
                continue

            if min_confidence is not None:
                record_conf = getattr(record, "confidence", 1.0)
                if record_conf < min_confidence:
                    continue

            results.append(
                ProtocolSearchResult(
                    memory_type=sr.record_type,
                    memory_id=record.id,
                    content=content,
                    score=sr.score,
                    metadata={
                        "confidence": getattr(record, "confidence", None),
                    },
                )
            )
        results = self._dispatch_on_search(query, results)
        return results

    # ---- Working Memory ----

    def load(
        self,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        context: Optional[str] = None,
        epoch_id: Optional[str] = None,
        max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
        track_access: bool = True,
    ) -> Dict[str, Any]:
        """Assemble working memory within a token budget.

        Args:
            token_budget: Token budget for memory assembly.
            context: Optional context string (unused, for future use).
            epoch_id: If set, filter candidates to this specific epoch.
            max_item_chars: Max characters per item when truncating.
            track_access: If True, record access for salience tracking.
        """
        budget = max(MIN_TOKEN_BUDGET, min(MAX_TOKEN_BUDGET, token_budget))
        remaining = budget

        # Fetch candidates from all types
        batched = self._backend.load_all(
            values_limit=None,
            beliefs_limit=None,
            goals_limit=None,
            goals_status="active",
            episodes_limit=None,
            notes_limit=None,
            drives_limit=None,
            relationships_limit=None,
            epoch_id=epoch_id,
        )

        if batched is None:
            # Fallback to individual queries
            result = {
                "values": [
                    {"id": v.id, "name": v.name, "statement": v.statement, "priority": v.priority}
                    for v in self._backend.get_values(limit=50)
                ],
                "beliefs": [
                    {"id": b.id, "statement": b.statement, "confidence": b.confidence}
                    for b in self._backend.get_beliefs(limit=50)
                ],
                "goals": [
                    {"id": g.id, "title": g.title, "status": g.status}
                    for g in self._backend.get_goals(limit=50)
                ],
                "episodes": [
                    {"id": e.id, "objective": e.objective, "outcome": e.outcome}
                    for e in self._backend.get_episodes(limit=20)
                ],
                "_meta": {"budget_used": budget, "budget_total": budget},
            }
            self._dispatch_on_load(result)
            return result

        # Filter by strength tier: load() only includes Strong + Fading (>= 0.5)
        for key in ("values", "beliefs", "goals", "drives", "episodes", "notes", "relationships"):
            batched[key] = self._filter_by_strength(batched.get(key, []))

        # Build candidate list with priorities
        candidates = []
        for v in batched.get("values", []):
            candidates.append((_compute_priority_score("value", v), "value", v))
        for b in batched.get("beliefs", []):
            candidates.append((_compute_priority_score("belief", b), "belief", b))
        for g in batched.get("goals", []):
            candidates.append((_compute_priority_score("goal", g), "goal", g))
        for d in batched.get("drives", []):
            candidates.append((_compute_priority_score("drive", d), "drive", d))
        for e in batched.get("episodes", []):
            candidates.append((_compute_priority_score("episode", e), "episode", e))
        for n in batched.get("notes", []):
            candidates.append((_compute_priority_score("note", n), "note", n))
        for r in batched.get("relationships", []):
            candidates.append((_compute_priority_score("relationship", r), "relationship", r))

        # Summaries
        all_summaries = self._backend.list_summaries(self.stack_id)
        superseded_ids = set()
        for s in all_summaries:
            if s.supersedes:
                superseded_ids.update(s.supersedes)
        for s in all_summaries:
            if s.id not in superseded_ids:
                scope_key = f"summary_{s.scope}"
                candidates.append((_compute_priority_score(scope_key, s), "summary", s))

        # Self-narratives
        active_narratives = self._backend.list_self_narratives(self.stack_id, active_only=True)
        for n in active_narratives:
            candidates.append((_compute_priority_score("self_narrative", n), "self_narrative", n))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected: Dict[str, list] = {
            "values": [],
            "beliefs": [],
            "goals": [],
            "drives": [],
            "episodes": [],
            "notes": [],
            "relationships": [],
            "summaries": [],
            "self_narratives": [],
        }

        excluded = []
        budget_exhausted = False
        for priority, memory_type, record in candidates:
            excluded_record = {
                "memory_type": memory_type,
                "memory_id": getattr(record, "id", None),
                "record": record,
                "priority": priority,
            }
            if budget_exhausted:
                excluded.append(excluded_record)
                continue
            text = self._record_to_text(memory_type, record)
            text = _truncate_at_word_boundary(text, max_item_chars)
            tokens = _estimate_tokens(text)
            if tokens <= remaining:
                key = (
                    memory_type + "s"
                    if memory_type not in ("summary", "self_narrative")
                    else ("summaries" if memory_type == "summary" else "self_narratives")
                )
                if key in selected:
                    selected[key].append(record)
                remaining -= tokens
            else:
                excluded.append(excluded_record)
            if remaining <= 0:
                budget_exhausted = True

        excluded_count = len(excluded)

        # Format output (truncate text fields to max_item_chars)
        result: Dict[str, Any] = {
            "values": [
                {
                    "id": v.id,
                    "name": v.name,
                    "statement": _truncate_at_word_boundary(v.statement, max_item_chars),
                    "priority": v.priority,
                }
                for v in selected["values"]
            ],
            "beliefs": [
                {
                    "id": b.id,
                    "statement": _truncate_at_word_boundary(b.statement, max_item_chars),
                    "belief_type": b.belief_type,
                    "confidence": b.confidence,
                }
                for b in selected["beliefs"]
            ],
            "goals": [
                {
                    "id": g.id,
                    "title": g.title,
                    "description": _truncate_at_word_boundary(g.description or "", max_item_chars),
                    "priority": g.priority,
                    "status": g.status,
                }
                for g in selected["goals"]
            ],
            "drives": [
                {
                    "id": d.id,
                    "drive_type": d.drive_type,
                    "intensity": d.intensity,
                    "focus_areas": d.focus_areas,
                }
                for d in selected["drives"]
            ],
            "episodes": [
                {
                    "id": e.id,
                    "objective": _truncate_at_word_boundary(e.objective or "", max_item_chars),
                    "outcome": _truncate_at_word_boundary(e.outcome or "", max_item_chars),
                    "outcome_type": getattr(e, "outcome_type", None),
                    "tags": e.tags,
                    "lessons": e.lessons if e.lessons else [],
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in selected["episodes"]
            ],
            "notes": [
                {
                    "id": n.id,
                    "content": _truncate_at_word_boundary(n.content or "", max_item_chars),
                    "note_type": n.note_type,
                    "tags": n.tags,
                    "speaker": getattr(n, "speaker", None),
                    "reason": getattr(n, "reason", None),
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                }
                for n in selected["notes"]
            ],
            "relationships": [
                {
                    "entity_name": r.entity_name,
                    "entity_type": r.entity_type,
                    "sentiment": r.sentiment,
                    "interaction_count": getattr(r, "interaction_count", 0),
                    "last_interaction": (
                        r.last_interaction.isoformat()
                        if getattr(r, "last_interaction", None)
                        else None
                    ),
                    "notes": _truncate_at_word_boundary(r.notes or "", max_item_chars),
                }
                for r in selected["relationships"]
            ],
            "_meta": {
                "budget_used": budget - remaining,
                "budget_total": budget,
                "excluded_count": excluded_count,
                "_excluded_candidates": excluded,
            },
        }

        if selected["summaries"]:
            result["summaries"] = [
                {
                    "id": s.id,
                    "scope": s.scope,
                    "content": _truncate_at_word_boundary(s.content, max_item_chars),
                }
                for s in selected["summaries"]
            ]

        if selected["self_narratives"]:
            result["self_narratives"] = [
                {
                    "id": sn.id,
                    "narrative_type": sn.narrative_type,
                    "content": _truncate_at_word_boundary(sn.content, max_item_chars),
                    "key_themes": sn.key_themes,
                    "unresolved_tensions": sn.unresolved_tensions,
                }
                for sn in selected["self_narratives"]
            ]

        # Track access for salience
        if track_access:
            accesses = []
            for key in (
                "values",
                "beliefs",
                "goals",
                "drives",
                "episodes",
                "notes",
                "relationships",
            ):
                for rec in selected[key]:
                    type_name = key.rstrip("s") if key != "values" else "value"
                    accesses.append((type_name, rec.id))
            if accesses:
                self._backend.record_access_batch(accesses)

        self._dispatch_on_load(result)
        return result

    # ---- Meta-Memory ----

    def record_access(self, memory_type: str, memory_id: str) -> bool:
        return self._backend.record_access(memory_type, memory_id)

    def update_memory_meta(
        self,
        memory_type: str,
        memory_id: str,
        *,
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        return self._backend.update_memory_meta(memory_type, memory_id, confidence=confidence)

    def forget_memory(
        self,
        memory_type: str,
        memory_id: str,
        reason: str,
    ) -> bool:
        success = self._backend.forget_memory(memory_type, memory_id, reason)
        if success:
            # Cascade: flag direct children via audit entries
            children = self._backend.get_memories_derived_from(memory_type, memory_id)
            for child_type, child_id in children:
                self._backend.log_audit(
                    child_type,
                    child_id,
                    "cascade_flag",
                    "system",
                    {"cascade_source": f"{memory_type}:{memory_id}", "reason": "source_forgotten"},
                )
        return success

    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        return self._backend.recover_memory(memory_type, memory_id)

    def protect_memory(
        self,
        memory_type: str,
        memory_id: str,
        protected: bool = True,
    ) -> bool:
        return self._backend.protect_memory(memory_type, memory_id, protected)

    def weaken_memory(
        self,
        memory_type: str,
        memory_id: str,
        amount: float,
    ) -> bool:
        # Check current strength before weakening to determine if cascade needed
        memory = self._backend.get_memory(memory_type, memory_id)
        old_strength = getattr(memory, "strength", 1.0) if memory else 1.0

        success = self._backend.weaken_memory(memory_type, memory_id, amount)
        if success:
            new_strength = old_strength - abs(amount)
            if new_strength < 0.0:
                new_strength = 0.0
            # Cascade only if strength drops below 0.2 (dormant threshold)
            if new_strength < 0.2 and old_strength >= 0.2:
                children = self._backend.get_memories_derived_from(memory_type, memory_id)
                for child_type, child_id in children:
                    self._backend.log_audit(
                        child_type,
                        child_id,
                        "cascade_flag",
                        "system",
                        {
                            "cascade_source": f"{memory_type}:{memory_id}",
                            "reason": "source_dormant",
                        },
                    )
        return success

    def verify_memory(
        self,
        memory_type: str,
        memory_id: str,
    ) -> bool:
        success = self._backend.verify_memory(memory_type, memory_id)
        if success:
            # Boost source memories referenced in derived_from
            memory = self._backend.get_memory(memory_type, memory_id)
            if memory:
                derived_from = getattr(memory, "derived_from", None) or []
                for ref in derived_from:
                    if not ref or ":" not in ref:
                        continue
                    ref_type, ref_id = ref.split(":", 1)
                    # Skip annotation refs
                    if ref_type in ANNOTATION_REF_TYPES:
                        continue
                    self._backend.boost_memory_strength(ref_type, ref_id, 0.02)
        return success

    # ---- Cascade Queries ----

    def get_memories_derived_from(self, memory_type: str, memory_id: str) -> List[tuple]:
        """Find all memories that cite 'type:id' in their derived_from."""
        return self._backend.get_memories_derived_from(memory_type, memory_id)

    def get_ungrounded_memories(self) -> List[tuple]:
        """Find memories where ALL source refs have strength 0.0 or don't exist."""
        return self._backend.get_ungrounded_memories(self.stack_id)

    def log_audit(
        self,
        memory_type: str,
        memory_id: str,
        operation: str,
        *,
        actor: str = "system",
        details: Optional[Any] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        return self._backend.log_audit(
            memory_type,
            memory_id,
            operation,
            actor,
            details,
            correlation_id,
        )

    def get_audit_log(
        self,
        *,
        memory_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Any]:
        return self._backend.get_audit_log(
            memory_type=memory_type,
            memory_id=memory_id,
            operation=operation,
            correlation_id=correlation_id,
            limit=limit,
        )

    # ---- Processing ----

    def get_processing_config(self) -> List[Dict[str, Any]]:
        """Get all processing configuration entries."""
        return self._backend.get_processing_config()

    def set_processing_config(
        self,
        layer_transition: str,
        **kwargs: Any,
    ) -> bool:
        """Update processing configuration for a layer transition."""
        _validate_literal_value(
            layer_transition, VALID_PROCESSING_TRANSITIONS, "processing transition"
        )
        return self._backend.set_processing_config(layer_transition, **kwargs)

    def mark_episode_processed(self, episode_id: str) -> bool:
        """Mark an episode as processed."""
        return self._backend.mark_episode_processed(episode_id)

    def mark_note_processed(self, note_id: str) -> bool:
        """Mark a note as processed."""
        return self._backend.mark_note_processed(note_id)

    def mark_belief_processed(self, belief_id: str) -> bool:
        """Mark a belief as processed."""
        return self._backend.mark_belief_processed(belief_id)

    # ---- Stack Settings ----

    def get_stack_setting(self, key: str) -> Optional[str]:
        """Get a stack setting value by key."""
        return self._backend.get_stack_setting(key)

    def set_stack_setting(self, key: str, value: str) -> None:
        """Set a stack setting (upsert). Updates in-memory state for known keys."""
        self._backend.set_stack_setting(key, value)
        # Sync in-memory flags so changes take effect immediately
        if key == "enforce_provenance":
            self._enforce_provenance = value == "true"
        elif key == "stack_state" and value in StackState.__members__:
            self._state = StackState[value]

    def get_all_stack_settings(self) -> Dict[str, str]:
        """Get all stack settings as a dict."""
        return self._backend.get_all_stack_settings()

    # ---- Trust Layer ----

    def save_trust_assessment(self, assessment: TrustAssessment) -> str:
        return self._backend.save_trust_assessment(assessment)

    def get_trust_assessments(
        self,
        *,
        entity_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[TrustAssessment]:
        assessments = self._backend.get_trust_assessments()
        if entity_id:
            assessments = [a for a in assessments if a.entity == entity_id]
        return assessments

    def compute_trust(
        self,
        entity_id: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate trust for an entity."""
        assessment = self._backend.get_trust_assessment(entity_id)
        if not assessment:
            return {
                "entity": entity_id,
                "domain": domain or "general",
                "score": 0.5,
                "source": "default",
            }
        dimensions = assessment.dimensions or {}
        d = domain or "general"
        dim_data = dimensions.get(d, {})
        score = dim_data.get("score", 0.5) if isinstance(dim_data, dict) else 0.5
        return {
            "entity": entity_id,
            "domain": d,
            "score": score,
            "source": "assessment",
        }

    # ---- Features ----

    def consolidate(
        self,
        *,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run memory consolidation."""
        episodes = self._backend.get_episodes(limit=50)
        if len(episodes) < 3:
            return {
                "consolidated": 0,
                "lessons_found": 0,
                "message": "Need at least 3 episodes to consolidate",
            }
        all_lessons = []
        for ep in episodes:
            if ep.lessons:
                all_lessons.extend(ep.lessons)
        lesson_counts = Counter(all_lessons)
        common = [lesson for lesson, cnt in lesson_counts.items() if cnt >= 2]
        return {
            "consolidated": len(episodes),
            "lessons_found": len(common),
            "common_lessons": common[:5],
        }

    def apply_forgetting(
        self,
        *,
        protect_identity: bool = True,
    ) -> Dict[str, Any]:
        """Apply salience-based forgetting."""
        report = self.run_forgetting_cycle(
            threshold=0.3,
            limit=10,
            dry_run=False,
        )
        return {
            "forgotten": report.get("forgotten", 0),
            "candidates": report.get("candidate_count", 0),
            "protected": report.get("protected", 0),
        }

    # ---- Sync ----

    def sync(self) -> ProtocolSyncResult:
        storage_result = self._backend.sync()
        return ProtocolSyncResult(
            pushed=storage_result.pushed,
            pulled=storage_result.pulled,
            conflicts=storage_result.conflict_count,
            errors=storage_result.errors,
        )

    def pull_changes(self, *, since: Optional[datetime] = None) -> ProtocolSyncResult:
        storage_result = self._backend.pull_changes(since=since)
        return ProtocolSyncResult(
            pushed=storage_result.pushed,
            pulled=storage_result.pulled,
            conflicts=storage_result.conflict_count,
            errors=storage_result.errors,
        )

    def get_pending_sync_count(self) -> int:
        return self._backend.get_pending_sync_count()

    def is_online(self) -> bool:
        return self._backend.is_online()

    # ---- Stats & Export ----

    def get_stats(self) -> Dict[str, int]:
        return self._backend.get_stats()

    def dump(
        self,
        *,
        format: str = "markdown",
        include_raw: bool = True,
        include_forgotten: bool = False,
    ) -> str:
        """Export all memories as a formatted string."""
        _validate_literal_value(format, VALID_DUMP_FORMATS, "dump format")
        if format == "json":
            return self._dump_json(include_raw, include_forgotten)
        return self._dump_markdown(include_raw, include_forgotten)

    def export(self, path: str, *, format: str = "markdown") -> None:
        """Export all memories to a file."""
        _validate_literal_value(format, VALID_DUMP_FORMATS, "export format")
        content = self.dump(format=format)
        if format == "markdown" and path.endswith(".json"):
            content = self.dump(format="json")
        elif format == "json" and (path.endswith(".md") or path.endswith(".markdown")):
            content = self.dump(format="markdown")
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(content, encoding="utf-8")

    # ---- Composition Hooks ----

    def on_attach(
        self,
        core_id: str,
        inference: Optional[InferenceService] = None,
    ) -> None:
        self._attached_core_id = core_id
        if inference is not None:
            self._inference = inference
            self._set_inference_for_components(inference)
        # Transition to ACTIVE on first attach (provenance enforcement begins)
        if self._state == StackState.INITIALIZING:
            self._state = StackState.ACTIVE
            self._backend.set_stack_setting("stack_state", StackState.ACTIVE.name)

    def on_detach(self, core_id: str) -> None:
        self._attached_core_id = None
        self._inference = None
        self._set_inference_for_components(None)

    def on_model_changed(
        self,
        inference: Optional[InferenceService],
    ) -> None:
        self._inference = inference
        self._set_inference_for_components(inference)

    # ---- Lint Helpers ----

    def _get_lint_config(self) -> Dict[str, Any]:
        """Get lint configuration from stack settings."""
        from kernle.lint import get_lint_config

        return get_lint_config(self.get_stack_setting)

    def _lint_belief(self, belief: Belief) -> Any:
        """Run lint on a belief's statement. Returns a LintResult."""
        from kernle.lint import lint_belief

        config = self._get_lint_config()
        return lint_belief(belief.statement, config)

    def _lint_value(self, value: Value) -> Any:
        """Run lint on a value's name and statement. Returns a LintResult."""
        from kernle.lint import lint_value

        config = self._get_lint_config()
        return lint_value(value.name, value.statement, config)

    def _redirect_to_suggestion(self, memory_type: str, memory: Any, lint_result: Any) -> str:
        """Redirect a lint-failed memory to a suggestion and log to audit.

        Instead of saving the malformed memory directly, store it as a
        MemorySuggestion with status='rejected' and resolution_reason
        indicating the lint failure. This keeps the structured memory
        clean while preserving the content for manual review.

        Returns the suggestion ID (prefixed with 'suggestion:' to indicate
        it was redirected).
        """
        suggestion_id = str(uuid.uuid4())

        # Build content dict from the memory object
        if memory_type == "belief":
            content = {
                "statement": memory.statement,
                "belief_type": getattr(memory, "belief_type", "fact"),
                "confidence": getattr(memory, "confidence", 0.8),
            }
        elif memory_type == "value":
            content = {
                "name": memory.name,
                "statement": memory.statement,
                "priority": getattr(memory, "priority", 50),
            }
        else:
            content = {"raw": str(memory)}

        # Build source_raw_ids from derived_from if available
        derived_from = getattr(memory, "derived_from", None) or []
        source_raw_ids = [ref for ref in derived_from if ref]

        suggestion = MemorySuggestion(
            id=suggestion_id,
            stack_id=self.stack_id,
            memory_type=memory_type,
            content=content,
            confidence=0.0,
            source_raw_ids=source_raw_ids,
            status="rejected",
            resolution_reason=f"lint_failed: {lint_result.summary}",
            created_at=datetime.now(timezone.utc),
            resolved_at=datetime.now(timezone.utc),
        )
        self._backend.save_suggestion(suggestion)

        # Log lint failure to audit
        self._backend.log_audit(
            memory_type,
            suggestion_id,
            "suggestion.resolved",
            "system",
            {
                "resolution": "lint_rejected",
                "suggestion_id": suggestion_id,
                "failures": lint_result.failures,
                "redirected_to": f"suggestion:{suggestion_id}",
            },
        )

        logger.info(
            "Lint rejected %s, redirected to suggestion:%s: %s",
            memory_type,
            suggestion_id,
            lint_result.summary,
        )

        return f"suggestion:{suggestion_id}"

    # ---- Private Helpers ----

    @staticmethod
    def _record_to_text(memory_type: str, record: Any) -> str:
        """Get text representation of a record for token estimation."""
        if memory_type == "value":
            return f"{record.name}: {record.statement}"
        elif memory_type == "belief":
            return record.statement
        elif memory_type == "goal":
            return f"{record.title} {record.description or ''}"
        elif memory_type == "drive":
            return f"{record.drive_type}: {record.focus_areas or ''}"
        elif memory_type == "episode":
            return f"{record.objective} {record.outcome}"
        elif memory_type == "note":
            return record.content
        elif memory_type == "relationship":
            return f"{record.entity_name}: {record.notes or ''}"
        elif memory_type == "summary":
            return f"[{record.scope}] {record.content}"
        elif memory_type == "self_narrative":
            return f"[{record.narrative_type}] {record.content}"
        return str(record)

    def _dump_markdown(self, include_raw: bool, include_forgotten: bool) -> str:
        """Export memory as markdown."""
        lines = [
            f"# Memory Dump for {self.stack_id}",
            f"_Exported at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
        ]
        values = self._backend.get_values(limit=100)
        if values:
            lines.append("## Values")
            for v in sorted(values, key=lambda x: x.priority, reverse=True):
                lines.append(f"- **{v.name}** (priority {v.priority}): {v.statement}")
            lines.append("")

        beliefs = self._backend.get_beliefs(limit=100)
        if beliefs:
            lines.append("## Beliefs")
            for b in sorted(beliefs, key=lambda x: x.confidence, reverse=True):
                lines.append(f"- [{b.confidence:.0%}] {b.statement}")
            lines.append("")

        goals = self._backend.get_goals(status=None, limit=100)
        if goals:
            lines.append("## Goals")
            for g in goals:
                icon = "+" if g.status == "completed" else "o" if g.status == "active" else "-"
                lines.append(f"- {icon} [{g.priority}] {g.title}")
            lines.append("")

        episodes = self._backend.get_episodes(limit=100)
        if episodes:
            lines.append("## Episodes")
            for e in episodes:
                date_str = e.created_at.strftime("%Y-%m-%d") if e.created_at else "unknown"
                lines.append(f"### {e.objective}")
                lines.append(f"*{date_str}* | {e.outcome}")
                if e.lessons:
                    for lesson in e.lessons:
                        lines.append(f"  - {lesson}")
                lines.append("")

        notes = self._backend.get_notes(limit=100)
        if notes:
            lines.append("## Notes")
            for n in notes:
                lines.append(f"- [{n.note_type}] {n.content}")
            lines.append("")

        if include_raw:
            raw_entries = self._backend.list_raw(limit=100)
            if raw_entries:
                lines.append("## Raw Entries")
                for r in raw_entries:
                    status = "done" if r.processed else "pending"
                    lines.append(f"- [{status}] {r.content or r.blob or ''}")
                lines.append("")

        return "\n".join(lines)

    def _dump_json(self, include_raw: bool, include_forgotten: bool) -> str:
        """Export memory as JSON."""

        def _dt(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None

        data: Dict[str, Any] = {
            "stack_id": self.stack_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "values": [
                {"id": v.id, "name": v.name, "statement": v.statement, "priority": v.priority}
                for v in self._backend.get_values(limit=100)
            ],
            "beliefs": [
                {"id": b.id, "statement": b.statement, "confidence": b.confidence}
                for b in self._backend.get_beliefs(limit=100)
            ],
            "goals": [
                {"id": g.id, "title": g.title, "status": g.status, "priority": g.priority}
                for g in self._backend.get_goals(status=None, limit=100)
            ],
            "episodes": [
                {
                    "id": e.id,
                    "objective": e.objective,
                    "outcome": e.outcome,
                    "created_at": _dt(e.created_at),
                }
                for e in self._backend.get_episodes(limit=100)
            ],
            "notes": [
                {"id": n.id, "content": n.content, "note_type": n.note_type}
                for n in self._backend.get_notes(limit=100)
            ],
        }
        if include_raw:
            data["raw_entries"] = [
                {"id": r.id, "content": r.content, "processed": r.processed}
                for r in self._backend.list_raw(limit=100)
            ]
        return json.dumps(data, indent=2, default=str)

    # ---- Stack-level helpers ----

    def list_raw(
        self, processed: Optional[bool] = None, limit: int = 100, offset: int = 0
    ) -> List[Any]:
        """List raw entries."""
        return self._backend.list_raw(processed=processed, limit=limit, offset=offset)

    def get_identity_confidence(self) -> float:
        """Get identity confidence from values and beliefs.

        Used by AnxietyComponent to compute identity_coherence dimension.
        """
        values = self._backend.get_values(limit=10)
        beliefs = self._backend.get_beliefs(limit=20)
        if not values and not beliefs:
            return 0.0
        total_conf = 0.0
        count = 0
        for v in values:
            total_conf += v.confidence
            count += 1
        for b in beliefs:
            total_conf += b.confidence
            count += 1
        return total_conf / count if count > 0 else 0.0
