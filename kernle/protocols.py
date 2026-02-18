"""
kernle Protocol Definitions
============================

The complete interface contracts for the kernle component system.

Protocol version: 1

Components and their roles:
- Core (torso):  The bus. Connects everything. Has an ID. Routes operations.
- Stack (head):  A memory system. Self-contained. Attachable/detachable.
- Plugin (limb): A capability. Manages its own state. Removable without residue.
- Model (heart): The thinking engine. Required. Interchangeable. Affects behavior.

The "entity" is the composition — no single component IS the entity.

Package structure:
- kernle-core:  Core, protocols, shared types, plugin/stack management, CLI
- kernle-stack: Default memory implementation (SQLite, embeddings, features)
- plugins:      Independent packages discovered via entry points

Design principles:
- Components are peers, not parent-child
- core_id and stack_id are both first-class identifiers
- Plugins never touch the stack schema — they write memories through the core
- When a plugin is removed, only memories remain (in the stack)
- Everything works locally (local model) or with cloud providers
- Stacks are many-to-many with cores (multiple heads, shared stacks)
- Interchangeability of all components is a feature, not a limitation

Error handling philosophy:
- Routed operations (core -> stack) raise NoActiveStackError if no stack attached
- Plugin context methods return None if no active stack (plugins handle gracefully)
- Invalid arguments raise ValueError
- Storage failures raise StorageError (stack-specific subclass)
- Plugin failures are isolated — one plugin's error doesn't crash the core
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from kernle.types import (
    Belief,
    Drive,
    Episode,
    Epoch,
    Goal,
    MemorySuggestion,
    MemoryType,
    Note,
    Playbook,
    RawEntry,
    Relationship,
    SearchResult,
    SelfNarrative,
    SourceType,
    Summary,
    TrustAssessment,
    Value,
)

BeliefType = Literal[
    "assumption",
    "causal",
    "constraint",
    "evaluative",
    "fact",
    "factual",
    "hypothesis",
    "learned",
    "model",
    "observation",
    "opinion",
    "pattern",
    "preference",
    "principle",
    "procedural",
    "rule",
    "strategy",
]

NoteType = Literal[
    "decision",
    "diagnostic",
    "fact",
    "insight",
    "note",
    "observation",
    "procedure",
    "quote",
    "reference",
]

InferenceScope = Literal["none", "fast", "capable", "embedding"]

ProcessingTransition = Literal[
    "raw_to_episode",
    "raw_to_note",
    "episode_to_belief",
    "episode_to_goal",
    "episode_to_relationship",
    "belief_to_value",
    "episode_to_drive",
]

GoalType = Literal["task", "aspiration", "commitment", "exploration"]
DriveType = Literal["existence", "growth", "curiosity", "connection", "reproduction"]
GoalStatus = Literal["active", "completed", "paused"]
SuggestionStatus = Literal[
    "pending",
    "promoted",
    "modified",
    "rejected",
    "dismissed",
    "expired",
]
SuggestionMemoryType = Literal[
    "belief", "note", "episode", "goal", "relationship", "value", "drive"
]
SearchRecordType = Literal["episode", "note", "belief", "value", "goal"]
DumpFormat = Literal["markdown", "json"]
ModelRole = Literal["system", "user", "assistant", "tool"]


# =============================================================================
# PROTOCOL VERSION
# =============================================================================
# Bumped when protocols change in incompatible ways. Components check this
# to verify they're speaking the same language.

PROTOCOL_VERSION = 1


# =============================================================================
# ERRORS
# =============================================================================


class KernleError(Exception):
    """Base for all kernle errors."""

    pass


class NoActiveStackError(KernleError):
    """Raised when a routed operation has no active stack to route to."""

    pass


class PluginError(KernleError):
    """Raised when a plugin fails. Isolated — doesn't crash the core."""

    pass


class StorageError(KernleError):
    """Raised by stack implementations on storage failures."""

    pass


class ProvenanceError(KernleError):
    """Raised when provenance validation fails.

    E.g., creating a belief without citing an episode/note,
    or citing a raw directly as a belief source.
    """

    pass


class MaintenanceModeError(KernleError):
    """Raised when an operation requires maintenance mode but the stack is not in it."""

    pass


class InferenceRequiredError(KernleError):
    """Raised when an operation requires a bound inference model but none is available."""

    pass


# =============================================================================
# SHARED TYPES
# =============================================================================
# These live in kernle-core. Imported by stacks, plugins, and the core itself.
#
# Memory dataclasses (Episode, Belief, Value, Goal, Note, Drive, Relationship,
# RawEntry, Playbook, TrustAssessment, EntityModel, Epoch, Summary,
# SelfNarrative, MemorySuggestion) are defined in kernle.types.
#
# They are the shared vocabulary. A plugin creates an Episode; the stack
# stores it. The types are the contract between them.
#
# Trust and Relationships:
# - Relationship is a record of knowing another entity (with a trust_level
#   convenience field for quick access).
# - TrustAssessment is a detailed, domain-scoped trust evaluation with
#   evidence and scoring. Multiple assessments can exist per relationship.
# - compute_trust() aggregates TrustAssessments into a score.
# - relationship.trust_level is typically the latest compute_trust() result
#   for that entity. The stack may update it automatically or on request.
# =============================================================================


class StackState(str, Enum):
    """Lifecycle state for a memory stack."""

    INITIALIZING = "initializing"  # Seed writes allowed without provenance
    ACTIVE = "active"  # All writes require valid provenance
    MAINTENANCE = "maintenance"  # Only controlled admin operations


ProtocolSearchResult = SearchResult


@dataclass
class SyncResult:
    """Result of a sync operation."""

    pushed: int = 0
    pulled: int = 0
    conflicts: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def conflict_count(self) -> int:
        """Number of conflicts encountered."""
        if isinstance(self.conflicts, int):
            return self.conflicts
        return len(self.conflicts)


@dataclass
class ModelCapabilities:
    """What a model implementation can do."""

    model_id: str
    provider: str  # "anthropic", "openai", "ollama", "llamacpp", etc.
    context_window: int
    max_output_tokens: int = 4096
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True


@dataclass
class ModelMessage:
    """A message in a conversation."""

    role: ModelRole  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


@dataclass
class ModelResponse:
    """Complete response from a model."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    model_id: Optional[str] = None


@dataclass
class ModelChunk:
    """A streaming chunk from a model."""

    content: str = ""
    tool_call: Optional[dict[str, Any]] = None
    is_final: bool = False
    usage: Optional[dict[str, int]] = None


ModelErrorClass = Literal["rate_limit", "auth", "timeout", "server", "unknown"]


@dataclass
class ModelStatus:
    """Status envelope for model operations."""

    provider: str  # "openai", "anthropic", "ollama"
    available: bool  # True if last call succeeded
    error_class: Optional[ModelErrorClass] = None
    error_message: Optional[str] = None
    degraded: bool = False  # True when fallback was used


@dataclass
class ToolDefinition:
    """A tool that can be offered to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Optional[Callable[[dict[str, Any]], Any]] = None


@dataclass
class PluginHealth:
    """Health check result for a plugin."""

    healthy: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StackInfo:
    """Metadata about an attached stack."""

    stack_id: str
    alias: Optional[str] = None
    schema_version: int = 0
    stats: dict[str, int] = field(default_factory=dict)
    is_active: bool = False


@dataclass
class PluginInfo:
    """Metadata about a plugin (loaded or discovered)."""

    name: str
    version: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    is_loaded: bool = False


@dataclass
class Binding:
    """A snapshot of the current composition.

    Captures which model, stacks, and plugins are bound to a core.
    Used by checkpoint() for persistence. Restore via from_checkpoint().
    """

    core_id: str
    model_config: dict[str, Any]  # provider, model_id (both required for model restoration)
    stacks: dict[str, Optional[str]]  # stack_id -> optional alias
    active_stack_id: Optional[str] = None
    plugins: list[str] = field(default_factory=list)  # plugin names
    created_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MODEL PROTOCOL — Heart/Lungs
# =============================================================================
# Required. Interchangeable. Differences in model produce meaningfully
# different behavior — that's acknowledged, not hidden. Swapping from
# Claude to Llama changes how the entity thinks. That's fine.
# =============================================================================


@runtime_checkable
class ModelProtocol(Protocol):
    """Interface for the thinking engine.

    Implementations: AnthropicModel, OpenAIModel, OllamaModel,
    LlamaCppModel, VLLMModel, etc.
    """

    @property
    def model_id(self) -> str:
        """Identifier (e.g., 'claude-sonnet-4-5-20250929', 'llama3:8b')."""
        ...

    @property
    def capabilities(self) -> ModelCapabilities:
        """What this model can do."""
        ...

    def generate(
        self,
        messages: list[ModelMessage],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> ModelResponse:
        """Generate a complete response."""
        ...

    def stream(
        self,
        messages: list[ModelMessage],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> Iterator[ModelChunk]:
        """Stream a response chunk by chunk."""
        ...


# =============================================================================
# INFERENCE SERVICE — Narrow model access for stack components
# =============================================================================
# When the core attaches a stack, it can provide an InferenceService.
# This gives stack components (consolidation, suggestions, etc.) access
# to model capabilities WITHOUT the stack knowing about ModelProtocol.
#
# If no core is attached, or the core has no model, inference_service
# is None. Components that need inference degrade gracefully.
# =============================================================================


@runtime_checkable
class InferenceService(Protocol):
    """Narrow interface the core provides to stacks for inference.

    Not the full ModelProtocol — just what stack components need.
    The core implements this by routing to whatever model is bound.
    """

    def infer(self, prompt: str, *, system: Optional[str] = None) -> str:
        """Generate text from a prompt. Returns the response string.

        Used by stack components for:
        - Consolidation: synthesize episodes into beliefs
        - Suggestions: extract patterns from recent memories
        - Emotional tagging: detect valence/arousal from text
        """
        ...

    def embed(self, text: str) -> list[float]:
        """Embed text into a vector. Returns floats.

        Used by embedding components. The core routes this to
        whatever embedding provider or model is configured.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed. Default: loop over embed()."""
        ...

    @property
    def embedding_dimension(self) -> int:
        """Dimension of vectors produced by embed()."""
        ...

    @property
    def embedding_provider_id(self) -> str:
        """Stable ID for the current embedding source.

        Used by stacks to detect when re-indexing is needed.
        e.g., 'ngram-v1', 'sentence-transformers/all-MiniLM-L6-v2'
        """
        ...


# =============================================================================
# STACK COMPONENT PROTOCOL — Swappable sub-components of the stack
# =============================================================================
# The stack has a component system, parallel to how the core has plugins.
# Components extend what the stack can do. Some are required (embedding
# for search), most are optional (emotional tagging, anxiety, etc.).
#
# Components hook into the stack's lifecycle: memory saves, loads,
# searches, and periodic maintenance. They receive an InferenceService
# if one is available (provided by the core on attach).
#
# The current kernle mixins (Anxiety, Emotions, Consolidation, Forgetting,
# Knowledge, MetaMemory, Suggestions) map directly to stack components.
#
# Discovery via entry point: kernle.stack_components
#
# Required vs optional:
# - A component declares required=True if the stack can't function without it
# - The stack refuses to operate if a required component is missing
# - Optional components enhance behavior — their absence means graceful
#   degradation, not failure
# =============================================================================


@runtime_checkable
class StackComponentProtocol(Protocol):
    """Interface for stack sub-components.

    Implementations: EmbeddingComponent, ForgettingComponent,
    ConsolidationComponent, EmotionalTaggingComponent,
    AnxietyComponent, SuggestionComponent, MetaMemoryComponent, etc.
    """

    @property
    def name(self) -> str:
        """Component identifier (e.g., 'embedding', 'forgetting')."""
        ...

    @property
    def version(self) -> str:
        """Semantic version."""
        ...

    @property
    def required(self) -> bool:
        """Whether the stack needs this component to function.

        True: stack refuses to operate without it (e.g., embedding).
        False: stack works without it, just with reduced capability.
        """
        ...

    @property
    def needs_inference(self) -> bool:
        """Whether this component needs model inference.

        If True and no InferenceService is available, the component
        should degrade gracefully (skip inference-dependent operations,
        not crash).
        """
        ...

    @property
    def inference_scope(self) -> InferenceScope:
        """Hint about what kind of inference this component needs.

        Values:
          'fast'      — tagging, classification (cheap/fast model)
          'capable'   — synthesis, reasoning (capable model)
          'embedding' — vector generation (embedding model)
          'none'      — no inference needed

        Default: 'capable'

        This is a declaration only — scope-based routing to different
        models is future work. Components declare their needs so a
        future core can route infer() calls to appropriate models.
        """
        ...

    @property
    def priority(self) -> int:
        """Ordering priority for hook dispatch. Lower runs first.

        Priority ranges:
          0-99    infrastructure (embedding, indexing)
          100-199 enrichment (emotional tagging, meta-memory)
          200-299 analysis (consolidation, suggestions, knowledge)
          300+    maintenance (forgetting, cleanup)

        Default: 200.

        Components are sorted by priority in the stack's component
        registry. on_save, on_search, on_load, and on_maintenance
        hooks fire in priority order.
        """
        ...

    def attach(
        self,
        stack_id: str,
        inference: Optional[InferenceService] = None,
    ) -> None:
        """Called when the component is added to a stack.

        Args:
            stack_id: The stack this component is joining.
            inference: Model access, if available. None if the stack
                      is detached from any core, or the core has no model.
        """
        ...

    def detach(self) -> None:
        """Called when the component is removed from a stack."""
        ...

    def set_inference(self, inference: Optional[InferenceService]) -> None:
        """Update the inference service.

        Called when the stack is attached/detached from a core,
        or when the core's model changes. None means inference
        is no longer available.
        """
        ...

    def set_storage(self, storage: Any) -> None:
        """Provide storage access to the component.

        Called by the stack after attach() to give the component
        direct access to the storage backend. Components that need
        to query or write memories during maintenance, on_load, etc.
        use this reference.

        Args:
            storage: The stack's storage backend instance.
        """
        ...

    # ---- Lifecycle Hooks ----
    # The stack calls these at appropriate moments. Components
    # implement the ones they care about. Default: no-op.

    def on_save(self, memory_type: MemoryType, memory_id: str, memory: Any) -> Optional[dict]:
        """Called after a memory is saved.

        Return a dict of metadata fields to persist on the memory
        (e.g., emotional tags). The stack writes returned fields to the
        database. Return None to leave unchanged.

        Examples:
        - EmotionalTagging: detect valence/arousal, return {"valence": 0.8}
        - MetaMemory: initialize confidence tracking
        - Embedding: generate and store vector
        """
        ...

    def on_search(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Called after search results are assembled.

        Can re-rank, filter, or augment results.
        Return the (possibly modified) results list.

        Examples:
        - Forgetting: filter out low-salience results
        - Emotional: boost emotionally relevant results
        """
        ...

    def on_load(self, context: dict[str, Any]) -> None:
        """Called during load() assembly.

        Contribute to the working memory context.

        Examples:
        - Anxiety: add current anxiety levels
        - Forgetting: add forgetting statistics
        """
        ...

    def on_maintenance(self) -> dict[str, Any]:
        """Called during periodic maintenance.

        Do background work: forgetting sweeps, suggestion extraction,
        consolidation passes, embedding re-indexing, etc.

        Returns stats about what was done.
        """
        ...


# =============================================================================
# STACK SUB-PROTOCOLS — Granular interfaces for type narrowing
# =============================================================================
# Each sub-protocol captures a cohesive subset of stack behavior.
# Code that only needs to read memories can accept StackReaderProtocol;
# code that only needs search can accept StackSearchProtocol; etc.
#
# The composite StackProtocol below inherits from all of them,
# so existing code that depends on StackProtocol is unchanged.
# =============================================================================


@runtime_checkable
class StackLifecycleProtocol(Protocol):
    """Stack lifecycle state and composition hooks."""

    @property
    def state(self) -> StackState:
        """Current lifecycle state (INITIALIZING, ACTIVE, or MAINTENANCE)."""
        ...

    def enter_maintenance(self) -> None:
        """Enter maintenance mode. Only controlled admin operations allowed."""
        ...

    def exit_maintenance(self) -> None:
        """Exit maintenance mode, returning to ACTIVE state."""
        ...

    def on_attach(
        self,
        core_id: str,
        inference: Optional[InferenceService] = None,
    ) -> None:
        """Called when this stack is attached to a core.

        The stack should:
        - Track the core_id (for shared stack coordination)
        - Pass the inference service to components that need it
        - Enable inference-dependent features if inference is provided

        Args:
            core_id: The core attaching this stack.
            inference: Model access for components. None if the core
                      has no model bound yet.
        """
        ...

    def on_detach(self, core_id: str) -> None:
        """Called when this stack is detached from a core.

        The stack should:
        - Stop tracking this core_id
        - If no other cores are attached, set inference to None
          on all components (graceful degradation)
        """
        ...

    def on_model_changed(
        self,
        inference: Optional[InferenceService],
    ) -> None:
        """Called when the attached core's model changes.

        The stack should update the inference service on all
        components. If inference is None, the model was removed.
        """
        ...


@runtime_checkable
class StackComponentManagerProtocol(Protocol):
    """Component management for stacks."""

    @property
    def components(self) -> dict[str, StackComponentProtocol]:
        """All loaded components, keyed by name."""
        ...

    def add_component(self, component: StackComponentProtocol) -> None:
        """Add a component to this stack.

        Calls component.attach() with the stack_id and current
        inference service (if any).
        """
        ...

    def remove_component(self, name: str) -> None:
        """Remove a component.

        Raises if the component is required and no replacement
        is provided.
        """
        ...

    def get_component(self, name: str) -> Optional[StackComponentProtocol]:
        """Get a component by name. Returns None if not loaded."""
        ...

    def maintenance(self) -> dict[str, Any]:
        """Run maintenance on all components.

        Calls on_maintenance() on each component. Returns
        combined stats keyed by component name.
        """
        ...


@runtime_checkable
class StackWriterProtocol(Protocol):
    """Low-level write operations for memories."""

    def save_episode(self, episode: Episode) -> str: ...
    def save_belief(self, belief: Belief) -> str: ...
    def save_value(self, value: Value) -> str: ...
    def save_goal(self, goal: Goal) -> str: ...
    def save_note(self, note: Note) -> str: ...
    def save_drive(self, drive: Drive) -> str: ...
    def save_relationship(self, relationship: Relationship) -> str: ...
    def save_raw(self, raw: RawEntry) -> str: ...
    def save_playbook(self, playbook: Playbook) -> str: ...
    def save_epoch(self, epoch: Epoch) -> str: ...
    def save_summary(self, summary: Summary) -> str: ...
    def save_self_narrative(self, narrative: SelfNarrative) -> str: ...
    def save_suggestion(self, suggestion: MemorySuggestion) -> str: ...

    def update_drive_atomic(self, drive: Drive, expected_version: Optional[int] = None) -> bool: ...
    def update_relationship_atomic(
        self, relationship: Relationship, expected_version: Optional[int] = None
    ) -> bool: ...

    def save_episodes_batch(self, episodes: list[Episode]) -> list[str]: ...
    def save_beliefs_batch(self, beliefs: list[Belief]) -> list[str]: ...
    def save_notes_batch(self, notes: list[Note]) -> list[str]: ...


@runtime_checkable
class StackReaderProtocol(Protocol):
    """Read operations for memories."""

    def get_episodes(
        self,
        *,
        limit: int = 50,
        tags: Optional[list[str]] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> list[Episode]: ...

    def get_beliefs(
        self,
        *,
        limit: int = 50,
        belief_type: Optional[BeliefType] = None,
        min_confidence: Optional[float] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> list[Belief]: ...

    def get_values(
        self,
        *,
        limit: int = 50,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> list[Value]: ...

    def get_goals(
        self,
        *,
        limit: int = 50,
        status: Optional[GoalStatus] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> list[Goal]: ...

    def get_notes(
        self,
        *,
        limit: int = 50,
        note_type: Optional[NoteType] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> list[Note]: ...

    def get_drives(self, *, include_expired: bool = False) -> list[Drive]: ...

    def get_drive(self, drive_type: str) -> Optional[Drive]: ...

    def get_relationships(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_trust: Optional[float] = None,
    ) -> list[Relationship]: ...

    def get_relationship(self, entity_name: str) -> Optional[Relationship]: ...

    def get_raw(
        self,
        *,
        limit: int = 50,
        tags: Optional[list[str]] = None,
    ) -> list[RawEntry]: ...

    def get_memory(self, memory_type: MemoryType, memory_id: str) -> Optional[Any]:
        """Get any single memory by type and ID."""
        ...


@runtime_checkable
class StackSearchProtocol(Protocol):
    """Semantic search across memory types."""

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[list[SearchRecordType]] = None,
        context: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> list[SearchResult]:
        """Semantic search across all memory types."""
        ...


@runtime_checkable
class StackLoaderProtocol(Protocol):
    """Working memory assembly."""

    def load(
        self,
        *,
        token_budget: int = 8000,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Assemble working memory within a token budget.

        Returns a structured dict of the most relevant current memories.
        """
        ...


@runtime_checkable
class StackSuggestionProtocol(Protocol):
    """Suggestion lifecycle management."""

    def get_suggestion(self, suggestion_id: str) -> Optional[MemorySuggestion]:
        """Retrieve a single suggestion by ID."""
        ...

    def get_suggestions(
        self,
        status: Optional[SuggestionStatus] = None,
        memory_type: Optional[SuggestionMemoryType] = None,
        limit: int = 100,
        min_confidence: Optional[float] = None,
        max_age_hours: Optional[float] = None,
        source_raw_id: Optional[str] = None,
    ) -> list[MemorySuggestion]:
        """Retrieve suggestions with optional filters."""
        ...

    def accept_suggestion(
        self,
        suggestion_id: str,
        modifications: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Accept a pending suggestion and promote it to a structured memory.

        Returns the ID of the created memory, or None if the suggestion
        was not found or not pending.
        """
        ...

    def dismiss_suggestion(
        self,
        suggestion_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Dismiss a pending suggestion (will not be promoted)."""
        ...


@runtime_checkable
class StackMetaMemoryProtocol(Protocol):
    """Meta-memory operations: access tracking, forgetting, auditing."""

    def record_access(self, memory_type: MemoryType, memory_id: str) -> bool:
        """Record that a memory was accessed (strengthens salience)."""
        ...

    def update_memory_meta(
        self,
        memory_type: MemoryType,
        memory_id: str,
        *,
        confidence: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Update metadata on an existing memory."""
        ...

    def forget_memory(
        self,
        memory_type: MemoryType,
        memory_id: str,
        reason: str,
    ) -> bool:
        """Soft-delete a memory (can be recovered)."""
        ...

    def recover_memory(self, memory_type: MemoryType, memory_id: str) -> bool:
        """Recover a forgotten memory."""
        ...

    def protect_memory(
        self,
        memory_type: MemoryType,
        memory_id: str,
        protected: bool = True,
    ) -> bool:
        """Protect/unprotect a memory from forgetting."""
        ...

    def weaken_memory(
        self,
        memory_type: MemoryType,
        memory_id: str,
        amount: float,
    ) -> bool:
        """Reduce a memory's strength by a given amount."""
        ...

    def verify_memory(
        self,
        memory_type: MemoryType,
        memory_id: str,
    ) -> bool:
        """Verify a memory: boost strength and increment verification count."""
        ...

    def log_audit(
        self,
        memory_type: MemoryType,
        memory_id: str,
        operation: str,
        *,
        actor: str = "system",
        details: Optional[Any] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Log an audit entry for a memory operation."""
        ...

    def get_audit_log(
        self,
        *,
        memory_type: Optional[MemoryType] = None,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get audit log entries."""
        ...


@runtime_checkable
class StackProcessingProtocol(Protocol):
    """Processing configuration and state management."""

    def get_processing_config(self) -> list[dict[str, Any]]:
        """Get all processing configuration entries."""
        ...

    def set_processing_config(
        self,
        layer_transition: ProcessingTransition,
        **kwargs: Any,
    ) -> bool:
        """Update processing configuration."""
        ...

    def mark_episode_processed(self, episode_id: str) -> bool:
        """Mark an episode as processed."""
        ...

    def mark_note_processed(self, note_id: str) -> bool:
        """Mark a note as processed."""
        ...


@runtime_checkable
class StackSettingsProtocol(Protocol):
    """Stack-level settings management."""

    def get_stack_setting(self, key: str) -> Optional[str]:
        """Get a stack setting value by key."""
        ...

    def set_stack_setting(self, key: str, value: str) -> None:
        """Set a stack setting (upsert)."""
        ...

    def get_all_stack_settings(self) -> dict[str, str]:
        """Get all stack settings as a dict."""
        ...


@runtime_checkable
class StackTrustProtocol(Protocol):
    """Trust assessment and computation."""

    def save_trust_assessment(self, assessment: TrustAssessment) -> str: ...

    def get_trust_assessments(
        self,
        *,
        entity_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[TrustAssessment]: ...

    def compute_trust(
        self,
        entity_id: str,
        domain: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compute aggregate trust for an entity."""
        ...


@runtime_checkable
class StackFeatureProtocol(Protocol):
    """Higher-level memory features: consolidation and forgetting."""

    def consolidate(
        self,
        *,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run memory consolidation.

        Finds patterns across episodes, strengthens repeated lessons
        into beliefs, surfaces suggestions for review.
        """
        ...

    def apply_forgetting(
        self,
        *,
        protect_identity: bool = True,
    ) -> dict[str, Any]:
        """Apply salience-based forgetting."""
        ...


@runtime_checkable
class StackSyncProtocol(Protocol):
    """Remote sync operations."""

    def sync(self) -> SyncResult:
        """Sync with remote storage (if configured)."""
        ...

    def pull_changes(self, *, since: Optional[datetime] = None) -> SyncResult:
        """Pull changes from remote."""
        ...

    def get_pending_sync_count(self) -> int: ...
    def is_online(self) -> bool: ...


@runtime_checkable
class StackExportProtocol(Protocol):
    """Stats and export operations."""

    def get_stats(self) -> dict[str, int]:
        """Counts of each memory type."""
        ...

    def dump(
        self,
        *,
        format: DumpFormat = "markdown",
        include_raw: bool = True,
        include_forgotten: bool = False,
    ) -> str:
        """Export all memories as a formatted string."""
        ...

    def export(self, path: str, *, format: DumpFormat = "markdown") -> None:
        """Export all memories to a file."""
        ...


# =============================================================================
# STACK PROTOCOL — Head
# =============================================================================
# A self-contained memory system. Knows nothing about models or core plugins.
# Stores, retrieves, searches, and maintains memories. That's it.
#
# A stack can be:
# - Attached to one core, or many, or none
# - Shared between entities (collaborative memory)
# - Forked/snapshotted for experimentation
# - Swapped at runtime (context switching)
#
# The stack's schema is sovereign. Nothing outside the stack modifies it.
#
# The stack has components (sub-plugins) that extend its capabilities.
# Components are discovered via the kernle.stack_components entry point
# and are swappable at runtime. Embedding is just one such component.
#
# PROVENANCE AND WRITE DISCIPLINE:
#
# Memories should be written through an attached core following protocol
# standards. The core ensures full provenance: source attribution,
# timestamps, context tags, derived_from chains, and source_type tracking.
# This provenance is not just metadata — it is functionally important.
# Consolidation, trust computation, forgetting, and meta-memory all
# depend on knowing where a memory came from and how it was formed.
#
# A detached stack is a portable data artifact — it can be opened,
# queried, exported, synced, and maintained by scripts or tooling.
# But it is not an autonomous system. Direct writes to a detached
# stack produce memories with incomplete provenance (no source
# attribution, no context from the core's current composition).
# This is sometimes necessary for migration, repair, or bootstrap,
# but it degrades the stack's ability to reason about its own contents.
# Like doctoring medical records — sometimes clinically necessary,
# never clean, and should be minimized.
#
# The save_*() methods on the stack are low-level storage operations.
# They accept whatever is passed to them. The core's routed methods
# (episode(), belief(), etc.) are the protocol-compliant entry points
# that ensure provenance is complete before calling save_*().
# =============================================================================


@runtime_checkable
class StackProtocol(
    StackLifecycleProtocol,
    StackComponentManagerProtocol,
    StackWriterProtocol,
    StackReaderProtocol,
    StackSearchProtocol,
    StackLoaderProtocol,
    StackSuggestionProtocol,
    StackMetaMemoryProtocol,
    StackProcessingProtocol,
    StackSettingsProtocol,
    StackTrustProtocol,
    StackFeatureProtocol,
    StackSyncProtocol,
    StackExportProtocol,
    Protocol,
):
    """Full stack interface — backwards compatible composite.

    A self-contained, portable unit of memory. Sub-protocols can
    be used independently for type narrowing.

    The intended write path is: core -> stack (with full provenance).
    Direct writes are possible but produce incomplete provenance.

    Implementations: SQLiteStack (default), InMemoryStack, and custom backends.
    """

    @property
    def stack_id(self) -> str:
        """Unique identifier for this stack."""
        ...

    @property
    def schema_version(self) -> int:
        """Current schema version of this stack's storage."""
        ...


# =============================================================================
# PLUGIN CONTEXT — What the core gives to plugins
# =============================================================================
# This is the plugin's window into the system. Mediated access only.
# The plugin never sees raw stacks, raw models, or raw connections.
#
# Memory writes go through here, tagged with the plugin's identity
# as the source. When the plugin is removed, these memories remain
# in the stack — they belong to the stack now, attributed to the plugin.
# =============================================================================


@runtime_checkable
class PluginContext(Protocol):
    """Mediated interface from core to an active plugin."""

    @property
    def core_id(self) -> str:
        """The core this plugin is attached to."""
        ...

    @property
    def active_stack_id(self) -> Optional[str]:
        """The currently active stack's ID (None if no stack)."""
        ...

    @property
    def plugin_name(self) -> str:
        """This plugin's name (used for source attribution)."""
        ...

    # ---- Memory Write ----
    # All writes are automatically attributed with source=f"plugin:{plugin_name}"
    # so the stack knows where memories came from.

    def episode(
        self,
        objective: str,
        outcome: str,
        *,
        lessons: Optional[list[str]] = None,
        repeat: Optional[list[str]] = None,
        avoid: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        """Write an episode. Returns memory ID or None if no active stack."""
        ...

    def belief(
        self,
        statement: str,
        *,
        belief_type: BeliefType = "fact",
        confidence: float = 0.8,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        """Write a belief."""
        ...

    def value(
        self,
        name: str,
        statement: str,
        *,
        priority: int = 50,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        """Write a value."""
        ...

    def goal(
        self,
        title: str,
        *,
        description: Optional[str] = None,
        goal_type: GoalType = "task",
        priority: str = "medium",
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        """Write a goal."""
        ...

    def note(
        self,
        content: str,
        *,
        note_type: NoteType = "note",
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        context: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        """Write a note."""
        ...

    def relationship(
        self,
        other_entity_id: str,
        *,
        trust_level: Optional[float] = None,
        interaction_type: Optional[str] = None,
        notes: Optional[str] = None,
        entity_type: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> Optional[str]:
        """Write or update a relationship."""
        ...

    def raw(
        self,
        content: str,
    ) -> Optional[str]:
        """Write a raw entry."""
        ...

    # ---- Memory Read ----

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[list[SearchRecordType]] = None,
        context: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search the active stack."""
        ...

    def get_relationships(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_trust: Optional[float] = None,
    ) -> list[Relationship]:
        """Read relationships from the active stack."""
        ...

    def get_goals(
        self,
        *,
        status: Optional[GoalStatus] = None,
        context: Optional[str] = None,
    ) -> list[Goal]:
        """Read goals from the active stack."""
        ...

    # ---- Trust ----

    def trust_set(
        self,
        entity: str,
        domain: str,
        score: float,
        *,
        evidence: Optional[str] = None,
    ) -> Optional[str]:
        """Set trust for an entity in a domain."""
        ...

    def trust_get(
        self,
        entity: str,
        *,
        domain: Optional[str] = None,
    ) -> list[TrustAssessment]:
        """Get trust assessments for an entity."""
        ...

    # ---- Plugin's Own Storage ----

    def get_data_dir(self) -> Path:
        """Directory for this plugin's operational state.

        Scoped: ~/.kernle/plugins/{plugin_name}/data/
        The plugin owns this directory entirely. It is expected to
        clean it up in deactivate(). If the plugin is uninstalled,
        the core removes this directory.
        """
        ...

    def get_config(self, key: str, default: Any = None) -> Any:
        """Read plugin-specific configuration.

        Reads from the plugin's config namespace. Configuration
        is stored by the core, not by the plugin.
        """
        ...

    def get_secret(self, key: str) -> Optional[str]:
        """Read a secret (API key, credential).

        Secrets are stored in-memory by the core, scoped to the
        plugin. Not persisted in memory stacks or across restarts.
        """
        ...


# =============================================================================
# PLUGIN PROTOCOL — Limbs
# =============================================================================
# A capability extension. Manages its own operational state. Removable.
#
# Lifecycle:
#   1. Discovery:  Core finds plugin via entry point (kernle.plugins group)
#   2. Load:       Core calls activate(context) — plugin initializes
#   3. Use:        Plugin provides CLI commands, MCP tools, actions
#   4. Unload:     Core calls deactivate() — plugin cleans up everything
#
# After unload, the only trace is memories written to the stack through
# the context during the plugin's lifetime. Those memories belong to the
# stack now. The plugin itself is gone completely.
# =============================================================================


@runtime_checkable
class PluginProtocol(Protocol):
    """Interface for capability extensions.

    Implementations: analytics, web-search, code-executor, etc.
    """

    @property
    def name(self) -> str:
        """Plugin identifier. Used for namespacing, config, data dir."""
        ...

    @property
    def version(self) -> str:
        """Semantic version of this plugin."""
        ...

    @property
    def protocol_version(self) -> int:
        """The PROTOCOL_VERSION this plugin was built against.

        The core checks this on load. If it doesn't match, the core
        warns (or refuses to load if major incompatibility).
        """
        ...

    @property
    def description(self) -> str:
        """One-line description."""
        ...

    def capabilities(self) -> list[str]:
        """What this plugin can do.

        Used for discovery, status display, and source tagging.
        e.g., ['analytics', 'data-pipeline', 'reporting']
        """
        ...

    def activate(self, context: PluginContext) -> None:
        """Initialize the plugin.

        Called when loaded into a core. The plugin should:
        - Store the context reference for later use
        - Initialize its own operational state
        - Set up any connections it needs (APIs, services)
        - Restore state from get_data_dir() if applicable

        Args:
            context: Mediated access to core, stacks, and config.
        """
        ...

    def deactivate(self) -> None:
        """Shut down the plugin.

        Called when unloaded from a core. The plugin MUST:
        - Close all connections
        - Flush any pending operations
        - Clean up files in get_data_dir() if ephemeral
        - Release all resources

        After this returns, the plugin is gone. The only residue
        is memories written to the stack through the context.
        """
        ...

    def register_cli(self, subparsers: Any) -> None:
        """Add subcommands to the core's CLI.

        Args:
            subparsers: The argparse subparsers action from the core CLI.
                       Add commands like: subparsers.add_parser('wallet', ...)
        """
        ...

    def register_tools(self) -> list[ToolDefinition]:
        """Return tool definitions for MCP / model integration.

        These tools become available to the model when this plugin
        is active. The core registers them with the MCP server.
        """
        ...

    def on_load(self, load_context: dict[str, Any]) -> None:
        """Contribute to `kernle load` working memory output.

        Called when the core assembles working memory. The plugin
        can add operational state that the model should know about.

        Args:
            load_context: The working memory dict being built.
                         Add your key: load_context['my_plugin'] = {...}
        """
        ...

    def on_status(self, status: dict[str, Any]) -> None:
        """Contribute to `kernle status` output.

        Args:
            status: The status dict being built.
        """
        ...

    def health_check(self) -> PluginHealth:
        """Check if the plugin is functioning correctly.

        Called by `kernle doctor` or on demand.
        """
        ...

    def handle_cli(self, args: Any, k: Any) -> None:
        """Dispatch a CLI command for this plugin.

        Called by the core CLI when a command registered by this plugin
        is invoked. Receives parsed args and the Kernle instance.

        Args:
            args: Parsed argparse namespace.
            k: The Kernle instance (provides stack, entity, storage).
        """
        ...

    def cli_commands(self) -> list[str]:
        """Return top-level CLI command names registered by this plugin.

        Used by the core CLI dispatcher to route commands to this plugin.
        Must match the parser names added in register_cli().

        Returns:
            List of command name strings, e.g. ["wallet", "job", "skills"].
        """
        ...


# =============================================================================
# CORE PROTOCOL — Torso
# =============================================================================
# The bus. Connects stacks, plugins, and the model. Has an ID.
# Routes operations. Manages the composition.
#
# The core is not "the entity." The entity is the composition.
# But the core is what holds the composition together.
#
# The core_id persists across reconfigurations. Swap the model,
# change the stacks, add/remove plugins — the core_id stays.
# =============================================================================


@runtime_checkable
class CoreProtocol(Protocol):
    """Interface for the coordinator.

    The default implementation is kernle.entity.Entity.
    """

    @property
    def core_id(self) -> str:
        """This core's persistent identifier."""
        ...

    # ---- Model ----

    @property
    def model(self) -> Optional[ModelProtocol]:
        """The currently bound model (None if not yet configured)."""
        ...

    def set_model(self, model: ModelProtocol) -> None:
        """Bind a model to this core.

        Replaces any previously bound model.
        """
        ...

    # ---- Stacks ----

    @property
    def active_stack(self) -> Optional[StackProtocol]:
        """The currently active stack (None if none attached)."""
        ...

    @property
    def stacks(self) -> dict[str, StackInfo]:
        """All attached stacks, keyed by stack_id."""
        ...

    def attach_stack(
        self,
        stack: StackProtocol,
        *,
        alias: Optional[str] = None,
        set_active: bool = True,
    ) -> str:
        """Attach a memory stack.

        Args:
            stack: The stack to attach.
            alias: Optional cosmetic display name. Stored as-is (None when omitted).
            set_active: Make this the active stack.

        Returns:
            The stack_id of the attached stack.

        Raises:
            ValueError: If a stack with the same stack_id is already attached.
        """
        ...

    def detach_stack(self, stack_id: str) -> None:
        """Detach a stack.

        The stack continues to exist independently — it's just
        no longer connected to this core.
        """
        ...

    def set_active_stack(self, stack_id: str) -> None:
        """Switch which attached stack is active.

        All routed memory operations go to the active stack.
        """
        ...

    # ---- Plugins ----

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        """All loaded plugins, keyed by name."""
        ...

    def load_plugin(self, plugin: PluginProtocol) -> None:
        """Load and activate a plugin.

        Creates a PluginContext for this plugin and calls activate().
        Registers the plugin's CLI commands and MCP tools.
        """
        ...

    def unload_plugin(self, name: str) -> None:
        """Deactivate and unload a plugin.

        Calls deactivate(), unregisters CLI commands and tools,
        removes the plugin from the core.
        """
        ...

    def discover_plugins(self) -> list[PluginInfo]:
        """Discover installed plugins via entry points.

        Scans the 'kernle.plugins' entry point group for available
        plugins, whether currently loaded or not.
        """
        ...

    # ---- Routed Memory Operations ----
    # The proper write path for memories. These methods:
    # 1. Populate provenance fields (source, source_type, timestamps)
    # 2. Set context from the core's current composition
    # 3. Route to the active stack's save_*() methods
    # 4. Trigger component hooks (on_save) via the stack
    #
    # This is how memories should be created. Direct stack.save_*()
    # calls bypass provenance and should be reserved for maintenance.
    #
    # Raise NoActiveStackError if no stack is attached.

    def episode(
        self,
        objective: str,
        outcome: str,
        *,
        lessons: Optional[list[str]] = None,
        repeat: Optional[list[str]] = None,
        avoid: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str: ...

    def belief(
        self,
        statement: str,
        *,
        type: BeliefType = "fact",
        confidence: float = 0.8,
        foundational: bool = False,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str: ...

    def value(
        self,
        name: str,
        statement: str,
        *,
        priority: int = 50,
        type: str = "core_value",
        foundational: bool = False,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str: ...

    def goal(
        self,
        title: str,
        *,
        description: Optional[str] = None,
        goal_type: GoalType = "task",
        priority: str = "medium",
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str: ...

    def note(
        self,
        content: str,
        *,
        type: NoteType = "note",
        speaker: Optional[str] = None,
        reason: Optional[str] = None,
        tags: Optional[list[str]] = None,
        protect: bool = False,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str: ...

    def drive(
        self,
        drive_type: DriveType,
        *,
        intensity: float = 0.5,
        focus_areas: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str: ...

    def relationship(
        self,
        other_stack_id: str,
        *,
        trust_level: Optional[float] = None,
        notes: Optional[str] = None,
        interaction_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str: ...

    def raw(
        self,
        content: str,
        *,
        source: Optional[str] = None,
    ) -> str: ...

    # ---- Routed Search & Load ----

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[list[SearchRecordType]] = None,
        context: Optional[str] = None,
    ) -> list[SearchResult]: ...

    def load(
        self,
        *,
        token_budget: int = 8000,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Assemble working memory.

        Loads from the active stack, then calls on_load() on all
        active plugins so they can contribute operational state.
        """
        ...

    def status(self) -> dict[str, Any]:
        """Full system status.

        Includes: core info, model info, stack stats, plugin status.
        Calls on_status() on all active plugins.
        """
        ...

    # ---- Routed Trust ----

    def trust_set(
        self,
        entity: str,
        domain: str,
        score: float,
        *,
        evidence: Optional[str] = None,
    ) -> str: ...

    def trust_get(
        self,
        entity: str,
        *,
        domain: Optional[str] = None,
    ) -> list[TrustAssessment]: ...

    def trust_list(
        self,
        *,
        domain: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> list[TrustAssessment]: ...

    # ---- Routed Memory Control ----

    def weaken(
        self,
        memory_type: MemoryType,
        memory_id: str,
        amount: float,
        *,
        reason: Optional[str] = None,
    ) -> bool:
        """Reduce a memory's strength by a given amount."""
        ...

    def forget(
        self,
        memory_type: MemoryType,
        memory_id: str,
        reason: str,
    ) -> bool:
        """Forget a memory (set strength to 0.0)."""
        ...

    def recover(
        self,
        memory_type: MemoryType,
        memory_id: str,
    ) -> bool:
        """Recover a forgotten memory (restore strength to 0.2)."""
        ...

    def verify(
        self,
        memory_type: MemoryType,
        memory_id: str,
        *,
        evidence: Optional[str] = None,
    ) -> bool:
        """Verify a memory: boost strength and increment verification count."""
        ...

    def protect(
        self,
        memory_type: MemoryType,
        memory_id: str,
        protected: bool = True,
    ) -> bool:
        """Protect or unprotect a memory from forgetting/decay."""
        ...

    def process(
        self,
        transition: Optional[ProcessingTransition] = None,
        *,
        force: bool = False,
        allow_no_inference_override: bool = False,
        auto_promote: bool = False,
    ) -> list:
        """Run memory processing sessions.

        By default, creates suggestions for review rather than directly
        promoting memories. Set auto_promote=True to directly write memories.

        When no model is bound (inference unavailable), identity-layer
        transitions are blocked by the no-inference safety policy.
        Values can never be created without inference. Other identity
        layers require explicit override with force=True and
        allow_no_inference_override=True.

        Args:
            transition: Specific layer transition to process (None = check all)
            force: Process even if triggers aren't met
            allow_no_inference_override: Allow identity-layer writes without
                inference (except values). Only effective with force=True.
            auto_promote: If True, directly write memories. If False (default),
                create suggestions for review.

        Returns:
            List of ProcessingResult for each transition that ran.
        """
        ...

    # ---- Routed Sync ----

    def sync(self) -> SyncResult: ...
    def checkpoint(self, message: str = "") -> str: ...

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
    ) -> "CoreProtocol":
        """Restore a core from a checkpoint file.

        Reads checkpoint JSON, validates schema version, extracts
        binding, and delegates to from_binding(Binding) for rehydration.
        """
        ...

    # ---- Binding Management ----

    def get_binding(self) -> Binding:
        """Snapshot the current composition."""
        ...

    @classmethod
    def from_binding(
        cls,
        binding: Binding,
        *,
        data_dir: Optional[Path] = None,
    ) -> "CoreProtocol":
        """Restore a core from a Binding object.

        Reconnects the model, attaches stacks, loads plugins.
        """
        ...


# =============================================================================
# ENTRY POINTS
# =============================================================================
# All components register via pyproject.toml entry points.
# The core discovers them at runtime via importlib.metadata.
#
# Core plugins (limbs):
#   [project.entry-points."kernle.plugins"]
#   my-plugin = "my_plugin:MyPlugin"
#
# Stack implementations:
#   [project.entry-points."kernle.stacks"]
#   sqlite = "kernle_stack:SQLiteStack"
#
# Model implementations:
#   [project.entry-points."kernle.models"]
#   anthropic = "kernle_anthropic:AnthropicModel"
#   ollama = "kernle_ollama:OllamaModel"
#
# Stack components (stack sub-plugins):
#   [project.entry-points."kernle.stack_components"]
#   embedding-ngram = "kernle_stack:LocalNgramEmbedding"
#   embedding-st = "kernle_embeddings_st:SentenceTransformerEmbedding"
#   embedding-ollama = "kernle_ollama:OllamaEmbedding"
#   forgetting = "kernle_stack:SalienceForgetting"
#   consolidation = "kernle_stack:EpisodicConsolidation"
#   emotional-tagging = "kernle_stack:EmotionalTagging"
#   anxiety = "kernle_stack:AnxietyDetection"
#   suggestions = "kernle_stack:SuggestionExtraction"
#   meta-memory = "kernle_stack:MetaMemoryTracking"
#
# =============================================================================

ENTRY_POINT_GROUP_PLUGINS = "kernle.plugins"
ENTRY_POINT_GROUP_STACKS = "kernle.stacks"
ENTRY_POINT_GROUP_MODELS = "kernle.models"
ENTRY_POINT_GROUP_STACK_COMPONENTS = "kernle.stack_components"
