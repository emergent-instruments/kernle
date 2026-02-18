"""
Shared memory types for kernle.

All memory dataclasses live here. These are the shared vocabulary between
core, stacks, plugins, and models. A plugin creates an Episode; the stack
stores it. The types are the contract between them.

Previously these lived in kernle.storage.base. They are re-exported from
there for backwards compatibility.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# === Shared Utility Functions ===


def utc_now() -> str:
    """Get current timestamp as ISO string in UTC."""
    return datetime.now(timezone.utc).isoformat()


class ParseDatetimeError(ValueError):
    """Structured parse failure for ISO datetime strings."""

    def __init__(self, value: str, cause: Exception):
        super().__init__(f"Invalid ISO datetime string: {value!r}")
        self.value = value
        self.cause = cause


def parse_datetime(
    s: Optional[str], *, strict: bool = False
) -> Optional[datetime] | ParseDatetimeError:
    """Parse ISO datetime string."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        result = ParseDatetimeError(s, exc)
        if strict:
            raise result from exc
        return result


# === Sync Queue State Constants ===
# Integer values stored in sync_queue.synced column.
SYNC_PENDING = 0  # Not yet synced / awaiting push
SYNC_COMPLETED = 1  # Successfully synced to cloud
SYNC_DEAD_LETTER = 2  # Permanently failed after max retries


# === Enums ===


class SourceType(str, Enum):
    """How a memory was created/acquired.

    This is the single authoritative taxonomy for source_type values.
    All components, CLI, MCP, and processing code must use these values.
    """

    DIRECT_EXPERIENCE = "direct_experience"  # Directly observed/experienced
    INFERENCE = "inference"  # Inferred from other memories
    EXTERNAL = "external"  # Information received from another being (entity-neutral)
    CONSOLIDATION = "consolidation"  # Created during consolidation
    PROCESSING = "processing"  # Created by automated memory processing/promotion
    SEED = "seed"  # Initial seed memories provided at setup
    OBSERVATION = "observation"  # Passive observation (not direct interaction)
    IMPORTED = "imported"  # One-time direct import from another stack or file
    UNKNOWN = "unknown"  # Legacy or untracked


# Canonical set of valid source_type string values, derived from the enum.
# Use this for validation instead of maintaining separate lists.
VALID_SOURCE_TYPE_VALUES = frozenset(st.value for st in SourceType)


class MemoryType(str, Enum):
    """Canonical memory record type values."""

    EPISODE = "episode"
    BELIEF = "belief"
    VALUE = "value"
    GOAL = "goal"
    NOTE = "note"
    DRIVE = "drive"
    RELATIONSHIP = "relationship"
    RAW = "raw"
    PLAYBOOK = "playbook"
    TRUST_ASSESSMENT = "trust_assessment"
    ENTITY_MODEL = "entity_model"
    EPOCH = "epoch"
    SUMMARY = "summary"
    SELF_NARRATIVE = "self_narrative"
    SUGGESTION = "suggestion"


VALID_MEMORY_TYPE_VALUES = frozenset(m.value for m in MemoryType)


class SyncStatus(Enum):
    """Sync status for a record."""

    LOCAL_ONLY = "local_only"  # Not yet synced to cloud
    SYNCED = "synced"  # In sync with cloud
    PENDING_PUSH = "pending_push"  # Local changes need to be pushed
    PENDING_PULL = "pending_pull"  # Cloud has newer version
    CONFLICT = "conflict"  # Conflicting changes


# === Errors ===


class VersionConflictError(Exception):
    """Raised when a record's version doesn't match the expected version.

    This indicates a concurrent modification - another process updated the
    record between when we read it and when we tried to save our changes.
    """

    def __init__(self, table: str, record_id: str, expected_version: int, actual_version: int):
        self.table = table
        self.record_id = record_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Version conflict on {table}/{record_id}: "
            f"expected version {expected_version}, found {actual_version}"
        )


# === Sync Types ===


@dataclass
class SyncConflict:
    """Details of a sync conflict that was resolved.

    When local and cloud versions of a record differ, the sync engine
    resolves the conflict using last-write-wins and records the details
    here for user visibility.
    """

    id: str  # Unique ID for this conflict record
    table: str  # Table name (episodes, notes, beliefs, etc.)
    record_id: str  # ID of the record that had a conflict
    local_version: Dict[str, Any]  # Snapshot of local version before resolution
    cloud_version: Dict[str, Any]  # Snapshot of cloud version
    resolution: str  # "local_wins" or "cloud_wins"
    resolved_at: datetime  # When the conflict was resolved
    local_summary: Optional[str] = None  # Human-readable summary of local content
    cloud_summary: Optional[str] = None  # Human-readable summary of cloud content
    source: Optional[str] = None  # Origin of this conflict record (e.g., sync_engine)
    diff_hash: Optional[str] = None  # Stable hash of compared values
    policy_decision: Optional[str] = None  # Resolution policy used


@dataclass
class SyncResult:
    """Result of a sync operation."""

    pushed: int = 0  # Records pushed to cloud
    pulled: int = 0  # Records pulled from cloud
    conflicts: List[SyncConflict] = field(default_factory=list)  # Detailed conflict records
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def conflict_count(self) -> int:
        """Number of conflicts encountered."""
        return len(self.conflicts)


@dataclass
class QueuedChange:
    """A change queued for sync."""

    id: int
    table_name: str
    record_id: str
    operation: str  # 'insert', 'update', 'delete'
    payload: Optional[str] = None  # JSON payload for the change
    queued_at: Optional[datetime] = None
    # Retry tracking for resilient sync
    retry_count: int = 0
    last_error: Optional[str] = None
    last_attempt_at: Optional[datetime] = None


# === Meta-Memory Types ===


@dataclass
class ConfidenceChange:
    """A record of confidence change for tracking history."""

    timestamp: datetime
    old_confidence: float
    new_confidence: float
    reason: Optional[str] = None


@dataclass
class MemoryLineage:
    """Provenance chain for a memory."""

    source_type: SourceType
    source_episodes: List[str]  # Episode IDs that support this memory
    derived_from: List[str]  # Memory IDs this was derived from (format: type:id)
    confidence_history: List[ConfidenceChange]


# === Memory Dataclasses ===


@dataclass
class RawEntry:
    """A raw memory entry - unstructured blob capture for later processing.

    The raw layer is designed for zero-friction brain dumps. The agent dumps
    whatever they want into the blob field; the system only tracks housekeeping
    metadata like when it was captured and whether it's been processed.

    Note: For backward compatibility, blob/captured_at can be None if the legacy
    content/timestamp fields are provided. New code should always use blob/captured_at.
    """

    id: str
    stack_id: str
    # Primary fields (new schema)
    blob: Optional[str] = None  # The unstructured brain dump - no validation, no length limits
    captured_at: Optional[datetime] = None  # When the entry was captured
    source: str = "unknown"  # Auto-populated: cli|mcp|sdk|import|unknown
    processed: bool = False
    processed_into: Optional[List[str]] = None  # Audit trail: ["episode:abc", "note:xyz"]
    # Sync fields
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False

    # DEPRECATED fields - kept for backward compatibility during migration
    # These will be removed in a future version
    content: Optional[str] = None  # Use blob instead
    timestamp: Optional[datetime] = None  # Use captured_at instead
    tags: Optional[List[str]] = None  # Include in blob text instead
    confidence: float = 1.0  # Not meaningful for raw dumps
    source_type: SourceType = SourceType.DIRECT_EXPERIENCE  # Meta-memory concept, not for raw

    def __post_init__(self):
        """Handle backward compatibility for blob/captured_at."""
        # If blob is not set but content is, use content as blob
        if self.blob is None and self.content is not None:
            self.blob = self.content
        # If captured_at is not set but timestamp is, use timestamp as captured_at
        if self.captured_at is None and self.timestamp is not None:
            self.captured_at = self.timestamp
        # For backward compatibility, also populate legacy fields from new fields
        if self.content is None and self.blob is not None:
            self.content = self.blob
        if self.timestamp is None and self.captured_at is not None:
            self.timestamp = self.captured_at


@dataclass
class Episode:
    """An episode/experience record."""

    id: str
    stack_id: str
    objective: str
    outcome: str
    outcome_type: Optional[str] = None
    lessons: Optional[List[str]] = None
    repeat: Optional[List[str]] = None  # Patterns to replicate
    avoid: Optional[List[str]] = None  # Patterns to avoid
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    # Emotional memory fields
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    emotional_arousal: float = 0.0  # 0.0 (calm) to 1.0 (intense)
    emotional_tags: Optional[List[str]] = None  # ["joy", "frustration", "curiosity"]
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: SourceType = SourceType.DIRECT_EXPERIENCE  # SourceType value
    source_entity: Optional[str] = None  # Who provided it (name, email, or ID; entity-neutral)
    source_episodes: Optional[List[str]] = None  # IDs of related episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Strength and access fields
    strength: float = 1.0  # 0.0 (forgotten) to 1.0 (strong) — replaces binary is_forgotten
    times_accessed: int = 0  # Number of times this memory was retrieved
    last_accessed: Optional[datetime] = None  # When last accessed/retrieved
    is_protected: bool = False  # Never decay (core identity memories)
    # Processing state
    processed: bool = False  # Whether this episode has been processed for promotion
    # Context/scope fields for project-specific memories
    context: Optional[str] = None  # e.g., "project:api-service", "repo:myorg/myrepo"
    context_tags: Optional[List[str]] = None  # Additional context tags for filtering
    # Epoch tracking
    epoch_id: Optional[str] = None
    # Privacy fields (Phase 8a)
    subject_ids: Optional[List[str]] = None  # Who/what is this about
    access_grants: Optional[List[str]] = None  # Who can see this (empty = private to self)
    consent_grants: Optional[List[str]] = None  # Who authorized sharing


@dataclass
class Belief:
    """A belief record."""

    id: str
    stack_id: str
    statement: str
    belief_type: str = "fact"
    confidence: float = 0.8
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    source_type: SourceType = SourceType.DIRECT_EXPERIENCE  # SourceType value
    source_entity: Optional[str] = None  # Who provided it (name, email, or ID; entity-neutral)
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Belief revision fields
    # DEPRECATED (v0.14+): supersedes/superseded_by are no longer written.
    # Revision tracking uses audit log (belief.revised/belief.deactivated).
    # These fields remain for reading pre-v0.14 data; will be removed in v0.15.
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    times_reinforced: int = 0  # How many times confirmed
    is_active: bool = True  # False if revised/archived
    # Strength and access fields
    strength: float = 1.0  # 0.0 (forgotten) to 1.0 (strong)
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = False
    # Context/scope fields for project-specific memories
    context: Optional[str] = None  # e.g., "project:api-service", "repo:myorg/myrepo"
    context_tags: Optional[List[str]] = None  # Additional context tags for filtering
    # Epoch tracking
    epoch_id: Optional[str] = None
    # Privacy fields (Phase 8a)
    subject_ids: Optional[List[str]] = None  # Who/what is this about
    access_grants: Optional[List[str]] = None  # Who can see this (empty = private to self)
    consent_grants: Optional[List[str]] = None  # Who authorized sharing
    # Processing state
    processed: bool = False  # Whether this belief has been processed for promotion
    # Belief scope and domain metadata (KEP v3)
    belief_scope: str = "world"  # 'self' | 'world' | 'relational'
    source_domain: Optional[str] = None  # "coding", "communication", etc.
    cross_domain_applications: Optional[List[str]] = None  # domains this belief applies to
    abstraction_level: str = "specific"  # 'specific' | 'domain' | 'universal'


@dataclass
class Value:
    """A value record."""

    id: str
    stack_id: str
    name: str
    statement: str
    priority: int = 50
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.9  # Values tend to be high-confidence
    source_type: SourceType = SourceType.DIRECT_EXPERIENCE  # SourceType value
    source_entity: Optional[str] = None  # Who provided it (name, email, or ID; entity-neutral)
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Strength and access fields
    strength: float = 1.0  # 0.0 (forgotten) to 1.0 (strong)
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = True  # Values are protected by default
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None
    # Epoch tracking
    epoch_id: Optional[str] = None
    # Privacy fields (Phase 8a)
    subject_ids: Optional[List[str]] = None  # Who/what is this about
    access_grants: Optional[List[str]] = None  # Who can see this (empty = private to self)
    consent_grants: Optional[List[str]] = None  # Who authorized sharing


@dataclass
class Goal:
    """A goal record."""

    id: str
    stack_id: str
    title: str
    description: Optional[str] = None
    goal_type: str = "task"  # task, aspiration, commitment, exploration
    priority: str = "medium"
    status: str = "active"
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: SourceType = SourceType.DIRECT_EXPERIENCE  # SourceType value
    source_entity: Optional[str] = None  # Who provided it (name, email, or ID; entity-neutral)
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Strength and access fields
    strength: float = 1.0  # 0.0 (forgotten) to 1.0 (strong)
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = False
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None
    # Epoch tracking
    epoch_id: Optional[str] = None
    # Privacy fields (Phase 8a)
    subject_ids: Optional[List[str]] = None  # Who/what is this about
    access_grants: Optional[List[str]] = None  # Who can see this (empty = private to self)
    consent_grants: Optional[List[str]] = None  # Who authorized sharing


@dataclass
class Note:
    """A note/memory record."""

    id: str
    stack_id: str
    content: str
    note_type: str = "note"
    speaker: Optional[str] = None
    reason: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: SourceType = SourceType.DIRECT_EXPERIENCE  # SourceType value
    source_entity: Optional[str] = None  # Who provided it (name, email, or ID; entity-neutral)
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Strength and access fields
    strength: float = 1.0  # 0.0 (forgotten) to 1.0 (strong)
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = False
    # Processing state
    processed: bool = False  # Whether this note has been processed for promotion
    # Context/scope fields for project-specific memories
    context: Optional[str] = None  # e.g., "project:api-service", "repo:myorg/myrepo"
    context_tags: Optional[List[str]] = None  # Additional context tags for filtering
    # Epoch tracking
    epoch_id: Optional[str] = None
    # Privacy fields (Phase 8a)
    subject_ids: Optional[List[str]] = None  # Who/what is this about
    access_grants: Optional[List[str]] = None  # Who can see this (empty = private to self)
    consent_grants: Optional[List[str]] = None  # Who authorized sharing


@dataclass
class Drive:
    """A drive/motivation record."""

    id: str
    stack_id: str
    drive_type: str
    intensity: float = 0.5
    focus_areas: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: SourceType = SourceType.DIRECT_EXPERIENCE  # SourceType value
    source_entity: Optional[str] = None  # Who provided it (name, email, or ID; entity-neutral)
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Strength and access fields
    strength: float = 1.0  # 0.0 (forgotten) to 1.0 (strong)
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = True  # Drives are protected by default
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None
    # Epoch tracking
    epoch_id: Optional[str] = None
    # Privacy fields (Phase 8a)
    subject_ids: Optional[List[str]] = None  # Who/what is this about
    access_grants: Optional[List[str]] = None  # Who can see this (empty = private to self)
    consent_grants: Optional[List[str]] = None  # Who authorized sharing


@dataclass
class Relationship:
    """A relationship record."""

    id: str
    stack_id: str
    entity_name: str
    entity_type: str
    relationship_type: str
    notes: Optional[str] = None
    sentiment: float = 0.0
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: SourceType = SourceType.DIRECT_EXPERIENCE  # SourceType value
    source_entity: Optional[str] = None  # Who provided it (name, email, or ID; entity-neutral)
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Strength and access fields
    strength: float = 1.0  # 0.0 (forgotten) to 1.0 (strong)
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = False
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None
    # Epoch tracking
    epoch_id: Optional[str] = None
    # Privacy fields (Phase 8a)
    subject_ids: Optional[List[str]] = None  # Who/what is this about
    access_grants: Optional[List[str]] = None  # Who can see this (empty = private to self)
    consent_grants: Optional[List[str]] = None  # Who authorized sharing


@dataclass
class RelationshipHistoryEntry:
    """A history entry for relationship changes.

    Tracks changes to relationships over time, including trust changes,
    type changes, interactions, and notes. This enables understanding
    how a relationship has evolved.
    """

    id: str
    stack_id: str
    relationship_id: str  # FK to relationships.id
    entity_name: str  # Denormalized for easy querying
    event_type: str  # interaction, trust_change, type_change, note
    old_value: Optional[str] = None  # JSON: previous state
    new_value: Optional[str] = None  # JSON: new state
    episode_id: Optional[str] = None  # Related episode if applicable
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class Epoch:
    """A temporal epoch - a named era in the agent's timeline.

    Epochs mark significant phases or transitions in the agent's experience.
    They enable epoch-scoped loading and temporal navigation.
    """

    id: str
    stack_id: str
    epoch_number: int
    name: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None  # None = still active
    trigger_type: str = "declared"  # declared, detected, system
    trigger_description: Optional[str] = None
    summary: Optional[str] = None
    key_belief_ids: Optional[List[str]] = None
    key_relationship_ids: Optional[List[str]] = None
    key_goal_ids: Optional[List[str]] = None
    dominant_drive_ids: Optional[List[str]] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class TrustAssessment:
    """A trust assessment for an entity (KEP v3 section 8)."""

    id: str
    stack_id: str
    entity: str
    dimensions: Dict[str, Any]
    authority: Optional[List[Dict[str, Any]]] = None
    evidence_episode_ids: Optional[List[str]] = None
    last_updated: Optional[datetime] = None
    created_at: Optional[datetime] = None
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class EntityModel:
    """A mental model of an entity (person, agent, organization).

    Entity models capture behavioral observations, preferences, and
    capabilities that the agent has learned about other entities through
    interaction. They support building richer relationship understanding.
    """

    id: str
    stack_id: str
    entity_name: str  # The entity this model is about
    model_type: str  # behavioral, preference, capability
    observation: str  # The actual observation/model content
    confidence: float = 0.7
    source_episodes: Optional[List[str]] = None  # Episodes supporting this observation
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Privacy fields - subject_ids auto-populated from entity_name
    subject_ids: Optional[List[str]] = None
    access_grants: Optional[List[str]] = None
    consent_grants: Optional[List[str]] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class DiagnosticSession:
    """A formal diagnostic session for structured memory health checks.

    Diagnostic sessions provide a controlled framework for examining
    memory system health with explicit consent and access boundaries.

    session_type values:
    - self_requested: SI initiated the session
    - routine: Scheduled/periodic check
    - anomaly_triggered: Triggered by detected anomaly
    - operator_initiated: Human operator requested

    access_level values:
    - structural: IDs and scores only (default, privacy-safe)
    - content: Can read statement text
    - full: Complete access
    """

    id: str
    stack_id: str
    session_type: str = "self_requested"
    access_level: str = "structural"
    status: str = "active"
    consent_given: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class DiagnosticReport:
    """A report produced by a diagnostic session.

    Contains structural findings only -- IDs, scores, counts, never
    memory content. Each finding has a severity, category, description,
    and recommendation.
    """

    id: str
    stack_id: str
    session_id: str  # References DiagnosticSession.id
    findings: Optional[List[Dict[str, Any]]] = None  # JSONB list of findings
    summary: Optional[str] = None
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class SelfNarrative:
    """An autobiographical self-narrative (KEP v3 section 9).

    Self-narratives are the agent's story about itself -- who it is,
    how it has developed, and what it aspires to become. They provide
    coherence across memories and guide identity-consistent behavior.

    narrative_type values:
    - identity: Who I am right now
    - developmental: How I got here / my growth story
    - aspirational: Who I want to become
    """

    id: str
    stack_id: str
    content: str
    narrative_type: str = "identity"  # identity | developmental | aspirational
    epoch_id: Optional[str] = None
    key_themes: Optional[List[str]] = None  # JSON array
    unresolved_tensions: Optional[List[str]] = None  # JSON array
    is_active: bool = True
    supersedes: Optional[str] = None  # ID of narrative this replaced
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class Summary:
    """An agent summary - fractal narrative compression across time scales.

    Summaries provide hierarchical compression of the agent's experience
    at different temporal scopes: month, quarter, year, decade, epoch.
    Higher-scope summaries supersede lower-scope ones, enabling efficient
    memory loading by skipping covered lower-scope summaries.
    """

    id: str
    stack_id: str
    scope: str  # 'month' | 'quarter' | 'year' | 'decade' | 'epoch'
    period_start: str  # ISO date string
    period_end: str  # ISO date string
    content: str  # SI-written narrative compression
    epoch_id: Optional[str] = None
    key_themes: Optional[List[str]] = None  # JSON array
    supersedes: Optional[List[str]] = None  # JSON array of lower-scope summary IDs
    is_protected: bool = True  # Never forgotten
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Sync metadata
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class Playbook:
    """A playbook/procedural memory record.

    Playbooks are "how I do things" memory - executable procedures
    learned from experience. They encode successful workflows as
    reusable step sequences with applicability conditions and failure modes.
    """

    id: str
    stack_id: str
    name: str  # "Deploy to production"
    description: str  # What this playbook does
    trigger_conditions: List[str]  # When to use this
    steps: List[Dict[str, Any]]  # [{action, details, adaptations}]
    failure_modes: List[str]  # What can go wrong
    recovery_steps: Optional[List[str]] = None  # How to recover
    mastery_level: str = "novice"  # novice/competent/proficient/expert
    times_used: int = 0
    success_rate: float = 0.0
    source_episodes: Optional[List[str]] = None  # Where this was learned
    tags: Optional[List[str]] = None
    # Meta-memory fields
    confidence: float = 0.8
    last_used: Optional[datetime] = None
    created_at: Optional[datetime] = None
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None
    # Privacy fields (Phase 8a)
    subject_ids: Optional[List[str]] = None  # Who/what is this about
    access_grants: Optional[List[str]] = None  # Who can see this (empty = private to self)
    consent_grants: Optional[List[str]] = None  # Who authorized sharing
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class MemorySuggestion:
    """A suggested memory extracted from raw entries.

    MemorySuggestions are auto-extracted patterns from raw entries that
    require agent review before being promoted to structured memories.
    This enables auto-extraction while keeping the agent in control.

    Workflow:
    1. Raw entry captured (manual or auto-capture)
    2. System extracts suggestions based on patterns
    3. SI reviews: approve (promote to memory), modify, or reject
    4. Approved suggestions become Episode, Belief, or Note records

    Status values:
    - pending: Awaiting review
    - promoted: Accepted and converted to structured memory
    - modified: Accepted with modifications
    - rejected: Declined (with optional reason)
    """

    id: str
    stack_id: str
    memory_type: str  # "episode", "belief", "note"
    content: Dict[str, Any]  # Structured data for the suggested memory
    confidence: float  # System confidence in this suggestion (0.0-1.0)
    source_raw_ids: List[str]  # Which raw entries this came from
    status: str = "pending"  # pending, promoted, modified, rejected
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_reason: Optional[str] = None
    # Link to promoted memory (if status is promoted/modified)
    promoted_to: Optional[str] = None  # Format: "type:id", e.g., "episode:abc123"
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


# Canonical set of valid suggestion status values.
# Use this for validation instead of maintaining separate lists.
VALID_SUGGESTION_STATUSES = frozenset(
    ["pending", "promoted", "modified", "rejected", "dismissed", "expired"]
)

# Canonical set of memory types that the suggestion system supports.
# Used by processing.py (producer) and sqlite_stack.py (resolver).
SUGGESTION_MEMORY_TYPES = frozenset(
    {"episode", "belief", "note", "goal", "relationship", "value", "drive"}
)


@dataclass
class SearchResult:
    """A search result with compatibility for both legacy protocol and stack shapes.

    Historically this class had two different shapes in `kernle.types` and
    `kernle.protocols`. Both are represented here to avoid silent shape drift.

    Preferred shape:
    - `record` + `record_type` for protocol-native callers.
    - `memory_type` + `memory_id` + `content` for storage-native callers.
    """

    record: Any = None  # Legacy protocol shape.
    record_type: str = ""
    score: float = 0.0
    memory_type: Optional[str] = None  # Storage-native shape.
    memory_id: Optional[str] = None
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_type and self.memory_type:
            self.record_type = self.memory_type
        if not self.memory_id and self.record is not None:
            self.memory_id = getattr(self.record, "id", None)
        if not self.memory_type:
            self.memory_type = self.record_type


# === Trust Constants ===


TRUST_THRESHOLDS: Dict[str, float] = {
    "suggest_belief": 0.3,
    "contradict_world_belief": 0.6,
    "contradict_self_belief": 0.7,
    "suggest_value_change": 0.8,
    "request_deletion": 0.9,
    "diagnostic": 0.85,
}

# Dynamic trust constants (KEP v3 section 8.6-8.7)
DEFAULT_TRUST: float = 0.5  # Neutral trust for unknown entities
TRUST_DECAY_RATE: float = 0.01  # Per-day decay factor toward neutral
TRUST_DEPTH_DECAY: float = 0.85  # 15% decay per hop in transitive chains
SELF_TRUST_FLOOR: float = 0.5  # Minimum self-trust (overridden by accuracy)

SEED_TRUST: List[Dict[str, Any]] = [
    {
        "entity": "stack-owner",
        "dimensions": {"general": {"score": 0.95}},
        "authority": [{"scope": "all"}],
    },
    {
        "entity": "self",
        "dimensions": {"general": {"score": 0.8}},
        "authority": [{"scope": "belief_revision", "requires_evidence": True}],
    },
    {
        "entity": "web-search",
        "dimensions": {"general": {"score": 0.5}, "medical": {"score": 0.3}},
        "authority": [{"scope": "information_only"}],
    },
    {
        "entity": "context-injection",
        "dimensions": {"general": {"score": 0.0}},
        "authority": [],
    },
]
