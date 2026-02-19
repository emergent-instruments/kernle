"""Storage protocol for Kernle backends.

This defines the interface that all storage backends must implement.
Currently supported:
- SQLiteStorage: Local-first storage with sqlite-vec for semantic search

Memory dataclasses and shared types have been moved to kernle.types.
They are re-exported here for backwards compatibility.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# DEPRECATED: Re-exports for backwards compatibility only.
# Import from kernle.types (canonical) or kernle.storage (convenience) instead.
# These re-exports will be removed in a future version.
from kernle.types import (  # noqa: F401
    DEFAULT_TRUST,
    SEED_TRUST,
    SELF_TRUST_FLOOR,
    TRUST_DECAY_RATE,
    TRUST_DEPTH_DECAY,
    TRUST_THRESHOLDS,
    Belief,
    ConfidenceChange,
    DiagnosticReport,
    DiagnosticSession,
    Drive,
    EntityModel,
    Episode,
    Epoch,
    Goal,
    MemoryLineage,
    MemorySuggestion,
    Note,
    Playbook,
    QueuedChange,
    RawEntry,
    Relationship,
    RelationshipHistoryEntry,
    SearchResult,
    SelfNarrative,
    SourceType,
    Summary,
    SyncConflict,
    SyncResult,
    SyncStatus,
    TrustAssessment,
    Value,
    VersionConflictError,
    parse_datetime,
    utc_now,
)

# === Sub-protocols ===


@runtime_checkable
class EpisodeStorage(Protocol):
    """Storage interface for episodes and emotional memory."""

    @abstractmethod
    def save_episode(self, episode: Episode) -> str:
        """Save an episode. Returns the episode ID."""
        ...

    @abstractmethod
    def get_episodes(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        processed: Optional[bool] = None,
    ) -> List[Episode]:
        """Get episodes, optionally filtered.

        Args:
            limit: Maximum number of episodes to return
            since: Only return episodes created after this time
            tags: Filter by tags
            processed: If True, only processed; if False, only unprocessed; None = all
        """
        ...

    @abstractmethod
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID."""
        ...

    @abstractmethod
    def update_episode_emotion(
        self, episode_id: str, valence: float, arousal: float, tags: Optional[List[str]] = None
    ) -> bool:
        """Update emotional associations for an episode.

        Args:
            episode_id: The episode to update
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal (0.0 to 1.0)
            tags: Emotional tags (e.g., ["joy", "excitement"])

        Returns:
            True if updated, False if episode not found
        """
        ...

    @abstractmethod
    def get_emotional_episodes(self, days: int = 7, limit: int = 100) -> List[Episode]:
        """Get episodes with emotional data for summary calculations.

        Args:
            days: Number of days to look back
            limit: Maximum episodes to retrieve

        Returns:
            Episodes with non-zero emotional data
        """
        ...

    @abstractmethod
    def search_by_emotion(
        self,
        valence_range: Optional[tuple] = None,
        arousal_range: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """Find episodes matching emotional criteria.

        Args:
            valence_range: (min, max) valence filter, e.g. (0.5, 1.0) for positive
            arousal_range: (min, max) arousal filter, e.g. (0.7, 1.0) for high arousal
            tags: Emotional tags to match (any match)
            limit: Maximum results

        Returns:
            List of matching episodes
        """
        ...


@runtime_checkable
class BeliefStorage(Protocol):
    """Storage interface for beliefs."""

    @abstractmethod
    def save_belief(self, belief: Belief) -> str:
        """Save a belief. Returns the belief ID."""
        ...

    @abstractmethod
    def get_beliefs(
        self,
        limit: int = 100,
        include_inactive: bool = False,
        processed: Optional[bool] = None,
    ) -> List[Belief]:
        """Get beliefs.

        Args:
            limit: Maximum number of beliefs to return
            include_inactive: If True, include superseded/archived beliefs
            processed: If True, only processed; if False, only unprocessed; None = all
        """
        ...

    @abstractmethod
    def find_belief(self, statement: str) -> Optional[Belief]:
        """Find a belief by statement (for deduplication)."""
        ...


@runtime_checkable
class ValueStorage(Protocol):
    """Storage interface for values."""

    @abstractmethod
    def save_value(self, value: Value) -> str:
        """Save a value. Returns the value ID."""
        ...

    @abstractmethod
    def get_values(self, limit: int = 100) -> List[Value]:
        """Get values, ordered by priority."""
        ...


@runtime_checkable
class GoalStorage(Protocol):
    """Storage interface for goals."""

    @abstractmethod
    def save_goal(self, goal: Goal) -> str:
        """Save a goal. Returns the goal ID."""
        ...

    @abstractmethod
    def get_goals(self, status: Optional[str] = "active", limit: int = 100) -> List[Goal]:
        """Get goals, optionally filtered by status."""
        ...


@runtime_checkable
class NoteStorage(Protocol):
    """Storage interface for notes."""

    @abstractmethod
    def save_note(self, note: Note) -> str:
        """Save a note. Returns the note ID."""
        ...

    @abstractmethod
    def get_notes(
        self, limit: int = 100, since: Optional[datetime] = None, note_type: Optional[str] = None
    ) -> List[Note]:
        """Get notes, optionally filtered."""
        ...


@runtime_checkable
class DriveStorage(Protocol):
    """Storage interface for drives."""

    @abstractmethod
    def save_drive(self, drive: Drive) -> str:
        """Save or update a drive. Returns the drive ID."""
        ...

    @abstractmethod
    def get_drives(self) -> List[Drive]:
        """Get all drives for the agent."""
        ...

    @abstractmethod
    def get_drive(self, drive_type: str) -> Optional[Drive]:
        """Get a specific drive by type."""
        ...


@runtime_checkable
class RelationshipStorage(Protocol):
    """Storage interface for relationships, relationship history, and entity models."""

    @abstractmethod
    def save_relationship(self, relationship: Relationship) -> str:
        """Save or update a relationship. Returns the relationship ID."""
        ...

    @abstractmethod
    def get_relationships(self, entity_type: Optional[str] = None) -> List[Relationship]:
        """Get relationships, optionally filtered by entity type."""
        ...

    @abstractmethod
    def get_relationship(self, entity_name: str) -> Optional[Relationship]:
        """Get a specific relationship by entity name."""
        ...

    # === Relationship History ===

    def save_relationship_history(self, entry: "RelationshipHistoryEntry") -> str:
        """Save a relationship history entry. Returns the entry ID.

        Args:
            entry: The history entry to save

        Returns:
            The entry ID
        """
        return entry.id

    def get_relationship_history(
        self,
        entity_name: str,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List["RelationshipHistoryEntry"]:
        """Get history entries for a relationship.

        Args:
            entity_name: Entity to get history for
            event_type: Filter by event type (interaction, trust_change, etc.)
            limit: Maximum entries to return

        Returns:
            List of history entries, most recent first
        """
        return []

    # === Entity Models ===

    def save_entity_model(self, model: "EntityModel") -> str:
        """Save an entity model. Returns the model ID.

        Args:
            model: The entity model to save

        Returns:
            The model ID
        """
        return model.id

    def get_entity_models(
        self,
        entity_name: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 100,
    ) -> List["EntityModel"]:
        """Get entity models, optionally filtered.

        Args:
            entity_name: Filter by entity name
            model_type: Filter by model type (behavioral, preference, capability)
            limit: Maximum models to return

        Returns:
            List of entity models
        """
        return []

    def get_entity_model(self, model_id: str) -> Optional["EntityModel"]:
        """Get a specific entity model by ID.

        Args:
            model_id: ID of the entity model

        Returns:
            The entity model or None if not found
        """
        return None


@runtime_checkable
class TrustStorage(Protocol):
    """Storage interface for trust assessments."""

    def save_trust_assessment(self, assessment: "TrustAssessment") -> str:
        """Save or update a trust assessment. Returns the assessment ID."""
        return assessment.id

    def get_trust_assessment(self, entity: str) -> Optional["TrustAssessment"]:
        """Get a trust assessment for a specific entity."""
        return None

    def get_trust_assessments(self) -> List["TrustAssessment"]:
        """Get all trust assessments for the agent."""
        return []

    def delete_trust_assessment(self, entity: str) -> bool:
        """Delete a trust assessment (soft delete)."""
        return False


@runtime_checkable
class PlaybookStorage(Protocol):
    """Storage interface for playbooks (procedural memory)."""

    @abstractmethod
    def save_playbook(self, playbook: "Playbook") -> str:
        """Save a playbook. Returns the playbook ID."""
        ...

    @abstractmethod
    def get_playbook(self, playbook_id: str) -> Optional["Playbook"]:
        """Get a specific playbook by ID."""
        ...

    @abstractmethod
    def list_playbooks(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List["Playbook"]:
        """Get playbooks, optionally filtered by tags."""
        ...

    @abstractmethod
    def search_playbooks(self, query: str, limit: int = 10) -> List["Playbook"]:
        """Search playbooks by name, description, or triggers."""
        ...

    @abstractmethod
    def update_playbook_usage(self, playbook_id: str, success: bool) -> bool:
        """Update playbook usage statistics.

        Args:
            playbook_id: ID of the playbook
            success: Whether the usage was successful

        Returns:
            True if updated, False if playbook not found
        """
        ...


@runtime_checkable
class RawStorage(Protocol):
    """Storage interface for raw entries."""

    @abstractmethod
    def save_raw(
        self,
        blob: str,
        source: str = "unknown",
    ) -> str:
        """Save a raw entry for later processing. Returns the entry ID."""
        ...

    @abstractmethod
    def get_raw(self, raw_id: str) -> Optional[RawEntry]:
        """Get a specific raw entry by ID."""
        ...

    @abstractmethod
    def list_raw(
        self, processed: Optional[bool] = None, limit: int = 100, offset: int = 0
    ) -> List[RawEntry]:
        """Get raw entries, optionally filtered by processed state."""
        ...

    @abstractmethod
    def mark_raw_processed(self, raw_id: str, processed_into: List[str]) -> bool:
        """Mark a raw entry as processed into other memories.

        Args:
            raw_id: ID of the raw entry
            processed_into: List of memory refs (format: type:id)

        Returns:
            True if updated, False if not found
        """
        ...


@runtime_checkable
class SuggestionStorage(Protocol):
    """Storage interface for memory suggestions."""

    def save_suggestion(self, suggestion: MemorySuggestion) -> str:
        """Save a memory suggestion. Returns the suggestion ID.

        Args:
            suggestion: The suggestion to save

        Returns:
            The suggestion ID
        """
        return suggestion.id  # Default: just return ID (no-op)

    def get_suggestion(self, suggestion_id: str) -> Optional[MemorySuggestion]:
        """Get a specific suggestion by ID.

        Args:
            suggestion_id: ID of the suggestion

        Returns:
            The suggestion or None if not found
        """
        return None

    def get_suggestions(
        self,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemorySuggestion]:
        """Get suggestions, optionally filtered.

        Args:
            status: Filter by status (pending, promoted, modified, rejected)
            memory_type: Filter by suggested memory type (episode, belief, note)
            limit: Maximum suggestions to return

        Returns:
            List of suggestions matching the filters
        """
        return []

    def update_suggestion_status(
        self,
        suggestion_id: str,
        status: str,
        resolution_reason: Optional[str] = None,
        promoted_to: Optional[str] = None,
    ) -> bool:
        """Update the status of a suggestion.

        Args:
            suggestion_id: ID of the suggestion to update
            status: New status (pending, promoted, modified, rejected)
            resolution_reason: Optional reason for the resolution
            promoted_to: Reference to promoted memory (format: type:id)

        Returns:
            True if updated, False if suggestion not found
        """
        return False

    def delete_suggestion(self, suggestion_id: str) -> bool:
        """Delete a suggestion (soft delete by marking deleted=1).

        Args:
            suggestion_id: ID of the suggestion to delete

        Returns:
            True if deleted, False if not found
        """
        return False

    def expire_suggestions(self, max_age_hours: float = 168.0) -> List[str]:
        """Auto-dismiss pending suggestions older than max_age_hours.

        Args:
            max_age_hours: Age threshold in hours (default: 168 = 7 days)

        Returns:
            List of expired suggestion IDs
        """
        return []


@runtime_checkable
class SearchStorage(Protocol):
    """Storage interface for search and cloud search."""

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        prefer_cloud: bool = True,
    ) -> List[SearchResult]:
        """Search across memories using hybrid cloud/local strategy.

        Strategy:
        1. If cloud credentials are configured and prefer_cloud=True,
           try cloud search first with timeout
        2. On cloud failure or no credentials, fall back to local search
        3. Local search uses semantic vectors (if available) or text matching

        Args:
            query: Search query
            limit: Maximum results
            record_types: Filter by type (episode, note, belief, etc.)
            prefer_cloud: If True, try cloud search first (default True)

        Returns:
            List of SearchResult objects
        """
        ...

    def has_cloud_credentials(self) -> bool:
        """Check if cloud credentials are available for hybrid search.

        Returns:
            True if backend_url and auth_token are configured.
        """
        return False  # Default: no cloud credentials

    def cloud_health_check(self, timeout: float = 3.0) -> Dict[str, Any]:
        """Test cloud backend connectivity.

        Args:
            timeout: Request timeout in seconds (default 3s)

        Returns:
            Dict with keys:
            - 'healthy': bool indicating if cloud is reachable
            - 'latency_ms': response time in milliseconds (if healthy)
            - 'error': error message (if not healthy)
        """
        return {
            "healthy": False,
            "error": "Cloud search not supported by this storage backend",
        }


@runtime_checkable
class MetaMemoryStorage(Protocol):
    """Storage interface for meta-memory operations."""

    @abstractmethod
    def get_memory(self, memory_type: str, memory_id: str) -> Optional[Any]:
        """Get a memory by type and ID.

        Args:
            memory_type: Type of memory (episode, belief, value, goal, note, drive, relationship)
            memory_id: ID of the memory

        Returns:
            The memory record or None if not found
        """
        ...

    @abstractmethod
    def memory_exists(self, memory_type: str, memory_id: str) -> bool:
        """Check if a memory exists in the stack.

        Args:
            memory_type: Type of memory (episode, belief, value, goal, note, drive, relationship, raw)
            memory_id: ID of the memory

        Returns:
            True if the memory exists and is not deleted
        """
        ...

    @abstractmethod
    def update_strength(self, memory_type: str, memory_id: str, strength: float) -> bool:
        """Update the strength field of a memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            strength: New strength value (clamped to 0.0-1.0)

        Returns:
            True if updated, False if memory not found
        """
        ...

    @abstractmethod
    def update_memory_meta(
        self,
        memory_type: str,
        memory_id: str,
        confidence: Optional[float] = None,
        source_type: Optional[str] = None,
        source_episodes: Optional[List[str]] = None,
        derived_from: Optional[List[str]] = None,
        last_verified: Optional[datetime] = None,
        verification_count: Optional[int] = None,
        confidence_history: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Update meta-memory fields for a memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            confidence: New confidence value
            source_type: New source type
            source_episodes: New source episodes list
            derived_from: New derived_from list
            last_verified: New verification timestamp
            verification_count: New verification count
            confidence_history: New confidence history

        Returns:
            True if updated, False if memory not found
        """
        ...

    @abstractmethod
    def get_memories_by_confidence(
        self,
        threshold: float,
        below: bool = True,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories filtered by confidence threshold.

        Args:
            threshold: Confidence threshold
            below: If True, get memories below threshold; if False, above
            memory_types: Filter by type (episode, belief, etc.)
            limit: Maximum results

        Returns:
            List of matching memories with their types
        """
        ...

    @abstractmethod
    def get_memories_by_source(
        self,
        source_type: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories filtered by source type.

        Args:
            source_type: Source type to filter by
            memory_types: Filter by memory type
            limit: Maximum results

        Returns:
            List of matching memories
        """
        ...

    def get_all_active_memories(self, memory_types: Optional[List[str]] = None) -> List[tuple]:
        """Get all active (non-deleted, non-forgotten) memories for strength decay.

        Args:
            memory_types: Types to include (default: all except raw)

        Returns:
            List of (memory_type, record) tuples
        """
        return []

    def update_strength_batch(self, updates: List[tuple]) -> int:
        """Update strength for multiple memories in a single operation.

        Args:
            updates: List of (memory_type, memory_id, strength) tuples

        Returns:
            Number of memories successfully updated
        """
        count = 0
        for memory_type, memory_id, strength in updates:
            if self.update_strength(memory_type, memory_id, strength):
                count += 1
        return count

    def update_memory_metadata(
        self,
        memory_type: str,
        memory_id: str,
        **kwargs: Any,
    ) -> bool:
        """Update arbitrary metadata fields on a memory record.

        This is a general-purpose metadata update that supports any field
        the storage backend recognizes. Used by Stack for lazy decay persistence
        and other metadata updates that don't fit specific protocol methods.

        Args:
            memory_type: Type of memory (episode, belief, note, etc.)
            memory_id: ID of the memory
            **kwargs: Field name/value pairs to update

        Returns:
            True if updated, False if memory not found
        """
        return False


@runtime_checkable
class ForgettingStorage(Protocol):
    """Storage interface for forgetting, memory access tracking, and audit logging."""

    @abstractmethod
    def record_access(self, memory_type: str, memory_id: str) -> bool:
        """Record that a memory was accessed (for salience tracking).

        Increments times_accessed and updates last_accessed timestamp.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if updated, False if memory not found
        """
        ...

    def record_access_batch(self, accesses: List[tuple[str, str]]) -> int:
        """Record multiple memory accesses in a single operation.

        This is an optimization for bulk access tracking, such as when
        loading working memory or returning search results.

        Args:
            accesses: List of (memory_type, memory_id) tuples

        Returns:
            Number of memories successfully updated
        """
        # Default implementation: call record_access for each item
        # Storage backends can override for better performance
        count = 0
        for memory_type, memory_id in accesses:
            if self.record_access(memory_type, memory_id):
                count += 1
        return count

    @abstractmethod
    def forget_memory(
        self,
        memory_type: str,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Tombstone a memory (mark as forgotten, don't delete).

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            reason: Optional reason for forgetting

        Returns:
            True if forgotten, False if not found or already forgotten
        """
        ...

    @abstractmethod
    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if recovered, False if not found or not forgotten
        """
        ...

    @abstractmethod
    def protect_memory(self, memory_type: str, memory_id: str, protected: bool = True) -> bool:
        """Mark a memory as protected from forgetting.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            protected: True to protect, False to unprotect

        Returns:
            True if updated, False if memory not found
        """
        ...

    def weaken_memory(self, memory_type: str, memory_id: str, amount: float) -> bool:
        """Reduce a memory's strength by a given amount.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            amount: Amount to reduce strength by (positive value)

        Returns:
            True if updated, False if not found or protected
        """
        ...

    def verify_memory(self, memory_type: str, memory_id: str) -> bool:
        """Verify a memory: boost strength and increment verification count.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if updated, False if not found
        """
        ...

    def log_audit(
        self,
        memory_type: str,
        memory_id: str,
        operation: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Log an audit entry for a memory operation.

        Args:
            memory_type: Type of memory affected
            memory_id: ID of the memory affected
            operation: Operation name (forget, recover, protect, weaken, verify,
                or dotted format like belief.revised, suggestion.created)
            actor: Who performed the operation
            details: Optional JSON-serializable details
            correlation_id: Optional ID linking related audit entries

        Returns:
            The audit entry ID
        """
        ...

    def get_audit_log(
        self,
        *,
        memory_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries.

        Args:
            memory_type: Filter by memory type
            memory_id: Filter by memory ID
            operation: Filter by operation type
            correlation_id: Filter by correlation ID
            limit: Max entries to return

        Returns:
            List of audit entry dicts
        """
        ...

    @abstractmethod
    def get_forgetting_candidates(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
        threshold: float = 0.5,
    ) -> List[SearchResult]:
        """Get memories that are candidates for forgetting.

        Returns memories that are:
        - Not protected
        - Not already forgotten (strength > 0.0)
        - Below the strength threshold
        - Sorted by strength (lowest first)

        Args:
            memory_types: Filter by memory type
            limit: Maximum results
            threshold: Strength threshold (memories below this are candidates)

        Returns:
            List of candidate memories with computed salience scores
        """
        ...

    @abstractmethod
    def get_forgotten_memories(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get all forgotten (tombstoned) memories.

        Args:
            memory_types: Filter by memory type
            limit: Maximum results

        Returns:
            List of forgotten memories
        """
        ...


@runtime_checkable
class SyncStorage(Protocol):
    """Storage interface for sync operations."""

    @abstractmethod
    def sync(self) -> SyncResult:
        """Sync local changes with cloud.

        For cloud-only storage, this is a no-op.
        For local storage, this pushes/pulls changes.
        """
        ...

    @abstractmethod
    def pull_changes(self, since: Optional[datetime] = None) -> SyncResult:
        """Pull changes from cloud since the given timestamp.

        Args:
            since: Pull changes since this time. If None, uses last sync time.

        Returns:
            SyncResult with pulled count and any conflicts.
        """
        ...

    @abstractmethod
    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        ...

    @abstractmethod
    def is_online(self) -> bool:
        """Check if cloud storage is reachable.

        Returns True if connected, False if offline.
        """
        ...

    # === Sync Queue Methods (for CLI sync commands) ===

    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return utc_now()

    def _clear_queued_change(self, conn: Any, change_id: str) -> None:
        """Clear a queued sync change after successful sync."""
        pass

    def _mark_synced(self, conn: Any, table: str, record_id: str) -> None:
        """Mark a record as synced."""
        pass

    def _set_sync_meta(self, key: str, value: str) -> None:
        """Set sync metadata."""
        pass

    def get_queued_changes(self, limit: int = 100) -> List[Any]:
        """Get pending sync queue changes."""
        return []

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get timestamp of last successful sync."""
        return None

    def get_sync_conflicts(self, limit: int = 100) -> List[SyncConflict]:
        """Get recent sync conflict history.

        Args:
            limit: Maximum number of conflicts to return

        Returns:
            List of SyncConflict records, most recent first
        """
        return []

    def save_sync_conflict(self, conflict: SyncConflict) -> str:
        """Save a sync conflict record.

        Args:
            conflict: The conflict to save

        Returns:
            The conflict ID
        """
        return conflict.id

    def clear_sync_conflicts(self, before: Optional[datetime] = None) -> int:
        """Clear sync conflict history.

        Args:
            before: If provided, only clear conflicts before this timestamp.
                    If None, clear all conflicts.

        Returns:
            Number of conflicts cleared
        """
        return 0

    def _connect(self) -> Any:
        """Get database connection (for sync operations)."""
        raise NotImplementedError("Subclass must implement _connect")

    def _get_record_for_push(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a record formatted for push sync."""
        return None


@runtime_checkable
class StatsStorage(Protocol):
    """Storage interface for statistics."""

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Get counts of each record type."""
        ...


@runtime_checkable
class BatchStorage(Protocol):
    """Storage interface for batch operations."""

    def save_episodes_batch(self, episodes: List[Episode]) -> List[str]:
        """Save multiple episodes in a single transaction.

        This is an optional optimization that storage backends can implement
        to batch multiple database writes into a single transaction, improving
        performance when processing large codebases or bulk imports.

        Default implementation falls back to individual saves.

        Args:
            episodes: List of Episode objects to save

        Returns:
            List of episode IDs (in the same order as input)
        """
        return [self.save_episode(ep) for ep in episodes]

    def save_beliefs_batch(self, beliefs: List[Belief]) -> List[str]:
        """Save multiple beliefs in a single transaction.

        This is an optional optimization that storage backends can implement
        to batch multiple database writes into a single transaction, improving
        performance when processing large codebases or bulk imports.

        Default implementation falls back to individual saves.

        Args:
            beliefs: List of Belief objects to save

        Returns:
            List of belief IDs (in the same order as input)
        """
        return [self.save_belief(b) for b in beliefs]

    def save_notes_batch(self, notes: List[Note]) -> List[str]:
        """Save multiple notes in a single transaction.

        This is an optional optimization that storage backends can implement
        to batch multiple database writes into a single transaction, improving
        performance when processing large codebases or bulk imports.

        Default implementation falls back to individual saves.

        Args:
            notes: List of Note objects to save

        Returns:
            List of note IDs (in the same order as input)
        """
        return [self.save_note(n) for n in notes]

    def load_all(
        self,
        values_limit: Optional[int] = 10,
        beliefs_limit: Optional[int] = 20,
        goals_limit: Optional[int] = 10,
        goals_status: str = "active",
        episodes_limit: Optional[int] = 20,
        notes_limit: Optional[int] = 5,
        drives_limit: Optional[int] = None,
        relationships_limit: Optional[int] = None,
        epoch_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load all memory types in a single operation (optional optimization).

        This is an optional method that storage backends can implement to
        batch multiple queries into a single database connection, avoiding
        N+1 query patterns.

        Default implementation returns None, indicating the caller should
        fall back to individual get_* methods.

        Args:
            values_limit: Max values to load (None = use high limit for budget loading)
            beliefs_limit: Max beliefs to load (None = use high limit for budget loading)
            goals_limit: Max goals to load (None = use high limit for budget loading)
            goals_status: Goal status filter
            episodes_limit: Max episodes to load (None = use high limit for budget loading)
            notes_limit: Max notes to load (None = use high limit for budget loading)
            drives_limit: Max drives to load (None = all drives)
            relationships_limit: Max relationships to load (None = all relationships)
            epoch_id: If set, filter candidates to this epoch only

        Returns:
            Dict with keys: values, beliefs, goals, drives, episodes, notes, relationships
            Or None if not implemented (caller should use individual methods)
        """
        return None  # Default: not implemented, use individual methods


@runtime_checkable
class DiagnosticStorage(Protocol):
    """Storage interface for diagnostic sessions and reports."""

    def save_diagnostic_session(self, session: "DiagnosticSession") -> str:
        """Save a diagnostic session. Returns the session ID."""
        return session.id

    def get_diagnostic_session(self, session_id: str) -> Optional["DiagnosticSession"]:
        """Get a specific diagnostic session by ID."""
        return None

    def get_diagnostic_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List["DiagnosticSession"]:
        """Get diagnostic sessions, optionally filtered by status."""
        return []

    def complete_diagnostic_session(self, session_id: str) -> bool:
        """Mark a diagnostic session as completed. Returns True if updated."""
        return False

    def save_diagnostic_report(self, report: "DiagnosticReport") -> str:
        """Save a diagnostic report. Returns the report ID."""
        return report.id

    def get_diagnostic_report(self, report_id: str) -> Optional["DiagnosticReport"]:
        """Get a specific diagnostic report by ID."""
        return None

    def get_diagnostic_reports(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List["DiagnosticReport"]:
        """Get diagnostic reports, optionally filtered by session."""
        return []

    def get_episodes_by_source_entity(
        self, source_entity: str, limit: int = 500
    ) -> List["Episode"]:
        """Get episodes associated with a source entity for trust computation."""
        return []


@runtime_checkable
class SummaryStorage(Protocol):
    """Storage interface for fractal summarization."""

    def save_summary(self, summary: "Summary") -> str:
        """Save a summary. Returns the summary ID."""
        return summary.id

    def get_summary(self, summary_id: str) -> Optional["Summary"]:
        """Get a specific summary by ID."""
        return None

    def list_summaries(self, stack_id: str, scope: Optional[str] = None) -> List["Summary"]:
        """Get summaries, optionally filtered by scope."""
        return []


@runtime_checkable
class NarrativeStorage(Protocol):
    """Storage interface for self-narratives."""

    def save_self_narrative(self, narrative: "SelfNarrative") -> str:
        """Save a self-narrative. Returns the narrative ID."""
        return narrative.id  # Default no-op

    def get_self_narrative(self, narrative_id: str) -> Optional["SelfNarrative"]:
        """Get a specific self-narrative by ID."""
        return None

    def list_self_narratives(
        self,
        stack_id: str,
        narrative_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List["SelfNarrative"]:
        """Get self-narratives, optionally filtered.

        Args:
            stack_id: Stack ID to filter by
            narrative_type: Filter by type (identity, developmental, aspirational)
            active_only: If True, only return active narratives

        Returns:
            List of matching self-narratives
        """
        return []

    def deactivate_self_narratives(self, stack_id: str, narrative_type: str) -> int:
        """Deactivate all active narratives of a given type.

        Used before saving a new narrative to ensure only one is active per type.

        Args:
            stack_id: Agent ID
            narrative_type: Narrative type to deactivate

        Returns:
            Number of narratives deactivated
        """
        return 0


@runtime_checkable
class EpochStorage(Protocol):
    """Storage interface for epochs."""

    def save_epoch(self, epoch: "Epoch") -> str:
        """Save an epoch. Returns the epoch ID."""
        return epoch.id  # Default no-op

    def get_epoch(self, epoch_id: str) -> Optional["Epoch"]:
        """Get a specific epoch by ID."""
        return None

    def get_epochs(self, limit: int = 100) -> List["Epoch"]:
        """Get all epochs, ordered by epoch_number DESC."""
        return []

    def get_current_epoch(self) -> Optional["Epoch"]:
        """Get the currently active (open) epoch, if any."""
        return None

    def close_epoch(self, epoch_id: str, summary: Optional[str] = None) -> bool:
        """Close an epoch by setting ended_at. Returns True if closed."""
        return False


# === New Sub-protocols (v0.15.0 — Storage Decoupling) ===


@runtime_checkable
class SettingsStorage(Protocol):
    """Storage interface for stack configuration settings (key-value store).

    Override guidance for custom backends:
        All methods have safe no-op defaults. A backend that doesn't override
        these will still construct and run, but Stack settings (lifecycle state,
        enforce_provenance flag) won't persist across restarts — the stack will
        always start in INITIALIZING state.

        **Recommended override** for any backend that persists data:
        implement all three methods.
    """

    def get_stack_setting(self, key: str) -> Optional[str]:
        """Get a stack setting value by key.

        Args:
            key: Setting key name

        Returns:
            The setting value, or None if not found
        """
        return None

    def set_stack_setting(self, key: str, value: str) -> None:
        """Set a stack setting (upsert).

        Args:
            key: Setting key name
            value: Setting value
        """
        pass

    def get_all_stack_settings(self) -> Dict[str, str]:
        """Get all stack settings as a dict.

        Returns:
            Dict of key → value for all settings
        """
        return {}


@runtime_checkable
class AtomicUpdateStorage(Protocol):
    """Storage interface for optimistic-locking updates.

    These methods update an entire record atomically, optionally checking
    a version field to prevent lost writes.

    Override guidance for custom backends:
        Defaults return ``False`` (update silently fails). This is safe for
        construction but means belief revision, goal updates, and relationship
        changes will be no-ops. **Required override** for any backend used
        with Stack's strict mode or the processing pipeline.
    """

    def update_belief_atomic(
        self, belief: "Belief", expected_version: Optional[int] = None
    ) -> bool:
        """Update a belief with optimistic concurrency control.

        Args:
            belief: The belief to update (must have valid .id)
            expected_version: If set, only update if current version matches

        Returns:
            True if updated, False if not found or version mismatch
        """
        return False

    def update_goal_atomic(self, goal: "Goal", expected_version: Optional[int] = None) -> bool:
        """Update a goal with optimistic concurrency control.

        Args:
            goal: The goal to update (must have valid .id)
            expected_version: If set, only update if current version matches

        Returns:
            True if updated, False if not found or version mismatch
        """
        return False

    def update_drive_atomic(self, drive: "Drive", expected_version: Optional[int] = None) -> bool:
        """Update a drive with optimistic concurrency control.

        Args:
            drive: The drive to update (must have valid .id)
            expected_version: If set, only update if current version matches

        Returns:
            True if updated, False if not found or version mismatch
        """
        return False

    def update_relationship_atomic(
        self, relationship: "Relationship", expected_version: Optional[int] = None
    ) -> bool:
        """Update a relationship with optimistic concurrency control.

        Args:
            relationship: The relationship to update (must have valid .id)
            expected_version: If set, only update if current version matches

        Returns:
            True if updated, False if not found or version mismatch
        """
        return False

    def update_episode_atomic(
        self, episode: "Episode", expected_version: Optional[int] = None
    ) -> bool:
        """Update an episode with optimistic concurrency control.

        Args:
            episode: The episode to update (must have valid .id)
            expected_version: If set, only update if current version matches

        Returns:
            True if updated, False if not found or version mismatch
        """
        return False


@runtime_checkable
class ProcessingStorage(Protocol):
    """Storage interface for memory processing state and configuration.

    Override guidance for custom backends:
        Defaults return ``False``/empty. A backend with defaults will allow
        Stack construction but the processing pipeline (``kernle process run``)
        won't mark memories as processed or persist configuration.
        **Required override** if using memory processing (exhaust/promote).
    """

    def mark_episode_processed(self, episode_id: str) -> bool:
        """Mark an episode as processed.

        Args:
            episode_id: ID of the episode to mark

        Returns:
            True if marked, False if not found
        """
        return False

    def mark_note_processed(self, note_id: str) -> bool:
        """Mark a note as processed.

        Args:
            note_id: ID of the note to mark

        Returns:
            True if marked, False if not found
        """
        return False

    def mark_belief_processed(self, belief_id: str) -> bool:
        """Mark a belief as processed.

        Args:
            belief_id: ID of the belief to mark

        Returns:
            True if marked, False if not found
        """
        return False

    def get_processing_config(self) -> List[Dict[str, Any]]:
        """Get all processing configuration entries.

        Returns:
            List of processing config dicts with keys like
            layer_transition, enabled, model_id, etc.
        """
        return []

    def set_processing_config(
        self,
        layer_transition: str,
        **kwargs: Any,
    ) -> bool:
        """Update processing configuration for a layer transition.

        Args:
            layer_transition: The transition name (e.g. "raw_to_episode")
            **kwargs: Configuration fields to update

        Returns:
            True if updated, False otherwise
        """
        return False


@runtime_checkable
class LineageStorage(Protocol):
    """Storage interface for provenance/lineage queries and strength cascading.

    Override guidance for custom backends:
        Defaults return empty lists/tuples and ``False``. Stack cascade logic
        (forget → flag children, weaken → flag dormant, verify → boost sources)
        depends on ``get_memories_derived_from`` and ``boost_memory_strength``.
        **Required override** if using forgetting, strength cascade, or belief
        revision features.
    """

    def get_memories_derived_from(self, memory_type: str, memory_id: str) -> List[tuple]:
        """Find all memories that cite 'type:id' in their derived_from.

        Args:
            memory_type: Type of the source memory
            memory_id: ID of the source memory

        Returns:
            List of (child_type, child_id) tuples
        """
        return []

    def get_ungrounded_memories(self, stack_id: str) -> List[tuple]:
        """Find memories where ALL source refs have strength 0.0 or don't exist.

        Args:
            stack_id: Stack ID to scope the query

        Returns:
            List of (memory_type, memory_id) tuples
        """
        return []

    def log_belief_revision(
        self,
        old_id: str,
        new_id: str,
        reason: Optional[str] = None,
        actor: str = "system",
        correlation_id: Optional[str] = None,
    ) -> tuple:
        """Log audit entries for a belief revision (old → new).

        Args:
            old_id: ID of the superseded belief
            new_id: ID of the new belief
            reason: Reason for revision
            actor: Who performed the revision
            correlation_id: Optional ID linking related entries

        Returns:
            Tuple of (old_audit_id, new_audit_id)
        """
        return ("", "")

    def boost_memory_strength(self, memory_type: str, memory_id: str, amount: float) -> bool:
        """Boost a memory's strength by a given amount (capped at 1.0).

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            amount: Amount to boost (positive value)

        Returns:
            True if updated, False if not found
        """
        return False


@runtime_checkable
class EmbeddingStorage(Protocol):
    """Storage interface for embedding statistics and observability.

    Override guidance for custom backends:
        Default returns a zeroed stats dict. **Optional override** — only
        needed if the backend manages its own embedding storage and wants
        to report provider statistics via ``kernle status``.
    """

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding provider statistics for observability.

        Returns:
            Dict with keys:
            - total: total embeddings stored
            - by_provider: counts per embedding_provider
            - fallback_count: number of fallback embeddings
            - current_provider: name of active provider
            - is_degraded: whether using fallback
        """
        return {
            "total": 0,
            "by_provider": {},
            "fallback_count": 0,
            "current_provider": "none",
            "is_degraded": False,
        }


# === Composite Storage Protocol ===


@runtime_checkable
class Storage(
    EpisodeStorage,
    BeliefStorage,
    ValueStorage,
    GoalStorage,
    NoteStorage,
    DriveStorage,
    RelationshipStorage,
    TrustStorage,
    PlaybookStorage,
    RawStorage,
    SuggestionStorage,
    SearchStorage,
    MetaMemoryStorage,
    ForgettingStorage,
    SyncStorage,
    StatsStorage,
    BatchStorage,
    DiagnosticStorage,
    SummaryStorage,
    NarrativeStorage,
    EpochStorage,
    SettingsStorage,
    AtomicUpdateStorage,
    ProcessingStorage,
    LineageStorage,
    EmbeddingStorage,
    Protocol,
):
    """Full storage interface — backwards compatible composite.

    All storage backends must implement this interface.
    Sub-protocols can be used independently for type narrowing.
    """

    stack_id: str
