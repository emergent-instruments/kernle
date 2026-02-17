"""Memory write operations for Kernle."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from kernle.core.enrichment import (
    DRIVE_TYPES,
    VALID_BELIEF_TYPES,
    VALID_NOTE_TYPES,
    build_derived_from,
    clamp_confidence,
    clamp_intensity,
    format_note_content,
    infer_outcome_type,
    normalize_belief_type,
    normalize_note_type,
    normalize_source_type,
    validate_drive_type,
    validate_goal_type,
)
from kernle.logging_config import log_save
from kernle.storage import Belief, Drive, Episode, Goal, Note, Relationship, Value
from kernle.types import SourceType

logger = logging.getLogger(__name__)


class WritersMixin:
    """Memory write operations for Kernle."""

    # Backward-compatible class attributes — canonical source is enrichment module.
    _VALID_BELIEF_TYPES: frozenset[str] = VALID_BELIEF_TYPES
    _VALID_NOTE_TYPES: frozenset[str] = VALID_NOTE_TYPES
    DRIVE_TYPES = DRIVE_TYPES

    # Backward-compatible aliases — delegate to enrichment module.
    _normalize_source_type = staticmethod(normalize_source_type)
    _normalize_belief_type = staticmethod(normalize_belief_type)
    _normalize_note_type = staticmethod(normalize_note_type)

    # =========================================================================
    # EPISODES
    # =========================================================================

    def episode(
        self,
        objective: str,
        outcome: str,
        lessons: Optional[List[str]] = None,
        repeat: Optional[List[str]] = None,
        avoid: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        derived_from: Optional[List[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str:
        """Record an episodic experience.

        Args:
            derived_from: List of memory IDs this episode was derived from (for linking)
            source: Source context (e.g., 'session with Sean', 'heartbeat check')
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
            source_type: How this memory was acquired (default: direct_experience)
        """
        # Validate inputs
        objective = self._validate_string_input(objective, "objective", 1000)
        outcome = self._validate_string_input(outcome, "outcome", 1000)

        if lessons:
            lessons = [self._validate_string_input(lesson, "lesson", 500) for lesson in lessons]
        if repeat:
            repeat = [self._validate_string_input(r, "repeat pattern", 500) for r in repeat]
        if avoid:
            avoid = [self._validate_string_input(a, "avoid pattern", 500) for a in avoid]
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]

        episode_id = str(uuid.uuid4())

        outcome_type = infer_outcome_type(outcome)

        resolved = self._normalize_source_type(source_type)

        derived_from_value = build_derived_from(derived_from, source)

        episode = Episode(
            id=episode_id,
            stack_id=self.stack_id,
            objective=objective,
            outcome=outcome,
            outcome_type=outcome_type,
            lessons=lessons if lessons else None,
            repeat=repeat if repeat else None,
            avoid=avoid if avoid else None,
            tags=tags or ["manual"],
            created_at=datetime.now(timezone.utc),
            confidence=0.8,
            source_type=resolved.value,
            source_episodes=None,  # Reserved for supporting evidence (episode IDs)
            derived_from=derived_from_value,
            # Context/scope fields
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_episode(episode)

        # Log the episode save
        log_save(
            self.stack_id,
            memory_type="episode",
            memory_id=episode_id,
            summary=objective[:50],
        )

        return episode_id

    def update_episode(
        self,
        episode_id: str,
        outcome: Optional[str] = None,
        lessons: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Update an existing episode."""
        # Validate inputs
        episode_id = self._validate_string_input(episode_id, "episode_id", 100)

        # Get the existing episode via _write_backend (respects strict mode)
        existing = self._write_backend.get_memory("episode", episode_id)

        if not existing:
            return False

        if outcome is not None:
            outcome = self._validate_string_input(outcome, "outcome", 1000)
            existing.outcome = outcome
            existing.outcome_type = infer_outcome_type(outcome)

        if lessons:
            lessons = [self._validate_string_input(lesson, "lesson", 500) for lesson in lessons]
            # Merge with existing lessons
            existing_lessons = existing.lessons or []
            existing.lessons = list(set(existing_lessons + lessons))

        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]
            # Merge with existing tags
            existing_tags = existing.tags or []
            existing.tags = list(set(existing_tags + tags))

        # Use atomic update with optimistic concurrency control
        # This prevents race conditions where concurrent updates could overwrite each other
        self._storage.update_episode_atomic(existing)
        return True

    # =========================================================================
    # NOTES
    # =========================================================================

    def note(
        self,
        content: str,
        type: str = "note",
        speaker: Optional[str] = None,
        reason: Optional[str] = None,
        tags: Optional[List[str]] = None,
        protect: bool = False,
        derived_from: Optional[List[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str:
        """Capture a quick note (decision, insight, quote).

        Args:
            derived_from: List of memory IDs this note was derived from (for linking)
            source: Source context (e.g., 'conversation with X', 'reading Y')
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
            source_type: How this memory was acquired (default: direct_experience)
        """
        # Validate inputs
        content = self._validate_string_input(content, "content", 2000)
        type = self._normalize_note_type(type)

        if speaker:
            speaker = self._validate_string_input(speaker, "speaker", 200)
        if reason:
            reason = self._validate_string_input(reason, "reason", 1000)
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]

        note_id = str(uuid.uuid4())

        formatted = format_note_content(content, type, speaker=speaker, reason=reason)

        resolved = self._normalize_source_type(source_type)

        derived_from_value = build_derived_from(derived_from, source)

        note = Note(
            id=note_id,
            stack_id=self.stack_id,
            content=formatted,
            note_type=type,
            speaker=speaker,
            reason=reason,
            tags=tags or [],
            created_at=datetime.now(timezone.utc),
            source_type=resolved.value,
            source_episodes=None,  # Reserved for supporting evidence (episode IDs)
            derived_from=derived_from_value,
            is_protected=protect,
            # Context/scope fields
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_note(note)
        return note_id

    # =========================================================================
    # RAW ENTRIES (Zero-friction capture)
    # =========================================================================

    def raw(
        self,
        blob: str,
        source: str = "unknown",
    ) -> str:
        """Quick capture of unstructured brain dump for later processing.

        The raw layer is designed for zero-friction capture. Dump whatever you
        want into the blob field; the system only tracks housekeeping metadata.

        Args:
            blob: The raw brain dump content (no validation, no length limits).
            source: Auto-populated source identifier (cli|mcp|sdk|import|unknown).

        Returns:
            Raw entry ID
        """
        # Basic validation - no length limit, but sanitize control chars
        blob = self._validate_string_input(blob, "blob", max_length=None)

        if self._strict:
            from kernle.types import RawEntry

            raw_entry = RawEntry(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                blob=blob,
                source=source,
            )
            return self._write_backend.save_raw(raw_entry)
        return self._storage.save_raw(blob=blob, source=source)

    def list_raw(
        self, processed: Optional[bool] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List raw entries, optionally filtered by processed state.

        Args:
            processed: Filter by processed state (None = all, True = processed, False = unprocessed)
            limit: Maximum entries to return
            offset: Number of entries to skip

        Returns:
            List of raw entry dicts with blob as primary content field
        """
        entries = self._storage.list_raw(processed=processed, limit=limit, offset=offset)
        return [
            {
                "id": e.id,
                "blob": e.blob,  # Primary content field
                "captured_at": e.captured_at.isoformat() if e.captured_at else None,
                "source": e.source,
                "processed": e.processed,
                "processed_into": e.processed_into,
                # Legacy fields for backward compatibility
                "content": e.blob,  # Alias for blob
                "timestamp": e.captured_at.isoformat() if e.captured_at else None,  # Alias
                "tags": e.tags,  # Deprecated but included for compatibility
            }
            for e in entries
        ]

    def get_raw(self, raw_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific raw entry by ID.

        Args:
            raw_id: ID of the raw entry

        Returns:
            Raw entry dict with blob as primary content field, or None if not found
        """
        entry = self._storage.get_raw(raw_id)
        if entry:
            return {
                "id": entry.id,
                "blob": entry.blob,  # Primary content field
                "captured_at": entry.captured_at.isoformat() if entry.captured_at else None,
                "source": entry.source,
                "processed": entry.processed,
                "processed_into": entry.processed_into,
                # Legacy fields for backward compatibility
                "content": entry.blob,  # Alias for blob
                "timestamp": entry.captured_at.isoformat() if entry.captured_at else None,
                "tags": entry.tags,  # Deprecated
            }
        return None

    def search_raw(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search raw entries using keyword search (FTS5).

        This is a safety net for when backlogs accumulate. For semantic search
        across all memory types, use the regular search() method instead.

        Args:
            query: FTS5 search query (supports AND, OR, NOT, phrases in quotes)
            limit: Maximum number of results

        Returns:
            List of matching raw entry dicts, ordered by relevance.
        """
        entries = self._storage.search_raw_fts(query, limit=limit)
        return [
            {
                "id": e.id,
                "blob": e.blob,
                "captured_at": e.captured_at.isoformat() if e.captured_at else None,
                "source": e.source,
                "processed": e.processed,
                "processed_into": e.processed_into,
                # Legacy fields
                "content": e.blob,
                "timestamp": e.captured_at.isoformat() if e.captured_at else None,
                "tags": e.tags,
            }
            for e in entries
        ]

    def process(
        self,
        transition=None,
        force=False,
        allow_no_inference_override=False,
        auto_promote=False,
        batch_size=None,
    ):
        """Run memory processing.

        By default, creates suggestions for review rather than directly
        promoting memories. Set auto_promote=True to directly write memories.

        When no model is bound, identity-layer transitions are blocked
        by the no-inference safety policy. Values can never be created
        without inference.

        Args:
            transition: Specific layer transition to process (None = check all)
            force: Process even if triggers aren't met
            allow_no_inference_override: Allow identity-layer writes without
                inference (except values). Only effective with force=True.
            auto_promote: If True, directly write memories. If False (default),
                create suggestions for review.
            batch_size: Override the per-transition batch size (None = use config).

        Returns:
            List of ProcessingResult for each transition that ran
        """
        entity = self.entity
        if entity is None:
            raise RuntimeError("process() requires Entity (use SQLite storage)")
        # Ensure stack is attached — handles lazy property order where
        # checkpoint() creates _stack before _entity exists, so the
        # stack property's attach_stack() call never ran.
        if entity.active_stack is None and self.stack is not None:
            entity.attach_stack(self.stack, alias="default", set_active=True)
        return entity.process(
            transition=transition,
            force=force,
            allow_no_inference_override=allow_no_inference_override,
            auto_promote=auto_promote,
            batch_size=batch_size,
        )

    def process_raw(
        self,
        raw_id: str,
        as_type: str,
        **kwargs,
    ) -> str:
        """Convert a raw entry into a structured memory.

        Args:
            raw_id: ID of the raw entry to process
            as_type: Type to convert to (episode, note, belief)
            **kwargs: Additional arguments for the target type

        Returns:
            ID of the created memory

        Raises:
            ValueError: If raw entry not found or invalid as_type
        """
        entry = self._storage.get_raw(raw_id)
        if not entry:
            raise ValueError(f"Raw entry {raw_id} not found")

        if entry.processed:
            raise ValueError(f"Raw entry {raw_id} already processed")

        # Create the appropriate memory type
        memory_id = None
        memory_ref = None

        raw_ref = f"raw:{raw_id}"

        if as_type == "episode":
            # Extract or use provided objective/outcome
            # Use blob (preferred) or content (deprecated) for backwards compatibility
            content = entry.blob or entry.content or ""
            objective = kwargs.get("objective") or content[:100]
            outcome = kwargs.get("outcome", "completed")
            lessons = kwargs.get("lessons") or ([content] if len(content) > 100 else None)
            tags = kwargs.get("tags") or entry.tags or []
            if "raw" not in tags:
                tags.append("raw")

            memory_id = self.episode(
                objective=objective,
                outcome=outcome,
                lessons=lessons,
                tags=tags,
                source="raw-processing",
                derived_from=[raw_ref],
            )
            memory_ref = f"episode:{memory_id}"

        elif as_type == "note":
            content = entry.blob or entry.content or ""
            note_type = kwargs.get("type", "note")
            tags = kwargs.get("tags") or entry.tags or []
            if "raw" not in tags:
                tags.append("raw")

            memory_id = self.note(
                content=content,
                type=note_type,
                speaker=kwargs.get("speaker"),
                reason=kwargs.get("reason"),
                tags=tags,
                source="raw-processing",
                derived_from=[raw_ref],
            )
            memory_ref = f"note:{memory_id}"

        elif as_type == "belief":
            content = entry.blob or entry.content or ""
            confidence = kwargs.get("confidence", 0.7)
            belief_type = kwargs.get("type", "observation")

            memory_id = self.belief(
                statement=content,
                type=belief_type,
                confidence=confidence,
                source="raw-processing",
                derived_from=[raw_ref],
            )
            memory_ref = f"belief:{memory_id}"

        else:
            raise ValueError(f"Invalid as_type: {as_type}. Must be one of: episode, note, belief")

        # Mark the raw entry as processed
        self._storage.mark_raw_processed(raw_id, [memory_ref])

        return memory_id

    # =========================================================================
    # BATCH INSERTION
    # =========================================================================

    def episodes_batch(self, episodes: List[Dict[str, Any]]) -> List[str]:
        """Save multiple episodes in a single transaction for bulk imports.

        This method optimizes performance when saving many episodes at once,
        such as when importing from external sources or processing large codebases.
        All episodes are saved in a single database transaction.

        Args:
            episodes: List of episode dicts with keys:
                - objective (str, required): What you were trying to accomplish
                - outcome (str, required): What actually happened
                - outcome_type (str, optional): "success", "failure", or "partial"
                - lessons (List[str], optional): Lessons learned
                - tags (List[str], optional): Tags for categorization
                - confidence (float, optional): Confidence level 0.0-1.0

        Returns:
            List of episode IDs (in the same order as input)

        Example:
            ids = k.episodes_batch([
                {"objective": "Fix login bug", "outcome": "Successfully fixed"},
                {"objective": "Add tests", "outcome": "Added 10 unit tests"},
            ])
        """
        episode_objects = []
        for ep_data in episodes:
            objective = self._validate_string_input(ep_data.get("objective", ""), "objective", 1000)
            outcome = self._validate_string_input(ep_data.get("outcome", ""), "outcome", 1000)

            # Infer outcome_type if not explicitly provided
            outcome_type = ep_data.get("outcome_type") or infer_outcome_type(outcome)

            derived_from_value = build_derived_from(
                ep_data.get("derived_from"), ep_data.get("source")
            )

            episode = Episode(
                id=ep_data.get("id", str(uuid.uuid4())),
                stack_id=self.stack_id,
                objective=objective,
                outcome=outcome,
                outcome_type=outcome_type,
                lessons=ep_data.get("lessons"),
                tags=ep_data.get("tags", ["batch"]),
                created_at=datetime.now(timezone.utc),
                confidence=ep_data.get("confidence", 0.8),
                source_type=normalize_source_type(ep_data.get("source_type")),
                derived_from=derived_from_value,
            )
            episode_objects.append(episode)

        # Use batch method if available, otherwise fall back to individual saves
        backend = self._write_backend
        if hasattr(backend, "save_episodes_batch"):
            return backend.save_episodes_batch(episode_objects)
        else:
            return [backend.save_episode(ep) for ep in episode_objects]

    def beliefs_batch(self, beliefs: List[Dict[str, Any]]) -> List[str]:
        """Save multiple beliefs in a single transaction for bulk imports.

        This method optimizes performance when saving many beliefs at once,
        such as when importing knowledge from external sources.
        All beliefs are saved in a single database transaction.

        Args:
            beliefs: List of belief dicts with keys:
                - statement (str, required): The belief statement
                - type (str, optional): "fact", "opinion", "principle", "strategy", or "model"
                - confidence (float, optional): Confidence level 0.0-1.0

        Returns:
            List of belief IDs (in the same order as input)

        Example:
            ids = k.beliefs_batch([
                {"statement": "Python uses indentation for blocks", "confidence": 1.0},
                {"statement": "Type hints improve code quality", "confidence": 0.9},
            ])
        """
        belief_objects = []
        for b_data in beliefs:
            statement = self._validate_string_input(b_data.get("statement", ""), "statement", 1000)
            belief_type = normalize_belief_type(
                b_data.get("type", b_data.get("belief_type", "fact"))
            )

            derived_from_value = build_derived_from(
                b_data.get("derived_from"), b_data.get("source")
            )

            belief = Belief(
                id=b_data.get("id", str(uuid.uuid4())),
                stack_id=self.stack_id,
                statement=statement,
                belief_type=belief_type,
                confidence=clamp_confidence(b_data.get("confidence", 0.8)),
                created_at=datetime.now(timezone.utc),
                source_type=normalize_source_type(b_data.get("source_type")),
                derived_from=derived_from_value,
            )
            belief_objects.append(belief)

        # Use batch method if available, otherwise fall back to individual saves
        backend = self._write_backend
        if hasattr(backend, "save_beliefs_batch"):
            return backend.save_beliefs_batch(belief_objects)
        else:
            return [backend.save_belief(b) for b in belief_objects]

    def notes_batch(self, notes: List[Dict[str, Any]]) -> List[str]:
        """Save multiple notes in a single transaction for bulk imports.

        This method optimizes performance when saving many notes at once,
        such as when importing from external sources or ingesting documents.
        All notes are saved in a single database transaction.

        Args:
            notes: List of note dicts with keys:
                - content (str, required): The note content
                - type (str, optional): "note", "decision", "insight", or "quote"
                - speaker (str, optional): Who said this (for quotes)
                - reason (str, optional): Why this note matters
                - tags (List[str], optional): Tags for categorization

        Returns:
            List of note IDs (in the same order as input)

        Example:
            ids = k.notes_batch([
                {"content": "Users prefer dark mode", "type": "insight"},
                {"content": "Use TypeScript for new services", "type": "decision"},
            ])
        """
        note_objects = []
        for n_data in notes:
            content = self._validate_string_input(n_data.get("content", ""), "content", 2000)
            note_type = normalize_note_type(n_data.get("type", n_data.get("note_type", "note")))

            formatted = format_note_content(
                content,
                note_type,
                speaker=n_data.get("speaker"),
                reason=n_data.get("reason"),
            )

            derived_from_value = build_derived_from(
                n_data.get("derived_from"), n_data.get("source")
            )

            note = Note(
                id=n_data.get("id", str(uuid.uuid4())),
                stack_id=self.stack_id,
                content=formatted,
                note_type=note_type,
                speaker=n_data.get("speaker"),
                reason=n_data.get("reason"),
                tags=n_data.get("tags", []),
                created_at=datetime.now(timezone.utc),
                source_type=normalize_source_type(n_data.get("source_type")),
                derived_from=derived_from_value,
            )
            note_objects.append(note)

        # Use batch method if available, otherwise fall back to individual saves
        backend = self._write_backend
        if hasattr(backend, "save_notes_batch"):
            return backend.save_notes_batch(note_objects)
        else:
            return [backend.save_note(n) for n in note_objects]

    # =========================================================================
    # BELIEFS & VALUES & GOALS
    # =========================================================================

    def belief(
        self,
        statement: str,
        type: str = "fact",
        confidence: float = 0.8,
        foundational: bool = False,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        derived_from: Optional[List[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str:
        """Add or update a belief.

        Args:
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
            source: Source context (e.g., 'raw-processing', 'consolidation', 'told by Claire')
            derived_from: List of memory refs this was derived from (format: type:id)
            source_type: How this memory was acquired (default: direct_experience)
        """
        confidence = clamp_confidence(confidence)
        belief_id = str(uuid.uuid4())

        resolved = self._normalize_source_type(source_type)

        derived_from_value = build_derived_from(derived_from, source)
        derived_from_value = self._validate_derived_from(derived_from_value)

        belief = Belief(
            id=belief_id,
            stack_id=self.stack_id,
            statement=statement,
            belief_type=normalize_belief_type(type),
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            source_type=resolved.value,
            derived_from=derived_from_value,
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_belief(belief)
        return belief_id

    def value(
        self,
        name: str,
        statement: str,
        priority: int = 50,
        type: str = "core_value",
        foundational: bool = False,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        derived_from: Optional[List[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str:
        """Add or affirm a value.

        Args:
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
            source: Source context (e.g., 'consolidation', 'told by X', 'raw-processing')
            derived_from: List of memory refs this was derived from (format: type:id)
            source_type: How this memory was acquired (default: direct_experience)
        """
        value_id = str(uuid.uuid4())

        resolved = self._normalize_source_type(source_type)

        derived_from_value = build_derived_from(derived_from, source)
        derived_from_value = self._validate_derived_from(derived_from_value)

        value = Value(
            id=value_id,
            stack_id=self.stack_id,
            name=name,
            statement=statement,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            source_type=resolved.value,
            derived_from=derived_from_value if derived_from_value else None,
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_value(value)
        return value_id

    def goal(
        self,
        title: str,
        description: Optional[str] = None,
        goal_type: str = "task",
        priority: str = "medium",
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        derived_from: Optional[List[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str:
        """Add a goal.

        Args:
            goal_type: Type of goal (task, aspiration, commitment, exploration)
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
            source: Source context (e.g., 'consolidation', 'told by X', 'raw-processing')
            derived_from: List of memory refs this was derived from (format: type:id)
            source_type: How this memory was acquired (default: direct_experience)
        """
        validate_goal_type(goal_type)

        goal_id = str(uuid.uuid4())

        # Set protection based on goal_type
        is_protected = goal_type in ("aspiration", "commitment")

        resolved = self._normalize_source_type(source_type)

        derived_from_value = build_derived_from(derived_from, source)
        derived_from_value = self._validate_derived_from(derived_from_value)

        goal = Goal(
            id=goal_id,
            stack_id=self.stack_id,
            title=title,
            description=description or title,
            goal_type=goal_type,
            priority=priority,
            status="active",
            created_at=datetime.now(timezone.utc),
            is_protected=is_protected,
            source_type=resolved.value,
            derived_from=derived_from_value if derived_from_value else None,
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_goal(goal)

        # Protect aspiration/commitment goals from forgetting (via _write_backend)
        if is_protected:
            self._write_backend.protect_memory("goal", goal_id, protected=True)

        return goal_id

    def update_goal(
        self,
        goal_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update a goal's status, priority, or description."""
        # Validate inputs
        goal_id = self._validate_string_input(goal_id, "goal_id", 100)

        # Get goals via _write_backend (respects strict mode)
        goals = self._write_backend.get_goals(status=None, limit=1000)
        existing = None
        for g in goals:
            if g.id == goal_id:
                existing = g
                break

        if not existing:
            return False

        if status is not None:
            if status not in ("active", "completed", "paused"):
                raise ValueError("Invalid status. Must be one of: active, completed, paused")
            existing.status = status

        if priority is not None:
            if priority not in ("low", "medium", "high"):
                raise ValueError("Invalid priority. Must be one of: low, medium, high")
            existing.priority = priority

        if description is not None:
            description = self._validate_string_input(description, "description", 1000)
            existing.description = description

        self._write_backend.update_goal_atomic(existing)
        return True

    # =========================================================================
    # DRIVES (Motivation System)
    # =========================================================================

    def drive(
        self,
        drive_type: str,
        intensity: float = 0.5,
        focus_areas: Optional[List[str]] = None,
        decay_hours: int = 24,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        derived_from: Optional[List[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
    ) -> str:
        """Set or update a drive.

        Args:
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
            source: Source context (e.g., 'consolidation', 'told by X', 'raw-processing')
            derived_from: List of memory refs this was derived from (format: type:id)
            source_type: How this memory was acquired (default: direct_experience)
        """
        validate_drive_type(drive_type)

        resolved = self._normalize_source_type(source_type)

        derived_from_value = build_derived_from(derived_from, source)
        derived_from_value = self._validate_derived_from(derived_from_value)

        # Check if drive exists via _write_backend (respects strict mode)
        existing = next(
            (d for d in self._write_backend.get_drives() if d.drive_type == drive_type),
            None,
        )

        now = datetime.now(timezone.utc)

        if existing:
            existing.intensity = clamp_intensity(intensity)
            existing.focus_areas = focus_areas or []
            existing.updated_at = now
            if context is not None:
                existing.context = context
            if context_tags is not None:
                existing.context_tags = context_tags
            existing.source_type = resolved.value
            if derived_from_value:
                existing.derived_from = derived_from_value
            self._write_backend.update_drive_atomic(existing)
            return existing.id
        else:
            drive_id = str(uuid.uuid4())
            drive = Drive(
                id=drive_id,
                stack_id=self.stack_id,
                drive_type=drive_type,
                intensity=clamp_intensity(intensity),
                focus_areas=focus_areas or [],
                created_at=now,
                updated_at=now,
                source_type=resolved.value,
                derived_from=derived_from_value if derived_from_value else None,
                context=context,
                context_tags=context_tags,
            )
            self._write_backend.save_drive(drive)
            return drive_id

    def satisfy_drive(self, drive_type: str, amount: float = 0.2) -> bool:
        """Record satisfaction of a drive (reduces intensity toward baseline)."""
        existing = next(
            (d for d in self._write_backend.get_drives() if d.drive_type == drive_type),
            None,
        )

        if existing:
            new_intensity = max(0.1, existing.intensity - amount)
            existing.intensity = new_intensity
            existing.updated_at = datetime.now(timezone.utc)
            self._write_backend.update_drive_atomic(existing)
            return True
        return False

    # =========================================================================
    # RELATIONAL MEMORY (Models of Other Entities)
    # =========================================================================

    def relationship(
        self,
        other_stack_id: str,
        trust_level: Optional[float] = None,
        notes: Optional[str] = None,
        interaction_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        derived_from: Optional[List[str]] = None,
        source_type: Optional[Union[str, SourceType]] = None,
        source_entity: Optional[str] = None,
    ) -> str:
        """Update relationship model for another entity.

        Args:
            other_stack_id: Name/identifier of the other entity
            trust_level: Trust level 0.0-1.0 (converted to sentiment -1 to 1)
            notes: Notes about the relationship
            interaction_type: Type of interaction being logged
            entity_type: Type of entity (person, agent, organization, system)
            derived_from: Memory IDs this relationship was derived from
            source_type: How this memory was acquired (default: direct_experience)
            source_entity: Who/what created this memory
        """
        resolved = self._normalize_source_type(source_type)

        # Check existing — use write backend for consistent strict-mode path
        backend = self._write_backend
        existing = backend.get_relationship(other_stack_id)

        now = datetime.now(timezone.utc)

        if existing:
            if trust_level is not None:
                # Convert trust_level (0-1) to sentiment (-1 to 1)
                existing.sentiment = max(-1.0, min(1.0, (trust_level * 2) - 1))
            if notes:
                existing.notes = notes
            if entity_type:
                existing.entity_type = entity_type
            if derived_from:
                existing.derived_from = derived_from
            if source_type is not None:
                existing.source_type = resolved.value
            if source_entity is not None:
                existing.source_entity = source_entity
            existing.interaction_count += 1
            existing.last_interaction = now
            self._write_backend.update_relationship_atomic(existing)
            return existing.id
        else:
            rel_id = str(uuid.uuid4())
            relationship = Relationship(
                id=rel_id,
                stack_id=self.stack_id,
                entity_name=other_stack_id,
                entity_type=entity_type or "person",
                relationship_type=interaction_type or "interaction",
                notes=notes,
                sentiment=((trust_level * 2) - 1) if trust_level is not None else 0.0,
                interaction_count=1,
                last_interaction=now,
                created_at=now,
                source_type=resolved.value,
                source_entity=source_entity,
                derived_from=derived_from,
            )
            self._write_backend.save_relationship(relationship)
            return rel_id
