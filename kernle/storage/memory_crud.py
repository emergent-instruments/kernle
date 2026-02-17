"""Memory CRUD row deserializers for Kernle storage.

Extracted from sqlite.py. Contains module-level functions for converting
database rows to dataclass instances for all memory types.

SQLiteStorage keeps thin wrapper methods that delegate here.
"""

import json
import logging
import sqlite3
from typing import Any, Optional

from .base import (
    Belief,
    DiagnosticReport,
    DiagnosticSession,
    Drive,
    EntityModel,
    Episode,
    Epoch,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    Relationship,
    RelationshipHistoryEntry,
    SelfNarrative,
    Summary,
    Value,
    parse_datetime,
)

logger = logging.getLogger(__name__)


def _from_json(s: Optional[str]) -> Any:
    """Parse JSON string."""
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _parse_dt(s: Optional[str]) -> Any:
    """Parse ISO datetime string."""
    return parse_datetime(s)


def _safe_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """Safely get value from row."""
    try:
        value = row[key]
        return value if value is not None else default
    except (IndexError, KeyError):
        return default


def _row_to_episode(row: sqlite3.Row) -> Episode:
    """Convert a row to an Episode."""
    return Episode(
        id=row["id"],
        stack_id=row["stack_id"],
        objective=row["objective"],
        outcome=row["outcome"],
        outcome_type=row["outcome_type"],
        lessons=_from_json(row["lessons"]),
        tags=_from_json(row["tags"]),
        created_at=_parse_dt(row["created_at"]),
        emotional_valence=(
            row["emotional_valence"] if row["emotional_valence"] is not None else 0.0
        ),
        emotional_arousal=(
            row["emotional_arousal"] if row["emotional_arousal"] is not None else 0.0
        ),
        emotional_tags=_from_json(row["emotional_tags"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        # Meta-memory fields
        confidence=_safe_get(row, "confidence", 0.8),
        source_type=_safe_get(row, "source_type", "direct_experience"),
        source_episodes=_from_json(_safe_get(row, "source_episodes", None)),
        derived_from=_from_json(_safe_get(row, "derived_from", None)),
        last_verified=_parse_dt(_safe_get(row, "last_verified", None)),
        verification_count=_safe_get(row, "verification_count", 0),
        confidence_history=_from_json(_safe_get(row, "confidence_history", None)),
        # Forgetting fields
        times_accessed=_safe_get(row, "times_accessed", 0),
        last_accessed=_parse_dt(_safe_get(row, "last_accessed", None)),
        is_protected=bool(_safe_get(row, "is_protected", 0)),
        strength=float(_safe_get(row, "strength", 1.0)),
        processed=bool(_safe_get(row, "processed", 0)),
        # Context/scope fields
        context=_safe_get(row, "context", None),
        context_tags=_from_json(_safe_get(row, "context_tags", None)),
        # Entity-neutral sourcing
        source_entity=_safe_get(row, "source_entity", None),
        # Privacy fields (Phase 8a)
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        # Epoch tracking
        epoch_id=_safe_get(row, "epoch_id", None),
        # Repeat/avoid patterns
        repeat=_from_json(_safe_get(row, "repeat", None)),
        avoid=_from_json(_safe_get(row, "avoid", None)),
    )


def _row_to_belief(row: sqlite3.Row) -> Belief:
    """Convert a row to a Belief."""
    is_active_val = _safe_get(row, "is_active", 1)
    return Belief(
        id=row["id"],
        stack_id=row["stack_id"],
        statement=row["statement"],
        belief_type=row["belief_type"],
        confidence=row["confidence"],
        created_at=_parse_dt(row["created_at"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        # Meta-memory fields
        source_type=_safe_get(row, "source_type", "direct_experience"),
        source_episodes=_from_json(_safe_get(row, "source_episodes", None)),
        derived_from=_from_json(_safe_get(row, "derived_from", None)),
        last_verified=_parse_dt(_safe_get(row, "last_verified", None)),
        verification_count=_safe_get(row, "verification_count", 0),
        confidence_history=_from_json(_safe_get(row, "confidence_history", None)),
        # Belief revision fields
        supersedes=_safe_get(row, "supersedes", None),
        superseded_by=_safe_get(row, "superseded_by", None),
        times_reinforced=_safe_get(row, "times_reinforced", 0),
        is_active=bool(is_active_val) if is_active_val is not None else True,
        # Forgetting fields
        times_accessed=_safe_get(row, "times_accessed", 0),
        last_accessed=_parse_dt(_safe_get(row, "last_accessed", None)),
        is_protected=bool(_safe_get(row, "is_protected", 0)),
        strength=float(_safe_get(row, "strength", 1.0)),
        # Context/scope fields
        context=_safe_get(row, "context", None),
        context_tags=_from_json(_safe_get(row, "context_tags", None)),
        # Entity-neutral sourcing
        source_entity=_safe_get(row, "source_entity", None),
        # Privacy fields (Phase 8a)
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        # Processing state
        processed=bool(_safe_get(row, "processed", 0)),
        # Belief scope and domain metadata (KEP v3)
        belief_scope=_safe_get(row, "belief_scope", "world"),
        source_domain=_safe_get(row, "source_domain", None),
        cross_domain_applications=_from_json(_safe_get(row, "cross_domain_applications", None)),
        abstraction_level=_safe_get(row, "abstraction_level", "specific"),
        # Epoch tracking
        epoch_id=_safe_get(row, "epoch_id", None),
    )


def _row_to_value(row: sqlite3.Row) -> Value:
    """Convert a row to a Value."""
    return Value(
        id=row["id"],
        stack_id=row["stack_id"],
        name=row["name"],
        statement=row["statement"],
        priority=row["priority"],
        created_at=_parse_dt(row["created_at"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        # Meta-memory fields
        confidence=_safe_get(row, "confidence", 0.9),
        source_type=_safe_get(row, "source_type", "direct_experience"),
        source_episodes=_from_json(_safe_get(row, "source_episodes", None)),
        derived_from=_from_json(_safe_get(row, "derived_from", None)),
        last_verified=_parse_dt(_safe_get(row, "last_verified", None)),
        verification_count=_safe_get(row, "verification_count", 0),
        confidence_history=_from_json(_safe_get(row, "confidence_history", None)),
        # Forgetting fields (values protected by default)
        times_accessed=_safe_get(row, "times_accessed", 0),
        last_accessed=_parse_dt(_safe_get(row, "last_accessed", None)),
        is_protected=bool(_safe_get(row, "is_protected", 1)),  # Default protected
        strength=float(_safe_get(row, "strength", 1.0)),
        # Context/scope fields
        context=_safe_get(row, "context", None),
        context_tags=_from_json(_safe_get(row, "context_tags", None)),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        # Epoch tracking
        epoch_id=_safe_get(row, "epoch_id", None),
    )


def _row_to_goal(row: sqlite3.Row) -> Goal:
    """Convert a row to a Goal."""
    return Goal(
        id=row["id"],
        stack_id=row["stack_id"],
        title=row["title"],
        description=row["description"],
        goal_type=_safe_get(row, "goal_type", "task"),
        priority=row["priority"],
        status=row["status"],
        created_at=_parse_dt(row["created_at"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        # Meta-memory fields
        confidence=_safe_get(row, "confidence", 0.8),
        source_type=_safe_get(row, "source_type", "direct_experience"),
        source_episodes=_from_json(_safe_get(row, "source_episodes", None)),
        derived_from=_from_json(_safe_get(row, "derived_from", None)),
        last_verified=_parse_dt(_safe_get(row, "last_verified", None)),
        verification_count=_safe_get(row, "verification_count", 0),
        confidence_history=_from_json(_safe_get(row, "confidence_history", None)),
        # Forgetting fields
        times_accessed=_safe_get(row, "times_accessed", 0),
        last_accessed=_parse_dt(_safe_get(row, "last_accessed", None)),
        is_protected=bool(_safe_get(row, "is_protected", 0)),
        strength=float(_safe_get(row, "strength", 1.0)),
        # Context/scope fields
        context=_safe_get(row, "context", None),
        context_tags=_from_json(_safe_get(row, "context_tags", None)),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        # Epoch tracking
        epoch_id=_safe_get(row, "epoch_id", None),
    )


def _row_to_note(row: sqlite3.Row) -> Note:
    """Convert a row to a Note."""
    return Note(
        id=row["id"],
        stack_id=row["stack_id"],
        content=row["content"],
        note_type=row["note_type"],
        speaker=row["speaker"],
        reason=row["reason"],
        tags=_from_json(row["tags"]),
        created_at=_parse_dt(row["created_at"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        # Meta-memory fields
        confidence=_safe_get(row, "confidence", 0.8),
        source_type=_safe_get(row, "source_type", "direct_experience"),
        source_episodes=_from_json(_safe_get(row, "source_episodes", None)),
        derived_from=_from_json(_safe_get(row, "derived_from", None)),
        last_verified=_parse_dt(_safe_get(row, "last_verified", None)),
        verification_count=_safe_get(row, "verification_count", 0),
        confidence_history=_from_json(_safe_get(row, "confidence_history", None)),
        # Forgetting fields
        times_accessed=_safe_get(row, "times_accessed", 0),
        last_accessed=_parse_dt(_safe_get(row, "last_accessed", None)),
        is_protected=bool(_safe_get(row, "is_protected", 0)),
        strength=float(_safe_get(row, "strength", 1.0)),
        processed=bool(_safe_get(row, "processed", 0)),
        # Context/scope fields
        context=_safe_get(row, "context", None),
        context_tags=_from_json(_safe_get(row, "context_tags", None)),
        # Entity-neutral sourcing
        source_entity=_safe_get(row, "source_entity", None),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        # Epoch tracking
        epoch_id=_safe_get(row, "epoch_id", None),
    )


def _row_to_drive(row: sqlite3.Row) -> Drive:
    """Convert a row to a Drive."""
    return Drive(
        id=row["id"],
        stack_id=row["stack_id"],
        drive_type=row["drive_type"],
        intensity=row["intensity"],
        focus_areas=_from_json(row["focus_areas"]),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        # Meta-memory fields
        confidence=_safe_get(row, "confidence", 0.8),
        source_type=_safe_get(row, "source_type", "direct_experience"),
        source_episodes=_from_json(_safe_get(row, "source_episodes", None)),
        derived_from=_from_json(_safe_get(row, "derived_from", None)),
        last_verified=_parse_dt(_safe_get(row, "last_verified", None)),
        verification_count=_safe_get(row, "verification_count", 0),
        confidence_history=_from_json(_safe_get(row, "confidence_history", None)),
        # Forgetting fields (drives protected by default)
        times_accessed=_safe_get(row, "times_accessed", 0),
        last_accessed=_parse_dt(_safe_get(row, "last_accessed", None)),
        is_protected=bool(_safe_get(row, "is_protected", 1)),  # Default protected
        strength=float(_safe_get(row, "strength", 1.0)),
        # Context/scope fields
        context=_safe_get(row, "context", None),
        context_tags=_from_json(_safe_get(row, "context_tags", None)),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        # Epoch tracking
        epoch_id=_safe_get(row, "epoch_id", None),
    )


def _row_to_relationship(row: sqlite3.Row) -> Relationship:
    """Convert a row to a Relationship."""
    return Relationship(
        id=row["id"],
        stack_id=row["stack_id"],
        entity_name=row["entity_name"],
        entity_type=row["entity_type"],
        relationship_type=row["relationship_type"],
        notes=row["notes"],
        sentiment=row["sentiment"],
        interaction_count=row["interaction_count"],
        last_interaction=_parse_dt(row["last_interaction"]),
        created_at=_parse_dt(row["created_at"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        # Meta-memory fields
        confidence=_safe_get(row, "confidence", 0.8),
        source_type=_safe_get(row, "source_type", "direct_experience"),
        source_episodes=_from_json(_safe_get(row, "source_episodes", None)),
        derived_from=_from_json(_safe_get(row, "derived_from", None)),
        last_verified=_parse_dt(_safe_get(row, "last_verified", None)),
        verification_count=_safe_get(row, "verification_count", 0),
        confidence_history=_from_json(_safe_get(row, "confidence_history", None)),
        # Forgetting fields
        times_accessed=_safe_get(row, "times_accessed", 0),
        last_accessed=_parse_dt(_safe_get(row, "last_accessed", None)),
        is_protected=bool(_safe_get(row, "is_protected", 0)),
        strength=float(_safe_get(row, "strength", 1.0)),
        # Context/scope fields
        context=_safe_get(row, "context", None),
        context_tags=_from_json(_safe_get(row, "context_tags", None)),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        # Entity-neutral sourcing
        source_entity=_safe_get(row, "source_entity", None),
        # Epoch tracking
        epoch_id=_safe_get(row, "epoch_id", None),
    )


def _row_to_epoch(row: sqlite3.Row) -> Epoch:
    """Convert a row to an Epoch."""
    return Epoch(
        id=row["id"],
        stack_id=row["stack_id"],
        epoch_number=row["epoch_number"],
        name=row["name"],
        started_at=_parse_dt(row["started_at"]),
        ended_at=_parse_dt(row["ended_at"]),
        trigger_type=_safe_get(row, "trigger_type", "declared"),
        trigger_description=_safe_get(row, "trigger_description", None),
        summary=_safe_get(row, "summary", None),
        key_belief_ids=_from_json(_safe_get(row, "key_belief_ids", None)),
        key_relationship_ids=_from_json(_safe_get(row, "key_relationship_ids", None)),
        key_goal_ids=_from_json(_safe_get(row, "key_goal_ids", None)),
        dominant_drive_ids=_from_json(_safe_get(row, "dominant_drive_ids", None)),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(_safe_get(row, "cloud_synced_at", None)),
        version=row["version"],
        deleted=bool(row["deleted"]),
    )


def _row_to_summary(row: sqlite3.Row) -> Summary:
    """Convert a row to a Summary."""
    return Summary(
        id=row["id"],
        stack_id=row["stack_id"],
        scope=row["scope"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        epoch_id=_safe_get(row, "epoch_id", None),
        content=row["content"],
        key_themes=_from_json(_safe_get(row, "key_themes", None)),
        supersedes=_from_json(_safe_get(row, "supersedes", None)),
        is_protected=bool(_safe_get(row, "is_protected", 1)),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        cloud_synced_at=_parse_dt(_safe_get(row, "cloud_synced_at", None)),
        version=row["version"],
        deleted=bool(row["deleted"]),
    )


def _row_to_self_narrative(row: sqlite3.Row) -> SelfNarrative:
    """Convert a row to a SelfNarrative."""
    return SelfNarrative(
        id=row["id"],
        stack_id=row["stack_id"],
        content=row["content"],
        narrative_type=_safe_get(row, "narrative_type", "identity"),
        epoch_id=_safe_get(row, "epoch_id", None),
        key_themes=_from_json(_safe_get(row, "key_themes", None)),
        unresolved_tensions=_from_json(_safe_get(row, "unresolved_tensions", None)),
        is_active=bool(_safe_get(row, "is_active", 1)),
        supersedes=_safe_get(row, "supersedes", None),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        cloud_synced_at=_parse_dt(_safe_get(row, "cloud_synced_at", None)),
        version=row["version"],
        deleted=bool(row["deleted"]),
    )


def _row_to_diagnostic_session(row: sqlite3.Row) -> DiagnosticSession:
    """Convert a database row to a DiagnosticSession."""
    return DiagnosticSession(
        id=row["id"],
        stack_id=row["stack_id"],
        session_type=row["session_type"],
        access_level=row["access_level"],
        status=row["status"],
        consent_given=bool(row["consent_given"]),
        started_at=parse_datetime(row["started_at"]),
        completed_at=parse_datetime(row["completed_at"]) if row["completed_at"] else None,
        local_updated_at=(
            parse_datetime(row["local_updated_at"]) if row["local_updated_at"] else None
        ),
        cloud_synced_at=(
            parse_datetime(row["cloud_synced_at"]) if row["cloud_synced_at"] else None
        ),
        version=row["version"],
        deleted=bool(row["deleted"]),
    )


def _row_to_diagnostic_report(row: sqlite3.Row) -> DiagnosticReport:
    """Convert a database row to a DiagnosticReport."""
    return DiagnosticReport(
        id=row["id"],
        stack_id=row["stack_id"],
        session_id=row["session_id"],
        findings=json.loads(row["findings"]) if row["findings"] else None,
        summary=row["summary"],
        created_at=parse_datetime(row["created_at"]) if row["created_at"] else None,
        local_updated_at=(
            parse_datetime(row["local_updated_at"]) if row["local_updated_at"] else None
        ),
        cloud_synced_at=(
            parse_datetime(row["cloud_synced_at"]) if row["cloud_synced_at"] else None
        ),
        version=row["version"],
        deleted=bool(row["deleted"]),
    )


def _row_to_relationship_history(row: sqlite3.Row) -> RelationshipHistoryEntry:
    """Convert a row to a RelationshipHistoryEntry."""
    return RelationshipHistoryEntry(
        id=row["id"],
        stack_id=row["stack_id"],
        relationship_id=row["relationship_id"],
        entity_name=row["entity_name"],
        event_type=row["event_type"],
        old_value=row["old_value"],
        new_value=row["new_value"],
        episode_id=_safe_get(row, "episode_id", None),
        notes=_safe_get(row, "notes", None),
        created_at=_parse_dt(row["created_at"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(_safe_get(row, "cloud_synced_at", None)),
        version=_safe_get(row, "version", 1),
        deleted=bool(_safe_get(row, "deleted", 0)),
    )


def _row_to_entity_model(row: sqlite3.Row) -> EntityModel:
    """Convert a row to an EntityModel."""
    return EntityModel(
        id=row["id"],
        stack_id=row["stack_id"],
        entity_name=row["entity_name"],
        model_type=row["model_type"],
        observation=row["observation"],
        confidence=_safe_get(row, "confidence", 0.7),
        source_episodes=_from_json(_safe_get(row, "source_episodes", None)),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(_safe_get(row, "updated_at", None)),
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(_safe_get(row, "cloud_synced_at", None)),
        version=_safe_get(row, "version", 1),
        deleted=bool(_safe_get(row, "deleted", 0)),
    )


def _row_to_playbook(row: sqlite3.Row) -> Playbook:
    """Convert a row to a Playbook."""
    return Playbook(
        id=row["id"],
        stack_id=row["stack_id"],
        name=row["name"],
        description=row["description"],
        trigger_conditions=_from_json(row["trigger_conditions"]) or [],
        steps=_from_json(row["steps"]) or [],
        failure_modes=_from_json(row["failure_modes"]) or [],
        recovery_steps=_from_json(row["recovery_steps"]),
        mastery_level=row["mastery_level"],
        times_used=row["times_used"],
        success_rate=row["success_rate"],
        source_episodes=_from_json(row["source_episodes"]),
        tags=_from_json(row["tags"]),
        confidence=_safe_get(row, "confidence", 0.8),
        last_used=_parse_dt(_safe_get(row, "last_used", None)),
        created_at=_parse_dt(row["created_at"]),
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
        consent_grants=_from_json(_safe_get(row, "consent_grants", None)),
        access_grants=_from_json(_safe_get(row, "access_grants", None)),
        subject_ids=_from_json(_safe_get(row, "subject_ids", None)),
        # Privacy fields (Phase 8a)
    )


def _row_to_suggestion(row: sqlite3.Row) -> MemorySuggestion:
    """Convert a row to a MemorySuggestion."""
    return MemorySuggestion(
        id=row["id"],
        stack_id=row["stack_id"],
        memory_type=row["memory_type"],
        content=_from_json(row["content"]) or {},
        confidence=row["confidence"],
        source_raw_ids=_from_json(row["source_raw_ids"]) or [],
        status=row["status"],
        created_at=_parse_dt(row["created_at"]),
        resolved_at=_parse_dt(row["resolved_at"]),
        resolution_reason=row["resolution_reason"],
        promoted_to=row["promoted_to"],
        local_updated_at=_parse_dt(row["local_updated_at"]),
        cloud_synced_at=_parse_dt(row["cloud_synced_at"]),
        version=row["version"],
        deleted=bool(row["deleted"]),
    )
