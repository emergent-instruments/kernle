"""Database schema and migration logic for Kernle SQLite storage.

Extracted from sqlite.py. Contains:
- Schema DDL constants (SCHEMA, VECTOR_SCHEMA)
- Schema version tracking (SCHEMA_VERSION)
- Table allowlist (ALLOWED_TABLES, validate_table_name)
- Database initialization (init_db)
- Schema migration (migrate_schema)
- FTS5 setup (ensure_raw_fts5)
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 27  # v0.14.00: add correlation_id to memory_audit, use_legacy_heuristics flag

# Allowed table names for SQL queries (security: prevents SQL injection via table names)
ALLOWED_TABLES = frozenset(
    {
        "episodes",
        "beliefs",
        "agent_values",
        "goals",
        "notes",
        "drives",
        "relationships",
        "playbooks",
        "raw_entries",
        "checkpoints",
        "memory_suggestions",
        "schema_version",
        "sync_queue",
        "sync_conflicts",
        "embeddings",
        "health_check_events",
        "boot_config",
        "trust_assessments",  # KEP v3 trust layer
        "epochs",  # KEP v3 temporal epochs
        "diagnostic_sessions",  # KEP v3 diagnostic sessions
        "diagnostic_reports",  # KEP v3 diagnostic reports
        "summaries",  # KEP v3 fractal summarization
        "self_narratives",  # KEP v3 self-narrative layer
        "memory_audit",  # v0.9.0 memory audit trail
        "processing_config",  # v0.9.0 processing configuration
        "stack_settings",  # per-stack feature flags
    }
)


def validate_table_name(table: str) -> str:
    """Validate table name against allowlist to prevent SQL injection.

    Args:
        table: Table name to validate

    Returns:
        The validated table name

    Raises:
        ValueError: If table name is not in allowlist
    """
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    return table


SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Episodes (experiences/work logs)
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    outcome TEXT NOT NULL,
    outcome_type TEXT,
    lessons TEXT,  -- JSON array
    tags TEXT,     -- JSON array
    created_at TEXT NOT NULL,
    -- Emotional memory fields
    emotional_valence REAL DEFAULT 0.0,  -- -1.0 (negative) to 1.0 (positive)
    emotional_arousal REAL DEFAULT 0.0,  -- 0.0 (calm) to 1.0 (intense)
    emotional_tags TEXT,  -- JSON array ["joy", "frustration", "curiosity"]
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,  -- JSON array of episode IDs
    derived_from TEXT,     -- JSON array of memory IDs (format: type:id)
    last_verified TEXT,    -- ISO timestamp
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,  -- JSON array of confidence changes
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    strength REAL DEFAULT 1.0,
    processed INTEGER DEFAULT 0,   -- Whether episode has been processed for promotion
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    source_entity TEXT,
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Epoch tracking
    epoch_id TEXT,
    -- Repeat/avoid patterns
    repeat TEXT,   -- JSON array of patterns to replicate
    avoid TEXT,    -- JSON array of patterns to avoid
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(stack_id);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_sync ON episodes(cloud_synced_at);
CREATE INDEX IF NOT EXISTS idx_episodes_valence ON episodes(emotional_valence);
CREATE INDEX IF NOT EXISTS idx_episodes_arousal ON episodes(emotional_arousal);
CREATE INDEX IF NOT EXISTS idx_episodes_confidence ON episodes(confidence);
CREATE INDEX IF NOT EXISTS idx_episodes_source_type ON episodes(source_type);
CREATE INDEX IF NOT EXISTS idx_episodes_strength ON episodes(strength);
CREATE INDEX IF NOT EXISTS idx_episodes_is_protected ON episodes(is_protected);
CREATE INDEX IF NOT EXISTS idx_episodes_epoch ON episodes(epoch_id);

-- Beliefs
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    belief_type TEXT DEFAULT 'fact',
    confidence REAL DEFAULT 0.8,
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,  -- JSON array of episode IDs
    derived_from TEXT,     -- JSON array of memory IDs
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Belief revision fields
    supersedes TEXT,           -- ID of belief this replaced
    superseded_by TEXT,        -- ID of belief that replaced this
    times_reinforced INTEGER DEFAULT 0,  -- How many times confirmed
    is_active INTEGER DEFAULT 1,  -- 0 if superseded/archived
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    strength REAL DEFAULT 1.0,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    source_entity TEXT,
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Epoch tracking
    epoch_id TEXT,
    -- Processing state
    processed INTEGER DEFAULT 0,
    -- Belief scope and domain metadata (KEP v3)
    belief_scope TEXT DEFAULT 'world',
    source_domain TEXT,
    cross_domain_applications TEXT,
    abstraction_level TEXT DEFAULT 'specific',
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_beliefs_agent ON beliefs(stack_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence);
CREATE INDEX IF NOT EXISTS idx_beliefs_source_type ON beliefs(source_type);
CREATE INDEX IF NOT EXISTS idx_beliefs_is_active ON beliefs(is_active);
CREATE INDEX IF NOT EXISTS idx_beliefs_supersedes ON beliefs(supersedes);
CREATE INDEX IF NOT EXISTS idx_beliefs_superseded_by ON beliefs(superseded_by);
CREATE INDEX IF NOT EXISTS idx_beliefs_strength ON beliefs(strength);
CREATE INDEX IF NOT EXISTS idx_beliefs_is_protected ON beliefs(is_protected);
CREATE INDEX IF NOT EXISTS idx_beliefs_belief_scope ON beliefs(belief_scope);
CREATE INDEX IF NOT EXISTS idx_beliefs_source_domain ON beliefs(source_domain);
CREATE INDEX IF NOT EXISTS idx_beliefs_abstraction_level ON beliefs(abstraction_level);
CREATE INDEX IF NOT EXISTS idx_beliefs_epoch ON beliefs(epoch_id);

-- Values
CREATE TABLE IF NOT EXISTS agent_values (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    name TEXT NOT NULL,
    statement TEXT NOT NULL,
    priority INTEGER DEFAULT 50,
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.9,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 1,  -- Values protected by default
    strength REAL DEFAULT 1.0,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    source_entity TEXT,
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Epoch tracking
    epoch_id TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_values_stack ON agent_values(stack_id);
CREATE INDEX IF NOT EXISTS idx_values_confidence ON agent_values(confidence);
CREATE INDEX IF NOT EXISTS idx_values_strength ON agent_values(strength);
CREATE INDEX IF NOT EXISTS idx_values_is_protected ON agent_values(is_protected);
CREATE INDEX IF NOT EXISTS idx_values_epoch ON agent_values(epoch_id);

-- Goals
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    goal_type TEXT DEFAULT 'task',
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    strength REAL DEFAULT 1.0,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    source_entity TEXT,
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Epoch tracking
    epoch_id TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_goals_agent ON goals(stack_id);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_confidence ON goals(confidence);
CREATE INDEX IF NOT EXISTS idx_goals_strength ON goals(strength);
CREATE INDEX IF NOT EXISTS idx_goals_is_protected ON goals(is_protected);
CREATE INDEX IF NOT EXISTS idx_goals_epoch ON goals(epoch_id);

-- Notes (memories)
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    content TEXT NOT NULL,
    note_type TEXT DEFAULT 'note',
    speaker TEXT,
    reason TEXT,
    tags TEXT,  -- JSON array
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    strength REAL DEFAULT 1.0,
    processed INTEGER DEFAULT 0,   -- Whether note has been processed for promotion
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    source_entity TEXT,
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Epoch tracking
    epoch_id TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notes_agent ON notes(stack_id);
CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at);
CREATE INDEX IF NOT EXISTS idx_notes_confidence ON notes(confidence);
CREATE INDEX IF NOT EXISTS idx_notes_strength ON notes(strength);
CREATE INDEX IF NOT EXISTS idx_notes_is_protected ON notes(is_protected);
CREATE INDEX IF NOT EXISTS idx_notes_epoch ON notes(epoch_id);

-- Drives
CREATE TABLE IF NOT EXISTS drives (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    drive_type TEXT NOT NULL,
    intensity REAL DEFAULT 0.5,
    focus_areas TEXT,  -- JSON array
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 1,  -- Drives protected by default
    strength REAL DEFAULT 1.0,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    source_entity TEXT,
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Epoch tracking
    epoch_id TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,
    UNIQUE(stack_id, drive_type)
);
CREATE INDEX IF NOT EXISTS idx_drives_agent ON drives(stack_id);
CREATE INDEX IF NOT EXISTS idx_drives_confidence ON drives(confidence);
CREATE INDEX IF NOT EXISTS idx_drives_strength ON drives(strength);
CREATE INDEX IF NOT EXISTS idx_drives_is_protected ON drives(is_protected);
CREATE INDEX IF NOT EXISTS idx_drives_epoch ON drives(epoch_id);

-- Relationships
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    notes TEXT,
    sentiment REAL DEFAULT 0.0,
    interaction_count INTEGER DEFAULT 0,
    last_interaction TEXT,
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    strength REAL DEFAULT 1.0,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    source_entity TEXT,
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Epoch tracking
    epoch_id TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,
    UNIQUE(stack_id, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_relationships_agent ON relationships(stack_id);
CREATE INDEX IF NOT EXISTS idx_relationships_confidence ON relationships(confidence);
CREATE INDEX IF NOT EXISTS idx_relationships_strength ON relationships(strength);
CREATE INDEX IF NOT EXISTS idx_relationships_is_protected ON relationships(is_protected);
CREATE INDEX IF NOT EXISTS idx_relationships_epoch ON relationships(epoch_id);

-- Trust assessments (KEP v3 section 8)
CREATE TABLE IF NOT EXISTS trust_assessments (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    dimensions TEXT NOT NULL,       -- JSON: {"general": {"score": 0.95}, ...}
    authority TEXT DEFAULT '[]',    -- JSON: [{"scope": "all"}, ...]
    evidence_episode_ids TEXT,      -- JSON array of episode IDs
    last_updated TEXT,
    created_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,
    UNIQUE(stack_id, entity)
);
CREATE INDEX IF NOT EXISTS idx_trust_agent ON trust_assessments(stack_id);
CREATE INDEX IF NOT EXISTS idx_trust_entity ON trust_assessments(stack_id, entity);

-- Relationship history (tracking changes over time)
CREATE TABLE IF NOT EXISTS relationship_history (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    relationship_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- interaction, trust_change, type_change, note
    old_value TEXT,            -- JSON: previous state
    new_value TEXT,            -- JSON: new state
    episode_id TEXT,           -- Related episode if applicable
    notes TEXT,
    created_at TEXT NOT NULL,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_rel_history_agent ON relationship_history(stack_id);
CREATE INDEX IF NOT EXISTS idx_rel_history_relationship ON relationship_history(relationship_id);
CREATE INDEX IF NOT EXISTS idx_rel_history_entity ON relationship_history(entity_name);
CREATE INDEX IF NOT EXISTS idx_rel_history_event_type ON relationship_history(event_type);

-- Entity models (mental models of other entities)
CREATE TABLE IF NOT EXISTS entity_models (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    model_type TEXT NOT NULL,  -- behavioral, preference, capability
    observation TEXT NOT NULL,
    confidence REAL DEFAULT 0.7,
    source_episodes TEXT,      -- JSON array of episode IDs
    created_at TEXT NOT NULL,
    updated_at TEXT,
    -- Privacy fields
    subject_ids TEXT,          -- JSON array, auto-populated from entity_name
    access_grants TEXT,
    consent_grants TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_entity_models_agent ON entity_models(stack_id);
CREATE INDEX IF NOT EXISTS idx_entity_models_entity ON entity_models(entity_name);
CREATE INDEX IF NOT EXISTS idx_entity_models_type ON entity_models(model_type);


-- Epochs (temporal era tracking)
CREATE TABLE IF NOT EXISTS epochs (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    epoch_number INTEGER NOT NULL,
    name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,              -- NULL = still active
    trigger_type TEXT DEFAULT 'declared',  -- declared, detected, system
    trigger_description TEXT,
    summary TEXT,
    key_belief_ids TEXT,        -- JSON array
    key_relationship_ids TEXT,  -- JSON array
    key_goal_ids TEXT,          -- JSON array
    dominant_drive_ids TEXT,    -- JSON array
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,
    UNIQUE(stack_id, epoch_number)
);
CREATE INDEX IF NOT EXISTS idx_epochs_agent ON epochs(stack_id);
CREATE INDEX IF NOT EXISTS idx_epochs_number ON epochs(stack_id, epoch_number);
CREATE INDEX IF NOT EXISTS idx_epochs_active ON epochs(ended_at);

-- Playbooks (procedural memory - "how I do things")
CREATE TABLE IF NOT EXISTS playbooks (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    trigger_conditions TEXT NOT NULL,  -- JSON array
    steps TEXT NOT NULL,               -- JSON array of {action, details, adaptations}
    failure_modes TEXT NOT NULL,       -- JSON array
    recovery_steps TEXT,               -- JSON array (optional)
    mastery_level TEXT DEFAULT 'novice',  -- novice/competent/proficient/expert
    times_used INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    source_episodes TEXT,              -- JSON array of episode IDs
    tags TEXT,                         -- JSON array
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    last_used TEXT,
    created_at TEXT NOT NULL,
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_playbooks_agent ON playbooks(stack_id);
CREATE INDEX IF NOT EXISTS idx_playbooks_mastery ON playbooks(mastery_level);
CREATE INDEX IF NOT EXISTS idx_playbooks_times_used ON playbooks(times_used);
CREATE INDEX IF NOT EXISTS idx_playbooks_confidence ON playbooks(confidence);

-- Raw entries (unstructured blob capture for later processing)
-- The blob field is the primary storage - unvalidated, high limit brain dumps
CREATE TABLE IF NOT EXISTS raw_entries (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    blob TEXT NOT NULL,  -- Unstructured brain dump (primary field)
    captured_at TEXT NOT NULL,  -- When captured (was: timestamp)
    source TEXT DEFAULT 'unknown',  -- Auto-populated: cli|mcp|sdk|import|unknown
    processed INTEGER DEFAULT 0,
    processed_into TEXT,  -- JSON array of memory refs (type:id)
    -- DEPRECATED columns (kept for migration, will be removed)
    content TEXT,  -- Use blob instead
    timestamp TEXT,  -- Use captured_at instead
    tags TEXT,  -- Include in blob text instead
    confidence REAL DEFAULT 1.0,  -- Not meaningful for raw
    source_type TEXT DEFAULT 'direct_experience',  -- Meta-memory, not for raw
    -- Privacy fields (Phase 8a)
    subject_ids TEXT,       -- JSON array of entity IDs this memory is about
    access_grants TEXT,     -- JSON array of entity IDs who can see this (empty = private)
    consent_grants TEXT,    -- JSON array of entity IDs who authorized sharing
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_raw_agent ON raw_entries(stack_id);
CREATE INDEX IF NOT EXISTS idx_raw_processed ON raw_entries(stack_id, processed);
CREATE INDEX IF NOT EXISTS idx_raw_captured_at ON raw_entries(stack_id, captured_at DESC, id DESC);
-- Partial index for anxiety system queries (unprocessed entries)
CREATE INDEX IF NOT EXISTS idx_raw_unprocessed ON raw_entries(captured_at)
    WHERE processed = 0 AND deleted = 0;

-- FTS5 virtual table for raw blob keyword search (safety net for backlogs)
-- Note: FTS5 table is created separately in _ensure_fts5_tables() due to SQLite version differences

-- Memory suggestions (auto-extracted suggestions awaiting review)
CREATE TABLE IF NOT EXISTS memory_suggestions (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- episode, belief, note
    content TEXT NOT NULL,  -- JSON object with structured data
    confidence REAL DEFAULT 0.5,
    source_raw_ids TEXT NOT NULL,  -- JSON array of raw entry IDs
    status TEXT DEFAULT 'pending',  -- pending, promoted, modified, rejected
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution_reason TEXT,
    promoted_to TEXT,  -- Format: type:id (e.g., episode:abc123)
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_suggestions_agent ON memory_suggestions(stack_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_status ON memory_suggestions(stack_id, status);
CREATE INDEX IF NOT EXISTS idx_suggestions_type ON memory_suggestions(stack_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_suggestions_created ON memory_suggestions(created_at);

-- Health check events (compliance tracking)
CREATE TABLE IF NOT EXISTS health_check_events (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    anxiety_score INTEGER,
    source TEXT DEFAULT 'cli',  -- cli, mcp
    triggered_by TEXT DEFAULT 'manual'  -- boot, heartbeat, manual
);
CREATE INDEX IF NOT EXISTS idx_health_check_agent ON health_check_events(stack_id);
CREATE INDEX IF NOT EXISTS idx_health_check_checked_at ON health_check_events(checked_at);

-- Sync queue for offline changes (enhanced v2)
CREATE TABLE IF NOT EXISTS sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    operation TEXT NOT NULL,  -- insert, update, delete
    data TEXT,  -- JSON payload of the record data
    local_updated_at TEXT NOT NULL,
    synced INTEGER DEFAULT 0,  -- 0 = pending, 1 = synced
    payload TEXT,  -- Legacy: JSON payload (kept for backward compatibility)
    queued_at TEXT,  -- Legacy: kept for backward compatibility
    -- Retry tracking for resilient sync (v0.2.5)
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,
    last_attempt_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_queue_table ON sync_queue(table_name);
CREATE INDEX IF NOT EXISTS idx_sync_queue_record ON sync_queue(record_id);
CREATE INDEX IF NOT EXISTS idx_sync_queue_synced ON sync_queue(synced);
-- Unique partial index for atomic UPSERT on unsynced entries
CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_queue_unsynced_unique
    ON sync_queue(table_name, record_id) WHERE synced = 0;

-- Sync metadata (tracks last sync time and state)
CREATE TABLE IF NOT EXISTS sync_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Sync conflict history (tracks resolved conflicts for user visibility)
CREATE TABLE IF NOT EXISTS sync_conflicts (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    local_version TEXT NOT NULL,   -- JSON snapshot of local version
    cloud_version TEXT NOT NULL,   -- JSON snapshot of cloud version
    resolution TEXT NOT NULL,      -- "local_wins" or "cloud_wins"
    resolved_at TEXT NOT NULL,
    local_summary TEXT,            -- Human-readable summary
    cloud_summary TEXT,
    source TEXT,
    diff_hash TEXT,
    policy_decision TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_resolved ON sync_conflicts(resolved_at);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_record ON sync_conflicts(table_name, record_id);

-- Boot config (always-available key/value config for agents)
CREATE TABLE IF NOT EXISTS boot_config (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4)))),
    stack_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stack_id, key)
);
CREATE INDEX IF NOT EXISTS idx_boot_agent ON boot_config(stack_id);

-- Diagnostic sessions (formal health check sessions)
CREATE TABLE IF NOT EXISTS diagnostic_sessions (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    session_type TEXT DEFAULT 'self_requested',  -- self_requested, routine, anomaly_triggered, operator_initiated
    access_level TEXT DEFAULT 'structural',       -- structural, content, full
    status TEXT DEFAULT 'active',                 -- active, completed, cancelled
    consent_given INTEGER DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_diag_sessions_agent ON diagnostic_sessions(stack_id);
CREATE INDEX IF NOT EXISTS idx_diag_sessions_status ON diagnostic_sessions(stack_id, status);

-- Diagnostic reports (findings from diagnostic sessions)
CREATE TABLE IF NOT EXISTS diagnostic_reports (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    session_id TEXT NOT NULL,  -- References diagnostic_sessions.id
    findings TEXT,              -- JSON array of findings
    summary TEXT,
    created_at TEXT NOT NULL,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_diag_reports_agent ON diagnostic_reports(stack_id);
CREATE INDEX IF NOT EXISTS idx_diag_reports_session ON diagnostic_reports(session_id);

-- Agent summaries (fractal summarization)
CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    scope TEXT NOT NULL,                         -- 'month' | 'quarter' | 'year' | 'decade' | 'epoch'
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    epoch_id TEXT,
    content TEXT NOT NULL,                        -- Agent-written narrative compression
    key_themes TEXT,                              -- JSON array
    supersedes TEXT,                              -- JSON array of lower-scope summary IDs this covers
    is_protected INTEGER DEFAULT 1,              -- Never forgotten
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_summaries_stack ON summaries(stack_id);
CREATE INDEX IF NOT EXISTS idx_summaries_scope ON summaries(stack_id, scope);
CREATE INDEX IF NOT EXISTS idx_summaries_period ON summaries(stack_id, period_start, period_end);

-- Self-narrative (autobiographical identity model, KEP v3)
CREATE TABLE IF NOT EXISTS self_narratives (
    id TEXT PRIMARY KEY,
    stack_id TEXT NOT NULL,
    epoch_id TEXT,
    narrative_type TEXT DEFAULT 'identity',   -- 'identity' | 'developmental' | 'aspirational'
    content TEXT NOT NULL,
    key_themes TEXT,                          -- JSON array
    unresolved_tensions TEXT,                 -- JSON array
    is_active INTEGER DEFAULT 1,
    supersedes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_narrative_stack ON self_narratives(stack_id);
CREATE INDEX IF NOT EXISTS idx_narrative_active ON self_narratives(stack_id, narrative_type, is_active);

-- Embedding metadata (tracks what's been embedded)
CREATE TABLE IF NOT EXISTS embedding_meta (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- To detect when re-embedding needed
    created_at TEXT NOT NULL,
    embedding_provider TEXT,     -- Provider that generated this embedding
    fallback_used INTEGER DEFAULT 0  -- 1 if fallback embedder was used
);
CREATE INDEX IF NOT EXISTS idx_embedding_meta_record ON embedding_meta(table_name, record_id);

-- Memory audit trail (v0.9.0, correlation_id added v0.14.0)
CREATE TABLE IF NOT EXISTS memory_audit (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    details TEXT,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    correlation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_memory ON memory_audit(memory_type, memory_id);
CREATE INDEX IF NOT EXISTS idx_audit_operation ON memory_audit(operation);
CREATE INDEX IF NOT EXISTS idx_audit_created ON memory_audit(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_correlation ON memory_audit(correlation_id);

-- Processing configuration (v0.9.0)
CREATE TABLE IF NOT EXISTS processing_config (
    layer_transition TEXT PRIMARY KEY,
    enabled INTEGER DEFAULT 1,
    model_id TEXT,
    quantity_threshold INTEGER,
    valence_threshold REAL,
    time_threshold_hours INTEGER,
    batch_size INTEGER DEFAULT 10,
    max_sessions_per_day INTEGER,
    updated_at TEXT
);

-- Stack settings (per-stack feature flags)
CREATE TABLE IF NOT EXISTS stack_settings (
    stack_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (stack_id, key)
);
"""

# Virtual table for vector search (created when sqlite-vec is available)
VECTOR_SCHEMA = """
-- Vector table using sqlite-vec
-- dimension should match embedding provider
CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""


def init_db(
    conn: sqlite3.Connection,
    stack_id: str,
    has_vec: bool,
    embedder_dimension: int,
    load_vec_fn,
    db_path,
    agent_dir,
) -> None:
    """Initialize the database schema.

    Args:
        conn: Database connection.
        stack_id: The stack identifier (needed for stack_settings migration).
        has_vec: Whether sqlite-vec is available.
        embedder_dimension: Dimension for vector embeddings.
        load_vec_fn: Function to load sqlite-vec extension into a connection.
        db_path: Path to the database file (for permissions).
        agent_dir: Path to the agent directory (for permissions).
    """
    # First, run migrations if needed (before executing full schema)
    migrate_schema(conn, stack_id)

    # Now execute full schema (CREATE TABLE IF NOT EXISTS is safe)
    conn.executescript(SCHEMA)

    # Check/set schema version
    cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
    row = cur.fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    else:
        # Update schema version
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    # Create vector table if sqlite-vec is available
    if has_vec:
        load_vec_fn(conn)
        vec_schema = VECTOR_SCHEMA.format(dim=embedder_dimension)
        try:
            conn.executescript(vec_schema)
        except sqlite3.OperationalError as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create vector table: {e}", exc_info=True)

    # Create FTS5 table for raw blob keyword search
    ensure_raw_fts5(conn)

    conn.commit()

    # Set secure file permissions (owner read/write only)
    import os

    try:
        os.chmod(db_path, 0o600)
        os.chmod(db_path.parent, 0o700)
        if agent_dir.exists():
            os.chmod(agent_dir, 0o700)
    except OSError as e:
        logger.warning(f"Could not set secure permissions: {e}", exc_info=True)


def ensure_raw_fts5(conn: sqlite3.Connection) -> None:
    """Create FTS5 virtual table for raw blob keyword search.

    FTS5 provides fast keyword search on raw blobs as a safety net
    when backlogs accumulate. This is separate from semantic search.
    """
    try:
        # Check if FTS5 table already exists
        result = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='raw_fts'
        """).fetchone()

        if result is None:
            # Create FTS5 content table (external content mode)
            # This links to raw_entries.blob without duplicating data
            conn.execute("""
                CREATE VIRTUAL TABLE raw_fts USING fts5(
                    blob,
                    content=raw_entries,
                    content_rowid=rowid
                )
            """)
            logger.info("Created raw_fts FTS5 table for keyword search")

            # Populate FTS index from existing data
            conn.execute("INSERT INTO raw_fts(raw_fts) VALUES('rebuild')")
            logger.info("Rebuilt raw_fts index")

    except sqlite3.OperationalError as e:
        # FTS5 might not be available in all SQLite builds
        if "no such module: fts5" in str(e).lower():
            logger.warning(
                "FTS5 not available in this SQLite build - keyword search disabled", exc_info=True
            )
        elif "already exists" in str(e).lower():
            pass  # Table already exists, that's fine
        else:
            logger.warning(f"Could not create FTS5 table: {e}", exc_info=True)
    except Exception as e:
        logger.warning(f"FTS5 setup failed: {e}", exc_info=True)


def migrate_schema(conn: sqlite3.Connection, stack_id: str) -> None:
    """Run schema migrations for existing databases.

    Handles adding new columns to existing tables.

    Args:
        conn: Database connection.
        stack_id: The stack identifier (needed for stack_settings migration).
    """
    # Check if tables exist first
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {t[0] for t in tables}

    if "episodes" not in table_names:
        # Fresh database, no migration needed
        return

    # Get current columns for each table
    def get_columns(table: str) -> set:
        try:
            validate_table_name(table)  # Security: defense-in-depth
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            return {c[1] for c in cols}
        except (TypeError, ValueError):
            return set()

    # Migrations for episodes table
    episode_cols = get_columns("episodes")
    migrations = []

    if "emotional_valence" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN emotional_valence REAL DEFAULT 0.0")
    if "emotional_arousal" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN emotional_arousal REAL DEFAULT 0.0")
    if "emotional_tags" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN emotional_tags TEXT")
    if "confidence" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN confidence REAL DEFAULT 0.8")
    if "source_type" not in episode_cols:
        migrations.append(
            "ALTER TABLE episodes ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
        )
    if "source_episodes" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN source_episodes TEXT")
    if "derived_from" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN derived_from TEXT")
    if "last_verified" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN last_verified TEXT")
    if "verification_count" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN verification_count INTEGER DEFAULT 0")
    if "confidence_history" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN confidence_history TEXT")
    if "repeat" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN repeat TEXT")
    if "avoid" not in episode_cols:
        migrations.append("ALTER TABLE episodes ADD COLUMN avoid TEXT")

    # Migrations for beliefs table
    belief_cols = get_columns("beliefs")
    if "beliefs" in table_names:
        if "source_type" not in belief_cols:
            migrations.append(
                "ALTER TABLE beliefs ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
            )
        if "source_episodes" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN source_episodes TEXT")
        if "derived_from" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN derived_from TEXT")
        if "last_verified" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN last_verified TEXT")
        if "verification_count" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN verification_count INTEGER DEFAULT 0")
        if "confidence_history" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN confidence_history TEXT")
        # Belief revision fields
        if "supersedes" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN supersedes TEXT")
        if "superseded_by" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN superseded_by TEXT")
        if "times_reinforced" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN times_reinforced INTEGER DEFAULT 0")
        if "is_active" not in belief_cols:
            migrations.append("ALTER TABLE beliefs ADD COLUMN is_active INTEGER DEFAULT 1")

    # Migrations for values table
    value_cols = get_columns("agent_values")
    if "agent_values" in table_names:
        if "confidence" not in value_cols:
            migrations.append("ALTER TABLE agent_values ADD COLUMN confidence REAL DEFAULT 0.9")
        if "source_type" not in value_cols:
            migrations.append(
                "ALTER TABLE agent_values ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
            )
        if "source_episodes" not in value_cols:
            migrations.append("ALTER TABLE agent_values ADD COLUMN source_episodes TEXT")
        if "derived_from" not in value_cols:
            migrations.append("ALTER TABLE agent_values ADD COLUMN derived_from TEXT")
        if "last_verified" not in value_cols:
            migrations.append("ALTER TABLE agent_values ADD COLUMN last_verified TEXT")
        if "verification_count" not in value_cols:
            migrations.append(
                "ALTER TABLE agent_values ADD COLUMN verification_count INTEGER DEFAULT 0"
            )
        if "confidence_history" not in value_cols:
            migrations.append("ALTER TABLE agent_values ADD COLUMN confidence_history TEXT")

    # Migrations for goals table
    goal_cols = get_columns("goals")
    if "goals" in table_names:
        if "confidence" not in goal_cols:
            migrations.append("ALTER TABLE goals ADD COLUMN confidence REAL DEFAULT 0.8")
        if "source_type" not in goal_cols:
            migrations.append(
                "ALTER TABLE goals ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
            )
        if "source_episodes" not in goal_cols:
            migrations.append("ALTER TABLE goals ADD COLUMN source_episodes TEXT")
        if "derived_from" not in goal_cols:
            migrations.append("ALTER TABLE goals ADD COLUMN derived_from TEXT")
        if "last_verified" not in goal_cols:
            migrations.append("ALTER TABLE goals ADD COLUMN last_verified TEXT")
        if "verification_count" not in goal_cols:
            migrations.append("ALTER TABLE goals ADD COLUMN verification_count INTEGER DEFAULT 0")
        if "confidence_history" not in goal_cols:
            migrations.append("ALTER TABLE goals ADD COLUMN confidence_history TEXT")
        if "goal_type" not in goal_cols:
            migrations.append("ALTER TABLE goals ADD COLUMN goal_type TEXT DEFAULT 'task'")

    # Migrations for notes table
    note_cols = get_columns("notes")
    if "notes" in table_names:
        if "confidence" not in note_cols:
            migrations.append("ALTER TABLE notes ADD COLUMN confidence REAL DEFAULT 0.8")
        if "source_type" not in note_cols:
            migrations.append(
                "ALTER TABLE notes ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
            )
        if "source_episodes" not in note_cols:
            migrations.append("ALTER TABLE notes ADD COLUMN source_episodes TEXT")
        if "derived_from" not in note_cols:
            migrations.append("ALTER TABLE notes ADD COLUMN derived_from TEXT")
        if "last_verified" not in note_cols:
            migrations.append("ALTER TABLE notes ADD COLUMN last_verified TEXT")
        if "verification_count" not in note_cols:
            migrations.append("ALTER TABLE notes ADD COLUMN verification_count INTEGER DEFAULT 0")
        if "confidence_history" not in note_cols:
            migrations.append("ALTER TABLE notes ADD COLUMN confidence_history TEXT")

    # Migrations for drives table
    drive_cols = get_columns("drives")
    if "drives" in table_names:
        if "confidence" not in drive_cols:
            migrations.append("ALTER TABLE drives ADD COLUMN confidence REAL DEFAULT 0.8")
        if "source_type" not in drive_cols:
            migrations.append(
                "ALTER TABLE drives ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
            )
        if "source_episodes" not in drive_cols:
            migrations.append("ALTER TABLE drives ADD COLUMN source_episodes TEXT")
        if "derived_from" not in drive_cols:
            migrations.append("ALTER TABLE drives ADD COLUMN derived_from TEXT")
        if "last_verified" not in drive_cols:
            migrations.append("ALTER TABLE drives ADD COLUMN last_verified TEXT")
        if "verification_count" not in drive_cols:
            migrations.append("ALTER TABLE drives ADD COLUMN verification_count INTEGER DEFAULT 0")
        if "confidence_history" not in drive_cols:
            migrations.append("ALTER TABLE drives ADD COLUMN confidence_history TEXT")

    # Migrations for relationships table
    rel_cols = get_columns("relationships")
    if "relationships" in table_names:
        if "confidence" not in rel_cols:
            migrations.append("ALTER TABLE relationships ADD COLUMN confidence REAL DEFAULT 0.8")
        if "source_type" not in rel_cols:
            migrations.append(
                "ALTER TABLE relationships ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
            )
        if "source_episodes" not in rel_cols:
            migrations.append("ALTER TABLE relationships ADD COLUMN source_episodes TEXT")
        if "derived_from" not in rel_cols:
            migrations.append("ALTER TABLE relationships ADD COLUMN derived_from TEXT")
        if "last_verified" not in rel_cols:
            migrations.append("ALTER TABLE relationships ADD COLUMN last_verified TEXT")
        if "verification_count" not in rel_cols:
            migrations.append(
                "ALTER TABLE relationships ADD COLUMN verification_count INTEGER DEFAULT 0"
            )
        if "confidence_history" not in rel_cols:
            migrations.append("ALTER TABLE relationships ADD COLUMN confidence_history TEXT")

    # Migrations for sync_queue table (enhanced v2)
    sync_cols = get_columns("sync_queue")
    if "sync_queue" in table_names:
        if "payload" not in sync_cols:
            migrations.append("ALTER TABLE sync_queue ADD COLUMN payload TEXT")
        if "data" not in sync_cols:
            migrations.append("ALTER TABLE sync_queue ADD COLUMN data TEXT")
        if "local_updated_at" not in sync_cols:
            migrations.append("ALTER TABLE sync_queue ADD COLUMN local_updated_at TEXT")
        if "synced" not in sync_cols:
            migrations.append("ALTER TABLE sync_queue ADD COLUMN synced INTEGER DEFAULT 0")
        # Retry tracking for resilient sync (v0.2.5)
        if "retry_count" not in sync_cols:
            migrations.append("ALTER TABLE sync_queue ADD COLUMN retry_count INTEGER DEFAULT 0")
        if "last_error" not in sync_cols:
            migrations.append("ALTER TABLE sync_queue ADD COLUMN last_error TEXT")
        if "last_attempt_at" not in sync_cols:
            migrations.append("ALTER TABLE sync_queue ADD COLUMN last_attempt_at TEXT")

    # Migrations for sync_conflicts metadata (v0.13.06)
    sync_conflict_cols = get_columns("sync_conflicts")
    if "sync_conflicts" in table_names:
        if "source" not in sync_conflict_cols:
            migrations.append("ALTER TABLE sync_conflicts ADD COLUMN source TEXT")
        if "diff_hash" not in sync_conflict_cols:
            migrations.append("ALTER TABLE sync_conflicts ADD COLUMN diff_hash TEXT")
        if "policy_decision" not in sync_conflict_cols:
            migrations.append("ALTER TABLE sync_conflicts ADD COLUMN policy_decision TEXT")

    # === Forgetting field migrations ===
    # Add forgetting fields to all memory tables
    forgetting_tables = [
        ("episodes", False),  # (table_name, protected_by_default)
        ("beliefs", False),
        ("agent_values", True),  # Values protected by default
        ("goals", False),
        ("notes", False),
        ("drives", True),  # Drives protected by default
        ("relationships", False),
    ]

    for table, protected_default in forgetting_tables:
        if table not in table_names:
            continue
        cols = get_columns(table)
        protected_val = 1 if protected_default else 0

        if "times_accessed" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN times_accessed INTEGER DEFAULT 0")
        if "last_accessed" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN last_accessed TEXT")
        if "is_protected" not in cols:
            migrations.append(
                f"ALTER TABLE {table} ADD COLUMN is_protected INTEGER DEFAULT {protected_val}"
            )
        if "is_forgotten" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN is_forgotten INTEGER DEFAULT 0")
        if "forgotten_at" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN forgotten_at TEXT")
        if "forgotten_reason" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN forgotten_reason TEXT")

    # === Context/scope field migrations ===
    # Add context fields to all memory tables
    context_tables = [
        "episodes",
        "beliefs",
        "agent_values",
        "goals",
        "notes",
        "drives",
        "relationships",
    ]

    for table in context_tables:
        if table not in table_names:
            continue
        cols = get_columns(table)

        if "context" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN context TEXT")
        if "context_tags" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN context_tags TEXT")

    # === Privacy field migrations (Phase 8a) ===
    # Add privacy fields to all memory tables
    privacy_tables = [
        "episodes",
        "beliefs",
        "agent_values",
        "goals",
        "notes",
        "drives",
        "relationships",
        "playbooks",
        "raw_entries",
    ]

    for table in privacy_tables:
        if table not in table_names:
            continue
        cols = get_columns(table)

        if "subject_ids" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN subject_ids TEXT")
        if "access_grants" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN access_grants TEXT")
        if "consent_grants" not in cols:
            migrations.append(f"ALTER TABLE {table} ADD COLUMN consent_grants TEXT")

    # === Raw entries blob migration ===
    # Migrate from structured content/tags to blob-based storage
    raw_cols = get_columns("raw_entries")
    if "raw_entries" in table_names:
        # Add blob column if it doesn't exist
        if "blob" not in raw_cols:
            migrations.append("ALTER TABLE raw_entries ADD COLUMN blob TEXT")
        # Add captured_at column if it doesn't exist
        if "captured_at" not in raw_cols:
            migrations.append("ALTER TABLE raw_entries ADD COLUMN captured_at TEXT")

        # Execute pending schema migrations first so blob column exists
        for migration in migrations:
            try:
                conn.execute(migration)
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"Migration failed: {migration}: {e}", exc_info=True)
        migrations.clear()

        # Migrate data from content/tags to blob with natural language format
        # Now safe to run since blob column is guaranteed to exist
        try:
            # Check if migration is needed (blob is NULL but content exists)
            needs_migration = conn.execute("""
                SELECT COUNT(*) FROM raw_entries
                WHERE (blob IS NULL OR blob = '') AND content IS NOT NULL AND content != ''
            """).fetchone()[0]

            if needs_migration > 0:
                # Migrate content + tags to blob in natural language format
                conn.execute("""
                    UPDATE raw_entries SET blob =
                        content ||
                        CASE WHEN source IS NOT NULL AND source != 'manual' AND source != '' AND source != 'unknown'
                             THEN ' (from ' || source || ')' ELSE '' END ||
                        CASE WHEN tags IS NOT NULL AND tags != '[]' AND tags != 'null' AND tags != ''
                             THEN ' [tags: ' ||
                                  REPLACE(REPLACE(REPLACE(tags, '["', ''), '"]', ''), '","', ', ') ||
                                  ']'
                             ELSE '' END
                    WHERE (blob IS NULL OR blob = '') AND content IS NOT NULL
                """)
                # Copy timestamp to captured_at
                conn.execute("""
                    UPDATE raw_entries SET captured_at = timestamp
                    WHERE captured_at IS NULL AND timestamp IS NOT NULL
                """)
                # Normalize source to enum values
                conn.execute("""
                    UPDATE raw_entries SET source =
                        CASE
                            WHEN source IN ('cli', 'mcp', 'sdk', 'import') THEN source
                            WHEN source = 'manual' THEN 'cli'
                            WHEN source LIKE '%auto%' THEN 'sdk'
                            ELSE 'unknown'
                        END
                """)
                logger.info(f"Migrated {needs_migration} raw entries to blob format")
        except Exception as e:
            logger.warning(f"Raw blob data migration failed: {e}", exc_info=True)

    # Create health_check_events table if it doesn't exist
    if "health_check_events" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS health_check_events (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                anxiety_score INTEGER,
                source TEXT DEFAULT 'cli',
                triggered_by TEXT DEFAULT 'manual'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_health_check_agent ON health_check_events(stack_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_health_check_checked_at ON health_check_events(checked_at)"
        )
        logger.info("Created health_check_events table")

    # Execute migrations
    for migration in migrations:
        try:
            conn.execute(migration)
            logger.debug(f"Migration applied: {migration}")
        except Exception as e:
            logger.warning(
                f"Migration failed (may already exist): {migration} - {e}", exc_info=True
            )

    if migrations:
        conn.commit()
        logger.info(f"Applied {len(migrations)} schema migrations")

    # v13: Add source_entity column for entity-neutral provenance (Phase 5)
    source_entity_tables = [
        "episodes",
        "beliefs",
        "agent_values",
        "goals",
        "notes",
        "drives",
        "relationships",
    ]
    for table in source_entity_tables:
        if table in table_names:
            cols = get_columns(table)
            if "source_entity" not in cols:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN source_entity TEXT")
                    logger.info(f"Added source_entity column to {table}")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(
                            f"Failed to add source_entity to {table}: {e}", exc_info=True
                        )
    conn.commit()

    # v14: Add privacy fields (Phase 8a) - subject_ids, access_grants, consent_grants
    privacy_tables = [
        "episodes",
        "beliefs",
        "agent_values",
        "goals",
        "notes",
        "drives",
        "relationships",
        "playbooks",
        "raw_entries",
    ]
    privacy_fields = ["subject_ids", "access_grants", "consent_grants"]
    for table in privacy_tables:
        if table in table_names:
            cols = get_columns(table)
            for field in privacy_fields:
                if field not in cols:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {field} TEXT")
                        logger.info(f"Added {field} column to {table}")
                    except Exception as e:
                        if "duplicate column" not in str(e).lower():
                            logger.warning(f"Failed to add {field} to {table}: {e}", exc_info=True)
    conn.commit()

    # v15: Sync queue payload/data consistency fix
    # Migrate data -> payload where payload is NULL (fixes orphaned entry bug #70)
    if "sync_queue" in table_names:
        try:
            fixed = conn.execute(
                "UPDATE sync_queue SET payload = data WHERE payload IS NULL AND data IS NOT NULL"
            ).rowcount
            if fixed > 0:
                logger.info(f"Fixed {fixed} sync queue entries (data -> payload)")
            conn.commit()
        except Exception as e:
            logger.warning(f"Sync queue payload migration failed: {e}", exc_info=True)

    # v15: Boot config table (Phase 9) - always-available key/value config
    if "boot_config" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS boot_config (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4)))),
                stack_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stack_id, key)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_boot_agent ON boot_config(stack_id)")
        logger.info("Created boot_config table")
        conn.commit()

    # Create trust_assessments table if it doesn't exist
    if "trust_assessments" not in table_names and "agent_trust_assessments" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trust_assessments (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                entity TEXT NOT NULL,
                dimensions TEXT NOT NULL,
                authority TEXT DEFAULT '[]',
                evidence_episode_ids TEXT,
                last_updated TEXT,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                cloud_synced_at TEXT,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0,
                UNIQUE(stack_id, entity)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trust_agent " "ON trust_assessments(stack_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trust_entity " "ON trust_assessments(stack_id, entity)"
        )
        logger.info("Created trust_assessments table")
        conn.commit()
        # NOTE: Self-trust bootstrapping (identity entity with score 1.0) is
        # handled at runtime by SQLiteStack._ensure_self_trust(), not during
        # schema migration.  This keeps migrations DDL-only and avoids
        # coupling migration code to stack_id / runtime context.

    # v20: Add diagnostic sessions and reports tables
    if "diagnostic_sessions" not in table_names and "agent_diagnostic_sessions" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnostic_sessions (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                session_type TEXT DEFAULT 'self_requested',
                access_level TEXT DEFAULT 'structural',
                status TEXT DEFAULT 'active',
                consent_given INTEGER DEFAULT 0,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                local_updated_at TEXT NOT NULL,
                cloud_synced_at TEXT,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_diag_sessions_agent " "ON diagnostic_sessions(stack_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_diag_sessions_status "
            "ON diagnostic_sessions(stack_id, status)"
        )
        logger.info("Created diagnostic_sessions table")
        conn.commit()

    if "diagnostic_reports" not in table_names and "agent_diagnostic_reports" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnostic_reports (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                findings TEXT,
                summary TEXT,
                created_at TEXT NOT NULL,
                local_updated_at TEXT NOT NULL,
                cloud_synced_at TEXT,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_diag_reports_agent " "ON diagnostic_reports(stack_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_diag_reports_session "
            "ON diagnostic_reports(session_id)"
        )
        logger.info("Created diagnostic_reports table")
        conn.commit()

    # v18: Add belief_scope and domain metadata (KEP v3)
    if "beliefs" in table_names:
        belief_cols = get_columns("beliefs")
        belief_scope_fields = {
            "belief_scope": "TEXT DEFAULT 'world'",
            "source_domain": "TEXT",
            "cross_domain_applications": "TEXT",
            "abstraction_level": "TEXT DEFAULT 'specific'",
        }
        for field_name, field_type in belief_scope_fields.items():
            if field_name not in belief_cols:
                try:
                    conn.execute(f"ALTER TABLE beliefs ADD COLUMN {field_name} {field_type}")
                    logger.info(f"Added {field_name} column to beliefs")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"Failed to add {field_name} to beliefs: {e}", exc_info=True)
        conn.commit()

    # v19: Add epoch_id columns and epochs table (KEP v3)
    epoch_tables = [
        "episodes",
        "beliefs",
        "agent_values",
        "goals",
        "notes",
        "drives",
        "relationships",
    ]
    for tbl in epoch_tables:
        if tbl in table_names:
            cols = get_columns(tbl)
            if "epoch_id" not in cols:
                try:
                    conn.execute(f"ALTER TABLE {tbl} ADD COLUMN epoch_id TEXT")
                    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_epoch ON {tbl}(epoch_id)")
                    logger.info(f"Added epoch_id column to {tbl}")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"Failed to add epoch_id to {tbl}: {e}", exc_info=True)

    # v21: Add summaries table (fractal summarization)
    if "summaries" not in table_names and "agent_summaries" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                epoch_id TEXT,
                content TEXT NOT NULL,
                key_themes TEXT,
                supersedes TEXT,
                is_protected INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                cloud_synced_at TEXT,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_stack " "ON summaries(stack_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_scope " "ON summaries(stack_id, scope)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_period "
            "ON summaries(stack_id, period_start, period_end)"
        )
        logger.info("Created summaries table")
        conn.commit()

    if "epochs" not in table_names and "agent_epochs" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS epochs (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                epoch_number INTEGER NOT NULL,
                name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                trigger_type TEXT DEFAULT 'declared',
                trigger_description TEXT,
                summary TEXT,
                key_belief_ids TEXT,
                key_relationship_ids TEXT,
                key_goal_ids TEXT,
                dominant_drive_ids TEXT,
                local_updated_at TEXT NOT NULL,
                cloud_synced_at TEXT,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0,
                UNIQUE(stack_id, epoch_number)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_epochs_agent ON epochs(stack_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_epochs_number " "ON epochs(stack_id, epoch_number)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_epochs_active ON epochs(ended_at)")
        logger.info("Created epochs table")
    else:
        # Migrate existing epochs table
        epoch_cols = get_columns("epochs")
        if "trigger_description" not in epoch_cols:
            try:
                conn.execute("ALTER TABLE epochs ADD COLUMN trigger_description TEXT")
                logger.info("Added trigger_description column to epochs")
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"Failed to add trigger_description: {e}", exc_info=True)
    conn.commit()

    # v22: Add self_narratives table (self-narrative layer)
    if "self_narratives" not in table_names and "agent_self_narrative" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS self_narratives (
                id TEXT PRIMARY KEY,
                stack_id TEXT NOT NULL,
                epoch_id TEXT,
                narrative_type TEXT DEFAULT 'identity',
                content TEXT NOT NULL,
                key_themes TEXT,
                unresolved_tensions TEXT,
                is_active INTEGER DEFAULT 1,
                supersedes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                cloud_synced_at TEXT,
                version INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_narrative_stack " "ON self_narratives(stack_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_narrative_active "
            "ON self_narratives(stack_id, narrative_type, is_active)"
        )
        logger.info("Created self_narratives table")
        conn.commit()

    # v23: Rename agent_id -> stack_id columns and drop agent_ table prefixes
    # Check if migration is needed by looking for agent_id column in episodes
    episode_cols = get_columns("episodes")
    if "agent_id" in episode_cols and "stack_id" not in episode_cols:
        logger.info("Migrating schema v22 -> v23: agent_id -> stack_id")

        # 1. Rename agent_id column to stack_id in all tables
        # Include both old and new table names to handle partial migrations
        col_rename_tables = [
            "episodes",
            "beliefs",
            "notes",
            "goals",
            "drives",
            "relationships",
            "playbooks",
            "raw_entries",
            "sync_queue",
            "health_check_events",
            "boot_config",
            "checkpoints",
            "memory_suggestions",
            "agent_values",
            "relationship_history",
            "entity_models",
            # Old names (pre-rename) and new names (post-rename)
            "trust_assessments",
            "agent_trust_assessments",
            "epochs",
            "agent_epochs",
            "diagnostic_sessions",
            "agent_diagnostic_sessions",
            "diagnostic_reports",
            "agent_diagnostic_reports",
            "summaries",
            "agent_summaries",
            "self_narratives",
            "agent_self_narrative",
        ]

        for tbl in col_rename_tables:
            if tbl in table_names:
                cols = get_columns(tbl)
                if "agent_id" in cols:
                    try:
                        conn.execute(f"ALTER TABLE {tbl} RENAME COLUMN agent_id TO stack_id")
                        logger.info(f"Renamed agent_id -> stack_id in {tbl}")
                    except Exception as e:
                        logger.warning(f"Column rename failed for {tbl}: {e}", exc_info=True)

        # 2. Rename tables that had agent_ prefix
        # Note: agent_values is NOT renamed (values is a SQL reserved word)
        table_renames = {
            "agent_trust_assessments": "trust_assessments",
            "agent_epochs": "epochs",
            "agent_diagnostic_sessions": "diagnostic_sessions",
            "agent_diagnostic_reports": "diagnostic_reports",
            "agent_summaries": "summaries",
            "agent_self_narrative": "self_narratives",
        }
        for old_name, new_name in table_renames.items():
            if old_name in table_names:
                try:
                    conn.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")
                    logger.info(f"Renamed table {old_name} -> {new_name}")
                except Exception as e:
                    logger.warning(
                        f"Table rename failed {old_name} -> {new_name}: {e}", exc_info=True
                    )

        conn.commit()
        logger.info("Schema migration v23 complete (agent -> stack)")

    # v23 catch-up: fix any tables missed by a partial migration
    # Scans ALL tables for any remaining agent_id columns
    catchup_tables = [
        t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    for tbl in catchup_tables:
        if tbl in table_names:
            cols = get_columns(tbl)
            if "agent_id" in cols and "stack_id" not in cols:
                try:
                    conn.execute(f"ALTER TABLE {tbl} RENAME COLUMN agent_id TO stack_id")
                    logger.info(f"v23 catch-up: renamed agent_id -> stack_id in {tbl}")
                except Exception as e:
                    logger.warning(f"v23 catch-up failed for {tbl}: {e}", exc_info=True)
    conn.commit()

    # v24: Memory integrity -- replace is_forgotten/forgotten_at/forgotten_reason with strength
    # Add strength column and processed column, migrate existing data
    strength_tables = [
        "episodes",
        "beliefs",
        "agent_values",
        "goals",
        "notes",
        "drives",
        "relationships",
    ]
    for tbl in strength_tables:
        if tbl not in table_names:
            continue
        cols = get_columns(tbl)

        # Add strength column if not present
        if "strength" not in cols:
            try:
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN strength REAL DEFAULT 1.0")
                logger.info(f"Added strength column to {tbl}")
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"Failed to add strength to {tbl}: {e}", exc_info=True)

        # Migrate existing data: forgotten memories get strength = 0.0
        if "is_forgotten" in cols:
            try:
                migrated = conn.execute(
                    f"UPDATE {tbl} SET strength = 0.0 WHERE is_forgotten = 1 AND (strength IS NULL OR strength = 1.0)"
                ).rowcount
                if migrated > 0:
                    logger.info(f"Migrated {migrated} forgotten records in {tbl} to strength=0.0")
            except Exception as e:
                logger.warning(f"Failed to migrate is_forgotten data in {tbl}: {e}", exc_info=True)

    # Add processed column to episodes, notes, and beliefs
    for tbl in ["episodes", "notes", "beliefs"]:
        if tbl not in table_names:
            continue
        cols = get_columns(tbl)
        if "processed" not in cols:
            try:
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN processed INTEGER DEFAULT 0")
                logger.info(f"Added processed column to {tbl}")
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"Failed to add processed to {tbl}: {e}", exc_info=True)

    # Create memory_audit table if it doesn't exist
    if "memory_audit" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_audit (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                details TEXT,
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_memory ON memory_audit(memory_type, memory_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_operation ON memory_audit(operation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON memory_audit(created_at)")
        logger.info("Created memory_audit table")

    # Create processing_config table if it doesn't exist
    if "processing_config" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_config (
                layer_transition TEXT PRIMARY KEY,
                enabled INTEGER DEFAULT 1,
                model_id TEXT,
                quantity_threshold INTEGER,
                valence_threshold REAL,
                time_threshold_hours INTEGER,
                batch_size INTEGER DEFAULT 10,
                max_sessions_per_day INTEGER,
                updated_at TEXT
            )
        """)
        logger.info("Created processing_config table")

    # Create stack_settings table if it doesn't exist
    if "stack_settings" not in table_names:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stack_settings (
                stack_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (stack_id, key)
            )
        """)
        logger.info("Created stack_settings table")
    else:
        # Migrate existing table: add stack_id if missing
        cols = {row[1] for row in conn.execute("PRAGMA table_info(stack_settings)").fetchall()}
        if "stack_id" not in cols:
            conn.execute("ALTER TABLE stack_settings RENAME TO stack_settings_old")
            conn.execute("""
                CREATE TABLE stack_settings (
                    stack_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (stack_id, key)
                )
            """)
            conn.execute(
                """
                INSERT INTO stack_settings (stack_id, key, value, updated_at)
                SELECT ?, key, value, updated_at FROM stack_settings_old
            """,
                (stack_id,),
            )
            conn.execute("DROP TABLE stack_settings_old")
            logger.info("Migrated stack_settings table to include stack_id")

    conn.commit()

    # v25: Backfill captured_at from timestamp and create compound index for pagination
    if "raw_entries" in table_names:
        try:
            backfilled = conn.execute("""
                UPDATE raw_entries SET captured_at = timestamp
                WHERE captured_at IS NULL AND timestamp IS NOT NULL
            """).rowcount
            if backfilled > 0:
                logger.info(f"v25: Backfilled captured_at for {backfilled} raw entries")
        except Exception as e:
            logger.warning(f"v25: captured_at backfill failed: {e}", exc_info=True)

        # Replace old single-column index with compound index for pagination
        try:
            idx_info = conn.execute("""
                SELECT sql FROM sqlite_master
                WHERE type='index' AND name='idx_raw_captured_at'
            """).fetchone()
            needs_rebuild = True
            if idx_info and idx_info[0] and "id DESC" in idx_info[0]:
                needs_rebuild = False  # Already has compound index

            if needs_rebuild:
                conn.execute("DROP INDEX IF EXISTS idx_raw_captured_at")
                conn.execute("""
                    CREATE INDEX idx_raw_captured_at
                    ON raw_entries(stack_id, captured_at DESC, id DESC)
                """)
                logger.info("v25: Created compound index idx_raw_captured_at")
        except Exception as e:
            logger.warning(f"v25: Index rebuild failed: {e}", exc_info=True)

    # v26: Add embedding observability columns to embedding_meta
    if "embedding_meta" in table_names:
        emb_cols = get_columns("embedding_meta")
        if "embedding_provider" not in emb_cols:
            try:
                conn.execute("ALTER TABLE embedding_meta ADD COLUMN embedding_provider TEXT")
                logger.info("v26: Added embedding_provider column to embedding_meta")
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"v26: Failed to add embedding_provider: {e}", exc_info=True)
        if "fallback_used" not in emb_cols:
            try:
                conn.execute(
                    "ALTER TABLE embedding_meta ADD COLUMN fallback_used INTEGER DEFAULT 0"
                )
                logger.info("v26: Added fallback_used column to embedding_meta")
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"v26: Failed to add fallback_used: {e}", exc_info=True)
        conn.commit()

        conn.commit()

    # v27: Add correlation_id to memory_audit for processing session tracking
    if "memory_audit" in table_names:
        audit_cols = get_columns("memory_audit")
        if "correlation_id" not in audit_cols:
            try:
                conn.execute("ALTER TABLE memory_audit ADD COLUMN correlation_id TEXT")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_correlation ON memory_audit(correlation_id)"
                )
                logger.info("v27: Added correlation_id column to memory_audit")
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    logger.warning(f"v27: Failed to add correlation_id: {e}", exc_info=True)

    # v27: Set use_legacy_heuristics=true for existing stacks
    if "stack_settings" in table_names:
        try:
            # Check if any stack data exists (indicating an existing stack)
            has_data = conn.execute(
                "SELECT 1 FROM stack_settings WHERE stack_id = ? LIMIT 1",
                (stack_id,),
            ).fetchone()
            if has_data:
                # Only set if not already configured
                existing = conn.execute(
                    "SELECT 1 FROM stack_settings WHERE stack_id = ? AND key = ?",
                    (stack_id, "use_legacy_heuristics"),
                ).fetchone()
                if not existing:
                    from datetime import datetime, timezone

                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        "INSERT INTO stack_settings (stack_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                        (stack_id, "use_legacy_heuristics", "true", now),
                    )
                    logger.info(
                        "v27: Set use_legacy_heuristics=true for existing stack %s", stack_id
                    )
        except Exception as e:
            logger.warning(f"v27: Failed to set use_legacy_heuristics: {e}", exc_info=True)

    conn.commit()
