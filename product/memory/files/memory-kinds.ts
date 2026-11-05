// Fields of each memory kind, stored in memory.payload (DATA-MEMORY).
// The canonical text of each kind (memory.content) is named per kind below.
// Shared fields (confidence, trust_tier, source_type, source_entity, context,
// context_tags, derived_from, is_protected, foundational) live in columns or links,
// not in the payload.

export type MemoryKind =
  | "episode"
  | "note"
  | "belief"
  | "value"
  | "goal"
  | "drive"
  | "relationship"
  | "playbook"
  | "entity_model"
  | "guardrail";

/** content = objective + " → " + outcome */
export interface EpisodePayload {
  objective: string;
  outcome: string;
  outcome_type?: "success" | "failure" | "mixed" | "neutral";
  lessons?: string[];
  repeat?: string[];
  avoid?: string[];
  tags?: string[];
  emotional_tags?: string[];
  /** Present when the episode was segmented from a runtime transcript. */
  session?: { runtime: "claude-code" | "codex" | "openclaw"; session_id: string; first_seq: number; last_seq: number };
}

/** content = content */
export interface NotePayload {
  content: string;
  note_type:
    | "decision"
    | "diagnostic"
    | "fact"
    | "insight"
    | "note"
    | "observation"
    | "procedure"
    | "quote"
    | "reference";
  speaker?: string;
  reason?: string;
  tags?: string[];
}

/** content = statement */
export interface BeliefPayload {
  statement: string;
  belief_type:
    | "assumption"
    | "causal"
    | "constraint"
    | "evaluative"
    | "fact"
    | "hypothesis"
    | "learned"
    | "model"
    | "observation"
    | "opinion"
    | "pattern"
    | "preference"
    | "principle"
    | "procedural"
    | "rule"
    | "strategy";
  times_reinforced: number;
  belief_scope?: "self" | "world" | "relational";
  source_domain?: string;
  cross_domain_applications?: string[];
  abstraction_level?: "specific" | "domain" | "universal";
}

/** content = name + ": " + statement */
export interface ValuePayload {
  name: string;
  statement: string;
  priority: number; // 0..100
  value_type: "core_value" | "principle" | "commitment";
}

/** content = title + " — " + description */
export interface GoalPayload {
  title: string;
  description?: string;
  goal_type: "task" | "aspiration" | "commitment" | "exploration";
  priority: "low" | "medium" | "high" | "urgent";
  status: "active" | "completed" | "paused" | "abandoned";
}

/** content = drive_type + " (" + focus_areas + ")" */
export interface DrivePayload {
  drive_type: "existence" | "growth" | "curiosity" | "connection" | "reproduction";
  intensity: number; // 0..1, lowered by satisfy_drive
  focus_areas?: string[];
}

/** content = entity_name + ": " + notes */
export interface RelationshipPayload {
  entity_name: string;
  entity_type: "person" | "agent" | "organization" | "system" | "other";
  relationship_type?: string;
  notes?: string;
  sentiment: number; // -1..1
  interaction_count: number;
  last_interaction?: string; // ISO-8601 UTC
}

/** content = name + ": " + description */
export interface PlaybookPayload {
  name: string;
  description: string;
  trigger_conditions: string[];
  steps: string[];
  failure_modes?: string[];
  recovery_steps?: string[];
  mastery_level: "novice" | "competent" | "proficient" | "expert";
  times_used: number;
  success_rate: number; // 0..1
  last_used?: string;
}

/** content = entity_name + ": " + observation */
export interface EntityModelPayload {
  entity_name: string;
  model_type: "behavioral" | "preference" | "capability";
  observation: string;
}

/** content = rule; checked before matching tool calls (recall gut check). */
export interface GuardrailPayload {
  rule: string; // "Before X, check Y"
  why?: string;
  triggers: {
    tools?: string[]; // tool names, e.g. "Bash"
    patterns?: string[]; // case-insensitive substrings, or /regex/, matched against the tool input
    paths?: string[]; // glob patterns matched against file paths in the tool input
  };
}

export type MemoryPayload =
  | { kind: "episode"; payload: EpisodePayload }
  | { kind: "note"; payload: NotePayload }
  | { kind: "belief"; payload: BeliefPayload }
  | { kind: "value"; payload: ValuePayload }
  | { kind: "goal"; payload: GoalPayload }
  | { kind: "drive"; payload: DrivePayload }
  | { kind: "relationship"; payload: RelationshipPayload }
  | { kind: "playbook"; payload: PlaybookPayload }
  | { kind: "entity_model"; payload: EntityModelPayload }
  | { kind: "guardrail"; payload: GuardrailPayload };
