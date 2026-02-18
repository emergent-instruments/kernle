"""Golden snapshot end-to-end pipeline regression test.

Catches silent changes in:
  - Memory counts by type (raw, episodes, notes, beliefs, values, goals, relationships, drives)
  - Provenance chain integrity (all derived_from references valid)
  - source_type correctness (all canonical SourceType enum values)
  - Strength tier assignment
  - Inference safety gating (no identity-layer writes without inference)

Variants:
  - inference-on: mock model returns structured JSON responses
  - inference-off: no model, verify safe-mode behavior

HOW TO UPDATE THE GOLDEN SNAPSHOT:
  When processing logic intentionally changes (new transitions, different
  promotion behavior, etc.), run the test, inspect the diff, and update
  the EXPECTED_* constants at the top of this file to match the new
  correct output. Each constant has a comment explaining what it tracks.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Optional

import pytest

from kernle.entity import Entity
from kernle.processing import VALID_TRANSITIONS
from kernle.protocols import ModelResponse
from kernle.stack.sqlite_stack import SQLiteStack
from kernle.types import (
    VALID_SOURCE_TYPE_VALUES,
    SourceType,
)

# =============================================================================
# Deterministic Helpers
# =============================================================================

_UUID_COUNTER = 0


def _deterministic_uuid():
    """Generate predictable UUIDs for deterministic test output."""
    global _UUID_COUNTER
    _UUID_COUNTER += 1
    return f"00000000-0000-0000-0000-{_UUID_COUNTER:012d}"


FIXED_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


# =============================================================================
# Golden Snapshot Expected Values
# =============================================================================

# Expected memory counts after full pipeline run with inference-on.
# Keys: memory type -> expected count after processing.
EXPECTED_COUNTS_INFERENCE_ON = {
    "raw": 12,  # 12 raw entries ingested
    "episodes": 3,  # mock model produces 3 episodes from raw_to_episode
    "notes": 2,  # mock model produces 2 notes from raw_to_note
    "beliefs": 0,  # blocked by promotion gate (evidence < belief_min_evidence=3)
    "values": 0,  # blocked: no beliefs exist to promote to values
    "goals": 1,  # gate-blocked episodes stay unprocessed, available for goal promotion
    "relationships": 0,  # episodes consumed by episode_to_goal (none left)
    "drives": 0,  # episodes consumed by episode_to_goal (none left)
}

# Expected memory counts when inference is off (no model bound).
# Identity-layer transitions are blocked; raw-layer transitions fail at
# inference call level (no model to call), so nothing is created.
EXPECTED_COUNTS_INFERENCE_OFF = {
    "raw": 12,
    "episodes": 0,
    "notes": 0,
    "beliefs": 0,
    "values": 0,
    "goals": 0,
    "relationships": 0,
    "drives": 0,
}

# Expected transitions that produce results with inference-on.
EXPECTED_TRANSITIONS_WITH_RESULTS = {
    "raw_to_episode",
    "raw_to_note",
    "episode_to_belief",
    "episode_to_goal",
    "episode_to_relationship",
    "episode_to_drive",
}

# Expected transitions blocked without inference (all transitions).
EXPECTED_BLOCKED_TRANSITIONS = set(VALID_TRANSITIONS)


# =============================================================================
# Mock Model
# =============================================================================


class DeterministicMockModel:
    """Mock ModelProtocol that returns deterministic JSON responses.

    Each transition gets a fixed, valid JSON response that the
    MemoryProcessor can parse and write to the stack.
    """

    def __init__(self):
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return "mock-deterministic-v1"

    @property
    def capabilities(self):
        from kernle.protocols import ModelCapabilities

        return ModelCapabilities(streaming=False, tool_use=False, vision=False)

    def generate(
        self,
        messages: list,
        *,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> ModelResponse:
        """Return deterministic responses based on the system prompt."""
        self._call_count += 1
        prompt_text = messages[0].content if messages else ""

        if system and "episodic memories" in system:
            return self._episode_response(prompt_text)
        elif system and "factual notes" in system:
            return self._note_response(prompt_text)
        elif system and "beliefs" in system and "emerge from experiences" in system:
            return self._belief_response(prompt_text)
        elif system and "goals" in system:
            return self._goal_response(prompt_text)
        elif system and "relationships" in system:
            return self._relationship_response(prompt_text)
        elif system and "core values" in system:
            return self._value_response(prompt_text)
        elif system and "drives" in system or system and "motivational" in system:
            return self._drive_response(prompt_text)
        else:
            return ModelResponse(content="[]")

    def _episode_response(self, prompt: str) -> ModelResponse:
        """Extract raw IDs from prompt and produce episodes referencing them."""
        raw_ids = self._extract_ids(prompt)
        episodes = []
        # Group into batches of 4 (3 episodes from 12 raw entries)
        for i in range(0, len(raw_ids), 4):
            batch = raw_ids[i : i + 4]
            if not batch:
                break
            episodes.append(
                {
                    "objective": f"Completed task batch {i // 4 + 1}",
                    "outcome": f"Successfully processed {len(batch)} items",
                    "outcome_type": "success",
                    "lessons": [f"Lesson from batch {i // 4 + 1}"],
                    "source_raw_ids": batch,
                }
            )
        return ModelResponse(content=json.dumps(episodes))

    def _note_response(self, prompt: str) -> ModelResponse:
        raw_ids = self._extract_ids(prompt)
        notes = []
        # Produce 2 notes from the first 2 raw IDs
        for i, rid in enumerate(raw_ids[:2]):
            notes.append(
                {
                    "content": f"Factual observation {i + 1} from raw data",
                    "note_type": "observation",
                    "source_raw_ids": [rid],
                }
            )
        return ModelResponse(content=json.dumps(notes))

    def _belief_response(self, prompt: str) -> ModelResponse:
        ep_ids = self._extract_ids(prompt)
        if not ep_ids:
            return ModelResponse(content="[]")
        beliefs = [
            {
                "statement": "Systematic processing leads to better outcomes",
                "belief_type": "causal",
                "confidence": 0.75,
                "source_episode_ids": ep_ids[:2],
            }
        ]
        return ModelResponse(content=json.dumps(beliefs))

    def _goal_response(self, prompt: str) -> ModelResponse:
        ep_ids = self._extract_ids(prompt)
        if not ep_ids:
            return ModelResponse(content="[]")
        goals = [
            {
                "title": "Improve processing efficiency",
                "description": "Optimize the batch processing pipeline",
                "goal_type": "task",
                "priority": "medium",
                "source_episode_ids": ep_ids[:1],
            }
        ]
        return ModelResponse(content=json.dumps(goals))

    def _relationship_response(self, prompt: str) -> ModelResponse:
        ep_ids = self._extract_ids(prompt)
        if not ep_ids:
            return ModelResponse(content="[]")
        rels = [
            {
                "entity_name": "processing-system",
                "entity_type": "system",
                "sentiment": 0.5,
                "context_note": "Reliable automated system",
                "source_episode_ids": ep_ids[:1],
            }
        ]
        return ModelResponse(content=json.dumps(rels))

    def _value_response(self, prompt: str) -> ModelResponse:
        belief_ids = self._extract_ids(prompt)
        if not belief_ids:
            return ModelResponse(content="[]")
        values = [
            {
                "name": "Systematic Approach",
                "statement": "Systematic approaches yield better results",
                "priority": 75,
                "source_belief_ids": belief_ids[:1],
            }
        ]
        return ModelResponse(content=json.dumps(values))

    def _drive_response(self, prompt: str) -> ModelResponse:
        ep_ids = self._extract_ids(prompt)
        if not ep_ids:
            return ModelResponse(content="[]")
        drives = [
            {
                "drive_type": "efficiency",
                "intensity": 0.7,
                "source_episode_ids": ep_ids[:1],
            }
        ]
        return ModelResponse(content=json.dumps(drives))

    @staticmethod
    def _extract_ids(prompt: str) -> list:
        """Extract bracketed IDs like [abc-123] from prompt text."""
        import re

        return re.findall(r"\[([a-f0-9-]+)\]", prompt)


# =============================================================================
# Fixed Raw Inputs
# =============================================================================

RAW_INPUTS = [
    "Debugged the authentication module. Found the root cause in token refresh logic.",
    "Team standup: discussed priorities for the sprint. Alice is handling frontend.",
    "Deployed v2.3.1 to staging. All smoke tests passed on first attempt.",
    "Reviewed PR #445 from Bob. Good code quality, minor style issues noted.",
    "Infrastructure alert resolved: disk usage was at 92%, cleaned up old logs.",
    "Customer feedback: users love the new search feature but want better filters.",
    "Refactored the database connection pooling. Reduced idle connections by 40%.",
    "Wrote documentation for the new API endpoints. Added example requests.",
    "Mentored junior developer on testing best practices. Covered mocking strategies.",
    "Performance optimization: query time reduced from 2.1s to 0.3s with proper indexing.",
    "Security audit completed: no critical vulnerabilities found. Two medium risks addressed.",
    "Released version 3.0 to production. Zero downtime deployment achieved.",
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def stack(tmp_path):
    """Create a bare SQLiteStack (no default components) for test isolation."""
    db_path = tmp_path / "golden_test.db"
    s = SQLiteStack(
        stack_id="golden-test",
        db_path=db_path,
        components=[],  # bare stack, no auto-components
        enforce_provenance=False,  # disable during seed phase
    )
    yield s


@pytest.fixture
def entity_with_stack(stack):
    """Create an Entity with an attached stack in INITIALIZING state."""
    entity = Entity(core_id="golden-test-core")
    entity.attach_stack(stack, alias="golden")
    return entity, stack


@pytest.fixture
def entity_with_model(entity_with_stack):
    """Entity with a deterministic mock model bound."""
    entity, stack = entity_with_stack
    model = DeterministicMockModel()
    entity.set_model(model)
    return entity, stack, model


# =============================================================================
# Helper Functions
# =============================================================================


def ingest_raw_entries(entity: Entity) -> list:
    """Ingest all fixed raw inputs and return their IDs."""
    ids = []
    for content in RAW_INPUTS:
        raw_id = entity.raw(content)
        ids.append(raw_id)
    return ids


def get_memory_counts(stack: SQLiteStack) -> Dict[str, int]:
    """Get counts of all memory types in the stack."""
    backend = stack._backend
    return {
        "raw": len(backend.list_raw(limit=1000)),
        "episodes": len(stack.get_episodes(limit=1000, include_forgotten=True)),
        "notes": len(stack.get_notes(limit=1000, include_forgotten=True)),
        "beliefs": len(stack.get_beliefs(limit=1000, include_forgotten=True)),
        "values": len(stack.get_values(limit=1000, include_forgotten=True)),
        "goals": len(stack.get_goals(limit=1000, include_forgotten=True)),
        "relationships": len(stack.get_relationships()),
        "drives": len(stack.get_drives()),
    }


def collect_all_derived_from(stack: SQLiteStack) -> Dict[str, list]:
    """Collect all derived_from refs across all memory types."""
    refs: Dict[str, list] = {}

    for ep in stack.get_episodes(limit=1000, include_forgotten=True):
        if ep.derived_from:
            refs[f"episode:{ep.id}"] = ep.derived_from

    for note in stack.get_notes(limit=1000, include_forgotten=True):
        if note.derived_from:
            refs[f"note:{note.id}"] = note.derived_from

    for belief in stack.get_beliefs(limit=1000, include_forgotten=True):
        if belief.derived_from:
            refs[f"belief:{belief.id}"] = belief.derived_from

    for value in stack.get_values(limit=1000, include_forgotten=True):
        if value.derived_from:
            refs[f"value:{value.id}"] = value.derived_from

    for goal in stack.get_goals(limit=1000, include_forgotten=True):
        if goal.derived_from:
            refs[f"goal:{goal.id}"] = goal.derived_from

    for rel in stack.get_relationships():
        if rel.derived_from:
            refs[f"relationship:{rel.id}"] = rel.derived_from

    for drive in stack.get_drives():
        if drive.derived_from:
            refs[f"drive:{drive.id}"] = drive.derived_from

    return refs


def collect_all_source_types(stack: SQLiteStack) -> Dict[str, str]:
    """Collect source_type from all memories."""
    types: Dict[str, str] = {}

    for ep in stack.get_episodes(limit=1000, include_forgotten=True):
        types[f"episode:{ep.id}"] = ep.source_type

    for note in stack.get_notes(limit=1000, include_forgotten=True):
        types[f"note:{note.id}"] = note.source_type

    for belief in stack.get_beliefs(limit=1000, include_forgotten=True):
        types[f"belief:{belief.id}"] = belief.source_type

    for value in stack.get_values(limit=1000, include_forgotten=True):
        types[f"value:{value.id}"] = value.source_type

    for goal in stack.get_goals(limit=1000, include_forgotten=True):
        types[f"goal:{goal.id}"] = goal.source_type

    for rel in stack.get_relationships():
        types[f"relationship:{rel.id}"] = rel.source_type

    for drive in stack.get_drives():
        types[f"drive:{drive.id}"] = drive.source_type

    return types


def collect_all_strengths(stack: SQLiteStack) -> Dict[str, float]:
    """Collect strength values from all memories."""
    strengths: Dict[str, float] = {}

    for ep in stack.get_episodes(limit=1000, include_forgotten=True):
        strengths[f"episode:{ep.id}"] = ep.strength

    for note in stack.get_notes(limit=1000, include_forgotten=True):
        strengths[f"note:{note.id}"] = note.strength

    for belief in stack.get_beliefs(limit=1000, include_forgotten=True):
        strengths[f"belief:{belief.id}"] = belief.strength

    for value in stack.get_values(limit=1000, include_forgotten=True):
        strengths[f"value:{value.id}"] = value.strength

    for goal in stack.get_goals(limit=1000, include_forgotten=True):
        strengths[f"goal:{goal.id}"] = goal.strength

    for rel in stack.get_relationships():
        strengths[f"relationship:{rel.id}"] = rel.strength

    for drive in stack.get_drives():
        strengths[f"drive:{drive.id}"] = drive.strength

    return strengths


def get_existing_memory_ids(stack: SQLiteStack) -> set:
    """Get all existing memory IDs as 'type:id' refs."""
    ids = set()
    backend = stack._backend

    for raw in backend.list_raw(limit=1000):
        ids.add(f"raw:{raw.id}")

    for ep in stack.get_episodes(limit=1000, include_forgotten=True):
        ids.add(f"episode:{ep.id}")

    for note in stack.get_notes(limit=1000, include_forgotten=True):
        ids.add(f"note:{note.id}")

    for belief in stack.get_beliefs(limit=1000, include_forgotten=True):
        ids.add(f"belief:{belief.id}")

    for value in stack.get_values(limit=1000, include_forgotten=True):
        ids.add(f"value:{value.id}")

    for goal in stack.get_goals(limit=1000, include_forgotten=True):
        ids.add(f"goal:{goal.id}")

    for rel in stack.get_relationships():
        ids.add(f"relationship:{rel.id}")

    for drive in stack.get_drives():
        ids.add(f"drive:{drive.id}")

    return ids


# =============================================================================
# VARIANT 1: Inference-On (Mock Model)
# =============================================================================


class TestPipelineGoldenInferenceOn:
    """Full pipeline test with a deterministic mock model.

    Ingests 12 raw entries, runs all processing transitions with
    force=True and auto_promote=True, then asserts exact counts,
    provenance integrity, source_type correctness, and strength tiers.
    """

    def _run_pipeline(self, entity, stack):
        """Run the full pipeline: ingest + process all transitions in order."""
        raw_ids = ingest_raw_entries(entity)

        # Process raw -> episode and raw -> note first
        results_raw = entity.process(
            transition="raw_to_episode",
            force=True,
            auto_promote=True,
        )

        results_note = entity.process(
            transition="raw_to_note",
            force=True,
            auto_promote=True,
        )

        # Now process episode -> belief
        results_belief = entity.process(
            transition="episode_to_belief",
            force=True,
            auto_promote=True,
        )

        # episode -> goal
        results_goal = entity.process(
            transition="episode_to_goal",
            force=True,
            auto_promote=True,
        )

        # episode -> relationship
        results_rel = entity.process(
            transition="episode_to_relationship",
            force=True,
            auto_promote=True,
        )

        # episode -> drive
        results_drive = entity.process(
            transition="episode_to_drive",
            force=True,
            auto_promote=True,
        )

        # belief -> value (may be blocked by quantity threshold or lack of beliefs)
        results_value = entity.process(
            transition="belief_to_value",
            force=True,
            auto_promote=True,
        )

        all_results = (
            results_raw
            + results_note
            + results_belief
            + results_goal
            + results_rel
            + results_drive
            + results_value
        )
        return raw_ids, all_results

    def test_memory_counts_match_golden(self, entity_with_model):
        """Assert exact memory counts match the golden snapshot."""
        entity, stack, model = entity_with_model
        raw_ids, results = self._run_pipeline(entity, stack)

        actual_counts = get_memory_counts(stack)
        for mem_type, expected in EXPECTED_COUNTS_INFERENCE_ON.items():
            actual = actual_counts[mem_type]
            assert actual == expected, (
                f"Memory count mismatch for '{mem_type}': "
                f"expected {expected}, got {actual}. "
                f"Full counts: {json.dumps(actual_counts, indent=2)}"
            )

    def test_provenance_chain_integrity(self, entity_with_model):
        """Every derived_from reference must point to an existing memory."""
        entity, stack, model = entity_with_model
        self._run_pipeline(entity, stack)

        all_refs = collect_all_derived_from(stack)
        existing_ids = get_existing_memory_ids(stack)

        broken_refs = []
        for memory_ref, derived_list in all_refs.items():
            for ref in derived_list:
                if ref not in existing_ids:
                    broken_refs.append(
                        {
                            "memory": memory_ref,
                            "broken_ref": ref,
                        }
                    )

        assert broken_refs == [], "Broken provenance references found:\n" + "\n".join(
            f"  {b['memory']} -> {b['broken_ref']}" for b in broken_refs
        )

    def test_all_derived_from_are_present(self, entity_with_model):
        """All processed memories (except raw) should have derived_from set."""
        entity, stack, model = entity_with_model
        self._run_pipeline(entity, stack)

        missing_provenance = []

        for ep in stack.get_episodes(limit=1000, include_forgotten=True):
            if not ep.derived_from:
                missing_provenance.append(f"episode:{ep.id}")

        for note in stack.get_notes(limit=1000, include_forgotten=True):
            if not note.derived_from:
                missing_provenance.append(f"note:{note.id}")

        for belief in stack.get_beliefs(limit=1000, include_forgotten=True):
            if not belief.derived_from:
                missing_provenance.append(f"belief:{belief.id}")

        for goal in stack.get_goals(limit=1000, include_forgotten=True):
            if not goal.derived_from:
                missing_provenance.append(f"goal:{goal.id}")

        for rel in stack.get_relationships():
            if not rel.derived_from:
                missing_provenance.append(f"relationship:{rel.id}")

        for drive in stack.get_drives():
            if not drive.derived_from:
                missing_provenance.append(f"drive:{drive.id}")

        assert missing_provenance == [], "Memories missing derived_from provenance:\n" + "\n".join(
            f"  {m}" for m in missing_provenance
        )

    def test_source_type_correctness(self, entity_with_model):
        """All source_type values must be valid SourceType enum values."""
        entity, stack, model = entity_with_model
        self._run_pipeline(entity, stack)

        all_types = collect_all_source_types(stack)
        invalid = []
        for memory_ref, source_type in all_types.items():
            if source_type not in VALID_SOURCE_TYPE_VALUES:
                invalid.append(
                    {
                        "memory": memory_ref,
                        "source_type": source_type,
                    }
                )

        assert invalid == [], (
            "Invalid source_type values found:\n"
            + "\n".join(f"  {i['memory']}: '{i['source_type']}'" for i in invalid)
            + f"\nValid values: {sorted(VALID_SOURCE_TYPE_VALUES)}"
        )

    def test_processing_source_type(self, entity_with_model):
        """All memories created by processing should have source_type='processing'."""
        entity, stack, model = entity_with_model
        self._run_pipeline(entity, stack)

        all_types = collect_all_source_types(stack)
        non_processing = []
        for memory_ref, source_type in all_types.items():
            if source_type != SourceType.PROCESSING.value:
                non_processing.append(
                    {
                        "memory": memory_ref,
                        "source_type": source_type,
                    }
                )

        assert (
            non_processing == []
        ), "Memories created by processing should have source_type='processing':\n" + "\n".join(
            f"  {i['memory']}: '{i['source_type']}'" for i in non_processing
        )

    def test_strength_tier_assignment(self, entity_with_model):
        """All newly created memories should start at strength 1.0."""
        entity, stack, model = entity_with_model
        self._run_pipeline(entity, stack)

        strengths = collect_all_strengths(stack)
        non_full_strength = []
        for memory_ref, strength in strengths.items():
            if strength != 1.0:
                non_full_strength.append(
                    {
                        "memory": memory_ref,
                        "strength": strength,
                    }
                )

        assert (
            non_full_strength == []
        ), "Newly created memories should start at strength 1.0:\n" + "\n".join(
            f"  {i['memory']}: {i['strength']}" for i in non_full_strength
        )

    def test_processing_results_structure(self, entity_with_model):
        """Processing results should contain expected transitions."""
        entity, stack, model = entity_with_model
        raw_ids, results = self._run_pipeline(entity, stack)

        # Filter to results that actually ran (not skipped)
        ran_transitions = {
            r.layer_transition for r in results if not r.skipped and r.source_count > 0
        }

        # At minimum, raw_to_episode and raw_to_note should run
        assert (
            "raw_to_episode" in ran_transitions
        ), f"raw_to_episode should have run. Got transitions: {ran_transitions}"
        assert (
            "raw_to_note" in ran_transitions
        ), f"raw_to_note should have run. Got transitions: {ran_transitions}"

    def test_raw_entries_marked_processed(self, entity_with_model):
        """After processing, raw entries should be marked as processed."""
        entity, stack, model = entity_with_model
        self._run_pipeline(entity, stack)

        backend = stack._backend
        unprocessed = backend.list_raw(processed=False, limit=1000)
        all_raw = backend.list_raw(limit=1000)

        # All raw entries should be marked processed (both raw_to_episode
        # and raw_to_note process them)
        assert len(unprocessed) == 0, (
            f"Expected 0 unprocessed raw entries after pipeline, "
            f"got {len(unprocessed)} out of {len(all_raw)} total"
        )

    def test_episodes_marked_processed(self, entity_with_model):
        """After episode->belief/goal/relationship/drive processing,
        episodes should be marked as processed."""
        entity, stack, model = entity_with_model
        self._run_pipeline(entity, stack)

        episodes = stack.get_episodes(limit=1000, include_forgotten=True)
        unprocessed = [ep for ep in episodes if not ep.processed]

        assert len(unprocessed) == 0, (
            f"Expected 0 unprocessed episodes after pipeline, "
            f"got {len(unprocessed)} out of {len(episodes)} total"
        )

    def test_no_duplicate_memories_on_reprocess(self, entity_with_model):
        """Running processing a second time should not create duplicates."""
        entity, stack, model = entity_with_model
        self._run_pipeline(entity, stack)

        counts_first = get_memory_counts(stack)

        # Run processing again — should be idempotent
        entity.process(force=True, auto_promote=True)

        counts_second = get_memory_counts(stack)

        for mem_type in counts_first:
            assert counts_first[mem_type] == counts_second[mem_type], (
                f"Duplicate memories created for '{mem_type}' on reprocess: "
                f"first run={counts_first[mem_type]}, "
                f"second run={counts_second[mem_type]}"
            )

    def test_auto_promote_writes_directly(self, entity_with_model):
        """With auto_promote=True, memories should be written directly
        (not as suggestions)."""
        entity, stack, model = entity_with_model
        raw_ids, results = self._run_pipeline(entity, stack)

        for r in results:
            if r.skipped:
                continue
            assert r.auto_promote is True, (
                f"Result for {r.layer_transition} should have auto_promote=True, "
                f"got {r.auto_promote}"
            )
            # Suggestions should be empty when auto_promote is True
            assert len(r.suggestions) == 0, (
                f"Result for {r.layer_transition} should have 0 suggestions with "
                f"auto_promote=True, got {len(r.suggestions)}"
            )


# =============================================================================
# VARIANT 2: Inference-Off (No Model)
# =============================================================================


class TestPipelineGoldenInferenceOff:
    """Pipeline test without a model bound.

    Verifies the no-inference safety policy:
    - Identity-layer transitions are blocked
    - Raw-layer transitions fail gracefully (no model to call)
    - No memories are written to identity layers
    """

    def _run_pipeline_no_model(self, entity, stack):
        """Ingest raw entries and attempt to process without a model."""
        raw_ids = ingest_raw_entries(entity)

        results = entity.process(
            force=True,
            auto_promote=True,
        )
        return raw_ids, results

    def test_memory_counts_no_inference(self, entity_with_stack):
        """Without a model, no structured memories should be created."""
        entity, stack = entity_with_stack
        raw_ids, results = self._run_pipeline_no_model(entity, stack)

        actual_counts = get_memory_counts(stack)

        # Verify raw entries were ingested
        assert actual_counts["raw"] == EXPECTED_COUNTS_INFERENCE_OFF["raw"], (
            f"Raw count mismatch: expected {EXPECTED_COUNTS_INFERENCE_OFF['raw']}, "
            f"got {actual_counts['raw']}"
        )

        # Verify no identity-layer memories were created
        for mem_type in ("beliefs", "values", "goals", "relationships", "drives"):
            assert actual_counts[mem_type] == 0, (
                f"Identity-layer '{mem_type}' should have 0 memories without "
                f"inference, got {actual_counts[mem_type]}"
            )

    def test_identity_transitions_blocked(self, entity_with_stack):
        """Identity-layer transitions should report inference_blocked."""
        entity, stack = entity_with_stack
        raw_ids, results = self._run_pipeline_no_model(entity, stack)

        blocked_results = [r for r in results if r.inference_blocked]
        blocked_transitions = {r.layer_transition for r in blocked_results}

        assert blocked_transitions == EXPECTED_BLOCKED_TRANSITIONS, (
            f"Expected blocked transitions: {sorted(EXPECTED_BLOCKED_TRANSITIONS)}\n"
            f"Got blocked transitions: {sorted(blocked_transitions)}"
        )

    def test_blocked_results_have_skip_reason(self, entity_with_stack):
        """Blocked results should include a human-readable skip reason."""
        entity, stack = entity_with_stack
        raw_ids, results = self._run_pipeline_no_model(entity, stack)

        for r in results:
            if r.inference_blocked:
                assert (
                    r.skip_reason is not None
                ), f"Blocked result for {r.layer_transition} missing skip_reason"
                assert "inference" in r.skip_reason.lower(), (
                    f"Skip reason for {r.layer_transition} should mention inference: "
                    f"'{r.skip_reason}'"
                )

    def test_no_identity_layer_writes(self, entity_with_stack):
        """Even with force=True, no identity-layer memories should exist."""
        entity, stack = entity_with_stack
        self._run_pipeline_no_model(entity, stack)

        all_types = collect_all_source_types(stack)
        for memory_ref in all_types:
            mem_type = memory_ref.split(":")[0]
            assert mem_type not in (
                "belief",
                "value",
                "goal",
                "drive",
            ), f"Identity-layer memory {memory_ref} should not exist without inference"

    def test_belief_to_value_always_blocked(self, entity_with_stack):
        """belief_to_value should always be blocked without inference."""
        entity, stack = entity_with_stack
        raw_ids, results = self._run_pipeline_no_model(entity, stack)

        value_results = [r for r in results if r.layer_transition == "belief_to_value"]
        for r in value_results:
            assert (
                r.inference_blocked is True
            ), "belief_to_value should always be blocked without inference"
            assert (
                "cannot promote to identity layer" in r.skip_reason.lower()
                or "value creation requires inference" in r.skip_reason.lower()
                or "inference unavailable" in r.skip_reason.lower()
            ), (f"Skip reason should mention inference requirement: " f"'{r.skip_reason}'")


# =============================================================================
# VARIANT 3: Provenance Hierarchy Validation
# =============================================================================


class TestPipelineProvenanceHierarchy:
    """Verify that provenance follows the correct hierarchy:
    raw -> episode/note -> belief -> value
    raw -> episode -> goal, relationship, drive
    """

    def test_episode_derived_from_raw(self, entity_with_model):
        """Episodes should only reference raw entries in derived_from."""
        entity, stack, model = entity_with_model
        ingest_raw_entries(entity)
        entity.process(transition="raw_to_episode", force=True, auto_promote=True)

        for ep in stack.get_episodes(limit=1000, include_forgotten=True):
            for ref in ep.derived_from or []:
                ref_type = ref.split(":")[0]
                assert ref_type == "raw", f"Episode {ep.id} has non-raw derived_from ref: {ref}"

    def test_note_derived_from_raw(self, entity_with_model):
        """Notes should only reference raw entries in derived_from."""
        entity, stack, model = entity_with_model
        ingest_raw_entries(entity)
        entity.process(transition="raw_to_note", force=True, auto_promote=True)

        for note in stack.get_notes(limit=1000, include_forgotten=True):
            for ref in note.derived_from or []:
                ref_type = ref.split(":")[0]
                assert ref_type == "raw", f"Note {note.id} has non-raw derived_from ref: {ref}"

    def test_belief_derived_from_episode(self, entity_with_model):
        """Beliefs should reference episodes in derived_from."""
        entity, stack, model = entity_with_model
        ingest_raw_entries(entity)
        entity.process(transition="raw_to_episode", force=True, auto_promote=True)
        entity.process(transition="episode_to_belief", force=True, auto_promote=True)

        for belief in stack.get_beliefs(limit=1000, include_forgotten=True):
            for ref in belief.derived_from or []:
                ref_type = ref.split(":")[0]
                assert (
                    ref_type == "episode"
                ), f"Belief {belief.id} has non-episode derived_from ref: {ref}"

    def test_goal_derived_from_episode(self, entity_with_model):
        """Goals should reference episodes in derived_from."""
        entity, stack, model = entity_with_model
        ingest_raw_entries(entity)
        entity.process(transition="raw_to_episode", force=True, auto_promote=True)
        entity.process(transition="episode_to_goal", force=True, auto_promote=True)

        for goal in stack.get_goals(limit=1000, include_forgotten=True):
            for ref in goal.derived_from or []:
                ref_type = ref.split(":")[0]
                assert (
                    ref_type == "episode"
                ), f"Goal {goal.id} has non-episode derived_from ref: {ref}"

    def test_relationship_derived_from_episode(self, entity_with_model):
        """Relationships should reference episodes in derived_from."""
        entity, stack, model = entity_with_model
        ingest_raw_entries(entity)
        entity.process(transition="raw_to_episode", force=True, auto_promote=True)
        entity.process(transition="episode_to_relationship", force=True, auto_promote=True)

        for rel in stack.get_relationships():
            for ref in rel.derived_from or []:
                ref_type = ref.split(":")[0]
                assert (
                    ref_type == "episode"
                ), f"Relationship {rel.id} has non-episode derived_from ref: {ref}"

    def test_drive_derived_from_episode(self, entity_with_model):
        """Drives should reference episodes in derived_from."""
        entity, stack, model = entity_with_model
        ingest_raw_entries(entity)
        entity.process(transition="raw_to_episode", force=True, auto_promote=True)
        entity.process(transition="episode_to_drive", force=True, auto_promote=True)

        for drive in stack.get_drives():
            for ref in drive.derived_from or []:
                ref_type = ref.split(":")[0]
                assert (
                    ref_type == "episode"
                ), f"Drive {drive.id} has non-episode derived_from ref: {ref}"


# =============================================================================
# VARIANT 4: Suggestion Mode (auto_promote=False)
# =============================================================================


class TestPipelineSuggestionMode:
    """Verify that without auto_promote, memories go to suggestions."""

    def test_suggestions_created_not_memories(self, entity_with_model):
        """With auto_promote=False, processing should create suggestions
        instead of direct memories."""
        entity, stack, model = entity_with_model
        ingest_raw_entries(entity)

        results = entity.process(
            transition="raw_to_episode",
            force=True,
            auto_promote=False,
        )

        # Should have suggestions, not created memories
        for r in results:
            if r.skipped:
                continue
            assert (
                len(r.suggestions) > 0
            ), f"Expected suggestions for {r.layer_transition}, got none"
            assert len(r.created) == 0, (
                f"Expected no direct creates for {r.layer_transition} "
                f"when auto_promote=False, got {len(r.created)}"
            )

        # Verify no episodes were directly created
        episodes = stack.get_episodes(limit=1000, include_forgotten=True)
        assert (
            len(episodes) == 0
        ), f"Expected 0 episodes with auto_promote=False, got {len(episodes)}"
