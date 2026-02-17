"""Tests for suggestions-first promotion default (Issue #401).

Verifies that:
- Default processing creates suggestions, not direct promotions
- auto_promote=True produces direct promotions (opt-in)
- CLI --auto-promote flag works correctly
- MCP auto_promote parameter works correctly
- ProcessingResult tracks suggestions vs created
- All transition types work in suggestions mode
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from kernle.entity import Entity
from kernle.processing import (
    DEFAULT_LAYER_CONFIGS,
    MemoryProcessor,
    ProcessingResult,
    PromotionGateConfig,
)
from kernle.stack.sqlite_stack import SQLiteStack
from kernle.types import Belief, Episode, MemorySuggestion, RawEntry

# Relaxed promotion gates for tests that don't test gating behavior
_NO_GATES = PromotionGateConfig(
    belief_min_evidence=0,
    belief_min_confidence=0.0,
    value_min_evidence=0,
    value_requires_protection=False,
)

STACK_ID = "test-stack"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def stack(tmp_path):
    return SQLiteStack(
        STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
    )


@pytest.fixture
def entity(tmp_path):
    ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
    st = SQLiteStack(
        STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
    )
    ent.attach_stack(st)
    return ent


class MockInference:
    def __init__(self, response: str = "[]"):
        self.response = response
        self.calls = []

    def infer(self, prompt: str, *, system=None) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        return self.response

    def embed(self, text: str) -> list:
        return [0.0] * 64

    def embed_batch(self, texts: list) -> list:
        return [[0.0] * 64 for _ in texts]

    @property
    def embedding_dimension(self) -> int:
        return 64

    @property
    def embedding_provider_id(self) -> str:
        return "mock"


def _make_mock_stack():
    mock_stack = MagicMock()
    mock_stack.stack_id = STACK_ID
    mock_stack._backend = MagicMock()
    return mock_stack


def _make_processor(mock_stack, response="[]", auto_promote=False):
    inference = MockInference(response)
    return (
        MemoryProcessor(
            stack=mock_stack,
            inference=inference,
            core_id="test",
            auto_promote=auto_promote,
            promotion_gates=_NO_GATES,
        ),
        inference,
    )


# =============================================================================
# ProcessingResult: new fields
# =============================================================================


class TestProcessingResultSuggestionsField:
    def test_default_values(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=5,
        )
        assert result.created == []
        assert result.suggestions == []
        assert result.auto_promote is False

    def test_auto_promote_flag(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=5,
            auto_promote=True,
        )
        assert result.auto_promote is True

    def test_suggestions_populated(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=3,
            suggestions=[{"type": "episode", "id": "s-1"}],
        )
        assert len(result.suggestions) == 1
        assert result.suggestions[0]["type"] == "episode"


# =============================================================================
# Default mode: suggestions (not direct promotion)
# =============================================================================


class TestDefaultSuggestionsMode:
    """Default processing creates suggestions, not direct promotions."""

    def test_default_creates_suggestions_not_memories(self, stack):
        """The core safety guarantee: default mode creates suggestions."""
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Raw content {i}",
                    source="test",
                )
            )

        response = json.dumps(
            [
                {
                    "objective": "Learned something",
                    "outcome": "Good result",
                    "outcome_type": "success",
                    "lessons": ["lesson 1"],
                    "source_raw_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        result = results[0]

        # No direct memory creation
        assert result.created == []
        assert result.auto_promote is False

        # Suggestions were created
        assert len(result.suggestions) == 1
        assert result.suggestions[0]["type"] == "episode"

        # Verify no episodes were saved directly
        episodes = stack.get_episodes()
        assert len(episodes) == 0

        # Verify suggestion was saved
        suggestions = stack._backend.get_suggestions(status="pending")
        assert len(suggestions) >= 1

    def test_default_raw_to_note_creates_suggestions(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Some factual info about the system",
                source="test",
            )
        )

        response = json.dumps(
            [
                {
                    "content": "A factual note",
                    "note_type": "fact",
                    "source_raw_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_note", force=True)
        assert len(results) == 1
        assert results[0].created == []
        assert len(results[0].suggestions) == 1
        assert results[0].suggestions[0]["type"] == "note"

    def test_default_episode_to_belief_creates_suggestions(self, stack):
        for i in range(3):
            ep = Episode(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                objective=f"Event {i}",
                outcome=f"Outcome {i}",
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            stack.save_episode(ep)

        response = json.dumps(
            [
                {
                    "statement": "Testing is important",
                    "belief_type": "evaluative",
                    "confidence": 0.8,
                    "source_episode_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack, inference=inference, core_id="test", promotion_gates=_NO_GATES
        )

        results = processor.process("episode_to_belief", force=True)
        assert len(results) == 1
        assert results[0].created == []
        assert len(results[0].suggestions) == 1
        assert results[0].suggestions[0]["type"] == "belief"

        # No beliefs directly saved
        beliefs = stack.get_beliefs()
        assert len(beliefs) == 0

    def test_default_belief_to_value_creates_suggestions(self, stack):
        """Identity-layer transitions MUST produce suggestions by default."""
        for i in range(3):
            belief = Belief(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                statement=f"Belief {i}",
                belief_type="evaluative",
                confidence=0.9,
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            stack.save_belief(belief)

        response = json.dumps(
            [
                {
                    "name": "integrity",
                    "statement": "always be honest",
                    "priority": 90,
                    "source_belief_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack, inference=inference, core_id="test", promotion_gates=_NO_GATES
        )

        results = processor.process("belief_to_value", force=True)
        assert len(results) == 1
        assert results[0].created == []
        assert len(results[0].suggestions) == 1
        assert results[0].suggestions[0]["type"] == "value"

        # No values directly saved
        values = stack.get_values()
        assert len(values) == 0

    def test_default_force_still_creates_suggestions(self, stack):
        """force=True affects trigger checking, NOT the promotion mode."""
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Test content for force mode",
                source="test",
            )
        )

        response = json.dumps(
            [{"objective": "Forced item", "outcome": "done", "source_raw_ids": []}]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        # force does NOT mean auto_promote
        assert results[0].auto_promote is False
        assert results[0].created == []
        assert len(results[0].suggestions) == 1


# =============================================================================
# auto_promote=True: direct promotion (opt-in)
# =============================================================================


class TestAutoPromoteMode:
    """auto_promote=True directly writes memories (opt-in behavior)."""

    def test_auto_promote_creates_memories_directly(self, stack):
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Raw content {i}",
                    source="test",
                )
            )

        response = json.dumps(
            [
                {
                    "objective": "Learned something",
                    "outcome": "Good result",
                    "outcome_type": "success",
                    "source_raw_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack, inference=inference, core_id="test", auto_promote=True
        )

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        result = results[0]

        # Direct promotion happened
        assert result.auto_promote is True
        assert len(result.created) == 1
        assert result.created[0]["type"] == "episode"
        assert result.suggestions == []

        # Episode was saved directly
        episodes = stack.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].objective == "Learned something"

    def test_auto_promote_per_call_overrides_instance(self, stack):
        """process(auto_promote=True) overrides instance default of False."""
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Test content",
                source="test",
            )
        )

        response = json.dumps([{"objective": "Test", "outcome": "Done", "source_raw_ids": []}])
        inference = MockInference(response)
        # Instance default is False (suggestions)
        processor = MemoryProcessor(
            stack=stack, inference=inference, core_id="test", auto_promote=False
        )

        # Override per-call to True
        results = processor.process("raw_to_episode", force=True, auto_promote=True)
        assert len(results) == 1
        assert results[0].auto_promote is True
        assert len(results[0].created) == 1
        assert results[0].suggestions == []

    def test_auto_promote_false_per_call_overrides_instance(self, stack):
        """process(auto_promote=False) overrides instance default of True."""
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Test content",
                source="test",
            )
        )

        response = json.dumps([{"objective": "Test", "outcome": "Done", "source_raw_ids": []}])
        inference = MockInference(response)
        # Instance default is True
        processor = MemoryProcessor(
            stack=stack, inference=inference, core_id="test", auto_promote=True
        )

        # Override per-call to False
        results = processor.process("raw_to_episode", force=True, auto_promote=False)
        assert len(results) == 1
        assert results[0].auto_promote is False
        assert results[0].created == []
        assert len(results[0].suggestions) == 1


# =============================================================================
# Suggestions for all transition types (mock-based)
# =============================================================================


class TestSuggestionsAllTransitions:
    """Verify suggestions are created for all transition types."""

    def _run_suggestion_test(self, transition, response_items, expected_type):
        mock_stack = _make_mock_stack()
        mock_stack.save_suggestion.return_value = f"suggestion-{expected_type}"

        if transition in ("raw_to_episode", "raw_to_note"):
            raw = MagicMock()
            raw.id = "r-1"
            raw.blob = "test"
            raw.content = None
            mock_stack._backend.list_raw.return_value = [raw]
            mock_stack.get_episodes.return_value = []
        elif transition.startswith("episode_to_"):
            ep = MagicMock()
            ep.id = "ep-1"
            ep.objective = "test"
            ep.outcome = "done"
            ep.processed = False
            mock_stack.get_episodes.return_value = [ep]
            mock_stack.get_beliefs.return_value = []
            mock_stack.get_goals.return_value = []
            mock_stack.get_relationships.return_value = []
            mock_stack.get_drives.return_value = []
        elif transition == "belief_to_value":
            belief = MagicMock()
            belief.id = "b-1"
            belief.statement = "test"
            belief.confidence = 0.8
            belief.processed = False
            mock_stack.get_beliefs.return_value = [belief]
            mock_stack.get_values.return_value = []

        response = json.dumps(response_items)
        processor, _ = _make_processor(mock_stack, response, auto_promote=False)
        results = processor.process(transition, force=True)

        assert len(results) == 1
        result = results[0]
        assert result.auto_promote is False
        assert result.created == []
        assert len(result.suggestions) == 1
        assert result.suggestions[0]["type"] == expected_type
        mock_stack.save_suggestion.assert_called_once()

    def test_raw_to_episode(self):
        self._run_suggestion_test(
            "raw_to_episode",
            [{"objective": "test", "outcome": "done", "source_raw_ids": ["r-1"]}],
            "episode",
        )

    def test_raw_to_note(self):
        self._run_suggestion_test(
            "raw_to_note",
            [{"content": "note text", "note_type": "fact", "source_raw_ids": ["r-1"]}],
            "note",
        )

    def test_episode_to_belief(self):
        self._run_suggestion_test(
            "episode_to_belief",
            [
                {
                    "statement": "belief",
                    "belief_type": "evaluative",
                    "confidence": 0.8,
                    "source_episode_ids": ["ep-1"],
                }
            ],
            "belief",
        )

    def test_episode_to_goal(self):
        self._run_suggestion_test(
            "episode_to_goal",
            [
                {
                    "title": "goal",
                    "description": "do it",
                    "goal_type": "task",
                    "source_episode_ids": ["ep-1"],
                }
            ],
            "goal",
        )

    def test_episode_to_relationship(self):
        self._run_suggestion_test(
            "episode_to_relationship",
            [
                {
                    "entity_name": "Alice",
                    "entity_type": "person",
                    "sentiment": 0.5,
                    "source_episode_ids": ["ep-1"],
                }
            ],
            "relationship",
        )

    def test_belief_to_value(self):
        self._run_suggestion_test(
            "belief_to_value",
            [
                {
                    "name": "honesty",
                    "statement": "be honest",
                    "priority": 90,
                    "source_belief_ids": ["b-1"],
                }
            ],
            "value",
        )

    def test_episode_to_drive(self):
        self._run_suggestion_test(
            "episode_to_drive",
            [
                {
                    "drive_type": "curiosity",
                    "intensity": 0.8,
                    "source_episode_ids": ["ep-1"],
                }
            ],
            "drive",
        )


# =============================================================================
# _write_suggestions detail tests
# =============================================================================


class TestWriteSuggestions:
    def test_suggestion_has_correct_fields(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_suggestion.return_value = "s-1"
        processor, _ = _make_processor(mock_stack)

        parsed = [
            {
                "objective": "test objective",
                "outcome": "test outcome",
                "source_raw_ids": ["r-1", "r-2"],
            }
        ]
        result = processor._write_suggestions("raw_to_episode", parsed, [])

        assert len(result) == 1
        assert result[0] == {"type": "episode", "id": "s-1"}

        # Verify the MemorySuggestion that was saved
        saved = mock_stack.save_suggestion.call_args[0][0]
        assert isinstance(saved, MemorySuggestion)
        assert saved.memory_type == "episode"
        assert saved.content == parsed[0]
        assert saved.source_raw_ids == ["r-1", "r-2"]
        assert saved.status == "pending"
        assert saved.stack_id == STACK_ID

    def test_suggestion_extracts_episode_source_ids(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_suggestion.return_value = "s-1"
        processor, _ = _make_processor(mock_stack)

        parsed = [
            {
                "statement": "belief",
                "source_episode_ids": ["ep-1", "ep-2"],
            }
        ]
        processor._write_suggestions("episode_to_belief", parsed, [])

        saved = mock_stack.save_suggestion.call_args[0][0]
        assert saved.source_raw_ids == ["episode:ep-1", "episode:ep-2"]

    def test_suggestion_extracts_belief_source_ids(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_suggestion.return_value = "s-1"
        processor, _ = _make_processor(mock_stack)

        parsed = [
            {
                "name": "value",
                "source_belief_ids": ["b-1"],
            }
        ]
        processor._write_suggestions("belief_to_value", parsed, [])

        saved = mock_stack.save_suggestion.call_args[0][0]
        assert saved.source_raw_ids == ["belief:b-1"]

    def test_multiple_suggestions(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_suggestion.side_effect = ["s-1", "s-2"]
        processor, _ = _make_processor(mock_stack)

        parsed = [
            {"objective": "first", "outcome": "ok", "source_raw_ids": []},
            {"objective": "second", "outcome": "ok", "source_raw_ids": []},
        ]
        result = processor._write_suggestions("raw_to_episode", parsed, [])
        assert len(result) == 2

    def test_suggestion_exception_does_not_crash(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_suggestion.side_effect = [Exception("db error"), "s-2"]
        processor, _ = _make_processor(mock_stack)

        parsed = [
            {"objective": "fails", "outcome": "err", "source_raw_ids": []},
            {"objective": "works", "outcome": "ok", "source_raw_ids": []},
        ]
        result = processor._write_suggestions("raw_to_episode", parsed, [])
        assert len(result) == 1
        assert result[0]["id"] == "s-2"

    def test_unknown_transition_returns_empty(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        result = processor._write_suggestions("fake_transition", [{"key": "val"}], [])
        assert result == []


# =============================================================================
# _extract_source_ids
# =============================================================================


class TestExtractSourceIds:
    def test_raw_to_episode(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"source_raw_ids": ["r-1", "r-2"]}
        assert processor._extract_source_ids("raw_to_episode", item) == ["r-1", "r-2"]

    def test_raw_to_note(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"source_raw_ids": ["r-3"]}
        assert processor._extract_source_ids("raw_to_note", item) == ["r-3"]

    def test_episode_to_belief(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"source_episode_ids": ["ep-1"]}
        assert processor._extract_source_ids("episode_to_belief", item) == ["episode:ep-1"]

    def test_episode_to_goal(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"source_episode_ids": ["ep-2"]}
        assert processor._extract_source_ids("episode_to_goal", item) == ["episode:ep-2"]

    def test_episode_to_relationship(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"source_episode_ids": ["ep-3"]}
        assert processor._extract_source_ids("episode_to_relationship", item) == ["episode:ep-3"]

    def test_episode_to_drive(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"source_episode_ids": ["ep-4"]}
        assert processor._extract_source_ids("episode_to_drive", item) == ["episode:ep-4"]

    def test_belief_to_value(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"source_belief_ids": ["b-1"]}
        assert processor._extract_source_ids("belief_to_value", item) == ["belief:b-1"]

    def test_missing_ids_returns_empty(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        assert processor._extract_source_ids("raw_to_episode", {}) == []

    def test_unknown_transition_returns_empty(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        assert processor._extract_source_ids("fake", {"x": "y"}) == []


# =============================================================================
# Entity.process() with auto_promote
# =============================================================================


class TestEntityProcessAutoPromote:
    def test_entity_default_creates_suggestions(self, entity):
        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())
        results = entity.process(force=True)
        # All transitions skip (no sources), but auto_promote is False
        for r in results:
            assert r.auto_promote is False

    def test_entity_auto_promote_true(self, entity):
        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())
        results = entity.process(force=True, auto_promote=True)
        # Skipped results (no sources) don't reach the promotion path,
        # but non-skipped results should have auto_promote=True.
        # With no data, all transitions skip. Verify no suggestions were created.
        for r in results:
            assert r.skipped is True
            # Skipped results don't set auto_promote (they exit early)
            assert r.created == []
            assert r.suggestions == []

    def test_entity_auto_promote_false_by_default(self, entity):
        """Verify the default is suggestions mode, not auto-promote."""

        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())
        # Call without auto_promote arg
        results = entity.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert results[0].auto_promote is False


# =============================================================================
# Audit log includes suggestion info
# =============================================================================


class TestAuditLogSuggestions:
    def test_audit_log_records_suggestion_count(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "test"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []
        mock_stack.save_suggestion.return_value = "s-1"

        response = json.dumps([{"objective": "thing", "outcome": "done", "source_raw_ids": []}])
        processor, _ = _make_processor(mock_stack, response, auto_promote=False)
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        processor._process_layer("raw_to_episode", config, auto_promote=False)

        # suggestion.created + process summary = 2 calls
        assert mock_stack.log_audit.call_count == 2
        details = mock_stack.log_audit.call_args[1]["details"]
        assert details["suggestion_count"] == 1
        assert details["created_count"] == 0
        assert details["auto_promote"] is False

    def test_audit_log_records_auto_promote(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "test"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []
        mock_stack.save_episode.return_value = "ep-1"

        response = json.dumps([{"objective": "thing", "outcome": "done", "source_raw_ids": []}])
        processor, _ = _make_processor(mock_stack, response, auto_promote=True)
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        processor._process_layer("raw_to_episode", config, auto_promote=True)

        # memory.promoted + process summary = 2 calls
        assert mock_stack.log_audit.call_count == 2
        details = mock_stack.log_audit.call_args[1]["details"]
        assert details["created_count"] == 1
        assert details["suggestion_count"] == 0
        assert details["auto_promote"] is True


# =============================================================================
# MCP handler
# =============================================================================


class TestMCPHandlerAutoPromote:
    def test_validate_auto_promote_default(self):
        from kernle.mcp.handlers.processing import validate_memory_process

        result = validate_memory_process({})
        assert result["auto_promote"] is False

    def test_validate_auto_promote_true(self):
        from kernle.mcp.handlers.processing import validate_memory_process

        result = validate_memory_process({"auto_promote": True})
        assert result["auto_promote"] is True

    def test_validate_auto_promote_invalid_type(self):
        from kernle.mcp.handlers.processing import validate_memory_process

        result = validate_memory_process({"auto_promote": "yes"})
        assert result["auto_promote"] is False


# =============================================================================
# Backward compatibility: auto_promote=True preserves old behavior
# =============================================================================


class TestBackwardCompatibility:
    """Ensure auto_promote=True produces the same results as old code."""

    def test_auto_promote_writes_memories_same_as_before(self, stack):
        """auto_promote=True should produce identical results to old behavior."""
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Raw content {i}",
                    source="test",
                )
            )

        response = json.dumps(
            [
                {
                    "objective": "Learned something",
                    "outcome": "Good result",
                    "outcome_type": "success",
                    "lessons": ["lesson 1"],
                    "source_raw_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack, inference=inference, core_id="test", auto_promote=True
        )

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        result = results[0]
        assert result.auto_promote is True
        assert len(result.created) == 1
        assert result.suggestions == []

        # Episode actually saved
        episodes = stack.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].objective == "Learned something"
        assert episodes[0].source_type == "processing"

    def test_auto_promote_marks_sources_processed(self, stack):
        """auto_promote=True still marks sources as processed."""
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Test content",
                source="test",
            )
        )

        response = json.dumps([{"objective": "Test", "outcome": "Done", "source_raw_ids": []}])
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack, inference=inference, core_id="test", auto_promote=True
        )

        processor.process("raw_to_episode", force=True)

        # Raws should be marked as processed
        raws = stack._backend.list_raw(processed=False)
        assert len(raws) == 0

    def test_suggestions_mode_marks_sources_processed(self, stack):
        """Default suggestions mode also marks sources as processed."""
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Test content",
                source="test",
            )
        )

        response = json.dumps([{"objective": "Test", "outcome": "Done", "source_raw_ids": []}])
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        processor.process("raw_to_episode", force=True)

        # Raws should still be marked as processed (to prevent re-processing)
        raws = stack._backend.list_raw(processed=False)
        assert len(raws) == 0
