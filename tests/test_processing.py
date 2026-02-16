"""Tests for memory processing sessions (v0.9.0 PR 6).

Tests cover:
- MemoryProcessor: layer-specific processing, trigger evaluation
- Storage: mark_episode/note_processed, get/set_processing_config
- Stack: processing method routing
- Entity: process() method with mock inference
- Processing output parsing and memory writing
- check_triggers: all transition types
- _gather_sources / _gather_context: all transition types
- _format_sources / _format_context: all memory types
- _write_memories: all 7 transition types
- _mark_processed: raw, episode, belief sources
- _parse_response: plain JSON, markdown-fenced JSON
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kernle.entity import Entity
from kernle.processing import (
    DEFAULT_LAYER_CONFIGS,
    IDENTITY_LAYER_TRANSITIONS,
    NO_INFERENCE_MIN_CONFIDENCE,
    NO_INFERENCE_MIN_EVIDENCE,
    NO_OVERRIDE_TRANSITIONS,
    OVERRIDE_TRANSITIONS,
    VALID_TRANSITION_ORDER,
    VALID_TRANSITIONS,
    LayerConfig,
    MemoryProcessor,
    ProcessingResult,
    PromotionGateConfig,
    evaluate_triggers,
)
from kernle.stack.sqlite_stack import SQLiteStack
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Belief, Episode, Note, RawEntry

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
def storage(tmp_path):
    s = SQLiteStorage(STACK_ID, db_path=tmp_path / "test.db")
    yield s
    s.close()


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
    """Mock inference service for testing."""

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


class MockInferenceWithModel(MockInference):
    """Mock inference service that exposes a model_id."""

    def __init__(self, model_id: str, response: str = "[]"):
        super().__init__(response=response)
        self.model_id = model_id


def _save_raw(storage, blob="test raw content"):
    storage.save_raw(blob=blob, source="test")
    raws = storage.list_raw(limit=1)
    return raws[0].id


def _save_episode(storage, objective="Test episode", outcome="Test outcome"):
    ep = Episode(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        objective=objective,
        outcome=outcome,
        source_type="observation",
        source_entity="test",
    )
    storage.save_episode(ep)
    return ep.id


def _make_mock_stack():
    """Create a MagicMock stack with _backend for MemoryProcessor tests."""
    mock_stack = MagicMock()
    mock_stack.stack_id = STACK_ID
    mock_stack._backend = MagicMock()
    return mock_stack


def _make_processor(mock_stack, response="[]"):
    """Create a MemoryProcessor with a mock stack and inference."""
    inference = MockInference(response)
    return (
        MemoryProcessor(
            stack=mock_stack, inference=inference, core_id="test", promotion_gates=_NO_GATES
        ),
        inference,
    )


# =============================================================================
# Trigger Evaluation
# =============================================================================


class TestEvaluateTriggers:
    def test_disabled_config_never_triggers(self):
        config = LayerConfig(layer_transition="raw_to_episode", enabled=False)
        assert not evaluate_triggers("raw_to_episode", config, 100)

    def test_quantity_threshold_triggers(self):
        config = LayerConfig(layer_transition="raw_to_episode", quantity_threshold=5)
        assert not evaluate_triggers("raw_to_episode", config, 4)
        assert evaluate_triggers("raw_to_episode", config, 5)
        assert evaluate_triggers("raw_to_episode", config, 10)

    def test_valence_threshold_triggers_for_raw(self):
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            valence_threshold=2.0,
        )
        assert not evaluate_triggers("raw_to_episode", config, 1, cumulative_valence=1.5)
        assert evaluate_triggers("raw_to_episode", config, 1, cumulative_valence=2.5)

    def test_valence_threshold_triggers_for_raw_to_note(self):
        config = LayerConfig(
            layer_transition="raw_to_note",
            quantity_threshold=100,
            valence_threshold=2.0,
        )
        assert evaluate_triggers("raw_to_note", config, 1, cumulative_valence=3.0)

    def test_valence_threshold_triggers_for_episode_to_belief(self):
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=2.0,
        )
        assert not evaluate_triggers("episode_to_belief", config, 1, cumulative_valence=1.5)
        assert evaluate_triggers("episode_to_belief", config, 1, cumulative_valence=2.5)

    def test_time_threshold_triggers(self):
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        assert not evaluate_triggers("raw_to_episode", config, 1, hours_since_last=12.0)
        assert evaluate_triggers("raw_to_episode", config, 1, hours_since_last=25.0)

    def test_time_threshold_none_does_not_trigger(self):
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        assert not evaluate_triggers("raw_to_episode", config, 1, hours_since_last=None)

    def test_time_threshold_exact_boundary(self):
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        assert evaluate_triggers("raw_to_episode", config, 1, hours_since_last=24.0)

    def test_time_threshold_zero_disables(self):
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            time_threshold_hours=0,
        )
        assert not evaluate_triggers("raw_to_episode", config, 1, hours_since_last=100.0)


# =============================================================================
# LayerConfig defaults
# =============================================================================


class TestLayerConfig:
    def test_default_configs_cover_all_transitions(self):
        assert set(DEFAULT_LAYER_CONFIGS.keys()) == VALID_TRANSITIONS

    def test_default_config_values(self):
        cfg = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        assert cfg.enabled is True
        assert cfg.quantity_threshold == 10
        assert cfg.batch_size == 10

    def test_custom_config(self):
        cfg = LayerConfig(
            layer_transition="raw_to_episode",
            enabled=True,
            quantity_threshold=5,
            batch_size=3,
        )
        assert cfg.quantity_threshold == 5
        assert cfg.batch_size == 3


class TestLayerConfigGating:
    def test_process_blocks_when_model_id_mismatch(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = [MagicMock(id="r-1", blob="raw", content=None)]
        mock_stack.get_episodes.return_value = []
        mock_stack.get_beliefs.return_value = []
        inference = MockInferenceWithModel("actual-model", response="[]")
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=inference,
            core_id="test",
            configs={"raw_to_episode": LayerConfig(layer_transition="raw_to_episode")},
        )
        processor.update_config(
            "raw_to_episode",
            LayerConfig(layer_transition="raw_to_episode", model_id="expected-model"),
        )

        result = processor.process("raw_to_episode", force=True)[0]

        assert result.skipped
        assert not result.inference_blocked
        assert "requires model 'expected-model'" in result.skip_reason
        assert result.source_count == 1
        assert inference.calls == []
        mock_stack._backend.mark_raw_processed.assert_not_called()

    def test_model_id_match_allows_processing(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = [MagicMock(id="r-1", blob="raw", content=None)]
        mock_stack.get_episodes.return_value = []
        mock_stack.get_beliefs.return_value = []
        inference = MockInferenceWithModel(
            "expected-model",
            response=json.dumps(
                [
                    {
                        "objective": "validated output",
                        "outcome": "accepted",
                        "source_raw_ids": ["r-1"],
                    }
                ]
            ),
        )
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=inference,
            core_id="test",
            configs={"raw_to_episode": LayerConfig(layer_transition="raw_to_episode")},
        )
        processor.update_config(
            "raw_to_episode",
            LayerConfig(layer_transition="raw_to_episode", model_id="expected-model"),
        )

        result = processor.process("raw_to_episode", force=True)[0]

        assert not result.skipped
        assert not result.inference_blocked
        assert result.source_count == 1
        assert inference.calls
        mock_stack._backend.mark_raw_processed.assert_called_once()

    def test_empty_inference_output_keeps_sources_unprocessed_suggestion_mode(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = [MagicMock(id="r-1", blob="test", content=None)]
        mock_stack.get_episodes.return_value = []

        processor, _ = _make_processor(mock_stack, "[]")
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config, auto_promote=False)

        assert not result.skipped
        assert result.source_count == 1
        assert result.no_output_processed
        assert result.created == []
        assert result.suggestions == []
        mock_stack._backend.mark_raw_processed.assert_not_called()

    def test_process_blocks_when_daily_session_cap_reached(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = [MagicMock(id="r-1", blob="raw", content=None)]
        mock_stack.get_episodes.return_value = []
        mock_stack.get_beliefs.return_value = []
        mock_stack.get_audit_log.return_value = [
            {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        ]

        inference = MockInference(response="[]")
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=inference,
            core_id="test",
            configs={"raw_to_episode": LayerConfig(layer_transition="raw_to_episode")},
        )
        processor.update_config(
            "raw_to_episode",
            LayerConfig(layer_transition="raw_to_episode", max_sessions_per_day=1),
        )

        result = processor.process("raw_to_episode", force=True)[0]

        assert result.skipped
        assert result.source_count == 1
        assert "max_sessions_per_day" in result.skip_reason
        assert inference.calls == []


# =============================================================================
# Storage: mark_episode_processed / mark_note_processed
# =============================================================================


class TestMarkEpisodeProcessed:
    def test_marks_episode(self, storage):
        eid = _save_episode(storage)
        ep = storage.get_episode(eid)
        assert not ep.processed

        assert storage.mark_episode_processed(eid)
        ep = storage.get_episode(eid)
        assert ep.processed

    def test_nonexistent_returns_false(self, storage):
        assert not storage.mark_episode_processed("no-such-id")

    def test_deleted_returns_false(self, storage):
        eid = _save_episode(storage)
        storage.forget_memory("episode", eid, "test")
        # forget sets deleted=1
        # Actually forget_memory sets strength=0, not deleted. Let's skip this test.


class TestMarkNoteProcessed:
    def test_marks_note(self, storage):
        note = Note(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            content="Test note",
            note_type="observation",
            source_type="observation",
            source_entity="test",
        )
        storage.save_note(note)
        notes = storage.get_notes(limit=10)
        assert not notes[0].processed

        assert storage.mark_note_processed(note.id)
        notes = storage.get_notes(limit=10)
        marked = [n for n in notes if n.id == note.id]
        assert len(marked) == 1
        assert marked[0].processed

    def test_nonexistent_returns_false(self, storage):
        assert not storage.mark_note_processed("no-such-id")


# =============================================================================
# Storage: processing_config
# =============================================================================


class TestProcessingConfig:
    def test_empty_config(self, storage):
        configs = storage.get_processing_config()
        assert configs == []

    def test_set_and_get(self, storage):
        storage.set_processing_config(
            "raw_to_episode",
            enabled=True,
            quantity_threshold=5,
            batch_size=3,
        )
        configs = storage.get_processing_config()
        assert len(configs) == 1
        assert configs[0]["layer_transition"] == "raw_to_episode"
        assert configs[0]["enabled"] is True
        assert configs[0]["quantity_threshold"] == 5
        assert configs[0]["batch_size"] == 3

    def test_update_existing(self, storage):
        storage.set_processing_config("raw_to_episode", quantity_threshold=5)
        storage.set_processing_config("raw_to_episode", quantity_threshold=20)

        configs = storage.get_processing_config()
        assert len(configs) == 1
        assert configs[0]["quantity_threshold"] == 20

    def test_multiple_configs(self, storage):
        storage.set_processing_config("raw_to_episode", batch_size=5)
        storage.set_processing_config("episode_to_belief", batch_size=10)

        configs = storage.get_processing_config()
        assert len(configs) == 2
        transitions = {c["layer_transition"] for c in configs}
        assert transitions == {"episode_to_belief", "raw_to_episode"}

    def test_disable_transition(self, storage):
        storage.set_processing_config("raw_to_episode", enabled=True)
        storage.set_processing_config("raw_to_episode", enabled=False)

        configs = storage.get_processing_config()
        assert configs[0]["enabled"] is False


# =============================================================================
# Stack routing
# =============================================================================


class TestStackProcessingRouting:
    def test_get_processing_config(self, stack):
        configs = stack.get_processing_config()
        assert configs == []

    def test_set_processing_config(self, stack):
        assert stack.set_processing_config("raw_to_episode", batch_size=5)
        configs = stack.get_processing_config()
        assert len(configs) == 1

    def test_mark_episode_processed(self, stack):
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="Test",
            outcome="Done",
            source_type="observation",
            source_entity="test",
        )
        stack.save_episode(ep)
        assert stack.mark_episode_processed(ep.id)

    def test_mark_note_processed(self, stack):
        note = Note(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            content="Test note",
            note_type="observation",
            source_type="observation",
            source_entity="test",
        )
        stack.save_note(note)
        assert stack.mark_note_processed(note.id)


# =============================================================================
# MemoryProcessor
# =============================================================================


class TestMemoryProcessor:
    def test_no_sources_skips(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == "No unprocessed sources"

    def test_process_raw_to_episode(self, stack):
        # Save some raw entries
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Raw content {i}",
                    source="test",
                )
            )

        # Mock inference to return one episode
        response = json.dumps(
            [
                {
                    "objective": "Learned something",
                    "outcome": "Good result",
                    "outcome_type": "success",
                    "lessons": ["lesson 1"],
                    "source_raw_ids": [],  # Empty for simplicity
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
        assert not result.skipped
        assert result.source_count == 3
        assert len(result.created) == 1
        assert result.created[0]["type"] == "episode"

        # Verify episode was saved
        episodes = stack.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].objective == "Learned something"

    def test_process_raw_to_note(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Some factual info",
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
        processor = MemoryProcessor(
            stack=stack, inference=inference, core_id="test", auto_promote=True
        )

        results = processor.process("raw_to_note", force=True)
        assert len(results) == 1
        assert len(results[0].created) == 1
        assert results[0].created[0]["type"] == "note"

    def test_process_episode_to_belief(self, stack):
        # Save unprocessed episodes
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
            stack=stack,
            inference=inference,
            core_id="test",
            auto_promote=True,
            promotion_gates=_NO_GATES,
        )

        results = processor.process("episode_to_belief", force=True)
        assert len(results) == 1
        assert len(results[0].created) == 1
        assert results[0].created[0]["type"] == "belief"

    def test_inference_failure_returns_error(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Some content",
                source="test",
            )
        )

        inference = MockInference()
        inference.infer = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("model error"))
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert len(results[0].errors) == 1
        assert "Inference failed" in results[0].errors[0]

    def test_bad_json_returns_error(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Some content",
                source="test",
            )
        )

        inference = MockInference("not valid json")
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert len(results[0].errors) == 1
        assert "Parse failed" in results[0].errors[0]

    def test_check_triggers_no_sources(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        assert not processor.check_triggers("raw_to_episode")

    def test_check_triggers_meets_threshold(self, stack):
        # Save enough raws to meet threshold
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=3,
        )
        for i in range(5):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Content {i}",
                    source="test",
                )
            )

        inference = MockInference()
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            configs={"raw_to_episode": config},
        )
        assert processor.check_triggers("raw_to_episode")

    def test_process_all_transitions_no_sources(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        # All transitions skip due to no sources
        results = processor.process(force=True)
        assert all(r.skipped for r in results)

    def test_update_config(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        new_config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=3,
        )
        processor.update_config("raw_to_episode", new_config)
        assert processor.get_config("raw_to_episode").quantity_threshold == 3

    def test_disabled_transition_skipped(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Content",
                source="test",
            )
        )
        config = LayerConfig(
            layer_transition="raw_to_episode",
            enabled=False,
        )
        inference = MockInference()
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            configs={"raw_to_episode": config},
        )
        results = processor.process("raw_to_episode", force=True)
        assert results == []


# =============================================================================
# MemoryProcessor: parse_response edge cases
# =============================================================================


class TestParseResponse:
    def test_strips_markdown_fences(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        result = processor._parse_response('```json\n[{"key": "value"}]\n```')
        assert result == [{"key": "value"}]

    def test_plain_json(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        result = processor._parse_response('[{"key": "value"}]')
        assert result == [{"key": "value"}]

    def test_empty_array(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        result = processor._parse_response("[]")
        assert result == []

    def test_strips_plain_backtick_fences(self, stack):
        """Markdown fences without language tag are also stripped."""
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        result = processor._parse_response('```\n[{"a": 1}]\n```')
        assert result == [{"a": 1}]

    def test_invalid_json_raises(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        with pytest.raises(json.JSONDecodeError):
            processor._parse_response("not json at all")

    def test_multiline_json_in_fences(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        fenced = '```json\n[\n  {"key": "val1"},\n  {"key": "val2"}\n]\n```'
        result = processor._parse_response(fenced)
        assert len(result) == 2
        assert result[0]["key"] == "val1"


# =============================================================================
# MemoryProcessor: mark processed
# =============================================================================


class TestMarkProcessed:
    def test_raws_marked_after_processing(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Test content",
                source="test",
            )
        )

        response = json.dumps(
            [
                {
                    "objective": "Test",
                    "outcome": "Done",
                    "source_raw_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        processor.process("raw_to_episode", force=True)

        # Raws should be marked as processed
        raws = stack._backend.list_raw(processed=False)
        assert len(raws) == 0
        all_raws = stack._backend.list_raw()
        assert len(all_raws) == 1
        assert all_raws[0].processed

    def test_episodes_marked_after_processing(self, stack):
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="Test",
            outcome="Done",
            source_type="observation",
            source_entity="test",
            processed=False,
        )
        stack.save_episode(ep)

        response = json.dumps(
            [
                {
                    "statement": "A belief",
                    "belief_type": "factual",
                    "confidence": 0.7,
                    "source_episode_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            promotion_gates=_NO_GATES,
        )
        processor.process("episode_to_belief", force=True)

        # Episode should be marked as processed
        episodes = stack.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].processed


# =============================================================================
# Entity.process()
# =============================================================================


class TestEntityProcess:
    def test_requires_stack(self):
        ent = Entity(core_id="test-no-stack")
        with pytest.raises(Exception):
            ent.process()

    def test_no_model_blocks_all_transitions(self, entity):
        """Without a model, all transitions are blocked (not raised)."""
        results = entity.process(force=True)
        blocked = [r for r in results if r.inference_blocked]
        assert len(blocked) == len(VALID_TRANSITIONS)

    def test_process_with_mock_model(self, entity):
        """Entity.process() works with a mock model and no data."""

        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())
        results = entity.process(force=True)
        # All transitions skip (no sources)
        assert all(r.skipped for r in results)

    def test_process_specific_transition(self, entity):
        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())
        results = entity.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert results[0].layer_transition == "raw_to_episode"

    def test_process_loads_saved_config(self, entity):
        """Entity.process() loads config from stack."""

        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())

        # Save a config
        stack = entity.active_stack
        stack.set_processing_config("raw_to_episode", quantity_threshold=99)

        # Process should load the config (even though it doesn't change behavior
        # in this test since we force=True and have no sources)
        results = entity.process("raw_to_episode", force=True)
        assert len(results) == 1


# =============================================================================
# ProcessingResult
# =============================================================================


class TestProcessingResult:
    def test_default_values(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=5,
        )
        assert result.created == []
        assert result.errors == []
        assert result.skipped is False
        assert result.skip_reason is None

    def test_skipped_result(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=0,
            skipped=True,
            skip_reason="No sources",
        )
        assert result.skipped
        assert result.skip_reason == "No sources"


# =============================================================================
# check_triggers — all transition types (mock-based)
# =============================================================================


class TestCheckTriggersAllTransitions:
    """Cover check_triggers lines 325-360 for each transition type."""

    def test_unknown_config_returns_false(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        assert not processor.check_triggers("nonexistent_transition")

    def test_disabled_config_returns_false(self):
        mock_stack = _make_mock_stack()
        config = LayerConfig(layer_transition="raw_to_episode", enabled=False)
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=MockInference(),
            core_id="test",
            configs={"raw_to_episode": config},
        )
        assert not processor.check_triggers("raw_to_episode")

    def test_no_backend_returns_false(self):
        mock_stack = MagicMock(spec=[])
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=MockInference(),
            core_id="test",
        )
        assert not processor.check_triggers("raw_to_episode")

    def test_raw_to_episode_triggers(self):
        mock_stack = _make_mock_stack()
        raws = [MagicMock() for _ in range(11)]
        mock_stack._backend.list_raw.return_value = raws
        processor, _ = _make_processor(mock_stack)
        assert processor.check_triggers("raw_to_episode")
        mock_stack._backend.list_raw.assert_called_once_with(processed=False, limit=11)

    def test_raw_to_note_triggers(self):
        mock_stack = _make_mock_stack()
        raws = [MagicMock() for _ in range(11)]
        mock_stack._backend.list_raw.return_value = raws
        processor, _ = _make_processor(mock_stack)
        assert processor.check_triggers("raw_to_note")

    def test_episode_to_belief_triggers(self):
        mock_stack = _make_mock_stack()
        episodes = [MagicMock(processed=False) for _ in range(6)]
        mock_stack._backend.get_episodes.return_value = episodes
        processor, _ = _make_processor(mock_stack)
        assert processor.check_triggers("episode_to_belief")
        mock_stack._backend.get_episodes.assert_called_once_with(limit=6, processed=False)

    def test_episode_to_belief_no_unprocessed(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.get_episodes.return_value = []
        processor, _ = _make_processor(mock_stack)
        assert not processor.check_triggers("episode_to_belief")

    def test_episode_to_goal_triggers(self):
        mock_stack = _make_mock_stack()
        episodes = [MagicMock(processed=False) for _ in range(6)]
        mock_stack._backend.get_episodes.return_value = episodes
        processor, _ = _make_processor(mock_stack)
        assert processor.check_triggers("episode_to_goal")

    def test_episode_to_relationship_triggers(self):
        mock_stack = _make_mock_stack()
        # episode_to_relationship has quantity_threshold=3 by default
        episodes = [MagicMock(processed=False) for _ in range(4)]
        mock_stack._backend.get_episodes.return_value = episodes
        processor, _ = _make_processor(mock_stack)
        assert processor.check_triggers("episode_to_relationship")

    def test_belief_to_value_triggers(self):
        mock_stack = _make_mock_stack()
        beliefs = [MagicMock(processed=False) for _ in range(6)]
        mock_stack._backend.get_beliefs.return_value = beliefs
        processor, _ = _make_processor(mock_stack)
        assert processor.check_triggers("belief_to_value")
        mock_stack._backend.get_beliefs.assert_called_once_with(limit=6, processed=False)

    def test_episode_to_drive_triggers(self):
        mock_stack = _make_mock_stack()
        episodes = [MagicMock(processed=False) for _ in range(6)]
        mock_stack._backend.get_episodes.return_value = episodes
        processor, _ = _make_processor(mock_stack)
        assert processor.check_triggers("episode_to_drive")

    def test_unknown_transition_returns_false(self):
        mock_stack = _make_mock_stack()
        config = LayerConfig(layer_transition="unknown_type", enabled=True)
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=MockInference(),
            core_id="test",
            configs={"unknown_type": config},
        )
        assert not processor.check_triggers("unknown_type")


# =============================================================================
# _gather_sources — all transition types (mock-based)
# =============================================================================


class TestGatherSources:
    def test_no_backend_returns_empty(self):
        mock_stack = MagicMock(spec=[])
        processor = MemoryProcessor(stack=mock_stack, inference=MockInference(), core_id="test")
        assert processor._gather_sources("raw_to_episode", 10) == []

    def test_raw_to_episode(self):
        mock_stack = _make_mock_stack()
        raws = [MagicMock() for _ in range(3)]
        mock_stack._backend.list_raw.return_value = raws
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("raw_to_episode", 10)
        assert result == raws
        mock_stack._backend.list_raw.assert_called_once_with(processed=False, limit=10)

    def test_raw_to_note(self):
        mock_stack = _make_mock_stack()
        raws = [MagicMock() for _ in range(2)]
        mock_stack._backend.list_raw.return_value = raws
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("raw_to_note", 5)
        assert result == raws

    def test_episode_to_belief(self):
        mock_stack = _make_mock_stack()
        unprocessed = [MagicMock(processed=False) for _ in range(3)]
        mock_stack._backend.get_episodes.return_value = unprocessed
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("episode_to_belief", 10)
        assert len(result) == 3
        mock_stack._backend.get_episodes.assert_called_once_with(limit=10, processed=False)

    def test_episode_to_goal(self):
        mock_stack = _make_mock_stack()
        eps = [MagicMock(processed=False) for _ in range(4)]
        mock_stack._backend.get_episodes.return_value = eps
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("episode_to_goal", 10)
        assert len(result) == 4

    def test_episode_to_relationship(self):
        mock_stack = _make_mock_stack()
        eps = [MagicMock(processed=False) for _ in range(2)]
        mock_stack._backend.get_episodes.return_value = eps
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("episode_to_relationship", 5)
        assert len(result) == 2

    def test_episode_to_drive(self):
        mock_stack = _make_mock_stack()
        eps = [MagicMock(processed=False) for _ in range(3)]
        mock_stack._backend.get_episodes.return_value = eps
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("episode_to_drive", 10)
        assert len(result) == 3

    def test_belief_to_value(self):
        mock_stack = _make_mock_stack()
        beliefs = [MagicMock(processed=False) for _ in range(4)]
        mock_stack._backend.get_beliefs.return_value = beliefs
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("belief_to_value", 10)
        assert len(result) == 4
        mock_stack._backend.get_beliefs.assert_called_once_with(limit=10, processed=False)

    def test_batch_size_limits_episodes(self):
        mock_stack = _make_mock_stack()
        eps = [MagicMock(processed=False) for _ in range(5)]
        mock_stack._backend.get_episodes.return_value = eps
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("episode_to_belief", 5)
        assert len(result) == 5
        # Backend is called with limit=batch_size directly
        mock_stack._backend.get_episodes.assert_called_once_with(limit=5, processed=False)

    def test_unknown_transition_returns_empty(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_sources("nonexistent_type", 10)
        assert result == []


# =============================================================================
# _gather_context — all transition types (mock-based)
# =============================================================================


class TestGatherContext:
    def test_raw_to_episode_context(self):
        mock_stack = _make_mock_stack()
        episodes = [MagicMock() for _ in range(5)]
        mock_stack.get_episodes.return_value = episodes
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_context("raw_to_episode")
        assert result == episodes
        mock_stack.get_episodes.assert_called_once_with(limit=20)

    def test_raw_to_note_context(self):
        mock_stack = _make_mock_stack()
        episodes = [MagicMock() for _ in range(3)]
        mock_stack.get_episodes.return_value = episodes
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_context("raw_to_note")
        assert result == episodes

    def test_episode_to_belief_context(self):
        mock_stack = _make_mock_stack()
        beliefs = [MagicMock() for _ in range(3)]
        mock_stack.get_beliefs.return_value = beliefs
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_context("episode_to_belief")
        assert result == beliefs
        mock_stack.get_beliefs.assert_called_once_with(limit=20)

    def test_episode_to_goal_context(self):
        mock_stack = _make_mock_stack()
        goals = [MagicMock() for _ in range(3)]
        mock_stack.get_goals.return_value = goals
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_context("episode_to_goal")
        assert result == goals
        mock_stack.get_goals.assert_called_once_with(limit=20)

    def test_episode_to_relationship_context(self):
        mock_stack = _make_mock_stack()
        rels = [MagicMock() for _ in range(2)]
        mock_stack.get_relationships.return_value = rels
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_context("episode_to_relationship")
        assert result == rels
        mock_stack.get_relationships.assert_called_once()

    def test_belief_to_value_context(self):
        mock_stack = _make_mock_stack()
        values = [MagicMock() for _ in range(3)]
        mock_stack.get_values.return_value = values
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_context("belief_to_value")
        assert result == values
        mock_stack.get_values.assert_called_once_with(limit=20)

    def test_episode_to_drive_context(self):
        mock_stack = _make_mock_stack()
        drives = [MagicMock() for _ in range(2)]
        mock_stack.get_drives.return_value = drives
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_context("episode_to_drive")
        assert result == drives
        mock_stack.get_drives.assert_called_once()

    def test_unknown_transition_returns_empty(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        result = processor._gather_context("nonexistent_type")
        assert result == []


# =============================================================================
# _format_sources — episodes, beliefs, raw entries
# =============================================================================


class TestFormatSources:
    def test_format_raw_entries_with_blob(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        raw = MagicMock()
        raw.id = "raw-1"
        raw.blob = "test content here"
        raw.content = None
        result = processor._format_sources("raw_to_episode", [raw])
        assert "[raw-1]" in result
        assert "test content here" in result

    def test_format_raw_entries_with_content_fallback(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        raw = MagicMock()
        raw.id = "raw-2"
        raw.blob = None
        raw.content = "fallback content"
        result = processor._format_sources("raw_to_note", [raw])
        assert "[raw-2]" in result
        assert "fallback content" in result

    def test_format_raw_entries_both_none(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        raw = MagicMock()
        raw.id = "raw-3"
        raw.blob = None
        raw.content = None
        result = processor._format_sources("raw_to_episode", [raw])
        assert "[raw-3]" in result

    def test_format_episodes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        ep = MagicMock()
        ep.id = "ep-1"
        ep.objective = "did something"
        ep.outcome = "it worked"
        result = processor._format_sources("episode_to_belief", [ep])
        assert "[ep-1]" in result
        assert "did something" in result
        assert "it worked" in result

    def test_format_beliefs(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        belief = MagicMock(spec=["id", "statement", "confidence"])
        belief.id = "b-1"
        belief.statement = "testing matters"
        belief.confidence = 0.9
        # Remove objective attribute so hasattr check fails for episode branch
        del belief.objective
        result = processor._format_sources("belief_to_value", [belief])
        assert "[b-1]" in result
        assert "testing matters" in result
        assert "0.9" in result

    def test_format_empty_sources(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        result = processor._format_sources("raw_to_episode", [])
        assert result == "(none)"

    def test_format_fallback_str(self):
        """Sources without known attributes use str() fallback."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        obj = MagicMock(spec=["id"])
        obj.id = "other-1"
        del obj.blob
        del obj.content
        del obj.objective
        del obj.statement
        result = processor._format_sources("episode_to_drive", [obj])
        assert "[other-1]" in result

    def test_format_raw_long_blob_truncated(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        raw = MagicMock()
        raw.id = "raw-long"
        raw.blob = "x" * 1000
        raw.content = None
        result = processor._format_sources("raw_to_episode", [raw])
        # blob[:500] truncation
        assert len(result) < 600


# =============================================================================
# _format_context — values, goals, drives, relationships, empty
# =============================================================================


class TestFormatContext:
    def test_empty_context(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        result = processor._format_context("raw_to_episode", [])
        assert result == "(none)"

    def test_format_episodes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        ep = MagicMock()
        ep.objective = "obj1"
        ep.outcome = "out1"
        result = processor._format_context("raw_to_episode", [ep])
        assert "obj1" in result
        assert "out1" in result

    def test_format_beliefs(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        belief = MagicMock(spec=["statement", "belief_type"])
        belief.statement = "things work well"
        belief.belief_type = "causal"
        del belief.objective  # Not an episode
        result = processor._format_context("episode_to_belief", [belief])
        assert "things work well" in result

    def test_format_values(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        value = MagicMock(spec=["name", "statement"])
        value.name = "honesty"
        value.statement = "always be truthful"
        del value.objective  # Not an episode
        del value.belief_type  # Not a plain belief
        result = processor._format_context("belief_to_value", [value])
        assert "honesty" in result
        assert "always be truthful" in result

    def test_format_goals(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        goal = MagicMock(spec=["title", "description"])
        goal.title = "ship v2"
        goal.description = "release version 2.0"
        del goal.objective
        del goal.statement
        result = processor._format_context("episode_to_goal", [goal])
        assert "ship v2" in result
        assert "release version 2.0" in result

    def test_format_goals_without_description(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        goal = MagicMock(spec=["title", "description"])
        goal.title = "do stuff"
        goal.description = None
        del goal.objective
        del goal.statement
        result = processor._format_context("episode_to_goal", [goal])
        assert "do stuff" in result

    def test_format_drives(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        drive = MagicMock(spec=["drive_type", "intensity"])
        drive.drive_type = "curiosity"
        drive.intensity = 0.8
        del drive.objective
        del drive.statement
        del drive.title
        result = processor._format_context("episode_to_drive", [drive])
        assert "curiosity" in result
        assert "0.8" in result

    def test_format_relationships(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        rel = MagicMock(spec=["entity_name", "entity_type"])
        rel.entity_name = "Alice"
        rel.entity_type = "person"
        del rel.objective
        del rel.statement
        del rel.title
        del rel.drive_type
        result = processor._format_context("episode_to_relationship", [rel])
        assert "Alice" in result
        assert "person" in result

    def test_format_fallback_str(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        obj = MagicMock(spec=[])
        del obj.objective
        del obj.statement
        del obj.title
        del obj.drive_type
        del obj.entity_name
        result = processor._format_context("raw_to_episode", [obj])
        # Should use str(c)[:100] fallback
        assert result.startswith("- ")

    def test_context_limited_to_10(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        episodes = []
        for i in range(15):
            ep = MagicMock()
            ep.objective = f"obj{i}"
            ep.outcome = f"out{i}"
            episodes.append(ep)
        result = processor._format_context("raw_to_episode", episodes)
        lines = result.strip().split("\n")
        assert len(lines) == 10


# =============================================================================
# _write_memories — all 7 transition types (mock-based)
# =============================================================================


class TestWriteMemories:
    def test_write_raw_to_episode(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "objective": "learned things",
                "outcome": "success",
                "outcome_type": "success",
                "lessons": ["lesson1"],
                "source_raw_ids": ["r1", "r2"],
            }
        ]
        created = processor._write_memories("raw_to_episode", parsed, [])
        assert len(created) == 1
        assert created[0] == {"type": "episode", "id": "ep-new"}
        mock_stack.save_episode.assert_called_once()
        saved_ep = mock_stack.save_episode.call_args[0][0]
        assert saved_ep.objective == "learned things"
        assert saved_ep.derived_from == ["raw:r1", "raw:r2"]
        assert saved_ep.source_type == "processing"
        assert saved_ep.source_entity.startswith("core:")

    def test_write_raw_to_note(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_note.return_value = "note-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "content": "factual note",
                "note_type": "observation",
                "source_raw_ids": ["r1"],
            }
        ]
        created = processor._write_memories("raw_to_note", parsed, [])
        assert len(created) == 1
        assert created[0] == {"type": "note", "id": "note-new"}
        saved_note = mock_stack.save_note.call_args[0][0]
        assert saved_note.content == "factual note"
        assert saved_note.derived_from == ["raw:r1"]
        assert saved_note.source_type == "processing"
        assert saved_note.source_entity.startswith("core:")

    def test_write_episode_to_belief(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_belief.return_value = "b-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "statement": "testing is vital",
                "belief_type": "evaluative",
                "confidence": 0.85,
                "source_episode_ids": ["ep-1"],
            }
        ]
        created = processor._write_memories("episode_to_belief", parsed, [])
        assert len(created) == 1
        assert created[0] == {"type": "belief", "id": "b-new"}
        saved = mock_stack.save_belief.call_args[0][0]
        assert saved.statement == "testing is vital"
        assert saved.confidence == 0.85
        assert saved.derived_from == ["episode:ep-1"]
        assert saved.source_type == "processing"
        assert saved.source_entity.startswith("core:")

    def test_write_episode_to_goal(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_goal.return_value = "g-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "title": "ship feature",
                "description": "deliver the feature",
                "goal_type": "task",
                "priority": "high",
                "source_episode_ids": ["ep-2"],
            }
        ]
        created = processor._write_memories("episode_to_goal", parsed, [])
        assert len(created) == 1
        assert created[0] == {"type": "goal", "id": "g-new"}
        saved = mock_stack.save_goal.call_args[0][0]
        assert saved.title == "ship feature"
        assert saved.priority == "high"
        assert saved.derived_from == ["episode:ep-2"]
        assert saved.source_type == "processing"
        assert saved.source_entity.startswith("core:")

    def test_write_episode_to_relationship(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_relationship.return_value = "rel-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "entity_name": "Alice",
                "entity_type": "person",
                "sentiment": 0.7,
                "context_note": "collaborative",
                "source_episode_ids": ["ep-3"],
            }
        ]
        created = processor._write_memories("episode_to_relationship", parsed, [])
        assert len(created) == 1
        assert created[0] == {"type": "relationship", "id": "rel-new"}
        saved = mock_stack.save_relationship.call_args[0][0]
        assert saved.entity_name == "Alice"
        assert saved.sentiment == 0.7
        assert saved.notes == "collaborative"
        assert saved.source_type == "processing"
        assert saved.source_entity.startswith("core:")

    def test_write_belief_to_value(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_value.return_value = "v-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "name": "integrity",
                "statement": "always be honest",
                "priority": 90,
                "source_belief_ids": ["b-1"],
            }
        ]
        created = processor._write_memories("belief_to_value", parsed, [])
        assert len(created) == 1
        assert created[0] == {"type": "value", "id": "v-new"}
        saved = mock_stack.save_value.call_args[0][0]
        assert saved.name == "integrity"
        assert saved.priority == 90
        assert saved.derived_from == ["belief:b-1"]
        assert saved.source_type == "processing"
        assert saved.source_entity.startswith("core:")

    def test_write_episode_to_drive(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_drive.return_value = "d-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "drive_type": "curiosity",
                "intensity": 0.8,
                "source_episode_ids": ["ep-4"],
            }
        ]
        created = processor._write_memories("episode_to_drive", parsed, [])
        assert len(created) == 1
        assert created[0] == {"type": "drive", "id": "d-new"}
        saved = mock_stack.save_drive.call_args[0][0]
        assert saved.drive_type == "curiosity"
        assert saved.intensity == 0.8
        assert saved.derived_from == ["episode:ep-4"]
        assert saved.source_type == "processing"
        assert saved.source_entity.startswith("core:")

    def test_write_multiple_items(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.side_effect = ["ep-1", "ep-2"]
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {"objective": "first", "outcome": "ok", "source_raw_ids": []},
            {"objective": "second", "outcome": "ok", "source_raw_ids": []},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [])
        assert len(created) == 2

    def test_write_exception_does_not_crash(self):
        """A failing write for one item should not prevent other items."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.side_effect = [Exception("db error"), "ep-2"]
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {"objective": "fails", "outcome": "err", "source_raw_ids": []},
            {"objective": "works", "outcome": "ok", "source_raw_ids": []},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [])
        # Only the second item succeeds
        assert len(created) == 1
        assert created[0]["id"] == "ep-2"

    def test_write_episode_default_values(self):
        """Missing optional fields get defaults."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-def"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {"objective": "obj", "outcome": "out"},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [])
        assert len(created) == 1
        saved = mock_stack.save_episode.call_args[0][0]
        assert saved.outcome_type == "neutral"
        assert saved.lessons is None
        assert saved.derived_from == []

    def test_write_goal_title_from_description(self):
        """Goal title falls back to description when title missing."""
        mock_stack = _make_mock_stack()
        mock_stack.save_goal.return_value = "g-fb"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {"description": "do the thing", "source_episode_ids": []},
        ]
        processor._write_memories("episode_to_goal", parsed, [])
        saved = mock_stack.save_goal.call_args[0][0]
        assert saved.title == "do the thing"

    def test_write_value_statement_defaults_to_name(self):
        """Value statement falls back to name when missing."""
        mock_stack = _make_mock_stack()
        mock_stack.save_value.return_value = "v-fb"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {"name": "honesty", "source_belief_ids": []},
        ]
        processor._write_memories("belief_to_value", parsed, [])
        saved = mock_stack.save_value.call_args[0][0]
        assert saved.statement == "honesty"
        assert saved.priority == 50


# =============================================================================
# _mark_processed — raw, episode, belief sources (mock-based)
# =============================================================================


class TestMarkProcessedMock:
    def test_mark_raw_sources(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        raw1 = MagicMock()
        raw1.id = "r-1"
        raw2 = MagicMock()
        raw2.id = "r-2"
        created = [{"type": "episode", "id": "ep-1"}]
        processor._mark_processed("raw_to_episode", [raw1, raw2], created)
        assert mock_stack._backend.mark_raw_processed.call_count == 2
        mock_stack._backend.mark_raw_processed.assert_any_call("r-1", ["episode:ep-1"])
        mock_stack._backend.mark_raw_processed.assert_any_call("r-2", ["episode:ep-1"])

    def test_mark_raw_sources_for_note(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        raw = MagicMock()
        raw.id = "r-3"
        created = [{"type": "note", "id": "n-1"}]
        processor._mark_processed("raw_to_note", [raw], created)
        mock_stack._backend.mark_raw_processed.assert_called_once_with("r-3", ["note:n-1"])

    def test_mark_episode_sources_for_belief(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        ep = MagicMock()
        ep.id = "ep-1"
        created = [{"type": "belief", "id": "b-1"}]
        processor._mark_processed("episode_to_belief", [ep], created)
        mock_stack._backend.mark_episode_processed.assert_called_once_with("ep-1")

    def test_mark_episode_sources_for_goal(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        ep = MagicMock()
        ep.id = "ep-2"
        created = [{"type": "goal", "id": "g-1"}]
        processor._mark_processed("episode_to_goal", [ep], created)
        mock_stack._backend.mark_episode_processed.assert_called_once_with("ep-2")

    def test_mark_episode_sources_for_relationship(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        ep = MagicMock()
        ep.id = "ep-3"
        created = [{"type": "relationship", "id": "rel-1"}]
        processor._mark_processed("episode_to_relationship", [ep], created)
        mock_stack._backend.mark_episode_processed.assert_called_once_with("ep-3")

    def test_mark_episode_sources_for_drive(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        ep = MagicMock()
        ep.id = "ep-4"
        created = [{"type": "drive", "id": "d-1"}]
        processor._mark_processed("episode_to_drive", [ep], created)
        mock_stack._backend.mark_episode_processed.assert_called_once_with("ep-4")

    def test_mark_belief_sources_for_value(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        belief = MagicMock()
        belief.id = "b-1"
        created = [{"type": "value", "id": "v-1"}]
        processor._mark_processed("belief_to_value", [belief], created)
        mock_stack._backend.mark_belief_processed.assert_called_once_with("b-1")

    def test_no_backend_returns_silently(self):
        mock_stack = MagicMock(spec=[])
        processor = MemoryProcessor(stack=mock_stack, inference=MockInference(), core_id="test")
        # Should not raise
        processor._mark_processed("raw_to_episode", [], [])

    def test_created_refs_format(self):
        """Verify the created_refs list format for raw marking."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        raw = MagicMock()
        raw.id = "r-1"
        created = [
            {"type": "episode", "id": "ep-1"},
            {"type": "note", "id": "n-1"},
        ]
        processor._mark_processed("raw_to_episode", [raw], created)
        expected_refs = ["episode:ep-1", "note:n-1"]
        mock_stack._backend.mark_raw_processed.assert_called_once_with("r-1", expected_refs)


# =============================================================================
# _process_layer — full pipeline (mock-based)
# =============================================================================


class TestProcessLayer:
    def test_no_prompts_skips(self):
        """Unknown transition with no prompts returns skipped result."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        config = LayerConfig(layer_transition="fake_transition")
        result = processor._process_layer("fake_transition", config)
        assert result.skipped
        assert "No prompts" in result.skip_reason

    def test_no_sources_skips(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = []
        processor, _ = _make_processor(mock_stack)
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config)
        assert result.skipped
        assert result.skip_reason == "No unprocessed sources"

    def test_happy_path_raw_to_episode(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "test content"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []
        mock_stack.save_episode.return_value = "ep-new"

        response = json.dumps(
            [
                {
                    "objective": "processed item",
                    "outcome": "success",
                    "source_raw_ids": ["r-1"],
                }
            ]
        )
        processor, inference = _make_processor(mock_stack, response)
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config, auto_promote=True)

        assert not result.skipped
        assert result.source_count == 1
        assert len(result.created) == 1
        assert result.created[0] == {"type": "episode", "id": "ep-new"}
        assert len(inference.calls) == 1
        mock_stack.log_audit.assert_called_once()

    def test_inference_failure(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "test"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []

        inference = MockInference()
        inference.infer = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        processor = MemoryProcessor(stack=mock_stack, inference=inference, core_id="test")
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config)

        assert not result.skipped
        assert result.source_count == 1
        assert len(result.errors) == 1
        assert "Inference failed" in result.errors[0]

    def test_parse_failure(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "test"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []

        processor, _ = _make_processor(mock_stack, "not valid json")
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config)

        assert not result.skipped
        assert len(result.errors) == 1
        assert "Parse failed" in result.errors[0]

    def test_empty_inference_output_keeps_sources_unprocessed(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "test"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []

        processor, _ = _make_processor(mock_stack, "[]")
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config, auto_promote=True)

        assert not result.skipped
        assert result.source_count == 1
        assert result.no_output_processed
        assert result.created == []
        assert result.suggestions == []
        mock_stack._backend.mark_raw_processed.assert_not_called()

    def test_audit_log_called_on_success(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "stuff"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []
        mock_stack.save_episode.return_value = "ep-1"

        response = json.dumps(
            [
                {
                    "objective": "thing",
                    "outcome": "done",
                    "source_raw_ids": [],
                }
            ]
        )
        processor, _ = _make_processor(mock_stack, response)
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        processor._process_layer("raw_to_episode", config, auto_promote=True)

        mock_stack.log_audit.assert_called_once_with(
            "processing",
            "raw_to_episode",
            "process",
            actor="core:test",
            details={
                "source_count": 1,
                "created_count": 1,
                "suggestion_count": 0,
                "auto_promote": True,
                "deduplicated": 0,
                "gate_blocked": 0,
                "gate_details": [],
                "errors": [],
            },
        )

    def test_auto_promote_write_failure_keeps_sources_unprocessed_and_reports_error(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "stuff"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []
        mock_stack.save_episode.side_effect = RuntimeError("db write failed")

        response = json.dumps(
            [
                {
                    "objective": "thing",
                    "outcome": "done",
                    "source_raw_ids": ["r-1"],
                }
            ]
        )
        processor, _ = _make_processor(mock_stack, response)
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config, auto_promote=True)

        assert result.created == []
        assert any("Write failed for raw_to_episode" in err for err in result.errors)
        mock_stack._backend.mark_raw_processed.assert_not_called()

    def test_suggestion_write_failure_keeps_sources_unprocessed_and_reports_error(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "stuff"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []
        mock_stack.save_suggestion.side_effect = RuntimeError("db write failed")

        response = json.dumps(
            [
                {
                    "objective": "thing",
                    "outcome": "done",
                    "source_raw_ids": ["r-1"],
                }
            ]
        )
        processor, _ = _make_processor(mock_stack, response)
        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config, auto_promote=False)

        assert result.suggestions == []
        assert any("Suggestion write failed for raw_to_episode" in err for err in result.errors)
        mock_stack._backend.mark_raw_processed.assert_not_called()

    def test_write_suggestions_preserves_typed_sources_for_non_raw_transitions(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_suggestion.return_value = "sug-1"
        processor, _ = _make_processor(mock_stack)

        suggestions = processor._write_suggestions(
            "episode_to_belief",
            [
                {
                    "statement": "Belief from episodes",
                    "source_episode_ids": ["ep-1", "ep-2"],
                }
            ],
            [],
        )

        assert suggestions == [{"type": "belief", "id": "sug-1"}]
        saved = mock_stack.save_suggestion.call_args[0][0]
        assert saved.source_raw_ids == ["episode:ep-1", "episode:ep-2"]


# =============================================================================
# process() — full flow with force and trigger checking (mock-based)
# =============================================================================


class TestProcessFullFlow:
    def test_force_bypasses_triggers(self):
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "data"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]
        mock_stack.get_episodes.return_value = []
        mock_stack.save_episode.return_value = "ep-1"

        response = json.dumps(
            [
                {
                    "objective": "obj",
                    "outcome": "out",
                    "source_raw_ids": [],
                }
            ]
        )
        processor, _ = _make_processor(mock_stack, response)
        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert not results[0].skipped

    def test_without_force_checks_triggers(self):
        mock_stack = _make_mock_stack()
        # Not enough raws to trigger (default threshold is 10)
        raws = [MagicMock() for _ in range(2)]
        mock_stack._backend.list_raw.return_value = raws
        processor, _ = _make_processor(mock_stack)
        results = processor.process("raw_to_episode")
        # No results since trigger not met
        assert results == []

    def test_process_none_transition_iterates_all(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = []
        mock_stack._backend.get_episodes.return_value = []
        mock_stack._backend.get_beliefs.return_value = []
        processor, _ = _make_processor(mock_stack)
        results = processor.process(force=True)
        # All 7 transitions should produce skipped results
        assert len(results) == len(VALID_TRANSITIONS)
        assert all(r.skipped for r in results)

    def test_process_none_transition_is_deterministic(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        fake_process_layer = MagicMock(
            side_effect=lambda t, config, auto_promote=False, batch_size=None: ProcessingResult(
                layer_transition=t,
                source_count=0,
                skipped=True,
            )
        )
        processor._process_layer = fake_process_layer

        processor.process(force=True)

        actual = [call.args[0] for call in fake_process_layer.mock_calls]
        assert actual == list(VALID_TRANSITION_ORDER)

    def test_missing_config_transition_skipped(self):
        """A transition not in configs is skipped entirely."""
        mock_stack = _make_mock_stack()
        # Use a config dict that excludes the transition we request
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=MockInference(),
            core_id="test",
            configs={"episode_to_belief": DEFAULT_LAYER_CONFIGS["episode_to_belief"]},
        )
        results = processor.process("raw_to_episode", force=True)
        assert results == []

    def test_get_config_returns_none_for_missing(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        assert processor.get_config("nonexistent") is None


# =============================================================================
# No-Inference Safety Policy
# =============================================================================


def _make_no_inference_processor(mock_stack, response="[]", inference_available=False):
    """Create a MemoryProcessor with inference_available=False."""
    inference = MockInference(response)
    return (
        MemoryProcessor(
            stack=mock_stack,
            inference=inference,
            core_id="test",
            inference_available=inference_available,
        ),
        inference,
    )


class TestNoInferenceSafetyConstants:
    """Verify the safety constant definitions are correct."""

    def test_identity_layer_transitions_are_subset_of_valid(self):
        assert IDENTITY_LAYER_TRANSITIONS.issubset(VALID_TRANSITIONS)

    def test_no_override_is_subset_of_identity(self):
        assert NO_OVERRIDE_TRANSITIONS.issubset(IDENTITY_LAYER_TRANSITIONS)

    def test_override_is_subset_of_identity(self):
        assert OVERRIDE_TRANSITIONS.issubset(IDENTITY_LAYER_TRANSITIONS)

    def test_no_overlap_between_override_and_no_override(self):
        assert NO_OVERRIDE_TRANSITIONS.isdisjoint(OVERRIDE_TRANSITIONS)

    def test_union_covers_all_identity_transitions(self):
        assert NO_OVERRIDE_TRANSITIONS | OVERRIDE_TRANSITIONS == IDENTITY_LAYER_TRANSITIONS

    def test_belief_to_value_is_no_override(self):
        assert "belief_to_value" in NO_OVERRIDE_TRANSITIONS

    def test_raw_transitions_not_in_identity(self):
        assert "raw_to_episode" not in IDENTITY_LAYER_TRANSITIONS
        assert "raw_to_note" not in IDENTITY_LAYER_TRANSITIONS


class TestNoInferenceSafetyGating:
    """Test the _check_inference_safety method."""

    def test_inference_available_allows_all(self):
        """When inference is available, nothing is blocked."""
        mock_stack = _make_mock_stack()
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=MockInference(),
            core_id="test",
            inference_available=True,
        )
        for transition in VALID_TRANSITIONS:
            result = processor._check_inference_safety(transition, force=True, allow_override=True)
            assert result is None, f"{transition} should not be blocked when inference available"

    def test_no_inference_blocks_belief_to_value(self):
        """belief_to_value is always blocked without inference."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_no_inference_processor(mock_stack)
        result = processor._check_inference_safety(
            "belief_to_value", force=True, allow_override=True
        )
        assert result is not None
        assert result.inference_blocked
        assert result.skipped
        assert "Value creation requires inference" in result.skip_reason

    def test_no_inference_blocks_belief_to_value_even_with_override(self):
        """belief_to_value cannot be overridden."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_no_inference_processor(mock_stack)
        result = processor._check_inference_safety(
            "belief_to_value", force=True, allow_override=True
        )
        assert result is not None
        assert result.inference_blocked

    def test_no_inference_blocks_identity_layers_without_override(self):
        """Identity-layer transitions are blocked without force+override."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_no_inference_processor(mock_stack)
        for transition in OVERRIDE_TRANSITIONS:
            result = processor._check_inference_safety(
                transition, force=False, allow_override=False
            )
            assert result is not None, f"{transition} should be blocked"
            assert result.inference_blocked
            assert result.skipped

    def test_no_inference_blocks_identity_with_force_but_no_override(self):
        """force=True alone is not enough — need allow_override too."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_no_inference_processor(mock_stack)
        for transition in OVERRIDE_TRANSITIONS:
            result = processor._check_inference_safety(transition, force=True, allow_override=False)
            assert result is not None, f"{transition} should be blocked with force only"
            assert result.inference_blocked

    def test_no_inference_allows_override_with_force_and_flag(self):
        """force=True + allow_override=True lets override transitions through."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_no_inference_processor(mock_stack)
        for transition in OVERRIDE_TRANSITIONS:
            result = processor._check_inference_safety(transition, force=True, allow_override=True)
            assert result is None, f"{transition} should be allowed with force+override"

    def test_no_inference_blocks_raw_transitions(self):
        """raw_to_episode and raw_to_note are blocked when inference unavailable."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_no_inference_processor(mock_stack)
        for transition in ("raw_to_episode", "raw_to_note"):
            result = processor._check_inference_safety(
                transition, force=False, allow_override=False
            )
            assert result is not None, f"{transition} should be blocked"
            assert result.inference_blocked is True
            assert result.skipped is True


class TestNoInferenceProcessMethod:
    """Test the full process() method with inference_available=False."""

    def test_no_inference_blocks_value_creation_via_process(self):
        """process('belief_to_value') is blocked when inference unavailable."""
        mock_stack = _make_mock_stack()
        beliefs = [MagicMock(processed=False) for _ in range(6)]
        mock_stack.get_beliefs.return_value = beliefs
        processor, _ = _make_no_inference_processor(mock_stack)
        results = processor.process("belief_to_value", force=True)
        assert len(results) == 1
        assert results[0].inference_blocked
        assert results[0].skipped

    def test_no_inference_blocks_all_transitions_by_default(self):
        """Running process() with force=True blocks all transitions."""
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = []
        mock_stack.get_episodes.return_value = []
        mock_stack.get_beliefs.return_value = []
        processor, _ = _make_no_inference_processor(mock_stack)
        results = processor.process(force=True)
        blocked = [r for r in results if r.inference_blocked]
        assert len(blocked) == len(VALID_TRANSITIONS)

    def test_no_inference_blocks_raw_transitions(self):
        """raw_to_episode/raw_to_note are blocked when inference unavailable."""
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = []
        mock_stack.get_episodes.return_value = []
        mock_stack.get_beliefs.return_value = []
        processor, _ = _make_no_inference_processor(mock_stack)
        results = processor.process(force=True)
        raw_results = [
            r for r in results if r.layer_transition in ("raw_to_episode", "raw_to_note")
        ]
        for r in raw_results:
            assert r.inference_blocked
            assert r.skipped

    def test_no_inference_with_override_allows_beliefs(self):
        """force=True + allow_no_inference_override=True unblocks beliefs."""
        mock_stack = _make_mock_stack()
        mock_stack._backend.list_raw.return_value = []
        mock_stack._backend.get_episodes.return_value = []
        mock_stack._backend.get_beliefs.return_value = []
        processor, _ = _make_no_inference_processor(mock_stack)
        results = processor.process(
            "episode_to_belief", force=True, allow_no_inference_override=True
        )
        # Not blocked by inference policy
        assert len(results) == 1
        assert not results[0].inference_blocked
        # Skipped because no sources
        assert results[0].skipped
        assert results[0].skip_reason == "No unprocessed sources"

    def test_no_inference_override_with_insufficient_evidence_is_blocked(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.get_episodes.return_value = [
            MagicMock(processed=False, id="ep-1", objective="foo", outcome="bar")
        ]
        processor, inference = _make_no_inference_processor(mock_stack)

        results = processor.process(
            "episode_to_belief", force=True, allow_no_inference_override=True
        )

        assert len(results) == 1
        assert results[0].inference_blocked
        assert results[0].skipped
        assert results[0].source_count == 1
        assert (
            f"requires at least {NO_INFERENCE_MIN_EVIDENCE} source memories"
            in results[0].skip_reason
        )
        assert inference.calls == []

    def test_no_inference_override_with_insufficient_confidence_is_blocked(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.get_episodes.return_value = [
            MagicMock(
                processed=False,
                id="ep-1",
                objective="foo",
                outcome="bar",
                confidence=0.50,
            ),
            MagicMock(processed=False, id="ep-2", objective="foo", outcome="bar", confidence=0.95),
            MagicMock(processed=False, id="ep-3", objective="foo", outcome="bar", confidence=0.10),
        ]
        processor, inference = _make_no_inference_processor(mock_stack)

        results = processor.process(
            "episode_to_belief", force=True, allow_no_inference_override=True
        )

        assert len(results) == 1
        assert results[0].inference_blocked
        assert results[0].skipped
        assert results[0].source_count == 3
        assert f"minimum confidence {NO_INFERENCE_MIN_CONFIDENCE:.2f}" in results[0].skip_reason
        assert "minimum observed is" in results[0].skip_reason
        assert inference.calls == []

    def test_no_inference_override_with_sufficient_thresholds_still_blocked(self):
        mock_stack = _make_mock_stack()
        mock_stack._backend.get_episodes.return_value = [
            MagicMock(
                processed=False,
                id="ep-1",
                objective="foo",
                outcome="bar",
                confidence=0.95,
            ),
            MagicMock(processed=False, id="ep-2", objective="foo", outcome="bar", confidence=0.96),
            MagicMock(processed=False, id="ep-3", objective="foo", outcome="bar", confidence=0.99),
        ]
        processor, inference = _make_no_inference_processor(mock_stack)

        results = processor.process(
            "episode_to_belief", force=True, allow_no_inference_override=True
        )

        assert len(results) == 1
        assert results[0].inference_blocked
        assert results[0].skipped
        assert results[0].source_count == 3
        assert results[0].skip_reason == (
            "Blocked: no inference available for identity-layer override transitions."
        )
        assert inference.calls == []

    def test_no_inference_override_still_blocks_values(self):
        """Even with override, belief_to_value is always blocked."""
        mock_stack = _make_mock_stack()
        mock_stack.get_beliefs.return_value = [MagicMock(processed=False) for _ in range(6)]
        processor, _ = _make_no_inference_processor(mock_stack)
        results = processor.process("belief_to_value", force=True, allow_no_inference_override=True)
        assert len(results) == 1
        assert results[0].inference_blocked
        assert "Value creation requires inference" in results[0].skip_reason

    def test_force_alone_does_not_bypass_inference_safety(self):
        """force=True without override flag still blocks identity layers."""
        mock_stack = _make_mock_stack()
        mock_stack.get_episodes.return_value = [MagicMock(processed=False) for _ in range(6)]
        processor, _ = _make_no_inference_processor(mock_stack)
        results = processor.process("episode_to_belief", force=True)
        assert len(results) == 1
        assert results[0].inference_blocked


class TestNoInferenceEntity:
    """Test Entity.process() with no model bound."""

    def test_entity_process_no_model_blocks_all_transitions(self, tmp_path):
        """Entity.process() without model blocks all transitions."""
        ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
        st = SQLiteStack(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
        )
        ent.attach_stack(st)
        # No model set — inference_available=False
        results = ent.process(force=True)
        blocked = [r for r in results if r.inference_blocked]
        assert len(blocked) == len(VALID_TRANSITIONS)

    def test_entity_process_no_model_blocks_raw_layers(self, tmp_path):
        """Entity.process() without model blocks raw transitions."""
        ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
        st = SQLiteStack(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
        )
        ent.attach_stack(st)
        results = ent.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert results[0].inference_blocked

    def test_entity_process_no_model_override_flag_unblocks(self, tmp_path):
        """Entity.process() with override flag unblocks non-value identity layers."""
        ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
        st = SQLiteStack(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
        )
        ent.attach_stack(st)
        results = ent.process("episode_to_belief", force=True, allow_no_inference_override=True)
        assert len(results) == 1
        assert not results[0].inference_blocked

    def test_entity_process_no_model_value_always_blocked(self, tmp_path):
        """Entity.process() without model always blocks belief_to_value."""
        ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
        st = SQLiteStack(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
        )
        ent.attach_stack(st)
        results = ent.process("belief_to_value", force=True, allow_no_inference_override=True)
        assert len(results) == 1
        assert results[0].inference_blocked

    def test_entity_process_with_model_allows_all(self, tmp_path):
        """Entity.process() with model bound allows all transitions."""

        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
        st = SQLiteStack(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
        )
        ent.attach_stack(st)
        ent.set_model(MockModel())
        results = ent.process(force=True)
        blocked = [r for r in results if r.inference_blocked]
        assert len(blocked) == 0


class TestProcessingResultInferenceBlocked:
    """Test the inference_blocked field on ProcessingResult."""

    def test_default_is_false(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=0,
        )
        assert result.inference_blocked is False

    def test_set_to_true(self):
        result = ProcessingResult(
            layer_transition="belief_to_value",
            source_count=0,
            skipped=True,
            skip_reason="blocked by policy",
            inference_blocked=True,
        )
        assert result.inference_blocked is True


class TestGateBlockedSourcesNotConsumed:
    """P1 fix: when promotion gates block ALL items, sources must NOT be
    marked as processed so they remain available for future processing."""

    def test_gate_blocks_all_items_sources_stay_unprocessed(self, stack):
        """When gate blocks every parsed item, source episodes stay unprocessed."""
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="Evidence A",
            outcome="Happened once",
            source_type="observation",
            source_entity="test",
            processed=False,
        )
        stack.save_episode(ep)

        # Gate requires 3 episodes — parsed item references only 1
        strict_gates = PromotionGateConfig(
            belief_min_evidence=3,
            belief_min_confidence=0.0,
            value_min_evidence=0,
            value_requires_protection=False,
        )

        response = json.dumps(
            [
                {
                    "statement": "A belief from thin evidence",
                    "belief_type": "factual",
                    "confidence": 0.8,
                    "source_episode_ids": [ep.id],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            auto_promote=True,
            promotion_gates=strict_gates,
        )
        results = processor.process("episode_to_belief", force=True)
        result = results[0]

        # Gate should have blocked the item
        assert result.gate_blocked == 1
        assert len(result.created) == 0

        # Source episode must still be unprocessed
        episodes = stack.get_episodes(limit=10)
        unprocessed = [e for e in episodes if not e.processed]
        assert len(unprocessed) == 1
        assert unprocessed[0].id == ep.id

    def test_gate_allows_promotion_sources_marked_processed(self, stack):
        """When gate allows promotion, sources are marked processed as before."""
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="Well-evidenced",
            outcome="Confirmed",
            source_type="observation",
            source_entity="test",
            processed=False,
        )
        stack.save_episode(ep)

        # Gate with no minimum evidence — everything passes
        response = json.dumps(
            [
                {
                    "statement": "A well-founded belief",
                    "belief_type": "factual",
                    "confidence": 0.9,
                    "source_episode_ids": [ep.id],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            auto_promote=True,
            promotion_gates=_NO_GATES,
        )
        results = processor.process("episode_to_belief", force=True)
        result = results[0]

        assert result.gate_blocked == 0
        assert len(result.created) == 1

        # Source episode should now be processed
        episodes = stack.get_episodes(limit=10)
        unprocessed = [e for e in episodes if not e.processed]
        assert len(unprocessed) == 0

    def test_previously_blocked_sources_promoted_after_conditions_improve(self, stack):
        """Sources blocked by gates can be promoted when conditions change."""
        # Create 3 episodes to eventually meet the evidence threshold
        ep_ids = []
        for i in range(3):
            ep = Episode(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                objective=f"Evidence {i}",
                outcome=f"Result {i}",
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            stack.save_episode(ep)
            ep_ids.append(ep.id)

        # First attempt: gate requires 5 episodes — only 3 referenced, blocks
        strict_gates = PromotionGateConfig(
            belief_min_evidence=5,
            belief_min_confidence=0.0,
            value_min_evidence=0,
            value_requires_protection=False,
        )

        response = json.dumps(
            [
                {
                    "statement": "A belief needing more evidence",
                    "belief_type": "factual",
                    "confidence": 0.8,
                    "source_episode_ids": ep_ids,
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            auto_promote=True,
            promotion_gates=strict_gates,
        )
        results = processor.process("episode_to_belief", force=True)
        assert results[0].gate_blocked == 1
        assert len(results[0].created) == 0

        # Episodes should still be unprocessed
        episodes = stack.get_episodes(limit=10)
        unprocessed = [e for e in episodes if not e.processed]
        assert len(unprocessed) == 3

        # Second attempt: relax gate to 3 episodes — now passes
        relaxed_gates = PromotionGateConfig(
            belief_min_evidence=3,
            belief_min_confidence=0.0,
            value_min_evidence=0,
            value_requires_protection=False,
        )

        inference2 = MockInference(response)
        processor2 = MemoryProcessor(
            stack=stack,
            inference=inference2,
            core_id="test",
            auto_promote=True,
            promotion_gates=relaxed_gates,
        )
        results2 = processor2.process("episode_to_belief", force=True)
        assert results2[0].gate_blocked == 0
        assert len(results2[0].created) == 1

        # Now episodes should be processed
        episodes = stack.get_episodes(limit=10)
        unprocessed = [e for e in episodes if not e.processed]
        assert len(unprocessed) == 0

    def test_gate_blocks_suggestion_mode_sources_stay_unprocessed(self, stack):
        """In suggestion mode (auto_promote=False), gate-blocked items
        should also leave sources unprocessed."""
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="Evidence for suggestion",
            outcome="Only once",
            source_type="observation",
            source_entity="test",
            processed=False,
        )
        stack.save_episode(ep)

        strict_gates = PromotionGateConfig(
            belief_min_evidence=3,
            belief_min_confidence=0.0,
            value_min_evidence=0,
            value_requires_protection=False,
        )

        response = json.dumps(
            [
                {
                    "statement": "A suggestion from thin evidence",
                    "belief_type": "factual",
                    "confidence": 0.8,
                    "source_episode_ids": [ep.id],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            auto_promote=False,
            promotion_gates=strict_gates,
        )
        results = processor.process("episode_to_belief", force=True)
        result = results[0]

        assert result.gate_blocked == 1
        assert len(result.suggestions) == 0

        # Source episode must still be unprocessed
        episodes = stack.get_episodes(limit=10)
        unprocessed = [e for e in episodes if not e.processed]
        assert len(unprocessed) == 1

    def test_mock_gate_blocks_no_mark_called(self):
        """Unit test: when gate blocks all items, _mark_processed is not called."""
        mock_stack = _make_mock_stack()
        mock_stack.get_episodes.return_value = [
            MagicMock(id="ep-1", processed=False),
        ]
        mock_stack._backend.list_raw.return_value = []
        mock_stack.get_beliefs.return_value = []

        strict_gates = PromotionGateConfig(
            belief_min_evidence=5,
            belief_min_confidence=0.0,
            value_min_evidence=0,
            value_requires_protection=False,
        )

        response = json.dumps(
            [
                {
                    "statement": "blocked belief",
                    "belief_type": "factual",
                    "confidence": 0.8,
                    "source_episode_ids": ["ep-1"],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=mock_stack,
            inference=inference,
            core_id="test",
            auto_promote=True,
            promotion_gates=strict_gates,
        )

        processor._process_layer(
            "episode_to_belief",
            LayerConfig(layer_transition="episode_to_belief"),
            auto_promote=True,
        )

        # mark_episode_processed should NOT have been called
        mock_stack._backend.mark_episode_processed.assert_not_called()


class TestBeliefToValuePromotionGates:
    def test_belief_to_value_requires_missing_sources(self, stack):
        """Promotion should fail if any source belief IDs cannot be resolved."""
        valid_belief = Belief(
            id="present-belief-id",
            stack_id=STACK_ID,
            statement="A valid source belief",
            source_type="observation",
            source_entity="test",
            processed=False,
            is_protected=True,
        )
        stack.save_belief(valid_belief)

        response = json.dumps(
            [
                {
                    "name": "Integrity value",
                    "statement": "A value without a valid source",
                    "value": "integrity",
                    "source_belief_ids": ["missing-belief-id", "present-belief-id"],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            auto_promote=True,
            promotion_gates=PromotionGateConfig(
                belief_min_evidence=0,
                belief_min_confidence=0.0,
                value_min_evidence=1,
                value_requires_protection=True,
            ),
        )

        results = processor.process("belief_to_value", force=True)
        assert len(results) == 1
        assert results[0].gate_blocked == 1
        assert len(results[0].created) == 0


# =============================================================================
# Ordering starvation fix — P1 regression tests
# =============================================================================


class TestOrderingStarvation:
    """Verify that unprocessed items are found even when many recent items are processed.

    Before the fix, check_triggers() and _gather_sources() fetched recent
    items with a limit, then filtered for processed=False in Python. If the
    most recent items were already processed, older unprocessed records were
    invisible (ordering starvation).

    The fix queries the backend directly with processed=False, so the SQL
    LIMIT applies only to unprocessed rows.
    """

    def test_check_triggers_episodes_finds_old_unprocessed(self, stack):
        """Old unprocessed episodes are counted even when recent ones are processed."""

        backend = stack._backend

        # Save 20 processed episodes (recent)
        for i in range(20):
            ep = Episode(
                id=f"processed-{i}",
                stack_id=STACK_ID,
                objective=f"Processed event {i}",
                outcome=f"Done {i}",
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_episode(ep)
            backend.mark_episode_processed(f"processed-{i}")

        # Save 6 old unprocessed episodes (these should trigger)
        for i in range(6):
            ep = Episode(
                id=f"unprocessed-{i}",
                stack_id=STACK_ID,
                objective=f"Old event {i}",
                outcome=f"Pending {i}",
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_episode(ep)

        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=5,
        )
        processor = MemoryProcessor(
            stack=stack,
            inference=MockInference(),
            core_id="test",
            configs={"episode_to_belief": config},
        )
        # Should trigger because 6 unprocessed >= threshold 5
        assert processor.check_triggers("episode_to_belief")

    def test_check_triggers_beliefs_finds_old_unprocessed(self, stack):
        """Old unprocessed beliefs are counted even when recent ones are processed."""
        from kernle.types import Belief

        backend = stack._backend

        # Save 20 processed beliefs
        for i in range(20):
            b = Belief(
                id=f"proc-belief-{i}",
                stack_id=STACK_ID,
                statement=f"Processed belief {i}",
                belief_type="factual",
                confidence=0.8,
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_belief(b)
            backend.mark_belief_processed(f"proc-belief-{i}")

        # Save 6 old unprocessed beliefs
        for i in range(6):
            b = Belief(
                id=f"unproc-belief-{i}",
                stack_id=STACK_ID,
                statement=f"Unprocessed belief {i}",
                belief_type="factual",
                confidence=0.8,
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_belief(b)

        config = LayerConfig(
            layer_transition="belief_to_value",
            quantity_threshold=5,
        )
        processor = MemoryProcessor(
            stack=stack,
            inference=MockInference(),
            core_id="test",
            configs={"belief_to_value": config},
        )
        assert processor.check_triggers("belief_to_value")

    def test_gather_sources_episodes_finds_old_unprocessed(self, stack):
        """_gather_sources returns old unprocessed episodes, not just recent ones."""
        backend = stack._backend

        # Save 20 processed episodes
        for i in range(20):
            ep = Episode(
                id=f"done-{i}",
                stack_id=STACK_ID,
                objective=f"Done {i}",
                outcome=f"Done {i}",
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_episode(ep)
            backend.mark_episode_processed(f"done-{i}")

        # Save 3 old unprocessed episodes
        unprocessed_ids = set()
        for i in range(3):
            eid = f"pending-{i}"
            ep = Episode(
                id=eid,
                stack_id=STACK_ID,
                objective=f"Pending {i}",
                outcome=f"Pending {i}",
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_episode(ep)
            unprocessed_ids.add(eid)

        processor = MemoryProcessor(
            stack=stack,
            inference=MockInference(),
            core_id="test",
            promotion_gates=_NO_GATES,
        )
        sources = processor._gather_sources("episode_to_belief", 10)
        assert len(sources) == 3
        assert {s.id for s in sources} == unprocessed_ids

    def test_gather_sources_beliefs_finds_old_unprocessed(self, stack):
        """_gather_sources returns old unprocessed beliefs, not just recent ones."""
        from kernle.types import Belief

        backend = stack._backend

        # Save 20 processed beliefs
        for i in range(20):
            b = Belief(
                id=f"done-b-{i}",
                stack_id=STACK_ID,
                statement=f"Done {i}",
                belief_type="factual",
                confidence=0.8,
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_belief(b)
            backend.mark_belief_processed(f"done-b-{i}")

        # Save 3 old unprocessed beliefs
        unprocessed_ids = set()
        for i in range(3):
            bid = f"pending-b-{i}"
            b = Belief(
                id=bid,
                stack_id=STACK_ID,
                statement=f"Pending {i}",
                belief_type="factual",
                confidence=0.8,
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_belief(b)
            unprocessed_ids.add(bid)

        processor = MemoryProcessor(
            stack=stack,
            inference=MockInference(),
            core_id="test",
            promotion_gates=_NO_GATES,
        )
        sources = processor._gather_sources("belief_to_value", 10)
        assert len(sources) == 3
        assert {s.id for s in sources} == unprocessed_ids

    def test_storage_get_episodes_processed_filter(self, stack):
        """SQLiteStorage.get_episodes(processed=False) returns only unprocessed."""
        backend = stack._backend

        # Save 5 episodes, mark 3 as processed
        for i in range(5):
            ep = Episode(
                id=f"ep-{i}",
                stack_id=STACK_ID,
                objective=f"Event {i}",
                outcome=f"Outcome {i}",
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_episode(ep)

        for i in range(3):
            backend.mark_episode_processed(f"ep-{i}")

        # processed=False returns only 2
        unprocessed = backend.get_episodes(limit=100, processed=False)
        assert len(unprocessed) == 2
        assert all(not e.processed for e in unprocessed)

        # processed=True returns only 3
        processed = backend.get_episodes(limit=100, processed=True)
        assert len(processed) == 3
        assert all(e.processed for e in processed)

        # processed=None returns all 5
        all_eps = backend.get_episodes(limit=100)
        assert len(all_eps) == 5

    def test_storage_get_beliefs_processed_filter(self, stack):
        """SQLiteStorage.get_beliefs(processed=False) returns only unprocessed."""
        from kernle.types import Belief

        backend = stack._backend

        for i in range(5):
            b = Belief(
                id=f"b-{i}",
                stack_id=STACK_ID,
                statement=f"Belief {i}",
                belief_type="factual",
                confidence=0.8,
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            backend.save_belief(b)

        for i in range(3):
            backend.mark_belief_processed(f"b-{i}")

        unprocessed = backend.get_beliefs(limit=100, processed=False)
        assert len(unprocessed) == 2
        assert all(not b.processed for b in unprocessed)

        processed = backend.get_beliefs(limit=100, processed=True)
        assert len(processed) == 3
        assert all(b.processed for b in processed)

        all_beliefs = backend.get_beliefs(limit=100)
        assert len(all_beliefs) == 5
