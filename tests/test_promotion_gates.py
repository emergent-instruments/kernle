"""Tests for promotion gates (Issue #405).

Promotion gates enforce evidence-count, confidence floor, and trust floor
criteria before allowing memories to be promoted up the hierarchy.

Tests cover:
- PromotionGateConfig defaults and custom values
- PromotionGateResult summary formatting
- _check_promotion_gate: belief gates (evidence count, confidence floor)
- _check_promotion_gate: value gates (evidence count, protection requirement)
- _check_promotion_gate: non-gated transitions pass through
- Gate integration with _write_memories (items blocked vs allowed)
- Gate integration with _write_suggestions (items blocked vs allowed)
- Gate results propagated to ProcessingResult
- Gate configuration via stack settings in Entity.process()
- CLI output includes gate status
- MCP handler includes gate status
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from kernle.entity import Entity
from kernle.processing import (
    DEFAULT_LAYER_CONFIGS,
    DEFAULT_PROMOTION_GATES,
    MemoryProcessor,
    ProcessingResult,
    PromotionGateConfig,
    PromotionGateResult,
)
from kernle.stack.sqlite_stack import Stack

STACK_ID = "test-stack"


# =============================================================================
# Fixtures
# =============================================================================


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


def _make_mock_stack():
    """Create a MagicMock stack with _backend for MemoryProcessor tests."""
    mock_stack = MagicMock()
    mock_stack.stack_id = STACK_ID
    mock_stack._backend = MagicMock()
    return mock_stack


def _make_processor(mock_stack, response="[]", gates=None):
    """Create a MemoryProcessor with a mock stack, inference, and optional gates."""
    inference = MockInference(response)
    return (
        MemoryProcessor(
            stack=mock_stack,
            inference=inference,
            core_id="test",
            promotion_gates=gates,
        ),
        inference,
    )


@pytest.fixture
def stack(tmp_path):
    return Stack.from_sqlite(
        STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
    )


# =============================================================================
# PromotionGateConfig
# =============================================================================


class TestPromotionGateConfig:
    def test_default_values(self):
        config = PromotionGateConfig()
        assert config.belief_min_evidence == 3
        assert config.belief_min_confidence == 0.6
        assert config.value_min_evidence == 5
        assert config.value_requires_protection is True

    def test_custom_values(self):
        config = PromotionGateConfig(
            belief_min_evidence=5,
            belief_min_confidence=0.8,
            value_min_evidence=10,
            value_requires_protection=False,
        )
        assert config.belief_min_evidence == 5
        assert config.belief_min_confidence == 0.8
        assert config.value_min_evidence == 10
        assert config.value_requires_protection is False

    def test_default_singleton_exists(self):
        assert DEFAULT_PROMOTION_GATES is not None
        assert DEFAULT_PROMOTION_GATES.belief_min_evidence == 3


# =============================================================================
# PromotionGateResult
# =============================================================================


class TestPromotionGateResult:
    def test_passed_summary(self):
        result = PromotionGateResult(passed=True, transition="episode_to_belief")
        assert result.summary == "promotion gate passed"

    def test_failed_summary_single(self):
        result = PromotionGateResult(
            passed=False,
            transition="episode_to_belief",
            failures=["insufficient evidence: 1 episodes (need >= 3)"],
        )
        assert "insufficient evidence" in result.summary

    def test_failed_summary_multiple(self):
        result = PromotionGateResult(
            passed=False,
            transition="episode_to_belief",
            failures=[
                "insufficient evidence: 1 episodes (need >= 3)",
                "confidence too low: 0.40 (need >= 0.60)",
            ],
        )
        assert "insufficient evidence" in result.summary
        assert "confidence too low" in result.summary
        assert ";" in result.summary


# =============================================================================
# _check_promotion_gate: Belief Gates
# =============================================================================


class TestBeliefPromotionGate:
    def test_passes_with_enough_evidence_and_confidence(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {
            "statement": "testing matters",
            "belief_type": "evaluative",
            "confidence": 0.8,
            "source_episode_ids": ["ep-1", "ep-2", "ep-3"],
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert result.passed
        assert result.failures == []

    def test_blocked_insufficient_evidence(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {
            "statement": "testing matters",
            "confidence": 0.8,
            "source_episode_ids": ["ep-1"],  # Only 1, need 3
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert not result.passed
        assert len(result.failures) == 1
        assert "insufficient evidence" in result.failures[0]
        assert "1 episodes" in result.failures[0]
        assert "need >= 3" in result.failures[0]

    def test_blocked_low_confidence(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {
            "statement": "maybe testing matters",
            "confidence": 0.4,  # Below 0.6
            "source_episode_ids": ["ep-1", "ep-2", "ep-3"],
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert not result.passed
        assert len(result.failures) == 1
        assert "confidence too low" in result.failures[0]
        assert "0.40" in result.failures[0]

    def test_blocked_both_failures(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {
            "statement": "weak claim",
            "confidence": 0.3,
            "source_episode_ids": ["ep-1"],
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert not result.passed
        assert len(result.failures) == 2

    def test_exact_threshold_passes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {
            "statement": "edge case",
            "confidence": 0.6,  # Exactly at threshold
            "source_episode_ids": ["ep-1", "ep-2", "ep-3"],  # Exactly 3
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert result.passed

    def test_missing_source_episode_ids_uses_empty(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {
            "statement": "no refs",
            "confidence": 0.8,
            # No source_episode_ids key
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert not result.passed
        assert "insufficient evidence" in result.failures[0]
        assert "0 episodes" in result.failures[0]

    def test_missing_confidence_uses_default(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {
            "statement": "uses default confidence",
            "source_episode_ids": ["ep-1", "ep-2", "ep-3"],
            # No confidence key — defaults to 0.7
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert result.passed  # 0.7 > 0.6

    def test_custom_gate_thresholds(self):
        mock_stack = _make_mock_stack()
        gates = PromotionGateConfig(belief_min_evidence=1, belief_min_confidence=0.3)
        processor, _ = _make_processor(mock_stack, gates=gates)
        item = {
            "statement": "low bar",
            "confidence": 0.4,
            "source_episode_ids": ["ep-1"],
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert result.passed

    def test_strict_gate_thresholds(self):
        mock_stack = _make_mock_stack()
        gates = PromotionGateConfig(belief_min_evidence=10, belief_min_confidence=0.95)
        processor, _ = _make_processor(mock_stack, gates=gates)
        item = {
            "statement": "high bar",
            "confidence": 0.9,
            "source_episode_ids": ["ep-1", "ep-2", "ep-3", "ep-4", "ep-5"],
        }
        result = processor._check_promotion_gate("episode_to_belief", item)
        assert not result.passed
        assert len(result.failures) == 2  # Both evidence and confidence fail


# =============================================================================
# _check_promotion_gate: Value Gates
# =============================================================================


class TestValuePromotionGate:
    def test_passes_with_enough_protected_beliefs(self):
        mock_stack = _make_mock_stack()

        # Mock get_memory to return protected beliefs
        def mock_get_memory(mtype, mid):
            return MagicMock(is_protected=True)

        mock_stack.get_memory = mock_get_memory
        processor, _ = _make_processor(mock_stack)

        item = {
            "name": "integrity",
            "statement": "be honest",
            "source_belief_ids": ["b-1", "b-2", "b-3", "b-4", "b-5"],
        }
        result = processor._check_promotion_gate("belief_to_value", item)
        assert result.passed

    def test_blocked_insufficient_evidence(self):
        mock_stack = _make_mock_stack()
        mock_stack.get_memory = lambda mt, mid: MagicMock(is_protected=True)
        processor, _ = _make_processor(mock_stack)

        item = {
            "name": "thin value",
            "source_belief_ids": ["b-1", "b-2"],  # Only 2, need 5
        }
        result = processor._check_promotion_gate("belief_to_value", item)
        assert not result.passed
        assert "insufficient evidence" in result.failures[0]
        assert "2 beliefs" in result.failures[0]

    def test_blocked_unprotected_beliefs(self):
        mock_stack = _make_mock_stack()

        # Mix of protected and unprotected
        def mock_get_memory(mtype, mid):
            if mid in ("b-1", "b-2", "b-3"):
                return MagicMock(is_protected=True)
            return MagicMock(is_protected=False)

        mock_stack.get_memory = mock_get_memory
        processor, _ = _make_processor(mock_stack)

        item = {
            "name": "mixed protection",
            "source_belief_ids": ["b-1", "b-2", "b-3", "b-4", "b-5"],
        }
        result = processor._check_promotion_gate("belief_to_value", item)
        assert not result.passed
        assert any("unprotected" in f for f in result.failures)

    def test_protection_not_required_when_disabled(self):
        mock_stack = _make_mock_stack()
        mock_stack.get_memory = lambda mt, mid: MagicMock(is_protected=False)
        gates = PromotionGateConfig(value_requires_protection=False)
        processor, _ = _make_processor(mock_stack, gates=gates)

        item = {
            "name": "no protection needed",
            "source_belief_ids": ["b-1", "b-2", "b-3", "b-4", "b-5"],
        }
        result = processor._check_promotion_gate("belief_to_value", item)
        assert result.passed

    def test_missing_source_belief_ids_uses_empty(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"name": "no refs"}
        result = processor._check_promotion_gate("belief_to_value", item)
        assert not result.passed
        assert "insufficient evidence" in result.failures[0]

    def test_empty_evidence_skips_protection_check(self):
        """When evidence list is empty, protection check is skipped (redundant)."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"name": "no refs", "source_belief_ids": []}
        result = processor._check_promotion_gate("belief_to_value", item)
        assert not result.passed
        # Only evidence count failure, not protection
        assert len(result.failures) == 1
        assert "insufficient evidence" in result.failures[0]

    def test_nonexistent_belief_treated_as_missing(self):
        """If get_memory returns None for a belief, it's flagged as missing."""
        mock_stack = _make_mock_stack()

        def mock_get_memory(mtype, mid):
            if mid == "b-exists":
                return MagicMock(is_protected=True)
            return None  # Doesn't exist

        mock_stack.get_memory = mock_get_memory
        processor, _ = _make_processor(mock_stack)

        item = {
            "name": "refs missing belief",
            "source_belief_ids": ["b-exists", "b-gone-1", "b-gone-2", "b-gone-3", "b-gone-4"],
        }
        result = processor._check_promotion_gate("belief_to_value", item)
        # Nonexistent beliefs should be flagged as "missing source beliefs"
        # rather than silently skipped — this is the promotion gate safety contract
        assert not result.passed
        assert any("missing source beliefs" in f for f in result.failures)


# =============================================================================
# _check_promotion_gate: Non-gated Transitions
# =============================================================================


class TestNonGatedTransitions:
    def test_raw_to_episode_always_passes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"objective": "test", "outcome": "done", "source_raw_ids": []}
        result = processor._check_promotion_gate("raw_to_episode", item)
        assert result.passed

    def test_raw_to_note_always_passes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"content": "a note", "source_raw_ids": []}
        result = processor._check_promotion_gate("raw_to_note", item)
        assert result.passed

    def test_episode_to_goal_always_passes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"title": "do stuff", "source_episode_ids": []}
        result = processor._check_promotion_gate("episode_to_goal", item)
        assert result.passed

    def test_episode_to_relationship_always_passes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"entity_name": "Alice", "source_episode_ids": []}
        result = processor._check_promotion_gate("episode_to_relationship", item)
        assert result.passed

    def test_episode_to_drive_always_passes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        item = {"drive_type": "curiosity", "source_episode_ids": []}
        result = processor._check_promotion_gate("episode_to_drive", item)
        assert result.passed


# =============================================================================
# Gate Integration: _write_memories
# =============================================================================


class TestGateIntegrationWriteMemories:
    def test_belief_blocked_by_gate_in_write_memories(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_belief.return_value = "b-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "statement": "weak claim",
                "belief_type": "evaluative",
                "confidence": 0.8,
                "source_episode_ids": ["ep-1"],  # Only 1, need 3
            }
        ]
        created = processor._write_memories("episode_to_belief", parsed, [])
        assert len(created) == 0  # Blocked by gate
        assert processor._last_gate_blocked == 1
        mock_stack.save_belief.assert_not_called()

    def test_belief_allowed_by_gate_in_write_memories(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_belief.return_value = "b-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "statement": "well-supported claim",
                "belief_type": "evaluative",
                "confidence": 0.85,
                "source_episode_ids": ["ep-1", "ep-2", "ep-3"],
            }
        ]
        created = processor._write_memories("episode_to_belief", parsed, [])
        assert len(created) == 1
        assert processor._last_gate_blocked == 0

    def test_mixed_items_some_blocked(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_belief.return_value = "b-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "statement": "weak",
                "confidence": 0.8,
                "source_episode_ids": ["ep-1"],  # Blocked
            },
            {
                "statement": "strong",
                "confidence": 0.85,
                "source_episode_ids": ["ep-1", "ep-2", "ep-3"],  # Passes
            },
        ]
        created = processor._write_memories("episode_to_belief", parsed, [])
        assert len(created) == 1
        assert processor._last_gate_blocked == 1

    def test_raw_to_episode_not_gated(self):
        """Non-identity transitions are not affected by gates."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)
        parsed = [
            {
                "objective": "learned",
                "outcome": "success",
                "source_raw_ids": [],  # Empty is fine for raw_to_episode
            }
        ]
        created = processor._write_memories("raw_to_episode", parsed, [])
        assert len(created) == 1
        assert processor._last_gate_blocked == 0


# =============================================================================
# Gate Integration: _write_suggestions
# =============================================================================


class TestGateIntegrationWriteSuggestions:
    def test_belief_suggestion_blocked_by_gate(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        parsed = [
            {
                "statement": "thin belief",
                "confidence": 0.8,
                "source_episode_ids": ["ep-1"],  # Only 1
            }
        ]
        sources = [MagicMock()]
        suggestions = processor._write_suggestions("episode_to_belief", parsed, sources)
        assert len(suggestions) == 0
        assert processor._last_gate_blocked == 1

    def test_belief_suggestion_allowed_by_gate(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        parsed = [
            {
                "statement": "well-supported",
                "confidence": 0.8,
                "source_episode_ids": ["ep-1", "ep-2", "ep-3"],
            }
        ]
        sources = [MagicMock()]
        suggestions = processor._write_suggestions("episode_to_belief", parsed, sources)
        assert len(suggestions) == 1
        assert processor._last_gate_blocked == 0


# =============================================================================
# Gate Results in ProcessingResult
# =============================================================================


class TestGateResultsInProcessingResult:
    def test_default_gate_values(self):
        result = ProcessingResult(
            layer_transition="episode_to_belief",
            source_count=5,
        )
        assert result.gate_blocked == 0
        assert result.gate_details == []

    def test_gate_values_set(self):
        result = ProcessingResult(
            layer_transition="episode_to_belief",
            source_count=5,
            gate_blocked=2,
            gate_details=["reason 1", "reason 2"],
        )
        assert result.gate_blocked == 2
        assert len(result.gate_details) == 2

    def test_process_layer_propagates_gate_results(self):
        """Full pipeline: gate results flow from _write_memories to ProcessingResult."""
        mock_stack = _make_mock_stack()
        eps = [MagicMock(processed=False, id=f"ep-{i}") for i in range(3)]
        for ep in eps:
            ep.objective = f"obj-{ep.id}"
            ep.outcome = f"out-{ep.id}"
        mock_stack._backend.list_raw.return_value = []
        mock_stack.get_episodes.return_value = eps
        mock_stack.get_beliefs.return_value = []

        # Response with beliefs that have insufficient evidence
        response = json.dumps(
            [
                {
                    "statement": "weak belief",
                    "belief_type": "causal",
                    "confidence": 0.8,
                    "source_episode_ids": ["ep-0"],  # Only 1, need 3
                }
            ]
        )
        processor, _ = _make_processor(mock_stack, response)
        config = DEFAULT_LAYER_CONFIGS["episode_to_belief"]
        result = processor._process_layer("episode_to_belief", config, auto_promote=True)

        assert not result.skipped
        assert result.gate_blocked == 1
        assert len(result.gate_details) == 1
        assert "insufficient evidence" in result.gate_details[0]
        assert len(result.created) == 0


# =============================================================================
# Gate Configuration via Stack Settings
# =============================================================================


class TestGateConfigViaStackSettings:
    def test_entity_loads_gate_config_from_stack_settings(self, tmp_path):
        ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
        st = Stack.from_sqlite(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
        )
        ent.attach_stack(st)

        # Set custom gate thresholds
        st.set_stack_setting("promotion_gate_belief_min_evidence", "5")
        st.set_stack_setting("promotion_gate_belief_min_confidence", "0.75")
        st.set_stack_setting("promotion_gate_value_min_evidence", "10")
        st.set_stack_setting("promotion_gate_value_requires_protection", "false")

        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        ent.set_model(MockModel())

        # Process — all skip (no sources), but gates should be configured
        results = ent.process("episode_to_belief", force=True)
        assert len(results) == 1
        assert results[0].skipped  # No sources
        # We verified it doesn't crash with custom settings

    def test_entity_uses_defaults_when_no_settings(self, tmp_path):
        ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
        st = Stack.from_sqlite(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
        )
        ent.attach_stack(st)

        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        ent.set_model(MockModel())
        results = ent.process(force=True)
        # All skip (no sources) — verifies no crash with defaults
        assert all(r.skipped for r in results)


# =============================================================================
# CLI Gate Status Output
# =============================================================================


class TestCLIGateOutput:
    def test_json_output_includes_gate_fields(self, tmp_path, capsys):
        """Verify JSON output from cmd_process includes gate_blocked and gate_details."""
        from kernle.cli.commands.process import cmd_process

        # Use a mock Kernle that returns gate results
        k = MagicMock()
        gate_result = ProcessingResult(
            layer_transition="episode_to_belief",
            source_count=3,
            gate_blocked=1,
            gate_details=["insufficient evidence: 1 episodes (need >= 3)"],
        )
        k.process.return_value = [gate_result]

        # Build args
        args = MagicMock()
        args.process_action = "run"
        args.transition = "episode_to_belief"
        args.force = True
        args.allow_no_inference_override = False
        args.auto_promote = False
        args.json = True

        cmd_process(args, k)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output[0]["gate_blocked"] == 1
        assert len(output[0]["gate_details"]) == 1
        assert "insufficient evidence" in output[0]["gate_details"][0]


# =============================================================================
# MCP Handler Gate Status Output
# =============================================================================


class TestMCPGateOutput:
    def test_mcp_output_includes_gate_status(self):
        from kernle.mcp.handlers.processing import handle_memory_process

        k = MagicMock()
        gate_result = ProcessingResult(
            layer_transition="episode_to_belief",
            source_count=3,
            auto_promote=True,
            gate_blocked=2,
            gate_details=[
                "insufficient evidence: 1 episodes (need >= 3)",
                "confidence too low: 0.40 (need >= 0.60)",
            ],
        )
        k.process.return_value = [gate_result]

        args = {
            "transition": "episode_to_belief",
            "force": True,
            "auto_promote": True,
            "allow_no_inference_override": False,
        }
        output = handle_memory_process(args, k)
        assert "Gate blocked: 2 item(s)" in output
        assert "insufficient evidence" in output
        assert "confidence too low" in output

    def test_mcp_output_no_gate_status_when_zero(self):
        from kernle.mcp.handlers.processing import handle_memory_process

        k = MagicMock()
        no_gate_result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=3,
            auto_promote=True,
            created=[{"type": "episode", "id": "ep-12345678"}],
        )
        k.process.return_value = [no_gate_result]

        args = {
            "transition": "raw_to_episode",
            "force": True,
            "auto_promote": True,
            "allow_no_inference_override": False,
        }
        output = handle_memory_process(args, k)
        assert "Gate blocked" not in output
