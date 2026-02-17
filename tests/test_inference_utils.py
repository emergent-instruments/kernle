"""Tests for Phase 2: Shared inference contract (inference_utils).

TDD tests for:
- parse_inference_json() parsing and validation
- InferenceResult dataclass
- use_legacy_heuristics flag read/write/defaults
"""

import json
import logging

import pytest

from kernle.core.inference_utils import InferenceResult, parse_inference_json

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logger():
    return logging.getLogger("test_inference_utils")


# ---------------------------------------------------------------------------
# parse_inference_json() tests
# ---------------------------------------------------------------------------


class TestParseInferenceJson:
    def test_parse_valid_json(self, logger):
        """Well-formed response with required fields parsed correctly."""
        raw = json.dumps({"valence": 0.8, "arousal": 0.6, "tags": ["joy"]})
        result = parse_inference_json(
            raw,
            required_fields=["valence", "arousal"],
            fallback={"valence": 0.0, "arousal": 0.0, "tags": []},
            logger=logger,
        )
        assert isinstance(result, InferenceResult)
        assert result.data["valence"] == 0.8
        assert result.data["arousal"] == 0.6
        assert result.data["tags"] == ["joy"]
        assert result.raw == raw
        assert result.fallback_used is False

    def test_parse_malformed_json_returns_fallback(self, logger):
        """Garbage input returns fallback with flag set."""
        fallback = {"valence": 0.0, "arousal": 0.0, "tags": []}
        result = parse_inference_json(
            "not valid json {{{",
            required_fields=["valence"],
            fallback=fallback,
            logger=logger,
        )
        assert result.data == fallback
        assert result.fallback_used is True

    def test_parse_missing_required_fields_returns_fallback(self, logger):
        """Partial JSON missing required fields returns fallback."""
        raw = json.dumps({"valence": 0.8})  # missing "arousal"
        fallback = {"valence": 0.0, "arousal": 0.0}
        result = parse_inference_json(
            raw,
            required_fields=["valence", "arousal"],
            fallback=fallback,
            logger=logger,
        )
        assert result.data == fallback
        assert result.fallback_used is True

    def test_parse_extra_fields_accepted(self, logger):
        """Extra fields beyond required don't cause failure."""
        raw = json.dumps(
            {"valence": 0.5, "arousal": 0.3, "extra_field": "bonus", "tags": ["neutral"]}
        )
        result = parse_inference_json(
            raw,
            required_fields=["valence", "arousal"],
            fallback={"valence": 0.0, "arousal": 0.0},
            logger=logger,
        )
        assert result.data["valence"] == 0.5
        assert result.data["extra_field"] == "bonus"
        assert result.fallback_used is False

    def test_parse_empty_string_returns_fallback(self, logger):
        """Empty string returns fallback."""
        fallback = {"valence": 0.0}
        result = parse_inference_json(
            "", required_fields=["valence"], fallback=fallback, logger=logger
        )
        assert result.data == fallback
        assert result.fallback_used is True

    def test_parse_json_array_returns_fallback(self, logger):
        """JSON array (not object) returns fallback."""
        raw = json.dumps([{"valence": 0.8}])
        fallback = {"valence": 0.0}
        result = parse_inference_json(
            raw, required_fields=["valence"], fallback=fallback, logger=logger
        )
        assert result.data == fallback
        assert result.fallback_used is True

    def test_parse_no_required_fields(self, logger):
        """With empty required_fields, any valid JSON object succeeds."""
        raw = json.dumps({"anything": "works"})
        result = parse_inference_json(raw, required_fields=[], fallback={}, logger=logger)
        assert result.data["anything"] == "works"
        assert result.fallback_used is False

    def test_parse_json_with_markdown_fencing(self, logger):
        """JSON wrapped in markdown code fences is extracted and parsed."""
        raw = '```json\n{"valence": 0.7, "arousal": 0.4}\n```'
        result = parse_inference_json(
            raw,
            required_fields=["valence", "arousal"],
            fallback={"valence": 0.0, "arousal": 0.0},
            logger=logger,
        )
        assert result.data["valence"] == 0.7
        assert result.fallback_used is False
