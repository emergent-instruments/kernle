"""Shared inference response validation and error handling.

All inference-dependent operations use these utilities for consistent
error handling, JSON parsing, and legacy heuristic gating.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class InferenceResult:
    """Parsed inference response."""

    data: Dict[str, Any]
    raw: str
    fallback_used: bool = False


# Regex to extract JSON from markdown code fences
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def parse_inference_json(
    raw_response: str,
    required_fields: List[str],
    fallback: Dict[str, Any],
    logger: logging.Logger,
) -> InferenceResult:
    """Parse JSON from inference response with validation.

    Handles common LLM output patterns:
    - Plain JSON objects
    - JSON wrapped in markdown code fences (```json ... ```)

    Returns InferenceResult. On parse/validation failure, returns
    fallback with fallback_used=True. Never raises.

    Args:
        raw_response: Raw string from inference call
        required_fields: List of field names that must be present in the parsed dict
        fallback: Default dict to return on failure
        logger: Logger for warnings

    Returns:
        InferenceResult with parsed data or fallback
    """
    if not raw_response or not raw_response.strip():
        logger.warning("Empty inference response, using fallback")
        return InferenceResult(data=dict(fallback), raw=raw_response or "", fallback_used=True)

    text = raw_response.strip()

    # Try extracting from markdown fences first
    fence_match = _FENCED_JSON_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse inference JSON: %s", e)
        return InferenceResult(data=dict(fallback), raw=raw_response, fallback_used=True)

    if not isinstance(parsed, dict):
        logger.warning("Inference response is not a JSON object (got %s)", type(parsed).__name__)
        return InferenceResult(data=dict(fallback), raw=raw_response, fallback_used=True)

    # Check required fields
    missing = [f for f in required_fields if f not in parsed]
    if missing:
        logger.warning("Inference response missing required fields: %s", missing)
        return InferenceResult(data=dict(fallback), raw=raw_response, fallback_used=True)

    return InferenceResult(data=parsed, raw=raw_response, fallback_used=False)
