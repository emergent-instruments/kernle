"""Tests for MCP handler boundary validation — input sanitization and handler callability.

Exercises negative-path scenarios for MCP validators: oversized inputs,
NaN/Infinity rejection, and verifies all registered handlers are callable.
"""

import math

import pytest

# ---------------------------------------------------------------------------
# Import validators from identity handlers
# ---------------------------------------------------------------------------
from kernle.mcp.handlers.identity import (
    HANDLERS as IDENTITY_HANDLERS,
)
from kernle.mcp.handlers.identity import (
    VALIDATORS as IDENTITY_VALIDATORS,
)
from kernle.mcp.handlers.identity import (
    validate_memory_drive,
    validate_memory_goal,
    validate_memory_value,
)

# ---------------------------------------------------------------------------
# Import validators from memory handlers
# ---------------------------------------------------------------------------
from kernle.mcp.handlers.memory import (
    HANDLERS as MEMORY_HANDLERS,
)
from kernle.mcp.handlers.memory import (
    VALIDATORS as MEMORY_VALIDATORS,
)
from kernle.mcp.handlers.memory import (
    validate_memory_checkpoint_save,
    validate_memory_episode,
    validate_memory_load,
    validate_memory_note,
    validate_memory_raw,
    validate_memory_raw_search,
    validate_memory_search,
)
from kernle.mcp.sanitize import (
    sanitize_string,
    validate_enum,
    validate_number,
)

# ===========================================================================
# Test: validate_number rejects NaN and Infinity
# ===========================================================================


class TestValidateNumberRejectsNonFinite:
    """validate_number must reject NaN, +Infinity, and -Infinity."""

    def test_rejects_nan(self):
        """float('nan') is not a valid numeric input."""
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_number(float("nan"), "test_field")

    def test_rejects_positive_infinity(self):
        """float('inf') is not a valid numeric input."""
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_number(float("inf"), "test_field")

    def test_rejects_negative_infinity(self):
        """float('-inf') is not a valid numeric input."""
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_number(float("-inf"), "test_field")

    def test_rejects_math_nan(self):
        """math.nan is not a valid numeric input."""
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_number(math.nan, "test_field")

    def test_rejects_math_inf(self):
        """math.inf is not a valid numeric input."""
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_number(math.inf, "test_field")

    def test_accepts_valid_float(self):
        """Normal float values should pass through."""
        assert validate_number(0.75, "test_field") == 0.75

    def test_accepts_valid_int(self):
        """Integer values should be accepted and returned as float."""
        assert validate_number(10, "test_field") == 10.0

    def test_accepts_zero(self):
        """Zero is a valid finite number."""
        assert validate_number(0, "test_field") == 0.0

    def test_accepts_negative(self):
        """Negative values are valid when no min constraint is set."""
        assert validate_number(-5.0, "test_field") == -5.0


# ===========================================================================
# Test: Memory validators reject oversized inputs
# ===========================================================================


class TestMemoryValidatorsRejectOversizedInputs:
    """Memory validators should reject strings that exceed their max_length."""

    def test_checkpoint_task_too_long(self):
        """checkpoint_save rejects task strings longer than 500 chars."""
        oversized_task = "x" * 501
        with pytest.raises(ValueError, match="too long"):
            validate_memory_checkpoint_save({"task": oversized_task})

    def test_episode_objective_too_long(self):
        """episode rejects objective strings longer than 1000 chars."""
        oversized_objective = "y" * 1001
        with pytest.raises(ValueError, match="too long"):
            validate_memory_episode({"objective": oversized_objective, "outcome": "ok"})

    def test_episode_outcome_too_long(self):
        """episode rejects outcome strings longer than 1000 chars."""
        oversized_outcome = "z" * 1001
        with pytest.raises(ValueError, match="too long"):
            validate_memory_episode({"objective": "test", "outcome": oversized_outcome})

    def test_note_content_too_long(self):
        """note rejects content strings longer than 2000 chars."""
        oversized_content = "a" * 2001
        with pytest.raises(ValueError, match="too long"):
            validate_memory_note({"content": oversized_content})

    def test_search_query_too_long(self):
        """search rejects query strings longer than 500 chars."""
        oversized_query = "b" * 501
        with pytest.raises(ValueError, match="too long"):
            validate_memory_search({"query": oversized_query})

    def test_raw_blob_too_long(self):
        """raw rejects blob strings longer than 1_000_000 chars."""
        oversized_blob = "c" * 1_000_001
        with pytest.raises(ValueError, match="too long"):
            validate_memory_raw({"blob": oversized_blob})

    def test_raw_search_query_too_long(self):
        """raw_search rejects query strings longer than 1000 chars."""
        oversized_query = "d" * 1001
        with pytest.raises(ValueError, match="too long"):
            validate_memory_raw_search({"query": oversized_query})


# ===========================================================================
# Test: Identity validators reject oversized inputs
# ===========================================================================


class TestIdentityValidatorsRejectOversizedInputs:
    """Identity validators should reject strings that exceed their max_length."""

    def test_value_name_too_long(self):
        """value rejects name strings longer than 100 chars."""
        oversized_name = "n" * 101
        with pytest.raises(ValueError, match="too long"):
            validate_memory_value({"name": oversized_name, "statement": "valid statement"})

    def test_value_statement_too_long(self):
        """value rejects statement strings longer than 1000 chars."""
        oversized_statement = "v" * 1001
        with pytest.raises(ValueError, match="too long"):
            validate_memory_value({"name": "valid-name", "statement": oversized_statement})

    def test_goal_title_too_long(self):
        """goal rejects title strings longer than 200 chars."""
        oversized_title = "t" * 201
        with pytest.raises(ValueError, match="too long"):
            validate_memory_goal({"title": oversized_title})

    def test_goal_description_too_long(self):
        """goal rejects description strings longer than 1000 chars."""
        oversized_description = "d" * 1001
        with pytest.raises(ValueError, match="too long"):
            validate_memory_goal({"title": "valid title", "description": oversized_description})


# ===========================================================================
# Test: Memory validators reject missing required fields
# ===========================================================================


class TestMemoryValidatorsRejectMissingRequired:
    """Validators should raise when required fields are missing or empty."""

    def test_checkpoint_missing_task(self):
        """checkpoint_save requires a task string."""
        with pytest.raises(ValueError):
            validate_memory_checkpoint_save({})

    def test_episode_missing_objective(self):
        """episode requires an objective string."""
        with pytest.raises(ValueError):
            validate_memory_episode({"outcome": "some outcome"})

    def test_episode_missing_outcome(self):
        """episode requires an outcome string."""
        with pytest.raises(ValueError):
            validate_memory_episode({"objective": "some objective"})

    def test_note_missing_content(self):
        """note requires a content string."""
        with pytest.raises(ValueError):
            validate_memory_note({})

    def test_search_missing_query(self):
        """search requires a query string."""
        with pytest.raises(ValueError):
            validate_memory_search({})

    def test_raw_missing_blob(self):
        """raw requires a blob string."""
        with pytest.raises(ValueError):
            validate_memory_raw({})

    def test_value_missing_name(self):
        """value requires a name string."""
        with pytest.raises(ValueError):
            validate_memory_value({"statement": "valid"})

    def test_value_missing_statement(self):
        """value requires a statement string."""
        with pytest.raises(ValueError):
            validate_memory_value({"name": "valid"})

    def test_goal_missing_title(self):
        """goal requires a title string."""
        with pytest.raises(ValueError):
            validate_memory_goal({})

    def test_drive_missing_drive_type(self):
        """drive requires a drive_type enum value."""
        with pytest.raises(ValueError):
            validate_memory_drive({})


# ===========================================================================
# Test: validate_number with range constraints used by validators
# ===========================================================================


class TestValidateNumberRangeConstraints:
    """validate_number should enforce min/max constraints correctly."""

    def test_drive_intensity_nan_rejected(self):
        """NaN intensity is rejected by drive validator."""
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_memory_drive({"drive_type": "curiosity", "intensity": float("nan")})

    def test_drive_intensity_inf_rejected(self):
        """Infinite intensity is rejected by drive validator."""
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_memory_drive({"drive_type": "curiosity", "intensity": float("inf")})

    def test_value_priority_above_max(self):
        """Priority > 100 is rejected by value validator."""
        with pytest.raises(ValueError, match="must be <="):
            validate_memory_value({"name": "test", "statement": "test", "priority": 150})

    def test_load_budget_nan_rejected(self):
        """NaN budget is rejected by load validator."""
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_memory_load({"budget": float("nan")})


# ===========================================================================
# Test: validate_enum rejects invalid enum values
# ===========================================================================


class TestValidateEnumBoundaries:
    """validate_enum should reject values not in the allowed list."""

    def test_invalid_note_type(self):
        """Note type must be one of the allowed values."""
        with pytest.raises(ValueError, match="must be one of"):
            validate_enum("invalid_type", "type", ["note", "decision", "insight", "quote"])

    def test_invalid_format(self):
        """Format must be 'text' or 'json'."""
        with pytest.raises(ValueError, match="must be one of"):
            validate_enum("xml", "format", ["text", "json"])

    def test_non_string_enum_raises(self):
        """Non-string values are rejected by validate_enum."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_enum(42, "format", ["text", "json"])


# ===========================================================================
# Test: All registered handlers and validators are callable
# ===========================================================================


class TestHandlerRegistryCallability:
    """Every function registered in HANDLERS and VALIDATORS dicts must be callable."""

    def test_all_memory_handlers_are_callable(self):
        """All memory handler functions should be callable."""
        for name, handler in MEMORY_HANDLERS.items():
            assert callable(handler), f"Memory handler '{name}' is not callable"

    def test_all_memory_validators_are_callable(self):
        """All memory validator functions should be callable."""
        for name, validator in MEMORY_VALIDATORS.items():
            assert callable(validator), f"Memory validator '{name}' is not callable"

    def test_all_identity_handlers_are_callable(self):
        """All identity handler functions should be callable."""
        for name, handler in IDENTITY_HANDLERS.items():
            assert callable(handler), f"Identity handler '{name}' is not callable"

    def test_all_identity_validators_are_callable(self):
        """All identity validator functions should be callable."""
        for name, validator in IDENTITY_VALIDATORS.items():
            assert callable(validator), f"Identity validator '{name}' is not callable"

    def test_memory_handlers_match_validators(self):
        """Every memory handler should have a corresponding validator."""
        handler_names = set(MEMORY_HANDLERS.keys())
        validator_names = set(MEMORY_VALIDATORS.keys())
        assert handler_names == validator_names, (
            f"Handler/validator mismatch.\n"
            f"  Handlers without validators: {handler_names - validator_names}\n"
            f"  Validators without handlers: {validator_names - handler_names}"
        )

    def test_identity_handlers_match_validators(self):
        """Every identity handler should have a corresponding validator."""
        handler_names = set(IDENTITY_HANDLERS.keys())
        validator_names = set(IDENTITY_VALIDATORS.keys())
        assert handler_names == validator_names, (
            f"Handler/validator mismatch.\n"
            f"  Handlers without validators: {handler_names - validator_names}\n"
            f"  Validators without handlers: {validator_names - handler_names}"
        )


# ===========================================================================
# Test: Other handler modules are importable and have registries
# ===========================================================================


class TestOtherHandlerModules:
    """Handlers from processing, sync, seed, and temporal should be importable."""

    def test_processing_handlers_importable(self):
        """Processing handler module should export HANDLERS and VALIDATORS."""
        from kernle.mcp.handlers.processing import HANDLERS, VALIDATORS

        assert isinstance(HANDLERS, dict)
        assert isinstance(VALIDATORS, dict)
        for name, handler in HANDLERS.items():
            assert callable(handler), f"Processing handler '{name}' is not callable"

    def test_sync_handlers_importable(self):
        """Sync handler module should export HANDLERS and VALIDATORS."""
        from kernle.mcp.handlers.sync import HANDLERS, VALIDATORS

        assert isinstance(HANDLERS, dict)
        assert isinstance(VALIDATORS, dict)
        for name, handler in HANDLERS.items():
            assert callable(handler), f"Sync handler '{name}' is not callable"

    def test_seed_handlers_importable(self):
        """Seed handler module should export HANDLERS and VALIDATORS."""
        from kernle.mcp.handlers.seed import HANDLERS, VALIDATORS

        assert isinstance(HANDLERS, dict)
        assert isinstance(VALIDATORS, dict)
        for name, handler in HANDLERS.items():
            assert callable(handler), f"Seed handler '{name}' is not callable"

    def test_temporal_handlers_importable(self):
        """Temporal handler module should export HANDLERS and VALIDATORS."""
        from kernle.mcp.handlers.temporal import HANDLERS, VALIDATORS

        assert isinstance(HANDLERS, dict)
        assert isinstance(VALIDATORS, dict)
        for name, handler in HANDLERS.items():
            assert callable(handler), f"Temporal handler '{name}' is not callable"


# ===========================================================================
# Test: sanitize_string rejects oversized inputs directly
# ===========================================================================


class TestSanitizeStringBoundaries:
    """sanitize_string should enforce max_length and reject non-strings."""

    def test_rejects_oversized_string(self):
        """Strings exceeding max_length are rejected."""
        with pytest.raises(ValueError, match="too long"):
            sanitize_string("x" * 11, "test_field", max_length=10)

    def test_rejects_non_string_type(self):
        """Non-string values are rejected with a type error."""
        with pytest.raises(ValueError, match="must be a string"):
            sanitize_string(12345, "test_field")

    def test_rejects_empty_when_required(self):
        """Empty strings are rejected when required=True."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_string("", "test_field", required=True)

    def test_allows_empty_when_not_required(self):
        """Empty strings are accepted when required=False."""
        result = sanitize_string("", "test_field", required=False)
        assert result == ""

    def test_at_boundary_length_accepted(self):
        """A string exactly at max_length should be accepted."""
        result = sanitize_string("x" * 10, "test_field", max_length=10)
        assert result == "x" * 10

    def test_strips_null_bytes(self):
        """Null bytes and control characters are stripped."""
        result = sanitize_string("hello\x00world\x01", "test_field")
        assert result == "helloworld"
