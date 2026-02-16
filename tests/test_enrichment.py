"""Unit tests for kernle.core.enrichment — pure enrichment functions."""

import pytest

from kernle.core.enrichment import (
    DRIVE_TYPES,
    VALID_GOAL_TYPES,
    build_derived_from,
    clamp_confidence,
    clamp_intensity,
    format_note_content,
    infer_outcome_type,
    normalize_belief_type,
    normalize_note_type,
    normalize_source_type,
    validate_drive_type,
    validate_goal_type,
)
from kernle.types import SourceType

# =========================================================================
# infer_outcome_type
# =========================================================================


class TestInferOutcomeType:
    @pytest.mark.parametrize(
        "outcome",
        [
            "Fixed successfully",
            "Task done",
            "Deployment completed",
            "All tests finished",
            "Goal accomplished",
        ],
    )
    def test_success_keywords(self, outcome):
        assert infer_outcome_type(outcome) == "success"

    @pytest.mark.parametrize(
        "outcome",
        [
            "Tests failed with 3 errors",
            "Got an error from the API",
            "The build broke",
            "Unable to connect",
            "Couldn't find the file",
        ],
    )
    def test_failure_keywords(self, outcome):
        assert infer_outcome_type(outcome) == "failure"

    def test_partial_default(self):
        assert infer_outcome_type("Made some progress on the task") == "partial"

    def test_case_insensitive(self):
        assert infer_outcome_type("SUCCESSFULLY deployed") == "success"
        assert infer_outcome_type("FAILED to start") == "failure"

    def test_empty_string(self):
        assert infer_outcome_type("") == "partial"


# =========================================================================
# format_note_content
# =========================================================================


class TestFormatNoteContent:
    def test_decision_with_reason(self):
        result = format_note_content("Use TypeScript", "decision", reason="Better type safety")
        assert result == "**Decision**: Use TypeScript\n**Reason**: Better type safety"

    def test_decision_without_reason(self):
        result = format_note_content("Use TypeScript", "decision")
        assert result == "**Decision**: Use TypeScript"

    def test_quote_with_speaker(self):
        result = format_note_content("Move fast", "quote", speaker="Zuck")
        assert result == '> "Move fast"\n> — Zuck'

    def test_quote_without_speaker(self):
        result = format_note_content("Move fast", "quote")
        assert result == '> "Move fast"\n> — Unknown'

    def test_insight(self):
        result = format_note_content("Users prefer dark mode", "insight")
        assert result == "**Insight**: Users prefer dark mode"

    def test_plain_note(self):
        result = format_note_content("Just a note", "note")
        assert result == "Just a note"

    def test_unknown_type_passes_through(self):
        result = format_note_content("Some content", "observation")
        assert result == "Some content"


# =========================================================================
# build_derived_from
# =========================================================================


class TestBuildDerivedFrom:
    def test_with_source(self):
        result = build_derived_from(["episode:ep1"], "session with Sean")
        assert result == ["episode:ep1", "context:session with Sean"]

    def test_without_source(self):
        result = build_derived_from(["episode:ep1"])
        assert result == ["episode:ep1"]

    def test_none_derived_from_with_source(self):
        result = build_derived_from(None, "heartbeat")
        assert result == ["context:heartbeat"]

    def test_none_both(self):
        result = build_derived_from(None)
        assert result is None

    def test_empty_list_no_source(self):
        result = build_derived_from([])
        assert result is None

    def test_does_not_mutate_input(self):
        original = ["episode:ep1"]
        build_derived_from(original, "src")
        assert original == ["episode:ep1"]


# =========================================================================
# clamp_confidence / clamp_intensity
# =========================================================================


class TestClampConfidence:
    def test_normal_value(self):
        assert clamp_confidence(0.5) == 0.5

    def test_below_zero(self):
        assert clamp_confidence(-0.3) == 0.0

    def test_above_one(self):
        assert clamp_confidence(1.5) == 1.0

    def test_boundary_zero(self):
        assert clamp_confidence(0.0) == 0.0

    def test_boundary_one(self):
        assert clamp_confidence(1.0) == 1.0


class TestClampIntensity:
    def test_normal_value(self):
        assert clamp_intensity(0.7) == 0.7

    def test_below_zero(self):
        assert clamp_intensity(-1.0) == 0.0

    def test_above_one(self):
        assert clamp_intensity(2.0) == 1.0


# =========================================================================
# validate_goal_type / validate_drive_type
# =========================================================================


class TestValidateGoalType:
    @pytest.mark.parametrize("goal_type", VALID_GOAL_TYPES)
    def test_valid_types(self, goal_type):
        assert validate_goal_type(goal_type) == goal_type

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid goal_type"):
            validate_goal_type("fantasy")


class TestValidateDriveType:
    @pytest.mark.parametrize("drive_type", DRIVE_TYPES)
    def test_valid_types(self, drive_type):
        validate_drive_type(drive_type)  # should not raise

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid drive type"):
            validate_drive_type("hunger")


# =========================================================================
# normalize_source_type
# =========================================================================


class TestNormalizeSourceType:
    def test_none_defaults_to_direct_experience(self):
        assert normalize_source_type(None) == SourceType.DIRECT_EXPERIENCE

    def test_enum_passthrough(self):
        assert normalize_source_type(SourceType.INFERENCE) == SourceType.INFERENCE

    def test_valid_string(self):
        assert normalize_source_type("inference") == SourceType.INFERENCE

    def test_case_insensitive(self):
        assert normalize_source_type("EXTERNAL") == SourceType.EXTERNAL

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid source_type"):
            normalize_source_type("telepathy")

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            normalize_source_type(42)


# =========================================================================
# normalize_belief_type / normalize_note_type
# =========================================================================


class TestNormalizeBeliefType:
    def test_none_defaults_to_fact(self):
        assert normalize_belief_type(None) == "fact"

    def test_valid_type(self):
        assert normalize_belief_type("hypothesis") == "hypothesis"

    def test_case_insensitive(self):
        assert normalize_belief_type("FACT") == "fact"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid belief type"):
            normalize_belief_type("unicorn")


class TestNormalizeNoteType:
    def test_none_defaults_to_note(self):
        assert normalize_note_type(None) == "note"

    def test_valid_type(self):
        assert normalize_note_type("decision") == "decision"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid note type"):
            normalize_note_type("poem")
