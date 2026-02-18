"""Unit tests for WritersMixin methods in kernle/core/writers.py.

These tests verify the validation, normalization, and branching logic
inside the writer methods (episode, belief, goal, drive, note, etc.).
We use a mocked Kernle instance (same pattern as test_strict_write_path.py)
so we can focus on the WritersMixin logic without needing a real database.

Key areas tested:
- episode() outcome_type detection from outcome text
- episode() source_type inference from source string
- belief() normalization of belief_type values
- belief() confidence clamping to [0.0, 1.0]
- goal() protection for aspiration/commitment types
- goal() rejection of invalid goal_type values
- drive() update-vs-create branching
- drive() intensity clamping
- drive() rejection of invalid drive_type values
- note() formatting by note_type (decision, quote, insight)
- _normalize_source_type validation
- _normalize_belief_type validation
- _normalize_note_type validation
"""

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kernle.core import Kernle
from kernle.core.writers import WritersMixin
from kernle.storage import Drive
from kernle.types import SourceType
from tests.conftest import bind_noop_model

# =========================================================================
# Helpers
# =========================================================================


def _uid():
    """Generate a unique ID for test data."""
    return str(uuid.uuid4())


def _now():
    """Get the current UTC timestamp."""
    return datetime.now(timezone.utc)


@pytest.fixture
def mocked_kernle():
    """Create a Kernle with separately mockable _write_backend and _storage.

    Yields (kernle, mock_write_backend, mock_storage).

    Uses patch.object on the Kernle class to override the _write_backend
    property safely, ensuring cleanup even if the test fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Create a real Kernle in non-strict mode (to avoid stack creation)
        k = Kernle(
            stack_id="test-writer-methods",
            checkpoint_dir=Path(tmp) / "cp",
            strict=False,
        )

        # Bind noop model BEFORE replacing storage — this caches
        # the SQLiteStack (with _inference set) so _require_inference passes.
        bind_noop_model(k)

        mock_storage = MagicMock()
        mock_storage.get_stack_setting.return_value = None
        mock_write_backend = MagicMock()

        k._storage = mock_storage

        # Patch the _write_backend property on the class
        with patch.object(
            type(k),
            "_write_backend",
            new_callable=PropertyMock,
            return_value=mock_write_backend,
        ):
            yield k, mock_write_backend, mock_storage


# =========================================================================
# Episode: outcome_type detection
# =========================================================================


class TestEpisodeOutcomeTypeDetection:
    """episode() should infer outcome_type from the outcome text."""

    def test_success_outcome_detected(self, mocked_kernle):
        """Outcome containing 'success' should set outcome_type to 'success'."""
        k, mock_wb, _ = mocked_kernle

        k.episode(objective="Fix the bug", outcome="Fixed successfully")

        # Inspect the Episode object passed to save_episode
        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.outcome_type == "success"

    def test_completed_outcome_detected(self, mocked_kernle):
        """Outcome containing 'completed' should set outcome_type to 'success'."""
        k, mock_wb, _ = mocked_kernle

        k.episode(objective="Deploy app", outcome="Deployment completed")

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.outcome_type == "success"

    def test_failure_outcome_detected(self, mocked_kernle):
        """Outcome containing 'fail' should set outcome_type to 'failure'."""
        k, mock_wb, _ = mocked_kernle

        k.episode(objective="Run tests", outcome="Tests failed with 3 errors")

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.outcome_type == "failure"

    def test_error_outcome_detected(self, mocked_kernle):
        """Outcome containing 'error' should set outcome_type to 'failure'."""
        k, mock_wb, _ = mocked_kernle

        k.episode(objective="Run query", outcome="Database error occurred")

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.outcome_type == "failure"

    def test_partial_outcome_as_default(self, mocked_kernle):
        """Outcome without success/failure keywords defaults to 'partial'."""
        k, mock_wb, _ = mocked_kernle

        k.episode(
            objective="Investigate issue",
            outcome="Found some clues but need more data",
        )

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.outcome_type == "partial"


# =========================================================================
# Episode: source_type inference from source string
# =========================================================================


class TestEpisodeSourceType:
    """episode() should handle source_type correctly."""

    def test_no_source_type_defaults_to_direct_experience(self, mocked_kernle):
        """When source_type is None, it should default to DIRECT_EXPERIENCE."""
        k, mock_wb, _ = mocked_kernle

        k.episode(objective="Test", outcome="Done")

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.source_type == SourceType.DIRECT_EXPERIENCE

    def test_source_without_source_type_defaults_to_direct_experience(self, mocked_kernle):
        """When source is set but source_type is not, it should still default to DIRECT_EXPERIENCE."""
        k, mock_wb, _ = mocked_kernle

        k.episode(
            objective="API limitation",
            outcome="Learned about rate limits",
            source="told by team lead",
        )

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.source_type == SourceType.DIRECT_EXPERIENCE

    def test_explicit_enum_source_type(self, mocked_kernle):
        """When source_type is a SourceType enum, it should be used directly."""
        k, mock_wb, _ = mocked_kernle

        k.episode(
            objective="Pattern analysis",
            outcome="Found correlation",
            source_type=SourceType.INFERENCE,
        )

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.source_type == SourceType.INFERENCE

    def test_explicit_string_source_type(self, mocked_kernle):
        """When source_type is a valid string, it should be converted to SourceType."""
        k, mock_wb, _ = mocked_kernle

        k.episode(
            objective="Test",
            outcome="Done",
            source="told by someone",
            source_type="seed",
        )

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.source_type == SourceType.SEED

    def test_invalid_source_type_raises_error(self, mocked_kernle):
        """When source_type is an invalid string, it should raise ValueError."""
        k, _, _ = mocked_kernle

        with pytest.raises(ValueError, match="Invalid source_type"):
            k.episode(objective="Test", outcome="Done", source_type="telepathy")


# =========================================================================
# Episode: derived_from lineage tracking
# =========================================================================


class TestEpisodeDerivedFrom:
    """episode() should properly build the derived_from list."""

    def test_source_appended_as_context_ref(self, mocked_kernle):
        """When source is provided, it should be appended as a context: ref."""
        k, mock_wb, _ = mocked_kernle

        k.episode(
            objective="Debug session",
            outcome="Found the bug",
            source="session with Sean",
        )

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert "context:session with Sean" in saved_episode.derived_from

    def test_explicit_derived_from_preserved(self, mocked_kernle):
        """Explicit derived_from IDs should be included in the final list."""
        k, mock_wb, _ = mocked_kernle

        ref_id = _uid()
        k.episode(
            objective="Follow-up",
            outcome="Addressed the issue",
            derived_from=[f"episode:{ref_id}"],
        )

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert f"episode:{ref_id}" in saved_episode.derived_from

    def test_no_source_no_derived_from_gives_none(self, mocked_kernle):
        """With no source and no derived_from, derived_from should be None."""
        k, mock_wb, _ = mocked_kernle

        k.episode(objective="Solo task", outcome="Completed alone")

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert saved_episode.derived_from is None


# =========================================================================
# Belief: type normalization and confidence clamping
# =========================================================================


class TestBeliefMethods:
    """Tests for belief() validation and normalization."""

    def test_default_belief_type_is_fact(self, mocked_kernle):
        """belief() with no type should default to 'fact'."""
        k, mock_wb, _ = mocked_kernle

        k.belief(statement="Python is interpreted")

        saved_belief = mock_wb.save_belief.call_args[0][0]
        assert saved_belief.belief_type == "fact"

    def test_belief_type_normalized_to_lowercase(self, mocked_kernle):
        """belief() should normalize the type to lowercase."""
        k, mock_wb, _ = mocked_kernle

        k.belief(statement="Testing helps", type="HYPOTHESIS")

        saved_belief = mock_wb.save_belief.call_args[0][0]
        assert saved_belief.belief_type == "hypothesis"

    def test_invalid_belief_type_raises_error(self, mocked_kernle):
        """belief() should raise ValueError for an unknown belief type."""
        k, _, _ = mocked_kernle

        with pytest.raises(ValueError, match="Invalid belief type"):
            k.belief(statement="Test", type="bogus_type")

    def test_confidence_clamped_to_max_one(self, mocked_kernle):
        """Confidence values above 1.0 should be clamped to 1.0."""
        k, mock_wb, _ = mocked_kernle

        k.belief(statement="Very certain", confidence=1.5)

        saved_belief = mock_wb.save_belief.call_args[0][0]
        assert saved_belief.confidence == 1.0

    def test_confidence_clamped_to_min_zero(self, mocked_kernle):
        """Confidence values below 0.0 should be clamped to 0.0."""
        k, mock_wb, _ = mocked_kernle

        k.belief(statement="Very uncertain", confidence=-0.5)

        saved_belief = mock_wb.save_belief.call_args[0][0]
        assert saved_belief.confidence == 0.0

    def test_belief_default_source_type(self, mocked_kernle):
        """belief() with no source_type should default to DIRECT_EXPERIENCE."""
        k, mock_wb, _ = mocked_kernle

        k.belief(statement="Pattern holds", source="consolidation pass")

        saved_belief = mock_wb.save_belief.call_args[0][0]
        assert saved_belief.source_type == SourceType.DIRECT_EXPERIENCE

    def test_belief_explicit_enum_source_type(self, mocked_kernle):
        """belief() with SourceType enum should use it directly."""
        k, mock_wb, _ = mocked_kernle

        k.belief(
            statement="Pattern holds",
            source="consolidation pass",
            source_type=SourceType.CONSOLIDATION,
        )

        saved_belief = mock_wb.save_belief.call_args[0][0]
        assert saved_belief.source_type == SourceType.CONSOLIDATION

    def test_belief_explicit_string_source_type(self, mocked_kernle):
        """belief() with valid string source_type should convert to SourceType."""
        k, mock_wb, _ = mocked_kernle

        k.belief(
            statement="Initial setup fact",
            source="seed data",
            source_type="seed",
        )

        saved_belief = mock_wb.save_belief.call_args[0][0]
        assert saved_belief.source_type == SourceType.SEED

    def test_belief_invalid_source_type_raises_error(self, mocked_kernle):
        """belief() with invalid source_type should raise ValueError."""
        k, _, _ = mocked_kernle

        with pytest.raises(ValueError, match="Invalid source_type"):
            k.belief(statement="Test", source_type="telepathy")


# =========================================================================
# Goal: protection and validation
# =========================================================================


class TestGoalProtectionAndValidation:
    """Tests for goal() protection logic and input validation."""

    def test_aspiration_goal_is_protected(self, mocked_kernle):
        """Aspiration goals should have is_protected=True on the Goal object."""
        k, mock_wb, _ = mocked_kernle

        k.goal(title="Learn Rust", goal_type="aspiration")

        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.is_protected is True

    def test_commitment_goal_is_protected(self, mocked_kernle):
        """Commitment goals should have is_protected=True on the Goal object."""
        k, mock_wb, _ = mocked_kernle

        k.goal(title="Ship v1.0", goal_type="commitment")

        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.is_protected is True

    def test_task_goal_is_not_protected(self, mocked_kernle):
        """Task goals should have is_protected=False."""
        k, mock_wb, _ = mocked_kernle

        k.goal(title="Fix login bug", goal_type="task")

        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.is_protected is False

    def test_exploration_goal_is_not_protected(self, mocked_kernle):
        """Exploration goals should have is_protected=False."""
        k, mock_wb, _ = mocked_kernle

        k.goal(title="Evaluate new DB", goal_type="exploration")

        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.is_protected is False

    def test_invalid_goal_type_raises_error(self, mocked_kernle):
        """goal() should raise ValueError for unknown goal_type."""
        k, _, _ = mocked_kernle

        with pytest.raises(ValueError, match="Invalid goal_type"):
            k.goal(title="Bad goal", goal_type="dream")

    def test_goal_default_type_is_task(self, mocked_kernle):
        """goal() should default goal_type to 'task'."""
        k, mock_wb, _ = mocked_kernle

        k.goal(title="Default type goal")

        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.goal_type == "task"

    def test_goal_description_defaults_to_title(self, mocked_kernle):
        """goal() description should default to the title when not provided."""
        k, mock_wb, _ = mocked_kernle

        k.goal(title="My goal title")

        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.description == "My goal title"


# =========================================================================
# Drive: update vs create and validation
# =========================================================================


class TestDriveMethods:
    """Tests for drive() update-vs-create logic and validation."""

    def test_creates_new_drive_when_none_exists(self, mocked_kernle):
        """drive() should create a new Drive when no matching type exists."""
        k, mock_wb, _ = mocked_kernle

        mock_wb.get_drives.return_value = []

        drive_id = k.drive("curiosity", intensity=0.7)

        # Should have called save_drive with a new Drive object
        assert drive_id is not None
        mock_wb.save_drive.assert_called_once()
        saved_drive = mock_wb.save_drive.call_args[0][0]
        assert saved_drive.drive_type == "curiosity"
        assert saved_drive.intensity == pytest.approx(0.7)

    def test_updates_existing_drive_when_match_found(self, mocked_kernle):
        """drive() should update an existing drive when the type matches."""
        k, mock_wb, _ = mocked_kernle

        existing_drive = Drive(
            id=_uid(),
            stack_id="test-writer-methods",
            drive_type="curiosity",
            intensity=0.3,
            focus_areas=["reading"],
            created_at=_now(),
            updated_at=_now(),
        )
        mock_wb.get_drives.return_value = [existing_drive]

        result_id = k.drive("curiosity", intensity=0.8, focus_areas=["coding"])

        # Should return the existing drive's ID (not a new one)
        assert result_id == existing_drive.id
        mock_wb.update_drive_atomic.assert_called_once()
        saved_drive = mock_wb.update_drive_atomic.call_args[0][0]
        assert saved_drive.intensity == pytest.approx(0.8)
        assert saved_drive.focus_areas == ["coding"]

    def test_invalid_drive_type_raises_error(self, mocked_kernle):
        """drive() should raise ValueError for an unknown drive_type."""
        k, _, _ = mocked_kernle

        with pytest.raises(ValueError, match="Invalid drive type"):
            k.drive("happiness", intensity=0.5)

    def test_intensity_clamped_to_max_one(self, mocked_kernle):
        """Intensity values above 1.0 should be clamped to 1.0."""
        k, mock_wb, _ = mocked_kernle

        mock_wb.get_drives.return_value = []

        k.drive("growth", intensity=2.0)

        saved_drive = mock_wb.save_drive.call_args[0][0]
        assert saved_drive.intensity == 1.0

    def test_intensity_clamped_to_min_zero(self, mocked_kernle):
        """Intensity values below 0.0 should be clamped to 0.0."""
        k, mock_wb, _ = mocked_kernle

        mock_wb.get_drives.return_value = []

        k.drive("existence", intensity=-0.5)

        saved_drive = mock_wb.save_drive.call_args[0][0]
        assert saved_drive.intensity == 0.0

    def test_passes_original_version_for_atomic_update(self, mocked_kernle):
        """drive() passes the original version for optimistic concurrency control."""
        k, mock_wb, _ = mocked_kernle

        existing_drive = Drive(
            id=_uid(),
            stack_id="test-writer-methods",
            drive_type="connection",
            intensity=0.5,
            focus_areas=[],
            created_at=_now(),
            updated_at=_now(),
            version=3,
        )
        mock_wb.get_drives.return_value = [existing_drive]

        k.drive("connection", intensity=0.6)

        # Atomic method receives the drive with original version (SQL handles increment)
        saved_drive = mock_wb.update_drive_atomic.call_args[0][0]
        assert saved_drive.version == 3


# =========================================================================
# Note: formatting by type
# =========================================================================


class TestNoteFormatting:
    """Tests for note() content formatting based on note_type."""

    def test_decision_note_formatting(self, mocked_kernle):
        """Decision notes should be prefixed with **Decision**: ."""
        k, mock_wb, _ = mocked_kernle

        k.note(content="Use PostgreSQL", type="decision", reason="Better JSON support")

        saved_note = mock_wb.save_note.call_args[0][0]
        assert saved_note.content.startswith("**Decision**: Use PostgreSQL")
        assert "**Reason**: Better JSON support" in saved_note.content

    def test_quote_note_formatting(self, mocked_kernle):
        """Quote notes should be formatted with blockquote and speaker."""
        k, mock_wb, _ = mocked_kernle

        k.note(content="Move fast and break things", type="quote", speaker="Zuckerberg")

        saved_note = mock_wb.save_note.call_args[0][0]
        assert "Move fast and break things" in saved_note.content
        assert "Zuckerberg" in saved_note.content

    def test_quote_without_speaker_uses_unknown(self, mocked_kernle):
        """Quote notes without a speaker should default to 'Unknown'."""
        k, mock_wb, _ = mocked_kernle

        k.note(content="Some wisdom", type="quote")

        saved_note = mock_wb.save_note.call_args[0][0]
        assert "Unknown" in saved_note.content

    def test_insight_note_formatting(self, mocked_kernle):
        """Insight notes should be prefixed with **Insight**: ."""
        k, mock_wb, _ = mocked_kernle

        k.note(content="Caching reduces latency", type="insight")

        saved_note = mock_wb.save_note.call_args[0][0]
        assert saved_note.content.startswith("**Insight**: Caching reduces latency")

    def test_plain_note_no_formatting(self, mocked_kernle):
        """Plain notes should have no special formatting."""
        k, mock_wb, _ = mocked_kernle

        k.note(content="Just a regular note")

        saved_note = mock_wb.save_note.call_args[0][0]
        assert saved_note.content == "Just a regular note"

    def test_invalid_note_type_raises_error(self, mocked_kernle):
        """note() should raise ValueError for unknown note_type."""
        k, _, _ = mocked_kernle

        with pytest.raises(ValueError, match="Invalid note type"):
            k.note(content="Test", type="journal")


# =========================================================================
# Static normalizers (_normalize_source_type, _normalize_belief_type, etc.)
# =========================================================================


class TestNormalizeSourceType:
    """Tests for WritersMixin._normalize_source_type static method."""

    def test_none_defaults_to_direct_experience(self):
        """None should return DIRECT_EXPERIENCE."""
        result = WritersMixin._normalize_source_type(None)
        assert result == SourceType.DIRECT_EXPERIENCE

    def test_source_type_enum_passed_through(self):
        """A SourceType enum value should be returned as-is."""
        result = WritersMixin._normalize_source_type(SourceType.SEED)
        assert result == SourceType.SEED

    def test_valid_string_converted_to_enum(self):
        """A valid string value should be converted to the SourceType enum."""
        result = WritersMixin._normalize_source_type("inference")
        assert result == SourceType.INFERENCE

    def test_case_insensitive_string_normalization(self):
        """Strings should be normalized case-insensitively."""
        result = WritersMixin._normalize_source_type("  EXTERNAL  ")
        assert result == SourceType.EXTERNAL

    def test_invalid_string_raises_error(self):
        """Invalid strings should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid source_type"):
            WritersMixin._normalize_source_type("telepathy")

    def test_non_string_non_enum_raises_error(self):
        """Non-string, non-enum types should raise ValueError."""
        with pytest.raises(ValueError, match="source_type must be a string"):
            WritersMixin._normalize_source_type(42)


class TestNormalizeBeliefType:
    """Tests for WritersMixin._normalize_belief_type class method."""

    def test_none_defaults_to_fact(self):
        """None should return 'fact'."""
        result = WritersMixin._normalize_belief_type(None)
        assert result == "fact"

    def test_valid_type_normalized(self):
        """Valid belief types should be normalized to lowercase."""
        result = WritersMixin._normalize_belief_type("HYPOTHESIS")
        assert result == "hypothesis"

    def test_whitespace_stripped(self):
        """Whitespace should be stripped from the input."""
        result = WritersMixin._normalize_belief_type("  opinion  ")
        assert result == "opinion"

    def test_invalid_type_raises_error(self):
        """Invalid belief types should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid belief type"):
            WritersMixin._normalize_belief_type("guess")

    def test_non_string_raises_error(self):
        """Non-string types should raise ValueError."""
        with pytest.raises(ValueError, match="belief_type must be a string"):
            WritersMixin._normalize_belief_type(123)


class TestNormalizeNoteType:
    """Tests for WritersMixin._normalize_note_type class method."""

    def test_none_defaults_to_note(self):
        """None should return 'note'."""
        result = WritersMixin._normalize_note_type(None)
        assert result == "note"

    def test_valid_type_normalized(self):
        """Valid note types should be normalized to lowercase."""
        result = WritersMixin._normalize_note_type("DECISION")
        assert result == "decision"

    def test_invalid_type_raises_error(self):
        """Invalid note types should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid note type"):
            WritersMixin._normalize_note_type("journal")

    def test_non_string_raises_error(self):
        """Non-string types should raise ValueError."""
        with pytest.raises(ValueError, match="note_type must be a string"):
            WritersMixin._normalize_note_type(True)


# =========================================================================
# Batch method enrichment
# =========================================================================


class TestEpisodesBatchEnrichment:
    """episodes_batch() should infer outcome_type and build derived_from."""

    def test_infers_outcome_type_success(self, mocked_kernle):
        """Batch episodes should infer outcome_type from outcome text."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_episodes_batch = MagicMock(return_value=["id1"])

        k.episodes_batch(
            [
                {"objective": "Fix bug", "outcome": "Fixed successfully"},
            ]
        )

        saved = mock_wb.save_episodes_batch.call_args[0][0]
        assert saved[0].outcome_type == "success"

    def test_infers_outcome_type_failure(self, mocked_kernle):
        """Batch episodes should detect failure keywords."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_episodes_batch = MagicMock(return_value=["id1"])

        k.episodes_batch(
            [
                {"objective": "Deploy", "outcome": "Build failed"},
            ]
        )

        saved = mock_wb.save_episodes_batch.call_args[0][0]
        assert saved[0].outcome_type == "failure"

    def test_explicit_outcome_type_preserved(self, mocked_kernle):
        """Explicit outcome_type should not be overridden by inference."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_episodes_batch = MagicMock(return_value=["id1"])

        k.episodes_batch(
            [
                {
                    "objective": "Fix bug",
                    "outcome": "Fixed successfully",
                    "outcome_type": "partial",
                },
            ]
        )

        saved = mock_wb.save_episodes_batch.call_args[0][0]
        assert saved[0].outcome_type == "partial"

    def test_builds_derived_from_with_source(self, mocked_kernle):
        """Batch episodes should build derived_from with source context marker."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_episodes_batch = MagicMock(return_value=["id1"])

        k.episodes_batch(
            [
                {
                    "objective": "Fix bug",
                    "outcome": "Done",
                    "derived_from": ["episode:ep1"],
                    "source": "code review",
                },
            ]
        )

        saved = mock_wb.save_episodes_batch.call_args[0][0]
        assert saved[0].derived_from == ["episode:ep1", "context:code review"]


class TestBeliefsBatchEnrichment:
    """beliefs_batch() should clamp confidence and build derived_from."""

    def test_clamps_confidence_above_one(self, mocked_kernle):
        """Batch beliefs should clamp confidence > 1.0 to 1.0."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_beliefs_batch = MagicMock(return_value=["id1"])

        k.beliefs_batch(
            [
                {"statement": "Python is great", "confidence": 1.5},
            ]
        )

        saved = mock_wb.save_beliefs_batch.call_args[0][0]
        assert saved[0].confidence == 1.0

    def test_clamps_confidence_below_zero(self, mocked_kernle):
        """Batch beliefs should clamp confidence < 0.0 to 0.0."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_beliefs_batch = MagicMock(return_value=["id1"])

        k.beliefs_batch(
            [
                {"statement": "PHP is bad", "confidence": -0.5},
            ]
        )

        saved = mock_wb.save_beliefs_batch.call_args[0][0]
        assert saved[0].confidence == 0.0

    def test_builds_derived_from_with_source(self, mocked_kernle):
        """Batch beliefs should build derived_from with source context marker."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_beliefs_batch = MagicMock(return_value=["id1"])

        k.beliefs_batch(
            [
                {
                    "statement": "Tests matter",
                    "derived_from": ["note:n1"],
                    "source": "reading docs",
                },
            ]
        )

        saved = mock_wb.save_beliefs_batch.call_args[0][0]
        assert saved[0].derived_from == ["note:n1", "context:reading docs"]


class TestNotesBatchEnrichment:
    """notes_batch() should format content and build derived_from."""

    def test_formats_decision_content(self, mocked_kernle):
        """Batch notes should format decision-type content."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_notes_batch = MagicMock(return_value=["id1"])

        k.notes_batch(
            [
                {"content": "Use TypeScript", "type": "decision", "reason": "Type safety"},
            ]
        )

        saved = mock_wb.save_notes_batch.call_args[0][0]
        assert saved[0].content == "**Decision**: Use TypeScript\n**Reason**: Type safety"

    def test_formats_quote_content(self, mocked_kernle):
        """Batch notes should format quote-type content."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_notes_batch = MagicMock(return_value=["id1"])

        k.notes_batch(
            [
                {"content": "Move fast", "type": "quote", "speaker": "Zuck"},
            ]
        )

        saved = mock_wb.save_notes_batch.call_args[0][0]
        assert saved[0].content == '> "Move fast"\n> — Zuck'

    def test_formats_insight_content(self, mocked_kernle):
        """Batch notes should format insight-type content."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_notes_batch = MagicMock(return_value=["id1"])

        k.notes_batch(
            [
                {"content": "Users prefer dark mode", "type": "insight"},
            ]
        )

        saved = mock_wb.save_notes_batch.call_args[0][0]
        assert saved[0].content == "**Insight**: Users prefer dark mode"

    def test_builds_derived_from_with_source(self, mocked_kernle):
        """Batch notes should build derived_from with source context marker."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.save_notes_batch = MagicMock(return_value=["id1"])

        k.notes_batch(
            [
                {
                    "content": "Important note",
                    "derived_from": ["episode:ep1"],
                    "source": "standup",
                },
            ]
        )

        saved = mock_wb.save_notes_batch.call_args[0][0]
        assert saved[0].derived_from == ["episode:ep1", "context:standup"]


# =========================================================================
# source_type parameter coverage for all writer methods
# =========================================================================


class TestNoteSourceType:
    """note() should handle source_type parameter correctly."""

    def test_default_source_type(self, mocked_kernle):
        """note() with no source_type should default to direct_experience."""
        k, mock_wb, _ = mocked_kernle
        k.note(content="Test note")
        saved_note = mock_wb.save_note.call_args[0][0]
        assert saved_note.source_type == SourceType.DIRECT_EXPERIENCE

    def test_enum_source_type(self, mocked_kernle):
        """note() with SourceType enum should use it directly."""
        k, mock_wb, _ = mocked_kernle
        k.note(content="Test note", source_type=SourceType.EXTERNAL)
        saved_note = mock_wb.save_note.call_args[0][0]
        assert saved_note.source_type == SourceType.EXTERNAL

    def test_string_source_type(self, mocked_kernle):
        """note() with valid string source_type should convert to SourceType."""
        k, mock_wb, _ = mocked_kernle
        k.note(content="Test note", source_type="external")
        saved_note = mock_wb.save_note.call_args[0][0]
        assert saved_note.source_type == SourceType.EXTERNAL

    def test_invalid_source_type_raises_error(self, mocked_kernle):
        """note() with invalid source_type should raise ValueError."""
        k, _, _ = mocked_kernle
        with pytest.raises(ValueError, match="Invalid source_type"):
            k.note(content="Test note", source_type="telepathy")


class TestValueSourceType:
    """value() should handle source_type parameter correctly."""

    def test_default_source_type(self, mocked_kernle):
        """value() with no source_type should default to direct_experience."""
        k, mock_wb, _ = mocked_kernle
        k.value(name="honesty", statement="Be truthful")
        saved_value = mock_wb.save_value.call_args[0][0]
        assert saved_value.source_type == SourceType.DIRECT_EXPERIENCE

    def test_enum_source_type(self, mocked_kernle):
        """value() with SourceType enum should use it directly."""
        k, mock_wb, _ = mocked_kernle
        k.value(name="honesty", statement="Be truthful", source_type=SourceType.EXTERNAL)
        saved_value = mock_wb.save_value.call_args[0][0]
        assert saved_value.source_type == SourceType.EXTERNAL

    def test_string_source_type(self, mocked_kernle):
        """value() with valid string source_type should convert to SourceType."""
        k, mock_wb, _ = mocked_kernle
        k.value(name="honesty", statement="Be truthful", source_type="external")
        saved_value = mock_wb.save_value.call_args[0][0]
        assert saved_value.source_type == SourceType.EXTERNAL

    def test_invalid_source_type_raises_error(self, mocked_kernle):
        """value() with invalid source_type should raise ValueError."""
        k, _, _ = mocked_kernle
        with pytest.raises(ValueError, match="Invalid source_type"):
            k.value(name="honesty", statement="Be truthful", source_type="telepathy")


class TestGoalSourceType:
    """goal() should handle source_type parameter correctly."""

    def test_default_source_type(self, mocked_kernle):
        """goal() with no source_type should default to direct_experience."""
        k, mock_wb, _ = mocked_kernle
        k.goal(title="Learn Rust")
        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.source_type == SourceType.DIRECT_EXPERIENCE

    def test_enum_source_type(self, mocked_kernle):
        """goal() with SourceType enum should use it directly."""
        k, mock_wb, _ = mocked_kernle
        k.goal(title="Learn Rust", source_type=SourceType.EXTERNAL)
        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.source_type == SourceType.EXTERNAL

    def test_string_source_type(self, mocked_kernle):
        """goal() with valid string source_type should convert to SourceType."""
        k, mock_wb, _ = mocked_kernle
        k.goal(title="Learn Rust", source_type="external")
        saved_goal = mock_wb.save_goal.call_args[0][0]
        assert saved_goal.source_type == SourceType.EXTERNAL

    def test_invalid_source_type_raises_error(self, mocked_kernle):
        """goal() with invalid source_type should raise ValueError."""
        k, _, _ = mocked_kernle
        with pytest.raises(ValueError, match="Invalid source_type"):
            k.goal(title="Learn Rust", source_type="telepathy")


class TestDriveSourceType:
    """drive() should handle source_type parameter correctly."""

    def test_default_source_type(self, mocked_kernle):
        """drive() with no source_type should default to direct_experience."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.get_drives.return_value = []
        k.drive("curiosity", intensity=0.7)
        saved_drive = mock_wb.save_drive.call_args[0][0]
        assert saved_drive.source_type == SourceType.DIRECT_EXPERIENCE

    def test_enum_source_type(self, mocked_kernle):
        """drive() with SourceType enum should use it directly."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.get_drives.return_value = []
        k.drive("curiosity", intensity=0.7, source_type=SourceType.CONSOLIDATION)
        saved_drive = mock_wb.save_drive.call_args[0][0]
        assert saved_drive.source_type == SourceType.CONSOLIDATION

    def test_string_source_type(self, mocked_kernle):
        """drive() with valid string source_type should convert to SourceType."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.get_drives.return_value = []
        k.drive("curiosity", intensity=0.7, source_type="consolidation")
        saved_drive = mock_wb.save_drive.call_args[0][0]
        assert saved_drive.source_type == SourceType.CONSOLIDATION

    def test_invalid_source_type_raises_error(self, mocked_kernle):
        """drive() with invalid source_type should raise ValueError."""
        k, _, _ = mocked_kernle
        with pytest.raises(ValueError, match="Invalid source_type"):
            k.drive("curiosity", intensity=0.7, source_type="telepathy")


class TestRelationshipSourceType:
    """relationship() should handle source_type and source_entity parameters correctly."""

    def test_default_source_type(self, mocked_kernle):
        """relationship() with no source_type should default to direct_experience."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.get_relationship.return_value = None
        k.relationship(other_stack_id="Alice", trust_level=0.8)
        saved_rel = mock_wb.save_relationship.call_args[0][0]
        assert saved_rel.source_type == SourceType.DIRECT_EXPERIENCE

    def test_enum_source_type(self, mocked_kernle):
        """relationship() with SourceType enum should use it directly."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.get_relationship.return_value = None
        k.relationship(other_stack_id="Alice", trust_level=0.8, source_type=SourceType.EXTERNAL)
        saved_rel = mock_wb.save_relationship.call_args[0][0]
        assert saved_rel.source_type == SourceType.EXTERNAL

    def test_string_source_type(self, mocked_kernle):
        """relationship() with valid string source_type should convert to SourceType."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.get_relationship.return_value = None
        k.relationship(other_stack_id="Alice", trust_level=0.8, source_type="external")
        saved_rel = mock_wb.save_relationship.call_args[0][0]
        assert saved_rel.source_type == SourceType.EXTERNAL

    def test_invalid_source_type_raises_error(self, mocked_kernle):
        """relationship() with invalid source_type should raise ValueError."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.get_relationship.return_value = None
        with pytest.raises(ValueError, match="Invalid source_type"):
            k.relationship(other_stack_id="Alice", source_type="telepathy")

    def test_source_entity_passed_through(self, mocked_kernle):
        """relationship() should pass source_entity to the Relationship object."""
        k, mock_wb, _ = mocked_kernle
        mock_wb.get_relationship.return_value = None
        k.relationship(
            other_stack_id="Alice",
            trust_level=0.8,
            source_entity="user:bob",
        )
        saved_rel = mock_wb.save_relationship.call_args[0][0]
        assert saved_rel.source_entity == "user:bob"


# =========================================================================
# process_raw: blob/content field usage (#835)
# =========================================================================


class TestProcessRawBlobUsage:
    """process_raw() should use entry.blob (preferred) with content as fallback."""

    def test_process_raw_note_uses_blob(self, mocked_kernle):
        """When a raw entry has blob set, process_raw should use blob as note content."""
        k, mock_wb, mock_storage = mocked_kernle

        raw_entry = MagicMock()
        raw_entry.blob = "This is the blob content"
        raw_entry.content = "This is the old content field"
        raw_entry.processed = False
        raw_entry.tags = []
        mock_storage.get_raw.return_value = raw_entry

        k.process_raw(raw_id="raw-123", as_type="note")

        saved_note = mock_wb.save_note.call_args[0][0]
        assert "blob content" in saved_note.content

    def test_process_raw_note_falls_back_to_content(self, mocked_kernle):
        """When blob is None, process_raw should fall back to content for notes."""
        k, mock_wb, mock_storage = mocked_kernle

        raw_entry = MagicMock()
        raw_entry.blob = None
        raw_entry.content = "Legacy content value"
        raw_entry.processed = False
        raw_entry.tags = []
        mock_storage.get_raw.return_value = raw_entry

        k.process_raw(raw_id="raw-123", as_type="note")

        saved_note = mock_wb.save_note.call_args[0][0]
        assert "Legacy content value" in saved_note.content

    def test_process_raw_episode_uses_blob(self, mocked_kernle):
        """Episode path should already use blob - verify it works correctly."""
        k, mock_wb, mock_storage = mocked_kernle

        raw_entry = MagicMock()
        raw_entry.blob = "Completed the deployment with zero downtime"
        raw_entry.content = "Old content"
        raw_entry.processed = False
        raw_entry.tags = []
        mock_storage.get_raw.return_value = raw_entry

        k.process_raw(raw_id="raw-123", as_type="episode")

        saved_episode = mock_wb.save_episode.call_args[0][0]
        assert "Completed the deployment" in saved_episode.objective
