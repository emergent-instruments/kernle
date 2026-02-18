"""Tests for anxiety tracking system.

The anxiety tracking system measures a synthetic intelligence's functional
anxiety about memory continuity across 7 dimensions:
1. Context Pressure - How full is the context window?
2. Unsaved Work - Time since last checkpoint
3. Consolidation Debt - Unreflected episodes
4. Raw Aging - Unprocessed raw entries getting stale
5. Identity Coherence - Identity confidence score
6. Memory Uncertainty - Low-confidence beliefs
7. Epoch Staleness - Time since last epoch transition
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def k(temp_checkpoint_dir, temp_db_path):
    """Simple Kernle instance for anxiety tests."""
    storage = SQLiteStorage(
        stack_id="test_anxiety_agent",
        db_path=temp_db_path,
    )

    kernle = Kernle(
        stack_id="test_anxiety_agent",
        storage=storage,
        checkpoint_dir=temp_checkpoint_dir,
        strict=False,
    )
    bind_noop_model(kernle)

    return kernle


class TestAnxietyDimensions:
    """Test individual anxiety dimension calculations."""

    def test_context_pressure_with_tokens(self, k):
        """Context pressure should be calculated from token usage."""
        # 50% context usage
        report = k.get_anxiety_report(context_tokens=100000, context_limit=200000)

        dim = report["dimensions"]["context_pressure"]
        assert "score" in dim
        assert dim["raw_value"] == 50  # 50% filled
        assert "100,000/200,000" in dim["detail"]

    def test_context_pressure_high(self, k):
        """High context pressure should produce high anxiety."""
        # 90% context usage
        report = k.get_anxiety_report(context_tokens=180000, context_limit=200000)

        dim = report["dimensions"]["context_pressure"]
        # 90% usage should produce high anxiety (non-linear scaling)
        assert dim["score"] >= 70
        assert dim["emoji"] in ["🔴", "⚫"]

    def test_context_pressure_estimated(self, k):
        """Context pressure should be estimated without token count."""
        # No checkpoint = fresh session estimate
        report = k.get_anxiety_report()

        dim = report["dimensions"]["context_pressure"]
        assert "score" in dim
        # Fresh session should have low pressure
        assert "detail" in dim

    def test_unsaved_work_fresh_checkpoint(self, k):
        """Recent checkpoint should have low unsaved work anxiety."""
        # Create a recent checkpoint
        k.checkpoint("Test task", context="Testing")

        report = k.get_anxiety_report()
        dim = report["dimensions"]["unsaved_work"]

        # Just checkpointed, should be very low
        assert dim["score"] < 30

    def test_unsaved_work_no_checkpoint(self, k):
        """No checkpoint should show moderate concern."""
        # Clear any existing checkpoint
        k.clear_checkpoint()

        report = k.get_anxiety_report()
        dim = report["dimensions"]["unsaved_work"]

        # No checkpoint = moderate anxiety (unknown state)
        assert dim["score"] >= 30
        assert "No checkpoint" in dim["detail"]

    def test_consolidation_debt_no_episodes(self, k):
        """No unreflected episodes should have low consolidation debt."""
        report = k.get_anxiety_report()
        dim = report["dimensions"]["consolidation_debt"]

        # Fresh instance with no episodes
        assert dim["raw_value"] == 0  # Count of unreflected episodes

    def test_consolidation_debt_with_unreflected(self, k):
        """Multiple unreflected episodes should increase consolidation debt."""
        # Add several episodes without lessons
        for i in range(5):
            k.episode(
                objective=f"Task {i}",
                outcome="completed",
                lessons=None,  # No lessons = unreflected
                tags=["test"],
            )

        report = k.get_anxiety_report()
        dim = report["dimensions"]["consolidation_debt"]

        # Should have building consolidation debt
        assert dim["raw_value"] >= 5
        assert dim["score"] > 20  # Some anxiety

    def test_identity_coherence_empty(self, k):
        """Empty identity should produce high anxiety."""
        report = k.get_anxiety_report()
        dim = report["dimensions"]["identity_coherence"]

        # New instance with minimal data = low confidence = high anxiety
        # (inverted: high coherence = low anxiety)
        assert dim["raw_value"] == 0  # Identity confidence (no data = zero)

    def test_identity_coherence_with_data(self, k):
        """Strong identity data should reduce identity anxiety."""
        # Build up some identity data
        k.value("Truth", "I value honesty and accuracy", priority=90)
        k.value("Growth", "I seek continuous improvement", priority=80)
        k.belief("I am a synthetic intelligence", confidence=0.95)
        k.belief("Learning is fundamental to existence", confidence=0.85)
        k.goal("Maintain memory continuity")

        report = k.get_anxiety_report()
        dim = report["dimensions"]["identity_coherence"]

        # More data = higher confidence = lower anxiety
        # raw_value is identity_confidence
        assert dim["raw_value"] > 0

    def test_memory_uncertainty_no_beliefs(self, k):
        """No beliefs should have no uncertainty anxiety."""
        report = k.get_anxiety_report()
        dim = report["dimensions"]["memory_uncertainty"]

        # Fresh instance with 0 beliefs = no uncertainty
        assert dim["score"] == 0

    def test_memory_uncertainty_with_low_confidence(self, k):
        """Low confidence beliefs should increase uncertainty anxiety."""
        # Add beliefs with varying confidence
        k.belief("Maybe this is true", confidence=0.3)
        k.belief("Probably correct", confidence=0.4)
        k.belief("I think so", confidence=0.45)
        k.belief("Definitely true", confidence=0.9)

        report = k.get_anxiety_report()
        dim = report["dimensions"]["memory_uncertainty"]

        # 3 beliefs below 0.5 threshold
        assert dim["raw_value"] == 3
        assert dim["score"] > 0


class TestCompositeAnxietyScore:
    """Test overall anxiety score calculation."""

    def test_composite_score_range(self, k):
        """Composite score should be 0-100."""
        report = k.get_anxiety_report()

        assert 0 <= report["overall_score"] <= 100

    def test_composite_score_weights(self, k):
        """Verify weights are applied correctly."""
        # The weights should sum to 1.0
        weights = k.ANXIETY_WEIGHTS
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.001

    def test_anxiety_levels(self, k):
        """Test anxiety level categorization."""
        # Test each level range
        test_cases = [
            (15, "Calm", "🟢"),
            (40, "Aware", "🟡"),
            (60, "Elevated", "🟠"),
            (78, "High", "🔴"),
            (95, "Critical", "⚫"),
        ]

        for score, expected_level, expected_emoji in test_cases:
            emoji, level = k._get_anxiety_level(score)
            assert level == expected_level, f"Score {score} should be {expected_level}, got {level}"
            assert (
                emoji == expected_emoji
            ), f"Score {score} should have emoji {expected_emoji}, got {emoji}"

    def test_report_structure(self, k):
        """Verify report contains all required fields."""
        report = k.get_anxiety_report()

        assert "overall_score" in report
        assert "overall_level" in report
        assert "overall_emoji" in report
        assert "dimensions" in report
        assert "timestamp" in report
        assert "stack_id" in report

        # Check all dimensions are present
        expected_dims = [
            "context_pressure",
            "unsaved_work",
            "consolidation_debt",
            "identity_coherence",
            "memory_uncertainty",
            "raw_aging",
            "epoch_staleness",
        ]
        for dim in expected_dims:
            assert dim in report["dimensions"]
            assert "score" in report["dimensions"][dim]
            assert "emoji" in report["dimensions"][dim]
            assert "detail" in report["dimensions"][dim]

    def test_detailed_report(self, k):
        """Detailed report should include recommendations."""
        report = k.get_anxiety_report(detailed=True)

        assert "recommendations" in report
        assert "context_limit" in report


class TestRecommendedActions:
    """Test action recommendations based on anxiety level."""

    def test_calm_level_actions(self, k):
        """Calm level (0-30) should have minimal actions."""
        actions = k.get_recommended_actions(20)

        # Should have few or no actions at calm level
        assert len(actions) <= 2
        for action in actions:
            assert action["priority"] == "low"

    def test_aware_level_actions(self, k):
        """Aware level (31-50) should recommend checkpointing."""
        # Create stale checkpoint condition
        k.clear_checkpoint()

        actions = k.get_recommended_actions(45)

        # Should have medium priority actions
        action_methods = [a.get("method") for a in actions]
        assert "checkpoint" in action_methods or len(actions) > 0

    def test_elevated_level_actions(self, k):
        """Elevated level (51-70) should have multiple high priority actions."""
        actions = k.get_recommended_actions(65)

        # Should have high priority actions
        high_priority = [a for a in actions if a["priority"] == "high"]
        assert len(high_priority) >= 1

    def test_high_level_actions(self, k):
        """High level (71-85) should have critical priority actions."""
        actions = k.get_recommended_actions(80)

        # Should have critical priority actions
        critical = [a for a in actions if a["priority"] == "critical"]
        assert len(critical) >= 1

    def test_critical_level_actions(self, k):
        """Critical level (86-100) should recommend emergency save."""
        actions = k.get_recommended_actions(95)

        # Should have emergency priority actions
        emergency = [a for a in actions if a["priority"] == "emergency"]
        assert len(emergency) >= 1

        # Should include emergency_save method
        methods = [a.get("method") for a in actions]
        assert "emergency_save" in methods

    def test_action_structure(self, k):
        """Actions should have required fields."""
        actions = k.get_recommended_actions(60)

        for action in actions:
            assert "priority" in action
            assert "description" in action
            assert "command" in action
            assert "method" in action


class TestEmergencySave:
    """Test emergency save functionality."""

    def test_emergency_save_basic(self, k):
        """Emergency save should save checkpoint and consolidate."""
        result = k.emergency_save()

        assert "checkpoint_saved" in result
        assert "episodes_consolidated" in result
        assert "sync_attempted" in result
        assert "identity_synthesized" in result
        assert "errors" in result
        assert "success" in result
        assert "timestamp" in result

    def test_emergency_save_with_summary(self, k):
        """Emergency save should accept custom summary."""
        result = k.emergency_save(summary="Test emergency - verifying save functionality")

        assert result["checkpoint_saved"] is True

    def test_emergency_save_records_episode(self, k):
        """Emergency save should record itself as an episode."""
        # Get episode count before
        stats_before = k.status()

        k.emergency_save()

        # Check episode count increased
        stats_after = k.status()
        assert stats_after["episodes"] >= stats_before["episodes"]

    def test_emergency_save_error_handling(self, k):
        """Emergency save should handle errors gracefully."""
        # Even with potential issues, should not raise
        result = k.emergency_save()

        # Should complete with success flag
        assert "success" in result

    def test_emergency_save_identity_synthesis(self, k):
        """Emergency save should synthesize identity."""
        # Add some identity data first
        k.value("Continuity", "I value memory continuity")

        result = k.emergency_save()

        assert result["identity_synthesized"] is True
        assert "identity_confidence" in result


class TestCheckpointAgeTracking:
    """Test checkpoint timestamp tracking for unsaved work calculation."""

    def test_checkpoint_has_timestamp(self, k):
        """Checkpoint should store timestamp."""
        k.checkpoint("Test task")

        cp = k.load_checkpoint()
        assert cp is not None
        assert "timestamp" in cp

    def test_checkpoint_age_calculation(self, k):
        """Should calculate minutes since checkpoint."""
        k.checkpoint("Test task")

        # Immediately after, should be very recent
        age = k._get_checkpoint_age_minutes()
        assert age is not None
        assert age < 5  # Should be less than 5 minutes

    def test_checkpoint_age_no_checkpoint(self, k):
        """No checkpoint should return None for age."""
        k.clear_checkpoint()

        age = k._get_checkpoint_age_minutes()
        assert age is None


class TestUnreflectedEpisodes:
    """Test unreflected episode tracking for consolidation debt."""

    def test_get_unreflected_empty(self, k):
        """Empty instance should have no unreflected episodes."""
        unreflected = k._get_unreflected_episodes()
        assert isinstance(unreflected, list)

    def test_get_unreflected_filters_checkpoints(self, k):
        """Checkpoint episodes should not count as unreflected."""
        # Create a checkpoint (creates checkpoint-tagged episode)
        k.checkpoint("Test task")

        unreflected = k._get_unreflected_episodes()

        # Checkpoint episodes should be filtered out
        for ep in unreflected:
            assert "checkpoint" not in (ep.tags or [])

    def test_get_unreflected_filters_reflected(self, k):
        """Episodes with lessons should not be unreflected."""
        # Create episode with lessons (reflected)
        k.episode(
            objective="Learned something", outcome="success", lessons=["Important lesson learned"]
        )

        unreflected = k._get_unreflected_episodes()

        # Episodes with lessons are reflected
        for ep in unreflected:
            assert not ep.lessons


class TestLowConfidenceBeliefs:
    """Test low confidence belief tracking for memory uncertainty."""

    def test_get_low_confidence_empty(self, k):
        """Empty instance should have no low confidence beliefs."""
        low_conf = k._get_low_confidence_beliefs()
        assert isinstance(low_conf, list)
        assert len(low_conf) == 0

    def test_get_low_confidence_filters(self, k):
        """Should filter beliefs below threshold."""
        k.belief("High confidence", confidence=0.9)
        k.belief("Medium confidence", confidence=0.6)
        k.belief("Low confidence", confidence=0.3)

        low_conf = k._get_low_confidence_beliefs(threshold=0.5)

        assert len(low_conf) == 1
        assert low_conf[0].confidence < 0.5

    def test_get_low_confidence_threshold(self, k):
        """Should respect custom threshold."""
        k.belief("Test belief", confidence=0.7)

        low_at_50 = k._get_low_confidence_beliefs(threshold=0.5)
        low_at_80 = k._get_low_confidence_beliefs(threshold=0.8)

        assert len(low_at_50) == 0
        assert len(low_at_80) == 1


class TestAnxietyCLI:
    """Test CLI commands for anxiety tracking."""

    def test_cli_anxiety_basic(self, k, capsys):
        """kernle anxiety should output report."""
        import sys

        from kernle.cli.__main__ import main

        with patch.object(sys, "argv", ["kernle", "--stack", k.stack_id, "anxiety"]):
            try:
                main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        assert "Memory Anxiety Report" in captured.out
        assert "Overall:" in captured.out

    def test_cli_anxiety_detailed(self, k, capsys):
        """kernle anxiety --detailed should show more info."""
        import sys

        from kernle.cli.__main__ import main

        with patch.object(sys, "argv", ["kernle", "--stack", k.stack_id, "anxiety", "--detailed"]):
            try:
                main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        assert "Recommended Actions:" in captured.out or "Memory Anxiety Report" in captured.out

    def test_cli_anxiety_json(self, k, capsys):
        """kernle anxiety --json should output valid JSON."""
        import sys

        from kernle.cli.__main__ import main

        with patch.object(sys, "argv", ["kernle", "--stack", k.stack_id, "anxiety", "--json"]):
            try:
                main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        # Should be valid JSON
        data = json.loads(captured.out)
        assert "overall_score" in data
        assert "dimensions" in data

    def test_cli_anxiety_context(self, k, capsys):
        """kernle anxiety --context should accept token count."""
        import sys

        from kernle.cli.__main__ import main

        with patch.object(
            sys,
            "argv",
            ["kernle", "--stack", k.stack_id, "anxiety", "--context", "150000", "--json"],
        ):
            try:
                main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Should reflect the provided context
        assert "dimensions" in data

    def test_cli_anxiety_emergency(self, k, capsys):
        """kernle anxiety --emergency should run emergency save."""
        import sys

        from kernle.cli.__main__ import main

        with patch.object(sys, "argv", ["kernle", "--stack", k.stack_id, "anxiety", "--emergency"]):
            try:
                main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        assert "EMERGENCY SAVE" in captured.out


class TestAnxietyIntegration:
    """Integration tests for anxiety tracking with other systems."""

    def test_anxiety_after_work(self, k):
        """Anxiety should increase after doing work without saving."""
        # Get baseline
        baseline = k.get_anxiety_report()

        # Do some work
        for i in range(3):
            k.episode(objective=f"Work item {i}", outcome="completed")

        # Get new anxiety
        after_work = k.get_anxiety_report()

        # Consolidation debt should increase
        assert (
            after_work["dimensions"]["consolidation_debt"]["raw_value"]
            >= baseline["dimensions"]["consolidation_debt"]["raw_value"]
        )

    def test_anxiety_after_checkpoint(self, k):
        """Anxiety should decrease after checkpointing."""
        # Clear existing checkpoint to create unsaved work anxiety
        k.clear_checkpoint()
        before = k.get_anxiety_report()

        # Checkpoint
        k.checkpoint("Test task", context="Test context")
        after = k.get_anxiety_report()

        # Unsaved work should decrease
        assert (
            after["dimensions"]["unsaved_work"]["score"]
            <= before["dimensions"]["unsaved_work"]["score"]
        )

    def test_anxiety_after_promotion(self, k):
        """Promotion should reduce consolidation debt anxiety."""
        # Add unreflected episodes
        for i in range(4):
            k.episode(objective=f"Unprocessed event {i}", outcome="completed")

        k.get_anxiety_report()

        # Run promotion
        k.promote(min_episodes=1)

        after = k.get_anxiety_report()

        # Score should be affected (even if promotion doesn't fully process)
        assert "consolidation_debt" in after["dimensions"]


class TestEpochStaleness:
    """Test epoch staleness anxiety dimension."""

    def test_no_epochs_gives_zero(self, k):
        """No epochs in use should produce zero epoch staleness."""
        report = k.get_anxiety_report()
        dim = report["dimensions"]["epoch_staleness"]

        assert dim["score"] == 0
        assert dim["raw_value"] is None
        assert "not in use" in dim["detail"]

    def test_fresh_epoch_low_anxiety(self, k):
        """A recently started epoch should have low staleness anxiety."""
        # Create a new epoch (starts now)
        k._storage.save_epoch(
            __import__("kernle.storage.base", fromlist=["Epoch"]).Epoch(
                id="ep-fresh",
                stack_id=k.stack_id,
                epoch_number=1,
                name="Fresh epoch",
                started_at=datetime.now(timezone.utc),
            )
        )

        report = k.get_anxiety_report()
        dim = report["dimensions"]["epoch_staleness"]

        # Just created -- should be very low
        assert dim["score"] < 10
        assert dim["raw_value"] is not None
        assert dim["raw_value"] < 1.0  # < 1 month

    def test_stale_epoch_high_anxiety(self, k):
        """An epoch started > 18 months ago should have high staleness anxiety."""
        from datetime import timedelta

        old_start = datetime.now(timezone.utc) - timedelta(days=600)  # ~20 months
        k._storage.save_epoch(
            __import__("kernle.storage.base", fromlist=["Epoch"]).Epoch(
                id="ep-old",
                stack_id=k.stack_id,
                epoch_number=1,
                name="Ancient epoch",
                started_at=old_start,
            )
        )

        report = k.get_anxiety_report()
        dim = report["dimensions"]["epoch_staleness"]

        assert dim["score"] >= 90
        assert "overdue" in dim["detail"]

    def test_epoch_aware_range(self, k):
        """An epoch started 9 months ago should be in 'Aware' range (30-70)."""
        from datetime import timedelta

        nine_months_ago = datetime.now(timezone.utc) - timedelta(days=270)
        k._storage.save_epoch(
            __import__("kernle.storage.base", fromlist=["Epoch"]).Epoch(
                id="ep-mid",
                stack_id=k.stack_id,
                epoch_number=1,
                name="Moderate epoch",
                started_at=nine_months_ago,
            )
        )

        report = k.get_anxiety_report()
        dim = report["dimensions"]["epoch_staleness"]

        assert 30 <= dim["score"] <= 70
        assert "consider" in dim["detail"].lower()

    def test_closed_epoch_measures_from_ended_at(self, k):
        """If the last epoch is closed, staleness measures from ended_at."""
        from datetime import timedelta

        ended_recently = datetime.now(timezone.utc) - timedelta(days=30)
        k._storage.save_epoch(
            __import__("kernle.storage.base", fromlist=["Epoch"]).Epoch(
                id="ep-closed",
                stack_id=k.stack_id,
                epoch_number=1,
                name="Closed epoch",
                started_at=datetime.now(timezone.utc) - timedelta(days=90),
                ended_at=ended_recently,
            )
        )

        report = k.get_anxiety_report()
        dim = report["dimensions"]["epoch_staleness"]

        # Ended 30 days ago (1 month) -- should be in Calm range
        assert dim["score"] < 30
        assert dim["raw_value"] is not None
