"""Additional tests for anxiety.py to improve coverage.

Targets uncovered lines:
- Line 53: _get_anxiety_level fallback for out-of-range scores
- Lines 66-67: _get_checkpoint_age_minutes error handling
- Lines 96-110: _get_aging_raw_entries loop with timestamps
- Lines 124, 139: String epoch timestamps
- Lines 144-146: Exception handling in _get_epoch_staleness_months
- Lines 215-220: Unsaved work for 60+ min checkpoint age
- Lines 239-244: Consolidation debt for 8-15 unreflected episodes
- Lines 258, 260: Identity coherence detail text for low/mid confidence
- Lines 291-292: High uncertainty with > 5 low-confidence beliefs
- Lines 309-322: Raw aging for 4-7 and 8+ aging entries
- Lines 351-352: Epoch staleness elevated (12-18 months)
- Line 401: anxiety() alias
- Lines 424, 446, 467, 485, 523: Recommended actions edge cases
- Lines 593-623: Emergency save error handling paths
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def k(temp_checkpoint_dir, temp_db_path):
    """Kernle instance for anxiety coverage tests."""
    storage = SQLiteStorage(
        stack_id="test_anxiety_cov",
        db_path=temp_db_path,
    )
    inst = Kernle(
        stack_id="test_anxiety_cov",
        storage=storage,
        checkpoint_dir=temp_checkpoint_dir,
        strict=False,
    )
    bind_noop_model(inst)
    return inst


class TestGetAnxietyLevelFallback:
    """Test _get_anxiety_level fallback path (line 53)."""

    def test_score_above_100_returns_critical(self, k):
        """Score > 100 should fall through ranges and return Critical fallback."""
        emoji, label = k._get_anxiety_level(150)
        assert emoji == "\u26ab"
        assert label == "Critical"

    def test_score_negative_returns_critical(self, k):
        """Negative score should fall through ranges and return Critical fallback."""
        emoji, label = k._get_anxiety_level(-5)
        assert emoji == "\u26ab"
        assert label == "Critical"


class TestCheckpointAgeErrorHandling:
    """Test _get_checkpoint_age_minutes error handling (lines 66-67)."""

    def test_invalid_timestamp_format_returns_none(self, k):
        """Invalid timestamp in checkpoint should return None."""
        # Create a checkpoint then patch its timestamp with invalid data
        k.checkpoint("Test task")
        with patch.object(k, "load_checkpoint", return_value={"timestamp": "not-a-date"}):
            age = k._get_checkpoint_age_minutes()
            assert age is None

    def test_value_error_with_z_timestamp_returns_none(self, k):
        """Timestamp with Z suffix but invalid body should return None (line 67)."""
        # "not-a-dateZ" -> "not-a-date+00:00" which raises ValueError in fromisoformat
        with patch.object(k, "load_checkpoint", return_value={"timestamp": "badZ"}):
            age = k._get_checkpoint_age_minutes()
            assert age is None


class TestGetAgingRawEntries:
    """Test _get_aging_raw_entries (lines 96-110)."""

    def test_aging_entries_counted(self, k):
        """Raw entries older than age_hours should be counted."""
        # Create raw entries with timestamps in the past
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        mock_entries = [
            SimpleNamespace(id="1", captured_at=old_time, timestamp=old_time),
            SimpleNamespace(id="2", captured_at=recent_time, timestamp=recent_time),
            SimpleNamespace(id="3", captured_at=old_time, timestamp=old_time),
        ]
        with patch.object(k, "list_raw", return_value=mock_entries):
            total, aging, oldest, _ = k._get_aging_raw_entries(24)
            assert total == 3
            assert aging == 2  # Two entries > 24h old
            assert oldest > 24

    def test_no_aging_entries(self, k):
        """All fresh entries should have aging_count 0."""
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_entries = [
            SimpleNamespace(id="1", captured_at=recent_time, timestamp=None),
            SimpleNamespace(id="2", captured_at=recent_time, timestamp=None),
        ]
        with patch.object(k, "list_raw", return_value=mock_entries):
            total, aging, oldest, _ = k._get_aging_raw_entries(24)
            assert total == 2
            assert aging == 0
            assert oldest < 24

    def test_empty_raw_entries(self, k):
        """No raw entries should return zeros."""
        total, aging, oldest, _ = k._get_aging_raw_entries(24)
        assert total == 0
        assert aging == 0
        assert oldest == 0

    def test_invalid_timestamp_in_entry_skipped(self, k):
        """Entries with invalid timestamps should be skipped (line 109-110)."""
        mock_entries = [
            SimpleNamespace(id="1", captured_at="bad-date", timestamp="bad-date"),
            SimpleNamespace(id="2", captured_at=None, timestamp=None),
            SimpleNamespace(id="3", captured_at=None, timestamp=None),
        ]
        with patch.object(k, "list_raw", return_value=mock_entries):
            total, aging, oldest, _ = k._get_aging_raw_entries(24)
            assert total == 3
            assert aging == 0  # All skipped due to invalid timestamps

    def test_uses_captured_at_over_timestamp(self, k):
        """Should prefer captured_at over deprecated timestamp field."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        mock_entries = [
            SimpleNamespace(id="1", captured_at=old_time, timestamp=recent_time),
        ]
        with patch.object(k, "list_raw", return_value=mock_entries):
            total, aging, oldest, _ = k._get_aging_raw_entries(24)
            assert aging == 1  # Uses captured_at (old), not timestamp (recent)

    def test_falls_back_to_timestamp_field(self, k):
        """Should fall back to timestamp when captured_at is absent."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()

        mock_entries = [
            SimpleNamespace(id="1", captured_at=None, timestamp=old_time),
        ]
        with patch.object(k, "list_raw", return_value=mock_entries):
            total, aging, oldest, _ = k._get_aging_raw_entries(24)
            assert aging == 1


class TestEpochStalenessStringTimestamps:
    """Test string epoch timestamps (lines 124, 139)."""

    def test_string_started_at_parsed(self, k):
        """String started_at should be parsed correctly (line 124)."""
        from kernle.storage import Epoch

        # Create an epoch with string started_at
        epoch = Epoch(
            id="ep-str",
            stack_id=k.stack_id,
            epoch_number=1,
            name="String timestamp epoch",
            started_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        k._storage.save_epoch(epoch)

        # Mock so started_at is returned as a string
        original_get = k._storage.get_current_epoch

        def mock_current_epoch():
            ep = original_get()
            if ep:
                # Simulate string timestamp from database
                ep.started_at = (
                    ep.started_at.isoformat()
                    if isinstance(ep.started_at, datetime)
                    else ep.started_at
                )
            return ep

        with patch.object(k._storage, "get_current_epoch", side_effect=mock_current_epoch):
            months = k._get_epoch_staleness_months()
            assert months is not None
            assert months < 3  # ~1 month

    def test_string_ended_at_parsed(self, k):
        """String ended_at should be parsed correctly (line 139)."""
        from kernle.storage import Epoch

        # Create a closed epoch with ended_at
        epoch = Epoch(
            id="ep-closed-str",
            stack_id=k.stack_id,
            epoch_number=1,
            name="Closed with string timestamp",
            started_at=datetime.now(timezone.utc) - timedelta(days=90),
            ended_at=datetime.now(timezone.utc) - timedelta(days=15),
        )
        k._storage.save_epoch(epoch)

        # Mock: no current epoch, and last epoch has string ended_at
        def mock_no_current():
            return None

        original_get_epochs = k._storage.get_epochs

        def mock_get_epochs(limit=1):
            epochs = original_get_epochs(limit=limit)
            for ep in epochs:
                if ep.ended_at and isinstance(ep.ended_at, datetime):
                    ep.ended_at = ep.ended_at.isoformat()
                if ep.started_at and isinstance(ep.started_at, datetime):
                    ep.started_at = ep.started_at.isoformat()
            return epochs

        with patch.object(k._storage, "get_current_epoch", side_effect=mock_no_current):
            with patch.object(k._storage, "get_epochs", side_effect=mock_get_epochs):
                months = k._get_epoch_staleness_months()
                assert months is not None
                assert months < 2  # ~0.5 months


class TestEpochStalenessExceptionHandling:
    """Test exception handling in _get_epoch_staleness_months (lines 144-146)."""

    def test_exception_returns_none(self, k):
        """Exception in epoch staleness should return None gracefully."""
        with patch.object(k._storage, "get_current_epoch", side_effect=RuntimeError("DB error")):
            result = k._get_epoch_staleness_months()
            assert result is None


class TestUnsavedWorkHighAge:
    """Test unsaved work for 60+ minute checkpoint age (lines 215-220)."""

    def test_stale_checkpoint_score(self, k):
        """Checkpoint > 60 min old should produce high score with STALE label."""
        k.checkpoint("Test task")
        # Mock checkpoint age to 90 minutes
        with patch.object(k, "_get_checkpoint_age_minutes", return_value=90):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["unsaved_work"]
            assert dim["score"] >= 80
            assert "STALE" in dim["detail"]

    def test_very_old_checkpoint_caps_at_100(self, k):
        """Very old checkpoint should cap score at 100."""
        with patch.object(k, "_get_checkpoint_age_minutes", return_value=300):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["unsaved_work"]
            assert dim["score"] == 100


class TestConsolidationDebtHighCounts:
    """Test consolidation debt for 8-15 unreflected episodes (lines 239-244)."""

    def test_10_unreflected_significant_backlog(self, k):
        """10 unreflected episodes should show significant backlog."""
        # Create 10 episodes without lessons
        for i in range(10):
            k.episode(objective=f"Task {i}", outcome="completed", tags=["test"])

        report = k.get_anxiety_report()
        dim = report["dimensions"]["consolidation_debt"]
        assert dim["raw_value"] >= 10
        assert "significant backlog" in dim["detail"]
        assert dim["score"] >= 61  # Should be in the 61+ range

    def test_16_unreflected_urgent(self, k):
        """16+ unreflected episodes should show URGENT."""
        for i in range(16):
            k.episode(objective=f"Task {i}", outcome="completed", tags=["test"])

        report = k.get_anxiety_report()
        dim = report["dimensions"]["consolidation_debt"]
        assert dim["raw_value"] >= 16
        assert "URGENT" in dim["detail"]
        assert dim["score"] >= 93


class TestIdentityCoherenceDetail:
    """Test identity coherence detail text (lines 258, 260)."""

    def test_weak_identity_detail(self, k):
        """Identity confidence < 0.5 should show WEAK (line 258/262)."""
        # No values, no beliefs = confidence 0
        report = k.get_anxiety_report()
        dim = report["dimensions"]["identity_coherence"]
        # With no data, confidence is 0 which is < 0.5
        assert "WEAK" in dim["detail"] or dim["raw_value"] == 0

    def test_developing_identity_detail(self, k):
        """Identity confidence 0.5-0.8 should show 'developing' (line 260)."""
        # Mock get_identity_confidence to return 0.6
        with patch.object(k, "get_identity_confidence", return_value=0.6):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["identity_coherence"]
            assert "developing" in dim["detail"]


class TestMemoryUncertaintyHigh:
    """Test high uncertainty with > 5 low-confidence beliefs (lines 291-292)."""

    def test_many_low_confidence_beliefs(self, k):
        """6+ low confidence beliefs should show HIGH uncertainty."""
        for i in range(8):
            k.belief(f"Uncertain belief {i} about topic {i}", confidence=0.2 + i * 0.03)

        report = k.get_anxiety_report()
        dim = report["dimensions"]["memory_uncertainty"]
        assert dim["raw_value"] >= 6
        assert "HIGH uncertainty" in dim["detail"]
        assert dim["score"] >= 75


class TestRawAgingHighCounts:
    """Test raw aging for high counts (lines 309-322)."""

    def test_4_to_7_aging_entries(self, k):
        """4-7 aging entries should show days-old info."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        mock_entries = [
            SimpleNamespace(id=str(i), captured_at=old_time, timestamp=None) for i in range(5)
        ]
        with patch.object(k, "list_raw", return_value=mock_entries):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["raw_aging"]
            assert dim["score"] >= 60
            assert "oldest" in dim["detail"]

    def test_8_plus_aging_entries_stale(self, k):
        """8+ aging entries should show STALE with review needed."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=96)).isoformat()
        mock_entries = [
            SimpleNamespace(id=str(i), content="x" * 250, captured_at=old_time, timestamp=None)
            for i in range(10)
        ]
        with patch.object(k, "list_raw", return_value=mock_entries):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["raw_aging"]
            assert dim["score"] >= 92
            assert "STALE" in dim["detail"]
            assert "review needed" in dim["detail"]


class TestEpochStalenessElevated:
    """Test epoch staleness in elevated range 12-18 months (lines 351-352)."""

    def test_14_month_old_epoch(self, k):
        """14-month old epoch should be in elevated range (70-90)."""
        from kernle.storage import Epoch

        old_start = datetime.now(timezone.utc) - timedelta(days=425)  # ~14 months
        k._storage.save_epoch(
            Epoch(
                id="ep-elevated",
                stack_id=k.stack_id,
                epoch_number=1,
                name="Elevated epoch",
                started_at=old_start,
            )
        )

        report = k.get_anxiety_report()
        dim = report["dimensions"]["epoch_staleness"]
        assert 70 <= dim["score"] <= 90
        assert "significant time" in dim["detail"] or "deep reflection" in dim["detail"]


class TestAnxietyAlias:
    """Test anxiety() alias method (line 401)."""

    def test_anxiety_returns_same_as_get_anxiety_report(self, k):
        """anxiety() should return same report as get_anxiety_report()."""
        report1 = k.get_anxiety_report(context_tokens=50000)
        report2 = k.anxiety(context_tokens=50000)

        # Same structure
        assert report1["overall_level"] == report2["overall_level"]
        assert set(report1["dimensions"].keys()) == set(report2["dimensions"].keys())

    def test_anxiety_alias_passes_args(self, k):
        """anxiety() should pass through all arguments."""
        report = k.anxiety(context_tokens=180000, context_limit=200000, detailed=True)
        assert "recommendations" in report
        assert report["dimensions"]["context_pressure"]["raw_value"] == 90


class TestRecommendedActionsEdgeCases:
    """Test edge cases in get_recommended_actions (lines 424, 446, 467, 485, 523)."""

    def test_elevated_with_low_identity_recommends_synthesis(self, k):
        """Elevated level with identity_conf < 0.7 should recommend identity synthesis."""
        with patch.object(k, "get_identity_confidence", return_value=0.4):
            actions = k.get_recommended_actions(65)
            methods = [a.get("method") for a in actions]
            assert "synthesize_identity" in methods

    def test_elevated_with_low_confidence_beliefs(self, k):
        """Elevated level with low confidence beliefs should recommend review."""
        # Create low confidence beliefs
        for i in range(3):
            k.belief(f"Uncertain belief {i}", confidence=0.3)

        actions = k.get_recommended_actions(65)
        methods = [a.get("method") for a in actions]
        assert "get_uncertain_memories" in methods

    def test_elevated_with_no_unreflected_no_promote_action(self, k):
        """Elevated level with 0 unreflected should still add checkpoint."""
        actions = k.get_recommended_actions(65)
        methods = [a.get("method") for a in actions]
        assert "checkpoint" in methods

    def test_high_level_includes_identity_synthesis(self, k):
        """High level (71-85) should include identity synthesis."""
        actions = k.get_recommended_actions(80)
        methods = [a.get("method") for a in actions]
        assert "synthesize_identity" in methods

    def test_high_level_with_online_sync(self, k):
        """High level with online sync should recommend sync."""
        with patch.object(k, "get_sync_status", return_value={"online": True}):
            actions = k.get_recommended_actions(80)
            methods = [a.get("method") for a in actions]
            assert "sync" in methods

    def test_high_level_without_online_sync(self, k):
        """High level without online sync should not recommend sync."""
        with patch.object(k, "get_sync_status", return_value={"online": False}):
            actions = k.get_recommended_actions(80)
            methods = [a.get("method") for a in actions]
            assert "sync" not in methods

    def test_calm_with_unreflected_suggests_promote(self, k):
        """Calm level with some unreflected episodes should suggest promote."""
        k.episode(objective="Some work", outcome="completed", tags=["test"])

        actions = k.get_recommended_actions(20)
        methods = [a.get("method") for a in actions]
        assert "promote" in methods
        assert all(a["priority"] == "low" for a in actions)

    def test_calm_with_no_unreflected_empty_actions(self, k):
        """Calm level with 0 unreflected should return empty actions."""
        actions = k.get_recommended_actions(10)
        assert len(actions) == 0

    def test_aware_with_old_checkpoint_recommends_checkpoint(self, k):
        """Aware level with checkpoint_age > 15 should recommend checkpoint."""
        k.clear_checkpoint()
        actions = k.get_recommended_actions(40)
        methods = [a.get("method") for a in actions]
        assert "checkpoint" in methods

    def test_aware_with_many_unreflected_recommends_promote(self, k):
        """Aware level with > 3 unreflected should recommend promote."""
        for i in range(5):
            k.episode(objective=f"Work {i}", outcome="completed", tags=["test"])

        actions = k.get_recommended_actions(45)
        methods = [a.get("method") for a in actions]
        assert "promote" in methods

    def test_elevated_with_unreflected_recommends_consolidate(self, k):
        """Elevated level with unreflected should recommend consolidation."""
        k.episode(objective="Work", outcome="completed", tags=["test"])

        actions = k.get_recommended_actions(60)
        # Should have consolidation action
        descriptions = [a.get("description", "") for a in actions]
        assert any("Consolidate" in d or "unreflected" in d for d in descriptions)


class TestEmergencySaveErrorPaths:
    """Test emergency_save error handling (lines 593-623)."""

    def test_checkpoint_failure_recorded(self, k):
        """Checkpoint failure should be recorded in errors."""
        with patch.object(k, "checkpoint", side_effect=RuntimeError("DB locked")):
            result = k.emergency_save()
            assert result["checkpoint_saved"] is False
            assert any("Checkpoint failed" in e for e in result["errors"])

    def test_consolidation_failure_recorded(self, k):
        """Promotion failure should be recorded in errors."""
        with patch.object(k, "promote", side_effect=RuntimeError("Promotion failed")):
            result = k.emergency_save()
            assert any("Consolidation failed" in e for e in result["errors"])

    def test_identity_failure_recorded(self, k):
        """Identity synthesis failure should be recorded in errors."""
        with patch.object(k, "synthesize_identity", side_effect=RuntimeError("Synthesis error")):
            result = k.emergency_save()
            assert result["identity_synthesized"] is False
            assert any("Identity synthesis failed" in e for e in result["errors"])

    def test_sync_failure_recorded(self, k):
        """Sync failure should be recorded in errors."""
        with patch.object(k, "get_sync_status", return_value={"online": True}):
            with patch.object(k, "sync", side_effect=RuntimeError("Network error")):
                result = k.emergency_save()
                assert any("Sync failed" in e for e in result["errors"])

    def test_episode_recording_failure(self, k):
        """Episode recording failure should be recorded in errors."""
        # We need to allow everything else to succeed, but fail episode()
        call_count = [0]
        original_episode = k.episode

        def failing_episode(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 0 and kwargs.get("tags") and "emergency" in kwargs["tags"]:
                raise RuntimeError("Episode write failed")
            return original_episode(*args, **kwargs)

        with patch.object(k, "episode", side_effect=failing_episode):
            result = k.emergency_save()
            assert any("Episode recording failed" in e for e in result["errors"])

    def test_sync_not_attempted_when_offline(self, k):
        """Sync should not be attempted when offline."""
        result = k.emergency_save()
        assert result["sync_attempted"] is False

    def test_success_flag_true_when_no_errors(self, k):
        """Success flag should be True when no errors occur."""
        result = k.emergency_save()
        assert result["success"] is True


class TestRawAgingReportIntegration:
    """Test raw aging paths in get_anxiety_report (lines 310-311, 313-314)."""

    def test_fresh_unprocessed_entries_report(self, k):
        """Unprocessed but fresh entries should show 'all fresh' detail."""
        recent = datetime.now(timezone.utc) - timedelta(hours=2)
        entries = [
            SimpleNamespace(content="x" * 100, captured_at=recent, timestamp=None) for _ in range(5)
        ]
        with patch.object(k, "_get_aging_raw_entries", return_value=(5, 0, 2, entries)):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["raw_aging"]
            assert "all fresh" in dim["detail"]

    def test_few_aging_entries_report(self, k):
        """1-3 aging entries should show >24h old detail."""
        old = datetime.now(timezone.utc) - timedelta(hours=30)
        recent = datetime.now(timezone.utc) - timedelta(hours=2)
        entries = [
            SimpleNamespace(content="x" * 100, captured_at=old, timestamp=None),
            SimpleNamespace(content="x" * 100, captured_at=old, timestamp=None),
        ] + [
            SimpleNamespace(content="x" * 100, captured_at=recent, timestamp=None) for _ in range(6)
        ]
        with patch.object(k, "_get_aging_raw_entries", return_value=(8, 2, 30, entries)):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["raw_aging"]
            assert ">24h old" in dim["detail"]


class TestIdentityCoherenceStrong:
    """Test strong identity coherence path (line 258)."""

    def test_strong_identity_shows_strong_detail(self, k):
        """Identity confidence >= 0.8 should show 'strong' detail."""
        with patch.object(k, "get_identity_confidence", return_value=0.85):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["identity_coherence"]
            assert "strong" in dim["detail"]


class TestEmergencySaveSyncSuccess:
    """Test successful sync path in emergency_save (lines 618-619)."""

    def test_successful_sync_records_result(self, k):
        """Successful sync should record sync_success and sync_result."""
        with patch.object(k, "get_sync_status", return_value={"online": True}):
            with patch.object(k, "sync", return_value={"success": True, "pushed": 5}):
                result = k.emergency_save()
                assert result["sync_attempted"] is True
                assert result["sync_success"] is True
                assert result["sync_result"]["pushed"] == 5


class TestEpochStalenessException:
    """Test exception and edge cases in _get_epoch_staleness_months."""

    def test_get_epochs_raises_returns_none(self, k):
        """Exception during get_epochs call should return None (lines 145-146)."""
        with patch.object(k._storage, "get_current_epoch", return_value=None):
            with patch.object(k._storage, "get_epochs", side_effect=Exception("table missing")):
                result = k._get_epoch_staleness_months()
                assert result is None

    def test_closed_epoch_without_ended_at_returns_none(self, k):
        """Closed epoch with no ended_at should return None (line 144)."""
        from kernle.storage import Epoch

        # Create an epoch without ended_at
        mock_epoch = Epoch(
            id="ep-no-end",
            stack_id=k.stack_id,
            epoch_number=1,
            name="No ended_at",
            started_at=datetime.now(timezone.utc) - timedelta(days=30),
            ended_at=None,
        )
        with patch.object(k._storage, "get_current_epoch", return_value=None):
            with patch.object(k._storage, "get_epochs", return_value=[mock_epoch]):
                result = k._get_epoch_staleness_months()
                assert result is None


class TestContextPressureRanges:
    """Test context pressure non-linear scaling."""

    def test_below_50_pct(self, k):
        """Below 50% context should use 0.6 multiplier."""
        report = k.get_anxiety_report(context_tokens=60000, context_limit=200000)
        dim = report["dimensions"]["context_pressure"]
        # 30% * 0.6 = 18
        assert dim["score"] == 18

    def test_50_to_70_pct(self, k):
        """50-70% context should use steeper scaling."""
        report = k.get_anxiety_report(context_tokens=120000, context_limit=200000)
        dim = report["dimensions"]["context_pressure"]
        # 60% -> 30 + (60-50) * 1.5 = 45
        assert dim["score"] == 45

    def test_70_to_85_pct(self, k):
        """70-85% context should use even steeper scaling."""
        report = k.get_anxiety_report(context_tokens=150000, context_limit=200000)
        dim = report["dimensions"]["context_pressure"]
        # 75% -> 60 + (75-70) * 2 = 70
        assert dim["score"] == 70

    def test_above_85_pct(self, k):
        """Above 85% context should use high scaling."""
        report = k.get_anxiety_report(context_tokens=190000, context_limit=200000)
        dim = report["dimensions"]["context_pressure"]
        # 95% -> 90 + (95-85) * 0.67 = 96
        assert dim["score"] == 96


class TestUnsavedWorkRanges:
    """Test unsaved work score ranges."""

    def test_15_to_60_min_range(self, k):
        """15-60 min checkpoint age should use middle scaling."""
        with patch.object(k, "_get_checkpoint_age_minutes", return_value=30):
            report = k.get_anxiety_report()
            dim = report["dimensions"]["unsaved_work"]
            # 30 + (30-15) * 1.1 = 46.5 -> 46
            assert dim["score"] == 46
