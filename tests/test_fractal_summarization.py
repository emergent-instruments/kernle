"""Tests for fractal summarization (summaries memory type).

Covers:
- Storage: save, get, list
- Core: create, list, get
- Priority scoring per scope
- Supersession logic in load()
- is_protected prevents forgetting
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle import Kernle
from kernle.core import MEMORY_TYPE_PRIORITIES, compute_priority_score
from kernle.storage import Summary


@pytest.fixture
def stack_id():
    return f"test-agent-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def k(stack_id):
    return Kernle(stack_id=stack_id, strict=False)


# === Dataclass Tests ===


class TestSummaryDataclass:
    def test_summary_defaults(self):
        s = Summary(
            id="s1",
            stack_id="agent1",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            content="Q1 summary content",
        )
        assert s.is_protected is True
        assert s.version == 1
        assert s.deleted is False
        assert s.key_themes is None
        assert s.supersedes is None
        assert s.epoch_id is None

    def test_summary_with_all_fields(self):
        now = datetime.now(timezone.utc)
        s = Summary(
            id="s2",
            stack_id="agent1",
            scope="year",
            period_start="2025-01-01",
            period_end="2025-12-31",
            content="Annual summary",
            epoch_id="epoch-1",
            key_themes=["growth", "learning"],
            supersedes=["s1-q1", "s1-q2", "s1-q3", "s1-q4"],
            is_protected=True,
            created_at=now,
            updated_at=now,
        )
        assert s.scope == "year"
        assert len(s.supersedes) == 4
        assert s.key_themes == ["growth", "learning"]


# === Storage Tests ===


class TestStorageSummary:
    def test_save_and_get_summary(self, k):
        summary_id = k.summary_save(
            content="Test summary content for the quarter",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            key_themes=["testing", "development"],
        )

        assert summary_id is not None
        assert len(summary_id) > 0

        # Retrieve and verify
        summary = k.summary_get(summary_id)
        assert summary is not None
        assert summary.id == summary_id
        assert summary.scope == "quarter"
        assert summary.period_start == "2025-01-01"
        assert summary.period_end == "2025-03-31"
        assert summary.content == "Test summary content for the quarter"
        assert summary.key_themes == ["testing", "development"]
        assert summary.is_protected is True

    def test_get_nonexistent_summary(self, k):
        result = k.summary_get("nonexistent-id")
        assert result is None

    def test_list_summaries_all(self, k):
        # Create summaries at different scopes
        k.summary_save(
            content="Monthly summary",
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
        )
        k.summary_save(
            content="Quarterly summary",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
        )
        k.summary_save(
            content="Yearly summary",
            scope="year",
            period_start="2025-01-01",
            period_end="2025-12-31",
        )

        summaries = k.summary_list()
        assert len(summaries) == 3

    def test_list_summaries_by_scope(self, k):
        k.summary_save(
            content="Month 1",
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
        )
        k.summary_save(
            content="Month 2",
            scope="month",
            period_start="2025-02-01",
            period_end="2025-02-28",
        )
        k.summary_save(
            content="Quarter 1",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
        )

        month_summaries = k.summary_list(scope="month")
        assert len(month_summaries) == 2

        quarter_summaries = k.summary_list(scope="quarter")
        assert len(quarter_summaries) == 1

    def test_save_summary_with_epoch_id(self, k):
        epoch_id = k.epoch_create(name="test-epoch")
        summary_id = k.summary_save(
            content="Epoch summary",
            scope="epoch",
            period_start="2025-01-01",
            period_end="2025-06-30",
            epoch_id=epoch_id,
        )

        summary = k.summary_get(summary_id)
        assert summary.epoch_id == epoch_id
        assert summary.scope == "epoch"


# === Core API Tests ===


class TestCoreSummary:
    def test_summary_save_validates_scope(self, k):
        with pytest.raises(ValueError, match="scope must be one of"):
            k.summary_save(
                content="Invalid",
                scope="invalid_scope",
                period_start="2025-01-01",
                period_end="2025-01-31",
            )

    def test_summary_save_validates_content(self, k):
        # Content too long (>10000)
        with pytest.raises(ValueError):
            k.summary_save(
                content="x" * 10001,
                scope="month",
                period_start="2025-01-01",
                period_end="2025-01-31",
            )

    def test_summary_list_validates_scope(self, k):
        with pytest.raises(ValueError, match="scope must be one of"):
            k.summary_list(scope="invalid")

    def test_summary_list_no_scope_filter(self, k):
        # Should not raise with None scope
        result = k.summary_list(scope=None)
        assert isinstance(result, list)

    def test_valid_scopes(self, k):
        for scope in ("month", "quarter", "year", "decade", "epoch"):
            sid = k.summary_save(
                content=f"Summary for {scope}",
                scope=scope,
                period_start="2025-01-01",
                period_end="2025-12-31",
            )
            assert sid is not None

    def test_summary_with_supersedes(self, k):
        # Create lower-scope summaries
        m1 = k.summary_save(
            content="January summary",
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
        )
        m2 = k.summary_save(
            content="February summary",
            scope="month",
            period_start="2025-02-01",
            period_end="2025-02-28",
        )
        m3 = k.summary_save(
            content="March summary",
            scope="month",
            period_start="2025-03-01",
            period_end="2025-03-31",
        )

        # Create higher-scope summary that supersedes them
        q1 = k.summary_save(
            content="Q1 summary covering all three months",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            supersedes=[m1, m2, m3],
        )

        summary = k.summary_get(q1)
        assert summary.supersedes == [m1, m2, m3]


# === Priority Scoring Tests ===


class TestPriorityScoring:
    def test_scope_priorities_in_type_map(self):
        assert "summary_decade" in MEMORY_TYPE_PRIORITIES
        assert "summary_epoch" in MEMORY_TYPE_PRIORITIES
        assert "summary_year" in MEMORY_TYPE_PRIORITIES
        assert "summary_quarter" in MEMORY_TYPE_PRIORITIES
        assert "summary_month" in MEMORY_TYPE_PRIORITIES

    def test_scope_priority_ordering(self):
        """Higher scopes should have higher priority."""
        assert MEMORY_TYPE_PRIORITIES["summary_decade"] > MEMORY_TYPE_PRIORITIES["summary_epoch"]
        assert MEMORY_TYPE_PRIORITIES["summary_epoch"] > MEMORY_TYPE_PRIORITIES["summary_year"]
        assert MEMORY_TYPE_PRIORITIES["summary_year"] > MEMORY_TYPE_PRIORITIES["summary_quarter"]
        assert MEMORY_TYPE_PRIORITIES["summary_quarter"] > MEMORY_TYPE_PRIORITIES["summary_month"]

    def test_decade_summary_higher_than_values(self):
        assert MEMORY_TYPE_PRIORITIES["summary_decade"] > MEMORY_TYPE_PRIORITIES["value"]

    def test_compute_priority_for_summary(self):
        s = Summary(
            id="s1",
            stack_id="a1",
            scope="year",
            period_start="2025-01-01",
            period_end="2025-12-31",
            content="Year summary",
        )
        score = compute_priority_score("summary_year", s)
        assert 0.0 < score <= 1.0

    def test_different_scopes_different_scores(self):
        s = Summary(
            id="s1",
            stack_id="a1",
            scope="decade",
            period_start="2020-01-01",
            period_end="2029-12-31",
            content="Decade summary",
        )
        decade_score = compute_priority_score("summary_decade", s)
        month_score = compute_priority_score("summary_month", s)
        assert decade_score > month_score


# === Supersession Logic Tests ===


class TestSupersessionLogic:
    def test_superseded_summaries_excluded_from_load(self, k):
        """When a higher-scope summary supersedes lower ones, load() should skip the lower ones."""
        # Create monthly summaries
        m1 = k.summary_save(
            content="January: Focused on onboarding and learning the codebase",
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
        )
        m2 = k.summary_save(
            content="February: Started contributing to core features",
            scope="month",
            period_start="2025-02-01",
            period_end="2025-02-28",
        )
        m3 = k.summary_save(
            content="March: Led first project independently",
            scope="month",
            period_start="2025-03-01",
            period_end="2025-03-31",
        )

        # Create quarter summary superseding the months
        q1 = k.summary_save(
            content="Q1: Transitioned from onboarding to independent contributor",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            supersedes=[m1, m2, m3],
        )

        # Load should include the quarter summary but not the superseded months
        result = k.load(budget=50000)
        summaries = result.get("summaries", [])

        summary_ids = {s["id"] for s in summaries}
        assert q1 in summary_ids, "Quarter summary should be included"
        assert m1 not in summary_ids, "Superseded month 1 should be excluded"
        assert m2 not in summary_ids, "Superseded month 2 should be excluded"
        assert m3 not in summary_ids, "Superseded month 3 should be excluded"

    def test_non_superseded_summaries_included(self, k):
        """Summaries not superseded by higher scope should appear in load()."""
        m1 = k.summary_save(
            content="April summary: new quarter",
            scope="month",
            period_start="2025-04-01",
            period_end="2025-04-30",
        )

        result = k.load(budget=50000)
        summaries = result.get("summaries", [])
        summary_ids = {s["id"] for s in summaries}
        assert m1 in summary_ids


# === Protection Tests ===


class TestSummaryProtection:
    def test_summaries_are_protected_by_default(self, k):
        sid = k.summary_save(
            content="Protected summary",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
        )

        summary = k.summary_get(sid)
        assert summary.is_protected is True

    def test_protected_summary_not_in_forgetting_candidates(self, k):
        """Protected summaries should not appear as forgetting candidates."""
        k.summary_save(
            content="This should never be forgotten",
            scope="year",
            period_start="2025-01-01",
            period_end="2025-12-31",
        )

        # Get forgetting candidates - summaries shouldn't be there
        # because they are protected by default and stored in a separate table
        candidates = k._storage.get_forgetting_candidates()
        for candidate in candidates:
            assert not isinstance(candidate.record, Summary)


# === Schema Version Test ===


class TestSchemaVersion:
    def test_schema_version_updated(self):
        from kernle.storage.sqlite import SCHEMA_VERSION

        assert SCHEMA_VERSION == 27
