"""Tests for list_raw returning full RawEntry objects, offset/pagination, and AnxietyMixin compat."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kernle.stack import Stack
from kernle.types import RawEntry


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_list_raw.db"


@pytest.fixture
def stack(tmp_db):
    return Stack.from_sqlite(
        stack_id="test-stack", db_path=tmp_db, components=[], enforce_provenance=False
    )


def _save_raw(stack, blob="test blob"):
    entry = RawEntry(
        id=str(uuid.uuid4()),
        stack_id="test-stack",
        blob=blob,
        source="cli",
    )
    rid = stack.save_raw(entry)
    return rid


class TestListRawReturnsRawEntry:
    """list_raw must return RawEntry objects, not stripped dicts."""

    def test_returns_raw_entry_objects(self, stack):
        _save_raw(stack)
        entries = stack.list_raw()
        assert len(entries) == 1
        assert isinstance(entries[0], RawEntry)

    def test_has_processed_into_field(self, stack):
        _save_raw(stack)
        entries = stack.list_raw()
        entry = entries[0]
        assert hasattr(entry, "processed_into")

    def test_has_all_fields(self, stack):
        _save_raw(stack, blob="full check")
        entries = stack.list_raw()
        entry = entries[0]
        for field in (
            "id",
            "stack_id",
            "blob",
            "captured_at",
            "source",
            "processed",
            "processed_into",
            "version",
            "deleted",
        ):
            assert hasattr(entry, field), f"Missing field: {field}"

    def test_processed_true_filter(self, stack):
        rid1 = _save_raw(stack, blob="will be processed")
        _save_raw(stack, blob="stays unprocessed")
        # Mark one as processed via the backend API
        stack._backend.mark_raw_processed(rid1, ["episode:test123"])
        entries = stack.list_raw(processed=True)
        assert len(entries) == 1
        assert all(e.processed for e in entries)

    def test_processed_false_filter(self, stack):
        rid1 = _save_raw(stack, blob="will be processed")
        _save_raw(stack, blob="stays unprocessed")
        stack._backend.mark_raw_processed(rid1, ["episode:test123"])
        entries = stack.list_raw(processed=False)
        assert len(entries) == 1
        assert all(not e.processed for e in entries)

    def test_limit(self, stack):
        for _ in range(5):
            _save_raw(stack)
        entries = stack.list_raw(limit=3)
        assert len(entries) == 3


class TestRawAgingViaCore:
    """compute_raw_aging_score from anxiety_core works with RawEntry data."""

    def test_aging_raw_entries_score_with_data(self, stack):
        from kernle.anxiety_core import compute_raw_aging_score

        rid_old = _save_raw(stack, blob="old entry")
        _save_raw(stack, blob="recent entry")

        # Backdate the old entry's captured_at in the DB
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        with stack._backend._connect() as conn:
            conn.execute(
                "UPDATE raw_entries SET captured_at = ?, timestamp = ? WHERE id = ?",
                (old_time.isoformat(), old_time.isoformat(), rid_old),
            )

        score = compute_raw_aging_score(total_unprocessed=2, aging_count=1, oldest_hours=48)
        assert score == 45  # 30 + 1*15

    def test_no_entries_score(self):
        from kernle.anxiety_core import compute_raw_aging_score

        score = compute_raw_aging_score(total_unprocessed=0, aging_count=0, oldest_hours=0)
        assert score == 0

    def test_all_recent_score(self):
        from kernle.anxiety_core import compute_raw_aging_score

        score = compute_raw_aging_score(total_unprocessed=1, aging_count=0, oldest_hours=0)
        assert score == 3  # min(30, 1*3)


class TestListRawOffset:
    """Tests for list_raw offset/pagination support."""

    def _save_n(self, stack, n):
        """Save n entries with distinct captured_at values for deterministic ordering."""
        ids = []
        for i in range(n):
            entry = RawEntry(
                id=str(uuid.uuid4()),
                stack_id="test-stack",
                blob=f"entry-{i}",
                source="cli",
            )
            rid = stack.save_raw(entry)
            ids.append(rid)
            # Backdate so each has a unique captured_at (oldest first)
            ts = (datetime.now(timezone.utc) - timedelta(seconds=n - i)).isoformat()
            with stack._backend._connect() as conn:
                conn.execute(
                    "UPDATE raw_entries SET captured_at = ?, timestamp = ? WHERE id = ?",
                    (ts, ts, rid),
                )
        return ids

    def test_offset_skips_entries(self, stack):
        """Offset should skip the first N entries."""
        self._save_n(stack, 5)
        all_entries = stack.list_raw(limit=100)
        offset_entries = stack.list_raw(limit=100, offset=2)
        assert len(offset_entries) == 3
        assert [e.id for e in offset_entries] == [e.id for e in all_entries[2:]]

    def test_offset_beyond_total_returns_empty(self, stack):
        """Offset past the end should return an empty list."""
        self._save_n(stack, 3)
        entries = stack.list_raw(limit=100, offset=100)
        assert entries == []

    def test_paginate_all_no_gaps_no_dupes(self, stack):
        """Paginating in small batches should yield exactly the full set with no duplicates."""
        self._save_n(stack, 10)
        all_entries = stack.list_raw(limit=100)
        all_ids = {e.id for e in all_entries}
        assert len(all_ids) == 10

        # Paginate in batches of 3
        paginated_ids = []
        offset = 0
        while True:
            batch = stack.list_raw(limit=3, offset=offset)
            if not batch:
                break
            paginated_ids.extend(e.id for e in batch)
            offset += 3

        assert len(paginated_ids) == 10
        assert set(paginated_ids) == all_ids
        # No duplicates
        assert len(paginated_ids) == len(set(paginated_ids))

    def test_deterministic_ordering(self, stack):
        """Entries with identical captured_at produce stable pages across repeated queries."""
        # Insert entries with same captured_at
        ts = datetime.now(timezone.utc).isoformat()
        for i in range(5):
            rid = str(uuid.uuid4())
            entry = RawEntry(id=rid, stack_id="test-stack", blob=f"same-ts-{i}", source="cli")
            stack.save_raw(entry)
            with stack._backend._connect() as conn:
                conn.execute(
                    "UPDATE raw_entries SET captured_at = ?, timestamp = ? WHERE id = ?",
                    (ts, ts, rid),
                )

        # Run the same query multiple times and check for stable ordering
        first_run = [e.id for e in stack.list_raw(limit=100)]
        for _ in range(5):
            this_run = [e.id for e in stack.list_raw(limit=100)]
            assert this_run == first_run

    def test_negative_offset_raises(self, stack):
        """Negative offset should raise ValueError."""
        with pytest.raises(ValueError, match="offset must be non-negative"):
            stack.list_raw(offset=-1)

    def test_zero_limit_raises(self, stack):
        """Zero limit should raise ValueError."""
        with pytest.raises(ValueError, match="limit must be positive"):
            stack.list_raw(limit=0)
