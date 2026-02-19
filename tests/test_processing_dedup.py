"""Tests for transition deduplication (Issue #403).

Tests cover:
- compute_provenance_hash: stable hashing, order independence, empty inputs
- compute_content_hash: normalization, whitespace handling, case insensitivity
- _extract_content_text: all transition types
- _extract_derived_from: all transition types
- _build_existing_index: content index building (content-only dedup)
- _check_duplicate: content match, no match
- _write_memories with dedup: skipping duplicates, counting deduplicated
- Full pipeline: re-running same cycle produces 0 new items
- New sources still create new items after dedup
- Edge cases: empty provenance, large provenance sets, partial overlaps
- Random ingestion order: dedup is order-independent
"""

from __future__ import annotations

import json
import random
import uuid
from unittest.mock import MagicMock

import pytest

from kernle.processing import (
    MemoryProcessor,
    ProcessingResult,
    PromotionGateConfig,
    _extract_content_text,
    _extract_derived_from,
    compute_content_hash,
    compute_provenance_hash,
)
from kernle.stack.sqlite_stack import Stack
from kernle.types import RawEntry

# Relaxed promotion gates for tests that don't test gating behavior
_NO_GATES = PromotionGateConfig(
    belief_min_evidence=0,
    belief_min_confidence=0.0,
    value_min_evidence=0,
    value_requires_protection=False,
)

STACK_ID = "test-stack"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def stack(tmp_path):
    return Stack.from_sqlite(
        STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=False
    )


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


def _make_processor(mock_stack, response="[]"):
    """Create a MemoryProcessor with a mock stack and inference."""
    inference = MockInference(response)
    return (
        MemoryProcessor(
            stack=mock_stack, inference=inference, core_id="test", promotion_gates=_NO_GATES
        ),
        inference,
    )


# =============================================================================
# compute_provenance_hash
# =============================================================================


class TestComputeProvenanceHash:
    def test_empty_list_returns_empty(self):
        assert compute_provenance_hash([]) == ""

    def test_single_ref(self):
        h = compute_provenance_hash(["raw:abc"])
        assert len(h) == 64  # SHA-256 hex digest
        assert h == compute_provenance_hash(["raw:abc"])  # deterministic

    def test_order_independent(self):
        """Hash should be the same regardless of input order."""
        h1 = compute_provenance_hash(["raw:abc", "raw:def"])
        h2 = compute_provenance_hash(["raw:def", "raw:abc"])
        assert h1 == h2

    def test_different_refs_different_hash(self):
        h1 = compute_provenance_hash(["raw:abc"])
        h2 = compute_provenance_hash(["raw:xyz"])
        assert h1 != h2

    def test_subset_different_hash(self):
        """A subset of refs should produce a different hash."""
        h1 = compute_provenance_hash(["raw:abc", "raw:def"])
        h2 = compute_provenance_hash(["raw:abc"])
        assert h1 != h2

    def test_large_provenance_set(self):
        """Performance: hashing 1000 refs should complete quickly."""
        refs = [f"raw:{uuid.uuid4()}" for _ in range(1000)]
        h = compute_provenance_hash(refs)
        assert len(h) == 64

    def test_duplicate_refs_treated_as_same(self):
        """Duplicate refs in the list should still produce a consistent hash."""
        h1 = compute_provenance_hash(["raw:abc", "raw:abc"])
        h2 = compute_provenance_hash(["raw:abc", "raw:abc"])
        assert h1 == h2

    def test_mixed_ref_types(self):
        h = compute_provenance_hash(["episode:e1", "raw:r1", "belief:b1"])
        assert len(h) == 64
        # Order independence
        h2 = compute_provenance_hash(["belief:b1", "episode:e1", "raw:r1"])
        assert h == h2


# =============================================================================
# compute_content_hash
# =============================================================================


class TestComputeContentHash:
    def test_empty_returns_empty(self):
        assert compute_content_hash("") == ""

    def test_deterministic(self):
        h1 = compute_content_hash("Hello world")
        h2 = compute_content_hash("Hello world")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = compute_content_hash("Hello World")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_whitespace_normalized(self):
        h1 = compute_content_hash("Hello   world")
        h2 = compute_content_hash("Hello world")
        assert h1 == h2

    def test_leading_trailing_whitespace(self):
        h1 = compute_content_hash("  Hello world  ")
        h2 = compute_content_hash("Hello world")
        assert h1 == h2

    def test_newlines_normalized(self):
        h1 = compute_content_hash("Hello\nworld")
        h2 = compute_content_hash("Hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_content_hash("Hello world")
        h2 = compute_content_hash("Goodbye world")
        assert h1 != h2


# =============================================================================
# _extract_content_text
# =============================================================================


class TestExtractContentText:
    def test_raw_to_episode(self):
        item = {"objective": "did something", "outcome": "it worked"}
        text = _extract_content_text("raw_to_episode", item)
        assert "did something" in text
        assert "it worked" in text

    def test_raw_to_note(self):
        item = {"content": "a factual note"}
        text = _extract_content_text("raw_to_note", item)
        assert text == "a factual note"

    def test_episode_to_belief(self):
        item = {"statement": "testing is important"}
        text = _extract_content_text("episode_to_belief", item)
        assert text == "testing is important"

    def test_episode_to_goal(self):
        item = {"title": "ship v2", "description": "release it"}
        text = _extract_content_text("episode_to_goal", item)
        assert "ship v2" in text
        assert "release it" in text

    def test_episode_to_relationship(self):
        item = {"entity_name": "Alice"}
        text = _extract_content_text("episode_to_relationship", item)
        assert text == "Alice"

    def test_belief_to_value(self):
        item = {"name": "honesty", "statement": "be truthful"}
        text = _extract_content_text("belief_to_value", item)
        assert "honesty" in text
        assert "be truthful" in text

    def test_episode_to_drive(self):
        item = {"drive_type": "curiosity"}
        text = _extract_content_text("episode_to_drive", item)
        assert text == "curiosity"

    def test_unknown_transition(self):
        text = _extract_content_text("unknown", {"stuff": "val"})
        assert text == ""

    def test_missing_fields(self):
        text = _extract_content_text("raw_to_episode", {})
        assert text == ""


# =============================================================================
# _extract_derived_from
# =============================================================================


class TestExtractDerivedFrom:
    def test_raw_to_episode(self):
        item = {"source_raw_ids": ["r1", "r2"]}
        result = _extract_derived_from("raw_to_episode", item)
        assert result == ["raw:r1", "raw:r2"]

    def test_raw_to_note(self):
        item = {"source_raw_ids": ["r1"]}
        result = _extract_derived_from("raw_to_note", item)
        assert result == ["raw:r1"]

    def test_episode_to_belief(self):
        item = {"source_episode_ids": ["ep1"]}
        result = _extract_derived_from("episode_to_belief", item)
        assert result == ["episode:ep1"]

    def test_episode_to_goal(self):
        item = {"source_episode_ids": ["ep1", "ep2"]}
        result = _extract_derived_from("episode_to_goal", item)
        assert result == ["episode:ep1", "episode:ep2"]

    def test_episode_to_relationship(self):
        item = {"source_episode_ids": ["ep1"]}
        result = _extract_derived_from("episode_to_relationship", item)
        assert result == ["episode:ep1"]

    def test_episode_to_drive(self):
        item = {"source_episode_ids": ["ep1"]}
        result = _extract_derived_from("episode_to_drive", item)
        assert result == ["episode:ep1"]

    def test_belief_to_value(self):
        item = {"source_belief_ids": ["b1", "b2"]}
        result = _extract_derived_from("belief_to_value", item)
        assert result == ["belief:b1", "belief:b2"]

    def test_missing_ids(self):
        result = _extract_derived_from("raw_to_episode", {})
        assert result == []

    def test_unknown_transition(self):
        result = _extract_derived_from("unknown", {"source_raw_ids": ["r1"]})
        assert result == []


# =============================================================================
# _build_existing_index
# =============================================================================


class TestBuildExistingIndex:
    def test_empty_context(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        index = processor._build_existing_index("raw_to_episode", [])
        assert index["content"] == {}
        assert "provenance" not in index

    def test_indexes_episodes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        ep = MagicMock()
        ep.id = "ep-1"
        ep.objective = "did something"
        ep.outcome = "it worked"
        ep.derived_from = ["raw:r1", "raw:r2"]
        index = processor._build_existing_index("raw_to_episode", [ep])
        # Content hash should be populated
        chash = compute_content_hash("did something it worked")
        assert chash in index["content"]
        assert "provenance" not in index

    def test_indexes_beliefs(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        belief = MagicMock(spec=["id", "statement", "belief_type", "derived_from"])
        belief.id = "b-1"
        belief.statement = "testing matters"
        belief.belief_type = "evaluative"
        belief.derived_from = ["episode:ep-1"]
        # Remove episode attributes so hasattr picks up belief branch
        del belief.objective
        index = processor._build_existing_index("episode_to_belief", [belief])
        chash = compute_content_hash("testing matters")
        assert chash in index["content"]
        assert "provenance" not in index

    def test_no_derived_from_still_indexes_content(self):
        """Records without derived_from are still indexed by content."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        ep = MagicMock()
        ep.id = "ep-1"
        ep.objective = "obj"
        ep.outcome = "out"
        ep.derived_from = None
        index = processor._build_existing_index("raw_to_episode", [ep])
        # Content should still be indexed
        assert len(index["content"]) == 1
        chash = compute_content_hash("obj out")
        assert chash in index["content"]

    def test_indexes_notes(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        note = MagicMock(spec=["id", "content", "note_type", "derived_from"])
        note.id = "n-1"
        note.content = "a factual note"
        note.note_type = "observation"
        note.derived_from = ["raw:r1"]
        del note.objective
        del note.statement
        index = processor._build_existing_index("raw_to_note", [note])
        chash = compute_content_hash("a factual note")
        assert chash in index["content"]

    def test_indexes_goals(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        goal = MagicMock(spec=["id", "title", "description", "derived_from"])
        goal.id = "g-1"
        goal.title = "ship v2"
        goal.description = "release it"
        goal.derived_from = ["episode:ep-1"]
        del goal.objective
        del goal.statement
        del goal.belief_type
        index = processor._build_existing_index("episode_to_goal", [goal])
        chash = compute_content_hash("ship v2 release it")
        assert chash in index["content"]

    def test_indexes_drives(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        drive = MagicMock(spec=["id", "drive_type", "derived_from"])
        drive.id = "d-1"
        drive.drive_type = "curiosity"
        drive.derived_from = ["episode:ep-1"]
        del drive.objective
        del drive.statement
        del drive.title
        del drive.entity_name
        del drive.content
        del drive.name
        index = processor._build_existing_index("episode_to_drive", [drive])
        chash = compute_content_hash("curiosity")
        assert chash in index["content"]

    def test_indexes_relationships(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        rel = MagicMock(spec=["id", "entity_name", "derived_from"])
        rel.id = "rel-1"
        rel.entity_name = "Alice"
        rel.derived_from = ["episode:ep-1"]
        del rel.objective
        del rel.statement
        del rel.title
        del rel.drive_type
        del rel.content
        del rel.name
        index = processor._build_existing_index("episode_to_relationship", [rel])
        chash = compute_content_hash("Alice")
        assert chash in index["content"]

    def test_indexes_values(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        value = MagicMock(spec=["id", "name", "statement", "derived_from"])
        value.id = "v-1"
        value.name = "honesty"
        value.statement = "always be truthful"
        value.derived_from = ["belief:b-1"]
        del value.objective
        del value.belief_type
        del value.title
        del value.drive_type
        del value.entity_name
        del value.content
        index = processor._build_existing_index("belief_to_value", [value])
        chash = compute_content_hash("honesty always be truthful")
        assert chash in index["content"]


# =============================================================================
# _check_duplicate
# =============================================================================


class TestCheckDuplicate:
    def test_same_provenance_different_content_not_duplicate(self):
        """Same provenance but different content is NOT a duplicate (content-only dedup)."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        dedup_index = {"content": {}}
        item = {"objective": "new obj", "outcome": "new out", "source_raw_ids": ["r1", "r2"]}
        result = processor._check_duplicate("raw_to_episode", item, dedup_index)
        assert result is None

    def test_content_match(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        existing = MagicMock()
        existing.id = "existing-ep"
        chash = compute_content_hash("did something it worked")
        dedup_index = {"content": {chash: existing}}
        item = {"objective": "did something", "outcome": "it worked", "source_raw_ids": []}
        result = processor._check_duplicate("raw_to_episode", item, dedup_index)
        assert result == "existing-ep"

    def test_no_match(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        dedup_index = {"content": {}}
        item = {"objective": "new thing", "outcome": "happened", "source_raw_ids": ["r99"]}
        result = processor._check_duplicate("raw_to_episode", item, dedup_index)
        assert result is None

    def test_content_match_is_only_dedup_mechanism(self):
        """Only content hash is checked for dedup; provenance is ignored."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        content_existing = MagicMock()
        content_existing.id = "content-match"
        chash = compute_content_hash("did something it worked")
        dedup_index = {"content": {chash: content_existing}}
        item = {"objective": "did something", "outcome": "it worked", "source_raw_ids": ["r1"]}
        result = processor._check_duplicate("raw_to_episode", item, dedup_index)
        assert result == "content-match"

    def test_content_dedup_without_source_ids(self):
        """Items without source IDs are still deduped by content."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        existing = MagicMock()
        existing.id = "content-match"
        chash = compute_content_hash("test statement")
        dedup_index = {"content": {chash: existing}}
        item = {"statement": "test statement", "source_episode_ids": []}
        result = processor._check_duplicate("episode_to_belief", item, dedup_index)
        assert result == "content-match"

    def test_all_transition_types_content_dedup(self):
        """Verify content-based dedup works for each transition type."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        cases = [
            ("raw_to_episode", {"objective": "obj", "outcome": "out", "source_raw_ids": ["r1"]}),
            ("raw_to_note", {"content": "a note", "source_raw_ids": ["r1"]}),
            ("episode_to_belief", {"statement": "belief", "source_episode_ids": ["ep1"]}),
            (
                "episode_to_goal",
                {"title": "goal", "description": "desc", "source_episode_ids": ["ep1"]},
            ),
            ("episode_to_relationship", {"entity_name": "Alice", "source_episode_ids": ["ep1"]}),
            ("belief_to_value", {"name": "val", "statement": "stmt", "source_belief_ids": ["b1"]}),
            ("episode_to_drive", {"drive_type": "curiosity", "source_episode_ids": ["ep1"]}),
        ]

        for transition, item in cases:
            content_text = _extract_content_text(transition, item)
            chash = compute_content_hash(content_text)
            existing = MagicMock()
            existing.id = f"existing-{transition}"
            dedup_index = {"content": {chash: existing}}
            result = processor._check_duplicate(transition, item, dedup_index)
            assert result == f"existing-{transition}", f"Failed for {transition}"


# =============================================================================
# _write_memories with dedup
# =============================================================================


class TestWriteMemoriesWithDedup:
    def test_allows_same_provenance_different_content(self):
        """Same provenance but different content is NOT skipped (content-only dedup)."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        dedup_index = {"content": {}}

        parsed = [
            {"objective": "new interpretation", "outcome": "different", "source_raw_ids": ["r1"]},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 1
        assert processor._last_deduplicated == 0
        mock_stack.save_episode.assert_called_once()

    def test_skips_duplicate_by_content(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_belief.return_value = "b-new"
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "existing-belief"
        chash = compute_content_hash("testing is vital")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {"statement": "Testing is vital", "source_episode_ids": []},
        ]
        created = processor._write_memories("episode_to_belief", parsed, [], dedup_index)
        assert len(created) == 0
        assert processor._last_deduplicated == 1
        mock_stack.save_belief.assert_not_called()

    def test_creates_new_when_no_duplicate(self):
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        dedup_index = {"content": {}}

        parsed = [
            {"objective": "brand new", "outcome": "novel", "source_raw_ids": ["r99"]},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 1
        assert processor._last_deduplicated == 0
        mock_stack.save_episode.assert_called_once()

    def test_mixed_dedup_and_new(self):
        """Some items are content duplicates, some are new."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "existing-ep"
        chash = compute_content_hash("duplicate skip")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {"objective": "duplicate", "outcome": "skip", "source_raw_ids": ["r1"]},
            {"objective": "new item", "outcome": "keep", "source_raw_ids": ["r99"]},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 1
        assert processor._last_deduplicated == 1
        mock_stack.save_episode.assert_called_once()

    def test_no_dedup_index_creates_all(self):
        """Without dedup_index, all items are created (backward compat)."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.side_effect = ["ep-1", "ep-2"]
        processor, _ = _make_processor(mock_stack)

        parsed = [
            {"objective": "first", "outcome": "ok", "source_raw_ids": []},
            {"objective": "second", "outcome": "ok", "source_raw_ids": []},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [])
        assert len(created) == 2
        assert processor._last_deduplicated == 0


# =============================================================================
# ProcessingResult.deduplicated field
# =============================================================================


class TestProcessingResultDedup:
    def test_default_deduplicated(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=5,
        )
        assert result.deduplicated == 0

    def test_with_deduplicated(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=5,
            deduplicated=3,
        )
        assert result.deduplicated == 3


# =============================================================================
# Full pipeline: idempotent re-processing
# =============================================================================


class TestIdempotentProcessing:
    def test_rerun_same_cycle_zero_new_items(self, stack):
        """Re-running the same processing cycle should produce 0 new items."""
        # 1. Save raw entries
        raw_ids = []
        for i in range(3):
            rid = stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Raw content {i}",
                    source="test",
                )
            )
            raw_ids.append(rid)

        # 2. First processing pass — creates episodes
        response = json.dumps(
            [
                {
                    "objective": "Learned something",
                    "outcome": "Good result",
                    "outcome_type": "success",
                    "lessons": ["lesson 1"],
                    "source_raw_ids": raw_ids,
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        results1 = processor.process("raw_to_episode", force=True, auto_promote=True)
        assert len(results1) == 1
        assert len(results1[0].created) == 1
        assert results1[0].deduplicated == 0

        # 3. Save more raw entries (to trigger second pass)
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"New raw content {i}",
                    source="test",
                )
            )

        # 4. Second processing pass — same episode should be deduplicated
        results2 = processor.process("raw_to_episode", force=True, auto_promote=True)
        assert len(results2) == 1
        # The episode from pass 1 now exists in context, so it should be deduped
        assert results2[0].deduplicated == 1
        assert len(results2[0].created) == 0

        # Total episodes should be 1, not 2
        episodes = stack.get_episodes()
        assert len(episodes) == 1

    def test_new_sources_create_new_items(self, stack):
        """Processing with genuinely new content should create new items."""
        # 1. Save raw entries for first batch
        rid1 = stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="First raw content",
                source="test",
            )
        )

        # 2. First processing pass
        response1 = json.dumps(
            [
                {
                    "objective": "First episode",
                    "outcome": "First outcome",
                    "source_raw_ids": [rid1],
                }
            ]
        )
        inference1 = MockInference(response1)
        processor1 = MemoryProcessor(stack=stack, inference=inference1, core_id="test")
        results1 = processor1.process("raw_to_episode", force=True, auto_promote=True)
        assert len(results1[0].created) == 1

        # 3. Save new raw entries
        for i in range(2):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Different raw {i}",
                    source="test",
                )
            )

        # 4. Second processing pass with different content
        response2 = json.dumps(
            [
                {
                    "objective": "Second episode",
                    "outcome": "Different outcome",
                    "source_raw_ids": [],
                }
            ]
        )
        inference2 = MockInference(response2)
        processor2 = MemoryProcessor(stack=stack, inference=inference2, core_id="test")
        results2 = processor2.process("raw_to_episode", force=True, auto_promote=True)
        assert len(results2[0].created) == 1
        assert results2[0].deduplicated == 0

        # Total episodes should be 2
        episodes = stack.get_episodes()
        assert len(episodes) == 2

    def test_content_dedup_across_transitions(self, stack):
        """Content-based dedup should catch semantically identical content."""
        # 1. Save raw entries
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Some content",
                source="test",
            )
        )

        # 2. First pass
        response = json.dumps(
            [
                {
                    "content": "A factual note about testing",
                    "note_type": "fact",
                    "source_raw_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        results1 = processor.process("raw_to_note", force=True, auto_promote=True)
        assert len(results1[0].created) == 1

        # 3. Save more raws to trigger second pass
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="More content",
                source="test",
            )
        )

        # 4. Second pass with same note content (different casing/whitespace)
        response2 = json.dumps(
            [
                {
                    "content": "  A Factual Note   About Testing  ",
                    "note_type": "fact",
                    "source_raw_ids": [],
                }
            ]
        )
        inference2 = MockInference(response2)
        processor2 = MemoryProcessor(stack=stack, inference=inference2, core_id="test")
        results2 = processor2.process("raw_to_note", force=True, auto_promote=True)
        assert results2[0].deduplicated == 1
        assert len(results2[0].created) == 0

        notes = stack.get_notes()
        assert len(notes) == 1


# =============================================================================
# Full pipeline: dedup in _process_layer
# =============================================================================


class TestProcessLayerDedup:
    def test_audit_log_includes_deduplicated_count(self):
        """The audit log should include the dedup count."""
        mock_stack = _make_mock_stack()
        raw = MagicMock()
        raw.id = "r-1"
        raw.blob = "test content"
        raw.content = None
        mock_stack._backend.list_raw.return_value = [raw]

        # Context includes an episode that will match
        existing_ep = MagicMock()
        existing_ep.id = "existing-ep"
        existing_ep.objective = "processed item"
        existing_ep.outcome = "success"
        existing_ep.derived_from = ["raw:r-1"]
        mock_stack.get_episodes.return_value = [existing_ep]

        response = json.dumps(
            [
                {
                    "objective": "processed item",
                    "outcome": "success",
                    "source_raw_ids": ["r-1"],
                }
            ]
        )
        processor, _ = _make_processor(mock_stack, response)

        from kernle.processing import DEFAULT_LAYER_CONFIGS

        config = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        result = processor._process_layer("raw_to_episode", config, auto_promote=True)

        assert result.deduplicated == 1
        assert len(result.created) == 0

        # Verify audit log was called with deduplicated count
        mock_stack.log_audit.assert_called_once()
        call_kwargs = mock_stack.log_audit.call_args
        details = call_kwargs[1]["details"] if "details" in call_kwargs[1] else call_kwargs[0][4]
        assert details["deduplicated"] == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_empty_parsed_list(self):
        """Empty parsed list should produce no items and no errors."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)
        dedup_index = {"content": {}}
        created = processor._write_memories("raw_to_episode", [], [], dedup_index)
        assert created == []
        assert processor._last_deduplicated == 0

    def test_all_items_deduplicated(self):
        """All items being content duplicates should produce 0 created items."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        existing1 = MagicMock()
        existing1.id = "ex-1"
        existing2 = MagicMock()
        existing2.id = "ex-2"
        chash1 = compute_content_hash("dup1 out1")
        chash2 = compute_content_hash("dup2 out2")
        dedup_index = {"content": {chash1: existing1, chash2: existing2}}

        parsed = [
            {"objective": "dup1", "outcome": "out1", "source_raw_ids": ["r1"]},
            {"objective": "dup2", "outcome": "out2", "source_raw_ids": ["r2"]},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 0
        assert processor._last_deduplicated == 2

    def test_content_hash_catches_whitespace_only_difference(self):
        """Content that differs only by whitespace should be deduplicated."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "ex-1"
        chash = compute_content_hash("hello world")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {"statement": "  Hello   World  ", "source_episode_ids": []},
        ]
        created = processor._write_memories("episode_to_belief", parsed, [], dedup_index)
        assert len(created) == 0
        assert processor._last_deduplicated == 1

    def test_large_provenance_set_performance(self):
        """Dedup should handle items with many provenance refs efficiently."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        # Build a large provenance set
        large_refs = [f"raw:{uuid.uuid4()}" for _ in range(500)]
        existing = MagicMock()
        existing.id = "existing"
        existing.objective = "big episode"
        existing.outcome = "lots of sources"
        existing.derived_from = large_refs

        index = processor._build_existing_index("raw_to_episode", [existing])
        chash = compute_content_hash("big episode lots of sources")
        assert chash in index["content"]

    def test_different_content_not_deduped(self):
        """Items with different content are NOT duplicates."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "existing"
        chash = compute_content_hash("existing obj existing out")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {"objective": "partial", "outcome": "overlap", "source_raw_ids": ["r1"]},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 1
        assert processor._last_deduplicated == 0

    def test_dedup_belief_to_value(self):
        """Test content dedup for belief_to_value transition."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "existing-value"
        chash = compute_content_hash("honesty be truthful")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {"name": "honesty", "statement": "be truthful", "source_belief_ids": ["b1"]},
        ]
        created = processor._write_memories("belief_to_value", parsed, [], dedup_index)
        assert len(created) == 0
        assert processor._last_deduplicated == 1

    def test_dedup_episode_to_drive(self):
        """Test content dedup for episode_to_drive transition."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "existing-drive"
        chash = compute_content_hash("curiosity")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {"drive_type": "Curiosity", "source_episode_ids": []},
        ]
        created = processor._write_memories("episode_to_drive", parsed, [], dedup_index)
        assert len(created) == 0
        assert processor._last_deduplicated == 1

    def test_dedup_episode_to_relationship(self):
        """Test content dedup for episode_to_relationship transition."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "existing-rel"
        chash = compute_content_hash("Alice")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {"entity_name": "alice", "entity_type": "person", "source_episode_ids": []},
        ]
        created = processor._write_memories("episode_to_relationship", parsed, [], dedup_index)
        assert len(created) == 0
        assert processor._last_deduplicated == 1


# =============================================================================
# Parametrized tests for all transition types
# =============================================================================


ALL_TRANSITIONS_DATA = [
    (
        "raw_to_episode",
        {"objective": "obj", "outcome": "out", "source_raw_ids": ["r1"]},
        "save_episode",
        "ep-new",
        "episode",
    ),
    (
        "raw_to_note",
        {"content": "a note", "note_type": "fact", "source_raw_ids": ["r1"]},
        "save_note",
        "n-new",
        "note",
    ),
    (
        "episode_to_belief",
        {"statement": "belief text", "confidence": 0.8, "source_episode_ids": ["ep1"]},
        "save_belief",
        "b-new",
        "belief",
    ),
    (
        "episode_to_goal",
        {"title": "do stuff", "description": "desc", "source_episode_ids": ["ep1"]},
        "save_goal",
        "g-new",
        "goal",
    ),
    (
        "episode_to_relationship",
        {"entity_name": "Bob", "entity_type": "person", "source_episode_ids": ["ep1"]},
        "save_relationship",
        "rel-new",
        "relationship",
    ),
    (
        "belief_to_value",
        {"name": "integrity", "statement": "be honest", "source_belief_ids": ["b1"]},
        "save_value",
        "v-new",
        "value",
    ),
    (
        "episode_to_drive",
        {"drive_type": "ambition", "intensity": 0.7, "source_episode_ids": ["ep1"]},
        "save_drive",
        "d-new",
        "drive",
    ),
]


@pytest.mark.parametrize(
    "transition,item,save_method,save_return,memory_type",
    ALL_TRANSITIONS_DATA,
    ids=[t[0] for t in ALL_TRANSITIONS_DATA],
)
class TestDedupAllTransitions:
    def test_content_dedup_skips(self, transition, item, save_method, save_return, memory_type):
        """Content dedup should skip creation for each transition type."""
        mock_stack = _make_mock_stack()
        getattr(mock_stack, save_method).return_value = save_return
        processor, _ = _make_processor(mock_stack)

        content_text = _extract_content_text(transition, item)
        existing = MagicMock()
        existing.id = f"existing-{memory_type}"
        chash = compute_content_hash(content_text)
        dedup_index = {"content": {chash: existing} if chash else {}}

        created = processor._write_memories(transition, [item], [], dedup_index)
        if chash:
            assert len(created) == 0
            assert processor._last_deduplicated == 1
            getattr(mock_stack, save_method).assert_not_called()

    def test_no_match_creates(self, transition, item, save_method, save_return, memory_type):
        """Without a match, item should be created normally."""
        mock_stack = _make_mock_stack()
        getattr(mock_stack, save_method).return_value = save_return
        processor, _ = _make_processor(mock_stack)

        dedup_index = {"content": {}}
        created = processor._write_memories(transition, [item], [], dedup_index)
        assert len(created) == 1
        assert created[0]["type"] == memory_type
        assert processor._last_deduplicated == 0
        getattr(mock_stack, save_method).assert_called_once()


# =============================================================================
# Random ingestion order tests (property-based without hypothesis)
# =============================================================================


class TestRandomIngestionOrder:
    """Verify dedup is order-independent by shuffling source refs and items."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024])
    def test_provenance_hash_order_independent_random(self, seed):
        """Provenance hash is the same regardless of ref list ordering."""
        rng = random.Random(seed)
        refs = [f"raw:{uuid.uuid4()}" for _ in range(rng.randint(2, 20))]
        h_original = compute_provenance_hash(refs)

        for _ in range(10):
            shuffled = list(refs)
            rng.shuffle(shuffled)
            assert compute_provenance_hash(shuffled) == h_original

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024])
    def test_dedup_order_independent_for_items(self, seed):
        """Processing items in any order produces same dedup result."""
        rng = random.Random(seed)
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        # Create N items, some of which are duplicates
        ref_pool = [f"raw:r{i}" for i in range(5)]
        unique_items = [
            {"objective": f"obj-{i}", "outcome": f"out-{i}", "source_raw_ids": [ref_pool[i]]}
            for i in range(5)
        ]

        # Build dedup index from first 3 items (simulating existing memories)
        existing_records = []
        for i in range(3):
            rec = MagicMock()
            rec.id = f"existing-{i}"
            rec.objective = f"obj-{i}"
            rec.outcome = f"out-{i}"
            rec.derived_from = [ref_pool[i]]
            existing_records.append(rec)

        # Shuffle the items and verify same dedup outcome
        for _ in range(5):
            shuffled = list(unique_items)
            rng.shuffle(shuffled)
            mock_stack.save_episode.reset_mock()
            mock_stack.save_episode.return_value = "ep-new"
            # Rebuild index each iteration since _write_memories mutates it
            fresh_index = processor._build_existing_index("raw_to_episode", existing_records)
            created = processor._write_memories("raw_to_episode", shuffled, [], fresh_index)
            # Items 0-2 are duplicates, items 3-4 are new
            assert (
                processor._last_deduplicated == 3
            ), f"Expected 3 deduped, got {processor._last_deduplicated}"
            assert len(created) == 2, f"Expected 2 created, got {len(created)}"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_content_hash_dedup_order_independent(self, seed):
        """Content dedup finds duplicates regardless of item ordering."""
        rng = random.Random(seed)
        mock_stack = _make_mock_stack()
        mock_stack.save_belief.return_value = "b-new"
        processor, _ = _make_processor(mock_stack)

        items = [{"statement": f"belief {i}", "source_episode_ids": []} for i in range(6)]

        # Index first 4 beliefs by content
        existing = []
        for i in range(4):
            rec = MagicMock(spec=["id", "statement", "belief_type", "derived_from"])
            rec.id = f"existing-{i}"
            rec.statement = f"belief {i}"
            rec.belief_type = "factual"
            rec.derived_from = None
            del rec.objective
            existing.append(rec)

        for _ in range(5):
            shuffled = list(items)
            rng.shuffle(shuffled)
            mock_stack.save_belief.reset_mock()
            mock_stack.save_belief.return_value = "b-new"
            # Rebuild index each iteration since _write_memories mutates it
            fresh_index = processor._build_existing_index("episode_to_belief", existing)
            created = processor._write_memories("episode_to_belief", shuffled, [], fresh_index)
            assert processor._last_deduplicated == 4
            assert len(created) == 2

    def test_full_pipeline_order_independent(self, stack):
        """Full integration: save raws in random order, process, re-process."""
        rng = random.Random(42)
        raw_ids = []
        blobs = [f"Raw content item {i}" for i in range(5)]
        rng.shuffle(blobs)

        for blob in blobs:
            rid = stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=blob,
                    source="test",
                )
            )
            raw_ids.append(rid)

        # First processing pass
        response = json.dumps(
            [
                {
                    "objective": "Synthesized episode",
                    "outcome": "From random-order raws",
                    "source_raw_ids": raw_ids[:3],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        results1 = processor.process("raw_to_episode", force=True, auto_promote=True)
        assert len(results1[0].created) == 1

        # Save more raws (to have unprocessed sources for second pass)
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Extra raw {i}",
                    source="test",
                )
            )

        # Second pass with same response (should be deduped)
        results2 = processor.process("raw_to_episode", force=True, auto_promote=True)
        assert results2[0].deduplicated == 1
        assert len(results2[0].created) == 0

        # Only 1 episode total
        episodes = stack.get_episodes()
        assert len(episodes) == 1


# =============================================================================
# Updated content with same provenance (skip policy)
# =============================================================================


class TestContentOnlyDedupPolicy:
    def test_same_provenance_different_content_allowed(self):
        """Same provenance but different content is NOT a duplicate (content-only dedup)."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        # Existing episode with specific content indexed
        existing = MagicMock()
        existing.id = "existing-ep"
        chash = compute_content_hash("Original interpretation First version")
        dedup_index = {"content": {chash: existing}}

        # New item from same provenance but DIFFERENT content -> should be created
        parsed = [
            {
                "objective": "Different interpretation",
                "outcome": "Second version",
                "source_raw_ids": ["r1", "r2"],
            }
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 1
        assert processor._last_deduplicated == 0
        mock_stack.save_episode.assert_called_once()

    def test_same_content_same_provenance_skips(self):
        """Same content (regardless of provenance) is caught by content dedup."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "existing-ep"
        chash = compute_content_hash("Same interpretation Same version")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {
                "objective": "Same interpretation",
                "outcome": "Same version",
                "source_raw_ids": ["r1"],
            }
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 0
        assert processor._last_deduplicated == 1

    def test_content_only_match_different_provenance(self):
        """Content match with different provenance still deduplicates."""
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.id = "existing-belief"
        chash = compute_content_hash("important insight")
        dedup_index = {"content": {chash: existing}}

        # Same content, different provenance
        item = {"statement": "Important insight", "source_episode_ids": ["ep999"]}
        result = processor._check_duplicate("episode_to_belief", item, dedup_index)
        assert result == "existing-belief"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_mixed_new_and_content_duplicates(self, seed):
        """Mix of new items and content duplicates (provenance is irrelevant)."""
        rng = random.Random(seed)
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.side_effect = ["ep-new-1", "ep-new-2"]
        processor, _ = _make_processor(mock_stack)

        # Existing memories
        existing1 = MagicMock()
        existing1.id = "ex-1"
        existing1.objective = "original obj 1"
        existing1.outcome = "original out 1"
        existing1.derived_from = ["raw:r1"]

        existing2 = MagicMock()
        existing2.id = "ex-2"
        existing2.objective = "original obj 2"
        existing2.outcome = "original out 2"
        existing2.derived_from = ["raw:r2"]

        dedup_index = processor._build_existing_index("raw_to_episode", [existing1, existing2])

        items = [
            # Exact content duplicate of existing1 -> deduplicated
            {"objective": "original obj 1", "outcome": "original out 1", "source_raw_ids": ["r1"]},
            # Same provenance as existing2, DIFFERENT content -> NOT deduped (content-only)
            {"objective": "UPDATED obj 2", "outcome": "CHANGED out 2", "source_raw_ids": ["r2"]},
            # Genuinely new
            {"objective": "brand new", "outcome": "novel", "source_raw_ids": ["r99"]},
        ]

        rng.shuffle(items)
        created = processor._write_memories("raw_to_episode", items, [], dedup_index)
        assert len(created) == 2  # Updated content + genuinely new
        assert processor._last_deduplicated == 1  # Only the exact content duplicate


# =============================================================================
# Intra-batch dedup (P1 regression: duplicate items within a single batch)
# =============================================================================


class TestIntraBatchDedup:
    """Regression tests for intra-batch dedup gap (P1 bug).

    The bug: _write_memories() checked dedup only against the prebuilt
    dedup_index but did NOT update the index as new items were written
    during the same batch. Two identical parsed items in one run were
    both created (deduplicated=0).
    """

    def test_identical_items_in_same_batch_by_content(self):
        """Two identical parsed items in one batch: only 1 created, 1 deduped."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        dedup_index = {"content": {}}

        parsed = [
            {"objective": "same thing", "outcome": "same result", "source_raw_ids": ["r1"]},
            {"objective": "same thing", "outcome": "same result", "source_raw_ids": ["r1"]},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 1, f"Expected 1 created, got {len(created)}"
        assert (
            processor._last_deduplicated == 1
        ), f"Expected 1 deduplicated, got {processor._last_deduplicated}"
        mock_stack.save_episode.assert_called_once()

    def test_same_provenance_different_content_both_created(self):
        """Two items with same provenance but different content: both created (content-only dedup)."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.side_effect = ["ep-1", "ep-2"]
        processor, _ = _make_processor(mock_stack)

        dedup_index = {"content": {}}

        parsed = [
            {
                "objective": "interpretation A",
                "outcome": "version A",
                "source_raw_ids": ["r1", "r2"],
            },
            {
                "objective": "interpretation B",
                "outcome": "version B",
                "source_raw_ids": ["r1", "r2"],
            },
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 2
        assert processor._last_deduplicated == 0
        assert mock_stack.save_episode.call_count == 2

    def test_three_identical_items_in_batch(self):
        """Three identical items: only 1 created, 2 deduped."""
        mock_stack = _make_mock_stack()
        mock_stack.save_belief.return_value = "b-new"
        processor, _ = _make_processor(mock_stack)

        dedup_index = {"content": {}}

        parsed = [
            {"statement": "testing is important", "source_episode_ids": ["ep1"]},
            {"statement": "testing is important", "source_episode_ids": ["ep1"]},
            {"statement": "testing is important", "source_episode_ids": ["ep1"]},
        ]
        created = processor._write_memories("episode_to_belief", parsed, [], dedup_index)
        assert len(created) == 1
        assert processor._last_deduplicated == 2
        mock_stack.save_belief.assert_called_once()

    def test_intra_batch_mixed_unique_and_duplicate(self):
        """Batch with 2 unique + 2 duplicates: 2 created, 2 deduped."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.side_effect = ["ep-1", "ep-2"]
        processor, _ = _make_processor(mock_stack)

        dedup_index = {"content": {}}

        parsed = [
            {"objective": "first unique", "outcome": "out1", "source_raw_ids": ["r1"]},
            {"objective": "second unique", "outcome": "out2", "source_raw_ids": ["r2"]},
            {"objective": "first unique", "outcome": "out1", "source_raw_ids": ["r1"]},
            {"objective": "second unique", "outcome": "out2", "source_raw_ids": ["r2"]},
        ]
        created = processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert len(created) == 2
        assert processor._last_deduplicated == 2

    def test_intra_batch_content_only_no_provenance(self):
        """Content-only dedup within batch (no provenance refs)."""
        mock_stack = _make_mock_stack()
        mock_stack.save_belief.return_value = "b-new"
        processor, _ = _make_processor(mock_stack)

        dedup_index = {"content": {}}

        parsed = [
            {"statement": "Honesty matters", "source_episode_ids": []},
            {"statement": "honesty   matters", "source_episode_ids": []},
        ]
        created = processor._write_memories("episode_to_belief", parsed, [], dedup_index)
        assert len(created) == 1
        assert processor._last_deduplicated == 1

    def test_full_pipeline_intra_batch_dedup(self, stack):
        """Full integration: model returns duplicate items in one response."""
        # Save raw entries
        raw_ids = []
        for i in range(3):
            rid = stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Content {i}",
                    source="test",
                )
            )
            raw_ids.append(rid)

        # Model returns two identical episodes in one response
        response = json.dumps(
            [
                {
                    "objective": "Learned about testing",
                    "outcome": "Testing is important",
                    "outcome_type": "success",
                    "source_raw_ids": raw_ids,
                },
                {
                    "objective": "Learned about testing",
                    "outcome": "Testing is important",
                    "outcome_type": "success",
                    "source_raw_ids": raw_ids,
                },
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        results = processor.process("raw_to_episode", force=True, auto_promote=True)

        assert len(results) == 1
        assert len(results[0].created) == 1, f"Expected 1 created, got {len(results[0].created)}"
        assert (
            results[0].deduplicated == 1
        ), f"Expected 1 deduplicated, got {results[0].deduplicated}"

        # Only 1 episode in the stack
        episodes = stack.get_episodes()
        assert len(episodes) == 1


# =============================================================================
# P3 Regression: stale _last_deduplicated in suggestions mode
# =============================================================================


class TestStaleDedupMetricInSuggestionsMode:
    """Regression tests for P3: _write_suggestions() did not reset
    _last_deduplicated, so a subsequent suggestions-mode run could report
    a stale dedup count from a prior auto_promote run.
    """

    def test_suggestions_mode_resets_deduplicated_count(self):
        """After auto_promote run with dedup > 0, suggestions mode must report 0."""
        mock_stack = _make_mock_stack()
        mock_stack.save_episode.return_value = "ep-new"
        processor, _ = _make_processor(mock_stack)

        # 1. Run _write_memories with a content duplicate -> sets _last_deduplicated = 1
        existing = MagicMock()
        existing.id = "existing-ep"
        chash = compute_content_hash("dup skip")
        dedup_index = {"content": {chash: existing}}

        parsed = [
            {"objective": "dup", "outcome": "skip", "source_raw_ids": ["r1"]},
        ]
        processor._write_memories("raw_to_episode", parsed, [], dedup_index)
        assert processor._last_deduplicated == 1

        # 2. Run _write_suggestions (suggestions mode) -> must reset to 0
        mock_stack.save_suggestion = MagicMock(return_value="sug-1")
        mock_stack._backend.execute.return_value = None
        suggestions_parsed = [
            {"objective": "new thing", "outcome": "novel", "source_raw_ids": ["r99"]},
        ]
        processor._write_suggestions("raw_to_episode", suggestions_parsed, [])
        assert processor._last_deduplicated == 0, (
            f"Expected _last_deduplicated=0 after _write_suggestions, "
            f"got {processor._last_deduplicated}"
        )

    def test_full_pipeline_suggestions_after_promote_no_stale_dedup(self, stack):
        """Integration: auto_promote run with dedup, then suggestions run reports 0."""
        # 1. Save raw entries
        raw_ids = []
        for i in range(3):
            rid = stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Content {i}",
                    source="test",
                )
            )
            raw_ids.append(rid)

        # 2. First pass: auto_promote=True, creates an episode
        response = json.dumps(
            [
                {
                    "objective": "Learned something",
                    "outcome": "Good result",
                    "outcome_type": "success",
                    "source_raw_ids": raw_ids,
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        results1 = processor.process("raw_to_episode", force=True, auto_promote=True)
        assert len(results1) == 1
        assert len(results1[0].created) == 1
        assert results1[0].deduplicated == 0

        # 3. Save more raws so second pass finds sources
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"New raw {i}",
                    source="test",
                )
            )

        # 4. Second pass: auto_promote=True, same response -> dedup fires
        results2 = processor.process("raw_to_episode", force=True, auto_promote=True)
        assert len(results2) == 1
        assert results2[0].deduplicated == 1

        # 5. Save more raws for third pass
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Third batch raw {i}",
                    source="test",
                )
            )

        # 6. Third pass: auto_promote=False (suggestions mode)
        #    Must NOT carry stale deduplicated=1 from pass 2
        results3 = processor.process("raw_to_episode", force=True, auto_promote=False)
        assert len(results3) == 1
        assert results3[0].deduplicated == 0, (
            f"Expected deduplicated=0 in suggestions mode, "
            f"got {results3[0].deduplicated} (stale value leaked)"
        )


# =============================================================================
# Issue #635: suggestion dedup/idempotency for partial-failure retries
# =============================================================================


class TestSuggestionDedupIdempotency:
    def test_write_suggestions_skips_existing_pending_duplicates(self):
        mock_stack = _make_mock_stack()
        processor, _ = _make_processor(mock_stack)

        existing = MagicMock()
        existing.content = {
            "objective": "already captured",
            "outcome": "keep once",
            "source_raw_ids": ["r-1"],
            "confidence": 0.7,
        }
        existing.source_raw_ids = ["r-1"]
        mock_stack.get_suggestions.return_value = [existing]
        mock_stack.save_suggestion.return_value = "s-new"

        parsed = [
            {
                "objective": "already captured",
                "outcome": "keep once",
                "source_raw_ids": ["r-1"],
                "confidence": 0.9,  # Different confidence should still dedup
            },
            {
                "objective": "new insight",
                "outcome": "create this one",
                "source_raw_ids": ["r-2"],
            },
        ]

        created = processor._write_suggestions("raw_to_episode", parsed, [])
        assert created == [{"type": "episode", "id": "s-new"}]
        assert mock_stack.save_suggestion.call_count == 1
        assert processor._last_deduplicated == 1

    def test_reprocess_after_partial_failure_does_not_duplicate_pending(self, stack, monkeypatch):
        """When a batch partially fails, retry should only create missing suggestions."""
        raw_ids = []
        for i in range(2):
            rid = stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Retry raw {i}",
                    source="test",
                )
            )
            raw_ids.append(rid)

        response = json.dumps(
            [
                {
                    "objective": "first suggestion",
                    "outcome": "failed first pass",
                    "source_raw_ids": raw_ids,
                },
                {
                    "objective": "second suggestion",
                    "outcome": "already saved on first pass",
                    "source_raw_ids": raw_ids,
                },
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        original_save = stack.save_suggestion
        fail_state = {"raised": False}

        def flaky_save(suggestion):
            if (
                suggestion.content.get("objective") == "first suggestion"
                and not fail_state["raised"]
            ):
                fail_state["raised"] = True
                raise RuntimeError("simulated partial failure")
            return original_save(suggestion)

        monkeypatch.setattr(stack, "save_suggestion", flaky_save)

        # First pass: one create + one write error, so sources remain unprocessed.
        results1 = processor.process("raw_to_episode", force=True, auto_promote=False)
        assert len(results1) == 1
        assert len(results1[0].suggestions) == 1
        assert any("Suggestion write failed for raw_to_episode" in e for e in results1[0].errors)

        pending_after_first = stack.get_suggestions(
            status="pending", memory_type="episode", limit=10
        )
        assert len(pending_after_first) == 1

        # Second pass retries same parsed output: dedup should skip existing pending
        # and create only the previously failed suggestion.
        results2 = processor.process("raw_to_episode", force=True, auto_promote=False)
        assert len(results2) == 1
        assert len(results2[0].suggestions) == 1
        assert results2[0].deduplicated == 1

        pending_after_second = stack.get_suggestions(
            status="pending", memory_type="episode", limit=10
        )
        assert len(pending_after_second) == 2
        objectives = sorted(s.content.get("objective") for s in pending_after_second)
        assert objectives == ["first suggestion", "second suggestion"]
