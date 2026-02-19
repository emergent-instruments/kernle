"""Tests for exact provenance matching in derived_from queries.

Verifies that get_memories_derived_from uses exact ID matching
rather than substring/prefix matching. LIKE-based SQL queries
can cause false positives when one ID is a prefix of another
(e.g., searching for "belief:abc" matching "belief:abcdef").

See: FIND-STOR-01
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from kernle.stack.sqlite_stack import Stack
from kernle.storage.memory_ops import _exact_derived_from_match
from kernle.types import Belief, Episode


def _id():
    return str(uuid.uuid4())


def _make_episode(stack_id, derived_from=None, strength=1.0):
    return Episode(
        id=_id(),
        stack_id=stack_id,
        objective="test objective",
        outcome="test outcome",
        created_at=datetime.now(timezone.utc),
        source_type="direct_experience",
        derived_from=derived_from,
        strength=strength,
    )


def _make_belief(stack_id, derived_from=None, strength=1.0):
    return Belief(
        id=_id(),
        stack_id=stack_id,
        statement="test belief",
        belief_type="fact",
        confidence=0.8,
        created_at=datetime.now(timezone.utc),
        source_type="direct_experience",
        derived_from=derived_from,
        strength=strength,
    )


@pytest.fixture
def stack(tmp_path):
    """Create an Stack with bare components for testing."""
    return Stack.from_sqlite(
        stack_id="test-provenance",
        db_path=tmp_path / "test.db",
        components=[],
    )


class TestExactDerivedFromMatch:
    """Unit tests for the _exact_derived_from_match helper."""

    def test_exact_match_returns_true(self):
        derived_json = json.dumps(["belief:abc"])
        assert _exact_derived_from_match(derived_json, "belief", "abc") is True

    def test_prefix_match_returns_false(self):
        """'belief:abc' must NOT match a record containing 'belief:abcdef'."""
        derived_json = json.dumps(["belief:abcdef"])
        assert _exact_derived_from_match(derived_json, "belief", "abc") is False

    def test_suffix_overlap_returns_false(self):
        """'belief:abcdef' must NOT match a record containing 'belief:abc'."""
        derived_json = json.dumps(["belief:abc"])
        assert _exact_derived_from_match(derived_json, "belief", "abcdef") is False

    def test_multiple_entries_exact_match(self):
        derived_json = json.dumps(["episode:xyz", "belief:abc", "note:123"])
        assert _exact_derived_from_match(derived_json, "belief", "abc") is True

    def test_multiple_entries_no_match(self):
        derived_json = json.dumps(["episode:xyz", "belief:abcdef", "note:123"])
        assert _exact_derived_from_match(derived_json, "belief", "abc") is False

    def test_malformed_json_returns_false(self):
        assert _exact_derived_from_match("not valid json", "belief", "abc") is False

    def test_none_returns_false(self):
        assert _exact_derived_from_match(None, "belief", "abc") is False

    def test_empty_string_returns_false(self):
        assert _exact_derived_from_match("", "belief", "abc") is False

    def test_json_object_not_list_returns_false(self):
        derived_json = json.dumps({"key": "belief:abc"})
        assert _exact_derived_from_match(derived_json, "belief", "abc") is False

    def test_json_null_returns_false(self):
        assert _exact_derived_from_match("null", "belief", "abc") is False

    def test_empty_list_returns_false(self):
        derived_json = json.dumps([])
        assert _exact_derived_from_match(derived_json, "belief", "abc") is False


class TestDerivedFromPrefixAmbiguity:
    """Integration tests: prefix IDs must not cause false matches."""

    def test_prefix_id_not_matched(self, stack):
        """Querying for 'belief:abc' must NOT match derived_from=['belief:abcdef']."""
        # Create the source episode
        ep = _make_episode(stack.stack_id)
        stack.save_episode(ep)

        # Create a belief whose derived_from references a LONGER id "abc" + "def"
        short_id = "abc"
        long_id = "abcdef"
        belief = _make_belief(stack.stack_id, derived_from=[f"belief:{long_id}"])
        stack.save_belief(belief)

        # Search for the SHORT id -- should NOT match the belief with the long id
        children = stack.get_memories_derived_from("belief", short_id)
        assert len(children) == 0

    def test_exact_id_matched(self, stack):
        """Querying for 'belief:abcdef' DOES match derived_from=['belief:abcdef']."""
        long_id = "abcdef"
        belief = _make_belief(stack.stack_id, derived_from=[f"belief:{long_id}"])
        belief_id = stack.save_belief(belief)

        children = stack.get_memories_derived_from("belief", long_id)
        assert len(children) >= 1
        child_ids = [cid for _, cid in children]
        assert belief_id in child_ids

    def test_prefix_ambiguity_with_uuid_like_ids(self, stack):
        """Realistic scenario: short UUID prefix should not match full UUID."""
        full_uuid = str(uuid.uuid4())
        prefix = full_uuid[:8]  # First 8 chars

        ep = _make_episode(stack.stack_id)
        stack.save_episode(ep)

        # Belief references the full UUID
        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{full_uuid}"])
        stack.save_belief(belief)

        # Search with the prefix -- should NOT match
        children = stack.get_memories_derived_from("episode", prefix)
        assert len(children) == 0

        # Search with the full UUID -- SHOULD match
        children = stack.get_memories_derived_from("episode", full_uuid)
        assert len(children) == 1

    def test_existing_exact_match_still_works(self, stack):
        """Regression: normal exact matching still works after the fix."""
        ep = _make_episode(stack.stack_id)
        ep_id = stack.save_episode(ep)

        belief = _make_belief(stack.stack_id, derived_from=[f"episode:{ep_id}"])
        belief_id = stack.save_belief(belief)

        children = stack.get_memories_derived_from("episode", ep_id)
        assert len(children) == 1
        assert children[0] == ("belief", belief_id)

    def test_multiple_refs_one_is_prefix(self, stack):
        """A belief with multiple refs where one is a prefix of the search term."""
        target_id = _id()
        prefix_id = target_id[:12]

        belief = _make_belief(
            stack.stack_id,
            derived_from=[f"episode:{prefix_id}", f"episode:{target_id}"],
        )
        stack.save_belief(belief)

        # Search for prefix_id should match (it's an exact member)
        children = stack.get_memories_derived_from("episode", prefix_id)
        assert len(children) == 1

        # Search for target_id should also match (it's an exact member)
        children = stack.get_memories_derived_from("episode", target_id)
        assert len(children) == 1
