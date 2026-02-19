"""Tests for accept_suggestion() lint-redirect handling.

Verifies that when save_belief()/save_value() returns a lint-redirect
(string starting with "suggestion:"), accept_suggestion() does NOT
mark the original suggestion as promoted.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from kernle.stack import Stack
from kernle.types import MemorySuggestion

STACK_ID = "lint-redirect-test"


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def stack(tmp_path):
    db_path = tmp_path / "lint_redirect.db"
    return Stack.from_sqlite(
        stack_id=STACK_ID, db_path=db_path, components=[], enforce_provenance=False
    )


def _make_suggestion(**kw) -> MemorySuggestion:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        memory_type="belief",
        content={"statement": "Testing is good", "belief_type": "fact"},
        confidence=0.8,
        source_raw_ids=["raw-1"],
        created_at=_now(),
    )
    defaults.update(kw)
    return MemorySuggestion(**defaults)


class TestSuggestionLintRedirect:
    """Verify accept_suggestion handles lint-redirect returns correctly."""

    def test_accept_lint_rejected_does_not_promote(self, stack):
        """When save_belief returns suggestion:*, original stays pending."""
        s = _make_suggestion()
        stack.save_suggestion(s)

        # Patch save_belief to simulate lint redirect
        with patch.object(stack, "save_belief", return_value="suggestion:fake-redirect-id"):
            result = stack.accept_suggestion(s.id)

        # Should return None (not the redirect ID)
        assert result is None

        # Original suggestion should NOT be marked promoted
        updated = stack.get_suggestion(s.id)
        assert updated is not None
        assert updated.status == "pending"

    def test_accept_lint_rejected_creates_new_suggestion(self, stack):
        """Lint-redirect creates a new suggestion; original stays pending."""
        s = _make_suggestion()
        stack.save_suggestion(s)

        new_suggestion_id = _uid()
        with patch.object(stack, "save_belief", return_value=f"suggestion:{new_suggestion_id}"):
            result = stack.accept_suggestion(s.id)

        assert result is None
        # Original is still pending
        assert stack.get_suggestion(s.id).status == "pending"

    def test_accept_creates_belief_count_increases(self, stack):
        """Normal accept (no lint redirect) creates a belief."""
        s = _make_suggestion()
        stack.save_suggestion(s)

        before = len(stack.get_beliefs(limit=1000))
        memory_id = stack.accept_suggestion(s.id)
        after = len(stack.get_beliefs(limit=1000))

        assert memory_id is not None
        assert after == before + 1

    def test_accept_lint_redirect_belief_count_unchanged(self, stack):
        """Lint-redirect does NOT create a belief."""
        s = _make_suggestion()
        stack.save_suggestion(s)

        before = len(stack.get_beliefs(limit=1000))
        with patch.object(stack, "save_belief", return_value="suggestion:fake-redirect"):
            result = stack.accept_suggestion(s.id)
        after = len(stack.get_beliefs(limit=1000))

        assert result is None
        assert after == before

    def test_promoted_to_never_contains_suggestion_prefix(self, stack):
        """After normal accept, promoted_to never starts with 'suggestion:'."""
        s = _make_suggestion()
        stack.save_suggestion(s)
        stack.accept_suggestion(s.id)

        updated = stack.get_suggestion(s.id)
        assert updated.status == "promoted"
        assert updated.promoted_to is not None
        assert not updated.promoted_to.startswith("suggestion:")
