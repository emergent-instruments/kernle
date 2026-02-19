"""Tests for Stack after AnxietyMixin removal (#571).

Verifies that:
- AnxietyMixin methods are no longer on the stack
- Component-level anxiety still works
- Core stack operations still work
- get_identity_confidence is retained
"""

import uuid

import pytest

from kernle.stack import Stack
from kernle.types import Value


@pytest.fixture
def stack(tmp_path):
    db_path = tmp_path / "test_no_mixin.db"
    return Stack.from_sqlite(
        stack_id="test-stack",
        db_path=db_path,
        enforce_provenance=False,
    )


@pytest.fixture
def bare_stack(tmp_path):
    db_path = tmp_path / "test_no_mixin_bare.db"
    return Stack.from_sqlite(
        stack_id="test-stack",
        db_path=db_path,
        components=[],
        enforce_provenance=False,
    )


class TestStackNoAnxietyMixin:
    """Stack no longer inherits AnxietyMixin."""

    def test_stack_no_mixin_anxiety_methods(self, stack):
        """Mixin-specific methods should not exist on the stack."""
        assert not hasattr(stack, "emergency_save")
        # The mixin's anxiety() method (alias for get_anxiety_report) should be gone
        # But the component's get_anxiety_report is called differently (via component)

    def test_stack_load_includes_component_anxiety(self, stack):
        """stack.load() result contains anxiety from AnxietyComponent's on_load."""
        result = stack.load()
        # The AnxietyComponent's on_load adds anxiety to context
        assert "anxiety" in result

    def test_stack_no_checkpoint_stubs(self, stack):
        """load_checkpoint and checkpoint stubs no longer exist."""
        assert not hasattr(stack, "load_checkpoint")
        assert not hasattr(stack, "checkpoint")

    def test_stack_basic_ops_still_work(self, stack):
        """save/load/search/maintenance all work."""
        # Save
        val = Value(
            id=str(uuid.uuid4()),
            stack_id="test-stack",
            name="test",
            statement="test value",
            source_type="direct_experience",
        )
        vid = stack.save_value(val)
        assert vid

        # Load
        result = stack.load()
        assert "values" in result

        # Search
        results = stack.search("test")
        assert isinstance(results, list)

        # Maintenance
        maint = stack.maintenance()
        assert isinstance(maint, dict)

    def test_stack_get_identity_confidence_retained(self, stack):
        """get_identity_confidence method still exists (used by AnxietyComponent)."""
        conf = stack.get_identity_confidence()
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0
