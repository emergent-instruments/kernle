"""Tests for Issue #354: Component ordering for on_save/on_search hooks.

Verifies:
- All built-in components declare a valid priority
- Components are sorted by priority in the stack's registry
- Hook dispatch follows priority order
- External components default to priority 200
"""

from unittest.mock import MagicMock

import pytest

from kernle.stack import Stack
from kernle.stack.components import (
    AnxietyComponent,
    ConsolidationComponent,
    EmbeddingComponent,
    EmotionalTaggingComponent,
    ForgettingComponent,
    KnowledgeComponent,
    MetaMemoryComponent,
    SuggestionComponent,
    get_default_components,
)

EXPECTED_PRIORITIES = {
    "embedding-ngram": 10,
    "emotions": 100,
    "metamemory": 110,
    "consolidation": 200,
    "suggestions": 210,
    "knowledge": 220,
    "anxiety": 250,
    "forgetting": 300,
}


class TestComponentPriority:
    """Verify priority is declared on all built-in components."""

    @pytest.mark.parametrize(
        "component_cls,expected_priority",
        [
            (EmbeddingComponent, 10),
            (EmotionalTaggingComponent, 100),
            (MetaMemoryComponent, 110),
            (ConsolidationComponent, 200),
            (SuggestionComponent, 210),
            (KnowledgeComponent, 220),
            (AnxietyComponent, 250),
            (ForgettingComponent, 300),
        ],
    )
    def test_priority_matches_expected(self, component_cls, expected_priority):
        component = component_cls()
        assert component.priority == expected_priority, (
            f"{component.name}: expected priority {expected_priority}, " f"got {component.priority}"
        )

    def test_all_defaults_have_priority(self):
        """All 8 default components should have a priority attribute."""
        components = get_default_components()
        for c in components:
            assert hasattr(c, "priority"), f"Component {c.name} missing priority"
            assert isinstance(c.priority, int), f"Component {c.name} priority not int"

    def test_priority_ranges(self):
        """Priorities should fall within documented ranges."""
        components = get_default_components()
        for c in components:
            assert (
                0 <= c.priority <= 999
            ), f"Component {c.name} has priority {c.priority} outside valid range"


class TestComponentOrdering:
    """Verify the stack sorts and dispatches components by priority."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return tmp_path / "test_ordering.db"

    def test_components_sorted_by_priority(self, tmp_db):
        """Default components should be sorted by priority in the registry."""
        stack = Stack.from_sqlite(
            stack_id="test-ordering",
            db_path=tmp_db,
            enforce_provenance=False,
        )
        priorities = [getattr(c, "priority", 200) for c in stack.components.values()]
        assert priorities == sorted(priorities), (
            f"Components not sorted by priority: {list(stack.components.keys())} "
            f"with priorities {priorities}"
        )

    def test_embedding_runs_first(self, tmp_db):
        """EmbeddingComponent (priority 10) should be first in the registry."""
        stack = Stack.from_sqlite(
            stack_id="test-ordering",
            db_path=tmp_db,
            enforce_provenance=False,
        )
        first_name = list(stack.components.keys())[0]
        assert first_name == "embedding-ngram", f"Expected embedding-ngram first, got {first_name}"

    def test_playbooks_runs_last(self, tmp_db):
        """PlaybookComponent (priority 400) should be last in the registry."""
        stack = Stack.from_sqlite(
            stack_id="test-ordering",
            db_path=tmp_db,
            enforce_provenance=False,
        )
        last_name = list(stack.components.keys())[-1]
        assert last_name == "playbooks", f"Expected playbooks last, got {last_name}"

    def test_add_component_maintains_order(self, tmp_db):
        """Adding a component re-sorts the registry by priority."""
        stack = Stack.from_sqlite(
            stack_id="test-ordering",
            db_path=tmp_db,
            components=[],
            enforce_provenance=False,
        )

        # Add in reverse priority order
        stack.add_component(ForgettingComponent())  # 300
        stack.add_component(EmbeddingComponent())  # 10
        stack.add_component(ConsolidationComponent())  # 200

        names = list(stack.components.keys())
        assert names == [
            "embedding-ngram",
            "consolidation",
            "forgetting",
        ], f"Components not in priority order: {names}"

    def test_external_component_default_priority(self, tmp_db):
        """Components without explicit priority default to 200."""

        class ExternalComponent:
            name = "external-test"
            version = "1.0.0"
            required = False
            needs_inference = False

            def attach(self, stack_id, inference=None):
                pass

            def detach(self):
                pass

            def set_inference(self, inference):
                pass

            def set_storage(self, storage):
                pass

            def on_save(self, memory_type, memory_id, memory):
                return None

            def on_search(self, query, results):
                return results

            def on_load(self, context):
                pass

            def on_maintenance(self):
                return {}

        stack = Stack.from_sqlite(
            stack_id="test-ordering",
            db_path=tmp_db,
            components=[],
            enforce_provenance=False,
        )

        stack.add_component(ForgettingComponent())  # 300
        stack.add_component(ExternalComponent())  # 200 (default)
        stack.add_component(EmbeddingComponent())  # 10

        names = list(stack.components.keys())
        assert names == [
            "embedding-ngram",
            "external-test",
            "forgetting",
        ], f"External component not sorted with default priority: {names}"

    def test_dispatch_order_on_save(self, tmp_db):
        """on_save dispatch should follow priority order."""
        call_order = []

        class TrackedComponent:
            required = False
            needs_inference = False
            inference_scope = "none"

            def __init__(self, name, prio):
                self.name = name
                self.version = "1.0.0"
                self.priority = prio

            def attach(self, stack_id, inference=None):
                pass

            def detach(self):
                pass

            def set_inference(self, inference):
                pass

            def set_storage(self, storage):
                pass

            def on_save(self, memory_type, memory_id, memory):
                call_order.append(self.name)
                return None

            def on_search(self, query, results):
                return results

            def on_load(self, context):
                pass

            def on_maintenance(self):
                return {}

        stack = Stack.from_sqlite(
            stack_id="test-dispatch",
            db_path=tmp_db,
            components=[],
            enforce_provenance=False,
        )

        # Add in scrambled order
        stack.add_component(TrackedComponent("third", 300))
        stack.add_component(TrackedComponent("first", 10))
        stack.add_component(TrackedComponent("second", 100))

        stack._dispatch_on_save("episode", "test-id", MagicMock())

        assert call_order == [
            "first",
            "second",
            "third",
        ], f"on_save dispatch order wrong: {call_order}"
