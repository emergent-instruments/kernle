"""Tests for stack component discovery and default configuration (#260)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kernle.stack.components import (
    AnxietyComponent,
    BeliefRevisionComponent,
    ConsolidationComponent,
    EmbeddingComponent,
    EmotionalTaggingComponent,
    ForgettingComponent,
    KnowledgeComponent,
    MetaMemoryComponent,
    PlaybookComponent,
    SuggestionComponent,
    TrustComponent,
    get_default_components,
    get_minimal_components,
    load_components_from_discovery,
)

# ---------------------------------------------------------------------------
# get_default_components / get_minimal_components
# ---------------------------------------------------------------------------


class TestGetDefaultComponents:
    def test_returns_11_components(self):
        components = get_default_components()
        assert len(components) == 11

    def test_component_types(self):
        components = get_default_components()
        types = {type(c) for c in components}
        expected = {
            EmbeddingComponent,
            TrustComponent,
            BeliefRevisionComponent,
            ForgettingComponent,
            ConsolidationComponent,
            EmotionalTaggingComponent,
            AnxietyComponent,
            SuggestionComponent,
            MetaMemoryComponent,
            KnowledgeComponent,
            PlaybookComponent,
        }
        assert types == expected

    def test_components_are_new_instances(self):
        """Each call returns fresh instances, not shared singletons."""
        a = get_default_components()
        b = get_default_components()
        for x, y in zip(a, b):
            assert x is not y

    def test_all_have_name_attribute(self):
        for c in get_default_components():
            assert isinstance(c.name, str)
            assert len(c.name) > 0

    def test_embedding_is_required(self):
        components = get_default_components()
        embedding = [c for c in components if isinstance(c, EmbeddingComponent)]
        assert len(embedding) == 1
        assert embedding[0].required is True


class TestGetMinimalComponents:
    def test_returns_1_component(self):
        components = get_minimal_components()
        assert len(components) == 1

    def test_is_embedding(self):
        components = get_minimal_components()
        assert isinstance(components[0], EmbeddingComponent)

    def test_returns_new_instance(self):
        a = get_minimal_components()
        b = get_minimal_components()
        assert a[0] is not b[0]


# ---------------------------------------------------------------------------
# load_components_from_discovery
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    """Minimal stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name: str, value: str, group: str):
        self.name = name
        self.value = value
        self.group = group
        self.dist = None

    def load(self):
        # Return the actual class for built-in components
        mapping = {
            "embedding": EmbeddingComponent,
            "forgetting": ForgettingComponent,
        }
        return mapping[self.name]


class TestLoadComponentsFromDiscovery:
    def test_discovers_and_instantiates(self, monkeypatch):
        """Monkeypatch importlib.metadata.entry_points to return fake eps."""
        fake_eps = [
            _FakeEntryPoint(
                "embedding",
                "kernle.stack.components.embedding:EmbeddingComponent",
                "kernle.stack_components",
            ),
            _FakeEntryPoint(
                "forgetting",
                "kernle.stack.components.forgetting:ForgettingComponent",
                "kernle.stack_components",
            ),
        ]

        monkeypatch.setattr(
            "kernle.discovery.importlib.metadata.entry_points",
            lambda group: fake_eps,
        )

        instances = load_components_from_discovery()
        assert len(instances) == 2
        assert isinstance(instances[0], EmbeddingComponent)
        assert isinstance(instances[1], ForgettingComponent)

    def test_handles_load_failure_gracefully(self, monkeypatch):
        """If one component fails to load, skip it and continue."""

        class _BadEP:
            name = "bad"
            value = "bad.module:BadClass"
            group = "kernle.stack_components"
            dist = None

            def load(self):
                raise ImportError("no such module")

        good_ep = _FakeEntryPoint(
            "embedding",
            "kernle.stack.components.embedding:EmbeddingComponent",
            "kernle.stack_components",
        )

        monkeypatch.setattr(
            "kernle.discovery.importlib.metadata.entry_points",
            lambda group: [_BadEP(), good_ep],
        )

        instances = load_components_from_discovery()
        assert len(instances) == 1
        assert isinstance(instances[0], EmbeddingComponent)

    def test_empty_when_nothing_registered(self, monkeypatch):
        monkeypatch.setattr(
            "kernle.discovery.importlib.metadata.entry_points",
            lambda group: [],
        )
        assert load_components_from_discovery() == []


# ---------------------------------------------------------------------------
# Stack constructor + add_component / set_storage
# ---------------------------------------------------------------------------


class TestStackComponents:
    """Tests for Stack component auto-loading and set_storage."""

    def test_constructor_loads_defaults(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-stack", db_path=tmp_path / "test.db", enforce_provenance=False
        )
        assert len(stack.components) == 11
        # Verify embedding is present
        names = set(stack.components.keys())
        assert "embedding-ngram" in names

    def test_constructor_respects_explicit_components(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        explicit = [EmbeddingComponent()]
        stack = Stack.from_sqlite(
            "test-stack",
            db_path=tmp_path / "test.db",
            components=explicit,
        )
        assert len(stack.components) == 1
        assert "embedding-ngram" in stack.components

    def test_constructor_empty_list_loads_nothing(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-stack",
            db_path=tmp_path / "test.db",
            components=[],
        )
        assert len(stack.components) == 0

    def test_add_component_calls_set_storage(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-stack",
            db_path=tmp_path / "test.db",
            components=[],
        )
        component = AnxietyComponent()
        assert component._storage is None
        stack.add_component(component)
        assert component._storage is not None
        assert component._storage is stack._storage

    def test_add_component_skips_set_storage_if_missing(self, tmp_path):
        """Components without set_storage don't cause errors."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-stack",
            db_path=tmp_path / "test.db",
            components=[],
        )

        class BareComponent:
            name = "bare"
            version = "1.0.0"
            required = False
            needs_inference = False

            def attach(self, stack_id, inference=None):
                pass

            def detach(self):
                pass

            def set_inference(self, inference):
                pass

        component = BareComponent()
        stack.add_component(component)
        assert "bare" in stack.components

    def test_components_receive_storage_on_default_load(self, tmp_path):
        """All default components with set_storage get storage assigned."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-stack", db_path=tmp_path / "test.db", enforce_provenance=False
        )
        for name, component in stack.components.items():
            if hasattr(component, "_storage") and hasattr(component, "set_storage"):
                assert (
                    component._storage is not None
                ), f"Component '{name}' has set_storage but _storage is None"

    def test_remove_component_raises_for_required(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-stack", db_path=tmp_path / "test.db", enforce_provenance=False
        )
        with pytest.raises(ValueError, match="Cannot remove required"):
            stack.remove_component("embedding-ngram")

    def test_remove_optional_component(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "test-stack", db_path=tmp_path / "test.db", enforce_provenance=False
        )
        assert "anxiety" in stack.components
        stack.remove_component("anxiety")
        assert "anxiety" not in stack.components


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestComponentLifecycle:
    def test_full_lifecycle(self, tmp_path):
        """Create stack -> components auto-loaded -> maintenance -> model change."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "lifecycle-test", db_path=tmp_path / "test.db", enforce_provenance=False
        )

        # Components auto-loaded
        assert len(stack.components) == 11

        # Maintenance runs without error
        results = stack.maintenance()
        assert isinstance(results, dict)

        # Model change propagates
        mock_inference = MagicMock()
        stack.on_model_changed(mock_inference)
        for component in stack._components.values():
            assert component._inference is mock_inference

    def test_add_remove_at_runtime(self, tmp_path):
        """Stack with minimal components can add/remove at runtime."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "runtime-test",
            db_path=tmp_path / "test.db",
            components=get_minimal_components(),
        )
        assert len(stack.components) == 1

        # Add a component at runtime
        anxiety = AnxietyComponent()
        stack.add_component(anxiety)
        assert len(stack.components) == 2
        assert "anxiety" in stack.components
        assert anxiety._storage is stack._storage

        # Remove it
        stack.remove_component("anxiety")
        assert len(stack.components) == 1
        assert "anxiety" not in stack.components

    def test_on_attach_propagates_inference(self, tmp_path):
        """on_attach propagates inference to all components."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "attach-test",
            db_path=tmp_path / "test.db",
            components=get_minimal_components(),
        )
        mock_inference = MagicMock()
        stack.on_attach("core-1", inference=mock_inference)

        for component in stack._components.values():
            assert component._inference is mock_inference

    def test_on_detach_clears_inference(self, tmp_path):
        """on_detach clears inference on all components."""
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            "detach-test",
            db_path=tmp_path / "test.db",
            components=get_minimal_components(),
        )
        mock_inference = MagicMock()
        stack.on_attach("core-1", inference=mock_inference)
        stack.on_detach("core-1")

        for component in stack._components.values():
            assert component._inference is None
