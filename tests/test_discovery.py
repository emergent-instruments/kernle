"""Tests for kernle.discovery entry point discovery."""

import importlib.metadata
from unittest.mock import MagicMock, patch

import pytest

from kernle.discovery import (
    DiscoveredComponent,
    _entry_point_to_component,
    discover_all,
    discover_models,
    discover_plugins,
    discover_stack_components,
    discover_stacks,
    load_component,
)
from kernle.protocols import (
    ENTRY_POINT_GROUP_MODELS,
    ENTRY_POINT_GROUP_PLUGINS,
    ENTRY_POINT_GROUP_STACK_COMPONENTS,
    ENTRY_POINT_GROUP_STACKS,
)


def _make_entry_point(name: str, value: str, group: str, dist_name: str = "test-pkg"):
    """Create a mock EntryPoint."""
    ep = MagicMock(spec=importlib.metadata.EntryPoint)
    ep.name = name
    ep.value = value
    ep.group = group
    ep.dist = MagicMock()
    ep.dist.name = dist_name
    ep.dist.version = "1.0.0"
    return ep


class TestDiscoveredComponent:
    def test_qualname(self):
        comp = DiscoveredComponent(
            name="test",
            group="kernle.plugins",
            module="test_pkg",
            attr="TestPlugin",
        )
        assert comp.qualname == "test_pkg:TestPlugin"

    def test_qualname_no_attr(self):
        comp = DiscoveredComponent(
            name="test",
            group="kernle.plugins",
            module="test_pkg",
            attr="",
        )
        assert comp.qualname == "test_pkg:"

    def test_defaults(self):
        comp = DiscoveredComponent(
            name="test",
            group="kernle.plugins",
            module="test_pkg",
            attr="TestPlugin",
        )
        assert comp.dist_name is None
        assert comp.dist_version is None
        assert comp.error is None
        assert comp.extras == {}


class TestDiscoverPlugins:
    @patch("kernle.discovery._get_entry_points")
    def test_empty(self, mock_eps):
        mock_eps.return_value = []
        result = discover_plugins()
        assert result == []
        mock_eps.assert_called_once_with(ENTRY_POINT_GROUP_PLUGINS)

    @patch("kernle.discovery._get_entry_points")
    def test_finds_plugins(self, mock_eps):
        mock_eps.return_value = [
            _make_entry_point("analytics", "analytics:AnalyticsPlugin", ENTRY_POINT_GROUP_PLUGINS),
            _make_entry_point(
                "web-search", "kernle_web:WebSearchPlugin", ENTRY_POINT_GROUP_PLUGINS
            ),
        ]
        result = discover_plugins()
        assert len(result) == 2
        assert result[0].name == "analytics"
        assert result[0].module == "analytics"
        assert result[0].attr == "AnalyticsPlugin"
        assert result[0].group == ENTRY_POINT_GROUP_PLUGINS
        assert result[0].dist_name == "test-pkg"
        assert result[1].name == "web-search"
        assert result[1].module == "kernle_web"
        assert result[1].attr == "WebSearchPlugin"


class TestDiscoverStacks:
    @patch("kernle.discovery._get_entry_points")
    def test_empty(self, mock_eps):
        mock_eps.return_value = []
        result = discover_stacks()
        assert result == []
        mock_eps.assert_called_once_with(ENTRY_POINT_GROUP_STACKS)

    @patch("kernle.discovery._get_entry_points")
    def test_finds_stacks(self, mock_eps):
        mock_eps.return_value = [
            _make_entry_point("sqlite", "kernle_stack:Stack", ENTRY_POINT_GROUP_STACKS),
        ]
        result = discover_stacks()
        assert len(result) == 1
        assert result[0].name == "sqlite"
        assert result[0].module == "kernle_stack"
        assert result[0].attr == "Stack"


class TestDiscoverModels:
    @patch("kernle.discovery._get_entry_points")
    def test_empty(self, mock_eps):
        mock_eps.return_value = []
        result = discover_models()
        assert result == []
        mock_eps.assert_called_once_with(ENTRY_POINT_GROUP_MODELS)

    @patch("kernle.discovery._get_entry_points")
    def test_finds_models(self, mock_eps):
        mock_eps.return_value = [
            _make_entry_point(
                "anthropic", "kernle_anthropic:AnthropicModel", ENTRY_POINT_GROUP_MODELS
            ),
            _make_entry_point("ollama", "kernle_ollama:OllamaModel", ENTRY_POINT_GROUP_MODELS),
        ]
        result = discover_models()
        assert len(result) == 2
        assert result[0].name == "anthropic"
        assert result[1].name == "ollama"


class TestDiscoverStackComponents:
    @patch("kernle.discovery._get_entry_points")
    def test_empty(self, mock_eps):
        mock_eps.return_value = []
        result = discover_stack_components()
        assert result == []
        mock_eps.assert_called_once_with(ENTRY_POINT_GROUP_STACK_COMPONENTS)

    @patch("kernle.discovery._get_entry_points")
    def test_finds_components(self, mock_eps):
        mock_eps.return_value = [
            _make_entry_point(
                "forgetting", "kernle_stack:SalienceForgetting", ENTRY_POINT_GROUP_STACK_COMPONENTS
            ),
        ]
        result = discover_stack_components()
        assert len(result) == 1
        assert result[0].name == "forgetting"
        assert result[0].attr == "SalienceForgetting"


class TestEntryPointParsing:
    def test_entry_point_to_component_prefers_module_attr(self):
        ep = MagicMock(spec=importlib.metadata.EntryPoint)
        ep.name = "manual"
        ep.value = "old.module:OldThing"
        ep.group = ENTRY_POINT_GROUP_PLUGINS
        ep.module = "actual.module"
        ep.attr = "ActualThing"
        ep.dist = None

        result = _entry_point_to_component(ep, ENTRY_POINT_GROUP_PLUGINS)

        assert result.module == "actual.module"
        assert result.attr == "ActualThing"

    def test_entry_point_to_component_trims_bracket_suffix(self):
        ep = _make_entry_point(
            "manual",
            "pkg.module:MyThing [extra]",
            ENTRY_POINT_GROUP_PLUGINS,
        )
        result = _entry_point_to_component(ep, ENTRY_POINT_GROUP_PLUGINS)
        assert result.module == "pkg.module"
        assert result.attr == "MyThing"


class TestDiscoverAll:
    @patch("kernle.discovery._get_entry_points")
    def test_returns_all_groups(self, mock_eps):
        mock_eps.return_value = []
        result = discover_all()
        assert ENTRY_POINT_GROUP_PLUGINS in result
        assert ENTRY_POINT_GROUP_STACKS in result
        assert ENTRY_POINT_GROUP_MODELS in result
        assert ENTRY_POINT_GROUP_STACK_COMPONENTS in result
        assert all(v == [] for v in result.values())

    @patch("kernle.discovery._get_entry_points")
    def test_aggregates_results(self, mock_eps):
        def side_effect(group):
            if group == ENTRY_POINT_GROUP_PLUGINS:
                return [_make_entry_point("p1", "pkg:P1", group)]
            if group == ENTRY_POINT_GROUP_MODELS:
                return [
                    _make_entry_point("m1", "pkg:M1", group),
                    _make_entry_point("m2", "pkg:M2", group),
                ]
            return []

        mock_eps.side_effect = side_effect
        result = discover_all()
        assert len(result[ENTRY_POINT_GROUP_PLUGINS]) == 1
        assert len(result[ENTRY_POINT_GROUP_MODELS]) == 2
        assert len(result[ENTRY_POINT_GROUP_STACKS]) == 0
        assert len(result[ENTRY_POINT_GROUP_STACK_COMPONENTS]) == 0

    @patch("kernle.discovery.importlib.metadata.entry_points")
    def test_discover_plugins_handles_entrypoint_metadata_errors(self, mock_entry_points):
        mock_entry_points.side_effect = RuntimeError("metadata unavailable")
        assert discover_plugins() == []


class TestLoadComponentFailure:
    @patch("kernle.discovery._get_entry_points")
    def test_load_component_wraps_load_failure(self, mock_eps):
        ep = _make_entry_point("broken", "pkg:Broken", ENTRY_POINT_GROUP_PLUGINS)
        ep.load.side_effect = ModuleNotFoundError("missing dependency")
        mock_eps.return_value = [ep]

        comp = DiscoveredComponent(
            name="broken",
            group=ENTRY_POINT_GROUP_PLUGINS,
            module="pkg",
            attr="Broken",
        )

        with pytest.raises(ImportError, match="Failed to load entry point 'broken'"):
            load_component(comp)


class TestLoadComponent:
    @patch("kernle.discovery._get_entry_points")
    def test_load_found(self, mock_eps):
        ep = _make_entry_point("test-plugin", "pkg:TestPlugin", ENTRY_POINT_GROUP_PLUGINS)
        ep.load.return_value = "loaded_class"
        mock_eps.return_value = [ep]

        comp = DiscoveredComponent(
            name="test-plugin",
            group=ENTRY_POINT_GROUP_PLUGINS,
            module="pkg",
            attr="TestPlugin",
        )
        result = load_component(comp)
        assert result == "loaded_class"
        ep.load.assert_called_once()

    @patch("kernle.discovery._get_entry_points")
    def test_load_not_found(self, mock_eps):
        mock_eps.return_value = []

        comp = DiscoveredComponent(
            name="nonexistent",
            group=ENTRY_POINT_GROUP_PLUGINS,
            module="pkg",
            attr="Nope",
        )
        with pytest.raises(ImportError, match="Entry point 'nonexistent' not found"):
            load_component(comp)

    @patch("kernle.discovery._get_entry_points")
    def test_load_ambiguous_raises(self, mock_eps):
        ep_a = _make_entry_point(
            "analytics", "analytics:AnalyticsPlugin", ENTRY_POINT_GROUP_PLUGINS, "pkg-a"
        )
        ep_b = _make_entry_point(
            "analytics", "analytics_v2:AnalyticsPlugin", ENTRY_POINT_GROUP_PLUGINS, "pkg-b"
        )
        mock_eps.return_value = [ep_a, ep_b]

        comp = DiscoveredComponent(
            name="analytics",
            group=ENTRY_POINT_GROUP_PLUGINS,
            module="analytics",
            attr="AnalyticsPlugin",
        )

        with pytest.raises(ImportError, match="Ambiguous entry point 'analytics' in group"):
            load_component(comp)

    @patch("kernle.discovery._get_entry_points")
    def test_load_resolves_matching_distribution(self, mock_eps):
        ep_a = _make_entry_point(
            "analytics", "analytics:AnalyticsPlugin", ENTRY_POINT_GROUP_PLUGINS, "pkg-a"
        )
        ep_b = _make_entry_point(
            "analytics", "analytics_v2:AnalyticsPlugin", ENTRY_POINT_GROUP_PLUGINS, "pkg-b"
        )
        ep_b.load.return_value = "loaded_from_pkg_b"
        mock_eps.return_value = [ep_a, ep_b]

        comp = DiscoveredComponent(
            name="analytics",
            group=ENTRY_POINT_GROUP_PLUGINS,
            module="analytics",
            attr="AnalyticsPlugin",
            dist_name="pkg-b",
        )

        assert load_component(comp) == "loaded_from_pkg_b"
        ep_b.load.assert_called_once()


class TestEntryPointNoDist:
    @patch("kernle.discovery._get_entry_points")
    def test_no_dist(self, mock_eps):
        """Entry points without dist info (e.g., editable installs) should work."""
        ep = MagicMock(spec=importlib.metadata.EntryPoint)
        ep.name = "test"
        ep.value = "pkg:Cls"
        ep.group = ENTRY_POINT_GROUP_PLUGINS
        ep.dist = None
        mock_eps.return_value = [ep]

        result = discover_plugins()
        assert len(result) == 1
        assert result[0].dist_name is None
        assert result[0].dist_version is None


class TestLiveDiscovery:
    """Tests against the real entry point registry (no mocking)."""

    def test_discover_plugins_returns_list(self):
        """discover_plugins should return a list (may be empty)."""
        result = discover_plugins()
        assert isinstance(result, list)

    def test_discover_all_returns_dict(self):
        """discover_all should return a dict with all four groups."""
        result = discover_all()
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            ENTRY_POINT_GROUP_PLUGINS,
            ENTRY_POINT_GROUP_STACKS,
            ENTRY_POINT_GROUP_MODELS,
            ENTRY_POINT_GROUP_STACK_COMPONENTS,
        }
