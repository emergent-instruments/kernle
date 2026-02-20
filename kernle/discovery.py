"""
Entry point discovery for kernle components.

Uses importlib.metadata.entry_points() to discover installed plugins,
stacks, models, and stack components registered via pyproject.toml
entry point groups.
"""

import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from kernle.protocols import (
    ENTRY_POINT_GROUP_MODELS,
    ENTRY_POINT_GROUP_PLUGINS,
    ENTRY_POINT_GROUP_STACK_COMPONENTS,
    ENTRY_POINT_GROUP_STACKS,
    ENTRY_POINT_GROUP_STORAGE,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredComponent:
    """Metadata about a discovered entry point component."""

    name: str
    group: str
    module: str
    attr: str
    dist_name: Optional[str] = None
    dist_version: Optional[str] = None
    error: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def qualname(self) -> str:
        """Full qualified reference: 'module:attr'."""
        return f"{self.module}:{self.attr}"


def _get_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    """Get entry points for a group, compatible with Python 3.10+."""
    try:
        return list(importlib.metadata.entry_points(group=group))
    except Exception as exc:
        logger.warning("Failed to read entry points for group '%s': %s", group, exc, exc_info=True)
        return []


def _entry_point_to_component(ep: importlib.metadata.EntryPoint, group: str) -> DiscoveredComponent:
    """Convert an EntryPoint to a DiscoveredComponent."""
    value = (ep.value or "").strip()
    raw_module = getattr(ep, "module", "") or ""
    raw_attr = getattr(ep, "attr", "") or ""

    module = raw_module.strip() if isinstance(raw_module, str) else ""
    attr = raw_attr.strip() if isinstance(raw_attr, str) else ""

    if not module or not attr:
        parts = value.split(":", 1)
        if not module:
            module = parts[0].strip() if parts else ""
        if not attr and len(parts) > 1:
            attr = parts[1].strip()

    # Defensive parse: some entry point metadata can be normalized with
    # optional bracketed suffixes (e.g. "module:Class [extra]").
    if " [" in attr and attr.endswith("]"):
        attr = attr.split(" [", 1)[0].strip()

    dist_name = None
    dist_version = None
    if ep.dist is not None:
        dist_name = ep.dist.name
        dist_version = ep.dist.version

    return DiscoveredComponent(
        name=ep.name,
        group=group,
        module=module,
        attr=attr,
        dist_name=dist_name,
        dist_version=dist_version,
    )


def discover_plugins() -> list[DiscoveredComponent]:
    """Discover installed plugins (kernle.plugins entry point group).

    Returns:
        List of discovered plugin components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_PLUGINS)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_PLUGINS) for ep in eps]


def discover_stacks() -> list[DiscoveredComponent]:
    """Discover installed stack implementations (kernle.stacks entry point group).

    Returns:
        List of discovered stack components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_STACKS)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_STACKS) for ep in eps]


def discover_models() -> list[DiscoveredComponent]:
    """Discover installed model implementations (kernle.models entry point group).

    Returns:
        List of discovered model components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_MODELS)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_MODELS) for ep in eps]


def discover_stack_components() -> list[DiscoveredComponent]:
    """Discover installed stack components (kernle.stack_components entry point group).

    Returns:
        List of discovered stack sub-components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_STACK_COMPONENTS)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_STACK_COMPONENTS) for ep in eps]


def discover_storage() -> list[DiscoveredComponent]:
    """Discover installed storage backends (kernle.storage entry point group).

    Returns:
        List of discovered storage backend components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_STORAGE)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_STORAGE) for ep in eps]


def discover_all() -> dict[str, list[DiscoveredComponent]]:
    """Discover all registered kernle components across all entry point groups.

    Returns:
        Dict keyed by group name, each value a list of DiscoveredComponent.
    """
    return {
        ENTRY_POINT_GROUP_PLUGINS: discover_plugins(),
        ENTRY_POINT_GROUP_STACKS: discover_stacks(),
        ENTRY_POINT_GROUP_MODELS: discover_models(),
        ENTRY_POINT_GROUP_STACK_COMPONENTS: discover_stack_components(),
        ENTRY_POINT_GROUP_STORAGE: discover_storage(),
    }


def load_component(component: DiscoveredComponent) -> Any:
    """Load (import) a discovered component's class/factory.

    Args:
        component: The component to load.

    Returns:
        The loaded class or callable.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute doesn't exist in the module.
    """
    eps = [ep for ep in _get_entry_points(component.group) if ep.name == component.name]
    if component.dist_name:
        dist_eps = [ep for ep in eps if ep.dist is not None and ep.dist.name == component.dist_name]
        if dist_eps:
            eps = dist_eps

    if component.module and component.attr and len(eps) > 1:
        qual_eps = [
            ep
            for ep in eps
            if (getattr(ep, "module", "") or "") == component.module
            and (getattr(ep, "attr", "") or "") == component.attr
        ]
        if qual_eps:
            eps = qual_eps

    if not eps:
        raise ImportError(f"Entry point '{component.name}' not found in group '{component.group}'")
    if len(eps) > 1:
        raise ImportError(
            f"Ambiguous entry point '{component.name}' in group '{component.group}': "
            f"{len(eps)} matches found"
        )
    try:
        return eps[0].load()
    except Exception as exc:
        raise ImportError(
            f"Failed to load entry point '{component.name}' from '{component.qualname}': {exc}"
        ) from exc
