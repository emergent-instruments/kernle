"""Storage backend factory via entry point discovery."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_storage(
    stack_id: str = "default",
    backend: str = "sqlite",
    **kwargs: Any,
) -> Any:
    """Create a storage backend by name via entry point discovery.

    Args:
        stack_id: Stack identifier for the storage instance.
        backend: Name of the storage backend entry point (default: "sqlite").
        **kwargs: Additional keyword arguments passed to the backend constructor.

    Returns:
        A storage backend instance implementing the Storage protocol.

    Raises:
        ValueError: If the backend name is not found in registered entry points.
        TypeError: If the created instance lacks required Storage protocol methods.
    """
    from kernle.discovery import discover_storage, load_component

    backends = discover_storage()
    match = next((b for b in backends if b.name == backend), None)

    if match is None:
        available = sorted(b.name for b in backends)
        raise ValueError(
            f"Unknown storage backend '{backend}'. "
            f"Available backends: {', '.join(available) or '(none registered)'}"
        )

    cls = load_component(match)
    instance = cls(stack_id=stack_id, **kwargs)

    # Verify the instance has core Storage protocol methods
    required = ("get_episodes", "get_beliefs", "get_stats", "save_raw")
    missing = [m for m in required if not callable(getattr(instance, m, None))]
    if missing:
        raise TypeError(
            f"Storage backend '{backend}' ({cls.__name__}) is missing required "
            f"methods: {', '.join(missing)}. Ensure it implements the Storage protocol."
        )

    return instance
