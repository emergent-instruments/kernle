"""Kernle class — main interface for memory operations.

This module defines the Kernle class skeleton, which inherits from
all extraction mixins and the existing feature mixins.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from kernle.core.checkpoint import CheckpointMixin
from kernle.core.identity import IdentityMixin
from kernle.core.loader import LoaderMixin
from kernle.core.managers import ManagersMixin
from kernle.core.serializers import SerializersMixin
from kernle.core.sync import SyncMixin
from kernle.core.validation import ValidationMixin
from kernle.core.writers import WritersMixin
from kernle.features import (
    AnxietyMixin,
    BeliefRevisionMixin,
    ConsolidationMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    PlaybookMixin,
    SuggestionsMixin,
    TrustMixin,
)
from kernle.storage import SQLiteStorage
from kernle.utils import get_kernle_home

if TYPE_CHECKING:
    from kernle.storage import Storage as StorageProtocol

logger = logging.getLogger(__name__)


class Kernle(
    LoaderMixin,
    WritersMixin,
    SerializersMixin,
    CheckpointMixin,
    IdentityMixin,
    ManagersMixin,
    SyncMixin,
    ValidationMixin,
    # Existing feature mixins:
    AnxietyMixin,
    BeliefRevisionMixin,
    ConsolidationMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    PlaybookMixin,
    SuggestionsMixin,
    TrustMixin,
):
    """Main interface for Kernle memory operations.

    This is the **legacy compatibility API**. Write methods (episode, belief,
    value, goal, note, drive, relationship, raw) write directly to the storage
    backend and do NOT enforce provenance hierarchy or maintenance mode.

    For enforced memory operations, use :attr:`entity` which routes writes
    through the stack with full provenance validation.

    Use ``strict=True`` to route all writes through the stack, which enforces
    maintenance mode blocking, provenance hierarchy (when enabled), and stack
    component hooks.

    Examples:
        # Legacy mode (default) — no enforcement
        k = Kernle(stack_id="my_agent")

        # Strict mode — writes routed through stack enforcement
        k = Kernle(stack_id="my_agent", strict=True)

        # Recommended: use Entity directly for full enforcement
        from kernle import Entity
        e = Entity(core_id="my_agent")
    """

    def __init__(
        self,
        stack_id: Optional[str] = None,
        storage: Optional["StorageProtocol"] = None,
        checkpoint_dir: Optional[Path] = None,
        strict: bool = True,
    ):
        """Initialize Kernle.

        Args:
            stack_id: Unique identifier for the agent
            storage: Optional storage backend. If None, auto-detects.
            checkpoint_dir: Directory for local checkpoints
            strict: If True, route writes through stack enforcement layer
                (maintenance mode, provenance validation, component hooks).
                Requires SQLite-backed storage.
        """
        self.stack_id = self._validate_stack_id(
            stack_id or os.environ.get("KERNLE_STACK_ID", "default")
        )
        self.checkpoint_dir = self._validate_checkpoint_dir(
            checkpoint_dir or get_kernle_home() / "checkpoints"
        )

        # Initialize storage
        if storage is not None:
            self._storage = storage
        else:
            self._storage = SQLiteStorage(
                stack_id=self.stack_id,
            )

        # Auto-sync configuration: enabled by default if sync is available
        # Can be disabled via KERNLE_AUTO_SYNC=false
        auto_sync_env = os.environ.get("KERNLE_AUTO_SYNC", "").lower()
        if auto_sync_env in ("false", "0", "no", "off"):
            self._auto_sync = False
        elif auto_sync_env in ("true", "1", "yes", "on"):
            self._auto_sync = True
        else:
            # Default: enabled if storage supports sync (has cloud_storage or is cloud-based)
            self._auto_sync = (
                self._storage.is_online() or self._storage.get_pending_sync_count() > 0
            )

        self._strict = strict

        logger.debug(
            f"Kernle initialized with storage: {type(self._storage).__name__}, "
            f"auto_sync: {self._auto_sync}, strict: {self._strict}"
        )

    @property
    def _write_backend(self):
        """Return the write target for memory operations.

        In strict mode, returns the stack (which enforces maintenance mode,
        provenance validation, and component hooks). In legacy mode, returns
        the storage backend directly (no enforcement).

        Raises:
            ValueError: If strict=True but no SQLite-backed stack is available.
        """
        if self._strict:
            stack = self.stack
            if stack is None:
                raise ValueError(
                    "strict=True requires SQLite-backed storage. "
                    "Use Entity for enforced writes with other storage backends."
                )
            return stack
        return self._storage

    def has_user_content(self) -> bool:
        """Return True if this stack contains any user-created memories."""
        stats = self._storage.get_stats()
        return any(
            stats.get(key, 0) > 0
            for key in (
                "episodes",
                "beliefs",
                "values",
                "goals",
                "notes",
                "drives",
                "relationships",
                "raw",
            )
        )

    @property
    def storage(self) -> "StorageProtocol":
        """Get the storage backend.

        .. deprecated:: 0.4.0
            Direct storage access will be deprecated in a future release.
            Prefer :attr:`entity` and :attr:`stack` for the new architecture.
        """
        return self._storage

    @property
    def entity(self):
        """Access the Entity (CoreProtocol) for new-style composition.

        The Entity is lazily created on first access. It provides the
        coordinator/bus for the new component architecture (v0.4.0+).

        Returns:
            Entity: The CoreProtocol implementation.
        """
        if not hasattr(self, "_entity"):
            from kernle.entity import Entity

            self._entity = Entity(core_id=self.stack_id)
        return self._entity

    @property
    def stack(self):
        """Access the Stack (StackProtocol) wrapper.

        The Stack is lazily created on first access. It wraps a
        *separate* SQLiteStorage pointing at the same database file,
        providing the StackProtocol interface.

        If the Entity has already been created, the stack is automatically
        attached as the active stack.

        Returns:
            Stack: The StackProtocol implementation, or None if the
            underlying storage is not SQLite-based.
        """
        if not hasattr(self, "_stack"):
            from kernle.storage.sqlite import SQLiteStorage as _SQLiteStorage

            if not isinstance(self._storage, _SQLiteStorage):
                return None

            from kernle.stack import Stack

            self._stack = Stack.from_sqlite(
                stack_id=self.stack_id,
                db_path=self._storage.db_path,
                enforce_provenance=self._strict,
            )
            if hasattr(self, "_entity"):
                self._entity.attach_stack(self._stack, alias="default", set_active=True)
            elif self._strict:
                # Strict mode: auto-attach so stack transitions to ACTIVE
                # (without Entity, on_attach is the only way to leave INITIALIZING)
                self._stack.on_attach(self.stack_id)
        return self._stack

    def process(
        self,
        transition: Optional[str] = None,
        force: bool = False,
        allow_no_inference_override: bool = False,
        auto_promote: bool = False,
        batch_size: Optional[int] = None,
    ):
        """Delegate memory processing to :class:`Entity` for deterministic behavior."""
        return self.entity.process(
            transition=transition,
            force=force,
            auto_promote=auto_promote,
            batch_size=batch_size,
        )

    @property
    def client(self):
        """Backwards-compatible access to Supabase client.

        DEPRECATED: Supabase storage has been removed from kernle core.
        Use kernle-cloud for cloud storage functionality.

        Raises:
            ValueError: Always — Supabase storage is no longer bundled.
        """
        raise ValueError(
            "Direct Supabase client access is no longer available. "
            "Supabase storage has been moved to kernle-cloud."
        )

    @property
    def auto_sync(self) -> bool:
        """Whether auto-sync is enabled.

        When enabled:
        - load() will pull remote changes first
        - checkpoint() will push local changes after saving
        """
        return self._auto_sync

    @auto_sync.setter
    def auto_sync(self, value: bool):
        """Enable or disable auto-sync."""
        self._auto_sync = value
