"""Kernle class — main interface for memory operations.

This module defines the Kernle class skeleton, which inherits from
all extraction mixins and the existing feature mixins.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

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
from kernle.utils import get_kernle_home

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

    All writes route through the Stack, which enforces maintenance mode
    blocking and runs component hooks (emotional tagging, etc.) regardless
    of the ``strict`` setting.

    The ``strict`` parameter controls:
    - **Provenance validation** — ``strict=True`` requires ``derived_from``
      on higher-tier memories and validates the provenance hierarchy.
    - **Save-time lint** — ``strict=True`` enables lint checks that may
      redirect low-quality writes to suggestions.
    - **Error propagation** — ``strict=True`` re-raises sync/checkpoint
      errors instead of logging and continuing.

    For full enforcement with Entity-level orchestration, use :attr:`entity`.

    Examples:
        # Non-strict mode — writes through stack, no provenance enforcement
        k = Kernle(stack_id="my_agent", strict=False)

        # Strict mode (default) — full enforcement
        k = Kernle(stack_id="my_agent", strict=True)

        # Recommended: use Entity directly for full orchestration
        from kernle import Entity
        e = Entity(core_id="my_agent")
    """

    def __init__(
        self,
        stack_id: Optional[str] = None,
        storage: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
        strict: bool = True,
    ):
        """Initialize Kernle.

        Args:
            stack_id: Unique identifier for the agent
            storage: Optional storage backend. If None, creates SQLiteStorage.
                Any object implementing the Storage protocol is accepted.
            checkpoint_dir: Directory for local checkpoints
            strict: If True, enable provenance validation, save-time lint,
                and strict error propagation. All writes route through Stack
                regardless of this setting.
        """
        self.stack_id = self._validate_stack_id(
            stack_id or os.environ.get("KERNLE_STACK_ID", "default")
        )
        self.checkpoint_dir = self._validate_checkpoint_dir(
            checkpoint_dir or get_kernle_home() / "checkpoints"
        )

        # Single storage instance — shared between Kernle and Stack
        if storage is not None:
            self._storage = storage
        else:
            from kernle.storage.factory import create_storage

            backend = os.environ.get("KERNLE_STORAGE_BACKEND", "sqlite")
            self._storage = create_storage(backend=backend, stack_id=self.stack_id)

        # Controls Stack enforcement (enforce_provenance, lint_on_save) and
        # error propagation in checkpoint/sync mixins.
        self._strict = strict

        # Eager Stack creation — shares the same storage instance
        from kernle.stack import Stack

        self._stack = Stack(
            stack_id=self.stack_id,
            storage=self._storage,
            enforce_provenance=strict,
            lint_on_save=strict,
        )
        self._stack.on_attach(self.stack_id)  # INITIALIZING → ACTIVE

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

        logger.debug(
            f"Kernle initialized with storage: {type(self._storage).__name__}, "
            f"auto_sync: {self._auto_sync}, strict: {self._strict}"
        )

    @property
    def _write_backend(self):
        """Return the write target for memory operations.

        Always returns the Stack, which handles enforcement based on its
        enforce_provenance setting. In strict mode, Stack validates provenance
        and runs component hooks. In non-strict mode, Stack still runs
        component hooks but skips provenance validation.
        """
        return self._stack

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
    def storage(self) -> Any:
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
        The eager Stack is automatically attached on first access.

        Returns:
            Entity: The CoreProtocol implementation.
        """
        if not hasattr(self, "_entity"):
            from kernle.entity import Entity

            self._entity = Entity(core_id=self.stack_id)
            self._entity.attach_stack(self._stack, alias="default", set_active=True)
        return self._entity

    @property
    def stack(self):
        """Access the Stack (StackProtocol) wrapper.

        The Stack is created eagerly during __init__ and shares the same
        storage instance as Kernle — no double instantiation.

        Returns:
            Stack: The StackProtocol implementation.
        """
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
