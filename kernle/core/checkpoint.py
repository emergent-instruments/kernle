"""Checkpoint operations for Kernle."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kernle.logging_config import log_checkpoint
from kernle.storage import Episode

logger = logging.getLogger(__name__)


class CheckpointMixin:
    """Checkpoint save/load/clear operations for Kernle."""

    def checkpoint(
        self,
        task: str,
        pending: Optional[list[str]] = None,
        context: Optional[str] = None,
        sync: Optional[bool] = None,
    ) -> dict:
        """Save current working state.

        If auto_sync is enabled (or sync=True), pushes local changes to remote
        after saving the checkpoint locally.

        Args:
            task: Description of the current task/state
            pending: List of pending items to continue later
            context: Additional context about the state
            sync: Override auto_sync setting. If None, uses self.auto_sync.

        Returns:
            Dict containing the checkpoint data
        """
        checkpoint_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stack_id": self.stack_id,
            "current_task": task,
            "pending": pending or [],
            "context": context,
        }

        # Save locally with proper error handling
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot create checkpoint directory: {e}", exc_info=True)
            raise ValueError(f"Cannot create checkpoint directory: {e}")

        checkpoint_file = self.checkpoint_dir / f"{self.stack_id}.json"

        existing = []
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except (json.JSONDecodeError, OSError, PermissionError) as e:
                logger.warning(f"Could not load existing checkpoint: {e}", exc_info=True)
                existing = []

        existing.append(checkpoint_data)
        existing = existing[-10:]  # Keep last 10

        try:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot save checkpoint: {e}", exc_info=True)
            raise ValueError(f"Cannot save checkpoint: {e}")

        # Also save as episode
        try:
            episode = Episode(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                objective=f"[CHECKPOINT] {self._validate_string_input(task, 'task', 500)}",
                outcome=self._validate_string_input(
                    context or "Working state checkpoint", "context", 1000
                ),
                outcome_type="partial",
                lessons=pending or [],
                tags=["checkpoint", "working_state"],
                source_type="observation",
                source_entity="kernle:checkpoint",
                created_at=datetime.now(timezone.utc),
            )
            self._write_backend.save_episode(episode)
        except (sqlite3.Error, OSError, IOError) as e:
            logger.warning(
                f"Failed to save checkpoint to database: {e}",
                extra={"operation": "checkpoint_episode_save", "error_type": type(e).__name__},
                exc_info=True,
            )
            if self._strict:
                raise

        # Auto-export boot file on checkpoint (keeps boot.md in sync)
        try:
            self._export_boot_file()
        except (OSError, IOError) as e:
            logger.warning(
                f"Failed to export boot file on checkpoint: {e}",
                extra={"operation": "checkpoint_boot_export", "error_type": type(e).__name__},
                exc_info=True,
            )
            if self._strict:
                raise

        # Sync after checkpoint if enabled
        should_sync = sync if sync is not None else self._auto_sync
        if should_sync:
            sync_result = self._sync_after_checkpoint()
            checkpoint_data["_sync"] = sync_result

        # Log the checkpoint save
        log_checkpoint(
            self.stack_id,
            task=task,
            context_len=len(context or ""),
        )

        return checkpoint_data

    # Maximum checkpoint file size (10MB) to prevent DoS via large files
    MAX_CHECKPOINT_SIZE = 10 * 1024 * 1024

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{self.stack_id}.json"
        if checkpoint_file.exists():
            try:
                # Check file size before loading to prevent DoS
                file_size = checkpoint_file.stat().st_size
                if file_size > self.MAX_CHECKPOINT_SIZE:
                    logger.error(
                        f"Checkpoint file too large ({file_size} bytes, max {self.MAX_CHECKPOINT_SIZE})"
                    )
                    raise ValueError(f"Checkpoint file too large ({file_size} bytes)")

                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoints = json.load(f)
                    if isinstance(checkpoints, list) and checkpoints:
                        return checkpoints[-1]
                    elif isinstance(checkpoints, dict):
                        return checkpoints
            except (json.JSONDecodeError, OSError, PermissionError) as e:
                logger.warning(f"Could not load checkpoint: {e}", exc_info=True)
        return None

    def clear_checkpoint(self) -> bool:
        """Clear local checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{self.stack_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False
