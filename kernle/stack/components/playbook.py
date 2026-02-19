"""Playbook stack component.

Provides procedural memory integration: playbook summaries in working memory,
health reporting during maintenance, and integration with the stack lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)


class PlaybookComponent:
    """Playbook (procedural memory) component.

    Integrates playbooks into the stack lifecycle by providing:
    - Working memory context with playbook summaries
    - Health statistics during maintenance sweeps
    - Playbook-specific search integration (handled by dedicated API)
    """

    # Component metadata - matches the pattern from ForgettingComponent
    name = "playbooks"
    version = "1.0.0"
    required = False
    needs_inference = False
    inference_scope = "none"
    priority = 400  # Lowest priority, runs last

    def __init__(self) -> None:
        """Initialize the playbook component.

        Sets up internal state tracking for stack attachment, inference
        service, and storage backend. All start as None until attach() is called.
        """
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._storage: Optional[Any] = None

    # ---- Component Lifecycle Methods ----

    def attach(self, stack_id: str, inference: Optional[InferenceService] = None) -> None:
        """Attach this component to a stack.

        Args:
            stack_id: The ID of the stack this component belongs to
            inference: Optional inference service (not used by playbooks)
        """
        self._stack_id = stack_id
        self._inference = inference

    def detach(self) -> None:
        """Detach this component from its stack.

        Clears all internal references to stack, inference, and storage.
        """
        self._stack_id = None
        self._inference = None
        self._storage = None

    def set_inference(self, inference: Optional[InferenceService]) -> None:
        """Set or update the inference service.

        Args:
            inference: The inference service to use (not used by playbooks)
        """
        self._inference = inference

    def set_storage(self, storage: Any) -> None:
        """Set the storage backend.

        Called by Stack after attach() to provide storage access.

        Args:
            storage: The storage backend instance
        """
        self._storage = storage

    # ---- Lifecycle Hooks ----

    def on_save(self, memory_type: str, memory_id: str, memory: Any) -> Any:
        """Hook called when a memory is saved.

        Playbooks are saved through the PlaybookMixin's dedicated API
        (kernle.playbook()), not through the generic save path. This hook
        doesn't need to do anything.

        Args:
            memory_type: Type of memory being saved
            memory_id: ID of the memory
            memory: The memory object

        Returns:
            None (no modification)
        """
        return None  # Playbooks don't act on generic saves

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Hook called when a search is performed.

        Playbooks have their own dedicated search API (kernle.search_playbooks(),
        kernle.find_playbook()) so they don't need to modify the generic search
        results. Keep it simple.

        Args:
            query: The search query
            results: Current search results

        Returns:
            Unmodified results list
        """
        return results  # Playbooks have dedicated search API

    def on_load(self, context: Dict[str, Any]) -> None:
        """Hook called when working memory context is loaded.

        Adds a summary of available playbooks to the working memory context.
        Includes:
        - Total playbook count
        - List of recently used playbooks (up to 3)

        Args:
            context: The working memory context dictionary to augment
        """
        if self._storage is None:
            logger.debug("PlaybookComponent: no storage, skipping context")
            return

        try:
            # Load all playbooks (up to 100) to get an accurate count
            # and find the most recently used ones
            playbooks = self._storage.list_playbooks(limit=100)

            # Sort by last_used (most recent first)
            # Playbooks that have never been used (last_used=None) go to the end
            sorted_playbooks = sorted(
                playbooks,
                key=lambda p: p.last_used if p.last_used else p.created_at,
                reverse=True,
            )

            # Build summary of recent playbooks (up to 3)
            recent = []
            for p in sorted_playbooks[:3]:
                recent.append(
                    {
                        "id": p.id,
                        "name": p.name,
                        "mastery": p.mastery_level,
                        "times_used": p.times_used,
                    }
                )

            # Add playbook summary to context
            context["playbooks"] = {
                "total": len(playbooks),
                "recent": recent,
            }

        except AttributeError:
            # Storage backend doesn't support playbooks - that's OK
            logger.debug("PlaybookComponent: storage doesn't support list_playbooks")

    def on_maintenance(self) -> Dict[str, Any]:
        """Hook called during maintenance sweeps.

        Reports playbook health statistics:
        - Total playbook count
        - Number of unused playbooks (times_used == 0)
        - Average mastery level

        Returns:
            Dictionary with health statistics
        """
        if self._storage is None:
            logger.debug("PlaybookComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        try:
            # Load all playbooks to compute statistics
            playbooks = self._storage.list_playbooks(limit=100)

            total_count = len(playbooks)

            # Count unused playbooks (never executed)
            unused_count = sum(1 for p in playbooks if p.times_used == 0)

            # Calculate average mastery level
            # Mastery levels: novice (0), competent (1), proficient (2), expert (3)
            mastery_levels = ["novice", "competent", "proficient", "expert"]
            mastery_values = []

            for p in playbooks:
                try:
                    mastery_index = mastery_levels.index(p.mastery_level)
                    mastery_values.append(mastery_index)
                except (ValueError, AttributeError):
                    # If mastery_level is missing or invalid, treat as novice (0)
                    mastery_values.append(0)

            if mastery_values:
                avg_mastery_value = sum(mastery_values) / len(mastery_values)
                # Convert back to level name (with fractional representation)
                avg_mastery_str = (
                    f"{avg_mastery_value:.2f} ({mastery_levels[int(avg_mastery_value)]})"
                )
            else:
                avg_mastery_str = "N/A"

            return {
                "total": total_count,
                "unused": unused_count,
                "avg_mastery": avg_mastery_str,
            }

        except AttributeError:
            # Storage backend doesn't support playbooks
            logger.debug("PlaybookComponent: storage doesn't support playbook methods")
            return {"skipped": True, "reason": "storage_unsupported"}
