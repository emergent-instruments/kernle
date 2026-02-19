"""Meta-memory stack component.

Provides confidence tracking, time-based decay, and memory provenance
analysis. Contributes uncertainty information to working memory context.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

# Default decay constants
DEFAULT_DECAY_RATE = 0.01
DEFAULT_DECAY_PERIOD_DAYS = 30
DEFAULT_DECAY_FLOOR = 0.5


class MetaMemoryComponent:
    """Meta-memory component for confidence tracking and decay.

    Monitors memory confidence over time, applies time-based decay,
    and surfaces uncertain memories during maintenance.
    """

    name = "metamemory"
    version = "1.0.0"
    required = False
    needs_inference = False
    inference_scope = "fast"
    priority = 110

    def __init__(
        self,
        *,
        decay_rate: float = DEFAULT_DECAY_RATE,
        decay_period_days: int = DEFAULT_DECAY_PERIOD_DAYS,
        decay_floor: float = DEFAULT_DECAY_FLOOR,
    ) -> None:
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._storage: Optional[Any] = None
        self._decay_rate = decay_rate
        self._decay_period_days = decay_period_days
        self._decay_floor = decay_floor

    def attach(self, stack_id: str, inference: Optional[InferenceService] = None) -> None:
        self._stack_id = stack_id
        self._inference = inference

    def detach(self) -> None:
        self._stack_id = None
        self._inference = None
        self._storage = None

    def set_inference(self, inference: Optional[InferenceService]) -> None:
        self._inference = inference

    def set_storage(self, storage: Any) -> None:
        """Called by Stack after attach to provide storage access."""
        self._storage = storage

    # ---- Lifecycle Hooks ----

    def on_save(self, memory_type: str, memory_id: str, memory: Any) -> Any:
        return None

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        return results

    def on_load(self, context: Dict[str, Any]) -> None:
        """Contribute uncertainty info to working memory context."""
        if self._storage is None:
            return
        uncertain = self._get_uncertain_count()
        if uncertain > 0:
            context["metamemory"] = {
                "uncertain_memories": uncertain,
            }

    def on_maintenance(self) -> Dict[str, Any]:
        """Report on memory confidence and decay status."""
        if self._storage is None:
            logger.debug("MetaMemoryComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        uncertain = self._get_uncertain_count()
        return {
            "uncertain_memories": uncertain,
            "decay_rate": self._decay_rate,
            "decay_period_days": self._decay_period_days,
            "decay_floor": self._decay_floor,
        }

    # ---- Core Logic ----

    def get_confidence_with_decay(self, memory: Any, memory_type: str) -> float:
        """Calculate confidence with time-based decay."""
        if getattr(memory, "is_protected", False):
            return getattr(memory, "confidence", 0.8)

        base_confidence = getattr(memory, "confidence", 0.8)
        last_verified = getattr(memory, "last_verified", None)
        if last_verified is None:
            last_verified = getattr(memory, "created_at", None)
        if last_verified is None:
            return base_confidence

        now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if last_verified.tzinfo is None:
            last_verified = last_verified.replace(tzinfo=timezone.utc)

        days_since = (now - last_verified).days
        if days_since <= 0:
            return base_confidence

        decay_periods = days_since / self._decay_period_days
        decay_amount = self._decay_rate * decay_periods
        return max(self._decay_floor, base_confidence - decay_amount)

    def _get_uncertain_count(self, threshold: float = 0.5) -> int:
        """Count memories with confidence below threshold."""
        if self._storage is None:
            return 0
        beliefs = self._storage.get_beliefs(limit=100)
        return sum(1 for b in beliefs if b.confidence < threshold)
