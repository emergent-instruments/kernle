"""Knowledge mapping stack component.

Provides meta-cognition capabilities: knowledge domain mapping,
competence boundary identification, and learning opportunity detection.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)


class KnowledgeComponent:
    """Knowledge mapping component.

    Analyzes beliefs, episodes, and notes to understand domain coverage,
    competence boundaries, and knowledge gaps. When inference is available,
    can generate richer domain analysis.
    """

    name = "knowledge"
    version = "1.0.0"
    required = False
    needs_inference = True
    inference_scope = "none"
    priority = 220

    def __init__(self) -> None:
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._storage: Optional[Any] = None

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
        pass

    def on_maintenance(self) -> Dict[str, Any]:
        """Generate knowledge map summary during maintenance."""
        if self._storage is None:
            logger.debug("KnowledgeComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        domain_stats = self._extract_domains()
        strengths = 0
        weaknesses = 0
        uncertain = 0

        for name, stats in domain_stats.items():
            if name in ("general", "manual", "auto-captured"):
                continue
            confidences = stats["belief_confidences"]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
            total = stats["belief_count"] + stats["episode_count"] + stats["note_count"]
            if total < 2:
                continue
            if avg_conf >= 0.7:
                strengths += 1
            elif avg_conf < 0.5:
                weaknesses += 1
            if stats["belief_count"] > 0 and avg_conf < 0.5:
                uncertain += 1

        result: Dict[str, Any] = {
            "domains_found": len(domain_stats),
            "strength_domains": strengths,
            "weakness_domains": weaknesses,
            "uncertain_areas": uncertain,
        }

        if self._inference is None:
            logger.debug("KnowledgeComponent: no inference, pattern-only analysis")
            result["inference_available"] = False
        else:
            result["inference_available"] = True

        return result

    # ---- Core Logic ----

    def _extract_domains(self) -> Dict[str, Dict[str, Any]]:
        """Extract knowledge domains from tags across memory types."""
        if self._storage is None:
            return {}

        domain_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "belief_count": 0,
                "belief_confidences": [],
                "episode_count": 0,
                "note_count": 0,
                "last_updated": None,
            }
        )

        beliefs = self._storage.get_beliefs(limit=500)
        for belief in beliefs:
            domain = getattr(belief, "belief_type", None) or "general"
            domain_stats[domain]["belief_count"] += 1
            domain_stats[domain]["belief_confidences"].append(belief.confidence)

        episodes = self._storage.get_episodes(limit=500)
        for episode in episodes:
            tags = episode.tags or []
            tags = [
                t
                for t in tags
                if t not in ("checkpoint", "working_state", "auto-captured", "manual")
            ]
            if tags:
                for tag in tags:
                    domain_stats[tag]["episode_count"] += 1
            else:
                domain_stats["general"]["episode_count"] += 1

        notes = self._storage.get_notes(limit=500)
        for note in notes:
            tags = note.tags or []
            if tags:
                for tag in tags:
                    domain_stats[tag]["note_count"] += 1
            else:
                domain_stats["general"]["note_count"] += 1

        return dict(domain_stats)
