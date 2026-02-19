"""Consolidation stack component.

Provides advanced memory consolidation: cross-domain pattern detection,
belief-to-value promotion, and entity model-to-belief promotion.
When inference is available, synthesizes episodes into thematic patterns;
falls back to keyword counting otherwise.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

# Thresholds for cross-domain pattern detection
CROSS_DOMAIN_MIN_EPISODES = 2
CROSS_DOMAIN_MIN_DOMAINS = 2


class ConsolidationComponent:
    """Consolidation component for memory pattern synthesis.

    Detects cross-domain patterns, identifies belief-to-value promotion
    candidates, and runs consolidation sweeps during maintenance.
    When inference is available, synthesizes themes from episodes;
    falls back to keyword counting without a model.
    """

    name = "consolidation"
    version = "1.0.0"
    required = False
    needs_inference = True
    inference_scope = "capable"
    priority = 200

    CROSS_DOMAIN_ALERT_THRESHOLD = 2
    LESSON_ALERT_THRESHOLD = 6
    CONSOLIDATED_VOLUME_ALERT_THRESHOLD = 10

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
        """Run consolidation: find common lessons across episodes."""
        if self._storage is None:
            logger.debug("ConsolidationComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        # Pattern detection works without inference
        episodes = self._storage.get_episodes(limit=50)
        if len(episodes) < 3:
            return {
                "consolidated": 0,
                "lessons_found": 0,
                "message": "Need at least 3 episodes to consolidate",
            }

        all_lessons: List[str] = []
        for ep in episodes:
            if ep.lessons:
                all_lessons.extend(ep.lessons)

        lesson_counts = Counter(all_lessons)
        common = [lesson for lesson, cnt in lesson_counts.items() if cnt >= 2]

        result: Dict[str, Any] = {
            "consolidated": len(episodes),
            "lessons_found": len(common),
            "common_lessons": common[:5],
        }

        # Cross-domain pattern detection (no inference needed)
        patterns = self._detect_cross_domain_patterns(episodes)
        if patterns:
            result["cross_domain_patterns"] = len(patterns)
        result["alerts"] = self._build_alerts(
            episodes_count=len(episodes), patterns=patterns, common_lessons=common
        )

        # Try inference-based synthesis
        synthesized = self._synthesize_via_inference(episodes)
        if synthesized is not None:
            result["inference_available"] = True
            result["synthesized_themes"] = synthesized
        else:
            if self._inference is None:
                logger.debug("ConsolidationComponent: no inference, skipping synthesis")
            result["inference_available"] = self._inference is not None

        return result

    def _build_alerts(
        self,
        episodes_count: int,
        patterns: List[Dict[str, Any]],
        common_lessons: List[str],
    ) -> List[Dict[str, Any]]:
        """Build maintenance alerts from consolidation heuristics."""
        alerts: List[Dict[str, Any]] = []

        pattern_count = len(patterns or [])
        if pattern_count >= self.CROSS_DOMAIN_ALERT_THRESHOLD:
            alerts.append(
                {
                    "type": "cross_domain_clustering",
                    "severity": "medium",
                    "message": "High cross-domain consolidation clustering detected.",
                    "value": pattern_count,
                }
            )

        if len(common_lessons) >= self.LESSON_ALERT_THRESHOLD:
            alerts.append(
                {
                    "type": "lesson_convergence",
                    "severity": "low",
                    "message": "High number of recurring lessons detected in recent episodes.",
                    "value": len(common_lessons),
                }
            )

        if episodes_count >= self.CONSOLIDATED_VOLUME_ALERT_THRESHOLD:
            alerts.append(
                {
                    "type": "consolidation_volume",
                    "severity": "medium",
                    "message": "Large episode volume processed during consolidation maintenance.",
                    "value": episodes_count,
                }
            )

        return alerts

    # ---- Core Logic ----

    def _synthesize_via_inference(self, episodes: list) -> Optional[List[str]]:
        """Attempt to synthesize themes from episodes using inference.

        Returns a list of theme strings on success, or None if inference
        is unavailable or returns invalid data.
        """
        if self._inference is None:
            return None

        # Build a summary of episodes for the model
        episode_summaries = []
        for ep in episodes[:20]:  # Limit to avoid overly long prompts
            objective = getattr(ep, "objective", "") or ""
            outcome = getattr(ep, "outcome", "") or ""
            lessons = getattr(ep, "lessons", None) or []
            lesson_str = "; ".join(lessons) if lessons else "none"
            episode_summaries.append(
                f"- Objective: {objective}, Outcome: {outcome}, Lessons: {lesson_str}"
            )

        episodes_text = "\n".join(episode_summaries)

        try:
            response = self._inference.infer(
                prompt=(
                    "Analyze these episodes and identify recurring themes and patterns.\n\n"
                    f"{episodes_text}\n\n"
                    "Return a JSON object: "
                    '{"themes": [list of theme strings describing recurring patterns]}'
                ),
                system="You are a pattern analysis system. Return only valid JSON.",
            )
            data = json.loads(response)
            themes = data.get("themes", [])

            if not isinstance(themes, list):
                return None

            return [str(t) for t in themes if isinstance(t, str)][:10]
        except Exception:
            logger.debug(
                "ConsolidationComponent: inference synthesis failed, skipping", exc_info=True
            )
            return None

    def _detect_cross_domain_patterns(self, episodes: list) -> List[Dict[str, Any]]:
        """Detect structural similarities in outcomes across domains."""
        lesson_domain_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for ep in episodes:
            if ep.strength == 0.0:
                continue
            tags = set(ep.tags or [])
            ctx_tags = getattr(ep, "context_tags", None) or []
            all_tags = tags | set(ctx_tags) or {"untagged"}
            otype = ep.outcome_type or "unknown"

            if ep.lessons:
                for lesson in ep.lessons:
                    normalized = lesson.strip().lower()
                    for tag in all_tags:
                        lesson_domain_map[normalized].append(
                            {"domain": tag, "outcome": otype, "episode_id": ep.id}
                        )

        patterns = []
        for lesson, occurrences in lesson_domain_map.items():
            domains_by_outcome: Dict[str, set] = defaultdict(set)
            for occ in occurrences:
                domains_by_outcome[occ["outcome"]].add(occ["domain"])

            for outcome, domains in domains_by_outcome.items():
                if len(domains) >= CROSS_DOMAIN_MIN_DOMAINS:
                    patterns.append(
                        {
                            "lesson": lesson,
                            "outcome": outcome,
                            "domains": sorted(domains),
                        }
                    )

        return patterns
