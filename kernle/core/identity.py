"""Identity synthesis and consolidation operations for Kernle."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class IdentityMixin:
    """Identity synthesis and consolidation operations for Kernle."""

    def promote(
        self,
        auto: bool = False,
        min_occurrences: int = 2,
        min_episodes: int = 3,
        confidence: float = 0.7,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Promote recurring patterns from episodes into beliefs.

        Scans recent episodes for recurring lessons and patterns. In auto
        mode, creates beliefs directly. In default mode, returns suggestions
        for the agent to review.

        This is the episodes -> beliefs promotion step. The agent controls
        when and whether promotion happens (SI autonomy principle).

        Args:
            auto: If True, create beliefs automatically. If False, return
                suggestions only (default: False).
            min_occurrences: Minimum times a lesson must appear across
                episodes to be considered for promotion (default: 2).
            min_episodes: Minimum episodes required to run promotion
                (default: 3).
            confidence: Initial confidence for auto-created beliefs
                (default: 0.7). Clamped to 0.1-0.95.
            limit: Maximum episodes to scan (default: 50).

        Returns:
            Dict with promotion results:
            - episodes_scanned: number of episodes analyzed
            - patterns_found: number of recurring patterns detected
            - suggestions: list of {lesson, count, source_episodes, promoted, belief_id}
            - beliefs_created: number of beliefs created (auto mode only)
        """

        confidence = max(0.1, min(0.95, confidence))
        limit = max(1, min(200, limit))
        min_occurrences = max(2, min_occurrences)

        episodes = self._storage.get_episodes(limit=limit)
        episodes = [ep for ep in episodes if ep.strength > 0.0]

        if len(episodes) < min_episodes:
            return {
                "episodes_scanned": len(episodes),
                "patterns_found": 0,
                "suggestions": [],
                "beliefs_created": 0,
                "message": f"Need at least {min_episodes} episodes (found {len(episodes)})",
            }

        # Map lessons to their source episodes
        lesson_sources: Dict[str, List[str]] = {}
        for ep in episodes:
            if ep.lessons:
                for lesson in ep.lessons:
                    # Normalize: strip whitespace, lowercase for matching
                    normalized = lesson.strip()
                    if not normalized:
                        continue
                    if normalized not in lesson_sources:
                        lesson_sources[normalized] = []
                    lesson_sources[normalized].append(ep.id)

        # Find recurring patterns
        recurring = [
            (lesson, ep_ids)
            for lesson, ep_ids in lesson_sources.items()
            if len(ep_ids) >= min_occurrences
        ]
        # Sort by frequency (most common first)
        recurring.sort(key=lambda x: -len(x[1]))

        # Check existing beliefs to avoid duplicates
        existing_beliefs = self._storage.get_beliefs(limit=10000)
        existing_statements = {b.statement.strip().lower() for b in existing_beliefs if b.is_active}

        suggestions = []
        beliefs_created = 0

        for lesson, source_ep_ids in recurring:
            # Skip if a very similar belief already exists
            if lesson.strip().lower() in existing_statements:
                suggestions.append(
                    {
                        "lesson": lesson,
                        "count": len(source_ep_ids),
                        "source_episodes": source_ep_ids[:5],
                        "promoted": False,
                        "skipped": "similar_belief_exists",
                    }
                )
                continue

            suggestion = {
                "lesson": lesson,
                "count": len(source_ep_ids),
                "source_episodes": source_ep_ids[:5],
                "promoted": False,
                "belief_id": None,
            }

            if auto:
                # Create belief with proper provenance
                derived_from = [f"episode:{eid}" for eid in source_ep_ids[:10]]
                belief_id = self.belief(
                    statement=lesson,
                    type="pattern",
                    confidence=confidence,
                    source="promotion",
                    derived_from=derived_from,
                    source_type="consolidation",
                )
                suggestion["promoted"] = True
                suggestion["belief_id"] = belief_id
                beliefs_created += 1
                # Add to existing set to prevent duplicates within same run
                existing_statements.add(lesson.strip().lower())

            suggestions.append(suggestion)

        return {
            "episodes_scanned": len(episodes),
            "patterns_found": len(recurring),
            "suggestions": suggestions[:20],  # Cap output
            "beliefs_created": beliefs_created,
        }

    def synthesize_identity(self) -> Dict[str, Any]:
        """Synthesize identity from memory.

        Combines values, beliefs, goals, and experiences into a coherent
        identity narrative.

        Returns:
            Identity synthesis including narrative and key components
        """
        values = self._storage.get_values(limit=10)
        beliefs = self._storage.get_beliefs(limit=20)
        goals = self._storage.get_goals(status="active", limit=10)
        episodes = self._storage.get_episodes(limit=20)
        drives = self._storage.get_drives()

        # Build narrative from components
        narrative_parts = []

        if values:
            top_value = max(values, key=lambda v: v.priority)
            narrative_parts.append(
                f"I value {top_value.name.lower()} highly: {top_value.statement}"
            )

        if beliefs:
            high_conf = [b for b in beliefs if b.confidence >= 0.8]
            if high_conf:
                narrative_parts.append(f"I believe: {high_conf[0].statement}")

        if goals:
            narrative_parts.append(f"I'm currently working on: {goals[0].title}")

        narrative = " ".join(narrative_parts) if narrative_parts else "Identity still forming."

        # Calculate confidence using the comprehensive scoring method
        confidence = self.get_identity_confidence()

        return {
            "narrative": narrative,
            "core_values": [
                {"name": v.name, "statement": v.statement, "priority": v.priority}
                for v in sorted(values, key=lambda v: v.priority, reverse=True)[:5]
            ],
            "key_beliefs": [
                {"statement": b.statement, "confidence": b.confidence, "foundational": False}
                for b in sorted(beliefs, key=lambda b: b.confidence, reverse=True)[:5]
            ],
            "active_goals": [{"title": g.title, "priority": g.priority} for g in goals[:5]],
            "drives": {d.drive_type: d.intensity for d in drives},
            "significant_episodes": [
                {
                    "objective": e.objective,
                    "outcome": e.outcome_type,
                    "lessons": e.lessons,
                }
                for e in episodes[:5]
            ],
            "confidence": confidence,
        }

    def get_identity_confidence(self) -> float:
        """Get overall identity confidence score.

        Calculates identity coherence based on:
        - Core values (20%): Having defined principles
        - Beliefs (20%): Both count and confidence quality
        - Goals (15%): Having direction and purpose
        - Episodes (20%): Experience count and reflection (lessons) rate
        - Drives (15%): Understanding intrinsic motivations
        - Relationships (10%): Modeling connections to others

        Returns:
            Confidence score (0.0-1.0) based on identity completeness and quality
        """
        # Get identity data
        values = self._storage.get_values(limit=10)
        beliefs = self._storage.get_beliefs(limit=20)
        goals = self._storage.get_goals(status="active", limit=10)
        episodes = self._storage.get_episodes(limit=50)
        drives = self._storage.get_drives()
        relationships = self._storage.get_relationships()

        # Values (20%): quantity x quality (priority)
        # Ideal: 3-5 values with high priority
        if values and len(values) > 0:
            value_count_score = min(1.0, len(values) / 5)
            avg_priority = sum(v.priority / 100 for v in values) / len(values)
            value_score = (value_count_score * 0.6 + avg_priority * 0.4) * 0.20
        else:
            value_score = 0.0

        # Beliefs (20%): quantity x quality (confidence)
        # Ideal: 5-10 beliefs with high confidence
        if beliefs and len(beliefs) > 0:
            avg_belief_conf = sum(b.confidence for b in beliefs) / len(beliefs)
            belief_count_score = min(1.0, len(beliefs) / 10)
            belief_score = (belief_count_score * 0.5 + avg_belief_conf * 0.5) * 0.20
        else:
            belief_score = 0.0

        # Goals (15%): having active direction
        # Ideal: 2-5 active goals
        goal_score = min(1.0, len(goals) / 5) * 0.15

        # Episodes (20%): experience x reflection
        # Ideal: 10-20 episodes with lessons extracted
        if episodes and len(episodes) > 0:
            with_lessons = sum(1 for e in episodes if e.lessons)
            lesson_rate = with_lessons / len(episodes)
            episode_count_score = min(1.0, len(episodes) / 20)
            episode_score = (episode_count_score * 0.5 + lesson_rate * 0.5) * 0.20
        else:
            episode_score = 0.0

        # Drives (15%): understanding motivations
        # Ideal: 2-3 drives defined (curiosity, growth, connection, etc.)
        drive_score = min(1.0, len(drives) / 3) * 0.15

        # Relationships (10%): modeling connections
        # Ideal: 3-5 key relationships tracked
        relationship_score = min(1.0, len(relationships) / 5) * 0.10

        total = (
            value_score
            + belief_score
            + goal_score
            + episode_score
            + drive_score
            + relationship_score
        )

        return round(total, 3)

    def detect_identity_drift(self, days: int = 30) -> Dict[str, Any]:
        """Detect changes in identity over time.

        Args:
            days: Number of days to analyze

        Returns:
            Drift analysis including changed values and evolved beliefs
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)

        # Get recent additions
        recent_episodes = self._storage.get_episodes(limit=50, since=since)

        # Simple drift detection based on episode count and themes
        drift_score = min(1.0, len(recent_episodes) / 20) * 0.5

        return {
            "period_days": days,
            "drift_score": drift_score,
            "changed_values": [],  # Would need historical comparison
            "evolved_beliefs": [],
            "new_experiences": [
                {
                    "objective": e.objective,
                    "outcome": e.outcome_type,
                    "lessons": e.lessons,
                    "date": e.created_at.strftime("%Y-%m-%d") if e.created_at else "",
                }
                for e in recent_episodes[:5]
            ],
        }

    def consolidate_epoch_closing(self, epoch_id: str) -> Dict[str, Any]:
        """Orchestrate full epoch-closing consolidation.

        A deeper consolidation sequence triggered when closing an epoch.
        Produces scaffold prompts for six steps of reflection.

        Args:
            epoch_id: ID of the epoch being closed

        Returns:
            Structured scaffold with all six epoch-closing steps
        """
        from kernle.features.consolidation import build_epoch_closing_scaffold

        epoch_id = self._validate_string_input(epoch_id, "epoch_id", 100)
        return build_epoch_closing_scaffold(self, epoch_id)
