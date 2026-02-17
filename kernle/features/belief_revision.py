"""Belief revision mixin for Kernle.

Provides belief update, contradiction detection, reinforcement,
supersession, and episode-based revision.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kernle.storage import Belief

if TYPE_CHECKING:
    from kernle.core import Kernle

logger = logging.getLogger(__name__)


class BeliefRevisionMixin:
    """Mixin providing belief revision capabilities."""

    def update_belief(
        self: "Kernle",
        belief_id: str,
        confidence: Optional[float] = None,
        is_active: Optional[bool] = None,
    ) -> bool:
        """Update a belief's confidence or deactivate it."""
        # Validate inputs
        belief_id = self._validate_string_input(belief_id, "belief_id", 100)

        # Get beliefs to find matching one (include inactive to allow reactivation)
        beliefs = self._storage.get_beliefs(limit=1000, include_inactive=True)
        existing = None
        for b in beliefs:
            if b.id == belief_id:
                existing = b
                break

        if not existing:
            return False

        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Confidence must be between 0.0 and 1.0")
            existing.confidence = confidence

        if is_active is not None:
            existing.is_active = is_active
            if not is_active:
                existing.deleted = True

        # Use atomic update with optimistic concurrency control
        self._storage.update_belief_atomic(existing)
        return True

    # =========================================================================
    # BELIEF REVISION
    # =========================================================================

    def find_contradictions(
        self: "Kernle",
        belief_statement: str,
        similarity_threshold: float = 0.6,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find beliefs that might contradict a statement.

        Uses inference-based contradiction detection when a model is available,
        otherwise returns an empty list as a safe default.

        Args:
            belief_statement: The statement to check for contradictions
            similarity_threshold: Minimum similarity score (0-1) for related beliefs
            limit: Maximum number of potential contradictions to return

        Returns:
            List of dicts with belief info and contradiction analysis
        """
        inference = self._get_inference()
        if inference is None:
            return []

        return self._find_contradictions_inference(
            belief_statement, inference, similarity_threshold, limit
        )

    def _find_contradictions_inference(
        self: "Kernle",
        belief_statement: str,
        inference,
        similarity_threshold: float = 0.6,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find contradictions via inference service."""
        from kernle.core.inference_utils import parse_inference_json

        # Use embedding search as first filter for candidate beliefs
        search_results = self._storage.search(
            belief_statement,
            limit=limit * 2,
            record_types=["belief"],
        )

        candidates = []
        stmt_lower = belief_statement.lower().strip()
        for result in search_results:
            if result.record_type != "belief":
                continue
            if result.score < similarity_threshold:
                continue
            belief = result.record
            if belief.statement.lower().strip() == stmt_lower:
                continue
            candidates.append(
                {
                    "belief_id": belief.id,
                    "statement": belief.statement,
                    "confidence": belief.confidence,
                    "similarity_score": round(result.score, 2),
                }
            )

        if not candidates:
            return []

        # Build prompt with candidate beliefs
        candidate_list = "\n".join(
            f"- [{c['belief_id']}] {c['statement']}" for c in candidates[:10]
        )
        prompt = (
            "Determine which of these existing beliefs contradict the given statement.\n\n"
            f"Statement: {belief_statement[:500]}\n\n"
            f"Existing beliefs:\n{candidate_list}\n\n"
            'Return JSON: {"contradictions": [{"belief_id": string, '
            '"contradiction_type": "direct_negation"|"comparative_opposition"|"preference_conflict"|"semantic", '
            '"contradiction_confidence": float 0-1, '
            '"explanation": string}]}\n'
            "Only include genuine contradictions with confidence >= 0.6."
        )

        try:
            raw = inference.infer(
                prompt=prompt,
                system="You are a contradiction detection system. Return only valid JSON.",
            )
        except Exception:
            logger.debug("Contradiction inference call failed", exc_info=True)
            return []

        result = parse_inference_json(
            raw,
            required_fields=["contradictions"],
            fallback={"contradictions": []},
            logger=logger,
        )

        if result.fallback_used:
            return []

        # Filter and enrich results
        candidate_map = {c["belief_id"]: c for c in candidates}
        contradictions = []
        for item in result.data.get("contradictions", []):
            bid = item.get("belief_id", "")
            conf = float(item.get("contradiction_confidence", 0))
            if conf < 0.6:
                continue
            candidate = candidate_map.get(bid)
            if not candidate:
                continue
            contradictions.append(
                {
                    "belief_id": bid,
                    "statement": candidate["statement"],
                    "confidence": candidate["confidence"],
                    "contradiction_type": item.get("contradiction_type", "semantic"),
                    "contradiction_confidence": round(conf, 2),
                    "explanation": item.get("explanation", ""),
                    "semantic_similarity": candidate["similarity_score"],
                }
            )

        contradictions.sort(key=lambda x: x["contradiction_confidence"], reverse=True)
        return contradictions[:limit]

    def find_semantic_contradictions(
        self: "Kernle",
        belief: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find beliefs that are semantically similar but may contradict.

        Uses inference-based contradiction detection when a model is available,
        otherwise returns an empty list as a safe default.

        Args:
            belief: The belief statement to check for contradictions
            similarity_threshold: Minimum similarity score (0-1) for related beliefs.
            limit: Maximum number of potential contradictions to return

        Returns:
            List of dicts with contradiction analysis
        """
        inference = self._get_inference()
        if inference is None:
            return []
        return self._find_contradictions_inference(belief, inference, similarity_threshold, limit)

    def reinforce_belief(
        self: "Kernle",
        belief_id: str,
        evidence_source: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """Increase reinforcement count when a belief is confirmed.

        Also slightly increases confidence (with diminishing returns).

        Args:
            belief_id: ID of the belief to reinforce
            evidence_source: What triggered this reinforcement (e.g., "episode:abc123")
            reason: Human-readable reason for reinforcement

        Returns:
            True if reinforced, False if belief not found
        """
        belief_id = self._validate_string_input(belief_id, "belief_id", 100)

        # Get the belief (include inactive to allow reinforcing superseded beliefs back)
        beliefs = self._storage.get_beliefs(limit=1000, include_inactive=True)
        existing = None
        for b in beliefs:
            if b.id == belief_id:
                existing = b
                break

        if not existing:
            return False

        # Store old confidence BEFORE modification for accurate history tracking
        old_confidence = existing.confidence

        # Increment reinforcement count first
        existing.times_reinforced += 1

        # Slightly increase confidence (diminishing returns)
        # Each reinforcement adds less confidence, capped at 0.99
        # Use (times_reinforced) which is already incremented, so first reinforcement uses 1
        confidence_boost = 0.05 * (1.0 / (1 + existing.times_reinforced * 0.1))
        room_to_grow = max(0.0, 0.99 - existing.confidence)  # Prevent negative when > 0.99
        existing.confidence = max(
            0.0, min(0.99, existing.confidence + room_to_grow * confidence_boost)
        )

        # Update confidence history with accurate old/new values
        history_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old": round(old_confidence, 3),
            "new": round(existing.confidence, 3),
            "reason": reason or f"Reinforced (count: {existing.times_reinforced})",
        }
        if evidence_source:
            history_entry["evidence_source"] = evidence_source

        history = existing.confidence_history or []
        history.append(history_entry)
        existing.confidence_history = history[-20:]  # Keep last 20 entries

        # Track supporting evidence in source_episodes
        if evidence_source and evidence_source.startswith("episode:"):
            existing.source_episodes = existing.source_episodes or []
            if evidence_source not in existing.source_episodes:
                existing.source_episodes.append(evidence_source)

        existing.last_verified = datetime.now(timezone.utc)
        existing.verification_count += 1

        # Use atomic update with optimistic concurrency control
        self._storage.update_belief_atomic(existing)
        return True

    def supersede_belief(
        self: "Kernle",
        old_id: str,
        new_statement: str,
        confidence: float = 0.8,
        reason: Optional[str] = None,
    ) -> str:
        """Replace an old belief with a new one.

        .. deprecated:: 0.14.0
            Use :meth:`revise_belief` instead. This method delegates to
            ``revise_belief`` and exists only for backward compatibility.
        """
        return self.revise_belief(old_id, new_statement, confidence, reason)

    def revise_belief(
        self: "Kernle",
        old_id: str,
        new_statement: str,
        confidence: float = 0.8,
        reason: Optional[str] = None,
    ) -> str:
        """Replace an old belief with a new one, tracked via audit log.

        Creates a new active belief and deactivates the old one. The revision
        relationship is recorded in the audit log (not via supersession chain
        fields). The new belief's ``derived_from`` links back to the old one.

        Args:
            old_id: ID of the belief being revised
            new_statement: The new belief statement
            confidence: Confidence in the new belief (clamped to 0.0-1.0)
            reason: Optional reason for the revision

        Returns:
            ID of the new belief

        Raises:
            ValueError: If old belief not found
        """
        old_id = self._validate_string_input(old_id, "old_id", 100)
        new_statement = self._validate_string_input(new_statement, "new_statement", 2000)

        # Get the old belief
        beliefs = self._storage.get_beliefs(limit=1000, include_inactive=True)
        old_belief = None
        for b in beliefs:
            if b.id == old_id:
                old_belief = b
                break

        if not old_belief:
            raise ValueError(f"Belief {old_id} not found")

        # Create the new belief — no supersession chain fields
        confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
        new_id = str(uuid.uuid4())
        new_belief = Belief(
            id=new_id,
            stack_id=self.stack_id,
            statement=new_statement,
            belief_type=old_belief.belief_type,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            source_type="inference",
            supersedes=None,
            superseded_by=None,
            times_reinforced=0,
            is_active=True,
            # Inherit source episodes from old belief
            source_episodes=old_belief.source_episodes,
            derived_from=[f"belief:{old_id}"],
            confidence_history=[
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "old": 0.0,
                    "new": confidence,
                    "reason": reason or f"Revised belief {old_id[:8]}",
                }
            ],
        )
        self._write_backend.save_belief(new_belief)

        # Deactivate the old belief — no superseded_by chain field
        old_belief.is_active = False

        # Add to confidence history
        hist = old_belief.confidence_history or []
        hist.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old": old_belief.confidence,
                "new": old_belief.confidence,
                "reason": f"Revised to belief {new_id[:8]}: {reason or 'no reason given'}",
            }
        )
        old_belief.confidence_history = hist[-20:]
        # Use atomic update with optimistic concurrency control
        self._storage.update_belief_atomic(old_belief)

        # Record revision in audit log (not chain fields)
        self._storage.log_belief_revision(
            old_id=old_id,
            new_id=new_id,
            reason=reason,
            actor=f"core:{getattr(self, 'core_id', 'unknown')}",
        )

        return new_id

    _EMPTY_REVISION_RESULT = {
        "reinforced": [],
        "contradicted": [],
        "suggested_new": [],
    }

    def revise_beliefs_from_episode(self: "Kernle", episode_id: str) -> Dict[str, Any]:
        """Analyze an episode and update relevant beliefs.

        Uses inference-based analysis when a model is available,
        otherwise returns an empty result as a safe default.

        Args:
            episode_id: ID of the episode to analyze

        Returns:
            Dict with keys: reinforced, contradicted, suggested_new
        """
        episode_id = self._validate_string_input(episode_id, "episode_id", 100)

        # Get the episode (needed for all paths)
        episode = self._storage.get_episode(episode_id)
        if not episode:
            return {
                "error": "Episode not found",
                **self._EMPTY_REVISION_RESULT,
            }

        inference = self._get_inference()
        if inference is None:
            return {"episode_id": episode_id, **self._EMPTY_REVISION_RESULT}

        return self._revise_beliefs_inference(episode_id, episode, inference)

    def _revise_beliefs_inference(
        self: "Kernle", episode_id: str, episode, inference
    ) -> Dict[str, Any]:
        """Revise beliefs from episode using inference service."""
        from kernle.core.inference_utils import parse_inference_json

        result = {
            "episode_id": episode_id,
            "reinforced": [],
            "contradicted": [],
            "suggested_new": [],
        }

        # Build evidence text from episode
        evidence_parts = [episode.objective, episode.outcome]
        if episode.lessons:
            evidence_parts.extend(episode.lessons)
        evidence_text = " ".join(evidence_parts)

        # Get active beliefs
        beliefs = self._storage.get_beliefs(limit=500)
        if not beliefs:
            return result

        belief_list = "\n".join(
            f"- [{b.id}] {b.statement} (confidence: {b.confidence:.1f})" for b in beliefs[:20]
        )

        prompt = (
            "Analyze how this episode relates to the existing beliefs.\n\n"
            f"Episode:\n"
            f"  Objective: {episode.objective}\n"
            f"  Outcome: {episode.outcome}\n"
            f"  Outcome type: {episode.outcome_type or 'unknown'}\n"
            f"  Lessons: {', '.join(episode.lessons or [])}\n\n"
            f"Existing beliefs:\n{belief_list}\n\n"
            'Return JSON: {"reinforced": [{"belief_id": string, "reason": string}], '
            '"contradicted": [{"belief_id": string, "reason": string}], '
            '"suggested_new": [{"statement": string, "confidence": float 0-1}]}'
        )

        try:
            raw = inference.infer(
                prompt=prompt,
                system="You are a belief revision system. Return only valid JSON.",
            )
        except Exception:
            logger.debug("Belief revision inference call failed", exc_info=True)
            return result

        parsed = parse_inference_json(
            raw,
            required_fields=["reinforced", "contradicted", "suggested_new"],
            fallback={"reinforced": [], "contradicted": [], "suggested_new": []},
            logger=logger,
        )

        if parsed.fallback_used:
            return result

        # Map belief IDs for validation
        belief_map = {b.id: b for b in beliefs}

        for item in parsed.data.get("reinforced", []):
            bid = item.get("belief_id", "")
            if bid in belief_map:
                self.reinforce_belief(
                    bid,
                    evidence_source=f"episode:{episode_id}",
                    reason=item.get("reason", ""),
                )
                result["reinforced"].append(
                    {
                        "belief_id": bid,
                        "statement": belief_map[bid].statement,
                        "evidence_source": f"episode:{episode_id}",
                    }
                )

        for item in parsed.data.get("contradicted", []):
            bid = item.get("belief_id", "")
            if bid in belief_map:
                result["contradicted"].append(
                    {
                        "belief_id": bid,
                        "statement": belief_map[bid].statement,
                        "evidence": evidence_text[:200],
                    }
                )

        for item in parsed.data.get("suggested_new", []):
            stmt = item.get("statement", "")
            if stmt:
                result["suggested_new"].append(
                    {
                        "statement": stmt,
                        "source_episode": episode_id,
                        "suggested_confidence": float(item.get("confidence", 0.6)),
                    }
                )

        return result

    def get_belief_history(self: "Kernle", belief_id: str) -> List[Dict[str, Any]]:
        """Get the revision history for a belief.

        Uses a dual-source strategy:
        1. **Primary**: Audit log entries (``belief.revised`` / ``belief.deactivated``)
        2. **Fallback**: Legacy supersession chain walk (for pre-v0.14 data)

        Args:
            belief_id: ID of the belief to trace

        Returns:
            List of beliefs in chronological order, with revision metadata
        """
        belief_id = self._validate_string_input(belief_id, "belief_id", 100)

        # Get all beliefs including inactive ones
        all_beliefs = self._storage.get_beliefs(limit=1000, include_inactive=True)
        belief_map = {b.id: b for b in all_beliefs}

        if belief_id not in belief_map:
            return []

        # --- Strategy 1: Audit log (primary) ---
        audit_history = self._get_belief_history_from_audit(belief_id, belief_map)
        if audit_history:
            return audit_history

        # --- Strategy 2: Legacy chain walk (fallback for pre-v0.14 data) ---
        belief = belief_map[belief_id]
        if belief.superseded_by or belief.supersedes:
            return self._walk_chain_legacy(belief_id, belief_map)

        # Single belief, no revisions
        return [self._belief_to_history_entry(belief, is_current=True)]

    def _get_belief_history_from_audit(
        self: "Kernle",
        belief_id: str,
        belief_map: Dict[str, "Belief"],
    ) -> List[Dict[str, Any]]:
        """Build belief history from audit log entries.

        Traces the full revision chain by following ``belief.deactivated``
        and ``belief.revised`` audit entries.

        Returns empty list if no audit entries found (caller should try
        legacy chain walk).
        """
        # Collect all belief IDs in the revision chain via audit log
        chain_ids = {belief_id}
        frontier = {belief_id}

        while frontier:
            next_frontier: set = set()
            for bid in frontier:
                # Find revisions that reference this belief
                deactivated = self._storage.get_audit_log(
                    memory_id=bid, operation="belief.deactivated"
                )
                for entry in deactivated:
                    details = entry.get("details") or {}
                    if isinstance(details, str):
                        import json

                        details = json.loads(details)
                    trigger_id = details.get("trigger_id")
                    if trigger_id and trigger_id not in chain_ids:
                        chain_ids.add(trigger_id)
                        next_frontier.add(trigger_id)

                revised = self._storage.get_audit_log(memory_id=bid, operation="belief.revised")
                for entry in revised:
                    details = entry.get("details") or {}
                    if isinstance(details, str):
                        import json

                        details = json.loads(details)
                    trigger_id = details.get("trigger_id")
                    if trigger_id and trigger_id not in chain_ids:
                        chain_ids.add(trigger_id)
                        next_frontier.add(trigger_id)

            frontier = next_frontier

        # If we only found the original belief ID (no audit trail), return empty
        if len(chain_ids) <= 1:
            # Check if there are any audit entries at all for this belief
            any_entries = self._storage.get_audit_log(
                memory_id=belief_id, operation="belief.deactivated"
            ) or self._storage.get_audit_log(memory_id=belief_id, operation="belief.revised")
            if not any_entries:
                return []

        # Build history entries sorted by creation date
        history = []
        for bid in chain_ids:
            if bid in belief_map:
                belief = belief_map[bid]
                entry = self._belief_to_history_entry(belief, is_current=(bid == belief_id))
                history.append(entry)

        # Sort by created_at (chronological)
        history.sort(key=lambda h: h.get("created_at") or "")
        return history

    def _walk_chain_legacy(
        self: "Kernle",
        belief_id: str,
        belief_map: Dict[str, "Belief"],
    ) -> List[Dict[str, Any]]:
        """Walk the supersession chain for pre-v0.14 data (fallback)."""
        history = []
        visited: set = set()

        # Walk backwards to find the root
        back_visited: set = set()

        def walk_back(bid: str) -> Optional[str]:
            if bid in back_visited or bid not in belief_map:
                return None
            back_visited.add(bid)
            belief = belief_map[bid]
            if belief.supersedes and belief.supersedes in belief_map:
                return belief.supersedes
            return None

        root_id = belief_id
        while True:
            prev = walk_back(root_id)
            if prev:
                root_id = prev
            else:
                break

        # Walk forward from root
        current_id: Optional[str] = root_id
        while current_id and current_id not in visited and current_id in belief_map:
            visited.add(current_id)
            belief = belief_map[current_id]
            entry = self._belief_to_history_entry(belief, is_current=(current_id == belief_id))

            # Add supersession reason from confidence history
            if belief.confidence_history:
                for h in reversed(belief.confidence_history):
                    reason = h.get("reason", "")
                    if "Superseded" in reason or "Revised" in reason:
                        entry["supersession_reason"] = reason
                        break

            history.append(entry)
            current_id = belief.superseded_by

        return history

    @staticmethod
    def _belief_to_history_entry(belief: "Belief", *, is_current: bool = False) -> Dict[str, Any]:
        """Convert a Belief to a history entry dict."""
        return {
            "id": belief.id,
            "statement": belief.statement,
            "confidence": belief.confidence,
            "times_reinforced": belief.times_reinforced,
            "is_active": belief.is_active,
            "is_current": is_current,
            "created_at": belief.created_at.isoformat() if belief.created_at else None,
            "supersedes": belief.supersedes,
            "superseded_by": belief.superseded_by,
        }
