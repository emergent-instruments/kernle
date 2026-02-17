"""Anxiety tracking mixin for Kernle.

This module provides memory anxiety tracking - measuring the functional
anxiety of a synthetic intelligence facing finite context and potential
memory loss.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kernle.anxiety_core import (
    SEVEN_DIM_WEIGHTS,
    apply_hysteresis,
    compute_consolidation_score,
    compute_context_pressure_score,
    compute_epoch_staleness_score,
    compute_identity_coherence_score,
    compute_memory_uncertainty_score,
    compute_raw_aging_score_weighted,
    compute_unsaved_work_score,
    score_with_confidence,
)

if TYPE_CHECKING:
    from kernle.core import Kernle

logger = logging.getLogger(__name__)

# Emoji mapping for anxiety levels (Kernle-layer presentation)
_LEVEL_EMOJIS = {
    "calm": "🟢",
    "aware": "🟡",
    "elevated": "🟠",
    "high": "🔴",
    "critical": "⚫",
}


class AnxietyMixin:
    """Mixin providing anxiety tracking capabilities.

    Measures anxiety across multiple dimensions:
    - Context pressure: How full is the context window?
    - Unsaved work: Time since last checkpoint
    - Consolidation debt: Unreflected episodes
    - Raw aging: Unprocessed raw entries getting stale
    - Identity coherence: Strength of self-model
    - Memory uncertainty: Low-confidence beliefs
    - Epoch staleness: Time since last epoch transition
    """

    # Anxiety level thresholds and colors
    ANXIETY_LEVELS = {
        (0, 30): ("🟢", "Calm"),
        (31, 50): ("🟡", "Aware"),
        (51, 70): ("🟠", "Elevated"),
        (71, 85): ("🔴", "High"),
        (86, 100): ("⚫", "Critical"),
    }

    # Dimension weights for composite score (7-factor model)
    ANXIETY_WEIGHTS = SEVEN_DIM_WEIGHTS

    def _get_anxiety_level(self: "Kernle", score: int) -> tuple:
        """Get emoji and label for an anxiety score."""
        for (low, high), (emoji, label) in self.ANXIETY_LEVELS.items():
            if low <= score <= high:
                return emoji, label
        return "⚫", "Critical"

    def _get_checkpoint_age_minutes(self: "Kernle") -> Optional[int]:
        """Get minutes since last checkpoint."""
        cp = self.load_checkpoint()
        if not cp or "timestamp" not in cp:
            return None

        try:
            cp_time = datetime.fromisoformat(cp["timestamp"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = now - cp_time
            return int(delta.total_seconds() / 60)
        except (ValueError, TypeError):
            return None

    def _get_unreflected_episodes(self: "Kernle") -> List[Any]:
        """Get episodes without lessons (unreflected experiences)."""
        episodes = self._storage.get_episodes(limit=100)
        # Filter out checkpoints and episodes that already have lessons
        unreflected = [
            e for e in episodes if (not e.tags or "checkpoint" not in e.tags) and not e.lessons
        ]
        return unreflected

    def _get_low_confidence_beliefs(self: "Kernle", threshold: float = 0.5) -> List[Any]:
        """Get beliefs with confidence below threshold."""
        beliefs = self._storage.get_beliefs(limit=100)
        return [b for b in beliefs if b.confidence < threshold]

    def _get_aging_raw_entries(self: "Kernle", age_hours: int = 24) -> tuple:
        """Get raw entries that are older than age_hours and unprocessed.

        Returns:
            Tuple of (total_unprocessed, aging_count, oldest_age_hours, entries)
        """
        raw_entries = self.list_raw(processed=False, limit=100)
        now = datetime.now(timezone.utc)

        aging_count = 0
        oldest_age_hours = 0

        for entry in raw_entries:
            try:
                entry_time = entry.captured_at or entry.timestamp
                if entry_time:
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                    age = now - entry_time
                    entry_hours = age.total_seconds() / 3600

                    if entry_hours > age_hours:
                        aging_count += 1

                    if entry_hours > oldest_age_hours:
                        oldest_age_hours = entry_hours
            except (ValueError, TypeError, AttributeError):
                continue  # Skip entries with unparseable timestamps

        return len(raw_entries), aging_count, oldest_age_hours, raw_entries

    def _get_epoch_staleness_months(self: "Kernle") -> Optional[float]:
        """Get months since the current epoch started, or since the last epoch ended.

        Returns None if the epochs table has no data (feature not in use).
        """
        try:
            current_epoch = self._storage.get_current_epoch()
            if current_epoch and current_epoch.started_at:
                started = current_epoch.started_at
                if isinstance(started, str):
                    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                delta = now - started
                return delta.total_seconds() / (30.44 * 86400)  # average days per month

            # No active epoch -- check if there are any closed epochs
            epochs = self._storage.get_epochs(limit=1)
            if not epochs:
                return None  # No epochs at all -- feature not in use

            # Last epoch was closed; measure from its ended_at
            last = epochs[0]
            if last.ended_at:
                ended = last.ended_at
                if isinstance(ended, str):
                    ended = datetime.fromisoformat(ended.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                delta = now - ended
                return delta.total_seconds() / (30.44 * 86400)

            return None
        except Exception as exc:
            logger.debug(
                "Swallowed %s in _months_since_last_epoch: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None  # Graceful degradation if epochs not available

    def get_anxiety_report(
        self: "Kernle",
        context_tokens: Optional[int] = None,
        context_limit: int = 200000,
        detailed: bool = False,
    ) -> dict:
        """Calculate memory anxiety across 7 dimensions.

        This measures the functional anxiety of a synthetic intelligence
        facing finite context and potential memory loss.

        Args:
            context_tokens: Current context token usage (if known)
            context_limit: Maximum context window size
            detailed: Include additional details in the report

        Returns:
            dict with:
            - overall_score: Composite anxiety score (0-100)
            - overall_level: Human-readable level (Calm, Aware, etc.)
            - overall_emoji: Level indicator emoji
            - dimensions: Per-dimension breakdown
            - recommendations: If detailed=True, includes recommended actions
        """
        dimensions = {}

        # 1. Context Pressure (0-100%)
        if context_tokens is not None:
            context_pressure_pct = min(100, int((context_tokens / context_limit) * 100))
            context_detail = f"{context_tokens:,}/{context_limit:,} tokens"
        else:
            checkpoint_age = self._get_checkpoint_age_minutes()
            if checkpoint_age is not None:
                estimated_tokens = checkpoint_age * 500
                context_pressure_pct = min(100, int((estimated_tokens / context_limit) * 100))
                context_detail = (
                    f"~{estimated_tokens:,} tokens (estimated from {checkpoint_age}min session)"
                )
            else:
                context_pressure_pct = 10
                context_detail = "No checkpoint (fresh session)"

        context_score = compute_context_pressure_score(context_pressure_pct)

        context_conf = score_with_confidence(min(100, context_score), 1)
        dimensions["context_pressure"] = {
            "score": context_conf["score"],
            "confidence": context_conf["confidence"],
            "raw_value": context_pressure_pct,
            "detail": context_detail,
            "emoji": self._get_anxiety_level(context_score)[0],
        }

        # 2. Unsaved Work (0-100%)
        checkpoint_age = self._get_checkpoint_age_minutes()
        unsaved_score = compute_unsaved_work_score(checkpoint_age)
        if checkpoint_age is None:
            unsaved_detail = "No checkpoint found"
        elif checkpoint_age < 60:
            unsaved_detail = f"{checkpoint_age} min since checkpoint"
        else:
            unsaved_detail = f"{checkpoint_age} min since checkpoint (STALE)"

        unsaved_conf = score_with_confidence(
            min(100, unsaved_score), 1 if checkpoint_age is not None else 0
        )
        dimensions["unsaved_work"] = {
            "score": unsaved_conf["score"],
            "confidence": unsaved_conf["confidence"],
            "raw_value": checkpoint_age,
            "detail": unsaved_detail,
            "emoji": self._get_anxiety_level(unsaved_score)[0],
        }

        # 3. Consolidation Debt (0-100%)
        unreflected = self._get_unreflected_episodes()
        unreflected_count = len(unreflected)
        consolidation_score = compute_consolidation_score(unreflected_count)

        if unreflected_count <= 3:
            consolidation_detail = f"{unreflected_count} unreflected episodes"
        elif unreflected_count <= 7:
            consolidation_detail = f"{unreflected_count} unreflected episodes (building up)"
        elif unreflected_count <= 15:
            consolidation_detail = f"{unreflected_count} unreflected episodes (significant backlog)"
        else:
            consolidation_detail = f"{unreflected_count} unreflected episodes (URGENT)"

        consolidation_conf = score_with_confidence(min(100, consolidation_score), unreflected_count)
        dimensions["consolidation_debt"] = {
            "score": consolidation_conf["score"],
            "confidence": consolidation_conf["confidence"],
            "raw_value": unreflected_count,
            "detail": consolidation_detail,
            "emoji": self._get_anxiety_level(consolidation_score)[0],
        }

        # 4. Identity Coherence (inverted - high coherence = low anxiety)
        identity_confidence = self.get_identity_confidence()
        identity_anxiety = compute_identity_coherence_score(identity_confidence)

        if identity_confidence >= 0.8:
            identity_detail = f"{identity_confidence:.0%} identity confidence (strong)"
        elif identity_confidence >= 0.5:
            identity_detail = f"{identity_confidence:.0%} identity confidence (developing)"
        else:
            identity_detail = f"{identity_confidence:.0%} identity confidence (WEAK)"

        # Count data points that contributed to identity confidence
        _id_sample_count = len(self._storage.get_values(limit=10)) + min(
            20, len(self._storage.get_beliefs(limit=100))
        )
        identity_conf = score_with_confidence(identity_anxiety, _id_sample_count)
        dimensions["identity_coherence"] = {
            "score": identity_conf["score"],
            "confidence": identity_conf["confidence"],
            "raw_value": identity_confidence,
            "detail": identity_detail,
            "emoji": self._get_anxiety_level(identity_anxiety)[0],
        }

        # 5. Memory Uncertainty (0-100%)
        low_conf_beliefs = self._get_low_confidence_beliefs(0.5)
        total_beliefs = len(self._storage.get_beliefs(limit=100))

        # Note: uncertainty_detail gets overwritten below based on low_conf count
        # so we just need to handle the no-beliefs case here
        if total_beliefs == 0:
            uncertainty_detail = "No beliefs yet"
        else:
            uncertainty_detail = ""  # Will be set below based on low_conf count

        uncertainty_score = compute_memory_uncertainty_score(len(low_conf_beliefs))
        if len(low_conf_beliefs) <= 2:
            uncertainty_detail = f"{len(low_conf_beliefs)} low-confidence beliefs"
        elif len(low_conf_beliefs) <= 5:
            uncertainty_detail = (
                f"{len(low_conf_beliefs)} low-confidence beliefs (some uncertainty)"
            )
        else:
            uncertainty_detail = (
                f"{len(low_conf_beliefs)} low-confidence beliefs (HIGH uncertainty)"
            )

        uncertainty_conf = score_with_confidence(min(100, uncertainty_score), total_beliefs)
        dimensions["memory_uncertainty"] = {
            "score": uncertainty_conf["score"],
            "confidence": uncertainty_conf["confidence"],
            "raw_value": len(low_conf_beliefs),
            "detail": uncertainty_detail,
            "emoji": self._get_anxiety_level(uncertainty_score)[0],
        }

        # 6. Raw Entry Aging (0-100%) — weighted by content length
        total_unprocessed, aging_count, oldest_hours, raw_entries = self._get_aging_raw_entries(24)
        raw_aging_score = compute_raw_aging_score_weighted(raw_entries, age_threshold_hours=24)

        if total_unprocessed == 0:
            raw_aging_detail = "No unprocessed raw entries"
        elif aging_count == 0:
            raw_aging_detail = f"{total_unprocessed} unprocessed (all fresh)"
        elif aging_count <= 3:
            raw_aging_detail = f"{aging_count}/{total_unprocessed} entries >24h old"
        elif aging_count <= 7:
            oldest_days = int(oldest_hours / 24)
            raw_aging_detail = f"{aging_count} entries aging (oldest: {oldest_days}d)"
        else:
            oldest_days = int(oldest_hours / 24)
            raw_aging_detail = (
                f"{aging_count} entries STALE (oldest: {oldest_days}d) - review needed"
            )

        raw_aging_conf = score_with_confidence(min(100, raw_aging_score), total_unprocessed)
        dimensions["raw_aging"] = {
            "score": raw_aging_conf["score"],
            "confidence": raw_aging_conf["confidence"],
            "raw_value": aging_count,
            "detail": raw_aging_detail,
            "emoji": self._get_anxiety_level(raw_aging_score)[0],
        }

        # 7. Epoch Staleness (0-100%)
        epoch_months = self._get_epoch_staleness_months()
        epoch_staleness_score = compute_epoch_staleness_score(epoch_months)

        if epoch_months is None:
            epoch_staleness_detail = "No epochs (not in use)"
        elif epoch_months < 6:
            epoch_staleness_detail = f"{epoch_months:.1f} months (current epoch is fresh)"
        elif epoch_months < 12:
            epoch_staleness_detail = (
                f"{epoch_months:.1f} months (consider whether current epoch is still accurate)"
            )
        elif epoch_months < 18:
            epoch_staleness_detail = (
                f"{epoch_months:.1f} months (significant time without deep reflection)"
            )
        else:
            epoch_staleness_detail = f"{epoch_months:.1f} months (likely overdue for epoch review)"

        epoch_sample_count = 1 if epoch_months is not None else 0
        epoch_conf = score_with_confidence(min(100, epoch_staleness_score), epoch_sample_count)
        dimensions["epoch_staleness"] = {
            "score": epoch_conf["score"],
            "confidence": epoch_conf["confidence"],
            "raw_value": epoch_months,
            "detail": epoch_staleness_detail,
            "emoji": self._get_anxiety_level(epoch_staleness_score)[0],
        }

        # Calculate composite score (weighted average)
        overall_score = 0
        for dim_name, weight in self.ANXIETY_WEIGHTS.items():
            overall_score += dimensions[dim_name]["score"] * weight
        overall_score = int(overall_score)

        overall_emoji, overall_level = self._get_anxiety_level(overall_score)

        # Apply hysteresis to prevent level oscillation near boundaries
        previous_level = getattr(self, "_previous_anxiety_level", None)
        hysteresis_level = apply_hysteresis(overall_score, previous_level)
        # Update the stored previous level for next call
        self._previous_anxiety_level = hysteresis_level  # type: ignore[attr-defined]
        # Translate key back to display label
        _key_to_label = {
            "calm": "Calm",
            "aware": "Aware",
            "elevated": "Elevated",
            "high": "High",
            "critical": "Critical",
        }
        overall_level = _key_to_label.get(hysteresis_level, overall_level)
        overall_emoji = _LEVEL_EMOJIS.get(hysteresis_level, overall_emoji)

        # Compute overall confidence (weighted average of dimension confidences)
        total_conf_weight = 0.0
        total_conf_sum = 0.0
        for dim_name, weight in self.ANXIETY_WEIGHTS.items():
            dim_confidence = dimensions[dim_name].get("confidence", 0.5)
            total_conf_sum += dim_confidence * weight
            total_conf_weight += weight
        overall_confidence = total_conf_sum / total_conf_weight if total_conf_weight > 0 else 0.0

        report = {
            "overall_score": overall_score,
            "overall_level": overall_level,
            "overall_emoji": overall_emoji,
            "overall_confidence": round(overall_confidence, 2),
            "dimensions": dimensions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stack_id": self.stack_id,
        }

        if detailed:
            report["recommendations"] = self.get_recommended_actions(overall_score)
            report["context_limit"] = context_limit
            report["context_tokens"] = context_tokens

        return report

    def anxiety(
        self: "Kernle",
        context_tokens: Optional[int] = None,
        context_limit: int = 200000,
        detailed: bool = False,
    ) -> dict:
        """Alias for get_anxiety_report() - more intuitive API.

        See get_anxiety_report() for full documentation.
        """
        return self.get_anxiety_report(context_tokens, context_limit, detailed)

    def get_recommended_actions(self: "Kernle", anxiety_level: int) -> List[Dict[str, Any]]:
        """Return prioritized actions based on anxiety level.

        Actions reference actual kernle commands/methods for execution.

        Args:
            anxiety_level: Overall anxiety score (0-100)

        Returns:
            List of action dicts with priority, description, command, and method
        """
        actions = []

        checkpoint_age = self._get_checkpoint_age_minutes()
        unreflected = self._get_unreflected_episodes()
        low_conf = self._get_low_confidence_beliefs(0.5)
        identity_conf = self.get_identity_confidence()

        # Calm (0-30): Continue normal work
        if anxiety_level <= 30:
            if len(unreflected) > 0:
                actions.append(
                    {
                        "priority": "low",
                        "description": f"Reflect on {len(unreflected)} recent experiences when convenient",
                        "command": "kernle promote",
                        "method": "promote",
                    }
                )
            return actions

        # Aware (31-50): Checkpoint and note major decisions
        if anxiety_level <= 50:
            if checkpoint_age is None or checkpoint_age > 15:
                actions.append(
                    {
                        "priority": "medium",
                        "description": "Checkpoint current work state",
                        "command": "kernle checkpoint save '<task>'",
                        "method": "checkpoint",
                    }
                )
            if len(unreflected) > 3:
                actions.append(
                    {
                        "priority": "medium",
                        "description": f"Process {len(unreflected)} unreflected episodes",
                        "command": "kernle promote",
                        "method": "promote",
                    }
                )
            return actions

        # Elevated (51-70): Full checkpoint, consolidate, verify
        if anxiety_level <= 70:
            actions.append(
                {
                    "priority": "high",
                    "description": "Full checkpoint with context",
                    "command": "kernle checkpoint save '<task>' --context '<summary>'",
                    "method": "checkpoint",
                }
            )
            if len(unreflected) > 0:
                actions.append(
                    {
                        "priority": "high",
                        "description": f"Consolidate {len(unreflected)} unreflected episodes",
                        "command": "kernle promote",
                        "method": "promote",
                    }
                )
            if identity_conf < 0.7:
                actions.append(
                    {
                        "priority": "medium",
                        "description": "Run identity synthesis to strengthen coherence",
                        "command": "kernle identity show",
                        "method": "synthesize_identity",
                    }
                )
            if len(low_conf) > 0:
                actions.append(
                    {
                        "priority": "low",
                        "description": f"Review {len(low_conf)} uncertain beliefs",
                        "command": "kernle meta uncertain",
                        "method": "get_uncertain_memories",
                    }
                )
            return actions

        # High (71-85): Priority memory work
        if anxiety_level <= 85:
            actions.append(
                {
                    "priority": "critical",
                    "description": "PRIORITY: Run full consolidation",
                    "command": "kernle promote",
                    "method": "promote",
                }
            )
            actions.append(
                {
                    "priority": "critical",
                    "description": "Full checkpoint with session summary",
                    "command": "kernle checkpoint save '<task>' --context '<full summary>'",
                    "method": "checkpoint",
                }
            )
            actions.append(
                {
                    "priority": "high",
                    "description": "Run identity synthesis and save",
                    "command": "kernle identity show",
                    "method": "synthesize_identity",
                }
            )
            sync_status = self.get_sync_status()
            if sync_status.get("online"):
                actions.append(
                    {
                        "priority": "high",
                        "description": "Sync to cloud storage",
                        "command": "kernle sync (if available)",
                        "method": "sync",
                    }
                )
            return actions

        # Critical (86-100): Emergency protocols
        actions.append(
            {
                "priority": "emergency",
                "description": "EMERGENCY: Run emergency_save immediately",
                "command": "kernle anxiety --emergency",
                "method": "emergency_save",
            }
        )
        actions.append(
            {
                "priority": "emergency",
                "description": "Final checkpoint with handoff note",
                "command": "kernle checkpoint save 'HANDOFF' --context '<state for next session>'",
                "method": "checkpoint",
            }
        )
        actions.append(
            {
                "priority": "critical",
                "description": "Accept some context will be lost - prioritize key insights",
                "command": None,
                "method": None,
            }
        )

        return actions

    def emergency_save(self: "Kernle", summary: Optional[str] = None) -> Dict[str, Any]:
        """Critical-level action: save everything possible.

        This is the nuclear option when anxiety hits critical levels.
        Performs all possible memory preservation actions.

        Args:
            summary: Optional session summary for the checkpoint

        Returns:
            dict with what was saved and any errors
        """
        results = {
            "checkpoint_saved": False,
            "episodes_consolidated": 0,
            "sync_attempted": False,
            "sync_success": False,
            "identity_synthesized": False,
            "errors": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Emergency checkpoint with full context
        try:
            checkpoint_summary = summary or "EMERGENCY SAVE - Critical anxiety level"
            cp = self.checkpoint(
                task="EMERGENCY_SAVE",
                pending=["Review previous session state"],
                context=checkpoint_summary,
            )
            results["checkpoint_saved"] = True
            results["checkpoint"] = cp
        except Exception as e:
            logger.warning("Emergency save checkpoint failed: %s", e, exc_info=True)
            results["errors"].append(f"Checkpoint failed: {str(e)}")

        # 2. Consolidate all unreflected episodes
        try:
            promotion = self.promote(min_episodes=1)
            results["episodes_consolidated"] = promotion.get("episodes_scanned", 0)
            results["consolidation_result"] = promotion
        except Exception as e:
            logger.warning("Emergency save consolidation failed: %s", e, exc_info=True)
            results["errors"].append(f"Consolidation failed: {str(e)}")

        # 3. Synthesize identity (to have a coherent state)
        try:
            identity = self.synthesize_identity()
            results["identity_synthesized"] = True
            results["identity_confidence"] = identity.get("confidence", 0)
        except Exception as e:
            logger.warning("Emergency save identity synthesis failed: %s", e, exc_info=True)
            results["errors"].append(f"Identity synthesis failed: {str(e)}")

        # 4. Attempt cloud sync
        try:
            sync_status = self.get_sync_status()
            if sync_status.get("online"):
                results["sync_attempted"] = True
                sync_result = self.sync()
                results["sync_success"] = sync_result.get("success", False)
                results["sync_result"] = sync_result
            else:
                results["sync_attempted"] = False
        except Exception as e:
            logger.warning("Emergency save sync failed: %s", e, exc_info=True)
            results["errors"].append(f"Sync failed: {str(e)}")

        # 5. Record this emergency save as an episode
        try:
            self.episode(
                objective="Emergency memory save",
                outcome="completed" if not results["errors"] else "partial",
                lessons=[
                    "Anxiety level hit critical - triggered emergency save",
                    f"Saved checkpoint: {results['checkpoint_saved']}",
                    f"Consolidated {results['episodes_consolidated']} episodes",
                ],
                tags=["emergency", "anxiety", "critical"],
            )
        except Exception as e:
            logger.warning("Emergency save episode recording failed: %s", e, exc_info=True)
            results["errors"].append(f"Episode recording failed: {str(e)}")

        results["success"] = len(results["errors"]) == 0
        return results
