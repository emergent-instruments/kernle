"""Anxiety tracking stack component.

Measures the functional anxiety of a synthetic intelligence across
multiple dimensions: context pressure, consolidation debt, memory
uncertainty, raw aging, identity coherence, and epoch staleness.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.anxiety_core import (
    FIVE_DIM_WEIGHTS,
    apply_hysteresis,
    compute_consolidation_score,
    compute_epoch_staleness_score,
    compute_identity_coherence_score,
    compute_memory_uncertainty_score,
    compute_raw_aging_score_weighted,
    score_with_confidence,
)
from kernle.anxiety_core import get_anxiety_level as _get_anxiety_level
from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

ANXIETY_WEIGHTS = FIVE_DIM_WEIGHTS


class AnxietyComponent:
    """Anxiety tracking component.

    Measures memory-related anxiety across multiple dimensions and
    contributes anxiety levels to working memory context during on_load.
    """

    name = "anxiety"
    version = "1.0.0"
    required = False
    needs_inference = False
    inference_scope = "none"
    priority = 250

    OVERALL_ALERT_SCORE = 70
    DIMENSION_ALERT_SCORE = 70

    def __init__(self) -> None:
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._storage: Optional[Any] = None
        self._previous_level: Optional[str] = None

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
        """Contribute anxiety levels to working memory context."""
        if self._storage is None:
            return
        report = self.get_anxiety_report()
        context["anxiety"] = {
            "overall_score": report["overall_score"],
            "overall_level": report["overall_level"],
        }

    def on_maintenance(self) -> Dict[str, Any]:
        """Report anxiety levels during maintenance."""
        if self._storage is None:
            logger.debug("AnxietyComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}
        report = self.get_anxiety_report()
        report["alerts"] = self._build_alerts(report)
        return report

    # ---- Core Logic ----

    def get_anxiety_report(self) -> Dict[str, Any]:
        """Calculate memory anxiety across available dimensions.

        Returns a simplified report with scores per dimension and an
        overall composite score.
        """
        if self._storage is None:
            return {"overall_score": 0, "overall_level": "Calm", "dimensions": {}}

        dimensions: Dict[str, Dict[str, Any]] = {}

        # Consolidation debt
        episodes = self._storage.get_episodes(limit=100)
        unreflected = [
            e for e in episodes if (not e.tags or "checkpoint" not in e.tags) and not e.lessons
        ]
        unreflected_count = len(unreflected)
        consolidation_score = compute_consolidation_score(unreflected_count)
        cons_conf = score_with_confidence(min(100, consolidation_score), unreflected_count)
        dimensions["consolidation_debt"] = {
            "score": cons_conf["score"],
            "confidence": cons_conf["confidence"],
            "detail": f"{unreflected_count} unreflected episodes",
        }

        # Memory uncertainty
        beliefs = self._storage.get_beliefs(limit=100)
        low_conf = [b for b in beliefs if b.confidence < 0.5]
        uncertainty_score = compute_memory_uncertainty_score(len(low_conf))
        unc_conf = score_with_confidence(min(100, uncertainty_score), len(beliefs))
        dimensions["memory_uncertainty"] = {
            "score": unc_conf["score"],
            "confidence": unc_conf["confidence"],
            "detail": f"{len(low_conf)} low-confidence beliefs",
        }

        # Identity coherence (inverted: high coherence = low anxiety)
        values = self._storage.get_values(limit=10)
        total_conf = 0.0
        count = 0
        for v in values:
            total_conf += getattr(v, "confidence", 0.8)
            count += 1
        for b in beliefs[:20]:
            total_conf += b.confidence
            count += 1
        identity_confidence = total_conf / count if count > 0 else 0.0
        identity_anxiety = compute_identity_coherence_score(identity_confidence)
        id_conf = score_with_confidence(identity_anxiety, count)
        dimensions["identity_coherence"] = {
            "score": id_conf["score"],
            "confidence": id_conf["confidence"],
            "detail": f"{identity_confidence:.0%} identity confidence",
        }

        # Raw aging
        raw_aging_score = 0
        raw_aging_detail = "No unprocessed raw entries"
        raw_sample_count = 0
        try:
            if hasattr(self._storage, "list_raw"):
                raw_entries = self._storage.list_raw(processed=False, limit=100)
                total_unprocessed = len(raw_entries)
                raw_sample_count = total_unprocessed
                raw_aging_score = compute_raw_aging_score_weighted(
                    raw_entries, age_threshold_hours=24
                )
                now = datetime.now(timezone.utc)
                aging_count = 0
                for entry in raw_entries:
                    try:
                        entry_time = getattr(entry, "captured_at", None) or getattr(
                            entry, "timestamp", None
                        )
                        if entry_time:
                            if isinstance(entry_time, str):
                                entry_time = datetime.fromisoformat(
                                    entry_time.replace("Z", "+00:00")
                                )
                            age = now - entry_time
                            if age.total_seconds() / 3600 > 24:
                                aging_count += 1
                    except (ValueError, TypeError, AttributeError):
                        continue  # Skip entries with unparseable timestamps
                if total_unprocessed == 0:
                    raw_aging_detail = "No unprocessed raw entries"
                elif aging_count == 0:
                    raw_aging_detail = f"{total_unprocessed} unprocessed (all fresh)"
                else:
                    raw_aging_detail = f"{aging_count}/{total_unprocessed} entries >24h old"
        except Exception as exc:
            logger.debug(
                "Swallowed %s computing raw_aging anxiety dimension: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
        raw_conf = score_with_confidence(raw_aging_score, raw_sample_count)
        dimensions["raw_aging"] = {
            "score": raw_conf["score"],
            "confidence": raw_conf["confidence"],
            "detail": raw_aging_detail,
        }

        # Epoch staleness
        epoch_staleness_score = 0
        epoch_sample_count = 0
        try:
            current_epoch = self._storage.get_current_epoch()
            if current_epoch and current_epoch.started_at:
                started = current_epoch.started_at
                if isinstance(started, str):
                    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                months = (now - started).total_seconds() / (30.44 * 86400)
                epoch_staleness_score = compute_epoch_staleness_score(months)
                epoch_sample_count = 1
        except Exception as exc:
            logger.debug(
                "Swallowed %s computing epoch_staleness anxiety dimension: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
        epoch_conf = score_with_confidence(min(100, epoch_staleness_score), epoch_sample_count)
        dimensions["epoch_staleness"] = {
            "score": epoch_conf["score"],
            "confidence": epoch_conf["confidence"],
            "detail": "Epoch staleness",
        }

        # Composite score
        overall_score = 0
        for dim_name, weight in ANXIETY_WEIGHTS.items():
            overall_score += dimensions[dim_name]["score"] * weight
        overall_score = int(overall_score)

        _, overall_level = _get_anxiety_level(overall_score)

        # Apply hysteresis to prevent level oscillation
        hysteresis_level = apply_hysteresis(overall_score, self._previous_level)
        self._previous_level = hysteresis_level
        _key_to_label = {
            "calm": "Calm",
            "aware": "Aware",
            "elevated": "Elevated",
            "high": "High",
            "critical": "Critical",
        }
        overall_level = _key_to_label.get(hysteresis_level, overall_level)

        return {
            "overall_score": overall_score,
            "overall_level": overall_level,
            "dimensions": dimensions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _build_alerts(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create threshold alerts for anxiety levels."""
        alerts: List[Dict[str, Any]] = []
        overall_score = report.get("overall_score", 0)
        if overall_score >= self.OVERALL_ALERT_SCORE:
            severity = "critical" if overall_score >= 90 else "high"
            alerts.append(
                {
                    "type": "overall_anxiety",
                    "severity": severity,
                    "message": f"Overall anxiety score is elevated: {overall_score}",
                    "value": overall_score,
                }
            )

        dimensions = report.get("dimensions", {})
        for dim_name, data in dimensions.items():
            score = int(data.get("score", 0))
            if score >= self.DIMENSION_ALERT_SCORE:
                alerts.append(
                    {
                        "type": f"dimension:{dim_name}",
                        "severity": "medium" if score < 90 else "high",
                        "message": f"Dimension '{dim_name}' breached threshold.",
                        "value": score,
                    }
                )

        return alerts
