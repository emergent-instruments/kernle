"""Trust stack component.

Provides trust-based memory gating, search result boosting, and trust decay
during maintenance. Gates memory input by source trust, boosts search results
from trusted sources, and applies trust decay to aging assessments.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult
from kernle.types import (
    DEFAULT_TRUST,
    SELF_TRUST_FLOOR,
    TRUST_DECAY_RATE,
    TRUST_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class TrustComponent:
    """Trust component for memory gating and trust-based operations.

    Gates memory input by checking source entity trust against action thresholds.
    Boosts search results from trusted sources. Applies trust decay to stale
    assessments during maintenance. Does not block saves, only annotates.
    """

    # Component metadata - these class attributes are required
    name = "trust"
    version = "1.0.0"
    required = False
    needs_inference = False
    inference_scope = "none"
    priority = 150  # Runs before emotions/forgetting but after embedding

    def __init__(self) -> None:
        """Initialize the TrustComponent.

        All components must initialize these three attributes to None.
        They will be set later when the component is attached to a stack.
        """
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._storage: Optional[Any] = None

    def attach(self, stack_id: str, inference: Optional[InferenceService] = None) -> None:
        """Attach the component to a stack.

        Called by the stack when the component is registered.

        Args:
            stack_id: The ID of the stack this component is attached to
            inference: Optional inference service for model-based operations
        """
        self._stack_id = stack_id
        self._inference = inference

    def detach(self) -> None:
        """Detach the component from its stack.

        Clears all references to stack resources.
        """
        self._stack_id = None
        self._inference = None
        self._storage = None

    def set_inference(self, inference: Optional[InferenceService]) -> None:
        """Update the inference service reference.

        Args:
            inference: The new inference service (or None to clear)
        """
        self._inference = inference

    def set_storage(self, storage: Any) -> None:
        """Set the storage backend reference.

        Called by Stack after attach to provide storage access.
        This allows the component to read/write trust assessments.

        Args:
            storage: The storage backend instance (typically SQLiteStorage)
        """
        self._storage = storage

    # ---- Lifecycle Hooks ----

    def on_save(self, memory_type: str, memory_id: str, memory: Any) -> Any:
        """Gate memory input by source trust.

        Checks if the memory has a source_entity attribute. If so, looks up
        the trust assessment for that entity. If trust is below the threshold
        for the action type, returns a warning dict. Does NOT block saves,
        only annotates with trust warnings.

        Args:
            memory_type: The type of memory being saved (e.g., "episode", "note")
            memory_id: The unique identifier for this memory
            memory: The memory object being saved

        Returns:
            None if no trust check needed, or a dict with trust warning if
            the source entity has low trust for this action type:
            {"trust_warning": str, "trust_level": float, "domain": str}
        """
        # If storage is not available, we can't check trust
        if self._storage is None:
            return None

        # Check if this memory has a source entity
        source_entity = getattr(memory, "source_entity", None)
        if source_entity is None:
            # No source entity = no trust check needed
            return None

        # Use the lowest TRUST_THRESHOLDS value as baseline for general saves.
        # TRUST_THRESHOLDS defines action-specific thresholds (suggest_belief,
        # request_deletion, etc.) but not per-memory-type save thresholds.
        # The lowest threshold ("suggest_belief": 0.3) is appropriate for
        # general memory saves since saving is a lower-privilege action.
        threshold = min(TRUST_THRESHOLDS.values()) if TRUST_THRESHOLDS else 0.3

        # Get the trust assessment for this source entity
        try:
            assessment = self._storage.get_trust_assessment(source_entity)
        except AttributeError:
            # Storage doesn't support trust assessments
            return None

        if assessment is None:
            # No assessment = default to warning about unknown source
            return {
                "trust_warning": f"No trust assessment for source: {source_entity}",
                "trust_level": 0.0,
                "domain": "general",
            }

        # Extract trust score from the assessment
        # Check authority first - if entity has "all" authority, they're trusted
        authority = getattr(assessment, "authority", None) or []
        has_all_authority = any(a.get("scope") == "all" for a in authority if isinstance(a, dict))

        if has_all_authority:
            # Entity has full authority, trust check passes
            return None

        # Get the trust score for the general domain
        dimensions = getattr(assessment, "dimensions", None) or {}
        domain = "general"
        domain_data = dimensions.get(domain, {})

        # Handle case where domain_data might not be a dict
        if isinstance(domain_data, dict):
            trust_score = domain_data.get("score", DEFAULT_TRUST)
        else:
            trust_score = DEFAULT_TRUST

        # Check if trust meets the threshold
        if trust_score >= threshold:
            # Trust check passes
            return None

        # Trust is below threshold - return warning
        return {
            "trust_warning": (
                f"Low trust ({trust_score:.2f}) for {source_entity} "
                f"(threshold: {threshold:.2f} for save_{memory_type})"
            ),
            "trust_level": trust_score,
            "domain": domain,
        }

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Boost search results from trusted sources.

        For each search result, checks if the underlying record has a
        source_entity attribute. If found, looks up trust and adjusts
        the result score accordingly. Higher trust = higher boost.

        Args:
            query: The search query string
            results: List of SearchResult objects from the search

        Returns:
            List of SearchResult objects with scores adjusted for trust.
            Results are re-sorted by the new scores.
        """
        # If storage is not available, can't check trust
        if self._storage is None:
            return results

        # If no results, nothing to boost
        if not results:
            return results

        # Process each result and adjust score based on trust
        adjusted_results = []
        for result in results:
            # Get the source entity from the record
            record = getattr(result, "record", None)
            if record is None:
                # No record = no trust boost
                adjusted_results.append(result)
                continue

            source_entity = getattr(record, "source_entity", None)
            if source_entity is None:
                # No source entity = no trust boost
                adjusted_results.append(result)
                continue

            # Look up trust assessment
            try:
                assessment = self._storage.get_trust_assessment(source_entity)
            except AttributeError:
                # Storage doesn't support trust assessments
                adjusted_results.append(result)
                continue

            if assessment is None:
                # No assessment = slight penalty (0.95x)
                boost_factor = 0.95
            else:
                # Extract trust score
                dimensions = getattr(assessment, "dimensions", None) or {}
                domain_data = dimensions.get("general", {})

                if isinstance(domain_data, dict):
                    trust_score = domain_data.get("score", DEFAULT_TRUST)
                else:
                    trust_score = DEFAULT_TRUST

                # Convert trust (0.0-1.0) to boost factor
                # trust=1.0 -> 1.3x boost
                # trust=0.5 -> 1.0x (no change)
                # trust=0.0 -> 0.7x penalty
                boost_factor = 0.7 + (trust_score * 0.6)

            # Apply boost to the result score
            original_score = result.score
            boosted_score = original_score * boost_factor

            # Create a new SearchResult with the boosted score
            # We need to preserve all original attributes
            adjusted_result = SearchResult(
                record_type=result.record_type,
                record=result.record,
                score=boosted_score,
            )
            adjusted_results.append(adjusted_result)

        # Re-sort results by the new scores (descending)
        adjusted_results.sort(key=lambda r: r.score, reverse=True)
        return adjusted_results

    def on_load(self, context: Dict[str, Any]) -> None:
        """Build trust summary for context.

        Reads all trust assessments and adds a trust_summary key to the
        context dict with entity→score mapping. This helps the agent
        understand current trust levels when loading working memory.

        Args:
            context: The context dict being built by the load operation.
                     This method modifies it in-place by adding trust_summary.
        """
        # If storage is not available, can't build summary
        if self._storage is None:
            return

        # Get all trust assessments
        try:
            assessments = self._storage.get_trust_assessments()
        except AttributeError:
            # Storage doesn't support trust assessments
            return

        if not assessments:
            # No assessments to summarize
            return

        # Build summary dict: entity -> {trust: score, authority: [scopes]}
        summary = {}
        for assessment in assessments:
            entity = getattr(assessment, "entity", None)
            if entity is None:
                continue

            # Extract general trust score
            dimensions = getattr(assessment, "dimensions", None) or {}
            general_data = dimensions.get("general", {})

            if isinstance(general_data, dict):
                score = general_data.get("score", DEFAULT_TRUST)
            else:
                score = DEFAULT_TRUST

            # Extract authority scopes
            authority = getattr(assessment, "authority", None) or []
            scopes = [a.get("scope", "unknown") for a in authority if isinstance(a, dict)]

            summary[entity] = {
                "trust": round(score, 2),
                "authority": scopes,
            }

        # Add summary to context
        context["trust_summary"] = summary

    def on_maintenance(self) -> Dict[str, Any]:
        """Apply trust decay to stale assessments.

        Reads all trust assessments and applies decay based on last_updated.
        Uses TRUST_DECAY_RATE from storage.base. For "self", applies
        SELF_TRUST_FLOOR to prevent decay below the floor.

        Returns:
            Dict with maintenance stats: assessments_checked, decayed,
            average_decay, skipped (if no storage)
        """
        # If storage is not available, can't run maintenance
        if self._storage is None:
            logger.debug("TrustComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        # Get all trust assessments
        try:
            assessments = self._storage.get_trust_assessments()
        except AttributeError:
            logger.debug("TrustComponent: storage doesn't support trust assessments")
            return {"skipped": True, "reason": "no_trust_support"}

        if not assessments:
            # No assessments to decay
            return {
                "assessments_checked": 0,
                "decayed": 0,
                "average_decay": 0.0,
            }

        # Track decay stats
        assessments_checked = 0
        decayed_count = 0
        total_decay_amount = 0.0

        now = datetime.now(timezone.utc)

        # Process each assessment
        for assessment in assessments:
            assessments_checked += 1

            # Get last_updated timestamp
            last_updated = getattr(assessment, "last_updated", None)
            if last_updated is None:
                # Use created_at as fallback
                last_updated = getattr(assessment, "created_at", None)

            if last_updated is None:
                # No timestamp = assume very old (365 days)
                days_since = 365.0
            else:
                # Calculate days since last update
                delta = now - last_updated
                days_since = delta.total_seconds() / 86400.0

            # Only decay if enough time has passed (at least 1 day)
            if days_since < 1.0:
                continue

            # Calculate decay factor: min(TRUST_DECAY_RATE * days, 1.0)
            # TRUST_DECAY_RATE is typically 0.001 per day
            decay_factor = min(TRUST_DECAY_RATE * days_since, 1.0)

            # Apply decay to each domain
            dimensions = getattr(assessment, "dimensions", None) or {}
            updated_dimensions = {}
            entity = getattr(assessment, "entity", "unknown")

            assessment_decayed = False

            for domain, domain_data in dimensions.items():
                # Handle case where domain_data might not be a dict
                if not isinstance(domain_data, dict):
                    updated_dimensions[domain] = domain_data
                    continue

                current_score = domain_data.get("score", DEFAULT_TRUST)

                # Decay toward DEFAULT_TRUST (0.5)
                # Formula: current + (DEFAULT_TRUST - current) * decay_factor
                decayed_score = current_score + (DEFAULT_TRUST - current_score) * decay_factor

                # For "self" entity, apply floor
                if entity == "self":
                    decayed_score = max(SELF_TRUST_FLOOR, decayed_score)

                # Round to 4 decimal places
                decayed_score = round(decayed_score, 4)

                # Track if any decay occurred
                if abs(decayed_score - current_score) > 0.0001:
                    total_decay_amount += abs(decayed_score - current_score)
                    assessment_decayed = True

                # Update domain data with new score
                updated_dimensions[domain] = {
                    **domain_data,
                    "score": decayed_score,
                }

            if assessment_decayed:
                # Update the assessment with decayed scores
                assessment.dimensions = updated_dimensions

                try:
                    self._storage.save_trust_assessment(assessment)
                    decayed_count += 1
                except AttributeError:
                    logger.warning(
                        "TrustComponent: storage doesn't support saving trust assessments"
                    )
            else:
                # No decay needed, keep original dimensions
                pass

        # Calculate average decay amount
        avg_decay = round(total_decay_amount / decayed_count, 4) if decayed_count > 0 else 0.0

        return {
            "assessments_checked": assessments_checked,
            "decayed": decayed_count,
            "average_decay": avg_decay,
        }
