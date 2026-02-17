"""Memory suggestion mixin for Kernle.

This module provides auto-extraction of memory suggestions from raw entries,
enabling the system to suggest structured memories while keeping the agent
in control of what gets promoted.

Uses inference for classification when available; returns empty results
otherwise.
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kernle.types import MemorySuggestion, RawEntry

if TYPE_CHECKING:
    from kernle.core import Kernle

logger = logging.getLogger(__name__)


def _normalize_suggestion_provenance_refs(source_refs: Optional[List[str]]) -> List[str]:
    """Normalize suggestion source refs to typed provenance refs.

    Backward compatibility:
    - Plain IDs are treated as raw IDs -> "raw:<id>"
    - Typed refs ("episode:<id>", "belief:<id>", etc.) are preserved
    """
    normalized: List[str] = []
    seen = set()
    for ref in source_refs or []:
        if not isinstance(ref, str):
            continue
        ref = ref.strip()
        if not ref:
            continue
        if ":" not in ref:
            ref = f"raw:{ref}"
        ref_type, ref_id = ref.split(":", 1)
        if not ref_type or not ref_id:
            continue
        canonical = f"{ref_type}:{ref_id}"
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


class SuggestionsMixin:
    """Mixin providing memory suggestion capabilities.

    Enables:
    - Pattern-based extraction of suggestions from raw entries
    - Review workflow (approve, modify, reject)
    - Promotion of suggestions to structured memories
    """

    def extract_suggestions(
        self: "Kernle",
        raw_entry: RawEntry,
        auto_save: bool = True,
    ) -> List[MemorySuggestion]:
        """Extract memory suggestions from a raw entry.

        Uses inference when available; returns empty list otherwise.

        Args:
            raw_entry: The raw entry to analyze
            auto_save: If True, save extracted suggestions to storage

        Returns:
            List of extracted suggestions
        """
        inference = self._get_inference()
        if inference is None:
            return []

        return self._extract_suggestions_inference(raw_entry, inference, auto_save)

    def _extract_suggestions_inference(
        self: "Kernle",
        raw_entry: RawEntry,
        inference,
        auto_save: bool = True,
    ) -> List[MemorySuggestion]:
        """Extract suggestions using inference (new path)."""
        from kernle.core.inference_utils import parse_inference_json

        content = raw_entry.blob or raw_entry.content or ""
        if not content.strip():
            return []

        prompt = (
            "Classify this text and suggest what type of memory it represents.\n\n"
            f"Text: {content[:500]}\n\n"
            'Return JSON: {"suggestions": [{"memory_type": "episode"|"belief"|"note", '
            '"confidence": float 0-1, "reason": string}]}\n'
            "Only include suggestions with confidence >= 0.5."
        )
        try:
            raw = inference.infer(
                prompt=prompt,
                system="You are a memory classification system. Return only valid JSON.",
            )
        except Exception:
            logger.debug("Suggestion inference call failed", exc_info=True)
            return []

        result = parse_inference_json(
            raw,
            required_fields=["suggestions"],
            fallback={"suggestions": []},
            logger=logger,
        )

        if result.fallback_used:
            return []

        suggestions = []
        for item in result.data.get("suggestions", []):
            memory_type = item.get("memory_type", "note")
            confidence = float(item.get("confidence", 0.5))
            if memory_type not in ("episode", "belief", "note"):
                memory_type = "note"
            if confidence < 0.5:
                continue

            if memory_type == "episode":
                suggestion = self._create_episode_suggestion(raw_entry, confidence)
            elif memory_type == "belief":
                suggestion = self._create_belief_suggestion(raw_entry, confidence)
            else:
                suggestion = self._create_note_suggestion(raw_entry, confidence)

            if suggestion:
                suggestions.append(suggestion)

        if auto_save:
            for suggestion in suggestions:
                self._storage.save_suggestion(suggestion)

        return suggestions

    def _create_episode_suggestion(
        self: "Kernle",
        raw_entry: RawEntry,
        confidence: float,
    ) -> Optional[MemorySuggestion]:
        """Create an episode suggestion from a raw entry.

        Args:
            raw_entry: Source raw entry
            confidence: Extraction confidence

        Returns:
            MemorySuggestion or None
        """
        content = raw_entry.blob or raw_entry.content or ""

        # Extract objective (first sentence or line)
        objective = self._extract_first_sentence(content)
        if not objective or len(objective) < 10:
            return None

        # Attempt to extract outcome
        outcome = self._extract_outcome(content)

        # Extract lessons if present
        lessons = self._extract_lessons(content)

        return MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id=self.stack_id,
            memory_type="episode",
            content={
                "objective": objective,
                "outcome": outcome or "Extracted from raw capture",
                "outcome_type": self._infer_outcome_type(content),
                "lessons": lessons,
            },
            confidence=confidence,
            source_raw_ids=[raw_entry.id],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

    def _create_belief_suggestion(
        self: "Kernle",
        raw_entry: RawEntry,
        confidence: float,
    ) -> Optional[MemorySuggestion]:
        """Create a belief suggestion from a raw entry.

        Args:
            raw_entry: Source raw entry
            confidence: Extraction confidence

        Returns:
            MemorySuggestion or None
        """
        content = raw_entry.blob or raw_entry.content or ""

        # Extract the belief statement
        statement = self._extract_belief_statement(content)
        if not statement or len(statement) < 10:
            return None

        # Infer belief type
        belief_type = self._infer_belief_type(content)

        return MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id=self.stack_id,
            memory_type="belief",
            content={
                "statement": statement,
                "belief_type": belief_type,
                "confidence": min(0.8, confidence),  # Start with modest confidence
            },
            confidence=confidence,
            source_raw_ids=[raw_entry.id],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

    def _create_note_suggestion(
        self: "Kernle",
        raw_entry: RawEntry,
        confidence: float,
    ) -> Optional[MemorySuggestion]:
        """Create a note suggestion from a raw entry.

        Args:
            raw_entry: Source raw entry
            confidence: Extraction confidence

        Returns:
            MemorySuggestion or None
        """
        content = (raw_entry.blob or raw_entry.content or "").strip()
        if len(content) < 10:
            return None

        # Infer note type
        note_type = self._infer_note_type(content)

        # Extract speaker if quote
        speaker = None
        if note_type == "quote":
            speaker = self._extract_speaker(content)

        # Extract reason if decision
        reason = None
        if note_type == "decision":
            reason = self._extract_reason(content)

        return MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id=self.stack_id,
            memory_type="note",
            content={
                "content": content,
                "note_type": note_type,
                "speaker": speaker,
                "reason": reason,
            },
            confidence=confidence,
            source_raw_ids=[raw_entry.id],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

    # === Helper Methods for Extraction ===

    def _extract_first_sentence(self: "Kernle", content: str) -> str:
        """Extract the first meaningful sentence from content."""
        # Split by sentence boundaries
        sentences = re.split(r"[.!?\n]", content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 10:
                return sentence
        return content[:200] if content else ""

    def _extract_outcome(self: "Kernle", content: str) -> Optional[str]:
        """Try to extract an outcome statement from content."""
        content_lower = content.lower()

        # Look for outcome indicators
        outcome_patterns = [
            r"(result(?:ed)?(?:\s+in)?|outcome|conclusion)[:\s]+(.+?)(?:\.|$)",
            r"(succeeded|failed|achieved|completed)[:\s]*(.+?)(?:\.|$)",
            r"(in the end|finally|ultimately)[,\s]+(.+?)(?:\.|$)",
        ]

        for pattern in outcome_patterns:
            match = re.search(pattern, content_lower)
            if match:
                return match.group(2).strip()[:200]

        return None

    def _extract_lessons(self: "Kernle", content: str) -> List[str]:
        """Extract lesson statements from content."""
        lessons = []
        content_lower = content.lower()

        # Look for lesson indicators
        lesson_patterns = [
            r"(?:learned|lesson|takeaway|insight)[:\s]+(.+?)(?:\.|$)",
            r"(?:realized|discovered|figured out)[:\s]+(.+?)(?:\.|$)",
            r"(?:key point|important)[:\s]+(.+?)(?:\.|$)",
        ]

        for pattern in lesson_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                lesson = match.strip()
                if len(lesson) >= 10 and lesson not in lessons:
                    lessons.append(lesson[:200])

        return lessons[:5]  # Limit to 5 lessons

    def _infer_outcome_type(self: "Kernle", content: str) -> str:
        """Infer outcome type from content."""
        content_lower = content.lower()

        # Check partial first since it may co-occur with success words
        if any(
            word in content_lower for word in ["partial", "partially", "mostly", "some progress"]
        ):
            return "partial"
        elif any(word in content_lower for word in ["failed", "failure", "blocked", "stuck"]):
            return "failure"
        elif any(
            word in content_lower
            for word in ["success", "succeeded", "achieved", "completed", "shipped"]
        ):
            return "success"
        else:
            return "unknown"

    def _extract_belief_statement(self: "Kernle", content: str) -> str:
        """Extract the core belief statement from content."""
        content = content.strip()

        # Try to find opinion phrases and extract the belief
        patterns = [
            r"i (?:think|believe|feel that)\s+(.+?)(?:\.|$)",
            r"(?:seems like|appears that)\s+(.+?)(?:\.|$)",
            r"(.+?)\s+(?:is better than|is worse than)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:500]

        # Fall back to first sentence
        return self._extract_first_sentence(content)

    def _infer_belief_type(self: "Kernle", content: str) -> str:
        """Infer belief type from content."""
        content_lower = content.lower()

        if any(word in content_lower for word in ["rule", "must", "should always", "never"]):
            return "rule"
        elif any(word in content_lower for word in ["prefer", "like", "favorite", "better"]):
            return "preference"
        elif any(word in content_lower for word in ["constraint", "limit", "cannot", "must not"]):
            return "constraint"
        elif any(word in content_lower for word in ["learned", "discovered", "realized"]):
            return "learned"
        else:
            return "fact"

    def _infer_note_type(self: "Kernle", content: str) -> str:
        """Infer note type from content."""
        content_lower = content.lower()

        # Check for quotes (has quoted text and attribution)
        if re.search(r'["\'].+["\']', content) and any(
            word in content_lower for word in ["said", "told", "mentioned"]
        ):
            return "quote"
        # Check for decisions
        elif any(
            word in content_lower for word in ["decided", "decision", "chose", "going to", "will"]
        ):
            return "decision"
        # Check for insights
        elif any(
            word in content_lower for word in ["insight", "realized", "noticed", "interesting"]
        ):
            return "insight"
        else:
            return "note"

    def _extract_speaker(self: "Kernle", content: str) -> Optional[str]:
        """Extract speaker name from a quote."""
        patterns = [
            r"(\w+)\s+said",
            r"(\w+)\s+told",
            r"(\w+)\s+mentioned",
            r"according to\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_reason(self: "Kernle", content: str) -> Optional[str]:
        """Extract reason from a decision."""
        patterns = [
            r"because\s+(.+?)(?:\.|$)",
            r"reason[:\s]+(.+?)(?:\.|$)",
            r"since\s+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]

        return None

    # === Suggestion Management ===

    def get_suggestions(
        self: "Kernle",
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
        min_confidence: Optional[float] = None,
        max_age_hours: Optional[float] = None,
        source_raw_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get memory suggestions.

        Args:
            status: Filter by status (pending, promoted, modified, rejected, dismissed, expired)
            memory_type: Filter by type (episode, belief, note)
            limit: Maximum suggestions to return
            min_confidence: Minimum confidence threshold
            max_age_hours: Only return suggestions created within this many hours
            source_raw_id: Filter to suggestions derived from this raw entry ID

        Returns:
            List of suggestion dicts
        """
        suggestions = self._storage.get_suggestions(
            status=status,
            memory_type=memory_type,
            limit=limit,
            min_confidence=min_confidence,
            max_age_hours=max_age_hours,
            source_raw_id=source_raw_id,
        )

        return [self._suggestion_to_dict(s) for s in suggestions]

    def get_suggestion(self: "Kernle", suggestion_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific suggestion by ID.

        Args:
            suggestion_id: ID of the suggestion

        Returns:
            Suggestion dict or None
        """
        suggestion = self._storage.get_suggestion(suggestion_id)
        if suggestion:
            return self._suggestion_to_dict(suggestion)
        return None

    def promote_suggestion(
        self: "Kernle",
        suggestion_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Promote a suggestion to a structured memory.

        Args:
            suggestion_id: ID of the suggestion to promote
            modifications: Optional modifications to apply before promotion

        Returns:
            ID of the created memory, or None if failed
        """
        # In strict mode, delegate to the stack's accept_suggestion which
        # handles provenance correctly (source_entity="kernle:suggestion-promotion").
        if getattr(self, "_strict", False):
            stack = getattr(self, "stack", None)
            if stack is not None and hasattr(stack, "accept_suggestion"):
                return stack.accept_suggestion(suggestion_id, modifications)

        suggestion = self._storage.get_suggestion(suggestion_id)
        if not suggestion or suggestion.status != "pending":
            return None

        # Apply modifications if provided
        content = suggestion.content.copy()
        if modifications:
            content.update(modifications)

        # Create the appropriate memory type
        memory_id = None
        memory_type = suggestion.memory_type

        # Preserve typed provenance when suggestions come from non-raw transitions.
        derived_from = _normalize_suggestion_provenance_refs(suggestion.source_raw_ids)

        if memory_type == "episode":
            memory_id = self.episode(
                objective=content.get("objective", ""),
                outcome=content.get("outcome", ""),
                lessons=content.get("lessons"),
                tags=["auto-suggested"],
                derived_from=derived_from,
                source_type="processing",
            )
        elif memory_type == "belief":
            memory_id = self.belief(
                statement=content.get("statement", ""),
                type=content.get("belief_type", "fact"),
                confidence=content.get("confidence", 0.7),
                derived_from=derived_from,
                source_type="processing",
            )
        elif memory_type == "note":
            memory_id = self.note(
                content=content.get("content", ""),
                type=content.get("note_type", "note"),
                speaker=content.get("speaker"),
                reason=content.get("reason"),
                tags=["auto-suggested"],
                derived_from=derived_from,
                source_type="processing",
            )
        elif memory_type == "goal":
            memory_id = self.goal(
                title=content.get("title", ""),
                description=content.get("description"),
                goal_type=content.get("goal_type", "task"),
                priority=content.get("priority", "medium"),
                derived_from=derived_from,
                source_type="processing",
            )
        elif memory_type == "value":
            memory_id = self.value(
                name=content.get("name", ""),
                statement=content.get("statement", ""),
                priority=content.get("priority", 50),
                derived_from=derived_from,
                source_type="processing",
            )
        elif memory_type == "relationship":
            memory_id = self.relationship(
                other_stack_id=content.get("entity_name", "unknown"),
                entity_type=content.get("entity_type"),
                interaction_type=content.get("relationship_type"),
                notes=content.get("notes"),
                derived_from=derived_from,
            )
        elif memory_type == "drive":
            memory_id = self.drive(
                drive_type=content.get("drive_type", "curiosity"),
                intensity=content.get("intensity", 0.5),
                focus_areas=content.get("focus_areas"),
                derived_from=derived_from,
                source_type="processing",
            )

        if memory_id:
            # Update suggestion status
            status = "modified" if modifications else "promoted"
            self._storage.update_suggestion_status(
                suggestion_id=suggestion_id,
                status=status,
                promoted_to=f"{memory_type}:{memory_id}",
            )

            # Mark only true raw refs as processed.
            for raw_ref in derived_from:
                ref_type, ref_id = raw_ref.split(":", 1)
                if ref_type != "raw":
                    continue
                self._storage.mark_raw_processed(
                    raw_id=ref_id,
                    processed_into=[f"{memory_type}:{memory_id}"],
                )

        return memory_id

    def reject_suggestion(
        self: "Kernle",
        suggestion_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Reject a suggestion.

        Args:
            suggestion_id: ID of the suggestion to reject
            reason: Optional reason for rejection

        Returns:
            True if rejected, False if failed
        """
        return self._storage.update_suggestion_status(
            suggestion_id=suggestion_id,
            status="rejected",
            resolution_reason=reason,
        )

    def accept_suggestion(
        self: "Kernle",
        suggestion_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Accept a suggestion and promote it to a structured memory.

        Alias for promote_suggestion with 'accepted' semantics.

        Args:
            suggestion_id: ID of the suggestion to accept
            modifications: Optional modifications to apply before promotion

        Returns:
            ID of the created memory, or None if failed
        """
        return self.promote_suggestion(suggestion_id, modifications)

    def dismiss_suggestion(
        self: "Kernle",
        suggestion_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Dismiss a suggestion (it will not be promoted).

        Sets status to 'dismissed' with an optional reason.

        Args:
            suggestion_id: ID of the suggestion to dismiss
            reason: Optional reason for dismissal

        Returns:
            True if dismissed, False if failed
        """
        return self._storage.update_suggestion_status(
            suggestion_id=suggestion_id,
            status="dismissed",
            resolution_reason=reason,
        )

    def expire_suggestions(
        self: "Kernle",
        max_age_hours: float = 168.0,
    ) -> List[str]:
        """Auto-dismiss pending suggestions older than max_age_hours.

        Args:
            max_age_hours: Age threshold in hours (default: 168 = 7 days)

        Returns:
            List of expired suggestion IDs
        """
        return self._storage.expire_suggestions(max_age_hours=max_age_hours)

    def _suggestion_to_dict(
        self: "Kernle",
        suggestion: MemorySuggestion,
    ) -> Dict[str, Any]:
        """Convert a suggestion to a dict representation."""
        return {
            "id": suggestion.id,
            "memory_type": suggestion.memory_type,
            "content": suggestion.content,
            "confidence": suggestion.confidence,
            "source_raw_ids": suggestion.source_raw_ids,
            "status": suggestion.status,
            "created_at": suggestion.created_at.isoformat() if suggestion.created_at else None,
            "resolved_at": suggestion.resolved_at.isoformat() if suggestion.resolved_at else None,
            "resolution_reason": suggestion.resolution_reason,
            "promoted_to": suggestion.promoted_to,
        }

    def extract_suggestions_from_unprocessed(
        self: "Kernle",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Extract suggestions from all unprocessed raw entries.

        Useful for batch processing raw entries that haven't been
        analyzed yet.

        Args:
            limit: Maximum raw entries to process

        Returns:
            List of all extracted suggestions
        """
        raw_entries = self._storage.list_raw(processed=False, limit=limit)
        all_suggestions = []

        for entry in raw_entries:
            suggestions = self.extract_suggestions(entry, auto_save=True)
            all_suggestions.extend([self._suggestion_to_dict(s) for s in suggestions])

        return all_suggestions
