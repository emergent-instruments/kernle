"""Suggestion extraction stack component.

Extracts memory suggestions from raw entries using inference.
Returns empty results when inference is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult
from kernle.types import MemorySuggestion

logger = logging.getLogger(__name__)


class SuggestionComponent:
    """Suggestion extraction component.

    Extracts memory suggestions from raw entries during maintenance.
    Uses inference for classification; returns empty results when
    inference is unavailable.
    """

    name = "suggestions"
    version = "1.0.0"
    required = False
    needs_inference = True
    inference_scope = "capable"
    priority = 210

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
        """Called by SQLiteStack after attach to provide storage access."""
        self._storage = storage

    # ---- Lifecycle Hooks ----

    def on_save(self, memory_type: str, memory_id: str, memory: Any) -> Any:
        return None

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        return results

    def on_load(self, context: Dict[str, Any]) -> None:
        pass

    def on_maintenance(self) -> Dict[str, Any]:
        """Extract suggestions from unprocessed raw entries."""
        if self._storage is None:
            logger.debug("SuggestionComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        raw_entries = self._storage.list_raw(processed=False, limit=50)
        total_suggestions = 0
        deduplicated = 0

        for entry in raw_entries:
            seen_signatures = self._get_existing_suggestion_signatures(entry.id)
            created_refs: List[str] = []

            suggestions = self._extract_suggestions(entry)
            for suggestion in suggestions:
                signature = self._suggestion_signature(entry.id, suggestion)
                if signature in seen_signatures:
                    deduplicated += 1
                    continue

                self._storage.save_suggestion(suggestion)
                seen_signatures.add(signature)
                created_refs.append(f"{suggestion.memory_type}:{suggestion.id}")
                total_suggestions += 1

            try:
                self._storage.mark_raw_processed(
                    raw_id=entry.id,
                    processed_into=created_refs,
                )
            except Exception as e:
                logger.debug(
                    "SuggestionComponent: failed to mark raw entry %s processed: %s",
                    entry.id,
                    e,
                    exc_info=True,
                )

        result: Dict[str, Any] = {
            "raw_entries_processed": len(raw_entries),
            "suggestions_extracted": total_suggestions,
            "deduplicated_suggestions": deduplicated,
        }

        if self._inference is None and total_suggestions == 0:
            logger.debug("SuggestionComponent: no inference, pattern-only extraction")
            result["inference_available"] = False
        else:
            result["inference_available"] = self._inference is not None

        return result

    def _get_existing_suggestion_signatures(self, raw_id: str) -> set:
        """Build dedupe signatures for suggestions already created from raw_id."""
        signatures: set[str] = set()
        if self._storage is None:
            return signatures

        try:
            existing = self._storage.get_suggestions(source_raw_id=raw_id, limit=200)
        except Exception as e:
            logger.debug(
                "SuggestionComponent: could not fetch existing suggestions for %s: %s",
                raw_id,
                e,
                exc_info=True,
            )
            return signatures

        for suggestion in existing:
            try:
                signatures.add(self._suggestion_signature(raw_id, suggestion))
            except Exception as exc:
                logger.debug(
                    "Swallowed %s computing suggestion signature: %s",
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                continue
        return signatures

    def _suggestion_signature(self, raw_entry_id: str, suggestion: Any) -> str:
        """Create a deterministic dedupe signature for a suggestion."""
        content = getattr(suggestion, "content", {}) or {}
        confidence = getattr(suggestion, "confidence", 0.0) or 0.0
        memory_type = getattr(suggestion, "memory_type", "")
        signature_payload = json.dumps(
            {
                "raw_id": raw_entry_id,
                "memory_type": memory_type,
                "confidence_bucket": round(float(confidence), 3),
                "content": content,
            },
            sort_keys=True,
        )
        return hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()

    # ---- Core Logic ----

    def _extract_suggestions(self, raw_entry: Any) -> List[MemorySuggestion]:
        """Extract memory suggestions from a raw entry.

        Uses inference when available; returns empty list otherwise.
        """
        if self._inference is None:
            return []
        return self._extract_suggestions_inference(raw_entry)

    def _extract_suggestions_inference(self, raw_entry: Any) -> List[MemorySuggestion]:
        """Extract suggestions using inference (non-legacy path)."""
        from kernle.core.inference_utils import parse_inference_json

        content = getattr(raw_entry, "blob", None) or getattr(raw_entry, "content", None) or ""
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
            raw = self._inference.infer(
                prompt=prompt,
                system="You are a memory classification system. Return only valid JSON.",
            )
        except Exception:
            logger.debug("SuggestionComponent: inference call failed", exc_info=True)
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
            suggestion = self._make_suggestion(raw_entry, memory_type, confidence)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _make_suggestion(
        self, raw_entry: Any, memory_type: str, confidence: float
    ) -> Optional[MemorySuggestion]:
        """Create a suggestion for the given memory type."""
        content_text = getattr(raw_entry, "blob", None) or getattr(raw_entry, "content", None) or ""
        if len(content_text.strip()) < 10:
            return None

        if memory_type == "episode":
            content_dict = {
                "objective": content_text[:200].strip(),
                "outcome": "Extracted from raw capture",
                "outcome_type": "unknown",
            }
        elif memory_type == "belief":
            content_dict = {
                "statement": content_text[:500].strip(),
                "belief_type": "fact",
                "confidence": min(0.8, confidence),
            }
        else:
            content_dict = {
                "content": content_text.strip(),
                "note_type": "note",
            }

        return MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id=self._stack_id or "",
            memory_type=memory_type,
            content=content_dict,
            confidence=confidence,
            source_raw_ids=[raw_entry.id],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
