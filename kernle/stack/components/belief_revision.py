"""Belief revision stack component.

Provides automatic contradiction detection when saving beliefs and tracks
stale belief patterns during maintenance. Helps maintain belief coherence
by flagging potential conflicts before they're committed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

# Top opposition pairs for contradiction detection
# These are the most important semantic oppositions to check
OPPOSITION_PAIRS = [
    # Frequency/Certainty
    ("always", "never"),
    ("should", "shouldn't"),
    ("good", "bad"),
    # Preferences
    ("important", "unnecessary"),
    ("prefer", "avoid"),
    ("like", "dislike"),
    # Comparatives
    ("more", "less"),
    ("better", "worse"),
    ("true", "false"),
    # Actions
    ("increase", "decrease"),
    ("improve", "worsen"),
    ("enable", "disable"),
    ("allow", "prevent"),
    ("accept", "reject"),
    # Safety and quality
    ("safe", "dangerous"),
    ("efficient", "inefficient"),
    ("reliable", "unreliable"),
    # Recommendations
    ("recommended", "discouraged"),
    ("love", "hate"),
    ("support", "block"),
]

# Stop words to exclude from topic overlap calculations
# These common words don't help identify meaningful topic overlap
STOP_WORDS = frozenset(
    [
        "i",
        "the",
        "a",
        "an",
        "to",
        "and",
        "or",
        "is",
        "are",
        "that",
        "this",
        "it",
        "be",
        "for",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
    ]
)


def detect_opposition(stmt1: str, stmt2: str) -> Dict[str, Any]:
    """Detect if two similar statements have opposing meanings.

    This function checks if two belief statements contain opposing words
    from the OPPOSITION_PAIRS list and have enough topic overlap to be
    considered a meaningful contradiction.

    Args:
        stmt1: First statement (should be lowercase)
        stmt2: Second statement (should be lowercase)

    Returns:
        Dict containing:
            - score: Opposition strength (0.0-1.0), 0 means no opposition detected
            - type: Type of opposition detected ("opposition_words" or "none")
            - explanation: Human-readable explanation of the detected opposition
            - topic_overlap: Number of shared content words between statements

    Example:
        >>> detect_opposition("testing is good", "testing is bad")
        {
            "score": 0.6,
            "type": "opposition_words",
            "explanation": "Opposing terms 'good' vs 'bad' with 1 shared topic words: testing",
            "topic_overlap": 1
        }
    """
    # Initialize the result with no opposition detected
    result = {
        "score": 0.0,
        "type": "none",
        "explanation": "",
        "topic_overlap": 0,
    }

    # Split statements into word sets for analysis
    words1 = set(stmt1.split())
    words2 = set(stmt2.split())

    # Calculate topic overlap by removing stop words
    # This tells us if the statements are about the same topic
    content_words1 = words1 - STOP_WORDS
    content_words2 = words2 - STOP_WORDS
    overlap = content_words1 & content_words2
    overlap_count = len(overlap)

    # Store overlap count for caller's use
    result["topic_overlap"] = overlap_count

    # We need at least 1 shared content word to consider this a meaningful contradiction
    # If statements don't share any topic words, they're probably about different things
    if overlap_count < 1:
        return result

    # Check each opposition pair to see if one word appears in stmt1 and its opposite in stmt2
    for word_a, word_b in OPPOSITION_PAIRS:
        # Check both directions: (word_a in stmt1 and word_b in stmt2) OR (word_b in stmt1 and word_a in stmt2)
        if (word_a in stmt1 and word_b in stmt2) or (word_b in stmt1 and word_a in stmt2):
            # Verify the words are actual tokens (not just substrings)
            # For example, "test" shouldn't match "testing"
            a_in_1 = word_a in words1
            b_in_2 = word_b in words2
            b_in_1 = word_b in words1
            a_in_2 = word_a in words2

            # If we found a real word-level opposition match
            if (a_in_1 and b_in_2) or (b_in_1 and a_in_2):
                # Calculate confidence score based on topic overlap
                # Base score of 0.5, plus 0.1 for each overlapping word, capped at 0.95
                score = min(0.5 + overlap_count * 0.1, 0.95)

                # Build explanation with the first few overlapping words as examples
                overlap_examples = ", ".join(list(overlap)[:3])

                return {
                    "score": score,
                    "type": "opposition_words",
                    "explanation": f"Opposing terms '{word_a}' vs '{word_b}' with {overlap_count} shared topic words: {overlap_examples}",
                    "topic_overlap": overlap_count,
                }

    # No opposition detected
    return result


class BeliefRevisionComponent:
    """Belief revision component.

    Automatically detects contradictions when beliefs are saved and tracks
    stale beliefs during maintenance. Helps maintain belief coherence by
    catching potential conflicts early.

    When a new belief is saved, this component:
    1. Searches for semantically similar existing beliefs
    2. Applies opposition detection to find contradictions
    3. Returns contradiction metadata for the stack to handle

    During maintenance, this component:
    1. Identifies stale beliefs (low confidence, never reinforced)
    2. Counts active beliefs for health monitoring
    3. Returns statistics for diagnostics
    """

    name = "belief_revision"
    version = "1.0.0"
    required = False
    needs_inference = False
    inference_scope = "none"
    priority = 200  # After trust (100), before consolidation (300)

    def __init__(self) -> None:
        """Initialize the belief revision component.

        Sets up the component with no stack attached. The stack will call
        attach() to bind this component to a specific stack instance.
        """
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._storage: Optional[Any] = None

    def attach(self, stack_id: str, inference: Optional[InferenceService] = None) -> None:
        """Attach this component to a specific stack.

        Called by the stack when the component is registered. Stores the
        stack ID and optional inference service for later use.

        Args:
            stack_id: The unique identifier for the stack
            inference: Optional inference service for advanced detection
        """
        self._stack_id = stack_id
        self._inference = inference

    def detach(self) -> None:
        """Detach this component from its stack.

        Clears all references to stack resources. Called when the component
        is being removed or the stack is being destroyed.
        """
        self._stack_id = None
        self._inference = None
        self._storage = None

    def set_inference(self, inference: Optional[InferenceService]) -> None:
        """Update the inference service.

        Called by the stack if the inference service changes after attachment.

        Args:
            inference: The new inference service to use (or None to disable)
        """
        self._inference = inference

    def set_storage(self, storage: Any) -> None:
        """Set the storage backend for this component.

        Called by Stack after attach() to provide access to the storage
        layer. This allows the component to query existing beliefs and memories.

        Args:
            storage: The storage backend (typically SQLiteStorage)
        """
        self._storage = storage

    # ---- Lifecycle Hooks ----

    def on_save(self, memory_type: str, memory_id: str, memory: Any) -> Any:
        """Check for contradictions when saving a belief.

        This is the main lifecycle hook for belief revision. When a new belief
        is being saved, we:
        1. Search for semantically similar existing beliefs
        2. Check each similar belief for opposition patterns
        3. Return contradiction metadata if any found

        Args:
            memory_type: The type of memory being saved (we only act on "belief")
            memory_id: The unique identifier for this memory
            memory: The memory object being saved (must have a "statement" attribute)

        Returns:
            Dict with {"contradictions": [...]} if any found, else None.
            Each contradiction dict contains:
                - belief_id: ID of the contradicting belief
                - statement: The contradicting belief's statement
                - opposition_score: Strength of the opposition (0.0-1.0)
                - opposition_type: Type of opposition detected
                - explanation: Human-readable explanation
                - topic_overlap: Number of shared content words
        """
        # Only process beliefs, skip all other memory types
        if memory_type != "belief":
            return None

        # If we don't have storage, we can't search for contradictions
        if self._storage is None:
            logger.debug("BeliefRevisionComponent: no storage, skipping contradiction check")
            return None

        # Extract the belief statement from the memory object
        # Use getattr with default to handle cases where statement might not exist
        statement = getattr(memory, "statement", "")
        if not statement or not statement.strip():
            # No statement to check, nothing to do
            return None

        # Search for semantically similar beliefs
        # We search more broadly than we need (limit * 2) to ensure we find all
        # potential contradictions even if some are filtered out
        try:
            search_results = self._storage.search(
                statement,
                limit=20,  # Get up to 20 similar beliefs to check
                record_types=["belief"],  # Only search beliefs
            )
        except Exception as e:
            # If search fails, log and continue without contradiction detection
            logger.warning(f"BeliefRevisionComponent: search failed: {e}")
            return None

        # Track any contradictions we find
        contradictions = []
        statement_lower = statement.lower().strip()

        # Check each similar belief for opposition
        for result in search_results:
            # Double-check the record type (should always be belief from our search)
            if result.record_type != "belief":
                continue

            # Get the existing belief from the search result
            existing_belief = result.record

            # Extract the belief's statement
            existing_statement = getattr(existing_belief, "statement", "")
            if not existing_statement:
                continue

            existing_lower = existing_statement.lower().strip()

            # Skip exact matches (same statement)
            if existing_lower == statement_lower:
                continue

            # Apply opposition detection to see if these beliefs contradict
            opposition = detect_opposition(statement_lower, existing_lower)

            # If we detected opposition (score > 0), record it
            if opposition["score"] > 0:
                contradictions.append(
                    {
                        "belief_id": existing_belief.id,
                        "statement": existing_statement,
                        "confidence": getattr(existing_belief, "confidence", 0.8),
                        "times_reinforced": getattr(existing_belief, "times_reinforced", 0),
                        "is_active": getattr(existing_belief, "is_active", True),
                        "opposition_score": round(opposition["score"], 3),
                        "opposition_type": opposition["type"],
                        "explanation": opposition["explanation"],
                        "topic_overlap": opposition["topic_overlap"],
                    }
                )

        # If we found any contradictions, return them for the stack to handle
        if contradictions:
            # Sort by opposition score (highest first) so most likely contradictions are first
            contradictions.sort(key=lambda x: x["opposition_score"], reverse=True)
            return {"contradictions": contradictions}

        # No contradictions found
        return None

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Process search results (no-op for belief revision).

        This component doesn't modify search results, so we pass them through
        unchanged.

        Args:
            query: The search query string
            results: The list of search results from the storage layer

        Returns:
            The unmodified results list
        """
        return results

    def on_load(self, context: Dict[str, Any]) -> None:
        """Contribute to working memory on load (no-op for belief revision).

        This component doesn't add anything to working memory, so this is
        a pass-through.

        Args:
            context: The working memory context being built
        """
        pass

    def on_maintenance(self) -> Dict[str, Any]:
        """Check for stale beliefs during maintenance.

        This runs periodically to identify beliefs that may need attention:
        1. Stale beliefs: low confidence (<0.3) and never reinforced
        2. Active beliefs: total count of active beliefs

        Returns:
            Dict with maintenance statistics:
                - stale_beliefs: Number of beliefs that are stale
                - active_beliefs: Total number of active beliefs
                - skipped: True if no storage (with reason)
        """
        # If we don't have storage, we can't check beliefs
        if self._storage is None:
            logger.debug("BeliefRevisionComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        # Try to get all beliefs from storage
        try:
            # Get all active beliefs (default limit may vary by storage implementation)
            beliefs = self._storage.get_beliefs()
        except AttributeError:
            # Storage doesn't have get_beliefs method
            logger.warning("BeliefRevisionComponent: storage doesn't support get_beliefs")
            return {"skipped": True, "reason": "no_get_beliefs"}
        except Exception as e:
            # Some other error occurred
            logger.error(f"BeliefRevisionComponent: error getting beliefs: {e}")
            return {"skipped": True, "reason": f"error: {e}"}

        # Count stale and active beliefs
        stale_count = 0
        active_count = 0

        for belief in beliefs:
            # Count active beliefs
            is_active = getattr(belief, "is_active", True)
            if is_active:
                active_count += 1

                # Check if this belief is stale
                # A belief is stale if:
                # 1. It has low confidence (<0.3)
                # 2. It has never been reinforced (times_reinforced == 0)
                confidence = getattr(belief, "confidence", 1.0)
                times_reinforced = getattr(belief, "times_reinforced", 0)

                if confidence < 0.3 and times_reinforced == 0:
                    stale_count += 1

        # Return statistics for the maintenance report
        return {
            "stale_beliefs": stale_count,
            "active_beliefs": active_count,
        }
