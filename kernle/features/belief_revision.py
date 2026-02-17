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

    # Opposition word pairs for semantic contradiction detection
    # Format: (word, opposite) - both directions are checked
    _OPPOSITION_PAIRS = [
        # Frequency/Certainty
        ("always", "never"),
        ("sometimes", "never"),
        ("often", "rarely"),
        ("frequently", "seldom"),
        ("constantly", "occasionally"),
        # Modal verbs and necessity
        ("should", "shouldn't"),
        ("must", "mustn't"),
        ("can", "cannot"),
        ("will", "won't"),
        ("would", "wouldn't"),
        ("could", "couldn't"),
        # Preferences and attitudes
        ("like", "dislike"),
        ("love", "hate"),
        ("prefer", "avoid"),
        ("enjoy", "despise"),
        ("favor", "oppose"),
        ("want", "reject"),
        ("appreciate", "resent"),
        ("embrace", "shun"),
        # Value judgments
        ("good", "bad"),
        ("best", "worst"),
        ("important", "unnecessary"),
        ("essential", "optional"),
        ("critical", "trivial"),
        ("valuable", "worthless"),
        ("beneficial", "harmful"),
        ("helpful", "unhelpful"),
        ("useful", "useless"),
        # Comparatives
        ("more", "less"),
        ("better", "worse"),
        ("faster", "slower"),
        ("higher", "lower"),
        ("greater", "lesser"),
        ("stronger", "weaker"),
        ("easier", "harder"),
        ("simpler", "complex"),
        ("safer", "riskier"),
        ("cheaper", "expensive"),
        ("larger", "smaller"),
        ("longer", "shorter"),
        # Actions and states
        ("increase", "decrease"),
        ("improve", "worsen"),
        ("enhance", "diminish"),
        ("enable", "disable"),
        ("allow", "prevent"),
        ("support", "block"),
        ("accept", "reject"),
        ("approve", "disapprove"),
        ("agree", "disagree"),
        ("include", "exclude"),
        ("add", "remove"),
        ("create", "destroy"),
        # Truth values
        ("true", "false"),
        ("right", "wrong"),
        ("correct", "incorrect"),
        ("accurate", "inaccurate"),
        ("valid", "invalid"),
        # Quality descriptors
        ("efficient", "inefficient"),
        ("effective", "ineffective"),
        ("reliable", "unreliable"),
        ("stable", "unstable"),
        ("secure", "insecure"),
        ("safe", "dangerous"),
        # Recommendations
        ("recommended", "discouraged"),
        ("advisable", "inadvisable"),
        ("encouraged", "forbidden"),
        ("suggested", "prohibited"),
    ]

    # Negation prefixes that can flip meaning
    _NEGATION_PREFIXES = ["not", "no", "non", "un", "in", "dis", "anti", "counter"]

    # Stop words to exclude from topic overlap calculations
    _STOP_WORDS = frozenset(
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
            "was",
            "were",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "for",
            "of",
            "in",
            "on",
            "at",
            "by",
            "with",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "but",
            "if",
            "then",
            "because",
            "while",
            "although",
            "though",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "me",
            "you",
            "him",
            "she",
            "we",
            "they",
            "who",
            "which",
            "what",
            "when",
            "where",
            "why",
            "how",
        ]
    )

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

        Gating:
        - ``use_legacy_heuristics=true``: heuristic pattern matching (unchanged)
        - ``use_legacy_heuristics=false`` + no model: empty list
        - ``use_legacy_heuristics=false`` + model: inference-based detection

        Args:
            belief_statement: The statement to check for contradictions
            similarity_threshold: Minimum similarity score (0-1) for related beliefs
            limit: Maximum number of potential contradictions to return

        Returns:
            List of dicts with belief info and contradiction analysis
        """
        if self._use_legacy_heuristics():
            return self._find_contradictions_heuristic(
                belief_statement, similarity_threshold, limit
            )

        inference = self._get_inference()
        if inference is None:
            return []

        return self._find_contradictions_inference(
            belief_statement, inference, similarity_threshold, limit
        )

    def _find_contradictions_heuristic(
        self: "Kernle",
        belief_statement: str,
        similarity_threshold: float = 0.6,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find contradictions using heuristic pattern matching (legacy)."""
        # Search for semantically similar beliefs
        search_results = self._storage.search(
            belief_statement,
            limit=limit * 2,
            record_types=["belief"],  # Get more to filter
        )

        contradictions = []
        stmt_lower = belief_statement.lower().strip()

        for result in search_results:
            if result.record_type != "belief":
                continue

            # Filter by similarity threshold
            if result.score < similarity_threshold:
                continue

            belief = result.record
            belief_stmt_lower = belief.statement.lower().strip()

            # Skip exact matches
            if belief_stmt_lower == stmt_lower:
                continue

            # Check for contradiction patterns
            contradiction_type = None
            confidence = 0.0
            explanation = ""

            # Negation patterns
            negation_pairs = [
                ("never", "always"),
                ("should not", "should"),
                ("cannot", "can"),
                ("don't", "do"),
                ("avoid", "prefer"),
                ("reject", "accept"),
                ("false", "true"),
                ("dislike", "like"),
                ("hate", "love"),
                ("wrong", "right"),
                ("bad", "good"),
            ]

            for neg, pos in negation_pairs:
                if (neg in stmt_lower and pos in belief_stmt_lower) or (
                    pos in stmt_lower and neg in belief_stmt_lower
                ):
                    # Check word overlap for topic relevance
                    words_stmt = set(stmt_lower.split()) - {
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
                    }
                    words_belief = set(belief_stmt_lower.split()) - {
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
                    }
                    overlap = len(words_stmt & words_belief)

                    if overlap >= 2:
                        contradiction_type = "direct_negation"
                        confidence = min(0.5 + overlap * 0.1 + result.score * 0.2, 0.95)
                        explanation = f"Negation conflict: '{neg}' vs '{pos}' with {overlap} overlapping terms"
                        break

            # Comparative opposition (more/less, better/worse, etc.)
            if not contradiction_type:
                comparative_pairs = [
                    ("more", "less"),
                    ("better", "worse"),
                    ("faster", "slower"),
                    ("higher", "lower"),
                    ("greater", "lesser"),
                    ("stronger", "weaker"),
                    ("easier", "harder"),
                    ("simpler", "more complex"),
                    ("safer", "riskier"),
                    ("cheaper", "more expensive"),
                    ("larger", "smaller"),
                    ("longer", "shorter"),
                    ("increase", "decrease"),
                    ("improve", "worsen"),
                    ("enhance", "diminish"),
                ]
                for comp_a, comp_b in comparative_pairs:
                    if (comp_a in stmt_lower and comp_b in belief_stmt_lower) or (
                        comp_b in stmt_lower and comp_a in belief_stmt_lower
                    ):
                        # Check word overlap for topic relevance (need high overlap for comparatives)
                        words_stmt = set(stmt_lower.split()) - {
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
                            "than",
                            comp_a,
                            comp_b,
                        }
                        words_belief = set(belief_stmt_lower.split()) - {
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
                            "than",
                            comp_a,
                            comp_b,
                        }
                        overlap = len(words_stmt & words_belief)

                        if overlap >= 2:
                            contradiction_type = "comparative_opposition"
                            # Higher confidence for comparative oppositions with strong topic overlap
                            confidence = min(0.6 + overlap * 0.08 + result.score * 0.2, 0.95)
                            explanation = f"Comparative opposition: '{comp_a}' vs '{comp_b}' with {overlap} overlapping terms"
                            break

            # Preference conflicts
            if not contradiction_type:
                preference_pairs = [
                    ("prefer", "avoid"),
                    ("like", "dislike"),
                    ("enjoy", "hate"),
                    ("favor", "oppose"),
                    ("support", "reject"),
                    ("want", "don't want"),
                ]
                for pref, anti in preference_pairs:
                    if (pref in stmt_lower and anti in belief_stmt_lower) or (
                        anti in stmt_lower and pref in belief_stmt_lower
                    ):
                        words_stmt = set(stmt_lower.split()) - {
                            "i",
                            "the",
                            "a",
                            "an",
                            "to",
                            "and",
                            "or",
                        }
                        words_belief = set(belief_stmt_lower.split()) - {
                            "i",
                            "the",
                            "a",
                            "an",
                            "to",
                            "and",
                            "or",
                        }
                        overlap = len(words_stmt & words_belief)

                        if overlap >= 2:
                            contradiction_type = "preference_conflict"
                            confidence = min(0.4 + overlap * 0.1 + result.score * 0.2, 0.85)
                            explanation = f"Preference conflict: '{pref}' vs '{anti}'"
                            break

            if contradiction_type:
                contradictions.append(
                    {
                        "belief_id": belief.id,
                        "statement": belief.statement,
                        "confidence": belief.confidence,
                        "times_reinforced": belief.times_reinforced,
                        "is_active": belief.is_active,
                        "contradiction_type": contradiction_type,
                        "contradiction_confidence": round(confidence, 2),
                        "explanation": explanation,
                        "semantic_similarity": round(result.score, 2),
                    }
                )

        # Sort by contradiction confidence
        contradictions.sort(key=lambda x: x["contradiction_confidence"], reverse=True)
        return contradictions[:limit]

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

        Gating:
        - ``use_legacy_heuristics=true``: heuristic opposition detection (unchanged)
        - ``use_legacy_heuristics=false`` + no model: empty list
        - ``use_legacy_heuristics=false`` + model: delegates to find_contradictions inference

        Args:
            belief: The belief statement to check for contradictions
            similarity_threshold: Minimum similarity score (0-1) for related beliefs.
            limit: Maximum number of potential contradictions to return

        Returns:
            List of dicts with contradiction analysis
        """
        if not self._use_legacy_heuristics():
            # In inference mode, find_contradictions handles everything
            inference = self._get_inference()
            if inference is None:
                return []
            return self._find_contradictions_inference(
                belief, inference, similarity_threshold, limit
            )

        return self._find_semantic_contradictions_heuristic(belief, similarity_threshold, limit)

    def _find_semantic_contradictions_heuristic(
        self: "Kernle",
        belief: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find semantic contradictions using heuristic opposition detection (legacy)."""
        belief = self._validate_string_input(belief, "belief", 2000)

        # Search for semantically similar beliefs
        search_results = self._storage.search(
            belief,
            limit=limit * 3,
            record_types=["belief"],  # Get more to filter by threshold
        )

        contradictions = []
        belief_lower = belief.lower().strip()

        for result in search_results:
            if result.record_type != "belief":
                continue

            # Filter by similarity threshold
            if result.score < similarity_threshold:
                continue

            existing_belief = result.record
            existing_lower = existing_belief.statement.lower().strip()

            # Skip exact matches
            if existing_lower == belief_lower:
                continue

            # Skip inactive beliefs by default
            if not existing_belief.is_active:
                continue

            # Detect opposition
            opposition = self._detect_opposition(belief_lower, existing_lower)

            if opposition["score"] > 0:
                contradictions.append(
                    {
                        "belief_id": existing_belief.id,
                        "statement": existing_belief.statement,
                        "confidence": existing_belief.confidence,
                        "times_reinforced": existing_belief.times_reinforced,
                        "is_active": existing_belief.is_active,
                        "similarity_score": round(result.score, 3),
                        "opposition_score": round(opposition["score"], 3),
                        "opposition_type": opposition["type"],
                        "explanation": opposition["explanation"],
                    }
                )

        # Sort by combined score (similarity * opposition)
        contradictions.sort(
            key=lambda x: x["similarity_score"] * x["opposition_score"], reverse=True
        )
        return contradictions[:limit]

    def _detect_opposition(
        self: "Kernle",
        stmt1: str,
        stmt2: str,
    ) -> Dict[str, Any]:
        """Detect if two similar statements have opposing meanings.

        Uses multiple heuristics:
        1. Direct opposition words (always/never, good/bad, etc.)
        2. Negation patterns (is vs is not, should vs shouldn't)
        3. Sentiment/valence indicators

        Args:
            stmt1: First statement (lowercase)
            stmt2: Second statement (lowercase)

        Returns:
            Dict with:
                - score: Opposition strength (0-1), 0 means no opposition detected
                - type: Type of opposition detected
                - explanation: Human-readable explanation
        """
        result = {"score": 0.0, "type": "none", "explanation": ""}

        words1 = set(stmt1.split())
        words2 = set(stmt2.split())

        # Calculate topic overlap (excluding stop words and opposition words)
        content_words1 = words1 - self._STOP_WORDS
        content_words2 = words2 - self._STOP_WORDS
        overlap = content_words1 & content_words2
        overlap_count = len(overlap)

        # Need some topic overlap to be a meaningful contradiction
        if overlap_count < 1:
            return result

        # 1. Check for direct opposition word pairs
        for word_a, word_b in self._OPPOSITION_PAIRS:
            # Check both directions
            if (word_a in stmt1 and word_b in stmt2) or (word_b in stmt1 and word_a in stmt2):
                # Verify words are used in meaningful context (not just substrings)
                a_in_1 = word_a in words1
                b_in_2 = word_b in words2
                b_in_1 = word_b in words1
                a_in_2 = word_a in words2

                if (a_in_1 and b_in_2) or (b_in_1 and a_in_2):
                    score = min(0.5 + overlap_count * 0.1, 0.95)
                    return {
                        "score": score,
                        "type": "opposition_words",
                        "explanation": f"Opposing terms '{word_a}' vs '{word_b}' with {overlap_count} shared topic words: {', '.join(list(overlap)[:3])}",
                    }

        # 2. Check for negation patterns
        negation_found = self._check_negation_pattern(stmt1, stmt2)
        if negation_found:
            score = min(0.4 + overlap_count * 0.1, 0.85)
            return {
                "score": score,
                "type": "negation",
                "explanation": f"Negation pattern detected with {overlap_count} shared topic words: {', '.join(list(overlap)[:3])}",
            }

        # 3. Check for sentiment opposition using positive/negative indicator words
        sentiment_opposition = self._check_sentiment_opposition(stmt1, stmt2)
        if sentiment_opposition["detected"]:
            score = min(0.3 + overlap_count * 0.1, 0.75)
            return {
                "score": score,
                "type": "sentiment_opposition",
                "explanation": f"Sentiment opposition: '{sentiment_opposition['word1']}' vs '{sentiment_opposition['word2']}' with topic overlap",
            }

        return result

    def _check_negation_pattern(self: "Kernle", stmt1: str, stmt2: str) -> bool:
        """Check if one statement negates the other.

        Looks for patterns like:
        - "X is good" vs "X is not good"
        - "should use X" vs "should not use X"
        - "I like X" vs "I don't like X"
        """
        # Common negation patterns
        negation_patterns = [
            ("is not", "is"),
            ("is", "is not"),
            ("are not", "are"),
            ("are", "are not"),
            ("do not", "do"),
            ("do", "do not"),
            ("does not", "does"),
            ("does", "does not"),
            ("should not", "should"),
            ("should", "should not"),
            ("shouldn't", "should"),
            ("should", "shouldn't"),
            ("can not", "can"),
            ("can", "can not"),
            ("cannot", "can"),
            ("can", "cannot"),
            ("can't", "can"),
            ("can", "can't"),
            ("won't", "will"),
            ("will", "won't"),
            ("don't", "do"),
            ("do", "don't"),
            ("doesn't", "does"),
            ("does", "doesn't"),
            ("isn't", "is"),
            ("is", "isn't"),
            ("aren't", "are"),
            ("are", "aren't"),
            ("wasn't", "was"),
            ("was", "wasn't"),
            ("weren't", "were"),
            ("were", "weren't"),
            ("not recommended", "recommended"),
            ("recommended", "not recommended"),
            ("not important", "important"),
            ("important", "not important"),
            ("no need", "need"),
            ("need", "no need"),
        ]

        for pattern_a, pattern_b in negation_patterns:
            if pattern_a in stmt1 and pattern_b in stmt2:
                # Make sure pattern_a is not a substring of pattern_b in stmt1
                if pattern_b not in stmt1 or stmt1.index(pattern_a) != stmt1.find(pattern_b):
                    return True
            if pattern_b in stmt1 and pattern_a in stmt2:
                if pattern_a not in stmt1 or stmt1.index(pattern_b) != stmt1.find(pattern_a):
                    return True

        return False

    def _check_sentiment_opposition(
        self: "Kernle",
        stmt1: str,
        stmt2: str,
    ) -> Dict[str, Any]:
        """Check for sentiment/valence opposition between statements.

        Looks for one statement having positive sentiment words and
        the other having negative sentiment words about the same topic.
        """
        positive_words = {
            "good",
            "great",
            "excellent",
            "important",
            "essential",
            "valuable",
            "helpful",
            "useful",
            "beneficial",
            "necessary",
            "crucial",
            "vital",
            "effective",
            "efficient",
            "reliable",
            "fast",
            "quick",
            "easy",
            "simple",
            "clear",
            "clean",
            "safe",
            "secure",
            "stable",
            "robust",
            "powerful",
            "flexible",
            "scalable",
            "maintainable",
            "readable",
            "elegant",
            "beautiful",
            "brilliant",
            "amazing",
            "wonderful",
            "love",
            "like",
            "enjoy",
            "prefer",
            "appreciate",
            "recommend",
            "success",
            "win",
            "gain",
            "improve",
            "enhance",
            "boost",
        }

        negative_words = {
            "bad",
            "poor",
            "terrible",
            "unimportant",
            "unnecessary",
            "worthless",
            "unhelpful",
            "useless",
            "harmful",
            "optional",
            "trivial",
            "minor",
            "ineffective",
            "inefficient",
            "unreliable",
            "slow",
            "sluggish",
            "hard",
            "complex",
            "confusing",
            "messy",
            "dangerous",
            "insecure",
            "unstable",
            "fragile",
            "weak",
            "rigid",
            "limited",
            "unmaintainable",
            "unreadable",
            "ugly",
            "awful",
            "horrible",
            "terrible",
            "disaster",
            "hate",
            "dislike",
            "avoid",
            "reject",
            "despise",
            "discourage",
            "failure",
            "loss",
            "degrade",
            "diminish",
            "reduce",
            "slows",
            "slow",
            "slowdown",
            "overhead",
            "bloat",
            "bloated",
            "waste",
            "wasted",
            "wastes",
            "wasting",
        }

        words1 = set(stmt1.split())
        words2 = set(stmt2.split())

        pos1 = words1 & positive_words
        neg1 = words1 & negative_words
        pos2 = words2 & positive_words
        neg2 = words2 & negative_words

        # Check for cross-sentiment: positive in one, negative in other
        if pos1 and neg2:
            return {
                "detected": True,
                "word1": list(pos1)[0],
                "word2": list(neg2)[0],
            }
        if neg1 and pos2:
            return {
                "detected": True,
                "word1": list(neg1)[0],
                "word2": list(pos2)[0],
            }

        return {"detected": False, "word1": "", "word2": ""}

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

        Gating:
        - ``use_legacy_heuristics=true``: word-overlap heuristic (unchanged)
        - ``use_legacy_heuristics=false`` + no model: empty result
        - ``use_legacy_heuristics=false`` + model: inference-based analysis

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

        if self._use_legacy_heuristics():
            return self._revise_beliefs_heuristic(episode_id, episode)

        inference = self._get_inference()
        if inference is None:
            return {"episode_id": episode_id, **self._EMPTY_REVISION_RESULT}

        return self._revise_beliefs_inference(episode_id, episode, inference)

    def _revise_beliefs_heuristic(self: "Kernle", episode_id: str, episode) -> Dict[str, Any]:
        """Revise beliefs from episode using word-overlap heuristic (legacy)."""
        result = {
            "episode_id": episode_id,
            "reinforced": [],
            "contradicted": [],
            "suggested_new": [],
        }

        # Build evidence text from episode
        evidence_parts = []
        if episode.outcome_type == "success":
            evidence_parts.append(f"Successfully: {episode.objective}")
        elif episode.outcome_type == "failure":
            evidence_parts.append(f"Failed: {episode.objective}")

        evidence_parts.append(episode.outcome)

        if episode.lessons:
            evidence_parts.extend(episode.lessons)

        evidence_text = " ".join(evidence_parts)

        # Get all active beliefs
        beliefs = self._storage.get_beliefs(limit=500)

        for belief in beliefs:
            belief_stmt_lower = belief.statement.lower()
            evidence_lower = evidence_text.lower()

            # Check for word overlap
            belief_words = set(belief_stmt_lower.split()) - {
                "i",
                "the",
                "a",
                "an",
                "to",
                "and",
                "or",
                "is",
                "are",
                "should",
                "can",
            }
            evidence_words = set(evidence_lower.split()) - {
                "i",
                "the",
                "a",
                "an",
                "to",
                "and",
                "or",
                "is",
                "are",
                "should",
                "can",
            }
            overlap = belief_words & evidence_words

            if len(overlap) < 2:
                continue  # Not related enough

            # Determine if evidence supports or contradicts
            is_supporting = False
            is_contradicting = False

            if episode.outcome_type == "success":
                # Success supports "should" beliefs about what worked
                if any(
                    word in belief_stmt_lower
                    for word in ["should", "prefer", "good", "important", "effective"]
                ):
                    is_supporting = True
                # Success contradicts "avoid" beliefs about what worked
                elif any(word in belief_stmt_lower for word in ["avoid", "never", "don't", "bad"]):
                    is_contradicting = True

            elif episode.outcome_type == "failure":
                # Failure contradicts "should" beliefs about what failed
                if any(
                    word in belief_stmt_lower
                    for word in ["should", "prefer", "good", "important", "effective"]
                ):
                    is_contradicting = True
                # Failure supports "avoid" beliefs
                elif any(word in belief_stmt_lower for word in ["avoid", "never", "don't", "bad"]):
                    is_supporting = True

            if is_supporting:
                # Reinforce the belief with episode as evidence
                self.reinforce_belief(
                    belief.id,
                    evidence_source=f"episode:{episode_id}",
                    reason=f"Confirmed by episode: {episode.objective[:50]}",
                )
                result["reinforced"].append(
                    {
                        "belief_id": belief.id,
                        "statement": belief.statement,
                        "overlap": list(overlap),
                        "evidence_source": f"episode:{episode_id}",
                    }
                )

            elif is_contradicting:
                # Flag as potentially contradicted
                result["contradicted"].append(
                    {
                        "belief_id": belief.id,
                        "statement": belief.statement,
                        "overlap": list(overlap),
                        "evidence": evidence_text[:200],
                    }
                )

        # Suggest new beliefs from lessons
        if episode.lessons:
            for lesson in episode.lessons:
                # Check if a similar belief already exists
                existing = self._storage.find_belief(lesson)
                if not existing:
                    # Check for similar beliefs via search
                    similar = self._storage.search(lesson, limit=3, record_types=["belief"])
                    if not any(r.score > 0.9 for r in similar):
                        result["suggested_new"].append(
                            {
                                "statement": lesson,
                                "source_episode": episode_id,
                                "suggested_confidence": (
                                    0.7 if episode.outcome_type == "success" else 0.6
                                ),
                            }
                        )

        # Link episode to affected beliefs
        for reinforced in result["reinforced"]:
            belief = next((b for b in beliefs if b.id == reinforced["belief_id"]), None)
            if belief:
                source_eps = belief.source_episodes or []
                if episode_id not in source_eps:
                    belief.source_episodes = source_eps + [episode_id]
                    self._write_backend.save_belief(belief)

        return result

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
