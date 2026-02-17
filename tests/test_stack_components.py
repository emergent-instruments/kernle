"""Tests for stack components — StackComponentProtocol implementations.

Tests cover:
- Protocol conformance (isinstance check against StackComponentProtocol)
- Attach/detach lifecycle
- Graceful degradation without inference for inference-needing components
- on_maintenance() behavior with and without storage
- set_storage() integration
- Component-specific logic (emotion detection, salience calculation, etc.)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from kernle.protocols import InferenceService, SearchResult, StackComponentProtocol
from kernle.stack.components import (
    AnxietyComponent,
    ConsolidationComponent,
    EmotionalTaggingComponent,
    ForgettingComponent,
    KnowledgeComponent,
    MetaMemoryComponent,
    SuggestionComponent,
)
from kernle.stack.components.suggestions import EPISODE_PATTERNS

# ============================================================================
# Helpers
# ============================================================================


def _make_mock_storage() -> MagicMock:
    """Create a mock storage backend with common method stubs."""
    storage = MagicMock()
    storage.get_episodes.return_value = []
    storage.get_beliefs.return_value = []
    storage.get_values.return_value = []
    storage.get_goals.return_value = []
    storage.get_notes.return_value = []
    storage.get_drives.return_value = []
    storage.get_relationships.return_value = []
    storage.get_memory.return_value = None
    storage.get_forgetting_candidates.return_value = []
    storage.get_current_epoch.return_value = None
    storage.list_raw.return_value = []
    storage.save_suggestion.return_value = "suggestion-id"
    storage.get_stack_setting.return_value = None
    return storage


def _make_mock_inference() -> MagicMock:
    """Create a mock inference service."""
    inference = MagicMock(spec=InferenceService)
    inference.infer.return_value = "mock response"
    inference.embed.return_value = [0.1] * 128
    return inference


def _make_mock_episode(
    id: str = "ep-1",
    objective: str = "Test objective",
    outcome: str = "Test outcome",
    outcome_type: str = "success",
    lessons: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    strength: float = 1.0,
    emotional_valence: float = 0.0,
    emotional_arousal: float = 0.0,
    emotional_tags: Optional[List[str]] = None,
    confidence: float = 0.8,
    created_at: Optional[datetime] = None,
) -> MagicMock:
    ep = MagicMock()
    ep.id = id
    ep.objective = objective
    ep.outcome = outcome
    ep.outcome_type = outcome_type
    ep.lessons = lessons or []
    ep.tags = tags or []
    ep.strength = strength
    ep.emotional_valence = emotional_valence
    ep.emotional_arousal = emotional_arousal
    ep.emotional_tags = emotional_tags or []
    ep.confidence = confidence
    ep.created_at = created_at or datetime.now(timezone.utc)
    ep.context_tags = []
    return ep


def _make_mock_belief(
    id: str = "belief-1",
    statement: str = "Test belief",
    confidence: float = 0.8,
    belief_type: str = "fact",
) -> MagicMock:
    b = MagicMock()
    b.id = id
    b.statement = statement
    b.confidence = confidence
    b.belief_type = belief_type
    return b


def _make_mock_value(
    id: str = "value-1",
    name: str = "Test value",
    statement: str = "Value statement",
    confidence: float = 0.9,
) -> MagicMock:
    v = MagicMock()
    v.id = id
    v.name = name
    v.statement = statement
    v.confidence = confidence
    return v


def _make_mock_memory(
    confidence: float = 0.8,
    times_accessed: int = 5,
    last_accessed: Optional[datetime] = None,
    created_at: Optional[datetime] = None,
    is_protected: bool = False,
    last_verified: Optional[datetime] = None,
    goal_type: str = "task",
) -> MagicMock:
    record = MagicMock()
    record.id = "mem-1"
    record.confidence = confidence
    record.times_accessed = times_accessed
    record.last_accessed = last_accessed or datetime.now(timezone.utc)
    record.created_at = created_at or datetime.now(timezone.utc)
    record.is_protected = is_protected
    record.last_verified = last_verified
    record.goal_type = goal_type
    return record


def _make_mock_raw_entry(
    id: str = "raw-1",
    content: str = "I completed the migration task successfully",
    blob: Optional[str] = None,
    processed: bool = False,
) -> MagicMock:
    entry = MagicMock()
    entry.id = id
    entry.content = content
    entry.blob = blob
    entry.processed = processed
    return entry


# ============================================================================
# All components: list for parametrized tests
# ============================================================================

ALL_COMPONENTS = [
    ForgettingComponent,
    AnxietyComponent,
    MetaMemoryComponent,
    ConsolidationComponent,
    EmotionalTaggingComponent,
    SuggestionComponent,
    KnowledgeComponent,
]

COMPONENT_METADATA = {
    ForgettingComponent: {"name": "forgetting", "required": False, "needs_inference": False},
    AnxietyComponent: {"name": "anxiety", "required": False, "needs_inference": False},
    MetaMemoryComponent: {"name": "metamemory", "required": False, "needs_inference": False},
    ConsolidationComponent: {"name": "consolidation", "required": False, "needs_inference": True},
    EmotionalTaggingComponent: {"name": "emotions", "required": False, "needs_inference": True},
    SuggestionComponent: {"name": "suggestions", "required": False, "needs_inference": True},
    KnowledgeComponent: {"name": "knowledge", "required": False, "needs_inference": True},
}


# ============================================================================
# Protocol Conformance
# ============================================================================


class TestProtocolConformance:
    """All components must satisfy StackComponentProtocol."""

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_isinstance_check(self, cls):
        component = cls()
        assert isinstance(component, StackComponentProtocol)

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_has_name(self, cls):
        component = cls()
        assert isinstance(component.name, str)
        assert len(component.name) > 0

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_has_version(self, cls):
        component = cls()
        assert isinstance(component.version, str)
        assert "." in component.version  # semver format

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_has_required(self, cls):
        component = cls()
        assert isinstance(component.required, bool)

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_has_needs_inference(self, cls):
        component = cls()
        assert isinstance(component.needs_inference, bool)

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_metadata_matches(self, cls):
        component = cls()
        expected = COMPONENT_METADATA[cls]
        assert component.name == expected["name"]
        assert component.required == expected["required"]
        assert component.needs_inference == expected["needs_inference"]


# ============================================================================
# Attach / Detach Lifecycle
# ============================================================================


class TestLifecycle:
    """Test attach/detach/set_inference lifecycle."""

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_attach(self, cls):
        component = cls()
        component.attach("stack-001")
        assert component._stack_id == "stack-001"
        assert component._inference is None

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_attach_with_inference(self, cls):
        component = cls()
        inference = _make_mock_inference()
        component.attach("stack-001", inference=inference)
        assert component._stack_id == "stack-001"
        assert component._inference is inference

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_detach(self, cls):
        component = cls()
        component.attach("stack-001")
        component.set_storage(_make_mock_storage())
        component.detach()
        assert component._stack_id is None
        assert component._inference is None
        assert component._storage is None

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_set_inference(self, cls):
        component = cls()
        inference = _make_mock_inference()
        component.set_inference(inference)
        assert component._inference is inference
        component.set_inference(None)
        assert component._inference is None

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_set_storage(self, cls):
        component = cls()
        storage = _make_mock_storage()
        component.set_storage(storage)
        assert component._storage is storage


# ============================================================================
# Graceful Degradation Without Storage
# ============================================================================


class TestNoStorage:
    """All components should handle missing storage gracefully."""

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_on_maintenance_without_storage(self, cls):
        component = cls()
        component.attach("stack-001")
        result = component.on_maintenance()
        assert isinstance(result, dict)
        assert result.get("skipped") is True
        assert isinstance(result.get("reason"), str) and result["reason"]

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_on_save_without_storage(self, cls):
        component = cls()
        result = component.on_save("episode", "ep-1", MagicMock())
        # Should return None or a dict, not raise
        assert result is None or isinstance(result, dict)

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_on_search_without_storage(self, cls):
        component = cls()
        results = [SearchResult(memory_type="episode", memory_id="ep-1", content="test", score=0.9)]
        returned = component.on_search("query", results)
        assert returned == results  # Should pass through

    @pytest.mark.parametrize("cls", ALL_COMPONENTS)
    def test_on_load_without_storage(self, cls):
        component = cls()
        context: Dict[str, Any] = {}
        component.on_load(context)
        # Should not raise


# ============================================================================
# Graceful Degradation Without Inference (for inference-needing components)
# ============================================================================

INFERENCE_COMPONENTS = [
    ConsolidationComponent,
    EmotionalTaggingComponent,
    SuggestionComponent,
    KnowledgeComponent,
]


class TestNoInference:
    """Inference-needing components should degrade gracefully without inference."""

    @pytest.mark.parametrize("cls", INFERENCE_COMPONENTS)
    def test_on_maintenance_without_inference(self, cls):
        component = cls()
        component.attach("stack-001")
        storage = _make_mock_storage()
        component.set_storage(storage)
        # No inference set
        result = component.on_maintenance()
        assert isinstance(result, dict)
        # Should still produce a result (partial work)


# ============================================================================
# ForgettingComponent Tests
# ============================================================================


class TestForgettingComponent:
    def test_calculate_salience_no_storage(self):
        c = ForgettingComponent()
        assert c.calculate_salience("episode", "ep-1") == -1.0

    def test_calculate_salience_not_found(self):
        c = ForgettingComponent()
        storage = _make_mock_storage()
        storage.get_memory.return_value = None
        c.set_storage(storage)
        assert c.calculate_salience("episode", "ep-1") == -1.0

    def test_calculate_salience_recent_accessed(self):
        c = ForgettingComponent()
        storage = _make_mock_storage()
        record = _make_mock_memory(
            confidence=0.9,
            times_accessed=10,
            last_accessed=datetime.now(timezone.utc),
        )
        storage.get_memory.return_value = record
        c.set_storage(storage)

        salience = c.calculate_salience("episode", "mem-1")
        assert salience > 0

    def test_calculate_salience_old_unaccessed(self):
        c = ForgettingComponent()
        storage = _make_mock_storage()
        record = _make_mock_memory(
            confidence=0.5,
            times_accessed=0,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=365),
        )
        storage.get_memory.return_value = record
        c.set_storage(storage)

        salience = c.calculate_salience("episode", "mem-1")
        # Old, unaccessed memory should have low salience
        assert salience < 0.1

    def test_goal_half_life_aspiration(self):
        c = ForgettingComponent()
        record = _make_mock_memory(goal_type="aspiration")
        half_life = c._get_half_life("goal", record)
        assert half_life == 180.0

    def test_goal_half_life_commitment(self):
        c = ForgettingComponent()
        record = _make_mock_memory(goal_type="commitment")
        half_life = c._get_half_life("goal", record)
        assert half_life == 365.0

    def test_on_maintenance_runs_forgetting(self):
        c = ForgettingComponent()
        storage = _make_mock_storage()

        candidate = MagicMock()
        candidate.score = 0.1
        candidate.record_type = "episode"
        candidate.record = MagicMock()
        candidate.record.id = "ep-old"
        storage.get_forgetting_candidates.return_value = [candidate]
        storage.forget_memory.return_value = True

        c.set_storage(storage)
        result = c.on_maintenance()
        assert result["forgotten"] == 1
        assert result["candidates_found"] == 1


# ============================================================================
# AnxietyComponent Tests
# ============================================================================


class TestAnxietyComponent:
    def test_get_anxiety_report_empty_storage(self):
        c = AnxietyComponent()
        storage = _make_mock_storage()
        c.set_storage(storage)
        report = c.get_anxiety_report()
        assert "overall_score" in report
        assert "overall_level" in report
        assert isinstance(report["overall_score"], int)

    def test_anxiety_increases_with_unreflected_episodes(self):
        c = AnxietyComponent()
        storage = _make_mock_storage()

        # 10 unreflected episodes
        episodes = [_make_mock_episode(id=f"ep-{i}", lessons=[], tags=["work"]) for i in range(10)]
        storage.get_episodes.return_value = episodes
        c.set_storage(storage)

        report = c.get_anxiety_report()
        consolidation_score = report["dimensions"]["consolidation_debt"]["score"]
        assert consolidation_score > 30  # Should be elevated

    def test_on_load_adds_anxiety(self):
        c = AnxietyComponent()
        storage = _make_mock_storage()
        c.set_storage(storage)

        context: Dict[str, Any] = {}
        c.on_load(context)
        assert "anxiety" in context
        assert "overall_score" in context["anxiety"]

    def test_build_alerts_overall_and_dimension_thresholds(self):
        c = AnxietyComponent()
        report: Dict[str, Any] = {
            "overall_score": 85,
            "dimensions": {
                "consolidation_debt": {"score": 70},
                "memory_uncertainty": {"score": 95},
            },
        }
        alerts = c._build_alerts(report)
        assert len(alerts) == 3
        assert alerts[0]["type"] == "overall_anxiety"
        assert alerts[0]["severity"] == "high"
        assert alerts[0]["value"] == 85

        types = {a["type"] for a in alerts}
        assert "dimension:consolidation_debt" in types
        assert "dimension:memory_uncertainty" in types

        assert any(
            a["type"] == "dimension:consolidation_debt" and a["severity"] == "medium"
            for a in alerts
        )
        assert any(
            a["type"] == "dimension:memory_uncertainty" and a["severity"] == "high" for a in alerts
        )

    def test_build_alerts_no_alerts_below_threshold(self):
        c = AnxietyComponent()
        alerts = c._build_alerts(
            {"overall_score": 69, "dimensions": {"identity_coherence": {"score": 1}}}
        )
        assert alerts == []


# ============================================================================
# MetaMemoryComponent Tests
# ============================================================================


class TestMetaMemoryComponent:
    def test_confidence_with_decay_protected(self):
        c = MetaMemoryComponent()
        memory = _make_mock_memory(confidence=0.9, is_protected=True)
        assert c.get_confidence_with_decay(memory, "belief") == 0.9

    def test_confidence_with_decay_old_memory(self):
        c = MetaMemoryComponent(decay_rate=0.01, decay_period_days=30, decay_floor=0.5)
        memory = _make_mock_memory(
            confidence=0.9,
            is_protected=False,
            last_verified=None,
            created_at=datetime.now(timezone.utc) - timedelta(days=300),
        )
        result = c.get_confidence_with_decay(memory, "belief")
        assert result < 0.9  # Should have decayed
        assert result >= 0.5  # Should not go below floor

    def test_confidence_no_decay_recent(self):
        c = MetaMemoryComponent()
        memory = _make_mock_memory(
            confidence=0.9,
            is_protected=False,
            last_verified=datetime.now(timezone.utc),
        )
        result = c.get_confidence_with_decay(memory, "belief")
        assert result == 0.9  # No decay for recently verified

    def test_on_load_adds_uncertainty_info(self):
        c = MetaMemoryComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = [
            _make_mock_belief(confidence=0.3),
            _make_mock_belief(confidence=0.2),
        ]
        c.set_storage(storage)

        context: Dict[str, Any] = {}
        c.on_load(context)
        assert "metamemory" in context
        assert context["metamemory"]["uncertain_memories"] == 2

    def test_on_load_no_uncertainty(self):
        c = MetaMemoryComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = [
            _make_mock_belief(confidence=0.9),
        ]
        c.set_storage(storage)

        context: Dict[str, Any] = {}
        c.on_load(context)
        # No uncertainty, so metamemory key should not be added
        assert "metamemory" not in context


# ============================================================================
# ConsolidationComponent Tests
# ============================================================================


class TestConsolidationComponent:
    def test_on_maintenance_too_few_episodes(self):
        c = ConsolidationComponent()
        storage = _make_mock_storage()
        storage.get_episodes.return_value = [_make_mock_episode(), _make_mock_episode(id="ep-2")]
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["consolidated"] == 0

    def test_on_maintenance_finds_common_lessons(self):
        c = ConsolidationComponent()
        storage = _make_mock_storage()
        storage.get_episodes.return_value = [
            _make_mock_episode(id="ep-1", lessons=["always test first"]),
            _make_mock_episode(id="ep-2", lessons=["always test first"]),
            _make_mock_episode(id="ep-3", lessons=["review before merge"]),
        ]
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["consolidated"] == 3
        assert result["lessons_found"] == 1
        assert "always test first" in result["common_lessons"]

    def test_on_maintenance_without_inference_still_works(self):
        c = ConsolidationComponent()
        storage = _make_mock_storage()
        storage.get_episodes.return_value = [
            _make_mock_episode(id=f"ep-{i}", lessons=["shared lesson"]) for i in range(5)
        ]
        c.set_storage(storage)
        # No inference
        result = c.on_maintenance()
        assert result["inference_available"] is False
        assert result["lessons_found"] >= 1

    def test_cross_domain_pattern_detection(self):
        c = ConsolidationComponent()
        episodes = [
            _make_mock_episode(
                id="ep-1", tags=["python"], lessons=["always test first"], outcome_type="success"
            ),
            _make_mock_episode(
                id="ep-2", tags=["rust"], lessons=["always test first"], outcome_type="success"
            ),
        ]
        patterns = c._detect_cross_domain_patterns(episodes)
        assert len(patterns) >= 1
        assert patterns[0]["lesson"] == "always test first"
        assert len(patterns[0]["domains"]) >= 2

    def test_build_alerts_thresholds(self):
        c = ConsolidationComponent()
        alerts = c._build_alerts(
            episodes_count=10,
            patterns=[{"lesson": "a"}, {"lesson": "b"}],
            common_lessons=["x"] * 6,
        )
        assert len(alerts) == 3
        alert_types = {alert["type"] for alert in alerts}
        assert alert_types == {
            "cross_domain_clustering",
            "lesson_convergence",
            "consolidation_volume",
        }

    def test_build_alerts_below_thresholds(self):
        c = ConsolidationComponent()
        alerts = c._build_alerts(
            episodes_count=9, patterns=[{"lesson": "a"}], common_lessons=["x"] * 5
        )
        assert alerts == []


# ============================================================================
# EmotionalTaggingComponent Tests
# ============================================================================


class TestEmotionalTaggingComponent:
    def test_detect_emotion_positive(self):
        c = EmotionalTaggingComponent()
        result = c.detect_emotion("I'm so happy and excited about this!")
        assert result["valence"] > 0
        assert result["confidence"] > 0
        assert len(result["tags"]) > 0

    def test_detect_emotion_negative(self):
        c = EmotionalTaggingComponent()
        result = c.detect_emotion("I'm frustrated and angry about the failure")
        assert result["valence"] < 0
        assert result["confidence"] > 0

    def test_detect_emotion_neutral(self):
        c = EmotionalTaggingComponent()
        result = c.detect_emotion("The meeting is at 3pm")
        assert result["confidence"] == 0.0
        assert result["tags"] == []

    def test_detect_emotion_empty(self):
        c = EmotionalTaggingComponent()
        result = c.detect_emotion("")
        assert result["confidence"] == 0.0

    def test_on_save_episode_detects_emotion(self):
        c = EmotionalTaggingComponent()
        memory = MagicMock()
        memory.objective = "Fix the frustrating bug"
        memory.outcome = "Finally resolved it, feeling satisfied"

        result = c.on_save("episode", "ep-1", memory)
        assert result is not None
        assert "emotional_valence" in result
        assert "emotional_tags" in result

    def test_on_save_non_episode_ignored(self):
        c = EmotionalTaggingComponent()
        result = c.on_save("belief", "b-1", MagicMock())
        assert result is None

    def test_on_maintenance_reports_emotions(self):
        c = EmotionalTaggingComponent()
        storage = _make_mock_storage()
        storage.get_episodes.return_value = [
            _make_mock_episode(
                emotional_valence=0.7, emotional_arousal=0.5, emotional_tags=["joy"]
            ),
            _make_mock_episode(
                emotional_valence=-0.3, emotional_arousal=0.8, emotional_tags=["frustration"]
            ),
        ]
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["episodes_with_emotions"] == 2
        assert result["avg_valence"] > -1.0


# ============================================================================
# SuggestionComponent Tests
# ============================================================================


class TestSuggestionComponent:
    def test_on_maintenance_extracts_suggestions(self):
        c = SuggestionComponent()
        c.attach("stack-001")
        storage = _make_mock_storage()
        storage.list_raw.return_value = [
            _make_mock_raw_entry(content="I completed the deployment and it succeeded"),
        ]
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["raw_entries_processed"] == 1
        assert result["suggestions_extracted"] >= 1
        assert storage.save_suggestion.called

    def test_on_maintenance_no_raw_entries(self):
        c = SuggestionComponent()
        storage = _make_mock_storage()
        storage.list_raw.return_value = []
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["raw_entries_processed"] == 0
        assert result["suggestions_extracted"] == 0

    def test_pattern_scoring(self):
        c = SuggestionComponent()
        episode_score = c._score_patterns(
            "i completed the task and shipped it, succeeded and learned a lot",
            EPISODE_PATTERNS,
        )
        assert episode_score > 0.4

    def test_pattern_scoring_no_match(self):
        c = SuggestionComponent()
        score = c._score_patterns("the sky is blue today", EPISODE_PATTERNS)
        assert score < 0.4

    def test_extract_suggestions_episode(self):
        c = SuggestionComponent()
        c._stack_id = "stack-001"
        entry = _make_mock_raw_entry(
            content="I completed the migration and it succeeded. Lesson: always test first."
        )
        suggestions = c._extract_suggestions(entry)
        assert len(suggestions) >= 1
        types = [s.memory_type for s in suggestions]
        assert "episode" in types

    def test_short_content_skipped(self):
        c = SuggestionComponent()
        c._stack_id = "stack-001"
        entry = _make_mock_raw_entry(content="hi")
        suggestions = c._extract_suggestions(entry)
        assert len(suggestions) == 0


# ============================================================================
# KnowledgeComponent Tests
# ============================================================================


class TestKnowledgeComponent:
    def test_on_maintenance_empty_storage(self):
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        c.set_storage(storage)

        result = c.on_maintenance()
        assert "domains_found" in result
        assert result["domains_found"] == 0

    def test_on_maintenance_with_data(self):
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = [
            _make_mock_belief(belief_type="python", confidence=0.9),
            _make_mock_belief(belief_type="python", confidence=0.85),
            _make_mock_belief(belief_type="docker", confidence=0.3),
        ]
        storage.get_episodes.return_value = [
            _make_mock_episode(tags=["python"]),
            _make_mock_episode(tags=["python"]),
        ]
        storage.get_notes.return_value = []
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["domains_found"] >= 2
        assert result["strength_domains"] >= 1  # python is a strength

    def test_on_maintenance_without_inference(self):
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        c.set_storage(storage)
        # No inference
        result = c.on_maintenance()
        assert result["inference_available"] is False

    def test_on_maintenance_with_inference(self):
        """When inference is available, inference_available is True."""
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        inference = _make_mock_inference()
        c.set_storage(storage)
        c.set_inference(inference)
        result = c.on_maintenance()
        assert result["inference_available"] is True

    def test_on_maintenance_skips_general_domains(self):
        """Domains named 'general', 'manual', 'auto-captured' are skipped in analysis."""
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = [
            _make_mock_belief(belief_type="general", confidence=0.9),
            _make_mock_belief(belief_type="general", confidence=0.85),
            _make_mock_belief(belief_type="manual", confidence=0.7),
            _make_mock_belief(belief_type="auto-captured", confidence=0.6),
        ]
        storage.get_episodes.return_value = []
        storage.get_notes.return_value = []
        c.set_storage(storage)

        result = c.on_maintenance()
        # These domains exist in domain_stats but are skipped in classification
        assert result["strength_domains"] == 0
        assert result["weakness_domains"] == 0

    def test_on_maintenance_weakness_detection(self):
        """Domains with avg confidence < 0.5 are counted as weaknesses."""
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = [
            _make_mock_belief(belief_type="rust", confidence=0.3),
            _make_mock_belief(belief_type="rust", confidence=0.4),
            _make_mock_belief(belief_type="rust", confidence=0.2),
        ]
        storage.get_episodes.return_value = [
            _make_mock_episode(tags=["rust"]),
        ]
        storage.get_notes.return_value = []
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["weakness_domains"] >= 1

    def test_on_maintenance_uncertain_areas(self):
        """Domains with beliefs and low confidence are marked uncertain."""
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = [
            _make_mock_belief(belief_type="quantum", confidence=0.3),
            _make_mock_belief(belief_type="quantum", confidence=0.4),
        ]
        storage.get_episodes.return_value = [
            _make_mock_episode(tags=["quantum"]),
        ]
        storage.get_notes.return_value = []
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["uncertain_areas"] >= 1

    def test_extract_domains_episodes_without_tags(self):
        """Episodes without tags fall into 'general' domain."""
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = []
        storage.get_episodes.return_value = [
            _make_mock_episode(tags=[]),
            _make_mock_episode(tags=None),
        ]
        # Set tags to None for second mock
        storage.get_episodes.return_value[1].tags = None
        storage.get_notes.return_value = []
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["domains_found"] >= 1  # 'general' domain

    def test_extract_domains_notes_with_tags(self):
        """Notes with tags contribute to domain stats."""
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        mock_note = MagicMock()
        mock_note.tags = ["python", "testing"]
        mock_note_no_tags = MagicMock()
        mock_note_no_tags.tags = []
        storage.get_beliefs.return_value = []
        storage.get_episodes.return_value = []
        storage.get_notes.return_value = [mock_note, mock_note_no_tags]
        c.set_storage(storage)

        result = c.on_maintenance()
        # Should have python, testing, and general domains
        assert result["domains_found"] >= 2

    def test_extract_domains_episodes_filter_meta_tags(self):
        """Meta tags like 'checkpoint', 'working_state' are filtered from episodes."""
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = []
        storage.get_episodes.return_value = [
            _make_mock_episode(tags=["checkpoint", "working_state", "python"]),
            _make_mock_episode(tags=["checkpoint"]),  # Only meta tags → general
        ]
        storage.get_notes.return_value = []
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["domains_found"] >= 1  # python + general

    def test_on_maintenance_no_storage(self):
        """Without storage, maintenance is skipped."""
        c = KnowledgeComponent()
        result = c.on_maintenance()
        assert result.get("skipped") is True

    def test_domain_below_threshold_skipped(self):
        """Domains with total < 2 records are skipped in classification."""
        c = KnowledgeComponent()
        storage = _make_mock_storage()
        storage.get_beliefs.return_value = [
            _make_mock_belief(belief_type="niche", confidence=0.9),
        ]
        storage.get_episodes.return_value = []
        storage.get_notes.return_value = []
        c.set_storage(storage)

        result = c.on_maintenance()
        # 'niche' has only 1 record (1 belief), so it shouldn't be classified
        assert result["strength_domains"] == 0


# ============================================================================
# EmotionalTaggingComponent Inference Tests
# ============================================================================


class TestEmotionalTaggingInference:
    """Test inference wiring and fallback in EmotionalTaggingComponent."""

    def test_with_inference_valid_json(self):
        """When inference returns valid JSON, uses inference result."""
        c = EmotionalTaggingComponent()
        inference = _make_mock_inference()
        inference.infer.return_value = (
            '{"valence": 0.6, "arousal": 0.4, "emotions": ["joy", "pride"]}'
        )
        c.set_inference(inference)

        result = c.detect_emotion("I completed the project successfully!")
        assert result["valence"] == 0.6
        assert result["arousal"] == 0.4
        assert result["tags"] == ["joy", "pride"]
        assert result["confidence"] == 0.9
        inference.infer.assert_called_once()

    def test_without_inference_falls_back_to_keywords(self):
        """Without inference, uses keyword-based detection."""
        c = EmotionalTaggingComponent()
        # No inference set
        result = c.detect_emotion("I'm so happy and excited!")
        assert result["valence"] > 0
        assert "joy" in result["tags"] or "excitement" in result["tags"]

    def test_inference_returns_bad_json_falls_back(self):
        """When inference returns invalid JSON, falls back to keywords."""
        c = EmotionalTaggingComponent()
        inference = _make_mock_inference()
        inference.infer.return_value = "not valid json at all"
        c.set_inference(inference)

        result = c.detect_emotion("I'm so happy and excited!")
        # Should still get a result from keyword fallback
        assert result["valence"] > 0
        assert len(result["tags"]) > 0
        inference.infer.assert_called_once()

    def test_inference_raises_exception_falls_back(self):
        """When inference raises, falls back to keywords gracefully."""
        c = EmotionalTaggingComponent()
        inference = _make_mock_inference()
        inference.infer.side_effect = RuntimeError("model unavailable")
        c.set_inference(inference)

        result = c.detect_emotion("I'm frustrated and angry")
        assert result["valence"] < 0
        assert len(result["tags"]) > 0

    def test_inference_missing_fields_falls_back(self):
        """When inference JSON lacks required fields, falls back."""
        c = EmotionalTaggingComponent()
        inference = _make_mock_inference()
        inference.infer.return_value = '{"valence": 0.5}'  # missing arousal and emotions
        c.set_inference(inference)

        result = c.detect_emotion("I'm so happy and excited!")
        # Should fall back to keywords since 'arousal' key missing
        assert result["valence"] > 0

    def test_inference_clamps_values(self):
        """Inference values are clamped to valid ranges."""
        c = EmotionalTaggingComponent()
        inference = _make_mock_inference()
        inference.infer.return_value = '{"valence": 5.0, "arousal": -3.0, "emotions": ["joy"]}'
        c.set_inference(inference)

        result = c.detect_emotion("extreme text")
        assert result["valence"] == 1.0  # clamped from 5.0
        assert result["arousal"] == 0.0  # clamped from -3.0

    def test_on_save_uses_inference_when_available(self):
        """on_save() uses inference path for emotion detection."""
        c = EmotionalTaggingComponent()
        inference = _make_mock_inference()
        inference.infer.return_value = '{"valence": -0.3, "arousal": 0.7, "emotions": ["anxiety"]}'
        c.set_inference(inference)

        memory = MagicMock()
        memory.objective = "Deploy to production"
        memory.outcome = "Deployment failed with errors"

        result = c.on_save("episode", "ep-1", memory)
        assert result is not None
        assert result["emotional_valence"] == -0.3
        assert result["emotional_arousal"] == 0.7
        assert result["emotional_tags"] == ["anxiety"]


# ============================================================================
# ConsolidationComponent Inference Tests
# ============================================================================


class TestConsolidationInference:
    """Test inference wiring and fallback in ConsolidationComponent."""

    def test_with_inference_returns_themes(self):
        """When inference returns valid themes, includes them in result."""
        c = ConsolidationComponent()
        inference = _make_mock_inference()
        inference.infer.return_value = (
            '{"themes": ["testing is important", "documentation prevents bugs"]}'
        )
        c.set_inference(inference)

        storage = _make_mock_storage()
        storage.get_episodes.return_value = [
            _make_mock_episode(id=f"ep-{i}", lessons=["always test first"]) for i in range(5)
        ]
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["inference_available"] is True
        assert "synthesized_themes" in result
        assert "testing is important" in result["synthesized_themes"]
        inference.infer.assert_called_once()

    def test_without_inference_still_finds_lessons(self):
        """Without inference, still finds common lessons via keyword counting."""
        c = ConsolidationComponent()
        storage = _make_mock_storage()
        storage.get_episodes.return_value = [
            _make_mock_episode(id="ep-1", lessons=["always test first"]),
            _make_mock_episode(id="ep-2", lessons=["always test first"]),
            _make_mock_episode(id="ep-3", lessons=["review before merge"]),
        ]
        c.set_storage(storage)
        # No inference set

        result = c.on_maintenance()
        assert result["inference_available"] is False
        assert result["lessons_found"] == 1
        assert "synthesized_themes" not in result

    def test_inference_bad_json_falls_back(self):
        """When inference returns bad JSON, still produces result without themes."""
        c = ConsolidationComponent()
        inference = _make_mock_inference()
        inference.infer.return_value = "not json"
        c.set_inference(inference)

        storage = _make_mock_storage()
        storage.get_episodes.return_value = [
            _make_mock_episode(id=f"ep-{i}", lessons=["shared lesson"]) for i in range(5)
        ]
        c.set_storage(storage)

        result = c.on_maintenance()
        # Should still have keyword-based results
        assert result["lessons_found"] >= 1
        # inference is available but synthesis failed, so no themes
        assert result["inference_available"] is True
        assert "synthesized_themes" not in result

    def test_inference_raises_exception_falls_back(self):
        """When inference raises, still produces keyword-based result."""
        c = ConsolidationComponent()
        inference = _make_mock_inference()
        inference.infer.side_effect = RuntimeError("model crashed")
        c.set_inference(inference)

        storage = _make_mock_storage()
        storage.get_episodes.return_value = [
            _make_mock_episode(id=f"ep-{i}", lessons=["shared lesson"]) for i in range(5)
        ]
        c.set_storage(storage)

        result = c.on_maintenance()
        assert result["consolidated"] == 5
        assert result["lessons_found"] >= 1


# ============================================================================
# Belief.processed Tests
# ============================================================================


class TestBeliefProcessed:
    """Test Belief.processed field and mark_belief_processed method."""

    def test_belief_processed_default_false(self):
        from kernle.types import Belief

        b = Belief(id="b-1", stack_id="s-1", statement="test")
        assert b.processed is False

    def test_belief_processed_can_be_set(self):
        from kernle.types import Belief

        b = Belief(id="b-1", stack_id="s-1", statement="test", processed=True)
        assert b.processed is True

    def test_mark_belief_processed_roundtrip(self, tmp_path):
        """Test save → mark processed → get roundtrip through DB."""
        from kernle.storage.sqlite import SQLiteStorage
        from kernle.types import Belief

        storage = SQLiteStorage(db_path=tmp_path / "test.db", stack_id="test-stack")

        b = Belief(id="b-1", stack_id="test-stack", statement="Test belief")
        storage.save_belief(b)

        # Verify starts unprocessed
        fetched = storage.get_belief("b-1")
        assert fetched is not None
        assert fetched.processed is False

        # Mark processed
        result = storage.mark_belief_processed("b-1")
        assert result is True

        # Verify now processed
        fetched2 = storage.get_belief("b-1")
        assert fetched2 is not None
        assert fetched2.processed is True

    def test_mark_belief_processed_nonexistent(self, tmp_path):
        """Marking a nonexistent belief returns False."""
        from kernle.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(db_path=tmp_path / "test.db", stack_id="test-stack")

        result = storage.mark_belief_processed("nonexistent")
        assert result is False

    def test_belief_processed_saved_to_db(self, tmp_path):
        """Test that processed=True is persisted when saving a belief."""
        from kernle.storage.sqlite import SQLiteStorage
        from kernle.types import Belief

        storage = SQLiteStorage(db_path=tmp_path / "test.db", stack_id="test-stack")

        b = Belief(id="b-2", stack_id="test-stack", statement="Processed belief", processed=True)
        storage.save_belief(b)

        fetched = storage.get_belief("b-2")
        assert fetched is not None
        assert fetched.processed is True


# ============================================================================
# Integration-style: Component with SQLiteStack lifecycle
# ============================================================================


class TestComponentIntegration:
    """Test how components work when added to a stack-like environment."""

    def test_full_lifecycle(self):
        """Component goes through full attach -> use -> detach cycle."""
        c = ForgettingComponent()
        storage = _make_mock_storage()
        inference = _make_mock_inference()

        # Attach
        c.attach("stack-001", inference=inference)
        c.set_storage(storage)
        assert c._stack_id == "stack-001"
        assert c._inference is inference
        assert c._storage is storage

        # Use
        result = c.on_maintenance()
        assert isinstance(result, dict)

        # Model change
        new_inference = _make_mock_inference()
        c.set_inference(new_inference)
        assert c._inference is new_inference

        # Remove inference
        c.set_inference(None)
        assert c._inference is None

        # Detach
        c.detach()
        assert c._stack_id is None
        assert c._storage is None

    def test_multiple_components_coexist(self):
        """Multiple components can be created and operate independently."""
        storage = _make_mock_storage()
        components = [cls() for cls in ALL_COMPONENTS]

        for comp in components:
            comp.attach("stack-001")
            comp.set_storage(storage)

        # Run maintenance on all
        results = {}
        for comp in components:
            results[comp.name] = comp.on_maintenance()

        assert len(results) == len(ALL_COMPONENTS)
        for name, result in results.items():
            assert isinstance(result, dict), f"{name} returned non-dict"

        # Detach all
        for comp in components:
            comp.detach()
            assert comp._stack_id is None
