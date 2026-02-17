"""Tests for advanced consolidation scaffolds (KEP v3, Issue #178).

Tests cover:
- Cross-domain pattern detection
- Belief-to-value promotion analysis
- Entity model-to-belief generalization
- Combined scaffold output
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import Belief, EntityModel, Episode
from kernle.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    """SQLite storage for consolidation scaffold tests."""
    db_path = tmp_path / "test_consolidation.db"
    s = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def kernle_instance(storage):
    """Kernle instance with test storage."""
    k = Kernle(stack_id="test_agent", storage=storage, strict=False)
    return k


def _make_episode(
    tags=None, outcome_type="success", lessons=None, objective="Test task", **kwargs
) -> Episode:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": "test_agent",
        "objective": objective,
        "outcome": f"Outcome for {objective}",
        "outcome_type": outcome_type,
        "tags": tags,
        "lessons": lessons,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(kwargs)
    return Episode(**defaults)


def _make_belief(
    statement="Test belief",
    confidence=0.85,
    created_at=None,
    times_reinforced=0,
    source_domain=None,
    cross_domain_applications=None,
    **kwargs,
) -> Belief:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": "test_agent",
        "statement": statement,
        "belief_type": "pattern",
        "confidence": confidence,
        "created_at": created_at or datetime.now(timezone.utc),
        "times_reinforced": times_reinforced,
        "source_domain": source_domain,
        "cross_domain_applications": cross_domain_applications,
    }
    defaults.update(kwargs)
    return Belief(**defaults)


def _make_entity_model(
    entity_name="Alice",
    model_type="behavioral",
    observation="Test observation",
    **kwargs,
) -> EntityModel:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": "test_agent",
        "entity_name": entity_name,
        "model_type": model_type,
        "observation": observation,
        "confidence": 0.8,
        "source_episodes": [str(uuid.uuid4())],
    }
    defaults.update(kwargs)
    return EntityModel(**defaults)


# === Cross-domain pattern scaffolding ===


class TestCrossDomainPatterns:
    def test_no_episodes(self, kernle_instance):
        result = kernle_instance.scaffold_cross_domain_patterns()
        assert result["episodes_scanned"] == 0
        assert result["patterns"] == []
        assert "No episodes" in result["scaffold"]

    def test_single_domain_no_cross_pattern(self, kernle_instance, storage):
        """Episodes in one domain only should not produce cross-domain patterns."""
        for i in range(3):
            ep = _make_episode(
                tags=["coding"],
                outcome_type="success",
                lessons=["tests help catch bugs"],
            )
            storage.save_episode(ep)

        result = kernle_instance.scaffold_cross_domain_patterns()
        assert result["episodes_scanned"] == 3
        assert result["domains_found"] >= 1
        # Single domain = no cross-domain patterns
        assert len(result["patterns"]) == 0

    def test_cross_domain_pattern_detected(self, kernle_instance, storage):
        """Same lesson in 2+ domains should be detected."""
        lesson = "shortcutting process leads to failure"

        # Domain 1: deployment
        for i in range(2):
            ep = _make_episode(
                tags=["deployment"],
                outcome_type="failure",
                lessons=[lesson],
                objective=f"Deploy task {i}",
            )
            storage.save_episode(ep)

        # Domain 2: relationships
        for i in range(2):
            ep = _make_episode(
                tags=["relationships"],
                outcome_type="failure",
                lessons=[lesson],
                objective=f"Relationship task {i}",
            )
            storage.save_episode(ep)

        result = kernle_instance.scaffold_cross_domain_patterns()
        assert result["episodes_scanned"] == 4
        assert result["domains_found"] >= 2

        # Should find the cross-domain pattern
        assert len(result["patterns"]) >= 1
        pattern = result["patterns"][0]
        assert pattern["lesson"] == lesson
        assert len(pattern["domains"]) >= 2
        assert "deployment" in pattern["domains"]
        assert "relationships" in pattern["domains"]

    def test_scaffold_text_contains_domains(self, kernle_instance, storage):
        """Scaffold text should list domains found."""
        ep1 = _make_episode(tags=["coding"], outcome_type="success")
        ep2 = _make_episode(tags=["writing"], outcome_type="failure")
        storage.save_episode(ep1)
        storage.save_episode(ep2)

        result = kernle_instance.scaffold_cross_domain_patterns()
        assert "coding" in result["scaffold"]
        assert "writing" in result["scaffold"]

    def test_outcome_patterns_detected(self, kernle_instance, storage):
        """Domains with dominant outcome types should be flagged."""
        for i in range(4):
            ep = _make_episode(
                tags=["testing"],
                outcome_type="success",
                objective=f"Test run {i}",
            )
            storage.save_episode(ep)
        # Add one failure for variety
        ep_fail = _make_episode(
            tags=["testing"],
            outcome_type="failure",
            objective="Test run fail",
        )
        storage.save_episode(ep_fail)

        result = kernle_instance.scaffold_cross_domain_patterns()
        # 4/5 = 80% success rate should be detected
        assert len(result.get("outcome_patterns", [])) >= 1
        op = result["outcome_patterns"][0]
        assert op["domain"] == "testing"
        assert op["outcome"] == "success"
        assert op["ratio"] >= 0.6

    def test_untagged_episodes_grouped(self, kernle_instance, storage):
        """Episodes without tags should be grouped under 'untagged'."""
        ep = _make_episode(tags=None, outcome_type="success")
        storage.save_episode(ep)

        result = kernle_instance.scaffold_cross_domain_patterns()
        assert result["episodes_scanned"] == 1
        assert result["domains_found"] >= 1


# === Belief-to-value promotion ===


class TestBeliefToValuePromotion:
    def test_no_beliefs(self, kernle_instance):
        result = kernle_instance.scaffold_belief_to_value()
        assert result["beliefs_scanned"] == 0
        assert result["candidates"] == []

    def test_young_belief_not_promoted(self, kernle_instance, storage):
        """Beliefs younger than 6 months should not be candidates."""
        belief = _make_belief(
            statement="Young belief",
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
            times_reinforced=5,
            source_domain="coding",
            cross_domain_applications=["testing", "deployment"],
        )
        storage.save_belief(belief)

        result = kernle_instance.scaffold_belief_to_value()
        assert result["beliefs_scanned"] == 1
        assert len(result["candidates"]) == 0

    def test_unreinforced_belief_not_promoted(self, kernle_instance, storage):
        """Beliefs with few reinforcements should not be candidates."""
        belief = _make_belief(
            statement="Unreinforced belief",
            created_at=datetime.now(timezone.utc) - timedelta(days=200),
            times_reinforced=1,
            source_domain="coding",
            cross_domain_applications=["testing", "deployment"],
        )
        storage.save_belief(belief)

        result = kernle_instance.scaffold_belief_to_value()
        assert result["beliefs_scanned"] == 1
        assert len(result["candidates"]) == 0

    def test_single_domain_belief_not_promoted(self, kernle_instance, storage):
        """Beliefs in fewer than 3 domains should not be candidates."""
        belief = _make_belief(
            statement="Single domain belief",
            created_at=datetime.now(timezone.utc) - timedelta(days=200),
            times_reinforced=5,
            source_domain="coding",
            cross_domain_applications=None,  # Only 1 domain
        )
        storage.save_belief(belief)

        result = kernle_instance.scaffold_belief_to_value()
        assert result["beliefs_scanned"] == 1
        assert len(result["candidates"]) == 0

    def test_contradicted_belief_not_promoted(self, kernle_instance, storage):
        """Inactive (revised) beliefs should not be candidates."""
        belief = _make_belief(
            statement="Contradicted belief",
            created_at=datetime.now(timezone.utc) - timedelta(days=200),
            times_reinforced=5,
            source_domain="coding",
            cross_domain_applications=["testing", "deployment"],
            is_active=False,  # v0.14+: use is_active instead of superseded_by
        )
        storage.save_belief(belief)

        result = kernle_instance.scaffold_belief_to_value()
        # Inactive beliefs are filtered out before scanning
        assert len(result["candidates"]) == 0

    def test_stable_belief_is_candidate(self, kernle_instance, storage):
        """A belief meeting all criteria should be a promotion candidate."""
        belief = _make_belief(
            statement="Iterative development leads to better outcomes",
            created_at=datetime.now(timezone.utc) - timedelta(days=200),
            times_reinforced=5,
            confidence=0.9,
            source_domain="coding",
            cross_domain_applications=["writing", "relationships"],
        )
        storage.save_belief(belief)

        result = kernle_instance.scaffold_belief_to_value()
        assert result["beliefs_scanned"] == 1
        assert len(result["candidates"]) == 1

        candidate = result["candidates"][0]
        assert candidate["statement"] == "Iterative development leads to better outcomes"
        assert candidate["times_reinforced"] == 5
        assert candidate["age_days"] >= 200
        assert len(candidate["domains"]) >= 3
        assert "coding" in candidate["domains"]

    def test_scaffold_text_contains_candidate(self, kernle_instance, storage):
        """Scaffold text should describe promotion candidates."""
        belief = _make_belief(
            statement="Quality matters more than speed",
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
            times_reinforced=8,
            confidence=0.95,
            source_domain="coding",
            cross_domain_applications=["writing", "design"],
        )
        storage.save_belief(belief)

        result = kernle_instance.scaffold_belief_to_value()
        assert "Quality matters more than speed" in result["scaffold"]
        assert "value-level stability" in result["scaffold"]

    def test_low_confidence_belief_not_promoted(self, kernle_instance, storage):
        """Beliefs with confidence below threshold should not be candidates."""
        belief = _make_belief(
            statement="Low confidence belief",
            created_at=datetime.now(timezone.utc) - timedelta(days=200),
            times_reinforced=5,
            confidence=0.5,
            source_domain="coding",
            cross_domain_applications=["testing", "deployment"],
        )
        storage.save_belief(belief)

        result = kernle_instance.scaffold_belief_to_value()
        assert len(result["candidates"]) == 0

    def test_custom_thresholds(self, kernle_instance, storage):
        """Custom thresholds should be respected."""
        belief = _make_belief(
            statement="Custom threshold belief",
            created_at=datetime.now(timezone.utc) - timedelta(days=60),
            times_reinforced=2,
            confidence=0.7,
            source_domain="coding",
            cross_domain_applications=["testing"],
        )
        storage.save_belief(belief)

        # With relaxed thresholds, should be a candidate
        result = kernle_instance.scaffold_belief_to_value(
            min_age_days=30,
            min_reinforcements=1,
            min_domains=2,
            min_confidence=0.6,
        )
        assert len(result["candidates"]) == 1


# === Entity model-to-belief promotion ===


class TestEntityModelToBeliefPromotion:
    def test_no_models(self, kernle_instance):
        result = kernle_instance.scaffold_entity_model_to_belief()
        assert result["models_scanned"] == 0
        assert result["generalizations"] == []
        assert "No entity models" in result["scaffold"]

    def test_single_entity_no_generalization(self, kernle_instance, storage):
        """Observations about a single entity should not generalize."""
        model = _make_entity_model(
            entity_name="Alice",
            observation="careful with code reviews",
        )
        storage.save_entity_model(model)

        result = kernle_instance.scaffold_entity_model_to_belief()
        assert result["models_scanned"] == 1
        assert len(result["generalizations"]) == 0

    def test_similar_observations_across_entities(self, kernle_instance, storage):
        """Similar observations about different entities should generalize."""
        model1 = _make_entity_model(
            entity_name="Alice",
            model_type="behavioral",
            observation="careful and thorough with code reviews and testing",
            source_episodes=["ep1", "ep2"],
        )
        model2 = _make_entity_model(
            entity_name="Bob",
            model_type="behavioral",
            observation="thorough and careful during code review process",
            source_episodes=["ep3"],
        )
        storage.save_entity_model(model1)
        storage.save_entity_model(model2)

        result = kernle_instance.scaffold_entity_model_to_belief()
        assert result["models_scanned"] == 2
        # These share "careful", "thorough", "code", "reviews" - should cluster
        assert len(result["generalizations"]) >= 1

        gen = result["generalizations"][0]
        assert gen["entity_count"] >= 2
        assert "Alice" in gen["entities"]
        assert "Bob" in gen["entities"]

    def test_dissimilar_observations_no_cluster(self, kernle_instance, storage):
        """Dissimilar observations should not cluster."""
        model1 = _make_entity_model(
            entity_name="Alice",
            model_type="behavioral",
            observation="prefers morning meetings for important decisions",
        )
        model2 = _make_entity_model(
            entity_name="Bob",
            model_type="behavioral",
            observation="enjoys working on database optimization tasks",
        )
        storage.save_entity_model(model1)
        storage.save_entity_model(model2)

        result = kernle_instance.scaffold_entity_model_to_belief()
        # These share no meaningful keywords - should not cluster
        assert len(result["generalizations"]) == 0

    def test_scaffold_text_contains_observations(self, kernle_instance, storage):
        """Scaffold text should include entity observations."""
        model1 = _make_entity_model(
            entity_name="Alice",
            model_type="capability",
            observation="strong debugging and problem solving skills",
            source_episodes=["ep1"],
        )
        model2 = _make_entity_model(
            entity_name="Bob",
            model_type="capability",
            observation="excellent debugging skills and problem analysis",
            source_episodes=["ep2"],
        )
        storage.save_entity_model(model1)
        storage.save_entity_model(model2)

        result = kernle_instance.scaffold_entity_model_to_belief()
        if result["generalizations"]:
            assert "Alice" in result["scaffold"]
            assert "Bob" in result["scaffold"]

    def test_min_supporting_episodes(self, kernle_instance, storage):
        """Generalizations need minimum supporting episodes."""
        model1 = _make_entity_model(
            entity_name="Alice",
            model_type="behavioral",
            observation="careful with code reviews and testing",
            source_episodes=[],  # No episodes
        )
        model2 = _make_entity_model(
            entity_name="Bob",
            model_type="behavioral",
            observation="thorough with code reviews and testing",
            source_episodes=[],  # No episodes
        )
        storage.save_entity_model(model1)
        storage.save_entity_model(model2)

        # With min_supporting_episodes=2, should not qualify (0 total)
        result = kernle_instance.scaffold_entity_model_to_belief(min_supporting_episodes=2)
        assert len(result["generalizations"]) == 0


# === Combined scaffold ===


class TestAdvancedConsolidation:
    def test_combined_scaffold_empty(self, kernle_instance):
        """Combined scaffold should work with no data."""
        result = kernle_instance.scaffold_advanced_consolidation()
        assert "cross_domain" in result
        assert "belief_to_value" in result
        assert "entity_to_belief" in result
        assert "scaffold" in result
        assert "Advanced Consolidation Scaffold" in result["scaffold"]

    def test_combined_scaffold_with_data(self, kernle_instance, storage):
        """Combined scaffold should integrate all three analyses."""
        # Add some episodes
        for i in range(3):
            ep = _make_episode(
                tags=["coding"],
                outcome_type="success",
                lessons=["testing helps"],
                objective=f"Code task {i}",
            )
            storage.save_episode(ep)

        # Add a stable belief
        belief = _make_belief(
            statement="Testing leads to quality",
            created_at=datetime.now(timezone.utc) - timedelta(days=200),
            times_reinforced=5,
            source_domain="coding",
            cross_domain_applications=["deployment", "writing"],
        )
        storage.save_belief(belief)

        result = kernle_instance.scaffold_advanced_consolidation()
        scaffold = result["scaffold"]

        # Should contain sections from all three
        assert "Cross-Domain Pattern Analysis" in scaffold
        assert "Belief-to-Value Promotion Analysis" in scaffold
        # Entity model section present (either header or "no models" message)
        assert "entity model" in scaffold.lower()

    def test_episode_limit_passed_through(self, kernle_instance, storage):
        """Episode limit should be passed to cross-domain analysis."""
        for i in range(10):
            ep = _make_episode(
                tags=[f"tag{i}"],
                objective=f"Task {i}",
            )
            storage.save_episode(ep)

        result = kernle_instance.scaffold_advanced_consolidation(episode_limit=5)
        assert result["cross_domain"]["episodes_scanned"] <= 5


# === Observation clustering ===


class TestObservationClustering:
    def test_cluster_similar_observations(self, kernle_instance):
        """Models with overlapping keywords should cluster."""
        models = [
            _make_entity_model(
                entity_name="A",
                observation="careful thorough code review process",
            ),
            _make_entity_model(
                entity_name="B",
                observation="thorough careful review approach",
            ),
        ]
        clusters = kernle_instance._cluster_observations(models)
        assert len(clusters) >= 1
        assert len(clusters[0]) >= 2

    def test_no_cluster_dissimilar(self, kernle_instance):
        """Models with no keyword overlap should not cluster."""
        models = [
            _make_entity_model(
                entity_name="A",
                observation="morning schedule breakfast routine",
            ),
            _make_entity_model(
                entity_name="B",
                observation="database optimization queries performance",
            ),
        ]
        clusters = kernle_instance._cluster_observations(models)
        # Should not form any cluster (no shared meaningful keywords)
        assert len(clusters) == 0

    def test_empty_models_list(self, kernle_instance):
        """Empty model list should return no clusters."""
        clusters = kernle_instance._cluster_observations([])
        assert clusters == []

    def test_stop_words_excluded(self, kernle_instance):
        """Stop words should not count as shared keywords."""
        models = [
            _make_entity_model(
                entity_name="A",
                observation="the cat is on the mat",
            ),
            _make_entity_model(
                entity_name="B",
                observation="the dog is in the garden",
            ),
        ]
        clusters = kernle_instance._cluster_observations(models)
        # Only stop words in common - should not cluster
        assert len(clusters) == 0
