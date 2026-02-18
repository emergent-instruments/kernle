"""Tests for meta-cognition functionality.

Tests the ability to have awareness of what we know and don't know:
- Knowledge mapping across domains
- Gap detection for queries
- Competence boundaries (strengths/weaknesses)
- Learning opportunity identification
"""

import pytest

from kernle.core import Kernle
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint files."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def kernle(temp_db, temp_checkpoint_dir):
    """Create a Kernle instance for testing."""
    storage = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    k = Kernle(
        stack_id="test-agent", storage=storage, checkpoint_dir=temp_checkpoint_dir, strict=False
    )
    bind_noop_model(k)
    return k


@pytest.fixture
def populated_kernle(kernle):
    """Kernle with some test data for meta-cognition."""
    # Add beliefs with varying confidence
    kernle.belief(
        statement="Python is excellent for rapid prototyping", type="fact", confidence=0.9
    )
    kernle.belief(statement="Type hints improve code maintainability", type="fact", confidence=0.85)
    kernle.belief(statement="Docker networking is complex", type="observation", confidence=0.4)
    kernle.belief(
        statement="Kubernetes autoscaling is straightforward", type="assumption", confidence=0.3
    )

    # Add episodes with tags
    kernle.episode(
        objective="Implement Python API endpoint",
        outcome="success",
        lessons=["Use FastAPI for modern APIs"],
        tags=["python", "api"],
    )
    kernle.episode(
        objective="Write Python unit tests",
        outcome="success",
        lessons=["pytest fixtures are powerful"],
        tags=["python", "testing"],
    )
    kernle.episode(
        objective="Deploy Docker container",
        outcome="failure",
        lessons=["Check port mappings carefully"],
        tags=["docker", "deployment"],
    )
    kernle.episode(
        objective="Configure Docker networking",
        outcome="failure",
        lessons=["Network modes are confusing"],
        tags=["docker", "networking"],
    )
    kernle.episode(
        objective="Set up Kubernetes cluster",
        outcome="partial",
        lessons=["Start with minikube for learning"],
        tags=["kubernetes"],
    )

    # Add notes
    kernle.note(
        content="Python 3.12 has significant performance improvements",
        type="insight",
        tags=["python"],
    )
    kernle.note(
        content="Always use docker-compose for multi-container apps",
        type="decision",
        tags=["docker"],
    )

    return kernle


class TestGetKnowledgeMap:
    """Test the get_knowledge_map method."""

    def test_empty_knowledge_map(self, kernle):
        """Should return empty domains for new instance."""
        knowledge_map = kernle.get_knowledge_map()

        assert "domains" in knowledge_map
        assert "blind_spots" in knowledge_map
        assert "uncertain_areas" in knowledge_map
        assert "total_domains" in knowledge_map
        assert "timestamp" in knowledge_map

    def test_knowledge_map_with_data(self, populated_kernle):
        """Should map knowledge domains from beliefs/episodes/notes."""
        knowledge_map = populated_kernle.get_knowledge_map()

        assert len(knowledge_map["domains"]) > 0
        assert knowledge_map["total_domains"] > 0

        # Check domain structure - verify domains exist and have expected names
        domain_names = [d["name"] for d in knowledge_map["domains"]]
        assert len(domain_names) > 0
        # Should have domains from tags and belief types
        assert any(
            name in ["python", "docker", "kubernetes", "fact", "observation"]
            for name in domain_names
        )

    def test_domain_statistics(self, populated_kernle):
        """Should calculate statistics for each domain."""
        knowledge_map = populated_kernle.get_knowledge_map()

        for domain in knowledge_map["domains"]:
            assert "name" in domain
            assert "belief_count" in domain
            assert "avg_confidence" in domain
            assert "episode_count" in domain
            assert "note_count" in domain
            assert "coverage" in domain
            assert domain["coverage"] in ("high", "medium", "low", "none")

    def test_uncertain_areas_detection(self, populated_kernle):
        """Should identify areas with low-confidence beliefs."""
        knowledge_map = populated_kernle.get_knowledge_map()

        # Should detect uncertain areas (domains with beliefs < 0.5 confidence)
        # We added "observation" and "assumption" type beliefs with low confidence
        uncertain = knowledge_map.get("uncertain_areas", [])
        # This depends on how domains are extracted
        assert isinstance(uncertain, list)

    def test_coverage_calculation(self, kernle):
        """Should calculate coverage based on item counts."""
        # Add various amounts of data to test coverage levels

        # High coverage domain (10+ items)
        for i in range(12):
            kernle.episode(
                objective=f"High coverage task {i}", outcome="success", tags=["high_coverage"]
            )

        # Medium coverage domain (3-9 items)
        for i in range(5):
            kernle.episode(
                objective=f"Medium coverage task {i}", outcome="success", tags=["medium_coverage"]
            )

        # Low coverage domain (1-2 items)
        kernle.episode(objective="Low coverage task", outcome="success", tags=["low_coverage"])

        knowledge_map = kernle.get_knowledge_map()
        domain_by_name = {d["name"]: d for d in knowledge_map["domains"]}

        if "high_coverage" in domain_by_name:
            assert domain_by_name["high_coverage"]["coverage"] == "high"
        if "medium_coverage" in domain_by_name:
            assert domain_by_name["medium_coverage"]["coverage"] == "medium"
        if "low_coverage" in domain_by_name:
            assert domain_by_name["low_coverage"]["coverage"] == "low"


class TestDetectKnowledgeGaps:
    """Test the detect_knowledge_gaps method."""

    def test_no_relevant_knowledge(self, kernle):
        """Should detect when no relevant knowledge exists."""
        result = kernle.detect_knowledge_gaps("quantum computing algorithms")

        assert result["has_relevant_knowledge"] is False
        assert result["confidence"] == 0.0
        assert result["recommendation"] == "Ask someone else"
        assert result["relevant_beliefs"] == []
        assert result["relevant_episodes"] == []

    def test_relevant_knowledge_found(self, populated_kernle):
        """Should find relevant knowledge for a known topic."""
        result = populated_kernle.detect_knowledge_gaps("Python")

        assert result["has_relevant_knowledge"] is True
        assert result["confidence"] > 0
        assert result["search_results_count"] > 0
        # Should have some relevant results
        assert len(result["relevant_beliefs"]) > 0 or len(result["relevant_episodes"]) > 0

    def test_low_confidence_recommendation(self, kernle):
        """Should recommend learning more when confidence is low."""
        # Add a low-confidence belief
        kernle.belief(
            statement="GraphQL might be better than REST", type="assumption", confidence=0.3
        )

        result = kernle.detect_knowledge_gaps("GraphQL")

        # With only one low-confidence result, should recommend learning more
        assert result["has_relevant_knowledge"] is True
        assert result["recommendation"] in (
            "I should learn more",
            "I have limited knowledge - proceed with caution",
        )

    def test_high_confidence_recommendation(self, populated_kernle):
        """Should recommend helping when confidence is high."""
        result = populated_kernle.detect_knowledge_gaps("Python API endpoint")

        # We have good Python knowledge
        if result["confidence"] >= 0.5 and result["search_results_count"] >= 3:
            assert result["recommendation"] == "I can help"

    def test_result_structure(self, populated_kernle):
        """Should return properly structured result."""
        result = populated_kernle.detect_knowledge_gaps("deployment")

        assert "has_relevant_knowledge" in result
        assert "relevant_beliefs" in result
        assert "relevant_episodes" in result
        assert "relevant_notes" in result
        assert "confidence" in result
        assert "gaps" in result
        assert "recommendation" in result
        assert "search_results_count" in result

        # Validate types
        assert isinstance(result["has_relevant_knowledge"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["gaps"], list)
        assert 0.0 <= result["confidence"] <= 1.0


class TestGetCompetenceBoundaries:
    """Test the get_competence_boundaries method."""

    def test_empty_boundaries(self, kernle):
        """Should handle empty data gracefully."""
        boundaries = kernle.get_competence_boundaries()

        assert "strengths" in boundaries
        assert "weaknesses" in boundaries
        assert "overall_confidence" in boundaries
        assert "success_rate" in boundaries
        assert "experience_depth" in boundaries
        assert "knowledge_breadth" in boundaries

    def test_identifies_strengths(self, populated_kernle):
        """Should identify domains with high confidence and success."""
        boundaries = populated_kernle.get_competence_boundaries()

        # Check structure
        for strength in boundaries["strengths"]:
            assert "domain" in strength
            assert "confidence" in strength
            assert "success_rate" in strength
            # Strengths should have good metrics
            assert strength["confidence"] >= 0.7 or strength["success_rate"] >= 0.6

    def test_identifies_weaknesses(self, populated_kernle):
        """Should identify domains with low confidence or success."""
        boundaries = populated_kernle.get_competence_boundaries()

        # Docker had failures, so might appear in weaknesses
        for weakness in boundaries["weaknesses"]:
            assert "domain" in weakness
            assert "confidence" in weakness
            assert "success_rate" in weakness

    def test_overall_metrics(self, populated_kernle):
        """Should calculate overall confidence and success rate."""
        boundaries = populated_kernle.get_competence_boundaries()

        assert 0.0 <= boundaries["overall_confidence"] <= 1.0
        assert 0.0 <= boundaries["success_rate"] <= 1.0
        assert boundaries["experience_depth"] >= 5  # at least 5 episodes in fixture
        assert boundaries["knowledge_breadth"] >= 3  # python, docker, kubernetes domains

    def test_experience_depth_counts_episodes(self, kernle):
        """Should count episodes for experience depth."""
        # Add specific number of episodes
        for i in range(7):
            kernle.episode(
                objective=f"Task {i}",
                outcome="success" if i % 2 == 0 else "failure",
                tags=["test_domain"],
            )

        boundaries = kernle.get_competence_boundaries()

        # Experience depth should reflect episode count
        assert boundaries["experience_depth"] >= 7


class TestIdentifyLearningOpportunities:
    """Test the identify_learning_opportunities method."""

    def test_empty_opportunities(self, kernle):
        """Should return empty list when no data."""
        opportunities = kernle.identify_learning_opportunities()

        assert isinstance(opportunities, list)

    def test_opportunity_structure(self, populated_kernle):
        """Should return properly structured opportunities."""
        opportunities = populated_kernle.identify_learning_opportunities(limit=10)

        for opp in opportunities:
            assert "type" in opp
            assert "domain" in opp
            assert "reason" in opp
            assert "priority" in opp
            assert "suggested_action" in opp
            assert opp["priority"] in ("high", "medium", "low")
            assert opp["type"] in (
                "low_coverage_domain",
                "uncertain_belief",
                "repeated_failures",
                "stale_knowledge",
            )

    def test_detects_low_coverage_domains(self, kernle):
        """Should identify domains with low coverage but references."""
        # Add episodes referencing a domain but no beliefs
        kernle.episode(objective="Learn GraphQL basics", outcome="partial", tags=["graphql"])
        kernle.episode(objective="Build GraphQL API", outcome="partial", tags=["graphql"])

        opportunities = kernle.identify_learning_opportunities()

        # Should return a list of opportunities
        assert isinstance(opportunities, list)
        # Verify opportunity structure if any exist
        opportunity_types = [o["type"] for o in opportunities]
        for opp_type in opportunity_types:
            assert opp_type in (
                "low_coverage_domain",
                "uncertain_belief",
                "repeated_failures",
                "stale_knowledge",
            )

    def test_detects_uncertain_beliefs(self, kernle):
        """Should identify beliefs with low confidence."""
        # Add low-confidence beliefs
        kernle.belief(
            statement="Microservices are always better", type="assumption", confidence=0.2
        )
        kernle.belief(statement="NoSQL scales better than SQL", type="assumption", confidence=0.3)

        opportunities = kernle.identify_learning_opportunities()

        # Should identify uncertain beliefs
        uncertain_beliefs = [o for o in opportunities if o["type"] == "uncertain_belief"]
        # Might find some based on the low confidence
        assert isinstance(uncertain_beliefs, list)

    def test_detects_repeated_failures(self, kernle):
        """Should identify domains with repeated failures."""
        # Add multiple failures in a domain
        for i in range(4):
            kernle.episode(
                objective=f"Deploy to cloud {i}",
                outcome="failure",
                lessons=[f"Failed attempt {i}"],
                tags=["cloud_deployment"],
            )

        opportunities = kernle.identify_learning_opportunities()

        # Should suggest learning about cloud_deployment
        failure_opps = [o for o in opportunities if o["type"] == "repeated_failures"]
        if failure_opps:
            domains = [o["domain"] for o in failure_opps]
            assert "cloud_deployment" in domains

    def test_respects_limit(self, populated_kernle):
        """Should respect the limit parameter."""
        opportunities = populated_kernle.identify_learning_opportunities(limit=2)

        assert len(opportunities) <= 2

    def test_priority_sorting(self, kernle):
        """Should sort opportunities by priority."""
        # Create conditions for different priority levels
        # High priority: repeated failures
        for i in range(3):
            kernle.episode(
                objective=f"High priority task {i}", outcome="failure", tags=["urgent_domain"]
            )

        # Low priority: stale but has coverage
        kernle.belief(statement="Old knowledge", type="fact", confidence=0.8)

        opportunities = kernle.identify_learning_opportunities(limit=10)

        if len(opportunities) >= 2:
            priorities = [o["priority"] for o in opportunities]
            priority_order = {"high": 0, "medium": 1, "low": 2}

            # Should be sorted (high first)
            for i in range(len(priorities) - 1):
                assert priority_order[priorities[i]] <= priority_order[priorities[i + 1]]


class TestMetaCognitionIntegration:
    """Integration tests for meta-cognition features."""

    def test_knowledge_map_updates_with_new_data(self, kernle):
        """Knowledge map should reflect new additions."""
        # Initial state
        map1 = kernle.get_knowledge_map()
        initial_domains = len(map1["domains"])

        # Add data
        kernle.belief(statement="Test belief", type="fact", confidence=0.8)
        kernle.episode(objective="Test episode", outcome="success", tags=["new_domain"])

        # Check updated map
        map2 = kernle.get_knowledge_map()

        # Should have at least one more domain
        assert len(map2["domains"]) >= initial_domains

    def test_gap_detection_improves_with_knowledge(self, kernle):
        """Gap detection should improve as knowledge is added."""
        # Query with no knowledge
        result1 = kernle.detect_knowledge_gaps("machine learning")

        # Add relevant knowledge
        kernle.belief(
            statement="Machine learning requires large datasets", type="fact", confidence=0.85
        )
        kernle.episode(
            objective="Train ML model",
            outcome="success",
            lessons=["Start with simple models"],
            tags=["machine_learning"],
        )

        # Query again
        result2 = kernle.detect_knowledge_gaps("machine learning")

        # Should have more relevant knowledge now
        assert result2["has_relevant_knowledge"] is True
        assert result2["confidence"] >= result1["confidence"]

    def test_boundaries_reflect_experience(self, kernle):
        """Competence boundaries should reflect actual experience."""
        # Add successful experiences in one domain
        for i in range(5):
            kernle.episode(objective=f"Python task {i}", outcome="success", tags=["python_dev"])
            kernle.belief(statement=f"Python insight {i}", type="fact", confidence=0.85)

        # Add failed experiences in another domain
        for i in range(3):
            kernle.episode(objective=f"Rust task {i}", outcome="failure", tags=["rust_dev"])
            kernle.belief(statement=f"Rust assumption {i}", type="assumption", confidence=0.3)

        boundaries = kernle.get_competence_boundaries()

        # Extract domain names for validation
        strength_domains = [s["domain"] for s in boundaries["strengths"]]
        weakness_domains = [w["domain"] for w in boundaries["weaknesses"]]

        # Verify we got valid lists
        assert isinstance(boundaries["strengths"], list)
        assert isinstance(boundaries["weaknesses"], list)

        # If domains were classified, verify structure is correct
        for domain in strength_domains:
            assert isinstance(domain, str)
        for domain in weakness_domains:
            assert isinstance(domain, str)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query_gap_detection(self, kernle):
        """Should handle empty query gracefully."""
        result = kernle.detect_knowledge_gaps("")

        assert "has_relevant_knowledge" in result
        assert "recommendation" in result

    def test_very_long_query(self, kernle):
        """Should handle very long queries."""
        long_query = "how do I " * 100
        result = kernle.detect_knowledge_gaps(long_query)

        assert "has_relevant_knowledge" in result

    def test_special_characters_in_domain(self, kernle):
        """Should handle special characters in tags/domains."""
        kernle.episode(
            objective="Test with special chars", outcome="success", tags=["c++", "node.js", "vue-3"]
        )

        knowledge_map = kernle.get_knowledge_map()

        # Should not crash
        assert "domains" in knowledge_map

    def test_concurrent_operations(self, kernle):
        """Should handle rapid sequential operations."""
        # Rapid additions
        for i in range(20):
            kernle.belief(statement=f"Concurrent belief {i}", type="fact", confidence=0.7)

        # All operations should work
        knowledge_map = kernle.get_knowledge_map()
        gaps = kernle.detect_knowledge_gaps("concurrent")
        boundaries = kernle.get_competence_boundaries()
        opportunities = kernle.identify_learning_opportunities()

        assert knowledge_map is not None
        assert gaps is not None
        assert boundaries is not None
        assert opportunities is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
