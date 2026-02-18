"""
Tests for logic audit fixes (LOGIC_AUDIT_2.md issues).

These tests verify the fixes for:
1. CRITICAL: Division by zero in get_identity_confidence()
2. CRITICAL: Race condition in _queue_sync() / queue_sync_operation()
3. HIGH: Off-by-one in confidence history tracking
4. HIGH: Null handling in detect_emotion()
"""

import threading
import uuid
from unittest.mock import patch

import pytest

from kernle import Kernle
from kernle.storage.sqlite import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def kernle_fresh(tmp_path):
    """Create a fresh Kernle instance for testing."""
    db_path = tmp_path / "test_logic_audit.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    k = Kernle(stack_id="test_agent", storage=storage, strict=False)
    bind_noop_model(k)
    return k


class TestDivisionByZeroProtection:
    """Test that get_identity_confidence() handles empty collections safely."""

    def test_identity_confidence_with_empty_collections(self, kernle_instance):
        """Should return 0.0 without division by zero when all collections are empty."""
        kernle, storage = kernle_instance

        # Ensure no data exists
        confidence = kernle.get_identity_confidence()

        # Should not crash and return 0.0
        assert confidence == 0.0

    def test_identity_confidence_with_proxy_like_empty_list(self, kernle_instance):
        """Should handle edge case where collection is truthy but len() is 0."""
        kernle, storage = kernle_instance

        # Create a mock storage that returns a truthy but empty object
        class TruthyEmptyList(list):
            """A list that's truthy even when empty (simulates proxy objects)."""

            def __bool__(self):
                return True

        # Patch the storage methods to return truthy empty lists
        with patch.object(storage, "get_values", return_value=TruthyEmptyList()):
            with patch.object(storage, "get_beliefs", return_value=TruthyEmptyList()):
                with patch.object(storage, "get_episodes", return_value=TruthyEmptyList()):
                    # This should not raise ZeroDivisionError
                    confidence = kernle.get_identity_confidence()

        # Should handle gracefully
        assert confidence >= 0.0
        assert confidence <= 1.0


class TestSyncQueueRaceCondition:
    """Test that sync queue operations are atomic and race-condition free."""

    def test_queue_sync_operation_is_atomic(self, sqlite_storage):
        """The queue_sync_operation should use UPSERT for atomicity."""
        storage = sqlite_storage

        # Queue the same operation twice quickly
        storage.queue_sync_operation("upsert", "episodes", "test-id-1", {"test": "data1"})
        storage.queue_sync_operation("upsert", "episodes", "test-id-1", {"test": "data2"})

        # Should not have duplicate entries - the second should update the first
        pending = storage.get_pending_sync_operations(limit=100)
        matching = [p for p in pending if p["record_id"] == "test-id-1"]

        # Should only have one entry (upsert behavior)
        assert len(matching) == 1
        # Should have the latest data
        assert matching[0]["data"]["test"] == "data2"

    def test_concurrent_queue_operations_no_duplicates(self, sqlite_storage):
        """Multiple threads queuing for the same record should not create duplicates."""
        storage = sqlite_storage
        record_id = f"concurrent-test-{uuid.uuid4()}"
        results = []
        errors = []

        def queue_operation(thread_num):
            try:
                result = storage.queue_sync_operation(
                    "upsert", "episodes", record_id, {"thread": thread_num}
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads that all try to queue for the same record
        threads = []
        for i in range(5):
            t = threading.Thread(target=queue_operation, args=(i,))
            threads.append(t)

        # Start all threads nearly simultaneously
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Check there's only one pending entry for this record
        pending = storage.get_pending_sync_operations(limit=100)
        matching = [p for p in pending if p["record_id"] == record_id]

        assert len(matching) == 1, f"Expected 1 entry, got {len(matching)}"


class TestConfidenceHistoryAccuracy:
    """Test that reinforce_belief tracks old/new confidence accurately."""

    def test_confidence_history_records_accurate_old_value(self, kernle_fresh):
        """The 'old' value in history should be the actual previous confidence."""
        k = kernle_fresh

        # Create belief with known confidence
        initial_confidence = 0.6
        belief_id = k.belief("Testing is important", confidence=initial_confidence)

        # Reinforce it
        k.reinforce_belief(belief_id)

        # Get the belief and check history
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)

        assert belief is not None
        assert belief.confidence_history is not None
        assert len(belief.confidence_history) > 0

        # The 'old' value should match the initial confidence
        history_entry = belief.confidence_history[-1]
        assert history_entry["old"] == round(initial_confidence, 3)

        # The 'new' value should be higher than old
        assert history_entry["new"] > history_entry["old"]

    def test_multiple_reinforcements_track_chain(self, kernle_fresh):
        """Multiple reinforcements should create accurate history chain."""
        k = kernle_fresh

        belief_id = k.belief("Testing is important", confidence=0.5)

        # Reinforce multiple times
        for _ in range(3):
            k.reinforce_belief(belief_id)

        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)

        # Check we have 3 history entries
        assert len(belief.confidence_history) == 3

        # Each 'new' should equal the next 'old'
        for i in range(len(belief.confidence_history) - 1):
            current_new = belief.confidence_history[i]["new"]
            next_old = belief.confidence_history[i + 1]["old"]
            assert current_new == next_old, f"History chain broken at index {i}"


class TestNullHandlingInEmotionDetection:
    """Test that detect_emotion handles None/empty input gracefully."""

    def test_detect_emotion_with_none(self, kernle_fresh):
        """Should return neutral result for None input, not crash."""
        k = kernle_fresh

        # This should not raise AttributeError
        result = k.detect_emotion(None)

        assert result is not None
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["tags"] == []
        assert result["confidence"] == 0.0

    def test_detect_emotion_with_empty_string(self, kernle_fresh):
        """Should return neutral result for empty string."""
        k = kernle_fresh

        result = k.detect_emotion("")

        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["tags"] == []
        assert result["confidence"] == 0.0

    def test_detect_emotion_with_valid_text(self, kernle_fresh):
        """Without inference, returns neutral defaults for valid text."""
        k = kernle_fresh

        result = k.detect_emotion("I'm really happy and excited!")

        assert result is not None
        # Without inference, returns neutral defaults
        assert result["valence"] == 0.0
        assert result["tags"] == []
