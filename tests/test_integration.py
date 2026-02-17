"""Integration tests for CLI → Kernle → Storage flow.

These tests verify that the full stack works end-to-end,
not just individual components in isolation.
"""

import pytest

from kernle import Kernle
from kernle.storage.sqlite import SQLiteStorage


class TestCLIIntegration:
    """Test CLI commands persist to actual storage."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary SQLite storage."""
        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(stack_id="test_integration", db_path=db_path)
        yield storage
        storage.close()

    @pytest.fixture
    def temp_kernle(self, tmp_path):
        """Create a Kernle instance with temp storage."""
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        storage = SQLiteStorage(stack_id="test_integration", db_path=db_path)
        k = Kernle(
            stack_id="test_integration",
            storage=storage,
            checkpoint_dir=checkpoint_dir,
            strict=False,
        )
        yield k
        storage.close()

    def test_episode_command_persists(self, temp_kernle):
        """CLI episode command should persist to actual storage."""
        k = temp_kernle

        # Record an episode
        episode_id = k.episode(
            objective="Test integration objective",
            outcome="Test completed successfully",
            lessons=["Integration tests are important"],
        )

        # Verify in storage
        episode = k._storage.get_episode(episode_id)

        assert episode is not None
        assert episode.objective == "Test integration objective"
        assert "Integration tests are important" in (episode.lessons or [])

    def test_note_command_persists(self, temp_kernle):
        """CLI note command should persist to actual storage."""
        k = temp_kernle

        # Record a note
        _note_id = k.note(
            content="Important decision about architecture",
            type="decision",
            reason="Better maintainability",
        )

        # Verify in storage
        notes = k._storage.get_notes(limit=10)

        assert len(notes) >= 1
        assert any("architecture" in n.content.lower() for n in notes)

    def test_checkpoint_save_load_cycle(self, temp_kernle):
        """Checkpoint should save and load correctly."""
        k = temp_kernle

        # Save checkpoint
        k.checkpoint(
            task="Integration testing",
            pending=["Add more tests", "Fix bugs"],
            context="Testing checkpoint persistence",
        )

        # Load checkpoint
        cp = k.load_checkpoint()

        assert cp is not None
        assert cp["current_task"] == "Integration testing"
        assert "Add more tests" in cp["pending"]

    def test_raw_capture_and_process(self, temp_kernle):
        """Raw capture and process should work end-to-end."""
        k = temp_kernle

        # Capture raw thought
        raw_id = k.raw("This is a raw thought that should become an episode")

        # Verify raw exists
        raw_entry = k.get_raw(raw_id)
        assert raw_entry is not None
        assert raw_entry["processed"] is False

        # Process into episode
        episode_id = k.process_raw(
            raw_id=raw_id,
            as_type="episode",
            objective="Captured thought",
            outcome="Processed successfully",
        )

        # Verify raw is marked processed
        raw_entry = k.get_raw(raw_id)
        assert raw_entry["processed"] is True

        # Verify episode exists
        episode = k._storage.get_episode(episode_id)
        assert episode is not None

    def test_belief_revision_flow(self, temp_kernle):
        """Belief creation, reinforcement, and supersession should work."""
        k = temp_kernle

        # Create initial belief
        belief_id = k.belief(statement="Integration tests catch bugs", confidence=0.7)

        # Reinforce it
        k.reinforce_belief(belief_id)

        # Check confidence increased - get beliefs and find ours
        beliefs = k._storage.get_beliefs(limit=100)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief is not None
        assert belief.confidence > 0.7
        assert belief.times_reinforced > 0

        # Supersede with new belief (delegates to revise_belief)
        k.supersede_belief(
            old_id=belief_id,
            new_statement="Integration tests are essential for quality",
            confidence=0.9,
            reason="Experience confirmed this",
        )

        # Verify old belief is inactive
        beliefs = k._storage.get_beliefs(limit=100, include_inactive=True)
        old_belief = next((b for b in beliefs if b.id == belief_id), None)
        assert old_belief is not None
        assert old_belief.is_active is False
        # v0.14+: chain fields no longer written; revision tracked via audit log
        assert old_belief.superseded_by is None

    def test_search_finds_recent_entries(self, temp_kernle):
        """Search should find recently added entries."""
        k = temp_kernle

        # Add some entries with unique content
        k.episode(
            objective="Implement search functionality",
            outcome="Search works correctly",
            lessons=["Indexing is important"],
        )

        k.note(content="Search should be fast and accurate", type="insight")

        k.belief(statement="Good search improves productivity", confidence=0.8)

        # Search should find them
        results = k.search("search", limit=10)

        assert len(results) >= 2  # Should find episode and note at minimum

    def test_load_returns_working_memory(self, temp_kernle):
        """Load should return comprehensive working memory."""
        k = temp_kernle

        # Add some data
        k.value("test_value", "Testing is important", priority=90)
        k.belief("Tests improve code quality", confidence=0.9)
        k.goal("Complete all tests", priority="high")
        k.episode("First test session", "Success")
        k.checkpoint("Integration testing", pending=["More tests"])

        # Load working memory
        memory = k.load()

        assert "checkpoint" in memory
        assert "values" in memory
        assert "beliefs" in memory
        assert "goals" in memory
        assert "lessons" in memory

        # Verify content
        assert any(v["name"] == "test_value" for v in memory["values"])
        assert any("high" in g["priority"] for g in memory["goals"])


class TestStorageIntegration:
    """Test storage layer integration."""

    def test_sqlite_roundtrip(self, tmp_path):
        """Data should survive SQLite save/load cycle."""
        db_path = tmp_path / "test.db"
        stack_id = "test_roundtrip"

        # Create and populate storage
        storage1 = SQLiteStorage(stack_id=stack_id, db_path=db_path)

        from datetime import datetime, timezone

        from kernle.storage import Belief, Episode

        episode = Episode(
            id="ep-test",
            stack_id=stack_id,
            objective="Test roundtrip",
            outcome="Successful",
            created_at=datetime.now(timezone.utc),
        )
        storage1.save_episode(episode)

        belief = Belief(
            id="bel-test",
            stack_id=stack_id,
            statement="SQLite is reliable",
            confidence=0.95,
            created_at=datetime.now(timezone.utc),
        )
        storage1.save_belief(belief)

        storage1.close()

        # Load in new storage instance
        storage2 = SQLiteStorage(stack_id=stack_id, db_path=db_path)

        loaded_episode = storage2.get_episode("ep-test")
        beliefs = storage2.get_beliefs(limit=100)
        loaded_belief = next((b for b in beliefs if b.id == "bel-test"), None)

        assert loaded_episode is not None
        assert loaded_episode.objective == "Test roundtrip"

        assert loaded_belief is not None
        assert loaded_belief.statement == "SQLite is reliable"
        assert loaded_belief.confidence == 0.95

        storage2.close()

    def test_unicode_content_preserved(self, tmp_path):
        """Unicode and emoji content should be preserved."""
        db_path = tmp_path / "test.db"
        stack_id = "test_unicode"

        storage = SQLiteStorage(stack_id=stack_id, db_path=db_path)

        from datetime import datetime, timezone

        from kernle.storage import Episode

        # Test various unicode content
        episode = Episode(
            id="ep-unicode",
            stack_id=stack_id,
            objective="Test 🎯 unicode with émojis and spëcial chârs",
            outcome="Success ✅ 日本語 العربية",
            lessons=["Unicode 🌍 works", "Émojis 😊 preserved"],
            created_at=datetime.now(timezone.utc),
        )
        storage.save_episode(episode)

        # Retrieve and verify
        loaded = storage.get_episode("ep-unicode")

        assert "🎯" in loaded.objective
        assert "✅" in loaded.outcome
        assert "日本語" in loaded.outcome
        assert any("🌍" in lesson for lesson in (loaded.lessons or []))

        storage.close()

    def test_large_content_handling(self, tmp_path):
        """Large content should be handled without truncation."""
        db_path = tmp_path / "test.db"
        stack_id = "test_large"

        storage = SQLiteStorage(stack_id=stack_id, db_path=db_path)

        from datetime import datetime, timezone

        from kernle.storage import Episode

        # Create large content (near the 2000 char limit)
        large_content = "x" * 1900

        episode = Episode(
            id="ep-large",
            stack_id=stack_id,
            objective=large_content,
            outcome="Completed",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_episode(episode)

        # Verify full content is preserved
        loaded = storage.get_episode("ep-large")
        assert len(loaded.objective) == 1900
        assert loaded.objective == large_content

        storage.close()


class TestAnxietyIntegration:
    """Test anxiety tracking integration."""

    def test_anxiety_reflects_state(self, tmp_path):
        """Anxiety should reflect actual memory state."""
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        stack_id = "test_anxiety"

        storage = SQLiteStorage(stack_id=stack_id, db_path=db_path)
        k = Kernle(stack_id=stack_id, storage=storage, checkpoint_dir=checkpoint_dir, strict=False)

        # Fresh state should have low anxiety
        report1 = k.get_anxiety_report()

        # Add some unreflected episodes (no lessons)
        for i in range(5):
            k.episode(f"Task {i}", "Done")  # No lessons = unreflected

        # Add some low-confidence beliefs
        for i in range(3):
            k.belief(f"Uncertain belief {i}", confidence=0.3)

        # Anxiety should be higher now
        report2 = k.get_anxiety_report()

        # Consolidation debt and memory uncertainty should increase
        assert (
            report2["dimensions"]["consolidation_debt"]["score"]
            > report1["dimensions"]["consolidation_debt"]["score"]
        )
        assert (
            report2["dimensions"]["memory_uncertainty"]["score"]
            >= report1["dimensions"]["memory_uncertainty"]["score"]
        )
