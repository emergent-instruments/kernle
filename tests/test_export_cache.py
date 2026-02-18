"""Tests for export_cache functionality."""

import os
import tempfile
from pathlib import Path

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance with isolated temp DB."""
    db_path = tmp_path / "test_export.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    storage = SQLiteStorage(stack_id="test-export", db_path=db_path)
    instance = Kernle(
        stack_id="test-export", storage=storage, checkpoint_dir=checkpoint_dir, strict=False
    )
    bind_noop_model(instance)
    yield instance
    storage.close()


class TestExportCache:
    def test_empty_cache(self, k):
        """Empty agent produces valid cache with header."""
        content = k.export_cache()
        assert "# MEMORY.md" in content
        assert "AUTO-GENERATED" in content
        assert "Do not edit manually" in content

    def test_beliefs_included(self, k):
        """Beliefs above min_confidence appear in cache."""
        k.belief("The sky is blue", confidence=0.9)
        k.belief("Water is wet", confidence=0.8)
        k.belief("Low confidence thing", confidence=0.2)

        content = k.export_cache(min_confidence=0.5)
        assert "sky is blue" in content
        assert "Water is wet" in content
        assert "Low confidence thing" not in content

    def test_beliefs_max_limit(self, k):
        """Max beliefs parameter limits output."""
        for i in range(10):
            k.belief(f"Belief number {i}", confidence=0.9 - i * 0.01)

        content = k.export_cache(max_beliefs=3)
        # Count belief lines specifically (format: "- [NN%] ...")
        belief_lines = [
            line for line in content.split("\n") if line.startswith("- [") and "%]" in line
        ]
        assert len(belief_lines) == 3

    def test_values_included(self, k):
        """Values appear in cache."""
        k.value("honesty", "Always tell the truth", priority=90)
        content = k.export_cache()
        assert "honesty" in content
        assert "Always tell the truth" in content

    def test_goals_included(self, k):
        """Active goals appear in cache."""
        k.goal("Ship v1.0", priority=80)
        content = k.export_cache()
        assert "Ship v1.0" in content

    def test_checkpoint_included(self, k):
        """Checkpoint appears in cache by default."""
        k.checkpoint("Working on export feature", pending=["Write tests"])
        content = k.export_cache()
        assert "Working on export feature" in content
        assert "Write tests" in content

    def test_checkpoint_excluded(self, k):
        """Checkpoint can be excluded."""
        k.checkpoint("Working on export feature")
        content = k.export_cache(include_checkpoint=False)
        assert "Working on export feature" not in content

    def test_write_to_file(self, k):
        """Can write cache to file."""
        k.belief("Test belief", confidence=0.9)

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name

        try:
            content = k.export_cache(path=path)
            assert Path(path).exists()
            file_content = Path(path).read_text()
            assert file_content == content
            assert "Test belief" in file_content
        finally:
            os.unlink(path)

    def test_relationships_included(self, k):
        """Key relationships appear in cache."""
        k.relationship("Sean", trust_level=0.9, notes="My human steward", entity_type="person")
        content = k.export_cache()
        assert "Sean" in content
        assert "person" in content

    def test_min_confidence_clamped(self, k):
        """min_confidence is clamped to 0.0-1.0 range."""
        k.belief("High conf", confidence=0.9)
        k.belief("Low conf", confidence=0.3)

        # min_confidence > 1.0 should be clamped to 1.0 (nothing passes)
        content = k.export_cache(min_confidence=5.0)
        assert "High conf" not in content

        # min_confidence < 0.0 should be clamped to 0.0 (everything passes)
        content = k.export_cache(min_confidence=-1.0)
        assert "High conf" in content
        assert "Low conf" in content

    def test_max_beliefs_clamped(self, k):
        """max_beliefs is clamped to 1-1000 range."""
        k.belief("Only belief", confidence=0.9)

        # max_beliefs=0 should be clamped to 1
        content = k.export_cache(max_beliefs=0)
        assert "Only belief" in content

    def test_relationship_notes_truncation(self, k):
        """Long relationship notes get truncated with ellipsis."""
        long_notes = "A" * 200
        k.relationship("Someone", trust_level=0.5, notes=long_notes, entity_type="person")
        content = k.export_cache()
        # Should be truncated with ...
        assert "A" * 80 + "..." in content
        assert "A" * 81 not in content

    def test_idempotent(self, k):
        """Running export twice produces same output (excluding timestamp)."""
        k.belief("Stable belief", confidence=0.9)
        k.value("consistency", "Be consistent", priority=85)

        content1 = k.export_cache()
        content2 = k.export_cache()

        # Remove timestamp lines for comparison
        def strip_timestamp(s):
            return "\n".join(line for line in s.split("\n") if "AUTO-GENERATED" not in line)

        assert strip_timestamp(content1) == strip_timestamp(content2)
