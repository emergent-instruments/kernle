"""Coverage tests for kernle/importers/json_importer.py.

Targets uncovered lines: duplicate checking for each import type,
error handling during import, and unknown type handling.
"""

import json

from kernle.importers.json_importer import (
    JsonImporter,
    JsonImportItem,
    _import_json_item,
)


class TestImportJsonItemDuplicateEpisode:
    """Test episode duplicate checking — covers lines 180-186."""

    def test_episode_skip_duplicate_by_objective_and_outcome(self, kernle_instance):
        """Skip episode import when duplicate found by objective+outcome."""
        k, storage = kernle_instance

        # Create an existing episode
        k.episode(objective="Existing objective", outcome="Existing outcome")

        item = JsonImportItem(
            type="episode",
            data={"objective": "Existing objective", "outcome": "Existing outcome"},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None

    def test_episode_import_when_no_duplicate(self, kernle_instance):
        """Import episode when no duplicate exists."""
        k, storage = kernle_instance

        item = JsonImportItem(
            type="episode",
            data={
                "objective": "New unique objective",
                "outcome": "New unique outcome",
                "outcome_type": "success",
                "tags": ["test"],
            },
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is not None


class TestImportJsonItemDuplicateNote:
    """Test note duplicate checking — covers lines 204-208."""

    def test_note_skip_duplicate_by_content(self, kernle_instance):
        """Skip note import when duplicate found by content."""
        k, storage = kernle_instance

        # Create an existing note
        k.note(content="Existing note content")

        item = JsonImportItem(
            type="note",
            data={"content": "Existing note content"},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None

    def test_note_import_when_no_duplicate(self, kernle_instance):
        """Import note when no duplicate exists."""
        k, storage = kernle_instance

        item = JsonImportItem(
            type="note",
            data={
                "content": "Unique note content",
                "type": "insight",
                "speaker": "Agent",
                "reason": "Observation",
                "tags": ["test"],
            },
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is not None


class TestImportJsonItemDuplicateValue:
    """Test value duplicate checking — covers lines 236-239."""

    def test_value_skip_duplicate_by_name(self, kernle_instance):
        """Skip value import when duplicate found by name."""
        k, storage = kernle_instance

        # Create an existing value
        k.value(name="Quality", statement="Code quality matters", priority=80)

        # Try to import same value name
        item = JsonImportItem(
            type="value",
            data={"name": "Quality", "statement": "Different statement"},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None

    def test_value_import_when_no_duplicate(self, kernle_instance):
        """Import value when no duplicate exists."""
        k, storage = kernle_instance

        item = JsonImportItem(
            type="value",
            data={
                "name": "Unique Value",
                "description": "Fallback description",
                "priority": 60,
            },
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is not None


class TestImportJsonItemDuplicateGoal:
    """Test goal duplicate checking — covers lines 251-254."""

    def test_goal_skip_duplicate_by_title(self, kernle_instance):
        """Skip goal import when duplicate found by title."""
        k, storage = kernle_instance

        # Create an existing goal
        k.goal(title="Ship v1.0", description="Release first version")

        # Try to import same goal title
        item = JsonImportItem(
            type="goal",
            data={"title": "Ship v1.0", "description": "Same goal"},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None

    def test_goal_skip_duplicate_by_description(self, kernle_instance):
        """Skip goal import when duplicate found by description."""
        k, storage = kernle_instance

        k.goal(title="Different title", description="Release first version")

        item = JsonImportItem(
            type="goal",
            data={"title": "Completely new title", "description": "Release first version"},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None

    def test_goal_import_when_no_duplicate(self, kernle_instance):
        """Import goal when no duplicate exists."""
        k, storage = kernle_instance

        item = JsonImportItem(
            type="goal",
            data={
                "title": "New unique goal",
                "description": "Unique description",
                "priority": "high",
            },
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is not None


class TestImportJsonItemDuplicateDrive:
    """Test drive duplicate checking — covers lines 266-268."""

    def test_drive_skip_duplicate_by_type(self, kernle_instance):
        """Skip drive import when duplicate found by drive_type."""
        k, storage = kernle_instance

        # Create an existing drive
        k.drive(drive_type="curiosity", intensity=0.8)

        # Try to import same drive type
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "curiosity", "intensity": 0.5},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None

    def test_drive_import_when_no_duplicate(self, kernle_instance):
        """Import drive when no duplicate exists."""
        k, storage = kernle_instance

        item = JsonImportItem(
            type="drive",
            data={
                "drive_type": "growth",
                "intensity": 0.7,
                "focus_areas": ["area1"],
            },
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is not None


class TestImportJsonItemDuplicateRelationship:
    """Test relationship duplicate checking — covers lines 280-282."""

    def test_relationship_skip_duplicate_by_entity_name(self, kernle_instance):
        """Skip relationship import when duplicate found by entity_name."""
        k, storage = kernle_instance

        # Create an existing relationship
        k.relationship(
            other_stack_id="Alice",
            entity_type="person",
            interaction_type="collaborator",
        )

        # Try to import same entity_name
        item = JsonImportItem(
            type="relationship",
            data={
                "entity_name": "Alice",
                "entity_type": "person",
                "relationship_type": "friend",
                "sentiment": 0.5,
            },
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None

    def test_relationship_import_when_no_duplicate(self, kernle_instance):
        """Import relationship when no duplicate exists."""
        k, storage = kernle_instance

        item = JsonImportItem(
            type="relationship",
            data={
                "entity_name": "UniqueEntity",
                "entity_type": "system",
                "relationship_type": "monitors",
                "sentiment": 0.3,
                "notes": "System monitoring",
            },
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is not None


class TestImportJsonItemDuplicateRaw:
    """Test raw duplicate checking — covers lines 302-305."""

    def test_raw_skip_duplicate_by_content(self, kernle_instance):
        """Skip raw import when duplicate found by content."""
        k, storage = kernle_instance

        # Create an existing raw entry
        k.raw(blob="Existing raw content", source="test")

        # Try to import same content
        item = JsonImportItem(
            type="raw",
            data={"content": "Existing raw content", "source": "import"},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None

    def test_raw_import_when_no_duplicate(self, kernle_instance):
        """Import raw when no duplicate exists."""
        k, storage = kernle_instance

        item = JsonImportItem(
            type="raw",
            data={"content": "Unique raw content", "source": "json-import"},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is not None


class TestImportJsonItemUnknownType:
    """Test unknown type handling — covers line 313."""

    def test_unknown_type_returns_none(self, kernle_instance):
        """Unknown memory type returns None (not imported)."""
        k, storage = kernle_instance

        item = JsonImportItem(
            type="unknown_type",
            data={"content": "something"},
        )
        result = _import_json_item(item, k, skip_duplicates=True)
        assert result is None


class TestImportToErrorHandling:
    """Test error handling in import_to() — covers lines 88-89."""

    def test_import_error_captured_in_errors_list(self, tmp_path, kernle_instance):
        """Errors during import are captured, not raised."""
        k, storage = kernle_instance

        # Build a JSON file with valid provenance: raw -> episode
        json_file = tmp_path / "errors.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw entry"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "good objective",
                            "outcome": "good outcome",
                            "derived_from": ["raw:r1"],
                        },
                        {
                            "id": "e2",
                            "objective": "",
                            "outcome": "will error",
                            "derived_from": ["raw:r1"],
                        },
                    ],
                }
            )
        )

        importer = JsonImporter(str(json_file))
        importer.parse()

        # Monkey-patch k.episode to raise on empty objectives
        original_episode = k.episode

        def error_episode(**kwargs):
            if kwargs.get("objective") == "":
                raise RuntimeError("Test error during import")
            return original_episode(**kwargs)

        k.episode = error_episode

        result = importer.import_to(k, dry_run=False, skip_duplicates=False)

        # The raw entry and good episode should import fine
        assert result["imported"].get("raw", 0) == 1
        assert result["imported"].get("episode", 0) == 1
        # The bad episode should have an error
        assert len(result["errors"]) >= 1
        assert "episode" in result["errors"][0]
