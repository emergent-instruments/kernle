"""Coverage tests for kernle/importers/json_importer.py.

Targets uncovered lines: duplicate checking for each import type,
error handling during import, and unknown type handling.

All tests use the public JsonImporter.import_to() API with JSON fixtures
containing proper provenance chains. Duplicate tests include duplicate
items within the same import payload (since import_to requires empty stacks).
"""

import json

from kernle.importers.json_importer import JsonImporter


def _write_json(tmp_path, name, data):
    """Write a JSON fixture file and return the path string."""
    json_file = tmp_path / name
    json_file.write_text(json.dumps(data))
    return str(json_file)


class TestImportJsonItemDuplicateEpisode:
    """Test episode duplicate checking via import_to with duplicate items in payload."""

    def test_episode_skip_duplicate_by_objective_and_outcome(self, tmp_path, kernle_instance):
        """Skip episode import when duplicate found by objective+outcome within same payload."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "dup_episodes.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "Same objective",
                        "outcome": "Same outcome",
                        "derived_from": ["raw:r1"],
                    },
                    {
                        "id": "e2",
                        "objective": "Same objective",
                        "outcome": "Same outcome",
                        "derived_from": ["raw:r1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("episode", 0) == 1
        assert result["skipped"].get("episode", 0) == 1

    def test_episode_import_when_no_duplicate(self, tmp_path, kernle_instance):
        """Import episode when no duplicate exists."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "unique_episodes.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "Unique objective A",
                        "outcome": "Unique outcome A",
                        "outcome_type": "success",
                        "tags": ["test"],
                        "derived_from": ["raw:r1"],
                    },
                    {
                        "id": "e2",
                        "objective": "Unique objective B",
                        "outcome": "Unique outcome B",
                        "derived_from": ["raw:r1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("episode", 0) == 2
        assert result["skipped"].get("episode", 0) == 0


class TestImportJsonItemDuplicateNote:
    """Test note duplicate checking via import_to with duplicate items in payload."""

    def test_note_skip_duplicate_by_content(self, tmp_path, kernle_instance):
        """Skip note import when duplicate found by content within same payload."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "dup_notes.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "notes": [
                    {
                        "id": "n1",
                        "content": "Duplicate note content",
                        "derived_from": ["raw:r1"],
                    },
                    {
                        "id": "n2",
                        "content": "Duplicate note content",
                        "derived_from": ["raw:r1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("note", 0) == 1
        assert result["skipped"].get("note", 0) == 1

    def test_note_import_when_no_duplicate(self, tmp_path, kernle_instance):
        """Import note when no duplicate exists."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "unique_notes.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "notes": [
                    {
                        "id": "n1",
                        "content": "Unique note A",
                        "type": "insight",
                        "speaker": "Agent",
                        "reason": "Observation",
                        "tags": ["test"],
                        "derived_from": ["raw:r1"],
                    },
                    {
                        "id": "n2",
                        "content": "Unique note B",
                        "derived_from": ["raw:r1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("note", 0) == 2
        assert result["skipped"].get("note", 0) == 0


class TestImportJsonItemDuplicateValue:
    """Test value duplicate checking via import_to with duplicate items in payload."""

    def test_value_skip_duplicate_by_name(self, tmp_path, kernle_instance):
        """Skip value import when duplicate found by name within same payload."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "dup_values.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "observe",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "beliefs": [
                    {
                        "id": "b1",
                        "statement": "Quality matters",
                        "confidence": 0.9,
                        "derived_from": ["episode:e1"],
                    }
                ],
                "values": [
                    {
                        "id": "v1",
                        "name": "Quality",
                        "statement": "Code quality matters",
                        "priority": 80,
                        "derived_from": ["belief:b1"],
                    },
                    {
                        "id": "v2",
                        "name": "Quality",
                        "statement": "Different statement about quality",
                        "priority": 60,
                        "derived_from": ["belief:b1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("value", 0) == 1
        assert result["skipped"].get("value", 0) == 1

    def test_value_import_when_no_duplicate(self, tmp_path, kernle_instance):
        """Import value when no duplicate exists."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "unique_values.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "observe",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "beliefs": [
                    {
                        "id": "b1",
                        "statement": "Values are important",
                        "confidence": 0.9,
                        "derived_from": ["episode:e1"],
                    }
                ],
                "values": [
                    {
                        "id": "v1",
                        "name": "Unique Value",
                        "description": "Fallback description",
                        "priority": 60,
                        "derived_from": ["belief:b1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("value", 0) == 1
        assert result["skipped"].get("value", 0) == 0


class TestImportJsonItemDuplicateGoal:
    """Test goal duplicate checking via import_to with duplicate items in payload."""

    def test_goal_skip_duplicate_by_title(self, tmp_path, kernle_instance):
        """Skip goal import when duplicate found by title within same payload."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "dup_goals_title.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "observe",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "goals": [
                    {
                        "id": "g1",
                        "title": "Ship v1.0",
                        "description": "Release first version",
                        "derived_from": ["episode:e1"],
                    },
                    {
                        "id": "g2",
                        "title": "Ship v1.0",
                        "description": "Different description for same goal",
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("goal", 0) == 1
        assert result["skipped"].get("goal", 0) == 1

    def test_goal_skip_duplicate_by_description(self, tmp_path, kernle_instance):
        """Skip goal import when duplicate found by description within same payload."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "dup_goals_desc.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "observe",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "goals": [
                    {
                        "id": "g1",
                        "title": "Ship v1.0",
                        "description": "Release first version",
                        "derived_from": ["episode:e1"],
                    },
                    {
                        "id": "g2",
                        "title": "Completely different title",
                        "description": "Release first version",
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("goal", 0) == 1
        assert result["skipped"].get("goal", 0) == 1

    def test_goal_import_when_no_duplicate(self, tmp_path, kernle_instance):
        """Import goal when no duplicate exists."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "unique_goals.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "observe",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "goals": [
                    {
                        "id": "g1",
                        "title": "New unique goal",
                        "description": "Unique description",
                        "priority": "high",
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("goal", 0) == 1
        assert result["skipped"].get("goal", 0) == 0


class TestImportJsonItemDuplicateDrive:
    """Test drive duplicate checking via import_to with duplicate items in payload."""

    def test_drive_skip_duplicate_by_type(self, tmp_path, kernle_instance):
        """Skip drive import when duplicate found by drive_type within same payload."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "dup_drives.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "observe",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "drives": [
                    {
                        "id": "d1",
                        "drive_type": "curiosity",
                        "intensity": 0.8,
                        "derived_from": ["episode:e1"],
                    },
                    {
                        "id": "d2",
                        "drive_type": "curiosity",
                        "intensity": 0.5,
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("drive", 0) == 1
        assert result["skipped"].get("drive", 0) == 1

    def test_drive_import_when_no_duplicate(self, tmp_path, kernle_instance):
        """Import drive when no duplicate exists."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "unique_drives.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "observe",
                        "outcome": "learned",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "drives": [
                    {
                        "id": "d1",
                        "drive_type": "growth",
                        "intensity": 0.7,
                        "focus_areas": ["area1"],
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("drive", 0) == 1
        assert result["skipped"].get("drive", 0) == 0


class TestImportJsonItemDuplicateRelationship:
    """Test relationship duplicate checking via import_to with duplicate items in payload."""

    def test_relationship_skip_duplicate_by_entity_name(self, tmp_path, kernle_instance):
        """Skip relationship import when duplicate found by entity_name within same payload."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "dup_relationships.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "meet Alice",
                        "outcome": "good collaboration",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "relationships": [
                    {
                        "id": "rel1",
                        "entity_name": "Alice",
                        "entity_type": "person",
                        "relationship_type": "collaborator",
                        "sentiment": 0.5,
                        "derived_from": ["episode:e1"],
                    },
                    {
                        "id": "rel2",
                        "entity_name": "Alice",
                        "entity_type": "person",
                        "relationship_type": "friend",
                        "sentiment": 0.8,
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("relationship", 0) == 1
        assert result["skipped"].get("relationship", 0) == 1

    def test_relationship_import_when_no_duplicate(self, tmp_path, kernle_instance):
        """Import relationship when no duplicate exists."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "unique_relationships.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw observation"}],
                "episodes": [
                    {
                        "id": "e1",
                        "objective": "observe systems",
                        "outcome": "gathered data",
                        "derived_from": ["raw:r1"],
                    }
                ],
                "relationships": [
                    {
                        "id": "rel1",
                        "entity_name": "UniqueEntity",
                        "entity_type": "system",
                        "relationship_type": "monitors",
                        "sentiment": 0.3,
                        "notes": "System monitoring",
                        "derived_from": ["episode:e1"],
                    },
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("relationship", 0) == 1
        assert result["skipped"].get("relationship", 0) == 0


class TestImportJsonItemDuplicateRaw:
    """Test raw duplicate checking via import_to with duplicate items in payload."""

    def test_raw_skip_duplicate_by_content(self, tmp_path, kernle_instance):
        """Skip raw import when duplicate found by content within same payload."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "dup_raws.json",
            {
                "raw_entries": [
                    {"id": "r1", "content": "Duplicate raw content", "source": "test"},
                    {"id": "r2", "content": "Duplicate raw content", "source": "test"},
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("raw", 0) == 1
        assert result["skipped"].get("raw", 0) == 1

    def test_raw_import_when_no_duplicate(self, tmp_path, kernle_instance):
        """Import raw when no duplicate exists."""
        k, storage = kernle_instance
        path = _write_json(
            tmp_path,
            "unique_raws.json",
            {
                "raw_entries": [
                    {"id": "r1", "content": "Unique raw content", "source": "json-import"},
                ],
            },
        )
        importer = JsonImporter(path)
        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("raw", 0) == 1
        assert result["skipped"].get("raw", 0) == 0


class TestImportJsonItemUnknownType:
    """Test unknown type handling via import_to."""

    def test_unknown_type_is_skipped(self, tmp_path, kernle_instance):
        """Unknown memory type is skipped during import (not imported, not errored).

        The parser only recognizes known keys, so an unknown type must be injected
        after parsing. We verify the importer gracefully skips unrecognized types.
        """
        k, storage = kernle_instance
        # Create a minimal valid JSON with a raw entry
        path = _write_json(
            tmp_path,
            "unknown_type.json",
            {
                "raw_entries": [{"id": "r1", "content": "raw data"}],
            },
        )
        importer = JsonImporter(path)
        importer.parse()

        # Inject an unknown type item after parsing to test the fallback path
        from kernle.importers.json_importer import JsonImportItem

        importer.items.append(
            JsonImportItem(type="unknown_type", data={"id": "u1", "content": "something"})
        )

        result = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result["imported"].get("raw", 0) == 1
        # Unknown type should be skipped (not in imported counts)
        assert "unknown_type" not in result["imported"]


class TestImportToErrorHandling:
    """Test error handling in import_to() — covers lines 88-89."""

    def test_import_error_captured_in_errors_list(self, tmp_path, kernle_instance):
        """Errors during import are captured, not raised."""
        k, storage = kernle_instance

        # Build a JSON file with valid provenance: raw -> episode
        path = _write_json(
            tmp_path,
            "errors.json",
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
            },
        )

        importer = JsonImporter(path)
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
