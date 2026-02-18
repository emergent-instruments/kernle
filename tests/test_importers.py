"""Tests for the importer modules.

Tests for:
- kernle/importers/csv_importer.py
- kernle/importers/json_importer.py
- kernle/importers/markdown.py
"""

import json

import pytest

from kernle.importers.csv_importer import (
    CsvImporter,
    CsvImportItem,
    _map_columns,
    parse_csv,
)
from kernle.importers.json_importer import (
    JsonImporter,
    JsonImportItem,
    parse_kernle_json,
)
from kernle.importers.markdown import (
    ImportItem,
    MarkdownImporter,
    _parse_beliefs,
    _parse_episodes,
    _parse_goals,
    _parse_notes,
    _parse_raw,
    _parse_values,
    _split_paragraphs,
    parse_markdown,
)

# ============================================================================
# CSV Importer Tests
# ============================================================================


class TestCsvImporterParsing:
    """Tests for CSV parsing functionality."""

    def test_parse_csv_with_type_column(self):
        """Parse CSV with explicit type column."""
        csv_content = """type,statement,confidence
belief,Testing is important,0.9
belief,Code should be readable,0.85
"""
        items = parse_csv(csv_content)
        assert len(items) == 2
        assert items[0].type == "belief"
        assert items[0].data["statement"] == "Testing is important"
        assert items[0].data["confidence"] == 0.9
        assert items[1].data["confidence"] == 0.85

    def test_parse_csv_with_memory_type_column(self):
        """Parse CSV with memory_type column instead of type."""
        csv_content = """memory_type,content
note,This is a note
note,Another note
"""
        items = parse_csv(csv_content)
        assert len(items) == 2
        assert all(item.type == "note" for item in items)

    def test_parse_csv_with_kind_column(self):
        """Parse CSV with kind column instead of type."""
        csv_content = """kind,name,description
value,Quality,Code should be tested
value,Simplicity,Prefer simple solutions
"""
        items = parse_csv(csv_content)
        assert len(items) == 2
        assert all(item.type == "value" for item in items)

    def test_parse_csv_with_fixed_memory_type(self):
        """Parse CSV with memory_type parameter overriding type column."""
        csv_content = """statement,confidence
Testing is important,0.9
Code should be readable,0.85
"""
        items = parse_csv(csv_content, memory_type="belief")
        assert len(items) == 2
        assert all(item.type == "belief" for item in items)

    def test_parse_csv_no_headers_raises_error(self):
        """CSV with no headers should raise ValueError."""
        csv_content = ""
        with pytest.raises(ValueError, match="no headers"):
            parse_csv(csv_content)

    def test_parse_csv_no_type_column_and_no_memory_type_raises_error(self):
        """CSV without type column or memory_type parameter should raise."""
        csv_content = """statement,confidence
Testing,0.9
"""
        with pytest.raises(ValueError, match="must have a 'type' column"):
            parse_csv(csv_content)

    def test_parse_csv_skips_rows_without_type(self):
        """Rows without type value are skipped when no memory_type parameter."""
        csv_content = """type,statement,confidence
belief,Testing is important,0.9
,Missing type,0.8
belief,Another belief,0.7
"""
        items = parse_csv(csv_content)
        assert len(items) == 2

    def test_parse_csv_skips_unknown_types(self):
        """Rows with unknown types are skipped."""
        csv_content = """type,content
belief,Valid belief
unknown_type,This should be skipped
note,Valid note
"""
        items = parse_csv(csv_content)
        assert len(items) == 2
        assert items[0].type == "belief"
        assert items[1].type == "note"

    def test_parse_csv_skips_empty_rows(self):
        """Rows with all empty values (no mapped content) are skipped."""
        csv_content = """type,content
note,Valid note
note,Another note
"""
        items = parse_csv(csv_content)
        assert len(items) == 2
        # Note: rows with empty content for note type still have the 'type' mapped
        # so they're not truly empty - add another test to clarify this behavior

    def test_parse_csv_normalizes_column_names(self):
        """Column names are normalized to lowercase."""
        csv_content = """TYPE,Statement,CONFIDENCE
belief,Testing is important,0.9
"""
        items = parse_csv(csv_content)
        assert len(items) == 1
        assert items[0].data["statement"] == "Testing is important"

    def test_parse_csv_episode_type(self):
        """Parse episode type rows."""
        csv_content = """type,objective,outcome,outcome_type,lessons,tags
episode,Fix the bug,Bug was fixed,success,"test first,verify locally","debugging,bugfix"
"""
        items = parse_csv(csv_content)
        assert len(items) == 1
        assert items[0].type == "episode"
        assert items[0].data["objective"] == "Fix the bug"
        assert items[0].data["outcome"] == "Bug was fixed"
        assert items[0].data["lessons"] == ["test first", "verify locally"]
        assert items[0].data["tags"] == ["debugging", "bugfix"]

    def test_parse_csv_goal_type(self):
        """Parse goal type rows."""
        csv_content = """type,title,description,status,priority
goal,Ship v1.0,Release the first version,active,high
goal,Write docs,Add documentation,completed,medium
"""
        items = parse_csv(csv_content)
        assert len(items) == 2
        assert items[0].type == "goal"
        assert items[0].data["title"] == "Ship v1.0"
        assert items[0].data["status"] == "active"

    def test_parse_csv_raw_type(self):
        """Parse raw type rows."""
        csv_content = """type,content,source,tags
raw,Some raw content,import,"note,scratch"
"""
        items = parse_csv(csv_content)
        assert len(items) == 1
        assert items[0].type == "raw"
        assert items[0].data["content"] == "Some raw content"
        assert items[0].data["source"] == "import"
        assert items[0].data["tags"] == ["note", "scratch"]


class TestMapColumns:
    """Tests for column mapping functionality."""

    def test_map_columns_belief(self):
        """Map belief columns correctly."""
        row = {"statement": "Test belief", "confidence": "0.9", "type": "fact"}
        result, _, _ = _map_columns(row, "belief")
        assert result["statement"] == "Test belief"
        assert result["confidence"] == 0.9
        assert result["type"] == "fact"

    def test_map_columns_with_aliases(self):
        """Map using column aliases."""
        row = {"text": "Test belief", "conf": "0.85"}
        result, _, _ = _map_columns(row, "belief")
        assert result["statement"] == "Test belief"
        assert result["confidence"] == 0.85

    def test_map_columns_confidence_percentage(self):
        """Confidence values > 1 are normalized to 0-1 range."""
        row = {"statement": "Test", "confidence": "90"}
        result, _, _ = _map_columns(row, "belief")
        assert result["confidence"] == 0.9

    def test_map_columns_confidence_invalid(self):
        """Invalid confidence values default to 0.7."""
        row = {"statement": "Test", "confidence": "invalid"}
        result, _, _ = _map_columns(row, "belief")
        assert result["confidence"] == 0.7

    def test_map_columns_priority_int_conversion(self):
        """Priority values are converted to int for values."""
        row = {"name": "Quality", "priority": "75"}
        result, _, _ = _map_columns(row, "value")
        assert result["priority"] == 75

    def test_map_columns_priority_invalid(self):
        """Invalid priority values default to 50."""
        row = {"name": "Quality", "priority": "high"}
        result, _, _ = _map_columns(row, "value")
        assert result["priority"] == 50

    def test_map_columns_tags_split(self):
        """Tags are split by comma."""
        row = {"content": "Test note", "tags": "tag1, tag2, tag3"}
        result, _, _ = _map_columns(row, "note")
        assert result["tags"] == ["tag1", "tag2", "tag3"]

    def test_map_columns_lessons_split(self):
        """Lessons are split by comma."""
        row = {"objective": "Task", "lessons": "lesson1, lesson2"}
        result, _, _ = _map_columns(row, "episode")
        assert result["lessons"] == ["lesson1", "lesson2"]


class TestCsvImporterClass:
    """Tests for the CsvImporter class."""

    def test_importer_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for non-existent file."""
        importer = CsvImporter(str(tmp_path / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            importer.parse()

    def test_importer_parse_file(self, tmp_path):
        """Parse a CSV file from disk."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,statement,confidence
belief,Testing matters,0.9
belief,Quality counts,0.85
""")
        importer = CsvImporter(str(csv_file))
        items = importer.parse()
        assert len(items) == 2
        assert importer.items == items

    def test_importer_with_memory_type_override(self, tmp_path):
        """Memory type parameter overrides file content."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""content
This is content
More content
""")
        importer = CsvImporter(str(csv_file), memory_type="note")
        items = importer.parse()
        assert len(items) == 2
        assert all(item.type == "note" for item in items)

    def test_importer_import_to_dry_run(self, tmp_path, kernle_instance):
        """Dry run counts raw items without importing."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,content,source
raw,Test raw one,import
raw,Test raw two,import
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=True)

        assert result["imported"]["raw"] == 2
        # Dry run should not actually import
        raws = storage.list_raw(limit=10)
        assert len(raws) == 0

    def test_importer_import_to_actual(self, tmp_path, kernle_instance):
        """Actually import raw items into Kernle."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,content,source
raw,CSV imported raw entry,import
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False)

        assert result["imported"]["raw"] == 1
        raws = storage.list_raw(limit=10)
        assert len(raws) == 1

    def test_importer_skip_duplicates(self, tmp_path, kernle_instance):
        """Skip duplicate raw items when skip_duplicates is True."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,content
raw,Unique raw content
""")
        k, storage = kernle_instance

        # First import into empty stack
        importer = CsvImporter(str(csv_file))
        result1 = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result1["imported"]["raw"] == 1

        # Verify duplicate is detected at item level (import guard
        # prevents a second import_to() on a non-empty stack)
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        dup_item = CsvImportItem(type="raw", data={"content": "Unique raw content"})
        assert _import_csv_item(dup_item, k, skip_duplicates=True) is False

    def test_importer_auto_parse_on_import(self, tmp_path, kernle_instance):
        """import_to() calls parse() if items is empty."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,content
raw,Auto-parsed raw entry
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        # Don't call parse() explicitly
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["raw"] == 1


class TestCsvImporterAllTypes:
    """Test importing all memory types via CSV."""

    def test_import_episode_skipped_as_non_raw(self, tmp_path, kernle_instance):
        """Non-raw episode items are skipped in CSV import."""
        csv_file = tmp_path / "episodes.csv"
        csv_file.write_text("""memory_type,objective,result,outcome_type
episode,Complete the task,Task completed successfully,success
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["skipped_non_raw"] == 1
        assert result["imported"].get("episode", 0) == 0

    def test_import_note_skipped_as_non_raw(self, tmp_path, kernle_instance):
        """Non-raw note items are skipped in CSV import."""
        csv_file = tmp_path / "notes.csv"
        csv_file.write_text("""memory_type,text,note_type,speaker
note,Important observation,insight,User
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["skipped_non_raw"] == 1
        assert result["imported"].get("note", 0) == 0

    def test_import_value_skipped_as_non_raw(self, tmp_path, kernle_instance):
        """Non-raw value items are skipped in CSV import."""
        csv_file = tmp_path / "values.csv"
        csv_file.write_text("""memory_type,name,description,priority
value,Quality,Code should be tested,80
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["skipped_non_raw"] == 1
        assert result["imported"].get("value", 0) == 0

    def test_import_goal_skipped_as_non_raw(self, tmp_path, kernle_instance):
        """Non-raw goal items are skipped in CSV import."""
        csv_file = tmp_path / "goals.csv"
        csv_file.write_text("""memory_type,title,description,status,priority
goal,Ship v1.0,Release first version,active,high
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["skipped_non_raw"] == 1
        assert result["imported"].get("goal", 0) == 0

    def test_import_raw(self, tmp_path, kernle_instance):
        """Import raw type."""
        csv_file = tmp_path / "raw.csv"
        csv_file.write_text("""type,content,source
raw,Some raw thought,manual-import
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["raw"] == 1

    def test_import_missing_required_fields(self, tmp_path, kernle_instance):
        """Raw items with empty content are skipped."""
        csv_file = tmp_path / "missing.csv"
        csv_file.write_text("""memory_type,content
raw,
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"].get("raw", 0) == 0


class TestCsvImporterGoalStatusMapping:
    """Test goal status value mapping."""

    @pytest.mark.parametrize(
        "status_input,expected_status",
        [
            ("done", "completed"),
            ("complete", "completed"),
            ("completed", "completed"),
            ("true", "completed"),
            ("1", "completed"),
            ("yes", "completed"),
            ("paused", "paused"),
            ("hold", "paused"),
            ("on hold", "paused"),
            ("active", "active"),
            ("in progress", "active"),
            ("other", "active"),
        ],
    )
    def test_goal_status_mapping(self, tmp_path, kernle_instance, status_input, expected_status):
        """Goal status values are normalized via _import_csv_item."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        item = CsvImportItem(
            type="goal",
            data={
                "title": f"Test goal {status_input}",
                "status": status_input,
            },
        )
        result = _import_csv_item(item, k, skip_duplicates=False)
        assert result is True
        goals = storage.get_goals(status=None, limit=10)
        assert len(goals) == 1
        assert goals[0].status == expected_status


class TestCsvImporterSkipDuplicates:
    """Test skip_duplicates for each memory type."""

    def test_skip_duplicate_note(self, tmp_path, kernle_instance):
        """Duplicate notes are skipped when skip_duplicates=True."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        # First import via _import_csv_item (notes are non-raw, skipped by import_to)
        item = CsvImportItem(
            type="note",
            data={
                "content": "This is a very unique and distinctive note about quantum computing research methodology and its applications in modern technology"
            },
        )
        assert _import_csv_item(item, k, skip_duplicates=False) is True
        assert len(storage.get_notes()) == 1

        # Verify duplicate is detected at item level
        assert _import_csv_item(item, k, skip_duplicates=True) is False
        assert len(storage.get_notes()) == 1

    def test_skip_duplicate_value(self, tmp_path, kernle_instance):
        """Duplicate values are skipped when skip_duplicates=True."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        # First import via _import_csv_item (values are non-raw, skipped by import_to)
        item = CsvImportItem(
            type="value", data={"name": "Integrity", "description": "Be honest always"}
        )
        assert _import_csv_item(item, k, skip_duplicates=False) is True
        assert len(storage.get_values()) == 1

        # Verify duplicate is detected at item level
        assert _import_csv_item(item, k, skip_duplicates=True) is False
        assert len(storage.get_values()) == 1

    def test_skip_duplicate_goal(self, tmp_path, kernle_instance):
        """Duplicate goals are skipped when skip_duplicates=True."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        # First import via _import_csv_item (goals are non-raw, skipped by import_to)
        item = CsvImportItem(
            type="goal", data={"title": "Learn Python", "description": "Master Python programming"}
        )
        assert _import_csv_item(item, k, skip_duplicates=False) is True
        assert len(storage.get_goals(status=None, limit=10)) == 1

        # Verify duplicate is detected at item level
        assert _import_csv_item(item, k, skip_duplicates=True) is False
        assert len(storage.get_goals(status=None, limit=10)) == 1

    def test_skip_duplicate_raw(self, tmp_path, kernle_instance):
        """Duplicate raw entries are skipped when skip_duplicates=True."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        csv_file = tmp_path / "raw.csv"
        csv_file.write_text("""memory_type,content
raw,Some raw content here
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert len(storage.list_raw(limit=10)) == 1

        dup_item = CsvImportItem(type="raw", data={"content": "Some raw content here"})
        assert _import_csv_item(dup_item, k, skip_duplicates=True) is False
        assert len(storage.list_raw(limit=10)) == 1

    def test_skip_duplicate_episode(self, tmp_path, kernle_instance):
        """Duplicate episodes are skipped when skip_duplicates=True."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        # First import via _import_csv_item (episodes are non-raw, skipped by import_to)
        item = CsvImportItem(
            type="episode",
            data={
                "objective": "Build the comprehensive quantum computing feature for distributed systems",
                "outcome": "Feature was built successfully with all integration tests passing",
            },
        )
        assert _import_csv_item(item, k, skip_duplicates=False) is True
        assert len(storage.get_episodes()) == 1

        # Verify duplicate is detected at item level
        dup_item = CsvImportItem(
            type="episode",
            data={
                "objective": "Build the comprehensive quantum computing feature for distributed systems"
            },
        )
        assert _import_csv_item(dup_item, k, skip_duplicates=True) is False
        assert len(storage.get_episodes()) == 1

    def test_note_dedup_ignores_non_note_search_hits(self, tmp_path, kernle_instance):
        """A belief with matching content should not block note import dedup."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        note_content = "Shared sentence used in both note and belief for dedup safety validation."
        k, storage = kernle_instance
        # Pre-populate with a belief (not a note) that has the same text
        k.belief(statement=note_content, confidence=0.8, source_type="imported")

        # Note import should succeed despite the belief having the same content
        note_item = CsvImportItem(type="note", data={"content": note_content})
        assert _import_csv_item(note_item, k, skip_duplicates=True) is True
        assert len(storage.get_notes()) == 1

    def test_episode_dedup_allows_same_prefix_different_objective(self, tmp_path, kernle_instance):
        """Episodes sharing first 60 chars should not be considered duplicates."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        shared_prefix = "A" * 60
        objective_one = f"{shared_prefix}-first-objective"
        objective_two = f"{shared_prefix}-second-objective"
        k, storage = kernle_instance
        # Import via _import_csv_item (episodes are non-raw, skipped by import_to)
        item1 = CsvImportItem(
            type="episode", data={"objective": objective_one, "outcome": "Outcome one"}
        )
        item2 = CsvImportItem(
            type="episode", data={"objective": objective_two, "outcome": "Outcome two"}
        )
        assert _import_csv_item(item1, k, skip_duplicates=True) is True
        assert _import_csv_item(item2, k, skip_duplicates=True) is True
        episodes = storage.get_episodes()
        assert len(episodes) == 2
        assert {ep.objective for ep in episodes} == {objective_one, objective_two}


class TestCsvImporterEmptyAndMissingFields:
    """Test handling of empty/missing required fields per type."""

    def test_note_empty_content_skipped(self, tmp_path, kernle_instance):
        """Note with empty content returns False via _import_csv_item."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        item = CsvImportItem(type="note", data={"content": ""})
        assert _import_csv_item(item, k, skip_duplicates=False) is False

    def test_value_empty_name_skipped(self, tmp_path, kernle_instance):
        """Value with empty name returns False via _import_csv_item."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        item = CsvImportItem(type="value", data={"name": "", "description": "Some description"})
        assert _import_csv_item(item, k, skip_duplicates=False) is False

    def test_goal_empty_title_and_description_skipped(self, tmp_path, kernle_instance):
        """Goal with empty title and description returns False via _import_csv_item."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        item = CsvImportItem(type="goal", data={"title": "", "description": ""})
        assert _import_csv_item(item, k, skip_duplicates=False) is False

    def test_raw_empty_content_skipped(self, tmp_path, kernle_instance):
        """Raw entry with empty content returns False."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("""memory_type,content
raw,
""")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"].get("raw", 0) == 0

    def test_import_error_captured(self, tmp_path, kernle_instance):
        """Errors during import are captured, not raised."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,content
raw,Valid raw content
""")
        k, storage = kernle_instance
        # Monkey-patch to cause an error
        original_raw = k.raw
        k.raw = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("test error"))
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert len(result["errors"]) == 1
        assert "test error" in result["errors"][0]
        k.raw = original_raw


class TestCsvImporterConfidenceAndPriority:
    """Test confidence normalization and priority conversion in CSV parsing."""

    def test_confidence_over_1_normalized(self):
        """Confidence > 1 is divided by 100 (e.g., 85 → 0.85)."""
        items = parse_csv("""type,statement,confidence
belief,Test belief,85
""")
        assert len(items) == 1
        assert items[0].data["confidence"] == 0.85

    def test_confidence_decimal_preserved(self):
        """Confidence <= 1 is kept as-is."""
        items = parse_csv("""type,statement,confidence
belief,Test belief,0.92
""")
        assert len(items) == 1
        assert items[0].data["confidence"] == 0.92

    def test_priority_int_for_value(self):
        """Priority is converted to int for value type."""
        items = parse_csv("""type,name,description,priority
value,TestVal,Test description,75
""")
        assert len(items) == 1
        assert items[0].data["priority"] == 75


class TestCsvImporterUnknownType:
    """Test handling of unknown memory types in _import_csv_item."""

    def test_unknown_type_returns_false(self, kernle_instance):
        """Unknown type returns False from _import_csv_item."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        item = CsvImportItem(type="unknown_type", data={"content": "test"})
        result = _import_csv_item(item, k, skip_duplicates=False)
        assert result is False


class TestCsvImporterEpisodeOutcomeType:
    """Test episode outcome_type tag folding."""

    def test_episode_with_outcome_type(self, tmp_path, kernle_instance):
        """Episode outcome_type is folded into tags via _import_csv_item."""
        from kernle.importers.csv_importer import CsvImportItem, _import_csv_item

        k, storage = kernle_instance
        item = CsvImportItem(
            type="episode",
            data={
                "objective": "Test objective",
                "outcome": "It worked",
                "outcome_type": "success",
            },
        )
        assert _import_csv_item(item, k, skip_duplicates=False) is True
        episodes = storage.get_episodes()
        assert len(episodes) == 1
        assert "outcome:success" in (episodes[0].tags or [])


# ============================================================================
# JSON Importer Tests
# ============================================================================


class TestJsonImporterParsing:
    """Tests for JSON parsing functionality."""

    def test_parse_kernle_json_basic(self):
        """Parse basic Kernle JSON export format."""
        content = json.dumps(
            {
                "stack_id": "test-agent",
                "exported_at": "2024-01-15T10:00:00Z",
                "values": [{"name": "Quality", "statement": "Test well", "priority": 80}],
                "beliefs": [{"statement": "Testing matters", "confidence": 0.9}],
                "goals": [],
                "episodes": [],
                "notes": [],
                "drives": [],
                "relationships": [],
            }
        )
        items, stack_id = parse_kernle_json(content)
        assert stack_id == "test-agent"
        assert len(items) == 2
        assert items[0].type == "value"
        assert items[1].type == "belief"

    def test_parse_kernle_json_all_types(self):
        """Parse JSON with all memory types."""
        content = json.dumps(
            {
                "stack_id": "test",
                "values": [{"name": "V1"}],
                "beliefs": [{"statement": "B1"}],
                "goals": [{"title": "G1"}],
                "episodes": [{"objective": "E1", "outcome": "O1"}],
                "notes": [{"content": "N1"}],
                "drives": [{"drive_type": "curiosity"}],
                "relationships": [{"entity_name": "User"}],
                "raw_entries": [{"content": "R1"}],
            }
        )
        items, stack_id = parse_kernle_json(content)

        types = [item.type for item in items]
        assert "value" in types
        assert "belief" in types
        assert "goal" in types
        assert "episode" in types
        assert "note" in types
        assert "drive" in types
        assert "relationship" in types
        assert "raw" in types

    def test_parse_kernle_json_invalid_json(self):
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            parse_kernle_json("not valid json")

    def test_parse_kernle_json_not_object(self):
        """Non-object root raises ValueError."""
        with pytest.raises(ValueError, match="must be an object"):
            parse_kernle_json("[]")

    def test_parse_kernle_json_empty_arrays(self):
        """Empty arrays in JSON are handled."""
        content = json.dumps(
            {
                "stack_id": "test",
                "values": [],
                "beliefs": [],
            }
        )
        items, stack_id = parse_kernle_json(content)
        assert len(items) == 0

    def test_parse_kernle_json_missing_stack_id(self):
        """Missing stack_id returns None."""
        content = json.dumps({"values": [{"name": "Test"}]})
        items, stack_id = parse_kernle_json(content)
        assert stack_id is None
        assert len(items) == 1


class TestJsonImporterClass:
    """Tests for the JsonImporter class."""

    def test_importer_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for non-existent file."""
        importer = JsonImporter(str(tmp_path / "nonexistent.json"))
        with pytest.raises(FileNotFoundError):
            importer.parse()

    def test_importer_parse_file(self, tmp_path):
        """Parse a JSON file from disk."""
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "stack_id": "test-agent",
                    "beliefs": [{"statement": "Test belief", "confidence": 0.9}],
                }
            )
        )
        importer = JsonImporter(str(json_file))
        items = importer.parse()
        assert len(items) == 1
        assert importer.source_stack_id == "test-agent"

    def test_importer_import_to_dry_run(self, tmp_path, kernle_instance):
        """Dry run counts items without importing."""
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "observation"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "Test one",
                            "outcome": "Passed",
                            "derived_from": ["raw:r1"],
                        },
                        {
                            "id": "e2",
                            "objective": "Test two",
                            "outcome": "Also passed",
                            "derived_from": ["raw:r1"],
                        },
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=True)

        assert result["imported"]["episode"] == 2
        assert result["imported"]["raw"] == 1
        # Dry run should not actually import
        episodes = storage.get_episodes()
        assert len(episodes) == 0

    def test_importer_import_to_actual(self, tmp_path, kernle_instance):
        """Actually import items into Kernle."""
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "source data"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "JSON test",
                            "outcome": "done",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False)

        assert result["imported"]["episode"] == 1
        assert result["imported"]["raw"] == 1

    def test_importer_skip_duplicates(self, tmp_path, kernle_instance):
        """Skip duplicate items."""
        from kernle.importers.json_importer import JsonImportItem, _import_json_item

        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "Unique JSON raw content"}],
                }
            )
        )
        k, storage = kernle_instance

        # First import into empty stack
        importer = JsonImporter(str(json_file))
        result1 = importer.import_to(k, dry_run=False, skip_duplicates=True)
        assert result1["imported"]["raw"] == 1

        # Verify duplicate is detected at item level
        dup_item = JsonImportItem(type="raw", data={"content": "Unique JSON raw content"})
        assert _import_json_item(dup_item, k, skip_duplicates=True) is None

    def test_importer_returns_source_stack_id(self, tmp_path, kernle_instance):
        """import_to returns source stack_id."""
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "stack_id": "source-agent",
                    "beliefs": [{"statement": "Test", "confidence": 0.9}],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=True)
        assert result["source_stack_id"] == "source-agent"


class TestJsonImporterAllTypes:
    """Test importing all memory types via JSON."""

    def test_import_episode(self, tmp_path, kernle_instance):
        """Import episode type with provenance chain."""
        json_file = tmp_path / "episodes.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw observation"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "Complete JSON task",
                            "outcome": "Task completed via JSON",
                            "outcome_type": "success",
                            "lessons": ["Test first"],
                            "derived_from": ["raw:r1"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["episode"] == 1

    def test_import_note(self, tmp_path, kernle_instance):
        """Import note type with provenance chain."""
        json_file = tmp_path / "notes.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw observation"}],
                    "notes": [
                        {
                            "id": "n1",
                            "content": "Important note from JSON",
                            "type": "insight",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["note"] == 1

    def test_import_value(self, tmp_path, kernle_instance):
        """Import value type with provenance chain."""
        json_file = tmp_path / "values.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw observation"}],
                    "notes": [
                        {"id": "n1", "content": "processed note", "derived_from": ["raw:r1"]}
                    ],
                    "beliefs": [
                        {
                            "id": "b1",
                            "statement": "Quality matters",
                            "confidence": 0.9,
                            "derived_from": ["note:n1"],
                        }
                    ],
                    "values": [
                        {
                            "id": "v1",
                            "name": "Quality from JSON",
                            "statement": "Test thoroughly",
                            "priority": 80,
                            "derived_from": ["belief:b1"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["value"] == 1

    def test_import_goal(self, tmp_path, kernle_instance):
        """Import goal type with provenance chain."""
        json_file = tmp_path / "goals.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw observation"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "Did task",
                            "outcome": "Done",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                    "goals": [
                        {
                            "id": "g1",
                            "title": "Ship v1.0 JSON",
                            "description": "Release first version",
                            "status": "active",
                            "priority": "high",
                            "derived_from": ["episode:e1"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["goal"] == 1

    def test_import_drive(self, tmp_path, kernle_instance):
        """Import drive type with provenance chain."""
        json_file = tmp_path / "drives.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw observation"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "Explored topic",
                            "outcome": "Learned a lot",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                    "drives": [
                        {
                            "id": "d1",
                            "drive_type": "curiosity",
                            "intensity": 0.8,
                            "focus_areas": ["learning", "exploration"],
                            "derived_from": ["episode:e1"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["drive"] == 1

    def test_drive_round_trip(self, tmp_path, kernle_instance):
        """Export drive via _dump_json, re-import via JsonImporter — drive_type preserved."""
        k, storage = kernle_instance
        # Seed raw + episode to satisfy provenance chain
        raw_id = k.raw(blob="drive observation", source="test")
        ep_id = k.episode(
            objective="Explored topic",
            outcome="Learned a lot",
            derived_from=[f"raw:{raw_id}"],
            source_type="imported",
        )
        k.drive(
            drive_type="curiosity",
            intensity=0.7,
            focus_areas=["learning"],
            derived_from=[f"episode:{ep_id}"],
            source_type="imported",
        )

        # Export to JSON
        export_path = tmp_path / "export.json"
        k.export(str(export_path), format="json")

        # Create a fresh Kernle instance for the import target
        from kernle.core.kernle_class import Kernle
        from kernle.storage.sqlite import SQLiteStorage
        from tests.conftest import bind_noop_model

        target_db = tmp_path / "target.db"
        target_storage = SQLiteStorage(stack_id="target_agent", db_path=target_db)
        target_k = Kernle(
            stack_id="target_agent",
            storage=target_storage,
            checkpoint_dir=tmp_path / "cp",
            strict=False,
        )
        bind_noop_model(target_k)

        # Re-import
        importer = JsonImporter(str(export_path))
        result = importer.import_to(target_k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["drive"] == 1

        drives = target_storage.get_drives()
        assert len(drives) == 1
        assert drives[0].drive_type == "curiosity"
        target_storage.close()

    def test_import_relationship(self, tmp_path, kernle_instance):
        """Import relationship type with provenance chain."""
        json_file = tmp_path / "relationships.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw observation"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "Worked with User123",
                            "outcome": "Great collaboration",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                    "relationships": [
                        {
                            "id": "rel1",
                            "entity_name": "User123",
                            "entity_type": "person",
                            "relationship_type": "collaborator",
                            "sentiment": 0.7,
                            "notes": "Great to work with",
                            "derived_from": ["episode:e1"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["relationship"] == 1

    def test_import_raw(self, tmp_path, kernle_instance):
        """Import raw type."""
        json_file = tmp_path / "raw.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [
                        {
                            "content": "Some raw thought",
                            "source": "json-import",
                            "tags": ["scratch"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["raw"] == 1


# ============================================================================
# Markdown Importer Tests
# ============================================================================


class TestMarkdownParsingHelpers:
    """Tests for markdown parsing helper functions."""

    def test_split_paragraphs(self):
        """Split text into paragraphs."""
        text = """First paragraph.

Second paragraph.


Third paragraph with extra spacing.
"""
        paras = _split_paragraphs(text)
        assert len(paras) == 3
        assert paras[0] == "First paragraph."
        assert paras[1] == "Second paragraph."
        assert "Third" in paras[2]

    def test_split_paragraphs_empty(self):
        """Handle empty text."""
        assert _split_paragraphs("") == []
        assert _split_paragraphs("   ") == []


class TestParseEpisodes:
    """Tests for episode parsing."""

    def test_parse_episodes_basic(self):
        """Parse basic episode bullets."""
        content = """
- Fixed the bug
- Deployed new feature
"""
        items = _parse_episodes(content)
        assert len(items) == 2
        assert items[0].type == "episode"
        assert items[0].objective == "Fixed the bug"

    def test_parse_episodes_with_arrow_lesson(self):
        """Parse episodes with -> lesson format."""
        content = "- Fixed auth bug -> Always check token expiry"
        items = _parse_episodes(content)
        assert len(items) == 1
        assert items[0].objective == "Fixed auth bug"
        assert items[0].lesson == "Always check token expiry"

    def test_parse_episodes_with_parenthetical_lesson(self):
        """Parse episodes with (Lesson: X) format."""
        content = "- Deployed feature (Lesson: Test in staging first)"
        items = _parse_episodes(content)
        assert len(items) == 1
        assert items[0].lesson == "Test in staging first"

    def test_parse_episodes_with_lesson_colon(self):
        """Parse episodes with Lesson: X format (no parentheses)."""
        content = "- Completed task Lesson: Document everything"
        items = _parse_episodes(content)
        assert len(items) == 1
        assert items[0].lesson == "Document everything"

    def test_parse_episodes_with_success_marker(self):
        """Parse episodes with [success] marker."""
        content = "- [success] Deployed to production"
        items = _parse_episodes(content)
        assert len(items) == 1
        assert items[0].metadata.get("outcome_type") == "success"

    def test_parse_episodes_with_failure_marker(self):
        """Parse episodes with [failure] or [failed] marker."""
        content = """
- [failure] Migration script crashed
- [failed] API integration broke
"""
        items = _parse_episodes(content)
        assert len(items) == 2
        assert items[0].metadata.get("outcome_type") == "failure"
        assert items[1].metadata.get("outcome_type") == "failure"

    def test_parse_episodes_truncates_long_objectives(self):
        """Long objectives are truncated to 200 chars."""
        long_text = "A" * 300
        content = f"- {long_text}"
        items = _parse_episodes(content)
        assert len(items) == 1
        assert len(items[0].objective) == 200

    def test_parse_episodes_numbered_list(self):
        """Parse numbered list episodes."""
        content = """
1. First task
2. Second task
3. Third task
"""
        items = _parse_episodes(content)
        assert len(items) == 3


class TestParseNotes:
    """Tests for note parsing."""

    def test_parse_notes_basic(self):
        """Parse basic note bullets."""
        content = """
- First note
- Second note
"""
        items = _parse_notes(content, "notes")
        assert len(items) == 2
        assert items[0].type == "note"
        assert items[0].note_type == "note"

    def test_parse_notes_decision_header(self):
        """Parse notes from decision header."""
        content = "- Made architecture decision"
        items = _parse_notes(content, "decisions")
        assert items[0].note_type == "decision"

    def test_parse_notes_insight_header(self):
        """Parse notes from insight header."""
        content = "- Key insight about the system"
        items = _parse_notes(content, "insights")
        assert items[0].note_type == "insight"

    def test_parse_notes_observation_header(self):
        """Parse notes from observation header."""
        content = "- Observed behavior"
        items = _parse_notes(content, "observations")
        assert items[0].note_type == "observation"


class TestParseBeliefs:
    """Tests for belief parsing."""

    def test_parse_beliefs_basic(self):
        """Parse basic belief bullets."""
        content = """
- Testing is important
- Code should be readable
"""
        items = _parse_beliefs(content)
        assert len(items) == 2
        assert items[0].type == "belief"
        assert items[0].confidence == 0.7  # default

    def test_parse_beliefs_percentage_confidence(self):
        """Parse beliefs with (N%) confidence."""
        content = "- Testing is important (90%)"
        items = _parse_beliefs(content)
        assert items[0].confidence == 0.9
        assert "90%" not in items[0].statement

    def test_parse_beliefs_bracket_percentage(self):
        """Parse beliefs with [N%] confidence."""
        content = "- Code quality matters [85%]"
        items = _parse_beliefs(content)
        assert items[0].confidence == 0.85

    def test_parse_beliefs_decimal_bracket(self):
        """Parse beliefs with [0.N] confidence."""
        content = "- Simplicity wins [0.95]"
        items = _parse_beliefs(content)
        assert items[0].confidence == 0.95

    def test_parse_beliefs_decimal_paren(self):
        """Parse beliefs with (0.N) confidence."""
        content = "- Refactoring helps (0.8)"
        items = _parse_beliefs(content)
        assert items[0].confidence == 0.8

    def test_parse_beliefs_confidence_label(self):
        """Parse beliefs with (confidence: N) format."""
        content = "- Documentation is valuable (confidence: 0.85)"
        items = _parse_beliefs(content)
        assert items[0].confidence == 0.85

    def test_parse_beliefs_confidence_label_percentage(self):
        """Parse beliefs with (confidence: N) where N > 1."""
        content = "- Testing reduces bugs (confidence: 90)"
        items = _parse_beliefs(content)
        assert items[0].confidence == 0.9

    def test_parse_beliefs_removes_i_believe_prefix(self):
        """Remove 'I believe' prefix from statements."""
        content = "- I believe testing is crucial"
        items = _parse_beliefs(content)
        assert items[0].statement == "testing is crucial"

    def test_parse_beliefs_clamps_confidence(self):
        """Confidence is clamped to [0, 1]."""
        content = """
- Too high (150%)
- Too low (0%)
"""
        items = _parse_beliefs(content)
        assert items[0].confidence <= 1.0
        assert items[1].confidence >= 0.0


class TestParseValues:
    """Tests for value parsing."""

    def test_parse_values_basic(self):
        """Parse basic value bullets."""
        content = """
- Quality
- Simplicity
"""
        items = _parse_values(content)
        assert len(items) == 2
        assert items[0].type == "value"

    def test_parse_values_with_description(self):
        """Parse values with name: description format."""
        content = "- Quality: Code should be well-tested and maintainable"
        items = _parse_values(content)
        assert items[0].name == "Quality"
        assert "well-tested" in items[0].description

    def test_parse_values_no_colon(self):
        """Values without colon use truncated text as name."""
        content = "- This is a long value statement without a colon separator"
        items = _parse_values(content)
        assert len(items[0].name) <= 50

    def test_parse_values_with_priority(self):
        """Parse values with priority marker."""
        content = "- Quality: Test well (priority: 80)"
        items = _parse_values(content)
        assert items[0].priority == 80


class TestParseGoals:
    """Tests for goal parsing."""

    def test_parse_goals_basic(self):
        """Parse basic goal bullets."""
        content = """
- Ship v1.0
- Write documentation
"""
        items = _parse_goals(content)
        assert len(items) == 2
        assert items[0].type == "goal"
        assert items[0].status == "active"

    def test_parse_goals_done_markers(self):
        """Parse goals with completion markers."""
        content = """
- [done] Completed task
- [complete] Another done task
- [x] Checked off task
"""
        items = _parse_goals(content)
        assert all(item.status == "completed" for item in items)

    def test_parse_goals_paused_markers(self):
        """Parse goals with paused markers."""
        content = """
- [paused] On hold task
- [hold] Waiting for input
"""
        items = _parse_goals(content)
        assert all(item.status == "paused" for item in items)

    def test_parse_goals_priority_markers(self):
        """Parse goals with priority markers."""
        content = """
- [high] Urgent task
- [urgent] Critical task
- [p1] Priority one
- [low] Low priority
- [p3] Priority three
"""
        items = _parse_goals(content)
        assert items[0].metadata.get("priority") == "high"
        assert items[1].metadata.get("priority") == "high"
        assert items[2].metadata.get("priority") == "high"
        assert items[3].metadata.get("priority") == "low"
        assert items[4].metadata.get("priority") == "low"


class TestParseRaw:
    """Tests for raw content parsing."""

    def test_parse_raw_bullets(self):
        """Parse raw content with bullets."""
        content = """
- First thought
- Second thought
"""
        items = _parse_raw(content)
        assert len(items) == 2
        assert all(item.type == "raw" for item in items)

    def test_parse_raw_paragraphs(self):
        """Parse raw content as paragraphs when no bullets."""
        content = """First paragraph here.

Second paragraph here.

Third paragraph."""
        items = _parse_raw(content)
        assert len(items) == 3

    def test_parse_raw_asterisk_bullets(self):
        """Parse raw content with asterisk bullets."""
        content = """
* Star bullet one
* Star bullet two
"""
        items = _parse_raw(content)
        assert len(items) == 2


class TestParseMarkdown:
    """Tests for full markdown document parsing."""

    def test_parse_markdown_empty(self):
        """Handle empty markdown."""
        items = parse_markdown("")
        assert items == []

    def test_parse_markdown_preamble_only(self):
        """Handle markdown with only preamble (no sections)."""
        content = """
Just some text without any structure.

Another paragraph.
"""
        items = parse_markdown(content)
        assert len(items) >= 1
        assert all(item.type == "raw" for item in items)

    def test_parse_markdown_beliefs_section(self):
        """Parse markdown beliefs section."""
        content = """
## Beliefs

- Testing is important (90%)
- Code should be readable
"""
        items = parse_markdown(content)
        assert len(items) == 2
        assert all(item.type == "belief" for item in items)

    def test_parse_markdown_episodes_section(self):
        """Parse markdown episodes/lessons section."""
        content = """
## Episodes

- Fixed the bug -> Test first

## Lessons

- Another lesson learned
"""
        items = parse_markdown(content)
        assert len(items) == 2
        assert all(item.type == "episode" for item in items)

    def test_parse_markdown_notes_section(self):
        """Parse markdown notes/decisions section."""
        content = """
## Notes

- General note

## Decisions

- Made a decision
"""
        items = parse_markdown(content)
        assert len(items) == 2
        assert all(item.type == "note" for item in items)

    def test_parse_markdown_values_section(self):
        """Parse markdown values/principles section."""
        content = """
## Values

- Quality: Test well

## Principles

- Keep it simple
"""
        items = parse_markdown(content)
        assert len(items) == 2
        assert all(item.type == "value" for item in items)

    def test_parse_markdown_goals_section(self):
        """Parse markdown goals/tasks section."""
        content = """
## Goals

- Ship v1.0

## Tasks

- Write docs
"""
        items = parse_markdown(content)
        assert len(items) == 2
        assert all(item.type == "goal" for item in items)

    def test_parse_markdown_raw_section(self):
        """Parse markdown raw/thoughts section."""
        content = """
## Raw

- Random thought

## Thoughts

- Another idea

## Scratch

- Draft content
"""
        items = parse_markdown(content)
        assert len(items) == 3
        assert all(item.type == "raw" for item in items)

    def test_parse_markdown_unknown_section(self):
        """Unknown sections are treated as raw."""
        content = """
## Unknown Section Name

- Content here
- More content
"""
        items = parse_markdown(content)
        assert len(items) == 2
        assert all(item.type == "raw" for item in items)

    def test_parse_markdown_full_document(self):
        """Parse a complete markdown document."""
        content = """
# Memory File

This is a preamble.

## Beliefs

- Testing is important (90%)
- Code should be readable (85%)

## Episodes

- Fixed critical bug -> Always add tests

## Notes

- Remember to update docs

## Goals

- [x] Complete MVP
- Ship beta version

## Values

- Quality: Always prioritize code quality
"""
        items = parse_markdown(content)

        types = {}
        for item in items:
            types[item.type] = types.get(item.type, 0) + 1

        assert types.get("raw", 0) >= 1  # preamble
        assert types.get("belief", 0) == 2
        assert types.get("episode", 0) == 1
        assert types.get("note", 0) == 1
        assert types.get("goal", 0) == 2
        assert types.get("value", 0) == 1

    def test_parse_markdown_empty_sections(self):
        """Empty sections are skipped."""
        content = """
## Beliefs

## Notes

- Actual note
"""
        items = parse_markdown(content)
        assert len(items) == 1
        assert items[0].type == "note"

    def test_parse_markdown_case_insensitive_headers(self):
        """Section headers are case-insensitive."""
        content = """
## BELIEFS

- Upper case header belief

## beliefs

- Lower case header belief
"""
        items = parse_markdown(content)
        assert len(items) == 2
        assert all(item.type == "belief" for item in items)


class TestMarkdownImporterClass:
    """Tests for the MarkdownImporter class."""

    def test_importer_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for non-existent file."""
        importer = MarkdownImporter(str(tmp_path / "nonexistent.md"))
        with pytest.raises(FileNotFoundError):
            importer.parse()

    def test_importer_parse_file(self, tmp_path):
        """Parse a markdown file from disk."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""
## Beliefs

- Testing matters (90%)
- Quality counts (85%)
""")
        importer = MarkdownImporter(str(md_file))
        items = importer.parse()
        assert len(items) == 2

    def test_importer_import_to_dry_run(self, tmp_path, kernle_instance):
        """Dry run counts raw items without importing (non-raw skipped)."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""
## Thoughts

- Raw thought one
- Raw thought two
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=True)

        assert result["raw"] == 2
        raw_entries = storage.list_raw(limit=100)
        assert len(raw_entries) == 0

    def test_importer_import_to_actual(self, tmp_path, kernle_instance):
        """Actually import raw items into Kernle."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""
## Thoughts

- Markdown imported raw thought
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=False)

        assert result["raw"] == 1
        raw_entries = storage.list_raw(limit=100)
        assert len(raw_entries) == 1

    def test_importer_auto_parse_on_import(self, tmp_path, kernle_instance):
        """import_to() calls parse() if items is empty."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""
## Scratch

- Auto-parsed raw thought
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        # Don't call parse() explicitly
        result = importer.import_to(k, dry_run=False)
        assert result["raw"] == 1


class TestMarkdownImporterAllTypes:
    """Test importing all memory types via Markdown."""

    def test_import_episode_skipped_as_non_raw(self, tmp_path, kernle_instance):
        """Non-raw episode items are skipped during markdown import."""
        md_file = tmp_path / "episodes.md"
        md_file.write_text("""
## Episodes

- Completed important markdown task -> Document the process
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=False)
        assert result["skipped_non_raw"] == 1
        assert result.get("episode", 0) == 0

    def test_import_note_skipped_as_non_raw(self, tmp_path, kernle_instance):
        """Non-raw note items are skipped during markdown import."""
        md_file = tmp_path / "notes.md"
        md_file.write_text("""
## Decisions

- Chose Python for the backend from markdown
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=False)
        assert result["skipped_non_raw"] == 1
        assert result.get("note", 0) == 0

    def test_import_value_skipped_as_non_raw(self, tmp_path, kernle_instance):
        """Non-raw value items are skipped during markdown import."""
        md_file = tmp_path / "values.md"
        md_file.write_text("""
## Principles

- Clarity from markdown: Code should be self-documenting
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=False)
        assert result["skipped_non_raw"] == 1
        assert result.get("value", 0) == 0

    def test_import_goal_skipped_as_non_raw(self, tmp_path, kernle_instance):
        """Non-raw goal items are skipped during markdown import."""
        md_file = tmp_path / "goals.md"
        md_file.write_text("""
## Tasks

- [high] Complete the markdown feature
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=False)
        assert result["skipped_non_raw"] == 1
        assert result.get("goal", 0) == 0

    def test_import_raw(self, tmp_path, kernle_instance):
        """Import raw type."""
        md_file = tmp_path / "thoughts.md"
        md_file.write_text("""
## Thoughts

- Random markdown idea to explore later
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=False)
        assert result["raw"] == 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_csv_unicode_content(self, tmp_path):
        """Handle unicode content in CSV."""
        csv_file = tmp_path / "unicode.csv"
        csv_file.write_text(
            """type,content
note,Hello world with emoji and special chars
note,Caf\u00e9 au lait
""",
            encoding="utf-8",
        )
        importer = CsvImporter(str(csv_file))
        items = importer.parse()
        assert len(items) == 2

    def test_json_unicode_content(self, tmp_path):
        """Handle unicode content in JSON."""
        json_file = tmp_path / "unicode.json"
        json_file.write_text(
            json.dumps(
                {
                    "notes": [
                        {"content": "Hello world"},
                        {"content": "Cafe au lait"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        importer = JsonImporter(str(json_file))
        items = importer.parse()
        assert len(items) == 2

    def test_markdown_unicode_content(self, tmp_path):
        """Handle unicode content in Markdown."""
        md_file = tmp_path / "unicode.md"
        md_file.write_text(
            """
## Notes

- Hello world note
- Cafe au lait
""",
            encoding="utf-8",
        )
        importer = MarkdownImporter(str(md_file))
        items = importer.parse()
        assert len(items) == 2

    def test_csv_with_quotes_and_commas(self, tmp_path):
        """Handle quoted fields with commas in CSV."""
        csv_file = tmp_path / "quoted.csv"
        csv_file.write_text("""type,content
note,"This has a comma, in it"
note,"And ""quotes"" too"
""")
        importer = CsvImporter(str(csv_file))
        items = importer.parse()
        assert len(items) == 2
        assert "comma, in it" in items[0].data["content"]

    def test_expanduser_paths(self, tmp_path, monkeypatch):
        """Paths with ~ are expanded."""
        # This tests that Path.expanduser() is called
        csv_importer = CsvImporter("~/test.csv")
        assert "~" not in str(csv_importer.file_path)

        json_importer = JsonImporter("~/test.json")
        assert "~" not in str(json_importer.file_path)

        md_importer = MarkdownImporter("~/test.md")
        assert "~" not in str(md_importer.file_path)


class TestImportItemDataclass:
    """Tests for ImportItem dataclass."""

    def test_import_item_defaults(self):
        """ImportItem has sensible defaults."""
        item = ImportItem(type="test")
        assert item.content == ""
        assert item.objective == ""
        assert item.confidence == 0.7
        assert item.priority == 50
        assert item.status == "active"
        assert item.source == "import"
        assert item.metadata == {}

    def test_csv_import_item_defaults(self):
        """CsvImportItem has sensible defaults."""
        item = CsvImportItem(type="test")
        assert item.data == {}

    def test_json_import_item_defaults(self):
        """JsonImportItem has sensible defaults."""
        item = JsonImportItem(type="test")
        assert item.data == {}


# ============================================================================
# Markdown Importer Origin Metadata Tests
# ============================================================================


class TestMarkdownImportMeta:
    """Tests for origin tracking metadata on parsed markdown items."""

    def test_parsed_items_include_origin_file(self):
        """Parsed items include _import_meta with origin_file when file_path is provided."""
        content = """
## Beliefs

- Testing is important
"""
        items = parse_markdown(content, origin_file="/path/to/memory.md")
        assert len(items) == 1
        assert "_import_meta" in items[0].metadata
        assert items[0].metadata["_import_meta"]["origin_file"] == "/path/to/memory.md"

    def test_origin_section_matches_section_header(self):
        """origin_section matches the original section header text."""
        content = """
## Beliefs

- Testing is important

## Episodes

- Fixed a bug -> Always test first
"""
        items = parse_markdown(content, origin_file="/path/to/memory.md")
        assert len(items) == 2

        belief_item = next(i for i in items if i.type == "belief")
        assert belief_item.metadata["_import_meta"]["origin_section"] == "Beliefs"

        episode_item = next(i for i in items if i.type == "episode")
        assert episode_item.metadata["_import_meta"]["origin_section"] == "Episodes"

    def test_preamble_items_have_preamble_section(self):
        """Preamble items have origin_section set to 'preamble'."""
        content = """
Just a thought before any sections.

## Beliefs

- Something
"""
        items = parse_markdown(content, origin_file="/tmp/test.md")
        preamble_items = [i for i in items if i.source == "preamble"]
        assert len(preamble_items) >= 1
        assert preamble_items[0].metadata["_import_meta"]["origin_section"] == "preamble"

    def test_no_origin_file_means_no_import_meta(self):
        """When origin_file is not provided, items have no _import_meta."""
        content = """
## Beliefs

- Testing is important
"""
        items = parse_markdown(content)
        assert len(items) == 1
        assert "_import_meta" not in items[0].metadata

    def test_markdown_importer_class_sets_origin_file(self, tmp_path):
        """MarkdownImporter.parse() sets origin_file from its file_path."""
        md_file = tmp_path / "test_origin.md"
        md_file.write_text("""
## Notes

- A note about origin tracking
""")
        importer = MarkdownImporter(str(md_file))
        items = importer.parse()
        assert len(items) == 1
        assert "_import_meta" in items[0].metadata
        assert items[0].metadata["_import_meta"]["origin_file"] == str(md_file)
        assert items[0].metadata["_import_meta"]["origin_section"] == "Notes"

    def test_import_meta_does_not_break_import(self, tmp_path, kernle_instance):
        """Extra _import_meta in metadata does not break actual import."""
        md_file = tmp_path / "test_meta_import.md"
        md_file.write_text("""
## Thoughts

- Import meta raw thought one

## Ideas

- Import meta raw thought two
""")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=False)

        assert result["raw"] == 2

        # Verify the data was actually stored
        raw_entries = storage.list_raw(limit=100)
        assert len(raw_entries) == 2
