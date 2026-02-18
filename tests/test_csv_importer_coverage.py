"""Unit tests for CSV importer edge cases in kernle/importers/csv_importer.py.

These tests cover:
- parse_csv with empty file (no headers)
- parse_csv with headers but no data rows
- parse_csv with malformed/unmapped headers
- parse_csv requiring a 'type' column or --type override
- _map_columns confidence conversion (including percentage values)
- _map_columns priority conversion for values
- _map_columns tag/lesson splitting
- CsvImporter with nonexistent file
- CsvImporter import_to with dry_run mode
- parse_csv skipping rows with unknown memory types
- parse_csv skipping rows without content
"""

import tempfile
from unittest.mock import MagicMock

import pytest

from kernle.importers.csv_importer import (
    COLUMN_MAPPINGS,
    CsvImporter,
    _map_columns,
    parse_csv,
)

# =========================================================================
# parse_csv: Empty / malformed files
# =========================================================================


class TestParseCsvEmptyFiles:
    """parse_csv should handle empty and malformed CSV content."""

    def test_empty_string_raises_value_error(self):
        """An empty string should raise ValueError due to missing headers."""
        with pytest.raises(ValueError, match="no headers"):
            parse_csv("")

    def test_headers_only_returns_empty_list(self):
        """CSV with headers but no data rows should return an empty list."""
        csv_content = "type,content,confidence\n"
        result = parse_csv(csv_content)
        assert result == []

    def test_no_type_column_without_override_raises_error(self):
        """CSV without 'type' column and no memory_type override should raise."""
        csv_content = "content,confidence\nSome belief,0.9\n"

        with pytest.raises(ValueError, match="type.*column"):
            parse_csv(csv_content)

    def test_no_type_column_with_override_succeeds(self):
        """CSV without 'type' column should work when memory_type is specified."""
        csv_content = "statement,confidence\nPython is great,0.9\n"
        result = parse_csv(csv_content, memory_type="belief")

        assert len(result) == 1
        assert result[0].type == "belief"
        assert result[0].data["statement"] == "Python is great"


# =========================================================================
# parse_csv: Malformed headers and column mapping
# =========================================================================


class TestParseCsvMalformedHeaders:
    """parse_csv should handle CSVs with unrecognized or mixed-case headers."""

    def test_unrecognized_headers_produce_empty_data(self):
        """Rows with only unrecognized extra columns should have minimal data.

        Note: the 'type' column itself maps to 'type' in belief's COLUMN_MAPPINGS,
        so the row still has data={type: 'belief'} and is not skipped. But extra
        columns like 'zzz_unknown' are ignored and not included in data.
        """
        csv_content = "type,zzz_unknown,yyy_nothing\nbelief,foo,bar\n"
        result = parse_csv(csv_content)

        # The row is not skipped because 'type' maps to belief's 'type' field
        assert len(result) == 1
        # But the unrecognized columns should not appear in the data
        assert "zzz_unknown" not in result[0].data
        assert "yyy_nothing" not in result[0].data

    def test_mixed_case_headers_normalized(self):
        """Headers should be normalized to lowercase for matching."""
        csv_content = "Type,Statement,Confidence\nbelief,Test statement,0.9\n"
        result = parse_csv(csv_content)

        assert len(result) == 1
        assert result[0].data["statement"] == "Test statement"

    def test_whitespace_in_headers_stripped(self):
        """Whitespace in header names should be stripped."""
        csv_content = " type , content , confidence \nbelief,My belief,0.8\n"
        result = parse_csv(csv_content)

        assert len(result) == 1
        assert result[0].type == "belief"

    def test_alias_headers_mapped_correctly(self):
        """Column aliases (e.g., 'text' for 'content') should be recognized."""
        csv_content = "type,text\nnote,My note text\n"
        result = parse_csv(csv_content)

        assert len(result) == 1
        assert result[0].data["content"] == "My note text"


# =========================================================================
# parse_csv: Row-level handling
# =========================================================================


class TestParseCsvRowHandling:
    """parse_csv should correctly handle various row conditions."""

    def test_unknown_type_rows_skipped(self):
        """Rows with unrecognized memory types should be silently skipped."""
        csv_content = (
            "type,content\n" "belief,Valid belief\n" "unicorn,Invalid type\n" "note,Valid note\n"
        )
        result = parse_csv(csv_content)

        assert len(result) == 2
        types = [item.type for item in result]
        assert "unicorn" not in types

    def test_rows_without_type_value_skipped(self):
        """Rows where the type column is empty should be skipped."""
        csv_content = "type,content\n,Empty type\nnote,Valid note\n"
        result = parse_csv(csv_content)

        assert len(result) == 1
        assert result[0].type == "note"

    def test_empty_data_rows_skipped(self):
        """Rows where all mapped fields are empty should be skipped.

        We use memory_type override so the 'type' column itself does not
        contribute mapped data, making the row truly empty.
        """
        csv_content = "statement\n\n"
        result = parse_csv(csv_content, memory_type="belief")

        # The row has no statement content, so data is empty and row is skipped
        assert len(result) == 0

    def test_multiple_valid_types_parsed(self):
        """CSV with multiple memory types should parse each correctly."""
        csv_content = (
            "type,content,statement,objective,outcome\n"
            "note,My note,,,\n"
            "belief,,My belief,,\n"
            "episode,,,Learn pytest,Tests passed\n"
        )
        result = parse_csv(csv_content)

        assert len(result) == 3
        types = [item.type for item in result]
        assert "note" in types
        assert "belief" in types
        assert "episode" in types


# =========================================================================
# _map_columns: Type conversions
# =========================================================================


class TestMapColumnsTypeConversions:
    """_map_columns should handle type conversions for specific fields."""

    def test_confidence_float_conversion(self):
        """Confidence should be converted to a float."""
        row = {"statement": "Test", "confidence": "0.85"}
        result, _, _ = _map_columns(row, "belief")

        assert result["confidence"] == pytest.approx(0.85)

    def test_confidence_percentage_conversion(self):
        """Confidence values > 1 should be treated as percentages and divided by 100."""
        row = {"statement": "Test", "confidence": "90"}
        result, _, _ = _map_columns(row, "belief")

        assert result["confidence"] == pytest.approx(0.9)

    def test_confidence_invalid_defaults_to_0_7(self):
        """Non-numeric confidence values should default to 0.7."""
        row = {"statement": "Test", "confidence": "very high"}
        result, _, _ = _map_columns(row, "belief")

        assert result["confidence"] == pytest.approx(0.7)

    def test_priority_int_conversion_for_values(self):
        """Priority for values should be converted to an integer."""
        row = {"name": "Test Value", "priority": "75"}
        result, _, _ = _map_columns(row, "value")

        assert result["priority"] == 75

    def test_priority_invalid_defaults_to_50(self):
        """Non-numeric priority should default to 50."""
        row = {"name": "Test Value", "priority": "high"}
        result, _, _ = _map_columns(row, "value")

        assert result["priority"] == 50

    def test_tags_split_by_comma(self):
        """Tags should be split by comma into a list."""
        row = {"objective": "Test", "tags": "python, testing, tdd"}
        result, _, _ = _map_columns(row, "episode")

        assert result["tags"] == ["python", "testing", "tdd"]

    def test_lessons_split_by_comma(self):
        """Lessons should be split by comma into a list."""
        row = {"objective": "Test", "lessons": "Use mocks, Write clean code"}
        result, _, _ = _map_columns(row, "episode")

        assert result["lessons"] == ["Use mocks", "Write clean code"]

    def test_empty_tags_filtered_out(self):
        """Empty strings from comma splitting should be filtered out."""
        row = {"objective": "Test", "tags": "python,,, testing,"}
        result, _, _ = _map_columns(row, "episode")

        assert result["tags"] == ["python", "testing"]


# =========================================================================
# CsvImporter: File handling
# =========================================================================


class TestCsvImporterFileHandling:
    """CsvImporter should handle file-level edge cases."""

    def test_nonexistent_file_raises_file_not_found(self):
        """parse() on a nonexistent file should raise FileNotFoundError."""
        importer = CsvImporter("/tmp/does_not_exist_12345.csv")

        with pytest.raises(FileNotFoundError, match="File not found"):
            importer.parse()

    def test_empty_file_raises_value_error(self):
        """parse() on a truly empty file should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
            f.flush()

            importer = CsvImporter(f.name)
            with pytest.raises(ValueError, match="no headers"):
                importer.parse()

    def test_valid_file_parsed_successfully(self):
        """parse() on a valid CSV file should return items."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("type,statement,confidence\nbelief,Python rocks,0.9\n")
            f.flush()

            importer = CsvImporter(f.name)
            items = importer.parse()

            assert len(items) == 1
            assert items[0].type == "belief"

    def test_memory_type_override_applied(self):
        """CsvImporter with memory_type should apply it to all rows."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("content\nFirst note\nSecond note\n")
            f.flush()

            importer = CsvImporter(f.name, memory_type="note")
            items = importer.parse()

            assert len(items) == 2
            assert all(item.type == "note" for item in items)


# =========================================================================
# CsvImporter: import_to dry_run
# =========================================================================


class TestCsvImporterDryRun:
    """import_to with dry_run=True should count items without importing."""

    def test_dry_run_counts_items_without_importing(self):
        """dry_run should return counts but not call any Kernle methods."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("type,content\n" "raw,First raw entry\n" "raw,Second raw entry\n")
            f.flush()

            importer = CsvImporter(f.name)
            importer.parse()

            # Use a MagicMock for the Kernle instance
            mock_k = MagicMock()

            result = importer.import_to(mock_k, dry_run=True)

            assert result["imported"]["raw"] == 2
            assert result["errors"] == []
            # Kernle methods should NOT have been called
            mock_k.raw.assert_not_called()
            mock_k.belief.assert_not_called()
            mock_k.episode.assert_not_called()


# =========================================================================
# CsvImporter: import_to auto-parse
# =========================================================================


class TestCsvImporterAutoparse:
    """import_to should auto-parse if items haven't been parsed yet."""

    def test_import_to_auto_parses_when_items_empty(self):
        """import_to should call parse() if self.items is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("type,content\nraw,Auto-parsed raw entry\n")
            f.flush()

            importer = CsvImporter(f.name)
            # Don't call parse() explicitly

            mock_k = MagicMock()
            result = importer.import_to(mock_k, dry_run=True)

            assert result["imported"]["raw"] == 1


# =========================================================================
# COLUMN_MAPPINGS: Coverage of alias support
# =========================================================================


class TestColumnMappings:
    """Verify that COLUMN_MAPPINGS covers expected aliases."""

    def test_episode_objective_aliases(self):
        """Episode 'objective' should have common aliases."""
        aliases = COLUMN_MAPPINGS["episode"]["objective"]
        assert "objective" in aliases
        assert "title" in aliases
        assert "task" in aliases

    def test_note_content_aliases(self):
        """Note 'content' should have common aliases."""
        aliases = COLUMN_MAPPINGS["note"]["content"]
        assert "content" in aliases
        assert "text" in aliases
        assert "body" in aliases

    def test_belief_statement_aliases(self):
        """Belief 'statement' should have common aliases."""
        aliases = COLUMN_MAPPINGS["belief"]["statement"]
        assert "statement" in aliases
        assert "belief" in aliases
        assert "content" in aliases

    def test_raw_content_aliases(self):
        """Raw 'content' should have common aliases."""
        aliases = COLUMN_MAPPINGS["raw"]["content"]
        assert "content" in aliases
        assert "raw" in aliases
        assert "data" in aliases
