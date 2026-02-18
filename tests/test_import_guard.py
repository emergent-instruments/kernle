"""Tests for the empty-stack import guard and has_user_content().

The import guard ensures that import_to() only works on empty stacks.
dry_run=True bypasses the guard so you can always preview imports.
"""

import json

import pytest

from kernle.importers.csv_importer import CsvImporter
from kernle.importers.json_importer import JsonImporter
from kernle.importers.markdown import MarkdownImporter


class TestHasUserContent:
    """Test Kernle.has_user_content() method."""

    def test_fresh_stack_has_no_content(self, kernle_instance):
        k, _ = kernle_instance
        assert not k.has_user_content()

    def test_returns_true_after_episode(self, kernle_instance):
        k, _ = kernle_instance
        k.episode(objective="test", outcome="passed")
        assert k.has_user_content()

    def test_returns_true_after_note(self, kernle_instance):
        k, _ = kernle_instance
        k.note(content="some note")
        assert k.has_user_content()

    def test_returns_true_after_belief(self, kernle_instance):
        k, _ = kernle_instance
        k.belief(statement="sky is blue", source_type="imported")
        assert k.has_user_content()

    def test_returns_true_after_raw(self, kernle_instance):
        k, _ = kernle_instance
        k.raw("brain dump")
        assert k.has_user_content()

    def test_returns_true_after_value(self, kernle_instance):
        k, _ = kernle_instance
        k.value(name="honesty", statement="be honest", source_type="imported")
        assert k.has_user_content()

    def test_returns_true_after_goal(self, kernle_instance):
        k, _ = kernle_instance
        k.goal(title="ship it", source_type="imported")
        assert k.has_user_content()

    def test_returns_true_after_drive(self, kernle_instance):
        k, _ = kernle_instance
        k.drive(drive_type="curiosity", source_type="imported")
        assert k.has_user_content()


class TestImportGuardCsvImporter:
    """Test that CsvImporter.import_to() rejects non-empty stacks."""

    def test_import_into_empty_stack_succeeds(self, tmp_path, kernle_instance):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("type,content\nraw,hello world\n")
        k, storage = kernle_instance
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False)
        assert result["imported"]["raw"] == 1

    def test_import_into_non_empty_stack_raises(self, tmp_path, kernle_instance):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("type,content\nraw,hello world\n")
        k, _ = kernle_instance
        k.raw("existing content")
        importer = CsvImporter(str(csv_file))
        with pytest.raises(ValueError, match="existing content"):
            importer.import_to(k, dry_run=False)

    def test_dry_run_bypasses_guard(self, tmp_path, kernle_instance):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("type,content\nraw,hello world\n")
        k, _ = kernle_instance
        k.raw("existing content")
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=True)
        assert result["imported"]["raw"] == 1


class TestImportGuardJsonImporter:
    """Test that JsonImporter.import_to() rejects non-empty stacks."""

    def test_import_into_empty_stack_succeeds(self, tmp_path, kernle_instance):
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "test data"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "test",
                            "outcome": "passed",
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

    def test_import_into_non_empty_stack_raises(self, tmp_path, kernle_instance):
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "test data"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "test",
                            "outcome": "passed",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                }
            )
        )
        k, _ = kernle_instance
        k.raw("existing content")
        importer = JsonImporter(str(json_file))
        with pytest.raises(ValueError, match="existing content"):
            importer.import_to(k, dry_run=False)

    def test_dry_run_bypasses_guard(self, tmp_path, kernle_instance):
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "test data"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "test",
                            "outcome": "passed",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                }
            )
        )
        k, _ = kernle_instance
        k.raw("existing content")
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=True)
        assert result["imported"]["episode"] == 1


class TestImportGuardMarkdownImporter:
    """Test that MarkdownImporter.import_to() rejects non-empty stacks."""

    def test_import_into_empty_stack_succeeds(self, tmp_path, kernle_instance):
        md_file = tmp_path / "test.md"
        md_file.write_text("## Thoughts\n\n- hello world\n")
        k, storage = kernle_instance
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=False)
        assert result.get("raw", 0) == 1

    def test_import_into_non_empty_stack_raises(self, tmp_path, kernle_instance):
        md_file = tmp_path / "test.md"
        md_file.write_text("## Thoughts\n\n- hello world\n")
        k, _ = kernle_instance
        k.raw("existing content")
        importer = MarkdownImporter(str(md_file))
        with pytest.raises(ValueError, match="existing content"):
            importer.import_to(k, dry_run=False)

    def test_dry_run_bypasses_guard(self, tmp_path, kernle_instance):
        md_file = tmp_path / "test.md"
        md_file.write_text("## Thoughts\n\n- hello world\n")
        k, _ = kernle_instance
        k.raw("existing content")
        importer = MarkdownImporter(str(md_file))
        result = importer.import_to(k, dry_run=True)
        assert result.get("raw", 0) == 1


class TestProvenanceChainValidation:
    """Test provenance chain enforcement on JSON import."""

    def test_valid_provenance_chain_succeeds(self, tmp_path, kernle_instance):
        """JSON with valid raw -> episode -> belief chain imports successfully."""
        json_file = tmp_path / "valid_chain.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "observation data"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "test",
                            "outcome": "passed",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                    "beliefs": [
                        {
                            "id": "b1",
                            "statement": "Testing works",
                            "confidence": 0.9,
                            "derived_from": ["episode:e1"],
                        }
                    ],
                }
            )
        )
        k, storage = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False)
        assert result["imported"]["raw"] == 1
        assert result["imported"]["episode"] == 1
        assert result["imported"]["belief"] == 1
        assert not result["errors"]

    def test_missing_derived_from_fails(self, tmp_path, kernle_instance):
        """Belief with no derived_from raises ValueError."""
        json_file = tmp_path / "no_provenance.json"
        json_file.write_text(
            json.dumps(
                {
                    "beliefs": [{"id": "b1", "statement": "No provenance", "confidence": 0.9}],
                }
            )
        )
        k, _ = kernle_instance
        importer = JsonImporter(str(json_file))
        with pytest.raises(ValueError, match="Provenance validation failed"):
            importer.import_to(k, dry_run=False)

    def test_missing_referenced_id_fails(self, tmp_path, kernle_instance):
        """Belief referencing non-existent episode raises ValueError."""
        json_file = tmp_path / "bad_ref.json"
        json_file.write_text(
            json.dumps(
                {
                    "beliefs": [
                        {
                            "id": "b1",
                            "statement": "Bad ref",
                            "confidence": 0.9,
                            "derived_from": ["episode:e_nonexistent"],
                        }
                    ],
                }
            )
        )
        k, _ = kernle_instance
        importer = JsonImporter(str(json_file))
        with pytest.raises(ValueError, match="Provenance validation failed"):
            importer.import_to(k, dry_run=False)

    def test_wrong_ref_type_fails(self, tmp_path, kernle_instance):
        """Belief referencing raw directly (must go through episode/note) raises ValueError."""
        json_file = tmp_path / "wrong_type.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw data"}],
                    "beliefs": [
                        {
                            "id": "b1",
                            "statement": "Wrong ref type",
                            "confidence": 0.9,
                            "derived_from": ["raw:r1"],
                        }
                    ],
                }
            )
        )
        k, _ = kernle_instance
        importer = JsonImporter(str(json_file))
        with pytest.raises(ValueError, match="Provenance validation failed"):
            importer.import_to(k, dry_run=False)

    def test_annotation_refs_dont_count_as_provenance(self, tmp_path, kernle_instance):
        """Item with only annotation refs (context:*) fails provenance check."""
        json_file = tmp_path / "annotation_only.json"
        json_file.write_text(
            json.dumps(
                {
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "test",
                            "outcome": "passed",
                            "derived_from": ["context:session-123"],
                        }
                    ],
                }
            )
        )
        k, _ = kernle_instance
        importer = JsonImporter(str(json_file))
        with pytest.raises(ValueError, match="Provenance validation failed"):
            importer.import_to(k, dry_run=False)

    def test_dry_run_reports_provenance_errors(self, tmp_path, kernle_instance):
        """dry_run with bad provenance returns errors, writes zero rows."""
        json_file = tmp_path / "bad_provenance.json"
        json_file.write_text(
            json.dumps(
                {
                    "beliefs": [{"id": "b1", "statement": "No chain", "confidence": 0.9}],
                }
            )
        )
        k, _ = kernle_instance
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=True)
        assert result["imported"] == {}
        assert len(result["errors"]) > 0
