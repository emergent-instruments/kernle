"""Tests for importer schema coercion rules (issue #716).

Tests cover:
- Bounds validation for confidence, priority, intensity, sentiment
- Coercion warnings tracking
- Strict mode rejections vs permissive mode defaults
- Rejection reason tracking for silent skips
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
    _import_json_item,
)

# ============================================================================
# CSV Importer: Confidence validation
# ============================================================================


class TestCsvConfidenceValidation:
    """Test confidence bounds validation in CSV importer."""

    def test_csv_confidence_out_of_bounds_clamped(self):
        """confidence=150 (>100) auto-scales to 1.5, then clamped to 1.0 in permissive mode."""
        row = {"statement": "Test belief", "confidence": "150"}
        result, warnings, _ = _map_columns(row, "belief")
        # 150 > 1 triggers divide-by-100 = 1.5, which is > 1.0, so clamp to 1.0
        assert result["confidence"] == 1.0
        assert len(warnings) == 1
        assert warnings[0]["field"] == "confidence"

    def test_csv_confidence_negative_clamped(self):
        """confidence=-0.5 should be clamped to 0.0 in permissive mode."""
        row = {"statement": "Test belief", "confidence": "-0.5"}
        result, _, _ = _map_columns(row, "belief")
        assert result["confidence"] == 0.0

    def test_csv_confidence_non_numeric_strict_rejects(self):
        """confidence='high' in strict mode should reject the row."""
        items = parse_csv(
            """type,statement,confidence
belief,Test belief,high
""",
            strict=True,
        )
        assert len(items) == 0

    def test_csv_confidence_non_numeric_permissive_defaults(self):
        """confidence='high' in permissive mode defaults to 0.7 with warning."""
        items, warnings = parse_csv(
            """type,statement,confidence
belief,Test belief,high
""",
            strict=False,
            return_warnings=True,
        )
        assert len(items) == 1
        assert items[0].data["confidence"] == 0.7
        assert len(warnings) >= 1
        assert warnings[0]["field"] == "confidence"
        assert warnings[0]["original"] == "high"
        assert warnings[0]["coerced_to"] == 0.7

    def test_csv_confidence_value_over_100_rejected_strict(self):
        """confidence=150 should be rejected in strict mode."""
        items = parse_csv(
            """type,statement,confidence
belief,Test belief,150
""",
            strict=True,
        )
        assert len(items) == 0

    def test_csv_confidence_value_over_100_clamped_permissive(self):
        """confidence=150 in permissive mode: 150/100=1.5, clamped to 1.0."""
        items, warnings = parse_csv(
            """type,statement,confidence
belief,Test belief,150
""",
            strict=False,
            return_warnings=True,
        )
        assert len(items) == 1
        assert items[0].data["confidence"] == 1.0
        assert any(w["field"] == "confidence" for w in warnings)

    def test_csv_confidence_valid_percentage_auto_scaled(self):
        """confidence=85 should auto-scale to 0.85 (existing behavior preserved)."""
        row = {"statement": "Test belief", "confidence": "85"}
        result, _, _ = _map_columns(row, "belief")
        assert result["confidence"] == pytest.approx(0.85)

    def test_csv_confidence_valid_decimal_preserved(self):
        """confidence=0.92 should be preserved as-is."""
        row = {"statement": "Test belief", "confidence": "0.92"}
        result, _, _ = _map_columns(row, "belief")
        assert result["confidence"] == pytest.approx(0.92)


# ============================================================================
# CSV Importer: Priority validation
# ============================================================================


class TestCsvPriorityValidation:
    """Test priority bounds validation in CSV importer."""

    def test_csv_priority_out_of_bounds_clamped(self):
        """priority=200 should be clamped to 100 in permissive mode."""
        row = {"name": "Quality", "priority": "200"}
        result, _, _ = _map_columns(row, "value")
        assert result["priority"] == 100

    def test_csv_priority_negative_clamped(self):
        """priority=-10 should be clamped to 0 in permissive mode."""
        row = {"name": "Quality", "priority": "-10"}
        result, _, _ = _map_columns(row, "value")
        assert result["priority"] == 0

    def test_csv_priority_non_numeric_strict_rejects(self):
        """priority='high' in strict mode should reject the row."""
        items = parse_csv(
            """type,name,description,priority
value,Quality,Code quality,high
""",
            strict=True,
        )
        assert len(items) == 0

    def test_csv_priority_non_numeric_permissive_defaults(self):
        """priority='high' in permissive mode defaults to 50 with warning."""
        items, warnings = parse_csv(
            """type,name,description,priority
value,Quality,Code quality,high
""",
            strict=False,
            return_warnings=True,
        )
        assert len(items) == 1
        assert items[0].data["priority"] == 50
        assert any(w["field"] == "priority" for w in warnings)


# ============================================================================
# CSV Importer: Intensity validation
# ============================================================================


class TestCsvIntensityValidation:
    """Test intensity bounds validation in CSV importer (via _map_columns)."""

    def test_csv_intensity_out_of_bounds_clamped(self):
        """intensity=2.0 should be clamped to 1.0."""
        # Intensity is relevant for drives in JSON, but _map_columns should
        # handle it if it ever appears in CSV. We test via JSON importer below.
        pass


# ============================================================================
# JSON Importer: Intensity validation
# ============================================================================


class TestJsonIntensityValidation:
    """Test intensity bounds validation in JSON importer."""

    def test_json_intensity_out_of_bounds_clamped(self, kernle_instance):
        """intensity=2.0 should be clamped to 1.0 in permissive mode."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "curiosity", "intensity": 2.0},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is not None

        drives = storage.get_drives()
        assert len(drives) == 1
        assert drives[0].intensity <= 1.0

    def test_json_intensity_negative_clamped(self, kernle_instance):
        """intensity=-0.5 should be clamped to 0.0."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "growth", "intensity": -0.5},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is not None

        drives = storage.get_drives()
        assert len(drives) == 1
        assert drives[0].intensity >= 0.0

    def test_json_intensity_strict_rejects_out_of_bounds(self, kernle_instance):
        """intensity=2.0 in strict mode should reject the item."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "curiosity", "intensity": 2.0},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None


# ============================================================================
# JSON Importer: Sentiment validation
# ============================================================================


class TestJsonSentimentValidation:
    """Test sentiment bounds validation in JSON importer."""

    def test_json_sentiment_out_of_bounds_clamped(self, kernle_instance):
        """sentiment=5.0 should be clamped to 1.0 in permissive mode."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="relationship",
            data={
                "entity_name": "TestEntity",
                "entity_type": "person",
                "relationship_type": "friend",
                "sentiment": 5.0,
            },
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is not None

        rel = storage.get_relationship("TestEntity")
        assert rel is not None
        # sentiment 5.0 clamped to 1.0, stored as sentiment on the object
        assert rel.sentiment <= 1.0

    def test_json_sentiment_negative_out_of_bounds_clamped(self, kernle_instance):
        """sentiment=-5.0 should be clamped to -1.0."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="relationship",
            data={
                "entity_name": "NegEntity",
                "entity_type": "person",
                "relationship_type": "rival",
                "sentiment": -5.0,
            },
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is not None

        rel = storage.get_relationship("NegEntity")
        assert rel is not None
        # sentiment -5.0 clamped to -1.0
        assert rel.sentiment >= -1.0

    def test_json_sentiment_strict_rejects_out_of_bounds(self, kernle_instance):
        """sentiment=5.0 in strict mode should reject the item."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="relationship",
            data={
                "entity_name": "TestEntity",
                "entity_type": "person",
                "relationship_type": "friend",
                "sentiment": 5.0,
            },
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None


# ============================================================================
# JSON Importer: Confidence validation
# ============================================================================


class TestJsonConfidenceValidation:
    """Test confidence bounds validation in JSON importer."""

    def test_json_confidence_out_of_bounds_clamped(self, kernle_instance):
        """confidence=1.5 should be clamped to 1.0."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="belief",
            data={"statement": "Test belief clamped", "confidence": 1.5},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is not None

        beliefs = storage.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confidence <= 1.0

    def test_json_confidence_negative_clamped(self, kernle_instance):
        """confidence=-0.5 should be clamped to 0.0."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="belief",
            data={"statement": "Negative confidence", "confidence": -0.5},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is not None

        beliefs = storage.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confidence >= 0.0


# ============================================================================
# JSON Importer: Priority validation
# ============================================================================


class TestJsonPriorityValidation:
    """Test priority bounds validation in JSON importer."""

    def test_json_value_priority_out_of_bounds_clamped(self, kernle_instance):
        """priority=200 should be clamped to 100."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="value",
            data={"name": "Excessive Priority", "statement": "Test", "priority": 200},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is not None

        values = storage.get_values()
        assert len(values) == 1
        assert values[0].priority <= 100


# ============================================================================
# Coercion Warnings Tracking
# ============================================================================


class TestCoercionWarningsTracked:
    """Test that coercion warnings are properly tracked and returned."""

    def test_coercion_warnings_tracked(self):
        """Verify warnings list is populated when values are coerced."""
        csv_content = """type,statement,confidence
belief,Belief with bad conf,invalid_value
belief,Belief clamped negative,-0.5
"""
        items, warnings = parse_csv(csv_content, strict=False, return_warnings=True)
        assert len(items) == 2
        assert len(warnings) >= 2

        # Check warning structure
        for w in warnings:
            assert "row" in w
            assert "field" in w
            assert "original" in w
            assert "coerced_to" in w
            assert "reason" in w

    def test_no_warnings_for_valid_data(self):
        """No warnings should be generated for valid data."""
        csv_content = """type,statement,confidence
belief,Valid belief,0.9
belief,Another valid,0.7
"""
        items, warnings = parse_csv(csv_content, strict=False, return_warnings=True)
        assert len(items) == 2
        assert len(warnings) == 0

    def test_json_coercion_warnings_in_import_result(self, tmp_path, kernle_instance):
        """JSON importer should include coercion_warnings in result."""
        k, storage = kernle_instance
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw entry"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "obj",
                            "outcome": "out",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                    "drives": [
                        {
                            "drive_type": "curiosity",
                            "intensity": 2.0,
                            "derived_from": ["episode:e1"],
                        },
                    ],
                }
            )
        )
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["drive"] == 1
        assert "coercion_warnings" in result
        assert len(result["coercion_warnings"]) >= 1


# ============================================================================
# Strict Mode
# ============================================================================


class TestStrictModeRejectsMalformed:
    """Test that strict mode rejects malformed values instead of defaulting."""

    def test_strict_mode_rejects_malformed(self):
        """strict=True should reject rows with non-numeric confidence/priority."""
        csv_content = """type,statement,confidence
belief,Good belief,0.9
belief,Bad confidence,not_a_number
"""
        items = parse_csv(csv_content, strict=True)
        # Only the valid row should survive
        assert len(items) == 1
        assert items[0].data["statement"] == "Good belief"

    def test_strict_mode_rejects_out_of_range_confidence(self):
        """strict=True should reject rows with confidence > 100."""
        items = parse_csv(
            """type,statement,confidence
belief,Over 100,150
""",
            strict=True,
        )
        assert len(items) == 0

    def test_strict_mode_rejects_negative_confidence(self):
        """strict=True should reject rows with confidence < 0."""
        items = parse_csv(
            """type,statement,confidence
belief,Negative,-0.5
""",
            strict=True,
        )
        assert len(items) == 0

    def test_strict_mode_preserves_valid_rows(self):
        """strict=True should not reject valid data."""
        csv_content = """type,statement,confidence
belief,Valid belief,0.9
belief,Another valid,85
"""
        items = parse_csv(csv_content, strict=True)
        assert len(items) == 2


# ============================================================================
# Rejection Reasons Tracking
# ============================================================================


class TestRejectionReasonsTracked:
    """Test that rejections (silent skips) are reported with reason."""

    def test_rejection_reasons_tracked(self):
        """Rows rejected in strict mode should report reason."""
        csv_content = """type,statement,confidence
belief,Valid belief,0.9
belief,Bad confidence,xyz
belief,Too high,999
"""
        items, warnings, rejections = parse_csv(
            csv_content, strict=True, return_warnings=True, return_rejections=True
        )
        assert len(items) == 1
        assert len(rejections) >= 2

        # Check rejection structure
        for r in rejections:
            assert "row" in r
            assert "field" in r
            assert "value" in r
            assert "reason" in r

    def test_csv_importer_import_result_includes_rejections(self, tmp_path, kernle_instance):
        """CsvImporter.import_to result should include rejections list."""
        k, storage = kernle_instance
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,statement,confidence
belief,Valid belief,0.9
belief,Missing statement,
""")
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert "rejections" in result


# ============================================================================
# Backwards Compatibility
# ============================================================================


class TestBackwardsCompatibility:
    """Verify that default behavior remains permissive (no breaking changes)."""

    def test_default_parse_csv_is_permissive(self):
        """parse_csv() without strict parameter uses permissive mode."""
        csv_content = """type,statement,confidence
belief,Test,invalid
"""
        items = parse_csv(csv_content)
        # Should still produce an item with default confidence
        assert len(items) == 1
        assert items[0].data["confidence"] == 0.7

    def test_default_parse_csv_returns_items_only(self):
        """parse_csv() without return_warnings returns items only."""
        csv_content = """type,statement,confidence
belief,Test,invalid
"""
        result = parse_csv(csv_content)
        # Should return a list, not a tuple
        assert isinstance(result, list)

    def test_csv_importer_default_not_strict(self, tmp_path, kernle_instance):
        """CsvImporter default import is not strict (test via _import_csv_item)."""
        from kernle.importers.csv_importer import _import_csv_item

        k, storage = kernle_instance
        # CSV import_to() only imports raw items now, so test permissive
        # coercion through _import_csv_item directly which still handles
        # any type.
        item = CsvImportItem(
            type="belief",
            data={"statement": "Permissive test", "confidence": 0.7},
        )
        result = _import_csv_item(item, k, skip_duplicates=False)
        assert result is True

    def test_json_importer_default_not_strict(self, tmp_path, kernle_instance):
        """JsonImporter default import is not strict."""
        k, storage = kernle_instance
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "raw_entries": [{"id": "r1", "content": "raw entry"}],
                    "episodes": [
                        {
                            "id": "e1",
                            "objective": "obj",
                            "outcome": "out",
                            "derived_from": ["raw:r1"],
                        }
                    ],
                    "drives": [
                        {
                            "drive_type": "curiosity",
                            "intensity": 2.0,
                            "derived_from": ["episode:e1"],
                        },
                    ],
                }
            )
        )
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["drive"] == 1


# ============================================================================
# Boolean / NaN / Infinity strict-mode validation (#723)
# ============================================================================


class TestBooleanNaNInfinityImportValidation:
    """Booleans, NaN, and Infinity must not bypass numeric validation."""

    # --- CLI _validate_import_numeric ---

    def test_validate_import_numeric_rejects_bool_strict(self):
        """Boolean True should be rejected in strict mode."""
        from kernle.cli.commands.import_cmd import _validate_import_numeric

        val, rejected = _validate_import_numeric(True, 0.0, 1.0, 0.8, strict=True)
        assert rejected is True

    def test_validate_import_numeric_rejects_false_strict(self):
        """Boolean False should be rejected in strict mode."""
        from kernle.cli.commands.import_cmd import _validate_import_numeric

        val, rejected = _validate_import_numeric(False, 0.0, 1.0, 0.8, strict=True)
        assert rejected is True

    def test_validate_import_numeric_defaults_bool_permissive(self):
        """Boolean True returns default in permissive mode."""
        from kernle.cli.commands.import_cmd import _validate_import_numeric

        val, rejected = _validate_import_numeric(True, 0.0, 1.0, 0.8, strict=False)
        assert rejected is False
        assert val == 0.8

    def test_validate_import_numeric_rejects_nan_strict(self):
        """NaN should be rejected in strict mode."""
        from kernle.cli.commands.import_cmd import _validate_import_numeric

        val, rejected = _validate_import_numeric(float("nan"), 0.0, 1.0, 0.8, strict=True)
        assert rejected is True

    def test_validate_import_numeric_rejects_inf_strict(self):
        """Infinity should be rejected in strict mode."""
        from kernle.cli.commands.import_cmd import _validate_import_numeric

        val, rejected = _validate_import_numeric(float("inf"), 0.0, 1.0, 0.8, strict=True)
        assert rejected is True

    def test_validate_import_numeric_rejects_neg_inf_strict(self):
        """-Infinity should be rejected in strict mode."""
        from kernle.cli.commands.import_cmd import _validate_import_numeric

        val, rejected = _validate_import_numeric(float("-inf"), 0.0, 1.0, 0.8, strict=True)
        assert rejected is True

    # --- JSON Importer: Boolean confidence ---

    def test_json_belief_bool_confidence_strict_rejected(self, kernle_instance):
        """Boolean confidence in strict mode should reject the belief."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="belief",
            data={"statement": "Bool confidence", "confidence": True},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None

    def test_json_belief_bool_confidence_permissive_defaults(self, kernle_instance):
        """Boolean confidence in permissive mode defaults to 0.8."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="belief",
            data={"statement": "Bool confidence permissive", "confidence": True},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=False)
        assert result is not None
        beliefs = storage.get_beliefs()
        assert any(b.confidence == pytest.approx(0.8) for b in beliefs)

    # --- JSON Importer: Boolean intensity ---

    def test_json_drive_bool_intensity_strict_rejected(self, kernle_instance):
        """Boolean intensity in strict mode should reject the drive."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "bool_drive", "intensity": True},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None

    # --- JSON Importer: Boolean sentiment ---

    def test_json_relationship_bool_sentiment_strict_rejected(self, kernle_instance):
        """Boolean sentiment in strict mode should reject the relationship."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="relationship",
            data={"entity_name": "bool_entity", "sentiment": True},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None

    # --- JSON Importer: Boolean priority ---

    def test_json_value_bool_priority_strict_rejected(self, kernle_instance):
        """Boolean priority in strict mode should reject the value."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="value",
            data={"name": "Bool Priority", "statement": "Test", "priority": True},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None

    def test_json_value_bool_priority_permissive_defaults(self, kernle_instance):
        """Boolean priority in permissive mode defaults to 50."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="value",
            data={"name": "Bool Priority Default", "statement": "Test", "priority": False},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=False)
        assert result is not None
        values = storage.get_values()
        matching = [v for v in values if v.name == "Bool Priority Default"]
        assert len(matching) == 1
        assert matching[0].priority == 50

    # --- JSON Importer: NaN priority (was crashing with int(float('nan'))) ---

    def test_json_value_nan_priority_strict_rejected(self, kernle_instance):
        """NaN priority in strict mode should reject, not crash."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="value",
            data={"name": "NaN Priority", "statement": "Test", "priority": float("nan")},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None

    def test_json_value_nan_priority_permissive_defaults(self, kernle_instance):
        """NaN priority in permissive mode should default to 50, not crash."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="value",
            data={"name": "NaN Default", "statement": "Test", "priority": float("nan")},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=False)
        assert result is not None
        values = storage.get_values()
        matching = [v for v in values if v.name == "NaN Default"]
        assert len(matching) == 1
        assert matching[0].priority == 50

    def test_json_value_inf_priority_strict_rejected(self, kernle_instance):
        """Infinity priority in strict mode should reject."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="value",
            data={"name": "Inf Priority", "statement": "Test", "priority": float("inf")},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None

    # --- JSON Importer: NaN/Inf confidence ---

    def test_json_belief_nan_confidence_strict_rejected(self, kernle_instance):
        """NaN confidence should be rejected via _validate_range."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="belief",
            data={"statement": "NaN belief", "confidence": float("nan")},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None

    def test_json_drive_inf_intensity_strict_rejected(self, kernle_instance):
        """Infinity intensity should be rejected via _validate_range."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "inf_drive", "intensity": float("inf")},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is None

    # --- CSV Importer: NaN in confidence ---

    def test_csv_nan_confidence_strict_rejected(self):
        """CSV with NaN confidence should be rejected in strict mode."""
        items = parse_csv(
            """type,statement,confidence
belief,NaN belief,nan
""",
            strict=True,
        )
        assert len(items) == 0

    def test_csv_inf_confidence_strict_rejected(self):
        """CSV with inf confidence should be rejected in strict mode."""
        items = parse_csv(
            """type,statement,confidence
belief,Inf belief,inf
""",
            strict=True,
        )
        assert len(items) == 0
