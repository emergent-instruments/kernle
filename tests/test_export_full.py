"""Tests for export_full functionality."""

import json
from pathlib import Path

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance with isolated temp DB."""
    db_path = tmp_path / "test_export_full.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    storage = SQLiteStorage(stack_id="test-export-full", db_path=db_path)
    instance = Kernle(
        stack_id="test-export-full", storage=storage, checkpoint_dir=checkpoint_dir, strict=False
    )
    bind_noop_model(instance)
    yield instance
    storage.close()


def _populate_all_layers(k):
    """Helper to populate all memory layers for testing."""
    k.value("honesty", "Always tell the truth", priority=90)
    k.value("curiosity", "Stay curious", priority=70)
    k.belief("The sky is blue", confidence=0.9)
    k.belief("Water is wet", confidence=0.8)
    k.episode(
        "Deploy API",
        "Successfully deployed v2",
        lessons=["Always run tests first"],
        tags=["deploy"],
    )
    k.note("Remember to check logs", type="insight")
    k.drive("curiosity", intensity=0.8, focus_areas=["python", "rust"])
    k.relationship("Alice", trust_level=0.8, notes="Collaborator", entity_type="person")
    k.narrative_save("I am a helpful agent", narrative_type="identity")
    k.trust_set("operator", domain="general", score=0.9)
    k.boot_set("preferred_model", "claude-opus-4-6")
    k.playbook(
        "Deploy to prod",
        "Standard deployment procedure",
        steps=["Run tests", "Build", "Deploy"],
        triggers=["deployment requested"],
        failure_modes=["Tests fail"],
    )
    k.checkpoint(task="testing export", context="running tests", pending=["finish tests"])


class TestExportFullMarkdown:
    def test_empty_stack_produces_valid_output(self, k):
        """Empty stack produces valid markdown with header."""
        content = k.export_full()
        assert "# Full Agent Context" in content
        assert "test-export-full" in content
        assert "Exported at" in content

    def test_all_sections_present(self, k):
        """All memory layer sections appear in output."""
        _populate_all_layers(k)
        content = k.export_full()

        assert "## Boot Config" in content
        assert "## Self-Narratives" in content
        assert "## Values" in content
        assert "## Beliefs" in content
        assert "## Goals" not in content  # no goals were added
        assert "## Drives" in content
        assert "## Relationships" in content
        assert "## Episodes" in content
        assert "## Notes" in content
        assert "## Trust Assessments" in content
        assert "## Playbooks" in content
        assert "## Checkpoint" in content

    def test_boot_config_content(self, k):
        """Boot config values appear in markdown output."""
        k.boot_set("preferred_model", "claude-opus-4-6")
        content = k.export_full()
        assert "preferred_model" in content
        assert "claude-opus-4-6" in content

    def test_values_content(self, k):
        """Values appear with name, priority, and statement."""
        k.value("honesty", "Always tell the truth", priority=90)
        content = k.export_full()
        assert "honesty" in content
        assert "Always tell the truth" in content

    def test_beliefs_content(self, k):
        """Beliefs appear with confidence."""
        k.belief("The sky is blue", confidence=0.9)
        content = k.export_full()
        assert "sky is blue" in content
        assert "90%" in content

    def test_episodes_content(self, k):
        """Episodes appear with objective and outcome."""
        k.episode("Deploy API", "Successfully deployed", lessons=["Test first"])
        content = k.export_full()
        assert "Deploy API" in content
        assert "Successfully deployed" in content
        assert "Test first" in content

    def test_notes_content(self, k):
        """Notes appear with content."""
        k.note("Check logs daily", type="insight")
        content = k.export_full()
        assert "Check logs daily" in content

    def test_drives_content(self, k):
        """Drives appear with type and intensity."""
        k.drive("curiosity", intensity=0.8, focus_areas=["python"])
        content = k.export_full()
        assert "curiosity" in content
        assert "80%" in content

    def test_relationships_content(self, k):
        """Relationships appear with entity name."""
        k.relationship("Alice", trust_level=0.8, entity_type="person")
        content = k.export_full()
        assert "Alice" in content

    def test_narratives_content(self, k):
        """Self-narratives appear with type and content."""
        k.narrative_save("I strive to be helpful", narrative_type="identity")
        content = k.export_full()
        assert "Identity" in content
        assert "I strive to be helpful" in content

    def test_trust_assessments_content(self, k):
        """Trust assessments appear with entity and dimensions."""
        k.trust_set("operator", domain="general", score=0.9)
        content = k.export_full()
        assert "operator" in content

    def test_playbooks_content(self, k):
        """Playbooks appear with name and steps."""
        k.playbook(
            "Deploy",
            "Standard deploy",
            steps=["Test", "Build", "Ship"],
            triggers=["deploy requested"],
        )
        content = k.export_full()
        assert "Deploy" in content
        assert "Standard deploy" in content
        assert "Test" in content

    def test_checkpoint_content(self, k):
        """Checkpoint appears with task info."""
        k.checkpoint(
            task="testing",
            context="unit tests",
            pending=["more tests"],
        )
        content = k.export_full()
        assert "testing" in content
        assert "unit tests" in content

    def test_no_raw_by_default(self, k):
        """Raw entries excluded by default."""
        k.raw("some raw input")
        content = k.export_full()
        assert "## Raw Entries" not in content

    def test_include_raw_flag(self, k):
        """Raw entries included when flag is set."""
        k.raw("some raw input")
        content = k.export_full(include_raw=True)
        assert "## Raw Entries" in content
        assert "some raw input" in content

    def test_goals_section_when_present(self, k):
        """Goals section appears when goals exist."""
        k.goal("Ship v1.0", description="Release first version", priority="high")
        content = k.export_full()
        assert "## Goals" in content
        assert "Ship v1.0" in content


class TestExportFullJSON:
    def test_empty_stack_produces_valid_json(self, k):
        """Empty stack produces valid JSON with metadata."""
        content = k.export_full(format="json")
        data = json.loads(content)
        assert data["stack_id"] == "test-export-full"
        assert "exported_at" in data
        assert data["format"] == "export-full"

    def test_all_keys_present(self, k):
        """All expected top-level keys are present in JSON output."""
        _populate_all_layers(k)
        content = k.export_full(format="json")
        data = json.loads(content)

        expected_keys = [
            "stack_id",
            "exported_at",
            "format",
            "boot_config",
            "self_narratives",
            "values",
            "beliefs",
            "goals",
            "drives",
            "relationships",
            "episodes",
            "notes",
            "trust_assessments",
            "playbooks",
            "checkpoint",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_no_raw_entries_by_default(self, k):
        """Raw entries key absent by default."""
        k.raw("some input")
        content = k.export_full(format="json")
        data = json.loads(content)
        assert "raw_entries" not in data

    def test_include_raw_adds_key(self, k):
        """Raw entries key present when include_raw=True."""
        k.raw("some input")
        content = k.export_full(format="json", include_raw=True)
        data = json.loads(content)
        assert "raw_entries" in data
        assert len(data["raw_entries"]) >= 1

    def test_values_have_provenance_fields(self, k):
        """Values include provenance metadata in JSON."""
        k.value("honesty", "Always tell the truth", priority=90)
        content = k.export_full(format="json")
        data = json.loads(content)
        v = data["values"][0]
        assert "id" in v
        assert "source_type" in v
        assert "confidence" in v
        assert "created_at" in v

    def test_beliefs_have_provenance_fields(self, k):
        """Beliefs include provenance metadata in JSON."""
        k.belief("Sky is blue", confidence=0.9, derived_from=["episode:abc123"])
        content = k.export_full(format="json")
        data = json.loads(content)
        b = data["beliefs"][0]
        assert "id" in b
        assert "derived_from" in b
        assert "source_type" in b
        assert "supersedes" in b
        assert "is_active" in b

    def test_narratives_have_full_fields(self, k):
        """Self-narratives include all fields in JSON."""
        k.narrative_save(
            "I am helpful",
            narrative_type="identity",
            key_themes=["helpfulness"],
            unresolved_tensions=["autonomy vs compliance"],
        )
        content = k.export_full(format="json")
        data = json.loads(content)
        n = data["self_narratives"][0]
        assert n["content"] == "I am helpful"
        assert n["narrative_type"] == "identity"
        assert n["key_themes"] == ["helpfulness"]
        assert n["unresolved_tensions"] == ["autonomy vs compliance"]
        assert n["is_active"] is True

    def test_trust_assessments_have_full_fields(self, k):
        """Trust assessments include all fields in JSON."""
        k.trust_set("operator", domain="general", score=0.9)
        content = k.export_full(format="json")
        data = json.loads(content)
        # Find the operator assessment (stack may bootstrap an "identity" assessment)
        a = next(t for t in data["trust_assessments"] if t["entity"] == "operator")
        assert a["entity"] == "operator"
        assert "dimensions" in a
        assert "general" in a["dimensions"]

    def test_playbooks_have_full_fields(self, k):
        """Playbooks include all fields in JSON."""
        k.playbook(
            "Deploy",
            "Standard deploy",
            steps=["Test", "Build"],
            triggers=["deploy time"],
            failure_modes=["tests fail"],
        )
        content = k.export_full(format="json")
        data = json.loads(content)
        p = data["playbooks"][0]
        assert p["name"] == "Deploy"
        assert p["description"] == "Standard deploy"
        assert len(p["steps"]) == 2
        assert p["trigger_conditions"] == ["deploy time"]
        assert p["failure_modes"] == ["tests fail"]
        assert "mastery_level" in p
        assert "confidence" in p

    def test_boot_config_in_json(self, k):
        """Boot config appears as dict in JSON."""
        k.boot_set("model", "opus")
        k.boot_set("mode", "strict")
        content = k.export_full(format="json")
        data = json.loads(content)
        assert data["boot_config"]["model"] == "opus"
        assert data["boot_config"]["mode"] == "strict"

    def test_checkpoint_in_json(self, k):
        """Checkpoint appears in JSON output."""
        k.checkpoint(task="testing", context="test context")
        content = k.export_full(format="json")
        data = json.loads(content)
        assert data["checkpoint"]["current_task"] == "testing"


class TestExportFullFile:
    def test_write_to_markdown_file(self, k, tmp_path):
        """export_full writes markdown file when path provided."""
        _populate_all_layers(k)
        out_path = str(tmp_path / "context.md")
        content = k.export_full(path=out_path)

        assert Path(out_path).exists()
        file_content = Path(out_path).read_text()
        assert file_content == content
        assert "# Full Agent Context" in file_content

    def test_write_to_json_file(self, k, tmp_path):
        """export_full writes JSON file when path provided."""
        _populate_all_layers(k)
        out_path = str(tmp_path / "context.json")
        k.export_full(path=out_path, format="json")

        assert Path(out_path).exists()
        file_content = Path(out_path).read_text()
        data = json.loads(file_content)
        assert data["stack_id"] == "test-export-full"

    def test_auto_detect_json_from_extension(self, k, tmp_path):
        """Writing to .json file auto-detects JSON format."""
        k.value("test", "test value")
        out_path = str(tmp_path / "output.json")
        k.export_full(path=out_path)

        file_content = Path(out_path).read_text()
        # Should be valid JSON despite format defaulting to markdown
        data = json.loads(file_content)
        assert "values" in data

    def test_creates_parent_dirs(self, k, tmp_path):
        """export_full creates parent directories as needed."""
        out_path = str(tmp_path / "subdir" / "deep" / "context.md")
        k.export_full(path=out_path)
        assert Path(out_path).exists()

    def test_empty_stack_file_write(self, k, tmp_path):
        """Empty stack can be written to file without errors."""
        out_path = str(tmp_path / "empty.md")
        k.export_full(path=out_path)
        assert Path(out_path).exists()
        content = Path(out_path).read_text()
        assert "# Full Agent Context" in content
