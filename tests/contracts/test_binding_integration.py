"""Integration tests for the Binding and checkpoint save/restore system.

Tests the full lifecycle of creating an Entity + Stack composition,
checkpointing, and restoring from checkpoint. Uses real Stack instances
(not mocks) to verify the binding and checkpoint system works end-to-end.
"""

import json
from unittest.mock import MagicMock, PropertyMock

import pytest

from kernle.entity import Entity
from kernle.protocols import Binding, PluginHealth
from kernle.stack import Stack

CORE_ID = "binding-test-core"
STACK_ID = "binding-test-stack"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "kernle_data"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "binding_test.db"


@pytest.fixture
def entity(data_dir):
    return Entity(core_id=CORE_ID, data_dir=data_dir)


@pytest.fixture
def stack(db_path):
    return Stack.from_sqlite(stack_id=STACK_ID, db_path=db_path, enforce_provenance=False)


def _make_mock_plugin(name="test-plugin"):
    plugin = MagicMock()
    type(plugin).name = PropertyMock(return_value=name)
    type(plugin).version = PropertyMock(return_value="1.0.0")
    type(plugin).protocol_version = PropertyMock(return_value=1)
    type(plugin).description = PropertyMock(return_value=f"Plugin: {name}")
    plugin.capabilities.return_value = ["testing"]
    plugin.activate.return_value = None
    plugin.deactivate.return_value = None
    plugin.health_check.return_value = PluginHealth(healthy=True)
    plugin.on_load.return_value = None
    plugin.on_status.return_value = None
    plugin.register_cli.return_value = None
    plugin.register_tools.return_value = []
    return plugin


def _make_mock_model(model_id="test-model", provider="anthropic"):
    model = MagicMock()
    type(model).model_id = PropertyMock(return_value=model_id)
    capabilities = MagicMock()
    type(capabilities).provider = PropertyMock(return_value=provider)
    type(model).capabilities = PropertyMock(return_value=capabilities)
    return model


# ============================================================================
# 1. Full Roundtrip: Create, Save, Restore
# ============================================================================


class TestBindingRoundtrip:
    def test_basic_roundtrip(self, entity, stack):
        """Create Entity + Stack, checkpoint, restore, verify."""
        entity.attach_stack(stack, alias="primary")
        entity.episode("Test roundtrip", "It works")

        # Save checkpoint
        entity.checkpoint("roundtrip test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        assert len(files) == 1

        # Restore
        restored = Entity.from_checkpoint(files[0])
        assert restored.core_id == CORE_ID

    def test_roundtrip_preserves_core_id(self, entity, stack):
        entity.attach_stack(stack)
        entity.checkpoint("test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        restored = Entity.from_checkpoint(files[0])
        assert restored.core_id == entity.core_id

    def test_roundtrip_with_model(self, entity, stack):
        model = _make_mock_model("claude-test")
        entity.set_model(model)
        entity.attach_stack(stack, alias="main")

        entity.checkpoint("model test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        data = json.loads(files[0].read_text())

        assert data["binding"]["model_config"]["model_id"] == "claude-test"
        assert data["binding"]["model_config"]["provider"] == "anthropic"

    def test_roundtrip_with_multiple_stacks(self, entity, tmp_path):
        s1 = Stack.from_sqlite(stack_id="s1", db_path=tmp_path / "s1.db", enforce_provenance=False)
        s2 = Stack.from_sqlite(stack_id="s2", db_path=tmp_path / "s2.db", enforce_provenance=False)

        entity.attach_stack(s1, alias="alpha")
        entity.attach_stack(s2, alias="beta", set_active=False)

        entity.checkpoint("multi stack")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        data = json.loads(files[0].read_text())

        assert data["binding"]["stacks"]["s1"] == "alpha"
        assert data["binding"]["stacks"]["s2"] == "beta"
        assert data["binding"]["active_stack_id"] == "s1"

    def test_roundtrip_with_plugins(self, entity, stack):
        entity.attach_stack(stack, alias="main")
        entity.load_plugin(_make_mock_plugin("plugin-a"))
        entity.load_plugin(_make_mock_plugin("plugin-b"))

        entity.checkpoint("plugins test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        data = json.loads(files[0].read_text())

        assert "plugin-a" in data["binding"]["plugins"]
        assert "plugin-b" in data["binding"]["plugins"]

    def test_roundtrip_restores_stack_aliases(self, entity, stack):
        entity.attach_stack(stack, alias="main")
        entity.episode("Restore", "from checkpoint")
        entity.checkpoint("stack restore test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        restored = Entity.from_checkpoint(files[0])
        assert restored.core_id == CORE_ID
        assert STACK_ID in restored.stacks
        assert restored.active_stack is not None
        assert restored.active_stack.stack_id == STACK_ID

    def test_roundtrip_restores_plugins(self, entity, stack, monkeypatch):
        entity.attach_stack(stack, alias="main")
        entity.load_plugin(_make_mock_plugin("restored-plugin"))
        entity.checkpoint("plugin restore test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        class _RestoredPlugin:
            name = "restored-plugin"
            version = "1.0.0"
            protocol_version = 1
            description = "Restored test plugin"

            def __init__(self):
                self.activated = False

            def capabilities(self):
                return ["testing"]

            def activate(self, context):
                self.activated = True
                self.context = context

            def deactivate(self):
                self.activated = False

            def health_check(self):
                return PluginHealth(healthy=True, message="ok")

            def on_load(self):
                return None

            def on_status(self):
                return None

            def register_cli(self, _subparsers):
                return None

            def register_tools(self):
                return []

        component = MagicMock()
        component.name = "restored-plugin"

        monkeypatch.setattr(
            "kernle.discovery.discover_plugins",
            lambda: [component],
        )
        monkeypatch.setattr(
            "kernle.discovery.load_component",
            lambda _comp: _RestoredPlugin,
        )

        restored = Entity.from_checkpoint(files[0])
        assert "restored-plugin" in restored.plugins


# ============================================================================
# 2. Checkpoint File Format
# ============================================================================


class TestCheckpointFileFormat:
    def test_checkpoint_is_valid_json(self, entity, stack):
        entity.attach_stack(stack, alias="main")
        entity.checkpoint("format test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        content = files[0].read_text()
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_checkpoint_is_human_readable(self, entity, stack):
        """Checkpoint file should be indented / pretty-printed."""
        entity.attach_stack(stack, alias="main")
        entity.checkpoint("pretty test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        content = files[0].read_text()
        assert "\n" in content
        assert "  " in content

    def test_checkpoint_has_required_fields(self, entity, stack):
        entity.attach_stack(stack, alias="main")
        entity.checkpoint("fields test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        data = json.loads(files[0].read_text())
        required = ["schema_version", "checkpoint_id", "message", "binding", "created_at"]
        for field in required:
            assert field in data, f"Missing required field: {field}"

    def test_checkpoint_has_schema_version(self, entity, stack):
        entity.attach_stack(stack, alias="main")
        entity.checkpoint("schema test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        data = json.loads(files[0].read_text())
        assert data["schema_version"] == 1

    def test_checkpoint_binding_has_stacks(self, entity, stack):
        entity.attach_stack(stack, alias="memory")
        entity.checkpoint("stacks test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        data = json.loads(files[0].read_text())
        assert data["binding"]["stacks"][STACK_ID] == "memory"


# ============================================================================
# 3. Memories Survive Binding Roundtrip
# ============================================================================


class TestMemoriesSurviveBinding:
    """Save memories through Entity, checkpoint, verify stack still has them."""

    def test_memories_persist_after_checkpoint(self, entity, stack):
        entity.attach_stack(stack, alias="main")

        ep_id = entity.episode("Learn Rust", "Built CLI tool")
        b_id = entity.belief("Rust is safe")
        v_id = entity.value("Safety", "Memory safety matters")
        g_id = entity.goal("Ship v2")
        n_id = entity.note("Check performance")

        entity.checkpoint("persist test")

        episodes = stack.get_episodes()
        assert any(e.id == ep_id for e in episodes)
        beliefs = stack.get_beliefs()
        assert any(b.id == b_id for b in beliefs)
        values = stack.get_values()
        assert any(v.id == v_id for v in values)
        goals = stack.get_goals()
        assert any(g.id == g_id for g in goals)
        notes = stack.get_notes()
        assert any(n.id == n_id for n in notes)

    def test_stack_accessible_after_restore(self, entity, stack, db_path):
        """After checkpoint restore, stack data is still on disk and accessible."""
        entity.attach_stack(stack, alias="main")
        ep_id = entity.episode("Persist test", "Memory survives")

        entity.checkpoint("access test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        restored = Entity.from_checkpoint(files[0])
        assert restored.core_id == CORE_ID

        reopened_stack = Stack.from_sqlite(
            stack_id=STACK_ID, db_path=db_path, enforce_provenance=False
        )
        episodes = reopened_stack.get_episodes()
        assert any(e.id == ep_id for e in episodes)


# ============================================================================
# 4. Edge Cases and Error Handling
# ============================================================================


class TestCheckpointEdgeCases:
    def test_from_checkpoint_invalid_path(self, tmp_path):
        bad_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            Entity.from_checkpoint(bad_path)

    def test_from_checkpoint_corrupted_json(self, tmp_path):
        bad_path = tmp_path / "corrupt.json"
        bad_path.write_text("not valid json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            Entity.from_checkpoint(bad_path)

    def test_get_binding_detached_entity(self, entity):
        """Entity with no stacks can still produce a binding."""
        binding = entity.get_binding()
        assert binding.core_id == CORE_ID
        assert binding.stacks == {}
        assert binding.active_stack_id is None

    def test_binding_after_detach(self, entity, stack):
        entity.attach_stack(stack, alias="temp")
        entity.detach_stack(STACK_ID)

        binding = entity.get_binding()
        assert binding.stacks == {}
        assert binding.active_stack_id is None


# ============================================================================
# 5. Binding Object (Dataclass)
# ============================================================================


class TestBindingDataclass:
    def test_binding_fields(self):
        b = Binding(
            core_id="test",
            model_config={"model_id": "m1"},
            stacks={"s1": "main"},
            active_stack_id="s1",
            plugins=["p1"],
        )
        assert b.core_id == "test"
        assert b.model_config == {"model_id": "m1"}
        assert b.stacks == {"s1": "main"}
        assert b.active_stack_id == "s1"
        assert b.plugins == ["p1"]

    def test_binding_defaults(self):
        b = Binding(core_id="test", model_config={}, stacks={})
        assert b.active_stack_id is None
        assert b.plugins == []
        assert b.created_at is None
        assert b.metadata == {}


# ============================================================================
# 6. Binding Format (stack_id-keyed)
# ============================================================================


class TestBindingFormat:
    """Verify stack_id-keyed binding format loads correctly."""

    def test_stack_id_keyed_format_loads(self):
        """Binding with {stack_id -> alias, active_stack_id} loads correctly."""
        binding = Binding(
            core_id="format-core",
            model_config={},
            stacks={"stack-1": "main", "stack-2": None},
            active_stack_id="stack-1",
            plugins=[],
        )

        restored = Entity.from_binding(binding)
        assert restored.core_id == "format-core"
        assert restored._restored_binding.stacks == {
            "stack-1": "main",
            "stack-2": None,
        }
        assert restored._restored_binding.active_stack_id == "stack-1"
