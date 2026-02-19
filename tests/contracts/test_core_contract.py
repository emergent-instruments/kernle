"""Contract tests for CoreProtocol.

Verifies that Entity conforms to the CoreProtocol contract using
real Stack instances for stack operations. Tests cover:
- Stack attach/detach/set_active
- Routed operations reach the stack
- NoActiveStackError when no stack
- Provenance populated on routed writes
- Plugin lifecycle with mock plugins
- Status assembly
- Checkpoint save/restore

Designed to be reusable: future core implementations can run the
same contract suite by substituting fixtures.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

import pytest

from kernle.entity import Entity
from kernle.protocols import (
    Binding,
    NoActiveStackError,
    PluginHealth,
    PluginInfo,
    StackInfo,
)
from kernle.stack import Stack

CORE_ID = "contract-test-core"
STACK_ID = "contract-test-stack"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "kernle_data"


@pytest.fixture
def entity(data_dir):
    return Entity(core_id=CORE_ID, data_dir=data_dir)


@pytest.fixture
def stack(tmp_path):
    db_path = tmp_path / "contract_core_test.db"
    return Stack.from_sqlite(stack_id=STACK_ID, db_path=db_path, enforce_provenance=False)


@pytest.fixture
def entity_with_stack(entity, stack):
    """Entity with a stack already attached and active."""
    entity.attach_stack(stack, alias="main")
    return entity, stack


def _make_mock_plugin(name="test-plugin", version="1.0.0"):
    """Create a mock PluginProtocol."""
    plugin = MagicMock()
    type(plugin).name = PropertyMock(return_value=name)
    type(plugin).version = PropertyMock(return_value=version)
    type(plugin).protocol_version = PropertyMock(return_value=1)
    type(plugin).description = PropertyMock(return_value=f"Test plugin: {name}")
    plugin.capabilities.return_value = ["testing"]
    plugin.activate.return_value = None
    plugin.deactivate.return_value = None
    plugin.health_check.return_value = PluginHealth(healthy=True, message="ok")
    plugin.on_load.return_value = None
    plugin.on_status.return_value = None
    plugin.register_cli.return_value = None
    plugin.register_tools.return_value = []
    return plugin


def _make_mock_model(model_id="test-model", provider="anthropic"):
    """Create a mock ModelProtocol."""
    model = MagicMock()
    type(model).model_id = PropertyMock(return_value=model_id)
    capabilities = MagicMock()
    type(capabilities).provider = PropertyMock(return_value=provider)
    type(model).capabilities = PropertyMock(return_value=capabilities)
    return model


# ============================================================================
# 1. Stack Attach / Detach / Set Active
# ============================================================================


class TestStackManagement:
    def test_attach_stack_returns_stack_id(self, entity, stack):
        result = entity.attach_stack(stack, alias="primary")
        assert result == STACK_ID
        assert entity.active_stack is stack

    def test_attach_uses_stack_id_as_key(self, entity, stack):
        result = entity.attach_stack(stack)
        assert result == STACK_ID

    def test_attach_calls_on_attach(self, entity, stack):
        entity.attach_stack(stack)
        assert stack._attached_core_id == CORE_ID

    def test_detach_stack(self, entity, stack):
        entity.attach_stack(stack, alias="temp")
        entity.detach_stack(STACK_ID)

        assert entity.active_stack is None
        assert stack._attached_core_id is None

    def test_detach_nonexistent_is_safe(self, entity):
        entity.detach_stack("nonexistent")  # Should not raise

    def test_set_active_stack(self, entity, tmp_path):
        db1 = tmp_path / "stack1.db"
        db2 = tmp_path / "stack2.db"
        s1 = Stack.from_sqlite(stack_id="s1", db_path=db1, enforce_provenance=False)
        s2 = Stack.from_sqlite(stack_id="s2", db_path=db2, enforce_provenance=False)

        entity.attach_stack(s1, alias="first", set_active=True)
        entity.attach_stack(s2, alias="second", set_active=False)

        assert entity.active_stack is s1

        entity.set_active_stack("s2")
        assert entity.active_stack is s2

    def test_set_active_missing_raises(self, entity):
        with pytest.raises(ValueError, match="No stack with stack_id"):
            entity.set_active_stack("missing")

    def test_stacks_property(self, entity, stack):
        entity.attach_stack(stack, alias="main")

        stacks = entity.stacks
        assert STACK_ID in stacks
        info = stacks[STACK_ID]
        assert isinstance(info, StackInfo)
        assert info.stack_id == STACK_ID
        assert info.alias == "main"
        assert info.is_active is True

    def test_multiple_stacks(self, entity, tmp_path):
        db1 = tmp_path / "s1.db"
        db2 = tmp_path / "s2.db"
        s1 = Stack.from_sqlite(stack_id="s1", db_path=db1, enforce_provenance=False)
        s2 = Stack.from_sqlite(stack_id="s2", db_path=db2, enforce_provenance=False)

        entity.attach_stack(s1, alias="alpha")
        entity.attach_stack(s2, alias="beta", set_active=False)

        stacks = entity.stacks
        assert len(stacks) == 2
        assert stacks["s1"].is_active is True
        assert stacks["s2"].is_active is False


# ============================================================================
# 2. Routed Ops Reach Stack
# ============================================================================


class TestRoutedOps:
    """Write through Entity, verify data in stack."""

    def test_episode_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        ep_id = entity.episode("Learn testing", "Tests passed")
        assert isinstance(ep_id, str)

        episodes = stack.get_episodes()
        found = [e for e in episodes if e.id == ep_id]
        assert len(found) == 1
        assert found[0].objective == "Learn testing"

    def test_belief_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        b_id = entity.belief("Code review helps")
        assert isinstance(b_id, str)

        beliefs = stack.get_beliefs()
        found = [b for b in beliefs if b.id == b_id]
        assert len(found) == 1
        assert found[0].statement == "Code review helps"

    def test_value_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        v_id = entity.value("Quality", "Ship reliable software", priority=80)
        assert isinstance(v_id, str)

        values = stack.get_values()
        found = [v for v in values if v.id == v_id]
        assert len(found) == 1
        assert found[0].name == "Quality"

    def test_goal_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        g_id = entity.goal("Release v1", description="Ship it")
        assert isinstance(g_id, str)

        goals = stack.get_goals()
        found = [g for g in goals if g.id == g_id]
        assert len(found) == 1
        assert found[0].title == "Release v1"

    def test_note_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        n_id = entity.note("Important insight", type="insight")
        assert isinstance(n_id, str)

        notes = stack.get_notes()
        found = [n for n in notes if n.id == n_id]
        assert len(found) == 1
        assert found[0].content == "**Insight**: Important insight"

    def test_drive_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        d_id = entity.drive("curiosity", intensity=0.8)
        assert isinstance(d_id, str)

        drives = stack.get_drives()
        found = [d for d in drives if d.id == d_id]
        assert len(found) == 1
        assert found[0].drive_type == "curiosity"

    def test_relationship_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        r_id = entity.relationship("partner-1", entity_type="agent", notes="Trusted")
        assert isinstance(r_id, str)

        rels = stack.get_relationships()
        found = [r for r in rels if r.id == r_id]
        assert len(found) == 1
        assert found[0].entity_name == "partner-1"

    def test_raw_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        r_id = entity.raw("Brain dump text")
        assert isinstance(r_id, str)

        raw = stack.get_raw()
        assert len(raw) >= 1

    def test_search_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        entity.note("Rust programming guide")

        results = entity.search("Rust")
        assert isinstance(results, list)

    def test_load_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        entity.value("Honesty", "Always be truthful")

        result = entity.load()
        assert isinstance(result, dict)
        assert "values" in result

    def test_trust_set_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        ta_id = entity.trust_set("bob", "general", 0.9)
        assert isinstance(ta_id, str)

        assessments = stack.get_trust_assessments(entity_id="bob")
        assert len(assessments) >= 1

    def test_trust_get_routed(self, entity_with_stack):
        entity, stack = entity_with_stack
        entity.trust_set("alice", "code", 0.85)

        assessments = entity.trust_get("alice")
        assert len(assessments) >= 1


# ============================================================================
# 3. NoActiveStackError
# ============================================================================


class TestNoActiveStack:
    def test_episode_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.episode("test", "test")

    def test_belief_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.belief("test")

    def test_value_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.value("test", "test")

    def test_goal_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.goal("test")

    def test_note_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.note("test")

    def test_drive_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.drive("test")

    def test_relationship_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.relationship("test")

    def test_raw_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.raw("test")

    def test_search_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.search("test")

    def test_load_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.load()

    def test_sync_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.sync()

    def test_trust_set_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.trust_set("e", "d", 0.5)

    def test_trust_get_no_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.trust_get("e")


# ============================================================================
# 4. Provenance on Routed Writes
# ============================================================================


class TestProvenance:
    def test_episode_has_source_entity(self, entity_with_stack):
        entity, stack = entity_with_stack
        ep_id = entity.episode("Test provenance", "It worked")

        episodes = stack.get_episodes()
        ep = [e for e in episodes if e.id == ep_id][0]
        assert ep.source_entity == f"core:{CORE_ID}"
        assert ep.source_type == "direct_experience"

    def test_episode_custom_source(self, entity_with_stack):
        entity, stack = entity_with_stack
        ep_id = entity.episode("Test", "Ok", source="user:alice")

        episodes = stack.get_episodes()
        ep = [e for e in episodes if e.id == ep_id][0]
        assert ep.source_entity == "user:alice"

    def test_belief_has_source_entity(self, entity_with_stack):
        entity, stack = entity_with_stack
        b_id = entity.belief("Provenance works")

        beliefs = stack.get_beliefs()
        b = [x for x in beliefs if x.id == b_id][0]
        assert b.source_entity == f"core:{CORE_ID}"

    def test_note_has_source_entity(self, entity_with_stack):
        entity, stack = entity_with_stack
        n_id = entity.note("Provenance test")

        notes = stack.get_notes()
        n = [x for x in notes if x.id == n_id][0]
        assert n.source_entity == f"core:{CORE_ID}"

    def test_episode_has_created_at(self, entity_with_stack):
        entity, stack = entity_with_stack
        before = datetime.now(timezone.utc)
        ep_id = entity.episode("Timestamp test", "Done")

        episodes = stack.get_episodes()
        ep = [e for e in episodes if e.id == ep_id][0]
        assert ep.created_at is not None
        assert ep.created_at >= before

    def test_context_tags_pass_through(self, entity_with_stack):
        entity, stack = entity_with_stack
        ep_id = entity.episode(
            "Tagged ep",
            "Done",
            context="project:test",
            context_tags=["ci", "pipeline"],
        )

        episodes = stack.get_episodes()
        ep = [e for e in episodes if e.id == ep_id][0]
        assert ep.context == "project:test"
        assert ep.context_tags == ["ci", "pipeline"]

    def test_derived_from_pass_through(self, entity_with_stack):
        entity, stack = entity_with_stack
        b_id = entity.belief(
            "Derived belief",
            derived_from=["episode:abc", "belief:xyz"],
        )

        beliefs = stack.get_beliefs()
        b = [x for x in beliefs if x.id == b_id][0]
        assert b.derived_from == ["episode:abc", "belief:xyz"]

    def test_episode_explicit_source_type(self, entity_with_stack):
        """Explicit source_type parameter overrides default."""
        entity, stack = entity_with_stack
        ep_id = entity.episode(
            "External observation",
            "Noted from partner",
            source_type="external",
        )

        episodes = stack.get_episodes()
        ep = [e for e in episodes if e.id == ep_id][0]
        assert ep.source_type == "external"

    def test_note_explicit_source_type(self, entity_with_stack):
        """Note with explicit source_type stores correctly."""
        entity, stack = entity_with_stack
        n_id = entity.note("Observation note", source_type="observation")

        notes = stack.get_notes()
        n = [x for x in notes if x.id == n_id][0]
        assert n.source_type == "observation"

    def test_relationship_has_source_entity(self, entity_with_stack):
        """Relationship created through Entity has source_entity set."""
        entity, stack = entity_with_stack
        r_id = entity.relationship("partner-test", entity_type="agent")

        rels = stack.get_relationships()
        r = [x for x in rels if x.id == r_id][0]
        assert r.source_type == "direct_experience"


# ============================================================================
# 5. Plugin Lifecycle
# ============================================================================


class TestPluginLifecycle:
    def test_load_plugin(self, entity):
        plugin = _make_mock_plugin("my-plugin")
        entity.load_plugin(plugin)

        assert "my-plugin" in entity.plugins
        plugin.activate.assert_called_once()

    def test_plugin_context_passed(self, entity):
        plugin = _make_mock_plugin("ctx-plugin")
        entity.load_plugin(plugin)

        call_args = plugin.activate.call_args
        ctx = call_args[0][0]
        assert ctx.core_id == CORE_ID
        assert ctx.plugin_name == "ctx-plugin"

    def test_unload_plugin(self, entity):
        plugin = _make_mock_plugin("temp-plugin")
        entity.load_plugin(plugin)
        entity.unload_plugin("temp-plugin")

        assert "temp-plugin" not in entity.plugins
        plugin.deactivate.assert_called_once()

    def test_unload_nonexistent_is_safe(self, entity):
        entity.unload_plugin("nonexistent")  # Should not raise

    def test_plugins_property(self, entity):
        plugin = _make_mock_plugin("info-plugin")
        entity.load_plugin(plugin)

        plugins = entity.plugins
        assert "info-plugin" in plugins
        info = plugins["info-plugin"]
        assert isinstance(info, PluginInfo)
        assert info.is_loaded is True

    def test_plugin_writes_through_context(self, entity_with_stack):
        entity, stack = entity_with_stack
        plugin = _make_mock_plugin("writer-plugin")

        def capture_activate(ctx):
            ctx.note("From plugin")

        plugin.activate.side_effect = capture_activate
        entity.load_plugin(plugin)

        notes = stack.get_notes()
        found = [n for n in notes if n.content == "From plugin"]
        assert len(found) == 1
        assert found[0].source_entity == "plugin:writer-plugin"

    def test_plugin_context_returns_none_without_stack(self, entity):
        plugin = _make_mock_plugin("stackless-plugin")
        results = {}

        def capture_activate(ctx):
            results["ep"] = ctx.episode("Test", "Outcome")
            results["search"] = ctx.search("query")
            results["rels"] = ctx.get_relationships()
            results["goals"] = ctx.get_goals()

        plugin.activate.side_effect = capture_activate
        entity.load_plugin(plugin)

        assert results["ep"] is None
        assert results["search"] == []
        assert results["rels"] == []
        assert results["goals"] == []


# ============================================================================
# 6. Status Assembly
# ============================================================================


class TestStatus:
    def test_status_shape(self, entity_with_stack):
        entity, stack = entity_with_stack

        status = entity.status()
        assert status["core_id"] == CORE_ID
        assert "stacks" in status
        assert "plugins" in status
        assert "model" in status

    def test_status_includes_stack_stats(self, entity_with_stack):
        entity, stack = entity_with_stack
        entity.episode("Status test", "Done")

        status = entity.status()
        assert STACK_ID in status["stacks"]
        stack_info = status["stacks"][STACK_ID]
        assert "stats" in stack_info
        assert stack_info["stats"].get("episodes", 0) >= 1

    def test_status_includes_model(self, entity):
        model = _make_mock_model("claude-test")
        entity.set_model(model)

        status = entity.status()
        assert status["model"] == "claude-test"

    def test_status_no_model(self, entity):
        status = entity.status()
        assert status["model"] is None

    def test_status_includes_plugin_health(self, entity_with_stack):
        entity, stack = entity_with_stack
        plugin = _make_mock_plugin("health-plugin")
        entity.load_plugin(plugin)

        status = entity.status()
        assert "health-plugin" in status["plugins"]
        assert status["plugins"]["health-plugin"]["health"]["healthy"] is True

    def test_status_plugin_on_status_called(self, entity_with_stack):
        entity, stack = entity_with_stack
        plugin = _make_mock_plugin("status-plugin")
        entity.load_plugin(plugin)

        entity.status()
        plugin.on_status.assert_called_once()


# ============================================================================
# 7. Model Binding
# ============================================================================


class TestModelBinding:
    def test_set_model(self, entity):
        model = _make_mock_model("test-model")
        entity.set_model(model)
        assert entity.model is model
        assert entity.model.model_id == "test-model"

    def test_model_initially_none(self, entity):
        assert entity.model is None

    def test_set_model_notifies_stacks(self, entity, stack):
        entity.attach_stack(stack, alias="main")
        model = _make_mock_model("new-model")
        entity.set_model(model)

        # Stack should have been notified via on_model_changed
        # with a real InferenceService wrapping the model (v0.5.0)
        from kernle.protocols import InferenceService

        assert stack._inference is not None
        assert isinstance(stack._inference, InferenceService)


# ============================================================================
# 8. Checkpoint Save/Restore
# ============================================================================


class TestCheckpointSaveRestore:
    def test_get_binding(self, entity_with_stack):
        entity, stack = entity_with_stack
        model = _make_mock_model("binding-model")
        entity.set_model(model)

        binding = entity.get_binding()
        assert isinstance(binding, Binding)
        assert binding.core_id == CORE_ID
        assert STACK_ID in binding.stacks
        assert binding.stacks[STACK_ID] == "main"
        assert binding.active_stack_id == STACK_ID
        assert binding.model_config.get("model_id") == "binding-model"
        assert binding.model_config.get("provider") == "anthropic"

    def test_checkpoint_creates_file(self, entity_with_stack, data_dir):
        entity, stack = entity_with_stack
        cp_id = entity.checkpoint("test")

        assert isinstance(cp_id, str)
        cp_dir = data_dir / "checkpoints"
        assert cp_dir.exists()
        files = list(cp_dir.glob("*.json"))
        assert len(files) >= 1

    def test_checkpoint_is_valid_json(self, entity_with_stack):
        entity, stack = entity_with_stack
        entity.checkpoint("json test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        data = json.loads(files[0].read_text())
        assert isinstance(data, dict)
        assert all(k in data for k in ["schema_version", "checkpoint_id", "binding"])

    def test_binding_includes_plugins(self, entity_with_stack):
        entity, stack = entity_with_stack
        plugin = _make_mock_plugin("bind-plugin")
        entity.load_plugin(plugin)

        binding = entity.get_binding()
        assert "bind-plugin" in binding.plugins

    def test_from_binding_object(self, entity_with_stack):
        entity, stack = entity_with_stack
        binding = entity.get_binding()

        restored = Entity.from_binding(binding)
        assert restored.core_id == CORE_ID

    def test_from_checkpoint(self, entity_with_stack):
        entity, stack = entity_with_stack
        entity.checkpoint("restore test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))

        restored = Entity.from_checkpoint(files[0])
        assert restored.core_id == CORE_ID

    def test_checkpoint_roundtrip_preserves_data(self, entity_with_stack):
        entity, stack = entity_with_stack
        model = _make_mock_model("roundtrip-model")
        entity.set_model(model)
        plugin = _make_mock_plugin("roundtrip-plugin")
        entity.load_plugin(plugin)

        entity.checkpoint("roundtrip test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        data = json.loads(files[0].read_text())

        assert data["binding"]["core_id"] == CORE_ID
        assert data["binding"]["stacks"][STACK_ID] == "main"
        assert data["binding"]["active_stack_id"] == STACK_ID
        assert data["binding"]["model_config"]["model_id"] == "roundtrip-model"
        assert data["binding"]["model_config"]["provider"] == "anthropic"
        assert "roundtrip-plugin" in data["binding"]["plugins"]


# ============================================================================
# 9. Core Properties
# ============================================================================


class TestCoreProperties:
    def test_core_id(self, entity):
        assert entity.core_id == CORE_ID

    def test_active_stack_none_initially(self, entity):
        assert entity.active_stack is None

    def test_stacks_empty_initially(self, entity):
        assert entity.stacks == {}

    def test_plugins_empty_initially(self, entity):
        assert entity.plugins == {}


# ============================================================================
# 10. Checkpoint
# ============================================================================


class TestCheckpoint:
    def test_checkpoint_creates_file(self, entity_with_stack, data_dir):
        entity, stack = entity_with_stack
        cp_id = entity.checkpoint("test checkpoint")

        assert isinstance(cp_id, str)
        cp_dir = data_dir / "checkpoints"
        assert cp_dir.exists()
        files = list(cp_dir.glob("*.json"))
        assert len(files) >= 1

    def test_checkpoint_no_stack_raises(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.checkpoint("no stack")


class TestRepeatAvoidRoundtrip:
    def test_repeat_avoid_roundtrip(self, entity_with_stack):
        entity, stack = entity_with_stack
        ep_id = entity.episode(
            "roundtrip obj",
            "roundtrip out",
            repeat=["good pattern"],
            avoid=["bad pattern"],
        )
        episodes = stack.get_episodes()
        found = [e for e in episodes if e.id == ep_id]
        assert len(found) == 1
        assert found[0].repeat == ["good pattern"]
        assert found[0].avoid == ["bad pattern"]


# ============================================================================
# 11. Public API Conformance
# ============================================================================


class TestPublicAPIConformance:
    """Verify public API exports and protocol conformance."""

    def test_entity_exported_from_package(self):
        """Entity should be importable from the kernle package."""
        from kernle import Entity

        assert Entity is not None

    def test_entity_satisfies_core_protocol(self):
        """Entity should satisfy CoreProtocol."""
        from kernle import Entity
        from kernle.protocols import CoreProtocol

        entity = Entity(core_id="api-test")
        assert isinstance(entity, CoreProtocol)

    def test_version_matches_pyproject(self):
        """Package version should match importlib.metadata."""
        from importlib.metadata import version

        import kernle

        expected = version("kernle")
        assert kernle.__version__ == expected
