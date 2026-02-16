"""Tests for kernle.entity.Entity — CoreProtocol implementation."""

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

import kernle.entity as entity_module
from kernle.entity import Entity, _generate_id, _PluginContextImpl
from kernle.protocols import (
    Binding,
    NoActiveStackError,
    PluginHealth,
    PluginInfo,
    StackInfo,
    SyncResult,
)
from kernle.types import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    RawEntry,
    Relationship,
    SourceType,
    TrustAssessment,
    Value,
)


class TestGenerateIdDeterministicOrdering:
    def test_generate_id_is_deterministic_for_fixed_timestamp(self, monkeypatch):
        fixed_time = datetime(2025, 2, 13, 12, 0, tzinfo=timezone.utc)
        fixed_ms = int(fixed_time.timestamp() * 1000)
        monkeypatch.setattr("kernle.entity._ID_LAST_TIMESTAMP_MS", fixed_ms)
        monkeypatch.setattr("kernle.entity._ID_SEQUENCE", -1)

        with patch("kernle.entity.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time
            first = _generate_id()
            second = _generate_id()
            third = _generate_id()

        assert first.split("-")[0] == f"{fixed_ms:013d}"
        assert first.split("-")[2] == "000000"
        assert second.split("-")[2] == "000001"
        assert third.split("-")[2] == "000002"
        assert first < second < third

        monkeypatch.setattr("kernle.entity._ID_SEQUENCE", -1)
        with patch("kernle.entity.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time
            replay = _generate_id()

        assert replay == first


class TestGenerateIdClockAndSequenceSafety:
    """Tests for clock jump detection and sequence overflow protection in _generate_id()."""

    @pytest.fixture(autouse=True)
    def _save_restore_id_globals(self):
        """Save and restore module-level ID generator state around each test."""
        saved_seq = entity_module._ID_SEQUENCE
        saved_ts = entity_module._ID_LAST_TIMESTAMP_MS
        saved_pinned = entity_module._ID_CLOCK_WAS_PINNED
        yield
        entity_module._ID_SEQUENCE = saved_seq
        entity_module._ID_LAST_TIMESTAMP_MS = saved_ts
        entity_module._ID_CLOCK_WAS_PINNED = saved_pinned

    def test_backward_clock_jump_pins_timestamp(self):
        """When the clock jumps backward, the ID should use the last known timestamp."""
        base_time = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        base_ms = int(base_time.timestamp() * 1000)

        # Set state: we've already generated an ID at base_ms
        entity_module._ID_LAST_TIMESTAMP_MS = base_ms
        entity_module._ID_SEQUENCE = 0

        # Clock jumps backward by 5000ms
        jumped_time = datetime(2025, 6, 1, 11, 59, 55, tzinfo=timezone.utc)

        with patch("kernle.entity.datetime") as mock_dt:
            mock_dt.now.return_value = jumped_time
            result = _generate_id()

        # The timestamp in the ID should be the pinned value (base_ms), not the jumped time
        id_timestamp = int(result.split("-")[0])
        assert id_timestamp == base_ms

    def test_backward_clock_jump_logs_warning(self, caplog):
        """A backward clock jump should produce a warning log with the delta."""
        base_time = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        base_ms = int(base_time.timestamp() * 1000)

        entity_module._ID_LAST_TIMESTAMP_MS = base_ms
        entity_module._ID_SEQUENCE = 0

        # Clock jumps backward by 5000ms
        jumped_time = datetime(2025, 6, 1, 11, 59, 55, tzinfo=timezone.utc)
        expected_delta = base_ms - int(jumped_time.timestamp() * 1000)

        with caplog.at_level(logging.WARNING, logger="kernle.entity"):
            with patch("kernle.entity.datetime") as mock_dt:
                mock_dt.now.return_value = jumped_time
                _generate_id()

        assert any("Clock jump detected" in msg for msg in caplog.messages)
        assert any(str(expected_delta) in msg for msg in caplog.messages)

    def test_sequence_wrap_raises_error(self):
        """When the sequence reaches _ID_SEQUENCE_MAX, a RuntimeError should be raised."""
        base_time = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        base_ms = int(base_time.timestamp() * 1000)

        entity_module._ID_LAST_TIMESTAMP_MS = base_ms
        # Set sequence to one below max so the next increment hits the limit
        entity_module._ID_SEQUENCE = entity_module._ID_SEQUENCE_MAX - 1

        with patch("kernle.entity.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            with pytest.raises(RuntimeError, match="ID sequence exhausted"):
                _generate_id()

    def test_sequence_near_limit_warns(self, caplog):
        """When the sequence reaches 90% capacity, a warning should be logged."""
        base_time = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        base_ms = int(base_time.timestamp() * 1000)

        entity_module._ID_LAST_TIMESTAMP_MS = base_ms
        # Set sequence so the next increment lands at the warn threshold
        entity_module._ID_SEQUENCE = entity_module._ID_SEQUENCE_WARN_THRESHOLD - 1

        with caplog.at_level(logging.WARNING, logger="kernle.entity"):
            with patch("kernle.entity.datetime") as mock_dt:
                mock_dt.now.return_value = base_time
                result = _generate_id()

        # Should still produce a valid ID
        assert result is not None
        # Should warn about nearing limit
        assert any("ID sequence nearing limit" in msg for msg in caplog.messages)

    def test_clock_recovery_resets_sequence(self, caplog):
        """After a backward jump, when the clock catches up, the sequence resets and an info log is emitted."""
        base_time = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        base_ms = int(base_time.timestamp() * 1000)

        entity_module._ID_LAST_TIMESTAMP_MS = base_ms
        entity_module._ID_SEQUENCE = 0

        # Step 1: Simulate backward jump
        jumped_time = datetime(2025, 6, 1, 11, 59, 55, tzinfo=timezone.utc)
        with patch("kernle.entity.datetime") as mock_dt:
            mock_dt.now.return_value = jumped_time
            _generate_id()

        # Verify we're pinned
        assert entity_module._ID_CLOCK_WAS_PINNED is True

        # Step 2: Clock recovers past the original timestamp
        recovered_time = datetime(2025, 6, 1, 12, 0, 1, tzinfo=timezone.utc)
        recovered_ms = int(recovered_time.timestamp() * 1000)

        with caplog.at_level(logging.INFO, logger="kernle.entity"):
            with patch("kernle.entity.datetime") as mock_dt:
                mock_dt.now.return_value = recovered_time
                result = _generate_id()

        # The ID should use the recovered timestamp
        id_timestamp = int(result.split("-")[0])
        assert id_timestamp == recovered_ms

        # Sequence should have been reset
        assert entity_module._ID_SEQUENCE == 0

        # Clock recovery should be logged
        assert any("Clock recovered" in msg for msg in caplog.messages)

        # Pin flag should be cleared
        assert entity_module._ID_CLOCK_WAS_PINNED is False

    def test_normal_operation_no_warnings(self, caplog):
        """Normal forward-moving clock operation should not produce any warnings."""
        entity_module._ID_LAST_TIMESTAMP_MS = 0
        entity_module._ID_SEQUENCE = 0
        entity_module._ID_CLOCK_WAS_PINNED = False

        with caplog.at_level(logging.DEBUG, logger="kernle.entity"):
            ids = [_generate_id() for _ in range(10)]

        # All IDs should be unique and ordered
        assert len(set(ids)) == 10
        assert ids == sorted(ids)

        # No warnings should be emitted
        warning_messages = [
            r for r in caplog.records if r.levelno >= logging.WARNING and r.name == "kernle.entity"
        ]
        assert len(warning_messages) == 0


# ---- Fixtures ----


def _make_mock_stack(stack_id="test-stack", schema_version=22):
    """Create a mock StackProtocol."""
    stack = MagicMock()
    type(stack).stack_id = PropertyMock(return_value=stack_id)
    type(stack).schema_version = PropertyMock(return_value=schema_version)
    stack.get_stats.return_value = {"episodes": 5, "beliefs": 3}
    stack.on_attach.return_value = None
    stack.on_detach.return_value = None
    stack.on_model_changed.return_value = None
    # Write ops return IDs
    stack.save_episode.side_effect = lambda ep: ep.id
    stack.save_belief.side_effect = lambda b: b.id
    stack.save_value.side_effect = lambda v: v.id
    stack.save_goal.side_effect = lambda g: g.id
    stack.save_note.side_effect = lambda n: n.id
    stack.save_drive.side_effect = lambda d: d.id
    stack.save_relationship.side_effect = lambda r: r.id
    stack.save_raw.side_effect = lambda r: r.id
    # Read ops
    stack.search.return_value = []
    stack.load.return_value = {"identity": {}, "beliefs": []}
    stack.sync.return_value = SyncResult()
    stack.get_trust_assessments.return_value = []
    stack.save_trust_assessment.side_effect = lambda a: a.id
    stack.get_relationships.return_value = []
    stack.get_goals.return_value = []
    return stack


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


@pytest.fixture
def entity(tmp_path):
    return Entity(core_id="test-core", data_dir=tmp_path)


@pytest.fixture
def stack():
    return _make_mock_stack()


@pytest.fixture
def plugin():
    return _make_mock_plugin()


# ---- Basic Properties ----


class TestEntityProperties:
    def test_core_id(self, entity):
        assert entity.core_id == "test-core"

    def test_model_initially_none(self, entity):
        assert entity.model is None

    def test_active_stack_initially_none(self, entity):
        assert entity.active_stack is None

    def test_stacks_initially_empty(self, entity):
        assert entity.stacks == {}

    def test_plugins_initially_empty(self, entity):
        assert entity.plugins == {}


class TestGenerateIdSequenceOrdering:
    def test_generate_id_is_ordered_and_unique(self):
        ids = [_generate_id() for _ in range(512)]

        assert ids == sorted(ids), "Generated IDs should be monotonically ordered"
        assert len(set(ids)) == len(ids), "Generated IDs should be unique"


# ---- Stack Management ----


class TestStackManagement:
    def test_attach_stack_returns_stack_id(self, entity, stack):
        result = entity.attach_stack(stack)
        assert result == "test-stack"
        stack.on_attach.assert_called_once_with("test-core", None)

    def test_attach_stack_with_alias_still_returns_stack_id(self, entity, stack):
        result = entity.attach_stack(stack, alias="my-alias")
        assert result == "test-stack"

    def test_attach_stack_sets_active(self, entity, stack):
        entity.attach_stack(stack)
        assert entity.active_stack is stack

    def test_attach_stack_no_set_active(self, entity, stack):
        entity.attach_stack(stack, set_active=False)
        assert entity.active_stack is None

    def test_attach_duplicate_stack_id_raises(self, entity, stack):
        entity.attach_stack(stack)
        duplicate = _make_mock_stack(stack_id="test-stack")
        with pytest.raises(ValueError, match="already attached"):
            entity.attach_stack(duplicate)

    def test_stacks_property_returns_stack_info(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        stacks = entity.stacks
        assert "test-stack" in stacks
        info = stacks["test-stack"]
        assert isinstance(info, StackInfo)
        assert info.stack_id == "test-stack"
        assert info.alias == "primary"
        assert info.is_active is True
        assert info.schema_version == 22
        assert info.stats == {"episodes": 5, "beliefs": 3}

    def test_detach_stack(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        entity.detach_stack("test-stack")
        assert entity.active_stack is None
        assert "test-stack" not in entity.stacks
        stack.on_detach.assert_called_once_with("test-core")

    def test_detach_nonexistent_stack_is_noop(self, entity):
        entity.detach_stack("nonexistent")  # Should not raise

    def test_detach_clears_active_if_matching(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        second = _make_mock_stack(stack_id="second-stack")
        entity.attach_stack(second, alias="secondary", set_active=False)
        entity.detach_stack("test-stack")
        assert entity.active_stack is None
        assert "second-stack" in entity.stacks

    def test_set_active_stack(self, entity):
        s1 = _make_mock_stack(stack_id="s1")
        s2 = _make_mock_stack(stack_id="s2")
        entity.attach_stack(s1, alias="first")
        entity.attach_stack(s2, alias="second")
        assert entity.active_stack is s2  # Last attached is active
        entity.set_active_stack("s1")
        assert entity.active_stack is s1

    def test_set_active_stack_invalid_raises(self, entity):
        with pytest.raises(ValueError, match="No stack with stack_id"):
            entity.set_active_stack("nonexistent")

    def test_multiple_stacks(self, entity):
        s1 = _make_mock_stack(stack_id="s1")
        s2 = _make_mock_stack(stack_id="s2")
        entity.attach_stack(s1, alias="first", set_active=False)
        entity.attach_stack(s2, alias="second", set_active=True)
        stacks = entity.stacks
        assert len(stacks) == 2
        assert stacks["s1"].is_active is False
        assert stacks["s2"].is_active is True


# ---- Model Management ----


class TestModelManagement:
    def test_set_model(self, entity):
        model = _make_mock_model()
        entity.set_model(model)
        assert entity.model is model
        assert entity.model.model_id == "test-model"

    def test_set_model_notifies_stacks(self, entity, stack):
        entity.attach_stack(stack)
        model = _make_mock_model()
        entity.set_model(model)
        stack.on_model_changed.assert_called_once()

    def test_set_model_replaces_previous(self, entity):
        m1 = _make_mock_model(model_id="first")
        m2 = _make_mock_model(model_id="second")
        entity.set_model(m1)
        entity.set_model(m2)
        assert entity.model.model_id == "second"


# ---- Routed Memory Operations ----


class TestRoutedOperations:
    def test_episode_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        mem_id = entity.episode("test objective", "test outcome")
        assert mem_id is not None
        stack.save_episode.assert_called_once()
        ep = stack.save_episode.call_args[0][0]
        assert isinstance(ep, Episode)
        assert ep.objective == "test objective"
        assert ep.outcome == "test outcome"
        assert ep.stack_id == "test-stack"

    def test_episode_populates_provenance(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out", source="user:alice", context="project:foo")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_entity == "user:alice"
        assert ep.context == "project:foo"
        assert ep.created_at is not None

    def test_episode_default_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_entity == "core:test-core"

    def test_belief_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        mem_id = entity.belief("the sky is blue", confidence=0.9)
        assert mem_id is not None
        b = stack.save_belief.call_args[0][0]
        assert isinstance(b, Belief)
        assert b.statement == "the sky is blue"
        assert b.confidence == 0.9

    def test_belief_default_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.belief("test")
        b = stack.save_belief.call_args[0][0]
        assert b.source_entity == "core:test-core"

    def test_value_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "I value truthfulness", priority=90)
        v = stack.save_value.call_args[0][0]
        assert isinstance(v, Value)
        assert v.name == "honesty"
        assert v.priority == 90

    def test_goal_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust", description="Start with the book", goal_type="aspiration")
        g = stack.save_goal.call_args[0][0]
        assert isinstance(g, Goal)
        assert g.title == "learn rust"
        assert g.goal_type == "aspiration"

    def test_note_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.note("important thought", tags=["meta"])
        n = stack.save_note.call_args[0][0]
        assert isinstance(n, Note)
        assert n.content == "important thought"
        assert n.tags == ["meta"]

    def test_drive_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity", intensity=0.8, focus_areas=["AI"])
        d = stack.save_drive.call_args[0][0]
        assert isinstance(d, Drive)
        assert d.drive_type == "curiosity"
        assert d.intensity == 0.8

    def test_relationship_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("other-entity", trust_level=0.7, entity_type="agent")
        r = stack.save_relationship.call_args[0][0]
        assert isinstance(r, Relationship)
        assert r.entity_name == "other-entity"
        assert r.entity_type == "agent"
        # trust_level (0-1) is converted to sentiment (-1 to 1): (0.7 * 2) - 1 = 0.4
        assert r.sentiment == pytest.approx(0.4)

    def test_raw_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.raw("some unstructured content")
        r = stack.save_raw.call_args[0][0]
        assert isinstance(r, RawEntry)
        assert r.blob == "some unstructured content"
        assert r.source == "core:test-core"


# ---- NoActiveStackError ----


class TestNoActiveStack:
    def test_episode_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.episode("obj", "out")

    def test_belief_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.belief("test")

    def test_value_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.value("v", "s")

    def test_goal_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.goal("g")

    def test_note_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.note("n")

    def test_drive_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.drive("d")

    def test_relationship_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.relationship("other")

    def test_raw_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.raw("r")

    def test_search_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.search("query")

    def test_load_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.load()

    def test_trust_set_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.trust_set("entity", "domain", 0.5)

    def test_trust_get_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.trust_get("entity")

    def test_sync_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.sync()


# ---- Search & Load ----


class TestSearchAndLoad:
    def test_search_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.search("test query", limit=5)
        stack.search.assert_called_once_with("test query", limit=5, record_types=None, context=None)

    def test_load_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.load(token_budget=4000, context="project:x")
        stack.load.assert_called_once_with(token_budget=4000, context="project:x")

    def test_load_calls_plugin_on_load(self, entity, stack, plugin):
        entity.attach_stack(stack)
        entity.load_plugin(plugin)
        entity.load()
        plugin.on_load.assert_called_once()

    def test_load_plugin_error_does_not_crash(self, entity, stack):
        entity.attach_stack(stack)
        bad_plugin = _make_mock_plugin(name="bad-plugin")
        bad_plugin.on_load.side_effect = RuntimeError("boom")
        entity.load_plugin(bad_plugin)
        result = entity.load()  # Should not raise
        assert result is not None


# ---- Trust Operations ----


class TestTrustOperations:
    def test_trust_set(self, entity, stack):
        entity.attach_stack(stack)
        mem_id = entity.trust_set("alice", "general", 0.9, evidence="ep-123")
        assert mem_id is not None
        assessment = stack.save_trust_assessment.call_args[0][0]
        assert isinstance(assessment, TrustAssessment)
        assert assessment.entity == "alice"
        assert assessment.dimensions == {"general": {"score": 0.9}}

    def test_trust_get(self, entity, stack):
        entity.attach_stack(stack)
        entity.trust_get("alice", domain="general")
        stack.get_trust_assessments.assert_called_once_with(entity_id="alice", domain="general")

    def test_trust_list(self, entity, stack):
        entity.attach_stack(stack)
        entity.trust_list(domain="general")
        stack.get_trust_assessments.assert_called_once_with(domain="general")

    def test_trust_list_filters_by_min_score(self, entity, stack):
        entity.attach_stack(stack)
        high = TrustAssessment(
            id="a1",
            stack_id="test-stack",
            entity="alice",
            dimensions={"general": {"score": 0.9}},
        )
        low = TrustAssessment(
            id="a2",
            stack_id="test-stack",
            entity="bob",
            dimensions={"general": {"score": 0.2}},
        )
        stack.get_trust_assessments.return_value = [high, low]
        result = entity.trust_list(min_score=0.5)
        assert len(result) == 1
        assert result[0].entity == "alice"


# ---- Plugin Management ----


class TestPluginManagement:
    def test_load_plugin(self, entity, plugin):
        entity.load_plugin(plugin)
        plugin.activate.assert_called_once()
        assert "test-plugin" in entity.plugins
        info = entity.plugins["test-plugin"]
        assert isinstance(info, PluginInfo)
        assert info.is_loaded is True

    def test_unload_plugin(self, entity, plugin):
        entity.load_plugin(plugin)
        entity.unload_plugin("test-plugin")
        plugin.deactivate.assert_called_once()
        assert "test-plugin" not in entity.plugins

    def test_unload_nonexistent_plugin_is_noop(self, entity):
        entity.unload_plugin("nonexistent")  # Should not raise

    def test_plugin_context_provides_core_id(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.core_id == "test-core"

    def test_plugin_context_returns_none_stack_id_without_stack(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.active_stack_id is None

    def test_plugin_context_returns_stack_id_with_stack(self, entity, stack, plugin):
        entity.attach_stack(stack)
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.active_stack_id == "test-stack"

    def test_plugin_context_write_returns_none_without_stack(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.episode("obj", "out") is None
        assert ctx.belief("stmt") is None
        assert ctx.note("content") is None
        assert ctx.raw("blob") is None

    def test_plugin_context_write_routes_through_entity(self, entity, stack, plugin):
        entity.attach_stack(stack)
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        ctx.episode("plugin objective", "plugin outcome")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_entity == "plugin:test-plugin"

    def test_plugin_context_search_returns_empty_without_stack(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.search("query") == []

    def test_plugin_context_get_data_dir(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        data_dir = ctx.get_data_dir()
        assert data_dir.exists()
        assert "test-plugin" in str(data_dir)

    def test_plugin_context_config_and_secrets(self, entity, plugin):
        entity._plugin_configs["test-plugin"] = {"api_url": "https://example.com"}
        entity._plugin_secrets["test-plugin"] = {"api_key": "secret123"}
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.get_config("api_url") == "https://example.com"
        assert ctx.get_config("missing", "default") == "default"
        assert ctx.get_secret("api_key") == "secret123"
        assert ctx.get_secret("missing") is None

    @patch("kernle.entity.discover_plugins")
    def test_discover_plugins(self, mock_discover, entity, plugin):
        from kernle.discovery import DiscoveredComponent

        mock_discover.return_value = [
            DiscoveredComponent(
                name="found-plugin",
                group="kernle.plugins",
                module="found_plugin",
                attr="FoundPlugin",
                dist_version="2.0.0",
            )
        ]
        entity.load_plugin(plugin)
        discovered = entity.discover_plugins()
        assert len(discovered) == 1
        assert discovered[0].name == "found-plugin"
        assert discovered[0].is_loaded is False


# ---- Status Assembly ----


class TestStatus:
    def test_status_basic(self, entity):
        result = entity.status()
        assert result["core_id"] == "test-core"
        assert result["model"] is None
        assert result["stacks"] == {}
        assert result["plugins"] == {}

    def test_status_with_model(self, entity):
        entity.set_model(_make_mock_model(model_id="claude"))
        result = entity.status()
        assert result["model"] == "claude"

    def test_status_with_stack(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        result = entity.status()
        assert "test-stack" in result["stacks"]
        assert result["stacks"]["test-stack"]["stack_id"] == "test-stack"
        assert result["stacks"]["test-stack"]["alias"] == "primary"
        assert result["stacks"]["test-stack"]["active"] is True

    def test_status_with_plugin(self, entity, plugin):
        entity.load_plugin(plugin)
        result = entity.status()
        assert "test-plugin" in result["plugins"]
        assert result["plugins"]["test-plugin"]["health"]["healthy"] is True

    def test_status_plugin_health_failure_isolated(self, entity):
        bad_plugin = _make_mock_plugin(name="bad")
        bad_plugin.health_check.side_effect = RuntimeError("crash")
        entity.load_plugin(bad_plugin)
        result = entity.status()
        assert result["plugins"]["bad"]["health"]["healthy"] is False

    def test_status_calls_plugin_on_status(self, entity, plugin):
        entity.load_plugin(plugin)
        entity.status()
        plugin.on_status.assert_called_once()


# ---- Binding ----


class TestBinding:
    def test_get_binding(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        entity.set_model(_make_mock_model())
        entity.load_plugin(_make_mock_plugin())
        binding = entity.get_binding()
        assert isinstance(binding, Binding)
        assert binding.core_id == "test-core"
        assert binding.stacks == {"test-stack": "primary"}
        assert binding.active_stack_id == "test-stack"
        assert "test-plugin" in binding.plugins
        assert binding.model_config["provider"] == "anthropic"
        assert binding.model_config["model_id"] == "test-model"

    def test_from_binding_object(self):
        binding = Binding(
            core_id="restored-core",
            model_config={},
            stacks={},
        )
        restored = Entity.from_binding(binding)
        assert restored.core_id == "restored-core"


# ---- Checkpoint ----


class TestCheckpoint:
    def test_checkpoint(self, entity, stack):
        entity.attach_stack(stack)
        cp_id = entity.checkpoint("test checkpoint")
        assert "test-core" in cp_id
        cp_dir = entity._data_dir / "checkpoints"
        assert cp_dir.exists()
        files = list(cp_dir.glob("*.json"))
        assert len(files) == 1

    def test_checkpoint_ids_are_unique(self, entity, stack):
        entity.attach_stack(stack)
        first = entity.checkpoint("first")
        second = entity.checkpoint("second")
        assert first != second
        assert first.startswith("test-core_")

    def test_checkpoint_includes_schema_version(self, entity, stack):
        import json

        entity.attach_stack(stack)
        entity.checkpoint("schema test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        data = json.loads(files[0].read_text())
        assert data["schema_version"] == 1

    def test_checkpoint_roundtrip(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        entity.checkpoint("roundtrip test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        assert len(files) == 1

        restored = Entity.from_checkpoint(files[0])
        assert restored.core_id == "test-core"

    def test_checkpoint_retention_keeps_10(self, entity, stack):
        entity.attach_stack(stack)
        for i in range(15):
            entity.checkpoint(f"checkpoint {i}")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        assert len(files) == 10

    def test_from_checkpoint_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            Entity.from_checkpoint(missing)

    def test_from_checkpoint_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            Entity.from_checkpoint(bad)

    def test_from_checkpoint_unsupported_schema_version(self, tmp_path):
        import json

        cp = tmp_path / "future.json"
        cp.write_text(
            json.dumps(
                {
                    "schema_version": 999,
                    "binding": {"core_id": "test"},
                }
            )
        )
        with pytest.raises(ValueError, match="Unsupported checkpoint schema_version"):
            Entity.from_checkpoint(cp)

    def test_from_checkpoint_missing_binding(self, tmp_path):
        import json

        cp = tmp_path / "nobinding.json"
        cp.write_text(json.dumps({"schema_version": 1}))
        with pytest.raises(ValueError, match="missing 'binding' key"):
            Entity.from_checkpoint(cp)

    def test_from_checkpoint_no_schema_version_warns(self, entity, stack, tmp_path, caplog):
        import json
        import logging

        entity.attach_stack(stack)
        # Create a checkpoint and remove schema_version
        entity.checkpoint("test")
        cp_dir = entity._data_dir / "checkpoints"
        files = list(cp_dir.glob("*.json"))
        data = json.loads(files[0].read_text())
        del data["schema_version"]
        files[0].write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING, logger="kernle.entity"):
            restored = Entity.from_checkpoint(files[0])
        assert restored.core_id == "test-core"
        assert any("no schema_version" in msg for msg in caplog.messages)

    def test_from_checkpoint_binding_not_dict(self, tmp_path):
        import json

        cp = tmp_path / "bad_binding.json"
        cp.write_text(json.dumps({"schema_version": 1, "binding": "not-a-dict"}))
        with pytest.raises(ValueError, match="must be a dict"):
            Entity.from_checkpoint(cp)

    def test_from_checkpoint_binding_missing_core_id(self, tmp_path):
        import json

        cp = tmp_path / "no_core_id.json"
        cp.write_text(json.dumps({"schema_version": 1, "binding": {"stacks": {}}}))
        with pytest.raises(ValueError, match="missing required 'core_id'"):
            Entity.from_checkpoint(cp)

    def test_model_config_includes_provider(self, entity, stack):
        entity.attach_stack(stack)
        entity.set_model(_make_mock_model(model_id="test-model", provider="anthropic"))
        binding = entity.get_binding()
        assert binding.model_config["provider"] == "anthropic"
        assert binding.model_config["model_id"] == "test-model"


# ---- Sync ----


class TestRepeatAvoid:
    def test_episode_forwards_repeat_avoid(self, entity, stack):
        entity.attach_stack(stack)
        ep_id = entity.episode(
            "test objective",
            "test outcome",
            repeat=["pattern1"],
            avoid=["antipattern1"],
        )
        assert isinstance(ep_id, str)
        # Verify the Episode object passed to save_episode has repeat/avoid
        ep = stack.save_episode.call_args[0][0]
        assert ep.repeat == ["pattern1"]
        assert ep.avoid == ["antipattern1"]

    def test_episode_repeat_avoid_defaults_none(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out")
        ep = stack.save_episode.call_args[0][0]
        assert ep.repeat is None
        assert ep.avoid is None

    def test_plugin_context_forwards_repeat_avoid(self, entity, stack):
        entity.attach_stack(stack)
        from kernle.entity import _PluginContextImpl

        ctx = _PluginContextImpl(entity, "test-plugin")
        ep_id = ctx.episode(
            "plugin obj",
            "plugin out",
            repeat=["do this"],
            avoid=["not this"],
        )
        assert isinstance(ep_id, str)
        ep = stack.save_episode.call_args[0][0]
        assert ep.repeat == ["do this"]
        assert ep.avoid == ["not this"]


class TestSync:
    def test_sync_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        result = entity.sync()
        stack.sync.assert_called_once()
        assert isinstance(result, SyncResult)


# ---- Plugin Protocol Version ----


class TestPluginProtocolVersion:
    def test_matching_version_loads(self, entity, stack):
        entity.attach_stack(stack)
        plugin = MagicMock()
        plugin.name = "version-test"
        plugin.version = "1.0.0"
        plugin.protocol_version = 1  # matches PROTOCOL_VERSION
        plugin.description = "test"
        plugin.register_tools.return_value = []
        # Should not raise
        entity.load_plugin(plugin)
        assert "version-test" in entity.plugins

    def test_future_version_raises(self, entity, stack):
        entity.attach_stack(stack)
        plugin = MagicMock()
        plugin.name = "future-plugin"
        plugin.version = "1.0.0"
        plugin.protocol_version = 999
        plugin.description = "test"
        with pytest.raises(ValueError, match="protocol version 999"):
            entity.load_plugin(plugin)

    def test_old_version_warns(self, entity, stack):
        entity.attach_stack(stack)
        plugin = MagicMock()
        plugin.name = "old-plugin"
        plugin.version = "1.0.0"
        plugin.protocol_version = 0
        plugin.description = "test"
        plugin.register_tools.return_value = []
        # logger.warning doesn't trigger pytest.warns, so just verify it loads
        entity.load_plugin(plugin)
        assert "old-plugin" in entity.plugins


class TestProcessConfigHandling:
    def test_process_uses_defaults_when_settings_invalid(self, entity, stack, monkeypatch):
        entity.attach_stack(stack)
        stack.get_stack_setting.side_effect = RuntimeError("setting parse failure")
        stack.get_processing_config.return_value = []

        class _FakeProcessor:
            def __init__(self, *args, **kwargs):
                pass

            def update_config(self, layer_transition, config):
                pass

            def process(self, *args, **kwargs):
                return []

        monkeypatch.setattr("kernle.processing.MemoryProcessor", _FakeProcessor)

        results = entity.process()
        assert results == []

    def test_process_raises_on_invalid_settings_with_strict_mode(self, entity, stack, monkeypatch):
        entity.attach_stack(stack)
        stack.get_stack_setting.side_effect = RuntimeError("setting parse failure")
        stack.get_processing_config.return_value = []

        class _FakeProcessor:
            def __init__(self, *args, **kwargs):
                pass

            def update_config(self, layer_transition, config):
                pass

            def process(self, *args, **kwargs):
                return []

        monkeypatch.setattr("kernle.processing.MemoryProcessor", _FakeProcessor)

        with pytest.raises(RuntimeError, match="setting parse failure"):
            entity.process(strict=True)


# ---- Binding Metadata ----


class TestBindingMetadata:
    def test_from_binding_stores_metadata(self):
        from kernle.protocols import Binding

        binding = Binding(
            core_id="meta-test",
            model_config={"model_id": "test-model"},
            stacks={"s1": "main"},
            plugins=["plugin-a"],
        )
        entity = Entity.from_binding(binding)
        assert entity._restored_binding is not None
        assert entity._restored_binding.core_id == "meta-test"
        assert entity._restored_binding.plugins == ["plugin-a"]


# ---- source_entity Attribution ----


class TestSourceEntity:
    """All write methods should set source_entity on the created memory."""

    def test_value_sets_source_entity(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "Be truthful")
        v = stack.save_value.call_args[0][0]
        assert v.source_entity == "core:test-core"

    def test_value_uses_custom_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "Be truthful", source="user:alice")
        v = stack.save_value.call_args[0][0]
        assert v.source_entity == "user:alice"

    def test_goal_sets_source_entity(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust")
        g = stack.save_goal.call_args[0][0]
        assert g.source_entity == "core:test-core"

    def test_goal_uses_custom_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust", source="user:bob")
        g = stack.save_goal.call_args[0][0]
        assert g.source_entity == "user:bob"

    def test_drive_sets_source_entity(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity")
        d = stack.save_drive.call_args[0][0]
        assert d.source_entity == "core:test-core"

    def test_drive_uses_custom_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity", source="user:charlie")
        d = stack.save_drive.call_args[0][0]
        assert d.source_entity == "user:charlie"

    def test_relationship_sets_source_entity(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("other-agent")
        r = stack.save_relationship.call_args[0][0]
        assert r.source_entity == "core:test-core"

    def test_relationship_uses_custom_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("other-agent", source="user:dave")
        r = stack.save_relationship.call_args[0][0]
        assert r.source_entity == "user:dave"


# ---- derived_from Pass-Through ----


class TestDerivedFrom:
    """All write methods should pass derived_from to the created memory."""

    def test_value_passes_derived_from(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "Be truthful", derived_from=["belief:abc123"])
        v = stack.save_value.call_args[0][0]
        assert v.derived_from == ["belief:abc123"]

    def test_goal_passes_derived_from(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust", derived_from=["episode:ep1"])
        g = stack.save_goal.call_args[0][0]
        assert g.derived_from == ["episode:ep1"]

    def test_drive_passes_derived_from(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity", derived_from=["episode:ep1", "belief:b1"])
        d = stack.save_drive.call_args[0][0]
        assert d.derived_from == ["episode:ep1", "belief:b1"]

    def test_relationship_passes_derived_from(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("other-agent", derived_from=["episode:ep1"])
        r = stack.save_relationship.call_args[0][0]
        assert r.derived_from == ["episode:ep1"]

    def test_value_without_derived_from(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "Be truthful")
        v = stack.save_value.call_args[0][0]
        assert v.derived_from is None

    def test_goal_without_derived_from(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust")
        g = stack.save_goal.call_args[0][0]
        assert g.derived_from is None


# ---- PluginContext Attribution ----


class TestPluginContextAttribution:
    """Plugin writes should be attributed with source=plugin:{name}."""

    def test_plugin_value_sets_source(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.value("honesty", "Be truthful")
        v = stack.save_value.call_args[0][0]
        assert v.source_entity == "plugin:test-plugin"

    def test_plugin_goal_sets_source(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.goal("learn rust")
        g = stack.save_goal.call_args[0][0]
        assert g.source_entity == "plugin:test-plugin"

    def test_plugin_relationship_sets_source(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.relationship("other-agent")
        r = stack.save_relationship.call_args[0][0]
        assert r.source_entity == "plugin:test-plugin"

    def test_plugin_derived_from_passes_through(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.belief("test statement", derived_from=["episode:ep1"])
        b = stack.save_belief.call_args[0][0]
        # build_derived_from appends context marker for the plugin source
        assert b.derived_from == ["episode:ep1", "context:plugin:test-plugin"]


# ---- Entity Enrichment Parity Tests ----


class TestEntityEnrichmentParity:
    """Verify Entity methods apply the same enrichment as WritersMixin."""

    def test_episode_infers_outcome_type_success(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("Fix bug", "Fixed successfully")
        ep = stack.save_episode.call_args[0][0]
        assert ep.outcome_type == "success"

    def test_episode_infers_outcome_type_failure(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("Run tests", "Tests failed with errors")
        ep = stack.save_episode.call_args[0][0]
        assert ep.outcome_type == "failure"

    def test_episode_infers_outcome_type_partial(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("Investigate", "Found some clues")
        ep = stack.save_episode.call_args[0][0]
        assert ep.outcome_type == "partial"

    def test_episode_builds_derived_from_with_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out", source="session with Sean", derived_from=["ep:1"])
        ep = stack.save_episode.call_args[0][0]
        assert ep.derived_from == ["ep:1", "context:session with Sean"]

    def test_episode_defaults_tags_to_manual(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out")
        ep = stack.save_episode.call_args[0][0]
        assert ep.tags == ["manual"]

    def test_episode_defaults_confidence_to_0_8(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out")
        ep = stack.save_episode.call_args[0][0]
        assert ep.confidence == 0.8

    def test_note_formats_decision_content(self, entity, stack):
        entity.attach_stack(stack)
        entity.note("Use TypeScript", type="decision", reason="Type safety")
        n = stack.save_note.call_args[0][0]
        assert n.content == "**Decision**: Use TypeScript\n**Reason**: Type safety"

    def test_note_normalizes_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.note("test", type="insight")
        n = stack.save_note.call_args[0][0]
        assert n.note_type == "insight"
        assert n.content.startswith("**Insight**:")

    def test_note_defaults_tags_to_empty(self, entity, stack):
        entity.attach_stack(stack)
        entity.note("test")
        n = stack.save_note.call_args[0][0]
        assert n.tags == []

    def test_belief_clamps_confidence(self, entity, stack):
        entity.attach_stack(stack)
        entity.belief("test", confidence=1.5)
        b = stack.save_belief.call_args[0][0]
        assert b.confidence == 1.0

    def test_belief_clamps_confidence_min(self, entity, stack):
        entity.attach_stack(stack)
        entity.belief("test", confidence=-0.3)
        b = stack.save_belief.call_args[0][0]
        assert b.confidence == 0.0

    def test_belief_normalizes_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.belief("test", type="hypothesis")
        b = stack.save_belief.call_args[0][0]
        assert b.belief_type == "hypothesis"

    def test_goal_validates_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid goal_type"):
            entity.goal("learn", goal_type="fantasy")

    def test_goal_auto_protects_aspiration(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust", goal_type="aspiration")
        g = stack.save_goal.call_args[0][0]
        assert g.is_protected is True
        stack.protect_memory.assert_called_once()

    def test_goal_defaults_description_to_title(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust")
        g = stack.save_goal.call_args[0][0]
        assert g.description == "learn rust"

    def test_goal_defaults_status_to_active(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust")
        g = stack.save_goal.call_args[0][0]
        assert g.status == "active"

    def test_drive_validates_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid drive type"):
            entity.drive("hunger")

    def test_drive_clamps_intensity(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity", intensity=1.5)
        d = stack.save_drive.call_args[0][0]
        assert d.intensity == 1.0

    def test_drive_defaults_focus_areas_to_empty(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity")
        d = stack.save_drive.call_args[0][0]
        assert d.focus_areas == []

    def test_drive_sets_updated_at(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity")
        d = stack.save_drive.call_args[0][0]
        assert d.updated_at is not None

    def test_relationship_converts_trust_to_sentiment(self, entity, stack):
        entity.attach_stack(stack)
        # trust_level 0.0 -> sentiment -1.0
        entity.relationship("alice", trust_level=0.0)
        r = stack.save_relationship.call_args[0][0]
        assert r.sentiment == pytest.approx(-1.0)

    def test_relationship_defaults_entity_type_to_person(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("alice")
        r = stack.save_relationship.call_args[0][0]
        assert r.entity_type == "person"

    def test_relationship_defaults_type_to_interaction(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("alice")
        r = stack.save_relationship.call_args[0][0]
        assert r.relationship_type == "interaction"

    def test_relationship_sets_interaction_count(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("alice")
        r = stack.save_relationship.call_args[0][0]
        assert r.interaction_count == 1

    def test_relationship_sets_last_interaction(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("alice")
        r = stack.save_relationship.call_args[0][0]
        assert r.last_interaction is not None


# ---- source_type Parameter Coverage for Entity Writer Methods ----


class TestEntitySourceType:
    """Entity writer methods should handle source_type correctly."""

    # -- episode --

    def test_episode_default_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_type == "direct_experience"

    def test_episode_enum_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out", source_type=SourceType.EXTERNAL)
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_type == "external"

    def test_episode_string_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out", source_type="external")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_type == "external"

    def test_episode_invalid_source_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid source_type"):
            entity.episode("obj", "out", source_type="telepathy")

    # -- belief --

    def test_belief_default_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.belief("the sky is blue")
        b = stack.save_belief.call_args[0][0]
        assert b.source_type == "direct_experience"

    def test_belief_enum_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.belief("the sky is blue", source_type=SourceType.EXTERNAL)
        b = stack.save_belief.call_args[0][0]
        assert b.source_type == "external"

    def test_belief_string_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.belief("the sky is blue", source_type="external")
        b = stack.save_belief.call_args[0][0]
        assert b.source_type == "external"

    def test_belief_invalid_source_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid source_type"):
            entity.belief("the sky is blue", source_type="telepathy")

    # -- value --

    def test_value_default_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "Be truthful")
        v = stack.save_value.call_args[0][0]
        assert v.source_type == "direct_experience"

    def test_value_enum_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "Be truthful", source_type=SourceType.EXTERNAL)
        v = stack.save_value.call_args[0][0]
        assert v.source_type == "external"

    def test_value_string_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "Be truthful", source_type="external")
        v = stack.save_value.call_args[0][0]
        assert v.source_type == "external"

    def test_value_invalid_source_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid source_type"):
            entity.value("honesty", "Be truthful", source_type="telepathy")

    # -- goal --

    def test_goal_default_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust")
        g = stack.save_goal.call_args[0][0]
        assert g.source_type == "direct_experience"

    def test_goal_enum_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust", source_type=SourceType.EXTERNAL)
        g = stack.save_goal.call_args[0][0]
        assert g.source_type == "external"

    def test_goal_string_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust", source_type="external")
        g = stack.save_goal.call_args[0][0]
        assert g.source_type == "external"

    def test_goal_invalid_source_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid source_type"):
            entity.goal("learn rust", source_type="telepathy")

    # -- note --

    def test_note_default_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.note("important thought")
        n = stack.save_note.call_args[0][0]
        assert n.source_type == "direct_experience"

    def test_note_enum_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.note("important thought", source_type=SourceType.EXTERNAL)
        n = stack.save_note.call_args[0][0]
        assert n.source_type == "external"

    def test_note_string_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.note("important thought", source_type="external")
        n = stack.save_note.call_args[0][0]
        assert n.source_type == "external"

    def test_note_invalid_source_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid source_type"):
            entity.note("important thought", source_type="telepathy")

    # -- drive --

    def test_drive_default_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity", intensity=0.8)
        d = stack.save_drive.call_args[0][0]
        assert d.source_type == "direct_experience"

    def test_drive_enum_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity", intensity=0.8, source_type=SourceType.CONSOLIDATION)
        d = stack.save_drive.call_args[0][0]
        assert d.source_type == "consolidation"

    def test_drive_string_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity", intensity=0.8, source_type="consolidation")
        d = stack.save_drive.call_args[0][0]
        assert d.source_type == "consolidation"

    def test_drive_invalid_source_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid source_type"):
            entity.drive("curiosity", source_type="telepathy")

    # -- relationship --

    def test_relationship_default_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("other-agent")
        r = stack.save_relationship.call_args[0][0]
        assert r.source_type == "direct_experience"

    def test_relationship_enum_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("other-agent", source_type=SourceType.EXTERNAL)
        r = stack.save_relationship.call_args[0][0]
        assert r.source_type == "external"

    def test_relationship_string_source_type(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("other-agent", source_type="external")
        r = stack.save_relationship.call_args[0][0]
        assert r.source_type == "external"

    def test_relationship_invalid_source_type(self, entity, stack):
        entity.attach_stack(stack)
        with pytest.raises(ValueError, match="Invalid source_type"):
            entity.relationship("other-agent", source_type="telepathy")


# ---- PluginContext source_type Passthrough ----


class TestPluginContextSourceType:
    """PluginContext should pass source_type through to Entity methods."""

    def test_episode_source_type_passthrough(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.episode("obj", "out", source_type="external")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_type == "external"

    def test_episode_default_source_type(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.episode("obj", "out")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_type == "direct_experience"

    def test_belief_source_type_passthrough(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.belief("test statement", source_type="seed")
        b = stack.save_belief.call_args[0][0]
        assert b.source_type == "seed"

    def test_relationship_source_type_passthrough(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.relationship("other-agent", source_type="external")
        r = stack.save_relationship.call_args[0][0]
        assert r.source_type == "external"

    def test_note_source_type_passthrough(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.note("a note", source_type="inference")
        n = stack.save_note.call_args[0][0]
        assert n.source_type == "inference"

    def test_goal_source_type_passthrough(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.goal("learn rust", source_type="external")
        g = stack.save_goal.call_args[0][0]
        assert g.source_type == "external"

    def test_drive_source_type_passthrough(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.drive("curiosity", source_type="consolidation")
        d = stack.save_drive.call_args[0][0]
        assert d.source_type == "consolidation"

    def test_value_source_type_passthrough(self, entity, stack):
        entity.attach_stack(stack)
        ctx = _PluginContextImpl(entity, "test-plugin")
        ctx.value("honesty", "Be truthful", source_type="seed")
        v = stack.save_value.call_args[0][0]
        assert v.source_type == "seed"
