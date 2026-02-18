"""Tests for memory processing MCP tools, CLI commands, and Kernle.process() delegation."""

import json
from unittest.mock import Mock, patch

import pytest

from kernle.mcp.server import (
    TOOLS,
    call_tool,
    validate_tool_input,
)
from kernle.processing import VALID_TRANSITIONS
from kernle.storage.sqlite import SQLiteStorage


@pytest.fixture
def kernle_instance(tmp_path):
    """Create a real Kernle instance with SQLite storage."""
    from kernle import Kernle

    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(stack_id="test-process", db_path=db_path)
    k = Kernle(stack_id="test-process", storage=storage, strict=False)
    yield k
    storage.close()


# =============================================================================
# Kernle.process() delegation tests
# =============================================================================


class TestKernleProcess:
    """Test Kernle.process() delegates to Entity.process()."""

    def test_process_delegates_to_entity(self, kernle_instance):
        """Kernle.process() should delegate to entity.process()."""
        k = kernle_instance

        mock_entity = Mock()
        mock_entity.process.return_value = []
        k._entity = mock_entity

        result = k.process(transition="raw_to_episode", force=True)
        mock_entity.process.assert_called_once_with(
            transition="raw_to_episode",
            force=True,
            auto_promote=False,
            batch_size=None,
        )
        assert result == []

    def test_process_no_args_delegates_defaults(self, kernle_instance):
        """Kernle.process() with no args passes defaults."""
        k = kernle_instance

        mock_entity = Mock()
        mock_entity.process.return_value = []
        k._entity = mock_entity

        k.process()
        mock_entity.process.assert_called_once_with(
            transition=None,
            force=False,
            auto_promote=False,
            batch_size=None,
        )

    def test_process_propagates_runtime_error(self, kernle_instance):
        """Kernle.process() propagates RuntimeError from entity."""
        k = kernle_instance

        mock_entity = Mock()
        mock_entity.process.side_effect = RuntimeError(
            "No model bound \u2014 processing requires inference"
        )
        k._entity = mock_entity

        with pytest.raises(RuntimeError, match="No model bound"):
            k.process()


# =============================================================================
# MCP tool definition tests
# =============================================================================


class TestMCPProcessToolDefinitions:
    """Test that memory_process and memory_process_status tools are defined."""

    def test_memory_process_tool_exists(self):
        """memory_process tool should be in TOOLS list."""
        tool_names = {t.name for t in TOOLS}
        assert "memory_process" in tool_names

    def test_memory_process_status_tool_exists(self):
        """memory_process_status tool should be in TOOLS list."""
        tool_names = {t.name for t in TOOLS}
        assert "memory_process_status" in tool_names

    def test_memory_process_schema(self):
        """memory_process tool should have correct input schema."""
        tool = next(t for t in TOOLS if t.name == "memory_process")
        schema = tool.inputSchema
        assert schema["type"] == "object"
        props = schema["properties"]
        assert "transition" in props
        assert "force" in props
        assert "enum" in props["transition"]
        assert set(props["transition"]["enum"]) == VALID_TRANSITIONS

    def test_memory_process_status_schema(self):
        """memory_process_status tool should have minimal schema."""
        tool = next(t for t in TOOLS if t.name == "memory_process_status")
        schema = tool.inputSchema
        assert schema["type"] == "object"


# =============================================================================
# MCP validation tests
# =============================================================================


class TestMCPProcessValidation:
    """Test validate_tool_input for process tools."""

    def test_validate_process_no_args(self):
        """Validate memory_process with no arguments."""
        result = validate_tool_input("memory_process", {})
        assert result["transition"] is None
        assert result["force"] is False

    def test_validate_process_with_transition(self):
        """Validate memory_process with valid transition."""
        result = validate_tool_input("memory_process", {"transition": "raw_to_episode"})
        assert result["transition"] == "raw_to_episode"

    def test_validate_process_invalid_transition(self):
        """Validate memory_process rejects invalid transition."""
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_tool_input("memory_process", {"transition": "invalid_thing"})

    def test_validate_process_with_force(self):
        """Validate memory_process with force flag."""
        result = validate_tool_input("memory_process", {"force": True})
        assert result["force"] is True

    def test_validate_process_force_non_bool(self):
        """Non-bool force defaults to False."""
        result = validate_tool_input("memory_process", {"force": "yes"})
        assert result["force"] is False

    def test_validate_process_status(self):
        """Validate memory_process_status with no args."""
        result = validate_tool_input("memory_process_status", {})
        assert result == {}


# =============================================================================
# MCP handler tests
# =============================================================================


@pytest.fixture
def mock_kernle_for_process():
    """Create mock Kernle for process tool tests."""
    mock = Mock()
    mock._storage = Mock()

    # Set up storage mock returns
    mock._storage.list_raw.return_value = []
    mock._storage.get_episodes.return_value = []
    mock._storage.get_beliefs.return_value = []

    return mock


@pytest.fixture
def patched_kernle_process(mock_kernle_for_process):
    """Patch get_kernle to return process mock."""
    with patch("kernle.mcp.server.get_kernle", return_value=mock_kernle_for_process):
        yield mock_kernle_for_process


class TestMCPProcessHandler:
    """Test memory_process MCP tool handler."""

    @pytest.mark.asyncio
    async def test_process_no_model_returns_error(self, patched_kernle_process):
        """memory_process with no model bound returns informative error."""
        patched_kernle_process.process.side_effect = RuntimeError("No model bound")

        result = await call_tool("memory_process", {})

        assert len(result) == 1
        assert "requires a bound model" in result[0].text

    @pytest.mark.asyncio
    async def test_process_no_triggers(self, patched_kernle_process):
        """memory_process with no triggers met returns message."""
        patched_kernle_process.process.return_value = []

        result = await call_tool("memory_process", {})

        assert len(result) == 1
        assert "No processing triggered" in result[0].text
        assert "force=true" in result[0].text

    @pytest.mark.asyncio
    async def test_process_force_no_sources(self, patched_kernle_process):
        """memory_process with force but no sources."""
        patched_kernle_process.process.return_value = []

        result = await call_tool("memory_process", {"force": True})

        assert len(result) == 1
        assert "No processing triggered" in result[0].text
        assert "No unprocessed memories found" in result[0].text

    @pytest.mark.asyncio
    async def test_process_successful(self, patched_kernle_process):
        """memory_process returns formatted results on success."""
        from kernle.processing import ProcessingResult

        mock_result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=3,
            created=[
                {"type": "episode", "id": "ep-12345678-abcd"},
                {"type": "episode", "id": "ep-87654321-dcba"},
            ],
            auto_promote=True,
        )
        patched_kernle_process.process.return_value = [mock_result]

        result = await call_tool("memory_process", {"force": True, "auto_promote": True})

        assert len(result) == 1
        text = result[0].text
        assert "Processing complete" in text
        assert "raw_to_episode" in text
        assert "3 sources" in text
        assert "2 created" in text

    @pytest.mark.asyncio
    async def test_process_with_errors(self, patched_kernle_process):
        """memory_process includes errors in output."""
        from kernle.processing import ProcessingResult

        mock_result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=5,
            errors=["Inference failed: connection timeout"],
        )
        patched_kernle_process.process.return_value = [mock_result]

        result = await call_tool("memory_process", {"force": True})

        text = result[0].text
        assert "Error: Inference failed" in text

    @pytest.mark.asyncio
    async def test_process_skipped(self, patched_kernle_process):
        """memory_process shows skip reason."""
        from kernle.processing import ProcessingResult

        mock_result = ProcessingResult(
            layer_transition="raw_to_note",
            source_count=0,
            skipped=True,
            skip_reason="No unprocessed sources",
        )
        patched_kernle_process.process.return_value = [mock_result]

        result = await call_tool("memory_process", {"force": True})

        text = result[0].text
        assert "skipped" in text
        assert "No unprocessed sources" in text


class TestMCPProcessStatusHandler:
    """Test memory_process_status MCP tool handler."""

    @pytest.mark.asyncio
    async def test_status_empty(self, patched_kernle_process):
        """memory_process_status with no unprocessed memories."""
        result = await call_tool("memory_process_status", {})

        assert len(result) == 1
        text = result[0].text
        assert "Memory Processing Status" in text
        assert "Unprocessed raw entries: 0" in text
        assert "Unprocessed episodes: 0" in text

    @pytest.mark.asyncio
    async def test_status_with_unprocessed(self, patched_kernle_process):
        """memory_process_status shows counts and trigger readiness."""
        # Create mock unprocessed items
        raw_entries = [Mock() for _ in range(15)]
        patched_kernle_process._storage.list_raw.return_value = raw_entries

        ep_processed = Mock()
        ep_processed.processed = True
        ep_unprocessed = Mock()
        ep_unprocessed.processed = False
        patched_kernle_process._storage.get_episodes.return_value = [
            ep_processed,
            ep_unprocessed,
            ep_unprocessed,
            ep_unprocessed,
            ep_unprocessed,
            ep_unprocessed,
        ]

        result = await call_tool("memory_process_status", {})

        text = result[0].text
        assert "Unprocessed raw entries: 15" in text
        assert "Unprocessed episodes: 5" in text
        assert "READY" in text  # raw_to_episode should be ready at 15 >= 10


# =============================================================================
# CLI command tests
# =============================================================================


class TestCLIProcessCommand:
    """Test CLI process command registration and execution."""

    def test_process_command_registered(self):
        """process command should be importable from commands."""
        from kernle.cli.commands import cmd_process

        assert callable(cmd_process)

    def test_process_run_no_model(self, kernle_instance, capsys):
        """kernle process run with no model shows error."""
        from kernle.cli.commands.process import cmd_process

        k = kernle_instance

        mock_entity = Mock()
        mock_entity.process.side_effect = RuntimeError(
            "No model bound \u2014 processing requires inference"
        )
        k._entity = mock_entity

        args = Mock()
        args.process_action = "run"
        args.transition = None
        args.force = False
        args.json = False

        cmd_process(args, k)

        captured = capsys.readouterr()
        assert "requires a bound model" in captured.out

    def test_process_run_success(self, kernle_instance, capsys):
        """kernle process run with successful processing (auto-promote mode)."""
        from kernle.cli.commands.process import cmd_process
        from kernle.processing import ProcessingResult

        k = kernle_instance

        mock_result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=3,
            created=[{"type": "episode", "id": "ep-12345678"}],
            auto_promote=True,
        )
        mock_entity = Mock()
        mock_entity.process.return_value = [mock_result]
        k._entity = mock_entity

        args = Mock()
        args.process_action = "run"
        args.transition = "raw_to_episode"
        args.force = True
        args.auto_promote = True
        args.json = False

        cmd_process(args, k)

        captured = capsys.readouterr()
        assert "Processing complete" in captured.out
        assert "raw_to_episode" in captured.out
        assert "ep-12345" in captured.out

    def test_process_run_json_output(self, kernle_instance, capsys):
        """kernle process run --json outputs JSON."""
        from kernle.cli.commands.process import cmd_process
        from kernle.processing import ProcessingResult

        k = kernle_instance

        mock_result = ProcessingResult(
            layer_transition="raw_to_note",
            source_count=2,
            created=[{"type": "note", "id": "note-abc123"}],
        )
        mock_entity = Mock()
        mock_entity.process.return_value = [mock_result]
        k._entity = mock_entity

        args = Mock()
        args.process_action = "run"
        args.transition = None
        args.force = True
        args.json = True

        cmd_process(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, list)
        assert output[0]["transition"] == "raw_to_note"
        assert output[0]["source_count"] == 2

    def test_process_status(self, kernle_instance, capsys):
        """kernle process status shows counts."""
        from kernle.cli.commands.process import cmd_process

        k = kernle_instance

        # Mock storage methods
        k._storage.list_raw = Mock(return_value=[Mock() for _ in range(5)])

        ep = Mock()
        ep.processed = False
        k._storage.get_episodes = Mock(return_value=[ep, ep])

        belief = Mock()
        belief.processed = False
        k._storage.get_beliefs = Mock(return_value=[belief])

        args = Mock()
        args.process_action = "status"
        args.json = False

        cmd_process(args, k)

        captured = capsys.readouterr()
        assert "Unprocessed raw entries:" in captured.out
        assert "5" in captured.out
        assert "Trigger Status:" in captured.out

    def test_process_status_json(self, kernle_instance, capsys):
        """kernle process status --json outputs JSON."""
        from kernle.cli.commands.process import cmd_process

        k = kernle_instance

        k._storage.list_raw = Mock(return_value=[])
        k._storage.get_episodes = Mock(return_value=[])
        k._storage.get_beliefs = Mock(return_value=[])

        args = Mock()
        args.process_action = "status"
        args.json = True

        cmd_process(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "unprocessed_raw" in output
        assert "unprocessed_episodes" in output
        assert "triggers" in output

    def test_process_invalid_transition(self, kernle_instance, capsys):
        """kernle process run with invalid transition shows error."""
        from kernle.cli.commands.process import cmd_process

        k = kernle_instance

        args = Mock()
        args.process_action = "run"
        args.transition = "invalid_transition"
        args.force = False
        args.json = False

        cmd_process(args, k)

        captured = capsys.readouterr()
        assert "Invalid transition" in captured.out

    def test_process_no_triggers_without_force(self, kernle_instance, capsys):
        """kernle process run without triggers hints about --force."""
        from kernle.cli.commands.process import cmd_process

        k = kernle_instance

        mock_entity = Mock()
        mock_entity.process.return_value = []
        k._entity = mock_entity

        args = Mock()
        args.process_action = "run"
        args.transition = None
        args.force = False
        args.json = False

        cmd_process(args, k)

        captured = capsys.readouterr()
        assert "--force" in captured.out


# =============================================================================
# Default config persistence test
# =============================================================================


class TestDefaultConfigPersistence:
    """Test that Entity.process() loads saved configs."""

    def test_entity_process_loads_saved_config(self):
        """Entity.process() loads config from stack."""
        from kernle.entity import Entity

        entity = Entity(core_id="test-config")

        # Mock stack with saved config
        mock_stack = Mock()
        mock_stack.get_processing_config.return_value = [
            {
                "layer_transition": "raw_to_episode",
                "enabled": True,
                "quantity_threshold": 20,
                "batch_size": 5,
            }
        ]

        # Mock inference
        mock_inference = Mock()

        entity._get_inference_service = Mock(return_value=mock_inference)
        entity._require_active_stack = Mock(return_value=mock_stack)

        # entity.process will create MemoryProcessor and load config
        # Since we mock everything, it should complete without error
        with patch("kernle.processing.MemoryProcessor") as mock_processor_cls:
            instance = mock_processor_cls.return_value
            instance.process.return_value = []

            entity.process(force=True)

            # Verify MemoryProcessor was created and config was loaded
            mock_processor_cls.assert_called_once()
            # Verify update_config was called with the saved config
            instance.update_config.assert_called_once()
            call_args = instance.update_config.call_args
            assert call_args[0][0] == "raw_to_episode"
            lc = call_args[0][1]
            assert lc.quantity_threshold == 20
            assert lc.batch_size == 5

    def test_entity_process_no_model_gates_all_transitions(self):
        """Entity.process() without model blocks all transitions."""
        from kernle.entity import Entity
        from kernle.processing import VALID_TRANSITIONS

        entity = Entity(core_id="test-no-model")

        mock_stack = Mock()
        mock_stack.get_processing_config.return_value = []
        mock_stack._backend.list_raw.return_value = []
        mock_stack.get_episodes.return_value = []
        mock_stack.get_beliefs.return_value = []
        entity._require_active_stack = Mock(return_value=mock_stack)
        entity._get_inference_service = Mock(return_value=None)

        results = entity.process(force=True)
        blocked = [r for r in results if r.inference_blocked]
        assert len(blocked) == len(VALID_TRANSITIONS)
