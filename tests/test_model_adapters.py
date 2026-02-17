"""Tests for inference passthrough model adapters.

Covers:
- CallableModelAdapter: wraps (prompt, system) -> str callables
- SamplingModelAdapter: bridges MCP sampling to ModelProtocol
- _get_mcp_session: safe session retrieval
- _maybe_bind_model: idempotent model binding
- call_tool dispatcher: executor dispatch for built-ins, in-loop for plugins
- End-to-end library integration
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from kernle.models.adapters import CallableModelAdapter, SamplingModelAdapter
from kernle.protocols import ModelCapabilities, ModelMessage, ModelResponse

# =============================================================================
# CallableModelAdapter tests
# =============================================================================


class TestCallableModelAdapter:
    def test_calls_fn_with_prompt(self):
        """Callable receives concatenated messages as prompt."""
        received = {}

        def capture(prompt: str, system: Optional[str]) -> str:
            received["prompt"] = prompt
            received["system"] = system
            return "response"

        adapter = CallableModelAdapter(capture)
        msgs = [ModelMessage(role="user", content="hello")]
        adapter.generate(msgs)

        assert received["prompt"] == "USER: hello"
        assert received["system"] is None

    def test_passes_system(self):
        """System prompt passed as second arg."""
        received = {}

        def capture(prompt: str, system: Optional[str]) -> str:
            received["system"] = system
            return "ok"

        adapter = CallableModelAdapter(capture)
        adapter.generate(
            [ModelMessage(role="user", content="hi")],
            system="be helpful",
        )
        assert received["system"] == "be helpful"

    def test_returns_model_response(self):
        """Return type is ModelResponse with .content."""
        adapter = CallableModelAdapter(lambda p, s: "the answer")
        result = adapter.generate([ModelMessage(role="user", content="q")])

        assert isinstance(result, ModelResponse)
        assert result.content == "the answer"

    def test_capabilities(self):
        """model_id/provider reflected in .capabilities."""
        adapter = CallableModelAdapter(lambda p, s: "", model_id="gpt-4o", provider="openai")
        caps = adapter.capabilities

        assert isinstance(caps, ModelCapabilities)
        assert caps.model_id == "gpt-4o"
        assert caps.provider == "openai"
        assert caps.supports_streaming is False

    def test_model_id_property(self):
        adapter = CallableModelAdapter(lambda p, s: "", model_id="my-model")
        assert adapter.model_id == "my-model"

    def test_stream_raises(self):
        """stream() raises NotImplementedError."""
        adapter = CallableModelAdapter(lambda p, s: "")
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            adapter.stream()

    def test_multiple_messages(self):
        """3-message list flattened correctly."""
        received = {}

        def capture(prompt: str, system: Optional[str]) -> str:
            received["prompt"] = prompt
            return "ok"

        adapter = CallableModelAdapter(capture)
        msgs = [
            ModelMessage(role="user", content="hello"),
            ModelMessage(role="assistant", content="hi there"),
            ModelMessage(role="user", content="how are you?"),
        ]
        adapter.generate(msgs)

        assert received["prompt"] == "USER: hello\n\nASSISTANT: hi there\n\nUSER: how are you?"

    def test_message_flattening_is_lossy(self):
        """Documents known limitation: only role+content, no structured data."""
        received = {}

        def capture(prompt: str, system: Optional[str]) -> str:
            received["prompt"] = prompt
            return "ok"

        adapter = CallableModelAdapter(capture)
        # ModelMessage with tool_calls — the adapter ignores them
        msgs = [
            ModelMessage(
                role="assistant",
                content="I'll call a tool",
                tool_calls=[{"id": "t1", "function": {"name": "search"}}],
            ),
        ]
        adapter.generate(msgs)

        # Only role and content survive — tool_calls not in prompt
        assert "search" not in received["prompt"]
        assert "ASSISTANT: I'll call a tool" in received["prompt"]


# =============================================================================
# SamplingModelAdapter tests
# =============================================================================


class TestSamplingModelAdapter:
    def _make_adapter(self, session=None, loop=None):
        """Helper to create adapter with defaults."""
        if session is None:
            session = MagicMock()
        if loop is None:
            loop = asyncio.new_event_loop()
        return SamplingModelAdapter(session=session, loop=loop), session, loop

    def test_calls_create_message(self):
        """create_message called with correct SamplingMessage list."""
        loop = asyncio.new_event_loop()

        async def fake_create_message(**kwargs):
            from mcp.types import TextContent as MCPTextContent

            return SimpleNamespace(
                content=MCPTextContent(type="text", text="response"),
                model="claude-sonnet-4-5-20250929",
                stopReason="end_turn",
            )

        session = MagicMock()
        session.create_message = MagicMock(side_effect=fake_create_message)

        adapter = SamplingModelAdapter(session=session, loop=loop)
        msgs = [ModelMessage(role="user", content="test")]

        # Run from a thread (as in executor)
        def run():
            return adapter.generate(msgs)

        # Run the loop in a thread so run_coroutine_threadsafe works
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        try:
            adapter.generate(msgs)
            assert session.create_message.called
            call_kwargs = session.create_message.call_args
            assert call_kwargs.kwargs["max_tokens"] == 4096
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)
            loop.close()

    def test_passes_max_tokens(self):
        """max_tokens forwarded."""
        loop = asyncio.new_event_loop()

        async def fake_create_message(**kwargs):
            from mcp.types import TextContent as MCPTextContent

            return SimpleNamespace(
                content=MCPTextContent(type="text", text="ok"),
            )

        session = MagicMock()
        session.create_message = MagicMock(side_effect=fake_create_message)

        adapter = SamplingModelAdapter(session=session, loop=loop)

        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        try:
            adapter.generate(
                [ModelMessage(role="user", content="test")],
                max_tokens=2048,
            )
            call_kwargs = session.create_message.call_args
            assert call_kwargs.kwargs["max_tokens"] == 2048
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)
            loop.close()

    def test_passes_system_prompt(self):
        """system forwarded as system_prompt."""
        loop = asyncio.new_event_loop()

        async def fake_create_message(**kwargs):
            from mcp.types import TextContent as MCPTextContent

            return SimpleNamespace(
                content=MCPTextContent(type="text", text="ok"),
            )

        session = MagicMock()
        session.create_message = MagicMock(side_effect=fake_create_message)

        adapter = SamplingModelAdapter(session=session, loop=loop)

        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        try:
            adapter.generate(
                [ModelMessage(role="user", content="test")],
                system="be concise",
            )
            call_kwargs = session.create_message.call_args
            assert call_kwargs.kwargs["system_prompt"] == "be concise"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)
            loop.close()

    def test_returns_model_response(self):
        """Text content extracted; model_id optional."""
        loop = asyncio.new_event_loop()

        async def fake_create_message(**kwargs):
            from mcp.types import TextContent as MCPTextContent

            return SimpleNamespace(
                content=MCPTextContent(type="text", text="the answer"),
                model="claude-sonnet-4-5-20250929",
                stopReason="end_turn",
            )

        session = MagicMock()
        session.create_message = MagicMock(side_effect=fake_create_message)
        adapter = SamplingModelAdapter(session=session, loop=loop)

        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        try:
            result = adapter.generate([ModelMessage(role="user", content="q")])
            assert isinstance(result, ModelResponse)
            assert result.content == "the answer"
            assert result.model_id == "claude-sonnet-4-5-20250929"
            assert result.stop_reason == "end_turn"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)
            loop.close()

    def test_model_id_falls_back_to_default(self):
        """When result has no model attr, uses adapter default."""
        loop = asyncio.new_event_loop()

        async def fake_create_message(**kwargs):
            from mcp.types import TextContent as MCPTextContent

            return SimpleNamespace(
                content=MCPTextContent(type="text", text="ok"),
            )

        session = MagicMock()
        session.create_message = MagicMock(side_effect=fake_create_message)
        adapter = SamplingModelAdapter(session=session, loop=loop)

        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        try:
            result = adapter.generate([ModelMessage(role="user", content="q")])
            assert result.model_id == "mcp-sampling"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)
            loop.close()

    def test_non_text_content_raises_valueerror(self):
        """ImageContent result -> ValueError, not KeyError/AttributeError."""
        loop = asyncio.new_event_loop()

        async def fake_create_message(**kwargs):
            # Return non-text content
            return SimpleNamespace(
                content=SimpleNamespace(type="image", data="abc"),
            )

        session = MagicMock()
        session.create_message = MagicMock(side_effect=fake_create_message)
        adapter = SamplingModelAdapter(session=session, loop=loop)

        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        try:
            with pytest.raises(ValueError, match="non-text content"):
                adapter.generate([ModelMessage(role="user", content="q")])
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)
            loop.close()

    def test_stream_raises(self):
        """stream() raises NotImplementedError."""
        adapter, _, loop = self._make_adapter()
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            adapter.stream()
        loop.close()

    def test_capabilities(self):
        adapter, _, loop = self._make_adapter()
        caps = adapter.capabilities
        assert caps.model_id == "mcp-sampling"
        assert caps.provider == "mcp"
        assert caps.supports_streaming is False
        loop.close()


# =============================================================================
# _get_mcp_session tests
# =============================================================================


class TestGetMcpSession:
    def test_returns_session_when_available(self):
        """Mock request_ctx.get().session -> returns session object."""
        mock_session = MagicMock()
        mock_ctx = SimpleNamespace(session=mock_session)

        with patch(
            "kernle.mcp.server.request_ctx",
            create=True,
        ):
            # We need to patch at the import location inside _get_mcp_session
            with patch("mcp.server.lowlevel.server.request_ctx") as patched_ctx:
                patched_ctx.get.return_value = mock_ctx

                from kernle.mcp.server import _get_mcp_session

                result = _get_mcp_session()
                assert result is mock_session

    def test_returns_none_on_import_error(self):
        """Import failure -> returns None, no exception."""
        from kernle.mcp.server import _get_mcp_session

        with patch.dict("sys.modules", {"mcp.server.lowlevel.server": None}):
            # Force import error by removing the module
            import sys

            saved = sys.modules.get("mcp.server.lowlevel.server")
            sys.modules["mcp.server.lowlevel.server"] = None  # type: ignore
            try:
                result = _get_mcp_session()
                assert result is None
            finally:
                if saved is not None:
                    sys.modules["mcp.server.lowlevel.server"] = saved
                else:
                    sys.modules.pop("mcp.server.lowlevel.server", None)

    def test_returns_none_on_lookup_error(self):
        """request_ctx.get() raises LookupError -> returns None."""
        from kernle.mcp.server import _get_mcp_session

        with patch("mcp.server.lowlevel.server.request_ctx") as patched_ctx:
            patched_ctx.get.side_effect = LookupError("no context")
            result = _get_mcp_session()
            assert result is None

    def test_returns_none_on_any_exception(self):
        """Generic exception -> returns None (never raises)."""
        from kernle.mcp.server import _get_mcp_session

        with patch("mcp.server.lowlevel.server.request_ctx") as patched_ctx:
            patched_ctx.get.side_effect = RuntimeError("unexpected")
            result = _get_mcp_session()
            assert result is None


# =============================================================================
# _maybe_bind_model tests
# =============================================================================


class TestMaybeBindModel:
    def _make_mock_kernle(self, *, has_model: bool = False) -> MagicMock:
        k = MagicMock()
        if has_model:
            k.entity.model = MagicMock()
            k.entity.model.model_id = "existing-model"
        else:
            k.entity.model = None
        return k

    def test_noop_if_already_set(self):
        """Model not replaced; no load_persisted_model called."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=True)
        loop = asyncio.new_event_loop()

        with patch("kernle.mcp.server.load_persisted_model", create=True):
            _maybe_bind_model(k, None, loop)

        k.entity.set_model.assert_not_called()
        loop.close()

    def test_logs_already_set(self, caplog):
        """Log message emitted with model_id."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=True)
        loop = asyncio.new_event_loop()

        import logging

        with caplog.at_level(logging.DEBUG, logger="kernle.mcp.server"):
            _maybe_bind_model(k, None, loop)

        assert any("already bound" in r.message.lower() for r in caplog.records)
        loop.close()

    def test_uses_persisted_config_first(self):
        """boot_config model loaded; sampling skipped."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=False)
        mock_persisted = MagicMock()
        mock_persisted.model_id = "anthropic-claude"
        loop = asyncio.new_event_loop()

        with patch(
            "kernle.cli.commands.model.load_persisted_model",
            return_value=mock_persisted,
        ):
            _maybe_bind_model(k, None, loop)

        k.entity.set_model.assert_called_once_with(mock_persisted)
        loop.close()

    def test_persisted_fails_falls_through_to_sampling(self, caplog):
        """load_persisted_model raises -> tries sampling; warning logged."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=False)
        loop = asyncio.new_event_loop()

        mock_session = MagicMock()
        mock_session.check_client_capability = MagicMock(return_value=True)

        import logging

        with (
            patch(
                "kernle.cli.commands.model.load_persisted_model",
                side_effect=RuntimeError("api key missing"),
            ),
            caplog.at_level(logging.WARNING, logger="kernle.mcp.server"),
        ):
            _maybe_bind_model(k, mock_session, loop)

        # Should have fallen through to sampling
        assert k.entity.set_model.called
        assert any("falling through to sampling" in r.message.lower() for r in caplog.records)
        loop.close()

    def test_persisted_returns_none_tries_sampling(self):
        """Returns None -> sampling attempted."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=False)
        loop = asyncio.new_event_loop()

        mock_session = MagicMock()
        mock_session.check_client_capability = MagicMock(return_value=True)

        with patch(
            "kernle.cli.commands.model.load_persisted_model",
            return_value=None,
        ):
            _maybe_bind_model(k, mock_session, loop)

        # Should bind sampling adapter
        assert k.entity.set_model.called
        call_arg = k.entity.set_model.call_args[0][0]
        assert isinstance(call_arg, SamplingModelAdapter)
        loop.close()

    def test_uses_sampling_if_supported(self):
        """check_client_capability True -> SamplingModelAdapter bound."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=False)
        loop = asyncio.new_event_loop()

        mock_session = MagicMock()
        mock_session.check_client_capability = MagicMock(return_value=True)

        with patch(
            "kernle.cli.commands.model.load_persisted_model",
            return_value=None,
        ):
            _maybe_bind_model(k, mock_session, loop)

        assert k.entity.set_model.called
        adapter = k.entity.set_model.call_args[0][0]
        assert isinstance(adapter, SamplingModelAdapter)
        assert adapter.model_id == "mcp-sampling"
        loop.close()

    def test_skips_sampling_if_not_supported(self, caplog):
        """Capability check False -> no model; info logged."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=False)
        loop = asyncio.new_event_loop()

        mock_session = MagicMock()
        mock_session.check_client_capability = MagicMock(return_value=False)

        import logging

        with (
            patch(
                "kernle.cli.commands.model.load_persisted_model",
                return_value=None,
            ),
            caplog.at_level(logging.DEBUG, logger="kernle.mcp.server"),
        ):
            _maybe_bind_model(k, mock_session, loop)

        k.entity.set_model.assert_not_called()
        loop.close()

    def test_no_session_skips_sampling(self, caplog):
        """session=None -> capture-only; no exception."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=False)
        loop = asyncio.new_event_loop()

        import logging

        with (
            patch(
                "kernle.cli.commands.model.load_persisted_model",
                return_value=None,
            ),
            caplog.at_level(logging.DEBUG, logger="kernle.mcp.server"),
        ):
            _maybe_bind_model(k, None, loop)

        k.entity.set_model.assert_not_called()
        assert any("capture-only" in r.message.lower() for r in caplog.records)
        loop.close()

    def test_sampling_fails_gracefully(self, caplog):
        """Sampling setup raises -> warning logged; continues."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=False)
        loop = asyncio.new_event_loop()

        mock_session = MagicMock()
        mock_session.check_client_capability = MagicMock(side_effect=RuntimeError("broken"))

        import logging

        with (
            patch(
                "kernle.cli.commands.model.load_persisted_model",
                return_value=None,
            ),
            caplog.at_level(logging.WARNING, logger="kernle.mcp.server"),
        ):
            # Should not raise
            _maybe_bind_model(k, mock_session, loop)

        k.entity.set_model.assert_not_called()
        assert any("capture-only" in r.message.lower() for r in caplog.records)
        loop.close()

    def test_malformed_persisted_plus_unsupported_sampling(self, caplog):
        """Both fail -> capture-only; both warnings logged."""
        from kernle.mcp.server import _maybe_bind_model

        k = self._make_mock_kernle(has_model=False)
        loop = asyncio.new_event_loop()

        mock_session = MagicMock()
        mock_session.check_client_capability = MagicMock(return_value=False)

        import logging

        with (
            patch(
                "kernle.cli.commands.model.load_persisted_model",
                side_effect=ValueError("bad config"),
            ),
            caplog.at_level(logging.DEBUG, logger="kernle.mcp.server"),
        ):
            _maybe_bind_model(k, mock_session, loop)

        k.entity.set_model.assert_not_called()
        # Should have warning about persisted failure
        assert any("falling through to sampling" in r.message.lower() for r in caplog.records)
        loop.close()


# =============================================================================
# call_tool dispatcher tests
# =============================================================================


class TestCallToolDispatcher:
    @pytest.mark.asyncio
    async def test_builtin_runs_in_executor(self):
        """Thread ID inside handler != event loop thread ID."""
        handler_thread_ids: list[int] = []
        event_loop_thread_id = threading.current_thread().ident

        def fake_handler(args: dict, k: Any) -> str:
            handler_thread_ids.append(threading.current_thread().ident)
            return "ok"

        with (
            patch("kernle.mcp.server.HANDLERS", {"test_tool": fake_handler}),
            patch("kernle.mcp.server.VALIDATORS", {"test_tool": lambda x: x}),
            patch("kernle.mcp.server.get_kernle", return_value=MagicMock()),
            patch("kernle.mcp.server._get_mcp_session", return_value=None),
            patch("kernle.mcp.server._maybe_bind_model"),
        ):
            from kernle.mcp.server import call_tool

            await call_tool("test_tool", {})

        assert len(handler_thread_ids) == 1
        assert handler_thread_ids[0] != event_loop_thread_id

    @pytest.mark.asyncio
    async def test_plugin_runs_in_loop(self):
        """Plugin handler runs in event loop thread."""
        handler_thread_ids: list[int] = []
        event_loop_thread_id = threading.current_thread().ident

        def fake_plugin(args: dict) -> str:
            handler_thread_ids.append(threading.current_thread().ident)
            return "plugin result"

        with (
            patch("kernle.mcp.server.HANDLERS", {}),
            patch("kernle.mcp.server._plugin_handlers", {"test_plugin": fake_plugin}),
            patch(
                "kernle.mcp.server.validate_tool_input",
                side_effect=lambda name, args: args,
            ),
        ):
            from kernle.mcp.server import call_tool

            await call_tool("test_plugin", {})

        assert len(handler_thread_ids) == 1
        # Plugin runs in event loop thread
        assert handler_thread_ids[0] == event_loop_thread_id

    @pytest.mark.asyncio
    async def test_plugin_does_not_receive_kernle(self):
        """Plugin handler called with only (args,) not (args, k)."""
        call_args_list: list[tuple] = []

        def fake_plugin(*args: Any) -> str:
            call_args_list.append(args)
            return "ok"

        with (
            patch("kernle.mcp.server.HANDLERS", {}),
            patch("kernle.mcp.server._plugin_handlers", {"test_plugin": fake_plugin}),
            patch(
                "kernle.mcp.server.validate_tool_input",
                side_effect=lambda name, args: args,
            ),
        ):
            from kernle.mcp.server import call_tool

            await call_tool("test_plugin", {"key": "val"})

        assert len(call_args_list) == 1
        # Plugin receives only (args,) — no k parameter
        assert len(call_args_list[0]) == 1
        assert call_args_list[0][0] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_session_unavailable_does_not_crash(self):
        """_get_mcp_session returns None -> handler runs, capture-only."""

        def fake_handler(args: dict, k: Any) -> str:
            return "works"

        with (
            patch("kernle.mcp.server.HANDLERS", {"test_tool": fake_handler}),
            patch("kernle.mcp.server.VALIDATORS", {"test_tool": lambda x: x}),
            patch("kernle.mcp.server.get_kernle", return_value=MagicMock()),
            patch("kernle.mcp.server._get_mcp_session", return_value=None),
            patch("kernle.mcp.server._maybe_bind_model") as mock_bind,
        ):
            from kernle.mcp.server import call_tool

            result = await call_tool("test_tool", {})

        assert result[0].text == "works"
        # _maybe_bind_model was still called (with session=None)
        mock_bind.assert_called_once()

    @pytest.mark.asyncio
    async def test_lock_serializes_concurrent_calls(self):
        """Two concurrent built-in calls complete in order; shared counter not corrupted."""
        import time

        counter = {"value": 0}
        results: list[int] = []

        def slow_handler(args: dict, k: Any) -> str:
            current = counter["value"]
            time.sleep(0.05)  # simulate work
            counter["value"] = current + 1
            results.append(counter["value"])
            return str(counter["value"])

        with (
            patch("kernle.mcp.server.HANDLERS", {"test_tool": slow_handler}),
            patch("kernle.mcp.server.VALIDATORS", {"test_tool": lambda x: x}),
            patch("kernle.mcp.server.get_kernle", return_value=MagicMock()),
            patch("kernle.mcp.server._get_mcp_session", return_value=None),
            patch("kernle.mcp.server._maybe_bind_model"),
        ):
            from kernle.mcp.server import call_tool

            # Launch two concurrent calls
            task1 = asyncio.create_task(call_tool("test_tool", {}))
            task2 = asyncio.create_task(call_tool("test_tool", {}))
            await asyncio.gather(task1, task2)

        # Without lock, counter would be corrupted (both read 0, both write 1)
        # With lock, we get [1, 2]
        assert sorted(results) == [1, 2]

    @pytest.mark.asyncio
    async def test_get_kernle_not_called_in_thread(self):
        """get_kernle called exactly once (in async scope), not inside worker thread."""
        get_kernle_thread_ids: list[int] = []
        event_loop_thread_id = threading.current_thread().ident

        def tracked_get_kernle():
            get_kernle_thread_ids.append(threading.current_thread().ident)
            return MagicMock()

        def fake_handler(args: dict, k: Any) -> str:
            return "ok"

        with (
            patch("kernle.mcp.server.HANDLERS", {"test_tool": fake_handler}),
            patch("kernle.mcp.server.VALIDATORS", {"test_tool": lambda x: x}),
            patch("kernle.mcp.server.get_kernle", side_effect=tracked_get_kernle),
            patch("kernle.mcp.server._get_mcp_session", return_value=None),
            patch("kernle.mcp.server._maybe_bind_model"),
        ):
            from kernle.mcp.server import call_tool

            await call_tool("test_tool", {})

        assert len(get_kernle_thread_ids) == 1
        # get_kernle was called from the event loop thread, not executor
        assert get_kernle_thread_ids[0] == event_loop_thread_id


# =============================================================================
# End-to-end library tests
# =============================================================================


class TestLibraryIntegration:
    def test_callable_adapter_full_episode_workflow(self, tmp_path):
        """Attach adapter to Kernle, raw() succeeds."""
        from kernle.core import Kernle
        from kernle.storage import SQLiteStorage

        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(stack_id="test-adapter", db_path=db_path)
        k = Kernle(stack_id="test-adapter", storage=storage, strict=False)

        adapter = CallableModelAdapter(lambda p, s: '{"valence": 0.5}', model_id="test-model")
        k.entity.set_model(adapter)

        assert k.entity.model is not None
        assert k.entity.model.model_id == "test-model"

        rid = k.raw("test capture via adapter")
        assert rid, "raw capture should return an ID"

        storage.close()

    def test_no_adapter_no_crash(self, tmp_path):
        """Kernle without model, raw() succeeds; no exception."""
        from kernle.core import Kernle
        from kernle.storage import SQLiteStorage

        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(stack_id="test-no-model", db_path=db_path)
        k = Kernle(stack_id="test-no-model", storage=storage, strict=False)

        assert k.entity.model is None

        # raw() should work without a model
        rid = k.raw("capture without model")
        assert rid, "raw capture should succeed without model"

        storage.close()

    def test_no_adapter_raw_does_not_invoke_inference(self, tmp_path):
        """raw() without a model does not attempt inference."""
        from kernle.core import Kernle
        from kernle.storage import SQLiteStorage

        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(stack_id="test-no-infer", db_path=db_path)
        k = Kernle(stack_id="test-no-infer", storage=storage, strict=False)

        assert k.entity.model is None

        # raw() should succeed without a model — no inference path triggered
        rid = k.raw("test capture without inference")
        assert rid, "raw capture should succeed"

        # Verify no model was somehow attached
        assert k.entity.model is None

        storage.close()

    def test_top_level_imports(self):
        """CallableModelAdapter importable from kernle top level."""
        from kernle import CallableModelAdapter

        assert CallableModelAdapter is not None

    def test_sampling_adapter_not_in_top_level(self):
        """SamplingModelAdapter is internal — not exported at package top level."""
        import kernle

        assert not hasattr(kernle, "SamplingModelAdapter")

    def test_sampling_adapter_importable_from_subpackage(self):
        """SamplingModelAdapter still accessible via kernle.models.adapters."""
        from kernle.models.adapters import SamplingModelAdapter

        assert SamplingModelAdapter is not None
