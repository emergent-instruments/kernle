"""Tests for ModelProtocol implementations (Anthropic + Ollama).

All tests work without actual API access — SDK/HTTP calls are mocked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kernle.protocols import (
    ModelCapabilities,
    ModelMessage,
    ModelProtocol,
    ModelResponse,
    ToolDefinition,
)

# =============================================================================
# AnthropicModel tests
# =============================================================================


class TestAnthropicModelProperties:
    """Property and construction tests for AnthropicModel."""

    def test_model_id(self, anthropic_model):
        assert anthropic_model.model_id == "claude-sonnet-4-5-20250929"

    def test_model_id_custom(self):
        with _mock_anthropic_sdk():
            from kernle.models.anthropic import AnthropicModel

            model = AnthropicModel(model_id="claude-opus-4-20250514", api_key="test-key")
        assert model.model_id == "claude-opus-4-20250514"

    def test_capabilities(self, anthropic_model):
        caps = anthropic_model.capabilities
        assert isinstance(caps, ModelCapabilities)
        assert caps.provider == "anthropic"
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        assert caps.context_window == 200_000

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
        mock_module = _MockAnthropicModule()
        with patch.dict("sys.modules", {"anthropic": mock_module}):
            from kernle.models.anthropic import AnthropicModel

            model = AnthropicModel()
        mock_module.Anthropic.assert_called_once_with(api_key="env-key-123")
        assert model is not None

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
        with _mock_anthropic_sdk():
            from kernle.models.anthropic import AnthropicModel

            with pytest.raises(ValueError, match="API key is required"):
                AnthropicModel()

    def test_missing_anthropic_raises_import_error(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            # Force re-import to trigger ImportError
            import importlib

            import kernle.models.anthropic as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="anthropic"):
                mod.AnthropicModel(api_key="test-key")


class TestAnthropicModelGenerate:
    """generate() tests for AnthropicModel."""

    def test_generate_converts_messages_correctly(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.create.return_value = _make_anthropic_response("Hi there")

        messages = [
            ModelMessage(role="user", content="Hello"),
        ]
        anthropic_model.generate(messages)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_generate_extracts_system_message(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.create.return_value = _make_anthropic_response("response")

        messages = [
            ModelMessage(role="system", content="You are helpful."),
            ModelMessage(role="user", content="Hi"),
        ]
        anthropic_model.generate(messages)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."
        # System message should NOT be in the messages list
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_generate_system_param_merged_with_system_message(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.create.return_value = _make_anthropic_response("response")

        messages = [
            ModelMessage(role="system", content="From message."),
            ModelMessage(role="user", content="Hi"),
        ]
        anthropic_model.generate(messages, system="From param.")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "From param." in call_kwargs["system"]
        assert "From message." in call_kwargs["system"]

    def test_generate_returns_model_response(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.create.return_value = _make_anthropic_response(
            "Hello!",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
            model="claude-sonnet-4-5-20250929",
        )

        result = anthropic_model.generate([ModelMessage(role="user", content="Hi")])
        assert isinstance(result, ModelResponse)
        assert result.content == "Hello!"
        assert result.usage == {"input_tokens": 10, "output_tokens": 5}
        assert result.stop_reason == "end_turn"
        assert result.model_id == "claude-sonnet-4-5-20250929"

    def test_generate_with_tools(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.create.return_value = _make_anthropic_response("ok")

        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get the weather",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ]
        anthropic_model.generate(
            [ModelMessage(role="user", content="Weather?")],
            tools=tools,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "get_weather"
        assert call_kwargs["tools"][0]["input_schema"]["type"] == "object"

    def test_generate_handles_tool_use_response(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.create.return_value = _make_anthropic_response_with_tool_use(
            tool_id="toolu_123",
            tool_name="get_weather",
            tool_input={"city": "London"},
        )

        result = anthropic_model.generate([ModelMessage(role="user", content="Weather?")])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "toolu_123"
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["input"] == {"city": "London"}

    def test_generate_with_temperature_and_max_tokens(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.create.return_value = _make_anthropic_response("ok")

        anthropic_model.generate(
            [ModelMessage(role="user", content="Hi")],
            temperature=0.5,
            max_tokens=100,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_generate_api_error_wrapped(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.create.side_effect = RuntimeError("rate limited")

        with pytest.raises(Exception, match="Anthropic API error"):
            anthropic_model.generate([ModelMessage(role="user", content="Hi")])


class TestAnthropicModelStream:
    """stream() tests for AnthropicModel."""

    def test_stream_yields_chunks(self, anthropic_model):
        mock_client = anthropic_model._client
        _setup_mock_stream(mock_client, ["Hello", " world"])

        chunks = list(anthropic_model.stream([ModelMessage(role="user", content="Hi")]))
        # text chunks + final
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[0].is_final is False
        assert chunks[1].is_final is False

    def test_stream_final_chunk_has_usage(self, anthropic_model):
        mock_client = anthropic_model._client
        _setup_mock_stream(mock_client, ["text"], input_tokens=8, output_tokens=3)

        chunks = list(anthropic_model.stream([ModelMessage(role="user", content="Hi")]))
        final = chunks[-1]
        assert final.is_final is True
        assert final.usage == {"input_tokens": 8, "output_tokens": 3}

    def test_stream_api_error_wrapped(self, anthropic_model):
        mock_client = anthropic_model._client
        mock_client.messages.stream.side_effect = RuntimeError("connection lost")

        with pytest.raises(Exception, match="Anthropic streaming error"):
            list(anthropic_model.stream([ModelMessage(role="user", content="Hi")]))


# =============================================================================
# OllamaModel tests
# =============================================================================


class TestOllamaModelProperties:
    """Property and construction tests for OllamaModel."""

    def test_model_id(self, ollama_model):
        assert ollama_model.model_id == "llama3.2:latest"

    def test_model_id_custom(self):
        with _mock_requests():
            from kernle.models.ollama import OllamaModel

            model = OllamaModel(model_id="mistral:7b")
        assert model.model_id == "mistral:7b"

    def test_capabilities(self, ollama_model):
        caps = ollama_model.capabilities
        assert isinstance(caps, ModelCapabilities)
        assert caps.provider == "ollama"
        assert caps.supports_tools is False
        assert caps.supports_vision is False
        assert caps.supports_streaming is True
        assert caps.context_window == 8192

    def test_custom_base_url(self):
        with _mock_requests():
            from kernle.models.ollama import OllamaModel

            model = OllamaModel(base_url="http://myhost:9999")
        assert model._base_url == "http://myhost:9999"

    def test_custom_context_window(self):
        with _mock_requests():
            from kernle.models.ollama import OllamaModel

            model = OllamaModel(context_window=32768)
        assert model.capabilities.context_window == 32768

    def test_missing_requests_raises_import_error(self):
        with patch.dict("sys.modules", {"requests": None}):
            import importlib

            import kernle.models.ollama as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="requests"):
                mod.OllamaModel()


class TestOllamaModelGenerate:
    """generate() tests for OllamaModel."""

    def test_generate_sends_correct_request(self, ollama_model, mock_requests_post):
        mock_requests_post.return_value = _make_ollama_response("Hello!")

        ollama_model.generate([ModelMessage(role="user", content="Hi")])

        mock_requests_post.assert_called_once()
        call_args = mock_requests_post.call_args
        assert "/api/chat" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3.2:latest"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]

    def test_generate_returns_model_response(self, ollama_model, mock_requests_post):
        mock_requests_post.return_value = _make_ollama_response(
            "Hello!",
            prompt_eval_count=10,
            eval_count=5,
        )

        result = ollama_model.generate([ModelMessage(role="user", content="Hi")])
        assert isinstance(result, ModelResponse)
        assert result.content == "Hello!"
        assert result.usage == {"input_tokens": 10, "output_tokens": 5}
        assert result.stop_reason == "stop"

    def test_generate_with_system_param(self, ollama_model, mock_requests_post):
        mock_requests_post.return_value = _make_ollama_response("ok")

        ollama_model.generate(
            [ModelMessage(role="user", content="Hi")],
            system="Be concise.",
        )

        payload = mock_requests_post.call_args[1]["json"]
        # System should be prepended as first message
        assert payload["messages"][0] == {"role": "system", "content": "Be concise."}
        assert payload["messages"][1] == {"role": "user", "content": "Hi"}

    def test_generate_with_temperature(self, ollama_model, mock_requests_post):
        mock_requests_post.return_value = _make_ollama_response("ok")

        ollama_model.generate(
            [ModelMessage(role="user", content="Hi")],
            temperature=0.7,
        )

        payload = mock_requests_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.7

    def test_connection_error_handling(self, ollama_model, mock_requests_post):
        mock_requests_post.side_effect = ollama_model._requests.ConnectionError("refused")

        with pytest.raises(Exception, match="Cannot connect to Ollama"):
            ollama_model.generate([ModelMessage(role="user", content="Hi")])

    def test_timeout_error_handling(self, ollama_model, mock_requests_post):
        mock_requests_post.side_effect = ollama_model._requests.Timeout("timed out")

        with pytest.raises(Exception, match="timed out"):
            ollama_model.generate([ModelMessage(role="user", content="Hi")])

    def test_non_200_response_handling(self, ollama_model, mock_requests_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_requests_post.return_value = mock_resp

        with pytest.raises(Exception, match="HTTP 500"):
            ollama_model.generate([ModelMessage(role="user", content="Hi")])


class TestOllamaModelStream:
    """stream() tests for OllamaModel."""

    def test_stream_yields_chunks(self, ollama_model, mock_requests_post):
        mock_requests_post.return_value = _make_ollama_stream_response(
            ["Hello", " world"],
        )

        chunks = list(ollama_model.stream([ModelMessage(role="user", content="Hi")]))
        assert len(chunks) == 3  # 2 content + 1 final
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[0].is_final is False
        assert chunks[1].is_final is False

    def test_stream_final_chunk(self, ollama_model, mock_requests_post):
        mock_requests_post.return_value = _make_ollama_stream_response(
            ["text"],
            prompt_eval_count=5,
            eval_count=2,
        )

        chunks = list(ollama_model.stream([ModelMessage(role="user", content="Hi")]))
        final = chunks[-1]
        assert final.is_final is True
        assert final.usage == {"input_tokens": 5, "output_tokens": 2}

    def test_stream_connection_error(self, ollama_model, mock_requests_post):
        mock_requests_post.side_effect = ollama_model._requests.ConnectionError("refused")

        with pytest.raises(Exception, match="Cannot connect"):
            list(ollama_model.stream([ModelMessage(role="user", content="Hi")]))

    def test_stream_non_200(self, ollama_model, mock_requests_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_requests_post.return_value = mock_resp

        with pytest.raises(Exception, match="HTTP 404"):
            list(ollama_model.stream([ModelMessage(role="user", content="Hi")]))


# =============================================================================
# Integration / Protocol conformance tests
# =============================================================================


class TestProtocolConformance:
    """Verify both models satisfy ModelProtocol isinstance checks."""

    def test_anthropic_isinstance_model_protocol(self, anthropic_model):
        assert isinstance(anthropic_model, ModelProtocol)

    def test_ollama_isinstance_model_protocol(self, ollama_model):
        assert isinstance(ollama_model, ModelProtocol)


class TestEntityIntegration:
    """Verify models wire correctly into Entity + InferenceService."""

    def test_entity_set_model_wires_inference(self, anthropic_model, tmp_path):
        """Setting a model on Entity creates an InferenceService."""
        from kernle.entity import Entity
        from kernle.stack.sqlite_stack import Stack

        entity = Entity(core_id="test-core")
        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        entity.attach_stack(stack)

        entity.set_model(anthropic_model)
        assert entity.model is anthropic_model

    def test_inference_service_infer_delegates_to_model(self, anthropic_model):
        """InferenceService.infer() routes to the model's generate()."""
        from kernle.inference import create_inference_service

        mock_client = anthropic_model._client
        mock_client.messages.create.return_value = _make_anthropic_response("answer here")

        svc = create_inference_service(anthropic_model)
        result = svc.infer("What is 2+2?")
        assert result == "answer here"
        mock_client.messages.create.assert_called_once()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def anthropic_model():
    """Create an AnthropicModel with a mocked SDK client."""
    with _mock_anthropic_sdk():
        from kernle.models.anthropic import AnthropicModel

        model = AnthropicModel(api_key="test-key-123")
    return model


@pytest.fixture
def ollama_model():
    """Create an OllamaModel with mocked requests module."""
    with _mock_requests():
        from kernle.models.ollama import OllamaModel

        model = OllamaModel()
    return model


@pytest.fixture
def mock_requests_post(ollama_model):
    """Patch requests.post on an existing OllamaModel instance."""
    with patch.object(ollama_model._requests, "post") as mock_post:
        yield mock_post


# =============================================================================
# Test helpers — Anthropic mocks
# =============================================================================


class _MockAnthropicModule:
    """Fake anthropic module for import mocking."""

    def __init__(self):
        self.Anthropic = MagicMock()


def _mock_anthropic_sdk():
    """Context manager that mocks the anthropic import."""
    mock_module = _MockAnthropicModule()
    return patch.dict("sys.modules", {"anthropic": mock_module})


def _make_anthropic_response(
    text: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-4-5-20250929",
) -> SimpleNamespace:
    """Create a fake Anthropic API response."""
    text_block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        content=[text_block],
        usage=usage,
        stop_reason=stop_reason,
        model=model,
    )


def _make_anthropic_response_with_tool_use(
    *,
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    text: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> SimpleNamespace:
    """Create a fake Anthropic response with a tool_use block."""
    blocks = []
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    blocks.append(SimpleNamespace(type="tool_use", id=tool_id, name=tool_name, input=tool_input))
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        content=blocks,
        usage=usage,
        stop_reason="tool_use",
        model="claude-sonnet-4-5-20250929",
    )


def _setup_mock_stream(
    mock_client: MagicMock,
    text_chunks: list[str],
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Configure mock_client.messages.stream() to yield text chunks."""
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    final_message = SimpleNamespace(usage=usage)

    # The stream context manager
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=stream_cm)
    stream_cm.__exit__ = MagicMock(return_value=False)
    stream_cm.text_stream = iter(text_chunks)
    stream_cm.get_final_message.return_value = final_message

    mock_client.messages.stream.return_value = stream_cm


# =============================================================================
# Test helpers — Ollama mocks
# =============================================================================


class _MockRequestsModule:
    """Fake requests module with exception classes."""

    class ConnectionError(Exception):
        pass

    class Timeout(Exception):  # noqa: N818 — matches requests.Timeout naming
        pass

    def post(self, *args, **kwargs):
        pass


def _mock_requests():
    """Context manager that mocks the requests import."""
    mock_module = _MockRequestsModule()
    return patch.dict("sys.modules", {"requests": mock_module})


def _make_ollama_response(
    content: str,
    *,
    prompt_eval_count: int = 0,
    eval_count: int = 0,
    model: str = "llama3.2:latest",
) -> MagicMock:
    """Create a fake requests.Response for Ollama non-streaming."""
    resp = MagicMock()
    resp.status_code = 200
    data: dict[str, Any] = {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    if prompt_eval_count:
        data["prompt_eval_count"] = prompt_eval_count
    if eval_count:
        data["eval_count"] = eval_count
    resp.json.return_value = data
    return resp


def _make_ollama_stream_response(
    text_chunks: list[str],
    *,
    prompt_eval_count: int = 0,
    eval_count: int = 0,
) -> MagicMock:
    """Create a fake streaming response for Ollama."""
    lines: list[str] = []
    for chunk in text_chunks:
        lines.append(
            json.dumps(
                {
                    "model": "llama3.2:latest",
                    "message": {"role": "assistant", "content": chunk},
                    "done": False,
                }
            )
        )
    # Final line
    final: dict[str, Any] = {
        "model": "llama3.2:latest",
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }
    if prompt_eval_count:
        final["prompt_eval_count"] = prompt_eval_count
    if eval_count:
        final["eval_count"] = eval_count
    lines.append(json.dumps(final))

    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp
