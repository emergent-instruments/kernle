"""Model adapters for inference passthrough.

Two adapters that bridge external inference sources to Kernle's
ModelProtocol, eliminating the need for users to configure a
separate model binding.

- CallableModelAdapter: wraps any ``(prompt, system) -> str`` callable
  for library embedding (``from kernle import Kernle``).
- SamplingModelAdapter: bridges MCP sampling (host agent's model)
  to ModelProtocol for MCP server deployments.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Iterator, Optional

from kernle.protocols import (
    ModelCapabilities,
    ModelChunk,
    ModelMessage,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class CallableModelAdapter:
    """Wraps any callable as a ModelProtocol implementation.

    The callable receives two arguments: a flattened prompt string (all
    messages concatenated as ``"ROLE: content\\n\\n..."``) and an optional
    system prompt string. This is a simplification — for full message-
    structure fidelity, implement ModelProtocol directly.

    Usage::

        adapter = CallableModelAdapter(my_generate_fn, model_id="gpt-4o")
        k.entity.set_model(adapter)
    """

    def __init__(
        self,
        fn: Callable[[str, Optional[str]], str],
        *,
        model_id: str = "callable",
        provider: str = "external",
    ):
        self._fn = fn
        self._model_id = model_id
        self._provider = provider

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self._model_id,
            provider=self._provider,
            context_window=0,  # unknown for external callables
            supports_streaming=False,
        )

    def generate(
        self,
        messages: list[ModelMessage],
        *,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        prompt = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        text = self._fn(prompt, system)
        return ModelResponse(content=text)

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelChunk]:
        raise NotImplementedError("CallableModelAdapter does not support streaming")


class SamplingModelAdapter:
    """Bridges MCP sampling to ModelProtocol.

    Construct in the async ``call_tool`` scope with ``session`` and ``loop``
    captured from ``request_ctx``. Use from within a ``run_in_executor``
    thread — calling ``generate()`` from the event loop thread will deadlock.

    Args:
        session: The MCP ServerSession from ``request_ctx.get().session``.
        loop: The asyncio event loop (``asyncio.get_event_loop()`` in async scope).
        model_id: Identifier for this adapter (default: ``"mcp-sampling"``).
        provider: Provider label (default: ``"mcp"``).
    """

    def __init__(
        self,
        session: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        model_id: str = "mcp-sampling",
        provider: str = "mcp",
    ):
        self._session = session
        self._loop = loop
        self._model_id = model_id
        self._provider = provider

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self._model_id,
            provider=self._provider,
            context_window=0,  # unknown for MCP sampling
            supports_streaming=False,
        )

    def generate(
        self,
        messages: list[ModelMessage],
        *,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ModelResponse:
        from mcp.types import SamplingMessage
        from mcp.types import TextContent as MCPTextContent

        sampling_msgs = [
            SamplingMessage(
                role=m.role,
                content=MCPTextContent(type="text", text=m.content),
            )
            for m in messages
        ]
        coro = self._session.create_message(
            messages=sampling_msgs,
            max_tokens=max_tokens,
            system_prompt=system,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        result = future.result(timeout=60)

        # result.content is a union: TextContent | ImageContent | AudioContent
        if not isinstance(result.content, MCPTextContent):
            raise ValueError(
                f"MCP sampling returned non-text content: {type(result.content).__name__}. "
                "Kernle inference requires a text response."
            )

        # model and stop_reason may be absent depending on MCP client
        model_id = getattr(result, "model", None) or self._model_id
        return ModelResponse(
            content=result.content.text,
            model_id=model_id,
            stop_reason=getattr(result, "stopReason", None),
        )

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelChunk]:
        raise NotImplementedError("SamplingModelAdapter does not support streaming")
