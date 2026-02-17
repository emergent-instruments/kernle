"""kernle model implementations.

Concrete ModelProtocol implementations for various providers.
Discovered via the ``kernle.models`` entry point group.
"""

from __future__ import annotations

from kernle.models.adapters import CallableModelAdapter
from kernle.models.anthropic import AnthropicModel
from kernle.models.ollama import OllamaModel

__all__ = [
    "AnthropicModel",
    "CallableModelAdapter",
    "OllamaModel",
]
