"""Noop inference model for import operations.

Imports happen only on empty stacks, so no real inference is needed.
The noop model satisfies the _require_inference() gate structurally,
allowing writers to run their full validation pipeline without bypass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle.core.kernle_class import Kernle


class _NoopImportModel:
    """Minimal model that satisfies the inference protocol for imports."""

    model_id = "noop-import-model"

    def generate(self, prompt, **kwargs):
        return ""


def bind_import_model(k: "Kernle") -> None:
    """Bind a no-op inference model to satisfy the inference gate during import."""
    from kernle.stack import Stack

    stack = k.stack
    if stack is None or not isinstance(stack, Stack):
        return
    if getattr(stack, "_inference", None) is not None:
        return  # Already has a model, don't override

    from kernle.inference import create_inference_service

    inference = create_inference_service(_NoopImportModel())
    stack.on_model_changed(inference)
