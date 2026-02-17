"""
Kernle - Stratified memory for synthetic intelligences.

Memory sovereignty for synthetic intelligences.
"""

from .core import Kernle
from .entity import Entity
from .models.adapters import CallableModelAdapter

try:
    from importlib.metadata import version

    __version__ = version("kernle")
except Exception:
    __version__ = "0.0.0"

__all__ = ["CallableModelAdapter", "Entity", "Kernle"]
