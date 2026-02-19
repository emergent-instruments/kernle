"""kernle.stack - Memory stack implementations.

The default Stack accepts any Storage protocol backend.
Use Stack.from_sqlite() for the common local-agent case.
"""

from kernle.stack.sqlite_stack import Stack

__all__ = ["Stack"]
