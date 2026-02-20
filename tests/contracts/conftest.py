"""Parameterized fixtures for contract tests.

Provides stack and storage fixtures parameterized by backend.
Future backends (e.g. kernle-supabase) add their factory to
STACK_FACTORIES / STORAGE_FACTORIES and get the full contract suite
for free.

Usage in external packages:

    # In your conftest.py:
    from kernle.storage.sqlite import SQLiteStorage
    from kernle.stack import Stack

    def _create_my_storage(tmp_path):
        return MyStorage(stack_id="test", ...)

    STORAGE_FACTORIES.append((_create_my_storage, "my-backend"))
"""

import pytest

from kernle.stack import Stack
from kernle.storage.sqlite import SQLiteStorage

STACK_ID = "contract-test-stack"


# ============================================================================
# Factory functions
# ============================================================================


def _create_sqlite_storage(tmp_path):
    """Create a SQLiteStorage instance for contract testing."""
    return SQLiteStorage(stack_id=STACK_ID, db_path=tmp_path / "contract_test.db")


def _create_sqlite_stack(tmp_path):
    """Create a Stack backed by SQLiteStorage for contract testing."""
    storage = SQLiteStorage(stack_id=STACK_ID, db_path=tmp_path / "contract_test.db")
    return Stack(
        stack_id=STACK_ID,
        storage=storage,
        components=[],
        enforce_provenance=False,
    )


# ============================================================================
# Registries — append to extend with new backends
# ============================================================================

STORAGE_FACTORIES = [(_create_sqlite_storage, "sqlite")]
STACK_FACTORIES = [(_create_sqlite_stack, "sqlite")]


# ============================================================================
# Parameterized fixtures
# ============================================================================


@pytest.fixture(
    params=[f for f, _ in STORAGE_FACTORIES],
    ids=[name for _, name in STORAGE_FACTORIES],
)
def storage(request, tmp_path):
    """Parameterized storage fixture — runs contract tests against each backend."""
    return request.param(tmp_path)


@pytest.fixture(
    params=[f for f, _ in STACK_FACTORIES],
    ids=[name for _, name in STACK_FACTORIES],
)
def stack(request, tmp_path):
    """Parameterized stack fixture — runs contract tests against each backend."""
    return request.param(tmp_path)
