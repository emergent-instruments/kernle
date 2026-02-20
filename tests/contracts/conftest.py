"""Parameterized fixtures for contract tests.

Provides stack and storage fixtures parameterized by backend.
Future backends (e.g. kernle-supabase) add their factory to
STACK_FACTORIES / STORAGE_FACTORIES and get the full contract suite
for free.

Extension model — in your package's conftest.py::

    from tests.contracts.conftest import STORAGE_FACTORIES, STACK_FACTORIES

    def _create_my_storage(tmp_path):
        return MyStorage(stack_id=CONTRACT_STACK_ID, ...)

    STORAGE_FACTORIES.append((_create_my_storage, "my-backend"))

Factories are read at test-generation time (not import time), so
appending to the registries from any conftest.py that loads before
test collection will work.
"""

import pytest

from kernle.stack import Stack
from kernle.storage.sqlite import SQLiteStorage

# Canonical stack_id used by all contract test factories and helpers.
# Individual test files import this to construct matching memory objects.
CONTRACT_STACK_ID = "contract-test-stack"


# ============================================================================
# Factory functions
#
# Each factory receives (tmp_path) — file-based backends get per-test
# isolation via tmp_path; non-file backends can use CONTRACT_STACK_ID
# combined with unique DB names or schemas to isolate.
# ============================================================================


def _create_sqlite_storage(tmp_path):
    """Create a SQLiteStorage instance for contract testing."""
    return SQLiteStorage(stack_id=CONTRACT_STACK_ID, db_path=tmp_path / "contract_test.db")


def _create_sqlite_stack(tmp_path):
    """Create a Stack backed by SQLiteStorage for contract testing."""
    storage = SQLiteStorage(stack_id=CONTRACT_STACK_ID, db_path=tmp_path / "contract_test.db")
    return Stack(
        stack_id=CONTRACT_STACK_ID,
        storage=storage,
        components=[],
        enforce_provenance=False,
    )


# ============================================================================
# Registries — append to extend with new backends
#
# Each entry is (factory_callable, human_readable_id).
# Registries are read at test-generation time via pytest_generate_tests,
# so appending from any conftest loaded before collection takes effect.
# ============================================================================

STORAGE_FACTORIES = [(_create_sqlite_storage, "sqlite")]
STACK_FACTORIES = [(_create_sqlite_stack, "sqlite")]


# ============================================================================
# Dynamic parameterization — reads registries at generation time
# ============================================================================


def pytest_generate_tests(metafunc):
    """Parameterize 'storage' and 'stack' fixtures from the registries.

    This runs at test-generation time (not import time), so factories
    appended to the registries by external conftest files are included.
    """
    if "storage" in metafunc.fixturenames:
        factories = [f for f, _ in STORAGE_FACTORIES]
        ids = [name for _, name in STORAGE_FACTORIES]
        metafunc.parametrize("storage", factories, ids=ids, indirect=True)

    if "stack" in metafunc.fixturenames:
        factories = [f for f, _ in STACK_FACTORIES]
        ids = [name for _, name in STACK_FACTORIES]
        metafunc.parametrize("stack", factories, ids=ids, indirect=True)


# ============================================================================
# Fixtures — receive the factory via indirect parametrize
# ============================================================================


@pytest.fixture
def storage(request, tmp_path):
    """Storage fixture — factory is injected via indirect parametrize."""
    factory = request.param
    return factory(tmp_path)


@pytest.fixture
def stack(request, tmp_path):
    """Stack fixture — factory is injected via indirect parametrize."""
    factory = request.param
    return factory(tmp_path)
