"""Parameterized fixtures for contract tests.

Provides stack and storage fixtures parameterized by backend.
Future backends (e.g. kernle-supabase) add their factory to
STACK_FACTORIES / STORAGE_FACTORIES and get the full contract suite
for free.

Extension model — in your package's conftest.py::

    from tests.contracts.conftest import (
        CONTRACT_STACK_ID, STORAGE_FACTORIES, STACK_FACTORIES,
    )

    def _create_my_storage(tmp_path, stack_id=CONTRACT_STACK_ID):
        return MyStorage(stack_id=stack_id, connection=..., ...)

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
# Each factory receives (tmp_path, stack_id).
# - tmp_path: per-test temp directory (isolates file-based backends)
# - stack_id: logical partition key (isolates non-file backends)
#
# For isolation tests, the factory is called twice with different
# stack_ids but the same tmp_path, producing two instances that share
# the underlying data store but see different partitions.
# ============================================================================


def _create_sqlite_storage(tmp_path, stack_id=CONTRACT_STACK_ID):
    """Create a SQLiteStorage instance for contract testing."""
    return SQLiteStorage(stack_id=stack_id, db_path=tmp_path / "contract_test.db")


def _create_sqlite_stack(tmp_path, stack_id=CONTRACT_STACK_ID):
    """Create a Stack backed by SQLiteStorage for contract testing."""
    storage = SQLiteStorage(stack_id=stack_id, db_path=tmp_path / "contract_test.db")
    return Stack(
        stack_id=stack_id,
        storage=storage,
        components=[],
        enforce_provenance=False,
    )


# ============================================================================
# Registries — append to extend with new backends
#
# Each entry is (factory_callable, human_readable_id).
# Factory signature: (tmp_path, stack_id=CONTRACT_STACK_ID) -> instance
# Registries are read at test-generation time via pytest_generate_tests,
# so appending from any conftest loaded before collection takes effect.
# ============================================================================

STORAGE_FACTORIES = [(_create_sqlite_storage, "sqlite")]
STACK_FACTORIES = [(_create_sqlite_stack, "sqlite")]


# ============================================================================
# Dynamic parameterization — reads registries at generation time
# ============================================================================


def pytest_generate_tests(metafunc):
    """Parameterize fixtures from the registries.

    This runs at test-generation time (not import time), so factories
    appended to the registries by external conftest files are included.

    The ``make_storage`` fixture is the parametrization root for all
    storage-related fixtures.  ``storage`` derives from it, so both
    share the same backend without creating a cross-product.
    """
    # Only parametrize when make_storage is in the fixture chain.
    # Tests using conftest's `storage` depend on `make_storage` transitively,
    # so it appears in fixturenames. Tests with their own local `storage`
    # fixture (e.g. test_storage_protocol_expansion.py) don't, and are skipped.
    if "make_storage" in metafunc.fixturenames:
        factories = [f for f, _ in STORAGE_FACTORIES]
        ids = [name for _, name in STORAGE_FACTORIES]
        metafunc.parametrize("make_storage", factories, ids=ids, indirect=True)

    if "stack" in metafunc.fixturenames:
        factories = [f for f, _ in STACK_FACTORIES]
        ids = [name for _, name in STACK_FACTORIES]
        metafunc.parametrize("stack", factories, ids=ids, indirect=True)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_storage(request, tmp_path):
    """Factory fixture — returns a callable ``(stack_id?) -> Storage``.

    Useful for isolation tests that need a second storage instance
    sharing the same data store but partitioned by a different stack_id.

    Example::

        def test_isolation(make_storage):
            primary = make_storage()                        # default stack_id
            other   = make_storage("other-stack")           # different partition
            primary.save_episode(ep)
            assert other.get_episodes() == []               # isolated
    """
    factory = request.param

    def _make(stack_id=CONTRACT_STACK_ID):
        return factory(tmp_path, stack_id)

    return _make


@pytest.fixture
def storage(make_storage):
    """Storage fixture — convenience wrapper around ``make_storage``."""
    return make_storage()


@pytest.fixture
def stack(request, tmp_path):
    """Stack fixture — factory is injected via indirect parametrize."""
    factory = request.param
    return factory(tmp_path)
