"""Engine-module isolation of the autouse ``_protect_engine_modules`` (#14381).

The fixture undoes ``sys.modules`` changes to engine modules after each test.
Discarding a genuine Pinocchio or Drake module that a test imported first
cannot undo native extension state; it only makes the next import fail
(``pinocchio_pywrap_default.rpy`` / ``pydrake.autodiffutils`` errors), so such
modules are kept. Stubs and ``src.engines`` additions are still discarded and
replaced modules are still restored.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from tests.conftest import _protect_engine_modules

pytestmark = pytest.mark.unit
_PROBE = "_ud14381_probe"


@pytest.fixture
def isolation() -> Iterator[Iterator[None]]:
    """A fresh run of the fixture body, outside pytest's own invocation."""
    run = _protect_engine_modules.__wrapped__()
    next(run)
    yield run
    for name in [n for n in list(sys.modules) if _PROBE in n]:
        del sys.modules[name]


def _finish(run: Iterator[None]) -> None:
    with pytest.raises(StopIteration):
        next(run)


@pytest.mark.parametrize("prefix", ["pinocchio", "pydrake"])
def test_first_imported_native_engine_module_is_kept(isolation, prefix) -> None:
    """A genuine module the test imported first survives teardown."""
    name = f"{prefix}.{_PROBE}"
    module = types.ModuleType(name)
    sys.modules[name] = module
    _finish(isolation)
    assert sys.modules.get(name) is module


@pytest.mark.parametrize("prefix", ["pinocchio", "pydrake"])
def test_engine_stub_added_by_a_test_is_discarded(isolation, prefix) -> None:
    """A ``MagicMock`` standing in for an engine module never leaks."""
    name = f"{prefix}.{_PROBE}"
    sys.modules[name] = MagicMock()
    _finish(isolation)
    assert name not in sys.modules


def test_src_engines_module_added_by_a_test_is_discarded(isolation) -> None:
    """Pure-Python engine wrappers are still dropped (re-importable by path)."""
    name = f"src.engines.{_PROBE}"
    sys.modules[name] = types.ModuleType(name)
    _finish(isolation)
    assert name not in sys.modules


def test_replaced_engine_module_is_restored() -> None:
    """A module present before the test is put back if a test replaces it."""
    name = f"pydrake.{_PROBE}"
    original = types.ModuleType(name)
    sys.modules[name] = original
    try:
        run = _protect_engine_modules.__wrapped__()
        next(run)
        sys.modules[name] = MagicMock()
        _finish(run)
        assert sys.modules[name] is original
    finally:
        sys.modules.pop(name, None)
