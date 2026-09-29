"""#9771: the ocp probes must survive a poisoned ``sys.modules``.

``tests/unit/conftest.py`` installs spec-less ``casadi``/``pinocchio``
MagicMocks process-wide in ``pytest_configure`` (and never removes them).
When a lane collects ``tests/unit/optimization`` and the ``ocp`` tests in
the same pytest process, those mocks shadow the genuinely installed
distributions: ``_compat.bioptim_available()`` went ``False`` and 16 ocp
tests skipped with "bioptim not installed" on a machine that had just
used bioptim.

The contract here, independent of whether the real stack is installed:

- a spec-less mock in ``sys.modules`` must not hide a real distribution
  from ``bioptim_available()``;
- ``require_bioptim()`` must import the *real* ``bioptim`` (which pulls
  the real ``casadi``) even with mocks in place;
- when only the mock exists, the probe still reports unavailable so the
  unit tree's degradation-path semantics are unchanged.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from src.shared.python.optimization.ocp import _compat

pytestmark = pytest.mark.integration

FAKE_VERSION = "9.9.9-fake"
_PROBED = ("casadi", "bioptim", "biorbd_casadi")


@pytest.fixture()
def fake_real_stack(monkeypatch, tmp_path):
    """Install fake *real* ``casadi``/``bioptim`` distributions on sys.path."""
    (tmp_path / "casadi").mkdir()
    (tmp_path / "casadi" / "__init__.py").write_text(
        "__version__ = '9.9.9'\nMX = object\nOpti = object\n"
        "MX_eye = object\nSX_eye = object\nDM_eye = object\n",
        encoding="utf-8",
    )
    (tmp_path / "bioptim").mkdir()
    (tmp_path / "bioptim" / "__init__.py").write_text(
        "import casadi  # the real bioptim imports casadi at module scope\n"
        f"__version__ = {FAKE_VERSION!r}\n"
        "OptimalControlProgram = object\n"
        "Solver = object\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    # The import machinery caches whatever ``require_bioptim`` imported; a
    # fake left in ``sys.modules`` would poison every later test in the
    # process, so evict anything that resolves back into the tmp tree.
    for name in list(sys.modules):
        if name.split(".")[0] not in _PROBED:
            continue
        module = sys.modules[name]
        origin = str(getattr(module, "__path__", ""))
        file_origin = str(getattr(module, "__file__", ""))
        if str(tmp_path) in origin or str(tmp_path) in file_origin:
            del sys.modules[name]


def _poison(name: str) -> MagicMock:
    """Spec-less mock exactly as ``tests/unit/conftest.py`` installs it."""
    return MagicMock(name=f"{name}_mock")


def test_probe_reports_absent_when_only_mock_exists(monkeypatch):
    # This case promises no genuine distribution even on an optional-stack host.
    finder = _compat.PathFinder.find_spec
    monkeypatch.setattr(
        _compat.PathFinder,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name in ("casadi", "bioptim") else finder(name, *args, **kwargs)
        ),
    )
    monkeypatch.setitem(sys.modules, "casadi", _poison("casadi"))
    monkeypatch.setitem(sys.modules, "bioptim", _poison("bioptim"))
    assert _compat.bioptim_available() is False


def test_mock_in_sys_modules_does_not_hide_real_stack(fake_real_stack, monkeypatch):
    monkeypatch.setitem(sys.modules, "casadi", _poison("casadi"))
    assert _compat.bioptim_available() is True


def test_require_bioptim_imports_real_stack_past_mocks(fake_real_stack, monkeypatch):
    poisoned = _poison("casadi")
    monkeypatch.setitem(sys.modules, "casadi", poisoned)
    monkeypatch.setitem(sys.modules, "bioptim", _poison("bioptim"))

    module = _compat.require_bioptim()

    assert module.__version__ == FAKE_VERSION
    # The real casadi replaced the poisoned entry so lazy ``import casadi``
    # inside the ocp tests (e.g. swing_ocp) sees the genuine module.
    assert sys.modules["casadi"] is not poisoned
    assert sys.modules["casadi"].__version__ == "9.9.9"
