"""Classify the ``pinocchio`` module seen by the inverse-dynamics tests.

Three outcomes, kept deliberately distinct (issue #14297):

* absent library -> the caller skips (``pytest.importorskip``);
* ``MagicMock`` contamination from another test's engine probe -> skip, with
  a reason naming the mock;
* a real Pinocchio whose ``Model()`` constructor raises or returns a non-int
  ``nq`` -> :class:`BrokenPinocchioError`. In a provisioned real-Pinocchio
  lane this must fail, never skip.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any
from unittest import mock


class BrokenPinocchioError(RuntimeError):
    """The real Pinocchio extension is importable but its constructor is broken."""


def _is_mock(obj: Any) -> bool:
    return isinstance(obj, mock.NonCallableMock)


def mock_skip_reason(pin: ModuleType | Any) -> str | None:
    """Return a skip reason if ``pin`` is a mock; raise if real but broken."""
    if _is_mock(pin) or _is_mock(getattr(pin, "Model", None)):
        return "pinocchio in sys.modules is a MagicMock - real library absent"
    try:
        model = pin.Model()
    except Exception as exc:
        raise BrokenPinocchioError(
            f"real pinocchio Model() constructor failed: {exc!r}"
        ) from exc
    nq = getattr(model, "nq", None)
    if _is_mock(model) or _is_mock(nq):
        return "pinocchio Model is a MagicMock - real library absent"
    if not isinstance(nq, int):
        raise BrokenPinocchioError(
            f"real pinocchio Model().nq is {type(nq).__name__}, expected int"
        )
    return None
