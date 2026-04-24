"""Tests for QP solver availability checks."""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from src.robotics.control.whole_body.qp_solver import ScipyQPSolver

pytestmark = pytest.mark.unit


def test_scipy_qp_solver_unavailable(monkeypatch) -> None:
    """ScipyQPSolver should report unavailable when scipy import fails."""
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs) -> Any:
        if name.startswith("scipy"):
            raise ImportError("scipy not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    solver = ScipyQPSolver()

    assert not solver.is_available()
