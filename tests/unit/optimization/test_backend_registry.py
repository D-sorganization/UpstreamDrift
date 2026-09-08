"""Unit tests for the swing-optimizer backend registry (#9760).

This tree's conftest installs spec-less ``casadi`` / ``pinocchio`` mocks, so
every engine backend must report itself unavailable here while ``scipy``
stays available.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.optimization import backend_registry as registry
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
)
from src.shared.python.optimization.casadi_backend import CasadiSwingResult

pytestmark = pytest.mark.unit


def test_builtin_backends_are_registered_in_order() -> None:
    names = [spec.name for spec in registry.list_backends()]
    assert names[:5] == [
        "scipy",
        "casadi",
        "casadi-multiple-shooting",
        "crocoddyl",
        "bioptim",
    ]


def test_scipy_is_the_only_available_backend_under_mocks() -> None:
    available = {spec.name for spec in registry.available_backends()}
    assert "scipy" in available
    assert (
        not {"casadi", "casadi-multiple-shooting", "crocoddyl", "bioptim"} & available
    )


def test_require_backend_raises_with_install_hint() -> None:
    with pytest.raises(registry.BackendNotAvailableError, match="optimal-control"):
        registry.require_backend("casadi")
    with pytest.raises(
        registry.BackendNotAvailableError, match=r"upstream-drift\[bioptim\]"
    ):
        registry.require_backend("bioptim")
    with pytest.raises(KeyError, match="unknown optimizer backend"):
        registry.require_backend("nope")
    assert registry.require_backend("scipy").solve is None


def test_unknown_solver_name_falls_through_to_scipy() -> None:
    """``OptimizationConfig(solver="SLSQP")`` must keep reaching scipy."""
    assert registry.get_backend("SLSQP") is None


def test_register_backend_replaces_and_solves_through_optimizer() -> None:
    calls: list[str] = []

    def fake_solve(golfer, club, config, torque_limits, joint_limits, x0):
        calls.append("solved")
        return CasadiSwingResult(
            success=True,
            x=np.asarray(x0, dtype=float),
            fun=0.0,
            message="fake",
            iterations=3,
            transcription="fake",
        )

    spec = registry.BackendSpec(
        name="fake-backend",
        problem_class="tests",
        description="records calls",
        available=lambda: True,
        install_hint="n/a",
        solve=fake_solve,
    )
    original = registry.get_backend("fake-backend")
    registry.register_backend(spec)
    try:
        from src.shared.python.optimization.swing_optimizer import SwingOptimizer

        optimizer = SwingOptimizer(
            GolferModel(),
            ClubModel(),
            OptimizationConfig(n_nodes=4, solver="fake-backend"),
        )
        result = optimizer.optimize()
        assert calls == ["solved"]
        assert result.success is True
        assert result.iterations == 3
    finally:
        registry._REGISTRY.pop("fake-backend", None)
        if original is not None:
            registry.register_backend(original)


def test_deprecated_backend_warns_when_selected() -> None:
    def fake_solve(golfer, club, config, torque_limits, joint_limits, x0):
        return CasadiSwingResult(
            success=True,
            x=np.asarray(x0, dtype=float),
            fun=0.0,
            message="",
            iterations=1,
        )

    registry.register_backend(
        registry.BackendSpec(
            name="old-backend",
            problem_class="tests",
            description="deprecated",
            available=lambda: True,
            install_hint="n/a",
            solve=fake_solve,
            deprecated=True,
        )
    )
    try:
        from src.shared.python.optimization.swing_optimizer import SwingOptimizer

        optimizer = SwingOptimizer(
            GolferModel(),
            ClubModel(),
            OptimizationConfig(n_nodes=4, solver="old-backend"),
        )
        with pytest.warns(DeprecationWarning, match="old-backend"):
            optimizer.optimize()
    finally:
        registry._REGISTRY.pop("old-backend", None)


def test_backend_spec_requires_a_name() -> None:
    with pytest.raises(ValueError, match="name"):
        registry.BackendSpec(
            name=" ",
            problem_class="",
            description="",
            available=lambda: True,
            install_hint="",
        )
