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


# --- Codex review on PR #9768 ------------------------------------------------


def test_backend_torques_reach_the_reported_trajectory() -> None:
    """A transcription backend's own controls must survive into the metrics.

    Without this the shared result builder falls back to
    ``vector_to_trajectory``'s lumped ``system_moi * accel * 0.1`` estimate,
    so torque and injury metrics would describe controls nobody solved for.
    """
    from src.shared.python.optimization._swing_kinematics import JOINTS
    from src.shared.python.optimization.swing_optimizer import SwingOptimizer

    n_nodes = 6
    solved = np.arange(len(JOINTS) * (n_nodes - 1), dtype=float).reshape(
        len(JOINTS), n_nodes - 1
    )

    def fake_solve(golfer, club, config, torque_limits, joint_limits, x0):
        return CasadiSwingResult(
            success=True,
            x=np.asarray(x0, dtype=float),
            fun=0.0,
            message="fake",
            iterations=2,
            torques=solved,
            transcription="fake",
        )

    registry.register_backend(
        registry.BackendSpec(
            name="torque-backend",
            problem_class="tests",
            description="returns known torques",
            available=lambda: True,
            install_hint="n/a",
            solve=fake_solve,
        )
    )
    try:
        config = OptimizationConfig(n_nodes=n_nodes, solver="torque-backend")
        result = SwingOptimizer(GolferModel(), ClubModel(), config).optimize()
        assert result.success and result.trajectory is not None
        recovered = result.trajectory.joint_torques
        for index, joint in enumerate(JOINTS):
            # One torque per node: the last interval is held through the end.
            np.testing.assert_allclose(recovered[joint][: n_nodes - 1], solved[index])
            assert recovered[joint][-1] == solved[index][-1]
    finally:
        registry._REGISTRY.pop("torque-backend", None)


def test_scipy_backend_name_reaches_a_real_scipy_method() -> None:
    """``solver="scipy"`` is a registry key, not a SciPy method name."""
    from src.shared.python.optimization.swing_optimizer import SwingOptimizer

    optimizer = SwingOptimizer(
        GolferModel(), ClubModel(), OptimizationConfig(n_nodes=5, solver="scipy")
    )
    # Previously raised ValueError: Unknown solver scipy.
    result = optimizer.optimize()
    assert isinstance(result.success, bool)
    assert result.trajectory is not None or not result.success


def test_legacy_scipy_method_names_are_untouched() -> None:
    from src.shared.python.optimization.swing_optimizer import SwingOptimizer

    optimizer = SwingOptimizer(
        GolferModel(), ClubModel(), OptimizationConfig(n_nodes=5, solver="SLSQP")
    )
    result = optimizer.optimize()
    assert isinstance(result.success, bool)
