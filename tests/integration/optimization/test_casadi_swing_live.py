"""Live CasADi tests for the swing direct-transcription backend.

Epic #8390 (B3/#8398). Lives outside ``tests/unit`` because that tree's
conftest replaces ``casadi`` (and ``pinocchio``) with spec-less MagicMocks.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


CASADI_AVAILABLE = _available("casadi")
PIN_AVAILABLE = _available("pinocchio")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_casadi,
    pytest.mark.skipif(not CASADI_AVAILABLE, reason="casadi not installed"),
]

from src.shared.python.optimization._swing_kinematics import (  # noqa: E402
    JOINTS,
    generate_initial_guess,
)
from src.shared.python.optimization._swing_models import (  # noqa: E402
    ClubModel,
    GolferModel,
    OptimizationConfig,
)
from src.shared.python.optimization.casadi_backend import (  # noqa: E402
    build_clubhead_position,
    build_symbolic_rnea,
    solve_swing_casadi,
)
from src.shared.python.optimization.model_provider import (  # noqa: E402
    swing_joint_limits,
)


def _torque_limits(golfer: GolferModel) -> dict[str, float]:
    return {
        "hip_rotation": golfer.max_hip_torque,
        "trunk_rotation": golfer.max_trunk_torque,
        "shoulder_horizontal": golfer.max_shoulder_torque,
        "shoulder_vertical": golfer.max_shoulder_torque,
        "elbow_flexion": golfer.max_elbow_torque,
        "wrist_cock": golfer.max_wrist_torque,
        "wrist_rotation": golfer.max_wrist_torque,
    }


@pytest.mark.skipif(not PIN_AVAILABLE, reason="pinocchio not installed")
def test_symbolic_rnea_matches_pinocchio_on_bridge_model() -> None:
    """The CasADi RNEA must agree with pin.rnea on the same URDF chain."""
    import pinocchio as pin

    from src.shared.python.optimization.model_provider import (
        build_pinocchio_model,
    )

    golfer, club = GolferModel(), ClubModel()
    model = build_pinocchio_model(golfer, club)
    data = model.createData()
    rnea_sym = build_symbolic_rnea(golfer, club)

    rng = np.random.default_rng(3)
    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, model.nq)
        v = rng.uniform(-5.0, 5.0, model.nv)
        a = rng.uniform(-20.0, 20.0, model.nv)
        tau_pin = pin.rnea(model, data, q, v, a)
        tau_ca = np.asarray(rnea_sym(q, v, a)).flatten()
        np.testing.assert_allclose(tau_ca, tau_pin, atol=1e-9)


def test_swing_solve_converges_and_respects_limits() -> None:
    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=10, swing_duration=1.0, max_iterations=200)
    joint_limits = swing_joint_limits(golfer)
    x0 = generate_initial_guess(golfer, config, joint_limits)

    result = solve_swing_casadi(
        golfer, club, config, _torque_limits(golfer), joint_limits, x0
    )
    assert result.success is True
    assert np.all(np.isfinite(result.x))

    n, nodes = len(JOINTS), config.n_nodes
    q = result.x[: n * nodes].reshape(n, nodes)
    flex = golfer.flexibility_factor
    for j, joint in enumerate(JOINTS):
        lo, hi = joint_limits[joint]
        assert np.all(q[j] >= lo * flex - 1e-6)
        assert np.all(q[j] <= hi * flex + 1e-6)


def test_swing_solve_improves_terminal_clubhead_speed() -> None:
    import casadi as ca

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=10, swing_duration=1.0, max_iterations=200)
    joint_limits = swing_joint_limits(golfer)
    x0 = generate_initial_guess(golfer, config, joint_limits)
    result = solve_swing_casadi(
        golfer, club, config, _torque_limits(golfer), joint_limits, x0
    )
    assert result.success

    n, nodes = len(JOINTS), config.n_nodes
    ch = build_clubhead_position(golfer, club)
    qs = ca.SX.sym("q", n)
    vs = ca.SX.sym("v", n)
    speed = ca.Function("s", [qs, vs], [ca.norm_2(ca.jtimes(ch(qs), qs, vs))])

    def terminal_speed(x: np.ndarray) -> float:
        q = x[: n * nodes].reshape(n, nodes)
        v = x[n * nodes :].reshape(n, nodes)
        return float(speed(q[:, -1], v[:, -1]))

    assert terminal_speed(result.x) > terminal_speed(np.asarray(x0))


def test_swing_optimizer_dispatches_casadi_solver() -> None:
    """OptimizationConfig(solver='casadi') routes through the backend and
    returns a populated OptimizationResult."""
    from src.shared.python.optimization.swing_optimizer import SwingOptimizer

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(
        n_nodes=8, swing_duration=1.0, max_iterations=500, solver="casadi"
    )
    optimizer = SwingOptimizer(golfer, club, config)
    result = optimizer.optimize()
    assert result.success is True
    assert result.trajectory is not None
    assert result.iterations > 0


def test_x0_shape_validated() -> None:
    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=10)
    with pytest.raises(ValueError, match="x0 must have length"):
        solve_swing_casadi(
            golfer,
            club,
            config,
            _torque_limits(golfer),
            swing_joint_limits(golfer),
            np.zeros(5),
        )
