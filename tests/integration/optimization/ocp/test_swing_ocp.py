"""Phase 2: the max-clubhead-speed OCP and the ``solver="bioptim"`` route."""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.shared.python.optimization.ocp import _compat

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_bioptim,
    pytest.mark.skipif(not _compat.bioptim_available(), reason="bioptim not installed"),
]

os.environ.setdefault("MPLBACKEND", "Agg")

from src.shared.python.optimization._swing_kinematics import (  # noqa: E402
    JOINTS,
    generate_initial_guess,
)
from src.shared.python.optimization._swing_models import (  # noqa: E402
    ClubModel,
    GolferModel,
    OptimizationConfig,
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


def test_tracking_objective_converges_and_respects_limits() -> None:
    from src.shared.python.optimization.casadi_backend import dynamics_defect
    from src.shared.python.optimization.ocp.swing_ocp import (
        DEFAULT_TARGET_SPEED,
        MaxSpeedOcpOptions,
        solve_max_speed_ocp,
    )

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=8, swing_duration=1.0, max_iterations=500)
    joint_limits = swing_joint_limits(golfer)
    limits = _torque_limits(golfer)
    x0 = generate_initial_guess(golfer, config, joint_limits)

    solution = solve_max_speed_ocp(
        golfer,
        club,
        config,
        limits,
        joint_limits,
        x0,
        options=MaxSpeedOcpOptions(ode="collocation"),
    )
    assert solution.success, solution.status
    n, nodes = len(JOINTS), config.n_nodes
    assert solution.q.shape == (n, nodes)
    assert solution.qdot.shape == (n, nodes)
    assert solution.tau.shape == (n, nodes - 1)
    assert solution.time.shape == (nodes,)
    np.testing.assert_allclose(
        solution.q[:, 0], x0[: n * nodes].reshape(n, nodes)[:, 0]
    )
    np.testing.assert_allclose(solution.qdot[:, 0], 0.0, atol=1e-9)
    flex = golfer.flexibility_factor
    for j, joint in enumerate(JOINTS):
        lo, hi = joint_limits[joint]
        assert np.all(solution.q[j] >= lo * flex - 1e-6)
        assert np.all(solution.q[j] <= hi * flex + 1e-6)
        assert np.all(np.abs(solution.tau[j]) <= limits[joint] + 1e-6)
    # The convex tracking objective reaches its target.
    assert solution.clubhead_speed == pytest.approx(DEFAULT_TARGET_SPEED, abs=1.0)
    # Flagship layout and torques travel on the result.
    result = solution.result
    assert result.x.shape == (2 * n * nodes,)
    assert result.torques is not None
    assert result.transcription == "bioptim-collocation"
    # Provenance stamp is populated.
    assert solution.provenance is not None
    assert solution.provenance.engine == "bioptim"
    assert solution.provenance.solver_settings["objective"] == "track_speed"
    assert solution.provenance.solver_settings["ode"] == "collocation"
    del dynamics_defect  # imported to document the cross-check below


@pytest.mark.slow
def test_rk4_solution_satisfies_the_dynamics_it_transcribed() -> None:
    """The OCP is a transcription: its torques reproduce its own nodes."""
    from src.shared.python.optimization.casadi_backend import dynamics_defect
    from src.shared.python.optimization.ocp.swing_ocp import (
        MaxSpeedOcpOptions,
        solve_max_speed_ocp,
    )

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=8, swing_duration=1.0, max_iterations=500)
    joint_limits = swing_joint_limits(golfer)
    x0 = generate_initial_guess(golfer, config, joint_limits)
    solution = solve_max_speed_ocp(
        golfer,
        club,
        config,
        _torque_limits(golfer),
        joint_limits,
        x0,
        options=MaxSpeedOcpOptions(ode="rk4", n_integration_steps=4),
    )
    assert solution.success, solution.status
    assert solution.result.transcription == "bioptim-rk4"
    own = dynamics_defect(
        golfer,
        club,
        config,
        solution.result.x,
        torques=solution.result.torques,
        n_substeps=4,
    )
    assert own.max_defect < 1e-4
    # The finite-difference backend violates the same ODE by O(1) rad.
    fd = dynamics_defect(golfer, club, config, x0)
    assert own.max_position_defect < 1e-3 * fd.max_position_defect


@pytest.mark.slow
def test_maximize_speed_objective_pushes_past_the_target() -> None:
    """Parity formulation: nonconvex, so only the physics is asserted.

    A negative-weight quadratic is concave, so IPOPT commonly stops at the
    iteration cap with a good but uncertified point (see the module
    docstring). What must hold is that it beats the tracking target and
    still respects every bound.
    """
    from src.shared.python.optimization.ocp.swing_ocp import (
        MaxSpeedOcpOptions,
        solve_max_speed_ocp,
    )

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=6, swing_duration=1.0, max_iterations=300)
    joint_limits = swing_joint_limits(golfer)
    limits = _torque_limits(golfer)
    x0 = generate_initial_guess(golfer, config, joint_limits)
    solution = solve_max_speed_ocp(
        golfer,
        club,
        config,
        limits,
        joint_limits,
        x0,
        options=MaxSpeedOcpOptions(objective="maximize_speed", ode="collocation"),
    )
    assert solution.clubhead_speed > 40.0
    for j, joint in enumerate(JOINTS):
        assert np.all(np.abs(solution.tau[j]) <= limits[joint] + 1e-6)


@pytest.mark.slow
def test_swing_optimizer_selects_bioptim_backend() -> None:
    from src.shared.python.optimization.backend_registry import require_backend
    from src.shared.python.optimization.swing_optimizer import SwingOptimizer

    assert require_backend("bioptim").solve is not None
    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(
        n_nodes=6, swing_duration=1.0, max_iterations=400, solver="bioptim"
    )
    result = SwingOptimizer(golfer, club, config).optimize()
    assert result.success is True, result.message
    assert result.trajectory is not None
    assert result.iterations > 0


def test_build_validates_inputs() -> None:
    from src.shared.python.optimization.ocp.swing_ocp import (
        MaxSpeedOcpOptions,
        build_max_speed_ocp,
    )

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=6)
    with pytest.raises(ValueError, match="x0 must have length"):
        build_max_speed_ocp(
            golfer,
            club,
            config,
            _torque_limits(golfer),
            swing_joint_limits(golfer),
            np.zeros(3),
        )
    with pytest.raises(ValueError, match="ode"):
        MaxSpeedOcpOptions(ode="euler")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="objective"):
        MaxSpeedOcpOptions(objective="fastest")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_speed"):
        MaxSpeedOcpOptions(target_speed=0.0)
