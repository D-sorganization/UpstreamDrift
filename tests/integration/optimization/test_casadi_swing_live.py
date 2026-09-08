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

# Import the C extension once at module scope: the root conftest snapshots
# ``pinocchio*`` sys.modules entries before each test and evicts anything
# added during it, and pinocchio_pywrap_default does not survive a second
# import in the same process.
if PIN_AVAILABLE:
    import pinocchio as pin

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
    CasadiSolveOptions,
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


# --- #9755: anthropometric inertials ----------------------------------------


@pytest.mark.skipif(not PIN_AVAILABLE, reason="pinocchio not installed")
@pytest.mark.parametrize("inertials", ["anthropometric", "placeholder"])
def test_dynamics_kernels_match_pinocchio(inertials: str) -> None:
    """RNEA, CRBA-by-RNEA and forward dynamics agree with Pinocchio on the
    same URDF, for both the physical and the legacy placeholder inertials."""
    from src.shared.python.optimization.casadi_backend import (
        build_forward_dynamics,
        build_mass_matrix,
    )
    from src.shared.python.optimization.model_provider import (
        build_pinocchio_model,
        placeholder_link_inertials,
    )

    golfer, club = GolferModel(), ClubModel()
    model = build_pinocchio_model(golfer, club, inertials=inertials)  # type: ignore[arg-type]
    data = model.createData()
    link_inertials = (
        placeholder_link_inertials() if inertials == "placeholder" else None
    )
    rnea = build_symbolic_rnea(golfer, club, link_inertials=link_inertials)
    mass = build_mass_matrix(golfer, club, link_inertials=link_inertials)
    forward = build_forward_dynamics(golfer, club, link_inertials=link_inertials)

    rng = np.random.default_rng(7)
    for _ in range(50):
        q = rng.uniform(-1.0, 1.0, model.nq)
        v = rng.uniform(-5.0, 5.0, model.nv)
        a = rng.uniform(-20.0, 20.0, model.nv)
        tau = rng.uniform(-50.0, 50.0, model.nv)
        np.testing.assert_allclose(
            np.asarray(rnea(q, v, a)).ravel(), pin.rnea(model, data, q, v, a), atol=1e-9
        )
        crba = pin.crba(model, data, q)
        crba = np.triu(crba) + np.triu(crba, 1).T
        np.testing.assert_allclose(np.asarray(mass(q)), crba, atol=1e-9)
        np.testing.assert_allclose(
            np.asarray(forward(q, v, tau)).ravel(),
            pin.aba(model, data, q, v, tau),
            atol=1e-8,
        )


def test_anthropometric_torques_follow_the_golfer_and_placeholders_do_not() -> None:
    """The point of #9755: gravity torques respond to the golfer's mass."""
    from src.shared.python.optimization.model_provider import (
        placeholder_link_inertials,
    )

    club = ClubModel()
    light, heavy = GolferModel(mass=60.0), GolferModel(mass=120.0)
    q = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0])  # arm + club horizontal
    zero = np.zeros(7)

    def shoulder_gravity_torque(golfer: GolferModel, **kwargs: object) -> float:
        rnea = build_symbolic_rnea(golfer, club, **kwargs)  # type: ignore[arg-type]
        return abs(float(np.asarray(rnea(q, zero, zero)).ravel()[3]))

    assert shoulder_gravity_torque(heavy) > 1.3 * shoulder_gravity_torque(light)
    placeholder = placeholder_link_inertials()
    assert shoulder_gravity_torque(heavy, link_inertials=placeholder) == pytest.approx(
        shoulder_gravity_torque(light, link_inertials=placeholder)
    )


# --- #9756: dynamics defect and multiple shooting ---------------------------


def test_finite_difference_solution_violates_the_ode() -> None:
    from src.shared.python.optimization.casadi_backend import dynamics_defect

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=10, swing_duration=1.0, max_iterations=200)
    joint_limits = swing_joint_limits(golfer)
    x0 = generate_initial_guess(golfer, config, joint_limits)
    result = solve_swing_casadi(
        golfer, club, config, _torque_limits(golfer), joint_limits, x0
    )
    assert result.success and result.torques is None
    report = dynamics_defect(golfer, club, config, result.x)
    assert report.torques.shape == (len(JOINTS), config.n_nodes - 1)
    # The FD path is a kinematic fit: the ODE is violated by O(1) rad.
    assert report.max_position_defect > 0.1
    assert set(report.to_dict()) == {
        "max_position_defect",
        "max_velocity_defect",
        "rms_position_defect",
        "rms_velocity_defect",
    }


@pytest.mark.slow
def test_multiple_shooting_satisfies_its_own_dynamics() -> None:
    from src.shared.python.optimization.casadi_backend import dynamics_defect

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=6, swing_duration=0.6, max_iterations=800)
    joint_limits = swing_joint_limits(golfer)
    x0 = generate_initial_guess(golfer, config, joint_limits)
    limits = _torque_limits(golfer)
    result = solve_swing_casadi(
        golfer,
        club,
        config,
        limits,
        joint_limits,
        x0,
        options=CasadiSolveOptions(transcription="multiple_shooting", n_substeps=8),
    )
    assert result.success, result.message
    assert result.transcription == "multiple_shooting"
    assert result.torques is not None
    assert result.torques.shape == (len(JOINTS), config.n_nodes - 1)
    for j, joint in enumerate(JOINTS):
        assert np.all(np.abs(result.torques[j]) <= limits[joint] + 1e-6)
    # Re-integrating on the solver's own grid reproduces the nodes.
    own = dynamics_defect(
        golfer, club, config, result.x, torques=result.torques, n_substeps=8
    )
    assert own.max_defect < 1e-5
    # A finer reference integrator shows only discretisation error, which is
    # small next to the finite-difference path's O(1) ODE violation.
    reference = dynamics_defect(golfer, club, config, result.x, torques=result.torques)
    fd = solve_swing_casadi(golfer, club, config, limits, joint_limits, x0)
    fd_reference = dynamics_defect(golfer, club, config, fd.x)
    assert reference.max_position_defect < 0.5
    assert reference.max_position_defect < 0.5 * fd_reference.max_position_defect


def test_transcription_argument_validated() -> None:
    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=4)
    with pytest.raises(ValueError, match="transcription"):
        CasadiSolveOptions(transcription="collocation")  # type: ignore[arg-type]
