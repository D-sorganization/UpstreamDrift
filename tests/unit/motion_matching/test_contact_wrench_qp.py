"""Contact-wrench tracking QP: wrench cone, support geometry, solver, controller (#11670)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import contact_wrench_qp as cw

pytestmark = pytest.mark.unit

NORMAL = np.array([0.0, 0.0, 1.0])
SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_v1.json"
)


def _foot() -> cw.FootSupport:
    return cw.FootSupport(
        name="r",
        centre=np.zeros(3),
        axis_long=np.array([1.0, 0.0, 0.0]),
        axis_lat=np.array([0.0, 1.0, 0.0]),
        extent_long=(-0.10, 0.15),
        extent_lat=(-0.04, 0.04),
        touching=True,
    )


def _rows(**kw) -> np.ndarray:
    args = {"friction": 0.8, "torsion_length_m": 0.03} | kw
    return cw.wrench_cone_rows(_foot(), NORMAL, **args)


def _violated(w: list[float], **kw) -> bool:
    return bool(np.any(_rows(**kw) @ np.asarray(w) > 1e-9))


def test_cone_accepts_a_centred_vertical_load() -> None:
    assert not _violated([0, 0, 400, 0, 0, 0])
    assert _rows().shape == (cw.CONE_ROWS_PER_FOOT, 6)


def test_cone_rejects_tension_and_excess_tangential_force() -> None:
    assert _violated([0, 0, -1, 0, 0, 0])
    mu = 0.8 / np.sqrt(2.0)
    assert not _violated([0.99 * mu * 400, 0, 400, 0, 0, 0])
    assert _violated([1.01 * mu * 400, 0, 400, 0, 0, 0])
    assert _violated([0, -1.01 * mu * 400, 400, 0, 0, 0])


def test_cone_bounds_the_centre_of_pressure_to_the_sole() -> None:
    load = 400.0
    # cop at x: M.e2 = -x fz ; at y: M.e1 = y fz
    assert not _violated([0, 0, load, 0, -0.14 * load, 0])  # x = +0.14 < 0.15
    assert _violated([0, 0, load, 0, -0.16 * load, 0])
    assert _violated([0, 0, load, 0, 0.11 * load, 0])  # x = -0.11 < -0.10
    assert not _violated([0, 0, load, 0.03 * load, 0, 0])
    assert _violated([0, 0, load, 0.05 * load, 0, 0])
    assert _violated([0, 0, load, -0.05 * load, 0, 0])


def test_cone_margin_shrinks_the_sole_and_torsion_is_bounded() -> None:
    load = 400.0
    assert not _violated([0, 0, load, 0, -0.14 * load, 0])
    assert _violated([0, 0, load, 0, -0.14 * load, 0], cop_margin_m=0.02)
    limit = 0.8 * 0.03 * load
    assert not _violated([0, 0, load, 0, 0, 0.99 * limit])
    assert _violated([0, 0, load, 0, 0, 1.01 * limit])
    assert _violated([0, 0, load, 0, 0, -1.01 * limit])


@pytest.mark.parametrize(
    "kw",
    [{"friction": 0.0}, {"torsion_length_m": -1.0}, {"cop_margin_m": -0.1}],
)
def test_cone_validates_its_arguments(kw: dict) -> None:
    with pytest.raises(ValueError):
        _rows(**kw)
    with pytest.raises(ValueError):
        cw.wrench_cone_rows(
            _foot(), np.array([0.0, 0.0, 2.0]), friction=0.8, torsion_length_m=0.03
        )


def test_foot_supports_measure_the_touching_sole() -> None:
    names = ["heel_r", "forefoot_r", "toe_r", "heel_l", "forefoot_l"]
    radius = np.full(5, 0.03)
    centres = np.array(
        [
            [0.0, 0.0, 0.03],
            [0.15, 0.0, 0.03],
            [0.2, 0.0, 0.2],  # lifted toe
            [0.0, 0.3, 0.2],  # lifted left foot
            [0.15, 0.3, 0.2],
        ]
    )
    feet = cw.foot_supports(names, centres, radius, NORMAL, 0.0)
    by_side = {f.name: f for f in feet}
    assert by_side["r"].touching and not by_side["l"].touching
    assert by_side["r"].centre[:2] == pytest.approx([0.075, 0.0])
    assert by_side["r"].extent_long == pytest.approx((-0.075, 0.075))
    assert by_side["r"].axis_long == pytest.approx([1.0, 0.0, 0.0])
    assert by_side["r"].axis_lat == pytest.approx([0.0, 1.0, 0.0])


def test_foot_supports_validate_shapes() -> None:
    with pytest.raises(ValueError, match="match the sphere names"):
        cw.foot_supports(["heel_r"], np.zeros((2, 3)), np.zeros(1), NORMAL, 0.0)


def _toy_problem(nominal_fx: float) -> cw.WrenchQPProblem:
    """One foot under the CoM; joint 0 pushes the body horizontally (x force)."""
    na = 2
    joint = np.zeros((6, na))
    joint[0, 0] = 1.0  # a[0] adds x force: m * da
    joint[1, 1] = 1.0
    wrench = np.array([[nominal_fx, 0.0, 700.0, 0.0, 0.0, 0.0]])
    a_target = np.array([0.0, 0.0])
    a_root = np.array([nominal_fx / 70.0, 0, 0, 0, 0, 0])  # m = 70 kg
    foot = _foot()
    return cw.WrenchQPProblem(
        a_target=a_target,
        a_root_nominal=a_root,
        a_root_target=a_root,
        wrenches_nominal=wrench,
        momentum_root=np.diag([70.0] * 3 + [10.0] * 3),
        momentum_joint=joint,
        wrench_map=np.eye(6),
        cone=cw.wrench_cone_rows(foot, NORMAL, friction=0.8, torsion_length_m=0.03)[
            None
        ],
        touching=[True],
        body_weight_n=700.0,
    )


def test_solver_reproduces_the_nominal_point_when_the_cone_is_satisfied() -> None:
    problem = _toy_problem(nominal_fx=100.0)  # 100 < 0.57 * 700
    solution = cw.solve_wrench_qp(problem, cw.WrenchQPConfig())
    assert solution.success
    assert solution.a_joint == pytest.approx(problem.a_target, abs=1e-5)
    assert solution.wrenches[0] == pytest.approx(problem.wrenches_nominal[0], abs=1e-2)
    assert solution.max_slack < 1e-6


def test_solver_bends_joint_accelerations_when_the_friction_cone_is_violated() -> None:
    problem = _toy_problem(nominal_fx=500.0)  # 500 > 0.566 * 700 = 396
    solution = cw.solve_wrench_qp(problem, cw.WrenchQPConfig(slack_weight=1e8))
    assert solution.success
    mu = 0.8 / np.sqrt(2.0)
    assert solution.wrenches[0, 0] <= mu * solution.wrenches[0, 2] + 1.0
    assert abs(solution.a_joint[0]) > 1e-3  # the joint took up the difference
    balance = (
        problem.momentum_joint @ solution.a_joint
        + problem.momentum_root @ solution.a_root
        - problem.wrench_map @ solution.wrenches.reshape(-1)
    )
    reference = (
        problem.momentum_joint @ problem.a_target
        + problem.momentum_root @ problem.a_root_nominal
        - problem.wrench_map @ problem.wrenches_nominal.reshape(-1)
    )
    assert balance == pytest.approx(reference, abs=1e-5)


def test_solver_forces_a_lifted_foot_wrench_to_zero() -> None:
    base = _toy_problem(nominal_fx=0.0)
    problem = cw.WrenchQPProblem(
        **{**base.__dict__, "touching": [False], "wrenches_nominal": np.zeros((1, 6))}
    )
    solution = cw.solve_wrench_qp(problem, cw.WrenchQPConfig())
    assert not solution.wrenches.any()


def test_problem_rejects_inconsistent_shapes() -> None:
    base = _toy_problem(nominal_fx=0.0)
    with pytest.raises(ValueError, match="foot count"):
        cw.WrenchQPProblem(**{**base.__dict__, "wrenches_nominal": np.zeros((2, 6))})
    with pytest.raises(ValueError, match="body weight"):
        cw.WrenchQPProblem(**{**base.__dict__, "body_weight_n": 0.0})


@pytest.mark.parametrize(
    "kw",
    [
        {"friction": -1.0},
        {"torsion_length_m": float("nan")},
        {"joint_weight": 0.0},
        {"root_gain": -0.1},
    ],
)
def test_config_validates(kw: dict) -> None:
    with pytest.raises(ValueError):
        cw.WrenchQPConfig(**kw)


@pytest.fixture(scope="module")
def standing():
    fs = pytest.importorskip(
        "src.shared.python.motion_matching.full_body_forward_dynamics"
    )
    model = pytest.importorskip(
        "src.engines.physics_engines.mujoco.python.full_body_model"
    )
    pytest.importorskip("pydrake.solvers")
    sim = fs.FullBodySimulator(model.NativeMujocoFullBodyModel(SPEC.read_bytes()))
    q = fs.preload_feet(sim, np.zeros(sim.nv))
    times = np.linspace(0.0, 1.0, 11)
    return fs, sim, q, times, np.tile(q, (len(times), 1))


def test_controller_equals_the_baseline_on_a_feasible_hold(standing) -> None:
    fs, sim, q, times, q_ref = standing
    v = np.zeros(sim.nv)
    base = fs.tracking_controller(
        sim, times, q_ref, omega_rad_s=30.0, zeta=1.0, balance=(60.0, 15.0)
    )
    qp = cw.contact_wrench_controller(
        sim, times, q_ref, omega_rad_s=30.0, zeta=1.0, balance=(60.0, 15.0)
    )
    assert np.abs(qp(0.2, q, v) - base(0.2, q, v)).max() < 1e-3
    assert qp.activations == 0 and qp.max_slack < 1e-3


def test_controller_acts_when_the_cone_is_made_unreachable(standing) -> None:
    fs, sim, q, times, q_ref = standing
    # A sliding stance foot makes the plant push at the Coulomb limit, which a
    # near frictionless cone cannot allow.
    moving = q_ref
    v = np.zeros(sim.nv)
    v[0] = 0.3
    config = cw.WrenchQPConfig(friction=0.02, slack_weight=1e6)
    base = fs.tracking_controller(
        sim, times, moving, omega_rad_s=30.0, zeta=1.0, balance=(60.0, 15.0)
    )
    qp = cw.contact_wrench_controller(
        sim,
        times,
        moving,
        omega_rad_s=30.0,
        zeta=1.0,
        balance=(60.0, 15.0),
        config=config,
    )
    tau = qp(0.5, q, v)
    assert qp.activations == 1
    assert np.abs(tau - base(0.5, q, v)).max() > 1e-3


def test_controller_rejects_a_bad_reference(standing) -> None:
    _, sim, _, times, q_ref = standing
    with pytest.raises(ValueError, match="Reference times"):
        cw.contact_wrench_controller(sim, times[::-1], q_ref, omega_rad_s=30.0)


def test_wrench_qp_is_a_registered_tracking_backend() -> None:
    from src.shared.python.motion_matching.pipeline.dynamics import (
        validate_tracking_backend,
        TRACKING_BACKENDS,
    )

    assert "wrench-qp" in TRACKING_BACKENDS
    assert validate_tracking_backend(" Wrench-QP ") == "wrench-qp"
    with pytest.raises(ValueError, match="Unknown tracking backend"):
        validate_tracking_backend("nope")
