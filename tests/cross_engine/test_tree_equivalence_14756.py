"""Drake/Pinocchio tree-equivalence CI fixture (#14756, panel th_76f1ff346817).

Runs on URDF changes (see ``.github/workflows/urdf-cross-engine-equivalence.yml``).
The engine-neutral checks always run; the Drake/Pinocchio adapter checks skip
when an engine is not installed. Pure read-only: nothing is written to
``models/generated/``, mesh caches or the datastore.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_generation.canonical_model import CanonicalModel
from src.shared.python.model_generation.converters.urdf_parser import URDFParser
from src.shared.python.model_generation.tree_equivalence import (
    GRAVITY,
    coupled_generalized_force,
    dof_order,
    drake_to_canonical_floating,
    external_wrench_torques,
    kinematics,
    gravity_torques,
    inertia_about_link_origin,
    joint_damping_torques,
    pinocchio_to_canonical_floating,
    potential_energy,
)
from src.shared.python.motion_matching.cross_engine_replay import (
    EngineReplayOutcome,
    StepSizeConvergenceResult,
    compare_engine_replays,
)
from src.shared.python.motion_matching.full_body_v2_coupling import (
    trunk_coupling_equalities,
)
from src.shared.python.motion_matching.tour_metrics import SharedMetrics

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
URDF = """<robot name="tree">
  <link name="pelvis"><inertial><mass value="2"/><inertia ixx=".1" iyy=".1" izz=".1"/></inertial></link>
  <link name="l1"><inertial><origin xyz="0 0 .1"/><mass value="3"/><inertia ixx=".2" iyy=".3" izz=".1"/></inertial></link>
  <link name="thorax"><inertial><origin xyz=".05 0 .2" rpy="0 0 .5"/><mass value="5"/><inertia ixx=".4" iyy=".2" izz=".3" ixy=".05"/></inertial></link>
  <link name="arm"><inertial><origin xyz="0 .1 0"/><mass value="1.5"/><inertia ixx=".05" iyy=".05" izz=".01"/></inertial></link>
  <joint name="lumbar_flexion" type="revolute"><parent link="pelvis"/><child link="l1"/>
    <origin xyz="0 0 .1"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="9" velocity="9"/><dynamics damping="0.7"/></joint>
  <joint name="thoracic_flexion" type="revolute"><parent link="l1"/><child link="thorax"/>
    <origin xyz="0 0 .2" rpy="0 .2 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="9" velocity="9"/><dynamics damping="1.1"/></joint>
  <joint name="arm_swing" type="revolute"><parent link="thorax"/><child link="arm"/>
    <origin xyz="0 .2 .3"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="9" velocity="9"/><dynamics damping="0.3"/></joint>
</robot>"""
Q = {"lumbar_flexion": 0.3, "thoracic_flexion": -0.2, "arm_swing": 0.6}


@pytest.fixture(scope="module")
def tree():
    parsed = URDFParser().parse_string(URDF)
    return parsed, "pelvis"


def _fd(fn, q: dict[str, float], names: list[str], h: float = 1e-6) -> np.ndarray:
    out = []
    for n in names:
        hi, lo = dict(q, **{n: q[n] + h}), dict(q, **{n: q[n] - h})
        out.append((fn(hi) - fn(lo)) / (2 * h))
    return np.array(out)


def test_parser_to_canonical_preserves_tree_and_dynamics(tree) -> None:
    parsed, _ = tree
    canon = CanonicalModel.from_core(
        name="tree", links=parsed.links, joints=parsed.joints
    )
    by_name = {j.name: j for j in canon.joints}
    assert {n: (j.parent, j.child) for n, j in by_name.items()} == {
        j.name: (j.parent, j.child) for j in parsed.joints
    }
    assert by_name["thoracic_flexion"].dynamics.damping == pytest.approx(1.1)
    assert by_name["arm_swing"].axis == (1.0, 0.0, 0.0)
    assert (
        canon.stable_hash()
        == CanonicalModel.from_core(
            name="tree", links=reversed(parsed.links), joints=reversed(parsed.joints)
        ).stable_hash()
    )


def test_dof_order_is_declaration_order_not_canonical_sort(tree) -> None:
    parsed, root = tree
    canon = CanonicalModel.from_core(
        name="tree", links=parsed.links, joints=parsed.joints
    )
    order = dof_order(parsed.joints, root)
    assert order == ["lumbar_flexion", "thoracic_flexion", "arm_swing"]
    assert [j.name for j in canon.joints] != order  # canonical sorts alphabetically


@pytest.mark.xfail(
    strict=True, reason="URDFParser drops <inertial><origin rpy> (inertia frame)"
)
def test_parser_keeps_inertial_frame_rotation(tree) -> None:
    parsed, _ = tree
    thorax = next(k for k in parsed.links if k.name == "thorax")
    assert getattr(thorax.inertia, "rpy", None) == pytest.approx((0.0, 0.0, 0.5))


@pytest.mark.xfail(
    strict=True,
    reason="a URDF joint with no <dynamics> parses to damping 0.5; engines use 0",
)
def test_missing_dynamics_means_zero_damping() -> None:
    xml = URDF.replace('<dynamics damping="0.3"/>', "")
    arm = next(
        j for j in URDFParser().parse_string(xml).joints if j.name == "arm_swing"
    )
    assert arm.dynamics.damping == 0.0


def test_floating_base_orderings_agree() -> None:
    p, quat_xyzw = [1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.9]
    pq, pv = pinocchio_to_canonical_floating(p + quat_xyzw, [1, 2, 3, 4, 5, 6])
    dq, dv = drake_to_canonical_floating([0.9, 0.1, 0.2, 0.3] + p, [4, 5, 6, 1, 2, 3])
    assert np.array_equal(pq, dq) and np.array_equal(pv, dv)
    assert list(pq) == [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3]


def test_inertia_about_origin_is_frame_and_parallel_axis_correct() -> None:
    mass, com = 2.0, np.array([0.1, -0.2, 0.3])
    cc = np.diag([0.4, 0.2, 0.3])
    expected = np.diag([0.4, 0.2, 0.3]) + mass * (
        com @ com * np.eye(3) - np.outer(com, com)
    )
    assert inertia_about_link_origin(mass, com, cc) == pytest.approx(expected)
    # The same body given in a yawed inertial frame: rotate the tensor back.
    c, s_ = np.cos(0.7), np.sin(0.7)
    rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    rotated = inertia_about_link_origin(mass, com, rz.T @ cc @ rz, (0, 0, 0.7))
    assert rotated == pytest.approx(expected)


def test_gravity_matches_potential_energy_gradient(tree) -> None:
    parsed, root = tree
    names = dof_order(parsed.joints, root)
    tau = gravity_torques(parsed.links, parsed.joints, root, Q)
    fd = _fd(lambda q: potential_energy(parsed.links, parsed.joints, root, q), Q, names)
    assert tau == pytest.approx(fd, abs=1e-6)
    assert np.linalg.norm(tau) > 1.0


def test_damping_is_ordered_and_dissipative(tree) -> None:
    parsed, root = tree
    qd = {"lumbar_flexion": 1.0, "thoracic_flexion": 2.0, "arm_swing": -3.0}
    tau = joint_damping_torques(parsed.joints, root, qd)
    assert tau == pytest.approx([-0.7, -2.2, 0.9])
    assert sum(t * qd[n] for t, n in zip(tau, qd, strict=True)) < 0.0


def test_external_wrench_obeys_virtual_work(tree) -> None:
    parsed, root = tree
    names = dof_order(parsed.joints, root)
    f, m, pt = np.array([3.0, -1.0, 2.0]), np.array([0.2, 0.4, -0.1]), [0.0, 0.1, 0.0]
    tau = external_wrench_torques(parsed.joints, root, Q, "arm", pt, f, m)

    def point(q):
        st = kinematics(parsed.joints, root, q)
        return st.link_pos["arm"] + st.link_rot["arm"] @ np.asarray(pt)

    dp = _fd(point, Q, names)  # (n, 3)
    expect = np.array(
        [
            dp[i] @ f + kinematics(parsed.joints, root, Q).joint_axis[n] @ m
            for i, n in enumerate(names)
        ]
    )
    assert tau == pytest.approx(expect, abs=1e-6)
    # A wrench on the pelvis moves no joint.
    assert not external_wrench_torques(parsed.joints, root, Q, "pelvis", pt, f).any()


def test_adr0059_default_is_coupled_and_reduces_force(tree) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    eqs = [
        e for e in trunk_coupling_equalities(spec) if e["joint1"] == "thoracic_flexion"
    ]
    assert eqs, "ADR-0059: trunk coupling must be enabled in the committed spec"
    parsed, root = tree
    names = dof_order(parsed.joints, root)
    ratio = eqs[0]["polycoef"][1]

    def coupled_energy(q):
        full = dict(q, thoracic_flexion=ratio * q["lumbar_flexion"])
        return potential_energy(parsed.links, parsed.joints, root, full)

    q = dict(Q, thoracic_flexion=ratio * Q["lumbar_flexion"])
    tau = gravity_torques(parsed.links, parsed.joints, root, q)
    reduced, kept = coupled_generalized_force(tau, names, q, eqs)
    assert kept == ["lumbar_flexion", "arm_swing"]
    fd = _fd(coupled_energy, q, kept)
    assert reduced == pytest.approx(fd, abs=1e-6)


def _outcome(engine: str, traj: np.ndarray, rmse: float) -> EngineReplayOutcome:
    conv = StepSizeConvergenceResult(0.001, 0.0005, 0.0, 0.0, True)
    metrics = SharedMetrics(rmse, rmse, rmse, rmse, 0.01)
    return EngineReplayOutcome(
        engine=engine,
        status="PASSED",
        shared_metrics=metrics,
        contact_audit=None,
        convergence=conv,
        max_closure_residual_m=0.0,
        aligned_model_markers_m=traj,
    )


def test_equal_rmse_different_trajectory_counterexample_is_rejected() -> None:
    """#10350: +100 mm and -100 mm trajectories share RMSE but differ by 200 mm."""
    plus, minus = np.zeros((10, 4, 3)), np.zeros((10, 4, 3))
    plus[..., 0], minus[..., 0] = 0.10, -0.10
    rmse = float(np.sqrt(np.mean(plus**2)))
    assert rmse == pytest.approx(float(np.sqrt(np.mean(minus**2))))
    report = compare_engine_replays(
        (_outcome("mujoco", plus, rmse), _outcome("drake", minus, rmse))
    ).to_parity_report()
    diff = next(iter(report.engine_rows["mujoco"].pointwise_differences.values()))
    assert diff.pass_gate is False
    assert diff.max_abs_diff == pytest.approx(0.20)


def test_pinocchio_gravity_matches_reference(tree, tmp_path: Path) -> None:
    pin = pytest.importorskip("pinocchio")
    parsed, root = tree
    urdf = tmp_path / "tree.urdf"
    urdf.write_text(URDF, encoding="utf-8")
    model = pin.buildModelFromUrdf(str(urdf))
    # Pinocchio defaults to g = 9.81; pin it to the canonical constant the
    # reference uses so the comparison tests tree dynamics, not the constant.
    model.gravity.linear = GRAVITY
    names = dof_order(parsed.joints, root)
    q = np.array([Q[n] for n in model.names[1:]])
    assert list(model.names[1:]) == names
    data = model.createData()
    g = pin.computeGeneralizedGravity(model, data, q)
    ref = gravity_torques(parsed.links, parsed.joints, root, Q)
    assert np.asarray(g) == pytest.approx(ref, abs=1e-8)
