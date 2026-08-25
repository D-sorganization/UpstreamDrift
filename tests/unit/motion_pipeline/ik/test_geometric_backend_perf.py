"""Performance-fix coverage for issue #8921 (geometric IK backend).

The fix hoists rig-topology computation (:func:`_dof_layout`,
:func:`_topological_order`) out of the forward-kinematics hot path and
replaces the central-difference Jacobian with the analytic revolute-joint
form, built from a single forward-kinematics pass per LM iteration instead
of ``2 * n_dof`` extra FK evaluations.

These tests are deliberately independent of the removed finite-difference
Jacobian implementation: the reference finite-difference Jacobian used for
the parity checks below is reimplemented locally (mirroring the pre-fix
algorithm) rather than calling into solver internals, so the comparison is
not tautological.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.ik import geometric_backend as gb
from src.shared.python.motion_pipeline.ik.base import IKConfig
from src.shared.python.motion_pipeline.ik.geometric_backend import (
    GeometricIKSolver,
    forward_kinematics,
)

pytestmark = pytest.mark.unit

_TOL_RAD = math.radians(2.0)


def _make_planar_arm() -> SkeletonRig:
    """Same fixture as test_geometric_backend.py's planar two-link arm."""
    joints = {
        "shoulder": JointDef(
            name="shoulder",
            parent=None,
            children=["elbow"],
            tpose_offset=[0.0, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.14, upper=3.14)],
        ),
        "elbow": JointDef(
            name="elbow",
            parent="shoulder",
            children=["wrist"],
            tpose_offset=[1.0, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.14, upper=3.14)],
        ),
        "wrist": JointDef(
            name="wrist",
            parent="elbow",
            children=[],
            tpose_offset=[1.0, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.14, upper=3.14)],
        ),
    }
    return SkeletonRig(id="planar_arm", joints=joints, root_joint="shoulder")


def _make_spherical_shoulder_rig() -> SkeletonRig:
    """A rig with a multi-axis (3-DOF) joint, to exercise composed axes.

    ``shoulder`` has all three rotation axes (X, Y, Z composed in that
    order) so the analytic Jacobian's "world axis before this joint's own
    later axes" logic is actually exercised, not just single-axis joints.
    """
    joints = {
        "shoulder": JointDef(
            name="shoulder",
            parent=None,
            children=["elbow"],
            tpose_offset=[0.0, 0.0, 0.0],
            axes=["X", "Y", "Z"],
            limits=[
                JointLimit(lower=-3.14, upper=3.14),
                JointLimit(lower=-3.14, upper=3.14),
                JointLimit(lower=-3.14, upper=3.14),
            ],
        ),
        "elbow": JointDef(
            name="elbow",
            parent="shoulder",
            children=["wrist"],
            tpose_offset=[1.0, 0.0, 0.0],
            axes=["Y", "Z"],
            limits=[
                JointLimit(lower=-3.14, upper=3.14),
                JointLimit(lower=-3.14, upper=3.14),
            ],
        ),
        "wrist": JointDef(
            name="wrist",
            parent="elbow",
            children=[],
            tpose_offset=[0.5, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.14, upper=3.14)],
        ),
    }
    return SkeletonRig(id="spherical_shoulder", joints=joints, root_joint="shoulder")


def _reference_finite_diff_jacobian(
    rig: SkeletonRig,
    q: np.ndarray,
    joint_names: list[str],
    weight_vec: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference Jacobian, reimplemented from the pre-fix algorithm.

    Independent of solver internals: goes through the public
    :func:`forward_kinematics` dict API only, so it validates the analytic
    Jacobian rather than assuming it.
    """
    n_dof = len(q)
    n_res = 3 * len(joint_names)
    jac = np.zeros((n_res, n_dof))
    for j in range(n_dof):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[j] += eps
        q_minus[j] -= eps
        pos_plus = forward_kinematics(rig, q_plus.tolist())
        pos_minus = forward_kinematics(rig, q_minus.tolist())
        current_plus = np.concatenate([np.asarray(pos_plus[n]) for n in joint_names])
        current_minus = np.concatenate([np.asarray(pos_minus[n]) for n in joint_names])
        jac[:, j] = (current_plus - current_minus) / (2.0 * eps)
    return weight_vec[:, np.newaxis] * jac


def test_analytic_jacobian_matches_finite_difference_reference() -> None:
    """The closed-form Jacobian must numerically match central differences.

    This is the core correctness check for the #8921 fix: it validates the
    ``axis_world x (p_i - p_pivot)`` formula directly against an independent
    finite-difference reimplementation, on a rig with a composed 3-axis
    joint (shoulder) so cross-axis coupling is exercised.
    """
    rig = _make_spherical_shoulder_rig()
    topo = gb._get_rig_topology(rig)
    rng = np.random.default_rng(1234)
    q = rng.uniform(-0.5, 0.5, size=len(topo.layout))

    joint_names = ["elbow", "wrist"]
    weight_vec = np.ones(3 * len(joint_names))

    positions, name_to_row, dof_axis_world = gb._forward_kinematics_full(rig, q, topo)
    analytic = GeometricIKSolver._analytic_jacobian(
        topo, positions, name_to_row, dof_axis_world, joint_names, weight_vec
    )
    reference = _reference_finite_diff_jacobian(rig, q, joint_names, weight_vec)

    assert analytic.shape == reference.shape
    np.testing.assert_allclose(analytic, reference, atol=1e-4, rtol=1e-4)


def test_geometric_ik_converges_to_equivalent_pose_as_finite_difference_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end IK-convergence parity: analytic vs. finite-diff LM solve.

    Runs the real solver (analytic Jacobian) and a locally reimplemented
    finite-difference LM solve on the same rig/target, and asserts they
    converge to equivalent joint configurations. Exact float parity is not
    the bar (the Jacobian itself changed) -- IK-convergence tolerance is.
    """
    rig = _make_spherical_shoulder_rig()
    topo = gb._get_rig_topology(rig)
    n_dof = len(topo.layout)
    planted = np.array([0.4, -0.3, 0.2, 0.5, -0.2, 0.1][:n_dof])
    full = forward_kinematics(rig, planted.tolist())
    targets = {"elbow": full["elbow"], "wrist": full["wrist"]}

    config = IKConfig(max_iterations=200, tolerance=1e-12)
    solver = GeometricIKSolver(config)
    q_analytic = np.asarray(solver.solve_frame(targets, rig))

    # Locally reimplemented finite-difference LM solve (pre-fix algorithm),
    # independent of solver internals.
    joint_names = ["elbow", "wrist"]
    target_vec = np.concatenate([np.asarray(targets[n]) for n in joint_names])
    weight_vec = np.ones(3 * len(joint_names))
    q_fd = np.zeros(n_dof)
    lam = 1e-2
    prev_cost = np.inf
    for _ in range(config.max_iterations):
        pos = forward_kinematics(rig, q_fd.tolist())
        current = np.concatenate([np.asarray(pos[n]) for n in joint_names])
        residual = weight_vec * (target_vec - current)
        cost = float(residual @ residual)
        if cost < config.tolerance:
            break
        jac = _reference_finite_diff_jacobian(rig, q_fd, joint_names, weight_vec)
        jtj = jac.T @ jac
        jtr = jac.T @ residual
        try:
            dq = np.linalg.solve(jtj + lam * np.eye(n_dof), jtr)
        except np.linalg.LinAlgError:
            break
        q_new = q_fd + dq
        pos_new = forward_kinematics(rig, q_new.tolist())
        current_new = np.concatenate([np.asarray(pos_new[n]) for n in joint_names])
        new_cost = float(
            (weight_vec * (target_vec - current_new))
            @ (weight_vec * (target_vec - current_new))
        )
        if new_cost < cost:
            q_fd = q_new
            lam = max(lam * 0.5, 1e-9)
            if abs(prev_cost - new_cost) < config.tolerance:
                break
            prev_cost = new_cost
        else:
            lam = min(lam * 2.0, 1e6)

    # Both solves should reach (near-)equivalent end-effector poses ...
    pos_analytic = forward_kinematics(rig, q_analytic.tolist())
    pos_fd = forward_kinematics(rig, q_fd.tolist())
    for name in joint_names:
        assert np.asarray(pos_analytic[name]) == pytest.approx(
            np.asarray(pos_fd[name]), abs=1e-3
        )
    # ... and (for this fully-observable rig) the same joint configuration.
    np.testing.assert_allclose(q_analytic, q_fd, atol=1e-4)


def test_forward_kinematics_calls_per_iteration_reduced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timing-sanity check via call count, not wall-clock (avoids CI flake).

    Before the fix: a finite-difference Jacobian needed ``2 * n_dof`` FK
    calls plus 2 more (residual + trial-step residual) per LM iteration.
    After the fix: exactly 2 forward-kinematics passes per iteration,
    regardless of ``n_dof``. Using ``tolerance=-1`` disables early
    convergence so the loop always runs the full ``max_iterations``,
    making the call count deterministic.
    """
    rig = _make_spherical_shoulder_rig()
    call_count = 0
    original = gb._forward_kinematics_full

    def _counting_fk(rig_arg, q_arg, topo_arg):  # noqa: ANN001 - test shim
        nonlocal call_count
        call_count += 1
        return original(rig_arg, q_arg, topo_arg)

    monkeypatch.setattr(gb, "_forward_kinematics_full", _counting_fk)

    max_iterations = 15
    config = IKConfig(max_iterations=max_iterations, tolerance=-1.0)
    solver = GeometricIKSolver(config)
    solver.solve_frame({"elbow": (1.0, 0.1, 0.0), "wrist": (1.4, 0.1, 0.0)}, rig)

    n_dof = len(gb._get_rig_topology(rig).layout)
    old_style_calls = (2 * n_dof + 2) * max_iterations
    assert call_count == 2 * max_iterations
    assert call_count < old_style_calls


def test_rig_topology_is_cached_across_forward_kinematics_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The O(n_joints) tree walk must not repeat on every FK call.

    :func:`_dof_layout`/:func:`_topological_order` (the recursive tree walk
    flagged in #8921) should run once per rig, not once per
    ``forward_kinematics`` call.
    """
    rig = _make_planar_arm()
    build_count = 0
    original_build = gb._build_rig_topology

    def _counting_build(rig_arg):  # noqa: ANN001 - test shim
        nonlocal build_count
        build_count += 1
        return original_build(rig_arg)

    monkeypatch.setattr(gb, "_build_rig_topology", _counting_build)

    for _ in range(25):
        forward_kinematics(rig, [0.1, 0.2, 0.3])

    assert build_count == 1


def test_solve_uses_per_call_config_not_solver_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: ``solve()`` must actually thread ``config`` through.

    Previously ``solve()`` resolved a local ``config`` (used only for
    trajectory metadata) while ``solve_frame`` unconditionally read
    ``self.config``, silently ignoring any per-call config passed to
    ``solve()``. This asserts a per-call ``max_iterations`` override is
    honored by counting forward-kinematics passes.
    """
    rig = _make_planar_arm()
    call_count = 0
    original = gb._forward_kinematics_full

    def _counting_fk(rig_arg, q_arg, topo_arg):  # noqa: ANN001 - test shim
        nonlocal call_count
        call_count += 1
        return original(rig_arg, q_arg, topo_arg)

    monkeypatch.setattr(gb, "_forward_kinematics_full", _counting_fk)

    # Solver's own default config uses many iterations; solve() is called
    # with an explicit low-iteration, never-converging override.
    solver = GeometricIKSolver(IKConfig(max_iterations=100, tolerance=1e-6))
    override = IKConfig(max_iterations=3, tolerance=-1.0)

    frame = MarkerFrame(
        timestamp=0.0,
        markers={
            "elbow": Marker(name="elbow", x=1.0, y=0.1, z=0.0),
            "wrist": Marker(name="wrist", x=1.4, y=0.1, z=0.0),
        },
        frame_index=0,
    )
    traj = MarkerTrajectory(id="single_frame", frames=[frame])

    solver.solve(traj, rig, config=override)

    # 2 FK calls/iteration * 3 iterations for the single frame == 6, not
    # the ~200 calls solver.config's max_iterations=100 would produce.
    assert call_count == 2 * override.max_iterations
