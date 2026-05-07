"""Heavy integration tests for the pure-pinocchio differential IK fallback.

Marked ``requires_pinocchio`` and ``live_simulation`` (the latter via
the directory-wide auto-marker in ``conftest.py``). These exercise the
full forward-kinematics + Jacobian pipeline against a real Pinocchio
model and therefore only run when pinocchio is properly installed.

Closes issue #4138.
"""

from __future__ import annotations

import numpy as np
import pytest

# pinocchio is optional; skip the whole module if it is unavailable or
# only the stub is installed.
pin = pytest.importorskip("pinocchio")
if not hasattr(pin, "Model"):
    pytest.skip(
        "pinocchio stub installed, not robotics library", allow_module_level=True
    )

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.diff_ik import (  # noqa: E402
    differential_ik,
    solve_dual_frame_ik,
)

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _build_three_dof_arm() -> tuple[pin.Model, pin.Data, str]:
    """Build a planar 3-DOF revolute arm with an end-effector frame."""
    model = pin.Model()
    inertia = pin.Inertia(1.0, np.zeros(3), np.eye(3))

    parent = 0
    placements = [
        pin.SE3.Identity(),
        pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.0])),
        pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.0])),
    ]
    for k, placement in enumerate(placements):
        joint_id = model.addJoint(
            parent, pin.JointModelRZ(), placement, f"joint{k + 1}"
        )
        model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
        parent = joint_id

    model.addFrame(
        pin.Frame(
            "end_effector",
            parent,
            0,
            pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.0])),
            pin.FrameType.OP_FRAME,
        )
    )
    data = model.createData()
    return model, data, "end_effector"


def _fk(model: pin.Model, data: pin.Data, q: np.ndarray, frame: str) -> pin.SE3:
    pin.forwardKinematics(model, data, q)
    fid = model.getFrameId(frame)
    pin.updateFramePlacement(model, data, fid)
    return data.oMf[fid].copy()


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.requires_pinocchio
class TestDifferentialIKRecovery:
    """Recovery: known truth → forward kinematics → IK → recover truth."""

    def test_recovers_known_configuration(self) -> None:
        model, data, ee = _build_three_dof_arm()
        rng = np.random.default_rng(42)
        q_truth = rng.uniform(-0.5, 0.5, size=model.nq)
        target = _fk(model, data, q_truth, ee)

        q0 = rng.uniform(-0.1, 0.1, size=model.nq)
        q_sol, converged = differential_ik(
            model,
            data,
            ee,
            target,
            q0,
            max_iters=200,
            damping=1e-6,
            tol=1e-6,
        )
        assert converged, "IK should converge for in-workspace target"

        # Check that the solution lands on target via forward kinematics
        recovered = _fk(model, data, q_sol, ee)
        twist = pin.log6(recovered.actInv(target)).vector
        assert float(np.linalg.norm(twist)) < 1e-4

    def test_dual_frame_recovery(self) -> None:
        """Dual-frame IK over redundant DOFs converges to the truth pose."""
        model, data, ee = _build_three_dof_arm()
        # Use joint2's frame as the second target — built-in by appendBodyToJoint.
        frame_b = ee
        # First target frame: we add a dedicated mid-link frame.
        mid_id = model.addFrame(
            pin.Frame(
                "mid_link",
                2,
                0,
                pin.SE3(np.eye(3), np.array([0.2, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        assert mid_id >= 0
        data = model.createData()  # rebuild after frame addition

        rng = np.random.default_rng(7)
        q_truth = rng.uniform(-0.4, 0.4, size=model.nq)
        target_a = _fk(model, data, q_truth, "mid_link")
        target_b = _fk(model, data, q_truth, frame_b)

        q0 = rng.uniform(-0.1, 0.1, size=model.nq)
        q_sol, converged = solve_dual_frame_ik(
            model,
            data,
            "mid_link",
            target_a,
            frame_b,
            target_b,
            q0,
            max_iters=400,
            damping=1e-5,
            tol=1e-5,
        )
        assert converged

        recovered_b = _fk(model, data, q_sol, frame_b)
        twist = pin.log6(recovered_b.actInv(target_b)).vector
        assert float(np.linalg.norm(twist)) < 1e-3


@pytest.mark.requires_pinocchio
class TestSingularityHandling:
    """Singular / out-of-workspace targets should not blow up."""

    def test_workspace_edge_does_not_diverge(self) -> None:
        """Target placed slightly beyond reach: solver returns finite q."""
        model, data, ee = _build_three_dof_arm()
        # Reach is ~1.2 m. Place target at 2.0 m → well outside workspace.
        unreachable = pin.SE3(np.eye(3), np.array([2.0, 0.0, 0.0]))
        q0 = np.zeros(model.nq)
        q_sol, converged = differential_ik(
            model,
            data,
            ee,
            unreachable,
            q0,
            max_iters=100,
            damping=1e-3,
            tol=1e-6,
        )
        # We do not require convergence — the target is unreachable.
        # We DO require finite output and bounded magnitude.
        assert np.all(np.isfinite(q_sol))
        assert float(np.linalg.norm(q_sol)) < 1e3
        # converged should be False — this is documented behaviour.
        assert converged is False

    def test_singular_start_recovers(self) -> None:
        """A fully extended (singular) initial pose should still progress."""
        model, data, ee = _build_three_dof_arm()
        # Singular config: arm fully extended along +x.
        q0 = np.zeros(model.nq)
        rng = np.random.default_rng(11)
        q_truth = rng.uniform(-0.3, 0.3, size=model.nq)
        target = _fk(model, data, q_truth, ee)

        q_sol, converged = differential_ik(
            model,
            data,
            ee,
            target,
            q0,
            max_iters=200,
            damping=1e-3,
            tol=1e-5,
        )
        assert np.all(np.isfinite(q_sol))
        # With damping, even singular start should reach a feasible target.
        assert converged


@pytest.mark.requires_pinocchio
class TestDeterminism:
    """Same inputs → byte-identical outputs (no internal RNG)."""

    def test_identical_runs_match_exactly(self) -> None:
        model, data, ee = _build_three_dof_arm()
        target = pin.SE3(np.eye(3), np.array([0.5, 0.3, 0.0]))
        q0 = np.array([0.1, -0.2, 0.05])
        q_a, conv_a = differential_ik(
            model, data, ee, target, q0, max_iters=50, damping=1e-4
        )
        q_b, conv_b = differential_ik(
            model, data, ee, target, q0, max_iters=50, damping=1e-4
        )
        assert conv_a == conv_b
        assert np.array_equal(q_a, q_b)


@pytest.mark.requires_pinocchio
class TestPreconditions:
    """Design-by-contract preconditions raise descriptive errors."""

    def test_unknown_frame_raises(self) -> None:
        model, data, _ = _build_three_dof_arm()
        with pytest.raises(ValueError, match="not found"):
            differential_ik(
                model,
                data,
                "no_such_frame",
                pin.SE3.Identity(),
                np.zeros(model.nq),
            )

    def test_wrong_q0_shape_raises(self) -> None:
        model, data, ee = _build_three_dof_arm()
        with pytest.raises(ValueError, match="shape"):
            differential_ik(
                model,
                data,
                ee,
                pin.SE3.Identity(),
                np.zeros(model.nq + 2),
            )
