"""Tests for Physics-Informed ShadowModel Motion-Matching Consumer (Issue #11028).

TDD test suite verifying:
1. ShadowReport holds peak_rigid_torques alongside peak_residuals.
2. ShadowModel.observe computes peak rigid torques per swing phase.
3. RigidCore supports in-memory model instantiation for programmatic and synthetic models.
4. ShadowModel.observe_motion_matching observes inverse-dynamics output without changing state.
5. PinocchioInverseDynMatchingSolver wires ShadowModel in observation mode.
6. Launcher truthfulness entries point to active issue #11028 citing the data blocker.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.launchers.task_launch_truthfulness import (
    LaunchDisposition,
    get_launch_truthfulness_audit,
)
from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.matching.inverse_dyn_pinocchio import (
    PinocchioInverseDynMatchingSolver,
)
from src.shared.python.physics_informed.rigid_core import RigidCore
from src.shared.python.physics_informed.shadow_model import (
    ShadowModel,
    ShadowReport,
    SwingPhase,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Test fixtures and stubs
# =============================================================================


def _make_stub_pinocchio():
    """Build a minimal pin-API surrogate with a closed-form RNEA and model building."""
    pin = types.SimpleNamespace()

    def _rnea(model, data, q, v, a):  # noqa: ARG001
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        v_arr = np.asarray(v, dtype=np.float64).flatten()
        a_arr = np.asarray(a, dtype=np.float64).flatten()
        return a_arr + 0.1 * v_arr + 0.05 * np.sin(q_arr)

    pin.rnea = _rnea

    class _MockModel:
        def __init__(self):
            self.nq = 1
            self.nv = 1
            self.name = "stub_model"
            self.names = ["universe", "hinge"]

        def addJoint(self, *args, **kwargs):
            return 1

        def appendBodyToJoint(self, *args, **kwargs):
            pass

        def createData(self):
            return types.SimpleNamespace()

        def rnea(self, m, d, q, v, a):
            return pin.rnea(m, d, q, v, a)

    pin.Model = _MockModel
    pin.SE3 = types.SimpleNamespace(
        Identity=lambda: types.SimpleNamespace(translation=np.zeros(3))
    )
    pin.JointModelRX = lambda: types.SimpleNamespace()
    pin.JointModelRY = lambda: types.SimpleNamespace()
    pin.JointModelRZ = lambda: types.SimpleNamespace()
    pin.Inertia = types.SimpleNamespace(FromSphere=lambda m, r: types.SimpleNamespace())
    return pin


def _pendulum_rig() -> SkeletonRig:
    """Single-DoF revolute rig about X."""
    return SkeletonRig(
        id="pendulum",
        joints={
            "hinge": JointDef(
                name="hinge",
                parent=None,
                children=[],
                tpose_offset=[0.0, 0.0, -1.0],
                axes=["X"],
            ),
        },
        root_joint="hinge",
    )


def _sinusoidal_traj(rig: SkeletonRig, n_frames: int = 15) -> JointTrajectory:
    """1-DoF trajectory with sinusoidal motion across frames."""
    times = np.linspace(0.0, 0.5, n_frames)
    frames = [
        JointStateFrame(
            timestamp=float(t),
            q=[float(np.sin(2.0 * np.pi * 2.0 * t))],
            qdot=[float(2.0 * np.pi * 2.0 * np.cos(2.0 * np.pi * 2.0 * t))],
            qddot=[float(-((2.0 * np.pi * 2.0) ** 2) * np.sin(2.0 * np.pi * 2.0 * t))],
            frame_index=i,
        )
        for i, t in enumerate(times)
    ]
    return JointTrajectory(id="sin-ref", skeleton=rig, frames=frames)


# =============================================================================
# 1. ShadowReport peak_rigid_torques
# =============================================================================


def test_shadow_report_has_peak_rigid_torques() -> None:
    """ShadowReport must have peak_rigid_torques dict defaulting to empty."""
    report = ShadowReport()
    assert hasattr(report, "peak_rigid_torques")
    assert isinstance(report.peak_rigid_torques, dict)
    assert report.peak_rigid_torques == {}

    r1 = ShadowReport()
    r2 = ShadowReport()
    r1.peak_rigid_torques["transition"] = 10.5
    assert "transition" not in r2.peak_rigid_torques


# =============================================================================
# 2. ShadowModel peak rigid torque calculation per phase
# =============================================================================


def test_shadow_model_observe_computes_peak_rigid_torques() -> None:
    """ShadowModel.observe must record peak rigid torques per phase."""
    rigid = MagicMock()
    # Return different torque magnitudes per call
    torques = [np.array([1.0]), np.array([5.0]), np.array([2.0])] * 4
    rigid.compute_torques.side_effect = torques

    mlp = MagicMock()
    mlp.return_value = np.zeros(1)

    sm = ShadowModel(rigid, mlp)
    frames = [
        {"q": np.zeros(1), "dq": np.zeros(1), "ddq": np.zeros(1)} for _ in range(12)
    ]
    report = sm.observe(frames)

    assert isinstance(report, ShadowReport)
    assert len(report.peak_rigid_torques) > 0
    valid_keys = {p.value for p in SwingPhase}
    assert set(report.peak_rigid_torques.keys()) <= valid_keys
    for phase_name, peak in report.peak_rigid_torques.items():
        assert peak >= 0.0, f"peak for {phase_name} must be >= 0"


# =============================================================================
# 3. RigidCore in-memory model support
# =============================================================================


def test_rigid_core_accepts_in_memory_model() -> None:
    """RigidCore must accept pre-constructed or duck-typed Pinocchio model."""
    stub_model = types.SimpleNamespace(
        name="stub_robot",
        nq=2,
        nv=2,
        createData=lambda: types.SimpleNamespace(),
        rnea=lambda m, d, q, v, a: np.array([1.5, 2.5]),
    )
    core = RigidCore(model=stub_model)
    assert core.nq == 2
    assert core.nv == 2
    tau = core.compute_torques(np.zeros(2), np.zeros(2), np.zeros(2))
    np.testing.assert_allclose(tau, [1.5, 2.5])


# =============================================================================
# 4. ShadowModel observation on motion-matching trajectory
# =============================================================================


def test_shadow_model_observe_motion_matching_changes_no_state() -> None:
    """observe_motion_matching must report peak rigid torques and leave trajectory untouched."""
    stub_model = types.SimpleNamespace(
        name="stub_robot",
        nq=1,
        nv=1,
        createData=lambda: types.SimpleNamespace(),
        rnea=lambda m, d, q, v, a: np.array([4.2]),
    )
    core = RigidCore(model=stub_model)
    mlp = MagicMock(return_value=np.zeros(1))
    sm = ShadowModel(core, mlp)

    rig = _pendulum_rig()
    traj = _sinusoidal_traj(rig, n_frames=12)
    orig_q = [list(f.q) for f in traj.frames]

    report = sm.observe_motion_matching(traj)
    assert isinstance(report, ShadowReport)
    assert len(report.peak_rigid_torques) > 0

    # Ensure no trajectory state was modified
    after_q = [list(f.q) for f in traj.frames]
    assert orig_q == after_q


# =============================================================================
# 5. PinocchioInverseDynMatchingSolver wiring
# =============================================================================


def test_pinocchio_solver_wires_shadow_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PinocchioInverseDynMatchingSolver must include shadow observation report."""
    pin_stub = _make_stub_pinocchio()
    # Stub pinocchio import in the solver
    import sys

    monkeypatch.setitem(sys.modules, "pinocchio", pin_stub)

    rig = _pendulum_rig()
    traj = _sinusoidal_traj(rig, n_frames=12)

    solver = PinocchioInverseDynMatchingSolver(enable_shadow=True)
    result = solver.match(traj, rig)

    assert result is not None
    assert "shadow_report" in result.metadata
    shadow_meta = result.metadata["shadow_report"]
    assert shadow_meta is not None
    assert "peak_rigid_torques" in shadow_meta
    assert len(shadow_meta["peak_rigid_torques"]) > 0


def test_pinocchio_solver_shadow_observation_changes_no_trajectory_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outputs from solver with enable_shadow=True and False must be identical."""
    pin_stub = _make_stub_pinocchio()
    import sys

    monkeypatch.setitem(sys.modules, "pinocchio", pin_stub)

    rig = _pendulum_rig()
    traj = _sinusoidal_traj(rig, n_frames=12)

    solver_no_shadow = PinocchioInverseDynMatchingSolver(enable_shadow=False)
    res_no_shadow = solver_no_shadow.match(traj, rig)

    solver_shadow = PinocchioInverseDynMatchingSolver(enable_shadow=True)
    res_shadow = solver_shadow.match(traj, rig)

    # Tracked trajectories must match
    assert len(res_no_shadow.tracked_trajectory.frames) == len(
        res_shadow.tracked_trajectory.frames
    )
    for f1, f2 in zip(
        res_no_shadow.tracked_trajectory.frames,
        res_shadow.tracked_trajectory.frames,
        strict=True,
    ):
        assert f1.q == f2.q

    # Torque trajectories must match exactly
    for f1, f2 in zip(
        res_no_shadow.torque_trajectory.frames,
        res_shadow.torque_trajectory.frames,
        strict=True,
    ):
        np.testing.assert_allclose(f1.tau, f2.tau)


# =============================================================================
# 6. Task launch truthfulness references active issue #11028
# =============================================================================


def test_pinn_audit_entries_reference_open_issue_11028() -> None:
    """PINN truthfulness entries must reference #11028 and explain data blocker."""
    for cap_id in ("pinn_pure_rigid", "pinn_hybrid"):
        audit = get_launch_truthfulness_audit(cap_id)
        assert audit is not None
        assert audit.disposition == LaunchDisposition.LIBRARY_ONLY
        assert "#11028" in audit.tracking_issue
        assert "#7984" not in audit.tracking_issue
        assert "#5419" not in audit.tracking_issue
        assert "#11028" in audit.next_action
        assert (
            "residual" in audit.explanation.lower()
            or "blocker" in audit.explanation.lower()
        )
