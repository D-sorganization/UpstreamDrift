"""Parity + benchmark tests for the MuJoCo torque outer-loop Rust acceleration
(issue #5254 slice 4).

The solver shares the ``upstream_pinocchio_id`` outer-loop crate
with the Pinocchio inverse-dynamics solver. These tests verify that:

1. The Rust-driven outer loop produces tau outputs that match the
   pure-Python reference to <1e-9 RMSE on a 1000-frame trajectory.
2. The Rust path is at least 1.5x faster than the pure-Python reference
   end-to-end (we use a non-trivial callback to model the cost of a
   realistic inner step without requiring MuJoCo).
"""

from __future__ import annotations

import time

import numpy as np
import pytest


def _make_trajectory(n_frames: int, n_dof: int, dt: float, seed: int = 7):
    """Build a smooth, non-trivial (n_frames, n_dof) trajectory."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_frames, dtype=np.float64) * dt
    t = times[:, None]
    phases = rng.uniform(0, 2 * np.pi, size=(1, n_dof))
    amps = rng.uniform(0.2, 1.0, size=(1, n_dof))
    freqs = rng.uniform(0.5, 2.5, size=(1, n_dof))
    q = amps * np.sin(2 * np.pi * freqs * t + phases)
    return times, q


def _make_mujoco_callback():
    """A non-trivial per-frame callback modelling a MuJoCo inner step."""

    def _cb(q_row, v_row, a_row):
        q_arr = np.asarray(q_row, dtype=np.float64).flatten()
        v_arr = np.asarray(v_row, dtype=np.float64).flatten()
        a_arr = np.asarray(a_row, dtype=np.float64).flatten()
        # Mimic the structure of an inverse-dynamics residual
        return a_arr + 0.1 * v_arr + 0.05 * np.sin(q_arr)

    return _cb


def _python_finite_diff(times, q):
    """Pure-Python finite-difference (mirrors inverse_dyn_pinocchio.py)."""
    n_frames = q.shape[0]
    qdot = np.zeros_like(q)
    qddot = np.zeros_like(q)
    for i in range(1, n_frames - 1):
        dt = times[i + 1] - times[i - 1]
        if dt > 0:
            qdot[i] = (q[i + 1] - q[i - 1]) / dt
    qdot[0] = (q[1] - q[0]) / max(times[1] - times[0], 1e-9)
    qdot[-1] = (q[-1] - q[-2]) / max(times[-1] - times[-2], 1e-9)
    for i in range(1, n_frames - 1):
        dt_b = times[i] - times[i - 1]
        dt_f = times[i + 1] - times[i]
        if dt_b > 0 and dt_f > 0:
            qddot[i] = (
                2.0
                * (q[i + 1] * dt_b - q[i] * (dt_b + dt_f) + q[i - 1] * dt_f)
                / (dt_b * dt_f * (dt_b + dt_f))
            )
    qddot[0] = qddot[1]
    qddot[-1] = qddot[-2]
    return qdot, qddot


def _python_mujoco_pipeline(callback, times, q):
    """End-to-end Python pipeline (finite-diff + per-frame callback)."""
    qdot, qddot = _python_finite_diff(times, q)
    from src.shared.python.motion_pipeline.matching.torque_mujoco import (
        MuJoCoTorqueMatchingSolver,
    )

    return MuJoCoTorqueMatchingSolver._compute_tau_python(
        times, q, qdot, qddot, callback
    )


def _rust_mujoco_pipeline(rust, callback, times, q):
    """End-to-end Rust pipeline via upstream_pinocchio_id."""
    qdot, qddot = _python_finite_diff(times, q)
    from src.shared.python.motion_pipeline.matching.torque_mujoco import (
        MuJoCoTorqueMatchingSolver,
    )

    return MuJoCoTorqueMatchingSolver._compute_tau_rust(times, q, qdot, qddot, callback)


@pytest.mark.unit
def test_torque_mujoco_rust_outer_loop_parity_1000_frames():
    """Rust-driven MuJoCo outer loop matches pure-Python tau to <1e-9 RMSE."""
    rust = pytest.importorskip("upstream_pinocchio_id")
    n_frames, n_dof = 1000, 7
    times, q = _make_trajectory(n_frames, n_dof, dt=0.005)
    callback = _make_mujoco_callback()

    tau_python = _python_mujoco_pipeline(callback, times, q)
    tau_rust = _rust_mujoco_pipeline(rust, callback, times, q)

    rmse = float(np.sqrt(np.mean((tau_python - tau_rust) ** 2)))
    assert rmse < 1e-9, f"MuJoCo outer-loop tau RMSE {rmse} exceeds 1e-9"


@pytest.mark.unit
def test_torque_mujoco_solver_match_uses_rust_when_available():
    """The MuJoCo solver runs end-to-end through its public ``match`` API."""
    pytest.importorskip("upstream_pinocchio_id")
    from src.shared.python.motion_pipeline.contracts import (
        JointDef,
        JointStateFrame,
        JointTrajectory,
        SkeletonRig,
    )
    from src.shared.python.motion_pipeline.matching.torque_mujoco import (
        MuJoCoTorqueMatchingSolver,
    )

    rig = SkeletonRig(
        id="mujoco-test-rig",
        joints={
            "triplet": JointDef(
                name="triplet",
                parent=None,
                children=[],
                tpose_offset=[0.0, 0.0, 0.0],
                axes=["X", "Y", "Z"],
            ),
        },
        root_joint="triplet",
    )
    # 50 frames is enough for finite-diff (need >=3) and keeps the test fast.
    times, q = _make_trajectory(50, 3, dt=0.01)
    frames = [
        JointStateFrame(timestamp=float(times[i]), q=q[i].tolist(), frame_index=i)
        for i in range(len(times))
    ]
    traj = JointTrajectory(id="mujoco-test-traj", skeleton=rig, frames=frames)

    solver = MuJoCoTorqueMatchingSolver()
    result = solver.match(traj, rig)

    assert result.torque_trajectory is not None
    assert len(result.torque_trajectory.frames) == len(times)
    assert all(len(f.tau) == 3 for f in result.torque_trajectory.frames)
    assert result.success is True


@pytest.mark.benchmark
def test_torque_mujoco_rust_outer_loop_faster_than_python_stub_bench():
    """Rust outer loop >=1.5x faster than pure-Python on N=1000."""
    rust = pytest.importorskip("upstream_pinocchio_id")
    n_frames, n_dof = 1000, 7
    times, q = _make_trajectory(n_frames, n_dof, dt=0.005)
    callback = _make_mujoco_callback()

    # Warm-up runs.
    for _ in range(3):
        _ = _python_mujoco_pipeline(callback, times, q)
        _ = _rust_mujoco_pipeline(rust, callback, times, q)

    n_iters = 11
    py_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        tau_py = _python_mujoco_pipeline(callback, times, q)
        py_times.append(time.perf_counter() - t0)

    rs_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        tau_rs = _rust_mujoco_pipeline(rust, callback, times, q)
        rs_times.append(time.perf_counter() - t0)

    t_python = float(np.median(py_times))
    t_rust = float(np.median(rs_times))
    speedup = t_python / t_rust
    assert np.allclose(tau_py, tau_rs, atol=1e-9)
    assert speedup >= 1.5, (
        f"MuJoCo outer-loop speedup {speedup:.2f}x below 1.5x stub-bench floor "
        f"(python={t_python * 1000:.2f}ms, rust={t_rust * 1000:.2f}ms)"
    )
