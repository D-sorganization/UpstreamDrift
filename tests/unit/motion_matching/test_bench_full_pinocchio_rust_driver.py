"""Heavy-integration benchmark for the end-to-end Pinocchio Rust driver.

Part of issue #5254 (slice 5).
Verifies that the Rust outer-loop acceleration (via upstream_pinocchio_id)
yields at least a 3x speedup on a realistic N=1000, DOF=30 trajectory
when using the real Pinocchio rnea inner-loop.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.matching.inverse_dyn_pinocchio import (
    _HAVE_RUST_PIN_ID,
    PinocchioInverseDynMatchingSolver,
)


def _make_heavy_rig(n_dof: int = 30) -> SkeletonRig:
    """Build a serial rig with `n_dof` DOFs to exercise Pinocchio."""
    joints = {}
    parent = None
    for i in range(n_dof):
        name = f"joint_{i}"
        joints[name] = JointDef(
            name=name,
            parent=parent,
            children=[] if i == n_dof - 1 else [f"joint_{i + 1}"],
            tpose_offset=[0.1, 0.0, 0.0],
            axes=["Z"],
        )
        parent = name
    return SkeletonRig(id=f"heavy-rig-{n_dof}", joints=joints, root_joint="joint_0")


def _make_heavy_trajectory(
    rig: SkeletonRig, n_frames: int = 1000, dt: float = 0.005
) -> JointTrajectory:
    """Build a kinematic trajectory for the heavy rig."""
    n_dof = sum(len(j.axes) for j in rig.joints.values())
    rng = np.random.default_rng(42)
    times = np.arange(n_frames, dtype=float) * dt
    t = times[:, None]

    phases = rng.uniform(0, 2 * np.pi, size=(1, n_dof))
    amps = rng.uniform(0.1, 0.5, size=(1, n_dof))
    freqs = rng.uniform(0.5, 2.0, size=(1, n_dof))
    q = amps * np.sin(2 * np.pi * freqs * t + phases)

    frames = [
        JointStateFrame(timestamp=float(times[i]), q=q[i].tolist(), frame_index=i)
        for i in range(n_frames)
    ]
    return JointTrajectory(id="heavy-traj", skeleton=rig, frames=frames)


@pytest.mark.benchmark
@pytest.mark.skipif(not _HAVE_RUST_PIN_ID, reason="upstream_pinocchio_id not installed")
def test_bench_full_pinocchio_rust_driver_speedup():
    """End-to-end benchmark: Rust driver vs pure-Python driver.

    Asserts >= 3x speedup on N=1000 x 30 DOFs.
    Also asserts strict numerical parity (<1e-9 RMSE).
    """
    pytest.importorskip("pinocchio")

    n_frames = 1000
    n_dof = 30
    rig = _make_heavy_rig(n_dof)
    traj = _make_heavy_trajectory(rig, n_frames=n_frames, dt=0.005)

    solver = PinocchioInverseDynMatchingSolver()

    # Run with Rust enabled
    os.environ["RUST_OUTER_LOOP"] = "1"

    # Warmup
    for _ in range(2):
        _ = solver.match(traj, rig)

    n_iters = 5
    rs_times = []
    rs_result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        rs_result = solver.match(traj, rig)
        rs_times.append(time.perf_counter() - t0)

    # Run with Rust disabled
    os.environ["RUST_OUTER_LOOP"] = "0"

    # Warmup
    for _ in range(2):
        _ = solver.match(traj, rig)

    py_times = []
    py_result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        py_result = solver.match(traj, rig)
        py_times.append(time.perf_counter() - t0)

    # Restore env
    os.environ.pop("RUST_OUTER_LOOP", None)

    # Analyze
    t_rust = float(np.median(rs_times))
    t_python = float(np.median(py_times))
    speedup = t_python / t_rust

    assert rs_result is not None and py_result is not None
    assert rs_result.success and py_result.success

    # Check parity
    for i in range(n_frames):
        tau_rs = np.array(rs_result.torque_trajectory.frames[i].tau)
        tau_py = np.array(py_result.torque_trajectory.frames[i].tau)
        assert np.allclose(tau_rs, tau_py, atol=1e-9)

    assert speedup >= 3.0, (
        f"End-to-end Pinocchio speedup {speedup:.2f}x below 3.0x target "
        f"(python={t_python * 1000:.2f}ms, rust={t_rust * 1000:.2f}ms)"
    )
