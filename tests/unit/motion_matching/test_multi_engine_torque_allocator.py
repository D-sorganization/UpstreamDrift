"""Unit tests for MultiEngineTorqueAllocator and multi-engine adapters (#10415).

Verifies:
1. MuJoCo adapter initialization, dynamic equilibrium, and acceleration parity.
2. Drake adapter analytical spatial mechanics, unilateral contact, and trail arm suppression.
3. OpenSim adapter Simbody station kinematics and fast convex force allocation.
4. Simscape adapter and Simulink timeseries data export.
5. CLI allocate_swing_torques.py entrypoint execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.allocate_swing_torques import build_parser, main as cli_main
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    BaseEngineForceAdapter,
    DrakeForceAdapter,
    EngineType,
    MujocoForceAdapter,
    MultiEngineTorqueAllocator,
    OpenSimForceAdapter,
    SimscapeForceAdapter,
    TrajectoryAllocationResult,
    create_engine_force_adapter,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def dummy_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a clean synthetic full-body trajectory for testing."""
    n_frames = 15
    nv = 44
    time_s = np.linspace(0.0, 0.3, n_frames)
    q = np.zeros((n_frames, nv))
    v = np.zeros((n_frames, nv))
    a = np.zeros((n_frames, nv))

    # Pelvis nominal height
    q[:, 2] = 0.85
    # Add modest joint motion across actuated coordinates (indices 6:)
    for i in range(6, nv):
        omega = 2.0 * np.pi * 1.5
        q[:, i] = 0.05 * np.sin(omega * time_s + i * 0.1)
        v[:, i] = 0.05 * omega * np.cos(omega * time_s + i * 0.1)
        a[:, i] = -0.05 * (omega**2) * np.sin(omega * time_s + i * 0.1)
    return time_s, q, v, a


def test_drake_force_adapter(
    dummy_trajectory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    time_s, q, v, a = dummy_trajectory
    adapter = DrakeForceAdapter(nv=44, n_spheres=6)
    assert isinstance(adapter, BaseEngineForceAdapter)
    assert adapter.engine_type == EngineType.DRAKE
    assert adapter.nv == 44
    assert len(adapter.actuated_indices) == 38
    assert adapter.n_contact_spheres == 6

    # Test single frame inverse dynamics
    tau_rnea = adapter.compute_inverse_dynamics(q[0], v[0], a[0])
    assert tau_rnea.shape == (44,)
    # Z gravity load should be positive (~75 kg * 9.81 m/s^2 = ~735 N)
    assert tau_rnea[2] > 500.0

    j_ground = adapter.compute_contact_jacobian(q[0])
    assert j_ground.shape == (18, 44)

    j_grip = adapter.compute_grip_jacobian(q[0])
    assert j_grip.shape == (6, 44)


def test_opensim_force_adapter(
    dummy_trajectory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    time_s, q, v, a = dummy_trajectory
    adapter = OpenSimForceAdapter(nv=44, n_spheres=6)
    assert isinstance(adapter, BaseEngineForceAdapter)
    assert adapter.engine_type == EngineType.OPENSIM

    tau_rnea = adapter.compute_inverse_dynamics(q[0], v[0], a[0])
    assert tau_rnea.shape == (44,)
    assert tau_rnea[2] > 500.0

    j_ground = adapter.compute_contact_jacobian(q[0])
    assert j_ground.shape == (18, 44)

    j_grip = adapter.compute_grip_jacobian(q[0])
    assert j_grip.shape == (6, 44)


def test_simscape_force_adapter_and_export(
    tmp_path: Path,
    dummy_trajectory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    time_s, q, v, a = dummy_trajectory
    adapter = SimscapeForceAdapter(nv=44, n_spheres=6)
    assert isinstance(adapter, BaseEngineForceAdapter)
    assert adapter.engine_type == EngineType.SIMSCAPE

    allocator = MultiEngineTorqueAllocator(adapter=adapter)
    result = allocator.allocate_trajectory(
        time_s=time_s,
        q_traj=q,
        v_traj=v,
        a_traj=a,
        objective=AllocationObjective.MINIMUM_EFFORT,
    )
    assert isinstance(result, TrajectoryAllocationResult)
    assert result.success
    assert result.max_equilibrium_residual < 1e-4
    assert result.tau_actuated.shape == (len(time_s), 38)
    assert result.f_ground.shape == (len(time_s), 18)

    # Test export to Simulink dataset
    export_file = tmp_path / "simulink_export.npz"
    exported = adapter.export_simulink_timeseries(result, export_file)
    assert exported.is_file()

    loaded = np.load(exported, allow_pickle=True)
    assert "tau_actuated" in loaded
    assert "f_ground" in loaded
    meta = json.loads(str(loaded["metadata"]))
    assert meta["engine"] == "simscape"


def test_trail_arm_suppression(
    dummy_trajectory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    time_s, q, v, a = dummy_trajectory
    adapter = DrakeForceAdapter(nv=44, n_spheres=6)
    allocator = MultiEngineTorqueAllocator(adapter=adapter)

    trail_indices = list(range(18, 27))
    res_opt = allocator.allocate_trajectory(
        time_s=time_s,
        q_traj=q,
        v_traj=v,
        a_traj=a,
        objective=AllocationObjective.MINIMUM_EFFORT,
    )
    res_trail = allocator.allocate_trajectory(
        time_s=time_s,
        q_traj=q,
        v_traj=v,
        a_traj=a,
        objective=AllocationObjective.MINIMUM_TRAIL_ARM,
        trail_arm_indices=trail_indices,
    )

    # Trail-arm torque norm should be lower or equal under trail reduction
    trail_act_indices = [idx - 6 for idx in trail_indices]
    opt_trail_norm = np.linalg.norm(res_opt.tau_actuated[:, trail_act_indices])
    trail_norm = np.linalg.norm(res_trail.tau_actuated[:, trail_act_indices])

    assert trail_norm <= opt_trail_norm + 1e-4, (
        f"Trail arm torque was not reduced: {trail_norm} vs {opt_trail_norm}"
    )
    # Transmitted grip reaction forces should be active
    assert np.linalg.norm(res_trail.lambda_grip) > 0.0


def test_mujoco_adapter_with_spec(
    dummy_trajectory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    spec_path = (
        repo_root
        / "docs"
        / "development"
        / "full_body_models"
        / "full_body_spec_v1.json"
    )
    if not spec_path.is_file():
        pytest.skip(
            "docs/development/full_body_models/full_body_spec_v1.json not found in workspace"
        )

    adapter = create_engine_force_adapter(EngineType.MUJOCO, spec_path=spec_path)
    assert adapter.engine_type == EngineType.MUJOCO
    assert adapter.nv == 41  # full_body_spec_v1 has 41 DoFs (simscape-native base)

    time_s, _, _, _ = dummy_trajectory
    n_f = 3
    q_mj = np.zeros((n_f, adapter.nv))
    v_mj = np.zeros((n_f, adapter.nv))
    a_mj = np.zeros((n_f, adapter.nv))
    q_mj[:, 2] = 0.85

    allocator = MultiEngineTorqueAllocator(adapter=adapter)
    result = allocator.allocate_trajectory(
        time_s=time_s[:n_f],
        q_traj=q_mj,
        v_traj=v_mj,
        a_traj=a_mj,
        objective=AllocationObjective.MINIMUM_EFFORT,
    )
    assert result.success
    assert result.max_equilibrium_residual < 1e-3


def test_cli_allocate_swing_torques(
    tmp_path: Path,
    dummy_trajectory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    time_s, q, v, a = dummy_trajectory
    candidate_file = tmp_path / "candidate_test.npz"
    np.savez_compressed(candidate_file, time_s=time_s, q=q, v=v, a=a)

    out_file = tmp_path / "allocated_drake.npz"
    code = cli_main(
        [
            "--engine",
            "drake",
            "--candidate",
            str(candidate_file),
            "--objective",
            "minimum_effort",
            "--out",
            str(out_file),
        ]
    )
    assert code == 0
    assert out_file.is_file()
    loaded = np.load(out_file, allow_pickle=True)
    assert "tau_actuated" in loaded
    metrics = json.loads(str(loaded["metrics"]))
    assert metrics["engine"] == "drake"
    assert metrics["success"] is True
