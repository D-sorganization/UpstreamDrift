"""Unit tests for MuJoCo native inverse dynamics and mj_inverse computed torque tracking (MS-16 #10366)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.engines.physics_engines.mujoco.python.inverse_dynamics import (
    MujocoInverseDynamics,
)
from src.shared.python.contracts import ContractViolationError
from src.shared.python.motion_matching import full_body_forward_dynamics as fs
from src.shared.python.motion_matching.pipeline.constants import SPEC


@pytest.fixture(scope="module")
def native_model() -> NativeMujocoFullBodyModel:
    """Load cached NativeMujocoFullBodyModel from canonical SPEC."""
    spec_bytes = SPEC.read_bytes()
    return NativeMujocoFullBodyModel(spec_bytes)


@pytest.fixture(scope="module")
def inv_dynamics(native_model: NativeMujocoFullBodyModel) -> MujocoInverseDynamics:
    """Instantiate MujocoInverseDynamics adapter."""
    return MujocoInverseDynamics(native_model)


@pytest.fixture(scope="module")
def canonical_ik_trajectory() -> dict[str, np.ndarray]:
    """Load canonical IK reference trajectory."""
    from pathlib import Path

    traj_path = Path(
        "docs/development/full_body_models/evidence/ground_support/ik_trajectory.npz"
    )
    if not traj_path.exists():
        pytest.skip(f"Canonical IK trajectory not found at {traj_path}")
    data = np.load(traj_path)
    return {
        "time_s": np.asarray(data["time_s"], dtype=float),
        "q_ref": np.asarray(data["q_ref"], dtype=float),
    }


@pytest.mark.unit
def test_mujoco_mj_inverse_equivalence_without_contact(
    native_model: NativeMujocoFullBodyModel,
    inv_dynamics: MujocoInverseDynamics,
) -> None:
    """Prove exact roundtrip equivalence between mj_inverse and native forward dynamics."""
    rng = np.random.default_rng(42)
    q = rng.normal(scale=0.05, size=native_model.model.nq)
    v = rng.normal(scale=0.05, size=native_model.model.nv)
    a = rng.normal(scale=0.2, size=native_model.model.nv)

    # Compute torques with native inverse dynamics
    inv_dynamics.data.qpos[inv_dynamics.spec_to_mj] = q
    inv_dynamics.data.qvel[inv_dynamics.spec_to_mj] = v
    inv_dynamics.data.qacc[inv_dynamics.spec_to_mj] = a
    inv_dynamics._mj.mj_inverse(inv_dynamics.model, inv_dynamics.data)
    tau = inv_dynamics.data.qfrc_inverse.copy()

    # Apply torques in forward dynamics
    inv_dynamics.data.qacc[:] = 0.0
    inv_dynamics.data.qfrc_applied[:] = tau
    inv_dynamics.data.xfrc_applied[:] = 0.0
    inv_dynamics._mj.mj_forward(inv_dynamics.model, inv_dynamics.data)
    a_fwd = inv_dynamics.data.qacc[inv_dynamics.spec_to_mj].copy()

    assert np.all(np.isfinite(a_fwd))
    assert np.max(np.abs(a - a_fwd)) < 1e-10


@pytest.mark.unit
def test_mujoco_mj_inverse_equivalence_with_contact(
    native_model: NativeMujocoFullBodyModel,
    inv_dynamics: MujocoInverseDynamics,
) -> None:
    """Prove roundtrip equivalence accounting for external Hunt-Crossley contact wrenches."""
    sim = fs.FullBodySimulator(native_model)
    q0 = np.zeros(native_model.model.nq)
    q_preloaded = fs.preload_feet(sim, q0)
    v = np.zeros(native_model.model.nv)
    a = np.ones(native_model.model.nv) * 0.1

    coord_map = {
        name: float(q_preloaded[i])
        for i, name in enumerate(native_model.coordinate_order)
    }
    rate_map = {
        name: float(v[i]) for i, name in enumerate(native_model.coordinate_order)
    }
    _, tau_contact, samples = native_model.generalized_forces(coord_map, rate_map)
    xfrc_applied = native_model.data.xfrc_applied.copy()

    # Inverse dynamics with contact compensation
    tau = inv_dynamics.compute_inverse_torques(
        q_preloaded, v, a, compensate_contact=True
    )
    assert np.all(tau[:6] == 0.0)

    # Full torque including root for exact acceleration recovery test
    inv_dynamics.data.qpos[inv_dynamics.spec_to_mj] = q_preloaded
    inv_dynamics.data.qvel[inv_dynamics.spec_to_mj] = v
    inv_dynamics.data.qacc[inv_dynamics.spec_to_mj] = a
    inv_dynamics._mj.mj_inverse(inv_dynamics.model, inv_dynamics.data)
    tau_full = inv_dynamics.data.qfrc_inverse - tau_contact

    inv_dynamics.data.qacc[:] = 0.0
    inv_dynamics.data.qfrc_applied[:] = tau_full
    inv_dynamics.data.xfrc_applied[:] = xfrc_applied
    inv_dynamics._mj.mj_forward(inv_dynamics.model, inv_dynamics.data)
    a_fwd = inv_dynamics.data.qacc[inv_dynamics.spec_to_mj].copy()

    assert np.max(np.abs(a - a_fwd)) < 1e-8


@pytest.mark.unit
def test_mujoco_mj_inverse_rollout_50_steps_without_contact(
    native_model: NativeMujocoFullBodyModel,
    inv_dynamics: MujocoInverseDynamics,
    canonical_ik_trajectory: dict[str, np.ndarray],
) -> None:
    """Verify 50-step rollout discretization error budget under timestep refinement without contact."""
    times = canonical_ik_trajectory["time_s"][:51]
    q_traj = canonical_ik_trajectory["q_ref"][:51]
    v_traj = np.gradient(q_traj, times, axis=0)
    a_traj = np.gradient(v_traj, times, axis=0)
    q_target = q_traj[-1]

    def simulate(dt: float) -> np.ndarray:
        duration = times[-1] - times[0]
        steps = int(round(duration / dt))
        q_curr = q_traj[0].copy()
        v_curr = v_traj[0].copy()
        t_curr = times[0]
        for _ in range(steps):
            q_ref_t = np.array(
                [np.interp(t_curr, times, q_traj[:, i]) for i in range(41)]
            )
            v_ref_t = np.array(
                [np.interp(t_curr, times, v_traj[:, i]) for i in range(41)]
            )
            a_ref_t = np.array(
                [np.interp(t_curr, times, a_traj[:, i]) for i in range(41)]
            )

            inv_dynamics.data.qpos[inv_dynamics.spec_to_mj] = q_ref_t
            inv_dynamics.data.qvel[inv_dynamics.spec_to_mj] = v_ref_t
            inv_dynamics.data.qacc[inv_dynamics.spec_to_mj] = a_ref_t
            inv_dynamics._mj.mj_inverse(inv_dynamics.model, inv_dynamics.data)
            tau = inv_dynamics.data.qfrc_inverse.copy()

            inv_dynamics.data.qpos[inv_dynamics.spec_to_mj] = q_curr
            inv_dynamics.data.qvel[inv_dynamics.spec_to_mj] = v_curr
            inv_dynamics.data.qfrc_applied[:] = tau
            inv_dynamics.data.xfrc_applied[:] = 0.0
            inv_dynamics._mj.mj_forward(inv_dynamics.model, inv_dynamics.data)

            a_curr = inv_dynamics.data.qacc[inv_dynamics.spec_to_mj].copy()
            v_curr += a_curr * dt
            q_curr += v_curr * dt
            t_curr += dt
        return q_curr

    dt_coarse = float(times[1] - times[0])
    dt_fine = dt_coarse / 2.0

    q_coarse = simulate(dt_coarse)
    q_fine = simulate(dt_fine)

    err_coarse = float(np.max(np.abs(q_coarse - q_target)))
    err_fine = float(np.max(np.abs(q_fine - q_target)))

    # Declared discretization error budget:
    # 1. Coarse error under 0.03 rad
    # 2. Refined error under 0.015 rad
    # 3. Monotonic improvement: fine < coarse
    # 4. Refinement ratio around 2.0 (semi-implicit Euler)
    assert err_coarse < 0.03, f"Coarse error {err_coarse} exceeds 0.03 rad budget"
    assert err_fine < 0.015, f"Fine error {err_fine} exceeds 0.015 rad budget"
    assert err_fine < err_coarse, "Timestep refinement failed to decrease error"
    assert 1.8 <= (err_coarse / err_fine) <= 2.2, (
        f"Refinement ratio {err_coarse / err_fine} deviates from 2.0"
    )


@pytest.mark.unit
def test_mujoco_mj_inverse_rollout_50_steps_with_contact(
    native_model: NativeMujocoFullBodyModel,
    inv_dynamics: MujocoInverseDynamics,
    canonical_ik_trajectory: dict[str, np.ndarray],
) -> None:
    """Verify 50-step rollout discretization error budget under timestep refinement with contact."""
    sim = fs.FullBodySimulator(native_model)
    times = canonical_ik_trajectory["time_s"][:51]
    q_traj = canonical_ik_trajectory["q_ref"][:51].copy()
    for k in range(len(q_traj)):
        q_traj[k] = fs.preload_feet(sim, q_traj[k])

    v_traj = np.gradient(q_traj, times, axis=0)
    a_traj = np.gradient(v_traj, times, axis=0)
    q_target = q_traj[-1]

    def simulate(dt: float) -> np.ndarray:
        duration = times[-1] - times[0]
        steps = int(round(duration / dt))
        q_curr = q_traj[0].copy()
        v_curr = v_traj[0].copy()
        t_curr = times[0]
        for _ in range(steps):
            q_ref_t = np.array(
                [np.interp(t_curr, times, q_traj[:, i]) for i in range(41)]
            )
            v_ref_t = np.array(
                [np.interp(t_curr, times, v_traj[:, i]) for i in range(41)]
            )
            a_ref_t = np.array(
                [np.interp(t_curr, times, a_traj[:, i]) for i in range(41)]
            )

            inv_dynamics.data.qpos[inv_dynamics.spec_to_mj] = q_ref_t
            inv_dynamics.data.qvel[inv_dynamics.spec_to_mj] = v_ref_t
            inv_dynamics.data.qacc[inv_dynamics.spec_to_mj] = a_ref_t
            inv_dynamics._mj.mj_inverse(inv_dynamics.model, inv_dynamics.data)

            coord_map = {
                name: float(q_ref_t[i])
                for i, name in enumerate(native_model.coordinate_order)
            }
            rate_map = {
                name: float(v_ref_t[i])
                for i, name in enumerate(native_model.coordinate_order)
            }
            _, tau_contact, _ = native_model.generalized_forces(coord_map, rate_map)
            tau = inv_dynamics.data.qfrc_inverse - tau_contact
            xfrc_copy = native_model.data.xfrc_applied.copy()

            inv_dynamics.data.qpos[inv_dynamics.spec_to_mj] = q_curr
            inv_dynamics.data.qvel[inv_dynamics.spec_to_mj] = v_curr
            inv_dynamics.data.qfrc_applied[:] = tau
            inv_dynamics.data.xfrc_applied[:] = xfrc_copy
            inv_dynamics._mj.mj_forward(inv_dynamics.model, inv_dynamics.data)

            a_curr = inv_dynamics.data.qacc[inv_dynamics.spec_to_mj].copy()
            v_curr += a_curr * dt
            q_curr += v_curr * dt
            t_curr += dt
        return q_curr

    dt_coarse = float(times[1] - times[0])
    dt_fine = dt_coarse / 2.0

    q_coarse = simulate(dt_coarse)
    q_fine = simulate(dt_fine)

    err_coarse = float(np.max(np.abs(q_coarse - q_target)))
    err_fine = float(np.max(np.abs(q_fine - q_target)))

    # Declared discretization error budget with contact:
    # 1. Coarse error under 0.25 rad
    # 2. Refined error under 0.15 rad
    # 3. Monotonic improvement: fine < coarse
    # 4. Refinement ratio around 2.0
    assert err_coarse < 0.25, (
        f"Coarse contact error {err_coarse} exceeds 0.25 rad budget"
    )
    assert err_fine < 0.15, f"Fine contact error {err_fine} exceeds 0.15 rad budget"
    assert err_fine < err_coarse, "Contact refinement failed to decrease error"
    assert 1.8 <= (err_coarse / err_fine) <= 2.2, (
        f"Contact refinement ratio {err_coarse / err_fine} deviates from 2.0"
    )


@pytest.mark.unit
def test_mujoco_mj_inverse_tracking_controller_reproduces_trajectory(
    native_model: NativeMujocoFullBodyModel,
    inv_dynamics: MujocoInverseDynamics,
    canonical_ik_trajectory: dict[str, np.ndarray],
) -> None:
    """Verify tracking controller produces stable forward dynamics tracking in FullBodySimulator."""
    sim = fs.FullBodySimulator(native_model)
    times = canonical_ik_trajectory["time_s"][:51]
    q_traj = canonical_ik_trajectory["q_ref"][:51].copy()

    q0 = fs.preload_feet(sim, q_traj[0])
    v0 = np.gradient(q_traj, times, axis=0)[0]

    controller = inv_dynamics.create_tracking_controller(
        sim, times, q_traj, omega_rad_s=25.0, zeta=1.0, balance=(20.0, 5.0)
    )

    record = sim.run(
        q0,
        v0,
        controller,
        duration_s=float(times[-1] - times[0]),
        dt_s=1e-3,
        record_every=1,
    )

    assert len(record.time_s) > 0
    assert np.all(np.isfinite(record.q))
    assert np.all(np.isfinite(record.v))
    assert np.all(np.isfinite(record.tau))
    # Root effort must be strictly zero across entire record
    assert np.all(record.tau[:, :6] == 0.0)

    # Actuated joint tracking error at end of 50 frames
    q_final = np.array(
        [np.interp(record.time_s[-1], times, q_traj[:, i]) for i in range(41)]
    )
    final_err = float(np.max(np.abs(record.q[-1, 6:] - q_final[6:])))
    assert final_err < 0.08, f"Tracking error {final_err} rad exceeds 0.08 rad limit"


@pytest.mark.unit
def test_mujoco_mj_inverse_fails_closed_on_invalid_inputs(
    inv_dynamics: MujocoInverseDynamics,
) -> None:
    """Verify DbC failure closed when given invalid, nonfinite or mismatched inputs."""
    valid_q = np.zeros(inv_dynamics.nv)
    valid_v = np.zeros(inv_dynamics.nv)
    valid_a = np.zeros(inv_dynamics.nv)

    with pytest.raises((ContractViolationError, AssertionError, ValueError)):
        inv_dynamics.compute_inverse_torques(
            np.array([np.nan] * inv_dynamics.nv), valid_v, valid_a
        )

    with pytest.raises((ContractViolationError, AssertionError, ValueError)):
        inv_dynamics.compute_inverse_torques(valid_q[:10], valid_v, valid_a)

    with pytest.raises((ContractViolationError, AssertionError, ValueError)):
        inv_dynamics.compute_inverse_torques(
            valid_q, valid_v, np.array([np.inf] * inv_dynamics.nv)
        )
