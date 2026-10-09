"""One-hinge native feedback execution and independent torque replay (F05).

The planner may inspect the current native state while driving the live plant.
The resulting Tools bundle contains only executed, post-limit time-only motor
torques. Its separate native replay receives no controller or observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    NativeTorqueReplay,
    build_native_torque_bundle,
    replay_native_torque_bundle,
)
from src.shared.python.motion_matching.bounded_nmpc import (
    BoundedNMPC,
    MPCCommandReceipt,
)

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle


def _frozen(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeNMPCTracking:
    """One live controlled run and independently reproduced native trajectory."""

    bundle: ExperimentReplayBundle
    replay: NativeTorqueReplay
    commands: tuple[MPCCommandReceipt, ...]
    native_qpos: NDArray[np.float64]
    native_qvel: NDArray[np.float64]
    native_integration_states: NDArray[np.float64]
    control_wall_s: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "native_qpos",
            "native_qvel",
            "native_integration_states",
            "control_wall_s",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name)))


def _admit_native_model(path: Path, controller: BoundedNMPC) -> Any:
    """Reject any topology or channel outside the direct hinge fixture."""
    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(path))
    if (
        model.nbody != 2
        or model.ngeom != 1
        or model.njnt != 1
        or model.nq != 1
        or model.nv != 1
        or model.nu != 1
    ):
        raise ValueError("F05 native tracking admits one contact-free hinge motor")
    joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, 0)
    actuator_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, 0)
    if (
        controller.problem.state_component_ids != (f"{joint_name}_q", f"{joint_name}_v")
        or controller.problem.state_units != ("rad", "rad/s")
        or controller.problem.input_channel_ids != (actuator_name,)
        or controller.problem.input_units != ("N*m",)
        or not np.isclose(
            model.opt.timestep, controller.problem.time_step_s, atol=1e-12, rtol=0
        )
    ):
        raise ValueError("controller state, actuator or native step identity differs")
    return model


def run_native_nmpc_tracking(
    model_path: str | Path,
    initial_integration_state: NDArray[np.float64],
    controller: BoundedNMPC,
    *,
    steps: int,
    experiment_id: str,
) -> NativeNMPCTracking:
    """Drive direct unit-motor MuJoCo, then certify frozen-input replay.

    Only a contact-free one-hinge fixture is admitted. The F06 bundle builder
    validates the loaded model, complete native state, channel semantics and
    exact ZOH policy before and after the controlled run.
    """
    import mujoco as mj

    path = Path(model_path)
    if steps < 1 or not experiment_id or controller.applied_history:
        raise ValueError("native run needs positive steps, ID and fresh controller")
    model = _admit_native_model(path, controller)
    dt = float(model.opt.timestep)
    times = np.arange(steps + 1, dtype=np.float64) * dt
    zeros = np.zeros((steps + 1, 1), dtype=np.float64)
    preflight = build_native_torque_bundle(
        path, initial_integration_state, times, zeros, experiment_id=experiment_id
    )
    data = mj.MjData(model)
    model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_AUTORESET)
    mj.mj_setState(
        model,
        data,
        np.asarray(initial_integration_state, dtype=np.float64),
        mj.mjtState.mjSTATE_INTEGRATION,
    )
    if data.time != 0.0 or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("native initial state contains time or external load")
    integration_size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION)
    full = np.empty((steps + 1, integration_size), dtype=np.float64)
    qpos = np.empty((steps + 1, 1), dtype=np.float64)
    qvel = np.empty((steps + 1, 1), dtype=np.float64)
    applied = np.empty((steps + 1, 1), dtype=np.float64)
    control_wall_s = np.empty(steps, dtype=np.float64)
    commands: list[MPCCommandReceipt] = []
    for row in range(steps + 1):
        mj.mj_getState(model, data, full[row], mj.mjtState.mjSTATE_INTEGRATION)
        qpos[row], qvel[row] = data.qpos, data.qvel
        if row == steps:
            break
        current = float(times[row])
        started = time.perf_counter()
        receipt = controller.command_for_step(
            row,
            np.array([data.qpos[0], data.qvel[0]], dtype=np.float64),
            observation_time_s=current,
            current_time_s=current,
        )
        control_wall_s[row] = time.perf_counter() - started
        if receipt.input_boundary != "post_limit_actuator_command":
            raise ValueError("controller output is not post-limit actuator input")
        applied[row] = receipt.applied
        commands.append(receipt)
        data.ctrl[:] = receipt.applied
        mj.mj_step(model, data)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isclose(data.time, times[row + 1], atol=1e-12, rtol=0)
            or data.ncon
        ):
            raise RuntimeError("native run diverged, reset or developed contact")
    applied[-1] = applied[-2]  # Terminal sentinel is never executed.
    bundle = build_native_torque_bundle(
        path, initial_integration_state, times, applied, experiment_id=experiment_id
    )
    if bundle.model != preflight.model or bundle.policy != preflight.policy:
        raise ValueError("native model or executed policy changed during tracking")
    replay = replay_native_torque_bundle(bundle, path)
    for direct, reproduced in (
        (qpos, replay.qpos),
        (qvel, replay.qvel),
        (full, replay.integration_states),
    ):
        if not np.allclose(direct, reproduced, atol=1e-12, rtol=0):
            raise RuntimeError("frozen-input replay differs from controlled native run")
    return NativeNMPCTracking(
        bundle,
        replay,
        tuple(commands),
        qpos,
        qvel,
        full,
        control_wall_s,
    )
