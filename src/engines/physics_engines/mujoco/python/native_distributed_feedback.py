"""F02 exact-state feedback on an admitted native MuJoCo configuration manifold.

The controller reads the live state once per integration step. Only its actual
post-limit motor torques enter the time-only Tools bundle and fresh native replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.native_nmpc_tracking import (
    NativeNMPCTracking,
    run_native_direct_torque_tracking,
)
from src.engines.physics_engines.mujoco.python.native_tangent_derivative import (
    linearize_native_tangent_step,
)
from src.shared.python.motion_matching.bounded_nmpc import MPCCommandReceipt
from src.shared.python.motion_matching.distributed_feedback import (
    ActuatorMap,
    ControlStep,
    DirectBoundedAllocator,
    DistributedFeedbackController,
    EffortAllocator,
    NominalTrajectory,
)

Array = NDArray[np.float64]


class NativeMuJoCoTangentModel:
    """MuJoCo's native current-minus-reference chart and physical mass matrix."""

    def __init__(self, model: Any, initial_integration_state: Array) -> None:
        derivative = linearize_native_tangent_step(
            model, initial_integration_state, np.zeros(model.nu)
        )
        self.model = model
        self.nq = int(model.nq)
        self.nv = int(model.nv)
        self.ordered_input_channel_ids = derivative.ordered_input_channel_ids

    def _configuration(self, q: Array) -> Array:
        import mujoco as mj

        values = np.asarray(q, dtype=np.float64)
        if values.shape != (self.nq,) or not np.isfinite(values).all():
            raise ValueError("native configuration requires finite nq values")
        normalized = values.copy()
        mj.mj_normalizeQuat(self.model, normalized)
        if not np.allclose(values, normalized, rtol=0, atol=1e-12):
            raise ValueError("native configuration quaternion must be normalized")
        return values

    def difference(self, q: Array, reference: Array) -> Array:
        import mujoco as mj

        current = self._configuration(q)
        nominal = self._configuration(reference)
        tangent = np.empty(self.nv, dtype=np.float64)
        mj.mj_differentiatePos(self.model, tangent, 1.0, nominal, current)
        if not np.isfinite(tangent).all():
            raise ValueError("native manifold difference is nonfinite")
        return tangent

    def mass_inverse(self, q: Array) -> Array:
        import mujoco as mj

        data = mj.MjData(self.model)
        data.qpos[:] = self._configuration(q)
        mj.mj_forward(self.model, data)
        mass = np.empty((self.nv, self.nv), dtype=np.float64)
        mj.mj_fullM(self.model, mass, data.qM)
        if not np.isfinite(mass).all():
            raise ValueError("native mass matrix is nonfinite")
        return np.asarray(np.linalg.inv(mass), dtype=np.float64)


class _FeedbackStepAdapter:
    """Translate F02 ControlStep into the shared native applied-input boundary."""

    def __init__(self, controller: DistributedFeedbackController, dt: float) -> None:
        self.controller = controller
        self.dt = dt
        self._commands: list[MPCCommandReceipt] = []
        self.control_steps: list[ControlStep] = []

    @property
    def applied_history(self) -> tuple[MPCCommandReceipt, ...]:
        return tuple(self._commands)

    def command_for_step(
        self,
        index: int,
        observed_state: Array,
        *,
        observation_time_s: float,
        current_time_s: float,
    ) -> MPCCommandReceipt:
        if index != len(self._commands) or observation_time_s != current_time_s:
            raise ValueError("feedback requires contiguous exact-state observation")
        tangent_model = self.controller.model
        nq = tangent_model.nq
        nv = tangent_model.nv
        if observed_state.shape != (nq + nv,):
            raise ValueError("feedback observation requires complete qpos and qvel")
        step = self.controller.command_for_step(
            current_time_s, observed_state[:nq], observed_state[nq:], self.dt
        )
        if (
            step.input_boundary != "actuator_torque"
            or step.contact_mode != "contact_free"
        ):
            raise ValueError(
                "native F02 requires post-limit direct torque and contact-free plant"
            )
        receipt = MPCCommandReceipt(
            status="feedback_applied" if self.controller.enabled else "nominal_only",
            applied=step.applied,
            fallback=step.nominal_feedforward,
            objective=None,
            fallback_objective=None,
            evaluations=0,
            elapsed_s=0.0,
            warm_started=False,
            scenario_count=1,
        )
        self.control_steps.append(step)
        self._commands.append(receipt)
        return receipt


@dataclass(frozen=True)
class NativeDistributedFeedbackTracking:
    """Native closed-loop rollout plus independently replayed frozen motor inputs."""

    tracking: NativeNMPCTracking
    control_steps: tuple[ControlStep, ...]


def run_native_distributed_feedback_tracking(
    model_path: str | Path,
    initial_integration_state: Array,
    schedule: NominalTrajectory,
    *,
    max_torque_rate_nm_s: Array,
    experiment_id: str,
    enabled: bool = True,
) -> NativeDistributedFeedbackTracking:
    """Apply F02 TVLQR to admitted 9/8/2 native motor plant, then replay."""
    import mujoco as mj

    path = Path(model_path)
    model = mj.MjModel.from_xml_path(str(path))
    tangent = NativeMuJoCoTangentModel(model, initial_integration_state)
    dt = float(model.opt.timestep)
    if schedule.channel_ids != tangent.ordered_input_channel_ids:
        raise ValueError("nominal channel order differs from native motor order")
    if not np.allclose(
        schedule.times, np.arange(schedule.steps + 1) * dt, rtol=0, atol=1e-12
    ):
        raise ValueError("nominal clock differs from native integration grid")
    for q in schedule.q:
        tangent._configuration(q)
    joints = model.actuator_trnid[:, 0]
    actuators = ActuatorMap(
        tangent.ordered_input_channel_ids,
        tuple(int(model.jnt_dofadr[joint]) for joint in joints),
        tuple(range(6)),
    )
    allocator = DirectBoundedAllocator(
        actuators,
        model.actuator_ctrlrange[:, 0].copy(),
        model.actuator_ctrlrange[:, 1].copy(),
        max_torque_rate_nm_s,
        contact_free=True,
    )
    # The shared EffortAllocator Protocol declares actuators mutable; the
    # concrete frozen allocator has the identical runtime field and method.
    controller = DistributedFeedbackController(
        tangent, actuators, schedule, cast(EffortAllocator, allocator), enabled=enabled
    )
    adapter = _FeedbackStepAdapter(controller, dt)
    tracking = run_native_direct_torque_tracking(
        path,
        model,
        initial_integration_state,
        adapter,
        lambda data: np.r_[data.qpos, data.qvel],
        steps=schedule.steps,
        experiment_id=experiment_id,
    )
    return NativeDistributedFeedbackTracking(tracking, tuple(adapter.control_steps))
