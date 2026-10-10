"""F04 native train/holdout loop evaluator and paired motor interventions.

The F02 controller, F06 exact torque bundle and native replay own physics.
This adapter only maps two bounded gain scales to F04 trial losses and
separately measures an in-model motor-to-joint intervention. Neither its
loss covariance nor its intervention matrix identifies human neural control.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.native_distributed_feedback import (
    NativeMuJoCoTangentModel,
    run_native_distributed_feedback_tracking,
)
from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    _load_native,
    build_native_torque_bundle,
    replay_native_torque_bundle,
)
from src.shared.python.motion_matching.control_loop_tuning import (
    LoopParameter,
    LoopTuningProblem,
    TuningEvaluation,
    TuningTrial,
)
from src.shared.python.motion_matching.distributed_feedback import NominalTrajectory

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

Array = NDArray[np.float64]
_GROUPS = ("hip", "knee")
_CHANNELS = ("hip_torque", "knee_torque")


@dataclass(frozen=True)
class NativeMotorInterventionEvidence:
    """Signed native joint response to symmetric, frozen motor perturbations."""

    response_rad_per_nm: Array
    channel_ids: tuple[str, str]
    response_joint_ids: tuple[str, str]
    applied_input_sha256: tuple[str, str, str, str]
    initial_state_sha256: str
    policy_sha256: str
    time_grid_sha256: str
    interpretation: str = "synthetic_plant_input_intervention_not_human_control"

    def __post_init__(self) -> None:
        values = np.asarray(self.response_rad_per_nm, dtype=np.float64)
        if values.shape != (2, 2) or not np.isfinite(values).all():
            raise ValueError("native intervention matrix must be finite 2x2")
        frozen = values.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "response_rad_per_nm", frozen)


class NativeCoupledLoopEvaluator:
    """Map frozen F02 row gains to actual native F04 train/holdout losses."""

    def __init__(
        self,
        model_path: str | Path,
        reference_bundle: ExperimentReplayBundle,
        schedule: NominalTrajectory,
        trial_states: Mapping[str, Array],
        *,
        rate_limit_nm_s: Array,
    ) -> None:
        path = Path(model_path)
        model, _ = _load_native(path)
        if (model.nq, model.nv, model.nu) != (9, 8, 2):
            raise ValueError("native loop tuning requires nq=9/nv=8/nu=2")
        reference = replay_native_torque_bundle(reference_bundle, path)
        if (
            schedule.channel_ids != _CHANNELS
            or tuple(reference_bundle.model.ordered_input_channel_ids) != _CHANNELS
            or len(reference.time_seconds) != schedule.steps + 1
            or not np.array_equal(reference.time_seconds, schedule.times)
            or not np.array_equal(reference.qpos[:-1], schedule.q)
            or not np.array_equal(reference.qvel[:-1], schedule.v)
            or not np.array_equal(
                reference.applied_actuator_torques, schedule.feedforward
            )
        ):
            raise ValueError("native reference policy differs from frozen replay")
        phases = tuple(dict.fromkeys(schedule.phase_names))
        if (
            len(phases) != 2
            or not trial_states
            or any(not name for name in trial_states)
        ):
            raise ValueError("two phases and named native trials are required")
        rate = np.asarray(rate_limit_nm_s, dtype=np.float64)
        if rate.shape != (2,) or not np.isfinite(rate).all() or np.any(rate <= 0):
            raise ValueError("native loop rate limits must be positive per motor")
        states: dict[str, Array] = {}
        for trial_id, initial in trial_states.items():
            native = build_native_torque_bundle(
                path,
                np.asarray(initial, dtype=np.float64),
                schedule.times,
                np.vstack((schedule.feedforward, schedule.feedforward[-1])),
                experiment_id=f"f04-preflight-{trial_id}",
            )
            if (
                native.model != reference_bundle.model
                or native.policy != reference_bundle.policy
            ):
                raise ValueError("native trial model or policy differs from reference")
            frozen = np.asarray(initial, dtype=np.float64).copy()
            frozen.setflags(write=False)
            states[trial_id] = frozen
        self.path = path
        self.model = model
        self.tangent = NativeMuJoCoTangentModel(model, reference.integration_states[0])
        self.reference = reference
        self.schedule = schedule
        self.phases = phases
        self.rate = rate.copy()
        self.rate.setflags(write=False)
        self.trial_states = MappingProxyType(states)
        self.reference_model = reference_bundle.model
        self.reference_policy = reference_bundle.policy
        self._trace: list[tuple[str, tuple[float, float]]] = []
        self._applied_hashes: list[str] = []
        self._max_replay_error = 0.0

    @property
    def evaluations_by_split(self) -> dict[str, int]:
        return {
            split: sum(item[0] == split for item in self._trace)
            for split in ("train", "holdout")
        }

    @property
    def uncertainty_status(self) -> str:
        """No interval is inferred from the small synthetic trial inventory."""
        return "unavailable_no_resampling_or_capture_noise_model"

    @property
    def applied_input_hashes(self) -> tuple[str, ...]:
        return tuple(self._applied_hashes)

    @property
    def max_full_state_replay_error(self) -> float:
        return self._max_replay_error

    @property
    def parameters_by_split(self) -> dict[str, tuple[tuple[float, float], ...]]:
        return {
            split: tuple(values for label, values in self._trace if label == split)
            for split in ("train", "holdout")
        }

    def problem(
        self,
        *,
        train_ids: tuple[str, ...],
        holdout_ids: tuple[str, ...],
        initial_scales: tuple[float, float],
    ) -> LoopTuningProblem:
        if (
            not train_ids
            or not holdout_ids
            or len(set((*train_ids, *holdout_ids))) != len(train_ids) + len(holdout_ids)
            or not set((*train_ids, *holdout_ids)) <= set(self.trial_states)
        ):
            raise ValueError("native train/holdout IDs must be disjoint known trials")
        starts = np.asarray(initial_scales, dtype=np.float64)
        if (
            starts.shape != (2,)
            or not np.isfinite(starts).all()
            or np.any((starts < 0) | (starts > 2))
        ):
            raise ValueError("native gain scales must lie in [0, 2]")
        return LoopTuningProblem(
            parameters=tuple(
                LoopParameter(
                    f"{group}_gain_scale", group, "gain", float(start), 0.0, 2.0, 1.0
                )
                for group, start in zip(_GROUPS, starts, strict=True)
            ),
            groups=_GROUPS,
            phases=self.phases,
            trials=tuple(
                [
                    *(TuningTrial(name, "train") for name in train_ids),
                    *(TuningTrial(name, "holdout") for name in holdout_ids),
                ]
            ),
            evaluate=self,
            effort_weight=1e-4,
            robustness_weight=0.0,
            regularization_weight=1e-5,
        )

    def __call__(
        self, values: Array, trial: TuningTrial, active: tuple[str, ...]
    ) -> TuningEvaluation:
        gains = np.asarray(values, dtype=np.float64)
        if (
            gains.shape != (2,)
            or not np.isfinite(gains).all()
            or np.any((gains < 0) | (gains > 2))
            or trial.trial_id not in self.trial_states
            or not active
            or not set(active) <= set(_GROUPS)
        ):
            raise ValueError("native trial, active groups or gain scales invalid")
        row_scale = np.array(
            [gains[i] if group in active else 0.0 for i, group in enumerate(_GROUPS)]
        )
        schedule = replace(
            self.schedule,
            gains=self.schedule.gains * row_scale[None, :, None],
        )
        tracking = run_native_distributed_feedback_tracking(
            self.path,
            self.trial_states[trial.trial_id],
            schedule,
            max_torque_rate_nm_s=self.rate,
            experiment_id=f"f04-{trial.trial_id}",
        )
        executed = tracking.tracking
        if (
            executed.bundle.model != self.reference_model
            or executed.bundle.policy != self.reference_policy
        ):
            raise ValueError("native tuning trial changed model or executed policy")
        if any(
            step.information_pattern != "exact_simulated_state"
            for step in tracking.control_steps
        ):
            raise ValueError("native tuning observed a changed information pattern")
        self._max_replay_error = max(
            self._max_replay_error,
            float(
                np.max(
                    np.abs(
                        executed.native_integration_states
                        - executed.replay.integration_states
                    )
                )
            ),
        )
        self._applied_hashes.append(executed.bundle.applied_input_sha256)
        phase_losses = np.zeros((2, 2), dtype=np.float64)
        counts = np.zeros(2, dtype=np.int64)
        for step, phase in enumerate(schedule.phase_names):
            phase_id = self.phases.index(phase)
            difference = self.tangent.difference(
                executed.native_qpos[step + 1], self.reference.qpos[step + 1]
            )
            phase_losses[phase_id] += difference[6:8] ** 2
            counts[phase_id] += 1
        if np.any(counts == 0):
            raise ValueError("native phase has no executed observations")
        phase_losses /= counts[:, None]
        applied = np.asarray(
            [step.applied for step in tracking.control_steps], dtype=np.float64
        )
        requested = np.asarray(
            [step.total_requested for step in tracking.control_steps],
            dtype=np.float64,
        )
        self._trace.append((trial.split, (float(gains[0]), float(gains[1]))))
        return TuningEvaluation(
            phase_losses,
            effort=float(np.mean(np.sum(applied**2, axis=1))),
            robustness=0.0,
            constraint_violation=0.0,
            saturation_fraction=float(
                np.mean(np.any(np.abs(requested - applied) > 1e-12, axis=1))
            ),
        )


def paired_native_motor_interventions(
    model_path: str | Path,
    reference_bundle: ExperimentReplayBundle,
    *,
    step: int,
    delta_nm: float,
) -> NativeMotorInterventionEvidence:
    """Perturb only one saved motor input at a time; replay both signs afresh."""
    import mujoco as mj

    path = Path(model_path)
    model, _ = _load_native(path)
    reference = replay_native_torque_bundle(reference_bundle, path)
    tangent = NativeMuJoCoTangentModel(model, reference.integration_states[0])
    if (
        (model.nq, model.nv, model.nu) != (9, 8, 2)
        or tuple(reference_bundle.model.ordered_input_channel_ids) != _CHANNELS
        or step < 0
        or step >= len(reference.time_seconds) - 1
        or not np.isfinite(delta_nm)
        or delta_nm <= 0
    ):
        raise ValueError("native intervention requires named motors, step and delta")
    inputs = np.asarray(reference_bundle.input_history.values, dtype=np.float64)
    initial = np.asarray(
        next(
            item.values
            for item in reference_bundle.initial_state
            if item.component_id == "integration"
        ),
        dtype=np.float64,
    )
    response = np.empty((2, 2), dtype=np.float64)
    hashes: list[str] = []
    joint_ids = tuple(
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in _GROUPS
    )
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("native intervention joints missing")
    dof_indices = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    for channel in range(2):
        outputs: list[Array] = []
        for sign in (1.0, -1.0):
            shifted = inputs.copy()
            shifted[step, channel] += sign * delta_nm
            if (
                shifted[step, channel] < model.actuator_ctrlrange[channel, 0]
                or shifted[step, channel] > model.actuator_ctrlrange[channel, 1]
            ):
                raise ValueError("native intervention exceeds motor bound")
            bundle = build_native_torque_bundle(
                path,
                initial,
                np.asarray(reference.time_seconds),
                shifted,
                experiment_id=f"f04-intervention-{channel}-{int(sign)}",
            )
            if (
                bundle.model != reference_bundle.model
                or bundle.policy != reference_bundle.policy
                or bundle.integrity.initial_state_sha256
                != reference_bundle.integrity.initial_state_sha256
            ):
                raise ValueError(
                    "native intervention changed model/policy/initial state"
                )
            fresh = replay_native_torque_bundle(bundle, path)
            outputs.append(fresh.qpos[-1])
            hashes.append(bundle.applied_input_sha256)
        response[:, channel] = tangent.difference(outputs[0], outputs[1])[
            dof_indices
        ] / (2 * delta_nm)
    return NativeMotorInterventionEvidence(
        response,
        _CHANNELS,
        _GROUPS,
        (hashes[0], hashes[1], hashes[2], hashes[3]),
        reference_bundle.integrity.initial_state_sha256,
        reference_bundle.policy_sha256,
        reference_bundle.time_grid_sha256,
    )
