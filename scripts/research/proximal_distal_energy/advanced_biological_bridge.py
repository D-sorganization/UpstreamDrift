"""Reference-frame and reduced muscle bridges for the advanced monograph.

The model deliberately stops at a reduced Hill-type actuator tier.  It tests
coordinate invariance, muscle redundancy, activation delay, and series-elastic
transmission without identifying anatomical muscles or a preferred technique.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.mechanism_ladder import (
    InteractionSample,
    rotation_matrix,
)
from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import (
    HillMuscleModel,
    MuscleParameters,
    MuscleState,
)

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RedundancySurface:
    """Matched-torque family with increasing antagonist coactivation."""

    coactivation: Array
    positive_activation: Array
    negative_activation: Array
    net_torque_nm: Array
    activation_sum: Array
    stiffness_proxy_nm_rad: Array
    series_elastic_energy_j: Array


@dataclass(frozen=True)
class BiologicalProgramResult:
    """One continuous preparation and delivery trace."""

    name: str
    time_s: Array
    target_arm_torque_nm: Array
    target_wrist_torque_nm: Array
    transmitted_arm_torque_nm: Array
    transmitted_wrist_torque_nm: Array
    arm_activation: Array
    wrist_activation: Array
    tendon_force_n: Array
    series_elastic_energy_j: Array
    preparation_duration_s: float
    post_target_torque_nm: float
    post_transition_error_impulse_nms: float
    minimum_tendon_force_n: float


@dataclass(frozen=True)
class BiologicalProgramStudy:
    """Matched persistent-direction and complete-role-reversal programs."""

    programs: dict[str, BiologicalProgramResult]
    claim_boundary: str


@dataclass(frozen=True)
class _MuscleChannel:
    name: str
    positive: HillMuscleModel
    negative: HillMuscleModel
    moment_arm_m: float
    tendon_stiffness_n_m: float

    @property
    def positive_capacity_nm(self) -> float:
        return self.positive.params.F_max * self.moment_arm_m

    @property
    def negative_capacity_nm(self) -> float:
        return self.negative.params.F_max * self.moment_arm_m


def _muscle(f_max_n: float, l_opt_m: float, l_slack_m: float) -> HillMuscleModel:
    return HillMuscleModel(
        MuscleParameters(F_max=f_max_n, l_opt=l_opt_m, l_slack=l_slack_m)
    )


def _channel(name: str, capacity_nm: float, moment_arm_m: float) -> _MuscleChannel:
    f_max = capacity_nm / moment_arm_m
    return _MuscleChannel(
        name=name,
        positive=_muscle(f_max, 0.12, 0.20),
        negative=_muscle(f_max, 0.12, 0.20),
        moment_arm_m=moment_arm_m,
        tendon_stiffness_n_m=150_000.0,
    )


def _force(model: HillMuscleModel, activation: float) -> float:
    return model.compute_force(
        MuscleState(
            activation=activation,
            l_CE=model.params.l_opt,
            v_CE=0.0,
            l_MT=model.params.l_opt + model.params.l_slack,
        )
    )


def build_frame_invariance_audit() -> dict[str, object]:
    """Return deterministic rigid-frame, point-transport, and power audits."""
    samples = (
        InteractionSample(
            model_tier="advanced-frame-audit",
            time_s=0.0,
            frame="laboratory",
            reference_point_m=np.array([0.31, -0.18, 1.04]),
            force_n=np.array([84.0, -31.0, 47.0]),
            couple_nm=np.array([2.8, -5.4, 7.1]),
            linear_velocity_m_s=np.array([3.2, -0.7, 1.4]),
            angular_velocity_rad_s=np.array([-1.1, 4.8, 13.2]),
        ),
        InteractionSample(
            model_tier="advanced-frame-audit",
            time_s=0.04,
            frame="laboratory",
            reference_point_m=np.array([-0.12, 0.26, 0.91]),
            force_n=np.array([-62.0, 55.0, 19.0]),
            couple_nm=np.array([4.1, 1.9, -6.3]),
            linear_velocity_m_s=np.array([1.5, 2.2, -0.4]),
            angular_velocity_rad_s=np.array([2.3, -3.7, 9.4]),
        ),
    )
    rotation_residuals: list[float] = []
    transport_residuals: list[float] = []
    virtual_work_residuals: list[float] = []
    for index, sample in enumerate(samples):
        rotation = rotation_matrix(np.array([1.0, -0.3, 0.7]), 0.61 + 0.2 * index)
        rotated = sample.rotate(rotation, frame="club-attached")
        moved = sample.transport(
            sample.reference_point_m + np.array([0.18, -0.09, 0.06])
        )
        rotation_residuals.append(abs(rotated.total_power_w - sample.total_power_w))
        transport_residuals.append(abs(moved.total_power_w - sample.total_power_w))

        jacobian = np.array(
            [
                [0.2, -0.1, 0.4, 0.0],
                [0.7, 0.3, -0.2, 0.1],
                [-0.1, 0.5, 0.2, 0.4],
                [0.0, 0.8, -0.3, 0.2],
                [0.6, -0.2, 0.1, 0.5],
                [-0.4, 0.3, 0.7, -0.1],
            ]
        )
        qdot = np.array([1.2, -0.8, 0.55, 2.1])
        twist = jacobian @ qdot
        wrench = np.concatenate((sample.force_n, sample.couple_nm))
        generalized_force = jacobian.T @ wrench
        virtual_work_residuals.append(
            abs(float(generalized_force @ qdot) - float(wrench @ twist))
        )
    return {
        "wrench_order": ["force_xyz_n", "couple_xyz_nm"],
        "twist_order": ["linear_xyz_m_s", "angular_xyz_rad_s"],
        "rotation_convention": "active proper rotation, right-handed Cartesian axes",
        "point_transport_convention": "same physical wrench and rigid-body twist",
        "maximum_rotation_power_residual_w": max(rotation_residuals),
        "maximum_transport_power_residual_w": max(transport_residuals),
        "maximum_virtual_work_residual_w": max(virtual_work_residuals),
    }


def build_pose_adapter_audit() -> dict[str, dict[str, object]]:
    """Round-trip one canonical pose through all five lightweight adapters.

    This executes coordinate-convention code without importing optional engine
    runtimes.  It tests representation parity only, not dynamics parity.
    """
    from src.shared.python.pose_interchange.adapters.drake import DrakeAdapter
    from src.shared.python.pose_interchange.adapters.mujoco import MujocoAdapter
    from src.shared.python.pose_interchange.adapters.myosuite import MyosuiteAdapter
    from src.shared.python.pose_interchange.adapters.opensim import OpenSimAdapter
    from src.shared.python.pose_interchange.adapters.pinocchio import (
        PinocchioAdapter,
    )
    from src.shared.python.pose_interchange.canonical import CanonicalPose

    source = CanonicalPose(
        pelvis_translation_m=np.array([0.12, -0.08, 0.94]),
        pelvis_rotation_xyz_deg=np.array([7.0, -11.0, 24.0]),
        joint_angles_deg={
            "HipStartPositionX": -6.0,
            "TorsoStartPosition": 31.0,
            "LSStartPositionX": 42.0,
            "REStartPosition": 68.0,
        },
    )
    adapters = {
        "mujoco": MujocoAdapter(),
        "pinocchio": PinocchioAdapter(),
        "drake": DrakeAdapter(),
        "opensim": OpenSimAdapter(),
        "myosuite": MyosuiteAdapter(),
    }
    records: dict[str, dict[str, object]] = {}
    for name, adapter in adapters.items():
        engine_q = adapter.from_canonical(source)
        recovered = adapter.to_canonical(engine_q)
        joint_residuals = [
            abs(recovered.angle_deg(joint) - source.angle_deg(joint))
            for joint in source.joint_angles_deg
        ]
        records[name] = {
            "status": "executed_adapter_round_trip",
            "scope": "coordinate_representation_only_no_engine_runtime",
            "native_q_size": int(engine_q.size),
            "maximum_translation_residual_m": float(
                np.max(
                    np.abs(recovered.pelvis_translation_m - source.pelvis_translation_m)
                )
            ),
            "maximum_rotation_residual_deg": float(
                np.max(
                    np.abs(
                        recovered.pelvis_rotation_xyz_deg
                        - source.pelvis_rotation_xyz_deg
                    )
                )
            ),
            "maximum_joint_residual_deg": max(joint_residuals, default=0.0),
        }
    return records


def build_redundancy_surface(
    *, target_torque_nm: float = 10.0, sample_count: int = 41
) -> RedundancySurface:
    """Build an isometric activation family for one matched net joint torque."""
    if not np.isfinite(target_torque_nm):
        raise ValueError("target_torque_nm must be finite")
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    channel = _channel("representative joint", capacity_nm=40.0, moment_arm_m=0.04)
    required_fraction = target_torque_nm / channel.positive_capacity_nm
    if not 0.0 <= required_fraction < 1.0:
        raise ValueError("target_torque_nm exceeds the declared activation family")
    maximum_coactivation = 1.0 - required_fraction
    coactivation = np.linspace(0.0, maximum_coactivation, sample_count)
    positive_activation = required_fraction + coactivation
    negative_activation = coactivation.copy()
    positive_force = np.array(
        [_force(channel.positive, value) for value in positive_activation]
    )
    negative_force = np.array(
        [_force(channel.negative, value) for value in negative_activation]
    )
    net_torque = channel.moment_arm_m * (positive_force - negative_force)
    stiffness = (
        (positive_force + negative_force)
        * channel.moment_arm_m**2
        / channel.positive.params.l_opt
    )
    elastic_energy = (positive_force**2 + negative_force**2) / (
        2.0 * channel.tendon_stiffness_n_m
    )
    return RedundancySurface(
        coactivation=coactivation,
        positive_activation=positive_activation,
        negative_activation=negative_activation,
        net_torque_nm=net_torque,
        activation_sum=positive_activation + negative_activation,
        stiffness_proxy_nm_rad=stiffness,
        series_elastic_energy_j=elastic_energy,
    )


def _target_activations(
    channel: _MuscleChannel, torque_nm: float, baseline: float
) -> tuple[float, float]:
    if torque_nm >= 0.0:
        positive = baseline + torque_nm / channel.positive_capacity_nm
        negative = baseline
    else:
        positive = baseline
        negative = baseline + abs(torque_nm) / channel.negative_capacity_nm
    if max(positive, negative) > 1.0:
        raise ValueError(f"{channel.name} torque target exceeds activation capacity")
    return positive, negative


def _simulate_program(
    *,
    name: str,
    arm_pre_nm: float,
    wrist_pre_nm: float,
    arm_post_nm: float,
    wrist_post_nm: float,
    step_s: float,
) -> BiologicalProgramResult:
    preparation_duration = 0.18
    post_duration = 0.12
    preparation_steps = int(round(preparation_duration / step_s))
    post_steps = int(round(post_duration / step_s))
    time = np.arange(-preparation_steps, post_steps + 1, dtype=float) * step_s
    arm = _channel("proximal arm", capacity_nm=40.0, moment_arm_m=0.04)
    wrist = _channel("direct wrist", capacity_nm=15.0, moment_arm_m=0.03)
    dynamics = ActivationDynamics(tau_act=0.010, tau_deact=0.040)
    baseline = 0.08
    tendon_time_constant_s = 0.012
    activations = np.full(4, dynamics.min_activation)
    tendon_forces = np.zeros(4)
    transmitted_arm = np.zeros_like(time)
    transmitted_wrist = np.zeros_like(time)
    activation_trace = np.zeros((time.size, 4))
    tendon_force_trace = np.zeros((time.size, 4))
    elastic_energy = np.zeros_like(time)
    target_arm = np.where(time < 0.0, arm_pre_nm, arm_post_nm)
    target_wrist = np.where(time < 0.0, wrist_pre_nm, wrist_post_nm)

    for index, _ in enumerate(time):
        arm_target = _target_activations(arm, target_arm[index], baseline)
        wrist_target = _target_activations(wrist, target_wrist[index], baseline)
        targets = np.array((*arm_target, *wrist_target))
        for muscle_index, target in enumerate(targets):
            activations[muscle_index] = dynamics.update(
                float(target), float(activations[muscle_index]), step_s
            )
        desired_forces = np.array(
            [
                _force(arm.positive, activations[0]),
                _force(arm.negative, activations[1]),
                _force(wrist.positive, activations[2]),
                _force(wrist.negative, activations[3]),
            ]
        )
        tendon_forces += (
            step_s / tendon_time_constant_s * (desired_forces - tendon_forces)
        )
        transmitted_arm[index] = arm.moment_arm_m * (
            tendon_forces[0] - tendon_forces[1]
        )
        transmitted_wrist[index] = wrist.moment_arm_m * (
            tendon_forces[2] - tendon_forces[3]
        )
        activation_trace[index] = activations
        tendon_force_trace[index] = tendon_forces
        elastic_energy[index] = np.sum(
            tendon_forces[:2] ** 2 / (2.0 * arm.tendon_stiffness_n_m)
        ) + np.sum(tendon_forces[2:] ** 2 / (2.0 * wrist.tendon_stiffness_n_m))

    target_net = target_arm + target_wrist
    transmitted_net = transmitted_arm + transmitted_wrist
    post = time >= 0.0
    error_impulse = float(
        np.trapezoid(np.abs(target_net[post] - transmitted_net[post]), time[post])
    )
    total_tendon_force = np.sum(tendon_force_trace, axis=1)
    return BiologicalProgramResult(
        name=name,
        time_s=time,
        target_arm_torque_nm=target_arm,
        target_wrist_torque_nm=target_wrist,
        transmitted_arm_torque_nm=transmitted_arm,
        transmitted_wrist_torque_nm=transmitted_wrist,
        arm_activation=activation_trace[:, :2],
        wrist_activation=activation_trace[:, 2:],
        tendon_force_n=tendon_force_trace,
        series_elastic_energy_j=elastic_energy,
        preparation_duration_s=preparation_duration,
        post_target_torque_nm=float(arm_post_nm + wrist_post_nm),
        post_transition_error_impulse_nms=error_impulse,
        minimum_tendon_force_n=float(np.min(total_tendon_force[post])),
    )


def simulate_biological_programs(*, step_s: float = 0.0002) -> BiologicalProgramStudy:
    """Compare matched net-torque programs with continuous muscle states."""
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("step_s must be positive and finite")
    if step_s > 0.001:
        raise ValueError("step_s must not exceed 1 ms for activation stability")
    persistent = _simulate_program(
        name="persistent_direction",
        arm_pre_nm=10.0,
        wrist_pre_nm=-4.0,
        arm_post_nm=16.0,
        wrist_post_nm=-6.0,
        step_s=step_s,
    )
    reversal = _simulate_program(
        name="complete_role_reversal",
        arm_pre_nm=-4.0,
        wrist_pre_nm=10.0,
        arm_post_nm=16.0,
        wrist_post_nm=-6.0,
        step_s=step_s,
    )
    return BiologicalProgramStudy(
        programs={persistent.name: persistent, reversal.name: reversal},
        claim_boundary=(
            "Reduced Hill-type channels demonstrate activation, redundancy, and "
            "series-elastic consequences; they do not identify scapular muscles, "
            "neural strategy, tissue parameters, or a preferred human technique."
        ),
    )
