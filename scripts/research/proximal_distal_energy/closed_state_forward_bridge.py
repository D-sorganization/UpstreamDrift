"""Map closed subject-scaled states into the canonical forward-contact tier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.spatial_forward_contract import (
    CanonicalSpatialState,
    SpatialContactParameters,
    canonical_spatial_state_digest,
    contact_pair,
    driver_target_velocities,
    driver_targets,
    rotation_matrix_from_quaternion,
    transport_wrench,
)
from scripts.research.proximal_distal_energy.spatial_forward_engines import (
    AppliedSpatialForces,
    make_spatial_forward_adapter,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
    point_contact_jacobians,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"


@dataclass(frozen=True, slots=True)
class ClosedStateBridgeConfig:
    """Predeclared mapping and constitutive-law gates."""

    canonical_club_center_m: tuple[float, float, float] = (0.58, 0.0, 1.18)
    position_closure_tolerance_m: float = 5.0e-4
    velocity_closure_tolerance_m_s: float = 5.0e-3
    zero_preload_force_tolerance_n: float = 1.0e-8
    work_power_tolerance_w: float = 1.0e-10
    perturbation_m: float = 1.0e-3

    def __post_init__(self) -> None:
        center = np.asarray(self.canonical_club_center_m, dtype=float)
        positive = (
            self.position_closure_tolerance_m,
            self.velocity_closure_tolerance_m_s,
            self.zero_preload_force_tolerance_n,
            self.work_power_tolerance_w,
            self.perturbation_m,
        )
        if center.shape != (3,) or np.any(~np.isfinite(center)):
            raise ValueError("canonical_club_center_m must be one finite 3-vector")
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("bridge tolerances and perturbation must be positive")


def _point_state(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    joint: int,
    local: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    kin = forward_kinematics(model, q)
    point, linear, angular = point_contact_jacobians(model, kin, joint, local)
    return point, linear @ qd, angular @ qd


def _map_one_state(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    span_m: float,
    hand_contact_x_m: float,
    config: ClosedStateBridgeConfig,
) -> tuple[CanonicalSpatialState, FloatArray, FloatArray]:
    hand_local = np.array([hand_contact_x_m, 0.0, 0.0])
    grip_locals = np.array([[0.0, span_m / 2.0, -0.03], [0.0, -span_m / 2.0, -0.03]])
    hand_points, hand_velocities = [], []
    grip_points, grip_velocities = [], []
    for hand_joint, grip_local in zip(
        (model.lead_hand_joint, model.trail_hand_joint), grip_locals, strict=True
    ):
        hand_point, hand_velocity, _ = _point_state(
            model, q, qd, hand_joint, hand_local
        )
        grip_point, grip_velocity, _ = _point_state(
            model, q, qd, model.club_frame_joint, grip_local
        )
        hand_points.append(hand_point)
        hand_velocities.append(hand_velocity)
        grip_points.append(grip_point)
        grip_velocities.append(grip_velocity)
    club_point, club_velocity, club_angular = _point_state(
        model, q, qd, model.club_frame_joint, np.zeros(3)
    )
    rotation = forward_kinematics(model, q).joint_rotation[model.club_frame_joint]
    center = np.asarray(config.canonical_club_center_m)

    def transform(value: FloatArray) -> FloatArray:
        return center + rotation.T @ (np.asarray(value) - club_point)

    state = CanonicalSpatialState(
        hand_positions=np.asarray([transform(value) for value in hand_points]),
        hand_velocities=(rotation.T @ np.asarray(hand_velocities).T).T,
        club_position=center,
        club_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        club_linear_velocity=rotation.T @ club_velocity,
        club_angular_velocity=rotation.T @ club_angular,
    )
    position_error = np.linalg.norm(np.asarray(hand_points) - grip_points, axis=1)
    velocity_error = np.linalg.norm(
        np.asarray(hand_velocities) - grip_velocities, axis=1
    )
    return state, position_error, velocity_error


def _constitutive_controls(
    state: CanonicalSpatialState,
    params: SpatialContactParameters,
    config: ClosedStateBridgeConfig,
) -> dict[str, float | bool]:
    forces = []
    for index, offset in enumerate((params.lead_grip_offset, params.trail_grip_offset)):
        force, opposite, _, power = contact_pair(
            hand_position=state.club_position + np.asarray(offset),
            hand_velocity=np.zeros(3),
            club_point_position=state.club_position + np.asarray(offset),
            club_point_velocity=np.zeros(3),
            stiffness=params.contact_stiffness,
            damping=params.contact_damping,
        )
        forces.append(np.linalg.norm(force))
        if not np.allclose(
            force + opposite, 0.0, atol=config.zero_preload_force_tolerance_n
        ):
            raise RuntimeError(f"action-reaction failed at contact {index}")
        if abs(power) > config.work_power_tolerance_w:
            raise RuntimeError(f"zero-state power failed at contact {index}")
    displacement = np.array([config.perturbation_m, 0.0, 0.0])
    velocity = np.array([-0.2, 0.0, 0.0])
    force, opposite, stored_power, dissipated_power = contact_pair(
        hand_position=displacement,
        hand_velocity=velocity,
        club_point_position=np.zeros(3),
        club_point_velocity=np.zeros(3),
        stiffness=params.contact_stiffness,
        damping=params.contact_damping,
    )
    return {
        "maximum_zero_preload_force_n": float(max(forces)),
        "action_reaction_residual_n": float(np.linalg.norm(force + opposite)),
        "stored_power_w": float(stored_power),
        "dissipated_power_w": float(dissipated_power),
        "damping_passive": bool(dissipated_power <= 0.0),
    }


def canonical_state_from_vector(values: FloatArray) -> CanonicalSpatialState:
    """Reconstruct one canonical state from the committed packed convention."""

    values = np.asarray(values, dtype=float)
    if values.shape != (25,) or np.any(~np.isfinite(values)):
        raise ValueError("values must be one finite 25-element state")
    return CanonicalSpatialState(
        hand_positions=values[:6].reshape(2, 3),
        hand_velocities=values[6:12].reshape(2, 3),
        club_position=values[12:15],
        club_quaternion_wxyz=values[15:19],
        club_linear_velocity=values[19:22],
        club_angular_velocity=values[22:25],
    )


def _bridge_forces(
    state: CanonicalSpatialState,
    params: SpatialContactParameters,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    rotation = rotation_matrix_from_quaternion(state.club_quaternion_wxyz)
    offsets = np.asarray([params.lead_grip_offset, params.trail_grip_offset])
    rotated = (rotation @ offsets.T).T
    club_points = state.club_position + rotated
    club_velocities = state.club_linear_velocity + np.cross(
        np.broadcast_to(state.club_angular_velocity, (2, 3)), rotated
    )
    forces = np.zeros((2, 3))
    storage = 0.0
    dissipation = 0.0
    for hand in range(2):
        force, _, _, dissipated = contact_pair(
            hand_position=state.hand_positions[hand],
            hand_velocity=state.hand_velocities[hand],
            club_point_position=club_points[hand],
            club_point_velocity=club_velocities[hand],
            stiffness=params.contact_stiffness,
            damping=params.contact_damping,
        )
        displacement = state.hand_positions[hand] - club_points[hand]
        forces[hand] = force
        storage += 0.5 * params.contact_stiffness * float(displacement @ displacement)
        dissipation += dissipated
    return club_points, club_velocities, forces, storage, dissipation


def run_bridge_trace(
    engine: str,
    params: SpatialContactParameters,
    initial_state: CanonicalSpatialState,
    *,
    duration_s: float = 0.004,
    driver_enabled: bool = True,
) -> dict[str, Any]:
    """Advance one closed state with common forces and named energy ledgers."""

    if not np.isfinite(duration_s) or not 0.0 < duration_s <= params.duration:
        raise ValueError("duration_s must be finite and in (0, params.duration]")
    adapter = make_spatial_forward_adapter(engine, params, initial_state)
    steps = int(round(duration_s / params.time_step))
    positions, quaternions, wrenches = [], [], []
    energies, driver_powers, dissipated_powers = [], [], []
    for index in range(steps + 1):
        state = adapter.canonical_state()
        club_points, _, contact_forces, storage, dissipation = _bridge_forces(
            state, params
        )
        sample_time = index * params.time_step
        driver_forces = np.zeros((2, 3))
        if driver_enabled:
            driver_forces = params.driver_stiffness * (
                driver_targets(sample_time, params) - state.hand_positions
            ) + params.driver_damping * (
                driver_target_velocities(sample_time, params) - state.hand_velocities
            )
        positions.append(state.club_position)
        quaternions.append(state.club_quaternion_wxyz)
        wrenches.append(
            transport_wrench(
                reference=state.club_position,
                points=club_points,
                forces=contact_forces,
            )
        )
        energies.append(adapter.native_mechanical_energy() + storage)
        driver_powers.append(float(np.sum(driver_forces * state.hand_velocities)))
        dissipated_powers.append(dissipation)
        if index < steps:
            adapter.step(
                AppliedSpatialForces(
                    hand_forces=driver_forces - contact_forces,
                    club_points=club_points,
                    club_forces=contact_forces,
                ),
                params.time_step,
            )
    total_energy = np.asarray(energies)
    input_power = np.asarray(driver_powers) + np.asarray(dissipated_powers)
    cumulative_input = np.zeros_like(input_power)
    cumulative_input[1:] = np.cumsum(
        0.5 * (input_power[1:] + input_power[:-1]) * params.time_step
    )
    return {
        "initial_state_digest": adapter.initial_state_digest,
        "club_position": np.asarray(positions),
        "club_quaternion_wxyz": np.asarray(quaternions),
        "contact_wrench": np.asarray(wrenches),
        "total_energy": total_energy,
        "driver_power": np.asarray(driver_powers),
        "contact_dissipation_power": np.asarray(dissipated_powers),
        "energy_balance_residual": total_energy - total_energy[0] - cumulative_input,
    }


def compare_bridge_traces(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    position_delta = left["club_position"] - right["club_position"]
    quaternion_dot = np.abs(
        np.sum(left["club_quaternion_wxyz"] * right["club_quaternion_wxyz"], axis=1)
    )
    wrench_delta = left["contact_wrench"] - right["contact_wrench"]
    wrench_scale = max(
        1.0,
        float(np.sqrt(np.mean(left["contact_wrench"] ** 2))),
        float(np.sqrt(np.mean(right["contact_wrench"] ** 2))),
    )
    energy_scale = max(
        1.0,
        float(np.ptp(left["total_energy"])),
        float(np.ptp(right["total_energy"])),
    )
    metrics = {
        "club_position_rms_m": float(np.sqrt(np.mean(position_delta**2))),
        "club_position_max_m": float(np.max(np.linalg.norm(position_delta, axis=1))),
        "club_orientation_max_rad": float(
            np.max(2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0)))
        ),
        "contact_wrench_relative_rms": float(
            np.sqrt(np.mean(wrench_delta**2)) / wrench_scale
        ),
        "normalized_energy_discrepancy": float(
            np.max(np.abs(left["total_energy"] - right["total_energy"])) / energy_scale
        ),
    }
    return {
        "observed_metrics": metrics,
        "trajectory_gate_passed": metrics["club_position_rms_m"] <= 0.003
        and metrics["club_position_max_m"] <= 0.009
        and metrics["club_orientation_max_rad"] <= 0.035,
        "wrench_gate_passed": metrics["contact_wrench_relative_rms"] <= 0.10,
        "energy_gate_passed": metrics["normalized_energy_discrepancy"] <= 0.08,
    }


def evaluate_spanning_forward_subset(
    arrays: dict[str, NDArray[Any]],
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run both native engines for all profiles/spans at early/mid/late phases."""

    profiles = default_synthetic_profiles()
    phase_indices = (0, arrays["time_s"].size // 2, arrays["time_s"].size - 1)
    rows, positions, wrenches = [], [], []
    for case in range(arrays["case_profile_index"].size):
        profile_index = int(arrays["case_profile_index"][case])
        _, metadata = build_subject_scaled_model(profiles[profile_index])
        span = float(arrays["case_grip_span_m"][case])
        offsets = ((0.0, span / 2.0, -0.03), (0.0, -span / 2.0, -0.03))
        for phase_index in phase_indices:
            state = canonical_state_from_vector(
                arrays["canonical_state"][case, phase_index]
            )
            params = SpatialContactParameters(
                hand_mass=float(metadata["represented_body_masses_kg"]["hand"]),
                lead_grip_offset=offsets[0],
                trail_grip_offset=offsets[1],
                club_initial_position=tuple(state.club_position),
            )
            traces = {
                engine: run_bridge_trace(engine, params, state)
                for engine in ("mujoco", "pinocchio")
            }
            comparison = compare_bridge_traces(traces["mujoco"], traces["pinocchio"])
            rows.append(
                {
                    "case_index": case,
                    "phase_index": phase_index,
                    "profile_id": profiles[profile_index].profile_id,
                    "grip_span_m": span,
                    "initial_state_digest_match": (
                        traces["mujoco"]["initial_state_digest"]
                        == traces["pinocchio"]["initial_state_digest"]
                    ),
                    **comparison,
                }
            )
            positions.append(
                np.stack(
                    [
                        traces[engine]["club_position"]
                        for engine in ("mujoco", "pinocchio")
                    ]
                )
            )
            wrenches.append(
                np.stack(
                    [
                        traces[engine]["contact_wrench"]
                        for engine in ("mujoco", "pinocchio")
                    ]
                )
            )
    gate_names = ("trajectory_gate_passed", "wrench_gate_passed", "energy_gate_passed")
    summary = {
        "subset_case_count": len(rows),
        "phase_indices": list(phase_indices),
        "declared_tolerances": {
            "club_position_rms_limit_m": 0.003,
            "club_position_max_limit_m": 0.009,
            "club_orientation_max_limit_rad": 0.035,
            "contact_wrench_relative_rms_limit": 0.10,
            "normalized_energy_discrepancy_limit": 0.08,
        },
        "tolerance_calibration": (
            "Existing reduced-forward engineering acceptance regions, applied "
            "unchanged to the 4 ms closed-state initialization audit; they are "
            "not empirical confidence intervals."
        ),
        "all_initial_state_digests_match": all(
            row["initial_state_digest_match"] for row in rows
        ),
        **{name: all(row[name] for row in rows) for name in gate_names},
        "cases": rows,
    }
    return summary, {
        "subset_club_position_m": np.asarray(positions),
        "subset_contact_wrench": np.asarray(wrenches),
    }


def map_closed_contact_atlas(
    config: ClosedStateBridgeConfig = ClosedStateBridgeConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Map all 234 atlas samples and report every predeclared gate."""

    source_record = json.loads(
        (DATA_DIR / "subject_scaled_closed_contact.json").read_text(encoding="utf-8")
    )
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as archive:
        source = {name: archive[name].copy() for name in archive.files}
    q = source["solution_q"]
    time_s = source["time_s"]
    qd = np.gradient(q, time_s, axis=1, edge_order=2)
    states = np.empty((*q.shape[:2], 25))
    position_error = np.empty((*q.shape[:2], 2))
    velocity_error = np.empty((*q.shape[:2], 2))
    digests: list[str] = []
    profiles = default_synthetic_profiles()
    for case in range(q.shape[0]):
        profile_index = int(source["case_profile_index"][case])
        model, metadata = build_subject_scaled_model(profiles[profile_index])
        for sample in range(q.shape[1]):
            state, position_error[case, sample], velocity_error[case, sample] = (
                _map_one_state(
                    model,
                    q[case, sample],
                    qd[case, sample],
                    float(source["case_grip_span_m"][case]),
                    float(metadata["hand_contact_local_x_m"]),
                    config,
                )
            )
            values = np.concatenate(
                (
                    state.hand_positions.ravel(),
                    state.hand_velocities.ravel(),
                    state.club_position,
                    state.club_quaternion_wxyz,
                    state.club_linear_velocity,
                    state.club_angular_velocity,
                )
            )
            states[case, sample] = values
            digests.append(canonical_spatial_state_digest(state))
    params = SpatialContactParameters()
    controls = _constitutive_controls(state, params, config)
    position_pass = bool(np.max(position_error) <= config.position_closure_tolerance_m)
    velocity_pass = bool(
        np.max(velocity_error) <= config.velocity_closure_tolerance_m_s
    )
    record = {
        "schema_version": "closed-state-forward-bridge/v1",
        "study_id": "subject-scaled-closed-state-forward-contact-bridge",
        "source_schema_version": source_record["schema_version"],
        "design": {
            "configuration": asdict(config),
            "case_count": int(q.shape[0]),
            "time_sample_count": int(q.shape[1]),
            "mapped_state_count": int(q.shape[0] * q.shape[1]),
            "velocity_estimator": "second-order finite differences along each closed IK path",
            "frame_map": "constant initial club-frame rotation and translation; velocities rotate without boost",
        },
        "results": {
            "maximum_position_closure_error_m": float(np.max(position_error)),
            "maximum_velocity_closure_error_m_s": float(np.max(velocity_error)),
            "position_closure_gate_passed": position_pass,
            "velocity_closure_gate_passed": velocity_pass,
            "unique_initial_state_digest_count": len(set(digests)),
            "constitutive_controls": controls,
        },
        "claim_boundary": {
            "reduced_state_initialization": "evaluated",
            "forward_trajectory_parity": "not_yet_evaluated",
            "equipment_calibration": "not_established",
            "anatomical_validity": "not_established",
            "human_or_coaching_strategy": "not_established",
        },
    }
    arrays = {
        "time_s": time_s,
        "case_profile_index": source["case_profile_index"],
        "case_grip_span_m": source["case_grip_span_m"],
        "canonical_state": states,
        "position_closure_error_m": position_error,
        "velocity_closure_error_m_s": velocity_error,
    }
    return record, arrays


__all__ = [
    "ClosedStateBridgeConfig",
    "canonical_state_from_vector",
    "compare_bridge_traces",
    "evaluate_spanning_forward_subset",
    "map_closed_contact_atlas",
    "run_bridge_trace",
]
