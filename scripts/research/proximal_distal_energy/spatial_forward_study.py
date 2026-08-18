"""Common forward runner and falsification metrics for two native engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.spatial_forward_contract import (
    SpatialContactParameters,
    contact_pair,
    driver_target_velocities,
    driver_targets,
    rotation_matrix_from_quaternion,
    transport_wrench,
)
from scripts.research.proximal_distal_energy.spatial_forward_engines import (
    AppliedSpatialForces,
    EngineIdentity,
    make_spatial_forward_adapter,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialForwardTrace:
    """Canonical output arrays for one engine and intervention."""

    time: FloatArray
    hand_positions: FloatArray
    club_position: FloatArray
    club_quaternion_wxyz: FloatArray
    club_axis: FloatArray
    club_angular_velocity: FloatArray
    contact_forces: FloatArray
    contact_points: FloatArray
    contact_wrench: FloatArray
    swing_normal_couple: FloatArray
    long_axis_couple: FloatArray
    swing_plane_tilt: FloatArray
    driver_forces: FloatArray
    ground_pathway_wrench: FloatArray
    driver_power: FloatArray
    contact_dissipation_power: FloatArray
    interface_storage_energy: FloatArray
    native_mechanical_energy: FloatArray
    total_energy: FloatArray
    action_reaction_force_residual: FloatArray
    interface_power_residual: FloatArray
    wrench_power_residual: FloatArray
    coincident_couple: FloatArray
    reversed_couple: FloatArray
    energy_balance_residual: FloatArray
    engine_identity: EngineIdentity
    model_digest: str
    driver_disabled_after_killswitch: bool


def _club_geometry(
    state: Any, params: SpatialContactParameters
) -> tuple[FloatArray, FloatArray, FloatArray]:
    rotation = rotation_matrix_from_quaternion(state.club_quaternion_wxyz)
    offsets = np.asarray([params.lead_grip_offset, params.trail_grip_offset])
    rotated_offsets = (rotation @ offsets.T).T
    points = state.club_position + rotated_offsets
    velocities = state.club_linear_velocity + np.cross(
        np.broadcast_to(state.club_angular_velocity, (2, 3)), rotated_offsets
    )
    return rotation, points, velocities


def _swing_normal(club_axis: FloatArray, club_velocity: FloatArray) -> FloatArray:
    normal = np.cross(club_axis, club_velocity)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    normal /= norm
    if normal[2] < 0.0:
        normal *= -1.0
    return normal


def run_engine_trace(
    engine: str,
    params: SpatialContactParameters,
    *,
    disable_driver_after_killswitch: bool,
) -> SpatialForwardTrace:
    """Run one actual native forward engine under the common contact law."""

    adapter = make_spatial_forward_adapter(engine, params)
    steps = int(round(params.duration / params.time_step))
    time = np.linspace(0.0, steps * params.time_step, steps + 1)
    arrays: dict[str, FloatArray] = {
        "hand_positions": np.zeros((steps + 1, 2, 3)),
        "club_position": np.zeros((steps + 1, 3)),
        "club_quaternion_wxyz": np.zeros((steps + 1, 4)),
        "club_axis": np.zeros((steps + 1, 3)),
        "club_angular_velocity": np.zeros((steps + 1, 3)),
        "contact_forces": np.zeros((steps + 1, 2, 3)),
        "contact_points": np.zeros((steps + 1, 2, 3)),
        "contact_wrench": np.zeros((steps + 1, 6)),
        "swing_normal_couple": np.zeros(steps + 1),
        "long_axis_couple": np.zeros(steps + 1),
        "swing_plane_tilt": np.zeros(steps + 1),
        "driver_forces": np.zeros((steps + 1, 2, 3)),
        "ground_pathway_wrench": np.zeros((steps + 1, 6)),
        "driver_power": np.zeros(steps + 1),
        "contact_dissipation_power": np.zeros(steps + 1),
        "interface_storage_energy": np.zeros(steps + 1),
        "native_mechanical_energy": np.zeros(steps + 1),
        "total_energy": np.zeros(steps + 1),
        "action_reaction_force_residual": np.zeros(steps + 1),
        "interface_power_residual": np.zeros(steps + 1),
        "wrench_power_residual": np.zeros(steps + 1),
        "coincident_couple": np.zeros(steps + 1),
        "reversed_couple": np.zeros(steps + 1),
    }

    for index, sample_time in enumerate(time):
        state = adapter.canonical_state()
        rotation, club_points, club_point_velocities = _club_geometry(state, params)
        contact_forces = np.zeros((2, 3))
        contact_dissipation = 0.0
        storage_energy = 0.0
        for hand_index in range(2):
            force_on_club, _, _, dissipated_power = contact_pair(
                hand_position=state.hand_positions[hand_index],
                hand_velocity=state.hand_velocities[hand_index],
                club_point_position=club_points[hand_index],
                club_point_velocity=club_point_velocities[hand_index],
                stiffness=params.contact_stiffness,
                damping=params.contact_damping,
            )
            contact_forces[hand_index] = force_on_club
            contact_dissipation += dissipated_power
            displacement = state.hand_positions[hand_index] - club_points[hand_index]
            storage_energy += (
                0.5 * params.contact_stiffness * float(displacement @ displacement)
            )
        target_positions = driver_targets(float(sample_time), params)
        target_velocities = driver_target_velocities(float(sample_time), params)
        driver_enabled = not (
            disable_driver_after_killswitch and sample_time >= params.killswitch_time
        )
        if driver_enabled:
            driver_forces = params.driver_stiffness * (
                target_positions - state.hand_positions
            ) + params.driver_damping * (target_velocities - state.hand_velocities)
        else:
            driver_forces = np.zeros((2, 3))
        hand_forces = driver_forces - contact_forces
        wrench = transport_wrench(
            reference=state.club_position,
            points=club_points,
            forces=contact_forces,
        )
        swing_normal = _swing_normal(rotation[:, 0], state.club_linear_velocity)
        coincident_wrench = transport_wrench(
            reference=state.club_position,
            points=np.broadcast_to(state.club_position, (2, 3)),
            forces=contact_forces,
        )
        reversed_points = state.club_position - (club_points - state.club_position)
        reversed_wrench = transport_wrench(
            reference=state.club_position,
            points=reversed_points,
            forces=contact_forces,
        )
        ground_wrench = transport_wrench(
            reference=np.zeros(3),
            points=state.hand_positions,
            forces=-driver_forces,
        )
        native_energy = adapter.native_mechanical_energy()

        arrays["hand_positions"][index] = state.hand_positions
        arrays["club_position"][index] = state.club_position
        arrays["club_quaternion_wxyz"][index] = state.club_quaternion_wxyz
        arrays["club_axis"][index] = rotation[:, 0]
        arrays["club_angular_velocity"][index] = state.club_angular_velocity
        arrays["contact_forces"][index] = contact_forces
        arrays["contact_points"][index] = club_points
        arrays["contact_wrench"][index] = wrench
        arrays["swing_normal_couple"][index] = float(swing_normal @ wrench[3:])
        arrays["long_axis_couple"][index] = float(rotation[:, 0] @ wrench[3:])
        arrays["swing_plane_tilt"][index] = float(
            np.arctan2(rotation[2, 0], np.linalg.norm(rotation[:2, 0]))
        )
        arrays["driver_forces"][index] = driver_forces
        arrays["ground_pathway_wrench"][index] = ground_wrench
        arrays["driver_power"][index] = float(
            np.sum(driver_forces * state.hand_velocities)
        )
        arrays["contact_dissipation_power"][index] = contact_dissipation
        arrays["interface_storage_energy"][index] = storage_energy
        arrays["native_mechanical_energy"][index] = native_energy
        arrays["total_energy"][index] = native_energy + storage_energy
        arrays["action_reaction_force_residual"][index] = float(
            np.linalg.norm(np.sum(contact_forces + (-contact_forces), axis=0))
        )
        total_contact_body_power = float(
            np.sum(contact_forces * club_point_velocities)
            - np.sum(contact_forces * state.hand_velocities)
        )
        storage_rate = 0.0
        for hand_index in range(2):
            displacement = state.hand_positions[hand_index] - club_points[hand_index]
            relative_velocity = (
                state.hand_velocities[hand_index] - club_point_velocities[hand_index]
            )
            storage_rate += params.contact_stiffness * float(
                displacement @ relative_velocity
            )
        arrays["interface_power_residual"][index] = (
            total_contact_body_power + storage_rate - contact_dissipation
        )
        wrench_power = float(
            wrench[:3] @ state.club_linear_velocity
            + wrench[3:] @ state.club_angular_velocity
        )
        point_power = float(np.sum(contact_forces * club_point_velocities))
        arrays["wrench_power_residual"][index] = wrench_power - point_power
        arrays["coincident_couple"][index] = float(swing_normal @ coincident_wrench[3:])
        arrays["reversed_couple"][index] = float(swing_normal @ reversed_wrench[3:])

        if index < steps:
            adapter.step(
                AppliedSpatialForces(
                    hand_forces=hand_forces,
                    club_points=club_points,
                    club_forces=contact_forces,
                ),
                params.time_step,
            )

    cumulative_input = _cumulative_trapezoid(
        arrays["driver_power"] + arrays["contact_dissipation_power"],
        params.time_step,
    )
    arrays["energy_balance_residual"] = (
        arrays["total_energy"] - arrays["total_energy"][0] - cumulative_input
    )
    return SpatialForwardTrace(
        time=time,
        engine_identity=adapter.engine_identity,
        model_digest=adapter.model_digest,
        driver_disabled_after_killswitch=disable_driver_after_killswitch,
        **arrays,
    )


def _cumulative_trapezoid(values: FloatArray, time_step: float) -> FloatArray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(0.5 * time_step * (values[1:] + values[:-1]))
    return result


def _quaternion_angle_error(left: FloatArray, right: FloatArray) -> FloatArray:
    dots = np.abs(np.sum(left * right, axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _negative_duration(trace: SpatialForwardTrace, start_time: float) -> float:
    """Return the longest contiguous negative interval after ``start_time``."""

    mask = (trace.time >= start_time) & (trace.swing_normal_couple < 0.0)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    if starts.size == 0:
        return 0.0
    longest_samples = int(np.max(stops - starts))
    return float(longest_samples * (trace.time[1] - trace.time[0]))


def compare_engine_traces(
    mujoco_trace: SpatialForwardTrace,
    pinocchio_trace: SpatialForwardTrace,
    params: SpatialContactParameters,
) -> dict[str, Any]:
    """Evaluate preregistered trajectory, wrench, event, and energy gates."""

    if not np.array_equal(mujoco_trace.time, pinocchio_trace.time):
        raise ValueError("engine traces must use the same time grid")
    position_difference = mujoco_trace.club_position - pinocchio_trace.club_position
    wrench_difference = mujoco_trace.contact_wrench - pinocchio_trace.contact_wrench
    position_rms = float(np.sqrt(np.mean(position_difference**2)))
    position_max = float(np.max(np.linalg.norm(position_difference, axis=1)))
    orientation_max = float(
        np.max(
            _quaternion_angle_error(
                mujoco_trace.club_quaternion_wxyz,
                pinocchio_trace.club_quaternion_wxyz,
            )
        )
    )
    wrench_rms = float(np.sqrt(np.mean(wrench_difference**2)))
    wrench_scale = max(
        1.0,
        float(np.sqrt(np.mean(mujoco_trace.contact_wrench**2))),
        float(np.sqrt(np.mean(pinocchio_trace.contact_wrench**2))),
    )
    relative_wrench_rms = wrench_rms / wrench_scale
    energy_scale = max(
        1.0,
        float(np.ptp(mujoco_trace.total_energy)),
        float(np.ptp(pinocchio_trace.total_energy)),
    )
    energy_discrepancy = float(
        np.max(np.abs(mujoco_trace.total_energy - pinocchio_trace.total_energy))
        / energy_scale
    )
    gates = {
        "club_position_rms_limit_m": 0.003,
        "club_position_max_limit_m": 0.009,
        "club_orientation_max_limit_rad": 0.035,
        "contact_wrench_relative_rms_limit": 0.10,
        "normalized_energy_discrepancy_limit": 0.08,
    }
    metrics = {
        "club_position_rms_m": position_rms,
        "club_position_max_m": position_max,
        "club_orientation_max_rad": orientation_max,
        "contact_wrench_rms": wrench_rms,
        "contact_wrench_relative_rms": relative_wrench_rms,
        "normalized_energy_discrepancy": energy_discrepancy,
    }
    trajectory_gate = (
        position_rms <= gates["club_position_rms_limit_m"]
        and position_max <= gates["club_position_max_limit_m"]
        and orientation_max <= gates["club_orientation_max_limit_rad"]
    )
    wrench_gate = relative_wrench_rms <= gates["contact_wrench_relative_rms_limit"]
    energy_gate = energy_discrepancy <= gates["normalized_energy_discrepancy_limit"]
    return {
        "declared_tolerances": gates,
        "tolerance_calibration": (
            "Engineering acceptance regions for a shared semi-implicit "
            "integrator. Exact common initial state, analytic action-reaction "
            "controls, timestep refinement, and two native rollouts audit the "
            "regions; they are not empirical confidence intervals."
        ),
        "observed_metrics": metrics,
        "trajectory_gate_passed": bool(trajectory_gate),
        "wrench_gate_passed": bool(wrench_gate),
        "energy_gate_passed": bool(energy_gate),
    }


def summarize_trace(
    trace: SpatialForwardTrace, params: SpatialContactParameters
) -> dict[str, Any]:
    """Return bounded mechanism, conservation, and pathway observables."""

    after_kill = trace.time >= params.killswitch_time
    swing_couple = trace.swing_normal_couple[after_kill]
    return {
        "minimum_swing_normal_couple_nm": float(np.min(trace.swing_normal_couple)),
        "minimum_post_killswitch_couple_nm": float(np.min(swing_couple)),
        "post_killswitch_negative_duration_s": _negative_duration(
            trace, params.killswitch_time
        ),
        "peak_long_axis_couple_abs_nm": float(np.max(np.abs(trace.long_axis_couple))),
        "club_axis_out_of_plane_range_deg": float(
            np.rad2deg(np.ptp(trace.swing_plane_tilt))
        ),
        "club_long_axis_rotation_rate_peak_rad_s": float(
            np.max(
                np.abs(np.sum(trace.club_angular_velocity * trace.club_axis, axis=1))
            )
        ),
        "ground_pathway_force_peak_n": float(
            np.max(np.linalg.norm(trace.ground_pathway_wrench[:, :3], axis=1))
        ),
        "ground_pathway_moment_peak_nm": float(
            np.max(np.linalg.norm(trace.ground_pathway_wrench[:, 3:], axis=1))
        ),
        "coincident_grip_couple_max_nm": float(np.max(np.abs(trace.coincident_couple))),
        "reversed_geometry_sign_residual_nm": float(
            np.max(np.abs(trace.swing_normal_couple + trace.reversed_couple))
        ),
        "action_reaction_force_residual_max_n": float(
            np.max(trace.action_reaction_force_residual)
        ),
        "interface_power_residual_max_w": float(
            np.max(np.abs(trace.interface_power_residual))
        ),
        "wrench_power_residual_max_w": float(
            np.max(np.abs(trace.wrench_power_residual))
        ),
        "energy_balance_residual_max_j": float(
            np.max(np.abs(trace.energy_balance_residual))
        ),
    }


def engine_identity_record(identity: EngineIdentity) -> dict[str, Any]:
    """Return the JSON-ready native engine identity."""

    return asdict(identity)


__all__ = [
    "SpatialForwardTrace",
    "compare_engine_traces",
    "engine_identity_record",
    "run_engine_trace",
    "summarize_trace",
]
