"""Per-case input contract, signals and metrics record for the WSCG audit."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.run_two_hand_wscg_analysis import (
    CASE_NAMES,
    OUT_OF_PLANE_AXIS,
    _crossing_record,
    _interval_signal_metrics,
    _resampling_sensitivity,
    _vector,
)
from src.shared.python.contracts import ensure, require


def _require_case_analysis_inputs(name: str, data: np.ndarray) -> None:
    if name not in CASE_NAMES:
        raise ValueError(f"unknown case {name!r}")
    time = np.asarray(data["time_s"], dtype=float)
    require(
        time.ndim == 1 and time.size >= 2 and bool(np.all(np.isfinite(time))),
        "time_s must be a finite one-dimensional series with at least two samples",
    )
    sample_count = int(time.size)
    for column in ("clubhead_speed_mph", "lead_linear_power_w", "trail_linear_power_w"):
        values = np.asarray(data[column], dtype=float)
        require(
            values.ndim == 1
            and values.size == sample_count
            and bool(np.all(np.isfinite(values))),
            f"{column} must be a finite one-dimensional series matching time_s",
        )


def _case_grip_wrench_signals(data: np.ndarray, geometry: np.ndarray) -> dict[str, Any]:
    time = data["time_s"]
    lead_position, trail_position, midpoint = geometry
    lead_force = _vector(data, "lead_force_global", "n")
    trail_force = _vector(data, "trail_force_global", "n")
    lead_torque = _vector(data, "lead_free_torque_global", "nm")
    trail_torque = _vector(data, "trail_free_torque_global", "nm")
    lead_velocity = _vector(data, "lead_velocity_global", "m_s")
    trail_velocity = _vector(data, "trail_velocity_global", "m_s")
    source_resultant = _vector(data, "resultant_force_global_source", "n")
    source_couple_global = _vector(
        data, "equivalent_midpoint_couple_global_source", "nm"
    )
    source_couple_local = _vector(data, "equivalent_midpoint_couple_local_source", "nm")

    grip_vector = trail_position - lead_position
    grip_separation = np.linalg.norm(grip_vector, axis=1)
    axial_axis = grip_vector / grip_separation[:, None]
    normal_axis = np.cross(OUT_OF_PLANE_AXIS, axial_axis)
    lead_axial = np.sum(lead_force * axial_axis, axis=1)
    trail_axial = np.sum(trail_force * axial_axis, axis=1)
    lead_normal = np.sum(lead_force * normal_axis, axis=1)
    trail_normal = np.sum(trail_force * normal_axis, axis=1)
    lead_out = lead_force @ OUT_OF_PLANE_AXIS
    trail_out = trail_force @ OUT_OF_PLANE_AXIS
    reconstructed_local_forces = np.stack(
        (
            np.column_stack((lead_axial, lead_normal, lead_out)),
            np.column_stack((trail_axial, trail_normal, trail_out)),
        )
    )
    archived_local_forces = np.stack(
        (
            _vector(data, "lead_force_local_source", "n"),
            _vector(data, "trail_force_local_source", "n"),
        )
    )
    maximum_in_plane_force = float(
        np.max(np.linalg.norm(reconstructed_local_forces[..., :2], axis=2))
    )
    max_out_of_plane_force_n = float(
        max(np.max(np.abs(lead_out)), np.max(np.abs(trail_out)))
    )

    resultant = lead_force + trail_force
    force_moment_global = np.cross(lead_position - midpoint, lead_force) + np.cross(
        trail_position - midpoint, trail_force
    )
    free_torque_global = lead_torque + trail_torque
    reconstructed_couple_global = force_moment_global + free_torque_global
    force_moment = force_moment_global @ OUT_OF_PLANE_AXIS
    free_torque = free_torque_global @ OUT_OF_PLANE_AXIS
    equivalent_couple = source_couple_local[:, 2]
    ensure(
        bool(np.all(np.isfinite(grip_separation)) and np.all(grip_separation > 0.0)),
        "grip separation must be finite and strictly positive",
    )
    return {
        "time": time,
        "lead_position": lead_position,
        "trail_position": trail_position,
        "midpoint": midpoint,
        "lead_force": lead_force,
        "trail_force": trail_force,
        "lead_axial": lead_axial,
        "trail_axial": trail_axial,
        "lead_normal": lead_normal,
        "trail_normal": trail_normal,
        "resultant": resultant,
        "force_moment": force_moment,
        "free_torque": free_torque,
        "equivalent_couple": equivalent_couple,
        "grip_separation": grip_separation,
        "source_resultant": source_resultant,
        "reconstructed_couple_global": reconstructed_couple_global,
        "source_couple_global": source_couple_global,
        "reconstructed_local_forces": reconstructed_local_forces,
        "archived_local_forces": archived_local_forces,
        "maximum_in_plane_force": maximum_in_plane_force,
        "max_out_of_plane_force_n": max_out_of_plane_force_n,
        "lead_velocity": lead_velocity,
        "trail_velocity": trail_velocity,
        "grip_vector": grip_vector,
    }


_POWER_INPUT_KEYS = (
    "lead_force",
    "trail_force",
    "lead_velocity",
    "trail_velocity",
    "grip_vector",
    "resultant",
    "force_moment",
)


def _case_power_channels(
    data: np.ndarray, grip: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Contact power, its rigid/deformation split and the archived power sums."""
    require(
        all(key in grip for key in _POWER_INPUT_KEYS),
        "grip signals must come from _case_grip_wrench_signals",
    )
    lead_force = grip["lead_force"]
    trail_force = grip["trail_force"]
    lead_velocity = grip["lead_velocity"]
    trail_velocity = grip["trail_velocity"]
    grip_vector = grip["grip_vector"]

    midpoint_velocity = 0.5 * (lead_velocity + trail_velocity)
    contact_force_power = np.sum(
        lead_force * lead_velocity + trail_force * trail_velocity, axis=1
    )
    source_linear_power = data["lead_linear_power_w"] + data["trail_linear_power_w"]
    source_angular_power = data["lead_angular_power_w"] + data["trail_angular_power_w"]
    relative_velocity = trail_velocity - lead_velocity
    differential_force = 0.5 * (trail_force - lead_force)
    contact_omega_rad_s = np.sum(
        np.cross(grip_vector, relative_velocity) * OUT_OF_PLANE_AXIS, axis=1
    ) / np.sum(grip_vector**2, axis=1)
    rigid_relative_velocity = contact_omega_rad_s[:, None] * np.cross(
        OUT_OF_PLANE_AXIS, grip_vector
    )
    deformation_power = np.sum(
        differential_force * (relative_velocity - rigid_relative_velocity), axis=1
    )
    rigid_force_power = (
        np.sum(grip["resultant"] * midpoint_velocity, axis=1)
        + grip["force_moment"] * contact_omega_rad_s
    )
    channels = {
        "contact_force_power": contact_force_power,
        "source_linear_power": source_linear_power,
        "source_angular_power": source_angular_power,
        "rigid_force_power": rigid_force_power,
        "deformation_power": deformation_power,
        "contact_power": source_linear_power + source_angular_power,
        "decomposed_force_power": rigid_force_power + deformation_power,
        "contact_omega_rad_s": contact_omega_rad_s,
    }
    sample_count = int(np.asarray(data["time_s"]).size)
    ensure(
        all(np.shape(values) == (sample_count,) for values in channels.values()),
        "every power channel must hold one sample per time step",
    )
    return channels


def _case_metrics_record(
    data: np.ndarray, grip: dict[str, Any], power: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Assemble the per-case audit metrics from grip signals and power channels."""
    require(
        "equivalent_couple" in grip and "contact_force_power" in power,
        "metrics need grip signals and power channels for the same case",
    )
    time = grip["time"]
    equivalent_couple = grip["equivalent_couple"]
    grip_separation = grip["grip_separation"]
    force_moment = grip["force_moment"]
    free_torque = grip["free_torque"]
    max_out_of_plane_force_n = grip["max_out_of_plane_force_n"]
    maximum_in_plane_force = grip["maximum_in_plane_force"]
    contact_force_power = power["contact_force_power"]
    source_linear_power = power["source_linear_power"]

    crossing = _crossing_record(time, equivalent_couple)
    minimum_index = int(np.argmin(equivalent_couple))
    negative = equivalent_couple < 0.0
    negative_impulse = float(np.trapezoid(np.minimum(equivalent_couple, 0.0), time))
    lead_command = data["lead_command_torque_nm"]
    trail_command = data["trail_command_torque_nm"]
    interval_metrics = _interval_signal_metrics(
        time, crossing, contact_force_power, source_linear_power
    )
    couple_residual = grip["reconstructed_couple_global"] - grip["source_couple_global"]
    local_residual = grip["reconstructed_local_forces"] - grip["archived_local_forces"]
    metrics: dict[str, Any] = {
        "sample_count": int(time.size),
        "time_step_s": float(np.median(np.diff(time))),
        "grip_separation_mean_m": float(np.mean(grip_separation)),
        "grip_separation_range_m": [
            float(np.min(grip_separation)),
            float(np.max(grip_separation)),
        ],
        "maximum_resultant_reconstruction_residual_n": float(
            np.max(np.abs(grip["resultant"] - grip["source_resultant"]))
        ),
        "maximum_couple_reconstruction_residual_nm": float(
            np.max(np.abs(couple_residual))
        ),
        "maximum_out_of_plane_hand_force_n": max_out_of_plane_force_n,
        "maximum_in_plane_hand_force_n": maximum_in_plane_force,
        "maximum_out_of_plane_to_in_plane_force_ratio": float(
            max_out_of_plane_force_n / maximum_in_plane_force
        ),
        "maximum_local_force_projection_residual_n": float(
            np.max(np.abs(local_residual))
        ),
        "maximum_internal_force_power_identity_residual_w": float(
            np.max(np.abs(contact_force_power - power["decomposed_force_power"]))
        ),
        "maximum_archived_linear_power_discrepancy_w": float(
            np.max(np.abs(contact_force_power - source_linear_power))
        ),
        "maximum_abs_deformation_power_w": float(
            np.max(np.abs(power["deformation_power"]))
        ),
        "minimum_equivalent_couple_nm": float(equivalent_couple[minimum_index]),
        "minimum_couple_time_s": float(time[minimum_index]),
        "force_moment_at_minimum_nm": float(force_moment[minimum_index]),
        "free_torque_at_minimum_nm": float(free_torque[minimum_index]),
        "reconstructed_force_power_at_minimum_w": float(
            contact_force_power[minimum_index]
        ),
        "clubhead_speed_at_minimum_mph": float(
            data["clubhead_speed_mph"][minimum_index]
        ),
        "maximum_clubhead_speed_mph": float(np.max(data["clubhead_speed_mph"])),
        "maximum_clubhead_speed_time_s": float(
            time[int(np.argmax(data["clubhead_speed_mph"]))]
        ),
        "negative_duration_s": float(np.sum(negative) * np.median(np.diff(time))),
        "negative_angular_impulse_nm_s": negative_impulse,
        "crossings": crossing,
        "late_crossing_resampling": _resampling_sensitivity(
            time, equivalent_couple, float(crossing["late"]["time_s"])
        ),
        "maximum_abs_individual_command_torque_nm": float(
            max(np.max(np.abs(lead_command)), np.max(np.abs(trail_command)))
        ),
        "maximum_abs_net_command_torque_nm": float(
            np.max(np.abs(lead_command + trail_command))
        ),
        "maximum_free_torque_command_residual_nm": float(
            np.max(np.abs(free_torque - lead_command - trail_command))
        ),
        **interval_metrics,
    }
    ensure(
        metrics["sample_count"] == int(np.asarray(data["time_s"]).size),
        "metrics sample_count must match the case time series",
    )
    return metrics
