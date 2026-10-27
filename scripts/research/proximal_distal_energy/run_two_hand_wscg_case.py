"""Per-case input contract and grip-wrench signals for the WSCG audit."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.run_two_hand_wscg_analysis import (
    CASE_NAMES,
    OUT_OF_PLANE_AXIS,
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
