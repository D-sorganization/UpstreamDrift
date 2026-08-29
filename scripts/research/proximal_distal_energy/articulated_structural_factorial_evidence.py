"""Versioned completeness and identity gates for structural time histories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

EVIDENCE_SIDECAR_SCHEMA = "articulated-structural-factorial-evidence/1.0.0"
REQUIRED_EVIDENCE_ARRAYS = (
    "time_s",
    "q",
    "qd",
    "elastic_coordinates",
    "elastic_velocities",
    "base_coordinates",
    "base_velocities",
    "station_force_on_club_n",
    "active_station",
    "active_set_transition",
    "net_club_force_n",
    "contact_power_w",
    "cumulative_contact_impulse_n_s",
    "cumulative_contact_work_j",
    "maximum_station_force_n",
    "active_station_count",
    "force_couple_vector_nm",
    "grip_strain_energy_j",
    "grip_dissipation_power_w",
    "virtual_power_residual_w",
    "shaft_strain_energy_j",
    "shaft_damping_power_w",
    "shaft_power_residual_w",
    "ground_force_n",
    "ground_intrinsic_free_moment_nm",
    "ground_transported_moment_nm",
    "ground_strain_energy_j",
    "ground_damping_power_w",
    "ground_power_residual_w",
    "total_mechanical_energy_j",
    "total_energy_j",
    "cumulative_dissipation_j",
    "work_energy_residual_j",
    "tip_bending_m",
    "twist_angle_rad",
    "base_translation_m",
    "base_pitch_rad",
)


def cumulative_trapezoid(
    values: NDArray[np.float64], time_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Integrate sample histories while retaining a zero-valued first row."""

    array = np.asarray(values, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or array.ndim < 1 or array.shape[0] != time.size:
        raise ValueError("trapezoid inputs must share one sample dimension")
    result = np.zeros_like(array, dtype=float)
    if time.size > 1:
        widths = np.diff(time).reshape((-1,) + (1,) * (array.ndim - 1))
        result[1:] = np.cumsum(0.5 * (array[1:] + array[:-1]) * widths, axis=0)
    return result


def _require_close(
    actual: NDArray[np.float64], expected: NDArray[np.float64], *, name: str
) -> None:
    scale = max(1.0, float(np.max(np.abs(expected))) if expected.size else 0.0)
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=1e-12, atol=1e-12 * scale
    ):
        raise ValueError(f"{name} identity does not close")


def validate_structural_evidence_arrays(arrays: Mapping[str, NDArray[Any]]) -> None:
    """Reject incomplete, malformed, or internally inconsistent evidence arrays."""

    missing = sorted(set(REQUIRED_EVIDENCE_ARRAYS) - set(arrays))
    if missing:
        raise ValueError("missing required arrays: " + ", ".join(missing))
    normalized = {name: np.asarray(arrays[name]) for name in REQUIRED_EVIDENCE_ARRAYS}
    if any(value.dtype.hasobject for value in normalized.values()):
        raise ValueError("evidence arrays must not use object dtype")
    if any(
        np.issubdtype(value.dtype, np.number) and np.any(~np.isfinite(value))
        for value in normalized.values()
    ):
        raise ValueError("numeric evidence arrays must be finite")
    time = np.asarray(normalized["time_s"], dtype=float)
    if (
        time.ndim != 1
        or time.size < 2
        or time[0] != 0.0
        or np.any(np.diff(time) <= 0.0)
    ):
        raise ValueError("time_s must start at zero and increase strictly")
    for name, value in normalized.items():
        if value.ndim < 1 or value.shape[0] != time.size:
            raise ValueError(f"{name} does not share the time sample dimension")
    station_force = normalized["station_force_on_club_n"]
    active_station = np.asarray(normalized["active_station"], dtype=bool)
    if station_force.ndim != 4 or station_force.shape[-1] != 3:
        raise ValueError(
            "station force history must have shape (time, hand, station, 3)"
        )
    if active_station.shape != station_force.shape[:-1]:
        raise ValueError("active station history does not match station forces")
    active_count = np.asarray(normalized["active_station_count"], dtype=int)
    if not np.array_equal(active_count, np.count_nonzero(active_station, axis=(1, 2))):
        raise ValueError("active station count does not match station history")
    expected_transition = np.zeros(time.size, dtype=bool)
    expected_transition[1:] = np.any(
        active_station[1:] != active_station[:-1], axis=(1, 2)
    )
    if not np.array_equal(
        np.asarray(normalized["active_set_transition"], dtype=bool), expected_transition
    ):
        raise ValueError("active-set transition history does not match station states")
    contact_force = np.asarray(normalized["net_club_force_n"], dtype=float)
    if contact_force.shape != (time.size, 3):
        raise ValueError("net club force history must have shape (time, 3)")
    _require_close(
        np.asarray(normalized["cumulative_contact_impulse_n_s"], dtype=float),
        cumulative_trapezoid(contact_force, time),
        name="cumulative contact impulse",
    )
    contact_power = np.asarray(normalized["contact_power_w"], dtype=float)
    if contact_power.shape != time.shape:
        raise ValueError("contact power history must have shape (time,)")
    _require_close(
        np.asarray(normalized["cumulative_contact_work_j"], dtype=float),
        cumulative_trapezoid(contact_power, time),
        name="cumulative contact work",
    )
    total_energy = np.asarray(normalized["total_energy_j"], dtype=float)
    cumulative_dissipation = np.asarray(
        normalized["cumulative_dissipation_j"], dtype=float
    )
    expected_residual = total_energy - total_energy[0] - cumulative_dissipation
    _require_close(
        np.asarray(normalized["work_energy_residual_j"], dtype=float),
        expected_residual,
        name="work-energy residual",
    )


__all__ = [
    "EVIDENCE_SIDECAR_SCHEMA",
    "REQUIRED_EVIDENCE_ARRAYS",
    "cumulative_trapezoid",
    "validate_structural_evidence_arrays",
]
