"""Audit the archived WSCG two-hand force system and passive couple reversal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.two_hand_wrench import (
    find_zero_crossings,
)
from src.shared.python.contracts import ensure, require

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
RAW_DIR = DATA_DIR / "wscg_two_hand_raw"
MATLAB_TABLE_DIR = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "2D_Golf_Model"
    / "matlab"
    / "Tables"
)
CASE_NAMES = ("base", "ztcf", "delta")
OUT_OF_PLANE_AXIS = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)
SCHEMA_VERSION = "wscg-two-hand-wrench-audit-v2"
JSON_PATH = DATA_DIR / "two_hand_wscg_analysis.json"
NPZ_PATH = DATA_DIR / "two_hand_wscg_analysis.npz"
_CASE_METRIC_KEYS = (
    "sample_count",
    "time_step_s",
    "grip_separation_mean_m",
    "grip_separation_range_m",
    "maximum_resultant_reconstruction_residual_n",
    "maximum_couple_reconstruction_residual_nm",
    "maximum_out_of_plane_hand_force_n",
    "maximum_in_plane_hand_force_n",
    "maximum_out_of_plane_to_in_plane_force_ratio",
    "maximum_local_force_projection_residual_n",
    "maximum_internal_force_power_identity_residual_w",
    "maximum_archived_linear_power_discrepancy_w",
    "maximum_abs_deformation_power_w",
    "minimum_equivalent_couple_nm",
    "minimum_couple_time_s",
    "force_moment_at_minimum_nm",
    "free_torque_at_minimum_nm",
    "reconstructed_force_power_at_minimum_w",
    "clubhead_speed_at_minimum_mph",
    "maximum_clubhead_speed_mph",
    "maximum_clubhead_speed_time_s",
    "negative_duration_s",
    "negative_angular_impulse_nm_s",
    "crossings",
    "late_crossing_resampling",
    "maximum_abs_individual_command_torque_nm",
    "maximum_abs_net_command_torque_nm",
    "maximum_free_torque_command_residual_nm",
    "negative_interval_duration_s",
    "negative_interval_reconstructed_force_work_j",
    "negative_interval_archived_linear_work_j",
    "negative_interval_reconstructed_force_power_negative_fraction",
    "negative_interval_force_power_sign_disagreement_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _source_hashes() -> dict[str, str]:
    """Hash the original tables, portable caches, and executable audit closure."""
    paths = (
        *(MATLAB_TABLE_DIR / f"{name.upper()}.mat" for name in CASE_NAMES),
        *(RAW_DIR / f"{name}.csv" for name in CASE_NAMES),
        REPO_ROOT
        / "scripts/research/proximal_distal_energy/export_two_hand_wscg_tables.m",
        REPO_ROOT
        / "scripts/research/proximal_distal_energy/run_two_hand_wscg_analysis.py",
        REPO_ROOT / "scripts/research/proximal_distal_energy/two_hand_wrench.py",
    )
    return {path.relative_to(REPO_ROOT).as_posix(): _sha256(path) for path in paths}


def _load_case(name: str) -> np.ndarray:
    if name not in CASE_NAMES:
        raise ValueError(f"unknown case {name!r}")
    data = np.genfromtxt(RAW_DIR / f"{name}.csv", delimiter=",", names=True)
    if data.ndim != 1 or data.size < 2:
        raise ValueError(f"{name} cache must contain at least two rows")
    return data


def _vector(data: np.ndarray, stem: str, unit: str) -> np.ndarray:
    names = tuple(f"{stem}_{axis}_{unit}" for axis in "xyz")
    missing = [name for name in names if name not in (data.dtype.names or ())]
    if missing:
        raise ValueError(f"missing vector columns: {missing}")
    result = np.column_stack([data[name] for name in names])
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{stem} contains non-finite values")
    return result


def _require_case_analysis_inputs(name: str, data: np.ndarray) -> None:
    """Validate case name and equal-length finite sample columns."""
    if name not in CASE_NAMES:
        raise ValueError(f"unknown case {name!r}")
    time = np.asarray(data["time_s"], dtype=float)
    require(time.ndim == 1, "time_s must be one-dimensional")
    require(time.size >= 2, "time_s must contain at least two samples")
    require(np.all(np.isfinite(time)), "time_s must be finite")
    sample_count = int(time.size)
    for column in ("clubhead_speed_mph", "lead_linear_power_w", "trail_linear_power_w"):
        values = np.asarray(data[column], dtype=float)
        require(values.ndim == 1, f"{column} must be one-dimensional")
        require(values.size == sample_count, f"{column} length must match time_s")
        require(np.all(np.isfinite(values)), f"{column} must be finite")


def _crossing_record(time: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    crossings = find_zero_crossings(time, values)
    if not crossings:
        raise ValueError("expected at least one equivalent-couple sign reversal")
    records: list[dict[str, float | str]] = []
    for crossing_time, direction in crossings:
        right = int(np.searchsorted(time, crossing_time))
        left = right - 1
        records.append(
            {
                "time_s": crossing_time,
                "direction": direction,
                "left_time_s": float(time[left]),
                "right_time_s": float(time[right]),
                "bracket_width_s": float(time[right] - time[left]),
            }
        )
    late = [record for record in records if float(record["time_s"]) >= 0.2]
    if len(late) != 1:
        raise ValueError(f"expected one late crossing, found {len(late)}")
    return {"all": records, "late": late[0]}


def _resampling_sensitivity(
    time: np.ndarray, values: np.ndarray, reference_time: float
) -> dict[str, Any]:
    estimates: list[dict[str, float | int]] = []
    for stride in (1, 2, 5, 10, 20):
        for offset in range(stride):
            sampled_time = time[offset::stride]
            sampled_values = values[offset::stride]
            candidates = [
                crossing
                for crossing, _ in find_zero_crossings(sampled_time, sampled_values)
                if crossing >= 0.2
            ]
            if len(candidates) == 1:
                estimates.append(
                    {
                        "stride": stride,
                        "offset": offset,
                        "effective_step_s": float(np.median(np.diff(sampled_time))),
                        "late_crossing_s": candidates[0],
                    }
                )
    differences = [
        abs(float(row["late_crossing_s"]) - reference_time) for row in estimates
    ]
    return {
        "estimates": estimates,
        "maximum_absolute_shift_s": max(differences),
        "maximum_effective_step_s": max(
            float(row["effective_step_s"]) for row in estimates
        ),
    }


def _interval_signal_metrics(
    time: np.ndarray,
    crossing: dict[str, Any],
    reconstructed_power: np.ndarray,
    archived_power: np.ndarray,
) -> dict[str, float]:
    """Integrate both power records over the interpolated negative-couple interval."""
    start = float(crossing["all"][0]["time_s"])
    stop = float(crossing["late"]["time_s"])
    interior = (time > start) & (time < stop)
    interval_time = np.concatenate(([start], time[interior], [stop]))
    reconstructed = np.concatenate(
        (
            [np.interp(start, time, reconstructed_power)],
            reconstructed_power[interior],
            [np.interp(stop, time, reconstructed_power)],
        )
    )
    archived = np.concatenate(
        (
            [np.interp(start, time, archived_power)],
            archived_power[interior],
            [np.interp(stop, time, archived_power)],
        )
    )
    return {
        "negative_interval_duration_s": stop - start,
        "negative_interval_reconstructed_force_work_j": float(
            np.trapezoid(reconstructed, interval_time)
        ),
        "negative_interval_archived_linear_work_j": float(
            np.trapezoid(archived, interval_time)
        ),
        "negative_interval_reconstructed_force_power_negative_fraction": float(
            np.mean(reconstructed < 0.0)
        ),
        "negative_interval_force_power_sign_disagreement_fraction": float(
            np.mean(np.sign(reconstructed) != np.sign(archived))
        ),
    }


@dataclass(frozen=True, slots=True)
class _CaseForceState:
    time: np.ndarray
    lead_position: np.ndarray
    trail_position: np.ndarray
    midpoint: np.ndarray
    lead_force: np.ndarray
    trail_force: np.ndarray
    lead_torque: np.ndarray
    trail_torque: np.ndarray
    lead_velocity: np.ndarray
    trail_velocity: np.ndarray
    source_resultant: np.ndarray
    source_couple_global: np.ndarray
    source_couple_local: np.ndarray
    grip_vector: np.ndarray
    grip_separation: np.ndarray
    lead_axial: np.ndarray
    trail_axial: np.ndarray
    lead_normal: np.ndarray
    trail_normal: np.ndarray
    lead_out: np.ndarray
    trail_out: np.ndarray
    reconstructed_local_forces: np.ndarray
    archived_local_forces: np.ndarray
    maximum_in_plane_force: float
    resultant: np.ndarray
    force_moment_global: np.ndarray
    free_torque_global: np.ndarray
    reconstructed_couple_global: np.ndarray
    force_moment: np.ndarray
    free_torque: np.ndarray
    equivalent_couple: np.ndarray
    contact_force_power: np.ndarray
    source_linear_power: np.ndarray
    source_angular_power: np.ndarray
    relative_velocity: np.ndarray
    differential_force: np.ndarray
    contact_omega_rad_s: np.ndarray
    rigid_relative_velocity: np.ndarray
    deformation_power: np.ndarray
    rigid_force_power: np.ndarray
    decomposed_force_power: np.ndarray
    contact_power: np.ndarray


def _projected_hand_local_forces(
    data: np.ndarray,
    lead_position: np.ndarray,
    trail_position: np.ndarray,
    lead_force: np.ndarray,
    trail_force: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Project global hand forces into the grip frame and compare to archived locals."""
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
    return (
        grip_vector,
        grip_separation,
        lead_axial,
        trail_axial,
        lead_normal,
        trail_normal,
        lead_out,
        trail_out,
        reconstructed_local_forces,
        archived_local_forces,
        maximum_in_plane_force,
    )


def _midpoint_power_decomposition(
    data: np.ndarray,
    *,
    lead_force: np.ndarray,
    trail_force: np.ndarray,
    lead_velocity: np.ndarray,
    trail_velocity: np.ndarray,
    grip_vector: np.ndarray,
    resultant: np.ndarray,
    force_moment: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Decompose internal force power into rigid and deformation channels."""
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
        np.sum(resultant * midpoint_velocity, axis=1)
        + force_moment * contact_omega_rad_s
    )
    decomposed_force_power = rigid_force_power + deformation_power
    contact_power = source_linear_power + source_angular_power
    return (
        contact_force_power,
        source_linear_power,
        source_angular_power,
        relative_velocity,
        differential_force,
        contact_omega_rad_s,
        rigid_relative_velocity,
        deformation_power,
        rigid_force_power,
        decomposed_force_power,
        contact_power,
    )


@dataclass(frozen=True, slots=True)
class _MidpointWrench:
    resultant: np.ndarray
    force_moment_global: np.ndarray
    free_torque_global: np.ndarray
    reconstructed_couple_global: np.ndarray
    force_moment: np.ndarray
    free_torque: np.ndarray
    equivalent_couple: np.ndarray


def _midpoint_wrench_components(
    lead_position: np.ndarray,
    trail_position: np.ndarray,
    midpoint: np.ndarray,
    lead_force: np.ndarray,
    trail_force: np.ndarray,
    lead_torque: np.ndarray,
    trail_torque: np.ndarray,
    source_couple_local: np.ndarray,
) -> _MidpointWrench:
    """Reconstruct the archived midpoint wrench and scalar couple channel."""
    resultant = lead_force + trail_force
    force_moment_global = np.cross(lead_position - midpoint, lead_force) + np.cross(
        trail_position - midpoint, trail_force
    )
    free_torque_global = lead_torque + trail_torque
    reconstructed_couple_global = force_moment_global + free_torque_global
    force_moment = force_moment_global @ OUT_OF_PLANE_AXIS
    free_torque = free_torque_global @ OUT_OF_PLANE_AXIS
    equivalent_couple = source_couple_local[:, 2]
    return _MidpointWrench(
        resultant=resultant,
        force_moment_global=force_moment_global,
        free_torque_global=free_torque_global,
        reconstructed_couple_global=reconstructed_couple_global,
        force_moment=force_moment,
        free_torque=free_torque,
        equivalent_couple=equivalent_couple,
    )


def _archived_wrench_vectors(
    data: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load archived global wrench columns for one case row."""
    return (
        _vector(data, "lead_force_global", "n"),
        _vector(data, "trail_force_global", "n"),
        _vector(data, "lead_free_torque_global", "nm"),
        _vector(data, "trail_free_torque_global", "nm"),
        _vector(data, "lead_velocity_global", "m_s"),
        _vector(data, "trail_velocity_global", "m_s"),
        _vector(data, "resultant_force_global_source", "n"),
        _vector(data, "equivalent_midpoint_couple_global_source", "nm"),
        _vector(data, "equivalent_midpoint_couple_local_source", "nm"),
    )


def _compose_case_force_state(
    time: np.ndarray,
    geometry: np.ndarray,
    archived_vectors: tuple[np.ndarray, ...],
    grip_frame: tuple[np.ndarray | float, ...],
    wrench: _MidpointWrench,
    power: tuple[np.ndarray, ...],
) -> _CaseForceState:
    """Materialize the frozen case state from reconstructed intermediates."""
    lead_position, trail_position, midpoint = geometry
    (
        lead_force,
        trail_force,
        lead_torque,
        trail_torque,
        lead_velocity,
        trail_velocity,
        source_resultant,
        source_couple_global,
        source_couple_local,
    ) = archived_vectors
    (
        grip_vector,
        grip_separation,
        lead_axial,
        trail_axial,
        lead_normal,
        trail_normal,
        lead_out,
        trail_out,
        reconstructed_local_forces,
        archived_local_forces,
        maximum_in_plane_force,
    ) = grip_frame
    (
        contact_force_power,
        source_linear_power,
        source_angular_power,
        relative_velocity,
        differential_force,
        contact_omega_rad_s,
        rigid_relative_velocity,
        deformation_power,
        rigid_force_power,
        decomposed_force_power,
        contact_power,
    ) = power
    return _CaseForceState(
        time=time,
        lead_position=lead_position,
        trail_position=trail_position,
        midpoint=midpoint,
        lead_force=lead_force,
        trail_force=trail_force,
        lead_torque=lead_torque,
        trail_torque=trail_torque,
        lead_velocity=lead_velocity,
        trail_velocity=trail_velocity,
        source_resultant=source_resultant,
        source_couple_global=source_couple_global,
        source_couple_local=source_couple_local,
        grip_vector=grip_vector,
        grip_separation=grip_separation,
        lead_axial=lead_axial,
        trail_axial=trail_axial,
        lead_normal=lead_normal,
        trail_normal=trail_normal,
        lead_out=lead_out,
        trail_out=trail_out,
        reconstructed_local_forces=reconstructed_local_forces,
        archived_local_forces=archived_local_forces,
        maximum_in_plane_force=maximum_in_plane_force,
        resultant=wrench.resultant,
        force_moment_global=wrench.force_moment_global,
        free_torque_global=wrench.free_torque_global,
        reconstructed_couple_global=wrench.reconstructed_couple_global,
        force_moment=wrench.force_moment,
        free_torque=wrench.free_torque,
        equivalent_couple=wrench.equivalent_couple,
        contact_force_power=contact_force_power,
        source_linear_power=source_linear_power,
        source_angular_power=source_angular_power,
        relative_velocity=relative_velocity,
        differential_force=differential_force,
        contact_omega_rad_s=contact_omega_rad_s,
        rigid_relative_velocity=rigid_relative_velocity,
        deformation_power=deformation_power,
        rigid_force_power=rigid_force_power,
        decomposed_force_power=decomposed_force_power,
        contact_power=contact_power,
    )


def _reconstruct_case_force_state(
    data: np.ndarray, geometry: np.ndarray
) -> _CaseForceState:
    """Rebuild forces, couples, and power decomposition for one archived case."""
    time = data["time_s"]
    lead_position, trail_position, midpoint = geometry
    archived_vectors = _archived_wrench_vectors(data)
    grip_frame = _projected_hand_local_forces(
        data,
        lead_position,
        trail_position,
        archived_vectors[0],
        archived_vectors[1],
    )
    wrench = _midpoint_wrench_components(
        lead_position,
        trail_position,
        midpoint,
        archived_vectors[0],
        archived_vectors[1],
        archived_vectors[2],
        archived_vectors[3],
        archived_vectors[8],
    )
    power = _midpoint_power_decomposition(
        data,
        lead_force=archived_vectors[0],
        trail_force=archived_vectors[1],
        lead_velocity=archived_vectors[4],
        trail_velocity=archived_vectors[5],
        grip_vector=grip_frame[0],
        resultant=wrench.resultant,
        force_moment=wrench.force_moment,
    )
    return _compose_case_force_state(
        time, geometry, archived_vectors, grip_frame, wrench, power
    )


def _assemble_case_metrics(
    data: np.ndarray,
    state: _CaseForceState,
    crossing: dict[str, Any],
    interval_metrics: dict[str, float],
) -> dict[str, Any]:
    """Assemble the per-case metrics record in the committed key order."""
    time = state.time
    equivalent_couple = state.equivalent_couple
    minimum_index = int(np.argmin(equivalent_couple))
    negative = equivalent_couple < 0.0
    negative_impulse = float(np.trapezoid(np.minimum(equivalent_couple, 0.0), time))
    lead_command = data["lead_command_torque_nm"]
    trail_command = data["trail_command_torque_nm"]
    metrics: dict[str, Any] = {
        "sample_count": int(time.size),
        "time_step_s": float(np.median(np.diff(time))),
        "grip_separation_mean_m": float(np.mean(state.grip_separation)),
        "grip_separation_range_m": [
            float(np.min(state.grip_separation)),
            float(np.max(state.grip_separation)),
        ],
        "maximum_resultant_reconstruction_residual_n": float(
            np.max(np.abs(state.resultant - state.source_resultant))
        ),
        "maximum_couple_reconstruction_residual_nm": float(
            np.max(
                np.abs(state.reconstructed_couple_global - state.source_couple_global)
            )
        ),
        "maximum_out_of_plane_hand_force_n": float(
            max(np.max(np.abs(state.lead_out)), np.max(np.abs(state.trail_out)))
        ),
        "maximum_in_plane_hand_force_n": state.maximum_in_plane_force,
        "maximum_out_of_plane_to_in_plane_force_ratio": float(
            max(np.max(np.abs(state.lead_out)), np.max(np.abs(state.trail_out)))
            / state.maximum_in_plane_force
        ),
        "maximum_local_force_projection_residual_n": float(
            np.max(
                np.abs(state.reconstructed_local_forces - state.archived_local_forces)
            )
        ),
        "maximum_internal_force_power_identity_residual_w": float(
            np.max(np.abs(state.contact_force_power - state.decomposed_force_power))
        ),
        "maximum_archived_linear_power_discrepancy_w": float(
            np.max(np.abs(state.contact_force_power - state.source_linear_power))
        ),
        "maximum_abs_deformation_power_w": float(
            np.max(np.abs(state.deformation_power))
        ),
        "minimum_equivalent_couple_nm": float(equivalent_couple[minimum_index]),
        "minimum_couple_time_s": float(time[minimum_index]),
        "force_moment_at_minimum_nm": float(state.force_moment[minimum_index]),
        "free_torque_at_minimum_nm": float(state.free_torque[minimum_index]),
        "reconstructed_force_power_at_minimum_w": float(
            state.contact_force_power[minimum_index]
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
            np.max(np.abs(state.free_torque - lead_command - trail_command))
        ),
        **interval_metrics,
    }
    ensure(
        tuple(metrics.keys()) == _CASE_METRIC_KEYS,
        "case metrics must preserve committed key order",
    )
    return metrics


def _assemble_case_output_arrays(
    data: np.ndarray, geometry: np.ndarray, state: _CaseForceState
) -> dict[str, np.ndarray]:
    """Collect plot-ready arrays for one case."""
    lead_position, trail_position, midpoint = geometry
    return {
        "time": state.time,
        "lead_position": lead_position,
        "trail_position": trail_position,
        "midpoint": midpoint,
        "lead_force": state.lead_force,
        "trail_force": state.trail_force,
        "lead_axial": state.lead_axial,
        "trail_axial": state.trail_axial,
        "lead_normal": state.lead_normal,
        "trail_normal": state.trail_normal,
        "resultant": state.resultant,
        "force_moment": state.force_moment,
        "free_torque": state.free_torque,
        "equivalent_couple": state.equivalent_couple,
        "contact_force_power": state.contact_force_power,
        "source_angular_power": state.source_angular_power,
        "rigid_force_power": state.rigid_force_power,
        "deformation_power": state.deformation_power,
        "contact_power": state.contact_power,
        "decomposed_force_power": state.decomposed_force_power,
        "contact_omega_rad_s": state.contact_omega_rad_s,
        "grip_separation": state.grip_separation,
        "clubhead_speed_mph": data["clubhead_speed_mph"],
    }


def _analyze_case(
    name: str, data: np.ndarray, geometry: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _require_case_analysis_inputs(name, data)
    state = _reconstruct_case_force_state(data, geometry)
    crossing = _crossing_record(state.time, state.equivalent_couple)
    interval_metrics = _interval_signal_metrics(
        state.time,
        crossing,
        state.contact_force_power,
        state.source_linear_power,
    )
    metrics = _assemble_case_metrics(data, state, crossing, interval_metrics)
    arrays = _assemble_case_output_arrays(data, geometry, state)
    return metrics, arrays


def _geometry_sweep(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    index = int(np.argmin(arrays["equivalent_couple"]))
    lead = arrays["lead_position"][index]
    trail = arrays["trail_position"][index]
    midpoint = arrays["midpoint"][index]
    lead_force = arrays["lead_force"][index]
    trail_force = arrays["trail_force"][index]
    half_lead = lead - midpoint
    half_trail = trail - midpoint

    scales = np.linspace(0.0, 2.0, 81)
    spacing_moment = np.array(
        [
            (
                np.cross(scale * half_lead, lead_force)
                + np.cross(scale * half_trail, trail_force)
            )
            @ OUT_OF_PLANE_AXIS
            for scale in scales
        ]
    )
    angles_deg = np.linspace(-120.0, 120.0, 241)
    orientation_moment: list[float] = []
    co_rotated_moment: list[float] = []
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        # Rotation in the declared model plane about its fixed normal.
        skew = np.array(
            [
                [0.0, -OUT_OF_PLANE_AXIS[2], OUT_OF_PLANE_AXIS[1]],
                [OUT_OF_PLANE_AXIS[2], 0.0, -OUT_OF_PLANE_AXIS[0]],
                [-OUT_OF_PLANE_AXIS[1], OUT_OF_PLANE_AXIS[0], 0.0],
            ]
        )
        rotation = (
            np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        )
        rotated_lead = rotation @ half_lead
        rotated_trail = rotation @ half_trail
        orientation_moment.append(
            float(
                (
                    np.cross(rotated_lead, lead_force)
                    + np.cross(rotated_trail, trail_force)
                )
                @ OUT_OF_PLANE_AXIS
            )
        )
        co_rotated_moment.append(
            float(
                (
                    np.cross(rotated_lead, rotation @ lead_force)
                    + np.cross(rotated_trail, rotation @ trail_force)
                )
                @ OUT_OF_PLANE_AXIS
            )
        )
    return {
        "source_index": np.array(index),
        "spacing_scale": scales,
        "spacing_force_moment_nm": spacing_moment,
        "orientation_deg": angles_deg,
        "orientation_force_moment_nm": np.asarray(orientation_moment),
        "co_rotated_force_moment_nm": np.asarray(co_rotated_moment),
    }


def _audit_provenance_block() -> dict[str, Any]:
    return {
        "source": "Archived WSCG/Simscape output tables",
        "source_table_sha256": {
            name.upper(): _sha256(MATLAB_TABLE_DIR / f"{name.upper()}.mat")
            for name in CASE_NAMES
        },
        "force_direction": "Force exerted by each wrist on the club",
        "lead_hand": "left wrist in the archived right-handed model",
        "local_axes": (
            "+x from lead to trail contact; +z out of the declared model plane; "
            "+y = +z cross +x"
        ),
        "moment_sign": "positive about +z by the right-hand rule",
        "counterfactual_contract": (
            "ZTCF recomputes reactions on the BASE state history with commanded "
            "joint torques and kill damping set to zero; it is pointwise, not a "
            "new forward rollout."
        ),
    }


def _shared_case_geometry(base: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            _vector(base, "lead_position", "m"),
            _vector(base, "trail_position", "m"),
            _vector(base, "midpoint_position", "m"),
        )
    )


def _state_match_block(loaded: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "maximum_contact_position_difference_m": float(
            max(
                np.max(
                    np.abs(
                        _vector(loaded["base"], name, "m")
                        - _vector(loaded["ztcf"], name, "m")
                    )
                )
                for name in (
                    "lead_position",
                    "trail_position",
                    "midpoint_position",
                )
            )
        ),
        "maximum_contact_velocity_difference_m_s": float(
            max(
                np.max(
                    np.abs(
                        _vector(loaded["base"], name, "m_s")
                        - _vector(loaded["ztcf"], name, "m_s")
                    )
                )
                for name in ("lead_velocity_global", "trail_velocity_global")
            )
        ),
        "maximum_clubhead_speed_difference_mph": float(
            np.max(
                np.abs(
                    loaded["base"]["clubhead_speed_mph"]
                    - loaded["ztcf"]["clubhead_speed_mph"]
                )
            )
        ),
        "interpretation": (
            "The exported ZTCF state matches BASE to the recorded cache precision; "
            "the residuals are reported rather than rounded to exact zero."
        ),
    }


def _decomposition_block(
    case_arrays: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    base_couple = case_arrays["base"]["equivalent_couple"]
    ztcf_couple = case_arrays["ztcf"]["equivalent_couple"]
    delta_couple = case_arrays["delta"]["equivalent_couple"]
    return {
        "maximum_abs_base_minus_ztcf_minus_delta_couple_residual_nm": float(
            np.max(np.abs(base_couple - ztcf_couple - delta_couple))
        ),
        "minimum_ztcf_to_base_couple_ratio_at_base_minimum": float(
            ztcf_couple[np.argmin(base_couple)] / np.min(base_couple)
        ),
        "interpretation": (
            "The nonzero ZTCF equivalent couple is a constraint-reaction wrench at "
            "the achieved BASE states. It is passive in the pointwise actuation "
            "sense, but it is not an unforced future trajectory."
        ),
    }


def _attach_geometry_sweep_summary(
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    case_arrays: dict[str, dict[str, np.ndarray]],
) -> None:
    sweep = _geometry_sweep(case_arrays["ztcf"])
    arrays.update({f"sweep__{key}": value for key, value in sweep.items()})
    record["geometry_sweep"] = {
        "source_case": "ztcf",
        "source_time_s": float(case_arrays["ztcf"]["time"][int(sweep["source_index"])]),
        "spacing_scale_range": [0.0, 2.0],
        "orientation_range_deg": [-120.0, 120.0],
        "maximum_co_rotation_residual_nm": float(
            np.max(
                np.abs(
                    sweep["co_rotated_force_moment_nm"]
                    - sweep["co_rotated_force_moment_nm"][120]
                )
            )
        ),
    }


def build_outputs() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the machine-readable audit and plot-ready arrays."""
    loaded = {name: _load_case(name) for name in CASE_NAMES}
    geometry = _shared_case_geometry(loaded["base"])
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": _source_hashes(),
        "provenance": _audit_provenance_block(),
        "cases": {},
    }
    arrays: dict[str, np.ndarray] = {}
    case_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, data in loaded.items():
        metrics, computed = _analyze_case(name, data, geometry)
        record["cases"][name] = metrics
        case_arrays[name] = computed
        arrays.update({f"{name}__{key}": value for key, value in computed.items()})

    record["state_match"] = _state_match_block(loaded)
    record["decomposition"] = _decomposition_block(case_arrays)
    _attach_geometry_sweep_summary(record, arrays, case_arrays)
    return record, arrays


def write_outputs(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Write JSON metrics and compressed plot arrays."""
    record, arrays = build_outputs()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_PATH.name
    npz_path = output_dir / NPZ_PATH.name
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path


def main() -> None:
    """Write JSON metrics and compressed plot arrays."""
    write_outputs()


if __name__ == "__main__":
    main()
