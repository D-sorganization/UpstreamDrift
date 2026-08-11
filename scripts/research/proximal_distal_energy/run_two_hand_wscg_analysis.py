"""Audit the archived WSCG two-hand force system and passive couple reversal."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.two_hand_wrench import (
    find_zero_crossings,
)

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
    / "Model Output"
    / "Tables"
)
CASE_NAMES = ("base", "ztcf", "delta")
OUT_OF_PLANE_AXIS = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


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


def _analyze_case(
    name: str, data: np.ndarray, geometry: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
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

    resultant = lead_force + trail_force
    force_moment_global = np.cross(lead_position - midpoint, lead_force) + np.cross(
        trail_position - midpoint, trail_force
    )
    free_torque_global = lead_torque + trail_torque
    reconstructed_couple_global = force_moment_global + free_torque_global
    force_moment = force_moment_global @ OUT_OF_PLANE_AXIS
    free_torque = free_torque_global @ OUT_OF_PLANE_AXIS
    equivalent_couple = source_couple_local[:, 2]

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

    crossing = _crossing_record(time, equivalent_couple)
    minimum_index = int(np.argmin(equivalent_couple))
    negative = equivalent_couple < 0.0
    negative_impulse = float(np.trapezoid(np.minimum(equivalent_couple, 0.0), time))
    metrics: dict[str, Any] = {
        "sample_count": int(time.size),
        "time_step_s": float(np.median(np.diff(time))),
        "grip_separation_mean_m": float(np.mean(grip_separation)),
        "grip_separation_range_m": [
            float(np.min(grip_separation)),
            float(np.max(grip_separation)),
        ],
        "maximum_resultant_reconstruction_residual_n": float(
            np.max(np.abs(resultant - source_resultant))
        ),
        "maximum_couple_reconstruction_residual_nm": float(
            np.max(np.abs(reconstructed_couple_global - source_couple_global))
        ),
        "maximum_out_of_plane_hand_force_n": float(
            max(np.max(np.abs(lead_out)), np.max(np.abs(trail_out)))
        ),
        "maximum_force_power_identity_residual_w": float(
            np.max(np.abs(contact_force_power - decomposed_force_power))
        ),
        "maximum_source_linear_power_residual_w": float(
            np.max(np.abs(contact_force_power - source_linear_power))
        ),
        "maximum_abs_deformation_power_w": float(np.max(np.abs(deformation_power))),
        "minimum_equivalent_couple_nm": float(equivalent_couple[minimum_index]),
        "minimum_couple_time_s": float(time[minimum_index]),
        "force_moment_at_minimum_nm": float(force_moment[minimum_index]),
        "free_torque_at_minimum_nm": float(free_torque[minimum_index]),
        "negative_duration_s": float(np.sum(negative) * np.median(np.diff(time))),
        "negative_angular_impulse_nm_s": negative_impulse,
        "crossings": crossing,
        "late_crossing_resampling": _resampling_sensitivity(
            time, equivalent_couple, float(crossing["late"]["time_s"])
        ),
        "maximum_abs_command_torque_nm": float(
            np.max(
                np.abs(data["lead_command_torque_nm"] + data["trail_command_torque_nm"])
            )
        ),
    }
    arrays = {
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
        "contact_force_power": contact_force_power,
        "source_angular_power": source_angular_power,
        "rigid_force_power": rigid_force_power,
        "deformation_power": deformation_power,
        "contact_power": contact_power,
        "decomposed_force_power": decomposed_force_power,
        "contact_omega_rad_s": contact_omega_rad_s,
        "grip_separation": grip_separation,
        "clubhead_speed_mph": data["clubhead_speed_mph"],
    }
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


def build_outputs() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the machine-readable audit and plot-ready arrays."""
    loaded = {name: _load_case(name) for name in CASE_NAMES}
    base = loaded["base"]
    geometry = np.stack(
        (
            _vector(base, "lead_position", "m"),
            _vector(base, "trail_position", "m"),
            _vector(base, "midpoint_position", "m"),
        )
    )
    record: dict[str, Any] = {
        "provenance": {
            "git_sha": _git_sha(),
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
        },
        "cases": {},
    }
    arrays: dict[str, np.ndarray] = {}
    case_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, data in loaded.items():
        metrics, computed = _analyze_case(name, data, geometry)
        record["cases"][name] = metrics
        case_arrays[name] = computed
        arrays.update({f"{name}__{key}": value for key, value in computed.items()})

    base_couple = case_arrays["base"]["equivalent_couple"]
    ztcf_couple = case_arrays["ztcf"]["equivalent_couple"]
    delta_couple = case_arrays["delta"]["equivalent_couple"]
    record["decomposition"] = {
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
    return record, arrays


def main() -> None:
    """Write JSON metrics and compressed plot arrays."""
    record, arrays = build_outputs()
    with (DATA_DIR / "two_hand_wscg_analysis.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    np.savez_compressed(DATA_DIR / "two_hand_wscg_analysis.npz", **arrays)


if __name__ == "__main__":
    main()
