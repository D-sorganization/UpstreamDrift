"""Audit the archived WSCG two-hand force system and passive couple reversal."""

from __future__ import annotations

import hashlib
import json
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
    / "Tables"
)
CASE_NAMES = ("base", "ztcf", "delta")
OUT_OF_PLANE_AXIS = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)
SCHEMA_VERSION = "wscg-two-hand-wrench-audit-v2"
JSON_PATH = DATA_DIR / "two_hand_wscg_analysis.json"
NPZ_PATH = DATA_DIR / "two_hand_wscg_analysis.npz"
# Plot-ready per-case arrays, in the archived NPZ key order.
_CASE_GRIP_ARRAY_KEYS = (
    "time",
    "lead_position",
    "trail_position",
    "midpoint",
    "lead_force",
    "trail_force",
    "lead_axial",
    "trail_axial",
    "lead_normal",
    "trail_normal",
    "resultant",
    "force_moment",
    "free_torque",
    "equivalent_couple",
)
_CASE_POWER_ARRAY_KEYS = (
    "contact_force_power",
    "source_angular_power",
    "rigid_force_power",
    "deformation_power",
    "contact_power",
    "decomposed_force_power",
    "contact_omega_rad_s",
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


def _analyze_case(
    name: str, data: np.ndarray, geometry: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return one case's audit metrics and plot-ready arrays."""
    from scripts.research.proximal_distal_energy.run_two_hand_wscg_case import (
        _case_grip_wrench_signals,
        _case_metrics_record,
        _case_power_channels,
        _require_case_analysis_inputs,
    )

    _require_case_analysis_inputs(name, data)
    grip = _case_grip_wrench_signals(data, geometry)
    power = _case_power_channels(data, grip)
    metrics = _case_metrics_record(data, grip, power)
    arrays = {
        **{key: grip[key] for key in _CASE_GRIP_ARRAY_KEYS},
        **{key: power[key] for key in _CASE_POWER_ARRAY_KEYS},
        "grip_separation": grip["grip_separation"],
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


_CONTACT_POSITION_STEMS = ("lead_position", "trail_position", "midpoint_position")
_CONTACT_VELOCITY_STEMS = ("lead_velocity_global", "trail_velocity_global")
# Index of the unrotated (0 deg) sample in the geometry sweep's orientation grid.
_SWEEP_REFERENCE_INDEX = 120


def _provenance_record() -> dict[str, Any]:
    """Describe the archived source tables and the sign and axis conventions."""
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


def _max_state_difference(
    loaded: dict[str, np.ndarray], stems: tuple[str, ...], unit: str
) -> float:
    """Largest BASE-minus-ZTCF absolute difference over the named vector stems."""
    return float(
        max(
            np.max(
                np.abs(
                    _vector(loaded["base"], stem, unit)
                    - _vector(loaded["ztcf"], stem, unit)
                )
            )
            for stem in stems
        )
    )


def _state_match_record(loaded: dict[str, np.ndarray]) -> dict[str, Any]:
    """Report how closely the exported ZTCF state history matches BASE."""
    speed_difference = (
        loaded["base"]["clubhead_speed_mph"] - loaded["ztcf"]["clubhead_speed_mph"]
    )
    return {
        "maximum_contact_position_difference_m": _max_state_difference(
            loaded, _CONTACT_POSITION_STEMS, "m"
        ),
        "maximum_contact_velocity_difference_m_s": _max_state_difference(
            loaded, _CONTACT_VELOCITY_STEMS, "m_s"
        ),
        "maximum_clubhead_speed_difference_mph": float(
            np.max(np.abs(speed_difference))
        ),
        "interpretation": (
            "The exported ZTCF state matches BASE to the recorded cache precision; "
            "the residuals are reported rather than rounded to exact zero."
        ),
    }


def _decomposition_record(
    case_arrays: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Check BASE = ZTCF + DELTA for the equivalent couple."""
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


def _geometry_sweep_record(
    ztcf_arrays: dict[str, np.ndarray], sweep: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Summarize the ZTCF contact-geometry sweep."""
    co_rotated = sweep["co_rotated_force_moment_nm"]
    return {
        "source_case": "ztcf",
        "source_time_s": float(ztcf_arrays["time"][int(sweep["source_index"])]),
        "spacing_scale_range": [0.0, 2.0],
        "orientation_range_deg": [-120.0, 120.0],
        "maximum_co_rotation_residual_nm": float(
            np.max(np.abs(co_rotated - co_rotated[_SWEEP_REFERENCE_INDEX]))
        ),
    }


def _analyze_all_cases(
    loaded: dict[str, np.ndarray], geometry: np.ndarray
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Run the per-case audit for every case, in ``CASE_NAMES`` order."""
    metrics: dict[str, Any] = {}
    case_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, data in loaded.items():
        metrics[name], case_arrays[name] = _analyze_case(name, data, geometry)
    return metrics, case_arrays


def build_outputs() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the machine-readable audit and plot-ready arrays.

    Postcondition: the record keys and the array keys keep the published order,
    because ``write_outputs`` serializes them as-is and the JSON is byte-compared.
    """
    loaded = {name: _load_case(name) for name in CASE_NAMES}
    geometry = np.stack(
        [_vector(loaded["base"], stem, "m") for stem in _CONTACT_POSITION_STEMS]
    )
    case_metrics, case_arrays = _analyze_all_cases(loaded, geometry)
    arrays: dict[str, np.ndarray] = {
        f"{name}__{key}": value
        for name, computed in case_arrays.items()
        for key, value in computed.items()
    }
    sweep = _geometry_sweep(case_arrays["ztcf"])
    arrays.update({f"sweep__{key}": value for key, value in sweep.items()})
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": _source_hashes(),
        "provenance": _provenance_record(),
        "cases": case_metrics,
        "state_match": _state_match_record(loaded),
        "decomposition": _decomposition_record(case_arrays),
        "geometry_sweep": _geometry_sweep_record(case_arrays["ztcf"], sweep),
    }
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
