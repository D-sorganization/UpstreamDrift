"""Generate the phase-resolved shoulder-velocity drift-transfer atlas."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.shoulder_velocity_transfer import (
    VelocitySweepRequest,
    classify_transfer_phase,
    evaluate_velocity_sweep,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
FIGURE_DIR = DATA_DIR / "shoulder_velocity_transfer/figures"
JSON_PATH = DATA_DIR / "shoulder_velocity_transfer_study.json"
NPZ_PATH = DATA_DIR / "shoulder_velocity_transfer_study.npz"
SCHEMA_VERSION = "shoulder-velocity-transfer-evidence-v2"
PHASE_FRACTIONS = (0.0, 0.15, 0.40, 0.70, 0.95)
MINIMUM_VELOCITY_SCALE_RAD_S = 4.0
VELOCITY_OFFSET_FRACTION = 0.75
VELOCITY_CONSTRAINTS = (
    "preserve_relative_club_rate",
    "preserve_absolute_club_rate",
)


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/research/proximal_distal_energy/interaction_forces.py",
        "scripts/research/proximal_distal_energy/run_experiments.py",
        "scripts/research/proximal_distal_energy/shoulder_velocity_transfer.py",
        "scripts/research/proximal_distal_energy/run_shoulder_velocity_transfer_study.py",
        "scripts/research/proximal_distal_energy/swing_model.py",
        "scripts/research/proximal_distal_energy/torque_programs.py",
        "src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py",
        "src/shared/python/simulation_backends/factory.py",
        "src/shared/python/simulation_backends/model_params.py",
        "src/shared/python/simulation_backends/ode_backend.py",
        "src/shared/python/simulation_backends/protocol.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _velocity_scale(reference_rate: float) -> float:
    return max(abs(reference_rate), MINIMUM_VELOCITY_SCALE_RAD_S)


def _velocity_grid(reference_rate: float) -> np.ndarray:
    """Return a reference-centered sweep along the achieved rate direction."""
    direction = 1.0 if reference_rate >= 0.0 else -1.0
    scale = _velocity_scale(reference_rate)
    offsets = np.linspace(-VELOCITY_OFFSET_FRACTION, VELOCITY_OFFSET_FRACTION, 9)
    return reference_rate + direction * offsets * scale


def _reference_trace() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    params = GolfModelParams.default()
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    time, q, velocity, control = rollout_program(params, program)
    impact = find_impact(time, q, velocity, PlanarInertials.from_params(params))
    if impact is None:
        raise RuntimeError("registered reference program did not reach impact")
    return time, q, velocity, control, float(impact[0])


def _rows() -> list[dict[str, Any]]:
    params = GolfModelParams.default()
    time, q, velocity, control, impact_time = _reference_trace()
    rows: list[dict[str, Any]] = []
    for fraction in PHASE_FRACTIONS:
        sample_time = fraction * impact_time
        index = int(np.argmin(np.abs(time - sample_time)))
        reference_rate = float(velocity[index, 0])
        sweep_scale = _velocity_scale(reference_rate)
        for constraint in VELOCITY_CONSTRAINTS:
            request = VelocitySweepRequest(
                q_rad=q[index],
                reference_velocity_rad_s=velocity[index],
                control_nm=control[index],
                proximal_velocity_rad_s=_velocity_grid(reference_rate),
                velocity_constraint=constraint,
            )
            for sample in evaluate_velocity_sweep(request, params):
                row = asdict(sample)
                row.update(
                    {
                        "normalized_downswing_time": fraction,
                        "phase": classify_transfer_phase(fraction),
                        "reference_time_s": float(time[index]),
                        "reference_sample_index": index,
                        "reference_proximal_velocity_rad_s": reference_rate,
                        "velocity_sweep_scale_rad_s": sweep_scale,
                        "velocity_constraint": constraint,
                    }
                )
                rows.append(row)
    return rows


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def _phase_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for phase in dict.fromkeys(row["phase"] for row in rows):
        for constraint in VELOCITY_CONSTRAINTS:
            selected = [
                row
                for row in rows
                if row["phase"] == phase and row["velocity_constraint"] == constraint
            ]
            velocity = _array(selected, "proximal_velocity_rad_s")
            drift_power = _array(selected, "drift_grip_power_w")
            braking = _array(selected, "braking_grip_power_w")
            coefficients = np.polyfit(velocity, drift_power, 1)
            fitted = np.polyval(coefficients, velocity)
            residual_sum_squares = float(np.sum((drift_power - fitted) ** 2))
            total_sum_squares = float(np.sum((drift_power - np.mean(drift_power)) ** 2))
            linear_r_squared = (
                1.0 - residual_sum_squares / total_sum_squares
                if total_sum_squares > 1e-20
                else 1.0
            )
            reference_rate = float(selected[0]["reference_proximal_velocity_rad_s"])
            center_index = len(velocity) // 2
            centered_slope = float(
                (drift_power[center_index + 1] - drift_power[center_index - 1])
                / (velocity[center_index + 1] - velocity[center_index - 1])
            )
            summaries.append(
                {
                    "phase": phase,
                    "velocity_constraint": constraint,
                    "reference_proximal_rate_rad_s": reference_rate,
                    "velocity_sweep_scale_rad_s": float(
                        selected[0]["velocity_sweep_scale_rad_s"]
                    ),
                    "reference_state_in_grid": bool(
                        np.any(np.isclose(velocity, reference_rate))
                    ),
                    "drift_power_slope_w_per_rad_s": float(coefficients[0]),
                    "reference_centered_drift_power_slope_w_per_rad_s": (
                        centered_slope
                    ),
                    "drift_power_linear_r_squared": linear_r_squared,
                    "drift_power_endpoint_delta_w": float(
                        drift_power[-1] - drift_power[0]
                    ),
                    "braking_power_slope_w_per_rad_s": float(
                        np.polyfit(velocity, braking, 1)[0]
                    ),
                    "minimum_braking_power_w": float(np.min(braking)),
                    "maximum_drift_grip_power_w": float(np.max(drift_power)),
                    "drift_power_monotonic_increasing": bool(
                        np.all(np.diff(drift_power) >= -1e-10)
                    ),
                }
            )
    return summaries


def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return deterministic metadata and machine-readable atlas arrays."""
    rows = _rows()
    scalar_keys = (
        "proximal_velocity_rad_s",
        "relative_club_velocity_rad_s",
        "club_angular_velocity_rad_s",
        "clubhead_speed_m_s",
        "distal_kinetic_energy_j",
        "total_club_angular_acceleration_rad_s2",
        "drift_club_angular_acceleration_rad_s2",
        "control_club_angular_acceleration_rad_s2",
        "acceleration_closure_residual_rad_s2",
        "total_grip_power_w",
        "drift_grip_power_w",
        "control_grip_power_w",
        "braking_grip_power_w",
        "force_closure_residual_n",
        "normalized_downswing_time",
        "reference_time_s",
    )
    arrays = {key: _array(rows, key) for key in scalar_keys}
    arrays["grip_force_total_n"] = np.asarray(
        [row["grip_force_total_n"] for row in rows], dtype=float
    )
    arrays["grip_force_drift_n"] = np.asarray(
        [row["grip_force_drift_n"] for row in rows], dtype=float
    )
    arrays["grip_force_control_n"] = np.asarray(
        [row["grip_force_control_n"] for row in rows], dtype=float
    )
    phases = list(dict.fromkeys(row["phase"] for row in rows))
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "phase-resolved-proximal-link-velocity-atlas-v2",
        "model_tier": "exact_planar_double_pendulum",
        "counterfactual_kind": "pointwise_state_matched_zero_applied_control",
        "proximal_coordinate_meaning": (
            "first-link angular velocity; not anatomical shoulder or torso velocity"
        ),
        "phase_labels": phases,
        "phase_fractions": list(PHASE_FRACTIONS),
        "velocity_constraints": list(VELOCITY_CONSTRAINTS),
        "velocity_sweep_contract": {
            "kind": "reference_centered_directional_rate_sweep",
            "reference_state_included": True,
            "offset_fraction_of_scale": VELOCITY_OFFSET_FRACTION,
            "minimum_scale_rad_s": MINIMUM_VELOCITY_SCALE_RAD_S,
            "scale_definition": "max(abs(reference proximal rate), minimum scale)",
            "zero_rate_direction": "positive model coordinate",
            "stored_energy_is_not_matched": True,
        },
        "rows": rows,
        "phase_summaries": _phase_summaries(rows),
        "claim_status": {
            "local_velocity_drift_interaction": "tested_in_planar_pointwise_model",
            "forward_release_strategy": "untested_by_this_atlas",
            "anatomical_shoulder_strategy": "untested",
            "torso_rotation_strategy": "untested",
            "two_hand_internal_force_allocation": "untested_by_this_atlas",
            "universal_coaching_instruction": "unsupported",
        },
        "falsification_tests": [
            "Match configuration and applied controls while sweeping proximal rate.",
            "Repeat while preserving relative and absolute club angular rate.",
            "Retain negative and nonmonotonic drift-power regions.",
            "Require acceleration and force drift-plus-control closure.",
            "Test forward branches before interpreting a pointwise benefit as speed gain.",
            "Test an independently actuated torso and measured bilateral grip wrenches.",
        ],
        "limitations": [
            "Changing velocity changes stored kinetic energy; the atlas is not a free-energy comparison.",
            "The fixed pivot has no torso, pelvis, moving shoulder center, or three-dimensional motion.",
            "The local force field does not identify how a player creates or regulates the state.",
            "Pointwise power does not establish accumulated work or impact speed.",
            "No measured human trajectory or bilateral grip wrench is fitted in this tier.",
        ],
        "source_sha256": _source_hashes(),
        "array_artifact": NPZ_PATH.name,
    }
    return record, arrays


def write_outputs(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Persist deterministic JSON and lossless NumPy evidence."""
    record, arrays = run_study()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_PATH.name
    npz_path = output_dir / NPZ_PATH.name
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cast(Any, np.savez_compressed)(npz_path, **arrays)
    return json_path, npz_path


def main() -> None:
    for path in write_outputs():
        print(path)


if __name__ == "__main__":
    main()
