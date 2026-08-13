"""Generate the trajectory-level proximal-speed strategy evidence bundle."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.shoulder_velocity_strategy_search import (
    ShoulderVelocityProgram,
    evaluate_programs,
    pareto_program_indices,
)
from scripts.research.proximal_distal_energy.run_experiments import DT
from src.shared.python.simulation_backends import GolfModelParams

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"
JSON_PATH = DATA_DIR / "shoulder_velocity_strategy_study.json"
NPZ_PATH = DATA_DIR / "shoulder_velocity_strategy_study.npz"

SHOULDER_CUTS_S = (0.12, 0.18, 0.24, 0.30)
SHOULDER_AFTER_NM = (0.0, 30.0, 60.0)
WRIST_RELEASES_S = (0.10, 0.14, 0.18, 0.22, 0.26)


def _programs() -> tuple[ShoulderVelocityProgram, ...]:
    return tuple(
        ShoulderVelocityProgram(cut, 60.0, after, release, 10.0, 15.0)
        for cut in SHOULDER_CUTS_S
        for after in SHOULDER_AFTER_NM
        for release in WRIST_RELEASES_S
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _standardized_regression(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "proximal_velocity_at_release_rad_s",
        "wrist_release_s",
        "shoulder_cut_s",
        "shoulder_torque_after_nm",
    )
    predictors = np.asarray([[row[key] for key in keys] for row in rows], dtype=float)
    outcome = np.asarray([row["impact_speed_m_s"] for row in rows], dtype=float)
    scale = np.std(predictors, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (predictors - np.mean(predictors, axis=0)) / scale
    outcome_scale = max(float(np.std(outcome)), 1e-12)
    standardized_outcome = (outcome - np.mean(outcome)) / outcome_scale
    design = np.column_stack((np.ones(len(rows)), standardized))
    coefficients, *_ = np.linalg.lstsq(design, standardized_outcome, rcond=None)
    return {
        key: float(value) for key, value in zip(keys, coefficients[1:], strict=True)
    }


def _standardized_regression_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = (
        "proximal_velocity_at_release_rad_s",
        "wrist_release_s",
        "shoulder_cut_s",
        "shoulder_torque_after_nm",
    )
    predictors = np.asarray([[row[key] for key in keys] for row in rows], dtype=float)
    scale = np.std(predictors, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (predictors - np.mean(predictors, axis=0)) / scale
    design = np.column_stack((np.ones(len(rows)), standardized))
    return {
        "observation_count": len(rows),
        "parameter_count_including_intercept": design.shape[1],
        "matrix_rank": int(np.linalg.matrix_rank(design)),
        "condition_number": float(np.linalg.cond(design)),
        "selection_rule": "valid_impact_only",
        "uncertainty_intervals_reported": False,
        "causal_estimand": False,
    }


def _outcome_rows() -> list[dict[str, Any]]:
    outcomes = evaluate_programs(_programs(), GolfModelParams.default())
    rows = []
    for index, outcome in enumerate(outcomes):
        row = asdict(outcome)
        row["program_index"] = index
        row.update(asdict(outcome.program))
        del row["program"]
        rows.append(row)
    return rows


def _pareto_indices(valid_rows: list[dict[str, Any]]) -> list[int]:
    objectives = np.asarray(
        [
            (
                row["impact_speed_m_s"],
                row["braking_grip_work_j"],
                row["peak_grip_force_n"],
            )
            for row in valid_rows
        ]
    )
    local = pareto_program_indices(objectives)
    return [valid_rows[index]["program_index"] for index in local]


def _association_record(valid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    velocity = np.asarray(
        [row["proximal_velocity_at_release_rad_s"] for row in valid_rows]
    )
    speed = np.asarray([row["impact_speed_m_s"] for row in valid_rows])
    braking = np.asarray([row["braking_grip_work_j"] for row in valid_rows])
    peak_force = np.asarray([row["peak_grip_force_n"] for row in valid_rows])
    return {
        "release_velocity_vs_impact_speed_pearson_r": _correlation(velocity, speed),
        "release_velocity_vs_braking_work_pearson_r": _correlation(velocity, braking),
        "release_velocity_vs_peak_force_pearson_r": _correlation(velocity, peak_force),
        "standardized_speed_regression": _standardized_regression(valid_rows),
        "standardized_speed_regression_diagnostics": (
            _standardized_regression_diagnostics(valid_rows)
        ),
    }


def _provenance() -> dict[str, str]:
    source_paths = (
        "scripts/research/proximal_distal_energy/double_pendulum_attribution.py",
        "scripts/research/proximal_distal_energy/interaction_forces.py",
        "scripts/research/proximal_distal_energy/run_experiments.py",
        "scripts/research/proximal_distal_energy/run_shoulder_velocity_strategy_study.py",
        "scripts/research/proximal_distal_energy/shoulder_velocity_strategy_search.py",
        "scripts/research/proximal_distal_energy/swing_model.py",
        "src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py",
        "src/shared/python/biomechanics/drift_control_transfer.py",
        "src/shared/python/simulation_backends/factory.py",
        "src/shared/python/simulation_backends/model_params.py",
        "src/shared/python/simulation_backends/ode_backend.py",
        "src/shared/python/simulation_backends/protocol.py",
    )
    return {path: _sha256(REPO_ROOT / path) for path in source_paths}


def _record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["valid_impact"]]
    pareto_indices = _pareto_indices(valid_rows)
    best_row = max(valid_rows, key=lambda row: row["impact_speed_m_s"])
    best_program = ShoulderVelocityProgram(
        best_row["shoulder_cut_s"],
        best_row["shoulder_torque_before_nm"],
        best_row["shoulder_torque_after_nm"],
        best_row["wrist_release_s"],
        best_row["wrist_restrain_nm"],
        best_row["wrist_drive_nm"],
    )
    refined = evaluate_programs(
        (best_program,), GolfModelParams.default(), dt_s=DT / 2.0
    )[0]
    return {
        "schema_version": "shoulder-velocity-strategy-evidence-v2",
        "study_id": "fixed-hub-proximal-drive-and-wrist-release-grid",
        "model_tier": "exact_planar_double_pendulum_fixed_hub",
        "program_count": len(rows),
        "valid_impact_count": len(valid_rows),
        "grid": {
            "shoulder_cut_s": list(SHOULDER_CUTS_S),
            "shoulder_torque_after_nm": list(SHOULDER_AFTER_NM),
            "wrist_release_s": list(WRIST_RELEASES_S),
            "shoulder_torque_before_nm": 60.0,
            "wrist_restrain_nm": 10.0,
            "wrist_drive_nm": 15.0,
        },
        "programs": rows,
        "pareto_program_indices": pareto_indices,
        "pareto_diagnostics": {
            "valid_program_count": len(valid_rows),
            "nondominated_program_count": len(pareto_indices),
            "nondominated_fraction": len(pareto_indices) / len(valid_rows),
            "screen_is_weakly_discriminating": bool(
                len(pareto_indices) / len(valid_rows) > 0.75
            ),
        },
        "associations_valid_impact_only": _association_record(valid_rows),
        "analysis_boundary": (
            "Grid associations are descriptive model associations, not causal human "
            "effects or a continuous optimal-control solution; conditioning on the "
            "valid-impact selection rule can bias the associations."
        ),
        "claim_status": {
            "proximal_link_velocity": "tested_model_coordinate",
            "anatomical_shoulder_velocity": "not_tested",
            "torso_rotation_strategy": "not_tested",
            "universal_coaching_strategy": "unsupported",
        },
        "registered_objectives": {
            "maximize": ["impact_speed_m_s"],
            "minimize": ["braking_grip_work_j", "peak_grip_force_n"],
        },
        "interface_force_boundary": (
            "peak_grip_force_n is the net modeled planar wrist-interface force on "
            "the club, not either hand's anatomical grip force."
        ),
        "fixed_program_timestep_sensitivity": {
            "program_index": best_row["program_index"],
            "reference_dt_s": DT,
            "refined_dt_s": DT / 2.0,
            "same_valid_impact_classification": bool(
                best_row["valid_impact"] == refined.valid_impact
            ),
            "absolute_impact_speed_difference_m_s": abs(
                best_row["impact_speed_m_s"] - refined.impact_speed_m_s
            ),
            "absolute_total_grip_work_difference_j": abs(
                best_row["total_grip_work_j"] - refined.total_grip_work_j
            ),
            "absolute_peak_interface_force_difference_n": abs(
                best_row["peak_grip_force_n"] - refined.peak_grip_force_n
            ),
            "selection_was_not_reoptimized": True,
        },
        "falsification_tests": [
            "A speed association that vanishes after timing and drive controls weakens the high-velocity interpretation.",
            "A candidate dominated on speed, braking work, and peak force is not an optimal strategy.",
            "A rotating-base two-hand model that reverses the result rejects transfer to torso strategy.",
        ],
        "source_sha256": _provenance(),
        "array_artifact": NPZ_PATH.name,
    }


def _arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    array_keys = (
        "impact_speed_m_s",
        "proximal_velocity_at_release_rad_s",
        "proximal_velocity_at_impact_rad_s",
        "total_grip_work_j",
        "drift_grip_work_j",
        "control_grip_work_j",
        "transfer_work_closure_residual_j",
        "braking_grip_work_j",
        "peak_grip_force_n",
    )
    result = {
        key: np.asarray([row[key] for row in rows], dtype=float) for key in array_keys
    }
    result["valid_impact"] = np.asarray(
        [row["valid_impact"] for row in rows], dtype=bool
    )
    result["shoulder_cut_s"] = np.asarray([row["shoulder_cut_s"] for row in rows])
    result["shoulder_torque_after_nm"] = np.asarray(
        [row["shoulder_torque_after_nm"] for row in rows]
    )
    result["wrist_release_s"] = np.asarray([row["wrist_release_s"] for row in rows])
    return result


def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the registered grid and return JSON- and NPZ-ready evidence."""
    rows = _outcome_rows()
    return _record(rows), _arrays(rows)


def write_outputs(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Write deterministic JSON and lossless NumPy evidence artifacts."""
    record, arrays = run_study()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_PATH.name
    npz_path = output_dir / NPZ_PATH.name
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path


def main() -> None:
    """Write deterministic JSON and NPZ evidence artifacts."""
    write_outputs()


if __name__ == "__main__":
    main()
