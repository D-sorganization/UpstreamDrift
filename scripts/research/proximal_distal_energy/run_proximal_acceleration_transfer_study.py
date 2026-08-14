"""Generate the phase-resolved proximal-acceleration intervention atlas."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.research.proximal_distal_energy.proximal_acceleration_transfer import (
    AccelerationSweepRequest,
    evaluate_acceleration_sweep,
)
from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.shoulder_velocity_transfer import (
    classify_transfer_phase,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
FIGURE_DIR = DATA_DIR / "proximal_acceleration_transfer/figures"
JSON_PATH = DATA_DIR / "proximal_acceleration_transfer_study.json"
NPZ_PATH = DATA_DIR / "proximal_acceleration_transfer_study.npz"
SCHEMA_VERSION = "proximal-acceleration-transfer-evidence-v1"
PHASE_FRACTIONS = (0.0, 0.15, 0.40, 0.70, 0.95)
MINIMUM_ACCELERATION_SCALE_RAD_S2 = 20.0
ACCELERATION_OFFSET_FRACTION = 0.75


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/research/proximal_distal_energy/interaction_forces.py",
        "scripts/research/proximal_distal_energy/proximal_acceleration_transfer.py",
        "scripts/research/proximal_distal_energy/run_experiments.py",
        "scripts/research/proximal_distal_energy/run_proximal_acceleration_transfer_study.py",
        "scripts/research/proximal_distal_energy/swing_model.py",
        "scripts/research/proximal_distal_energy/torque_programs.py",
        "src/shared/python/simulation_backends/model_params.py",
        "src/shared/python/simulation_backends/ode_backend.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


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
    backend = make_backend("ode", params)
    time, q, velocity, control, impact_time = _reference_trace()
    rows: list[dict[str, Any]] = []
    for fraction in PHASE_FRACTIONS:
        index = int(np.argmin(np.abs(time - fraction * impact_time)))
        reference_acceleration = float(
            backend.forward_dynamics(q[index], velocity[index], control[index])[0]
        )
        scale = max(abs(reference_acceleration), MINIMUM_ACCELERATION_SCALE_RAD_S2)
        targets = (
            reference_acceleration
            + np.linspace(
                -ACCELERATION_OFFSET_FRACTION,
                ACCELERATION_OFFSET_FRACTION,
                9,
            )
            * scale
        )
        request = AccelerationSweepRequest(
            q_rad=q[index],
            velocity_rad_s=velocity[index],
            reference_control_nm=control[index],
            proximal_acceleration_rad_s2=targets,
        )
        for sample in evaluate_acceleration_sweep(request, params):
            row = asdict(sample)
            row.update(
                {
                    "normalized_downswing_time": fraction,
                    "phase": classify_transfer_phase(fraction),
                    "reference_time_s": float(time[index]),
                    "reference_sample_index": index,
                    "reference_proximal_acceleration_rad_s2": reference_acceleration,
                    "acceleration_sweep_scale_rad_s2": scale,
                }
            )
            rows.append(row)
    return rows


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def _phase_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for phase in dict.fromkeys(row["phase"] for row in rows):
        selected = [row for row in rows if row["phase"] == phase]
        acceleration = _array(selected, "proximal_acceleration_rad_s2")
        grip_power = _array(selected, "total_grip_power_w")
        distal_acceleration = _array(selected, "total_club_angular_acceleration_rad_s2")
        summaries.append(
            {
                "phase": phase,
                "reference_proximal_acceleration_rad_s2": selected[0][
                    "reference_proximal_acceleration_rad_s2"
                ],
                "grip_power_slope_w_per_rad_s2": float(
                    np.polyfit(acceleration, grip_power, 1)[0]
                ),
                "club_acceleration_slope": float(
                    np.polyfit(acceleration, distal_acceleration, 1)[0]
                ),
                "grip_power_endpoint_delta_w": float(grip_power[-1] - grip_power[0]),
                "grip_power_monotonic_increasing": bool(
                    np.all(np.diff(grip_power) >= -1e-10)
                ),
                "minimum_grip_power_w": float(np.min(grip_power)),
                "maximum_peak_grip_force_n": float(
                    np.max(_array(selected, "peak_grip_force_n"))
                ),
            }
        )
    return summaries


def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return deterministic metadata and machine-readable arrays."""
    rows = _rows()
    array_keys = tuple(
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    arrays = {key: _array(rows, key) for key in array_keys}
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "pointwise-identical-state-proximal-acceleration-dose-v1",
        "model_tier": "exact_planar_double_pendulum",
        "case_count": len(rows),
        "phase_fractions": list(PHASE_FRACTIONS),
        "intervention_contract": {
            "held_fixed": [
                "configuration",
                "velocity",
                "model_parameters",
                "gravity",
                "distal_actuator_torque",
            ],
            "solved_intervention": "proximal actuator torque",
            "state_and_kinetic_energy_matched": True,
            "input_work_matched": False,
            "prior_reachability_matched": False,
        },
        "rows": rows,
        "phase_summaries": _phase_summaries(rows),
        "claim_status": {
            "pointwise_proximal_acceleration": "tested",
            "forward_acceleration_strategy": "untested",
            "human_proximal_acceleration": "untested",
            "universal_coaching_instruction": "unsupported",
        },
        "falsification_tests": [
            "Retain phase-dependent sign changes and adverse interface power.",
            "Require target, drift-control, and reaction-force closure.",
            "Reject inference from acceleration if required torque or load dominates.",
            "Repeat with forward work/load-matched controls and articulated spatial models.",
        ],
        "limitations": [
            "The intervention changes proximal actuator torque and instantaneous power.",
            "A pointwise acceleration target does not establish reachability or accumulated work.",
            "The proximal coordinate is a fixed-hub link, not an anatomical torso or shoulder.",
            "No measured human trajectory or bilateral grip wrench is fitted.",
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
