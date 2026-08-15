"""Run a finite jointly work- and load-matched proximal-rate screen.

The screen is deliberately a model falsification exercise, not a causal estimate.
Programs may differ in several actuator settings.  Matching limits observable work
and interface-load differences; it does not make proximal rate an isolated input.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.shoulder_velocity_strategy_search import (
    ShoulderVelocityProgram,
    evaluate_programs,
)
from src.shared.python.simulation_backends import GolfModelParams

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
JSON_PATH = DATA_DIR / "joint_matched_proximal_rate_study.json"

COMMON_RELEASE_S = 0.14
SHOULDER_BEFORE_NM = (35.0, 45.0, 55.0, 65.0, 75.0, 85.0)
SHOULDER_AFTER_NM = (0.0, 20.0, 40.0, 60.0)
WRIST_RESTRAIN_NM = (5.0, 10.0, 15.0)
WRIST_DRIVE_NM = (10.0, 20.0, 30.0)
MINIMUM_RATE_SEPARATION_RAD_S = 1.5
PRIMARY_WORK_TOLERANCE = 0.05
PRIMARY_LOAD_TOLERANCE = 0.10
WORK_TOLERANCES = (0.025, 0.05, 0.075)
LOAD_TOLERANCES = (0.05, 0.10, 0.15)


def _programs() -> tuple[ShoulderVelocityProgram, ...]:
    return tuple(
        ShoulderVelocityProgram(
            COMMON_RELEASE_S,
            shoulder_before,
            shoulder_after,
            COMMON_RELEASE_S,
            wrist_restrain,
            wrist_drive,
        )
        for shoulder_before, shoulder_after, wrist_restrain, wrist_drive in product(
            SHOULDER_BEFORE_NM,
            SHOULDER_AFTER_NM,
            WRIST_RESTRAIN_NM,
            WRIST_DRIVE_NM,
        )
    )


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(0.5 * (abs(left) + abs(right)), 1e-12)


def _outcome_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, outcome in enumerate(
        evaluate_programs(_programs(), GolfModelParams.default())
    ):
        row = asdict(outcome)
        row["program_index"] = index
        row.update(asdict(outcome.program))
        del row["program"]
        rows.append(row)
    return rows


def _candidate_pairs(
    valid_rows: list[dict[str, Any]], work_tolerance: float, load_tolerance: float
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for offset, left in enumerate(valid_rows):
        for right in valid_rows[offset + 1 :]:
            low, high = sorted(
                (left, right),
                key=lambda row: row["proximal_velocity_at_release_rad_s"],
            )
            rate_separation = (
                high["proximal_velocity_at_release_rad_s"]
                - low["proximal_velocity_at_release_rad_s"]
            )
            net_work_difference = _relative_difference(
                low["total_actuator_work_j"], high["total_actuator_work_j"]
            )
            positive_work_difference = _relative_difference(
                low["positive_actuator_work_j"], high["positive_actuator_work_j"]
            )
            force_difference = _relative_difference(
                low["peak_grip_force_n"], high["peak_grip_force_n"]
            )
            if (
                rate_separation < MINIMUM_RATE_SEPARATION_RAD_S
                or net_work_difference > work_tolerance
                or positive_work_difference > work_tolerance
                or force_difference > load_tolerance
            ):
                continue
            candidates.append(
                {
                    "lower_rate_program_index": low["program_index"],
                    "higher_rate_program_index": high["program_index"],
                    "lower_release_rate_rad_s": low[
                        "proximal_velocity_at_release_rad_s"
                    ],
                    "higher_release_rate_rad_s": high[
                        "proximal_velocity_at_release_rad_s"
                    ],
                    "release_rate_separation_rad_s": rate_separation,
                    "relative_net_work_difference": net_work_difference,
                    "relative_positive_work_difference": positive_work_difference,
                    "relative_peak_force_difference": force_difference,
                    "lower_impact_speed_m_s": low["impact_speed_m_s"],
                    "higher_impact_speed_m_s": high["impact_speed_m_s"],
                    "impact_speed_difference_higher_minus_lower_m_s": (
                        high["impact_speed_m_s"] - low["impact_speed_m_s"]
                    ),
                }
            )
    return candidates


def _independent_pairs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedily select disjoint pairs using a frozen, deterministic ordering."""
    ordered = sorted(
        candidates,
        key=lambda pair: (
            pair["relative_net_work_difference"]
            + pair["relative_positive_work_difference"]
            + pair["relative_peak_force_difference"],
            -pair["release_rate_separation_rad_s"],
            pair["lower_rate_program_index"],
            pair["higher_rate_program_index"],
        ),
    )
    used: set[int] = set()
    selected: list[dict[str, Any]] = []
    for pair in ordered:
        indices = {
            pair["lower_rate_program_index"],
            pair["higher_rate_program_index"],
        }
        if used.isdisjoint(indices):
            selected.append(pair)
            used.update(indices)
    return selected


def _match_record(
    valid_rows: list[dict[str, Any]], work_tolerance: float, load_tolerance: float
) -> dict[str, Any]:
    candidates = _candidate_pairs(valid_rows, work_tolerance, load_tolerance)
    pairs = _independent_pairs(candidates)
    differences = np.asarray(
        [pair["impact_speed_difference_higher_minus_lower_m_s"] for pair in pairs]
    )
    return {
        "work_tolerance": work_tolerance,
        "load_tolerance": load_tolerance,
        "minimum_release_rate_separation_rad_s": MINIMUM_RATE_SEPARATION_RAD_S,
        "maximum_relative_net_work_difference": work_tolerance,
        "maximum_relative_positive_work_difference": work_tolerance,
        "maximum_relative_peak_force_difference": load_tolerance,
        "candidate_pair_count": len(candidates),
        "independent_pair_count": len(pairs),
        "higher_rate_faster_pair_count": int(np.sum(differences > 0.0)),
        "higher_rate_slower_pair_count": int(np.sum(differences < 0.0)),
        "equal_speed_pair_count": int(np.sum(differences == 0.0)),
        "impact_speed_difference_range_m_s": (
            [float(np.min(differences)), float(np.max(differences))]
            if differences.size
            else []
        ),
        "mean_impact_speed_difference_m_s": (
            float(np.mean(differences)) if differences.size else None
        ),
        "median_impact_speed_difference_m_s": (
            float(np.median(differences)) if differences.size else None
        ),
        "selection_rule": (
            "Greedy disjoint matching ordered by summed relative net-work, "
            "positive-work, and peak-force mismatch; then larger rate separation "
            "and program indices. This is not globally optimal matching."
        ),
        "causal_estimand": False,
        "pairs": pairs,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def run_study() -> dict[str, Any]:
    """Evaluate the finite factorial and return the complete evidence record."""
    rows = _outcome_rows()
    valid_rows = [row for row in rows if row["valid_impact"]]
    primary = _match_record(valid_rows, PRIMARY_WORK_TOLERANCE, PRIMARY_LOAD_TOLERANCE)
    sensitivity = [
        {
            key: value
            for key, value in _match_record(
                valid_rows, work_tolerance, load_tolerance
            ).items()
            if key != "pairs"
        }
        for work_tolerance in WORK_TOLERANCES
        for load_tolerance in LOAD_TOLERANCES
    ]
    return {
        "schema_version": "joint-matched-proximal-rate/v1",
        "study_id": "fixed-release-joint-work-load-rate-screen",
        "model_tier": "exact_planar_double_pendulum_fixed_hub",
        "registration": {
            "common_shoulder_cut_and_wrist_release_s": COMMON_RELEASE_S,
            "factorial_grid": {
                "shoulder_torque_before_nm": list(SHOULDER_BEFORE_NM),
                "shoulder_torque_after_nm": list(SHOULDER_AFTER_NM),
                "wrist_restrain_nm": list(WRIST_RESTRAIN_NM),
                "wrist_drive_nm": list(WRIST_DRIVE_NM),
            },
            "primary_work_tolerance": PRIMARY_WORK_TOLERANCE,
            "primary_load_tolerance": PRIMARY_LOAD_TOLERANCE,
            "minimum_release_rate_separation_rad_s": (MINIMUM_RATE_SEPARATION_RAD_S),
        },
        "attempted_program_count": len(rows),
        "valid_impact_count": len(valid_rows),
        "programs": rows,
        "primary_match": primary,
        "tolerance_sensitivity": sensitivity,
        "conclusion": "mixed_nonmonotonic_model_response",
        "interpretation": (
            "At common release time and within declared actuator-work and peak "
            "interface-force tolerances, higher proximal rate is associated with "
            "both faster and slower delivery. The finite planar grid therefore "
            "rejects a monotonic maximize-proximal-rate rule."
        ),
        "analysis_boundary": (
            "The paired programs differ in actuator commands, the greedy matching "
            "is not a globally optimal assignment, and conditioning on valid impact "
            "can bias results. The screen is a synthetic model counterexample, not "
            "an isolated rate intervention, human causal estimate, or coaching rule."
        ),
        "falsification_tests": [
            "A monotonic benefit is rejected if any admissible higher-rate member is slower.",
            "A robust direction is unsupported if the sign mixture changes across tolerances.",
            "Transfer to humans fails without participant-held-out bilateral-wrench replication.",
        ],
        "source_sha256": {
            path: _sha256(REPO_ROOT / path)
            for path in (
                "scripts/research/proximal_distal_energy/run_joint_matched_proximal_rate_study.py",
                "scripts/research/proximal_distal_energy/shoulder_velocity_strategy_search.py",
                "src/shared/python/simulation_backends/model_params.py",
                "src/shared/python/simulation_backends/ode_backend.py",
            )
        },
    }


def write_outputs(output_dir: Path = DATA_DIR) -> Path:
    """Write the deterministic JSON evidence artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / JSON_PATH.name
    path.write_text(json.dumps(run_study(), indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """Write the registered evidence artifact."""
    write_outputs()


if __name__ == "__main__":
    main()
