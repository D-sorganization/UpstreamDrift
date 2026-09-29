"""Factorial planar screen of proximal acceleration, braking, and release timing."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from .run_experiments import DT, HORIZON, INITIAL_Q, rollout_controls
from .swing_model import PlanarInertials, find_impact
from src.shared.python.simulation_backends import GolfModelParams

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/timing_failure_mode_study.json"
)

PROXIMAL_ACCELERATION_ONSETS = (0.0, 0.04, 0.08)
PROXIMAL_BRAKING_ONSETS = (0.20, 0.26, 0.32)
DISTAL_RELEASE_ONSETS = (0.10, 0.18, 0.26)


def _first_crossing(
    time: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    indices = np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold))
    if indices.size == 0:
        return None
    index = int(indices[0])
    fraction = (threshold - values[index]) / (values[index + 1] - values[index])
    return float(time[index] + fraction * (time[index + 1] - time[index]))


def _controls(
    acceleration_onset: float, braking_onset: float, release_onset: float
) -> np.ndarray:
    time = np.arange(HORIZON) * DT
    controls = np.zeros((HORIZON, 2))
    controls[:, 0] = np.where(time < acceleration_onset, 20.0, 80.0)
    controls[:, 0] = np.where(time < braking_onset, controls[:, 0], 5.0)
    controls[:, 1] = np.where(time < release_onset, -5.0, 15.0)
    return controls


def main() -> None:
    params = GolfModelParams.default()
    inertials = PlanarInertials.from_params(params)
    rows = []
    for acceleration_onset, braking_onset, release_onset in product(
        PROXIMAL_ACCELERATION_ONSETS,
        PROXIMAL_BRAKING_ONSETS,
        DISTAL_RELEASE_ONSETS,
    ):
        controls = _controls(acceleration_onset, braking_onset, release_onset)
        time, q, velocity, _ = rollout_controls(params, controls)
        impact = find_impact(time, q, velocity, inertials)
        relative_rate = velocity[:, 1]
        angle_cast = _first_crossing(time, q[:, 1], -0.75)
        rate_cast = _first_crossing(time, relative_rate, 2.0)
        rows.append(
            {
                "proximal_acceleration_onset_s": acceleration_onset,
                "proximal_braking_onset_s": braking_onset,
                "distal_release_onset_s": release_onset,
                "angle_casting_event_s": angle_cast,
                "rate_casting_event_s": rate_cast,
                "casting_definitions_agree_within_20_ms": (
                    angle_cast is not None
                    and rate_cast is not None
                    and abs(angle_cast - rate_cast) <= 0.02
                ),
                "impact_status": "valid" if impact is not None else "invalid",
                "impact_time_s": None if impact is None else impact[0],
                "clubhead_speed_m_s": None if impact is None else impact[1],
                "initial_state": {"q_rad": list(INITIAL_Q), "v_rad_s": [0.0, 0.0]},
            }
        )
    valid = [row for row in rows if row["clubhead_speed_m_s"] is not None]
    fastest = max(valid, key=lambda row: row["clubhead_speed_m_s"])
    record = {
        "schema_version": "timing-failure-mode-study/v1",
        "study_id": "planar-proximal-acceleration-braking-release-factorial",
        "model_tier": "fixed_hub_double_pendulum",
        "registered_grid": {
            "proximal_acceleration_onset_s": list(PROXIMAL_ACCELERATION_ONSETS),
            "proximal_braking_onset_s": list(PROXIMAL_BRAKING_ONSETS),
            "distal_release_onset_s": list(DISTAL_RELEASE_ONSETS),
        },
        "case_count": len(rows),
        "valid_case_count": len(valid),
        "rows": rows,
        "fastest_valid_case": fastest,
        "casting_definitions": {
            "angle": "first relative wrist-angle crossing of -0.75 rad",
            "rate": "first relative wrist-rate crossing of +2.0 rad/s",
            "agreement_window_s": 0.02,
        },
        "claim_status": {
            "universal_casting_event": "unsupported",
            "universal_optimal_timing": "unsupported",
            "human_coaching_strategy": "untested",
        },
        "limitations": [
            "The factorial changes control work and does not yet provide a matched-work causal contrast.",
            "The fixed hub omits torso dynamics, articulated arms, bilateral contact, shaft flex, and impact collision.",
            "Casting thresholds are operational alternatives, not diagnoses of human technique.",
        ],
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
