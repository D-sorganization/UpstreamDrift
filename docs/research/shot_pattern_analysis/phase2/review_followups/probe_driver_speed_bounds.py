"""Separately authorized driver speed-bound diagnostic; no population execution."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_physics import MatchedShotPhysics
from src.tools.shot_pattern_analysis.nominal_matching import (
    MatchBounds,
    match_nominal_landing,
)


def main() -> None:
    before = matched_source_snapshot()
    physics = MatchedShotPhysics()
    reference = matched_preset("driver")
    config = replace(reference, shaft_lean_deg=5.0)
    target = physics.simulate(face_deg=0, path_deg=0, config=reference).carry_x_m
    bounds = MatchBounds(-6.0, 6.0, 33.75, 58.5)
    rows = []
    for ftp in (-3.0, 3.0):
        result = match_nominal_landing(
            physics,
            config=config,
            target_x_m=target,
            reference_face_deg=-ftp,
            face_to_path_deg=ftp,
            bounds=bounds,
            tolerance_m=0.05,
        )
        fine = physics.simulate(
            face_deg=result.face_deg,
            path_deg=result.path_deg,
            config=replace(
                config,
                club_speed_mps=result.speed_mps,
                steps_per_compression=512,
                dt_s=0.005,
            ),
        )
        rows.append(
            {
                "face_to_path_deg": ftp,
                "face_deg": result.face_deg,
                "path_deg": result.path_deg,
                "speed_mps": result.speed_mps,
                "residual_m": result.residual_m,
                "fine_replay_residual_m": math.hypot(
                    fine.carry_x_m - target, fine.carry_y_m
                ),
                "n_evaluations": result.n_evaluations,
            }
        )
    assert before == matched_source_snapshot()
    output = {
        "scope": "Deterministic diagnostic only; not Monte Carlo, acceptance, or empirical qualification",
        "target_x_m": target,
        "bounds": vars(bounds),
        "tolerance_m": 0.05,
        "protocol": "separate_driver_upper_speed_1.30_sensitivity",
        "club": "driver",
        "lean_deg": 5.0,
        "rows": rows,
        "execution": before,
        "source_unchanged": True,
        "python_executable": sys.executable,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
