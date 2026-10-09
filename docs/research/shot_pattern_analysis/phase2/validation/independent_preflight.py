"""Independent deterministic refinement/feasibility; no Monte Carlo population.

Run from the repository root with PYTHONPATH=. and the qualified Python runtime.
The 0.05 m gate is declared before execution and is never adjusted by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_physics import (
    MatchedShotPhysics,
    UnsupportedShot,
)
from src.tools.shot_pattern_analysis.nominal_matching import (
    MatchBounds,
    NominalMatchError,
    match_nominal_landing,
)

GATE_M = 0.05
SETTINGS = ((128, 0.01), (256, 0.01), (512, 0.01), (512, 0.02), (512, 0.005))


def endpoint(outcome):
    return [outcome.carry_x_m, outcome.carry_y_m]


def comparison(a, b):
    """Distinguish unsupported domain from numerical/status inconsistencies."""
    if a["status"] == b["status"] == "valid":
        change = math.dist(a["endpoint_m"], b["endpoint_m"])
        return {"status": "evaluated", "change_m": change, "pass": change <= GATE_M}
    if a["status"] == b["status"] == "unsupported" and a["reason"] == b["reason"]:
        return {"status": "same_unsupported_domain", "pass": None}
    return {"status": "unresolved", "pass": False}


def delivery(physics, config, twist, horizontal, vertical):
    try:
        shot = physics.simulate_delivery(
            0.0,
            0.0,
            config,
            shaft_rotation_deg=twist,
            contact_horizontal_m=horizontal,
            contact_vertical_m=vertical,
        ).outcome
        return {
            "status": "valid",
            "endpoint_m": endpoint(shot),
            "speed_mps": shot.ball_speed_mps,
            "spin_rpm": shot.spin_rpm,
        }
    except UnsupportedShot as exc:
        return {"status": "unsupported", "reason": exc.reason}
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return {"status": "numerical_error", "error": f"{type(exc).__name__}: {exc}"}


def refinement(physics):
    rows = []
    for club, lean, twist, h, v in itertools.product(
        ("driver", "seven_iron", "pitching_wedge"),
        (0.0, 5.0, 10.0),
        (-4.0, 0.0, 4.0),
        (-0.0127, 0.0, 0.0127),
        (-0.0127, 0.0, 0.0127),
    ):
        config = replace(matched_preset(club), shaft_lean_deg=lean)
        runs = [
            delivery(
                physics,
                replace(config, steps_per_compression=steps, dt_s=dt),
                twist,
                h,
                v,
            )
            for steps, dt in SETTINGS
        ]
        rows.append(
            {
                "club": club,
                "lean_deg": lean,
                "twist_deg": twist,
                "horizontal_m": h,
                "vertical_m": v,
                "runs": runs,
                "comparisons": {
                    name: comparison(runs[a], runs[b])
                    for name, a, b in (
                        ("contact_coarse", 0, 1),
                        ("contact_fine", 1, 2),
                        ("contact_primary_to_fine", 0, 2),
                        ("flight_coarse", 3, 2),
                        ("flight_fine", 2, 4),
                    )
                },
            }
        )
    return rows


def feasibility(physics):
    rows = []
    for club, lean, context, ftp in itertools.product(
        ("driver", "seven_iron", "pitching_wedge"),
        (0.0, 5.0, 10.0),
        ("within_lean", "across_lean"),
        (0.0, -1.5, 1.5, -3.0, 3.0),
    ):
        config = replace(matched_preset(club), shaft_lean_deg=lean)
        target_config = (
            config if context == "within_lean" else replace(config, shaft_lean_deg=0.0)
        )
        row = {
            "club": club,
            "lean_deg": lean,
            "context": context,
            "face_to_path_deg": ftp,
        }
        try:
            target = physics.simulate(
                face_deg=0.0, path_deg=0.0, config=target_config
            ).carry_x_m
            reference = -ftp
            bounds = MatchBounds(
                reference - 3.0,
                reference + 3.0,
                config.club_speed_mps * 0.75,
                config.club_speed_mps * 1.25,
            )
            row.update(target_x_m=target, bounds=vars(bounds))
            matched = match_nominal_landing(
                physics,
                config=config,
                target_x_m=target,
                reference_face_deg=reference,
                face_to_path_deg=ftp,
                bounds=bounds,
                tolerance_m=GATE_M,
            )
            fine = physics.simulate(
                face_deg=matched.face_deg,
                path_deg=matched.path_deg,
                config=replace(
                    config,
                    club_speed_mps=matched.speed_mps,
                    steps_per_compression=512,
                    dt_s=0.005,
                ),
            )
            fine_residual = math.dist(endpoint(fine), [target, 0.0])
            row.update(
                status="matched",
                face_deg=matched.face_deg,
                path_deg=matched.path_deg,
                speed_mps=matched.speed_mps,
                optimizer_residual_m=matched.residual_m,
                fine_replay_residual_m=fine_residual,
                numerical_endpoint_change_m=math.dist(
                    endpoint(matched.outcome), endpoint(fine)
                ),
                fine_replay_pass=fine_residual <= GATE_M,
                n_evaluations=matched.n_evaluations,
            )
        except UnsupportedShot as exc:
            row.update(status="unsupported", reason=exc.reason)
        except NominalMatchError as exc:
            row.update(status="bounded_unreachable", reason=str(exc))
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            row.update(status="numerical_error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = matched_source_snapshot()
    physics = MatchedShotPhysics()
    result = {
        "gate_m": GATE_M,
        "settings_steps_dt": SETTINGS,
        "scope": "Deterministic geometry grid; not probability-weighted or empirical qualification",
        "execution": before,
        "python_executable": sys.executable,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "refinement": refinement(physics),
        "feasibility": feasibility(physics),
    }
    result["source_unchanged"] = before == matched_source_snapshot()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["source_unchanged"]:
        raise RuntimeError(
            "Scientific sources changed during independent preflight; receipt invalid"
        )
    sys.stdout.write(
        f"Saved {len(result['refinement'])} refinement and {len(result['feasibility'])} feasibility cases"
    )


if __name__ == "__main__":
    main()
