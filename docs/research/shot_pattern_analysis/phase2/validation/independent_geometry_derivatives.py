"""Analytical geometry derivatives and fixed-delivery local carry sensitivity."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

from src.tools.shot_pattern_analysis.club_geometry import club_delivery
from src.tools.shot_pattern_analysis.matched_config import MatchedConfig, matched_preset
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_physics import MatchedShotPhysics


def _geometry_checks() -> list[dict[str, Any]]:
    """Evaluate the original ordered analytic/central-difference grid."""
    geometry: list[dict[str, Any]] = []
    for club, lean, face in itertools.product(
        ("driver", "seven_iron", "pitching_wedge"), (0.0, 5.0, 10.0), (-3.0, 0.0, 3.0)
    ):
        config = matched_preset(club)
        loft, beta, ell = map(math.radians, (config.loft_deg, config.lie_deg, lean))
        predicted_face = -math.sin(beta) * math.cos(loft) / math.cos(loft - ell)
        predicted_loft = -math.cos(beta)
        ratios = []
        for epsilon in (1e-4, 1e-5):
            plus = club_delivery(config.loft_deg, config.lie_deg, lean, face, epsilon)
            minus = club_delivery(config.loft_deg, config.lie_deg, lean, face, -epsilon)
            ratios.append(
                {
                    "epsilon_deg": epsilon,
                    "face_per_twist": (plus.face_deg - minus.face_deg) / (2 * epsilon),
                    "loft_per_twist": (plus.loft_deg - minus.loft_deg) / (2 * epsilon),
                }
            )
        geometry.append(
            {
                "club": club,
                "lean_deg": lean,
                "nominal_face_deg": face,
                "analytical_face_per_twist": predicted_face,
                "analytical_loft_per_twist": predicted_loft,
                "analytical_loft_per_face": predicted_loft / predicted_face,
                "finite_differences": ratios,
            }
        )
    return geometry


def _carry_sensitivities(pilot: Path) -> tuple[list[dict], dict[str, str]]:
    """Replay fixed saved nominals in the original dt/epsilon evaluation order."""
    physics = MatchedShotPhysics()
    carry = []
    sources = {}
    for club, lean, curve in itertools.product(
        ("driver", "seven_iron", "pitching_wedge"), (0, 5, 10), (1, 2)
    ):
        key = f"{club}_lean{lean}_curve{curve}_rotation1_shaft_only"
        path = pilot / key / "summary.json"
        data = json.loads(path.read_text())
        sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        if data["status"] != "completed":
            carry.append(
                {"cell": key, "status": "unavailable", "reason": data["reason"]}
            )
            continue
        config = MatchedConfig(**data["config"]["club"])
        for pattern, nominal in data["nominals"].items():
            fixed = replace(config, club_speed_mps=nominal["speed_mps"])
            derivatives = []
            for dt, epsilon in itertools.product((0.02, 0.005), (0.01, 0.005)):
                plus_shot = physics.simulate_delivery(
                    nominal["face_deg"],
                    nominal["path_deg"],
                    replace(fixed, dt_s=dt),
                    shaft_rotation_deg=epsilon,
                ).outcome
                minus_shot = physics.simulate_delivery(
                    nominal["face_deg"],
                    nominal["path_deg"],
                    replace(fixed, dt_s=dt),
                    shaft_rotation_deg=-epsilon,
                ).outcome
                derivatives.append(
                    {
                        "epsilon_deg": epsilon,
                        "flight_dt_s": dt,
                        "dx_m_per_twist_deg": (
                            plus_shot.carry_x_m - minus_shot.carry_x_m
                        )
                        / (2 * epsilon),
                        "dy_m_per_twist_deg": (
                            plus_shot.carry_y_m - minus_shot.carry_y_m
                        )
                        / (2 * epsilon),
                    }
                )
            carry.append(
                {
                    "cell": key,
                    "pattern": pattern,
                    "status": "evaluated",
                    "nominal": nominal,
                    "derivatives": derivatives,
                }
            )
    return carry, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = matched_source_snapshot()
    geometry = _geometry_checks()
    carry, sources = _carry_sensitivities(args.pilot)
    maximum = max(
        abs(check[field] - row["analytical_" + field])
        for row in geometry
        for check in row["finite_differences"]
        for field in ("face_per_twist", "loft_per_twist")
    )
    result = {
        "scope": "Local deterministic derivatives at zero twist; fixed nominal speed/path; no empirical uncertainty interpretation",
        "angle_units": "Public angle ratios deg/deg; carry derivatives m/degree of shaft twist",
        "geometry_tolerance": 1e-8,
        "geometry_max_error": maximum,
        "geometry_pass": maximum <= 1e-8,
        "execution": before,
        "source_unchanged": before == matched_source_snapshot(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "saved_nominal_summary_sha256": sources,
        "geometry": geometry,
        "carry": carry,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["source_unchanged"] or not result["geometry_pass"]:
        raise RuntimeError("Derivative verification failed or source changed")
    sys.stdout.write(
        f"Geometry max error {maximum:.3g}; saved {len(carry)} carry records\n"
    )


if __name__ == "__main__":
    main()
