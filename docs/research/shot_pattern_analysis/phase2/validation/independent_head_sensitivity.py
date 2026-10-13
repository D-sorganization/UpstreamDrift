"""Fixed-delivery head-assumption sensitivity, not an uncertainty interval."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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


def variants(config):
    yield "nominal", config
    for scale in (0.8, 1.2):
        yield (
            f"cg_depth_x{scale}",
            replace(config, cg_depth_m=config.cg_depth_m * scale),
        )
        yield (
            f"inertia_x{scale}",
            replace(
                config,
                inertia_diagonal_kg_m2=tuple(
                    v * scale for v in config.inertia_diagonal_kg_m2
                ),
            ),
        )
        if config.club_id == "driver":
            yield (
                f"curvature_radii_x{scale}",
                replace(
                    config,
                    bulge_radius_m=config.bulge_radius_m * scale,
                    roll_radius_m=config.roll_radius_m * scale,
                ),
            )
    yield (
        "newton_qp_comparator",
        replace(config, contact_law="maximum_dissipation_newton"),
    )


def simulate(physics, config, horizontal, vertical):
    try:
        result = physics.simulate_delivery(
            0.0,
            0.0,
            config,
            contact_horizontal_m=horizontal,
            contact_vertical_m=vertical,
        )
        return {
            "status": "valid",
            "outcome": asdict(result.outcome),
            "contact_face_deg": result.contact_face_deg,
            "contact_loft_deg": result.contact_loft_deg,
        }
    except UnsupportedShot as exc:
        return {"status": "unsupported", "reason": exc.reason}
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {"status": "numerical_error", "error": f"{type(exc).__name__}: {exc}"}


def compare(base, other):
    if base["status"] != "valid" or other["status"] != "valid":
        return {
            "status": "not_comparable",
            "base_status": base["status"],
            "variant_status": other["status"],
        }
    a, b = base["outcome"], other["outcome"]
    delta = {
        key: b[key] - a[key]
        for key in (
            "carry_x_m",
            "carry_y_m",
            "spin_rpm",
            "launch_elevation_deg",
            "ball_speed_mps",
        )
    }
    delta["endpoint_displacement_m"] = math.hypot(
        delta["carry_x_m"], delta["carry_y_m"]
    )
    return {"status": "compared", "delta": delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = matched_source_snapshot()
    physics = MatchedShotPhysics()
    rows = []
    strikes = (
        ("center", 0.0, 0.0),
        ("toe", 0.0127, 0.0),
        ("heel", -0.0127, 0.0),
        ("high", 0.0, 0.0127),
        ("low", 0.0, -0.0127),
    )
    for club, lean, (strike, h, v) in itertools.product(
        ("driver", "seven_iron", "pitching_wedge"), (0.0, 5.0, 10.0), strikes
    ):
        config = replace(matched_preset(club), shaft_lean_deg=lean)
        variants_data = [
            {"variant": name, "config": asdict(cfg), **simulate(physics, cfg, h, v)}
            for name, cfg in variants(config)
        ]
        base = variants_data[0]
        for item in variants_data:
            item["comparison_to_nominal"] = compare(base, item)
        rows.append(
            {
                "club": club,
                "lean_deg": lean,
                "strike": strike,
                "horizontal_m": h,
                "vertical_m": v,
                "variants": variants_data,
            }
        )
    result = {
        "scope": "Deterministic fixed-delivery sensitivity; not statistical or empirical uncertainty",
        "design": "One factor at a time; face=path=shaft twist=0, no rematching; preset speeds/AoA",
        "restitution_comparison": "Same numeric 0.83 means energetic primary versus Newton velocity coefficient comparator",
        "execution": before,
        "source_unchanged": before == matched_source_snapshot(),
        "python_executable": sys.executable,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["source_unchanged"]:
        raise RuntimeError("Scientific sources changed during sensitivity execution")
    sys.stdout.write(f"Saved {len(rows)} delivery cases\n")


if __name__ == "__main__":
    main()
