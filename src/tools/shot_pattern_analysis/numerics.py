"""Reproducible time-step refinement for corrected shot-pattern physics."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from .core import AnalysisConfig, PhysicsProtocol
from .physics import ShotPhysics
from .sensitivity import preset_configs, runtime_provenance, scientific_source_hashes

CRITERION_M = 0.05
TIMESTEPS_S = (0.02, 0.01, 0.005)
NOMINAL_PATTERNS = ((0.0, 0.0), (1.5, 3.0), (-1.5, -3.0), (3.0, 6.0), (-3.0, -6.0))


def build_refinement(
    configs: Mapping[str, AnalysisConfig], *, engine: PhysicsProtocol | None = None
) -> dict:
    """Compare landing coordinates at three fixed steps against a 5 cm gate.

    Uses both delivery modes, five nominal patterns, and three face errors.
    All comparisons concern discretization of this model, not its empirical
    accuracy. No claim of monotonic convergence or formal order is made;
    ground-contact interpolation can vary with crossing phase.
    """
    if not configs or any(
        not isinstance(value, AnalysisConfig) for value in configs.values()
    ):
        raise ValueError("at least one valid AnalysisConfig is required")
    solver = engine if engine is not None else ShotPhysics()
    cases = []
    for club, base in configs.items():
        for mode in ("fixed_loft", "shaft_rotation"):
            for nominal, path in NOMINAL_PATTERNS:
                for error in (-3.0, 0.0, 3.0):
                    outcomes = [
                        solver.simulate(
                            face_deg=nominal + error,
                            path_deg=path,
                            nominal_face_deg=nominal,
                            config=replace(base, delivery_mode=mode, dt_s=dt),
                        )
                        for dt in TIMESTEPS_S
                    ]
                    differences = [
                        math.hypot(a.carry_x_m - b.carry_x_m, a.carry_y_m - b.carry_y_m)
                        for a, b in zip(outcomes[:-1], outcomes[1:], strict=True)
                    ]
                    if not all(math.isfinite(value) for value in differences):
                        raise ValueError(
                            "landing refinement produced nonfinite differences"
                        )
                    cases.append(
                        {
                            "club_id": club,
                            "delivery_mode": mode,
                            "nominal_face_deg": nominal,
                            "path_deg": path,
                            "face_error_deg": error,
                            "dt_020_vs_010_m": differences[0],
                            "dt_010_vs_005_m": differences[1],
                        }
                    )
    coarse = max(row["dt_020_vs_010_m"] for row in cases)
    fine = max(row["dt_010_vs_005_m"] for row in cases)
    return {
        "schema": "shot-pattern-timestep-refinement/1",
        "status": "passed"
        if math.isfinite(max(coarse, fine)) and max(coarse, fine) <= CRITERION_M
        else "failed",
        "criterion_m": CRITERION_M,
        "method": "Landing Euclidean refinement at .02/.01/.005 seconds; three clubs, two delivery modes, five nominal patterns, errors -3/0/+3 degrees",
        "timesteps_s": list(TIMESTEPS_S),
        "configs": {
            key: {
                "club_speed_mps": value.club_speed_mps,
                "loft_deg": value.loft_deg,
                "attack_angle_deg": value.attack_angle_deg,
                "lie_deg": value.lie_deg,
                "shaft_lean_deg": value.shaft_lean_deg,
                "clubhead_mass_kg": value.clubhead_mass_kg,
                "max_time_s": value.max_time_s,
            }
            for key, value in configs.items()
        },
        "cases": cases,
        "max_dt_020_vs_010_m": coarse,
        "max_dt_010_vs_005_m": fine,
    }


def write_refinement(output: Path) -> dict:
    """Save the fixed three-club acceptance grid with shared exact provenance."""
    hashes = scientific_source_hashes()
    result = build_refinement(preset_configs())
    if scientific_source_hashes() != hashes:
        raise RuntimeError("scientific source changed during refinement run; retry")
    result["provenance"] = runtime_provenance(
        hashes,
        "python3 -m src.tools.shot_pattern_analysis.numerics --output docs/research/shot_pattern_analysis/corrected_numerical_validation.json",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_refinement(args.output)
    if result["status"] != "passed":
        raise SystemExit("Numerical refinement exceeded the unchanged 0.05 m gate")


if __name__ == "__main__":
    main()
