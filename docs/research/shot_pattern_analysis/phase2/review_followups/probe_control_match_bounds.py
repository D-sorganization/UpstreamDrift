"""Nominal-only fixed-target search diagnostics with unchanged speed limits."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from src.tools.shot_pattern_analysis.matched_config import MatchedConfig, matched_preset
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_physics import MatchedShotPhysics
from src.tools.shot_pattern_analysis.nominal_matching import (
    MatchBounds,
    NominalMatchError,
    match_nominal_landing,
)


def _diagnostic_multistart(
    physics: MatchedShotPhysics,
    config: MatchedConfig,
    reference: MatchedConfig,
    bounds: MatchBounds,
    ftp: float,
    target: float,
) -> list[dict[str, Any]]:
    """Probe the same bounded solver starts through the public physics adapter."""

    # Diagnostic solver exposes active bounds; all physics uses the public adapter.
    def residual(candidate, ftp=ftp, config=config, target=target):
        face, speed = candidate
        shot = physics.simulate(
            face_deg=float(face),
            path_deg=float(face - ftp),
            config=replace(config, club_speed_mps=float(speed)),
        )
        return np.array([shot.carry_x_m - target, shot.carry_y_m])

    diagnostics = []
    for initial_face in (-5.0, 0.0, 5.0):
        fit = least_squares(
            residual,
            [initial_face, reference.club_speed_mps],
            bounds=([-6, bounds.speed_min_mps], [6, bounds.speed_max_mps]),
            max_nfev=80,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        diagnostics.append(
            {
                "success": bool(fit.success),
                "face_deg": float(fit.x[0]),
                "speed_mps": float(fit.x[1]),
                "residual_xy_m": residual(fit.x).tolist(),
                "active_mask": fit.active_mask.tolist(),
                "optimality": float(fit.optimality),
                "n_evaluations": int(fit.nfev),
            }
        )
    return diagnostics


def main() -> None:
    before = matched_source_snapshot()
    physics = MatchedShotPhysics()
    rows = []
    for club in ("driver", "pitching_wedge"):
        reference = matched_preset(club)
        config = replace(reference, shaft_lean_deg=5.0)
        target = physics.simulate(
            face_deg=0.0, path_deg=0.0, config=reference
        ).carry_x_m
        bounds = MatchBounds(
            -6, 6, reference.club_speed_mps * 0.75, reference.club_speed_mps * 1.25
        )
        for ftp in (-3.0, 3.0):
            row: dict[str, Any] = {
                "club": club,
                "lean_deg": 5,
                "face_to_path_deg": ftp,
                "target_x_m": target,
                "bounds": vars(bounds),
            }
            try:
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
                row.update(
                    status="matched",
                    face_deg=result.face_deg,
                    path_deg=result.path_deg,
                    speed_mps=result.speed_mps,
                    residual_m=result.residual_m,
                    fine_replay_residual_m=math.hypot(
                        fine.carry_x_m - target, fine.carry_y_m
                    ),
                )
            except NominalMatchError as exc:
                diagnostics = _diagnostic_multistart(
                    physics, config, reference, bounds, ftp, target
                )
                row.update(
                    status="bounded_failure",
                    error=str(exc),
                    diagnostic_multistart=diagnostics,
                )
            rows.append(row)
    assert before == matched_source_snapshot()
    receipt = {
        "scope": "Deterministic nominal diagnostic only; no Monte Carlo or empirical qualification",
        "speed_limits_unchanged": True,
        "tolerance_m": 0.05,
        "rows": rows,
        "execution": before,
        "source_unchanged": True,
        "python_executable": sys.executable,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
