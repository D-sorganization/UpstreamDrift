"""Execution and reproduction runner for #9123."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    compute_trajectory_authority,
    generate_nominal_downswing_trajectory,
)


def run_qualification() -> dict[str, object]:
    """Execute the full trajectory control authority qualification."""
    states, controls = generate_nominal_downswing_trajectory(dt=0.002, steps=140)
    result = compute_trajectory_authority(states, controls, dt=0.002)

    evidence = {
        "status": (
            "PASSED"
            if result.is_transverse and result.tangent_rank_both == 3
            else "FAILED"
        ),
        "step_count": result.step_count,
        "dt_s": result.dt_s,
        "is_transverse": result.is_transverse,
        "transverse_inner_product": result.transverse_inner_product,
        "full_rank_both": result.full_rank_both,
        "full_rank_shoulder": result.full_rank_shoulder,
        "full_rank_wrist": result.full_rank_wrist,
        "full_rank_zero": result.full_rank_zero,
        "tangent_rank_both": result.tangent_rank_both,
        "tangent_rank_shoulder": result.tangent_rank_shoulder,
        "tangent_rank_wrist": result.tangent_rank_wrist,
        "tangent_rank_zero": result.tangent_rank_zero,
        "additivity_residual_norm": result.additivity_residual_norm,
        "pulse_agreement_relative_error": result.pulse_agreement_relative_error,
        "inference_boundary": result.inference_boundary,
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Run #9123 authority qualification.")
    parser.add_argument(
        "action", nargs="?", default="validate", choices=["validate", "report"]
    )
    args = parser.parse_args()

    evidence = run_qualification()
    print(json.dumps(evidence, indent=2))

    if evidence["status"] != "PASSED":
        return 1
    if evidence["additivity_residual_norm"] > 1e-10:
        return 1
    if evidence["full_rank_zero"] != 0 or evidence["tangent_rank_zero"] != 0:
        return 1
    if evidence["tangent_rank_both"] != 3:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
