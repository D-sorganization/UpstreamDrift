"""Execution and reproduction runner for #9124."""

from __future__ import annotations

import argparse
import json
import sys

from scripts.research.proximal_distal_energy.bounded_event_reachability import (
    run_bounded_reachability_suite,
)


def run_qualification() -> dict[str, object]:
    summary = run_bounded_reachability_suite()
    evidence = {
        "status": (
            "PASSED"
            if summary.small_amplitude_max_discrepancy < 0.20
            and summary.finite_amplitude_saturation_detected
            else "FAILED"
        ),
        "small_amplitude_max_discrepancy": summary.small_amplitude_max_discrepancy,
        "finite_amplitude_saturation_detected": summary.finite_amplitude_saturation_detected,
        "zero_authority_delta_norm": summary.zero_authority_delta_norm,
        "shoulder_only_feasible_count": summary.shoulder_only_feasible_count,
        "wrist_only_feasible_count": summary.wrist_only_feasible_count,
        "both_channels_feasible_count": summary.both_channels_feasible_count,
        "total_trials": summary.total_trials,
        "inference_boundary": summary.inference_boundary,
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run #9124 bounded reachability qualification."
    )
    parser.add_argument(
        "action", nargs="?", default="validate", choices=["validate", "report"]
    )
    args = parser.parse_args()

    evidence = run_qualification()
    print(json.dumps(evidence, indent=2))

    if evidence["status"] != "PASSED":
        return 1
    if not evidence["finite_amplitude_saturation_detected"]:
        return 1
    if evidence["small_amplitude_max_discrepancy"] > 0.20:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
