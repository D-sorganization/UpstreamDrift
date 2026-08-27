"""Execution and reproduction runner for #9125."""

from __future__ import annotations

import argparse
import json
import sys

from scripts.research.proximal_distal_energy.event_topology_robustness import (
    run_event_topology_suite,
)


def run_qualification() -> dict[str, object]:
    summary = run_event_topology_suite()
    evidence = {
        "status": (
            "PASSED"
            if summary.zero_perturbation_reproduces_nominal
            and summary.step_refinement_stable
            and summary.channel_coverage_passed
            and summary.noise_robustness_retained_unique_fraction >= 0.75
            else "FAILED"
        ),
        "zero_perturbation_reproduces_nominal": summary.zero_perturbation_reproduces_nominal,
        "nominal_first_crossing_time_s": summary.nominal_first_crossing_time_s,
        "max_tolerated_delay_s": summary.max_tolerated_delay_s,
        "noise_robustness_retained_unique_fraction": summary.noise_robustness_retained_unique_fraction,
        "channel_coverage_passed": summary.channel_coverage_passed,
        "step_refinement_stable": summary.step_refinement_stable,
        "total_trials": summary.total_trials,
        "inference_boundary": summary.inference_boundary,
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run #9125 event topology robustness mapping."
    )
    parser.add_argument(
        "action", nargs="?", default="validate", choices=["validate", "report"]
    )
    args = parser.parse_args()

    evidence = run_qualification()
    print(json.dumps(evidence, indent=2))

    if evidence["status"] != "PASSED":
        return 1
    if not evidence["zero_perturbation_reproduces_nominal"]:
        return 1
    if not evidence["step_refinement_stable"]:
        return 1
    if evidence["noise_robustness_retained_unique_fraction"] < 0.75:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
