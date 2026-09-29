"""Generate the registered pendulum force-source optimization artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for dependency_path in (
    REPO_ROOT / "src",
    REPO_ROOT / "src/shared/python",
    REPO_ROOT / "vendor/ud-tools/src/shared/python",
    REPO_ROOT / "vendor/ud-tools/src",
    REPO_ROOT / "vendor/ud-tools/src/python/src",
):
    resolved = str(dependency_path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

from scripts.research.proximal_distal_energy.force_source_optimization import (
    ForceSourceCandidate,
    evaluate_candidate,
    summarize_optimization,
)
from src.shared.python.biomechanics.force_source_attribution import (
    REQUIRED_FORCE_ATTRIBUTION_SCHEMA,
)
from src.shared.python.simulation_backends import GolfModelParams

OUTPUT_PATH = (
    REPO_ROOT
    / "docs/research/proximal_distal_energy_transfer/data/force_source_optimization.json"
)
SHOULDER_TORQUES_NM = (60.0, 80.0, 100.0)
WRIST_DRIVE_NM = 15.0
WRIST_RESTRAIN_NM = (0.0, 5.0, 10.0)
ONSET_TIMES_S = tuple(round(index * 0.025, 3) for index in range(15))


def registered_candidates() -> tuple[ForceSourceCandidate, ...]:
    """Return the frozen 135-cell open-loop search grid."""
    return tuple(
        ForceSourceCandidate(shoulder, WRIST_DRIVE_NM, restrain, onset)
        for shoulder in SHOULDER_TORQUES_NM
        for restrain in WRIST_RESTRAIN_NM
        for onset in ONSET_TIMES_S
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def build_artifact() -> dict[str, object]:
    """Evaluate the frozen grid and return its complete machine record."""
    params = GolfModelParams.default()
    candidates = registered_candidates()
    outcomes = tuple(evaluate_candidate(params, candidate) for candidate in candidates)
    return {
        "schema_version": "force-source-optimization/v1",
        "force_attribution_schema": REQUIRED_FORCE_ATTRIBUTION_SCHEMA,
        "provenance": {
            "git_sha": _git_sha(),
            "model": "GolfModelParams.default()",
            "backend": "ode",
            "coordinates": ["shoulder_absolute", "wrist_relative"],
            "endpoint": "wrist_hand_path",
            "mapping": "force_only_virtual_work_least_squares",
            "impact_rule": "first_club_vertical_crossing_in_registered_delivery_zone",
            "primary_objective": "absolute_coriolis_tangent_impulse_through_impact",
            "directional_companion": "signed_coriolis_tangent_impulse_through_impact",
            "search_type": "complete_registered_grid_not_continuous_optimal_control",
            "shoulder_torques_nm": list(SHOULDER_TORQUES_NM),
            "wrist_drive_nm": WRIST_DRIVE_NM,
            "wrist_restrain_nm": list(WRIST_RESTRAIN_NM),
            "onset_times_s": list(ONSET_TIMES_S),
        },
        "summary": summarize_optimization(outcomes),
        "outcomes": [asdict(outcome) for outcome in outcomes],
        "interpretation_limits": [
            "The Coriolis versus squared-speed split depends on the declared coordinates.",
            "The wrist endpoint is rank deficient for a two-coordinate force-only map; residual generalized couple is retained.",
            "Generalized-drive attribution is not a biological muscle-force decomposition.",
            "A grid-selected maximum is not a player prescription or a continuous optimum.",
        ],
    }


def main() -> None:
    """Write the deterministic JSON artifact."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
