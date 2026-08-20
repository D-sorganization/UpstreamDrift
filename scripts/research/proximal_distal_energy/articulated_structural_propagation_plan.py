"""Build the fail-closed structural-corner headline propagation plan."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
    SOURCE_PATHS as GROUND_SOURCE_PATHS,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)
from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
    SOURCE_PATHS as SHAFT_SOURCE_PATHS,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
CAMPAIGN = DATA / "articulated_structural_authority_campaign.json"
DEFAULT_OUTPUT = DATA / "articulated_structural_propagation_plan.json"
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/research/proximal_distal_energy/articulated_atlas_authority.py",
            "scripts/research/proximal_distal_energy/articulated_scaled_authority.py",
            "scripts/research/proximal_distal_energy/articulated_structural_authority_campaign.py",
            "scripts/research/proximal_distal_energy/articulated_structural_propagation_plan.py",
            "tests/research/test_articulated_structural_propagation_plan.py",
            *SHAFT_SOURCE_PATHS,
            *GROUND_SOURCE_PATHS,
        )
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scientific_configuration(config: Any) -> dict[str, Any]:
    """Exclude operational parallelism from the scientific design bind."""

    configuration = asdict(config)
    configuration.pop("worker_count")
    return configuration


def _planned_failures(
    authority: ArticulatedAtlasAuthority,
    states: tuple[tuple[int, int], ...],
) -> list[dict[str, int | str]]:
    requested = set(states)
    return [
        dict(failure)
        for failure in authority.selected_failures()
        if (int(failure["case_index"]), int(failure["phase_index"])) in requested
    ]


def _corner_plan(
    row: dict[str, Any],
    shaft: ArticulatedShaftAtlasConfig,
    ground: ArticulatedGroundAtlasConfig,
    data_directory: Path,
) -> dict[str, Any]:
    scaled = load_scaled_authority(
        data_directory / row["record_artifact"],
        data_directory / row["array_artifact"],
    )
    authority = ArticulatedAtlasAuthority.from_scaled(scaled)
    if authority.authority_sha256 != row["authority_sha256"]:
        raise RuntimeError("campaign and authority artifact digests do not match")
    if shaft.case_indices != ground.case_indices or (
        shaft.sample_indices != ground.sample_indices
    ):
        raise ValueError("shaft and ground atlases must use the same registered states")
    states = tuple(
        (case, phase) for case in shaft.case_indices for phase in shaft.sample_indices
    )
    feasible = authority.feasible_states(shaft.case_indices, shaft.sample_indices)
    failures = _planned_failures(authority, states)
    if len(feasible) + len(failures) != len(states):
        raise RuntimeError("every requested state must be feasible or retained failed")
    shaft_headline_cells = (
        len(feasible) * 2 * len(shaft.forward.time_steps_s) * 2 * len(shaft.horizons_s)
    )
    ground_headline_cells = (
        len(feasible)
        * 2
        * len(ground.forward.time_steps_s)
        * 2
        * len(ground.horizons_s)
    )
    return {
        "corner_id": row["corner_id"],
        "status": "ready_with_retained_failure" if failures else "ready",
        "requested_state_count": len(states),
        "feasible_state_count": len(feasible),
        "retained_failures": failures,
        "feasible_states": [list(state) for state in feasible],
        "expected_shaft_trajectory_count": len(feasible)
        * len(shaft.activations)
        * 2
        * len(shaft.forward.time_steps_s)
        * 2,
        "expected_ground_trajectory_count": len(feasible)
        * (len(ground.ground_activations) + len(ground.control_names))
        * 2
        * len(ground.forward.time_steps_s)
        * 2,
        "expected_shaft_headline_cell_count": shaft_headline_cells,
        "expected_ground_headline_cell_count": ground_headline_cells,
        "authority": authority.provenance_record(),
    }


def build_structural_propagation_plan(
    campaign_path: Path = CAMPAIGN,
    *,
    data_directory: Path = DATA,
) -> dict[str, Any]:
    """Bind every registered structural authority to both headline designs."""

    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("status") != "complete" or len(campaign.get("corners", [])) != 7:
        raise RuntimeError("the seven-corner authority campaign must be complete")
    shaft = ArticulatedShaftAtlasConfig()
    ground = ArticulatedGroundAtlasConfig()
    corners = [
        _corner_plan(row, shaft, ground, data_directory) for row in campaign["corners"]
    ]
    design = json.loads(
        json.dumps(
            {
                "case_indices": list(shaft.case_indices),
                "phase_indices": list(shaft.sample_indices),
                "shaft_configuration": _scientific_configuration(shaft),
                "ground_configuration": _scientific_configuration(ground),
                "parallelism": "worker_count is operational and excluded from the scientific design digest",
            }
        )
    )
    design_sha = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "articulated-structural-propagation-plan/v1",
        "status": "ready",
        "authority_campaign_sha256": _sha256(campaign_path),
        "design_sha256": design_sha,
        "design": design,
        "corners": corners,
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCE_PATHS},
        "limitations": {
            "scope": "engineering OAT corners, not a participant distribution",
            "evidence": "execution plan only; no headline result is implied",
            "human_inference": "none",
        },
    }


def validate_structural_propagation_plan(
    plan_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Reject a committed plan that differs from current governed inputs."""

    observed = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = build_structural_propagation_plan()
    if observed != expected:
        raise RuntimeError("committed structural propagation plan is stale or altered")
    return observed


def write_structural_propagation_plan(
    plan_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Atomically replace the governed propagation plan."""

    record = build_structural_propagation_plan()
    temporary = plan_path.with_suffix(plan_path.suffix + ".tmp")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(plan_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("write", "validate"), nargs="?", default="write"
    )
    args = parser.parse_args()
    if args.command == "validate":
        validate_structural_propagation_plan()
    else:
        write_structural_propagation_plan()


if __name__ == "__main__":
    main()
