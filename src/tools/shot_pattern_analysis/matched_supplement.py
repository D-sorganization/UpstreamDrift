"""Predeclared bounded-search supplements using the standard matrix runner.

These three protocols answer different bounded-search questions. They never
overwrite or reinterpret the original source-bound 324/72-cell matrices.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .matched_matrix import MatrixSelection, matched_matrix_cells, run_matched_matrix
from .matrix_cli import add_matrix_arguments


@dataclass(frozen=True)
class SupplementSpec:
    """One reviewed cell-selection and nominal-search protocol."""

    label: str
    club_id: str
    lean_deg: float
    profiles: tuple[str, ...] | None
    target_x_m: float | None
    speed_factors: tuple[float, float]


FACE_BOUNDS_DEG = (-6.0, 6.0)
SUPPLEMENTS = {
    "pw_primary_face": SupplementSpec(
        "pw-primary-face-bounds/1",
        "pitching_wedge",
        0.0,
        None,
        None,
        (0.75, 1.25),
    ),
    "pw_control_face": SupplementSpec(
        "pw-control-face-bounds/1",
        "pitching_wedge",
        5.0,
        ("shaft_only", "combined2"),
        77.40247724866299,
        (0.75, 1.25),
    ),
    "driver_control_speed": SupplementSpec(
        "driver-control-speed-bounds/1",
        "driver",
        5.0,
        ("shaft_only", "combined2"),
        178.94875782337786,
        (0.75, 1.30),
    ),
}


def supplement_cell_names(protocol: str) -> tuple[str, ...]:
    """Select only reviewed cells from the canonical 324-cell naming plan."""
    try:
        spec = SUPPLEMENTS[protocol]
    except KeyError as exc:
        raise ValueError(f"unknown supplement protocol: {protocol}") from exc
    return tuple(
        name
        for name, request in matched_matrix_cells(n_shots=32, profiles=spec.profiles)
        if request.club.club_id == spec.club_id
        and request.club.shaft_lean_deg == spec.lean_deg
        and request.curve_scale == 2.0
    )


def pw_control_alias_candidates() -> dict[str, str]:
    """Declare four possible PW0 references, without claiming proven equivalence.

    A separate source-bound review must compare target, delivery inputs, noise,
    nominals, scoring context, and outputs before using any alias in results.
    """
    return {
        name: name
        for name in supplement_cell_names("pw_primary_face")
        if name.endswith(("_shaft_only", "_combined2"))
    }


def expected_supplement_final_cells() -> dict[str, dict[str, dict]]:
    """Build the exact reviewed 26-cell authorization catalog from one source."""
    catalog = {}
    for protocol, spec in SUPPLEMENTS.items():
        targets = None if spec.target_x_m is None else {spec.club_id: spec.target_x_m}
        cells = matched_matrix_cells(
            n_shots=10000,
            common_targets=targets,
            profiles=spec.profiles,
            selected_cell_names=supplement_cell_names(protocol),
            match_face_bounds_deg=FACE_BOUNDS_DEG,
            match_speed_factors=spec.speed_factors,
        )
        catalog[spec.label] = {name: asdict(request) for name, request in cells}
    return json.loads(json.dumps(catalog))


def validate_supplement_acceptance(
    path: Path | None, protocol: str, execution: dict, requested_cells: dict[str, dict]
) -> None:
    """Bind a final supplement to all reviewed cells and source-matched pilots.

    The generic final-stage validator checks independent-review flags and all
    evidence hashes before this protocol-specific authorization is consulted.
    """
    if path is None:
        raise ValueError("supplement final stage requires an acceptance receipt")
    receipt_path = Path(path)
    receipt = json.loads(receipt_path.read_text())
    catalog = expected_supplement_final_cells()
    if protocol not in catalog:
        raise ValueError(f"supplement protocol was not reviewed: {protocol}")
    if json.loads(json.dumps(requested_cells)) != catalog[protocol]:
        raise ValueError("supplement requested cells or bounds were not reviewed")
    groups = receipt.get("supplement_protocols")
    if (
        receipt.get("supplement_stage") != "final"
        or not isinstance(groups, dict)
        or set(groups) != set(catalog)
        or receipt.get("execution") != execution
    ):
        raise ValueError("supplement acceptance stage, catalog, or source differs")
    evidence = receipt.get("evidence_files", {})
    for protocol_id, final_cells in catalog.items():
        group = groups[protocol_id]
        if (
            not isinstance(group, dict)
            or set(group) != {"final_cells", "pilot_plan", "pilot_plan_sha256"}
            or group["final_cells"] != final_cells
        ):
            raise ValueError(f"supplement final cell inventory differs: {protocol_id}")
        pilot_relative = group["pilot_plan"]
        expected_sha = group["pilot_plan_sha256"]
        if not isinstance(pilot_relative, str) or not isinstance(expected_sha, str):
            raise ValueError("supplement pilot plan reference is invalid")
        pilot_path = (receipt_path.parent / pilot_relative).resolve()
        if not pilot_path.is_relative_to(receipt_path.parent.resolve()):
            raise ValueError("supplement pilot plan escapes receipt directory")
        if (
            evidence.get(pilot_relative) != expected_sha
            or not pilot_path.is_file()
            or hashlib.sha256(pilot_path.read_bytes()).hexdigest() != expected_sha
        ):
            raise ValueError(f"supplement pilot plan hash differs: {protocol_id}")
        pilot = json.loads(pilot_path.read_text())
        expected_pilot_cells = {
            name: {**config, "n_shots": 32} for name, config in final_cells.items()
        }
        if (
            pilot.get("stage") != "pilot"
            or pilot.get("supplement_protocol") != protocol_id
            or pilot.get("execution") != execution
            or pilot.get("selected_cell_names") != list(final_cells)
            or pilot.get("cells") != expected_pilot_cells
        ):
            raise ValueError(f"supplement pilot plan does not match: {protocol_id}")


def run_supplement_matrix(
    protocol: str,
    output_root: Path,
    *,
    n_shots: int = 32,
    stage: str = "pilot",
    acceptance: Path | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    figures: str = "selected",
) -> list[dict]:
    """Delegate execution, immutable resume, and source checks to matrix runner."""
    if protocol not in SUPPLEMENTS:
        raise ValueError(f"unknown supplement protocol: {protocol}")
    spec = SUPPLEMENTS[protocol]
    targets = None if spec.target_x_m is None else {spec.club_id: spec.target_x_m}
    return run_matched_matrix(
        output_root,
        n_shots=n_shots,
        stage=stage,
        acceptance=acceptance,
        shard_index=shard_index,
        shard_count=shard_count,
        figures=figures,
        selection=MatrixSelection(
            common_targets=targets,
            profiles=spec.profiles,
            selected_cell_names=supplement_cell_names(protocol),
            match_face_bounds_deg=FACE_BOUNDS_DEG,
            match_speed_factors=spec.speed_factors,
            supplement_protocol=spec.label,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the existing command-line interface without starting a study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", choices=tuple(SUPPLEMENTS))
    add_matrix_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_supplement_matrix(
        args.protocol,
        args.output,
        n_shots=args.shots,
        stage=args.stage,
        acceptance=args.acceptance,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        figures=args.figures,
    )


if __name__ == "__main__":
    main()
