"""Predeclared bounded-search supplements using the standard matrix runner.

These three protocols answer different bounded-search questions. They never
overwrite or reinterpret the original source-bound 324/72-cell matrices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .matched_matrix import matched_matrix_cells, run_matched_matrix


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
        common_targets=targets,
        profiles=spec.profiles,
        selected_cell_names=supplement_cell_names(protocol),
        match_face_bounds_deg=FACE_BOUNDS_DEG,
        match_speed_factors=spec.speed_factors,
        supplement_protocol=spec.label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", choices=tuple(SUPPLEMENTS))
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--figures", choices=("none", "selected", "all"), default="selected"
    )
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
