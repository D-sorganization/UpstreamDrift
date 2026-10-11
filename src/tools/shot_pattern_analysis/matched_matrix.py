"""Pilot-first matched study inventory, immutable resume, and final-run gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .matched_config import matched_preset
from .matched_export import execute_and_export, matched_source_snapshot
from .matched_runner import ExperimentConfig
from .matrix_cli import add_matrix_arguments
from .noise import NoiseSpec
from .provenance import assert_source_unchanged, manifest_files_match

LOGGER = logging.getLogger(__name__)
PROFILES = (
    "shaft_only",
    "shaft_path",
    "shaft_speed1",
    "shaft_speed2",
    "shaft_strike",
    "combined1",
    "combined2",
    "cov_facepath_positive",
    "cov_facepath_negative",
)


def _selected_profiles(profiles: tuple[str, ...] | None) -> tuple[str, ...]:
    """Preserve catalog ordering while rejecting empty, duplicate, unknown input."""
    if profiles is None:
        return PROFILES
    if (
        not profiles
        or len(profiles) != len(set(profiles))
        or any(profile not in PROFILES for profile in profiles)
    ):
        raise ValueError("profiles must be nonempty, distinct known profile names")
    return tuple(profile for profile in PROFILES if profile in profiles)


def noise_profile(name: str, rotation_sd_deg: float) -> NoiseSpec:
    """Explicit Gaussian sensitivities; correlations are assumptions, not fits."""
    if name not in PROFILES:
        raise ValueError(f"unknown noise profile: {name}")
    path = (
        0.5
        if name
        in (
            "shaft_path",
            "combined1",
            "combined2",
            "cov_facepath_positive",
            "cov_facepath_negative",
        )
        else 0.0
    )
    speed = (
        2 * 0.44704
        if name in ("shaft_speed2", "combined2")
        else 0.44704
        if name
        in (
            "shaft_speed1",
            "combined1",
            "cov_facepath_positive",
            "cov_facepath_negative",
        )
        else 0.0
    )
    strike = (
        0.00635
        if name
        in (
            "shaft_strike",
            "combined1",
            "combined2",
            "cov_facepath_positive",
            "cov_facepath_negative",
        )
        else 0.0
    )
    corr = None
    if name.startswith("cov_"):
        # A constructive factor guarantees PSD; +shaft twist closes the face.
        factor = np.eye(7)
        factor[1, 0] = -0.6 if name == "cov_facepath_positive" else 0.6
        factor[1, 1] = 0.8
        for row, col, rho in (
            (2, 0, 0.2),
            (3, 1, 0.3),
            (4, 2, 0.25),
            (5, 0, 0.2),
            (6, 5, 0.2),
        ):
            factor[row, col] = rho
            factor[row, row] = float(np.sqrt(1 - rho * rho))
        matrix = factor @ factor.T
        corr = tuple(tuple(float(v) for v in row) for row in matrix)
    return NoiseSpec((rotation_sd_deg, path, speed, strike, strike, 0.0, 0.0), corr)


def matched_matrix_cells(
    *,
    n_shots: int = 32,
    seed: int = 20261009,
    common_targets: dict[str, float] | None = None,
    profiles: tuple[str, ...] | None = None,
    selected_cell_names: tuple[str, ...] | None = None,
    match_face_bounds_deg: tuple[float, float] | None = None,
    match_speed_factors: tuple[float, float] = (0.75, 1.25),
) -> tuple[tuple[str, ExperimentConfig], ...]:
    """Return requested factorial cells; default contains all 324 candidates."""
    selected_profiles = _selected_profiles(profiles)
    if selected_cell_names is not None and (
        not selected_cell_names
        or any(not isinstance(name, str) for name in selected_cell_names)
        or len(selected_cell_names) != len(set(selected_cell_names))
    ):
        raise ValueError("selected cell names must be nonempty, unique plan members")
    requested = set(selected_cell_names) if selected_cell_names is not None else None
    available: set[str] = set()
    cells = []
    for club_id in ("driver", "seven_iron", "pitching_wedge"):
        for lean in (0.0, 5.0, 10.0):
            club = replace(matched_preset(club_id), shaft_lean_deg=lean)
            for curve in (1.0, 2.0):
                for sd in (1.0, 2.0):
                    for profile in selected_profiles:
                        key = f"{club_id}_lean{lean:g}_curve{curve:g}_rotation{sd:g}_{profile}"
                        available.add(key)
                        if requested is not None and key not in requested:
                            continue
                        target = (
                            None if common_targets is None else common_targets[club_id]
                        )
                        cells.append(
                            (
                                key,
                                ExperimentConfig(
                                    club,
                                    noise_profile(profile, sd),
                                    n_shots=n_shots,
                                    seed=seed,
                                    curve_scale=curve,
                                    target_x_m=target,
                                    match_face_bounds_deg=match_face_bounds_deg,
                                    match_speed_factors=match_speed_factors,
                                ),
                            )
                        )
    if requested is not None and not requested <= available:
        raise ValueError("selected cell names must be nonempty, unique plan members")
    return tuple(cells)


def validate_final_acceptance(path: Path | None, execution: dict) -> None:
    """Require reviewed, source-bound evidence before a final matrix job."""
    if path is None:
        raise ValueError("final matrix requires an acceptance receipt")
    receipt = json.loads(Path(path).read_text())
    checks = receipt.get("checks", {})
    required = (
        "pilot",
        "numerical_refinement",
        "independent_review",
        "rendered_figures",
        "software_tests",
    )
    if receipt.get("status") != "accepted_for_conditional_simulation" or not all(
        checks.get(k) is True for k in required
    ):
        raise ValueError("acceptance receipt does not establish every required gate")
    assert_source_unchanged(receipt.get("execution", {}), execution)
    evidence = receipt.get("evidence_files", {})
    if not isinstance(evidence, dict) or len(evidence) < 5:
        raise ValueError(
            "acceptance requires hashed review, pilot, numerical, visual and test evidence"
        )
    for name, expected in evidence.items():
        file = Path(path).parent / name
        if (
            not file.is_file()
            or hashlib.sha256(file.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"acceptance evidence missing or changed: {name}")


def run_matched_matrix(
    output_root: Path,
    *,
    n_shots: int = 32,
    stage: str = "pilot",
    acceptance: Path | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    figures: str = "selected",
    common_targets: dict[str, float] | None = None,
    profiles: tuple[str, ...] | None = None,
    selected_cell_names: tuple[str, ...] | None = None,
    match_face_bounds_deg: tuple[float, float] | None = None,
    match_speed_factors: tuple[float, float] = (0.75, 1.25),
    supplement_protocol: str | None = None,
) -> list[dict]:
    """Publish immutable cells; resume verifies data, inputs, and execution sources."""
    if stage not in ("pilot", "final") or figures not in ("none", "selected", "all"):
        raise ValueError("invalid matrix stage or figure selection")
    if not isinstance(shard_count, int) or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be inside a positive shard count")
    if supplement_protocol is not None and (
        not supplement_protocol.strip() or selected_cell_names is None
    ):
        raise ValueError("supplement protocol requires selected cell names")
    if (
        stage == "final"
        and supplement_protocol is None
        and (
            selected_cell_names is not None
            or match_face_bounds_deg is not None
            or match_speed_factors != (0.75, 1.25)
        )
    ):
        raise ValueError(
            "final custom bounds or cell selection require a supplement protocol"
        )
    execution = matched_source_snapshot()
    if stage == "final":
        if n_shots != 10000:
            raise ValueError("the final report matrix uses10000 shots perpattern")
        validate_final_acceptance(acceptance, execution)
    elif n_shots > 512:
        raise ValueError(
            "pilot runs are limited to512 shots perpattern; use accepted final stage"
        )
    cells = matched_matrix_cells(
        n_shots=n_shots,
        common_targets=common_targets,
        profiles=profiles,
        selected_cell_names=selected_cell_names,
        match_face_bounds_deg=match_face_bounds_deg,
        match_speed_factors=match_speed_factors,
    )
    if stage == "final" and supplement_protocol is not None:
        from .matched_supplement import validate_supplement_acceptance

        validate_supplement_acceptance(
            acceptance,
            supplement_protocol,
            execution,
            {name: asdict(request) for name, request in cells},
        )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "matched-shot-matrix/1",
        "stage": stage,
        "execution": execution,
        "cells": {name: asdict(request) for name, request in cells},
    }
    if supplement_protocol is not None:
        plan["supplement_protocol"] = supplement_protocol
        plan["selected_cell_names"] = [name for name, _ in cells]
    plan = json.loads(json.dumps(plan))
    plan_path = output_root / "matrix_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("existing matrix plan differs; use a new output root")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for index, (name, request) in enumerate(cells):
        if index % shard_count != shard_index:
            continue
        assert_source_unchanged(execution, matched_source_snapshot())
        destination = output_root / name
        manifest = destination / "manifest.json"
        if destination.exists():
            if not manifest_files_match(manifest):
                raise RuntimeError(
                    f"incomplete or modified cell requires inspection: {name}"
                )
            start = json.loads((destination / "run_start.json").read_text())
            if start["config"] != json.loads(json.dumps(asdict(request))):
                raise RuntimeError(f"cell input mismatch: {name}")
            assert_source_unchanged(start["execution"], execution)
            LOGGER.info("Verified existing cell %s", name)
        else:
            make_figures = figures == "all" or (
                figures == "selected"
                and name.endswith("rotation1_combined2")
                and request.curve_scale == 2
            )
            LOGGER.info("[%d/%d] %s", index + 1, len(cells), name)
            execute_and_export(request, destination, figures=make_figures)
        assert_source_unchanged(execution, matched_source_snapshot())
        completed_start = json.loads((destination / "run_start.json").read_text())
        assert_source_unchanged(execution, completed_start["execution"])
        summary = json.loads((destination / "summary.json").read_text())
        results.append(
            {
                "cell": name,
                "status": summary["status"],
                "reason": summary.get("reason"),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    assert_source_unchanged(execution, matched_source_snapshot())
    (output_root / f"shard_{shard_index}_receipt.json").write_text(
        json.dumps(
            {"shard_count": shard_count, "shard_index": shard_index, "cells": results},
            indent=2,
        )
        + "\n"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the existing command-line interface without starting a study."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_matrix_arguments(parser)
    parser.add_argument(
        "--common-targets",
        type=Path,
        help="JSON club-to-target-distance mapping for across-lean control",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        help="optional profile subset; default includes all nine",
    )
    parser.add_argument(
        "--face-bounds-deg", nargs=2, type=float, metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--speed-factors",
        nargs=2,
        type=float,
        default=(0.75, 1.25),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--cells-file", type=Path, help="JSON list of selected planned cell names"
    )
    parser.add_argument(
        "--supplement-protocol",
        help="versioned protocol label for a selected supplement",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    targets = (
        json.loads(args.common_targets.read_text()) if args.common_targets else None
    )
    selection = json.loads(args.cells_file.read_text()) if args.cells_file else None
    if selection is not None and not isinstance(selection, list):
        parser.error("--cells-file must contain a JSON list")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_matched_matrix(
        args.output,
        n_shots=args.shots,
        stage=args.stage,
        acceptance=args.acceptance,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        figures=args.figures,
        common_targets=targets,
        profiles=tuple(args.profiles) if args.profiles else None,
        selected_cell_names=tuple(selection) if selection is not None else None,
        match_face_bounds_deg=tuple(args.face_bounds_deg)
        if args.face_bounds_deg
        else None,
        match_speed_factors=tuple(args.speed_factors),
        supplement_protocol=args.supplement_protocol,
    )


if __name__ == "__main__":
    main()
