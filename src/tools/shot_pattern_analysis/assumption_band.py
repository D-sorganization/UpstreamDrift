"""Head-assumption bands: rerun selected Monte Carlo cells under perturbed heads.

The frozen study fixes illustrative clubhead properties. This module repeats a
saved cell's complete protocol (nominal re-matching, the same seeded paired
delivery errors, scoring and paired bootstrap) once per one-factor head
variant. The spread of the resulting paired strokes-gained intervals is a
model-assumption band. It is a sensitivity range over declared ±20% property
changes and an alternative contact law, not a probabilistic confidence
interval and not empirical validation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Final

from .core import PATTERN_NAMES
from .matched_config import MatchedConfig
from .matched_physics import UnsupportedShot
from .matched_runner import ExperimentConfig, run_matched_analysis
from .nominal_matching import NominalMatchError
from .provenance import publish_new_text
from .uncertainty import PUBLICATION_RESAMPLES

SCHEMA: Final = "shot-pattern-assumption-band/1"
HEAD_PROPERTY_SCALES: Final = (0.8, 1.2)
ALTERNATIVE_CONTACT_LAW: Final = "maximum_dissipation_newton"
CURVED_PATTERNS: Final = tuple(name for name in PATTERN_NAMES if name != "Straight")
NOMINAL_REPRODUCTION: Final = "nominal_reproduction"

logger = logging.getLogger(__name__)


def head_variants(club: MatchedConfig) -> tuple[tuple[str, MatchedConfig], ...]:
    """Return the declared one-factor head variants, excluding the nominal head.

    CG depth and all three principal inertias scale by 0.8 and 1.2; driver
    bulge and roll radii scale together; the last variant swaps the energetic
    restitution law for the Newton velocity-restitution comparator.
    """
    if not isinstance(club, MatchedConfig):
        raise TypeError("club must be a MatchedConfig")
    variants: list[tuple[str, MatchedConfig]] = []
    for scale in HEAD_PROPERTY_SCALES:
        variants.append(
            (f"cg_depth_x{scale:g}", replace(club, cg_depth_m=club.cg_depth_m * scale))
        )
        variants.append(
            (
                f"inertia_x{scale:g}",
                replace(
                    club,
                    inertia_diagonal_kg_m2=(
                        club.inertia_diagonal_kg_m2[0] * scale,
                        club.inertia_diagonal_kg_m2[1] * scale,
                        club.inertia_diagonal_kg_m2[2] * scale,
                    ),
                ),
            )
        )
        if club.club_id == "driver":
            variants.append(
                (
                    f"face_radii_x{scale:g}",
                    replace(
                        club,
                        bulge_radius_m=club.bulge_radius_m * scale,
                        roll_radius_m=club.roll_radius_m * scale,
                    ),
                )
            )
    variants.append(
        ("newton_contact_law", replace(club, contact_law=ALTERNATIVE_CONTACT_LAW))
    )
    return tuple(variants)


def run_head_variant(
    config: dict[str, Any],
    variant: str,
    *,
    resamples: int = PUBLICATION_RESAMPLES,
    analysis: Callable[..., Any] = run_matched_analysis,
) -> dict[str, Any]:
    """Repeat one saved cell's protocol under one named head variant.

    Returns a JSON-safe record. A variant whose nominal search or nominal
    flight leaves the model domain is recorded as unavailable, never dropped.
    """
    request = ExperimentConfig.from_dict(config)
    variants = dict(head_variants(request.club))
    if variant not in variants:
        raise ValueError(f"unknown head variant {variant!r}")
    varied = replace(request, club=variants[variant])
    record: dict[str, Any] = {
        "variant": variant,
        "config": json.loads(json.dumps(asdict(varied))),
        "resamples": resamples,
    }
    try:
        result = analysis(varied, include_scoring=True, resamples=resamples)
    except (NominalMatchError, UnsupportedShot) as exc:
        return {**record, "status": "unavailable", "reason": str(exc)}
    scoring = result.course_scoring
    return {
        **record,
        "status": scoring.get("status", "unavailable"),
        "target_x_m": result.target_x_m,
        "paired_n": scoring.get("paired_n"),
        "paired_benefit_vs_straight": scoring.get("paired_benefit_vs_straight"),
        "mean_radial_error_difference_vs_straight": result.paired_comparisons.get(
            "mean_radial_error_difference_vs_straight"
        ),
        "supported_by_pattern": {
            name: item["supported_n"] for name, item in result.summary.items()
        },
    }


def assumption_envelope(
    nominal: dict[str, dict[str, float]], variants: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Combine nominal and variant paired-SG intervals into one band per pattern.

    ``nominal`` maps Draw/Fade to an interval dict. The band spans every
    available estimate and the outer 95% bounds, so it contains sampling and
    declared head-assumption variation together. Unavailable variants are
    counted, not ignored.
    """
    runs = list(variants)
    available = [run for run in runs if run.get("paired_benefit_vs_straight")]
    band: dict[str, Any] = {
        "variants_attempted": len(runs),
        "variants_available": len(available),
        "unavailable_variants": sorted(
            run["variant"] for run in runs if run not in available
        ),
    }
    for pattern in CURVED_PATTERNS:
        intervals = [nominal[pattern]] + [
            run["paired_benefit_vs_straight"][pattern] for run in available
        ]
        for interval in intervals:
            values = (interval["lower_95"], interval["estimate"], interval["upper_95"])
            if not all(math.isfinite(v) for v in values) or list(values) != sorted(
                values
            ):
                raise ValueError("paired interval must be finite and ordered")
        estimates = [interval["estimate"] for interval in intervals]
        lower = min(interval["lower_95"] for interval in intervals)
        upper = max(interval["upper_95"] for interval in intervals)
        band[pattern] = {
            "nominal": nominal[pattern],
            "estimate_min": min(estimates),
            "estimate_max": max(estimates),
            "band_lower_95": lower,
            "band_upper_95": upper,
            "direction": (
                "positive" if lower > 0 else "negative" if upper < 0 else "spans_zero"
            ),
            "variant_estimates": {
                run["variant"]: run["paired_benefit_vs_straight"][pattern]["estimate"]
                for run in available
            },
        }
    return band


def nominal_reproduction(
    config: dict[str, Any],
    saved_shots: Path,
    *,
    analysis: Callable[..., Any] = run_matched_analysis,
) -> dict[str, Any]:
    """Rerun a saved cell's nominal head and compare every shot with its archive.

    Evidence that the code running the variants reproduces the frozen study:
    statuses must agree and supported endpoints must coincide to rounding.
    """
    request = ExperimentConfig.from_dict(config)
    result = analysis(request, include_scoring=False)
    with gzip.open(saved_shots, "rt", newline="") as handle:
        saved = {
            (row["pattern"], int(row["shot_index"])): row
            for row in csv.DictReader(handle)
        }
    if len(saved) != len(result.shots):
        raise ValueError("saved and rerun shot counts differ")
    mismatched, worst = 0, 0.0
    for shot in result.shots:
        row = saved[(shot.pattern, shot.shot_index)]
        if row["status"] != shot.status:
            mismatched += 1
        elif shot.status == "supported":
            worst = max(
                worst,
                abs(float(row["carry_x_m"]) - shot.carry_x_m),
                abs(float(row["carry_y_m"]) - shot.carry_y_m),
            )
    return {
        "variant": NOMINAL_REPRODUCTION,
        "shots_compared": len(result.shots),
        "status_mismatches": mismatched,
        "max_abs_endpoint_difference_m": worst,
        "target_x_m": result.target_x_m,
    }


Task = tuple[str, dict[str, Any], str, str, int, str | None]


def _run_one(task: Task) -> str:
    cell, config, variant, output, resamples, saved_shots = task
    if variant == NOMINAL_REPRODUCTION:
        if saved_shots is None:
            raise ValueError("nominal reproduction needs the saved shots path")
        result = nominal_reproduction(config, Path(saved_shots))
    else:
        result = run_head_variant(config, variant, resamples=resamples)
    record = {"cell": cell, **result}
    destination = Path(output) / f"{cell}__{variant}.json"
    publish_new_text(destination, json.dumps(record, indent=2, allow_nan=False) + "\n")
    return destination.name


def plan_tasks(
    cells: dict[str, dict[str, Any]],
    output_dir: Path,
    resamples: int,
    saved_shots: dict[str, Path] | None = None,
) -> list[Task]:
    """List every (cell, variant) run not already published in ``output_dir``.

    With ``saved_shots`` (cell to archived ``shots.csv.gz``) each cell also
    gets a nominal reproduction run.
    """
    tasks: list[Task] = []
    for cell, config in sorted(cells.items()):
        names = [
            name for name, _ in head_variants(ExperimentConfig.from_dict(config).club)
        ]
        shots = None if saved_shots is None else str(saved_shots[cell])
        if shots is not None:
            names.append(NOMINAL_REPRODUCTION)
        for variant in names:
            if not (output_dir / f"{cell}__{variant}.json").exists():
                tasks.append((cell, config, variant, str(output_dir), resamples, shots))
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path, help="JSON {cell: saved config}")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--resamples", type=int, default=PUBLICATION_RESAMPLES)
    parser.add_argument(
        "--saved-shots",
        type=Path,
        help="JSON {cell: archived shots.csv.gz} enabling nominal reproduction runs",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cells = json.loads(args.selection.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = None
    if args.saved_shots is not None:
        saved = {
            cell: Path(path)
            for cell, path in json.loads(args.saved_shots.read_text()).items()
        }
    tasks = plan_tasks(cells, args.output_dir, args.resamples, saved)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            logger.info("[%d/%d] %s", done, len(tasks), future.result())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
