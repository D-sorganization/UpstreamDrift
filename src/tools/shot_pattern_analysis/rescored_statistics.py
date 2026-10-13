"""Recompute paired intervals for frozen matrix cells from their saved shots.

Frozen cells keep their original 500-resample intervals. This module reads
each integrity-checked cell's per-shot landings and recomputes the paired
radial-error, containment and strokes-gained intervals with a new resample
count. It never writes inside a cell; results go to a separate JSON record
that binds every input by hash. No flight is re-simulated.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Final

from . import matched_supplement_pipeline as pairs
from .headline_selection import FIXED_TARGET_CONTRASTS, INTERACTION_CONTRASTS
from .matched_report import load_validated_cell
from .matched_runner import paired_endpoint_statistics
from .matched_sensitivity import _source_hashes, _supported_endpoints
from .provenance import publish_new_text, sha256_file
from .uncertainty import PUBLICATION_RESAMPLES

SCHEMA: Final = "shot-pattern-rescored-statistics/1"
INPUT_FILES: Final = ("manifest.json", "run_start.json", "summary.json", "shots.csv.gz")

logger = logging.getLogger(__name__)


def rescore_saved_cell(
    cell_dir: Path,
    config: dict[str, Any],
    execution: dict[str, Any] | None = None,
    *,
    resamples: int = PUBLICATION_RESAMPLES,
) -> dict[str, Any]:
    """Return recomputed ``course_scoring`` and ``paired_comparisons`` for a cell.

    Preconditions: the cell passes manifest and config checks and completed.
    Postcondition: the population (paired n, conditioning) equals the saved
    summary's, so only the interval resampling can differ.
    """
    cell_dir = Path(cell_dir)
    summary = load_validated_cell(cell_dir, config, execution)
    if summary["status"] != "completed":
        raise ValueError(f"{cell_dir.name} is not a completed cell")
    endpoints, _ = _supported_endpoints(cell_dir / "shots.csv.gz", config["n_shots"])
    scoring, comparisons = paired_endpoint_statistics(
        endpoints,
        club_id=config["club"]["club_id"],
        target_x_m=float(summary["target_x_m"]),
        target_radius_m=config["target_radius_m"],
        n_shots=config["n_shots"],
        seed=config["seed"],
        resamples=resamples,
    )
    saved = summary["paired_comparisons"]
    for key in ("paired_n", "total_n", "conditioning"):
        if comparisons.get(key) != saved.get(key):
            raise RuntimeError(
                f"{cell_dir.name}: rescored {key} differs from the saved summary"
            )
    return {
        "course_scoring": scoring,
        "paired_comparisons": comparisons,
        "input_sha256": {name: sha256_file(cell_dir / name) for name in INPUT_FILES},
    }


def _rescore_entry(
    root: Path, plan: dict[str, Any], resamples: int, name: str
) -> dict[str, Any]:
    cell_dir = root / name
    if not (cell_dir / "summary.json").exists():
        return {"status": "not_run"}
    status = json.loads((cell_dir / "summary.json").read_text())["status"]
    if status != "completed":
        return {"status": status}
    rescored = rescore_saved_cell(
        cell_dir, plan["cells"][name], plan["execution"], resamples=resamples
    )
    logger.info("rescored %s", name)
    return {"status": "completed", **rescored}


def rescore_matrix(
    root: Path,
    *,
    resamples: int = PUBLICATION_RESAMPLES,
    cells: tuple[str, ...] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Rescore every completed cell (or the named ``cells``) of a saved matrix.

    Unavailable or never-run cells are listed with their status, not dropped.
    Results are identical for any ``workers`` count; each cell seeds its own
    bootstrap.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    root = Path(root)
    plan_path = root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    names = tuple(plan["cells"]) if cells is None else cells
    unknown = set(names) - set(plan["cells"])
    if unknown:
        raise ValueError(f"cells not in the matrix plan: {sorted(unknown)}")
    task = partial(_rescore_entry, root, plan, resamples)
    if workers == 1:
        entries = list(map(task, names))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            entries = list(pool.map(task, names))
    results = dict(zip(names, entries, strict=True))
    return {
        "schema": SCHEMA,
        "matrix": root.name,
        "resamples": resamples,
        "matrix_plan_sha256": sha256_file(plan_path),
        "source_sha256": _source_hashes(),
        "scope": (
            "Paired bootstrap over common supported shots; Monte Carlo sampling "
            "only. Excludes model, course and golfer uncertainty."
        ),
        "cells": results,
    }


def rescore_contrasts(
    phase2_root: Path,
    contrasts: tuple[tuple[str, str, str, str], ...] = (
        INTERACTION_CONTRASTS + FIXED_TARGET_CONTRASTS
    ),
    *,
    resamples: int = PUBLICATION_RESAMPLES,
) -> dict[str, Any]:
    """Recompute the article's explicit common-six contrasts at ``resamples``.

    Each ``(key, kind, first_cell, second_cell)`` must select exactly one
    validated explicit pair; the source protocols come from that pair.
    """
    specs = pairs.explicit_pair_specs(phase2_root)
    results: dict[str, Any] = {}
    for key, kind, first, second in contrasts:
        matches = [
            spec
            for spec in specs
            if (spec["comparison_kind"], spec["first"]["cell"], spec["second"]["cell"])
            == (kind, first, second)
        ]
        if len(matches) != 1:
            raise ValueError(f"{key} selects {len(matches)} explicit pairs, not one")
        result = pairs.analyze_pair_spec(matches[0], phase2_root, resamples=resamples)
        results[key] = {
            "comparison_kind": kind,
            "cells": [first, second],
            "source_protocols": result["source_protocols"],
            "status": result["status"],
            "target_x_m": result["target_x_m"],
            "common_six_n": result["common_supported_n"],
            "total_n_per_population": result["total_n_per_population"],
            "effect_modification": result.get("course_scoring_effect_modification"),
            "sources": result["sources"],
        }
        logger.info("rescored contrast %s", key)
    return {
        "schema": "shot-pattern-rescored-contrasts/1",
        "resamples": resamples,
        "source_sha256": _source_hashes(),
        "scope": (
            "(curve - Straight) in the second condition minus (curve - Straight) "
            "in the first, paired on the six-population common supported shots; "
            "Monte Carlo sampling only."
        ),
        "contrasts": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contrasts",
        action="store_true",
        help="treat the root as a restored phase2 tree and rescore the contrasts",
    )
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--resamples", type=int, default=PUBLICATION_RESAMPLES)
    parser.add_argument("--cells", nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.contrasts:
        record = rescore_contrasts(args.matrix_root, resamples=args.resamples)
    else:
        record = rescore_matrix(
            args.matrix_root,
            resamples=args.resamples,
            cells=None if args.cells is None else tuple(args.cells),
            workers=args.workers,
        )
    publish_new_text(
        args.destination, json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
