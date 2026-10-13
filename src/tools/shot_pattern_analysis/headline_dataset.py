"""Assemble the article's headline numbers into one hash-bound dataset.

Inputs are the integrity-checked saved cells, the 2,000-resample rescored
statistics, the explicit-contrast rescore and the head-assumption band runs.
The output JSON is the only source the LaTeX table renderer reads, so the
article's numbers can be regenerated and checked without the 10 GB archive.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Final

from .assumption_band import (
    CURVED_PATTERNS,
    NOMINAL_REPRODUCTION,
    assumption_envelope,
)
from .core import PATTERN_NAMES
from .headline_selection import (
    FIXED_TARGET_CONTRASTS,
    INTERACTION_CONTRASTS,
    PRIMARY_PROTOCOL,
    headline_cell_names,
    load_headline_cells,
    readable_condition,
)
from .provenance import publish_new_text, sha256_file

SCHEMA: Final = "shot-pattern-headline-dataset/1"
STATISTICS_DIR: Final = "statistics_v2"
GEOMETRY_FIELDS: Final = (
    "supported_n",
    "lateral_sd_m",
    "downrange_sd_m",
    "mean_carry_m",
    "target_rmse_m",
)
STATISTICS_FILES: Final = {
    PRIMARY_PROTOCOL: "final_v1.json",
    "pw-primary-face-bounds/1": "pw_primary_face.json",
}

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _band_runs(runs_dir: Path, cell: str) -> tuple[list[dict], dict]:
    """Return the cell's head-variant records and its nominal reproduction."""
    variants, reproduction = [], None
    for path in sorted(Path(runs_dir).glob(f"{cell}__*.json")):
        record = _load_json(path)
        if record.get("cell") != cell:
            raise ValueError(f"{path.name} names a different cell")
        if record["variant"] == NOMINAL_REPRODUCTION:
            reproduction = record
        else:
            variants.append(record)
    if reproduction is None or not variants:
        raise ValueError(f"{cell} lacks band variants or its nominal reproduction")
    return variants, reproduction


def _cell_entry(
    key: str,
    cell: str,
    source: dict[str, Any],
    rescored: dict[str, Any],
    runs_dir: Path,
) -> dict[str, Any]:
    summary = _load_json(source["dir"] / "summary.json")
    scoring = rescored["course_scoring"]
    paired = scoring["paired_benefit_vs_straight"]
    variants, reproduction = _band_runs(runs_dir, cell)
    resamples = {rescored["paired_comparisons"]["bootstrap_resamples"]}
    resamples |= {run["resamples"] for run in variants}
    if len(resamples) != 1:
        raise ValueError(f"{cell}: nominal and variant resamples differ: {resamples}")
    return {
        "key": key,
        "cell": cell,
        "condition": readable_condition(cell),
        "club_id": source["config"]["club"]["club_id"],
        "protocol": source["protocol"],
        "target_x_m": summary["target_x_m"],
        "scoring_status": scoring["status"],
        "paired_n": scoring["paired_n"],
        "geometry": {
            name: {field: summary["summary"][name][field] for field in GEOMETRY_FIELDS}
            for name in PATTERN_NAMES
        },
        "mean_strokes_gained": {
            name: scoring["patterns"][name]["mean_strokes_gained"]
            for name in PATTERN_NAMES
        },
        "paired_benefit_vs_straight": {name: paired[name] for name in CURVED_PATTERNS},
        "assumption_band": assumption_envelope(paired, variants),
        "nominal_reproduction": {
            field: reproduction[field]
            for field in (
                "shots_compared",
                "status_mismatches",
                "max_abs_endpoint_difference_m",
            )
        },
        "variant_status": {run["variant"]: run["status"] for run in variants},
        "variant_target_x_m": {
            run["variant"]: run.get("target_x_m") for run in variants
        },
    }


def build_headline_dataset(phase2_root: Path) -> dict[str, Any]:
    """Join saved cells, rescored statistics, contrasts and band runs.

    ``phase2_root`` is a restored phase-two tree whose ``statistics_v2``
    holds the rescored records and ``assumption_band/runs``. Fails closed when
    any input is missing or inconsistent; every input file's SHA-256 is
    recorded in ``inputs_sha256`` under its path relative to ``phase2_root``.
    """
    phase2_root = Path(phase2_root)
    statistics_dir = phase2_root / STATISTICS_DIR
    runs_dir = statistics_dir / "assumption_band/runs"
    sources = load_headline_cells(
        phase2_root / "final_v1",
        phase2_root / "supplement_final_v1/pw_primary_face",
    )
    statistics = {
        protocol: _load_json(statistics_dir / name)
        for protocol, name in STATISTICS_FILES.items()
    }
    contrasts = _load_json(statistics_dir / "contrasts.json")
    inputs = {
        f"{STATISTICS_DIR}/{name}": sha256_file(statistics_dir / name)
        for name in (*STATISTICS_FILES.values(), "contrasts.json")
    }
    cells = []
    for number, cell in enumerate(headline_cell_names(), start=1):
        source = sources[cell]
        rescored = statistics[source["protocol"]]["cells"][cell]
        if rescored["status"] != "completed":
            raise ValueError(f"{cell} has no rescored statistics")
        cells.append(_cell_entry(f"C{number:02d}", cell, source, rescored, runs_dir))
    for path in sorted(runs_dir.glob("*.json")):
        inputs[path.relative_to(phase2_root).as_posix()] = sha256_file(path)
    keys = [key for key, *_ in INTERACTION_CONTRASTS + FIXED_TARGET_CONTRASTS]
    if sorted(contrasts["contrasts"]) != sorted(keys):
        raise ValueError("contrast record does not cover exactly the article keys")
    resamples = {record["resamples"] for record in statistics.values()}
    resamples.add(contrasts["resamples"])
    if len(resamples) != 1:
        raise ValueError(f"inputs mix bootstrap resample counts: {resamples}")
    return {
        "schema": SCHEMA,
        "bootstrap_resamples": resamples.pop(),
        "units": {
            "distance": "metre",
            "strokes_gained": "strokes per modeled opportunity",
        },
        "cells": cells,
        "contrasts": {key: contrasts["contrasts"][key] for key in keys},
        "inputs_sha256": inputs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase2_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dataset = build_headline_dataset(args.phase2_root)
    publish_new_text(
        args.destination, json.dumps(dataset, indent=2, allow_nan=False) + "\n"
    )
    logger.info("wrote %s", args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
