"""Frozen source selection for the presentation's existing study plates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PHASE = Path(__file__).resolve().parent.parent


def orient_points(saved: object) -> np.ndarray:
    """Return fresh [lateral, downrange] metres from finite saved coordinates."""
    points = np.asarray(saved, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("saved points must be a nonempty finite N×2 coordinate array")
    return points[:, [1, 0]].copy()


def cell_dir(protocol: str, cell: str) -> Path:
    roots = {
        "original_primary/1": PHASE / "final_v1",
        "original_control/1": PHASE / "control_final_v1",
        "pw-primary-face-bounds/1": PHASE / "supplement_final_v1/pw_primary_face",
    }
    if protocol not in roots:
        raise ValueError(f"unmapped source protocol: {protocol}")
    result = roots[protocol] / cell
    if not result.is_dir():
        raise ValueError(f"saved source missing: {result}")
    manifest = json.loads((result / "manifest.json").read_text())
    for name in ("shots.csv.gz", "flight_samples.json"):
        path = result / name
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["files_sha256"][name]
        ):
            raise ValueError(f"saved source hash differs from manifest: {path}")
    return result


def selected_pair(comparisons: list[dict], club: str, kind: str) -> dict:
    prefix = (
        f"{club}_lean0_curve1_rotation{'2' if kind == 'across_lean' else '1'}_combined2"
    )
    matches = [
        row
        for row in comparisons
        if row["comparison_kind"] == kind
        and row["sources"][0]["cell"] == prefix
        and (
            kind != "across_lean"
            or row["sources"][1]["cell"].startswith(f"{club}_lean5")
        )
    ]
    if len(matches) != 1 or matches[0]["status"] not in (
        "available",
        "conditional_only",
    ):
        raise ValueError(f"expected one available {club} {kind} source pair")
    return matches[0]
