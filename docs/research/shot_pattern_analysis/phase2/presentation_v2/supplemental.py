"""Fail-closed source reader for optional supplemental flight replay plates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PHASE = Path(__file__).resolve().parent.parent
EXPECTED_CELLS = {
    "driver": PHASE / "final_v1/driver_lean0_curve1_rotation1_combined2",
    "seven_iron": PHASE / "final_v1/seven_iron_lean0_curve1_rotation1_combined2",
    "pitching_wedge": PHASE
    / "final_v1/pitching_wedge_lean0_curve1_rotation1_combined2",
    "driver_lean0": PHASE / "control_final_v1/driver_lean0_curve1_rotation2_combined2",
    "driver_lean5": PHASE / "control_final_v1/driver_lean5_curve1_rotation2_combined2",
}
SOURCE_FILES = ("shots.csv.gz", "flight_samples.json", "summary.json", "run_start.json")
PATTERNS = ("Straight", "Draw", "Fade")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SupplementalBundle:
    """Verified raw paths in saved [downrange, lateral] metres."""

    trajectories: dict[str, dict[str, list[list[list[float]]]]]
    sources: dict[str, str]
    shot_indices: tuple[int, ...]


def _verified_sources(
    receipt: dict[str, Any],
    cells: dict[str, Path],
    receipt_path: Path,
    trajectories_path: Path,
) -> dict[str, str]:
    """Bind every replay input to the frozen bytes named in the receipt."""
    source_hashes = receipt.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(cells):
        raise ValueError("supplemental source hashes incomplete")
    sources = {
        str(receipt_path): _sha(receipt_path),
        str(trajectories_path): _sha(trajectories_path),
    }
    for key, cell in cells.items():
        expected = source_hashes[key]
        if not isinstance(expected, dict) or set(expected) != set(SOURCE_FILES):
            raise ValueError("supplemental source hashes incomplete")
        for name in SOURCE_FILES:
            path = Path(cell) / name
            digest = _sha(path)
            if expected[name] != digest:
                raise ValueError(f"supplemental source hash differs: {key}/{name}")
            sources[str(path)] = digest
    return sources


def _verified_indices(receipt: dict[str, Any]) -> list[int]:
    """Require the recorded selection to be exactly 100 ordered common IDs."""
    indices = receipt.get("shot_indices")
    if (
        receipt.get("selected_paths_per_pattern") != 100
        or not isinstance(indices, list)
        or len(indices) != 100
        or any(type(i) is not int or i < 5 for i in indices)
        or len(set(indices)) != 100
        or indices != sorted(indices)
    ):
        raise ValueError("supplemental receipt must identify 100 distinct shot indices")
    return indices


def _verified_paths(
    records: Any, cells: dict[str, Path], indices: list[int]
) -> dict[str, dict[str, list[list[list[float]]]]]:
    """Check population identity, count, pairing IDs and coordinate shape."""
    if not isinstance(records, dict) or set(records) != set(cells):
        raise ValueError("supplemental trajectory cell keys differ")
    trajectories: dict[str, dict[str, list[list[list[float]]]]] = {}
    for key, groups in records.items():
        if not isinstance(groups, dict) or set(groups) != set(PATTERNS):
            raise ValueError("supplemental pattern keys differ")
        trajectories[key] = {}
        for pattern, paths in groups.items():
            if not isinstance(paths, list) or len(paths) != 100:
                raise ValueError("supplemental paths must contain 100 per pattern")
            if [item.get("shot_index") for item in paths] != indices:
                raise ValueError("supplemental shot indices differ across populations")
            raw = [item.get("path_xy_m") for item in paths]
            for path in raw:
                points = np.asarray(path, dtype=float)
                if points.shape != (41, 2) or not np.isfinite(points).all():
                    raise ValueError("supplemental path must be finite 41×2 metres")
            trajectories[key][pattern] = raw
    return trajectories


def load_supplemental(
    directory: Path,
    *,
    source_cells: dict[str, Path] | None = None,
) -> SupplementalBundle:
    """Load exactly 100 replayed shots per pattern after hash and shape checks."""
    directory = Path(directory).resolve()
    receipt_path = directory / "receipt.json"
    trajectories_path = directory / "trajectories.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "shot-pattern-supplemental-replay/1":
        raise ValueError("unsupported supplemental replay receipt")
    if _sha(trajectories_path) != receipt.get("trajectories_sha256"):
        raise ValueError("supplemental trajectories hash differs from receipt")
    cells = EXPECTED_CELLS if source_cells is None else source_cells
    if set(cells) != set(EXPECTED_CELLS):
        raise ValueError("supplemental source cell keys differ")
    sources = _verified_sources(receipt, cells, receipt_path, trajectories_path)
    indices = _verified_indices(receipt)
    trajectories = _verified_paths(
        json.loads(trajectories_path.read_text()), cells, indices
    )
    return SupplementalBundle(trajectories, sources, tuple(indices))
