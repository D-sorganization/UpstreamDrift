"""Generate the subject-scaled spatial contact-geometry evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    run_subject_scaled_geometry_atlas,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
JSON_PATH = DATA_DIR / "subject_scaled_spatial_geometry.json"
NPZ_PATH = DATA_DIR / "subject_scaled_spatial_geometry.npz"


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/research/proximal_distal_energy/spatial_full_body.py",
        "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
        "scripts/research/proximal_distal_energy/run_subject_scaled_spatial_geometry.py",
        "scripts/research/proximal_distal_energy/make_subject_scaled_spatial_geometry_figures.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def write_evidence() -> tuple[Path, Path]:
    """Execute the atlas and atomically replace its deterministic artifacts."""

    record, arrays = run_subject_scaled_geometry_atlas()
    record["source_sha256"] = _source_hashes()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH


def main() -> int:
    """Write the evidence bundle and print a compact scientific summary."""

    json_path, npz_path = write_evidence()
    record: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    closure = record["closure_tests"]
    geometry = record["geometry_tests"]
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    print(
        "hand-to-grip distance [min, median, max] m: "
        f"{closure['minimum_hand_to_grip_distance_m']:.6f}, "
        f"{closure['median_hand_to_grip_distance_m']:.6f}, "
        f"{closure['maximum_hand_to_grip_distance_m']:.6f}"
    )
    print(
        "contact Jacobian condition [min, median, max]: "
        f"{geometry['constraint_condition_number_minimum']:.3f}, "
        f"{geometry['constraint_condition_number_median']:.3f}, "
        f"{geometry['constraint_condition_number_maximum']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
