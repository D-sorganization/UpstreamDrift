"""Generate the subject-scaled closed-contact feasibility evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.subject_scaled_closed_contact import (
    run_closed_contact_feasibility_atlas,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
JSON_PATH = DATA_DIR / "subject_scaled_closed_contact.json"
NPZ_PATH = DATA_DIR / "subject_scaled_closed_contact.npz"


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/research/proximal_distal_energy/spatial_full_body.py",
        "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
        "scripts/research/proximal_distal_energy/subject_scaled_closed_contact.py",
        "scripts/research/proximal_distal_energy/run_subject_scaled_closed_contact.py",
        "scripts/research/proximal_distal_energy/make_subject_scaled_closed_contact_figures.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def write_evidence() -> tuple[Path, Path]:
    """Execute the atlas and replace its deterministic artifacts."""

    record, arrays = run_closed_contact_feasibility_atlas()
    record["source_sha256"] = _source_hashes()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH


def main() -> int:
    """Write the bundle and print the principal feasibility results."""

    json_path, npz_path = write_evidence()
    record: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    results = record["results"]
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    print(
        "feasible samples: "
        f"{results['feasible_sample_count']}/{results['total_sample_count']} "
        f"({results['feasible_fraction']:.3f})"
    )
    print(
        "max closure error / min joint margin / min collision clearance: "
        f"{results['maximum_contact_error_m']:.6g} m / "
        f"{results['minimum_joint_limit_margin_rad']:.6g} rad / "
        f"{results['minimum_collision_clearance_m']:.6g} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
