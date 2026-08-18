"""Generate the registered closed-state forward-contact horizon evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.research.proximal_distal_energy.forward_contact_validity_horizon import (
    run_validity_horizon_study,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
JSON_PATH = DATA_DIR / "forward_contact_validity_horizon.json"
NPZ_PATH = DATA_DIR / "forward_contact_validity_horizon.npz"
SOURCES = (
    "docs/research/proximal_distal_energy_transfer/data/closed_state_forward_bridge.json",
    "docs/research/proximal_distal_energy_transfer/data/closed_state_forward_bridge.npz",
    "scripts/research/proximal_distal_energy/spatial_forward_contract.py",
    "scripts/research/proximal_distal_energy/spatial_forward_engines.py",
    "scripts/research/proximal_distal_energy/closed_state_forward_bridge.py",
    "scripts/research/proximal_distal_energy/forward_contact_validity_horizon.py",
    "scripts/research/proximal_distal_energy/run_forward_contact_validity_horizon.py",
)


def write_evidence() -> tuple[Path, Path]:
    """Execute and write the deterministic JSON and compressed array authority."""

    record, arrays = run_validity_horizon_study()
    record["source_sha256"] = {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCES
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH


def main() -> int:
    """Generate evidence and print the first cross-engine failure boundaries."""

    json_path, npz_path = write_evidence()
    record = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    for variant in record["results"]["variants"]:
        print(
            variant["variant_id"],
            "first incomplete-pass horizon:",
            variant["first_incomplete_pass_horizon_s"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
