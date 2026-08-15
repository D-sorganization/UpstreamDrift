"""Generate the closed-state-to-forward-contact bridge evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.closed_state_forward_bridge import (
    evaluate_spanning_forward_subset,
    map_closed_contact_atlas,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
JSON_PATH = DATA_DIR / "closed_state_forward_bridge.json"
NPZ_PATH = DATA_DIR / "closed_state_forward_bridge.npz"
SOURCES = (
    "scripts/research/proximal_distal_energy/spatial_forward_contract.py",
    "scripts/research/proximal_distal_energy/spatial_forward_engines.py",
    "scripts/research/proximal_distal_energy/spatial_forward_study.py",
    "scripts/research/proximal_distal_energy/closed_state_forward_bridge.py",
    "scripts/research/proximal_distal_energy/run_closed_state_forward_bridge.py",
)


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCES
    }


def write_evidence() -> tuple[Path, Path]:
    """Execute all mapping and forward gates and replace deterministic artifacts."""

    record, arrays = map_closed_contact_atlas()
    forward, forward_arrays = evaluate_spanning_forward_subset(arrays)
    record["forward_subset"] = forward
    record["source_sha256"] = _source_hashes()
    arrays.update(forward_arrays)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH


def main() -> int:
    """Write evidence and report the principal falsification gates."""

    json_path, npz_path = write_evidence()
    record: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    print(json.dumps(record["results"], indent=2, sort_keys=True))
    print(
        "forward subset gates: ",
        {
            key: value
            for key, value in record["forward_subset"].items()
            if key.endswith(("passed", "match"))
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
