"""Generate governed native-constraint discrepancy evidence for issue #8911."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_native_constraint_discrepancy import (
    run_native_constraint_discrepancy,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data"
JSON_OUTPUT = DATA / "articulated_native_constraint_discrepancy.json"
NPZ_OUTPUT = DATA / "articulated_native_constraint_discrepancy.npz"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_contact_projection.py",
    "scripts/research/proximal_distal_energy/articulated_forward_contract.py",
    "scripts/research/proximal_distal_energy/articulated_forward_integration.py",
    "scripts/research/proximal_distal_energy/articulated_native_constraint_discrepancy.py",
    "scripts/research/proximal_distal_energy/make_articulated_native_constraint_discrepancy_figure.py",
    "scripts/research/proximal_distal_energy/run_articulated_native_constraint_discrepancy.py",
    "scripts/research/proximal_distal_energy/spatial_full_body.py",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
    "tests/research/test_articulated_native_constraint_discrepancy.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_evidence() -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Execute the preregistered comparison and bind every governing source."""

    import mujoco

    record, arrays = run_native_constraint_discrepancy()
    record["native_branch"]["engine_version"] = str(mujoco.__version__)
    record["classification"] = (
        "synthetic_formulation_discrepancy_not_human_or_engine_equivalence_evidence"
    )
    record["source_sha256"] = {
        relative: _sha256(ROOT / relative) for relative in SOURCE_PATHS
    }
    return record, arrays


def write_evidence(
    json_path: Path = JSON_OUTPUT,
    npz_path: Path = NPZ_OUTPUT,
) -> tuple[Path, Path]:
    """Atomically write the qualified JSON record and numeric arrays."""

    record, arrays = build_evidence()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    json_temporary.replace(json_path)
    npz_temporary = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with npz_temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    npz_temporary.replace(npz_path)
    return json_path, npz_path


def main() -> None:
    print(write_evidence())


if __name__ == "__main__":
    main()
