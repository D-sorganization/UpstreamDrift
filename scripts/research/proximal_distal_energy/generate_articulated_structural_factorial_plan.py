"""Write the canonical prospective structural-factorial design for #9153."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / (
    "docs/research/proximal_distal_energy_transfer/data/"
    "articulated_structural_factorial_plan.json"
)
AUTHORITIES = {
    "closed_state_npz": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "shaft_structural_basis_json": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_structural_basis.json",
    "shaft_structural_basis_npz": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_structural_basis.npz",
    "shaft_atlas_json": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_atlas.json",
    "shaft_atlas_npz": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_atlas.npz",
    "ground_atlas_json": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_ground_atlas.json",
    "ground_atlas_npz": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_ground_atlas.npz",
}


def main() -> None:
    """Hash the frozen authorities and write one deterministic JSON plan."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--design-authority-revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in AUTHORITIES.items()
    }
    manifest = StructuralFactorialPlan(
        design_authority_revision=args.design_authority_revision,
        authority_sha256=hashes,
    ).to_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
