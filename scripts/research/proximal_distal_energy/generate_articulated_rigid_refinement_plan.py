"""Write the canonical prospective rigid-refinement manifest for #9153."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_rigid_refinement_plan import (
    RigidRefinementExtensionPlan,
)


DEFAULT_OUTPUT = Path(
    "docs/research/proximal_distal_energy_transfer/data/"
    "articulated_rigid_refinement_plan.json"
)


def main() -> None:
    """Parse immutable identities and write one deterministic JSON manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-data-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = RigidRefinementExtensionPlan(
        source_revision=args.source_revision,
        source_data_sha256=args.source_data_sha256,
    ).to_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
