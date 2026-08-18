"""Generate the registered finite-ground atlas artifacts."""

from __future__ import annotations

import json

import numpy as np

from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    DATA,
    run_articulated_ground_atlas,
)


def main() -> None:
    record, arrays = run_articulated_ground_atlas()
    json_path = DATA / "articulated_ground_atlas.json"
    npz_path = DATA / "articulated_ground_atlas.npz"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    print(json_path)
    print(npz_path)
    print(json.dumps(record["results"], indent=2))


if __name__ == "__main__":
    main()
