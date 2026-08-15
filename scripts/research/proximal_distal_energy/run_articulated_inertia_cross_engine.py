"""Write the subject-scaled articulated MuJoCo/Pinocchio inertia evidence."""

from __future__ import annotations

import json

import numpy as np

from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    DATA_DIR,
    run_articulated_inertia_atlas,
)


def main() -> int:
    """Run the registered atlas and write deterministic JSON/NPZ evidence."""

    record, arrays = run_articulated_inertia_atlas()
    json_path = DATA_DIR / "articulated_inertia_cross_engine.json"
    npz_path = DATA_DIR / "articulated_inertia_cross_engine.npz"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    print(json_path)
    print(npz_path)
    print(json.dumps(record["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
