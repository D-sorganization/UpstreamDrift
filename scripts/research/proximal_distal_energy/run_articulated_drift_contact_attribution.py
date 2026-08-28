"""Write the articulated same-state attribution evidence for #9151."""

from __future__ import annotations

import json

import numpy as np

from scripts.research.proximal_distal_energy.articulated_drift_contact_attribution import (
    DATA_DIR,
    run_articulated_drift_contact_attribution,
)


def main() -> int:
    """Execute the registered atlas and write deterministic JSON/NPZ outputs."""

    record, arrays = run_articulated_drift_contact_attribution()
    json_path = DATA_DIR / "articulated_drift_contact_attribution.json"
    npz_path = DATA_DIR / "articulated_drift_contact_attribution.npz"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    print(json_path)
    print(npz_path)
    print(json.dumps(record["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
