"""Generate the registered articulated shaft atlas artifacts."""

from __future__ import annotations

import json

import numpy as np

from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    DATA_DIR,
    run_articulated_shaft_atlas,
)


def main() -> None:
    record, arrays = run_articulated_shaft_atlas()
    json_path = DATA_DIR / "articulated_shaft_atlas.json"
    npz_path = DATA_DIR / "articulated_shaft_atlas.npz"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    print(json_path)
    print(npz_path)
    print(json.dumps(record["results"], indent=2))


if __name__ == "__main__":
    main()
