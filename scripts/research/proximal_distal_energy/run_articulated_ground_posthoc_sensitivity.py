"""Generate the finite-ground post-hoc sensitivity artifact."""

from __future__ import annotations

import json

from scripts.research.proximal_distal_energy.articulated_ground_posthoc_sensitivity import (
    DATA,
    build_ground_posthoc_sensitivity,
)


def main() -> None:
    record = build_ground_posthoc_sensitivity()
    path = DATA / "articulated_ground_posthoc_sensitivity.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
