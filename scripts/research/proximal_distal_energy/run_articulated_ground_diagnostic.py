"""Generate the registered ground-pathway diagnostic artifact."""

from __future__ import annotations

import json

from scripts.research.proximal_distal_energy.articulated_ground_diagnostic import (
    DATA,
    run_articulated_ground_diagnostic,
)


def main() -> None:
    record = run_articulated_ground_diagnostic()
    path = DATA / "articulated_ground_diagnostic.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(path)
    print(
        json.dumps({key: record[key] for key in ("initialization", "parity")}, indent=2)
    )


if __name__ == "__main__":
    main()
