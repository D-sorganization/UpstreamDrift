"""Generate the limiting-cell articulated-shaft step diagnostic."""

from __future__ import annotations

import json

from scripts.research.proximal_distal_energy.articulated_shaft_atlas import DATA_DIR
from scripts.research.proximal_distal_energy.articulated_shaft_time_step_diagnostic import (
    run_articulated_shaft_time_step_diagnostic,
)


def main() -> None:
    record = run_articulated_shaft_time_step_diagnostic()
    path = DATA_DIR / "articulated_shaft_time_step_diagnostic.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(path)
    print(json.dumps(record["results"], indent=2))


if __name__ == "__main__":
    main()
