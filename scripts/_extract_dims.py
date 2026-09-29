"""Extract dimension/mass/inertia values from a Simscape dataset CSV.

This is a one-shot helper used while drafting shared/models/*.yaml from the
live Simscape model workspace as logged into a dataset trial CSV.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main(csv_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        first_row = next(reader)

    name_to_value = dict(zip(header, first_row, strict=True))
    keep_prefixes = ("model_",)
    keep_substrings = (
        "Length",
        "Width",
        "Mass",
        "Radius",
        "ClubLogs_ClubMass",
        "GolferMass",
        "SegmentInertiaLogs_",
    )

    out: dict[str, str] = {}
    for k, v in name_to_value.items():
        keep = (
            (k.startswith(keep_prefixes) and any(s in k for s in keep_substrings))
            or k in {"ClubLogs_ClubMass", "SegmentInertiaLogs_GolferMass"}
            or (
                k.startswith("SegmentInertiaLogs_")
                and any(s in k for s in ("Mass", "COM", "Inertia"))
            )
        )
        if keep:
            out[k] = v

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
