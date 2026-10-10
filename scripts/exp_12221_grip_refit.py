"""#12221 experiment (not for merge): fixture pipeline on a given spec document.

Runs ``regenerate_club_face_fixtures``' pipeline and bundle export for one
capture with the document replaced, then measures the 2 ms fixture swing the
same way the fixture script does (face events, GCV-20 speed timing). Writes
``<work>/<capture>_measure.json``; never touches ``tests/fixtures``.

    python3 -m scripts.exp_12221_grip_refit --capture iron \
        --document full_body_spec_anthro_iron7_g12221.json --work runs/g
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from scripts import regenerate_club_face_fixtures as rcf


def _plain(value):  # noqa: ANN001, ANN202
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    from src.shared.python.motion_matching.club_face_target import (
        FACE_ORIENTATION_WEIGHT,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", choices=sorted(rcf.CAPTURE_RUNS), required=True)
    parser.add_argument("--document", required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    club, _, flags = rcf.CAPTURE_RUNS[args.capture]
    rcf.CAPTURE_RUNS[args.capture] = (club, args.document, flags)
    run_dir, bundle = rcf.run_and_export(
        args.capture, args.work, FACE_ORIENTATION_WEIGHT, False
    )
    with np.load(bundle, allow_pickle=False) as data:
        q = np.asarray(data["reference_q"], dtype=float)[:: rcf.FIXTURE_STRIDE]
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    out = {
        "document": args.document,
        "marker_rms_m": {
            "ik": receipt["ik"]["marker_rms_m"],
            "reference": receipt["ik"]["reference"]["marker_rms_m"],
            "tracking": receipt["dynamics"]["marker_rms_m"],
        },
        "impact_split": receipt["ik"].get("impact_split"),
        "measure": _plain(rcf.measure_swing(run_dir, q, rcf.FIXTURE_DT_S)),
    }
    path = args.work / f"{args.capture}_measure.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
