"""OS-1: export the canonical tour capture as a TRC file with a bound receipt.

Reads data/C3D_TA_Driver.c3d through the frozen shared contract, restricts to
the 34 tracked labels (unassigned markers are excluded, never renamed), writes
evidence/tour_average_tracked.trc, verifies the file by roundtrip and records
hashes, counts and the marker-to-body map. No OpenSim binding is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.tour_matching import (  # noqa: E402
    GOLF_HUMANOID_MARKER_BODIES,
    read_trc,
    write_trc,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    TOUR_CAPTURE,
    load_tour_capture,
    tracked_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c3d", type=Path, default=ROOT / "data/C3D_TA_Driver.c3d")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence",
    )
    args = parser.parse_args()
    capture = load_tour_capture(args.c3d).subset(tracked_labels())
    args.output.mkdir(parents=True, exist_ok=True)
    trc_path = write_trc(
        capture, args.output / "tour_average_tracked.trc", rate_hz=TOUR_CAPTURE.rate_hz
    )
    back = read_trc(trc_path)
    if back.labels != capture.labels or not np.array_equal(back.valid, capture.valid):
        raise ValueError("TRC roundtrip changed labels or validity")
    roundtrip = float(
        np.max(np.abs(back.points_m[back.valid] - capture.points_m[capture.valid]))
    )
    receipt = {
        "work_package": "OS-1: Marker, Frame and Clock Contract",
        "epic": "#10003",
        "c3d_sha256": capture.source_sha256,
        "trc_sha256": hashlib.sha256(trc_path.read_bytes()).hexdigest(),
        "trc_path": str(trc_path.relative_to(ROOT)),
        "frames": capture.frames,
        "rate_hz": TOUR_CAPTURE.rate_hz,
        "duration_s": float(capture.time_s[-1]),
        "units": TOUR_CAPTURE.units,
        "vertical_axis": TOUR_CAPTURE.vertical_axis,
        "axis_policy": "C3D and OpenSim are both Y-up metres; no rotation or rescale",
        "tracked_labels": list(capture.labels),
        "excluded_labels": [
            label for label in TOUR_CAPTURE.labels if label not in capture.labels
        ],
        "valid_points": capture.valid_count(),
        "valid_points_per_label": {
            label: int(capture.valid[:, i].sum())
            for i, label in enumerate(capture.labels)
        },
        "roundtrip_max_abs_m": roundtrip,
        "marker_bodies": dict(GOLF_HUMANOID_MARKER_BODIES),
        "offsets": "not calibrated here; OS-3 owns marker placement",
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "src/shared/python/motion_matching/tour_capture_contract.py",
                ROOT
                / "src/engines/physics_engines/opensim/python/tour_matching/trc.py",
                ROOT
                / "src/engines/physics_engines/opensim/python/tour_matching/marker_map.py",
            )
        },
    }
    (args.output / "os1_trc_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return 0 if roundtrip <= 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
