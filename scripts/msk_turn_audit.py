#!/usr/bin/env python3
"""Turn audit of the OSV-9 musculoskeletal club retarget against the capture (#12042).

Tracks the generated club-face swing fixture (``tests/fixtures/club_face/
swing_q_<club>.npz``, the MuJoCo source of OSV-9) with
:func:`msk_club_tracking.track_swing` and reports the shared turn lines
(``swing_comparison.turn``) of the capture markers, the MuJoCo source (hip and
shoulder joint centres) and the Rajagopal retarget, at the capture's address,
top and impact. Modes:

* ``baseline``: the OSV-9 tracking before slice 4 (landmarks only, square
  synthetic stance);
* ``turn``: plus the capture's pelvis and upper-trunk turn targets;
* ``turn_feet``: plus the feet planted from the capture's foot markers.

The capture is private: pass its path (``--capture``); nothing from it is
written into the repository. Example::

    python3 scripts/msk_turn_audit.py --capture "$CAPTURE_DATA_DIR/<capture>.c3d" \\
        --club driver --mode turn_feet --out /tmp/msk_turn_driver.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engines.physics_engines.opensim.python import msk_club as mc  # noqa: E402
from src.engines.physics_engines.opensim.python import (  # noqa: E402
    msk_club_calibration as cal,
)
from src.engines.physics_engines.opensim.python import (  # noqa: E402
    msk_club_tracking as mt,
)
from src.engines.physics_engines.opensim.python import (  # noqa: E402
    msk_turn_targets as tt,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    load_tour_capture,
)
from src.shared.python.motion_matching.turn_receipt import (  # noqa: E402
    capture_turn_inputs,
)
from src.shared.python.swing_comparison.turn import (  # noqa: E402
    marker_turn_lines,
    model_turn_lines,
    turn_source_block,
)

logger = logging.getLogger(__name__)

MODES = ("baseline", "turn", "turn_feet")
#: Generated-model bodies whose origins are the source hip and shoulder centres.
SOURCE_BODIES = {
    "hip_l": "femur_l",
    "hip_r": "femur_r",
    "shoulder_l": "LS",
    "shoulder_r": "RS",
}
COORDINATES = ("pelvis_rotation", "lumbar_rotation", "hip_rotation_l", "hip_rotation_r")


def row_indices(n_rows: int, stride: int, event_times_s: list[float]) -> np.ndarray:
    """Every ``stride``-th fixture row plus the rows nearest the event times."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    events = [min(n_rows - 1, round(t / mt.FIXTURE_DT_S)) for t in event_times_s]
    return np.array(sorted(set(range(0, n_rows, stride)) | set(events)))


def _source_points(club: str, rows: np.ndarray) -> dict[str, np.ndarray]:
    generated = cal.GeneratedSwing(club)
    origins = [
        generated.body_origins(row, list(SOURCE_BODIES.values())) for row in rows
    ]
    return {k: np.array([o[b] for o in origins]) for k, b in SOURCE_BODIES.items()}


def audit(capture_path: Path, club: str, mode: str, stride: int) -> dict[str, Any]:
    """Run one retarget mode and return the turn table (JSON-ready)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    capture = load_tour_capture(capture_path)
    cap = capture_turn_inputs(capture)
    events = cap.events
    markers = marker_turn_lines(cap.markers, cap.t, events)
    fixture = mc.REPO_ROOT / "tests" / "fixtures" / "club_face" / f"swing_q_{club}.npz"
    all_rows = np.load(fixture)["q"]
    idx = row_indices(
        len(all_rows),
        stride,
        [events.address_time, events.top_time, events.impact_time],
    )
    rows, times = all_rows[idx], idx * mt.FIXTURE_DT_S
    targets = None if mode == "baseline" else tt.turn_targets_from_lines(markers, times)
    feet = tt.planted_feet_from_capture(capture) if mode == "turn_feet" else None
    start = time.monotonic()
    frames = mt.track_swing(
        mc.MODELS_DIR / "golf_humanoid.osim",
        rows,
        club=club,
        turn_targets=targets,
        feet=feet,
    )
    elapsed = time.monotonic() - start
    points = {
        k: np.array([f.turn_points[k] for f in frames]) for k in frames[0].turn_points
    }
    msk = model_turn_lines(points, times, events)
    source = model_turn_lines(_source_points(club, rows), times, events)
    at = {
        name: int(np.argmin(np.abs(times - t)))
        for name, t in (
            ("address", events.address_time),
            ("top", events.top_time),
            ("impact", events.impact_time),
        )
    }
    return {
        "club": club,
        "mode": mode,
        "stride": stride,
        "frames_tracked": len(frames),
        "track_seconds": round(elapsed, 1),
        "markers": turn_source_block(markers, events, "capture_markers"),
        "source": turn_source_block(source, events, "mujoco_fd_fixture_joint_centres"),
        "msk": turn_source_block(msk, events, f"opensim_msk_retarget_{mode}"),
        "max_lead_grip_error_m": max(f.lead_grip_error_m for f in frames),
        "max_trail_grip_gap_m": max(f.trail_grip_gap_m for f in frames),
        "landmark_rms_m": {k: frames[i].landmark_rms_m for k, i in at.items()},
        "coordinates_deg": {
            k: {c: float(np.degrees(frames[i].q[c])) for c in COORDINATES}
            for k, i in at.items()
        },
    }


def format_table(result: dict[str, Any]) -> str:
    """Plain-text turn table (top / impact, degrees) of an audit result."""
    lines = [
        f"{result['club']} {result['mode']}: turn from address, top / impact (deg)"
    ]
    keys = (
        "shoulder_girdle",
        "upper_trunk",
        "pelvis",
        "x_factor",
        "x_factor_shoulder_girdle",
    )
    for source in ("markers", "source", "msk"):
        cells = []
        for key in keys:
            line = result[source][key]
            vals = [line["top_deg"], line["impact_deg"]]
            cells.append(" / ".join("n/a" if v is None else f"{v:6.1f}" for v in vals))
        lines.append(f"{source:8s} " + " | ".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", type=Path, required=True, help="capture C3D")
    parser.add_argument("--club", default="driver", choices=("driver", "iron7"))
    parser.add_argument("--mode", default="turn_feet", choices=MODES)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True, help="output JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = audit(args.capture, args.club, args.mode, args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    logger.info("\n%s", format_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
