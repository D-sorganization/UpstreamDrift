"""OS-7 post-processing: when does the uninterrupted replay depart from the capture?

Pure numpy; runs anywhere. For every ``rungs/<ms>ms/replay_markers.npz`` under
an OS-7 evidence directory it computes the per-frame marker RMSE of the
replay against the capture (original validity mask, ``RShoulderTop``
excluded) and reports the first time the RMSE exceeds ``--threshold-mm``
(default 60 mm, 1.5x the OS-3b calibration floor). Output:
``replay_departure.json`` next to the receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.tour_matching.moco_g1 import (  # noqa: E402
    window_capture,
)
from src.engines.physics_engines.opensim.python.tour_matching.trc import (  # noqa: E402
    read_trc,
)

EXCLUDED = ("RShoulderTop",)


def per_frame_rmse_mm(
    points_m: np.ndarray, valid: np.ndarray, replay_m: np.ndarray
) -> np.ndarray:
    """Per-frame RMSE (mm) over valid markers; NaN where no marker is valid."""
    if points_m.shape != replay_m.shape or valid.shape != points_m.shape[:2]:
        raise ValueError("replay and capture arrays must share (frames, markers, 3)")
    sq = np.sum((replay_m - points_m) ** 2, axis=-1)
    sq = np.where(valid, sq, np.nan)
    with np.errstate(invalid="ignore"):
        return 1000.0 * np.sqrt(np.nanmean(sq, axis=1))


def departure_s(
    time_s: np.ndarray, rmse_mm: np.ndarray, threshold_mm: float
) -> float | None:
    """First time the RMSE exceeds the threshold, or None if it never does."""
    over = np.flatnonzero(np.nan_to_num(rmse_mm, nan=0.0) > threshold_mm)
    return None if over.size == 0 else float(time_s[over[0]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--trc", type=Path, required=True)
    parser.add_argument("--threshold-mm", type=float, default=60.0)
    args = parser.parse_args()
    capture = read_trc(args.trc)
    keep = [i for i, label in enumerate(capture.labels) if label not in EXCLUDED]
    report: dict[str, dict] = {}
    for npz in sorted(args.evidence.glob("rungs/*/replay_markers.npz")):
        with np.load(npz) as data:
            replay = np.asarray(data["markers_m"], dtype=float)
        window = window_capture(capture, float(capture.time_s[replay.shape[0] - 1]))
        window_frames = min(window.frames, replay.shape[0])
        points = window.points_m[:window_frames][:, keep]
        valid = window.valid[:window_frames][:, keep]
        curve = per_frame_rmse_mm(points, valid, replay[:window_frames][:, keep])
        finite = np.isfinite(replay[:window_frames]).all(axis=(1, 2))
        curve = np.where(finite, curve, np.inf)
        stride = max(1, window_frames // 40)
        report[npz.parent.name] = {
            "frames": int(window_frames),
            "departure_s": departure_s(
                window.time_s[:window_frames], curve, args.threshold_mm
            ),
            "threshold_mm": args.threshold_mm,
            "rmse_mm_at_0.05s_steps": [
                [round(float(t), 3), None if not np.isfinite(r) else round(float(r), 1)]
                for t, r in zip(window.time_s[::stride], curve[::stride], strict=False)
            ],
            "final_rmse_mm": None
            if not np.isfinite(curve[-1])
            else round(float(curve[-1]), 1),
        }
    out = args.evidence / "replay_departure.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, row in report.items():
        print(
            name,
            "departure_s",
            row["departure_s"],
            "final_rmse_mm",
            row["final_rmse_mm"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
