"""Regenerate golden-snapshot fixtures for the motion-matching pipeline.

Run this script when an intentional change to a loader, alignment routine, or
quaternion convention shifts the canonical pipeline output. It overwrites every
JSON snapshot under ``tests/fixtures/motion_matching/`` with freshly-computed
values rounded to 1e-9.

Usage::

    python3 tests/fixtures/motion_matching/regenerate.py

CI MUST NOT call this script — snapshot drift is intentional and reviewable.

Snapshot policy
---------------

* Last 10 frames only: keeps fixtures small and text-diffable.
* Rounded to 1e-9: stable across BLAS / numpy versions but tighter than the
  ``atol=1e-6`` tolerance the comparison uses.
* Generic key names: no source-specific identifiers in the snapshot keys.
* Body-marker snapshot is regenerated only when ``BodyTarget`` infrastructure
  has merged; otherwise it is left untouched and the comparison test skips.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = Path(__file__).resolve().parent
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"
LAST_N_FRAMES = 10
ROUND_DECIMALS = 9


def _round_array(arr: np.ndarray) -> list:
    """Round an ndarray to 1e-9 and return a JSON-safe nested list."""
    return np.round(np.asarray(arr, dtype=np.float64), ROUND_DECIMALS).tolist()


def regenerate_club_snapshot() -> None:
    """Regenerate the driver-C3D ``ClubTarget`` last-10-frames snapshot."""
    from src.shared.python.motion_matching.load_club_target import load_club_target
    from src.shared.python.motion_matching.target import AlignOptions

    target = load_club_target(DRIVER_C3D, opts=AlignOptions())
    sl = slice(-LAST_N_FRAMES, None)
    snapshot = {
        "schema_version": 1,
        "source_filename": DRIVER_C3D.name,
        "n_frames": int(target.time.shape[0]),
        "impact_idx": int(target.impact_idx),
        "last_n": LAST_N_FRAMES,
        "time": _round_array(target.time[sl]),
        "butt": _round_array(target.butt[sl]),
        "clubhead": _round_array(target.clubhead[sl]),
        "club_quat": _round_array(target.club_quat[sl]),
    }
    out_path = FIXTURES_DIR / "club_target_driver_last10.json"
    out_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)
    print(f"wrote {out_path}")


def regenerate_body_snapshot() -> None:
    """Regenerate the driver-C3D ``BodyTarget`` last-10-frames snapshot.

    Skipped (with a printed message) when ``BodyTarget`` infrastructure has not
    yet merged. The placeholder JSON is left in place so the comparison test
    can still detect drift once the loader lands.
    """
    try:  # pragma: no cover - exercised once #4481 / #4482 land.
        from src.shared.python.motion_matching.load_body_target import (  # type: ignore[import-not-found]
            load_body_target,
        )
        from src.shared.python.motion_matching.target import AlignOptions
    except ImportError:
        print("BodyTarget loader not yet present; skipping body snapshot regenerate.")
        return

    target = load_body_target(DRIVER_C3D, opts=AlignOptions())  # type: ignore[call-arg]
    marker_subset = ("MidHands", "ClubFace", "HeadTop", "WaistLeft", "WaistRight")
    sl = slice(-LAST_N_FRAMES, None)
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_filename": DRIVER_C3D.name,
        "n_frames": int(target.time.shape[0]),
        "impact_idx": int(getattr(target, "impact_idx", -1)),
        "last_n": LAST_N_FRAMES,
        "time": _round_array(target.time[sl]),
        "marker_xyz": {},
    }
    markers = getattr(target, "markers", {})
    for name in marker_subset:
        if name in markers:
            payload["marker_xyz"][name] = _round_array(markers[name][sl])
    out_path = FIXTURES_DIR / "body_target_driver_last10.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    """Regenerate every available snapshot."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    regenerate_club_snapshot()
    regenerate_body_snapshot()


if __name__ == "__main__":
    main()
