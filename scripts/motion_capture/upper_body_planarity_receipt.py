"""Create reproducible planarity-preflight receipts for TB-06 C3D campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.pendulum.python.motion_matching.upper_body_capture import (
    CaptureFrameCalibration,
    CaptureSourceClock,
    assess_planarity_lower_bound,
    calibrate_capture_frame,
)
from src.shared.python.motion_matching.body_target import BodyTarget
from src.shared.python.motion_matching.club_target import AlignOptions
from src.shared.python.motion_matching.loaders.c3d import load_club_target_c3d
from src.shared.python.motion_matching.loaders.c3d_body import load_body_target_c3d
from src.shared.python.sidekick.lab.bio.c3d_reader import C3DDataReader
from src.shared.python.tour_baselines.qualification_profiles import (
    UpperBodyGolferProfile,
)

MARKER_NAMES: tuple[str, ...] = (
    "RShoulderBack",
    "RElbowOut",
    "RWristTop",
    "LShoulderBack",
    "LElbowOut",
    "LWristTop",
)


def build_planarity_receipt(
    *,
    capture_name: str,
    body_target: BodyTarget,
    source_clock: CaptureSourceClock,
    calibration: CaptureFrameCalibration,
    max_marker_rmse_m: float,
) -> dict[str, Any]:
    """Build one traceable, non-promoting capture-planarity receipt."""
    if not capture_name:
        raise ValueError("capture_name must be non-empty")
    assessment = assess_planarity_lower_bound(
        calibration, max_marker_rmse_m=max_marker_rmse_m
    )
    return {
        "schema": "upper-body-planarity-preflight/1.0.0",
        "capture": capture_name,
        "model_id": "constrained_upper_body_golfer",
        "source": {
            "filename": body_target.source.filename,
            "sha256": body_target.source.sha256,
            "coordinate_frame": body_target.coordinate_frame,
        },
        "source_clock": {
            "raw_frame_count": source_clock.raw_frame_count,
            "raw_frame_rate_hz": source_clock.raw_frame_rate_hz,
        },
        "evaluation_clock": {
            "frame_count": len(body_target.time),
            "sample_rate_hz": float(1.0 / (body_target.time[1] - body_target.time[0])),
            "impact_index": body_target.impact_idx,
        },
        "marker_names": list(calibration.marker_names),
        "marker_coverage_fraction": {
            name: float(
                np.isfinite(
                    body_target.marker_xyz[:, body_target.marker_names.index(name), :]
                )
                .all(axis=-1)
                .mean()
            )
            for name in calibration.marker_names
        },
        "capture_plane": {
            "origin_m": calibration.frame.origin_m.tolist(),
            "basis": calibration.frame.plane_basis.tolist(),
            "rmse_m": calibration.plane_rmse_m,
            "max_deviation_m": calibration.plane_max_deviation_m,
        },
        "planarity_lower_bound": {
            "max_marker_rmse_m": assessment.max_marker_rmse_m,
            "marker_rmse_lower_bound_m": assessment.marker_rmse_lower_bound_m,
            "reason": assessment.reason,
        },
        "status": (
            "planar_fit_eligible_not_qualified"
            if assessment.planar_fit_is_eligible
            else "planar_fit_blocked_unqualified"
        ),
        "qualification": "not_assessed",
        "statuses": {
            "scientific_qualification": (
                "not_assessed" if assessment.planar_fit_is_eligible else "disqualified"
            ),
        },
    }


def _load_receipt(capture_name: str, c3d_path: Path) -> dict[str, Any]:
    reader = C3DDataReader(c3d_path)
    metadata = reader.get_metadata()
    source_clock = CaptureSourceClock(
        raw_frame_count=int(metadata.frame_count),
        raw_frame_rate_hz=float(metadata.frame_rate),
    )
    options = AlignOptions(
        sample_rate_hz=1000.0,
        simulation_time_s=0.3,
        time_alignment="impact",
        impact_target_t_s=0.25,
    )
    club_target = load_club_target_c3d(c3d_path, options)
    body_target = load_body_target_c3d(
        c3d_path,
        options,
        marker_set=MARKER_NAMES,
        impact_source=club_target,
    )
    calibration = calibrate_capture_frame(body_target, MARKER_NAMES)
    profile = UpperBodyGolferProfile()
    return build_planarity_receipt(
        capture_name=capture_name,
        body_target=body_target,
        source_clock=source_clock,
        calibration=calibration,
        max_marker_rmse_m=profile.max_whole_marker_rmse_m,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-c3d", type=Path, required=True)
    parser.add_argument("--iron-c3d", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Write preflight receipts for declared Driver and Iron source files."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for capture_name, c3d_path in (
        ("driver", args.driver_c3d),
        ("iron", args.iron_c3d),
    ):
        receipt = _load_receipt(capture_name, c3d_path)
        output_path = args.output_dir / f"tb06_{capture_name}_planarity_receipt.json"
        output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
