"""Bind reference rendering to existing session camera and timing evidence."""

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.motion_capture.reconstruct.intrinsics import IntrinsicsRecord
from src.motion_capture.reconstruct.overlay3d import calibration_for_view
from src.motion_capture.rig.alignment import view_timing
from src.shared.python.pose_estimation import CameraCalibration, CameraIntrinsics

from .evidence import CameraSnapshot, ViewClock


def session_camera(session: Path, variant: str, view: str) -> CameraSnapshot:
    """Keep the rig's camera selection; reapply measured lens evidence if present.

    No intrinsics are invented. Missing lens evidence is explicitly recorded as
    the existing reconstruction's ideal pinhole assumption. Conflicting measured
    lens dimensions or focal lengths require review rather than silent scaling.
    """
    record = calibration_for_view(session, variant, view)
    lens_description = (
        "stored distortion"
        if record.intrinsics.distortion is not None
        else "ideal pinhole, no lens evidence"
    )
    provenance = (
        f"Session reconstruction, variant {variant or 'default'}; {lens_description}"
    )
    lens_file = session / "intrinsics.json"
    if lens_file.is_file():
        if lens_file.stat().st_size > 1_000_000:
            raise ValueError("Intrinsics document is too large")
        rows = json.loads(lens_file.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Intrinsics document must contain calibration records")
        lenses = [IntrinsicsRecord.model_validate(row) for row in rows]
        matches = [lens for lens in lenses if lens.camera_id == view]
        if len(matches) > 1:
            raise ValueError("Duplicate lens records for this camera")
        if matches:
            lens = matches[0]
            if lens.image_size_px != record.image_size_px or not np.allclose(
                lens.matrix, record.intrinsics.matrix
            ):
                raise ValueError(
                    "Lens calibration differs from reconstruction; review alignment"
                )
            existing = record.intrinsics.distortion
            if existing is not None and (
                len(existing) != len(lens.distortion)
                or not np.allclose(existing, lens.distortion)
            ):
                raise ValueError(
                    "Lens distortion differs from reconstruction; review alignment"
                )
            record = CameraCalibration(
                record.camera_id,
                CameraIntrinsics(np.asarray(lens.matrix), np.asarray(lens.distortion)),
                record.extrinsics,
                record.image_size_px,
            )
            provenance = f"Session reconstruction, variant {variant or 'default'}; intrinsics.json RMS {lens.rms_px:g}px"
    return CameraSnapshot.from_calibration(record, provenance=provenance)


def session_clock(timing: Mapping[str, Any], view: str) -> ViewClock:
    """Use recorded alignment evidence, or label the nominal clock as unverified."""
    entry = view_timing(timing, view)
    if entry is None or entry.get("status") != "available":
        return ViewClock(view=view)
    return ViewClock.model_validate(
        {
            "view": view,
            "offset_ns": entry.get("offset_ns"),
            "uncertainty_ns": entry.get("uncertainty_ns"),
            "source": str(timing.get("method", "recorded timing")),
        }
    )
