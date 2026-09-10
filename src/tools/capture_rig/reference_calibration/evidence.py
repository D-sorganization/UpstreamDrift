"""Provider-independent review of saved camera-layout bindings for the wizard."""

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID
from datetime import datetime

import numpy as np

from src.motion_capture.reconstruct.intrinsics import IntrinsicsRecord
from src.motion_capture.rig.capture_notes import read_notes
from src.shared.python.pose_estimation.observations import CameraCalibration


def reviewed_intrinsics(payload: dict[str, Any], root: Path | None) -> bytes:
    """Validate the reviewed result and return its existing flat intrinsic records."""
    if root is None or payload.get("capture_id") != read_notes(root).capture_id:
        raise ValueError("Review this camera layout for the selected capture")
    if payload.get("operator_reviewed") is not True:
        raise ValueError("Review camera-layout evidence before using it")
    reviewed = payload.get("reviewed_utc")
    if (
        not isinstance(reviewed, str)
        or datetime.fromisoformat(reviewed).utcoffset() is None
    ):
        raise ValueError("Camera layout needs a dated operator review")
    revision = UUID(payload["reference_revision_id"])
    path = root / "reference_calibration" / f"{revision}.json"
    with path.open("rb") as stream:
        data = stream.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != payload.get(
        "revision_file_sha256"
    ):
        raise ValueError("Reference observations changed; review or solve them again")
    cameras = [CameraCalibration.from_dict(item) for item in payload["cameras"]]
    records = [
        IntrinsicsRecord.model_validate(item) for item in payload["intrinsic_records"]
    ]
    if len(cameras) != len(records) or len(
        {camera.camera_id for camera in cameras}
    ) != len(cameras):
        raise ValueError("Camera layout contains inconsistent view identities")
    lookup = {record.camera_id: record for record in records}
    if set(lookup) != {camera.camera_id for camera in cameras}:
        raise ValueError("Camera layout and intrinsic views do not match")
    for camera in cameras:
        record = lookup[camera.camera_id]
        intrinsic = camera.intrinsics
        coefficients = intrinsic.distortion
        if coefficients is None:
            coefficients = np.empty(0)
        if (
            camera.image_size_px != record.image_size_px
            or not np.array_equal(intrinsic.matrix, record.matrix)
            or not np.array_equal(coefficients, record.distortion)
        ):
            raise ValueError("Camera layout and selected lens calibration disagree")
    return json.dumps({"cameras": payload["intrinsic_records"]}).encode()
