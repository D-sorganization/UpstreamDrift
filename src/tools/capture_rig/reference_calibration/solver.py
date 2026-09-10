"""Worker-only consumption of Tools calibration; no camera solver lives here."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from shared.python.sidekick.lab.mocap.calibration import (
    DistortionCoefficients,
    DistortionModel,
    PinholeIntrinsics,
)
from shared.python.sidekick.lab.mocap.geometry import CoordinateFrame, RigidTransform
from shared.python.sidekick.lab.mocap.reference_placements import (
    ReferenceCamera,
    ReferenceLayoutResult,
    estimate_reference_layout,
)

from src.motion_capture.reconstruct.intrinsics import IntrinsicsRecord
from src.motion_capture.rig.documents import write_document
from src.shared.python.pose_estimation.observations import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
)
from src.shared.python.spatial_algebra.pose6dof import quaternion_to_rotation_matrix

from ..calibration_profiles import ProfileAssignment, write_profile_set
from .operations import verify_samples
from .session import ReferenceSession, save_revision
from .results import load_result
from .source_evidence import algorithm_evidence


def accept_result(request: dict[str, Any]) -> dict[str, Any]:
    """Publish an operator-reviewed copy after rechecking source bindings."""
    root = Path(request["workspace"])
    session = ReferenceSession.model_validate(request["session"])
    parameters = request["parameters"]
    if (
        parameters.get("reviewed") is not True
        or parameters.get("anchor_confirmed") is not True
    ):
        raise ValueError(
            "Review the camera evidence and world anchor before using the layout"
        )
    loaded = load_result(request)
    path = root / loaded["result_path"]
    document = loaded["result"]
    if parameters.get("result_sha256") != loaded["result_sha256"]:
        raise ValueError("Camera estimate changed since review; reopen it before use")
    if (
        document["anchor_placement_id"] != parameters["anchor_placement_id"]
        or document["anchor"]["translation_m"] != parameters["anchor_translation_m"]
    ):
        raise ValueError("World anchor changed; estimate the camera positions again")
    verify_samples(root, session)
    _intrinsics(session, confirmed=parameters.get("settings_confirmed", False))
    document.update(
        operator_reviewed=True,
        reviewed_utc=datetime.now(UTC).isoformat(),
        source_result_sha256=loaded["result_sha256"],
    )
    out = path.with_name(f"reviewed-{uuid4()}.json")
    write_document(out, document)
    return {"result_path": out.relative_to(root).as_posix(), "result": document}


def _intrinsics(
    session: ReferenceSession, *, confirmed: bool
) -> tuple[
    dict[str, ReferenceCamera], dict[str, IntrinsicsRecord], list[dict[str, Any]]
]:
    """Use the existing profile validator and export, retaining full distortion."""
    if confirmed is not True:
        raise ValueError(
            "Confirm the recorded lens, zoom, focus and camera positions before solving"
        )
    assignments = []
    for camera in session.cameras:
        if camera.profile is None:
            raise ValueError(f"{camera.view}: select a calibrated lens profile first")
        if any(
            getattr(camera.setup, field).casefold() in {"unknown", "?", "n/a"}
            for field in ("lens", "zoom", "focus", "sensor_mode")
        ):
            raise ValueError(
                f"{camera.view}: unknown optics cannot be used for a camera solve"
            )
        assignments.append(
            ProfileAssignment(camera.view, camera.profile, camera.setup, True)
        )
    with TemporaryDirectory(prefix="reference-intrinsics-") as temporary:
        path = Path(temporary) / "reviewed.json"
        write_profile_set(
            path,
            assignments,
            required_views=tuple(item.view for item in session.cameras),
        )
        payload = json.loads(path.read_bytes())
    records = {
        item["camera_id"]: IntrinsicsRecord.model_validate(item)
        for item in payload["cameras"]
    }
    result = {}
    for camera in session.cameras:
        record = records[camera.view]
        matrix = record.matrix
        coefficients = tuple(record.distortion)
        model = (
            DistortionModel.RATIONAL
            if len(coefficients) >= 8
            else DistortionModel.BROWN_CONRADY
        )
        lens = PinholeIntrinsics(
            matrix[0][0],
            matrix[1][1],
            matrix[0][2],
            matrix[1][2],
            record.image_size_px,
            DistortionCoefficients(model, coefficients),
            skew=matrix[0][1],
        )
        result[camera.view] = ReferenceCamera(camera.view, lens, camera.profile_key)
    return result, records, payload["profile_selections"]


def camera_records(
    result: ReferenceLayoutResult, records: dict[str, IntrinsicsRecord]
) -> list[dict[str, Any]]:
    """Adapt camera-from-world to the existing world-from-camera record once."""
    output = []
    poses = result.layout.camera_poses
    for view, pose in poses.items():
        lens = records[view]
        transform = pose.t_camera_from_world
        rotation = quaternion_to_rotation_matrix(list(transform.rotation_wxyz)).T
        calibration = CameraCalibration(
            camera_id=view,
            intrinsics=CameraIntrinsics(
                np.asarray(lens.matrix), np.asarray(lens.distortion)
            ),
            extrinsics=CameraExtrinsics(rotation, np.asarray(pose.camera_center_world)),
            image_size_px=lens.image_size_px,
        )
        output.append(calibration.to_dict())
    return output


def solve_reference(request: dict[str, Any]) -> dict[str, Any]:
    """Run the canonical fixed-intrinsics solver in this cancellable process.

    The UI cancels by terminating the isolated process. No file is published
    until the solve and source revalidation finish. A result is numerical
    evidence requiring operator review, not a physical-accuracy certificate.
    """
    root = Path(request["workspace"])
    session = ReferenceSession.model_validate(request["session"])
    parameters = request["parameters"]
    if parameters.get("anchor_confirmed") is not True:
        raise ValueError(
            "Confirm the anchor's measured offset, flat face and target-pointing arrow"
        )
    verify_samples(root, session)
    cameras, records, selections = _intrinsics(
        session, confirmed=parameters.get("settings_confirmed", False)
    )
    source_evidence = algorithm_evidence()
    anchor_id = parameters["anchor_placement_id"]
    world = CoordinateFrame.affinedrift_world_v1()
    anchor = RigidTransform(
        world.frame_id,
        f"placement:{anchor_id}",
        (1.0, 0.0, 0.0, 0.0),
        tuple(parameters["anchor_translation_m"]),
    )
    result = estimate_reference_layout(
        layout_id=str(uuid4()),
        world_frame=world,
        targets={target.reference_id: target for target in session.targets},
        cameras=cameras,
        observations=tuple(
            item.observation for item in session.samples if item.enabled
        ),
        anchor_placement_id=anchor_id,
        anchor=anchor,
    )
    verify_samples(root, session)
    _intrinsics(session, confirmed=True)
    if algorithm_evidence() != source_evidence:
        raise ValueError(
            "Calibration source changed during estimation; retry with a stable runtime"
        )
    revision_path = save_revision(root, session, capture_id=session.capture_id)
    document = {
        "schema_version": "capture-reference-solve/1",
        "capture_id": session.capture_id,
        "reference_revision_id": str(session.revision_id),
        "scene_id": session.scene_id,
        "revision_file_sha256": hashlib.sha256(revision_path.read_bytes()).hexdigest(),
        "session_sha256": hashlib.sha256(
            session.model_dump_json().encode()
        ).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "layout_id": result.layout.layout_id,
        "world_frame": asdict(world),
        "anchor": asdict(anchor),
        "anchor_placement_id": anchor_id,
        "cameras": camera_records(result, records),
        "profile_selections": selections,
        "intrinsic_records": [
            item.model_dump(mode="json") for item in records.values()
        ],
        "residuals": [asdict(item) for item in result.residuals],
        "limitations": list(result.limitations),
        "operator_reviewed": False,
        "placement_transforms": {
            key: asdict(value) for key, value in result.placement_transforms.items()
        },
        "opencv_version": cv2.__version__,
        "algorithm_evidence": source_evidence,
    }
    folder = root / "reference_calibration" / "results"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result.layout.layout_id}.json"
    write_document(path, document)
    return {
        "result_path": path.relative_to(root).as_posix(),
        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "result": document,
    }
