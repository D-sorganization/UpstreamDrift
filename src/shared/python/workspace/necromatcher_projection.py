"""Native image-space review of immutable, source-bound research fits."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.historical_fit import CameraProjection
from src.shared.python.motion_matching.pipeline.plant import get_plant

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary


def project_fit_frame(
    library: NecromatcherLibrary, fit_id: str, frame_index: int
) -> dict[str, Any]:
    """Project the bound native geometry, without certifying the camera or motion.

    Rebuilding a full-body definition must reproduce the actual stored model bytes.
    No geometry is inferred from observed landmarks or substituted from a fixture.
    """
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    fit = library.load_fit(fit_id)
    if type(frame_index) is not int:
        raise IndexError("Source frame index must be an integer")
    try:
        position = fit["frame_indices"].index(frame_index)
    except ValueError as exc:
        raise IndexError("Source frame has no stored fit sample") from exc
    model = library.load_asset(fit["model_id"])
    if model.metadata["engine"] != "mujoco":
        raise ValueError(
            "Native projection currently requires a MuJoCo full-body model"
        )
    try:
        definition = fit["provenance"]["native_definition"]
        original = fit["evidence"]["original_fit"]
        camera_record = original["camera"]
        attachments = original["attachments"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Fit lacks native definition, camera or attachment provenance"
        ) from exc
    model_bytes = json.dumps(definition, allow_nan=False).encode("utf-8")
    xml, _ = export_full_body_mjcf(model_bytes)
    if "sha256:" + hashlib.sha256(xml.encode("utf-8")).hexdigest() != fit["model_hash"]:
        raise ValueError("Fit definition does not reproduce the bound native model")
    if not isinstance(attachments, dict) or not attachments:
        raise ValueError("Fit requires named native marker attachments")
    for body, offset in attachments.values():
        if (
            not isinstance(body, str)
            or np.asarray(offset).shape != (3,)
            or not np.isfinite(np.asarray(offset, dtype=float)).all()
        ):
            raise ValueError(
                "Native marker attachments require a body and finite 3-vector"
            )
    camera = CameraProjection(**camera_record)
    native = get_plant("mujoco", model_bytes)
    if tuple(native.coordinate_order) != tuple(fit["coordinate_order"]):
        raise ValueError("Rebuilt native model coordinate order differs from fit")
    pixels = camera.project(
        native.marker_positions(np.asarray(fit["q"][position]), attachments)
    )
    if not np.isfinite(pixels).all():
        raise ValueError("Native model projection must be finite")
    return {
        "fit_id": fit_id,
        "capture_id": fit["capture_id"],
        "frame_index": frame_index,
        "frame": fit["frames"][position],
        "coordinates": "image_pixels",
        "qualification": "monocular_research_hypothesis",
        "camera_qualified": False,
        "physical_time_qualified": False,
        "points": {
            name: {"x": float(point[0]), "y": float(point[1]), "visibility": None}
            for name, point in zip(attachments, pixels, strict=True)
        },
    }
