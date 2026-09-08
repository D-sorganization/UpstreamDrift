"""Default mapping from video keypoint names to swing-model markers (Phase 3).

The swing rig has six markers (:data:`symbolic_model.MARKER_NAMES`); the
markerless pipeline emits keypoints in one of the schemas of
:data:`motion_pipeline.contracts.SchemaName`. This table names, per schema,
which keypoint drives which marker. Keypoints absent from the table are
ignored by :func:`tracking_ocp.build_tracking_ocp` with a warning; a marker
with no source keypoint is simply not tracked.

The chain is a single lead-arm model, so the *lead* (left, for a
right-handed golfer) shoulder, elbow and wrist are used. ``hip`` and
``trunk`` are driven by the same pelvis/torso landmarks when present.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["DEFAULT_KEYPOINT_MAP", "keypoint_map_for_schema"]

_MEDIAPIPE_33: dict[str, str] = {
    "left_hip": "hip",
    "left_shoulder": "shoulder",
    "left_elbow": "elbow",
    "left_wrist": "wrist",
    "clubhead": "clubhead",
}

_COCO_17: dict[str, str] = {
    "left_hip": "hip",
    "left_shoulder": "shoulder",
    "left_elbow": "elbow",
    "left_wrist": "wrist",
}

_BODY_25: dict[str, str] = {
    "MidHip": "hip",
    "Neck": "trunk",
    "LShoulder": "shoulder",
    "LElbow": "elbow",
    "LWrist": "wrist",
}

#: Marker-name identity map: keypoints already named like markers pass through.
_IDENTITY: dict[str, str] = {
    "hip": "hip",
    "trunk": "trunk",
    "shoulder": "shoulder",
    "elbow": "elbow",
    "wrist": "wrist",
    "clubhead": "clubhead",
    "clubface": "clubface",
}

DEFAULT_KEYPOINT_MAP: Mapping[str, Mapping[str, str]] = {
    "MediaPipe_33": _MEDIAPIPE_33,
    "COCO_17": _COCO_17,
    "BODY_25": _BODY_25,
    "OpenPose_25": _BODY_25,
    "custom": _IDENTITY,
}


def keypoint_map_for_schema(schema_name: str) -> dict[str, str]:
    """``keypoint name -> marker name`` for ``schema_name`` (identity fallback)."""
    table = DEFAULT_KEYPOINT_MAP.get(schema_name, _IDENTITY)
    return {**_IDENTITY, **table}
