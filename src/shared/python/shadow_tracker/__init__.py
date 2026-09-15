"""Shadow Tracker silhouette-to-forward-dynamics package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .camera_bridge import (
        from_pipeline_camera,
        to_pipeline_camera,
    )
    from .contracts import (
        CANDIDATE_RESULT_SCHEMA_VERSION,
        CAMERA_TRACK_SCHEMA_VERSION,
        FIT_REQUEST_SCHEMA_VERSION,
        FRAME_OBSERVATION_SCHEMA_VERSION,
        REPLAY_AUDIT_SCHEMA_VERSION,
        RESULT_BUNDLE_SCHEMA_VERSION,
        SHOT_SCHEMA_VERSION,
        SUBJECT_BINDING_SCHEMA_VERSION,
        CameraTrack,
        CandidateResult,
        FitRequest,
        ForwardModel,
        FrameObservation,
        ModelCapabilities,
        RenderRequest,
        RenderResult,
        ReplayAudit,
        ResultBundle,
        RolloutRequest,
        RolloutResult,
        SegmentationRequest,
        SegmentationResult,
        Segmenter,
        ShadowTrackerService,
        Shot,
        SilhouetteRenderer,
        SubjectModelBinding,
    )
    from .mask_records import (
        MaskFrame,
    )
    from .source_records import (
        FrameIdentity,
        RightsStatus,
        SourceAsset,
        validate_frame_sequence,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "SourceAsset": ".source_records",
    "FrameIdentity": ".source_records",
    "RightsStatus": ".source_records",
    "validate_frame_sequence": ".source_records",
    "MaskFrame": ".mask_records",
    "to_pipeline_camera": ".camera_bridge",
    "from_pipeline_camera": ".camera_bridge",
    "Shot": ".contracts",
    "FrameObservation": ".contracts",
    "CameraTrack": ".contracts",
    "SubjectModelBinding": ".contracts",
    "FitRequest": ".contracts",
    "ReplayAudit": ".contracts",
    "CandidateResult": ".contracts",
    "ResultBundle": ".contracts",
    "Segmenter": ".contracts",
    "SilhouetteRenderer": ".contracts",
    "ForwardModel": ".contracts",
    "ShadowTrackerService": ".contracts",
    "SegmentationRequest": ".contracts",
    "SegmentationResult": ".contracts",
    "RenderRequest": ".contracts",
    "RenderResult": ".contracts",
    "ModelCapabilities": ".contracts",
    "RolloutRequest": ".contracts",
    "RolloutResult": ".contracts",
    "SHOT_SCHEMA_VERSION": ".contracts",
    "FRAME_OBSERVATION_SCHEMA_VERSION": ".contracts",
    "CAMERA_TRACK_SCHEMA_VERSION": ".contracts",
    "SUBJECT_BINDING_SCHEMA_VERSION": ".contracts",
    "FIT_REQUEST_SCHEMA_VERSION": ".contracts",
    "REPLAY_AUDIT_SCHEMA_VERSION": ".contracts",
    "CANDIDATE_RESULT_SCHEMA_VERSION": ".contracts",
    "RESULT_BUNDLE_SCHEMA_VERSION": ".contracts",
}

__all__ = [
    "CANDIDATE_RESULT_SCHEMA_VERSION",
    "CAMERA_TRACK_SCHEMA_VERSION",
    "FIT_REQUEST_SCHEMA_VERSION",
    "FRAME_OBSERVATION_SCHEMA_VERSION",
    "REPLAY_AUDIT_SCHEMA_VERSION",
    "RESULT_BUNDLE_SCHEMA_VERSION",
    "SHOT_SCHEMA_VERSION",
    "SUBJECT_BINDING_SCHEMA_VERSION",
    "CameraTrack",
    "CandidateResult",
    "FitRequest",
    "ForwardModel",
    "FrameIdentity",
    "FrameObservation",
    "MaskFrame",
    "ModelCapabilities",
    "RenderRequest",
    "RenderResult",
    "ReplayAudit",
    "ResultBundle",
    "RightsStatus",
    "RolloutRequest",
    "RolloutResult",
    "SegmentationRequest",
    "SegmentationResult",
    "Segmenter",
    "ShadowTrackerService",
    "Shot",
    "SilhouetteRenderer",
    "SourceAsset",
    "SubjectModelBinding",
    "from_pipeline_camera",
    "to_pipeline_camera",
    "validate_frame_sequence",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
