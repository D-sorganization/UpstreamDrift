"""Saved expert video and motion references for camera comparison."""

from .comparison import (
    COMPARISON_EXPORT_SCHEMA,
    COMPARISON_SESSION_SCHEMA,
    ComparisonLayer,
    ComparisonSession,
    build_comparison_sidecar,
    comparison_session_path,
    load_comparison_session,
    save_comparison_session,
)
from .model import Asset, ReferenceMotion, ReferenceSource, ReferenceVideo
from .registration import (
    EventAnchors,
    ReferenceRegistration,
    ReferenceTransform,
    TimeMapping,
    canonical_z_up_to_adr0041_world,
    project_reference_to_camera,
    sample_reference_motion,
    transform_reference_motion,
)

__all__ = [
    "COMPARISON_EXPORT_SCHEMA",
    "COMPARISON_SESSION_SCHEMA",
    "Asset",
    "ComparisonLayer",
    "ComparisonSession",
    "EventAnchors",
    "ReferenceMotion",
    "ReferenceRegistration",
    "ReferenceSource",
    "ReferenceTransform",
    "ReferenceVideo",
    "TimeMapping",
    "build_comparison_sidecar",
    "canonical_z_up_to_adr0041_world",
    "comparison_session_path",
    "load_comparison_session",
    "project_reference_to_camera",
    "sample_reference_motion",
    "save_comparison_session",
    "transform_reference_motion",
]
