"""Camera rig capture orchestration for markerless motion capture.

UpstreamDrift owns session orchestration, project persistence and operator
workflows for the mocap lab (ADR-0041, #9422). Camera and capture *contracts*
belong to Tools ``sidekick.lab.mocap`` and are consumed through
:mod:`.tools_bridge` when the pinned Tools release ships them; nothing here
defines a competing schema.

Modules:

- :mod:`.topology` — USB hub-chain facts and the one-camera-per-root-port rule
- :mod:`.plan` — the rig plan: views bound to camera identities and capture modes
- :mod:`.sources` — frame sources (OpenCV/MSMF, deterministic synthetic)
- :mod:`.recorder` — compressed-stream recorders (ffmpeg DirectShow copy)
- :mod:`.session` — barrier-started multi-camera capture with typed outcomes
- :mod:`.bundle` — session bundles: plan, recordings index, manifest, validation
- :mod:`.ingest` — recordings to per-view 2-D observations with provenance
- :mod:`.alignment` — strobe offsets applied to observations as evidence
- :mod:`.sync` — strobe-based alignment of the cameras' arrival clocks
- :mod:`.tools_bridge` — fail-closed export to the Tools mocap schema
"""

from .bundle import BundleCheck, RecordingsIndex, check_bundle, write_bundle
from .plan import CameraBinding, CameraControls, CaptureMode, PlanCheck, RigPlan
from .session import (
    CameraStats,
    CaptureOutcome,
    CaptureSession,
    CaptureTuning,
    SessionManifest,
)
from .sources import Frame, FrameSource, SyntheticFrameSource
from .sync import TimingRecord, ViewTiming, align_views
from .topology import (
    PERIODIC_BUDGET_BYTES,
    RESERVE_BYTES,
    CameraLocation,
    derive_camera,
    parse_dshow_listing,
    predict_streaming,
)

__all__ = [
    "BundleCheck",
    "RecordingsIndex",
    "PERIODIC_BUDGET_BYTES",
    "RESERVE_BYTES",
    "CameraBinding",
    "CameraControls",
    "CameraLocation",
    "CameraStats",
    "CaptureMode",
    "CaptureOutcome",
    "CaptureSession",
    "CaptureTuning",
    "Frame",
    "FrameSource",
    "PlanCheck",
    "RigPlan",
    "SessionManifest",
    "SyntheticFrameSource",
    "TimingRecord",
    "ViewTiming",
    "derive_camera",
    "parse_dshow_listing",
    "predict_streaming",
    "align_views",
    "check_bundle",
    "write_bundle",
]
