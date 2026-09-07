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
- :mod:`.tools_bridge` — fail-closed export to the Tools mocap schema
"""

from .plan import CameraBinding, CameraControls, CaptureMode, PlanCheck, RigPlan
from .session import CameraStats, CaptureOutcome, CaptureSession, SessionManifest
from .sources import Frame, FrameSource, SyntheticFrameSource
from .topology import (
    PERIODIC_BUDGET_BYTES,
    RESERVE_BYTES,
    CameraLocation,
    derive_camera,
    parse_dshow_listing,
    predict_streaming,
)

__all__ = [
    "PERIODIC_BUDGET_BYTES",
    "RESERVE_BYTES",
    "CameraBinding",
    "CameraControls",
    "CameraLocation",
    "CameraStats",
    "CaptureMode",
    "CaptureOutcome",
    "CaptureSession",
    "Frame",
    "FrameSource",
    "PlanCheck",
    "RigPlan",
    "SessionManifest",
    "SyntheticFrameSource",
    "derive_camera",
    "parse_dshow_listing",
    "predict_streaming",
]
