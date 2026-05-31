"""Public surface of the mocap source-adapter framework."""

from __future__ import annotations

from src.shared.python.motion_pipeline.sources.base import (
    AdapterContractError,
    LoadedPayload,
    MocapSourceAdapter,
    SourceMetadata,
    UnsupportedFormatError,
)
from src.shared.python.motion_pipeline.sources.registry import (
    detect_format,
    list_formats,
    load_any,
    register_adapter,
    registered_adapters,
    unregister_adapter,
)

# Import side-effects register each adapter. Order matters: the first
# adapter whose ``supports(path)`` returns True wins, so more-specific
# JSON sniffers are imported before generic ones.
from src.shared.python.motion_pipeline.sources.bvh_adapter import BVHAdapter
from src.shared.python.motion_pipeline.sources.trc_adapter import TRCAdapter
from src.shared.python.motion_pipeline.sources.sto_mot_adapter import (
    OpenSimSTOMOTAdapter,
)
from src.shared.python.motion_pipeline.sources.mediapipe_json_adapter import (
    MediaPipeJSONAdapter,
)
from src.shared.python.motion_pipeline.sources.alphapose_json_adapter import (
    AlphaPoseJSONAdapter,
)
from src.shared.python.motion_pipeline.sources.hrnet_json_adapter import (
    HRNetJSONAdapter,
)
from src.shared.python.motion_pipeline.sources.openpose_json_adapter import (
    OpenPoseJSONAdapter,
)
from src.shared.python.motion_pipeline.sources.pose2sim_adapter import (
    Pose2SimAdapter,
    Pose2SimDetector,
    Pose2SimObservations,
    load_pose2sim_calibration,
    load_pose2sim_observations,
)
from src.shared.python.motion_pipeline.sources.csv_adapter import CSVAdapter
from src.shared.python.motion_pipeline.sources.c3d_adapter import C3DAdapter

__all__ = [
    "AdapterContractError",
    "AlphaPoseJSONAdapter",
    "BVHAdapter",
    "C3DAdapter",
    "CSVAdapter",
    "HRNetJSONAdapter",
    "LoadedPayload",
    "MediaPipeJSONAdapter",
    "MocapSourceAdapter",
    "OpenPoseJSONAdapter",
    "OpenSimSTOMOTAdapter",
    "Pose2SimAdapter",
    "Pose2SimDetector",
    "Pose2SimObservations",
    "SourceMetadata",
    "TRCAdapter",
    "UnsupportedFormatError",
    "detect_format",
    "list_formats",
    "load_pose2sim_calibration",
    "load_pose2sim_observations",
    "load_any",
    "register_adapter",
    "registered_adapters",
    "unregister_adapter",
]
