"""OpenCap Ingest Module (#11406).

Provides an isolated sidecar runner and adapter for OpenCap (Stanford NMBL),
conforming to ADR-0053. Spawns opencap-core via subprocess or Docker without
binding any OpenCap symbols or TensorFlow dependencies in UpstreamDrift core.
"""

from .augmenter import (
    OpenCapAugmenterConfig,
    OpenCapMarkerAugmenter,
    OpenCapPipelineResult,
    create_opencap_rig,
    run_opencap_pipeline_from_keypoints,
)
from .launcher import (
    OpenCapLaunchConfig,
    OpenCapLauncher,
    OpenCapLaunchResult,
    OpenCapSidecarNotFoundError,
    run_opencap_sidecar,
)
from .output_adapter import OpenCapOutputAdapter

__all__ = [
    "OpenCapAugmenterConfig",
    "OpenCapLaunchConfig",
    "OpenCapLaunchResult",
    "OpenCapLauncher",
    "OpenCapMarkerAugmenter",
    "OpenCapOutputAdapter",
    "OpenCapPipelineResult",
    "OpenCapSidecarNotFoundError",
    "create_opencap_rig",
    "run_opencap_pipeline_from_keypoints",
    "run_opencap_sidecar",
]
