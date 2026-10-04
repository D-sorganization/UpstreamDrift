"""
Motion Capture Module for UpstreamDrift.

This module provides motion capture integration capabilities,
including FreeMoCap sidecar pipeline support.
"""

from .freemocap_ingest import launcher, output_adapter
from .opencap_ingest import (
    OpenCapAugmenterConfig,
    OpenCapLaunchConfig,
    OpenCapLaunchResult,
    OpenCapLauncher,
    OpenCapMarkerAugmenter,
    OpenCapOutputAdapter,
    OpenCapPipelineResult,
    OpenCapSidecarNotFoundError,
    create_opencap_rig,
    run_opencap_pipeline_from_keypoints,
    run_opencap_sidecar,
)

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
    "launcher",
    "output_adapter",
    "run_opencap_pipeline_from_keypoints",
    "run_opencap_sidecar",
]
