"""
Motion Capture Module for UpstreamDrift.

This module provides motion capture integration capabilities,
including FreeMoCap sidecar pipeline support.
"""

from .freemocap_ingest import launcher, output_adapter
from .opencap_ingest import (
    OpenCapLaunchConfig,
    OpenCapLaunchResult,
    OpenCapLauncher,
    OpenCapOutputAdapter,
    OpenCapSidecarNotFoundError,
    run_opencap_sidecar,
)

__all__ = [
    "OpenCapLaunchConfig",
    "OpenCapLaunchResult",
    "OpenCapLauncher",
    "OpenCapOutputAdapter",
    "OpenCapSidecarNotFoundError",
    "launcher",
    "output_adapter",
    "run_opencap_sidecar",
]
