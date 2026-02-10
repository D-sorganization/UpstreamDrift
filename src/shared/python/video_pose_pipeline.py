"""Backward compatibility shim - module moved to gui_pkg.video_pose_pipeline."""

import sys as _sys

from .gui_pkg import video_pose_pipeline as _real_module  # noqa: E402
from .gui_pkg.video_pose_pipeline import (  # noqa: F401
    VideoPosePipeline,
    VideoProcessingConfig,
    VideoProcessingResult,
    logger,
)

_sys.modules[__name__] = _real_module
