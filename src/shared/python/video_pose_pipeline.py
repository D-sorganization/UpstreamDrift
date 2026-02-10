"""Backward compatibility shim - module moved to gui_pkg.video_pose_pipeline."""

import sys as _sys

from .gui_pkg import video_pose_pipeline as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
