"""Deprecated alias for :mod:`.swing_dataset_viewer` (issue #4480).

This shim preserves the old import path for one release.  New code must
import from :mod:`dtack.viz.swing_dataset_viewer`.
"""

from __future__ import annotations

import warnings

from .swing_dataset_viewer import SwingDatasetViewer

warnings.warn(
    "dtack.viz.rob_neal_viewer is deprecated; "
    "import from dtack.viz.swing_dataset_viewer instead. "
    "The old module name will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Backwards-compat alias for the renamed class.
RobNealDataViewer = SwingDatasetViewer

__all__ = ["RobNealDataViewer", "SwingDatasetViewer"]
