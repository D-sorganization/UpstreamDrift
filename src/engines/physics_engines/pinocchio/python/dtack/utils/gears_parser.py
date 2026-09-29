"""Deprecated alias for :mod:`.mat_dataset_parser` (issue #4480).

This shim preserves the old import path for one release.  New code must
import from :mod:`dtack.utils.mat_dataset_parser`.
"""

from __future__ import annotations

import warnings

from .mat_dataset_parser import MatDatasetParser

warnings.warn(
    "dtack.utils.gears_parser is deprecated; "
    "import from dtack.utils.mat_dataset_parser instead. "
    "The old module name will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Backwards-compat alias for the renamed class.
GearsParser = MatDatasetParser

__all__ = ["GearsParser", "MatDatasetParser"]
