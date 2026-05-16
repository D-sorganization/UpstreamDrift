"""Backwards-compatible shim. Use the canonical analog module instead.

Force-plate detection and extraction now live in
``src/shared/python/sidekick/lab/bio/_c3d_analog.py`` (issue
#4484).
"""

from sidekick.lab.bio._c3d_analog import *  # noqa: F401,F403
from sidekick.lab.bio._c3d_analog import (  # noqa: F401
    build_force_plate_dataframe,
    detect_force_plate_channels,
    force_plate_columns,
)
