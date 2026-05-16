"""Backwards-compatible shim. Use the canonical IO module instead.

Export helpers (``export_dataframe``, ``unit_scale``, ``sanitize_for_csv``,
``validate_export_path``) are now consolidated in
``src/shared/python/sidekick/lab/bio/_c3d_io.py`` (issue #4484).
"""

from sidekick.lab.bio._c3d_io import *  # noqa: F401,F403
from sidekick.lab.bio._c3d_io import (  # noqa: F401
    export_dataframe,
    sanitize_for_csv,
    unit_scale,
    validate_export_path,
)
