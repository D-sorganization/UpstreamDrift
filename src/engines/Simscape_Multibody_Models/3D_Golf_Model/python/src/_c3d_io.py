"""Backwards-compatible shim. Use the canonical IO module instead.

Canonical path:
``src/shared/python/sidekick/lab/bio/_c3d_io.py`` (issue #4484).
"""

from sidekick.lab.bio._c3d_io import *  # noqa: F401,F403
from sidekick.lab.bio._c3d_io import (  # noqa: F401
    build_metadata,
    load_c3d,
)

# Legacy alias retained for any out-of-tree callers still importing the
# old name. Prefer ``load_c3d``.
load_c3d_file = load_c3d
