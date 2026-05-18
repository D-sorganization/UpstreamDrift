"""Backwards-compatible shim. Use the canonical models module instead.

Canonical path:
``src/shared/python/sidekick/lab/bio/_c3d_models.py`` (issue #4484).
"""

from sidekick.lab.bio._c3d_models import *  # noqa: F401,F403
from sidekick.lab.bio._c3d_models import (  # noqa: F401
    BIOMECHANICAL_MARKER_MAX_M,
    BIOMECHANICAL_MARKER_MIN_M,
    SCHEMA_VERSION,
    C3DEvent,
    C3DMetadata,
)

# The canonical module does not define ``C3DMapping`` since it lives in the
# IO layer there; keep the legacy alias for shim compatibility.
C3DMapping = dict
