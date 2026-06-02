"""Impact event recording and solver API.

.. deprecated::
    This flat module is a **thin re-export shim** over the canonical
    :mod:`src.shared.python.physics.impact_model` package (#7053). The
    implementation previously lived here as a full copy of
    ``impact_model/solver.py``. The duplicate bodies were deleted to satisfy
    DRY; every symbol below is the *same object* as the canonical
    definition.

    New code should import from ``impact_model`` directly.
"""

from __future__ import annotations

from .impact_model.solver import (
    ImpactRecorder,
    ImpactSolverAPI,
)
from .impact_model.types import ImpactEvent

__all__ = ["ImpactEvent", "ImpactRecorder", "ImpactSolverAPI"]
