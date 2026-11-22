"""Kinematic replays carry no forces worth drawing (#15150).

A bundle whose every effort is zero moves the plant through a recorded motion
without actuation. The wrenches the overlay provider evaluates on it come from a
contact law applied to a motion nobody loaded, so they are not applied forces.
The export hides the vectors derived from efforts or contact unless the user
asks for them, and says so once in the log and on the HUD. Ellipsoids are
inertial properties of the pose and stay.

Detection reads the efforts, not a provenance label: the bundle writers share no
label, one can be absent or stale, and the efforts are what the forces depend on.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

NOTE = "Kinematic Replay: Forces Not Shown"
#: why a force-derived visual (shading, grip record, strip) is unavailable
UNAVAILABLE = "kinematic replay: bundle carries no efforts"


def note_for(withheld: Sequence[str]) -> str:
    """HUD note naming the requested visuals withheld besides the vectors."""
    return f"{NOTE} (also off: {', '.join(withheld)})" if withheld else NOTE


class VectorVisibility(NamedTuple):
    """Vectors drawn, and whether a kinematic replay withheld them."""

    show_force: bool
    show_torque: bool
    hidden: bool


def is_kinematic_replay(bundle: Any) -> bool:
    """True when every effort of ``bundle`` (steps and finer holds) is zero.

    Raises ``ValueError`` for an empty effort table.
    """
    holds = getattr(bundle, "hold_efforts", None)
    tables = [np.asarray(bundle.efforts), *([] if holds is None else [holds])]
    if tables[0].ndim != 2 or tables[0].shape[0] < 1:
        raise ValueError("bundle efforts must be a non-empty (steps, nv) table")
    return not any(np.any(table) for table in tables)


def withholds_visuals(bundle: Any, vectors_requested: bool) -> bool:
    """True when ``bundle`` is a kinematic replay and no vector flag was passed:
    shading, grip wrench, palm-thumb arrow and strips then carry no meaning."""
    return not vectors_requested and is_kinematic_replay(bundle)


def vector_visibility(bundle: Any, options: Any) -> VectorVisibility:
    """Force and torque visibility of ``bundle`` under ``options`` (an
    ``OverlayOptions``): its flags stand, except that a kinematic replay hides
    both unless ``forces_requested``. Logs one warning when it hides them."""
    hidden = is_kinematic_replay(bundle) and not options.forces_requested
    if hidden:
        logger.warning(
            "%s: the bundle has zero efforts (a kinematic replay), so force and "
            "torque vectors from efforts or contact are not drawn; pass "
            "--force-vectors or --torque-vectors to draw them",
            NOTE,
        )
    return VectorVisibility(
        options.show_force and not hidden, options.show_torque and not hidden, hidden
    )
