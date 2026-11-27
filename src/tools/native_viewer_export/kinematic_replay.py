"""Kinematic replays carry no forces worth drawing (#15150).

A bundle whose every effort is zero moves the plant through a recorded motion
without actuation. The wrenches the overlay provider evaluates on it come from a
contact law applied to a motion nobody loaded, so they are not applied forces.
The export hides the vectors derived from efforts or contact unless the user
asks for them, and says so once in the log and on the HUD. A vector request
draws the vectors it names and nothing else: shading, the grip record, the
palm-thumb arrow and strips stay withheld, and the HUD says the drawn vectors
are not measured forces. Ellipsoids are
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
NOTE_DRAWN = "Kinematic Replay: Drawn Vectors Are Not Measured Forces"
UNAVAILABLE = "kinematic replay: bundle carries no efforts"


def note_for(withheld: Sequence[str], vectors_drawn: bool = False) -> str:
    """HUD note naming the requested visuals withheld besides the vectors;
    ``vectors_drawn`` says the vectors on screen are not measured forces."""
    note = NOTE_DRAWN if vectors_drawn else NOTE
    return f"{note} (also off: {', '.join(withheld)})" if withheld else note


class VectorVisibility(NamedTuple):
    """Vectors drawn, whether a replay withheld them, and whether it is a replay."""

    show_force: bool
    show_torque: bool
    hidden: bool
    replay: bool = False


def is_kinematic_replay(bundle: Any) -> bool:
    """True when every effort of ``bundle`` (steps and finer holds) is zero.

    Raises ``ValueError`` for an empty effort table.
    """
    holds = getattr(bundle, "hold_efforts", None)
    tables = [np.asarray(bundle.efforts), *([] if holds is None else [holds])]
    if tables[0].ndim != 2 or tables[0].shape[0] < 1:
        raise ValueError("bundle efforts must be a non-empty (steps, nv) table")
    return not any(np.any(table) for table in tables)


def vector_visibility(bundle: Any, options: Any) -> VectorVisibility:
    """Force and torque visibility of ``bundle`` under ``options`` (an
    ``OverlayOptions``): its flags stand, except that a kinematic replay draws
    each vector only when that vector was explicitly requested. Logs one
    warning when no vector was requested and both are hidden.
    ``replay`` is true on a kinematic replay whatever the flags: it alone
    decides the withholding of every other effort-derived visual."""
    replay = is_kinematic_replay(bundle)
    hidden = replay and not (options.force_requested or options.torque_requested)
    if hidden:
        logger.warning(
            "%s: the bundle has zero efforts (a kinematic replay), so force and "
            "torque vectors from efforts or contact are not drawn; pass "
            "--force-vectors or --torque-vectors to draw them",
            NOTE,
        )
    show_force = options.show_force and (not replay or options.force_requested)
    show_torque = options.show_torque and (not replay or options.torque_requested)
    return VectorVisibility(show_force, show_torque, hidden, replay)
