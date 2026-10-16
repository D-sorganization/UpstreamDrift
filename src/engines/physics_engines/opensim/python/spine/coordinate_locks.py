"""Locked coordinates that the segmental drive moves (#12761).

A coordinate locked in the model file cannot follow the drive: OpenSim ignores
``setValue`` on it, forward dynamics holds it at zero acceleration while the
inverse-dynamics solver ignores the lock, and OpenSim reports no moment arm
about it.  The Bruno 2015 file locks 105 coordinates, including thoracic
lateral-bending and axial-rotation level coordinates.  A study therefore
detects the locked coordinates the drive moves and either fails naming them
(default) or unlocks exactly those (``unlock_driven_coordinates``).  Holding
them at their locked value instead would change the distribution (the other
levels must absorb the rotation, ADR-0055 composition), so it is a follow-up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.shared.python.engine_core.opensim_runtime import import_opensim

__all__ = [
    "DRIVEN_TOLERANCE",
    "driven_locked_coordinates",
    "format_names",
    "unlock_error",
]

#: A drive series moves a coordinate when its peak-to-peak exceeds this [rad|m].
DRIVEN_TOLERANCE = 1e-9
_MAX_LISTED = 8


def format_names(names: Sequence[str], limit: int = _MAX_LISTED) -> str:
    """``a, b, c`` truncated to ``limit`` names plus a remainder count."""
    shown = ", ".join(names[:limit])
    extra = len(names) - limit
    return shown if extra <= 0 else f"{shown}, ... (+{extra} more)"


def _locked_names(model_path: str | Path) -> tuple[str, ...]:
    osim = import_opensim()
    model = osim.Model(str(model_path))  # keep alive: the set is a view into it
    coords = model.getCoordinateSet()
    return tuple(
        str(coords.get(i).getName())
        for i in range(coords.getSize())
        if coords.get(i).get_locked()
    )


def driven_locked_coordinates(
    model_path: str | Path,
    coordinates: Mapping[str, np.ndarray],
    tolerance: float = DRIVEN_TOLERANCE,
) -> tuple[str, ...]:
    """Coordinates locked in the model whose drive series is not constant.

    Preconditions: ``model_path`` exists; ``tolerance`` is finite and positive.
    Postcondition: the names are in the order of ``coordinates``.
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be finite and positive, got {tolerance!r}")
    if not Path(model_path).is_file():
        raise ValueError(f"model file not found: {model_path}")
    locked = set(_locked_names(model_path))
    return tuple(
        name
        for name, values in coordinates.items()
        if name in locked and float(np.ptp(np.asarray(values))) > tolerance
    )


def unlock_error(names: Sequence[str]) -> ValueError:
    """The error raised when driven coordinates are locked and not unlocked."""
    return ValueError(
        f"{len(names)} coordinates are locked in the model but moved by the "
        f"drive: {format_names(names)}. Forward dynamics would hold them fixed "
        "and the inverse dynamics, static optimisation and joint reactions "
        "would disagree. Set unlock_driven_coordinates=true to unlock exactly "
        "these coordinates, or choose a coupling law that leaves them constant."
    )
