"""Simscape sole-contact export to the shared ground-reaction breakdown (#11709).

The canonical ``GolfSwing3D_Kinetic`` has no feet on the ground; the
exploratory ``GS3DX_FullBodyContact`` model has three sole spheres per foot
(heel, toe inside, toe outside). ``scripts/matlab/export_gs3dx_grf_fixture.m``
writes, per contact ``C``, the ground-on-foot force and the contact point in
the world frame (Z-up, SI)::

    GroundContactLogs_<C>Force_1..3, GroundContactLogs_<C>Point_1..3
    COMLogs_GlobalPosition_1..3, GroundContactLogs_GroundHeight

This module only assembles those per-contact samples into the GCV-1
breakdown (:func:`analyze_ground_reaction`); column parsing lives in
:mod:`src.engines.simscape.force_channels`, CoP and free moment in
:mod:`src.shared.python.biomechanics.ground_reaction`.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from src.shared.python.biomechanics.ground_reaction import (
        GroundReactionBreakdown,
    )
    from src.shared.python.force_overlay import OverlayWrench

__all__ = [
    "COM_PREFIX",
    "FOOT_CONTACTS",
    "GROUND_HEIGHT_COLUMN",
    "contact_columns",
    "ground_reaction_series",
    "overlay_wrenches",
]

#: Sole contacts per foot, in the order of the model's FootContactForces.
FOOT_CONTACTS: dict[str, tuple[str, str, str]] = {
    "left": ("LHeel", "LToeIn", "LToeOut"),
    "right": ("RHeel", "RToeIn", "RToeOut"),
}
COM_PREFIX = "COMLogs_GlobalPosition_"
GROUND_HEIGHT_COLUMN = "GroundContactLogs_GroundHeight"


def contact_columns(contact: str, quantity: str) -> tuple[str, str, str]:
    """Column names of one contact's ``Force`` or ``Point`` vector."""
    if quantity not in ("Force", "Point"):
        raise ValueError(f"quantity must be 'Force' or 'Point', got {quantity!r}")
    a, b, c = (f"GroundContactLogs_{contact}{quantity}_{k}" for k in "123")
    return (a, b, c)


def ground_reaction_series(
    forces: Mapping[str, np.ndarray],
    points: Mapping[str, np.ndarray],
    com: np.ndarray,
    ground_height_m: float,
) -> tuple[GroundReactionBreakdown, ...]:
    """Per-sample GCV-1 breakdown with feet labelled ``left`` / ``right``.

    Preconditions: ``forces`` and ``points`` map every contact of
    :data:`FOOT_CONTACTS` to a ``(T, 3)`` world array; ``com`` is ``(T, 3)``;
    ``ground_height_m`` is finite.

    Postconditions: one breakdown per sample; overlay labels from
    ``ground_reaction.to_overlay_wrenches`` are ``contact:grf_left``,
    ``contact:grf_right`` and ``contact:grf_net``.

    Raises:
        ValueError: missing contact, mismatched shapes or bad ground height.
    """
    if not math.isfinite(ground_height_m):
        raise ValueError(f"ground_height_m must be finite, got {ground_height_m!r}")
    names = [c for foot in FOOT_CONTACTS.values() for c in foot]
    absent = [n for n in names if n not in forces or n not in points]
    if absent:
        raise ValueError(f"missing contact data for: {', '.join(absent)}")
    com_arr = np.asarray(com, dtype=float)
    n_t = com_arr.shape[0] if com_arr.ndim == 2 else -1
    arrays = [np.asarray(forces[n]) for n in names] + [
        np.asarray(points[n]) for n in names
    ]
    if com_arr.shape != (n_t, 3) or any(a.shape != (n_t, 3) for a in arrays):
        raise ValueError("forces, points and com must all have shape (T, 3)")
    # Lazy: the biomechanics package is heavy and only needed with contacts.
    from src.shared.python.biomechanics.ground_reaction import (
        ContactSet,
        analyze_ground_reaction,
    )

    return tuple(
        analyze_ground_reaction(
            {
                foot: ContactSet(
                    np.stack([forces[c][t] for c in contacts]),
                    np.stack([points[c][t] for c in contacts]),
                )
                for foot, contacts in FOOT_CONTACTS.items()
            },
            com_arr[t],
            ground_height_m=ground_height_m,
        )
        for t in range(n_t)
    )


def overlay_wrenches(
    breakdown: GroundReactionBreakdown, *, source: str
) -> list[OverlayWrench]:
    """GCV-1 ``CONTACT`` overlay wrenches of one breakdown (shared helper)."""
    from src.shared.python.biomechanics.ground_reaction import to_overlay_wrenches

    return to_overlay_wrenches(breakdown, source=source)
