"""View a vertical plate trial as a shared :class:`GroundReactionSeries` (OF-01).

The series is the display and comparison currency of the ground-reaction core
(GCV-1).  A vertical plate observes only the vertical load and its CoP, so the
conversion fills the components it cannot see with NaN, never zero: shear,
free moment and every moment about the centre of mass (the plate does not know
where the centre of mass is).  :func:`observability` states this explicitly.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from src.shared.python.biomechanics.ground_reaction import (
    NET_LABEL,
    GroundReactionSeries,
)
from src.shared.python.force_plates.contracts import (
    FOOT_LABELS,
    Array,
    VerticalPlateTrial,
)
from src.shared.python.force_plates.placement import PlatePlacement

__all__ = ["observability", "to_ground_reaction_series"]

_OBSERVED = MappingProxyType(
    {"fx": False, "fy": False, "fz": True, "cop": True, "free_moment": False}
)


def observability() -> dict[str, bool]:
    """Which ground-reaction components a vertical plate observes."""
    return dict(_OBSERVED)


def _vertical(load_n: Array) -> Array:
    out = np.full((load_n.shape[0], 3), np.nan)
    out[:, 2] = load_n
    return out


def to_ground_reaction_series(
    trial: VerticalPlateTrial, placement: PlatePlacement
) -> GroundReactionSeries:
    """Ground-reaction series of ``trial`` placed in the world by ``placement``.

    Keys are the anatomical feet and ``"net"``.  Postconditions: shear, free
    moment, CoM moments and ``com_m`` are NaN; ``in_contact`` is true where the
    load reaches the trial's CoP threshold; CoPs lie on the plate surface.
    """
    if not isinstance(trial, VerticalPlateTrial):
        raise TypeError(
            f"trial must be a VerticalPlateTrial, got {type(trial).__name__}"
        )
    if not isinstance(placement, PlatePlacement):
        raise TypeError(
            f"placement must be a PlatePlacement, got {type(placement).__name__}"
        )
    loads = {**{k: trial.load_n[k] for k in FOOT_LABELS}, NET_LABEL: trial.net_load_n()}
    n = trial.n_samples
    unobserved = {k: np.full((n, 3), np.nan) for k in loads}
    return GroundReactionSeries(
        times_s=np.array(trial.times_s),
        force_n=MappingProxyType({k: _vertical(v) for k, v in loads.items()}),
        cop_m=MappingProxyType({k: placement.to_world(trial.cop_m[k]) for k in loads}),
        free_moment_nm=MappingProxyType(dict(unobserved)),
        moment_about_com_nm=MappingProxyType(dict(unobserved)),
        force_moment_about_com_nm=MappingProxyType(dict(unobserved)),
        in_contact=MappingProxyType(
            {k: v >= trial.cop_min_load_n for k, v in loads.items()}
        ),
        com_m=np.full((n, 3), np.nan),
        ground_height_m=np.full(n, placement.surface_height_m),
    )
