"""Shaft centreline at impact in a club frame shared by every flex (#15150 C5).

The grip-driven run records the chain points at impact (the joint centres grip
to tip, then the shaft tip site: ``n_links + 1`` points) in the rigid club-body
frame, i.e. the frame the prescribed grip motion gives the club at impact, in
which the straight shaft runs along +y from the grip to the head. Because every
flex is driven by the same grip motion, that frame is the same for all of them,
so their shapes compare directly.

:class:`ShaftShape` holds the points in that frame re-expressed as
``(along, lead, droop)`` columns (metres): origin at joint 0 (rigid with the
grip, so always on the straight-shaft axis), ``along`` the straight-shaft axis
grip to head, and ``lead``/``droop`` the axes of :func:`bend_trace.bend_axes`
(lead + with the head ahead, droop + toe-down at impact, #15777). A straight
shaft has zero lead and droop, and the tip's lead and droop equal the bend
trace's values at impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.club_fitting.bend_trace import SHAFT_CLUB, bend_axes
from src.shared.python.club_fitting.grip_driven_shaft import HeadGeometry, ShaftRun

SHAPE_KEY = "points_m"


@dataclass(frozen=True, eq=False)
class ShaftShape:
    """Chain points at impact, grip to tip: ``(n, 3)`` along, lead, droop (m)."""

    points_m: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.points_m, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.all(np.isfinite(p)):
            raise ValueError("points_m must be a finite (n >= 2, 3) array")
        object.__setattr__(self, "points_m", p)

    @property
    def along_m(self) -> np.ndarray:
        return self.points_m[:, 0]

    @property
    def lead_m(self) -> np.ndarray:
        return self.points_m[:, 1]

    @property
    def droop_m(self) -> np.ndarray:
        return self.points_m[:, 2]

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {SHAPE_KEY: self.points_m.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[list[float]]]) -> ShaftShape:
        """Inverse of :meth:`to_dict`. Raises ``ValueError`` on a bad shape."""
        if SHAPE_KEY not in data:
            raise ValueError(f"shaft shape is missing {SHAPE_KEY!r}")
        return cls(np.asarray(data[SHAPE_KEY]))


def shaft_shape(
    points_club_m: np.ndarray, head: HeadGeometry, rigid_rotation: np.ndarray
) -> ShaftShape:
    """Shape of chain points given in the rigid club frame at impact.

    Args:
        points_club_m: ``(n_links + 1, 3)`` joint centres then the tip.
        rigid_rotation: club-to-world rotation at impact (droop sign only).

    Raises:
        ValueError: fewer than two points, or no lead axis (see
            :func:`bend_axes`).
    """
    p = np.asarray(points_club_m, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
        raise ValueError("need (n >= 2, 3) chain points")
    lead_axis, droop_axis = bend_axes(head, np.asarray(rigid_rotation, dtype=float))
    rel = p - p[0]
    return ShaftShape(np.column_stack([rel @ SHAFT_CLUB, rel @ lead_axis,
                                       rel @ droop_axis]))  # fmt: skip


def shaft_shape_from_run(
    run: ShaftRun, head: HeadGeometry, impact_time_s: float
) -> ShaftShape | None:
    """Shape at impact of ``run``; ``None`` when the run has no chain points.

    Raises:
        ValueError: impact before the run starts.
    """
    if run.impact_chain_club_m is None:
        return None
    keep = run.times_s <= impact_time_s
    if not np.any(keep):
        raise ValueError("impact_time_s is before the run starts")
    return shaft_shape(run.impact_chain_club_m, head, run.rigid_rotation[keep][-1])
