"""Lead and droop of the shaft tip through the swing (#15150 C12, C7).

The tip deflection of a :class:`ShaftRun` (bent minus straight club) is split
in the rigid club frame (shaft along +y, grip to head): lead along the face
normal made perpendicular to the shaft (+ = head ahead, toward the target at
impact); droop along the remaining axis, signed to point down in the world at
impact (+ = toe-down). The trace ends at impact, after which the grip motion is
extrapolated and the bend is not physical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.club_fitting.grip_driven_shaft import HeadGeometry, ShaftRun

_SHAFT_CLUB = np.array([0.0, 1.0, 0.0])
_MIN_LEAD_AXIS = 1e-6
MM_PER_M = 1e3
_KEYS = ("time_s", "lead_mm", "droop_mm")


@dataclass(frozen=True, eq=False)
class BendTrace:
    """Lead and droop (mm) against time from impact (s; impact at 0)."""

    time_s: np.ndarray
    lead_mm: np.ndarray
    droop_mm: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(a, dtype=float) for a in self.to_dict().values()]
        if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 1:
            raise ValueError("time_s, lead_mm and droop_mm must be the same length")
        if arrays[0].size == 0 or not np.all(np.isfinite(arrays)):
            raise ValueError("a bend trace must be non-empty and finite")
        for name, a in zip(_KEYS, arrays, strict=True):
            object.__setattr__(self, name, a)

    def at_impact(self) -> tuple[float, float]:
        """``(lead_mm, droop_mm)`` interpolated at time 0."""
        return (
            float(np.interp(0.0, self.time_s, self.lead_mm)),
            float(np.interp(0.0, self.time_s, self.droop_mm)),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {k: np.asarray(getattr(self, k), dtype=float).tolist() for k in _KEYS}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> BendTrace:
        """Inverse of :meth:`to_dict`. Raises ``ValueError`` on a bad shape."""
        try:
            return cls(*(np.asarray(data[k]) for k in _KEYS))
        except KeyError as exc:
            raise ValueError(f"bend trace is missing {exc}") from exc


def bend_trace(run: ShaftRun, head: HeadGeometry, impact_time_s: float) -> BendTrace:
    """Lead and droop of ``run`` up to ``impact_time_s``.

    Raises:
        ValueError: impact before the run starts, or a face normal along the
            shaft (no lead direction).
    """
    keep = run.times_s <= impact_time_s
    if not np.any(keep):
        raise ValueError("impact_time_s is before the run starts")
    normal = head.normal_club
    lead_axis = normal - (normal @ _SHAFT_CLUB) * _SHAFT_CLUB
    if np.linalg.norm(lead_axis) < _MIN_LEAD_AXIS:
        raise ValueError("the face normal runs along the shaft; no lead axis")
    lead_axis = lead_axis / np.linalg.norm(lead_axis)
    droop_axis = np.cross(_SHAFT_CLUB, lead_axis)
    rotation = run.rigid_rotation[keep]
    if (rotation[-1] @ droop_axis)[2] > 0.0:
        droop_axis = -droop_axis
    local = np.einsum("nji,nj->ni", rotation, run.tip_deflection_m[keep])
    return BendTrace(
        run.times_s[keep] - impact_time_s,
        local @ lead_axis * MM_PER_M,
        local @ droop_axis * MM_PER_M,
    )
