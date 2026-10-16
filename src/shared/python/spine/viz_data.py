"""Read-only visualization adapter for spine kinematics and loads (SPN-15a).

Pure data: this module imports no plotting or GUI library and holds no model.
It turns the merged spine products into time-by-level arrays that a figure
(:mod:`src.shared.python.spine.plots`) or a later GUI (SPN-15b, #12772) can draw
without recomputing anything.

* :func:`level_angle_stack` evaluates
  :func:`~src.shared.python.spine.golf_terms.segmental_contributions` for a
  sequence of poses. Each column is a per-level *decomposition share* of the
  global joint-coordinate-system term (it telescopes to the global value), not
  the level's own Cardan angle and not additive in angle space. Degrees.
* :func:`load_heatmap` exposes one component of a
  :class:`~src.shared.python.spine.loads.SpinalLoadSeries`. The series
  convention (ADR-0055 item 5) is compression ``+F_y``, AP shear ``+F_x``
  (anterior), ML shear ``+F_z`` (right), wrench on the superior vertebra by
  the inferior one in the inferior ISB frame. The tension-positive view used by
  ``body_part_viz/axial_loads.py`` is produced only by
  :func:`to_tension_positive` (``tension = -compression``); the sign is
  declared in :attr:`HeatmapData.sign_convention`.

Every product carries a non-empty ``distribution_source`` and a
``scientific_status``. Status is fail-closed: it defaults to ``"unqualified"``,
only ``"unqualified"`` and ``"qualified"`` are accepted, and nothing in this
module promotes one to the other; the caller must pass ``"qualified"``
explicitly (for example from a receipt that earned it).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.contracts import ensure
from src.shared.python.spine.golf_terms import (
    contribution_remainder,
    segmental_contributions,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.loads import SpinalLoadSeries
from src.shared.python.spine.segmental import SegmentalPose

__all__ = [
    "SCIENTIFIC_STATUSES",
    "HeatmapData",
    "StackData",
    "level_angle_stack",
    "load_heatmap",
    "to_tension_positive",
]

SCIENTIFIC_STATUSES = ("unqualified", "qualified")
_QUANTITIES = ("compression", "tension", "ap_shear", "ml_shear")
_SIGN = {
    "compression": "compression-positive",
    "tension": "tension-positive",
    "ap_shear": "anterior-positive",
    "ml_shear": "right-positive",
}


def _check_status(scientific_status: object) -> str:
    if scientific_status not in SCIENTIFIC_STATUSES:
        raise ValueError(
            f"scientific_status must be one of {SCIENTIFIC_STATUSES}, "
            f"got {scientific_status!r}"
        )
    return str(scientific_status)


def _check_source(distribution_source: object) -> str:
    if not isinstance(distribution_source, str):
        raise TypeError("distribution_source must be a string")
    if not distribution_source.strip():
        raise ValueError("distribution_source must be a non-empty string")
    return distribution_source


def _readonly(value: ArrayLike) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _status_label(scientific_status: str) -> str:
    return "QUALIFIED" if scientific_status == "qualified" else "UNQUALIFIED"


@dataclass(frozen=True)
class StackData:
    """Per-level angle contributions, time by level (caudal to cranial).

    ``values_deg`` has shape ``(n, L)``; ``remainder_deg`` (shape ``(n,)``) is
    the global term minus the row sum (~0 by construction, NaN where a prefix
    is gimbal-locked). Arrays are read-only. The stack attributes the supplied
    chain only (``B = E = A = I``); use
    :func:`~src.shared.python.spine.golf_terms.segmental_contribution_breakdown`
    when a base, terminal or alignment rotation must be attributed.
    """

    time_s: np.ndarray
    levels: tuple[SpinalLevel, ...]
    values_deg: np.ndarray
    remainder_deg: np.ndarray
    term_id: str
    handedness: str
    distribution_source: str
    scientific_status: str
    units: str = "deg"

    def total_deg(self) -> np.ndarray:
        """Row sums of the contributions (the global term), shape ``(n,)``."""
        return self.values_deg.sum(axis=1)

    def status_label(self) -> str:
        """``"QUALIFIED"`` only if declared so, else ``"UNQUALIFIED"``."""
        return _status_label(self.scientific_status)


@dataclass(frozen=True)
class HeatmapData:
    """One load component, time by level (caudal to cranial).

    ``values`` has shape ``(n, L)`` in ``units`` (``"N"`` or ``"BW"``) under
    the named ``sign_convention``. Arrays are read-only.
    """

    time_s: np.ndarray
    levels: tuple[SpinalLevel, ...]
    values: np.ndarray
    quantity: str
    units: str
    sign_convention: str
    observable: str
    distribution_source: str
    scientific_status: str

    def status_label(self) -> str:
        """``"QUALIFIED"`` only if declared so, else ``"UNQUALIFIED"``."""
        return _status_label(self.scientific_status)


def to_tension_positive(compression: ArrayLike) -> np.ndarray:
    """Convert compression-positive values to tension-positive (``-c``).

    The single sign boundary between the spine load contract (``+F_y`` is
    compression) and ``body_part_viz/axial_loads.py`` consumers. Returns a new
    array; the input is unchanged.

    Raises:
        ValueError: if any value is not finite.
    """
    array = np.asarray(compression, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("compression must be finite (no NaN or inf)")
    return -array


def _check_time(time_s: ArrayLike, n_poses: int) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or time.size == 0 or n_poses == 0:
        raise ValueError("time_s and poses must be non-empty and time_s 1-D")
    if time.size != n_poses:
        raise ValueError(
            f"time_s and poses must have the same length, got {time.size} and {n_poses}"
        )
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    return time


def _check_poses(poses: Sequence[SegmentalPose]) -> tuple[SpinalLevel, ...]:
    if any(not isinstance(p, SegmentalPose) for p in poses):
        raise TypeError("every pose must be a SegmentalPose")
    levels = poses[0].levels
    if any(p.levels != levels for p in poses):
        raise ValueError("all poses must cover the same levels")
    return levels


def level_angle_stack(
    time_s: ArrayLike,
    poses: Sequence[SegmentalPose],
    term_id: str,
    *,
    handedness: Literal["right", "left"],
    distribution_source: str,
    scientific_status: str = "unqualified",
) -> StackData:
    """Per-level contributions of a JCS golf term over time.

    Preconditions: ``time_s`` is finite, strictly increasing and as long as
    ``poses``; ``poses`` is a non-empty sequence of :class:`SegmentalPose` over
    identical levels; ``term_id`` has a segmental decomposition (see
    :func:`segmental_contributions`); ``distribution_source`` is a non-empty
    string; ``scientific_status`` is ``"unqualified"`` (default) or
    ``"qualified"``. Postconditions: ``values_deg`` is ``(n, L)`` in degrees,
    columns caudal to cranial, read-only; metadata is carried unchanged.

    Raises:
        TypeError, ValueError: on a violated precondition.
    """
    source = _check_source(distribution_source)
    status = _check_status(scientific_status)
    time = _check_time(time_s, len(poses))
    levels = _check_poses(poses)
    rows = [segmental_contributions(term_id, p, handedness=handedness) for p in poses]
    values = np.array([[row[lv] for lv in levels] for row in rows], dtype=float)
    remainder = [
        contribution_remainder(term_id, p, handedness=handedness) for p in poses
    ]
    result = StackData(
        time_s=_readonly(time),
        levels=levels,
        values_deg=_readonly(values),
        remainder_deg=_readonly(remainder),
        term_id=term_id,
        handedness=handedness,
        distribution_source=source,
        scientific_status=status,
    )
    ensure(
        result.values_deg.shape == (time.size, len(levels)),
        "stack must be (n, L)",
        result.values_deg.shape,
    )
    return result


def load_heatmap(
    series: SpinalLoadSeries,
    quantity: str,
    *,
    scientific_status: str = "unqualified",
) -> HeatmapData:
    """Time-by-level array of one load component of ``series``.

    ``quantity`` is ``compression`` (``+F_y``), ``tension`` (``-F_y``, via
    :func:`to_tension_positive`), ``ap_shear`` (``+F_x``) or ``ml_shear``
    (``+F_z``). Units follow the series (``N`` or ``BW``). Provenance
    ``distribution_source`` and the observable are carried through;
    ``scientific_status`` defaults to ``"unqualified"``.

    Raises:
        TypeError: if ``series`` is not a :class:`SpinalLoadSeries`.
        ValueError: for an unknown ``quantity`` or ``scientific_status``.
    """
    if not isinstance(series, SpinalLoadSeries):
        raise TypeError(
            f"series must be a SpinalLoadSeries, got {type(series).__name__}"
        )
    if quantity not in _QUANTITIES:
        raise ValueError(f"quantity must be one of {_QUANTITIES}, got {quantity!r}")
    status = _check_status(scientific_status)
    source = _check_source(series.provenance.get("distribution_source"))
    values = {
        "compression": series.compression_n,
        "tension": lambda: to_tension_positive(series.compression_n()),
        "ap_shear": series.ap_shear_n,
        "ml_shear": series.ml_shear_n,
    }[quantity]()
    result = HeatmapData(
        time_s=_readonly(series.time_s),
        levels=series.levels,
        values=_readonly(values),
        quantity=quantity,
        units=series.force_unit,
        sign_convention=_SIGN[quantity],
        observable=series.observable.value,
        distribution_source=source,
        scientific_status=status,
    )
    ensure(
        result.values.shape == (series.time_s.size, len(series.levels)),
        "heatmap must be (n, L)",
        result.values.shape,
    )
    return result
