"""Playability window -- the primary scalar objective (issue #8614, W7).

Acushnet's adjustable-bounce patent US11766593B1 names both failure modes in one
breath: too little effective bounce and the head digs, losing clubhead speed;
too much and the leading edge reaches the ball. That is a **two-sided
constraint**, so the design goal is not to optimise a point value but to
**maximise the width of the acceptable window** across the delivery and
sand-condition distribution. This module measures that width as an area.

Definition
----------

Over a rectangular grid of two delivery or condition factors -- ``(entry
distance x attack angle)`` and ``(entry distance x sand firmness)`` are the two
the research names -- the window is the set of grid points whose carry lands
within ``tolerance_fraction`` of the target:

```
in_window(i, j)  <=>  |carry(i, j) - target| <= tolerance_fraction * target
```

Its **area** is the integral of that indicator over the grid, evaluated with
trapezoidal node weights, so a fully in-window grid returns exactly
``span_a * span_b`` and the measure is independent of how finely the grid was
sampled. Units are the product of the two axis units, e.g. m.rad for entry
distance against attack angle.

``area`` is the number to optimise: it is the formal version of what a fitter
means by forgiveness, and it is what separates a K-grind from a T-grind.
:func:`playability_objective` exposes it as a plain float for the study layer's
optimiser (:mod:`bunkershot3d.study.optimisation`).

Two honesty features:

* A **NaN carry means no carry was produced**, and the two ways that happens
  are different claims, so they are reported as two numbers. ``refused_fraction``
  is ADR-0032 refusal: the solver declined the query as outside its validity
  envelope. ``unmeasured_fraction`` is the solver answering and the answer not
  supporting a carry -- a head that buried, whose trace reverses inside its own
  crater, leaving no prismatic divot for the launch to divide by. Both count as
  outside the window; a large fraction of either means the window is unmeasured,
  not wide.

  These were one number until issue #9247. Nothing distinguished them while the
  delivery frame was mirrored, because every delivery in the registered sweep
  planed and burial was unreachable, so every NaN really was a refusal. Once the
  frame was corrected the steep end of the attack sweep buries, and the single
  number reported "the solver refused half this domain" about a solver that had
  refused nothing.
* The **largest connected** window is reported alongside the total. A design
  whose window is scattered islands is not forgiving, however much area the
  islands add up to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = [
    "DEFAULT_CARRY_TOLERANCE_FRACTION",
    "PlayabilityAxis",
    "PlayabilityWindow",
    "playability_objective",
    "playability_window",
]

#: Carry tolerance defining the window: +/-10 % of target.
DEFAULT_CARRY_TOLERANCE_FRACTION = 0.10


@dataclass(frozen=True)
class PlayabilityAxis:
    """One swept factor of the playability grid.

    Attributes:
        name: Factor name, e.g. ``"entry_distance"``.
        unit: SI unit symbol, e.g. ``"m"`` or ``"rad"``.
        values: ``(n,)`` strictly increasing sample stations.
    """

    name: str
    unit: str
    values: np.ndarray

    def __post_init__(self) -> None:
        """Validate the axis.

        Raises:
            ValueError: If the axis has fewer than two stations, holds a
                non-finite value, or is not strictly increasing (the node
                weights would then be negative and the area meaningless).
        """
        values = np.asarray(self.values, dtype=float).reshape(-1)
        if values.size < 2:
            raise ValueError(
                f"axis {self.name!r} needs at least 2 stations, got {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"axis {self.name!r} must be finite")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError(f"axis {self.name!r} must be strictly increasing")
        object.__setattr__(self, "values", values)

    @property
    def span(self) -> float:
        """Distance from the first station to the last."""
        return float(self.values[-1] - self.values[0])

    @property
    def node_weights(self) -> np.ndarray:
        """Trapezoidal weights, one per station; they sum to :attr:`span`."""
        values = self.values
        weights = np.empty_like(values)
        weights[0] = 0.5 * (values[1] - values[0])
        weights[-1] = 0.5 * (values[-1] - values[-2])
        weights[1:-1] = 0.5 * (values[2:] - values[:-2])
        return weights


@dataclass(frozen=True)
class PlayabilityWindow:
    """The measured window, and the pieces it was measured from.

    Attributes:
        axis_a: First swept factor (rows of ``carry_m``).
        axis_b: Second swept factor (columns of ``carry_m``).
        carry_m: ``(na, nb)`` carry distance [m]; NaN where the solver refused.
        target_carry_m: Carry the shot is aimed at [m].
        tolerance_fraction: Half-width of the acceptance band, as a fraction of
            the target.
        in_window: ``(na, nb)`` boolean acceptance mask.
        area: Window area, in ``axis_a.unit * axis_b.unit``.
        domain_area: ``axis_a.span * axis_b.span`` -- the area if every point
            were acceptable.
        fraction: ``area / domain_area``; dimensionless, so it survives a change
            of sweep ranges.
        largest_connected_area: Area of the largest 4-connected window region.
        refused_fraction: Weighted fraction of the domain the **envelope**
            refused, in the ADR-0032 sense: the solver declined to answer.
        unmeasured_fraction: Weighted fraction of the domain where the solver
            *did* answer but no carry could be derived from the answer --
            in practice a head that buried, whose trace reverses inside its
            own crater and so has no prismatic divot to divide by. Disjoint
            from :attr:`refused_fraction`; both count against the window.
        contains_nominal: Whether the nominal delivery is inside the window, or
            ``None`` when no nominal point was given.
    """

    axis_a: PlayabilityAxis
    axis_b: PlayabilityAxis
    carry_m: np.ndarray
    target_carry_m: float
    tolerance_fraction: float
    in_window: np.ndarray
    area: float
    domain_area: float
    fraction: float
    largest_connected_area: float
    refused_fraction: float
    unmeasured_fraction: float
    contains_nominal: bool | None

    @property
    def area_unit(self) -> str:
        """Unit symbol of :attr:`area`, e.g. ``"m.rad"``."""
        return f"{self.axis_a.unit}.{self.axis_b.unit}"

    @property
    def carry_band_m(self) -> tuple[float, float]:
        """Inclusive carry band that counts as acceptable [m]."""
        half = self.tolerance_fraction * self.target_carry_m
        return (self.target_carry_m - half, self.target_carry_m + half)


def _weight_grid(axis_a: PlayabilityAxis, axis_b: PlayabilityAxis) -> np.ndarray:
    """Return the ``(na, nb)`` outer product of the two axes' node weights."""
    return np.outer(axis_a.node_weights, axis_b.node_weights)


def _largest_connected_area(mask: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted area of the largest 4-connected true region.

    Args:
        mask: ``(na, nb)`` boolean acceptance mask.
        weights: ``(na, nb)`` node areas.

    Returns:
        The largest single region's area, or 0.0 when the mask is empty.
    """
    if not mask.any():
        return 0.0
    labels, count = ndimage.label(mask)
    areas = [float(weights[labels == index].sum()) for index in range(1, count + 1)]
    return max(areas)


def _nominal_in_window(
    axis_a: PlayabilityAxis,
    axis_b: PlayabilityAxis,
    mask: np.ndarray,
    nominal: tuple[float, float] | None,
) -> bool | None:
    """Return whether the grid node nearest ``nominal`` is inside the window.

    Args:
        axis_a: First axis.
        axis_b: Second axis.
        mask: Acceptance mask.
        nominal: ``(a, b)`` nominal delivery, or ``None``.

    Returns:
        The verdict at the nearest node, or ``None`` when no nominal was given.
    """
    if nominal is None:
        return None
    row = int(np.argmin(np.abs(axis_a.values - float(nominal[0]))))
    column = int(np.argmin(np.abs(axis_b.values - float(nominal[1]))))
    return bool(mask[row, column])


def playability_window(
    axis_a: PlayabilityAxis,
    axis_b: PlayabilityAxis,
    carry_m: np.ndarray,
    *,
    target_carry_m: float,
    tolerance_fraction: float = DEFAULT_CARRY_TOLERANCE_FRACTION,
    nominal: tuple[float, float] | None = None,
    refused: np.ndarray | None = None,
) -> PlayabilityWindow:
    """Measure the area over which carry stays within tolerance of the target.

    Args:
        axis_a: First swept factor; indexes the rows of ``carry_m``.
        axis_b: Second swept factor; indexes the columns.
        carry_m: ``(len(axis_a), len(axis_b))`` carry distances [m]. Use NaN for
            a point that produced no carry; it is counted as outside the window.
        target_carry_m: Target carry [m]; must be positive.
        tolerance_fraction: Half-width of the acceptance band as a fraction of
            the target. Must be in ``(0, 1]``.
        nominal: Optional ``(a, b)`` nominal delivery, reported as
            ``contains_nominal`` at the nearest grid node.
        refused: Optional boolean grid marking the points the **envelope**
            refused, in the ADR-0032 sense. Supply it whenever a NaN can
            arise for any other reason, so that the two are reported
            apart -- see :attr:`PlayabilityWindow.unmeasured_fraction`.
            Omitted, every NaN is attributed to refusal, which is what
            this function meant when refusal was the only way to get one.

    Returns:
        The measured window.

    Raises:
        ValueError: If the grid shape does not match the axes, the target is not
            positive, the tolerance is outside ``(0, 1]``, ``refused`` does not
            match the grid, or a refused point carries a finite carry.
    """
    carry = np.asarray(carry_m, dtype=float)
    expected = (axis_a.values.size, axis_b.values.size)
    if carry.shape != expected:
        raise ValueError(f"carry_m must have shape {expected}, got {carry.shape}")
    if not np.isfinite(target_carry_m) or target_carry_m <= 0.0:
        raise ValueError(
            f"target_carry_m must be positive and finite, got {target_carry_m}"
        )
    if not 0.0 < tolerance_fraction <= 1.0:
        raise ValueError(
            f"tolerance_fraction must be in (0, 1], got {tolerance_fraction}"
        )
    if np.any(np.isinf(carry)):
        raise ValueError("carry_m must be finite or NaN; found an infinity")
    weights = _weight_grid(axis_a, axis_b)
    domain_area = axis_a.span * axis_b.span
    missing = np.isnan(carry)
    if refused is None:
        refused_mask = missing
    else:
        refused_mask = np.asarray(refused, dtype=bool)
        if refused_mask.shape != expected:
            raise ValueError(
                f"refused must have shape {expected}, got {refused_mask.shape}"
            )
        if np.any(refused_mask & ~missing):
            raise ValueError(
                "a refused point cannot also report a carry; found a finite "
                "carry marked refused"
            )
    unmeasured = missing & ~refused_mask
    half_band = tolerance_fraction * target_carry_m
    with np.errstate(invalid="ignore"):
        mask = np.abs(carry - target_carry_m) <= half_band
    mask &= ~missing
    area = float((weights * mask).sum())
    return PlayabilityWindow(
        axis_a=axis_a,
        axis_b=axis_b,
        carry_m=carry,
        target_carry_m=float(target_carry_m),
        tolerance_fraction=float(tolerance_fraction),
        in_window=mask,
        area=area,
        domain_area=domain_area,
        fraction=area / domain_area,
        largest_connected_area=_largest_connected_area(mask, weights),
        refused_fraction=float((weights * refused_mask).sum()) / domain_area,
        unmeasured_fraction=float((weights * unmeasured).sum()) / domain_area,
        contains_nominal=_nominal_in_window(axis_a, axis_b, mask, nominal),
    )


def playability_objective(
    axis_a: PlayabilityAxis,
    axis_b: PlayabilityAxis,
    carry_m: np.ndarray,
    *,
    target_carry_m: float,
    tolerance_fraction: float = DEFAULT_CARRY_TOLERANCE_FRACTION,
    connected_only: bool = False,
) -> float:
    """Return the playability window area as a single number to maximise.

    This is the epic's **primary scalar objective**. Hand it to
    :mod:`bunkershot3d.study.optimisation` as the quantity a sole geometry is
    ranked by.

    Args:
        axis_a: First swept factor.
        axis_b: Second swept factor.
        carry_m: Carry grid [m]; NaN where the solver refused.
        target_carry_m: Target carry [m].
        tolerance_fraction: Half-width of the acceptance band.
        connected_only: When True, score only the largest connected window,
            which refuses to reward a design whose acceptable region is split
            into islands.

    Returns:
        The area, in ``axis_a.unit * axis_b.unit``. Larger is more forgiving.
    """
    window = playability_window(
        axis_a,
        axis_b,
        carry_m,
        target_carry_m=target_carry_m,
        tolerance_fraction=tolerance_fraction,
    )
    return window.largest_connected_area if connected_only else window.area
