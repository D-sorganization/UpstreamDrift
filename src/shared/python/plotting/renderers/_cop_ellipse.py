"""1-sigma ellipse geometry of the measured CoP markers (pure, no GUI).

:func:`covariance_ellipse` turns a 2x2 covariance ``(var_x, var_y, cov_xy)``
into the semi-axes and tilt of its 1-sigma ellipse: the points at Mahalanobis
distance one.  Its projection on any direction is that direction's ±1 sigma,
so its extent along x and y is ``sqrt(var_x)`` and ``sqrt(var_y)``; in 2-D it
holds about 39 % of the probability of a Gaussian, not 68 %.

:func:`marker_ellipse` picks the best description a payload carries at one
frame: the covariance, else the per-axis sigmas (axis-aligned), else a circle
of the worse-axis sigma.  The web mirror is ``covarianceEllipse`` and
``copMarkerEllipse`` in ``ui/src/components/analysis/groundReactionUncertainty.ts``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import NamedTuple

__all__ = [
    "COVARIANCE",
    "PER_AXIS",
    "MarkerEllipse",
    "covariance_ellipse",
    "marker_ellipse",
]

#: A marker drawn from the per-axis sigmas (or the worse axis), axis-aligned.
PER_AXIS = "per axis"
#: A marker drawn from the CoP x-y covariance, tilted with the plate.
COVARIANCE = "covariance"

#: Relative slack on ``cov_xy^2 <= var_x var_y`` for floating-point rounding.
_PSD_REL_SLACK = 1.0e-9

_Trace = Mapping[str, Sequence[float | None]]


class MarkerEllipse(NamedTuple):
    """Semi-axes along the (tilted) ellipse axes, tilt from +x, and its source."""

    semi_x: float
    semi_y: float
    angle_deg: float
    kind: str


def _finite_values(*values: float | None) -> list[float] | None:
    """``values`` as floats when every one is finite, else ``None``."""
    kept = [float(v) for v in values if v is not None and math.isfinite(v)]
    return kept if len(kept) == len(values) else None


def covariance_ellipse(
    var_x: float, var_y: float, cov_xy: float
) -> tuple[float, float, float]:
    """``(semi_major, semi_minor, angle_deg)`` of the 1-sigma ellipse.

    The angle is the major axis measured from +x towards +y, in ``(-90, 90]``
    degrees (``0`` for a circle).  Raises ``ValueError`` unless the input is a
    finite positive semi-definite covariance (within rounding).
    """
    if not all(math.isfinite(v) for v in (var_x, var_y, cov_xy)):
        raise ValueError("covariance terms must be finite")
    if var_x < 0.0 or var_y < 0.0:
        raise ValueError("covariance variances must be non-negative")
    if cov_xy**2 > var_x * var_y + _PSD_REL_SLACK * max(var_x, var_y) ** 2:
        raise ValueError("not a covariance: |cov_xy| > sqrt(var_x * var_y)")
    mean = 0.5 * (var_x + var_y)
    radius = math.hypot(0.5 * (var_x - var_y), cov_xy)
    angle = 0.5 * math.degrees(math.atan2(2.0 * cov_xy, var_x - var_y))
    # atan2 returns (-180, 180], so ``angle`` is already in (-90, 90].
    return math.sqrt(mean + radius), math.sqrt(max(mean - radius, 0.0)), angle


def marker_ellipse(sigma: _Trace, cov: _Trace | None, i: int) -> MarkerEllipse:
    """The frame-``i`` marker and what it shows (:data:`COVARIANCE` or :data:`PER_AXIS`).

    The covariance when every term is present, else the per-world-axis sigmas
    (``angle_deg`` 0), else a circle of the worse-axis sigma ``sigma["z"]``,
    which the caller must have checked is present.
    """
    terms = (
        None if cov is None else _finite_values(cov["x"][i], cov["y"][i], cov["z"][i])
    )
    if terms is not None:
        return MarkerEllipse(*covariance_ellipse(*terms), COVARIANCE)
    worse = sigma["z"][i]
    assert worse is not None, "cop_marker_indices keeps only frames with a sigma"
    axes = _finite_values(sigma["x"][i], sigma["y"][i])
    semi_x, semi_y = (worse, worse) if axes is None else axes
    return MarkerEllipse(semi_x, semi_y, 0.0, PER_AXIS)
