"""Parametric dimple-geometry drag-coefficient adjustment model.

This module provides a small, dependency-free parametric model that maps
golf-ball dimple geometry parameters (count, depth, diameter, surface
coverage) to an adjusted drag coefficient.

The model is intentionally heuristic and is calibrated to industry data
showing that tour-style dimple patterns reduce the drag coefficient at
typical ball-flight Reynolds numbers (Re ~ 1.5e5) from a smooth-sphere
value of ~0.45 down to ~0.22-0.28 (a factor of ~0.5).

Below the dimple-induced boundary-layer transition (Re < 4e4) the
dimples provide little benefit and the smooth-sphere drag is returned.

Created for issue #3504 (Dimple Geometry Optimization).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Tunable constants -------------------------------------------------------

#: Reynolds number above which dimple-induced turbulent transition is active.
RE_TRANSITION: float = 4.0e4

#: Maximum fractional drag reduction achievable with optimal geometry.
MAX_DRAG_REDUCTION: float = 0.5

#: Allowed parameter ranges (inclusive) used for DbC validation.
COUNT_MIN: int = 1
COUNT_MAX: int = 1000
DEPTH_MIN_MM: float = 0.0
DEPTH_MAX_MM: float = 1.0
DIAMETER_MIN_MM: float = 0.0
DIAMETER_MAX_MM: float = 10.0
COVERAGE_MIN: float = 0.0
COVERAGE_MAX: float = 1.0

#: Reference (optimal) values used by the efficiency heuristic.
COUNT_LOW: float = 280.0
COUNT_HIGH: float = 500.0
DEPTH_OPTIMAL_MM: float = 0.20
DEPTH_HALF_WIDTH_MM: float = 0.10
COVERAGE_LOW: float = 0.65
COVERAGE_HIGH: float = 0.85


# --- Helpers -----------------------------------------------------------------


def _clip(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the closed interval ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


# --- Public API --------------------------------------------------------------


@dataclass(frozen=True)
class DimpleGeometry:
    """Parametric description of a golf-ball dimple pattern.

    Parameters
    ----------
    count:
        Total number of dimples on the ball surface. Typical tour balls
        have between 280 and 500. Must satisfy ``1 <= count <= 1000``.
    depth_mm:
        Mean dimple depth in millimetres. Tour-typical range is
        0.10 - 0.30 mm. Must satisfy ``0 <= depth_mm <= 1``.
    diameter_mm:
        Mean dimple diameter in millimetres. Tour-typical range is
        2.5 - 4.5 mm. Must satisfy ``0 <= diameter_mm <= 10``.
    coverage_fraction:
        Fraction of the ball surface area covered by dimples. Tour-
        typical range is 0.65 - 0.85. Must satisfy
        ``0 <= coverage_fraction <= 1``.

    Raises
    ------
    TypeError
        If ``count`` is not an ``int`` or if numeric fields are not
        ``int``/``float``.
    ValueError
        If any field falls outside the documented allowed range.
    """

    count: int
    depth_mm: float
    diameter_mm: float
    coverage_fraction: float

    def __post_init__(self) -> None:
        # --- Type checks (DbC preconditions) --------------------------------
        if not isinstance(self.count, int) or isinstance(self.count, bool):
            raise TypeError(f"count must be an int, got {type(self.count).__name__}")
        for name in ("depth_mm", "diameter_mm", "coverage_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a real number, got {type(value).__name__}"
                )

        # --- Range checks ---------------------------------------------------
        if not (COUNT_MIN <= self.count <= COUNT_MAX):
            raise ValueError(
                f"count must be in [{COUNT_MIN}, {COUNT_MAX}], got {self.count}"
            )
        if not (DEPTH_MIN_MM <= float(self.depth_mm) <= DEPTH_MAX_MM):
            raise ValueError(
                f"depth_mm must be in [{DEPTH_MIN_MM}, {DEPTH_MAX_MM}], "
                f"got {self.depth_mm}"
            )
        if not (DIAMETER_MIN_MM <= float(self.diameter_mm) <= DIAMETER_MAX_MM):
            raise ValueError(
                f"diameter_mm must be in [{DIAMETER_MIN_MM}, "
                f"{DIAMETER_MAX_MM}], got {self.diameter_mm}"
            )
        if not (COVERAGE_MIN <= float(self.coverage_fraction) <= COVERAGE_MAX):
            raise ValueError(
                f"coverage_fraction must be in [{COVERAGE_MIN}, "
                f"{COVERAGE_MAX}], got {self.coverage_fraction}"
            )

    # ------------------------------------------------------------------
    # Heuristic factors
    # ------------------------------------------------------------------

    @property
    def count_factor(self) -> float:
        """Normalised count score in ``[0, 1]``.

        Linearly scales the dimple count between the typical lower
        (``COUNT_LOW``) and upper (``COUNT_HIGH``) bounds.
        """
        return _clip(
            (float(self.count) - COUNT_LOW) / (COUNT_HIGH - COUNT_LOW),
            0.0,
            1.0,
        )

    @property
    def depth_factor(self) -> float:
        """Normalised depth score in ``[0, 1]`` peaking at the optimum."""
        deviation = abs(float(self.depth_mm) - DEPTH_OPTIMAL_MM)
        return _clip(1.0 - deviation / DEPTH_HALF_WIDTH_MM, 0.0, 1.0)

    @property
    def coverage_factor(self) -> float:
        """Normalised coverage score in ``[0, 1]``."""
        return _clip(
            (float(self.coverage_fraction) - COVERAGE_LOW)
            / (COVERAGE_HIGH - COVERAGE_LOW),
            0.0,
            1.0,
        )

    @property
    def efficiency(self) -> float:
        """Combined efficiency score in ``[0, 1]``.

        The mean of the count, depth and coverage factors. A value of
        1 represents an "optimal" tour-style geometry; a value of 0
        represents a smooth ball or a wildly off-spec pattern.
        """
        return (self.count_factor + self.depth_factor + self.coverage_factor) / 3.0


def dimple_adjusted_cd(
    geometry: DimpleGeometry,
    reynolds: float,
    base_cd_smooth: float = 0.45,
) -> float:
    """Return the drag coefficient adjusted for dimple geometry.

    Parameters
    ----------
    geometry:
        Parametric dimple geometry description.
    reynolds:
        Free-stream Reynolds number based on ball diameter. Must be
        strictly positive.
    base_cd_smooth:
        Smooth-sphere drag coefficient at moderate Re (default 0.45).
        Must be strictly positive.

    Returns
    -------
    float
        The adjusted drag coefficient. For ``reynolds < RE_TRANSITION``
        this is exactly ``base_cd_smooth`` (dimples do not help while
        the boundary layer remains laminar). Above the transition a
        reduction up to ``MAX_DRAG_REDUCTION * base_cd_smooth`` is
        applied, scaled by the geometry's efficiency score, giving
        ``base_cd_smooth * (1 - 0.5 * efficiency)`` for an optimal
        geometry: ~0.225 for the default base.

    Raises
    ------
    TypeError
        If ``geometry`` is not a :class:`DimpleGeometry` or if numeric
        arguments are not real numbers.
    ValueError
        If ``reynolds`` or ``base_cd_smooth`` is not strictly positive.
    """
    # --- DbC preconditions --------------------------------------------------
    if not isinstance(geometry, DimpleGeometry):
        raise TypeError(
            f"geometry must be a DimpleGeometry instance, got {type(geometry).__name__}"
        )
    if not isinstance(reynolds, (int, float)) or isinstance(reynolds, bool):
        raise TypeError(
            f"reynolds must be a real number, got {type(reynolds).__name__}"
        )
    if not isinstance(base_cd_smooth, (int, float)) or isinstance(base_cd_smooth, bool):
        raise TypeError(
            f"base_cd_smooth must be a real number, got {type(base_cd_smooth).__name__}"
        )
    if float(reynolds) <= 0.0:
        raise ValueError(f"reynolds must be strictly positive, got {reynolds}")
    if float(base_cd_smooth) <= 0.0:
        raise ValueError(
            f"base_cd_smooth must be strictly positive, got {base_cd_smooth}"
        )

    base = float(base_cd_smooth)

    # Below the transition, dimples do nothing useful.
    if float(reynolds) < RE_TRANSITION:
        return base

    return base * (1.0 - MAX_DRAG_REDUCTION * geometry.efficiency)


__all__ = [
    "DimpleGeometry",
    "dimple_adjusted_cd",
    "RE_TRANSITION",
    "MAX_DRAG_REDUCTION",
]
