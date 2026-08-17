"""Parametric sole cross-section construction (issue #8609).

Everything is built in the Acushnet measurement plane - a vertical plane
perpendicular to the leading edge - with the leading-edge (LE) point at
the origin, ``x`` rearward and ``z`` up.  The ground plane is tangent to
the sole at the trailing contact point, so the whole profile lies in
``-d1 tan(theta) <= z <= 0``.

The sole is three segments:

1. a circular **leading roll** of radius ``rho1`` from the LE point
   through the 1.2 mm datum point ``(d2, -d3)`` - those two constraints
   fix the arc's start tangent and turn angle exactly;
2. a **camber** segment whose radius of curvature varies as
   ``R(u) = A exp(k u + c u (1 - u))`` along the turn.  Because the
   segment is parameterised by a *monotonically decreasing tangent
   angle* with a strictly positive radius, it is convex and descending
   **by construction** - no overshoot or waviness is representable.
   ``k`` is solved so the segment lands on its far endpoint and ``c`` so
   the sole camber area comes out at its declared value;
3. a circular **trailing roll** of radius ``rho2`` over the last 1.2 mm,
   arriving tangent to the ground at the trailing contact point.

Because the camber area is a *shape* parameter rather than a correction
applied afterwards, an unreachable request fails loudly and quotes the
range the rest of the schema admits (see
:func:`constructible_camber_range_m2`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BunkerShot3DValueError
from .wedge import WedgeGeometry

__all__ = [
    "InconstructibleCamberError",
    "SoleProfile",
    "build_section_polygon",
    "build_sole_profile",
    "constructible_camber_range_m2",
    "polygon_area_m2",
]

_QUADRATURE_POINTS = 257
_MIN_PROFILE_POINTS = 12
_MAX_BULGE = 12.0
_MAX_GROWTH = 80.0


class InconstructibleCamberError(BunkerShot3DValueError):
    """A camber area no convex, monotone sole of this width can realise.

    Carries the numbers a caller needs to react programmatically instead of
    parsing the message: what was asked for, the sole width it was asked of,
    and the band that width admits.  Also a :class:`ValueError`, so callers
    written against the previous behaviour keep working.

    Attributes:
        requested_camber_area_m2: The camber area that was asked for.
        sole_width_m: The sole width it was asked of.
        constructible_range_m2: ``(low, high)`` the width admits.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_camber_area_m2: float,
        sole_width_m: float,
        constructible_range_m2: tuple[float, float],
    ) -> None:
        """Build the error.

        Args:
            message: Human-readable explanation, quoting the numbers.
            requested_camber_area_m2: The camber area that was asked for.
            sole_width_m: The sole width it was asked of.
            constructible_range_m2: ``(low, high)`` the width admits.
        """
        super().__init__(message)
        self.requested_camber_area_m2 = float(requested_camber_area_m2)
        self.sole_width_m = float(sole_width_m)
        self.constructible_range_m2 = (
            float(constructible_range_m2[0]),
            float(constructible_range_m2[1]),
        )


@dataclass(frozen=True, eq=False)
class SoleProfile:
    """A sampled sole cross-section from the LE point to trailing contact.

    Attributes:
        points_m: ``(n, 2)`` polyline, ``x`` rearward and ``z`` up, with
            ``points_m[0]`` the leading-edge point at the origin and
            ``points_m[-1]`` the trailing contact point.
        camber_area_m2: Area between the profile and the LE/trailing
            chord, recomputed from the sampled polyline.
        leading_arc_radius_m: Realised ``rho1``.
        trailing_arc_radius_m: Realised ``rho2``.
        leading_tangent_deg: Sole angle below horizontal at the datum.
    """

    points_m: NDArray[np.float64]
    camber_area_m2: float
    leading_arc_radius_m: float
    trailing_arc_radius_m: float
    leading_tangent_deg: float

    def height_at_m(self, x_m: float) -> float:
        """Sole height at a rearward station, by linear interpolation."""
        return float(np.interp(x_m, self.points_m[:, 0], self.points_m[:, 1]))

    @property
    def trailing_contact_m(self) -> NDArray[np.float64]:
        """The trailing contact point."""
        return self.points_m[-1]


def _chord_area_m2(points: NDArray[np.float64]) -> float:
    """Area between the polyline and the chord joining its endpoints."""
    x_values = points[:, 0]
    chord = np.interp(
        x_values,
        [points[0, 0], points[-1, 0]],
        [points[0, 1], points[-1, 1]],
    )
    return float(np.trapezoid(chord - points[:, 1], x_values))


def polygon_area_m2(polygon: NDArray[np.float64]) -> float:
    """Signed shoelace area; positive for counter-clockwise winding."""
    x_values = polygon[:, 0]
    y_values = polygon[:, 1]
    return 0.5 * float(
        np.dot(x_values, np.roll(y_values, -1))
        - np.dot(y_values, np.roll(x_values, -1))
    )


def _leading_arc(
    radius_m: float, datum_offset_m: float, entry_height_m: float, n_points: int
) -> tuple[NDArray[np.float64], float]:
    """Circular roll from the LE point through the datum point.

    Returns the sampled arc and its end tangent (radians below the
    horizontal).
    """
    p = datum_offset_m / radius_m
    q = entry_height_m / radius_m
    half_chord = math.hypot(p, q) / 2.0
    if half_chord >= 1.0:
        raise ValueError(
            "leading-edge sole radius is too small for the entry geometry: "
            f"rho1 = {radius_m * 1e3:.3g} mm cannot span a "
            f"{math.hypot(datum_offset_m, entry_height_m) * 1e3:.3g} mm chord "
            "(the chord must be at most 2 rho1)"
        )
    mid = math.atan2(q, p)
    half_turn = math.asin(half_chord)
    start = mid + half_turn
    turns = np.linspace(0.0, 2.0 * half_turn, n_points)
    points = np.column_stack(
        [
            radius_m * (math.sin(start) - np.sin(start - turns)),
            radius_m * (math.cos(start) - np.cos(start - turns)),
        ]
    )
    points[0] = 0.0
    points[-1] = (datum_offset_m, -entry_height_m)
    return points, mid - half_turn


def _trailing_arc(
    radius_m: float,
    datum_offset_m: float,
    trailing_point_m: NDArray[np.float64],
    n_points: int,
) -> tuple[NDArray[np.float64], float]:
    """Circular roll arriving tangent to the ground at trailing contact."""
    if datum_offset_m >= radius_m:
        raise ValueError(
            "trailing-edge sole radius must exceed the measurement datum, "
            f"got rho2 = {radius_m * 1e3:.3g} mm"
        )
    turn = math.asin(datum_offset_m / radius_m)
    angles = np.linspace(turn, 0.0, n_points)
    points = np.column_stack(
        [
            trailing_point_m[0] - radius_m * np.sin(angles),
            trailing_point_m[1] + radius_m * (1.0 - np.cos(angles)),
        ]
    )
    points[-1] = trailing_point_m
    return points, turn


@dataclass(frozen=True, eq=False)
class _CamberQuadrature:
    """Pre-computed turn-angle quadrature for the camber segment."""

    samples: NDArray[np.float64]
    cosines: NDArray[np.float64]
    sines: NDArray[np.float64]
    parabola: NDArray[np.float64]

    @classmethod
    def build(cls, start_rad: float, end_rad: float) -> _CamberQuadrature:
        samples = np.linspace(0.0, 1.0, _QUADRATURE_POINTS, dtype=np.float64)
        angles = start_rad - samples * (start_rad - end_rad)
        return cls(
            samples=samples,
            cosines=np.cos(angles),
            sines=np.sin(angles),
            parabola=samples * (1.0 - samples),
        )

    def radius_weight(self, growth: float, bulge: float) -> NDArray[np.float64]:
        """Radius profile up to an arbitrary positive scale.

        The exponent is shifted by its maximum before exponentiating, so
        steep curvature schedules cannot overflow; the overall scale is
        divided out by the endpoint normalisation anyway.
        """
        exponent = growth * self.samples + bulge * self.parabola
        return np.exp(exponent - exponent.max())


def _cumulative_trapezoid(
    values: NDArray[np.float64], samples: NDArray[np.float64]
) -> NDArray[np.float64]:
    steps = 0.5 * (values[1:] + values[:-1]) * np.diff(samples)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _bracketed_root(
    function: Callable[[float], float],
    low: float,
    high: float,
    low_value: float,
    high_value: float,
    *,
    tolerance: float,
    max_iterations: int = 120,
) -> float:
    """Illinois (modified regula falsi) root find on a sign-changing bracket.

    Written out rather than imported so that this module - and therefore
    the whole geometry package - stays importable without SciPy, which a
    cross-engine import test in this package relies on.
    """
    guess = 0.5 * (low + high)
    for _ in range(max_iterations):
        if high_value == low_value:
            guess = 0.5 * (low + high)
        else:
            guess = (low * high_value - high * low_value) / (high_value - low_value)
            if not low < guess < high:
                guess = 0.5 * (low + high)
        value = function(guess)
        if value == 0.0 or (high - low) <= tolerance:
            return guess
        if value * low_value < 0.0:
            high, high_value = guess, value
            low_value *= 0.5
        else:
            low, low_value = guess, value
            high_value *= 0.5
    return guess


def _solve_growth(
    quadrature: _CamberQuadrature, span_x_m: float, drop_z_m: float, bulge: float
) -> float:
    """Find ``k`` so the camber segment lands on its far endpoint."""

    def residual(growth: float) -> float:
        """Mean-slope mismatch; bounded by the two endpoint tangents."""
        weight = quadrature.radius_weight(growth, bulge)
        cosine = float(np.trapezoid(weight * quadrature.cosines, quadrature.samples))
        sine = float(np.trapezoid(weight * quadrature.sines, quadrature.samples))
        return span_x_m * (sine / cosine) - drop_z_m

    low, high = -_MAX_GROWTH, _MAX_GROWTH
    low_value, high_value = residual(low), residual(high)
    if low_value * high_value > 0.0:
        mean_slope_deg = math.degrees(math.atan2(drop_z_m, span_x_m))
        raise ValueError(
            "sole camber segment is not constructible: behind the datum the "
            f"sole must fall {drop_z_m * 1e3:.4g} mm over "
            f"{span_x_m * 1e3:.4g} mm, a mean slope of "
            f"{mean_slope_deg:.3f} deg, which is too close to the "
            f"{math.degrees(math.atan2(quadrature.sines[-1], quadrature.cosines[-1])):.3f}"
            " deg trailing tangent. Widen the sole, lower the entry height, "
            "or reduce the heel/toe relief."
        )
    return _bracketed_root(residual, low, high, low_value, high_value, tolerance=1e-10)


def _camber_segment(
    start_point_m: NDArray[np.float64],
    end_point_m: NDArray[np.float64],
    quadrature: _CamberQuadrature,
    n_points: int,
    bulge: float,
) -> NDArray[np.float64]:
    """Monotone convex arc joining two points with prescribed tangents.

    ``bulge`` tightens (positive) or opens (negative) the mid-sole
    curvature, which is how the camber area is dialled in.
    """
    span_x = float(end_point_m[0] - start_point_m[0])
    drop_z = float(start_point_m[1] - end_point_m[1])
    growth = _solve_growth(quadrature, span_x, drop_z, bulge)
    weight = quadrature.radius_weight(growth, bulge)

    raw_x = _cumulative_trapezoid(weight * quadrature.cosines, quadrature.samples)
    raw_z = _cumulative_trapezoid(weight * quadrature.sines, quadrature.samples)
    # Normalise so both endpoints are hit exactly; the two scale factors
    # agree to ~1e-8, so the prescribed tangents survive.
    scaled_x = start_point_m[0] + raw_x * (span_x / raw_x[-1])
    scaled_z = start_point_m[1] - raw_z * (drop_z / raw_z[-1])

    # Sample by arc length, not by turn angle: a steep curvature schedule
    # barely moves near one end, and uniform-in-angle sampling would emit
    # coincident points there (degenerate triangles downstream).
    arc = _cumulative_trapezoid(weight, quadrature.samples)
    targets = np.linspace(0.0, float(arc[-1]), n_points)
    return np.column_stack(
        [np.interp(targets, arc, scaled_x), np.interp(targets, arc, scaled_z)]
    )


def _sample_counts(n_points: int) -> tuple[int, int, int]:
    """Split the sample budget across leading roll, camber, trailing roll."""
    if n_points < _MIN_PROFILE_POINTS:
        raise ValueError(
            f"n_points must be at least {_MIN_PROFILE_POINTS}, got {n_points}"
        )
    n_leading = max(4, n_points // 4)
    n_trailing = max(4, n_points // 8)
    n_camber = n_points - n_leading - n_trailing + 2
    if n_camber < 4:
        raise ValueError(f"n_points={n_points} is too small to sample the sole")
    return n_leading, n_camber, n_trailing


def _profile_assembler(
    geometry: WedgeGeometry, width_m: float, n_points: int
) -> tuple[Callable[[float], NDArray[np.float64]], float]:
    """Build the sole assembler for a (possibly relieved) sole width.

    Returns a function mapping the mid-sole bulge to the sampled profile,
    plus the sole's tangent angle at the datum in degrees.
    """
    datum_m = geometry.datum_offset_m
    if width_m <= 2.0 * datum_m:
        raise ValueError(
            f"sole width {width_m * 1e3:.3g} mm leaves no camber segment "
            f"between the two {datum_m * 1e3:.3g} mm measurement rolls"
        )
    drop_m = width_m * math.tan(geometry.geometric_bounce.angle_rad)
    n_leading, n_camber, n_trailing = _sample_counts(n_points)

    leading, leading_end_rad = _leading_arc(
        geometry.leading_edge_radius_m, datum_m, geometry.entry_height_m, n_leading
    )
    trailing, trailing_start_rad = _trailing_arc(
        geometry.trailing_edge_radius_m,
        datum_m,
        np.array([width_m, -drop_m]),
        n_trailing,
    )
    span_x = float(trailing[0, 0] - leading[-1, 0])
    drop_z = float(leading[-1, 1] - trailing[0, 1])
    if span_x <= 0.0 or drop_z <= 0.0:
        raise ValueError(
            "the sole must keep descending rearward between the datum and "
            f"the trailing roll (span {span_x * 1e3:.3g} mm, drop "
            f"{drop_z * 1e3:.3g} mm): the entry height is too deep for this "
            "bounce angle and sole width"
        )
    if leading_end_rad <= trailing_start_rad:
        raise ValueError(
            "the sole must flatten rearward: datum tangent "
            f"{math.degrees(leading_end_rad):.2f} deg is not steeper than the "
            f"trailing tangent {math.degrees(trailing_start_rad):.2f} deg"
        )
    quadrature = _CamberQuadrature.build(leading_end_rad, trailing_start_rad)

    def assemble(bulge: float) -> NDArray[np.float64]:
        camber = _camber_segment(leading[-1], trailing[0], quadrature, n_camber, bulge)
        return np.vstack([leading[:-1], camber, trailing[1:]])

    return assemble, math.degrees(leading_end_rad)


def constructible_camber_range_m2(
    geometry: WedgeGeometry,
    *,
    sole_width_m: float | None = None,
    n_points: int = 48,
) -> tuple[float, float]:
    """The camber areas this sole geometry can actually realise.

    The bounds come from the extremes of the mid-sole curvature family,
    which is why they are far tighter than the geometric ceiling
    ``d1 * drop / 2``: a real sole has to stay convex and monotone.
    """
    width_m = geometry.sole_width_m if sole_width_m is None else float(sole_width_m)
    assemble, _ = _profile_assembler(geometry, width_m, n_points)
    first = _chord_area_m2(assemble(-_MAX_BULGE))
    second = _chord_area_m2(assemble(_MAX_BULGE))
    return (min(first, second), max(first, second))


def build_sole_profile(
    geometry: WedgeGeometry,
    *,
    n_points: int = 48,
    sole_width_m: float | None = None,
    camber_area_m2: float | None = None,
) -> SoleProfile:
    """Build the sampled sole cross-section for a wedge.

    Args:
        geometry: The wedge design vector.
        n_points: Total number of sampled points along the sole.
        sole_width_m: Override ``d1`` - used by the lofting code to apply
            heel and toe relief, which move the trailing contact point
            forward.
        camber_area_m2: Override the declared camber area, which scales
            with the square of the sole width when relief is applied.

    Returns:
        The sampled profile, whose camber area equals the requested value
        to floating-point precision.

    Raises:
        ValueError: If the requested combination is not constructible - a
            leading radius too small for the entry chord, an entry height
            too deep for the bounce chord, or a camber area outside the
            range a convex monotone sole admits.
    """
    width_m = geometry.sole_width_m if sole_width_m is None else float(sole_width_m)
    target_area_m2 = (
        geometry.sole_camber_area_m2
        if camber_area_m2 is None
        else float(camber_area_m2)
    )
    if not math.isfinite(target_area_m2) or target_area_m2 <= 0.0:
        raise ValueError(f"camber area must be positive, got {target_area_m2!r}")

    assemble, leading_tangent_deg = _profile_assembler(geometry, width_m, n_points)
    lower = _chord_area_m2(assemble(-_MAX_BULGE))
    upper = _chord_area_m2(assemble(_MAX_BULGE))
    low_area, high_area = min(lower, upper), max(lower, upper)
    if not low_area <= target_area_m2 <= high_area:
        drop_m = width_m * math.tan(geometry.geometric_bounce.angle_rad)
        raise InconstructibleCamberError(
            f"a camber area of {target_area_m2 * 1e6:.4g} mm^2 is not "
            f"achievable for a {width_m * 1e3:.4g} mm sole at "
            f"{geometry.entry_height_m * 1e3:.3g} mm entry height and "
            f"{geometry.geometric_bounce.angle_deg:.2f} deg of geometric "
            f"bounce: a convex monotone sole admits "
            f"{low_area * 1e6:.4g} to {high_area * 1e6:.4g} mm^2 "
            f"(geometric ceiling {0.5 * width_m * drop_m * 1e6:.4g} mm^2)",
            requested_camber_area_m2=target_area_m2,
            sole_width_m=width_m,
            constructible_range_m2=(low_area, high_area),
        )

    bulge = _bracketed_root(
        lambda value: _chord_area_m2(assemble(value)) - target_area_m2,
        -_MAX_BULGE,
        _MAX_BULGE,
        lower - target_area_m2,
        upper - target_area_m2,
        tolerance=1e-12,
    )
    points = assemble(bulge)
    return SoleProfile(
        points_m=points,
        camber_area_m2=_chord_area_m2(points),
        leading_arc_radius_m=geometry.leading_edge_radius_m,
        trailing_arc_radius_m=geometry.trailing_edge_radius_m,
        leading_tangent_deg=leading_tangent_deg,
    )


def build_section_polygon(
    geometry: WedgeGeometry,
    *,
    n_points: int = 48,
    sole_width_m: float | None = None,
    camber_area_m2: float | None = None,
) -> NDArray[np.float64]:
    """Closed, convex, counter-clockwise head cross-section.

    The outline runs: leading-edge point -> sole -> trailing contact ->
    rear surface -> topline -> back down the lofted face.  Closure is
    implicit (the last vertex joins the first).

    Returns:
        ``(m, 2)`` polygon in the measurement plane, metres.
    """
    profile = build_sole_profile(
        geometry,
        n_points=n_points,
        sole_width_m=sole_width_m,
        camber_area_m2=camber_area_m2,
    )
    face_top = np.array(
        [
            geometry.face_height_m * math.sin(geometry.loft_rad),
            geometry.face_height_m * math.cos(geometry.loft_rad),
        ]
    )
    rear_x = max(
        float(profile.trailing_contact_m[0]),
        float(face_top[0]) + geometry.topline_width_m,
    )
    polygon = np.vstack([profile.points_m, [rear_x, face_top[1]], face_top])
    if polygon_area_m2(polygon) <= 0.0:
        raise ValueError(
            "head cross-section is degenerate or inverted; check face "
            "height, loft and topline width"
        )
    return polygon
