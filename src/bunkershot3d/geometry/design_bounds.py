"""Screen a design space against the constructible camber band (issue #8698).

A sensitivity study declares a box - bounce from 14 to 26 degrees, sole width
from 16 to 24 mm - and pins everything else, including the sole camber area.
But the camber areas a sole can actually realise depend on its width and its
bounce, so corners of that box can sit outside the band, and the lofter then
builds the nearest constructible section instead.  The sweep still runs; it
just answers a different question than the one asked, and
:class:`~bunkershot3d.study.morris.MorrisDesign`,
:class:`~bunkershot3d.study.sensitivity.SaltelliDesign` and
:class:`~bunkershot3d.study.sensitivity.SobolIndices` will attribute variance
to a factor the user believes is pinned.

:func:`check_camber_design_space` finds that before the first solver call.  It
evaluates the corners of the camber/width/bounce sub-box, because the band
moves monotonically with width and bounce, so an interior point cannot leave a
band both of its bracketing corners are inside.

Parameters are recognised by name, using the package's unit-suffix convention
(``bunkershot3d.units``): ``sole_width_mm``, ``sole_camber_area_mm2``,
``geometric_bounce_deg`` and so on.  ``bounce_deg`` is deliberately *not*
recognised - the geometric and marketed conventions differ by 6 to 10 degrees
here and guessing between them would be exactly the kind of silent
substitution this module exists to stop - but it is reported rather than
ignored.

This module lives in :mod:`bunkershot3d.geometry` rather than in
:mod:`bunkershot3d.study` so the study layer keeps its property of knowing
nothing about wedges; :meth:`bunkershot3d.study.design_space.DesignSpace.\
check_wedge_camber` is a thin delegating method.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from typing import Protocol

from .bounce import GeometricBounce, MarketedBounce, geometric_from_marketed
from .profile import constructible_camber_range_m2
from .wedge import WedgeGeometry

__all__ = ["check_camber_design_space"]

#: Parameter names that set the camber area, and their factor to m^2.
_CAMBER_NAMES: dict[str, float] = {
    "sole_camber_area_m2": 1.0,
    "sole_camber_area_mm2": 1e-6,
    "camber_area_m2": 1.0,
    "camber_area_mm2": 1e-6,
}

#: Parameter names that set the sole width, and their factor to metres.
_WIDTH_NAMES: dict[str, float] = {
    "sole_width_m": 1.0,
    "sole_width_mm": 1e-3,
}

#: Parameter names that set the bounce, and which convention they carry.
_BOUNCE_NAMES: dict[str, str] = {
    "geometric_bounce_deg": "geometric",
    "marketed_bounce_deg": "marketed",
}

#: Names that name a bounce without saying which convention it is in.
_AMBIGUOUS_BOUNCE_NAMES: frozenset[str] = frozenset({"bounce_deg", "bounce"})


class _BoundedParameter(Protocol):
    """The part of a design parameter this screen needs."""

    @property
    def name(self) -> str:
        """Parameter name."""

    @property
    def lower(self) -> float:
        """Inclusive lower bound."""

    @property
    def upper(self) -> float:
        """Inclusive upper bound."""


class DesignSpaceLike(Protocol):
    """The part of a design space this screen needs.

    Structural rather than nominal so :mod:`bunkershot3d.geometry` does not
    import :mod:`bunkershot3d.study`, which would drag SciPy into the import
    graph of every consumer of the geometry package.
    """

    @property
    def parameters(self) -> Sequence[_BoundedParameter]:
        """The space's parameters, in column order."""


#: One option along a screened axis: a label, and the SI value it stands for.
_Option = tuple[str | None, float | None]

#: One bounce option: a label, and the ``(convention, degrees)`` it stands for.
_BounceOption = tuple[str | None, tuple[str, float] | None]

#: Taken when a role is absent from the space: keep the base design's value.
_KEEP_BASE: _Option = (None, None)
_KEEP_BASE_BOUNCE: _BounceOption = (None, None)


def _scalar_axis(
    parameters: Sequence[_BoundedParameter],
    names: dict[str, float],
    scale: float,
    unit: str,
) -> tuple[_Option, ...]:
    """Turn one recognised role into the corner options to screen.

    Args:
        parameters: The space's parameters.
        names: Recognised names mapped to their factor to SI.
        scale: Factor from SI back to the display unit, for the label.
        unit: Display unit.

    Returns:
        Two options per matching parameter; ``(_KEEP_BASE,)`` when absent.
    """
    options = tuple(
        (
            f"{parameter.name} = {bound * names[parameter.name] * scale:.3f} {unit}",
            bound * names[parameter.name],
        )
        for parameter in parameters
        if parameter.name in names
        for bound in (float(parameter.lower), float(parameter.upper))
    )
    return options or (_KEEP_BASE,)


def _bounce_axis(
    parameters: Sequence[_BoundedParameter],
) -> tuple[_BounceOption, ...]:
    """Turn the recognised bounce parameters into corner options.

    Args:
        parameters: The space's parameters.

    Returns:
        Two options per recognised bounce parameter, each carrying its own
        convention; ``(_KEEP_BASE_BOUNCE,)`` when none is present.
    """
    options: list[_BounceOption] = [
        (f"{parameter.name} = {bound:.3f} deg", (_BOUNCE_NAMES[parameter.name], bound))
        for parameter in parameters
        if parameter.name in _BOUNCE_NAMES
        for bound in (float(parameter.lower), float(parameter.upper))
    ]
    return tuple(options) or (_KEEP_BASE_BOUNCE,)


def _corner_geometry(
    base: WedgeGeometry,
    *,
    camber_m2: float | None,
    width_m: float | None,
    bounce: tuple[str, float] | None,
) -> WedgeGeometry:
    """Build the design vector at one corner of the screened sub-box.

    Args:
        base: The design the study pins everything else from.
        camber_m2: Corner camber area, or ``None`` to keep the base's.
        width_m: Corner sole width, or ``None`` to keep the base's.
        bounce: ``(convention, degrees)``, or ``None`` to keep the base's.

    Returns:
        The corner design vector.

    Raises:
        ValueError: If the corner is not an admissible design vector at all.
    """
    sole_width_m = base.sole_width_m if width_m is None else width_m
    changes: dict[str, object] = {"sole_width_m": sole_width_m}
    if camber_m2 is not None:
        changes["sole_camber_area_m2"] = camber_m2
    if bounce is not None:
        convention, angle_deg = bounce
        changes["geometric_bounce"] = (
            GeometricBounce(angle_deg)
            if convention == "geometric"
            else geometric_from_marketed(
                MarketedBounce(angle_deg),
                sole_width_m=sole_width_m,
                entry_height_m=base.entry_height_m,
                datum_offset_m=base.datum_offset_m,
            )
        )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def _describe(labels: Sequence[str]) -> str:
    """Join corner labels into a readable clause.

    Args:
        labels: Per-parameter ``name = value unit`` strings.

    Returns:
        The joined clause, or ``"the declared design"`` when empty.
    """
    return " with ".join(labels) if labels else "the declared design"


def check_camber_design_space(
    space: DesignSpaceLike,
    geometry: WedgeGeometry,
    *,
    n_points: int = 48,
) -> tuple[str, ...]:
    """Report design-space corners whose camber area is not constructible.

    Args:
        space: The design space about to be sampled. Only its ``parameters``
            are read; the space is never mutated.
        geometry: The base design vector the study pins everything else from.
        n_points: Sole samples used to evaluate the band. Pass the same value
            the sweep will loft at - the band shifts by ~0.1% between 24 and
            48 samples.

    Returns:
        One human-readable finding per offending corner, empty when the whole
        sub-box is constructible. A space with no camber, sole-width or
        bounce parameter is not screened and returns an empty tuple.

    Raises:
        TypeError: If ``geometry`` is not a :class:`WedgeGeometry`.
    """
    if not isinstance(geometry, WedgeGeometry):
        raise TypeError(f"expected a WedgeGeometry, got {type(geometry).__name__}")
    parameters = tuple(space.parameters)

    findings: list[str] = [
        f"parameter {parameter.name!r} names a bounce without saying which "
        "convention it is in; the geometric (patent) and marketed "
        "conventions differ by 6-10 degrees here, so this space cannot be "
        "screened. Rename it geometric_bounce_deg or marketed_bounce_deg."
        for parameter in parameters
        if parameter.name in _AMBIGUOUS_BOUNCE_NAMES
    ]

    camber_axis = _scalar_axis(parameters, _CAMBER_NAMES, 1e6, "mm^2")
    width_axis = _scalar_axis(parameters, _WIDTH_NAMES, 1e3, "mm")
    bounce_axis = _bounce_axis(parameters)
    if (camber_axis, width_axis, bounce_axis) == (
        (_KEEP_BASE,),
        (_KEEP_BASE,),
        (_KEEP_BASE_BOUNCE,),
    ):
        return tuple(findings)

    for camber, width, bounce in itertools.product(
        camber_axis, width_axis, bounce_axis
    ):
        finding = _screen_corner(geometry, camber, width, bounce, n_points=n_points)
        if finding is not None:
            findings.append(finding)
    return tuple(findings)


def _screen_corner(
    geometry: WedgeGeometry,
    camber: _Option,
    width: _Option,
    bounce: _BounceOption,
    *,
    n_points: int,
) -> str | None:
    """Evaluate one corner, returning a finding when it is not constructible.

    Args:
        geometry: The base design vector.
        camber: ``(label, camber_area_m2)`` or ``(None, None)``.
        width: ``(label, sole_width_m)`` or ``(None, None)``.
        bounce: ``(label, (convention, degrees))`` or ``(None, None)``.
        n_points: Sole samples used to evaluate the band.

    Returns:
        The finding, or ``None`` when this corner is fine.
    """
    labels = [label for label, _ in (camber, width, bounce) if label is not None]
    described = _describe(labels)
    try:
        corner = _corner_geometry(
            geometry, camber_m2=camber[1], width_m=width[1], bounce=bounce[1]
        )
        low, high = constructible_camber_range_m2(corner, n_points=n_points)
    except (TypeError, ValueError) as error:
        return f"{described} is not an admissible design vector: {error}"

    declared_m2 = corner.sole_camber_area_m2
    if low <= declared_m2 <= high:
        return None
    effective_m2 = min(max(declared_m2, low), high)
    return (
        f"{described}: a convex monotone sole admits {low * 1e6:.3f} to "
        f"{high * 1e6:.3f} mm^2, so the declared "
        f"{declared_m2 * 1e6:.3f} mm^2 would be built as "
        f"{effective_m2 * 1e6:.3f} mm^2 instead"
    )
