"""The F1 validity envelope and the quantities ADR-0033 refuses outright.

Why F1 does not reuse ``evaluate_envelope``
-------------------------------------------

:func:`bunkershot3d.solvers.envelope.evaluate_envelope` judges a query
against **3D-RFT's** limits: a Froude ceiling of 0.4, a micro-inertial
limit of 0.1, Askari & Kamrin's ``I_G`` on the surface discretisation,
and the five Zhang & Goldman RFT failure modes attached unconditionally.
Three of those standing caveats -- no wake model, reduced accuracy at
sharp corners, steady-state assumption -- are statements about *resistive
force theory*.  Attaching them to a continuum solve would be a category
error, and a flattering one: it would describe F1's weaknesses using F0's
vocabulary and so hide the ones F1 actually has.

The shared machinery that is genuinely tier-independent **is** reused:
:class:`~bunkershot3d.solvers.envelope.ValidityVerdict`,
:class:`~bunkershot3d.solvers.envelope.EnvelopeStatus`,
:class:`~bunkershot3d.solvers.envelope.Caveat`,
:func:`~bunkershot3d.solvers.envelope.dimensionless_groups` and
:func:`~bunkershot3d.solvers.envelope.worst_of`.  A caller therefore
handles an F0 verdict and an F1 verdict identically, which is the point
of ADR-0032's one-protocol rule.

The status floor
----------------

**No F1 query can be better than ``BEYOND_VALIDATION``.**  That is not a
pessimistic default; it is arithmetic.  ``EXTRAPOLATED`` means "outside
the stated limits but inside the published validation set", and issue
#8616 found there is no published validation set for any quantity this
tier produces -- no measured sand velocity field, no measured divot
section, no measured clubhead deceleration in sand.  A status of
``WITHIN`` would have to be measured against something that does not
exist.  The NASA-STD-7009B self-assessment records validation at level 0
of 4, and this floor is how that number is prevented from quietly rising
because a new tier was added.

Refinement runs the other way here
----------------------------------

F0's trap is that refining the surface mesh *raises* ``I_G`` and makes
the fit less exact.  F1's is the mirror image: refining ``dx`` below a
few grain diameters means the continuum is being asked to resolve
structure the size of individual grains, where a continuum stress has no
meaning.  So this envelope refuses a grid that is **too fine** as well as
a feature that spans too few grains.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum

from ..envelope import (
    GRAVITY_M_S2,
    MARGINAL_CONTINUUM_SIZE_RATIO,
    MAX_VALIDATED_SPEED_M_S,
    MIN_CONTINUUM_SIZE_RATIO,
    Caveat,
    DimensionlessGroups,
    EnvelopeStatus,
    ValidityVerdict,
    dimensionless_groups,
)
from ..exceptions import OutOfEnvelopeError, SolverInputError

__all__ = [
    "F1_STANDING_CAVEATS",
    "MIN_CELLS_PER_GRAIN",
    "MIN_CELLS_PER_RESOLVED_FEATURE",
    "RefusedQuantity",
    "evaluate_f1_envelope",
    "require_quotable",
]

MIN_CELLS_PER_GRAIN = 2.0
"""Fewest grain diameters a cell may span before the continuum is a fiction.

A cell finer than about two grains is resolving structure the size of
individual grains with a model that has no grains in it.  Refining past
this point does not improve the answer; it changes what the answer is
about."""

MIN_CELLS_PER_RESOLVED_FEATURE = 4.0
"""Cells across a feature below which that feature is not resolved.

ADR-0033 puts the leading edge deliberately below this, so falling under
it raises :attr:`~bunkershot3d.solvers.envelope.Caveat.UNDER_RESOLVED_LEADING_EDGE`
rather than a refusal -- but it is stated on every verdict, because a
reader looking at a picture of the flow around the edge is entitled to
know the edge was never resolved."""

F1_STANDING_CAVEATS: tuple[Caveat, ...] = (
    Caveat.PLANE_STRAIN_NO_OUT_OF_PLANE,
    Caveat.RATE_INDEPENDENT_PLASTICITY,
    Caveat.DECLARED_EFFECTIVE_WIDTH,
    Caveat.BORROWED_COEFFICIENTS,
    Caveat.NO_MEASURED_COMPARISON,
)
"""Caveats that apply to every F1 result without exception.

Four of the five are structural -- they describe what the formulation
cannot carry -- and the fifth records that the sand constants are
borrowed, which F1 inherits from the sand package along with the
constants themselves."""


class RefusedQuantity(StrEnum):
    """Quantities ADR-0033 forbids F1 from reporting at all.

    "Refused" in that ADR means **the API raises**, not "discouraged by
    documentation".  The distinction is the whole point: a figure outlives
    its caption, and a number that can be read off an F1 result will be
    quoted eventually regardless of what the docstring said.
    """

    CLUB_FORCE = "club_force"
    """Absolute club force or wrench. F0 owns this; F1 is under-resolved
    at the leading edge, which is where club force lives."""

    BALL_LAUNCH = "ball_launch"
    """Ball speed, launch angle or spin. The F0 momentum-transfer path of
    #8657 remains the only route, and is itself uncalibrated."""

    OUT_OF_PLANE = "out_of_plane"
    """Any heel-toe or lateral distribution. The quantity does not exist
    in plane strain, so this is refused rather than approximated."""


_REFUSAL_TEXT: Mapping[RefusedQuantity, str] = {
    RefusedQuantity.CLUB_FORCE: (
        "F1 may not be quoted for absolute club force. It runs at bulk "
        "resolution (dx ~ 1-2 mm) and the leading edge, where club force is "
        "generated, spans only a few cells. F0's per-element decomposition "
        "owns this quantity. F1's wrench exists only for the shape-and-timing "
        "cross-check of ADR-0033's B5, and its magnitude additionally depends "
        "on a declared effective width that is an assumption, not a result."
    ),
    RefusedQuantity.BALL_LAUNCH: (
        "F1 may not be quoted for ball speed, launch angle or spin. Its ball "
        "exists to show what reaches the ball, and in plane strain it is an "
        "infinite cylinder rather than a sphere. Ball launch remains on F0's "
        "momentum-transfer path (#8657)."
    ),
    RefusedQuantity.OUT_OF_PLANE: (
        "F1 may not be quoted for any heel-toe or out-of-plane distribution. "
        "Plane strain has no third dimension to distribute anything over, so "
        "there is no approximate answer to give -- only a fabricated one."
    ),
}


def require_quotable(quantity: RefusedQuantity | str) -> None:
    """Raise for any quantity ADR-0033 forbids F1 from reporting.

    Args:
        quantity: The quantity a caller is about to read off an F1 result.

    Raises:
        OutOfEnvelopeError: Always, for a recognised refused quantity, in
            the same shape as the existing out-of-envelope refusals.
        SolverInputError: If the quantity is not one of the refused set --
            because silently passing an unrecognised name would turn this
            guard into a no-op the first time somebody misspells one.
    """
    try:
        key = RefusedQuantity(quantity)
    except ValueError:
        raise SolverInputError(
            f"{quantity!r} is not a refused F1 quantity; expected one of "
            + ", ".join(sorted(item.value for item in RefusedQuantity))
        ) from None
    raise OutOfEnvelopeError(
        f"F1 refuses to report {key.value}:\n  {_REFUSAL_TEXT[key]}",
        verdict=None,
    )


def evaluate_f1_envelope(
    *,
    speed_m_s: float,
    feature_lengths_m: Mapping[str, float],
    grain_diameter_m: float,
    cell_size_m: float,
    submerged_depth_m: float = 0.0,
    effective_width_m: float,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> ValidityVerdict:
    """Judge one F1 query at every supplied feature scale.

    Args:
        speed_m_s: Intrusion speed.
        feature_lengths_m: Named geometric scales.
        grain_diameter_m: Median grain diameter.
        cell_size_m: The grid ``dx``, which plays the role F0's surface
            discretisation length plays in its own envelope.
        submerged_depth_m: Deepest submerged point, positive downward.
        effective_width_m: The declared out-of-plane width, carried onto
            the verdict's details so a magnitude can never be reproduced
            without the assumption behind it.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The verdict. Nothing is raised here; refusal is a value.

    Raises:
        SolverInputError: If no feature scale is supplied, or if the cell
            size or width is not positive.
    """
    if not feature_lengths_m:
        raise SolverInputError(
            "at least one feature scale is required; judging the envelope on no "
            "scale at all is how a solver reports a flattering number"
        )
    size = float(cell_size_m)
    width = float(effective_width_m)
    if not math.isfinite(size) or size <= 0.0:
        raise SolverInputError(f"cell_size_m must be positive, got {cell_size_m!r}")
    if not math.isfinite(width) or width <= 0.0:
        raise SolverInputError(
            f"effective_width_m must be positive, got {effective_width_m!r}"
        )

    ordered = sorted(feature_lengths_m.items(), key=lambda item: item[1])
    groups = tuple(
        dimensionless_groups(
            speed_m_s=speed_m_s,
            feature_length_m=length,
            grain_diameter_m=grain_diameter_m,
            element_size_m=size,
            name=name,
            gravity_m_s2=gravity_m_s2,
        )
        for name, length in ordered
    )

    reasons: list[str] = [
        "no published measurement exists for any quantity this tier produces "
        "(issue #8616), so its validation level is 0 of 4 and no status better "
        "than BEYOND_VALIDATION is available to it"
    ]
    caveats: list[Caveat] = list(F1_STANDING_CAVEATS)
    refused = False

    cells_per_grain = size / float(grain_diameter_m)
    if cells_per_grain < MIN_CELLS_PER_GRAIN:
        refused = True
        reasons.append(
            f"the grid cell spans {cells_per_grain:.3g} grain diameters, under "
            f"{MIN_CELLS_PER_GRAIN:.0f}: refining a continuum below the grain "
            "scale does not improve the answer, it changes what the answer is "
            "about"
        )

    governing_index = 0
    for index, group in enumerate(groups):
        cells = group.scale.length_m / size
        if cells < MIN_CELLS_PER_RESOLVED_FEATURE:
            if Caveat.UNDER_RESOLVED_LEADING_EDGE not in caveats:
                caveats.append(Caveat.UNDER_RESOLVED_LEADING_EDGE)
            reasons.append(
                f"the {group.scale.name} scale spans {cells:.3g} cells, under "
                f"{MIN_CELLS_PER_RESOLVED_FEATURE:.0f}, so its local flow is not "
                "resolved"
            )
        if group.continuum_size_ratio < MARGINAL_CONTINUUM_SIZE_RATIO:
            if Caveat.MARGINAL_CONTINUUM not in caveats:
                caveats.append(Caveat.MARGINAL_CONTINUUM)
            reasons.append(
                f"the {group.scale.name} scale spans only "
                f"{group.continuum_size_ratio:.3g} grains"
            )
        if group.continuum_size_ratio < MIN_CONTINUUM_SIZE_RATIO:
            refused = True
            governing_index = index
            reasons.append(
                f"the {group.scale.name} scale spans only "
                f"{group.continuum_size_ratio:.3g} grains (minimum "
                f"{MIN_CONTINUUM_SIZE_RATIO:.0f}): there is no continuum there "
                "to solve"
            )

    if float(speed_m_s) > MAX_VALIDATED_SPEED_M_S:
        caveats.append(Caveat.BEYOND_PUBLISHED_SPEED)
        reasons.append(
            f"speed {float(speed_m_s):.3g} m/s is "
            f"{float(speed_m_s) / MAX_VALIDATED_SPEED_M_S:.0f}x the fastest "
            f"granular intrusion in the published corpus "
            f"({MAX_VALIDATED_SPEED_M_S} m/s)"
        )
    _add_shallow_caveat(groups[governing_index], submerged_depth_m, reasons, caveats)

    return ValidityVerdict(
        status=EnvelopeStatus.REFUSED if refused else EnvelopeStatus.BEYOND_VALIDATION,
        groups=groups,
        caveats=tuple(dict.fromkeys(caveats)),
        reasons=tuple(dict.fromkeys(reasons)),
        governing_index=governing_index,
        clamped_area_fraction=0.0,
        details={
            "submerged_depth_m": float(submerged_depth_m),
            "grain_diameter_m": float(grain_diameter_m),
            "cell_size_m": size,
            "effective_width_m": width,
            "cells_per_grain": cells_per_grain,
        },
    )


def _add_shallow_caveat(
    governing: DimensionlessGroups,
    submerged_depth_m: float,
    reasons: list[str],
    caveats: list[Caveat],
) -> None:
    """Flag an intrusion too shallow for the bed to develop a stress field."""
    limit = 2.0 * governing.grain_diameter_m
    if 0.0 < float(submerged_depth_m) < limit:
        caveats.append(Caveat.SHALLOW_INTRUSION)
        reasons.append(
            f"deepest submerged point is {float(submerged_depth_m) * 1e3:.3g} mm, "
            f"under {limit * 1e3:.3g} mm (two grain diameters)"
        )
