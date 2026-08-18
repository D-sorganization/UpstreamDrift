"""Solution verification: the Grid Convergence Index (issue #8616).

Celik, Ghia, Roache, Freitas, Coleman & Raad, "Procedure for Estimation
and Reporting of Uncertainty Due to Discretization in CFD Applications",
*J. Fluids Eng.* 130(7):078001 (2008).

This is the second of the three V&V steps and the one that produces
``u_num``.  It is **not** validation: no experimental data appears
anywhere in this module.

The procedure
-------------

With three solutions ``phi1`` (fine), ``phi2``, ``phi3`` (coarse) on grids
of representative size ``h1 < h2 < h3``::

    r21 = h2/h1,  r32 = h3/h2
    eps21 = phi2 - phi1,  eps32 = phi3 - phi2
    s = sign(eps32/eps21)

    p       = |ln|eps32/eps21| + q(p)| / ln(r21)          (3a)
    q(p)    = ln[(r21^p - s)/(r32^p - s)]                 (3b)-(3c)
    phi_ext = (r21^p phi1 - phi2)/(r21^p - 1)
    e_a21   = |(phi1 - phi2)/phi1|
    GCI     = Fs e_a21/(r21^p - 1),  Fs = 1.25 (>=3 grids), 3.0 (2 grids)

Equation (3a) is implicit in ``p`` and is solved by fixed-point
iteration; when ``r21 == r32`` the ``q(p)`` term vanishes identically and
one pass is exact.

Why the division by ``(r^p - 1)``
---------------------------------

The bare grid-to-grid difference ``e_a21`` is *not* the fine-grid error.
At ``r = 2`` and ``p = 2`` it **over-estimates the fine-grid error by a
factor of 3**, because ``phi2 - phi1`` spans the error on both grids and
the coarse error is ``r^p = 4`` times the fine one.  Dividing by
``r^p - 1 = 3`` removes exactly that over-estimate; see
:func:`error_amplification`, which is pinned by a test.

Oscillatory convergence
-----------------------

``eps32/eps21 < 0`` means the solution is oscillating rather than
converging monotonically, and the Richardson estimate is then not
trustworthy.  :class:`GCIStudy` reports the **percentage** of quantities
that oscillated instead of dropping them, because a study that silently
discards its oscillatory quantities reports a convergence it did not
achieve.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from src.shared.python.core.contracts import ensure

from .exceptions import SolutionVerificationError

__all__ = [
    "COMFORTABLE_REFINEMENT_RATIO",
    "FACTOR_OF_SAFETY_THREE_GRID",
    "FACTOR_OF_SAFETY_TWO_GRID",
    "GCI_COVERAGE_FACTOR",
    "ApparentOrder",
    "ConvergenceType",
    "GCIResult",
    "GCIStudy",
    "GridSolution",
    "apparent_order",
    "error_amplification",
    "grid_convergence_index",
    "richardson_extrapolate",
    "two_grid_gci",
]

FACTOR_OF_SAFETY_THREE_GRID = 1.25
"""``Fs`` for a study with three or more grids (Celik et al. 2008)."""

FACTOR_OF_SAFETY_TWO_GRID = 3.0
"""``Fs`` for two grids, where ``p`` has to be assumed rather than observed."""

COMFORTABLE_REFINEMENT_RATIO = 1.3
"""Celik's "desirably greater than" refinement ratio.

Below this the grid-to-grid differences are close enough together that
round-off and iteration error contaminate the order estimate."""

_MIN_DENOMINATOR = 1e-300


class ConvergenceType(StrEnum):
    """How a refinement triplet behaves, classified by ``R = eps21/eps32``."""

    MONOTONIC = "monotonic"
    """``0 < R < 1``: the intended case."""

    OSCILLATORY = "oscillatory"
    """``-1 < R < 0``: converging, but alternating. Richardson is unreliable."""

    OSCILLATORY_DIVERGENCE = "oscillatory_divergence"
    """``R < -1``: alternating and growing."""

    MONOTONIC_DIVERGENCE = "monotonic_divergence"
    """``R > 1``: the coarse grids are closer together than the fine ones."""

    EXACT = "exact"
    """``eps21 == 0``: the two finest grids agree to the last bit."""

    @property
    def is_oscillatory(self) -> bool:
        """True for either oscillatory class."""
        return self in (
            ConvergenceType.OSCILLATORY,
            ConvergenceType.OSCILLATORY_DIVERGENCE,
        )

    @property
    def supports_richardson(self) -> bool:
        """True only for monotonic convergence."""
        return self is ConvergenceType.MONOTONIC


@dataclass(frozen=True, slots=True)
class GridSolution:
    """One solution of one quantity on one grid.

    Attributes:
        cell_size_m: Representative cell size ``h``. For a surface
            discretisation this is the root mean element area.
        value: The quantity of interest on that grid.
        label: Optional human label for reports.
    """

    cell_size_m: float
    value: float
    label: str = ""

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            SolutionVerificationError: If ``h`` is not positive and finite
                or the value is not finite.
        """
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise SolutionVerificationError(
                f"cell size must be positive and finite, got {self.cell_size_m!r}"
            )
        if not math.isfinite(self.value):
            raise SolutionVerificationError(
                f"grid solution value must be finite, got {self.value!r}"
            )


@dataclass(frozen=True, slots=True)
class ApparentOrder:
    """The observed order of accuracy from a refinement triplet.

    Attributes:
        order: ``p`` from Eq. (3a).
        convergence: How the triplet behaved.
        iterations: Fixed-point iterations used.
        converged: Whether the iteration met its tolerance.
        ratio: ``eps32/eps21``, whose sign is the oscillation test.
    """

    order: float
    convergence: ConvergenceType
    iterations: int
    converged: bool
    ratio: float


def error_amplification(refinement_ratio: float, order: float) -> float:
    """``r^p - 1``: the factor GCI divides the grid-to-grid difference by.

    At ``r = 2`` and ``p = 2`` this is exactly ``3``, which is the amount
    by which the bare difference ``|phi2 - phi1|`` over-states the
    fine-grid error.

    Args:
        refinement_ratio: ``r``, strictly greater than one.
        order: ``p``.

    Returns:
        ``r ** p - 1``.

    Raises:
        SolutionVerificationError: If ``r <= 1`` or the result is not
            positive, which would make the GCI meaningless.
    """
    if not math.isfinite(refinement_ratio) or refinement_ratio <= 1.0:
        raise SolutionVerificationError(
            f"refinement ratio must exceed 1, got {refinement_ratio!r}; a ratio "
            "of 1 means the grids were not refined"
        )
    amplification = refinement_ratio**order - 1.0
    if not math.isfinite(amplification) or amplification <= 0.0:
        raise SolutionVerificationError(
            f"r^p - 1 = {amplification!r} for r = {refinement_ratio!r}, "
            f"p = {order!r}; a non-positive amplification means the apparent "
            "order is not usable for an error estimate"
        )
    return amplification


def _classify(epsilon_21: float, epsilon_32: float) -> tuple[ConvergenceType, float]:
    """Classify a refinement triplet and return ``eps32/eps21``."""
    if epsilon_21 == 0.0:
        return ConvergenceType.EXACT, 0.0
    ratio = epsilon_32 / epsilon_21
    inverse = epsilon_21 / epsilon_32 if epsilon_32 != 0.0 else math.inf
    if inverse > 1.0:
        return ConvergenceType.MONOTONIC_DIVERGENCE, ratio
    if inverse > 0.0:
        return ConvergenceType.MONOTONIC, ratio
    if inverse > -1.0:
        return ConvergenceType.OSCILLATORY, ratio
    return ConvergenceType.OSCILLATORY_DIVERGENCE, ratio


def _q_term(order: float, r21: float, r32: float, sign: float) -> float:
    """``q(p) = ln[(r21^p - s)/(r32^p - s)]``, Eq. (3b)."""
    numerator = r21**order - sign
    denominator = r32**order - sign
    if numerator <= 0.0 or denominator <= 0.0:
        raise SolutionVerificationError(
            f"q(p) is undefined at p = {order:.6g}: r^p - s is non-positive "
            f"({numerator:.6g}, {denominator:.6g}). The refinement triplet "
            "cannot support an apparent-order estimate."
        )
    return math.log(numerator / denominator)


def apparent_order(
    *,
    epsilon_21: float,
    epsilon_32: float,
    r21: float,
    r32: float,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> ApparentOrder:
    """Solve Eq. (3a) for the apparent order ``p``.

    Args:
        epsilon_21: ``phi2 - phi1``.
        epsilon_32: ``phi3 - phi2``.
        r21: ``h2/h1``.
        r32: ``h3/h2``.
        tolerance: Fixed-point convergence tolerance on ``p``.
        max_iterations: Iteration cap.

    Returns:
        The apparent order, its convergence classification, and whether
        the iteration converged.

    Raises:
        SolutionVerificationError: If either ratio is not above one, if
            the two finest grids agree exactly (no order is observable),
            or if the iteration cannot be evaluated.
    """
    for name, ratio in (("r21", r21), ("r32", r32)):
        if not math.isfinite(ratio) or ratio <= 1.0:
            raise SolutionVerificationError(f"{name} must exceed 1, got {ratio!r}")
    convergence, signed_ratio = _classify(epsilon_21, epsilon_32)
    if convergence is ConvergenceType.EXACT:
        raise SolutionVerificationError(
            "phi1 and phi2 are identical, so no order of accuracy is "
            "observable; the quantity is either grid-independent already or "
            "insensitive to this refinement"
        )
    magnitude = abs(signed_ratio)
    if magnitude < _MIN_DENOMINATOR:
        raise SolutionVerificationError(
            "eps32 is zero while eps21 is not; the coarse pair shows no "
            "change and the apparent order is undefined"
        )
    sign = math.copysign(1.0, signed_ratio)
    log_magnitude = math.log(magnitude)
    log_r21 = math.log(r21)

    order = abs(log_magnitude) / log_r21
    converged = False
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        updated = abs(log_magnitude + _q_term(order, r21, r32, sign)) / log_r21
        converged = abs(updated - order) < tolerance
        order = updated
        if converged:
            break
    ensure(
        math.isfinite(order),
        "the apparent-order iteration produced a non-finite order",
        value=order,
    )
    return ApparentOrder(
        order=order,
        convergence=convergence,
        iterations=iterations,
        converged=converged,
        ratio=signed_ratio,
    )


def richardson_extrapolate(
    *, fine_value: float, coarse_value: float, refinement_ratio: float, order: float
) -> float:
    """``phi_ext = (r^p phi1 - phi2)/(r^p - 1)``.

    Args:
        fine_value: ``phi1``, the finer of the two solutions.
        coarse_value: ``phi2``.
        refinement_ratio: ``r21 = h2/h1``.
        order: ``p``.

    Returns:
        The extrapolated zero-cell-size estimate.

    Raises:
        SolutionVerificationError: If ``r^p - 1`` is not usable.
    """
    amplification = error_amplification(refinement_ratio, order)
    return (refinement_ratio**order * fine_value - coarse_value) / amplification


GCI_COVERAGE_FACTOR = 2.0
"""Divisor turning a GCI into a V&V 20 *standard* numerical uncertainty.

The GCI is constructed as an approximately 95% error band, and ASME
V&V 20 multiplies ``u_val`` by ``k = 2`` to obtain a ~95% interval.
Feeding a GCI straight in as ``u_h`` would therefore expand a 95% band to
95% *twice*.  This package takes ``u_h = GCI * |phi1| / 2`` and says so;
the alternative convention (``u_h = GCI * |phi1|``, deliberately
conservative) is available as
:attr:`GCIResult.expanded_numerical_uncertainty`.
"""


@dataclass(frozen=True, slots=True)
class GCIResult:
    """One quantity's discretisation uncertainty.

    Attributes:
        quantity: What was refined, for reports.
        fine_value: ``phi1``.
        apparent_order: ``p``, observed for three grids, assumed for two.
        order_assumed: True when ``p`` was supplied rather than observed.
        order_converged: Whether the Eq. (3a) iteration converged.
        convergence: Monotonic, oscillatory or divergent.
        refinement_ratio: ``r21``.
        extrapolated_value: ``phi_ext21``, or ``None`` when the triplet
            does not support Richardson extrapolation.
        approximate_relative_error: ``e_a21``.
        extrapolated_relative_error: ``e_ext21``, or ``None``.
        factor_of_safety: ``Fs``.
        gci_fine: ``GCI_fine21``, a fraction of ``|phi1|``.
        n_grids: Number of grids the estimate used.
        comfortable_refinement: Whether ``r21`` cleared Celik's
            "desirably greater than 1.3".
    """

    quantity: str
    fine_value: float
    apparent_order: float
    order_assumed: bool
    order_converged: bool
    convergence: ConvergenceType
    refinement_ratio: float
    extrapolated_value: float | None
    approximate_relative_error: float
    extrapolated_relative_error: float | None
    factor_of_safety: float
    gci_fine: float
    n_grids: int
    comfortable_refinement: bool

    @property
    def is_oscillatory(self) -> bool:
        """True when ``eps32/eps21 < 0``."""
        return self.convergence.is_oscillatory

    @property
    def expanded_numerical_uncertainty(self) -> float:
        """``GCI * |phi1|`` in the units of the quantity: a ~95% band."""
        return self.gci_fine * abs(self.fine_value)

    @property
    def standard_numerical_uncertainty(self) -> float:
        """``u_h`` for V&V 20: the expanded band divided by ``k = 2``."""
        return self.expanded_numerical_uncertainty / GCI_COVERAGE_FACTOR

    def summary(self) -> str:
        """A statement fit for a run manifest."""
        order = (
            f"p={self.apparent_order:.3g} (assumed)"
            if self.order_assumed
            else f"p={self.apparent_order:.3g}"
        )
        lines = [
            f"{self.quantity or 'quantity'}: phi1={self.fine_value:.6g}, "
            f"{order}, r21={self.refinement_ratio:.3g}, "
            f"GCI={self.gci_fine:.3%} "
            f"(u_h={self.standard_numerical_uncertainty:.4g}), "
            f"{self.convergence.value} over {self.n_grids} grids"
        ]
        if self.is_oscillatory:
            lines.append(
                "  oscillatory convergence: the Richardson estimate is not "
                "trustworthy, and this quantity is counted in the study's "
                "oscillatory percentage rather than hidden"
            )
        if not self.comfortable_refinement:
            lines.append(
                f"  refinement ratio {self.refinement_ratio:.3g} is below the "
                f"{COMFORTABLE_REFINEMENT_RATIO} Celik calls desirable"
            )
        return "\n".join(lines)


def _relative_error(fine_value: float, other: float) -> float:
    """``|(phi1 - other)/phi1|``, with an explicit refusal at zero."""
    if abs(fine_value) < _MIN_DENOMINATOR:
        raise SolutionVerificationError(
            "the fine-grid value is zero, so a relative GCI is undefined; "
            "shift the quantity of interest or report an absolute band"
        )
    return abs((fine_value - other) / fine_value)


def grid_convergence_index(
    solutions: Sequence[GridSolution], *, quantity: str = ""
) -> GCIResult:
    """Celik's three-grid GCI, with the apparent order observed.

    Args:
        solutions: Three or more solutions. They are sorted by cell size
            and the three finest are used.
        quantity: Label for reports.

    Returns:
        The discretisation uncertainty of the finest solution.

    Raises:
        SolutionVerificationError: If fewer than three solutions are
            given, or the triplet cannot support an order estimate.
    """
    if len(solutions) < 3:
        raise SolutionVerificationError(
            f"a three-grid GCI needs at least three solutions, got "
            f"{len(solutions)}; use two_grid_gci() with an assumed order if "
            "only two grids exist"
        )
    ordered = sorted(solutions, key=lambda item: item.cell_size_m)
    fine, medium, coarse = ordered[0], ordered[1], ordered[2]
    r21 = medium.cell_size_m / fine.cell_size_m
    r32 = coarse.cell_size_m / medium.cell_size_m
    order = apparent_order(
        epsilon_21=medium.value - fine.value,
        epsilon_32=coarse.value - medium.value,
        r21=r21,
        r32=r32,
    )
    error_21 = _relative_error(fine.value, medium.value)
    amplification = error_amplification(r21, order.order)
    extrapolated: float | None = None
    extrapolated_error: float | None = None
    if order.convergence.supports_richardson:
        extrapolated = richardson_extrapolate(
            fine_value=fine.value,
            coarse_value=medium.value,
            refinement_ratio=r21,
            order=order.order,
        )
        extrapolated_error = _relative_error(extrapolated, fine.value)
    return GCIResult(
        quantity=quantity,
        fine_value=fine.value,
        apparent_order=order.order,
        order_assumed=False,
        order_converged=order.converged,
        convergence=order.convergence,
        refinement_ratio=r21,
        extrapolated_value=extrapolated,
        approximate_relative_error=error_21,
        extrapolated_relative_error=extrapolated_error,
        factor_of_safety=FACTOR_OF_SAFETY_THREE_GRID,
        gci_fine=FACTOR_OF_SAFETY_THREE_GRID * error_21 / amplification,
        n_grids=len(ordered),
        comfortable_refinement=r21 >= COMFORTABLE_REFINEMENT_RATIO,
    )


def two_grid_gci(
    fine: GridSolution,
    coarse: GridSolution,
    *,
    assumed_order: float,
    quantity: str = "",
) -> GCIResult:
    """The two-grid GCI, with ``Fs = 3.0`` because ``p`` had to be assumed.

    Celik's factor of safety triples when the order cannot be observed.
    That is not conservatism for its own sake: with two grids there is no
    evidence at all that the solution is in its asymptotic range.

    Args:
        fine: The finer solution.
        coarse: The coarser solution.
        assumed_order: The formal order of the scheme, assumed.
        quantity: Label for reports.

    Returns:
        The discretisation uncertainty of the finer solution.

    Raises:
        SolutionVerificationError: If the grids are not ordered, or the
            assumed order is not positive.
    """
    if coarse.cell_size_m <= fine.cell_size_m:
        raise SolutionVerificationError(
            f"the coarse grid must have the larger cell size, got "
            f"{coarse.cell_size_m!r} against {fine.cell_size_m!r}"
        )
    if not math.isfinite(assumed_order) or assumed_order <= 0.0:
        raise SolutionVerificationError(
            f"assumed order must be positive, got {assumed_order!r}"
        )
    r21 = coarse.cell_size_m / fine.cell_size_m
    error_21 = _relative_error(fine.value, coarse.value)
    amplification = error_amplification(r21, assumed_order)
    extrapolated = richardson_extrapolate(
        fine_value=fine.value,
        coarse_value=coarse.value,
        refinement_ratio=r21,
        order=assumed_order,
    )
    return GCIResult(
        quantity=quantity,
        fine_value=fine.value,
        apparent_order=assumed_order,
        order_assumed=True,
        order_converged=True,
        convergence=ConvergenceType.MONOTONIC,
        refinement_ratio=r21,
        extrapolated_value=extrapolated,
        approximate_relative_error=error_21,
        extrapolated_relative_error=_relative_error(extrapolated, fine.value),
        factor_of_safety=FACTOR_OF_SAFETY_TWO_GRID,
        gci_fine=FACTOR_OF_SAFETY_TWO_GRID * error_21 / amplification,
        n_grids=2,
        comfortable_refinement=r21 >= COMFORTABLE_REFINEMENT_RATIO,
    )


@dataclass(frozen=True)
class GCIStudy:
    """A set of GCI results, reporting oscillation as a percentage.

    Attributes:
        results: One entry per quantity of interest.
    """

    results: tuple[GCIResult, ...]

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            SolutionVerificationError: If the study is empty.
        """
        if not self.results:
            raise SolutionVerificationError(
                "a grid-convergence study needs at least one quantity"
            )

    @property
    def oscillatory_fraction(self) -> float:
        """Share of quantities that converged oscillatorily."""
        oscillatory = sum(1 for result in self.results if result.is_oscillatory)
        return oscillatory / len(self.results)

    @property
    def oscillatory_percentage(self) -> float:
        """:attr:`oscillatory_fraction` as a percentage."""
        return 100.0 * self.oscillatory_fraction

    @property
    def worst_gci(self) -> float:
        """Largest fractional GCI in the study."""
        return max(result.gci_fine for result in self.results)

    @property
    def numerical_uncertainty(self) -> float:
        """The largest ``u_h`` in the study, in the quantities' own units.

        Taking the maximum rather than a mean is deliberate: ``u_num``
        feeds a validation interval, and averaging away the worst-resolved
        quantity narrows that interval on quantities it was never
        estimated for.
        """
        return max(result.standard_numerical_uncertainty for result in self.results)

    def summary(self) -> str:
        """A multi-line statement fit for the credibility statement."""
        lines = [
            f"grid-convergence study over {len(self.results)} quantity(ies): "
            f"worst GCI {self.worst_gci:.3%}, "
            f"{self.oscillatory_percentage:.0f}% oscillatory"
        ]
        lines.extend(result.summary() for result in self.results)
        return "\n".join(lines)
