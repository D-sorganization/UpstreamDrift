"""Order of accuracy under refinement (issue #8616).

Code verification.  **No experimental data appears in this module.**

Two things are refined in this package, and they are different
--------------------------------------------------------------

* **The surface discretisation.**  The DRFT force is an integral of a
  local stress response over the swept surface, so halving the element
  size must make the quadrature error fall as ``h^p``.  With an exact
  answer available (:mod:`bunkershot3d.vandv.cases`) this is a true
  order-of-accuracy test against a known solution, not a Richardson
  estimate.
* **The timestep.**  Only for truncation-class conservation residuals.
  A round-off-class residual is refused outright, because it does not
  decay with the step at all.

Note the trap the addendum records for anyone extending this to an F1/F2
tier: at about 3.5 particles per cell, standard piecewise-linear MPM
**fails to converge beyond roughly 20 grid cells -- the error increases
with refinement**.  A grid-convergence study run on a linear-basis MPM
reports a number that means nothing.  The same direction-of-refinement
trap exists in F0's own envelope: ``I_G = v^2 d^2 / (g lambda^2)`` grows
as the surface mesh is refined, so a mesh fine enough to converge the
quadrature is a mesh far enough outside RFT's superposition argument that
the converged answer is a different kind of wrong.  The refinement study
is therefore verification and never validation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .conservation import ConservationResidual, residual_series
from .exceptions import VerificationError

__all__ = [
    "ObservedOrder",
    "RefinementLevel",
    "observed_order_from_errors",
    "observed_order_from_residuals",
    "refinement_errors",
]


@dataclass(frozen=True, slots=True)
class RefinementLevel:
    """One level of a refinement study.

    Attributes:
        cell_size_m: Representative cell size ``h``, or the timestep when
            the study refines in time.
        value: The computed quantity at this level.
        label: Optional human label.
    """

    cell_size_m: float
    value: float
    label: str = ""

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            VerificationError: If the size or value is unusable.
        """
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise VerificationError(
                f"refinement size must be positive and finite, got {self.cell_size_m!r}"
            )
        if not math.isfinite(self.value):
            raise VerificationError(
                f"refinement value must be finite, got {self.value!r}"
            )


@dataclass(frozen=True, slots=True)
class ObservedOrder:
    """The observed order of accuracy of a refinement series.

    Attributes:
        order: Least-squares slope of ``log|error|`` against ``log h``.
        pairwise_orders: One order per consecutive pair, coarse to fine.
        sizes: The refinement sizes, ascending.
        errors: The corresponding errors, in the same order.
        monotone: Whether the error fell at every refinement step.
    """

    order: float
    pairwise_orders: tuple[float, ...]
    sizes: tuple[float, ...]
    errors: tuple[float, ...]
    monotone: bool

    @property
    def spread(self) -> float:
        """Largest gap between any two pairwise orders.

        A wide spread means the series is not in its asymptotic range, so
        the fitted order is a summary of a transient rather than an
        observation of the scheme.
        """
        return max(self.pairwise_orders) - min(self.pairwise_orders)

    def summary(self) -> str:
        """A statement fit for a verification report."""
        pairs = ", ".join(f"{order:.3f}" for order in self.pairwise_orders)
        return (
            f"observed order {self.order:.3f} over {len(self.sizes)} levels "
            f"(pairwise {pairs}; spread {self.spread:.3f}; "
            f"{'monotone' if self.monotone else 'NOT monotone'})"
        )


def refinement_errors(
    levels: Sequence[RefinementLevel], *, exact_value: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Absolute errors of a refinement series against a known answer.

    Args:
        levels: The refinement levels, in any order.
        exact_value: The closed-form answer.

    Returns:
        ``(sizes, errors)`` sorted by ascending size.

    Raises:
        VerificationError: If fewer than two levels are given, if the
            exact value is not finite, or if any level lands exactly on
            the exact value -- where the log of the error is undefined and
            the study cannot proceed.
    """
    if len(levels) < 2:
        raise VerificationError(
            f"an order-of-accuracy study needs at least two levels, got {len(levels)}"
        )
    if not math.isfinite(exact_value):
        raise VerificationError(f"exact value must be finite, got {exact_value!r}")
    ordered = sorted(levels, key=lambda level: level.cell_size_m)
    sizes = tuple(level.cell_size_m for level in ordered)
    errors = tuple(abs(level.value - exact_value) for level in ordered)
    if any(error == 0.0 for error in errors):
        raise VerificationError(
            "a refinement level reproduced the exact value to the last bit, so "
            "its log-error is undefined. Either the quadrature is exact for "
            "this case -- in which case there is no order to observe -- or the "
            "'exact' value was taken from the same computation."
        )
    return (sizes, errors)


def observed_order_from_errors(
    sizes: Sequence[float], errors: Sequence[float]
) -> ObservedOrder:
    """Fit ``error ~ C h^p`` and report ``p``.

    The fit is least squares on ``log|error|`` against ``log h`` over all
    levels, and the pairwise orders are reported alongside so that a
    series which is not yet asymptotic is visible rather than averaged
    into a plausible-looking single number.

    Args:
        sizes: Refinement sizes, ascending or descending.
        errors: Absolute errors at each size.

    Returns:
        The observed order.

    Raises:
        VerificationError: If the series is too short, mismatched, or
            contains a non-positive size or error.
    """
    if len(sizes) != len(errors):
        raise VerificationError(
            f"got {len(sizes)} sizes and {len(errors)} errors; a refinement "
            "series needs one error per size"
        )
    if len(sizes) < 2:
        raise VerificationError(
            f"an order fit needs at least two levels, got {len(sizes)}"
        )
    size_array = np.asarray(sizes, dtype=np.float64)
    error_array = np.asarray(errors, dtype=np.float64)
    if not (np.all(size_array > 0.0) and np.all(np.isfinite(size_array))):
        raise VerificationError("every refinement size must be positive and finite")
    if not (np.all(error_array > 0.0) and np.all(np.isfinite(error_array))):
        raise VerificationError(
            "every error must be positive and finite; a zero error has no "
            "logarithm and usually means the reference came from the same code"
        )
    order_index = np.argsort(size_array)
    size_array = size_array[order_index]
    error_array = error_array[order_index]

    log_sizes = np.log(size_array)
    log_errors = np.log(error_array)
    slope = float(np.polyfit(log_sizes, log_errors, 1)[0])
    pairwise = tuple(
        float(
            (log_errors[index + 1] - log_errors[index])
            / (log_sizes[index + 1] - log_sizes[index])
        )
        for index in range(len(size_array) - 1)
    )
    monotone = bool(np.all(np.diff(error_array) > 0.0))
    return ObservedOrder(
        order=slope,
        pairwise_orders=pairwise,
        sizes=tuple(float(value) for value in size_array),
        errors=tuple(float(value) for value in error_array),
        monotone=monotone,
    )


def observed_order_from_residuals(
    residuals: Sequence[ConservationResidual],
) -> ObservedOrder:
    """Fit the decay order of a truncation-class residual series.

    This is the *only* correct test for the truncation class, and it is
    forbidden for the round-off class:
    :func:`~bunkershot3d.vandv.conservation.residual_series` raises if any
    residual is round-off class.

    Args:
        residuals: Truncation-class residuals across a step refinement.

    Returns:
        The observed order of the residual decay.

    Raises:
        ConservationClassError: If any residual is round-off class.
        VerificationError: If the series is unusable.
    """
    steps, values = residual_series(residuals)
    return observed_order_from_errors(steps, values)
