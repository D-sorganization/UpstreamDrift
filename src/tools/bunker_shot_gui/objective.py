"""May the carry objective rank two designs? (issues #9239, #9545).

The workbench's A/B comparison ranks two soles on absolute carry error over a
shared delivery sweep. Carry rests on a sand-to-ball transfer that no
measurement has qualified (#8616, #9543), and the corrected divot mass leaves
the nominal playability window empty (#9239). A finite, tightly banded carry
is therefore still not evidence that one design is better.

This module decides, once, what the comparison may claim. It reuses the
canonical :class:`~bunkershot3d.ball.ObjectiveDisposition` vocabulary rather
than inventing a parallel one, and adds the two grid facts that make a
*qualified* objective degenerate in practice: a window that is empty for both
designs, and two designs the objective cannot tell apart at any condition.

The exploratory diagnostics are kept -- the comparison still bootstraps and
bands the two designs -- but only a :attr:`CarryObjectiveStanding.predictive`
standing lets a winner be named.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.ball import ObjectiveDisposition

__all__ = [
    "MIN_RANKABLE_CONDITIONS",
    "SHIPPED_CARRY_DISPOSITION",
    "UNCALIBRATED_REASON",
    "CarryObjectiveStanding",
    "carry_objective_standing",
    "uncalibrated_standing",
]

SHIPPED_CARRY_DISPOSITION = ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED
"""The disposition of the carry objective in the shipped workbench.

No :class:`~bunkershot3d.ball.TransferQualification` ships, so this is the
only disposition :func:`bunkershot3d.ball.objective_disposition` can return
for it. A future qualification (#9694) changes this by supplying evidence,
never by editing the constant."""

MIN_RANKABLE_CONDITIONS = 2
"""Fewest shared delivery conditions a ranking can be built on."""

DESIGNS_COMPARED = 2
"""An A/B comparison: one row of objective values per design."""

_GRID_AXES = ("design", "condition")
"""The axes of the shared objective grid, in order."""

UNCALIBRATED_REASON = (
    "no sand-to-ball transfer qualification exists (#9239, #9543), so carry is "
    "a placeholder product and cannot rank designs as predictive"
)

_DISPOSITION_REASONS: dict[ObjectiveDisposition, str] = {
    ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED: UNCALIBRATED_REASON,
    ObjectiveDisposition.UNAVAILABLE_OUTSIDE_QUALIFIED_REGIME: (
        "the transfer qualification does not cover this strike, so carry "
        "cannot rank designs as predictive"
    ),
    ObjectiveDisposition.DEGENERATE_TARGET: (
        "the qualified nominal carry lies outside the target window, so the "
        "objective is degenerate at this target"
    ),
}
"""Why each non-supported disposition forbids a predictive ranking."""


@dataclass(frozen=True)
class CarryObjectiveStanding:
    """What a comparison on the carry objective is allowed to claim.

    Attributes:
        disposition: The canonical disposition of the objective.
        reasons: Why the objective cannot rank as predictive. Empty exactly
            when ``disposition`` is ``SUPPORTED``.

    Raises:
        ValueError: If a non-supported standing carries no reason, or a
            supported one carries a refusal reason.
    """

    disposition: ObjectiveDisposition
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        supported = self.disposition is ObjectiveDisposition.SUPPORTED
        if supported and self.reasons:
            raise ValueError(
                "a supported carry objective carries no refusal reason, got "
                f"{self.reasons!r}"
            )
        if not supported and not self.reasons:
            raise ValueError(
                f"a {self.disposition.value} carry objective must state the "
                "reason it cannot rank as predictive"
            )

    @property
    def predictive(self) -> bool:
        """True only when the objective is qualified and informative."""
        return self.disposition is ObjectiveDisposition.SUPPORTED


def uncalibrated_standing() -> CarryObjectiveStanding:
    """The standing of the carry objective in the shipped workbench."""
    return CarryObjectiveStanding(
        disposition=SHIPPED_CARRY_DISPOSITION, reasons=(UNCALIBRATED_REASON,)
    )


def _require_rankable_grid(errors: NDArray[np.float64]) -> None:
    """Preconditions on the shared objective grid.

    Raises:
        ValueError: If the grid is not ``(2, n)`` with ``n`` at least
            :data:`MIN_RANKABLE_CONDITIONS`, or holds a non-finite value.
    """
    if errors.ndim != len(_GRID_AXES) or errors.shape[0] != DESIGNS_COMPARED:
        raise ValueError(
            f"the objective grid must be ({DESIGNS_COMPARED}, n), got {errors.shape}"
        )
    if errors.shape[1] < MIN_RANKABLE_CONDITIONS:
        raise ValueError(
            f"a ranking needs at least {MIN_RANKABLE_CONDITIONS} shared "
            f"conditions, got {errors.shape[1]}"
        )
    if not np.all(np.isfinite(errors)):
        raise ValueError(
            "the objective grid must be finite; refused cells are dropped "
            "before the standing is judged"
        )


def _grid_degeneracy(
    errors: NDArray[np.float64], window_half_width_m: float
) -> tuple[str, ...]:
    """The reasons the grid itself leaves nothing to rank on."""
    reasons: list[str] = []
    if np.all(errors > window_half_width_m):
        reasons.append(
            "the playability window is empty for both designs on the shared "
            f"grid: no answered condition carries within {window_half_width_m:g} m "
            "of the target (#9239)"
        )
    if np.array_equal(errors[0], errors[1]):
        reasons.append(
            "the two designs produce an identical objective at every shared "
            "condition, so the objective cannot tell them apart"
        )
    return tuple(reasons)


def carry_objective_standing(
    errors: NDArray[np.float64],
    *,
    disposition: ObjectiveDisposition,
    window_half_width_m: float,
) -> CarryObjectiveStanding:
    """Judge whether the carry objective may rank this pair as predictive.

    Args:
        errors: ``(2, n)`` absolute carry error [m] of each design at the
            ``n`` delivery conditions both answered.
        disposition: The objective's disposition from its canonical owner,
            :func:`bunkershot3d.ball.objective_disposition`, or
            :data:`SHIPPED_CARRY_DISPOSITION` when no qualification exists.
        window_half_width_m: Half-width of the carry window [m]; a condition
            whose error exceeds it lies outside the playability window.

    Returns:
        The standing. A ``SUPPORTED`` disposition is demoted to
        ``DEGENERATE_TARGET`` when the grid leaves nothing to rank on; any
        other disposition is kept, with the grid's own reasons added.

    Raises:
        ValueError: If the grid violates :func:`_require_rankable_grid` or the
            window is not a positive finite width.
    """
    if not (np.isfinite(window_half_width_m) and window_half_width_m > 0.0):
        raise ValueError(
            f"window_half_width_m must be positive and finite, got "
            f"{window_half_width_m!r}"
        )
    _require_rankable_grid(errors)
    degeneracy = _grid_degeneracy(errors, window_half_width_m)
    if disposition is ObjectiveDisposition.SUPPORTED:
        if not degeneracy:
            return CarryObjectiveStanding(disposition=disposition)
        return CarryObjectiveStanding(
            disposition=ObjectiveDisposition.DEGENERATE_TARGET, reasons=degeneracy
        )
    return CarryObjectiveStanding(
        disposition=disposition,
        reasons=(_DISPOSITION_REASONS[disposition], *degeneracy),
    )
