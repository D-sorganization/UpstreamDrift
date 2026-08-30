"""The workbench's uncertainty surface (issue #9243).

Everything the workbench knows about *how wide* its answers are lives here:
the value objects that carry a band, the propagation of the accelerated-mass
interval through launch and flight, and the assembly of the budget a design
comparison is ranked on. Split out of :mod:`~src.tools.bunker_shot_gui.model`
rather than added to it because the module-size budget is 1200 lines and
because these are one concern -- a reader asking "where does the width come
from?" should have one file to open.

Nothing here imports the model, so there is no cycle; the model imports and
re-exports these names, and every existing ``from .model import ...`` keeps
working.

The three things this file is careful about
-------------------------------------------

**Propagation, not scaling.** :func:`propagate_carry_band` re-runs the launch
and flight models at each edge of
:class:`~bunkershot3d.metrics.divot.AcceleratedSandMass` rather than scaling
the central carry, because the chain is nowhere linear: ejecta speed goes as
``J / m``, the added-mass term ``m_b / (m_int + m_b)`` falls with the mass, and
carry against launch speed is a drag integral. The map is monotone
**decreasing**, so the interval's lower mass edge produces the band's *upper*
carry edge.

**Edge-wise aggregation.** :func:`objective_band` averages the per-condition
objective bands edge-wise, never in quadrature. One mass interval applies at
every delivery condition at once, so the width does not average out; treating
it as replicate noise would report a twenty-five point sweep as five times
more certain than a one-point one, on a width no number of extra conditions
can reduce.

**Terms with no number are named, not omitted.**
:data:`TRANSFER_EFFICIENCY_UNQUANTIFIED` and
:data:`LAUNCH_DIRECTION_UNQUANTIFIED` are model-form gaps that scale the
answer and that issue #8616 found no measurement anywhere to bound;
:data:`CARRY_NUMERICAL_UNQUANTIFIED` is the discretisation study nobody has
run for carry. Every budget this module builds carries all three, so its band
is always reported as a lower bound on the spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.ball import BunkerShotState, compute_bunker_launch
from bunkershot3d.metrics import PlayabilityWindow
from bunkershot3d.solvers import ValidityVerdict
from bunkershot3d.vandv.band import ConsistencyBand
from bunkershot3d.vandv.budget import (
    NumericalBasis,
    UncertaintyBudget,
    UncertaintyClass,
    UncertaintyTerm,
    UnquantifiedTerm,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "CARRY_BAND_SOURCE",
    "CARRY_NUMERICAL_UNQUANTIFIED",
    "LAUNCH_DIRECTION_UNQUANTIFIED",
    "TRANSFER_EFFICIENCY_UNQUANTIFIED",
    "CarryEstimate",
    "CarrySweep",
    "PlayabilityOutcome",
    "objective_band",
    "objective_budget",
    "propagate_carry_band",
]

TRANSFER_EFFICIENCY_UNQUANTIFIED = UnquantifiedTerm(
    name="ball momentum transfer efficiency",
    uncertainty_class=UncertaintyClass.MODEL_FORM,
    reason=(
        "BALL_MOMENTUM_TRANSFER_EFFICIENCY is a stated placeholder of 0.5 "
        "that scales the answer, and issue #8616 found no bunker, sand or "
        "wedge-interaction paper anywhere to bound it with. A band for it "
        "would have to be invented, which is the failure this whole surface "
        "exists to avoid, so it is named and excluded instead"
    ),
)
"""The uncalibrated constant no propagated band here can account for."""

LAUNCH_DIRECTION_UNQUANTIFIED = UnquantifiedTerm(
    name="launch direction out of the sand",
    uncertainty_class=UncertaintyClass.MODEL_FORM,
    reason=(
        "launch angle is taken from the effective loft, but the momentum the "
        "head puts into the bed points forward and *down* and it is the free "
        "surface -- not modelled at any tier -- that turns the ejecta up. No "
        "published launch angle out of sand exists to size the difference "
        "(issue #8616)"
    ),
)
"""The second model-form gap that carry rests on and nobody has measured."""

CARRY_NUMERICAL_UNQUANTIFIED = UnquantifiedTerm(
    name="discretisation uncertainty of carry",
    uncertainty_class=UncertaintyClass.NUMERICAL,
    basis=NumericalBasis.SPATIAL,
    reason=(
        "no grid- or step-convergence study has been run for carry itself. "
        "The shipped DRFT study (vandv.studies.surface_refinement_study) "
        "refines a cylinder inertial force, not this chain, and the F1 study "
        "the mass interval's in-plane factor was read off "
        "(solvers.mpm.verification.column_grid_convergence) holds the Courant "
        "number fixed, so its GCI is a SPACE-TIME band and cannot be reported "
        "as a spatial u_h (ADR-0033). Registered as unquantified rather than "
        "assumed zero"
    ),
)
"""The numerical term this comparison has not measured, said out loud."""

CARRY_BAND_SOURCE = (
    "propagated from DivotMetrics.accelerated_mass (issue #8659) by "
    "re-running the launch and flight models at each edge of the interval"
)
"""Where the carry band's width comes from, for the budget's provenance."""


@dataclass(frozen=True, slots=True)
class CarryEstimate:
    """A carry number and the verdict it may only ever be quoted with.

    Issue #8657: carry is derived from the impulse the solver delivered and
    the divot mass the metrics layer measured, through an **uncalibrated**
    transfer efficiency, and there is no published measurement of ball speed
    or launch angle out of sand to calibrate it against (issue #8616). Pairing
    the number with its verdict in one value object is what stops the two
    being separated on the way to a display.

    Since issue #9243 the mass is an interval rather than a point, so the
    carry is too: :attr:`band` is the same launch and flight models re-run at
    each edge of :class:`~bunkershot3d.metrics.divot.AcceleratedSandMass`.
    The edges swap on the way through -- the *lower* mass shares the delivered
    impulse among less sand and throws the ball *further* -- which is why the
    band is propagated rather than scaled.

    Attributes:
        carry_m: Carry distance [m], at the interval's central mass.
        verdict: The shot's verdict combined with the launch model's own,
            never better than ``BEYOND_VALIDATION``.
        band: Carry at the two edges of the accelerated-mass interval, or
            ``None`` when the delivery carried no interval to propagate.
        band_reasons: What had to be said about the band while building it --
            an edge clipped to the momentum floor, most often. Empty when the
            interval propagated untouched.
    """

    carry_m: float
    verdict: ValidityVerdict
    band: ConsistencyBand | None = None
    band_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Keep the band and the number it is the band of together.

        Raises:
            ValueError: If the band's centre is not the reported carry. A
                band around a different number is a different claim, and a
                display showing one beside the other would be reporting two
                shots as one.
        """
        if self.band is None:
            return
        if not math.isclose(self.band.central, self.carry_m, rel_tol=1e-12):
            raise ValueError(
                f"the carry band is centred on {self.band.central!r} m but the "
                f"reported carry is {self.carry_m!r} m; a band around a "
                "different number is a different claim"
            )


def propagate_carry_band(
    state: BunkerShotState,
    central_m: float,
    fly: Callable[[BunkerShotState], float],
) -> tuple[ConsistencyBand | None, tuple[str, ...]]:
    """Re-run launch and flight at each edge of the mass interval.

    Propagated rather than scaled: see the module docstring for why the chain
    is nowhere linear, and why the interval's lower mass edge produces the
    band's upper carry edge.

    The lower mass edge can be **inadmissible**: below
    :attr:`~bunkershot3d.ball.SandDelivery.admissible_mass_floor_kg` the
    delivered impulse would need ejecta faster than the head, and
    :class:`~bunkershot3d.ball.SandDelivery` refuses to be built there. The
    edge is clipped to that floor and the clip is reported, because a truncated
    band is a narrower claim than the interval it came from and must not read
    as though the interval were narrower.

    Args:
        state: The shot at the interval's central mass.
        central_m: Carry at that central mass [m].
        fly: Evaluates one shot state to a carry distance [m]. Injected rather
            than imported so this function has no opinion about which flight
            kernel is installed.

    Returns:
        The band and the reasons it carries, or ``(None, ())`` when the
        delivery had no interval behind it.
    """
    bounds = state.delivery.displaced_mass_bounds_kg
    if bounds is None:
        return None, ()
    floor_kg = state.delivery.admissible_mass_floor_kg
    reasons: list[str] = []
    carries: list[float] = []
    for edge_kg in bounds:
        usable_kg = max(float(edge_kg), floor_kg)
        if usable_kg > float(edge_kg):
            reasons.append(
                f"the mass interval's {float(edge_kg) * 1e3:.4g} g edge is "
                f"below the momentum floor of {floor_kg * 1e3:.4g} g, so the "
                "carry band is CLIPPED there and is narrower than the mass "
                "interval it was propagated from (issue #8659)"
            )
        carries.append(fly(_at_mass(state, usable_kg)))
    low, high = sorted(carries)
    if not low <= central_m <= high:
        reasons.append(
            f"the central carry {central_m:.4g} m fell outside the edges' "
            f"carries [{low:.4g}, {high:.4g}] m, so the band was widened to "
            "contain it rather than reported as an interval it is not inside; "
            "the mass-to-carry map is not monotone here"
        )
        low, high = min(low, central_m), max(high, central_m)
    return (
        ConsistencyBand(lower=low, central=central_m, upper=high),
        tuple(reasons),
    )


def _at_mass(state: BunkerShotState, mass_kg: float) -> BunkerShotState:
    """The same shot with its accelerated mass pinned to one value.

    The interval is dropped on the copy rather than carried: an edge
    evaluation is a probe of the band, not a claim that this mass came from
    an interval of its own.

    Args:
        state: The shot to copy.
        mass_kg: The mass to pin [kg].

    Returns:
        The pinned shot.
    """
    return replace(
        state,
        delivery=replace(
            state.delivery,
            displaced_mass_kg=mass_kg,
            displaced_mass_bounds_kg=None,
        ),
    )


@dataclass(frozen=True)
class PlayabilityOutcome:
    """The playability window, or the reason there is not one.

    Attributes:
        window: The measured window, or ``None`` when it could not be
            measured.
        unavailable_reason: Why, when ``window`` is ``None``.
        carry_m: ``(na, nb)`` carry grid [m]; NaN where the solver refused.
        attack_angle_deg: ``(na,)`` swept attack angles, for display.
        firmness_kg_per_cm2: ``(nb,)`` swept penetrometer readings.
        carry_verdict: The worst verdict over the answered grid points, which
            the whole grid must be read under. Present whenever any cell of
            ``carry_m`` is finite.
        carry_lower_m: ``(na, nb)`` carry at the accelerated-mass interval's
            *upper* mass edge -- the shortest carry the models are consistent
            with. NaN wherever ``carry_m`` is, and wherever the delivery
            carried no interval to propagate (issue #9243).
        carry_upper_m: ``(na, nb)`` carry at the interval's *lower* mass edge,
            the longest carry the models are consistent with.
        band_reasons: Everything the band propagation had to say over the
            whole grid, de-duplicated.
    """

    window: PlayabilityWindow | None
    unavailable_reason: str = ""
    carry_m: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    attack_angle_deg: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    firmness_kg_per_cm2: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    carry_verdict: ValidityVerdict | None = None
    carry_lower_m: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    carry_upper_m: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    band_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the carry-verdict rule on the grid as well as the shot.

        Raises:
            ValueError: If any grid point holds a carry number and no verdict
                accompanies the grid (issue #8657), or if a band grid is
                present and does not bracket the central grid it belongs to.
        """
        if bool(np.isfinite(self.carry_m).any()) and self.carry_verdict is None:
            raise ValueError(
                "a carry grid and its validity verdict travel together; "
                f"{int(np.isfinite(self.carry_m).sum())} point(s) carry a "
                "number with no verdict to read them under"
            )
        self._require_bracketing_bands()

    def _require_bracketing_bands(self) -> None:
        """Refuse band grids that do not contain the grid they describe.

        Raises:
            ValueError: If the band grids disagree in shape with the central
                grid, or if a banded cell does not bracket its central value.
        """
        for name in ("carry_lower_m", "carry_upper_m"):
            grid = getattr(self, name)
            if grid.size and grid.shape != self.carry_m.shape:
                raise ValueError(
                    f"{name} has shape {grid.shape} against a carry grid of "
                    f"{self.carry_m.shape}; a band that does not line up with "
                    "the number it bands is not a band"
                )
        if not (self.carry_lower_m.size and self.carry_upper_m.size):
            return
        banded = (
            np.isfinite(self.carry_lower_m)
            & np.isfinite(self.carry_upper_m)
            & np.isfinite(self.carry_m)
        )
        inside = (self.carry_lower_m <= self.carry_m) & (
            self.carry_m <= self.carry_upper_m
        )
        offenders = int(np.count_nonzero(banded & ~inside))
        if offenders:
            raise ValueError(
                f"{offenders} grid point(s) report a carry outside their own "
                "band; a point estimate outside its band is a different claim, "
                "not a narrower one"
            )

    @property
    def available(self) -> bool:
        """True when a window was measured."""
        return self.window is not None

    @property
    def has_bands(self) -> bool:
        """Whether any grid point carries a propagated band."""
        return bool(self.carry_lower_m.size and np.isfinite(self.carry_lower_m).any())


class CarrySweep:
    """Accumulator for one design's carry grid **and its band grids**.

    A class rather than three loose arrays: the central grid and its two edges
    are only meaningful together, and a caller that could fill one and forget
    another would produce a band that does not bracket its own value.
    :meth:`record` writes all three or none.
    """

    def __init__(self, points: int) -> None:
        """Allocate the three grids and the reason accumulators.

        Args:
            points: Stations per axis; the grids are ``points x points``.
        """
        shape = (points, points)
        self.carry = np.full(shape, np.nan, dtype=np.float64)
        self.lower = np.full(shape, np.nan, dtype=np.float64)
        self.upper = np.full(shape, np.nan, dtype=np.float64)
        self.reasons: list[str] = []
        self.verdicts: list[ValidityVerdict] = []
        self._band_reasons: list[str] = []

    def record(self, row: int, column: int, estimate: CarryEstimate | None) -> None:
        """Store one grid point, or leave it unanswered.

        Args:
            row: Attack-angle index.
            column: Firmness index.
            estimate: The carry at that point, or ``None`` when the point was
                refused.
        """
        if estimate is None:
            return
        self.carry[row, column] = estimate.carry_m
        self.verdicts.append(estimate.verdict)
        self._band_reasons.extend(estimate.band_reasons)
        if estimate.band is None:
            return
        self.lower[row, column] = estimate.band.lower
        self.upper[row, column] = estimate.band.upper

    def unique_band_reasons(self) -> tuple[str, ...]:
        """The band reasons, de-duplicated and in first-seen order.

        A clipped mass edge fires at most grid points at once and repeating it
        twenty-five times in a report buries it.

        Returns:
            The distinct reasons.
        """
        return tuple(dict.fromkeys(self._band_reasons))


def objective_band(
    playability: PlayabilityOutcome,
    shared: NDArray[np.bool_],
    target_carry_m: float,
) -> ConsistencyBand | None:
    """Mean absolute carry error, as a band, over the shared conditions.

    Two steps and both matter. Each grid point's carry band is mapped through
    ``|carry - target|``, which is V-shaped: a band that straddles the target
    reaches zero, so the objective's floor is not the image of a carry edge.
    Then the per-point bands are averaged **edge-wise**, for the reason the
    module docstring gives.

    Args:
        playability: The design's playability outcome, carrying the grids.
        shared: Boolean mask over the flattened grid.
        target_carry_m: The carry the objective measures distance to.

    Returns:
        The objective band, or ``None`` when no shared point carried one.
    """
    if not playability.has_bands:
        return None
    central = playability.carry_m.ravel()[shared]
    lower = playability.carry_lower_m.ravel()[shared]
    upper = playability.carry_upper_m.ravel()[shared]
    usable = np.isfinite(central) & np.isfinite(lower) & np.isfinite(upper)
    if not usable.any():
        return None
    bands = [
        ConsistencyBand(
            lower=float(low), central=float(mid), upper=float(high)
        ).absolute_deviation_from(target_carry_m)
        for low, mid, high in zip(
            lower[usable], central[usable], upper[usable], strict=True
        )
    ]
    return ConsistencyBand.mean(bands)


def objective_budget(
    design_name: str,
    playability: PlayabilityOutcome,
    shared: NDArray[np.bool_],
    *,
    target_carry_m: float,
    sampling_std_error: float,
) -> UncertaintyBudget:
    """Assemble one design's uncertainty budget for the ranking objective.

    The objective is the mean absolute carry error against the target over the
    shared delivery sweep, and three separate things make it uncertain:

    * ``MODEL_FORM`` -- the accelerated-mass interval of issue #8659,
      propagated through launch and flight at each grid point and then
      averaged edge-wise.
    * ``SAMPLING`` -- the bootstrap standard error of the mean over the
      answered conditions, which is the only uncertainty the shipped
      comparison ever saw.
    * ``NUMERICAL`` -- absent, and registered as **unquantified** rather than
      assumed zero: see :data:`CARRY_NUMERICAL_UNQUANTIFIED`.

    Args:
        design_name: The design the budget is for, for the report.
        playability: That design's playability outcome, carrying the grids.
        shared: Boolean mask over the flattened grid, true where both designs
            answered.
        target_carry_m: The carry the objective measures distance to.
        sampling_std_error: Bootstrap standard error of this design's mean
            objective, from
            :func:`~bunkershot3d.study.comparison.compare_designs`.

    Returns:
        The budget, whose centre is the mean absolute carry error.
    """
    central = float(
        np.mean(np.abs(playability.carry_m.ravel()[shared] - target_carry_m))
    )
    terms = [
        UncertaintyTerm.symmetric(
            name="delivery sweep (finite grid)",
            uncertainty_class=UncertaintyClass.SAMPLING,
            half_width=sampling_std_error,
            source=(
                "bootstrap standard error of the mean over the "
                f"{int(shared.sum())} shared delivery conditions"
            ),
        )
    ]
    band = objective_band(playability, shared, target_carry_m)
    if band is not None:
        terms.append(
            UncertaintyTerm.from_band(
                name="accelerated sand mass (#8659)",
                uncertainty_class=UncertaintyClass.MODEL_FORM,
                band=band,
                source=CARRY_BAND_SOURCE,
            )
        )
    return UncertaintyBudget(
        quantity=f"{design_name}: mean absolute carry error",
        central=central,
        terms=tuple(terms),
        unquantified=(
            TRANSFER_EFFICIENCY_UNQUANTIFIED,
            LAUNCH_DIRECTION_UNQUANTIFIED,
            CARRY_NUMERICAL_UNQUANTIFIED,
        ),
    )
