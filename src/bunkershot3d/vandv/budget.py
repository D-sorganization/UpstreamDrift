"""What a band is made of, kept in classes that do not mix (issue #9243).

Three classes, and the reason they stay apart
---------------------------------------------

:mod:`bunkershot3d.vandv.validation` already refuses to treat round-off and
truncation the same way -- ``u_h + u_it + u_ro`` add because they are three
correlated faces of one discrete solve, while ``u_num``, ``u_input`` and
``u_exp`` combine in quadrature because they come from independent sources.
This module extends that discipline one step outward, to the thing the
workbench actually ranks designs on:

``NUMERICAL``
    How wrong the *arithmetic* is: discretisation, iteration, round-off.
    Reducible by refining. This is the class V&V 20 calls ``u_num``, and the
    only class :meth:`UncertaintyBudget.as_v20_numerical` will map into one.

``MODEL_FORM``
    How wrong the *model* is: the accelerated-mass interval of issue #8659,
    an uncalibrated transfer efficiency, a wall shape nobody has measured.
    **Not** reducible by refining, and not a distribution. Refining the grid
    forever leaves it exactly where it was.

``SAMPLING``
    How few conditions were evaluated: the bootstrap spread over a delivery
    sweep. The one class here with an independence claim behind it, so the
    one class that combines in quadrature and does shrink with more points.

Summing a model-form band and a GCI into one number and calling the result
``u_num`` would say that a finer grid narrows the divot-mass interval. It does
not. So the three subtotals are always available separately
(:meth:`UncertaintyBudget.by_class`), the combined width is a named, documented
operation rather than a silent one (:meth:`UncertaintyBudget.band`), and
:meth:`UncertaintyBudget.as_v20_numerical` drops every non-numerical term on
the floor by construction.

The space-time trap ADR-0033 left behind
----------------------------------------

``bunkershot3d.solvers.mpm.verification.column_grid_convergence`` holds the
**Courant number fixed** while it refines, so its time step shrinks with its
cell size and the GCI it returns is a *space-time* band, not a spatial one.
That matters here because the in-plane half of the accelerated-mass interval
(:data:`bunkershot3d.metrics.divot.F1_ENTRAINMENT_FACTOR_BOUNDS`) was read off
that same F1 tier. Handing such a number to V&V 20 as ``u_h`` would report a
combined space-and-time discretisation error as a purely spatial one, and
would double-count if a separate temporal term were ever added beside it.

So every numerical term must declare a :class:`NumericalBasis`, and a
``SPACE_TIME`` term makes :meth:`UncertaintyBudget.as_v20_numerical` **refuse**
unless the caller passes ``space_time=`` to say, explicitly and in the call,
which V&V 20 component it is being folded into.

Terms nobody has a number for
-----------------------------

:data:`bunkershot3d.ball.splash.BALL_MOMENTUM_TRANSFER_EFFICIENCY` is 0.5 with
no measurement behind it and no published range to draw one from -- issue
#8616 found no bunker, sand or wedge-interaction paper at all. Inventing a
band for it would be fitting an uncalibrated constant, so it is registered as
an :class:`UnquantifiedTerm` instead: named, classed, sourced, and excluded
from the arithmetic. A budget holding one reports
:attr:`UncertaintyBudget.band_is_lower_bound`, and every statement it produces
says the width is a floor rather than a total.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from .band import ConsistencyBand
from .exceptions import VandVError
from .gci import ConvergenceType, GCIResult
from .validation import NumericalUncertainty

__all__ = [
    "DOMINANCE_SHARE",
    "ClassSubtotal",
    "DominantTerm",
    "NumericalBasis",
    "UncertaintyBudget",
    "UncertaintyClass",
    "UncertaintyTerm",
    "UnquantifiedTerm",
]

_DIVERGENT = frozenset(
    {
        ConvergenceType.OSCILLATORY_DIVERGENCE,
        ConvergenceType.MONOTONIC_DIVERGENCE,
    }
)
"""Refinement outcomes that produce no usable uncertainty at all."""

DOMINANCE_SHARE = 0.75
"""Share of the combined width above which one term is said to *swamp*.

A reporting threshold and not a physical one: past it, the comparison is
really a comparison about that single assumption, and saying so is more use
than the ranking it produced.
"""


class UncertaintyClass(StrEnum):
    """What kind of not-knowing a term describes."""

    NUMERICAL = "numerical"
    MODEL_FORM = "model-form"
    SAMPLING = "sampling"


class NumericalBasis(StrEnum):
    """What was refined to produce a numerical term.

    ``SPACE_TIME`` is the one that exists because of a real trap: a study that
    refines the mesh and the step together at fixed Courant number produces a
    band that is neither spatial nor temporal, and must not be reported as
    either without the caller saying so.
    """

    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    SPACE_TIME = "space-time"
    ITERATIVE = "iterative"
    ROUND_OFF = "round-off"


def _require_offset(name: str, value: float) -> float:
    """Coerce and check one side of a term.

    Args:
        name: Field name, for the message.
        value: The offset.

    Returns:
        The offset as a float.

    Raises:
        ValueError: If it is not finite and non-negative.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if number < 0.0:
        raise ValueError(
            f"{name} must be non-negative -- an offset is a distance from the "
            f"central value, never a direction -- got {value!r}"
        )
    return number


def _require_basis(
    uncertainty_class: UncertaintyClass, basis: NumericalBasis | None
) -> None:
    """Check that a term's basis matches its class.

    Args:
        uncertainty_class: The term's class.
        basis: The declared basis, or ``None``.

    Raises:
        ValueError: If a numerical term has no basis, or a non-numerical term
            has one.
    """
    if uncertainty_class is UncertaintyClass.NUMERICAL and basis is None:
        raise ValueError(
            "a numerical term must declare a basis: 'numerical' alone does not "
            "say what was refined, and a space-time GCI reported as a spatial "
            "u_h is the specific error this field exists to prevent (ADR-0033)"
        )
    if uncertainty_class is not UncertaintyClass.NUMERICAL and basis is not None:
        raise ValueError(
            f"a {uncertainty_class.value} term must not carry a numerical "
            f"basis, got {basis!r}: refining a grid does not narrow a "
            "model-form band, and saying it might is a category error"
        )


@dataclass(frozen=True, slots=True)
class UncertaintyTerm:
    """One named, sourced contribution to a quantity's band.

    Asymmetric on purpose. The accelerated-mass interval reaches much further
    up than down once it is pushed through the launch model, and collapsing it
    to a single half-width would throw that away.

    Attributes:
        name: What this term is, in a report.
        uncertainty_class: Which kind of not-knowing it is.
        lower_offset: How far below the central value this term reaches, in
            the quantity's own unit. Non-negative.
        upper_offset: How far above. Non-negative.
        source: Where the number came from. Required: a contribution with no
            provenance is not admissible in a budget that will be displayed.
        basis: What was refined, for numerical terms only.
    """

    name: str
    uncertainty_class: UncertaintyClass
    lower_offset: float
    upper_offset: float
    source: str
    basis: NumericalBasis | None = None

    def __post_init__(self) -> None:
        """Validate the term.

        Raises:
            ValueError: If an offset is negative or non-finite, if the source
                is blank, or if the basis does not match the class.
        """
        _require_offset("lower_offset", self.lower_offset)
        _require_offset("upper_offset", self.upper_offset)
        if not self.source.strip():
            raise ValueError(
                f"term {self.name!r} needs a source: a number with no "
                "provenance cannot be shown beside a design ranking"
            )
        _require_basis(self.uncertainty_class, self.basis)

    @classmethod
    def symmetric(
        cls,
        *,
        name: str,
        uncertainty_class: UncertaintyClass,
        half_width: float,
        source: str,
        basis: NumericalBasis | None = None,
    ) -> UncertaintyTerm:
        """A term that reaches equally far either side of the centre.

        Args:
            name: What the term is.
            uncertainty_class: Which kind of not-knowing.
            half_width: The offset, applied to both sides.
            source: Provenance.
            basis: What was refined, for numerical terms.

        Returns:
            The term.
        """
        return cls(
            name=name,
            uncertainty_class=uncertainty_class,
            lower_offset=half_width,
            upper_offset=half_width,
            source=source,
            basis=basis,
        )

    @classmethod
    def from_gci(
        cls,
        gci: GCIResult,
        *,
        basis: NumericalBasis,
        name: str | None = None,
    ) -> UncertaintyTerm:
        """Read a numerical term off a Grid Convergence Index study.

        The bridge from :mod:`bunkershot3d.vandv.gci` into a budget, so that a
        solution-verification result enters a design comparison as the
        ``NUMERICAL`` term it is rather than being re-derived by hand. The
        half-width is :attr:`~bunkershot3d.vandv.gci.GCIResult.standard_numerical_uncertainty`
        -- the expanded GCI band divided by ``k = 2``, which is what V&V 20
        calls ``u_h`` -- in the refined quantity's own units.

        ``basis`` is required and has no default, because the two shipped
        studies need different answers and getting it wrong is silent:
        ``vandv.studies.surface_refinement_study`` refines facet count at a
        fixed step and is ``SPATIAL``, while
        ``solvers.mpm.verification.column_grid_convergence`` holds the Courant
        number fixed so its step shrinks with its cell size, making it
        ``SPACE_TIME`` (ADR-0033). Declaring the second as the first would
        report a combined space-and-time error as a purely spatial one.

        Args:
            gci: The completed study for one quantity.
            basis: What the study actually refined.
            name: Term name; defaults to the GCI's own quantity label.

        Returns:
            The term, in the refined quantity's units. A caller applying it to
            a *different* quantity must scale it and say so in its source.

        Raises:
            VandVError: If the study diverged, where the Richardson estimate
                is not an uncertainty at all. Oscillatory convergence is
                admitted but the caveat is carried in the term's source, the
                same treatment :meth:`GCIResult.summary` gives it.
        """
        if gci.convergence in _DIVERGENT:
            raise VandVError(
                f"the GCI study for {gci.quantity!r} came back "
                f"{gci.convergence.value} -- it diverged under refinement, so "
                "its Richardson extrapolation is not an uncertainty and must "
                "not be entered in a budget as one"
            )
        source = (
            f"GCI ({basis.value}): {gci.gci_fine:.3%} of "
            f"phi1={gci.fine_value:.6g} over {gci.n_grids} grids, "
            f"p={gci.apparent_order:.3g}"
            f"{' (assumed)' if gci.order_assumed else ''}, "
            f"r21={gci.refinement_ratio:.3g}, {gci.convergence.value}"
        )
        if gci.is_oscillatory:
            source += (
                " -- OSCILLATORY, so the Richardson estimate is not "
                "trustworthy and this width is indicative"
            )
        return cls.symmetric(
            name=name or gci.quantity or "discretisation",
            uncertainty_class=UncertaintyClass.NUMERICAL,
            half_width=gci.standard_numerical_uncertainty,
            source=source,
            basis=basis,
        )

    @classmethod
    def from_band(
        cls,
        *,
        name: str,
        uncertainty_class: UncertaintyClass,
        band: ConsistencyBand,
        source: str,
        basis: NumericalBasis | None = None,
    ) -> UncertaintyTerm:
        """Read a term's two offsets off a propagated band.

        Args:
            name: What the term is.
            uncertainty_class: Which kind of not-knowing.
            band: The band the quantity was propagated into. Its centre is
                taken as the term's centre.
            source: Provenance.
            basis: What was refined, for numerical terms.

        Returns:
            The term, keeping the band's asymmetry.
        """
        return cls(
            name=name,
            uncertainty_class=uncertainty_class,
            lower_offset=band.central - band.lower,
            upper_offset=band.upper - band.central,
            source=source,
            basis=basis,
        )

    @property
    def width(self) -> float:
        """Total reach of the term, both sides together."""
        return self.lower_offset + self.upper_offset

    @property
    def half_width(self) -> float:
        """Half the total reach, for the symmetric V&V 20 components."""
        return 0.5 * self.width

    @property
    def uncertainty_class_value(self) -> str:
        """String representation of uncertainty class."""
        return self.uncertainty_class.value


@dataclass(frozen=True, slots=True)
class UnquantifiedTerm:
    """A contribution that is known to exist and has no number.

    Registering one is the honest alternative to two dishonest options:
    omitting it, which reads as a claim that it is zero, and inventing a range
    for it, which is fitting an uncalibrated constant.

    Attributes:
        name: What the term is.
        uncertainty_class: Which kind of not-knowing it would be.
        reason: Why there is no number, in the words a report can print.
        basis: What would be refined, for numerical terms.
    """

    name: str
    uncertainty_class: UncertaintyClass
    reason: str
    basis: NumericalBasis | None = None

    def __post_init__(self) -> None:
        """Validate the term.

        Raises:
            ValueError: If the reason is blank, or the basis does not match
                the class. Unsized does not mean unclassified.
        """
        if not self.reason.strip():
            raise ValueError(
                f"unquantified term {self.name!r} needs a reason; an unsized "
                "term with no explanation is indistinguishable from an omission"
            )
        _require_basis(self.uncertainty_class, self.basis)


@dataclass(frozen=True, slots=True)
class ClassSubtotal:
    """One class's combined reach, with the rule that combined it.

    Attributes:
        uncertainty_class: The class.
        lower_offset: Combined reach below the centre.
        upper_offset: Combined reach above it.
        rule: How the class's terms were combined, for the report.
        n_terms: How many terms went in.
    """

    uncertainty_class: UncertaintyClass
    lower_offset: float
    upper_offset: float
    rule: str
    n_terms: int

    @property
    def width(self) -> float:
        """Total reach of the class."""
        return self.lower_offset + self.upper_offset


@dataclass(frozen=True, slots=True)
class DominantTerm:
    """The widest single term in a budget, and how much of it that is.

    Attributes:
        term: The term itself.
        share: Its width over the combined width, in ``[0, 1]``.
        swamps: Whether the share clears :data:`DOMINANCE_SHARE`.
    """

    term: UncertaintyTerm
    share: float
    swamps: bool

    @property
    def term_name(self) -> str:
        """Name of the dominant uncertainty term."""
        return self.term.name

    @property
    def uncertainty_class_value(self) -> str:
        """Class value string of the dominant uncertainty term."""
        return self.term.uncertainty_class_value

    @property
    def source(self) -> str:
        """Provenance source string of the dominant uncertainty term."""
        return self.term.source


_QUADRATURE_CLASSES = frozenset({UncertaintyClass.SAMPLING})

_COMBINATION_RULES = {
    UncertaintyClass.NUMERICAL: (
        "simple addition (V&V 20: u_h, u_it and u_ro are correlated faces of "
        "one discrete solve)"
    ),
    UncertaintyClass.MODEL_FORM: (
        "simple addition (two uncalibrated models have no independence to "
        "root-sum-square)"
    ),
    UncertaintyClass.SAMPLING: (
        "quadrature (replicate spread is the one class here with an "
        "independence claim behind it)"
    ),
}


def _combine(offsets: list[float], uncertainty_class: UncertaintyClass) -> float:
    """Combine one side of one class's terms.

    Args:
        offsets: The per-term offsets on this side.
        uncertainty_class: Which class, which fixes the rule.

    Returns:
        The combined offset.
    """
    if uncertainty_class in _QUADRATURE_CLASSES:
        return math.sqrt(math.fsum(offset**2 for offset in offsets))
    return math.fsum(offsets)


@dataclass(frozen=True)
class UncertaintyBudget:
    """Everything known about how wide one quantity's band is, and why.

    Attributes:
        quantity: What the budget is about, in a report.
        central: The quantity's central value, in its own unit.
        terms: The sized contributions.
        unquantified: The contributions nobody has a number for.
    """

    quantity: str
    central: float
    terms: tuple[UncertaintyTerm, ...] = ()
    unquantified: tuple[UnquantifiedTerm, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the central value.

        Raises:
            ValueError: If the central value is not finite. A budget about NaN
                reaches a ranking and orders designs by accident.
        """
        if not math.isfinite(float(self.central)):
            raise ValueError(
                f"the central value of {self.quantity!r} must be finite, got "
                f"{self.central!r}"
            )

    def by_class(self) -> dict[UncertaintyClass, ClassSubtotal]:
        """Combine the terms within each class, keeping the classes apart.

        Returns:
            One subtotal per class that has at least one term, each naming the
            rule that combined it.
        """
        subtotals: dict[UncertaintyClass, ClassSubtotal] = {}
        for uncertainty_class in UncertaintyClass:
            members = [
                term
                for term in self.terms
                if term.uncertainty_class is uncertainty_class
            ]
            if not members:
                continue
            subtotals[uncertainty_class] = ClassSubtotal(
                uncertainty_class=uncertainty_class,
                lower_offset=_combine(
                    [term.lower_offset for term in members], uncertainty_class
                ),
                upper_offset=_combine(
                    [term.upper_offset for term in members], uncertainty_class
                ),
                rule=_COMBINATION_RULES[uncertainty_class],
                n_terms=len(members),
            )
        return subtotals

    def band(self) -> ConsistencyBand:
        """The quantity's band: the class subtotals added about the centre.

        Additive across classes, deliberately. Quadrature would need the
        classes to be independent draws from distributions, and a model-form
        interval is not a distribution at all -- so the conservative
        combination is the only one that can be defended, and it is named here
        rather than buried in a caller.

        When :attr:`band_is_lower_bound` is true this width is a floor and not
        a total; every statement this class produces says so.

        Returns:
            The consistency band.
        """
        subtotals = self.by_class().values()
        lower = math.fsum(subtotal.lower_offset for subtotal in subtotals)
        upper = math.fsum(subtotal.upper_offset for subtotal in subtotals)
        return ConsistencyBand(
            lower=self.central - lower,
            central=self.central,
            upper=self.central + upper,
        )

    @property
    def combined_width(self) -> float:
        """Total reach of the band, both sides together."""
        return self.band().width

    @property
    def band_is_lower_bound(self) -> bool:
        """Whether a known contribution was left out for want of a number."""
        return bool(self.unquantified)

    def dominant(self) -> DominantTerm | None:
        """The single widest term, and how much of the band it accounts for.

        Returns:
            The dominant term, or ``None`` when the budget has no sized terms.
        """
        if not self.terms:
            return None
        widest = max(self.terms, key=lambda term: term.width)
        total = self.combined_width
        share = 1.0 if total == 0.0 else widest.width / total
        return DominantTerm(term=widest, share=share, swamps=share >= DOMINANCE_SHARE)

    def as_v20_numerical(
        self, *, space_time: NumericalBasis | None = None
    ) -> NumericalUncertainty:
        """Map the **numerical** terms onto V&V 20's ``u_num`` components.

        Model-form and sampling terms are dropped by construction: ``u_num``
        is the uncertainty of the arithmetic, and a divot-mass interval that a
        finer grid cannot narrow has no place in it.

        Args:
            space_time: Where to put terms whose basis is
                :attr:`NumericalBasis.SPACE_TIME`. Omitted, such a term is
                **refused**, because
                ``bunkershot3d.solvers.mpm.verification.column_grid_convergence``
                holds the Courant number fixed and its band is therefore
                neither spatial nor temporal -- reporting it as ``u_h`` says a
                combined space-and-time error is purely spatial, and would
                double-count against any temporal term added beside it
                (ADR-0033). Pass :attr:`NumericalBasis.SPATIAL` or
                :attr:`NumericalBasis.TEMPORAL` to fold it in deliberately.

        Returns:
            The V&V 20 numerical uncertainty.

        Raises:
            VandVError: If a space-time term is present and ``space_time`` was
                not given, or if it names a basis that is itself space-time.
        """
        if space_time is NumericalBasis.SPACE_TIME:
            raise VandVError(
                "space_time must name the component to fold into, not "
                "'space-time' again"
            )
        buckets = {NumericalBasis.SPATIAL: 0.0, NumericalBasis.TEMPORAL: 0.0}
        iterative = 0.0
        round_off = 0.0
        for term in self.terms:
            if term.uncertainty_class is not UncertaintyClass.NUMERICAL:
                continue
            basis = term.basis
            if basis is NumericalBasis.SPACE_TIME:
                if space_time is None:
                    raise VandVError(_SPACE_TIME_REFUSAL.format(name=term.name))
                basis = space_time
            if basis is NumericalBasis.ITERATIVE:
                iterative += term.half_width
            elif basis is NumericalBasis.ROUND_OFF:
                round_off += term.half_width
            elif basis is not None:
                buckets[basis] += term.half_width
        return NumericalUncertainty(
            u_h=buckets[NumericalBasis.SPATIAL] + buckets[NumericalBasis.TEMPORAL],
            u_it=iterative,
            u_ro=round_off,
        )

    def statement(self, unit: str = "") -> str:
        """The budget as report lines: the split, the leader, the gaps.

        Args:
            unit: Unit suffix for the numbers.

        Returns:
            A multi-line statement. Never the width alone.
        """
        band = self.band()
        lines = [f"{self.quantity}: {band.statement(unit=unit)}"]
        for subtotal in self.by_class().values():
            lines.append(
                f"  {subtotal.uncertainty_class.value}: "
                f"-{subtotal.lower_offset:.4g}/+{subtotal.upper_offset:.4g}"
                f"{' ' + unit if unit else ''} over {subtotal.n_terms} term(s), "
                f"combined by {subtotal.rule}"
            )
        dominant = self.dominant()
        if dominant is not None:
            verb = "swamps the budget at" if dominant.swamps else "leads at"
            lines.append(
                f"  dominant term: {dominant.term_name} "
                f"({dominant.uncertainty_class_value}) {verb} "
                f"{dominant.share:.0%} of the width -- {dominant.source}"
            )
        for term in self.unquantified:
            lines.append(
                f"  UNQUANTIFIED ({term.uncertainty_class.value}): "
                f"{term.name} -- {term.reason}"
            )
        if self.band_is_lower_bound:
            lines.append(
                "  this band is a LOWER BOUND on the spread, not a total: "
                f"{len(self.unquantified)} known contribution(s) have no "
                "number and are excluded from the arithmetic above"
            )
        return "\n".join(lines)


_SPACE_TIME_REFUSAL = (
    "term {name!r} carries a SPACE-TIME numerical basis and cannot be mapped "
    "onto V&V 20 without being told which component it belongs to. The "
    "shipped column study holds the Courant number fixed, so its step shrinks "
    "with its cell size and the band it returns is neither a spatial u_h nor "
    "a temporal one; reporting it as u_h would claim a combined "
    "space-and-time discretisation error is purely spatial, and would "
    "double-count against any temporal term beside it (ADR-0033). Pass "
    "space_time=NumericalBasis.SPATIAL or TEMPORAL to fold it in deliberately"
)
"""Message of the refusal that keeps a space-time band out of ``u_h``."""
