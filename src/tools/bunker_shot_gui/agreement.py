"""The vocabulary of agreement between two fidelity tiers (issue #8713).

What is being agreed *about*, in what unit, what each tier means by it,
what a ratio of the two is worth, and -- the part the issue asks for by
name -- what agreement between them does and does not license.

Separate from :mod:`~src.tools.bunker_shot_gui.crosstier`, which owns the
probes and the comparison, because these are the terms any pair of tiers
would be compared in. The F1 tier is the first to be put beside F0; ADR-0032
specifies four.

Nothing here runs a solver, and nothing here draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from bunkershot3d.solvers.envelope import MAX_VALIDATED_SPEED_M_S
from bunkershot3d.vandv.credibility import (
    CREDIBILITY_ASSESSMENT,
    CredibilityFactor,
    envelope_exceedance,
)

__all__ = [
    "DECLARED_AGREEMENT_BAND",
    "AgreementClass",
    "ComparedQuantity",
    "DivergenceSpan",
    "QuantityAgreement",
    "licence_statement",
]

DECLARED_AGREEMENT_BAND = 0.25
"""Half-width of the agreement band, on ``|ln(F1 / F0)|``.

**Declared, not derived.** No measurement exists that could set it: issue
#8616 established there is no published data for any of these quantities
out of bunker sand, so there is no model error to calibrate a tolerance
against. It is a reporting threshold -- roughly "within 28 %, either way"
-- chosen so that the ratios ADR-0033 measured (1.49 to 2.68) read as the
divergence they are rather than as noise, and it travels with every
:class:`QuantityAgreement` so a reader can see what was applied.

The band is on the *logarithm* of the ratio because a factor of two too
large and a factor of two too small are the same disagreement, and a band
on the plain ratio would call one of them worse than the other."""

_STAMP_MAX_CHARS = 240


class ComparedQuantity(StrEnum):
    """A quantity both tiers produce -- and what each of them means by it.

    The note is not decoration. Two of these five are the same word for two
    different measurements, and drawing them on one axis without saying so
    would be the most misleading thing this view could do.
    """

    WRENCH = "wrench"
    SOLE_DEPTH = "sole_depth"
    DIVOT_SECTION = "divot_section"
    DIVOT_MASS = "divot_mass"
    SPEED_LOST = "speed_lost"

    @property
    def label(self) -> str:
        """A short heading for this quantity."""
        return _QUANTITY_LABEL[self]

    @property
    def unit(self) -> str:
        """The unit it is reported in. Stated, never assumed."""
        return _QUANTITY_UNIT[self]

    @property
    def display_scale(self) -> float:
        """Multiplier from SI to the unit in :attr:`unit`."""
        return _QUANTITY_SCALE[self]

    @property
    def note(self) -> str:
        """What each tier means by this quantity, where they differ."""
        return _QUANTITY_NOTE[self]


_QUANTITY_LABEL: dict[ComparedQuantity, str] = {
    ComparedQuantity.WRENCH: "Sand force on the head",
    ComparedQuantity.SOLE_DEPTH: "Sole depth",
    ComparedQuantity.DIVOT_SECTION: "Divot section",
    ComparedQuantity.DIVOT_MASS: "Divot mass",
    ComparedQuantity.SPEED_LOST: "Speed lost",
}

_QUANTITY_UNIT: dict[ComparedQuantity, str] = {
    ComparedQuantity.WRENCH: "N",
    ComparedQuantity.SOLE_DEPTH: "mm",
    ComparedQuantity.DIVOT_SECTION: "cm^2",
    ComparedQuantity.DIVOT_MASS: "g",
    ComparedQuantity.SPEED_LOST: "m/s",
}

_QUANTITY_SCALE: dict[ComparedQuantity, float] = {
    ComparedQuantity.WRENCH: 1.0,
    ComparedQuantity.SOLE_DEPTH: 1.0e3,
    ComparedQuantity.DIVOT_SECTION: 1.0e4,
    ComparedQuantity.DIVOT_MASS: 1.0e3,
    ComparedQuantity.SPEED_LOST: 1.0,
}

_QUANTITY_NOTE: dict[ComparedQuantity, str] = {
    ComparedQuantity.WRENCH: (
        "resultant magnitude. F0 integrates a fitted traction over the head; "
        "F1 reads an exact momentum ledger off the grid nodes the section "
        "projected, per unit width, so its magnitude is conditional on the "
        "declared effective width while its direction is not"
    ),
    ComparedQuantity.SOLE_DEPTH: (
        "the deepest submerged element of the shared query -- geometry, not "
        "physics, so this is a control row: a disagreement here would mean the "
        "pose did not reach one of the tiers unchanged. It is deliberately not "
        "either tier's own SolverResult.max_depth_m, because F0 reports its "
        "deepest *engaged* element there while F1 reports the deepest "
        "submerged one, and on a lofted head those differ by an order of "
        "magnitude"
    ),
    ComparedQuantity.DIVOT_SECTION: (
        "the same word for two different measurements. F0 transports no sand, "
        "so its divot is the swept lower envelope of the head itself -- where "
        "the head has been. F1's is where the sand ended up: the depression "
        "left in the free surface after the pass"
    ),
    ComparedQuantity.DIVOT_MASS: (
        "each tier's own divot section carried to a mass at one declared "
        "out-of-plane width and the bed's bulk density, so the comparison is "
        "not confounded by two different width assumptions; it inherits the "
        "envelope-versus-sand difference of the section it comes from"
    ),
    ComparedQuantity.SPEED_LOST: (
        "F0's alone. F1's section is driven kinematically at constant "
        "velocity, so it loses no speed of its own; the F1 figure is what its "
        "force would have taken off the same head over the same window, which "
        "is one-way coupled -- F1 was never asked what the slower head would "
        "have done"
    ),
}


class AgreementClass(StrEnum):
    """The verdict on one ratio, against the declared band."""

    CONSISTENT = "consistent"
    """Inside the band. Which licenses nothing; see :func:`licence_statement`."""

    DIVERGENT = "divergent"
    """Outside it. At least one tier is wrong, and that is worth knowing."""

    INCOMPARABLE = "incomparable"
    """One tier or both produced nothing, so there is no ratio to judge.

    Distinct from agreement on purpose: two zeroes are not a match, they
    are an unanswered question, and classing them as consistent would let
    a quantity nobody computed read as a quantity both tiers confirmed."""


@dataclass(frozen=True, slots=True)
class QuantityAgreement:
    """One quantity, both tiers' answers, and the verdict on their ratio.

    Attributes:
        quantity: Which quantity.
        f0_value: F0's answer, in SI.
        f1_value: F1's answer, in SI.
        band: Half-width of the agreement band on ``|ln ratio|``.
    """

    quantity: ComparedQuantity
    f0_value: float
    f1_value: float
    band: float = DECLARED_AGREEMENT_BAND

    def __post_init__(self) -> None:
        """Validate the pair.

        Raises:
            ValueError: If either value is negative or not finite, or if the
                band is not positive and finite. These are ``raise`` and not
                ``assert``: ``python -O`` strips assertions, and a zero band
                would silently class every finite discretisation error as a
                divergence.
        """
        for name, value in (("f0_value", self.f0_value), ("f1_value", self.f1_value)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative -- these are "
                    f"magnitudes -- got {value!r}"
                )
        if not math.isfinite(self.band) or self.band <= 0.0:
            raise ValueError(
                f"the agreement band must be positive and finite, got {self.band!r}"
            )

    @property
    def comparable(self) -> bool:
        """Whether both tiers produced something to form a ratio from."""
        return self.f0_value > 0.0 and self.f1_value > 0.0

    @property
    def ratio(self) -> float:
        """``F1 / F0``; ``nan`` when there is no ratio to take."""
        if not self.comparable:
            return math.nan
        return self.f1_value / self.f0_value

    @property
    def log_ratio(self) -> float:
        """``|ln(F1 / F0)|``, the symmetric measure of disagreement."""
        if not self.comparable:
            return math.nan
        return abs(math.log(self.ratio))

    @property
    def relative_difference(self) -> float:
        """``(F1 - F0) / F0``; ``nan`` when there is no ratio to take."""
        if not self.comparable:
            return math.nan
        return (self.f1_value - self.f0_value) / self.f0_value

    @property
    def agreement(self) -> AgreementClass:
        """The verdict on the ratio."""
        if not self.comparable:
            return AgreementClass.INCOMPARABLE
        return (
            AgreementClass.CONSISTENT
            if self.log_ratio <= self.band
            else AgreementClass.DIVERGENT
        )

    @property
    def diverged(self) -> bool:
        """Whether the two tiers left the declared band."""
        return self.agreement is AgreementClass.DIVERGENT

    def display(self, value: float) -> float:
        """Convert one SI value into this quantity's reporting unit."""
        return value * self.quantity.display_scale

    def summary(self) -> str:
        """A line fit for the agreement table drawn inside the view."""
        unit = self.quantity.unit
        head = (
            f"{self.quantity.label}: F0 {self.display(self.f0_value):.4g} {unit}, "
            f"F1 {self.display(self.f1_value):.4g} {unit}"
        )
        if not self.comparable:
            return f"{head} -- incomparable, at least one tier produced nothing"
        return (
            f"{head}, ratio {self.ratio:.3g}x "
            f"({self.relative_difference:+.0%}), "
            f"{self.agreement.value} against a declared band of "
            f"{self.band:g} on |ln ratio|"
        )


@dataclass(frozen=True, slots=True)
class DivergenceSpan:
    """A stretch of the record over which one quantity left the band.

    Attributes:
        quantity: Which quantity diverged.
        start_s: When the stretch opens [s].
        end_s: When it closes [s].
        worst_ratio: The largest departure from 1.0 inside it, as a ratio.
        n_probes: How many probes it covers.
    """

    quantity: ComparedQuantity
    start_s: float
    end_s: float
    worst_ratio: float
    n_probes: int

    @property
    def duration_s(self) -> float:
        """How long the stretch lasts [s]."""
        return self.end_s - self.start_s

    @property
    def label(self) -> str:
        """One line naming the quantity, the stretch and the worst ratio."""
        return (
            f"{self.quantity.label} diverges "
            f"{self.start_s * 1e3:.2f}-{self.end_s * 1e3:.2f} ms, "
            f"worst {self.worst_ratio:.3g}x "
            f"({self.n_probes} probe{'' if self.n_probes == 1 else 's'})"
        )


def licence_statement(
    *,
    speed_m_s: float,
    effective_width_m: float | None = None,
    feature_length_m: float = 0.100,
) -> str:
    """State what agreement between the two tiers does and does not license.

    Computed rather than quoted. The validation level comes from
    :data:`~bunkershot3d.vandv.credibility.CREDIBILITY_ASSESSMENT`, the
    exceedance from
    :func:`~bunkershot3d.vandv.credibility.envelope_exceedance`, and the
    speed ceiling from
    :data:`~bunkershot3d.solvers.envelope.MAX_VALIDATED_SPEED_M_S`, so the
    sentence cannot drift away from the code it is describing.

    Args:
        speed_m_s: The delivery speed the comparison was run at.
        effective_width_m: The declared out-of-plane width F1's magnitudes
            rest on; named when there is one, since no F1 magnitude may be
            reproduced without it.
        feature_length_m: Scale the Froude number is formed on.

    Returns:
        The statement, several sentences, meant to be drawn inside the
        view rather than captioned beneath it.
    """
    validation = next(
        item
        for item in CREDIBILITY_ASSESSMENT
        if item.factor is CredibilityFactor.VALIDATION
    )
    exceedance = envelope_exceedance(
        speed_m_s=speed_m_s, feature_length_m=feature_length_m
    )
    width = (
        ""
        if effective_width_m is None
        else (
            f" F1's magnitudes are per unit width, raised to forces at a "
            f"declared effective width of {effective_width_m * 1e3:g} mm; that "
            "assumption is not a result and no F1 magnitude may be reproduced "
            "without it."
        )
    )
    return (
        "What this comparison licenses: nothing about sand. It is a "
        "consistency check between two uncalibrated models, and agreement "
        "between two uncalibrated models is not validation. Neither tier's "
        "NASA-STD-7009B validation level moves because of anything on this "
        f"page: validation stands at {validation.achieved_level} of "
        f"{validation.threshold_level + 1} against a threshold of "
        f"{validation.threshold_level}, because neither tier is being "
        "compared to a measurement. What the comparison can do is falsify -- "
        "a disagreement beyond the declared band means at least one of the "
        "two is wrong, and that is the whole of its value.\n"
        f"The delivery point is {exceedance.speed_exceedance:.0f}x the "
        f"fastest intrusion in the published corpus "
        f"({MAX_VALIDATED_SPEED_M_S:g} m/s) and "
        f"{exceedance.froude_exceedance:.0f}x 3D-RFT's stated Froude limit at "
        f"the {feature_length_m * 1e3:g} mm clubhead scale, so the record is "
        "outside the published corpus from its first sample and never "
        f"returns.{width}"
    )
