"""The credibility statement, in NASA-STD-7009B's framing (issue #8616).

NASA-STD-7009B (2024-03-05; **7009A is superseded**) scores a model on
eight factors, each on a 0-4 scale, and requires the *achieved* level to
be reported next to the *threshold* the intended use demands.  Reporting
the achieved level alone is how a model that is nowhere near fit for
purpose reads as respectable; the gap is the information.

What this module asserts, and it is not flattering
--------------------------------------------------

* We run the F0 solver about **63x outside 3D-RFT's own stated Froude
  limit** of 0.4, and about **17x beyond the fastest intrusion anywhere
  in the published RFT/DRFT corpus** (1.44 m/s).  Those two numbers are
  computed here from the solver's own constants, not quoted, so they
  cannot drift away from the code.
* **``delta_h`` and ``lambda`` are uncalibrated for a wedge.**  No
  published wedge value exists for either.  ``lambda`` carries roughly
  90% of the load at greenside delivery speed and its published spread
  across motion types is 1.0 to 2.8, so the single most influential
  constant in the model is known to within a factor of nearly three.
* **Validation is at level 0.**  Not "limited", not "in progress": the
  only comparison that can be formed against a measurement is
  noise-limited, and a noise-limited comparison carries no information
  about model error.

The factor levels here are a self-assessment of the model, not of the
people who built it.  :attr:`CredibilityFactor.PEOPLE_QUALIFICATIONS` is
therefore recorded as **not assessed** rather than given a number, since
a self-scored competence rating is not evidence.

Where the numbers come from
---------------------------

Nowhere in this module.  :data:`CREDIBILITY_ASSESSMENT` is *derived* from
:data:`~bunkershot3d.vandv.roadmap.VALIDATION_LEDGER` and the shipped
measurement register, which is empty.  The ledger states, per factor, the
level it is held at, what holds it there, and -- for the three factors an
experiment could move -- the minimum measurement that would lift it one
level, with its conditions, its instrument class and its acceptance
criterion.  Keeping the assessment and the roadmap in one structure is not
tidiness: a published table saying 0 next to a plan written as though it
were 2 is the failure mode, and it is silent by nature.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..metrics.forgiveness import SWEEP_RANGES
from ..solvers.envelope import (
    GRAVITY_M_S2,
    MAX_VALIDATED_SPEED_M_S,
    RFT_FROUDE_LIMIT,
)
from .exceptions import VerificationError
from .ledger import (
    MAX_CREDIBILITY_LEVEL,
    CredibilityFactor,
    FactorAssessment,
)
from .measurement import MeasurementRegister
from .measurement_intake import shipped_register
from .reference_data import WIVOU_2016, DomainOverlap, domain_overlap
from .roadmap import VALIDATION_LEDGER

__all__ = [
    "CREDIBILITY_ASSESSMENT",
    "DESIGN_FEATURE_LENGTH_M",
    "DESIGN_SPEED_M_S",
    "MAX_CREDIBILITY_LEVEL",
    "CredibilityFactor",
    "EnvelopeExceedance",
    "FactorAssessment",
    "credibility_assessment",
    "credibility_table_markdown",
    "domain_of_applicability",
    "domain_table_markdown",
    "envelope_exceedance",
]

DESIGN_SPEED_M_S = 25.0
"""Greenside delivery speed the tool is built for (the 20-27 m/s band)."""

DESIGN_FEATURE_LENGTH_M = 0.100
"""Clubhead scale used for the headline Froude number.

This is the *most flattering* of the three scales the envelope judges:
the 30 mm sole width and the 5 mm leading edge are worse still.  It is
used for the headline number precisely so the headline cannot be accused
of being picked to look bad."""


def credibility_assessment(
    register: MeasurementRegister | None = None,
) -> tuple[FactorAssessment, ...]:
    """Derive the eight-factor assessment from the ledger.

    This is the only place a credibility level is produced.  There is no
    hand-maintained table of numbers to fall out of step with the roadmap,
    because there is no second table: :data:`VALIDATION_LEDGER` states the
    level each factor is held at, what holds it there, and which
    measurement would lift it, and the level reported here is that level
    plus whatever ``register`` has actually bought.

    Args:
        register: The measurements on hand.  ``None`` means the shipped
            register, which is empty and must stay empty until something is
            measured.

    Returns:
        One assessment per factor, in ledger order.
    """
    supplied = shipped_register() if register is None else register
    return VALIDATION_LEDGER.assessment(supplied)


CREDIBILITY_ASSESSMENT: tuple[FactorAssessment, ...] = credibility_assessment()
"""The credibility assessment, achieved level and gap per factor.

Derived from :data:`~bunkershot3d.vandv.roadmap.VALIDATION_LEDGER` and the
shipped measurement register.  The register is empty, so this is the level
the ledger holds -- validation at 0 of 4 -- and writing the roadmap did not
change it.  Supplying a measurement through
:mod:`bunkershot3d.vandv.measurement_intake` is the only thing that will."""


@dataclass(frozen=True, slots=True)
class EnvelopeExceedance:
    """How far outside its own envelope the tool is run.

    Attributes:
        speed_m_s: The design speed judged.
        feature_length_m: The scale the Froude number is formed on.
        froude: ``v / sqrt(g L)``.
        froude_limit: 3D-RFT's stated limit.
        froude_exceedance: ``froude / froude_limit``.
        max_validated_speed_m_s: Fastest intrusion in the published corpus.
        speed_exceedance: ``speed / max_validated_speed``.
    """

    speed_m_s: float
    feature_length_m: float
    froude: float
    froude_limit: float
    froude_exceedance: float
    max_validated_speed_m_s: float
    speed_exceedance: float

    def describe(self) -> str:
        """The headline sentence of the credibility statement."""
        return (
            f"At {self.speed_m_s:g} m/s on the {self.feature_length_m * 1e3:g} mm "
            f"clubhead scale, Fr = {self.froude:.1f} against 3D-RFT's stated "
            f"limit of {self.froude_limit:g}: about "
            f"{self.froude_exceedance:.0f}x outside the stated envelope, and "
            f"about {self.speed_exceedance:.0f}x beyond the fastest intrusion "
            f"({self.max_validated_speed_m_s:g} m/s) anywhere in the published "
            "RFT/DRFT validation corpus."
        )


def envelope_exceedance(
    *,
    speed_m_s: float = DESIGN_SPEED_M_S,
    feature_length_m: float = DESIGN_FEATURE_LENGTH_M,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> EnvelopeExceedance:
    """Compute how far outside the published envelope the design point sits.

    Computed from the solver's own constants rather than quoted, so the
    credibility statement cannot drift away from the code.

    Args:
        speed_m_s: Design delivery speed.
        feature_length_m: Feature scale for the Froude number.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The exceedance factors.

    Raises:
        VerificationError: If a length or speed is not positive.
    """
    for name, value in (
        ("speed_m_s", speed_m_s),
        ("feature_length_m", feature_length_m),
        ("gravity_m_s2", gravity_m_s2),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise VerificationError(f"{name} must be positive, got {value!r}")
    froude = speed_m_s / math.sqrt(gravity_m_s2 * feature_length_m)
    return EnvelopeExceedance(
        speed_m_s=float(speed_m_s),
        feature_length_m=float(feature_length_m),
        froude=froude,
        froude_limit=RFT_FROUDE_LIMIT,
        froude_exceedance=froude / RFT_FROUDE_LIMIT,
        max_validated_speed_m_s=MAX_VALIDATED_SPEED_M_S,
        speed_exceedance=speed_m_s / MAX_VALIDATED_SPEED_M_S,
    )


def domain_of_applicability() -> Mapping[str, DomainOverlap]:
    """How much of each declared sweep range published data covers.

    Only two factors have any published bunker measurement at all, and
    neither sweep sits inside it.  The rest of the design space has no
    measured domain of applicability whatsoever, which is why this
    mapping is short.

    Returns:
        Factor name to its overlap with the measured domain.
    """
    overlaps: dict[str, DomainOverlap] = {}
    for factor in WIVOU_2016.correlations:
        sweep = SWEEP_RANGES.get(factor)
        if sweep is None:  # pragma: no cover - registry drift guard
            continue
        overlaps[factor] = domain_overlap(
            (sweep.low, sweep.high), WIVOU_2016.value_range(factor)
        )
    return MappingProxyType(overlaps)


def credibility_table_markdown() -> str:
    """Render the eight-factor assessment as a Markdown table.

    Returns:
        The table, achieved level and gap side by side.
    """
    lines = [
        "| Factor | Achieved | Threshold | Gap |",
        "| ------ | -------- | --------- | --- |",
    ]
    lines.extend(
        f"| {item.factor.label} | {item.level_text()} | "
        f"{item.threshold_level} / {MAX_CREDIBILITY_LEVEL} | {item.gap_text()} |"
        for item in CREDIBILITY_ASSESSMENT
    )
    return "\n".join(lines)


def domain_table_markdown() -> str:
    """Render the domain-of-applicability overlaps as a Markdown table.

    Returns:
        The table, swept range against measured range.
    """
    lines = [
        "| Factor | Swept | Measured (Wivou 2016) | Inside measured domain |",
        "| ------ | ----- | --------------------- | ---------------------- |",
    ]
    for factor, overlap in domain_of_applicability().items():
        lines.append(
            f"| `{factor}` | {overlap.swept[0]:g} to {overlap.swept[1]:g} m | "
            f"{overlap.measured[0]:g} to {overlap.measured[1]:g} m | "
            f"{overlap.covered_fraction:.0%} |"
        )
    return "\n".join(lines)
