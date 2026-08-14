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
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ..metrics.forgiveness import SWEEP_RANGES
from ..solvers.envelope import (
    GRAVITY_M_S2,
    MAX_VALIDATED_SPEED_M_S,
    RFT_FROUDE_LIMIT,
)
from .exceptions import VerificationError
from .reference_data import WIVOU_2016, DomainOverlap, domain_overlap

__all__ = [
    "CREDIBILITY_ASSESSMENT",
    "DESIGN_FEATURE_LENGTH_M",
    "DESIGN_SPEED_M_S",
    "MAX_CREDIBILITY_LEVEL",
    "CredibilityFactor",
    "EnvelopeExceedance",
    "FactorAssessment",
    "credibility_table_markdown",
    "domain_of_applicability",
    "domain_table_markdown",
    "envelope_exceedance",
]

MAX_CREDIBILITY_LEVEL = 4
"""Top of the NASA-STD-7009B 0-4 scale."""

DESIGN_SPEED_M_S = 25.0
"""Greenside delivery speed the tool is built for (the 20-27 m/s band)."""

DESIGN_FEATURE_LENGTH_M = 0.100
"""Clubhead scale used for the headline Froude number.

This is the *most flattering* of the three scales the envelope judges:
the 30 mm sole width and the 5 mm leading edge are worse still.  It is
used for the headline number precisely so the headline cannot be accused
of being picked to look bad."""


class CredibilityFactor(StrEnum):
    """The eight NASA-STD-7009B credibility factors."""

    VERIFICATION = "verification"
    VALIDATION = "validation"
    INPUT_PEDIGREE = "input_pedigree"
    RESULTS_UNCERTAINTY = "results_uncertainty"
    RESULTS_ROBUSTNESS = "results_robustness"
    USE_HISTORY = "use_history"
    MS_MANAGEMENT = "ms_management"
    PEOPLE_QUALIFICATIONS = "people_qualifications"

    @property
    def title(self) -> str:
        """Human-readable factor name, as NASA-STD-7009B writes it."""
        if self is CredibilityFactor.MS_MANAGEMENT:
            return "M&S Management"
        return self.value.replace("_", " ").title()


@dataclass(frozen=True)
class FactorAssessment:
    """One factor's achieved level, its threshold, and the gap between them.

    Attributes:
        factor: Which factor.
        achieved_level: 0-4, or ``None`` when the factor cannot honestly
            be self-assessed.
        threshold_level: The level the intended use -- choosing between
            two wedge sole geometries and believing the answer -- demands.
        evidence: What the achieved level rests on.
        gap_statement: What is missing, stated as work rather than as a
            euphemism.
    """

    factor: CredibilityFactor
    achieved_level: int | None
    threshold_level: int
    evidence: str
    gap_statement: str

    def __post_init__(self) -> None:
        """Validate the levels.

        Raises:
            VerificationError: If a level falls outside 0-4, or the
                evidence or gap statement is empty.
        """
        for name in ("achieved_level", "threshold_level"):
            level = getattr(self, name)
            if level is None:
                continue
            if not isinstance(level, int) or not 0 <= level <= MAX_CREDIBILITY_LEVEL:
                raise VerificationError(
                    f"{self.factor.value}.{name} must be an integer in 0-"
                    f"{MAX_CREDIBILITY_LEVEL}, got {level!r}"
                )
        for name in ("evidence", "gap_statement"):
            if not getattr(self, name).strip():
                raise VerificationError(
                    f"{self.factor.value} has an empty {name}; an unexplained "
                    "credibility level is a number without evidence"
                )

    @property
    def is_assessed(self) -> bool:
        """False when the factor was deliberately not self-scored."""
        return self.achieved_level is not None

    @property
    def gap(self) -> int | None:
        """``threshold - achieved``, never negative; ``None`` if unassessed."""
        if self.achieved_level is None:
            return None
        return max(self.threshold_level - self.achieved_level, 0)

    @property
    def meets_threshold(self) -> bool:
        """True only when the factor is assessed and clears its threshold."""
        return self.achieved_level is not None and self.achieved_level >= (
            self.threshold_level
        )

    def level_text(self) -> str:
        """The achieved level rendered for a table cell."""
        if self.achieved_level is None:
            return "not assessed"
        return f"{self.achieved_level} / {MAX_CREDIBILITY_LEVEL}"

    def gap_text(self) -> str:
        """The gap rendered for a table cell."""
        if self.gap is None:
            return "n/a"
        return "met" if self.gap == 0 else f"{self.gap} level(s) short"


CREDIBILITY_ASSESSMENT: tuple[FactorAssessment, ...] = (
    FactorAssessment(
        factor=CredibilityFactor.VERIFICATION,
        achieved_level=2,
        threshold_level=3,
        evidence=(
            "Formal code verification of the F0 tier: conservation residuals "
            "split into round-off and truncation classes, an angular-momentum "
            "check against a naive per-element oracle, order of accuracy "
            "against a closed-form cylinder integral, and analytic flat-plate "
            "and zero-speed limits. Solution verification is implemented as a "
            "Celik GCI with Richardson extrapolation."
        ),
        gap_statement=(
            "No method of manufactured solutions for the coupled shot, and no "
            "verification at all of the F1, F2 or F3 tiers. The surface "
            "refinement study also runs into the package's own envelope: I_G "
            "grows as the mesh is refined, so a mesh fine enough to converge "
            "the quadrature is further outside RFT's superposition argument."
        ),
    ),
    FactorAssessment(
        factor=CredibilityFactor.VALIDATION,
        achieved_level=0,
        threshold_level=3,
        evidence=(
            "None. The one comparison that can be formed against a published "
            "measurement -- the material-scaling prediction of the vertical "
            "plate response against the Quikrete analogue's 2.02 N/cm^3 -- is "
            "noise-limited under V&V 20, so it carries no information about "
            "model error."
        ),
        gap_statement=(
            "No published data exists for ball launch angle, speed or spin "
            "from a splash shot, for clubhead deceleration in sand, for the "
            "energy split, or for ejecta mass. This is a gap in the field, "
            "not a search failure, so it cannot be closed by reading more. "
            "Closing it needs either the Wivou carry correlations compared "
            "against a computed model correlation, or an instrumented "
            "experiment: plate penetration at three plate areas, a 6x6 cm "
            "direct shear box, and one drag test at 20-27 m/s to fit lambda."
        ),
    ),
    FactorAssessment(
        factor=CredibilityFactor.INPUT_PEDIGREE,
        achieved_level=2,
        threshold_level=3,
        evidence=(
            "Every fitted constant is traced to a published analogue: the "
            "3D-RFT polynomial to a generic frictional-plastic medium, the "
            "friction angle and packing fraction to Quikrete medium sand, and "
            "lambda to plate-drag and wheel experiments. One fully "
            "characterised commercial bunker sand (Covia Signature 500, ASTM "
            "F1632 Method B and F1815) seeds the sand presets, and every entry "
            "carries a ProvenanceBasis."
        ),
        gap_statement=(
            "Nothing is measured on the sand actually being modelled, and the "
            "one characterised sand is a single lab report on a single "
            "commercial product, not a population. lambda and delta_h have no "
            "wedge value at all."
        ),
    ),
    FactorAssessment(
        factor=CredibilityFactor.RESULTS_UNCERTAINTY,
        achieved_level=2,
        threshold_level=3,
        evidence=(
            "Discretisation uncertainty is estimated by GCI and converted to a "
            "V&V 20 u_h; u_num is formed by simple addition of u_h, u_it and "
            "u_ro as the standard requires, and u_val by quadrature."
        ),
        gap_statement=(
            "No input uncertainty is propagated through a shot, and no "
            "model-form uncertainty is quantified anywhere -- which is the "
            "direct consequence of validation being at level 0. The reported "
            "u_num covers the numerics only and must not be read as an error "
            "bar on the physics."
        ),
    ),
    FactorAssessment(
        factor=CredibilityFactor.RESULTS_ROBUSTNESS,
        achieved_level=1,
        threshold_level=2,
        evidence=(
            "Metamorphic relations (translation, rotation, reflection, "
            "permutation, scaling, monotonicity) cover the solver, and the "
            "study package provides variance-based sensitivity over the "
            "declared sweep ranges."
        ),
        gap_statement=(
            "No sensitivity study over the constants that actually dominate "
            "the answer -- lambda across its published 1.0-2.8 spread and the "
            "delta_h saturation fraction -- and no independent review of the "
            "F0 tier by anyone who did not write it."
        ),
    ),
    FactorAssessment(
        factor=CredibilityFactor.USE_HISTORY,
        achieved_level=0,
        threshold_level=2,
        evidence=(
            "None. The F0 solver has never been used to make a design "
            "decision, and no predecessor of it has either."
        ),
        gap_statement=(
            "Use history accrues only by use. Until a design produced by this "
            "tool is built and measured, this factor cannot move."
        ),
    ),
    FactorAssessment(
        factor=CredibilityFactor.MS_MANAGEMENT,
        achieved_level=3,
        threshold_level=3,
        evidence=(
            "ADR-0032 records the architecture and its rejected alternatives; "
            "every run emits a manifest with config hash, physics hash, RNG "
            "seed entropy, library versions, git SHA, fidelity tier and "
            "validity verdict; and CI enforces lint, format, file-size, "
            "architecture and marker gates on every change."
        ),
        gap_statement=(
            "Threshold met. The remaining gap to level 4 is a formal "
            "configuration-management process with a defined release and "
            "approval path, which this repository does not have."
        ),
    ),
    FactorAssessment(
        factor=CredibilityFactor.PEOPLE_QUALIFICATIONS,
        achieved_level=None,
        threshold_level=2,
        evidence=(
            "Deliberately not self-assessed. A team scoring its own competence "
            "is not evidence, and a number here would dilute the seven factors "
            "that are backed by artefacts."
        ),
        gap_statement=(
            "Assess externally, or leave blank. Do not fill it in to make the "
            "table look complete."
        ),
    ),
)
"""The credibility assessment, achieved level and gap per factor."""


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
        f"| {item.factor.title} | {item.level_text()} | "
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
