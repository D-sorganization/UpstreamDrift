"""The validity envelope (issue #8611, research addendum section 1).

Why this module is the most important one in the package
--------------------------------------------------------

3D-RFT's own stated limit is ``Fr = v / sqrt(g L) < 0.4``, and the entire
RFT/DRFT validation corpus tops out at **1.44 m/s**.  A greenside bunker
shot is delivered at 20-27 m/s.  With ``L = 0.1 m`` that is ``Fr = 25``:
about **60x outside the stated envelope and ~20x beyond any published
validation**.

That does not invalidate the architecture -- DRFT is still the only
per-geometry method cheap enough for a design loop -- but it does make
ADR-0032's validity-envelope requirement *the* feature of this tier
rather than a nicety.  Published RFT coefficients are an initial guess,
not a validated model.  So this module:

* computes the dimensionless groups at every relevant feature scale
  (clubhead, sole width, leading edge) rather than one flattering one;
* refuses outright when the dynamic terms are switched off above
  ``Fr ~ 1``, where a quasi-static solver is wrong by an order of
  magnitude;
* attaches the five documented RFT failure modes (Zhang & Goldman 2014,
  *Phys. Fluids* 26:101308) to **every** verdict, because every one of
  them bites a bunker shot and none of them is modelled.

Reference values reproduced by :func:`dimensionless_groups` (addendum
section 1, ``v = 25 m/s``, ``d = 0.5 mm``):

==========================  =====  ================  =====
Feature scale L             Fr     micro-inertial I  d/L
==========================  =====  ================  =====
100 mm (clubhead)           25.2   0.126             0.005
30 mm (sole width)          46.1   0.768             0.017
5 mm (bounce / edge)        112.9  11.3              0.100
==========================  =====  ================  =====
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .exceptions import OutOfEnvelopeError, SolverInputError

__all__ = [
    "GRAVITY_M_S2",
    "MARGINAL_CONTINUUM_SIZE_RATIO",
    "MAX_ELEMENT_INERTIAL_NUMBER",
    "MAX_VALIDATED_SPEED_M_S",
    "MIN_CONTINUUM_SIZE_RATIO",
    "RFT_FROUDE_LIMIT",
    "RFT_INERTIAL_NUMBER_LIMIT",
    "RFT_QUASI_STATIC_FROUDE_CEILING",
    "Caveat",
    "DimensionlessGroups",
    "EnvelopeContext",
    "EnvelopeStatus",
    "FeatureScale",
    "RefusalPolicy",
    "STANDING_CAVEATS",
    "ValidityVerdict",
    "dimensionless_groups",
    "evaluate_envelope",
    "worst_of",
]

GRAVITY_M_S2 = 9.81
"""Standard gravity used throughout the F0 tier.

Pinned here because the material-scaling table in the research addendum
(``xi_n = rho_c * g * f_hat(mu)``) reproduces only with ``g = 9.81``, not
with 9.80665.
"""

RFT_FROUDE_LIMIT = 0.4
"""3D-RFT's own stated Froude limit (addendum section 1)."""

RFT_QUASI_STATIC_FROUDE_CEILING = 1.0
"""Above this, forces may not be reported unless the dynamic terms are on.

The addendum is explicit: the tool "must refuse to report forces at
Fr > ~1 unless the dynamic terms are active"."""

RFT_INERTIAL_NUMBER_LIMIT = 0.1
"""Micro-inertial number above which grain-rate effects survive in Psi."""

MAX_VALIDATED_SPEED_M_S = 1.44
"""The fastest intrusion anywhere in the published RFT/DRFT corpus."""

MIN_CONTINUUM_SIZE_RATIO = 5.0
"""Minimum feature-size-to-grain-size ratio for a continuum stress fit.

Below this the medium is a handful of grains, not a frictional-plastic
continuum, and superposition -- the entire justification for RFT -- has
nothing to superpose.

The textbook floor is usually quoted nearer 10, but the addendum's own
reference case (a 5 mm leading edge in ``d = 0.5 mm`` sand) sits at
exactly 10, so a limit of 10 would refuse the reference case and the
limit would be describing the rule rather than the physics.  5 is a
genuine backstop; :data:`MARGINAL_CONTINUUM_SIZE_RATIO` reports the
uncomfortable band above it."""

MARGINAL_CONTINUUM_SIZE_RATIO = 20.0
"""Below this many grains across a feature, say so loudly."""

MAX_ELEMENT_INERTIAL_NUMBER = 1.0
"""Askari & Kamrin's ``I_G`` above which the element size survives in Psi.

RFT is *exact* for a frictional-plastic medium because superposition
reproduces ``F = rho_c g L^3 Psi(beta, gamma)``.  It breaks when rate or
size effects let the discretisation length ``lambda`` survive in the
dimensionless groups.  For ``lambda = 2 mm``, ``d = 0.5 mm`` and
``v = 25 m/s``, ``I_G ~ 4.0`` -- not small.  Note the direction of the
trap: refining the surface mesh *raises* ``I_G``."""


class EnvelopeStatus(StrEnum):
    """How much of the answer, if any, may be believed.

    The ordering of the members is deliberate: :func:`worst_of` relies on
    it to combine per-timestep verdicts into a verdict for a whole shot.
    """

    WITHIN = "within"
    """Inside 3D-RFT's stated limits. Nothing in a bunker shot is."""

    EXTRAPOLATED = "extrapolated"
    """Outside the stated limits but inside the published validation set."""

    BEYOND_VALIDATION = "beyond_validation"
    """Past every published measurement. A number, loudly flagged."""

    REFUSED = "refused"
    """No number is reported. The query is unanswerable at this tier."""


_STATUS_ORDER: tuple[EnvelopeStatus, ...] = (
    EnvelopeStatus.WITHIN,
    EnvelopeStatus.EXTRAPOLATED,
    EnvelopeStatus.BEYOND_VALIDATION,
    EnvelopeStatus.REFUSED,
)


class RefusalPolicy(StrEnum):
    """What a solver does when it reaches :attr:`EnvelopeStatus.REFUSED`."""

    STRICT = "strict"
    """Raise :class:`~bunkershot3d.solvers.exceptions.OutOfEnvelopeError`."""

    REPORT = "report"
    """Return the result carrying a ``REFUSED`` verdict, for diagnostics."""


class Caveat(StrEnum):
    """A documented reason the answer is less trustworthy than it looks.

    The first five are the RFT failure modes catalogued by Zhang &
    Goldman (2014); every one of them applies to every bunker shot, so
    they are attached unconditionally by :data:`STANDING_CAVEATS`.
    """

    TRANSIENT_RESPONSE = "transient_response"
    DISTURBED_GROUND = "disturbed_ground"
    INCLINE = "incline"
    SHADOWING = "shadowing"
    SHARP_CORNERS = "sharp_corners"

    BORROWED_COEFFICIENTS = "borrowed_coefficients"
    UNCALIBRATED_STRUCTURAL_CORRECTION = "uncalibrated_structural_correction"
    UPWARD_FACING_LEADING_EDGE = "upward_facing_leading_edge"
    SUPERCRITICAL_FROUDE = "supercritical_froude"
    GRAIN_RATE_EFFECTS = "grain_rate_effects"
    BEYOND_PUBLISHED_SPEED = "beyond_published_speed"
    SHALLOW_INTRUSION = "shallow_intrusion"
    MARGINAL_CONTINUUM = "marginal_continuum"
    ELEMENT_SIZE_EFFECTS = "element_size_effects"


_CAVEAT_TEXT: Mapping[Caveat, str] = MappingProxyType(
    {
        Caveat.TRANSIENT_RESPONSE: (
            "RFT assumes a steady state; DEM needs about 1/5 of a stroke to "
            "reach one, and that transient caused a ~30% over-prediction in "
            "sand-swimming speed. A bunker shot is all transient."
        ),
        Caveat.DISTURBED_GROUND: (
            "Sand does not heal. Drag drops substantially near previously "
            "disturbed ground, and a divot is previously disturbed ground. "
            "Not modelled."
        ),
        Caveat.INCLINE: (
            "A 20 deg tilt drops drag by ~50%, and the authors state it is "
            "unclear how to incorporate inclines into RFT. Bunker faces are "
            "inclined. Not modelled."
        ),
        Caveat.SHADOWING: (
            "There is no wake model, so leading-edge elements sheltered "
            "behind other parts of the body are counted at full strength."
        ),
        Caveat.SHARP_CORNERS: (
            "Reduced accuracy along sharply varying surfaces -- which is "
            "exactly the leading edge and bounce surface being designed."
        ),
        Caveat.BORROWED_COEFFICIENTS: (
            "Every fitted coefficient is borrowed from a published analogue "
            "material; none is measured on golf bunker sand (issue #7999)."
        ),
        Caveat.UNCALIBRATED_STRUCTURAL_CORRECTION: (
            "The DRFT structural correction delta_h has no published "
            "wedge-specific form. The default model is a documented "
            "convention, not a calibration."
        ),
        Caveat.UPWARD_FACING_LEADING_EDGE: (
            "Some leading-edge elements face upward, outside the fitted "
            "domain of the 3D-RFT polynomial. Their depth term is clamped "
            "to the vertical-wall limit; their inertial term is exact."
        ),
        Caveat.SUPERCRITICAL_FROUDE: (
            "Froude number exceeds 3D-RFT's stated limit of 0.4, so the "
            "medium is not in the frictional-plastic regime the fit assumes."
        ),
        Caveat.GRAIN_RATE_EFFECTS: (
            "The micro-inertial number exceeds 0.1, so grain-scale rate "
            "effects survive in the dimensionless response and RFT's "
            "superposition argument no longer holds exactly."
        ),
        Caveat.BEYOND_PUBLISHED_SPEED: (
            "Speed exceeds 1.44 m/s, the fastest intrusion anywhere in the "
            "published RFT/DRFT validation corpus."
        ),
        Caveat.SHALLOW_INTRUSION: (
            "The body is barely submerged. RFT degrades near the free "
            "surface, where the depth-linear stress model has no room to "
            "develop."
        ),
        Caveat.MARGINAL_CONTINUUM: (
            "A feature spans only a few tens of grains, so treating the "
            "medium as a continuum over that feature is marginal."
        ),
        Caveat.ELEMENT_SIZE_EFFECTS: (
            "Askari & Kamrin's I_G exceeds 1, so the surface discretisation "
            "length survives in the dimensionless response and RFT's "
            "superposition argument is no longer exact. Refining the mesh "
            "makes this worse, not better."
        ),
    }
)

STANDING_CAVEATS: tuple[Caveat, ...] = (
    Caveat.TRANSIENT_RESPONSE,
    Caveat.DISTURBED_GROUND,
    Caveat.INCLINE,
    Caveat.SHADOWING,
    Caveat.SHARP_CORNERS,
    Caveat.BORROWED_COEFFICIENTS,
)
"""Caveats that apply to every F0 result without exception."""


@dataclass(frozen=True, slots=True)
class FeatureScale:
    """One geometric length at which the envelope is judged.

    Attributes:
        name: Human label, quoted in verdict text.
        length_m: The length itself.
    """

    name: str
    length_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.length_m) or self.length_m <= 0.0:
            raise SolverInputError(
                f"feature scale '{self.name}' must be a positive finite length, "
                f"got {self.length_m!r} m"
            )


@dataclass(frozen=True, slots=True)
class DimensionlessGroups:
    """The dimensionless state of one query at one feature scale.

    Attributes:
        scale: The feature length the groups were formed on.
        speed_m_s: Intrusion speed.
        grain_diameter_m: Median grain diameter ``d``.
        froude: ``v / sqrt(g L)``. RFT's stated limit is 0.4.
        micro_inertial_number: ``(v / L) * d / sqrt(g L)``, equivalently
            ``Fr * d / L``. Limit 0.1.
        grain_size_ratio: ``d / L``.
        continuum_size_ratio: ``L / d``, the number of grains across the
            feature. Below ~10 there is no continuum to fit.
        macro_inertial_number: Askari & Kamrin's ``I_G = v^2 d^2 /
            (g lambda^2)`` on the *discretisation* length ``lambda``.
            RFT is exact only while this stays small; at ``lambda = 2 mm``,
            ``d = 0.5 mm`` and ``v = 25 m/s`` it is about 4.0.
        element_size_m: The discretisation length ``lambda`` used.
    """

    scale: FeatureScale
    speed_m_s: float
    grain_diameter_m: float
    froude: float
    micro_inertial_number: float
    grain_size_ratio: float
    continuum_size_ratio: float
    macro_inertial_number: float
    element_size_m: float

    def describe(self) -> str:
        """One line per scale, suitable for a manifest."""
        return (
            f"{self.scale.name} (L={self.scale.length_m * 1e3:.3g} mm): "
            f"Fr={self.froude:.3g} (limit {RFT_FROUDE_LIMIT}), "
            f"I={self.micro_inertial_number:.3g} (limit "
            f"{RFT_INERTIAL_NUMBER_LIMIT}), "
            f"d/L={self.grain_size_ratio:.3g}, L/d={self.continuum_size_ratio:.3g}, "
            f"I_G={self.macro_inertial_number:.3g}"
        )


@dataclass(frozen=True, slots=True)
class EnvelopeContext:
    """What the *judging solver* brings to a verdict, not the query.

    Every other argument of :func:`evaluate_envelope` is a measurement of
    the intrusion being judged.  These three are supplied by the solver
    around it: the field its dimensionless groups are formed in, how much
    of its own surface it had to bend to stay inside the fitted domain,
    and anything further it wants said about the answer.  They travel
    together because they are all carried onto the verdict rather than
    judged, and grouping them puts two invariants on a constructor that
    previously rode all the way onto a public value object unchecked --
    a clamped fraction outside ``[0, 1]``, and a caveat list holding
    something that is not a :class:`Caveat` (which used to surface only
    as a ``KeyError`` from :meth:`ValidityVerdict.summary`).

    Attributes:
        gravity_m_s2: Gravitational acceleration the groups are formed
            in. Defaults to :data:`GRAVITY_M_S2`.
        clamped_area_fraction: Share of the active area whose orientation
            fell outside the fitted domain of the 3D-RFT polynomial and
            had to be clamped to the vertical-wall limit.
        extra_caveats: Further caveats the caller raises itself.
    """

    gravity_m_s2: float = GRAVITY_M_S2
    clamped_area_fraction: float = 0.0
    extra_caveats: tuple[Caveat, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise SolverInputError(
                f"gravity must be positive, got {self.gravity_m_s2!r}"
            )
        fraction = float(self.clamped_area_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise SolverInputError(
                f"clamped_area_fraction must lie in [0, 1], got "
                f"{self.clamped_area_fraction!r}"
            )
        caveats = tuple(self.extra_caveats)
        offenders = [item for item in caveats if not isinstance(item, Caveat)]
        if offenders:
            raise SolverInputError(
                f"extra_caveats must all be Caveat, got {offenders!r}"
            )
        object.__setattr__(self, "extra_caveats", caveats)


_DEFAULT_CONTEXT = EnvelopeContext()
"""The context of a query judged at standard gravity with nothing clamped."""


def dimensionless_groups(
    *,
    speed_m_s: float,
    feature_length_m: float,
    grain_diameter_m: float,
    element_size_m: float,
    name: str = "feature",
    gravity_m_s2: float = GRAVITY_M_S2,
) -> DimensionlessGroups:
    """Form every dimensionless group the envelope is judged on.

    Args:
        speed_m_s: Intrusion speed.
        feature_length_m: The geometric scale ``L``.
        grain_diameter_m: Median grain diameter ``d``.
        element_size_m: Surface discretisation length ``lambda``.
        name: Label for the scale.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The groups at this scale.

    Raises:
        SolverInputError: If any input is non-finite or non-positive
            where positivity is required.
    """
    scale = FeatureScale(name=name, length_m=feature_length_m)
    speed = float(speed_m_s)
    grain = float(grain_diameter_m)
    element = float(element_size_m)
    if not math.isfinite(speed) or speed < 0.0:
        raise SolverInputError(f"speed must be finite and non-negative, got {speed!r}")
    if not math.isfinite(grain) or grain <= 0.0:
        raise SolverInputError(f"grain diameter must be positive, got {grain!r} m")
    if not math.isfinite(element) or element <= 0.0:
        raise SolverInputError(f"element size must be positive, got {element!r} m")
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise SolverInputError(f"gravity must be positive, got {gravity_m_s2!r}")

    length = scale.length_m
    froude = speed / math.sqrt(gravity_m_s2 * length)
    return DimensionlessGroups(
        scale=scale,
        speed_m_s=speed,
        grain_diameter_m=grain,
        froude=froude,
        micro_inertial_number=froude * grain / length,
        grain_size_ratio=grain / length,
        continuum_size_ratio=length / grain,
        macro_inertial_number=(speed**2 * grain**2) / (gravity_m_s2 * element**2),
        element_size_m=element,
    )


@dataclass(frozen=True)
class ValidityVerdict:
    """The statement that travels with every force this package reports.

    Attributes:
        status: How much of the answer may be believed.
        groups: One :class:`DimensionlessGroups` per feature scale, in
            the order they were supplied. The *smallest* feature drives
            the verdict, because that is where the physics fails first.
        caveats: Documented reasons for distrust.
        reasons: Free-text findings, one per triggered rule.
        governing_index: Index into ``groups`` of the scale that set the
            status.
        clamped_area_fraction: Share of active area whose orientation had
            to be clamped into the fitted domain of the polynomial.
        details: Extra named scalars a caller may want in a manifest.
    """

    status: EnvelopeStatus
    groups: tuple[DimensionlessGroups, ...]
    caveats: tuple[Caveat, ...] = ()
    reasons: tuple[str, ...] = ()
    governing_index: int = 0
    clamped_area_fraction: float = 0.0
    details: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.groups:
            raise SolverInputError(
                "a validity verdict must be formed on at least one feature scale"
            )
        if not 0 <= self.governing_index < len(self.groups):
            raise SolverInputError(
                f"governing_index {self.governing_index} out of range for "
                f"{len(self.groups)} scale(s)"
            )
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def governing(self) -> DimensionlessGroups:
        """The groups at the scale that set the status."""
        return self.groups[self.governing_index]

    @property
    def is_refusal(self) -> bool:
        """True when no number may be reported."""
        return self.status is EnvelopeStatus.REFUSED

    @property
    def is_within_stated_envelope(self) -> bool:
        """True only inside 3D-RFT's own published limits."""
        return self.status is EnvelopeStatus.WITHIN

    def require_usable(self, policy: RefusalPolicy = RefusalPolicy.STRICT) -> None:
        """Refuse the query when the policy is strict and the verdict is a refusal.

        This is a plain ``raise``, not an ``assert`` and not a contract
        decorator: ``python -O`` strips assertions and ``DBC_LEVEL=off``
        disables contracts, and an envelope that evaporates under an
        optimisation flag is worse than no envelope at all.

        Raises:
            OutOfEnvelopeError: If refused under a strict policy.
        """
        if self.is_refusal and policy is RefusalPolicy.STRICT:
            raise OutOfEnvelopeError(
                "3D-RFT refuses this query:\n" + self.summary(),
                verdict=self,
            )

    def summary(self) -> str:
        """A multi-line statement fit for a run manifest or a log."""
        status = self.status.value
        lines = [f"validity: {status.upper()}"]
        lines.extend(f"  scale: {group.describe()}" for group in self.groups)
        if self.clamped_area_fraction > 0.0:
            lines.append(
                f"  {self.clamped_area_fraction:.1%} of the active area was "
                "clamped into the fitted orientation domain"
            )
        lines.extend(f"  reason: {reason}" for reason in self.reasons)
        lines.extend(f"  caveat: {_CAVEAT_TEXT[caveat]}" for caveat in self.caveats)
        return "\n".join(lines)


def worst_of(verdicts: Iterable[ValidityVerdict]) -> ValidityVerdict:
    """Combine per-step verdicts into the verdict for the whole trace.

    The combined verdict takes the worst status, the union of the
    caveats, and the groups of the step that was worst -- so a shot that
    was in-envelope for 99 steps and refused for one is reported as
    refused.

    Args:
        verdicts: Verdicts to combine, in any order.

    Returns:
        The combined verdict.

    Raises:
        SolverInputError: If ``verdicts`` is empty.
    """
    collected = list(verdicts)
    if not collected:
        raise SolverInputError("cannot combine an empty sequence of verdicts")
    worst = max(collected, key=lambda verdict: _STATUS_ORDER.index(verdict.status))
    caveats: dict[Caveat, None] = {}
    reasons: dict[str, None] = {}
    for verdict in collected:
        caveats.update(dict.fromkeys(verdict.caveats))
        reasons.update(dict.fromkeys(verdict.reasons))
    return ValidityVerdict(
        status=worst.status,
        groups=worst.groups,
        caveats=tuple(caveats),
        reasons=tuple(reasons),
        governing_index=worst.governing_index,
        clamped_area_fraction=max(v.clamped_area_fraction for v in collected),
        details=dict(worst.details),
    )


def _status_for(
    group: DimensionlessGroups,
    *,
    dynamic_terms_active: bool,
    reasons: list[str],
    caveats: list[Caveat],
) -> EnvelopeStatus:
    """Judge one feature scale, appending its findings."""
    status = EnvelopeStatus.WITHIN
    if group.froude > RFT_FROUDE_LIMIT:
        caveats.append(Caveat.SUPERCRITICAL_FROUDE)
        reasons.append(
            f"Fr = {group.froude:.3g} at the {group.scale.name} scale exceeds "
            f"3D-RFT's stated limit of {RFT_FROUDE_LIMIT} by "
            f"{group.froude / RFT_FROUDE_LIMIT:.0f}x"
        )
        status = EnvelopeStatus.EXTRAPOLATED
    if group.micro_inertial_number > RFT_INERTIAL_NUMBER_LIMIT:
        caveats.append(Caveat.GRAIN_RATE_EFFECTS)
        reasons.append(
            f"micro-inertial I = {group.micro_inertial_number:.3g} at the "
            f"{group.scale.name} scale exceeds {RFT_INERTIAL_NUMBER_LIMIT}"
        )
        status = max(status, EnvelopeStatus.EXTRAPOLATED, key=_STATUS_ORDER.index)
    if group.continuum_size_ratio < MARGINAL_CONTINUUM_SIZE_RATIO:
        caveats.append(Caveat.MARGINAL_CONTINUUM)
        reasons.append(
            f"the {group.scale.name} scale spans only "
            f"{group.continuum_size_ratio:.3g} grains"
        )
        status = max(status, EnvelopeStatus.EXTRAPOLATED, key=_STATUS_ORDER.index)
    if group.froude > RFT_QUASI_STATIC_FROUDE_CEILING and not dynamic_terms_active:
        reasons.append(
            f"Fr = {group.froude:.3g} > {RFT_QUASI_STATIC_FROUDE_CEILING} with the "
            "dynamic terms switched off: a quasi-static solver is wrong here by "
            "an order of magnitude, so no force is reported"
        )
        status = EnvelopeStatus.REFUSED
    if group.continuum_size_ratio < MIN_CONTINUUM_SIZE_RATIO:
        reasons.append(
            f"the {group.scale.name} scale spans only "
            f"{group.continuum_size_ratio:.3g} grains (minimum "
            f"{MIN_CONTINUUM_SIZE_RATIO:.0f}): there is no continuum to fit a "
            "stress response to"
        )
        status = EnvelopeStatus.REFUSED
    return status


def _groups_for(
    *,
    speed_m_s: float,
    feature_lengths_m: Mapping[str, float],
    grain_diameter_m: float,
    element_size_m: float,
    gravity_m_s2: float,
) -> tuple[DimensionlessGroups, ...]:
    """Form the groups at every supplied scale, smallest feature first.

    Raises:
        SolverInputError: If ``feature_lengths_m`` is empty or malformed.
    """
    if not feature_lengths_m:
        raise SolverInputError(
            "at least one feature scale is required; judging the envelope on no "
            "scale at all is how a solver reports a flattering number"
        )
    ordered = sorted(feature_lengths_m.items(), key=lambda item: item[1])
    return tuple(
        dimensionless_groups(
            speed_m_s=speed_m_s,
            feature_length_m=length,
            grain_diameter_m=grain_diameter_m,
            element_size_m=element_size_m,
            name=name,
            gravity_m_s2=gravity_m_s2,
        )
        for name, length in ordered
    )


def _apply_global_rules(
    governing: DimensionlessGroups,
    worst_rank: int,
    *,
    submerged_depth_m: float,
    structural_correction_calibrated: bool,
    context: EnvelopeContext,
    reasons: list[str],
    caveats: list[Caveat],
) -> int:
    """Judge the rules that do not depend on the feature scale.

    They live apart from :func:`_status_for` so that their finding is
    stated once against the governing scale, rather than repeated at
    every feature length.

    Args:
        governing: Groups at the scale that set the status.
        worst_rank: Rank in :data:`_STATUS_ORDER` reached so far.
        submerged_depth_m: Deepest submerged point, positive downward.
        structural_correction_calibrated: Whether ``delta_h`` has been
            calibrated for this body.
        context: What the judging solver brings to the verdict.
        reasons: Findings, appended in place.
        caveats: Caveats, appended in place.

    Returns:
        The rank after these rules, never below ``worst_rank``.
    """
    if governing.speed_m_s > MAX_VALIDATED_SPEED_M_S:
        caveats.append(Caveat.BEYOND_PUBLISHED_SPEED)
        reasons.append(
            f"speed {governing.speed_m_s:.3g} m/s is "
            f"{governing.speed_m_s / MAX_VALIDATED_SPEED_M_S:.0f}x the fastest "
            f"intrusion in the published corpus ({MAX_VALIDATED_SPEED_M_S} m/s)"
        )
        worst_rank = max(
            worst_rank, _STATUS_ORDER.index(EnvelopeStatus.BEYOND_VALIDATION)
        )
    if governing.macro_inertial_number > MAX_ELEMENT_INERTIAL_NUMBER:
        caveats.append(Caveat.ELEMENT_SIZE_EFFECTS)
        reasons.append(
            f"I_G = {governing.macro_inertial_number:.3g} on a "
            f"{governing.element_size_m * 1e3:.3g} mm surface discretisation "
            f"exceeds {MAX_ELEMENT_INERTIAL_NUMBER:.0f}"
        )
        worst_rank = max(worst_rank, _STATUS_ORDER.index(EnvelopeStatus.EXTRAPOLATED))
    if not structural_correction_calibrated:
        caveats.append(Caveat.UNCALIBRATED_STRUCTURAL_CORRECTION)
    if context.clamped_area_fraction > 0.0:
        caveats.append(Caveat.UPWARD_FACING_LEADING_EDGE)
    shallow_limit = 2.0 * governing.grain_diameter_m
    if 0.0 < submerged_depth_m < shallow_limit:
        caveats.append(Caveat.SHALLOW_INTRUSION)
        reasons.append(
            f"deepest submerged point is {submerged_depth_m * 1e3:.3g} mm, under "
            f"{shallow_limit * 1e3:.3g} mm (two grain diameters)"
        )
    caveats.extend(context.extra_caveats)
    return worst_rank


def evaluate_envelope(
    *,
    speed_m_s: float,
    feature_lengths_m: Mapping[str, float],
    grain_diameter_m: float,
    element_size_m: float,
    dynamic_terms_active: bool,
    submerged_depth_m: float = 0.0,
    structural_correction_calibrated: bool = False,
    context: EnvelopeContext = _DEFAULT_CONTEXT,
) -> ValidityVerdict:
    """Judge a query at every supplied feature scale.

    The governing scale is the *smallest* one that reaches the worst
    status, because the physics fails at the small end first: at 25 m/s
    the 5 mm leading edge is at ``I = 11.3`` while the 100 mm clubhead is
    at ``I = 0.126``.

    Args:
        speed_m_s: Intrusion speed.
        feature_lengths_m: Named geometric scales, e.g. ``{"clubhead":
            0.1, "sole width": 0.03, "leading edge": 0.005}``.
        grain_diameter_m: Median grain diameter.
        element_size_m: Surface discretisation length.
        dynamic_terms_active: Whether the DRFT inertial term is on.
        submerged_depth_m: Deepest submerged point, positive downward.
        structural_correction_calibrated: Whether ``delta_h`` has been
            calibrated for this body. It has not been, for any wedge.
        context: Gravity, the clamped area fraction and any caveats the
            judging solver raises itself. Defaults to standard gravity
            with nothing clamped.

    Returns:
        The verdict. Nothing is raised here; refusal is a *value*, and
        :meth:`ValidityVerdict.require_usable` turns it into an exception
        at the point a caller has declared a policy.

    Raises:
        SolverInputError: If ``feature_lengths_m`` is empty or malformed.
    """
    groups = _groups_for(
        speed_m_s=speed_m_s,
        feature_lengths_m=feature_lengths_m,
        grain_diameter_m=grain_diameter_m,
        element_size_m=element_size_m,
        gravity_m_s2=context.gravity_m_s2,
    )

    reasons: list[str] = []
    caveats: list[Caveat] = list(STANDING_CAVEATS)
    statuses = [
        _status_for(
            group,
            dynamic_terms_active=dynamic_terms_active,
            reasons=reasons,
            caveats=caveats,
        )
        for group in groups
    ]
    worst_rank = max(_STATUS_ORDER.index(status) for status in statuses)
    governing_index = next(
        index
        for index, status in enumerate(statuses)
        if _STATUS_ORDER.index(status) == worst_rank
    )
    worst_rank = _apply_global_rules(
        groups[governing_index],
        worst_rank,
        submerged_depth_m=submerged_depth_m,
        structural_correction_calibrated=structural_correction_calibrated,
        context=context,
        reasons=reasons,
        caveats=caveats,
    )

    return ValidityVerdict(
        status=_STATUS_ORDER[worst_rank],
        groups=groups,
        caveats=tuple(dict.fromkeys(caveats)),
        reasons=tuple(dict.fromkeys(reasons)),
        governing_index=governing_index,
        clamped_area_fraction=float(context.clamped_area_fraction),
        details={
            "submerged_depth_m": float(submerged_depth_m),
            "grain_diameter_m": float(grain_diameter_m),
            "element_size_m": float(element_size_m),
        },
    )
