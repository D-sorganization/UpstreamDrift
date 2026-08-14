"""Validity envelope for DRFT (issue #8611).

From research-digest-addendum.md:
- RFT's stated limit is Fr = v/sqrt(gL) < 0.4
- At v = 25 m/s, L = 0.1 m we are at Fr = 25 -- about 60x outside
- This makes validity-envelope reporting the most important feature

The solver must:
1. Compute Fr, I, d/L for every query
2. Refuse out-of-envelope queries when dynamic terms are inactive
3. Flag extrapolation when dynamic terms are active but Fr >> 1
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass


__all__ = [
    "DRFT_LAMBDA_OBLIQUE",
    "FR_HARD_LIMIT",
    "FR_STATED_LIMIT",
    "MICRO_INERTIAL_LIMIT",
    "EnvelopeStatus",
    "ValidityEnvelope",
    "ValidityVerdict",
    "compute_froude_number",
    "compute_micro_inertial",
]

#: Gravitational acceleration [m/s^2].
_GRAVITY_M_S2: float = 9.81

#: Stated RFT validity limit (Froude number).
FR_STATED_LIMIT: float = 0.4

#: Hard limit beyond which even DRFT is deep extrapolation.
#: Using > 1.0 rather than >= 1.0 so Fr=1.0 is still in REQUIRES_DYNAMIC.
FR_HARD_LIMIT: float = 1.01

#: Micro-inertial number limit from the RFT literature.
MICRO_INERTIAL_LIMIT: float = 0.1

#: Inertial scaling factor lambda for oblique plates (research addendum).
DRFT_LAMBDA_OBLIQUE: float = 1.1


class EnvelopeStatus(enum.Enum):
    """Classification of a query's position relative to the validity envelope."""

    #: Fr < 0.4: inside the stated RFT envelope, quasi-static OK.
    INSIDE_STATED = "inside_stated"

    #: 0.4 <= Fr < 1.0: requires dynamic terms, but still within reason.
    REQUIRES_DYNAMIC = "requires_dynamic"

    #: Fr >= 1.0: extrapolation, dynamic terms mandatory, flag the result.
    EXTRAPOLATION = "extrapolation"


def compute_froude_number(velocity_m_s: float, length_scale_m: float) -> float:
    """Compute Froude number Fr = v / sqrt(g * L).

    Args:
        velocity_m_s: Intrusion velocity [m/s].
        length_scale_m: Characteristic length (clubhead or sole width) [m].

    Returns:
        Froude number (dimensionless).

    Raises:
        ValueError: If length scale is not positive.
    """
    if length_scale_m <= 0:
        raise ValueError(f"length scale must be positive, got {length_scale_m}")
    if velocity_m_s < 0:
        raise ValueError(f"velocity must be non-negative, got {velocity_m_s}")

    return velocity_m_s / math.sqrt(_GRAVITY_M_S2 * length_scale_m)


def compute_micro_inertial(
    velocity_m_s: float,
    grain_diameter_m: float,
    intruder_scale_m: float,
) -> float:
    """Compute micro-inertial number I = v * d / sqrt(g * L^3).

    This measures the importance of grain-scale inertia relative to gravity.
    From research-digest-addendum.md Table 1: at v=25 m/s, d=0.5 mm, L=100 mm,
    micro-inertial I = 0.126.

    Args:
        velocity_m_s: Intrusion velocity [m/s].
        grain_diameter_m: Median grain diameter d50 [m].
        intruder_scale_m: Intruder length scale L [m].

    Returns:
        Micro-inertial number (dimensionless).
    """
    if intruder_scale_m <= 0:
        raise ValueError(f"intruder scale must be positive, got {intruder_scale_m}")
    if grain_diameter_m <= 0:
        raise ValueError(f"grain diameter must be positive, got {grain_diameter_m}")

    import math

    return (velocity_m_s * grain_diameter_m) / math.sqrt(
        _GRAVITY_M_S2 * intruder_scale_m**3
    )


@dataclass(frozen=True, slots=True)
class ValidityEnvelope:
    """Dimensionless groups defining the query's position in parameter space.

    Attributes:
        froude: Fr = v / sqrt(g*L), the primary validity indicator.
        micro_inertial: I = v^2*d^2 / (g*lambda^2), grain-scale inertia.
        depth_ratio: d/L, grain diameter to intruder scale.
    """

    froude: float
    micro_inertial: float
    depth_ratio: float

    @property
    def status(self) -> EnvelopeStatus:
        """Classify the query's envelope status."""
        if self.froude < FR_STATED_LIMIT:
            return EnvelopeStatus.INSIDE_STATED
        if self.froude < FR_HARD_LIMIT:
            return EnvelopeStatus.REQUIRES_DYNAMIC
        return EnvelopeStatus.EXTRAPOLATION


@dataclass(frozen=True, slots=True)
class ValidityVerdict:
    """Complete validity assessment for a DRFT query.

    Attributes:
        envelope: The dimensionless groups.
        should_refuse: If True, the solver must not return a force.
        is_extrapolation: If True, the result is outside validated range.
        reason: Human-readable explanation.
    """

    envelope: ValidityEnvelope
    should_refuse: bool
    is_extrapolation: bool
    reason: str

    @classmethod
    def evaluate(
        cls,
        froude: float,
        micro_inertial: float,
        depth_ratio: float,
        dynamic_terms_active: bool,
    ) -> ValidityVerdict:
        """Evaluate the validity of a query.

        Args:
            froude: Froude number Fr.
            micro_inertial: Micro-inertial number I.
            depth_ratio: Grain-to-intruder ratio d/L.
            dynamic_terms_active: Whether DRFT dynamic terms are enabled.

        Returns:
            A verdict with should_refuse, is_extrapolation, and reason.
        """
        envelope = ValidityEnvelope(
            froude=froude,
            micro_inertial=micro_inertial,
            depth_ratio=depth_ratio,
        )

        status = envelope.status

        if status == EnvelopeStatus.INSIDE_STATED:
            return cls(
                envelope=envelope,
                should_refuse=False,
                is_extrapolation=False,
                reason="Inside stated RFT envelope (Fr < 0.4).",
            )

        if status == EnvelopeStatus.REQUIRES_DYNAMIC:
            if dynamic_terms_active:
                return cls(
                    envelope=envelope,
                    should_refuse=False,
                    is_extrapolation=False,
                    reason=f"Fr = {froude:.2f} requires dynamic terms (active).",
                )
            return cls(
                envelope=envelope,
                should_refuse=True,
                is_extrapolation=False,
                reason=(
                    f"Fr = {froude:.2f} > 0.4: dynamic terms required but not "
                    "active. Enable DRFT dynamic correction or reduce velocity."
                ),
            )

        # status == EnvelopeStatus.EXTRAPOLATION
        if dynamic_terms_active:
            return cls(
                envelope=envelope,
                should_refuse=False,
                is_extrapolation=True,
                reason=(
                    f"Fr = {froude:.2f} >> 1: deep extrapolation beyond validated "
                    "RFT envelope. Result is indicative only; calibrate against "
                    "F1/F2 tier or experiment."
                ),
            )
        return cls(
            envelope=envelope,
            should_refuse=True,
            is_extrapolation=True,
            reason=(
                f"Fr = {froude:.2f} >> 1: quasi-static RFT is invalid. "
                "Dynamic terms are mandatory at bunker-shot speeds."
            ),
        )
