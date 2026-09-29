"""The DRFT structural correction ``delta_h`` (issue #8611).

DRFT has **two** corrections, not one::

    t = alpha(beta, gamma) * H(-z_tilde) * |z_tilde|  -  n_hat * lambda * rho * v_n^2
    z_tilde = z + delta_h

1. the inertial term ``lambda * rho * v_n^2``, and
2. the dynamic structural correction ``delta_h``, which lowers the
   *effective free surface* and feeds back through the depth linearity.

**The second is not optional.**  The source paper's central finding is
that applying ``lambda rho v^2`` *without* ``delta_h`` produced the wrong
**sign** of sinkage for wheels -- at every ``lambda`` from 1 to 100.  A
solver that implements only the inertial term is not a simplified DRFT,
it is a broken one.

What is and is not known
------------------------

The only published closed form is for a wheel: ``delta_h = r (r omega^2 / g)``,
which is ``v^2 / g`` at the rim.  The form is explicitly geometry-specific.
Vertical and horizontal *plate* intrusions are the cases where
``delta_h ~ 0``, which helps a leading edge but not a planing sole.

**No wedge-specific form exists.**  This module therefore exposes
``delta_h`` as a calibratable model with a documented default, and every
model reports ``is_calibrated_for_wedge = False``.  The solver turns that
flag into a standing caveat on every verdict.  Nothing here is a
calibration; it is a convention chosen so the model is well behaved while
the real one is missing.

The default, and why it is shaped the way it is
-----------------------------------------------

:class:`CrossoverSaturatingDepression` uses::

    delta_h = s * |z| * v_n^2 / (v_n^2 + v_ref^2)
    v_ref^2 = (depth stress scale) * |z| / (inertial stress scale)

``v_ref`` is the *local* speed at which the inertial stress equals the
depth stress -- about 7 m/s at a 40 mm divot, matching the 6.8 m/s
crossover in the research digest.  Three properties follow, and all three
are tested:

* **Quasi-static limit is the plate limit.**  ``v << v_ref`` gives
  ``delta_h -> 0``, which is the measured plate behaviour, so the model
  degrades to quasi-static RFT exactly where quasi-static RFT is right.
* **The depth force is monotonically increasing in both depth and
  speed.**  With ``K = v_ref^2 / |z|`` fixed, the effective depth is
  ``|z| - s |z| v^2/(v^2 + K|z|)``, whose derivative in ``|z|`` is
  ``1 - s v^4/(v^2 + K|z|)^2 >= 0`` for ``s <= 1``, and whose derivative
  in ``v`` is negative but bounded below by ``-2 v (depth scale) |z| s /
  v_ref^2 = -2 v (inertial scale) s``, which the inertial term's
  ``+2 v (inertial scale)`` always dominates for ``s <= 1``.  So neither
  monotonicity requirement can be violated by the correction.
* **The sign cannot flip.**  ``s < 1`` keeps ``z_tilde`` strictly
  negative, so the depth term can be attenuated but never inverted --
  the failure mode the source paper reported.

At 25 m/s and 40 mm the default leaves the depth term at about 1.3% of
the total, against the ~1% implied independently by Katsuragi & Durian's
``v^2 > 25 mu g z`` inertial-dominance criterion.  That agreement is a
weak sanity check on the default, **not** a calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..sand.provenance import PropertyProvenance, ProvenanceBasis
from .envelope import GRAVITY_M_S2
from .exceptions import CalibrationError

__all__ = [
    "CrossoverSaturatingDepression",
    "DepressionInputs",
    "StructuralCorrection",
    "WheelAnalogueDepression",
    "ZeroDepression",
    "default_structural_correction",
]


@dataclass(frozen=True)
class DepressionInputs:
    """Everything a ``delta_h`` model may look at, as arrays.

    Attributes:
        depth_m: Positive depth of each element below the *undisturbed*
            free surface, ``(m,)``.
        normal_speed_m_s: ``max(v . n_hat, 0)`` per element, ``(m,)``.
        depth_stress_scale_pa_per_m: ``xi_n * |alpha_n|`` per element --
            the local depth-linear stress gradient, ``(m,)``.
        inertial_stress_scale_pa_s2_per_m2: ``lambda * rho``, scalar.
        gravity_m_s2: Gravitational acceleration.
    """

    depth_m: NDArray[np.float64]
    normal_speed_m_s: NDArray[np.float64]
    depth_stress_scale_pa_per_m: NDArray[np.float64]
    inertial_stress_scale_pa_s2_per_m2: float
    gravity_m_s2: float = GRAVITY_M_S2


@runtime_checkable
class StructuralCorrection(Protocol):
    """A model for the dynamic free-surface depression ``delta_h``."""

    @property
    def name(self) -> str:
        """Short identifier for manifests."""
        ...

    @property
    def is_calibrated_for_wedge(self) -> bool:
        """Whether this model has been calibrated against wedge data.

        No implementation returns ``True``.  The method exists so the
        claim is a value in the result rather than a sentence in a
        docstring nobody reads.
        """
        ...

    @property
    def provenance(self) -> PropertyProvenance:
        """Where the model's form and constants came from."""
        ...

    def depression_m(self, inputs: DepressionInputs) -> NDArray[np.float64]:
        """Return ``delta_h >= 0`` per element, ``(m,)``."""
        ...


def _validate_inputs(inputs: DepressionInputs) -> None:
    """Reject malformed depression inputs with a plain raise."""
    if inputs.depth_m.shape != inputs.normal_speed_m_s.shape:
        raise CalibrationError(
            "depth and normal-speed arrays must have the same shape, got "
            f"{inputs.depth_m.shape} and {inputs.normal_speed_m_s.shape}"
        )
    if np.any(inputs.depth_m < 0.0):
        raise CalibrationError(
            "depth_m must be a positive depth below the free surface; negative "
            "entries mean the caller passed a signed z coordinate"
        )


@dataclass(frozen=True)
class ZeroDepression:
    """``delta_h = 0``: the measured plate limit.

    Vertical and horizontal plate intrusions are the cases where the
    depression vanishes, so this is the right model for a thin leading
    edge and the wrong one for a planing sole.  It is also the model that
    reproduces quasi-static RFT exactly, which makes it the reference the
    other models are compared against in tests.
    """

    @property
    def name(self) -> str:
        """Short identifier for manifests."""
        return "zero"

    @property
    def is_calibrated_for_wedge(self) -> bool:
        """False: measured for plates, never for a wedge sole."""
        return False

    @property
    def provenance(self) -> PropertyProvenance:
        """Plate-intrusion measurements, borrowed."""
        return PropertyProvenance(
            basis=ProvenanceBasis.BORROWED_ANALOGUE,
            source="vertical/horizontal plate intrusion, delta_h ~ 0 "
            "(research digest addendum, section 2)",
            note="Correct for plates. A wedge sole is not a plate, and the "
            "source paper reports that omitting delta_h inverted the sign of "
            "sinkage for wheels.",
        )

    def depression_m(self, inputs: DepressionInputs) -> NDArray[np.float64]:
        """Return an array of zeros shaped like the input depths."""
        _validate_inputs(inputs)
        return np.zeros_like(inputs.depth_m)


@dataclass(frozen=True)
class WheelAnalogueDepression:
    """``delta_h = c * v_n^2 / g``: the only published closed form.

    This is the wheel result ``r (r omega^2 / g)`` written at the rim.
    It is retained because it is the one form anybody measured, and
    because it makes the scale of the problem obvious: at 25 m/s it is
    **63.7 m**, which is not a free-surface depression, it is a
    demonstration that the wheel form does not transfer to a clubhead.

    Attributes:
        coefficient: ``c``, 1.0 for the published wheel form.
    """

    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise CalibrationError(
                f"coefficient must be finite and non-negative, got {self.coefficient!r}"
            )

    @property
    def name(self) -> str:
        """Short identifier for manifests."""
        return "wheel_analogue"

    @property
    def is_calibrated_for_wedge(self) -> bool:
        """False: calibrated for a grousered wheel, not a wedge."""
        return False

    @property
    def provenance(self) -> PropertyProvenance:
        """The published wheel form, borrowed."""
        return PropertyProvenance(
            basis=ProvenanceBasis.BORROWED_ANALOGUE,
            source="delta_h = r (r omega^2 / g) for a wheel "
            "(research digest addendum, section 2)",
            note="Geometry-specific by the source's own statement. At clubhead "
            "speed it evaluates to tens of metres, so it is a reference form "
            "rather than a usable default.",
        )

    def depression_m(self, inputs: DepressionInputs) -> NDArray[np.float64]:
        """Return ``c v_n^2 / g``, unbounded."""
        _validate_inputs(inputs)
        return self.coefficient * inputs.normal_speed_m_s**2 / inputs.gravity_m_s2


@dataclass(frozen=True)
class CrossoverSaturatingDepression:
    """The default: a depression that saturates at the local crossover speed.

    ``delta_h = saturation_fraction * |z| * v_n^2 / (v_n^2 + v_ref^2)``
    where ``v_ref`` is the local depth/inertia crossover speed.  See the
    module docstring for the three properties this shape guarantees and
    the proof sketch for each.

    Attributes:
        saturation_fraction: ``s``, the largest share of the local depth
            the effective free surface may be lowered by. Must be
            strictly below 1 so the depth term can never change sign.
        reference_speed_scale: Multiplies ``v_ref``. ``1.0`` puts the
            half-depression point exactly at the crossover speed.
        fallback_reference_speed_m_s: ``v_ref`` where the local depth
            stress scale is zero (an element carrying no depth response),
            so the expression stays finite.
    """

    saturation_fraction: float = 0.9
    reference_speed_scale: float = 1.0
    fallback_reference_speed_m_s: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.saturation_fraction < 1.0:
            raise CalibrationError(
                "saturation_fraction must lie in [0, 1); a value of 1 or more "
                "lets the effective free surface reach or pass the element and "
                "invert the depth term, which is the exact failure the "
                "structural correction exists to prevent. Got "
                f"{self.saturation_fraction!r}"
            )
        for name, value in (
            ("reference_speed_scale", self.reference_speed_scale),
            ("fallback_reference_speed_m_s", self.fallback_reference_speed_m_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise CalibrationError(f"{name} must be positive, got {value!r}")

    @property
    def name(self) -> str:
        """Short identifier for manifests."""
        return "crossover_saturating"

    @property
    def is_calibrated_for_wedge(self) -> bool:
        """False. This is a documented convention, not a calibration.

        Say it plainly: the wedge-specific form of ``delta_h`` is
        unknown.  Measuring it needs the F1/F2 tier or PIV, neither of
        which has been run.
        """
        return False

    @property
    def provenance(self) -> PropertyProvenance:
        """A modelling convention, explicitly not a measurement."""
        return PropertyProvenance(
            basis=ProvenanceBasis.CONVENTION,
            source="BunkerShot3D F0 default (issue #8611)",
            note=(
                "The wedge-specific form of delta_h is unknown. This shape was "
                "chosen so that it vanishes in the quasi-static plate limit, "
                "saturates below the element depth so the depth term cannot "
                "invert, and preserves monotonicity of force in both depth and "
                "speed. It is a convention, not a calibration, and must be "
                "replaced by an F1/F2 or PIV measurement before any absolute "
                "force from this solver is quoted."
            ),
        )

    def depression_m(self, inputs: DepressionInputs) -> NDArray[np.float64]:
        """Return the saturating depression, ``(m,)``."""
        _validate_inputs(inputs)
        inertial_scale = inputs.inertial_stress_scale_pa_s2_per_m2
        if not math.isfinite(inertial_scale) or inertial_scale <= 0.0:
            raise CalibrationError(
                "the inertial stress scale lambda*rho must be positive to place "
                f"the crossover speed, got {inertial_scale!r}"
            )
        reference_squared = np.where(
            inputs.depth_stress_scale_pa_per_m > 0.0,
            inputs.depth_stress_scale_pa_per_m * inputs.depth_m / inertial_scale,
            self.fallback_reference_speed_m_s**2,
        )
        reference_squared = reference_squared * self.reference_speed_scale**2
        speed_squared = inputs.normal_speed_m_s**2
        denominator = speed_squared + reference_squared
        fraction = np.where(denominator > 0.0, speed_squared / denominator, 0.0)
        return self.saturation_fraction * inputs.depth_m * fraction


def default_structural_correction() -> StructuralCorrection:
    """The documented, uncalibrated default ``delta_h`` model.

    Returns:
        A :class:`CrossoverSaturatingDepression` with default constants.
    """
    return CrossoverSaturatingDepression()
