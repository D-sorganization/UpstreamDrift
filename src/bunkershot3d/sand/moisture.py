"""Moisture regimes and the cavitation cap (issue #8610).

Wet sand is **two distinct regimes**, not one scalar knob (ADR-0032):

*damp / capillary*
    Menisci between grains carry a matric suction ``s ~ 2 sigma / r`` which
    appears as an apparent cohesion of order 1-10 kPa. Modelled with Bishop's
    effective stress, taking the effective-stress parameter ``chi`` as the
    degree of saturation: ``c_app = chi * s * tan(phi)``.

*saturated / cavitating*
    A 10 ms bunker impact in USGA-spec sand is globally **drained**
    (``k ~ 3e-4 m/s``, ``E_oed ~ 20 MPa`` give ``c_v ~ 0.61 m^2/s`` and a time
    factor ``T ~ 15`` over a 20 mm zone), so there is no undrained excess pore
    pressure to speak of. The real effect is *local shear-band dilation*, and
    the suction it generates is **capped by cavitation**: water cannot sustain
    an absolute pressure below its vapour pressure, so the gauge pore pressure
    cannot fall below about -100 kPa. That is worth roughly 65 kPa of extra
    shear strength -- of order 130 N on the club against a 200-600 N peak.

**The cap is mandatory.** Without it a poroelastic estimate invents multi-MPa
suction and overpredicts club force severalfold. Every suction term in this
package passes through :func:`clamp_pore_pressure_pa` or
:func:`clamp_suction_pa`, and both raise rather than assert, because
``python -O`` strips assertions.

All pressures are pascals; gauge unless the name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .exceptions import MoistureRegimeError

__all__ = [
    "ATMOSPHERIC_PRESSURE_PA",
    "CAVITATION_PORE_PRESSURE_PA",
    "CAVITATION_SUCTION_LIMIT_PA",
    "DRY_SATURATION_CEILING",
    "SATURATED_SATURATION_FLOOR",
    "WATER_DENSITY_KG_M3",
    "WATER_SURFACE_TENSION_N_PER_M",
    "WATER_VAPOUR_PRESSURE_PA",
    "MoistureRegime",
    "MoistureState",
    "capillary_apparent_cohesion_pa",
    "capillary_suction_pa",
    "cavitation_limited_strength_gain_pa",
    "clamp_pore_pressure_pa",
    "clamp_suction_pa",
    "classify_regime",
    "degree_of_saturation",
]

WATER_DENSITY_KG_M3 = 998.2
"""Density of fresh water at 20 C."""

WATER_SURFACE_TENSION_N_PER_M = 0.0728
"""Air-water surface tension at 20 C."""

ATMOSPHERIC_PRESSURE_PA = 101325.0
"""Standard atmosphere, the datum for gauge pressure."""

WATER_VAPOUR_PRESSURE_PA = 2339.0
"""Saturated vapour pressure of water at 20 C (absolute)."""

CAVITATION_PORE_PRESSURE_PA = WATER_VAPOUR_PRESSURE_PA - ATMOSPHERIC_PRESSURE_PA
"""Lowest attainable gauge pore pressure, about -99 kPa.

Derived rather than hard-coded: pore water cavitates once its *absolute*
pressure reaches the vapour pressure, so the gauge floor is
``p_vapour - p_atm``. The research digest quotes this as "about -100 kPa".
"""

CAVITATION_SUCTION_LIMIT_PA = -CAVITATION_PORE_PRESSURE_PA
"""Largest attainable suction magnitude, about +99 kPa."""

DRY_SATURATION_CEILING = 0.02
"""Below this degree of saturation there are no load-bearing menisci."""

SATURATED_SATURATION_FLOOR = 0.90
"""At or above this degree of saturation the menisci have merged."""


def _require_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise MoistureRegimeError(f"{name} must be finite, got {value!r}")
    return float(value)


class MoistureRegime(StrEnum):
    """Which moisture model applies. Selected explicitly, never inferred late."""

    DRY = "dry"
    DAMP_CAPILLARY = "damp_capillary"
    SATURATED = "saturated"


# --------------------------------------------------------------------------
# The cavitation cap
# --------------------------------------------------------------------------


def clamp_pore_pressure_pa(pore_pressure_pa: float) -> float:
    """Clamp a gauge pore pressure at the cavitation floor.

    Args:
        pore_pressure_pa: Requested gauge pore pressure. Negative is suction.

    Returns:
        ``max(pore_pressure_pa, CAVITATION_PORE_PRESSURE_PA)``.

    Raises:
        MoistureRegimeError: if the input is not finite. This is an explicit
            raise, not an assertion, so it survives ``python -O``.
    """
    value = _require_finite(pore_pressure_pa, "pore pressure")
    return max(value, CAVITATION_PORE_PRESSURE_PA)


def clamp_suction_pa(suction_pa: float) -> float:
    """Clamp a suction *magnitude* at the cavitation limit.

    Args:
        suction_pa: Non-negative suction magnitude.

    Returns:
        ``min(suction_pa, CAVITATION_SUCTION_LIMIT_PA)``.

    Raises:
        MoistureRegimeError: if the input is not finite or is negative.
    """
    value = _require_finite(suction_pa, "suction")
    if value < 0.0:
        raise MoistureRegimeError(
            f"suction magnitude must be non-negative, got {value!r} Pa; "
            "pass a gauge pore pressure to clamp_pore_pressure_pa instead"
        )
    return min(value, CAVITATION_SUCTION_LIMIT_PA)


# --------------------------------------------------------------------------
# Regime selection
# --------------------------------------------------------------------------


def degree_of_saturation(
    gravimetric_water_content: float,
    void_ratio: float,
    particle_density_kg_m3: float,
    water_density_kg_m3: float = WATER_DENSITY_KG_M3,
) -> float:
    """Return ``S = w * rho_s / (e * rho_w)``, clipped to [0, 1].

    Raises:
        MoistureRegimeError: on non-finite or non-physical arguments.
    """
    w = _require_finite(gravimetric_water_content, "water content")
    e = _require_finite(void_ratio, "void ratio")
    rho_s = _require_finite(particle_density_kg_m3, "particle density")
    rho_w = _require_finite(water_density_kg_m3, "water density")
    if w < 0.0:
        raise MoistureRegimeError(f"water content must not be negative, got {w!r}")
    if e <= 0.0 or rho_s <= 0.0 or rho_w <= 0.0:
        raise MoistureRegimeError(
            "void ratio, particle density and water density must all be "
            f"positive, got {e!r}, {rho_s!r}, {rho_w!r}"
        )
    return min(1.0, w * rho_s / (e * rho_w))


def classify_regime(saturation: float) -> MoistureRegime:
    """Map a degree of saturation onto a moisture regime.

    Raises:
        MoistureRegimeError: if ``saturation`` is outside [0, 1].
    """
    s = _require_finite(saturation, "degree of saturation")
    if not 0.0 <= s <= 1.0:
        raise MoistureRegimeError(f"degree of saturation must lie in [0, 1], got {s!r}")
    if s < DRY_SATURATION_CEILING:
        return MoistureRegime.DRY
    if s < SATURATED_SATURATION_FLOOR:
        return MoistureRegime.DAMP_CAPILLARY
    return MoistureRegime.SATURATED


# --------------------------------------------------------------------------
# Regime models, kept separate on purpose
# --------------------------------------------------------------------------


def capillary_suction_pa(
    meniscus_radius_m: float,
    surface_tension_n_per_m: float = WATER_SURFACE_TENSION_N_PER_M,
) -> float:
    """Return the matric suction ``2 sigma / r`` of a meniscus, cavitation-capped.

    Raises:
        MoistureRegimeError: if the radius or surface tension is not positive.
    """
    radius = _require_finite(meniscus_radius_m, "meniscus radius")
    sigma = _require_finite(surface_tension_n_per_m, "surface tension")
    if radius <= 0.0:
        raise MoistureRegimeError(f"meniscus radius must be positive, got {radius!r} m")
    if sigma <= 0.0:
        raise MoistureRegimeError(
            f"surface tension must be positive, got {sigma!r} N/m"
        )
    return clamp_suction_pa(2.0 * sigma / radius)


def capillary_apparent_cohesion_pa(
    suction_pa: float,
    saturation: float,
    friction_angle_rad: float,
) -> float:
    """Apparent cohesion of damp sand from Bishop's effective stress.

    ``c_app = chi * s * tan(phi)`` with the effective-stress parameter ``chi``
    approximated by the degree of saturation. For USGA-window sand this lands
    in the 1-10 kPa band reported for damp sand.

    Raises:
        MoistureRegimeError: on non-finite or out-of-range arguments.
    """
    s = clamp_suction_pa(suction_pa)
    chi = _require_finite(saturation, "degree of saturation")
    phi_rad = _require_finite(friction_angle_rad, "friction angle")
    if not 0.0 <= chi <= 1.0:
        raise MoistureRegimeError(
            f"degree of saturation must lie in [0, 1], got {chi!r}"
        )
    if not 0.0 < phi_rad < math.pi / 2.0:
        raise MoistureRegimeError(
            f"friction angle must lie in (0, pi/2) rad, got {phi_rad!r}"
        )
    return chi * s * math.tan(phi_rad)


def cavitation_limited_strength_gain_pa(
    requested_suction_pa: float,
    friction_angle_rad: float,
) -> float:
    """Extra shear strength from shear-band dilation suction, cavitation-capped.

    A poroelastic estimate of dilation suction in a shear band can reach
    several megapascals. Water cannot deliver it: the pore pressure cavitates
    at about -100 kPa gauge, so the strength gain saturates near
    ``CAVITATION_SUCTION_LIMIT_PA * tan(phi)``, about 65 kPa for phi = 34 deg.

    Raises:
        MoistureRegimeError: on non-finite, negative or out-of-range arguments.
    """
    capped = clamp_suction_pa(requested_suction_pa)
    phi_rad = _require_finite(friction_angle_rad, "friction angle")
    if not 0.0 < phi_rad < math.pi / 2.0:
        raise MoistureRegimeError(
            f"friction angle must lie in (0, pi/2) rad, got {phi_rad!r}"
        )
    return capped * math.tan(phi_rad)


@dataclass(frozen=True, slots=True)
class MoistureState:
    """The moisture condition of a sand bed, with its regime stated explicitly.

    Attributes:
        gravimetric_water_content: Mass of water per mass of dry solids.
        degree_of_saturation: Fraction of the void volume filled with water.
        regime: The declared regime. Must agree with the saturation.
        meniscus_radius_m: Characteristic meniscus neck radius used by the
            capillary model.
        surface_tension_n_per_m: Air-water surface tension.
    """

    gravimetric_water_content: float
    degree_of_saturation: float
    regime: MoistureRegime
    meniscus_radius_m: float
    surface_tension_n_per_m: float = WATER_SURFACE_TENSION_N_PER_M

    def __post_init__(self) -> None:
        w = _require_finite(self.gravimetric_water_content, "water content")
        s = _require_finite(self.degree_of_saturation, "degree of saturation")
        radius = _require_finite(self.meniscus_radius_m, "meniscus radius")
        sigma = _require_finite(self.surface_tension_n_per_m, "surface tension")
        if w < 0.0:
            raise MoistureRegimeError(f"water content must not be negative, got {w!r}")
        if radius <= 0.0:
            raise MoistureRegimeError(
                f"meniscus radius must be positive, got {radius!r} m"
            )
        if sigma <= 0.0:
            raise MoistureRegimeError(
                f"surface tension must be positive, got {sigma!r} N/m"
            )
        implied = classify_regime(s)
        if implied is not self.regime:
            raise MoistureRegimeError(
                f"declared regime {self.regime.value!r} disagrees with a "
                f"degree of saturation of {s:.4f}, which is "
                f"{implied.value!r}. Regime selection is explicit: either fix "
                "the water content or declare the regime the saturation "
                "implies."
            )

    @classmethod
    def from_water_content(
        cls,
        gravimetric_water_content: float,
        void_ratio: float,
        particle_density_kg_m3: float,
        meniscus_radius_m: float,
        water_density_kg_m3: float = WATER_DENSITY_KG_M3,
        surface_tension_n_per_m: float = WATER_SURFACE_TENSION_N_PER_M,
    ) -> MoistureState:
        """Build a state, classifying the regime from the degree of saturation."""
        saturation = degree_of_saturation(
            gravimetric_water_content=gravimetric_water_content,
            void_ratio=void_ratio,
            particle_density_kg_m3=particle_density_kg_m3,
            water_density_kg_m3=water_density_kg_m3,
        )
        return cls(
            gravimetric_water_content=gravimetric_water_content,
            degree_of_saturation=saturation,
            regime=classify_regime(saturation),
            meniscus_radius_m=meniscus_radius_m,
            surface_tension_n_per_m=surface_tension_n_per_m,
        )

    @property
    def matric_suction_pa(self) -> float:
        """Capillary suction magnitude; zero outside the damp regime."""
        if self.regime is not MoistureRegime.DAMP_CAPILLARY:
            return 0.0
        return capillary_suction_pa(
            meniscus_radius_m=self.meniscus_radius_m,
            surface_tension_n_per_m=self.surface_tension_n_per_m,
        )

    def cohesive_strength_pa(
        self,
        friction_angle_rad: float,
        dilation_suction_pa: float | None = None,
    ) -> float:
        """Return the moisture contribution to shear strength, by regime.

        Args:
            friction_angle_rad: Internal friction angle.
            dilation_suction_pa: Poroelastic estimate of shear-band dilation
                suction. Required in the saturated regime, rejected otherwise.

        Returns:
            Dry: 0. Damp: capillary apparent cohesion. Saturated: the
            cavitation-capped dilation strength gain.

        Raises:
            MoistureRegimeError: if ``dilation_suction_pa`` is supplied in the
                wrong regime, or omitted in the saturated regime.
        """
        if self.regime is MoistureRegime.SATURATED:
            if dilation_suction_pa is None:
                raise MoistureRegimeError(
                    "the saturated regime needs an explicit dilation suction: "
                    "pass dilation_suction_pa (it is cavitation-capped at "
                    f"{CAVITATION_SUCTION_LIMIT_PA:.0f} Pa). There is no "
                    "default, because a silent zero and a silent multi-MPa "
                    "value are both wrong."
                )
            return cavitation_limited_strength_gain_pa(
                requested_suction_pa=dilation_suction_pa,
                friction_angle_rad=friction_angle_rad,
            )
        if dilation_suction_pa is not None:
            raise MoistureRegimeError(
                "dilation_suction_pa applies only to the saturated regime; "
                f"this state is {self.regime.value!r}"
            )
        if self.regime is MoistureRegime.DRY:
            return 0.0
        return capillary_apparent_cohesion_pa(
            suction_pa=self.matric_suction_pa,
            saturation=self.degree_of_saturation,
            friction_angle_rad=friction_angle_rad,
        )
