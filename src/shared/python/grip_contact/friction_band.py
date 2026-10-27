"""Cited grip friction band and vector combined-load bound (#13890).

Replaces placeholder ContactMaterial friction with a cited friction band
per moisture regime (dry, moist, film) for reporting and sweeps, and provides
the vector combined-load bound helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

VALID_REGIMES: Final[tuple[str, ...]] = ("dry", "moist", "film")


@dataclass(frozen=True)
class GripFrictionBand:
    """Cited friction coefficient band for a contact regime.

    Preconditions:
        0 < mu_low <= mu_high, finite numbers.
        regime in ('dry', 'moist', 'film').
        source must carry a DOI/URL or 'provisional'.
    """

    regime: str
    mu_low: float
    mu_high: float
    source: str

    def __post_init__(self) -> None:
        if self.regime not in VALID_REGIMES:
            raise ValueError(
                f"regime must be one of {VALID_REGIMES}, got {self.regime!r}"
            )
        for name in ("mu_low", "mu_high"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        if self.mu_low <= 0.0:
            raise ValueError(f"mu_low must be positive, got {self.mu_low}")
        if self.mu_high < self.mu_low:
            raise ValueError(
                f"mu_high must be >= mu_low, got {self.mu_high} < {self.mu_low}"
            )
        src = self.source.lower()
        if not ("doi:" in src or "http" in src or "provisional" in src):
            raise ValueError(
                "source must carry a citation (DOI or URL) or the tag 'provisional'"
            )


GRIP_FRICTION_BANDS: Final[dict[str, GripFrictionBand]] = {
    "dry": GripFrictionBand(
        regime="dry",
        mu_low=0.5,
        mu_high=0.7,
        source=(
            "Derler & Gerhardt 2012, Tribol Lett 45(1):1–27, doi:10.1007/s11249-011-9854-y; "
            "Seo & Armstrong 2009, Ergonomics 52(5):609–616, doi:10.1080/00140130802471595"
        ),
    ),
    "moist": GripFrictionBand(
        regime="moist",
        mu_low=1.0,
        mu_high=1.4,
        source=(
            "Derler & Gerhardt 2012, Tribol Lett 45(1):1–27, doi:10.1007/s11249-011-9854-y; "
            "Seo & Armstrong 2009, Ergonomics 52(5):609–616, doi:10.1080/00140130802471595 (Table 1)"
        ),
    ),
    "film": GripFrictionBand(
        regime="film",
        mu_low=0.2,
        mu_high=0.4,
        source=(
            "provisional; Lewis, Carre & Tomlinson 2014, Procedia Eng 72:611–617, "
            "doi:10.1016/j.proeng.2014.06.064"
        ),
    ),
}


def get_friction_band(regime: str) -> GripFrictionBand:
    """Retrieve the canonical GripFrictionBand for the given regime."""
    if regime not in GRIP_FRICTION_BANDS:
        raise ValueError(
            f"Unknown regime {regime!r}; must be one of {tuple(GRIP_FRICTION_BANDS.keys())}"
        )
    return GRIP_FRICTION_BANDS[regime]


def combined_squeeze_n(
    peak_axial_force_n: float,
    peak_axial_torque_nm: float,
    peak_transverse_force_n: float,
    friction: float,
    grip_radius_m: float,
) -> float:
    """Vector combined-load reporting bound for one hand's squeeze demand.

    Assumes uniform pad load sharing on a cylindrical grip surface where axial
    and torque demands share one isotropic Coulomb cone:
        N >= sqrt(F_ax^2 + (tau_ax / r)^2) / mu
        N >= 2 * F_perp (unilateral contact condition)

    Args:
        peak_axial_force_n: Axial load along grip shaft (N), non-negative.
        peak_axial_torque_nm: Torque about grip axis (N·m), non-negative.
        peak_transverse_force_n: Transverse force across grip (N), non-negative.
        friction: Effective friction coefficient mu > 0.
        grip_radius_m: Grip radius r > 0 (m).

    Returns:
        Required squeeze normal force (N).
    """
    demands = (peak_axial_force_n, peak_axial_torque_nm, peak_transverse_force_n)
    if min(demands) < 0.0 or not all(math.isfinite(d) for d in demands):
        raise ValueError("demands must be finite and non-negative")
    if friction <= 0.0 or not math.isfinite(friction):
        raise ValueError("friction must be positive and finite")
    if grip_radius_m <= 0.0 or not math.isfinite(grip_radius_m):
        raise ValueError("grip_radius_m must be positive and finite")

    tangential_demand = math.sqrt(
        peak_axial_force_n**2 + (peak_axial_torque_nm / grip_radius_m) ** 2
    )
    return max(tangential_demand / friction, 2.0 * peak_transverse_force_n)


def governing_branch(
    peak_axial_force_n: float,
    peak_axial_torque_nm: float,
    peak_transverse_force_n: float,
    friction: float,
    grip_radius_m: float,
) -> Literal["axial", "torque", "transverse", "combined"]:
    """Identify the governing branch for the combined squeeze demand."""
    demands = (peak_axial_force_n, peak_axial_torque_nm, peak_transverse_force_n)
    if min(demands) < 0.0 or not all(math.isfinite(d) for d in demands):
        raise ValueError("demands must be finite and non-negative")
    if friction <= 0.0 or grip_radius_m <= 0.0:
        raise ValueError("friction and grip_radius_m must be positive")

    axial_demand = peak_axial_force_n / friction
    torque_demand = peak_axial_torque_nm / (friction * grip_radius_m)
    transverse_demand = 2.0 * peak_transverse_force_n
    tangential_demand = (
        math.sqrt(peak_axial_force_n**2 + (peak_axial_torque_nm / grip_radius_m) ** 2)
        / friction
    )

    if transverse_demand > tangential_demand:
        return "transverse"
    if torque_demand == 0.0 and axial_demand > 0.0:
        return "axial"
    if axial_demand == 0.0 and torque_demand > 0.0:
        return "torque"
    if axial_demand > 0.0 and torque_demand > 0.0:
        return "combined"
    return "axial"


def unilateral_friction_capacity(normal_force_n: float, friction: float) -> float:
    """Return friction capacity with unilateral contact: capacity is 0 for f_n <= 0."""
    if not math.isfinite(normal_force_n):
        raise ValueError("normal_force_n must be finite")
    if friction < 0.0 or not math.isfinite(friction):
        raise ValueError("friction must be non-negative and finite")
    return friction * max(0.0, float(normal_force_n))
