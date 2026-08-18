"""Compaction state of a sand bed (issue #8610).

Void ratio ``e``, solid fraction ``phi``, dry bulk density ``rho_d`` and
relative density ``Dr`` are one quantity seen four ways::

    phi    = 1 / (1 + e)
    e      = (1 - phi) / phi
    rho_d  = rho_s * phi
    Dr     = (e_max - e) / (e_max - e_min)

All densities are kg/m^3. The default packing limits are the classical
random-close and random-loose packing fractions for equal spheres; they are a
documented convention, not a measurement on bunker sand (see
:mod:`bunkershot3d.sand.provenance`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .exceptions import PackingStateError

__all__ = [
    "RANDOM_CLOSE_PACKING_SOLID_FRACTION",
    "RANDOM_LOOSE_PACKING_SOLID_FRACTION",
    "SAND_VOID_RATIO_MAX",
    "SAND_VOID_RATIO_MIN",
    "Angularity",
    "PackingState",
    "solid_fraction_from_void_ratio",
    "void_ratio_from_solid_fraction",
]

RANDOM_CLOSE_PACKING_SOLID_FRACTION = 0.64
"""Densest reproducible random packing of equal spheres."""

RANDOM_LOOSE_PACKING_SOLID_FRACTION = 0.55
"""Loosest packing that still supports its own weight."""

_RANGE_TOLERANCE = 1e-9


def _require_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise PackingStateError(f"{name} must be finite, got {value!r}")
    return float(value)


def void_ratio_from_solid_fraction(solid_fraction: float) -> float:
    """Return the void ratio ``e = (1 - phi) / phi``.

    Raises:
        PackingStateError: if ``solid_fraction`` is not in the open interval
            (0, 1).
    """
    phi = _require_finite(solid_fraction, "solid fraction")
    if not 0.0 < phi < 1.0:
        raise PackingStateError(
            f"solid fraction must lie strictly between 0 and 1, got {phi!r}"
        )
    return (1.0 - phi) / phi


def solid_fraction_from_void_ratio(void_ratio: float) -> float:
    """Return the solid fraction ``phi = 1 / (1 + e)``.

    Raises:
        PackingStateError: if ``void_ratio`` is negative.
    """
    e = _require_finite(void_ratio, "void ratio")
    if e < 0.0:
        raise PackingStateError(f"void ratio must not be negative, got {e!r}")
    return 1.0 / (1.0 + e)


SAND_VOID_RATIO_MIN = void_ratio_from_solid_fraction(
    RANDOM_CLOSE_PACKING_SOLID_FRACTION
)
"""Void ratio at random close packing (the densest state, Dr = 1)."""

SAND_VOID_RATIO_MAX = void_ratio_from_solid_fraction(
    RANDOM_LOOSE_PACKING_SOLID_FRACTION
)
"""Void ratio at random loose packing (the loosest state, Dr = 0)."""


class Angularity(Enum):
    """Grain shape class.

    USGA guidance is that angular sand interlocks, resists burial and does not
    crust, and is therefore desirable; rounded sand is not.
    """

    ROUNDED = "rounded"
    SUBROUNDED = "subrounded"
    SUBANGULAR = "subangular"
    ANGULAR = "angular"
    VERY_ANGULAR = "very_angular"

    @property
    def shape_index(self) -> int:
        """Ordinal from 0 (rounded) to 4 (very angular)."""
        return _ANGULARITY_ORDER[self]

    @property
    def is_usga_desirable(self) -> bool:
        """True for shapes the USGA rates desirable for bunker sand."""
        return self.shape_index >= _ANGULARITY_ORDER[Angularity.ANGULAR]


_ANGULARITY_ORDER: dict[Angularity, int] = {
    Angularity.ROUNDED: 0,
    Angularity.SUBROUNDED: 1,
    Angularity.SUBANGULAR: 2,
    Angularity.ANGULAR: 3,
    Angularity.VERY_ANGULAR: 4,
}


@dataclass(frozen=True, slots=True)
class PackingState:
    """The compaction state of a sand bed.

    Attributes:
        particle_density_kg_m3: Density of the mineral grains themselves.
        void_ratio: Volume of voids per unit volume of solids.
        void_ratio_min: Void ratio in the densest achievable state.
        void_ratio_max: Void ratio in the loosest achievable state.
    """

    particle_density_kg_m3: float
    void_ratio: float
    void_ratio_min: float = SAND_VOID_RATIO_MIN
    void_ratio_max: float = SAND_VOID_RATIO_MAX

    def __post_init__(self) -> None:
        rho_s = _require_finite(self.particle_density_kg_m3, "particle density")
        e = _require_finite(self.void_ratio, "void ratio")
        e_min = _require_finite(self.void_ratio_min, "void_ratio_min")
        e_max = _require_finite(self.void_ratio_max, "void_ratio_max")
        if rho_s <= 0.0:
            raise PackingStateError(
                f"particle density must be positive, got {rho_s!r} kg/m^3"
            )
        if e <= 0.0:
            raise PackingStateError(f"void ratio must be positive, got {e!r}")
        if e_min <= 0.0:
            raise PackingStateError(f"void_ratio_min must be positive, got {e_min!r}")
        if e_min >= e_max:
            raise PackingStateError(
                f"void_ratio_min ({e_min!r}) must be smaller than "
                f"void_ratio_max ({e_max!r})"
            )
        if not e_min - _RANGE_TOLERANCE <= e <= e_max + _RANGE_TOLERANCE:
            raise PackingStateError(
                f"void ratio {e!r} is outside the achievable range "
                f"[{e_min!r}, {e_max!r}]; relative density would be "
                f"{(e_max - e) / (e_max - e_min):.3f}, which is not a physical "
                "packing state"
            )

    # ----------------------------------------------------------- factories

    @classmethod
    def from_solid_fraction(
        cls,
        particle_density_kg_m3: float,
        solid_fraction: float,
        void_ratio_min: float = SAND_VOID_RATIO_MIN,
        void_ratio_max: float = SAND_VOID_RATIO_MAX,
    ) -> PackingState:
        """Build from a solid volume fraction."""
        return cls(
            particle_density_kg_m3=particle_density_kg_m3,
            void_ratio=void_ratio_from_solid_fraction(solid_fraction),
            void_ratio_min=void_ratio_min,
            void_ratio_max=void_ratio_max,
        )

    @classmethod
    def from_dry_bulk_density(
        cls,
        particle_density_kg_m3: float,
        dry_bulk_density_kg_m3: float,
        void_ratio_min: float = SAND_VOID_RATIO_MIN,
        void_ratio_max: float = SAND_VOID_RATIO_MAX,
    ) -> PackingState:
        """Build from a measured dry bulk density."""
        rho_s = _require_finite(particle_density_kg_m3, "particle density")
        rho_d = _require_finite(dry_bulk_density_kg_m3, "dry bulk density")
        if rho_s <= 0.0:
            raise PackingStateError(
                f"particle density must be positive, got {rho_s!r} kg/m^3"
            )
        return cls.from_solid_fraction(
            particle_density_kg_m3=rho_s,
            solid_fraction=rho_d / rho_s,
            void_ratio_min=void_ratio_min,
            void_ratio_max=void_ratio_max,
        )

    @classmethod
    def from_relative_density(
        cls,
        particle_density_kg_m3: float,
        relative_density: float,
        void_ratio_min: float = SAND_VOID_RATIO_MIN,
        void_ratio_max: float = SAND_VOID_RATIO_MAX,
    ) -> PackingState:
        """Build from a relative density (0 = loosest, 1 = densest).

        Raises:
            PackingStateError: if ``relative_density`` is outside [0, 1].
        """
        dr = _require_finite(relative_density, "relative density")
        if not 0.0 <= dr <= 1.0:
            raise PackingStateError(f"relative density must lie in [0, 1], got {dr!r}")
        e = void_ratio_max - dr * (void_ratio_max - void_ratio_min)
        return cls(
            particle_density_kg_m3=particle_density_kg_m3,
            void_ratio=e,
            void_ratio_min=void_ratio_min,
            void_ratio_max=void_ratio_max,
        )

    # ------------------------------------------------------------ derived

    @property
    def solid_fraction(self) -> float:
        """Volume fraction occupied by solid grains."""
        return solid_fraction_from_void_ratio(self.void_ratio)

    @property
    def porosity(self) -> float:
        """Volume fraction occupied by voids."""
        return 1.0 - self.solid_fraction

    @property
    def dry_bulk_density_kg_m3(self) -> float:
        """Mass of dry solids per unit bulk volume."""
        return self.particle_density_kg_m3 * self.solid_fraction

    @property
    def relative_density(self) -> float:
        """Dr in [0, 1]: 0 at the loosest packing, 1 at the densest."""
        span = self.void_ratio_max - self.void_ratio_min
        dr = (self.void_ratio_max - self.void_ratio) / span
        return min(1.0, max(0.0, dr))

    @property
    def relative_density_percent(self) -> float:
        """Relative density expressed as a percentage."""
        return 100.0 * self.relative_density

    @property
    def solid_fraction_min(self) -> float:
        """Solid fraction at the loosest packing."""
        return solid_fraction_from_void_ratio(self.void_ratio_max)

    @property
    def solid_fraction_max(self) -> float:
        """Solid fraction at the densest packing."""
        return solid_fraction_from_void_ratio(self.void_ratio_min)
