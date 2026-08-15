"""Discrete grain population and contact material (issue #8608).

These describe a *DEM discretisation*, not the sand. What the sand is -- its
sieve analysis, compaction, moisture regime and provenance -- is
:class:`~bunkershot3d.sand.state.SandState` (#8610). What a solver is asked to
instantiate as spheres, and with what contact law, is here. ADR-0032 records
that resolved DEM cannot reach the real scale anyway (2.1e8 grains), so this
pair exists to describe the F3 backends honestly, not to describe a bunker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..exceptions import DomainInvariantError
from ._validate import (
    require_non_negative,
    require_open_range,
    require_positive,
    require_positive_int,
    require_unit_interval,
)

__all__ = ["ContactMaterial", "GrainPopulation"]

#: Incompressibility limit. At nu = 0.5 the bulk modulus diverges and the
#: Hertzian effective modulus E* = E / (2 (1 - nu^2)) loses meaning.
_POISSON_LIMIT = 0.5


@dataclass(frozen=True, slots=True)
class GrainPopulation:
    """A log-normal population of spherical grains for a DEM backend.

    Attributes:
        count: Grains the configuration asks for, before coarse-graining.
        diameter_mean_m: Mean grain diameter.
        diameter_sigma_log: Sigma of the diameter distribution *in log space*.
            Zero is a monodisperse population and is admissible.
        density_kg_m3: Grain material density.
        coarse_graining_factor: How many real grains one simulated grain
            stands for. One means no coarse-graining; it can never be less.
    """

    count: int
    diameter_mean_m: float
    diameter_sigma_log: float
    density_kg_m3: float
    coarse_graining_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            DomainInvariantError: A field is outside its admissible range.
        """
        object.__setattr__(self, "count", require_positive_int(self.count, "count"))
        object.__setattr__(
            self,
            "diameter_mean_m",
            require_positive(self.diameter_mean_m, "diameter_mean_m"),
        )
        object.__setattr__(
            self,
            "diameter_sigma_log",
            require_non_negative(self.diameter_sigma_log, "diameter_sigma_log"),
        )
        object.__setattr__(
            self,
            "density_kg_m3",
            require_positive(self.density_kg_m3, "density_kg_m3"),
        )
        factor = require_positive(self.coarse_graining_factor, "coarse_graining_factor")
        if factor < 1.0:
            raise DomainInvariantError(
                "coarse_graining_factor must be at least 1 (one simulated grain "
                f"cannot stand for less than one real grain), got {factor!r}"
            )
        object.__setattr__(self, "coarse_graining_factor", factor)

    @property
    def radius_mean_m(self) -> float:
        """Half the mean diameter."""
        return 0.5 * self.diameter_mean_m

    @property
    def is_monodisperse(self) -> bool:
        """True when every grain has the same diameter."""
        return self.diameter_sigma_log == 0.0

    @property
    def effective_count(self) -> int:
        """Grains actually instantiated after coarse-graining, at least one."""
        return max(1, int(self.count / self.coarse_graining_factor))

    @property
    def mean_grain_mass_kg(self) -> float:
        """Mass of a sphere of the mean diameter, before coarse-graining."""
        return self.density_kg_m3 * (math.pi / 6.0) * self.diameter_mean_m**3


@dataclass(frozen=True, slots=True)
class ContactMaterial:
    """Hertz-Mindlin contact parameters shared by walls, grains and clubhead.

    Attributes:
        friction: Coulomb friction coefficient, dimensionless.
        restitution: Coefficient of restitution, dimensionless.
        youngs_modulus_pa: Young's modulus.
        poisson_ratio: Poisson's ratio, strictly inside ``(0, 0.5)``.
    """

    friction: float
    restitution: float
    youngs_modulus_pa: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            DomainInvariantError: A coefficient is outside its admissible range.
        """
        object.__setattr__(
            self, "friction", require_unit_interval(self.friction, "friction")
        )
        object.__setattr__(
            self, "restitution", require_unit_interval(self.restitution, "restitution")
        )
        object.__setattr__(
            self,
            "youngs_modulus_pa",
            require_positive(self.youngs_modulus_pa, "youngs_modulus_pa"),
        )
        object.__setattr__(
            self,
            "poisson_ratio",
            require_open_range(
                self.poisson_ratio, "poisson_ratio", 0.0, _POISSON_LIMIT
            ),
        )

    @property
    def shear_modulus_pa(self) -> float:
        """Isotropic shear modulus ``G = E / (2 (1 + nu))``."""
        return self.youngs_modulus_pa / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def effective_modulus_pa(self) -> float:
        """Hertzian effective modulus ``E* = E / (2 (1 - nu^2))``.

        The quantity that actually sets contact stiffness, and therefore the
        grain interpenetration at impact speed (defect B31).
        """
        return self.youngs_modulus_pa / (2.0 * (1.0 - self.poisson_ratio**2))
