"""The :class:`SandState` value object (issue #8610).

A narrow domain object, per ADR-0032 structural decision 1: it is passed
directly to the code that needs it instead of being reached through a root
configuration. It is frozen, so a solver cannot mutate the sand under itself.

Everything is SI internally, with unit-suffixed names. The penetrometer
firmness is stored in pascals and exposed in the published kg/cm^2 unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .bed import BunkerBedGeometry
from .exceptions import SandModelError
from .feasibility import (
    BedFeasibilityReport,
    evaluate_bed_feasibility,
    require_feasible_bed,
    required_grain_count,
)
from .firmness import (
    FirmnessRating,
    firmness_kg_per_cm2_from_pa,
    firmness_rating,
)
from .moisture import MoistureRegime, MoistureState
from .packing import Angularity, PackingState
from .provenance import SandProvenance
from .psd import ParticleSizeDistribution

__all__ = ["SandState"]


@dataclass(frozen=True, slots=True)
class SandState:
    """The complete state of a bunker sand bed.

    Attributes:
        name: Preset or run label.
        psd: Sieve analysis.
        packing: Compaction state.
        moisture: Water content and its regime.
        bed: Bed geometry the sand occupies.
        angularity: Grain shape class.
        friction_angle_deg: Internal friction angle. Borrowed from the
            Quikrete analogue; see :attr:`provenance`.
        penetrometer_firmness_pa: Firmness reading, SI.
        provenance: Where every honesty-critical value came from.
    """

    name: str
    psd: ParticleSizeDistribution
    packing: PackingState
    moisture: MoistureState
    bed: BunkerBedGeometry
    angularity: Angularity
    friction_angle_deg: float
    penetrometer_firmness_pa: float
    provenance: SandProvenance

    def __post_init__(self) -> None:
        if not math.isfinite(self.friction_angle_deg):
            raise SandModelError(
                f"friction angle must be finite, got {self.friction_angle_deg!r}"
            )
        if not 0.0 < self.friction_angle_deg < 90.0:
            raise SandModelError(
                "friction angle must lie strictly between 0 and 90 deg, got "
                f"{self.friction_angle_deg!r} deg"
            )
        if (
            not math.isfinite(self.penetrometer_firmness_pa)
            or self.penetrometer_firmness_pa <= 0.0
        ):
            raise SandModelError(
                "penetrometer firmness must be a positive finite pressure, got "
                f"{self.penetrometer_firmness_pa!r} Pa"
            )
        self.provenance.require_keys()

    # ------------------------------------------------------- delegated view

    @property
    def friction_angle_rad(self) -> float:
        """Internal friction angle in radians."""
        return math.radians(self.friction_angle_deg)

    @property
    def void_ratio(self) -> float:
        """Void ratio of the bed."""
        return self.packing.void_ratio

    @property
    def solid_fraction(self) -> float:
        """Solid volume fraction of the bed."""
        return self.packing.solid_fraction

    @property
    def relative_density(self) -> float:
        """Relative density in [0, 1]."""
        return self.packing.relative_density

    @property
    def dry_bulk_density_kg_m3(self) -> float:
        """Dry bulk density of the bed."""
        return self.packing.dry_bulk_density_kg_m3

    @property
    def bulk_density_kg_m3(self) -> float:
        """Moist bulk density, including the pore water."""
        return self.dry_bulk_density_kg_m3 * (
            1.0 + self.moisture.gravimetric_water_content
        )

    @property
    def regime(self) -> MoistureRegime:
        """The declared moisture regime."""
        return self.moisture.regime

    @property
    def d50_m(self) -> float:
        """Median grain diameter."""
        return self.psd.d50_m

    @property
    def uniformity_coefficient(self) -> float:
        """Cu = d60 / d10."""
        return self.psd.uniformity_coefficient

    @property
    def firmness_kg_per_cm2(self) -> float:
        """Penetrometer firmness in the published unit."""
        return firmness_kg_per_cm2_from_pa(self.penetrometer_firmness_pa)

    @property
    def firmness_rating(self) -> FirmnessRating:
        """USGA rating band for this firmness."""
        return firmness_rating(self.firmness_kg_per_cm2)

    # ------------------------------------------------------------ physics

    def cohesive_strength_pa(self, dilation_suction_pa: float | None = None) -> float:
        """Return the moisture contribution to shear strength.

        Dispatches on the declared regime; see
        :meth:`bunkershot3d.sand.moisture.MoistureState.cohesive_strength_pa`.
        """
        return self.moisture.cohesive_strength_pa(
            friction_angle_rad=self.friction_angle_rad,
            dilation_suction_pa=dilation_suction_pa,
        )

    # -------------------------------------------------------- feasibility

    def required_grain_count(self, grain_diameter_m: float | None = None) -> int:
        """Return how many grains it takes to fill this bed.

        Args:
            grain_diameter_m: Monodisperse grain diameter. When omitted, the
                gradation's own number-preserving diameter is used, which
                answers "given this bed depth, this PSD and this packing
                fraction, how many grains is that?" honestly -- the answer is
                dominated by the few percent of silt and is far larger than a
                d50-based estimate.
        """
        diameter_m = (
            self.psd.volume_equivalent_diameter_m
            if grain_diameter_m is None
            else grain_diameter_m
        )
        return required_grain_count(
            bulk_volume_m3=self.bed.bulk_volume_m3,
            solid_fraction=self.solid_fraction,
            grain_diameter_m=diameter_m,
        )

    def bed_feasibility(
        self,
        grain_count: int,
        grain_diameter_m: float,
        depth_tolerance: float | None = None,
        max_grain_count: int | None = None,
    ) -> BedFeasibilityReport:
        """Report whether a grain population can fill this bed."""
        kwargs = {} if depth_tolerance is None else {"depth_tolerance": depth_tolerance}
        return evaluate_bed_feasibility(
            bed=self.bed,
            grain_count=grain_count,
            grain_diameter_m=grain_diameter_m,
            target_solid_fraction=self.solid_fraction,
            max_grain_count=max_grain_count,
            **kwargs,
        )

    def require_feasible_bed(
        self,
        grain_count: int,
        grain_diameter_m: float,
        depth_tolerance: float | None = None,
        max_grain_count: int | None = None,
    ) -> BedFeasibilityReport:
        """Refuse a grain population that cannot physically fill this bed.

        Raises:
            InfeasibleBedError: with an actionable message.
        """
        kwargs = {} if depth_tolerance is None else {"depth_tolerance": depth_tolerance}
        return require_feasible_bed(
            bed=self.bed,
            grain_count=grain_count,
            grain_diameter_m=grain_diameter_m,
            target_solid_fraction=self.solid_fraction,
            max_grain_count=max_grain_count,
            **kwargs,
        )

    # ------------------------------------------------------------ derived

    def with_bed(self, bed: BunkerBedGeometry) -> SandState:
        """Return a copy of this state placed on a different bed."""
        return replace(self, bed=bed)
