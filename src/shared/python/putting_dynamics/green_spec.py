"""One description of a green's shape for every putting surface (#14463).

A :class:`GreenSpec` composes the core's own fields: a planar grade
(:meth:`~.surfaces.HeightField.planar`), an optional two-tier step, and
seeded undulation (:func:`~.surfaces.bumpy_height_field`). Routes, the
desktop tile and the studio describe a green with it, so every surface
rolls the same green (ADR-0058).

The named presets keep the legacy engine's preset names
(``GreenSurface.create_preset`` in
``src/engines/physics_engines/putting_green/python/_surface_presets.py``)
and approximate their shapes with these primitives; the two-tier green is
a smooth step rather than that engine's ridge. Turf speed stays separate
(:class:`~.turf.TurfSpec`): a spec is geometry only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require, require_finite

from .friction import FrictionParams, stimp_to_rolling_mu
from .surfaces import FrictionField, HeightField, SurfaceSpec, bumpy_height_field

#: Documented ranges, checked on construction.
GRADE_RANGE_PERCENT = (0.0, 10.0)
BUMP_HEIGHT_RANGE_M = (0.0, 0.05)
TIER_HEIGHT_RANGE_M = (0.0, 0.15)


@dataclass(frozen=True)
class GreenSpec:
    """Shape of a green, centred on the origin.

    Attributes:
        grade_percent: Planar grade [%].
        aspect_deg: Direction the green falls toward, CCW from +x [deg].
        bump_height_m: Standard deviation of the undulation [m].
        bump_length_m: Undulation correlation length [m].
        tier_height_m: Rise of the upper tier [m]; it lies uphill, across a
            step through the origin perpendicular to ``aspect_deg``.
        tier_width_m: Width of the step between tiers [m].
        seed: Undulation seed; the same spec always builds the same green.
    """

    grade_percent: float = 0.0
    aspect_deg: float = 0.0
    bump_height_m: float = 0.0
    bump_length_m: float = 1.5
    tier_height_m: float = 0.0
    tier_width_m: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name, value, (lo, hi) in (
            ("grade_percent", self.grade_percent, GRADE_RANGE_PERCENT),
            ("bump_height_m", self.bump_height_m, BUMP_HEIGHT_RANGE_M),
            ("tier_height_m", self.tier_height_m, TIER_HEIGHT_RANGE_M),
        ):
            require_finite(value, name)
            require(lo <= value <= hi, f"{name} in [{lo}, {hi}]", value)
        require_finite(self.aspect_deg, "aspect_deg")
        require(-360.0 <= self.aspect_deg <= 360.0, "aspect in [-360, 360]")
        require_finite(self.bump_length_m, "bump_length_m")
        require(0.0 < self.bump_length_m <= 20.0, "bump length in (0, 20] m")
        require_finite(self.tier_width_m, "tier_width_m")
        require(0.0 < self.tier_width_m <= 5.0, "tier width in (0, 5] m")
        require(
            isinstance(self.seed, int) and not isinstance(self.seed, bool),
            "seed must be an int",
        )
        require(self.seed >= 0, "seed must be >= 0", self.seed)

    def height(self, extent_m: float = 20.0) -> HeightField:
        """The elevation field over a square of side ``extent_m``."""
        field = HeightField.planar(self.grade_percent, self.aspect_deg, extent_m)
        if self.tier_height_m > 0.0:
            field = self._with_tier(field)
        return bumpy_height_field(
            seed=self.seed,
            amplitude_m=self.bump_height_m,
            correlation_length_m=self.bump_length_m,
            base=field,
        )

    def surface(self, stimp_ft: float = 10.0, extent_m: float = 20.0) -> SurfaceSpec:
        """This green at ``stimp_ft`` with uniform friction."""
        return SurfaceSpec(
            height=self.height(extent_m),
            friction_field=FrictionField.uniform(extent_m=extent_m),
            friction=FrictionParams(mu_roll0=stimp_to_rolling_mu(stimp_ft)),
        )

    def _with_tier(self, field: HeightField) -> HeightField:
        ny, nx = field.heights_m.shape
        x0, y0 = field.origin_m
        xs = x0 + np.arange(nx) * field.spacing_m
        ys = y0 + np.arange(ny) * field.spacing_m
        aspect = math.radians(self.aspect_deg)
        downhill = math.cos(aspect) * xs[None, :] + math.sin(aspect) * ys[:, None]
        step = 0.5 * self.tier_height_m * (1.0 - np.tanh(downhill / self.tier_width_m))
        return HeightField(
            heights_m=field.heights_m + step,
            spacing_m=field.spacing_m,
            origin_m=field.origin_m,
        )


GREEN_PRESETS: dict[str, GreenSpec] = {
    "flat_practice": GreenSpec(),
    "undulating_championship": GreenSpec(
        grade_percent=1.5, aspect_deg=26.6, bump_height_m=0.01, bump_length_m=3.0
    ),
    "severe_slopes": GreenSpec(
        grade_percent=5.0, tier_height_m=0.05, tier_width_m=2.0, aspect_deg=0.0
    ),
    "tiered": GreenSpec(aspect_deg=90.0, tier_height_m=0.08, tier_width_m=3.0),
}


def green_preset(name: str) -> GreenSpec:
    """The preset named ``name``; raises ``ValueError`` for an unknown one."""
    require(name in GREEN_PRESETS, f"unknown green preset {name!r}", name)
    return GREEN_PRESETS[name]
