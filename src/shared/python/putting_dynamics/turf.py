"""Turf model for putting_dynamics (epic #14141, putting v4 step 4b).

Migration source: ``src/engines/physics_engines/putting_green/python/
turf_properties.py``, which now imports :class:`GrassType` and
:class:`TurfCondition` from here so the tables have one home.

**One law.** ADR-0045 keeps ``ud-legacy-roll/1`` in the engine; the
shared core uses ``usga-stimp-roll/1`` only (Grober 2011 / USGA fixture).

Every table value below is an **uncited engineering default**. Grain is
stored as an angle: the engine's ``1 - g cos(theta_v - theta_grain)``
with ``theta_grain = atan2(grain_y, grain_x)`` is exactly
:func:`~.friction.grain_factor`.

Not ported (no shared-core hook; see step 5): the height-of-cut
multiplier (fitted to the legacy law), the compaction speed factor and
the cross-grain lateral curve. ``compaction_factor`` is metadata only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.shared.python.contracts import ensure, require, require_finite

from .friction import (
    DEFAULT_STATIC_MU,
    FrictionParams,
    require_grain,
    stimp_to_rolling_mu,
)
from .surfaces import SurfaceSpec

__all__ = [
    "GrassType",
    "TurfCondition",
    "TurfSpec",
]

#: Stimp bounds [ft]. Below ~3.92 ft the FROSTED multiplier (1.4) would push
#: ``mu_roll0`` past ``FrictionParams``' 0.2 ceiling.
_MIN_STIMP_FT = 4.0
_MAX_STIMP_FT = 16.0

#: Keeps static friction above kinetic on slow or frosted greens, where the
#: flat ``DEFAULT_STATIC_MU`` (0.08) would fall below ``mu_roll0``.
_STATIC_MU_MARGIN = 1.1


class GrassType(Enum):
    """Grass type on the green (migration source: ``turf_properties.py``)."""

    BENT_GRASS = "bent_grass"
    BERMUDA = "bermuda"
    POA_ANNUA = "poa_annua"
    FESCUE = "fescue"
    RYE_GRASS = "rye_grass"
    PASPALUM = "paspalum"
    ZOYSIA = "zoysia"

    @property
    def default_grain_strength(self) -> float:
        """Default grain-anisotropy amplitude (uncited engineering default)."""
        # Bermuda grows a visibly directional stolon grain; the rest are
        # ordered by common agronomic reputation, not a measured source.
        strengths = {
            GrassType.BENT_GRASS: 0.1,
            GrassType.BERMUDA: 0.4,
            GrassType.POA_ANNUA: 0.15,
            GrassType.FESCUE: 0.2,
            GrassType.RYE_GRASS: 0.15,
            GrassType.PASPALUM: 0.35,
            GrassType.ZOYSIA: 0.25,
        }
        return strengths.get(self, 0.15)

    @property
    def default_height_mm(self) -> float:
        """Default mowing height [mm] (uncited engineering default; informational
        only — :meth:`TurfSpec.to_friction_params` does not apply it)."""
        heights = {
            GrassType.BENT_GRASS: 3.0,
            GrassType.BERMUDA: 3.5,
            GrassType.POA_ANNUA: 3.2,
            GrassType.FESCUE: 4.0,
            GrassType.RYE_GRASS: 3.5,
            GrassType.PASPALUM: 3.5,
            GrassType.ZOYSIA: 3.8,
        }
        return heights.get(self, 3.5)


class TurfCondition(Enum):
    """Environmental condition (migration source: ``turf_properties.py``)."""

    DRY = "dry"
    NORMAL = "normal"
    WET = "wet"
    DEWY = "dewy"
    FROSTED = "frosted"
    RECENTLY_MOWED = "recently_mowed"
    AFTERNOON_GROWTH = "afternoon_growth"

    @property
    def friction_multiplier(self) -> float:
        """Multiplier on ``mu_roll0`` (uncited engineering default)."""
        # Ordered by common agronomic reputation (dry/mowed roll faster,
        # wet/dewy/frosted roll slower); no external fixture pins the
        # magnitudes.
        multipliers = {
            TurfCondition.DRY: 0.9,
            TurfCondition.NORMAL: 1.0,
            TurfCondition.WET: 1.3,
            TurfCondition.DEWY: 1.15,
            TurfCondition.FROSTED: 1.4,
            TurfCondition.RECENTLY_MOWED: 0.95,
            TurfCondition.AFTERNOON_GROWTH: 1.05,
        }
        return multipliers.get(self, 1.0)


@dataclass(frozen=True)
class TurfSpec:
    """The UD turf model (provisional), rendered through the USGA law only.

    Attributes:
        stimp_ft: Stimpmeter reading [feet], in
            ``[_MIN_STIMP_FT, _MAX_STIMP_FT]``.
        grass_type: Grass type; supplies the ``grain_strength`` default.
        condition: Its ``friction_multiplier`` scales ``mu_roll0`` once.
        grain_strength: In ``[0, 0.9)``; defaults from ``grass_type``.
        grain_direction_rad: Down-grain direction [rad] (``atan2(y, x)``).
        compaction_factor: Soil firmness in ``(0, 1]``; metadata only.
    """

    stimp_ft: float
    grass_type: GrassType = GrassType.BENT_GRASS
    condition: TurfCondition = TurfCondition.NORMAL
    grain_strength: float | None = None
    grain_direction_rad: float = 0.0
    compaction_factor: float = 0.8

    def __post_init__(self) -> None:
        require_finite(self.stimp_ft, "stimp_ft")
        require(
            _MIN_STIMP_FT <= self.stimp_ft <= _MAX_STIMP_FT,
            f"stimp_ft must be in [{_MIN_STIMP_FT}, {_MAX_STIMP_FT}] ft",
            self.stimp_ft,
        )
        require(
            isinstance(self.grass_type, GrassType),
            "grass_type must be a GrassType member",
            self.grass_type,
        )
        require(
            isinstance(self.condition, TurfCondition),
            "condition must be a TurfCondition member",
            self.condition,
        )
        grain_strength = self.grain_strength
        if grain_strength is None:
            grain_strength = self.grass_type.default_grain_strength
            object.__setattr__(self, "grain_strength", grain_strength)
        require_grain(grain_strength, self.grain_direction_rad)
        require_finite(self.compaction_factor, "compaction_factor")
        require(
            0.0 < self.compaction_factor <= 1.0,
            "compaction_factor in (0, 1]",
            self.compaction_factor,
        )

    def to_friction_params(self) -> FrictionParams:
        """Render onto :class:`~.friction.FrictionParams`.

        Postconditions: ``mu_roll0 > 0``, the condition multiplier is
        applied exactly once (never stacked with any other factor), and
        ``mu_static >= mu_roll0`` holds for every in-range stimp/condition
        combination (see :data:`_STATIC_MU_MARGIN`).
        """
        grain_strength = float(self.grain_strength or 0.0)
        mu_roll0 = (
            stimp_to_rolling_mu(self.stimp_ft) * self.condition.friction_multiplier
        )
        ensure(mu_roll0 > 0.0, "mu_roll0 must stay positive", mu_roll0)
        mu_static = max(DEFAULT_STATIC_MU, mu_roll0 * _STATIC_MU_MARGIN)
        return FrictionParams(
            mu_roll0=mu_roll0,
            mu_static=mu_static,
            grain_strength=grain_strength,
            grain_direction_rad=self.grain_direction_rad,
        )

    def to_surface_spec(self, extent_m: float | None = None) -> SurfaceSpec:
        """Render onto a flat, homogeneous :class:`~.surfaces.SurfaceSpec`.

        Args:
            extent_m: Grid side length [m]; omit to use
                :meth:`SurfaceSpec.flat_uniform`'s own default.
        """
        friction = self.to_friction_params()
        if extent_m is None:
            return SurfaceSpec.flat_uniform(stimp_ft=self.stimp_ft, friction=friction)
        return SurfaceSpec.flat_uniform(
            stimp_ft=self.stimp_ft, friction=friction, extent_m=extent_m
        )

    @classmethod
    def preset(cls, name: str) -> TurfSpec:
        """Named presets restated from the migration source's
        ``TurfProperties.create_preset``.

        Raises:
            ValueError: If ``name`` is not a known preset.
        """
        presets: dict[str, dict[str, Any]] = {
            "tournament_fast": {
                "stimp_ft": 12.0,
                "condition": TurfCondition.NORMAL,
                "grain_strength": 0.08,
            },
            "municipal_slow": {
                "stimp_ft": 8.0,
                "grass_type": GrassType.POA_ANNUA,
                "grain_strength": 0.2,
            },
            "bermuda_summer": {
                "stimp_ft": 9.0,
                "grass_type": GrassType.BERMUDA,
                "condition": TurfCondition.DRY,
                "grain_strength": 0.4,
                "grain_direction_rad": math.pi / 4.0,
            },
            "wet_morning": {
                "stimp_ft": 10.0,
                "condition": TurfCondition.DEWY,
                "grain_strength": 0.1,
            },
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets)}")
        return cls(**presets[name])
