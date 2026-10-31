"""Turf model for putting_dynamics (epic #14141, putting v4 step 4b).

Migration source: ``src/engines/physics_engines/putting_green/python/
turf_properties.py`` (``TurfProperties``, ``GrassType``, ``TurfCondition``).
That module is left untouched by this step (callers migrate in step 7,
duplicates are removed in step 10) — this module only adds the converged
model to the shared core.

**One law, not two.** ADR-0045 preserves two roll models in their existing
homes: ``ud-legacy-roll/1`` (the engine's agronomic-factor law) and
``usga-stimp-roll/1`` (:func:`~.friction.stimp_to_rolling_mu`). Per the
putting-v4 panel decision the shared core converges on one documented
model, built on ``usga-stimp-roll/1`` because it is the law consistent with
the external Grober (2011) / USGA Stimpmeter fixture. The legacy law is not
ported; it stays in the engine, unchanged, as ADR-0045 records.

Ported from the migration source, re-expressed through shared types:

* :class:`GrassType` / :class:`TurfCondition` — same members and numeric
  defaults as the migration source, whose citations (USGA, Haake 1989) are
  background only; every value below is an **uncited engineering default**.
* :class:`TurfSpec` — a frozen, DbC-checked descriptor that renders onto
  :class:`~.friction.FrictionParams` / :class:`~.surfaces.SurfaceSpec`
  through :func:`~.friction.stimp_to_rolling_mu` alone.

Grain mapping (exact, not an approximation): the migration source stores
grain as a 2-D unit vector and computes
``decel *= 1 - grain_strength * cos(theta_v - theta_grain)`` where
``theta_grain = atan2(grain_direction.y, grain_direction.x)``. The shared
:func:`~.friction.grain_factor` computes the identical
``1 - grain_strength * cos(theta_v - theta_grain)`` from
``grain_direction_rad`` directly. The two are the same formula under a
change of representation, so :class:`TurfSpec` stores the angle.

Deliberately **not** ported (no shared-core hook yet; tracked for step 5):
the height-of-cut multiplier (fit to the legacy base law), the compaction
speed factor and the cross-grain lateral curve (``simulate_ball`` consumes
only :class:`~.friction.FrictionParams`). ``compaction_factor`` is carried
as validated metadata only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.shared.python.contracts import ensure, require, require_finite

from .friction import DEFAULT_STATIC_MU, FrictionParams, stimp_to_rolling_mu
from .surfaces import SurfaceSpec

__all__ = [
    "GrassType",
    "TurfCondition",
    "TurfSpec",
]

#: Lower stimp bound for :class:`TurfSpec`, tightened from
#: :func:`~.friction.stimp_to_rolling_mu`'s own floor of 3 ft. Below ~3.92
#: ft the FROSTED condition multiplier (1.4, the largest) would push
#: ``mu_roll0`` past :class:`~.friction.FrictionParams`'s upper bound of
#: 0.2; 4.0 ft keeps every condition's headroom positive while matching
#: real-world green speeds (greens are not stimped below ~4-5 ft in
#: practice).
_MIN_STIMP_FT = 4.0
_MAX_STIMP_FT = 16.0

#: Static-over-kinetic headroom applied in :meth:`TurfSpec.to_friction_params`.
#: :data:`~.friction.DEFAULT_STATIC_MU` (0.08) is calibrated against a
#: typical stimp-10 green and is too small for slow/cold greens whose
#: condition-scaled ``mu_roll0`` already exceeds it; scaling by this margin
#: instead of using the flat default preserves the "classic static >
#: kinetic ordering" (see :mod:`.friction`'s module docstring) for every
#: in-range ``(stimp_ft, condition)`` combination.
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
    """Converged turf descriptor — the UD turf model (provisional).

    Renders onto :class:`~.friction.FrictionParams` /
    :class:`~.surfaces.SurfaceSpec` through :func:`stimp_to_rolling_mu`
    alone (``usga-stimp-roll/1``); see the module docstring for why the
    migration source's legacy roll law is not ported.

    Attributes:
        stimp_ft: Stimpmeter reading [feet], in
            ``[_MIN_STIMP_FT, _MAX_STIMP_FT]``.
        grass_type: Grass type; supplies the ``grain_strength`` default.
        condition: Environmental condition; its
            :attr:`~TurfCondition.friction_multiplier` scales
            ``mu_roll0`` exactly once, in :meth:`to_friction_params`.
        grain_strength: Grain-anisotropy amplitude in ``[0, 0.9)``;
            defaults to ``grass_type.default_grain_strength`` when
            omitted.
        grain_direction_rad: Down-grain direction [rad]; equivalent to
            the migration source's 2-D grain vector via
            ``atan2(y, x)`` (see module docstring).
        compaction_factor: Soil firmness in ``(0, 1]``; carried as
            descriptive metadata only (see module docstring).
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
        require_finite(grain_strength, "grain_strength")
        require(
            0.0 <= grain_strength < 0.9,
            "grain_strength in [0, 0.9)",
            grain_strength,
        )
        require_finite(self.grain_direction_rad, "grain_direction_rad")
        require(
            abs(self.grain_direction_rad) <= 2.0 * math.pi,
            "grain direction within +/- 2 pi",
            self.grain_direction_rad,
        )
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
