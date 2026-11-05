"""Request models of the 3-D putting routes (moved from the routes, #14466).

The green shape (#14463) and the /simulate-3d inputs, which
/solve-aim-3d shares, live here so routes/putting_green.py stays under
the monolith register's line threshold.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from src.shared.python.putting_dynamics.green_spec import (
    GREEN_PRESETS,
    GreenSpec,
    green_preset,
)
from src.shared.python.putting_dynamics.putter_presets import BALANCE_PRESETS
from src.shared.python.putting_dynamics.types import (
    TYPICAL_HEAD_MASS_KG,
    TYPICAL_HEAD_MOI_KG_M2,
    TYPICAL_LOFT_DEG,
)

#: The shared core's preset names, so an unknown one is a 422 in body or query.
GreenPresetName = Enum(  # type: ignore[misc]
    "GreenPresetName", {key: key for key in GREEN_PRESETS}, type=str
)


class GreenShapeRequest(BaseModel):
    """The green's shape (#14463): a named preset, or slope, tier and bumps.

    A ``preset`` replaces every other field with that preset's shape.
    """

    preset: GreenPresetName | None = Field(None, description="GET /green-presets")
    grade_percent: float = Field(0.0, ge=0.0, le=10.0)
    aspect_deg: float = Field(0.0, ge=-360.0, le=360.0, description="Falls toward")
    bump_height_m: float = Field(0.0, ge=0.0, le=0.05)
    bump_length_m: float = Field(1.5, gt=0.0, le=20.0)
    tier_height_m: float = Field(0.0, ge=0.0, le=0.15)
    tier_width_m: float = Field(1.0, gt=0.0, le=5.0)
    seed: int = Field(0, ge=0)

    def spec(self) -> GreenSpec:
        """The shared core's description of this green."""
        if self.preset is not None:
            return green_preset(self.preset.value)
        return GreenSpec(**self.model_dump(exclude={"preset"}))


class Putt3DSimulationRequest(BaseModel):
    """Physics and visualization inputs for one three-dimensional putt."""

    putter_speed_mps: float = Field(1.8, gt=0.0, le=10.0)
    loft_deg: float = Field(TYPICAL_LOFT_DEG, ge=-6.0, le=10.0)
    head_mass_kg: float = Field(TYPICAL_HEAD_MASS_KG, ge=0.1, le=1.0)
    head_moi_kg_m2: float = Field(TYPICAL_HEAD_MOI_KG_M2, ge=1e-5, le=1e-2)
    coefficient_of_restitution: float = Field(0.78, gt=0.0, lt=1.0)
    hosel_toe_m: float = Field(0.0, ge=-0.08, le=0.08)
    hosel_forward_m: float = Field(0.0, ge=-0.05, le=0.05)
    balance_type: str | None = Field(
        None,
        description=(
            "Named core putter head (see /putter-presets); when set it supplies "
            "head mass, MOI and hosel offsets, overriding those fields"
        ),
    )
    impact_toe_m: float = Field(
        0.0, ge=-0.08, le=0.08, description="Strike offset, toe + / heel - [m]"
    )
    stimp_rating: float = Field(10.0, ge=6.0, le=15.0)
    grade_percent: float = Field(0.0, ge=0.0, le=10.0)
    downhill_aspect_deg: float = Field(0.0, ge=-180.0, le=180.0)
    grain_strength: float = Field(0.0, ge=0.0, lt=0.9)
    grain_direction_deg: float = Field(0.0, ge=-180.0, le=180.0)
    rolling_velocity_coefficient: float = Field(0.0, ge=0.0, le=1.0)
    bump_height_m: float = Field(0.0, ge=0.0, le=0.01)
    friction_variation: float = Field(0.0, ge=0.0, le=0.9)
    random_seed: int = Field(8345, ge=0, le=2**31 - 1)
    hole_x_m: float = Field(3.0, ge=-9.0, le=9.0)
    hole_y_m: float = Field(0.0, ge=-9.0, le=9.0)
    aim_heading_deg: float = Field(
        0.0, ge=-180.0, le=180.0, description="Strike heading from +x, CCW [deg]"
    )
    green: GreenShapeRequest | None = Field(
        None,
        description=(
            "Core green shape (#14464); when set it replaces grade_percent, "
            "downhill_aspect_deg and bump_height_m"
        ),
    )

    @field_validator("balance_type")
    @classmethod
    def _known_balance_type(cls, value: str | None) -> str | None:
        if value is not None and value not in BALANCE_PRESETS:
            raise ValueError(f"balance_type must be one of {sorted(BALANCE_PRESETS)}")
        return value


class AimSolve3DResponse(BaseModel):
    """The pace and line that hole a /simulate-3d putt (the desktop read)."""

    putter_speed_mps: float
    aim_heading_deg: float
    break_m: float
    holed: bool
    converged: bool


__all__ = [
    "AimSolve3DResponse",
    "GreenPresetName",
    "GreenShapeRequest",
    "Putt3DSimulationRequest",
]
