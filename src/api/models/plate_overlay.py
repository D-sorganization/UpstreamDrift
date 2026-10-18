"""Request model for the measured plate overlay (OF-07, #13087).

Everything a swing file cannot carry is declared by the caller and validated
again by the force-plate domain objects: plate uncertainty and geometry, the
channels' timing skew, where the stance frame sits in the model's world, and
how the plate clock maps onto model time.  The trial itself is named by a path *relative* to the server's
allow-listed plate-trial root, never an arbitrary file.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlateDeclarationsModel(BaseModel):
    """Declared plate facts (see ``PlateDeclarations``)."""

    force_floor_n: float
    force_fraction: float
    cop_m: float
    sensing_depth_sigma_m: float
    uncertainty_source: str
    sensing_depth_m: float
    capacity_n: float
    device: str | None = None


class PlateReadingModel(BaseModel):
    """Declared channel timing (see ``DeclaredReading``)."""

    skew_s: float = Field(..., ge=0.0, description="Timing skew within a sample (s)")
    cop_skew: Literal["unknown", "independent", "common"] = Field(
        ..., description="How the two CoP channels' skews relate"
    )


class PlateAlignmentModel(BaseModel):
    """Mapping ``model_time = trial_time + offset_s`` with its provenance."""

    offset_s: float
    sigma_offset_s: float
    kind: Literal["absolute_utc", "shot_impact", "manual", "cross_correlation"]
    note: str
    derived_from_model: str | None = None


class PlateOverlayRequest(BaseModel):
    """Overlay one plate trial on a run's ground reaction."""

    run_id: str | None = Field(None, description="Run id; defaults to the active run")
    impact_time_s: float | None = Field(None, description="Impact event marker (s)")
    trial: str = Field(
        ..., description="Swing '<id>.json' relative to the plate-trial root"
    )
    handedness: Literal["RH", "LH"]
    origin_m: list[float] = Field(
        ..., min_length=3, max_length=3, description="Stance origin in the world (m)"
    )
    target_dir: list[float] = Field(
        default_factory=lambda: [0.0, -1.0, 0.0],
        min_length=3,
        max_length=3,
        description="Horizontal unit vector toward the target (default ADR-0041 -Y)",
    )
    declared: PlateDeclarationsModel
    alignment: PlateAlignmentModel
    reading: PlateReadingModel
    coverage: float = Field(2.0, description="Band factor for the interpolation bias")
