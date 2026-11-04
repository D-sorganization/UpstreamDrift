"""Counter-torque explorer payloads for the putting-green routes (#14469).

The settings of the shared core's two-hand couple model and its stroke
series.  The models live here so ``src.api.routes.putting_green`` stays
under the monolith threshold.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CounterTorqueRequest(BaseModel):
    """Counter-torque explorer settings (#14469); the core checks the ranges."""

    preload_nm: float = Field(0.60, description="In-plane grip preload couple (N m)")
    off_preload_nm: float = Field(0.55, description="Off-plane preload couple (N m)")
    noise_scale: float = Field(0.25, description="Hand-noise scale (0 = none)")
    gust_impulse_nm: float = Field(0.25, description="Gust pulse strength (N m)")
    lie_deg: float = Field(70.0, description="Shaft lie angle (deg)")
    putter: str = Field(
        "toe_hang", description="Putter head: a key of /putter-presets (#14461)"
    )
    n_points: int = Field(210, ge=1, le=2000, description="Intervals over the stroke")


class CounterTorqueResponse(BaseModel):
    """Stroke series of the core couple model, one list per sample field."""

    putter_name: str
    samples: dict[str, list[float]]
