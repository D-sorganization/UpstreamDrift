from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.engines.physics_engines.putting_green.python._surface_data import SlopeRegion
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)

if TYPE_CHECKING:
    from src.engines.physics_engines.putting_green.python.green_surface import (
        GreenSurface,
    )


class SurfacePresetsMixin:
    """Mixin providing factory class methods for GreenSurface."""

    @classmethod
    def from_heightmap(
        cls,
        heightmap: np.ndarray,
        width: float,
        height: float,
        turf: TurfProperties | None = None,
    ) -> GreenSurface:
        """Create green surface from heightmap array.

        Args:
            heightmap: 2D array of elevations
            width: Physical width [m]
            height: Physical height [m]
            turf: Turf properties

        Returns:
            GreenSurface instance
        """
        if heightmap is None:
            raise ValueError("heightmap must be provided")
        green = cls(width=width, height=height, turf=turf)  # type: ignore[call-arg]
        green.set_heightmap(heightmap)  # type: ignore[attr-defined]
        return green  # type: ignore[return-value]

    @classmethod
    def create_preset(cls, name: str) -> GreenSurface:
        """Create a preset green configuration.

        Available presets:
            - flat_practice: Flat practice green
            - undulating_championship: Championship undulating green
            - severe_slopes: Augusta-style severe slopes
            - tiered: Two-tier green with ridge

        Args:
            name: Name of preset

        Returns:
            Configured GreenSurface

        Raises:
            ValueError: If preset unknown
        """
        if name == "flat_practice":
            return cls(  # type: ignore[call-arg,return-value]
                width=15.0,
                height=15.0,
                turf=TurfProperties.create_preset("practice_green"),
            )

        if name == "undulating_championship":
            green = cls(  # type: ignore[call-arg]
                width=25.0,
                height=25.0,
                turf=TurfProperties.create_preset("tournament_fast"),
            )
            # Add multiple subtle slopes
            green.add_slope_region(  # type: ignore[attr-defined]
                SlopeRegion(
                    center=np.array([8.0, 8.0]),
                    radius=6.0,
                    slope_direction=np.array([1.0, 0.5]),
                    slope_magnitude=0.02,
                )
            )
            green.add_slope_region(  # type: ignore[attr-defined]
                SlopeRegion(
                    center=np.array([17.0, 17.0]),
                    radius=5.0,
                    slope_direction=np.array([-0.5, 1.0]),
                    slope_magnitude=0.015,
                )
            )
            green.add_depression(  # type: ignore[attr-defined]
                center=np.array([12.0, 12.0]),
                radius=3.0,
                depth=0.02,
            )
            green.set_hole_position(np.array([15.0, 15.0]))  # type: ignore[attr-defined]
            return green  # type: ignore[return-value]

        if name == "severe_slopes":
            green = cls(  # type: ignore[call-arg]
                width=20.0,
                height=20.0,
                turf=TurfProperties.create_preset("augusta_like"),
            )
            # Add severe tier
            green.add_slope_region(  # type: ignore[attr-defined]
                SlopeRegion(
                    center=np.array([10.0, 10.0]),
                    radius=8.0,
                    slope_direction=np.array([1.0, 0.0]),
                    slope_magnitude=0.05,  # 5% slope
                )
            )
            green.add_ridge(  # type: ignore[attr-defined]
                start=np.array([5.0, 15.0]),
                end=np.array([15.0, 15.0]),
                height=0.05,
                width=2.0,
            )
            green.set_hole_position(np.array([15.0, 10.0]))  # type: ignore[attr-defined]
            return green  # type: ignore[return-value]

        if name == "tiered":
            green = cls(  # type: ignore[call-arg]
                width=20.0,
                height=20.0,
                turf=TurfProperties.create_preset("tournament_standard"),
            )
            # Create a tier with ridge
            green.add_ridge(  # type: ignore[attr-defined]
                start=np.array([0.0, 10.0]),
                end=np.array([20.0, 10.0]),
                height=0.08,
                width=3.0,
            )
            green.set_hole_position(np.array([15.0, 5.0]))  # type: ignore[attr-defined]
            return green  # type: ignore[return-value]

        raise ValueError(f"Unknown preset: {name}")
