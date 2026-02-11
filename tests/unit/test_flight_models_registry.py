"""Tests for flight model registry coverage."""

from __future__ import annotations

import math

from src.shared.python.physics.flight_models import (
    FlightModelRegistry,
    UnifiedLaunchConditions,
)


def test_all_models_generate_trajectory() -> None:
    """All registered models should return a non-empty trajectory."""
    launch = UnifiedLaunchConditions(
        ball_speed=70.0,
        launch_angle=math.radians(12.0),
        azimuth_angle=0.0,
        spin_rate=2500.0,
    )

    for model in FlightModelRegistry.get_all_models():
        result = model.simulate(launch, max_time=1.0, dt=0.05)
        assert result.trajectory
