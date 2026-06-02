"""Tests for flight model registry coverage."""

from __future__ import annotations

import math

from src.shared.python.physics.flight_models import (
    ConstantCoefficientModel,
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


def test_constant_coefficient_specs_are_provenanced_and_in_range() -> None:
    """Each constant-Cd/Cl model is cited and within documented golf ranges.

    Value test for #7055. Wind-tunnel measurements of dimpled golf balls in
    the post-critical regime (Re ~ 1e5) put steady Cd in roughly 0.15-0.30 and
    Cl in roughly 0.10-0.30 (e.g. Bearman & Harvey 1976; cf.
    ``GOLF_BALL_DRAG_COEFFICIENT`` = 0.25). Pin every registered
    constant-coefficient spec to those ranges and require a non-empty
    reference so the coefficients cannot silently drift to non-physical
    values.
    """
    FlightModelRegistry.reset()
    constant_models = [
        m
        for m in FlightModelRegistry.get_all_models()
        if isinstance(m, ConstantCoefficientModel)
    ]
    assert constant_models, "expected constant-coefficient models in registry"

    for model in constant_models:
        spec = model._spec
        assert spec.reference.strip(), f"{spec.name} missing reference"
        assert 0.15 <= spec.cd <= 0.30, f"{spec.name} Cd={spec.cd} out of range"
        assert 0.10 <= spec.cl <= 0.30, f"{spec.name} Cl={spec.cl} out of range"
        assert spec.spin_decay >= 0.0, f"{spec.name} negative spin decay"
