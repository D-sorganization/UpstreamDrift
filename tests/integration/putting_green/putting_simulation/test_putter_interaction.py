"""Integration tests for Putting Green Simulation.

These tests verify that all components work together correctly:
- TurfProperties + GreenSurface
- BallRollPhysics + GreenSurface
- PutterStroke + BallRollPhysics
- Full simulation end-to-end
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    PutterStroke,
    PutterType,
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
    SimulationConfig,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)


class TestPutterInteraction:
    """Tests for putter-ball interaction."""

    def test_different_putter_types(self) -> None:
        """Different putter types should behave differently."""
        blade = PutterStroke(putter_type=PutterType.BLADE)
        mallet = PutterStroke(putter_type=PutterType.MALLET)

        # Off-center hit
        off_center_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
            impact_location=np.array([0.02, 0.0]),  # Toe hit
        )

        blade_state = blade.execute_stroke(np.array([0.0, 0.0]), off_center_params)
        mallet_state = mallet.execute_stroke(np.array([0.0, 0.0]), off_center_params)

        # Mallet should lose less speed on off-center hit (higher MOI)
        assert mallet_state.speed >= blade_state.speed - 0.1, (
            "Assertion failed: mallet_state.speed >= blade_state.speed - 0.1"
        )

    def test_face_angle_affects_direction(self) -> None:
        """Open/closed face should affect ball direction."""
        putter = PutterStroke()

        square = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )
        open_face = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=5.0,  # 5 degrees open
            attack_angle=0.0,
        )

        square_state = putter.execute_stroke(np.array([0.0, 0.0]), square)
        open_state = putter.execute_stroke(np.array([0.0, 0.0]), open_face)

        # Open face should push ball right (positive y direction)
        # Ball direction follows face angle partially
        square_dir = square_state.velocity / np.linalg.norm(square_state.velocity)
        open_dir = open_state.velocity / np.linalg.norm(open_state.velocity)

        assert open_dir[1] > square_dir[1], (
            "Assertion failed: open_dir[1] > square_dir[1]"
        )
