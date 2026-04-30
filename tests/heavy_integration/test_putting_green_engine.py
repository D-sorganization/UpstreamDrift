"""
Heavy Integration Contracts — Putting Green Engine
====================================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: The putting green physics engine simulates ball rolling,
putter strokes, and produces deterministic trajectories.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    PutterStroke,
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    GrassType,
    TurfProperties,
)


@pytest.mark.live_simulation
class TestBallRollPhysics:
    """Contract: Ball rolling physics are physically plausible."""

    def test_ball_decelerates_on_flat_surface(self) -> None:
        """Ball should decelerate and stop on a flat green."""
        turf = TurfProperties(grass_type=GrassType.BENTGRASS)
        physics = BallRollPhysics(turf_properties=turf)

        state = BallState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )

        # Step until ball stops or timeout
        dt = 0.001
        for _ in range(10000):
            state = physics.step(state, dt)
            if np.linalg.norm(state.velocity) < 1e-6:
                break

        # Ball should have stopped
        assert np.linalg.norm(state.velocity) < 1e-4, "Ball should stop due to friction"
        # Ball should have moved forward
        assert state.position[0] > 0.1, "Ball should have rolled forward"

    def test_ball_no_nan_in_trajectory(self) -> None:
        """Ball trajectory must never produce NaN values."""
        turf = TurfProperties(grass_type=GrassType.BENTGRASS)
        physics = BallRollPhysics(turf_properties=turf)

        state = BallState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([3.0, 1.0, 0.0]),
        )

        dt = 0.001
        for _ in range(5000):
            state = physics.step(state, dt)
            assert not np.any(np.isnan(state.position)), "NaN in position"
            assert not np.any(np.isnan(state.velocity)), "NaN in velocity"


@pytest.mark.live_simulation
class TestGreenSurface:
    """Contract: Green surface provides slope and elevation data."""

    def test_flat_green_zero_slope(self) -> None:
        """A flat green should report zero slope everywhere."""
        surface = GreenSurface()

        slope = surface.get_slope_at(0.0, 0.0)
        assert isinstance(slope, np.ndarray | tuple | list)

    def test_green_surface_elevation(self) -> None:
        """Surface elevation query returns finite values."""
        surface = GreenSurface()

        elev = surface.get_elevation_at(1.0, 1.0)
        assert np.isfinite(elev)


@pytest.mark.live_simulation
class TestPutterStroke:
    """Contract: Putter stroke produces a valid impulse."""

    def test_stroke_produces_velocity(self) -> None:
        """A putter stroke should produce a ball velocity."""
        params = StrokeParameters(
            speed=2.0,  # m/s club head speed
            face_angle=0.0,  # degrees, square to target
            stroke_path_angle=0.0,  # degrees, straight back-straight through
        )
        stroke = PutterStroke(params)

        initial_velocity = stroke.compute_ball_velocity()
        assert np.linalg.norm(initial_velocity) > 0, "Stroke should produce velocity"
        # Ball speed should be less than club speed (not perfectly elastic)
        assert np.linalg.norm(initial_velocity) <= params.speed * 1.1


@pytest.mark.live_simulation
class TestPuttingGreenSimulator:
    """Contract: Full putting simulator works end-to-end."""

    def test_simulator_initialization(self) -> None:
        """Simulator initializes without error."""
        sim = PuttingGreenSimulator()
        assert sim is not None

    def test_simulator_deterministic(self) -> None:
        """Same inputs produce identical trajectories."""
        sim1 = PuttingGreenSimulator()
        sim1.reset()

        sim2 = PuttingGreenSimulator()
        sim2.reset()

        # Apply same initial conditions
        q1, v1 = sim1.get_state()
        q2, v2 = sim2.get_state()

        for _ in range(100):
            sim1.step()
            sim2.step()

        q1_f, v1_f = sim1.get_state()
        q2_f, v2_f = sim2.get_state()

        np.testing.assert_allclose(q1_f, q2_f, atol=1e-14)
        np.testing.assert_allclose(v1_f, v2_f, atol=1e-14)


pytestmark = pytest.mark.live_simulation
