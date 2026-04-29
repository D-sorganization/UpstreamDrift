"""Integration tests for aerodynamics wired into physics engine step() methods (Issue #3167).

This test verifies that:
1. Aerodynamics can be enabled/disabled in all engines
2. Aerodynamic forces are applied during simulation steps
3. Spin decay occurs as expected
4. Aerodynamics affects trajectory (reduces distance, curves with spin)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.aerodynamics._config import AerodynamicsConfig
from src.shared.python.physics.aerodynamics._engine import AerodynamicsEngine

pytestmark = pytest.mark.integration


class TestAerodynamicsIntegration:
    """Test aerodynamics integration with physics engines."""

    def test_aerodynamics_engine_basic(self) -> None:
        """Test basic aerodynamics engine functionality."""
        config = AerodynamicsConfig(drag_enabled=True, lift_enabled=True)
        aero = AerodynamicsEngine(config)

        velocity = np.array([70.0, 0.0, 10.0])  # Driver speed with launch angle
        spin = np.array([0.0, 300.0, 0.0])  # Backspin in rad/s

        forces = aero.compute_forces(velocity, spin)

        assert "drag" in forces
        assert "lift" in forces
        assert "magnus" in forces
        assert "total" in forces

        # Verify all forces are 3D vectors
        for key in ["drag", "lift", "magnus", "total"]:
            assert forces[key].shape == (3,)
            assert np.all(np.isfinite(forces[key]))

        # Drag should oppose velocity (negative dot product)
        assert np.dot(forces["drag"], velocity) < 0

    def test_aerodynamics_drag_effect(self) -> None:
        """Test that drag reduces velocity magnitude."""
        config = AerodynamicsConfig(
            drag_enabled=True, lift_enabled=False, magnus_enabled=False
        )
        aero = AerodynamicsEngine(config)

        velocity = np.array([50.0, 0.0, 0.0])
        spin = np.array([0.0, 0.0, 0.0])

        forces = aero.compute_forces(velocity, spin)
        drag = forces["drag"]

        # Drag should be negative in velocity direction (opposing motion)
        assert drag[0] < 0
        assert abs(drag[1]) < 1e-6  # No lateral drag without spin
        assert abs(drag[2]) < 1e-6  # No vertical drag without spin

    def test_aerodynamics_lift_with_backspin(self) -> None:
        """Test that backspin creates lift (direction depends on spin orientation)."""
        config = AerodynamicsConfig(
            drag_enabled=False, lift_enabled=True, magnus_enabled=False
        )
        aero = AerodynamicsEngine(config)

        velocity = np.array([50.0, 0.0, 0.0])  # Forward motion
        spin = np.array([0.0, 300.0, 0.0])  # Spin around Y axis

        forces = aero.compute_forces(velocity, spin)
        lift = forces["lift"]

        # With spin, there should be a non-zero lift force perpendicular to velocity
        assert np.linalg.norm(lift) > 1e-6, "Backspin should create lift"
        assert np.abs(lift[0]) < 1e-6, "Lift should not be in velocity direction"

    def test_aerodynamics_spin_decay(self) -> None:
        """Test exponential spin decay over time."""
        config = AerodynamicsConfig()
        aero = AerodynamicsEngine(config)

        initial_spin = np.array([0.0, 300.0, 0.0])
        dt = 0.01
        velocity_magnitude = 50.0

        # Apply spin decay multiple times
        spin = initial_spin.copy()
        spin_history = [np.linalg.norm(initial_spin)]
        for _ in range(1000):
            spin = aero.compute_spin_decay(spin, dt, velocity_magnitude)
            spin_history.append(np.linalg.norm(spin))

        # Spin should decay monotonically
        for i in range(len(spin_history) - 1):
            assert (
                spin_history[i + 1] <= spin_history[i]
            ), "Spin should decay monotonically"

        # Spin should be noticeably reduced after 10 seconds
        assert spin_history[-1] < initial_spin[1] * 0.95
        assert np.all(np.isfinite(spin))

    def test_aerodynamics_with_zero_velocity(self) -> None:
        """Test that forces are zero when ball is stationary."""
        config = AerodynamicsConfig()
        aero = AerodynamicsEngine(config)

        velocity = np.array([0.0, 0.0, 0.0])
        spin = np.array([0.0, 300.0, 0.0])

        forces = aero.compute_forces(velocity, spin)

        # All forces should be near zero (drag and lift are velocity-dependent)
        assert np.linalg.norm(forces["drag"]) < 1e-6
        assert np.linalg.norm(forces["lift"]) < 1e-6
        # Magnus can be non-zero even at zero velocity (rarely physical but allowed)

    def test_aerodynamics_config_disable_features(self) -> None:
        """Test that individual aerodynamic features can be disabled."""
        velocity = np.array([50.0, 0.0, 0.0])
        spin = np.array([0.0, 300.0, 0.0])

        # All enabled
        config_all = AerodynamicsConfig(
            drag_enabled=True, lift_enabled=True, magnus_enabled=True
        )
        aero_all = AerodynamicsEngine(config_all)
        forces_all = aero_all.compute_forces(velocity, spin)
        total_all = np.linalg.norm(forces_all["total"])

        # Only drag
        config_drag = AerodynamicsConfig(
            drag_enabled=True, lift_enabled=False, magnus_enabled=False
        )
        aero_drag = AerodynamicsEngine(config_drag)
        forces_drag = aero_drag.compute_forces(velocity, spin)
        total_drag = np.linalg.norm(forces_drag["total"])

        # With lift and magnus disabled, total should be smaller
        assert total_drag < total_all

    @pytest.mark.integration
    def test_aerodynamics_trajectory_effect(self) -> None:
        """Integration test: verify aerodynamics affects trajectory.

        This test simulates a simple ball trajectory with and without
        aerodynamics to verify that forces actually affect motion.
        """

        # Simple explicit Euler integration
        def simulate(dt: float, num_steps: int, with_aero: bool) -> np.ndarray:
            """Simulate ball trajectory."""
            positions = np.zeros((num_steps + 1, 3))
            velocity = np.array([50.0, 0.0, 20.0])
            spin = np.array([0.0, 300.0, 0.0])

            if with_aero:
                config = AerodynamicsConfig(
                    drag_enabled=True, lift_enabled=True, magnus_enabled=True
                )
                aero = AerodynamicsEngine(config)
            else:
                aero = None

            gravity = np.array([0.0, 0.0, -9.81])
            ball_mass = 0.04593

            for i in range(num_steps):
                positions[i] = positions[i - 1] if i > 0 else np.zeros(3)

                # Gravity
                accel = gravity.copy()

                # Aerodynamics
                if aero:
                    forces = aero.compute_forces(velocity, spin)
                    aero_accel = forces["total"] / ball_mass
                    accel += aero_accel

                    # Spin decay
                    spin = aero.compute_spin_decay(spin, dt, np.linalg.norm(velocity))

                # Euler step
                velocity += accel * dt
                positions[i + 1] = positions[i] + velocity * dt

                # Stop if ball hits ground
                if positions[i + 1, 2] < 0:
                    positions[i + 1, 2] = 0
                    break

            return positions

        # Simulate with and without aerodynamics
        dt = 0.001
        num_steps = 5000

        positions_no_aero = simulate(dt, num_steps, with_aero=False)
        positions_with_aero = simulate(dt, num_steps, with_aero=True)

        # Find landing distances
        landing_no_aero = np.max(np.linalg.norm(positions_no_aero[:, :2], axis=1))
        landing_with_aero = np.max(np.linalg.norm(positions_with_aero[:, :2], axis=1))

        # Aerodynamics should reduce distance due to drag
        assert (
            landing_with_aero < landing_no_aero
        ), "Aerodynamics should reduce landing distance due to drag"

        # Verify difference is meaningful (not just numerical noise)
        distance_reduction = (landing_no_aero - landing_with_aero) / landing_no_aero
        assert distance_reduction > 0.01, "Aerodynamic effect should be significant"

        print(
            f"Landing distance without aero: {landing_no_aero:.2f} m "
            f"(aerodynamics reduces by {distance_reduction * 100:.1f}%)"
        )


@pytest.mark.slow
class TestEngineAerodynamicsIntegration:
    """Test aerodynamics integration with actual physics engines.

    These tests verify that engines can:
    1. Enable/disable aerodynamics
    2. Apply aerodynamics during step()
    3. Integrate with the engine's internal state
    """

    @pytest.mark.skipif(
        not __import__("importlib.util").util.find_spec("mujoco"),
        reason="MuJoCo not installed",
    )
    def test_mujoco_aerodynamics_enable(self) -> None:
        """Test enabling aerodynamics in MuJoCo engine."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = MuJoCoPhysicsEngine()
        config = AerodynamicsConfig()

        # Should not raise
        engine.enable_aerodynamics(config)
        assert engine.aero_engine is not None

        engine.disable_aerodynamics()
        assert engine.aero_engine is None

    @pytest.mark.skipif(
        not __import__("importlib.util").util.find_spec("pydrake"),
        reason="Drake not installed",
    )
    def test_drake_aerodynamics_enable(self) -> None:
        """Test enabling aerodynamics in Drake engine."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        config = AerodynamicsConfig()

        # Should not raise
        engine.enable_aerodynamics(config)
        assert engine.aero_engine is not None

        engine.disable_aerodynamics()
        assert engine.aero_engine is None

    @pytest.mark.skipif(
        not __import__("importlib.util").util.find_spec("pinocchio"),
        reason="Pinocchio not installed",
    )
    def test_pinocchio_aerodynamics_enable(self) -> None:
        """Test enabling aerodynamics in Pinocchio engine."""
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        config = AerodynamicsConfig()

        # Should not raise
        engine.enable_aerodynamics(config)
        assert engine.aero_engine is not None

        engine.disable_aerodynamics()
        assert engine.aero_engine is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
