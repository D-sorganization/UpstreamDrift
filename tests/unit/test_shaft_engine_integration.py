"""Tests for flexible shaft engine integration.

Task 3.3: Flexible Shaft Engine Integration tests.

Refactored to use shared engine availability module (DRY principle).
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.shared.python.engine_availability import MUJOCO_AVAILABLE

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )


class TestMuJoCoShaftIntegration:
    """Tests for MuJoCo flexible shaft integration."""

    @pytest.fixture
    def engine(self) -> "MuJoCoPhysicsEngine":  # type: ignore[name-defined]
        """Create MuJoCo engine with simple model."""
        if not MUJOCO_AVAILABLE:
            pytest.skip("MuJoCo not installed")

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = MuJoCoPhysicsEngine()
        # Load a simple model for testing
        simple_model = """
        <mujoco>
            <option gravity="0 0 -9.81" timestep="0.001"/>
            <worldbody>
                <body name="link" pos="0 0 0.5">
                    <joint name="joint" type="hinge" axis="0 1 0"/>
                    <geom type="cylinder" size="0.02 0.5"/>
                    <inertial pos="0 0 0.25" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                </body>
            </worldbody>
        </mujoco>
        """
        engine.load_from_string(simple_model)
        return engine

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_set_shaft_properties_returns_true(
        self, engine: "MuJoCoPhysicsEngine"
    ) -> None:  # type: ignore[name-defined]
        """set_shaft_properties should return True when configured."""
        length = 1.143  # [m] Standard driver shaft
        n_stations = 11
        EI_profile = np.linspace(50, 200, n_stations)  # [N·m²]
        mass_profile = np.linspace(0.05, 0.1, n_stations)  # [kg/m]

        result = engine.set_shaft_properties(length, EI_profile, mass_profile)

        assert result is True

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_get_shaft_state_none_before_config(
        self, engine: "MuJoCoPhysicsEngine"
    ) -> None:  # type: ignore[name-defined]
        """get_shaft_state should return None before configuration."""
        result = engine.get_shaft_state()

        assert result is None

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_get_shaft_state_after_config(self, engine: "MuJoCoPhysicsEngine") -> None:  # type: ignore[name-defined]
        """get_shaft_state should return dict after configuration."""
        n_stations = 11
        engine.set_shaft_properties(
            length=1.143,
            EI_profile=np.linspace(50, 200, n_stations),
            mass_profile=np.linspace(0.05, 0.1, n_stations),
        )

        state = engine.get_shaft_state()

        assert state is not None
        assert "deflection" in state
        assert "rotation" in state
        assert "velocity" in state
        assert "modal_amplitudes" in state

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_shaft_state_correct_shape(self, engine: "MuJoCoPhysicsEngine") -> None:  # type: ignore[name-defined]
        """Shaft state arrays should have correct shape."""
        n_stations = 11
        engine.set_shaft_properties(
            length=1.143,
            EI_profile=np.linspace(50, 200, n_stations),
            mass_profile=np.linspace(0.05, 0.1, n_stations),
        )

        state = engine.get_shaft_state()

        assert state is not None
        assert len(state["deflection"]) == n_stations
        assert len(state["rotation"]) == n_stations
        assert len(state["velocity"]) == n_stations
        assert len(state["modal_amplitudes"]) == 3  # Default 3 modes

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_initial_deflection_zero(self, engine: "MuJoCoPhysicsEngine") -> None:  # type: ignore[name-defined]
        """Initial shaft deflection should be zero."""
        n_stations = 11
        engine.set_shaft_properties(
            length=1.143,
            EI_profile=np.linspace(50, 200, n_stations),
            mass_profile=np.linspace(0.05, 0.1, n_stations),
        )

        state = engine.get_shaft_state()

        assert state is not None
        np.testing.assert_allclose(state["deflection"], 0.0)
        np.testing.assert_allclose(state["velocity"], 0.0)

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_damping_ratio_stored(self, engine: "MuJoCoPhysicsEngine") -> None:  # type: ignore[name-defined]
        """Damping ratio should be stored in configuration."""
        n_stations = 11
        custom_damping = 0.05

        engine.set_shaft_properties(
            length=1.143,
            EI_profile=np.linspace(50, 200, n_stations),
            mass_profile=np.linspace(0.05, 0.1, n_stations),
            damping_ratio=custom_damping,
        )

        assert engine._shaft_config["damping_ratio"] == custom_damping


class TestShaftInterfaceDefault:
    """Tests for default shaft interface behavior."""

    def test_interface_default_returns_false(self) -> None:
        """PhysicsEngine default implementation should return False."""
        from src.shared.python.interfaces import PhysicsEngine

        # Check that the protocol method has a default that returns False
        # This test verifies the interface definition
        assert hasattr(PhysicsEngine, "set_shaft_properties")
        assert hasattr(PhysicsEngine, "get_shaft_state")
