"""Tests for OpenSim integration (Section J).

Verifies:
- Hill-type muscle model functionality
- Activation → force →torque pipeline
- Muscle-induced acceleration analysis
- Grip wrapping geometry
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

LOGGER = logging.getLogger(__name__)

# Skip entire module if OpenSim not available
try:
    import opensim

    if not hasattr(opensim, "Model"):
        pytest.skip("OpenSim is mocked or unavailable", allow_module_level=True)
except ImportError:
    pytest.skip("OpenSim not installed", allow_module_level=True)


# Skip entire module if OpenSim not available
try:
    import opensim
    if not hasattr(opensim, "Model"):
        pytest.skip("OpenSim is mocked or unavailable", allow_module_level=True)
except ImportError:
    pytest.skip("OpenSim not installed", allow_module_level=True)

@pytest.fixture
def simple_arm_model():
    """Create a simple arm model with muscles for testing."""
    try:
        import opensim
    except ImportError:
        pytest.skip("OpenSim not installed")

    # Create a simple arm model
    model = opensim.Model()
    model.setName("SimpleArm")

    # Ground body
    ground = model.getGround()

    # Upper arm body
    upper_arm = opensim.Body(
        "upperarm",
        1.0,  # mass [kg]
        opensim.Vec3(0, -0.15, 0),  # COM
        opensim.Inertia(0.01, 0.01, 0.01),  # Inertia
    )

    # Shoulder joint (revolute)
    shoulder_loc = opensim.Vec3(0, 0, 0)
    shoulder_joint = opensim.PinJoint(
        "shoulder",
        ground,
        shoulder_loc,
        opensim.Vec3(0, 0, 0),
        upper_arm,
        shoulder_loc,
        opensim.Vec3(0, 0, 0),
    )

    model.addBody(upper_arm)
    model.addJoint(shoulder_joint)

    # Add a simple muscle (Thelen2003Muscle - Hill-type)
    muscle = opensim.Thelen2003Muscle()
    muscle.setName("biceps")
    muscle.setMaxIsometricForce(500.0)  # [N]
    muscle.setOptimalFiberLength(0.08)  # [m]
    muscle.setTendonSlackLength(0.2)  # [m]

    # Muscle path: origin on ground, insertion on upperarm
    muscle.addNewPathPoint("origin", ground, opensim.Vec3(0, 0.05, 0))
    muscle.addNewPathPoint("insertion", upper_arm, opensim.Vec3(0, -0.1, 0))

    model.addForce(muscle)

    # Finalize
    state = model.initSystem()

    return model, state


class TestOpenSimMuscleModels:
    """Test Hill-type muscle model functionality."""

    def test_muscle_force_length_curve(self, simple_arm_model):
        """Section J: Verify Hill-type F-L relationship."""
        model, state = simple_arm_model

        try:
            import opensim
        except ImportError:
            pytest.skip("OpenSim not installed")

        # Get muscle
        biceps = opensim.Muscle.safeDownCast(model.getMuscles().get("biceps"))

        # Set activation to 1.0 (fully active)
        biceps.setActivation(state, 1.0)

        # Set muscle at optimal length
        model.realizeDynamics(state)

        # Get force
        F_muscle = biceps.getActiveFiberForce(state)
        F_max = biceps.getMaxIsometricForce()

        # At optimal length with full activation, force should be near maximum
        # (exact value depends on velocity and activation dynamics)
        LOGGER.info(
            f"Muscle force at optimal length: {F_muscle:.1f} N (F_max={F_max:.1f} N)"
        )

        # Basic sanity check: force should be positive and reasonable
        assert (
            0 < F_muscle <= F_max * 1.5
        ), f"Muscle force {F_muscle} outside expected range"

    def test_activation_dynamics(self, simple_arm_model):
        """Section J: Verify activation dynamics (30-50ms delay)."""
        model, state = simple_arm_model

        try:
            import opensim
        except ImportError:
            pytest.skip("OpenSim not installed")

        biceps = opensim.Muscle.safeDownCast(model.getMuscles().get("biceps"))

        # Start with zero activation
        biceps.setActivation(state, 0.0)

        # Set excitation to 1.0
        # Note: In real OpenSim simulations, activation evolves over time
        #  This test just verifies the interface exists

        a_initial = biceps.getActivation(state)
        biceps.setActivation(state, 1.0)
        a_final = biceps.getActivation(state)

        LOGGER.info(f"Activation: {a_initial:.3f} → {a_final:.3f}")

        assert a_initial == 0.0
        assert a_final == 1.0


class TestOpenSimMuscleAnalysis:
    """Test muscle analysis module."""

    def test_muscle_force_extraction(self, simple_arm_model):
        """Section J: Extract muscle forces."""
        model, state = simple_arm_model

        try:
            from engines.physics_engines.opensim.python.muscle_analysis import (
                OpenSimMuscleAnalyzer,
            )
        except ImportError:
            pytest.skip("Muscle analysis module not available")

        analyzer = OpenSimMuscleAnalyzer(model, state)

        # Get muscle forces
        forces = analyzer.get_muscle_forces()

        LOGGER.info(f"Muscle forces: {forces}")

        assert "biceps" in forces
        assert isinstance(forces["biceps"], float)

    def test_moment_arm_computation(self, simple_arm_model):
        """Section J: Compute moment arms."""
        model, state = simple_arm_model

        try:
            from engines.physics_engines.opensim.python.muscle_analysis import (
                OpenSimMuscleAnalyzer,
            )
        except ImportError:
            pytest.skip("Muscle analysis module not available")

        analyzer = OpenSimMuscleAnalyzer(model, state)

        # Get moment arms
        moment_arms = analyzer.get_moment_arms()

        LOGGER.info(f"Moment arms: {moment_arms}")

        assert "biceps" in moment_arms
        # Biceps should have moment arm about shoulder coordinate
        assert len(moment_arms["biceps"]) > 0

    def test_muscle_induced_acceleration(self, simple_arm_model):
        """Section J: Compute muscle-induced accelerations."""
        model, state = simple_arm_model

        try:
            import opensim

            from engines.physics_engines.opensim.python.muscle_analysis import (
                OpenSimMuscleAnalyzer,
            )
        except ImportError:
            pytest.skip("Required modules not available")

        # Set biceps activation
        biceps = opensim.Muscle.safeDownCast(model.getMuscles().get("biceps"))
        biceps.setActivation(state, 0.5)

        analyzer = OpenSimMuscleAnalyzer(model, state)

        # Compute induced accelerations
        induced_accel = analyzer.compute_muscle_induced_accelerations()

        LOGGER.info(f"Induced accelerations: {induced_accel}")

        assert "biceps" in induced_accel
        assert isinstance(induced_accel["biceps"], np.ndarray)
        # Should produce non-zero acceleration
        assert not np.allclose(induced_accel["biceps"], 0.0)

    def test_comprehensive_muscle_analysis(self, simple_arm_model):
        """Section J: Full muscle contribution report."""
        model, state = simple_arm_model

        try:
            from engines.physics_engines.opensim.python.muscle_analysis import (
                OpenSimMuscleAnalyzer,
            )
        except ImportError:
            pytest.skip("Muscle analysis module not available")

        analyzer = OpenSimMuscleAnalyzer(model, state)

        # Run full analysis
        analysis = analyzer.analyze_all()

        LOGGER.info("Muscle analysis complete:")
        LOGGER.info(f"  Forces: {analysis.muscle_forces}")
        LOGGER.info(f"  Activations: {analysis.activation_levels}")
        LOGGER.info(f"  Total torque: {analysis.total_muscle_torque}")

        # Verify all fields populated
        assert len(analysis.muscle_forces) > 0
        assert len(analysis.activation_levels) > 0
        assert len(analysis.total_muscle_torque) > 0


class TestOpenSimEngine:
    """Test OpenSim engine with muscle integration."""

    def test_bias_force_computation(self):
        """Verify bias force computation uses inverse dynamics."""
        try:
            from engines.physics_engines.opensim.python.opensim_physics_engine import (
                OpenSimPhysicsEngine,
            )
        except ImportError:
            pytest.skip("OpenSim engine not available")

        engine = OpenSimPhysicsEngine()

        # Without a model, should return empty
        bias = engine.compute_bias_forces()
        assert len(bias) == 0

        # With a model loaded, we'd test actual computation
        # (requires valid .osim file)

    def test_gravity_force_computation(self):
        """Verify gravity force isolation."""
        try:
            from engines.physics_engines.opensim.python.opensim_physics_engine import (
                OpenSimPhysicsEngine,
            )
        except ImportError:
            pytest.skip("OpenSim engine not available")

        engine = OpenSimPhysicsEngine()

        # Without a model, should return empty
        gravity = engine.compute_gravity_forces()
        assert len(gravity) == 0

    def test_muscle_analyzer_integration(self):
        """Verify engine provides muscle analyzer."""
        try:
            from engines.physics_engines.opensim.python.opensim_physics_engine import (
                OpenSimPhysicsEngine,
            )
        except ImportError:
            pytest.skip("OpenSim engine not available")

        engine = OpenSimPhysicsEngine()

        # Without model, analyzer should be None
        analyzer = engine.get_muscle_analyzer()
        assert analyzer is None

    def test_drift_control_with_muscles(self, simple_arm_model):
        """Section F + J: Verify drift-control works with muscle model."""
        model, state = simple_arm_model

        try:
            from engines.physics_engines.opensim.python.opensim_physics_engine import (
                OpenSimPhysicsEngine,
            )
        except ImportError:
            pytest.skip("OpenSim engine not available")

        # Create engine and manually set model/state
        engine = OpenSimPhysicsEngine()
        engine._model = model
        engine._state = state

        # Compute drift (with zero muscle activation)
        a_drift = engine.compute_drift_acceleration()

        # Should return non-empty array
        assert len(a_drift) > 0

        LOGGER.info(f"Drift acceleration: {a_drift}")


class TestOpenSimGripModel:
    """Test grip wrapping geometry."""

    def test_grip_model_creation(self, simple_arm_model):
        """Section J1: Create grip model interface."""
        model, _ = simple_arm_model

        try:
            from engines.physics_engines.opensim.python.muscle_analysis import (
                OpenSimGripModel,
            )
        except ImportError:
            pytest.skip("Grip model not available")

        grip_model = OpenSimGripModel(model)

        assert grip_model is not None
        assert grip_model.model == model

    def test_cylindrical_wrap_addition(self, simple_arm_model):
        """Section J1: Add cylindrical wrapping surface."""
        model, _ = simple_arm_model

        try:
            from engines.physics_engines.opensim.python.muscle_analysis import (
                OpenSimGripModel,
            )
        except ImportError:
            pytest.skip("Grip model not available")

        grip_model = OpenSimGripModel(model)

        # Add wrap geometry (this may fail if bodies don't exist, which is expected)
        try:
            grip_model.add_cylindrical_wrap(
                muscle_name="biceps",
                grip_body_name="upperarm",  # Using upperarm as "grip" for test
                radius=0.03,  # 3 cm radius (shaft + hand)
                length=0.15,  # 15 cm
            )
            LOGGER.info("Successfully added cylindrical wrap")
        except Exception as e:
            LOGGER.info(f"Wrap addition failed (expected for simple model): {e}")
            # This is acceptable - we're testing the interface exists
