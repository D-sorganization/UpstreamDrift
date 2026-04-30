"""Tests for MyoSuite integration (Section K).

Verifies:
- MuJoCo muscle actuator integration
- Activation → force → torque pipeline
- Muscle-induced acceleration analysis
- Grip modeling via hand muscle forces
- Cross-validation with OpenSim

Refactored to use shared engine availability module (DRY principle).
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.engine_core.engine_availability import MYOSUITE_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@pytest.fixture
def myosuite_env_available() -> bool:
    """Check if MyoSuite is available."""
    if not MYOSUITE_AVAILABLE:
        pytest.skip("MyoSuite not installed")
    return True


class TestMyoSuiteMuscleAnalyzer:
    """Test MyoSuite muscle analysis module."""

    def test_muscle_actuator_identification(self, myosuite_env_available) -> None:
        """Section K: Identify muscle actuators from MuJoCo model."""
        import gym

        # Use a simple MyoSuite environment (if available)
        env = gym.make("myoElbowPose1D6MRandom-v0")
        env.reset()

        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim

        from src.engines.physics_engines.myosuite.python.muscle_analysis import (
            MyoSuiteMuscleAnalyzer,
        )

        analyzer = MyoSuiteMuscleAnalyzer(sim)

        # Should have identified some muscles
        assert len(analyzer.muscle_names) > 0, "No muscles found"
        logger.info(
            f"Found {len(analyzer.muscle_names)} muscles: {analyzer.muscle_names}"
        )

    def test_muscle_activation_extraction(self, myosuite_env_available) -> None:
        """Section K: Extract muscle activations from sim state."""
        import gym

        env = gym.make("myoElbowPose1D6MRandom-v0")
        env.reset()

        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim

        from src.engines.physics_engines.myosuite.python.muscle_analysis import (
            MyoSuiteMuscleAnalyzer,
        )

        analyzer = MyoSuiteMuscleAnalyzer(sim)

        # Get activations
        activations = analyzer.get_muscle_activations()

        assert len(activations) == len(analyzer.muscle_names)
        # Activations should be in [0, 1]
        assert np.all(activations >= 0.0) and np.all(activations <= 1.0)

        logger.info(f"Muscle activations: {activations}")

    def test_muscle_force_computation(self, myosuite_env_available) -> None:
        """Section K: Compute muscle forces from actuators."""
        import gym

        env = gym.make("myoElbowPose1D6MRandom-v0")
        # Take a few steps to build up muscle forces
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)

        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim

        from src.engines.physics_engines.myosuite.python.muscle_analysis import (
            MyoSuiteMuscleAnalyzer,
        )

        analyzer = MyoSuiteMuscleAnalyzer(sim)

        # Get forces
        forces = analyzer.get_muscle_forces()

        assert len(forces) == len(analyzer.muscle_names)
        # At least some muscles should have non-zero force
        assert np.any(forces != 0.0), "All muscle forces are zero"

        logger.info(f"Muscle forces: {forces}")

    def test_moment_arm_computation(self, myosuite_env_available) -> None:
        """Section K: Compute moment arms via finite differences."""
        import gym

        env = gym.make("myoElbowPose1D6MRandom-v0")
        env.reset()

        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim

        from src.engines.physics_engines.myosuite.python.muscle_analysis import (
            MyoSuiteMuscleAnalyzer,
        )

        analyzer = MyoSuiteMuscleAnalyzer(sim)

        # Compute moment arms
        moment_arms = analyzer.compute_moment_arms()

        assert len(moment_arms) == len(analyzer.muscle_names)

        # Log moment arm values
        for muscle_name, r in list(moment_arms.items())[:3]:  # First 3 muscles
            logger.info(f"Moment arms for {muscle_name}: {r}")

    def test_muscle_induced_acceleration(self, myosuite_env_available) -> None:
        """Section K: Compute muscle-induced accelerations."""
        import gym

        env = gym.make("myoElbowPose1D6MRandom-v0")
        env.reset()

        # Apply some muscle activation
        for _ in range(5):
            action = np.ones(env.action_space.shape) * 0.5  # 50% activation
            env.step(action)

        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim

        from src.engines.physics_engines.myosuite.python.muscle_analysis import (
            MyoSuiteMuscleAnalyzer,
        )

        analyzer = MyoSuiteMuscleAnalyzer(sim)

        # Compute induced accelerations
        induced = analyzer.compute_muscle_induced_accelerations()

        assert len(induced) == len(analyzer.muscle_names)

        # At least some muscles should produce non-zero acceleration
        non_zero_count = sum(1 for a in induced.values() if not np.allclose(a, 0.0))
        assert non_zero_count > 0, "All induced accelerations are zero"

        logger.info(f"Non-zero induced accelerations: {non_zero_count}/{len(induced)}")

    def test_comprehensive_muscle_analysis(self, myosuite_env_available) -> None:
        """Section K: Full muscle contribution report."""
        import gym

        env = gym.make("myoElbowPose1D6MRandom-v0")
        env.reset()

        # Apply activation
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)

        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim

        from src.engines.physics_engines.myosuite.python.muscle_analysis import (
            MyoSuiteMuscleAnalyzer,
        )

        analyzer = MyoSuiteMuscleAnalyzer(sim)

        # Full analysis
        analysis = analyzer.analyze_all()

        # Verify all fields populated
        assert analysis.muscle_state is not None
        assert len(analysis.muscle_state.muscle_names) > 0
        assert len(analysis.moment_arms) > 0
        assert len(analysis.joint_torques) > 0
        assert len(analysis.total_muscle_torque) > 0

        logger.info("Analysis complete:")
        logger.info(f"  Muscles: {len(analysis.muscle_state.muscle_names)}")
        logger.info(f"  Total torque: {analysis.total_muscle_torque}")
        logger.info(
            f"  Activation power: {list(analysis.activation_power.values())[:3]}"
        )


class TestMyoSuiteGripModel:
    """Test grip modeling via hand muscles."""

    def test_grip_muscle_identification(self, myosuite_env_available) -> None:
        """Section K1: Identify hand/finger muscles."""
        import gym

        # Use hand environment if available; fallback to elbow
        try:
            env = gym.make("myoHandPoseRandom-v0")
        except Exception:  # noqa: BLE001
            env = gym.make("myoElbowPose1D6MRandom-v0")

        env.reset()
        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim

        from src.engines.physics_engines.myosuite.python.muscle_analysis import (
            MyoSuiteGripModel,
            MyoSuiteMuscleAnalyzer,
        )

        analyzer = MyoSuiteMuscleAnalyzer(sim)
        grip_model = MyoSuiteGripModel(sim, analyzer)

        # Get grip muscles
        grip_muscles = grip_model.get_grip_muscles()

        logger.info(f"Grip muscles found: {grip_muscles}")

        # May or may not have grip muscles depending on model
        # Just verify interface works
        assert isinstance(grip_muscles, list)

    def test_total_grip_force_computation(self, myosuite_env_available) -> None:
        """Section K1: Compute total grip force."""
        import gym

        try:
            env = gym.make("myoHandPoseRandom-v0")
        except Exception:  # noqa: BLE001
            pytest.skip("Hand model not available")

        env.reset()

        # Apply grip activation
        for _ in range(10):
            action = np.ones(env.action_space.shape) * 0.7  # 70% grip
            env.step(action)

        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim

        from src.engines.physics_engines.myosuite.python.muscle_analysis import (
            MyoSuiteGripModel,
            MyoSuiteMuscleAnalyzer,
        )

        analyzer = MyoSuiteMuscleAnalyzer(sim)
        grip_model = MyoSuiteGripModel(sim, analyzer)

        # Compute grip force
        total_force = grip_model.compute_total_grip_force()

        logger.info(f"Total grip force: {total_force:.1f} N")

        # Should be positive with activation
        assert total_force >= 0.0


class TestMyoSuiteEngine:
    """Test MyoSuite engine with muscle integration."""

    def test_drift_control_with_muscles(self, myosuite_env_available) -> None:
        """Section F + K: Verify drift-control works with muscle model."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        engine.load_from_path("myoElbowPose1D6MRandom-v0")

        # Compute drift (with zero muscle activation)
        a_drift = engine.compute_drift_acceleration()

        assert len(a_drift) > 0, "Drift acceleration is empty"

        logger.info(f"Drift acceleration: {a_drift}")

        # Compute control
        nv = len(a_drift)
        tau = np.ones(nv) * 0.5
        a_control = engine.compute_control_acceleration(tau)

        assert len(a_control) == nv

        logger.info(f"Control acceleration: {a_control}")

    def test_muscle_analyzer_integration(self, myosuite_env_available) -> None:
        """Section K: Verify engine provides muscle analyzer."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        engine.load_from_path("myoElbowPose1D6MRandom-v0")

        # Get analyzer
        analyzer = engine.get_muscle_analyzer()

        assert analyzer is not None, "Analyzer should be available"
        assert len(analyzer.muscle_names) > 0

        logger.info(f"Analyzer muscles: {analyzer.muscle_names}")

    def test_muscle_activation_setting(self, myosuite_env_available) -> None:
        """Section K: Set muscle activations by name."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        engine.load_from_path("myoElbowPose1D6MRandom-v0")

        # Get muscle names
        analyzer = engine.get_muscle_analyzer()
        if analyzer is None:
            pytest.skip("No muscles available")

        if not hasattr(analyzer, "muscle_names") or len(analyzer.muscle_names) == 0:
            pytest.skip("No muscles available")

        # Set activation for first muscle
        muscle_name = analyzer.muscle_names[0]
        engine.set_muscle_activations({muscle_name: 0.8})

        logger.info(f"Set {muscle_name} activation to 0.8")

        # Verify it was set (by checking control vector)
        if hasattr(analyzer, "muscle_actuator_ids") and analyzer.muscle_actuator_ids:
            actuator_id = analyzer.muscle_actuator_ids[0]
            ctrl_value = engine.sim.data.ctrl[actuator_id]
            assert 0.7 <= ctrl_value <= 0.9, (
                f"Activation not set correctly: {ctrl_value}"
            )


class TestCrossValidation:
    """Cross-validation with OpenSim."""

    @pytest.mark.xfail(
        strict=False, reason="Cross-validation test pending matching models"
    )
    def test_muscle_force_comparison(self) -> None:
        """Section K2: Compare MyoSuite vs OpenSim muscle forces."""
        # This test requires both engines with comparable models
        # Placeholder for future cross-validation
        logger.info("Cross-validation: Placeholder for MyoSuite ↔ OpenSim comparison")
        raise NotImplementedError("Cross-validation requires matching models")

    @pytest.mark.xfail(strict=False, reason="Pending multi-engine grip models")
    def test_grip_force_validation(self) -> None:
        """Section K1 + J1: Compare grip forces across engines."""
        # Grip force should agree within ±15% (Section K2)
        logger.info("Grip cross-validation: Placeholder")
        raise NotImplementedError("Pending multi-engine grip models")
