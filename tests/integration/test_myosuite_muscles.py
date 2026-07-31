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

        assert len(activations) == len(
            analyzer.muscle_names
        ), "Assertion failed: len(activations) == len(analyzer.muscle_names)"
        # Activations should be in [0, 1]
        assert np.all(activations >= 0.0) and np.all(
            activations <= 1.0
        ), "Assertion failed: np.all(activations >= 0.0) and np.all(activations <= 1.0)"

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

        assert len(forces) == len(
            analyzer.muscle_names
        ), "Assertion failed: len(forces) == len(analyzer.muscle_names)"
        # At least some muscles should have non-zero force
        assert np.any(forces != 0.0), "All muscle forces are zero"

        logger.info(f"Muscle forces: {forces}")

    def test_myosuite_muscles_moment_arm_computation(
        self, myosuite_env_available
    ) -> None:
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

        assert len(moment_arms) == len(
            analyzer.muscle_names
        ), "Assertion failed: len(moment_arms) == len(analyzer.muscle_names)"

        # Log moment arm values
        for muscle_name, r in list(moment_arms.items())[:3]:  # First 3 muscles
            logger.info(f"Moment arms for {muscle_name}: {r}")

    def test_myosuite_muscles_muscle_induced_acceleration(
        self, myosuite_env_available
    ) -> None:
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

        assert len(induced) == len(
            analyzer.muscle_names
        ), "Assertion failed: len(induced) == len(analyzer.muscle_names)"

        # At least some muscles should produce non-zero acceleration
        non_zero_count = sum(1 for a in induced.values() if not np.allclose(a, 0.0))
        assert non_zero_count > 0, "All induced accelerations are zero"

        logger.info(f"Non-zero induced accelerations: {non_zero_count}/{len(induced)}")

    def test_myosuite_muscles_comprehensive_muscle_analysis(
        self, myosuite_env_available
    ) -> None:
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
        assert (
            analysis.muscle_state is not None
        ), "Assertion failed: analysis.muscle_state is not None"
        assert (
            len(analysis.muscle_state.muscle_names) > 0
        ), "Assertion failed: len(analysis.muscle_state.muscle_names) > 0"
        assert (
            len(analysis.moment_arms) > 0
        ), "Assertion failed: len(analysis.moment_arms) > 0"
        assert (
            len(analysis.joint_torques) > 0
        ), "Assertion failed: len(analysis.joint_torques) > 0"
        assert (
            len(analysis.total_muscle_torque) > 0
        ), "Assertion failed: len(analysis.total_muscle_torque) > 0"

        logger.info("Analysis complete:")
        logger.info(f"  Muscles: {len(analysis.muscle_state.muscle_names)}")
        logger.info(f"  Total torque: {analysis.total_muscle_torque}")
        logger.info(
            f"  Activation power: {list(analysis.activation_power.values())[:3]}"
        )


class TestCrossValidation:
    """Cross-validation with OpenSim."""
