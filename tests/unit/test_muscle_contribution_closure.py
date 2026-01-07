"""Tests for muscle contribution closure in induced acceleration analysis.

Scientific Background:
    Induced Acceleration Analysis decomposes total acceleration into contributions:
        a_total = Σ a_muscle_i + a_passive + a_external

    This test verifies the fundamental closure property: the sum of all muscle-induced
    accelerations should equal the total acceleration (when no external torques applied).

References:
    - Zajac, F. E. (2002). "Understanding muscle coordination of the human leg with
      dynamical simulations." Journal of Biomechanics.
    - Anderson, F. C., & Pandy, M. G. (2003). "Individual muscle contributions to
      support in normal walking." Gait & Posture.
"""

import numpy as np
import pytest

from engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine,
)


class TestMuscleContributionClosure:
    """Test that muscle contributions sum to total acceleration (closure property)."""

    @pytest.fixture
    def elbow_engine(self) -> MyoSuitePhysicsEngine:
        """Create MyoSuite elbow model for testing."""
        engine = MyoSuitePhysicsEngine()
        # Load simple elbow model (1-DOF, 6 muscles)
        engine.load_from_path("myoElbowPose1D6MRandom-v0")
        return engine

    def test_muscle_induced_acceleration_closure_zero_torque(
        self, elbow_engine: MyoSuitePhysicsEngine
    ):
        """Verify muscle contributions sum to total when no external torques applied.

        Physics:
            With τ_external = 0 and q_init = 0, the system dynamics become:
                M(q) * a = C(q, q̇) + g(q) + τ_muscle

            where τ_muscle = Σ τ_muscle_i for each muscle.

            The induced acceleration of muscle i is defined as:
                a_muscle_i = M(q)^-1 * τ_muscle_i

            Closure property requires:
                Σ a_muscle_i = a_total
        """
        # Set initial state (neutral pose, zero velocity)
        q_init = np.zeros(elbow_engine.model.nq)
        qd_init = np.zeros(elbow_engine.model.nv)
        elbow_engine.set_state(q_init, qd_init)

        # Apply zero external control
        elbow_engine.set_control(np.zeros(elbow_engine.model.nu))

        # Get total acceleration from forward dynamics
        elbow_engine.step(dt=0.001)
        a_total = elbow_engine.get_acceleration()

        # Get muscle analyzer
        analyzer = elbow_engine.get_muscle_analyzer()
        assert analyzer is not None, "Muscle analyzer not available"

        # Compute muscle-induced accelerations
        induced_accels = analyzer.compute_muscle_induced_accelerations()

        # Sum all muscle contributions
        a_muscle_sum = np.zeros_like(a_total)
        for _muscle_name, a_muscle in induced_accels.items():
            a_muscle_sum += a_muscle

        # Verify closure: Σ a_muscle_i ≈ a_total
        np.testing.assert_allclose(
            a_muscle_sum,
            a_total,
            atol=1e-5,  # 10 µrad/s² tolerance
            rtol=1e-4,  # 0.01% relative tolerance
            err_msg=f"Muscle contributions don't sum to total acceleration.\\n"
            f"Sum of muscle accelerations: {a_muscle_sum}\\n"
            f"Total acceleration: {a_total}\\n"
            f"Difference: {a_muscle_sum - a_total}"
            f"\\nThis violates the fundamental closure property of induced acceleration analysis.",
        )

    def test_muscle_contribution_closure_with_gravity(
        self, elbow_engine: MyoSuitePhysicsEngine
    ):
        """Verify closure holds even with gravitational loading.

        This test ensures the decomposition works correctly when passive forces
        (gravity) are present.
        """
        # Set initial state at 90° elbow flexion (gravity effects significant)
        q_init = np.array([np.pi / 2])  # 90 degrees
        qd_init = np.zeros(elbow_engine.model.nv)
        elbow_engine.set_state(q_init, qd_init)

        # Zero external control
        elbow_engine.set_control(np.zeros(elbow_engine.model.nu))

        # Forward dynamics
        elbow_engine.step(dt=0.001)
        a_total = elbow_engine.get_acceleration()

        # Muscle-induced accelerations
        analyzer = elbow_engine.get_muscle_analyzer()
        induced_accels = analyzer.compute_muscle_induced_accelerations()

        # Sum contributions
        a_muscle_sum = sum(induced_accels.values())

        # Closure should still hold
        np.testing.assert_allclose(
            a_muscle_sum,
            a_total,
            atol=1e-4,  # Slightly looser tolerance due to numerical errors with gravity
            rtol=1e-3,
            err_msg="Closure property violated under gravitational loading",
        )

    def test_individual_muscle_contributions_physical(
        self, elbow_engine: MyoSuitePhysicsEngine
    ):
        """Verify individual muscle contributions are physically reasonable.

        Each muscle should produce acceleration in its expected direction based on
        anatomy (flexors → positive acceleration, extensors → negative).
        """
        # Neutral position
        q_init = np.zeros(elbow_engine.model.nq)
        qd_init = np.zeros(elbow_engine.model.nv)
        elbow_engine.set_state(q_init, qd_init)

        # Compute induced accelerations
        analyzer = elbow_engine.get_muscle_analyzer()
        assert analyzer is not None, "Muscle analyzer not available"
        induced_accels = analyzer.compute_muscle_induced_accelerations()

        # Verify each muscle produces non-zero acceleration
        for muscle_name, a_muscle in induced_accels.items():
            assert not np.allclose(
                a_muscle, 0, atol=1e-10
            ), f"Muscle {muscle_name} produces zero acceleration (check muscle activation)"

            # Log for inspection (useful for understanding muscle function)
            print(f"{muscle_name}: a = {a_muscle[0]:.6f} rad/s²")

    @pytest.mark.parametrize(
        "activation_level",
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    def test_closure_holds_at_different_activations(
        self, elbow_engine: MyoSuitePhysicsEngine, activation_level: float
    ):
        """Verify closure property at various muscle activation levels.

        The closure test should hold regardless of muscle activation state.
        """
        # Set all muscles to same activation
        elbow_engine.set_muscle_activations(
            dict.fromkeys(elbow_engine.get_muscle_names(), activation_level)
        )

        # Rest of test same as base closure test
        q_init = np.zeros(elbow_engine.model.nq)
        qd_init = np.zeros(elbow_engine.model.nv)
        elbow_engine.set_state(q_init, qd_init)

        elbow_engine.step(dt=0.001)
        a_total = elbow_engine.get_acceleration()

        analyzer = elbow_engine.get_muscle_analyzer()
        assert analyzer is not None, "Muscle analyzer not available"
        induced_accels = analyzer.compute_muscle_induced_accelerations()

        a_muscle_sum = sum(induced_accels.values())

        np.testing.assert_allclose(
            a_muscle_sum,
            a_total,
            atol=1e-5,
            rtol=1e-4,
            err_msg=f"Closure violated at activation={activation_level}",
        )


@pytest.mark.slow
class TestMuscleContributionComplexModels:
    """Test closure property on more complex musculoskeletal models."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "myoElbowPose1D6MRandom-v0",  # Simple
            pytest.param(
                "myoHandPose1D20MRandom-v0",  # Complex hand
                marks=pytest.mark.skipif(
                    "not config.getvalue('--run-slow')",
                    reason="Slow test, run with --run-slow",
                ),
            ),
        ],
    )
    def test_closure_across_models(self, model_name: str):
        """Verify closure holds for various MyoSuite models."""
        engine = MyoSuitePhysicsEngine()
        try:
            engine.load_from_path(model_name)
        except Exception as e:
            pytest.skip(f"Model {model_name} not available: {e}")

        # Standard closure test
        q_init = np.zeros(engine.model.nq)
        qd_init = np.zeros(engine.model.nv)
        engine.set_state(q_init, qd_init)

        engine.set_control(np.zeros(engine.model.nu))
        engine.step(dt=0.001)

        a_total = engine.get_acceleration()

        analyzer = engine.get_muscle_analyzer()
        assert analyzer is not None, "Muscle analyzer not available"
        induced_accels = analyzer.compute_muscle_induced_accelerations()

        a_muscle_sum = sum(induced_accels.values())

        np.testing.assert_allclose(
            a_muscle_sum,
            a_total,
            atol=1e-4,  # Looser for complex models
            rtol=1e-3,
            err_msg=f"Closure failed for {model_name}",
        )
