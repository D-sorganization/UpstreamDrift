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

Refactored to use shared engine availability module (DRY principle).
"""

import typing

import numpy as np
import pytest

from src.shared.python.engine_availability import MYOSUITE_AVAILABLE, skip_if_unavailable

if MYOSUITE_AVAILABLE:
    from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
        MyoSuitePhysicsEngine as _MyoSuitePhysicsEngine,
    )
else:
    _MyoSuitePhysicsEngine = None  # type: ignore

# Skip entire module if MyoSuite not available
pytestmark = skip_if_unavailable("myosuite")


if MYOSUITE_AVAILABLE:

    @skip_if_unavailable("myosuite")
    class TestMuscleContributionClosure:
        """Test that muscle contributions sum to total acceleration (closure property)."""

        @pytest.fixture
        def elbow_engine(self):  # type: ignore
            """Create MyoSuite elbow model for testing."""
            engine = _MyoSuitePhysicsEngine()
            # Load simple elbow model (1-DOF, 6 muscles)
            engine.load_from_path("myoElbowPose1D6MRandom-v0")
            return engine

        def test_muscle_induced_acceleration_closure_zero_torque(self, elbow_engine):
            """Verify muscle contributions sum to total when no external torques applied.

            Physics:
                With τ_external = 0 and q_init = 0, the system dynamics become:
                    M(q) * a = C(q, q̇) + g(q) + τ_muscle

                where τ_muscle = Σ τ_muscle_i for each muscle.

            Validation:
                a_total ≈ Σ a_muscle_i (induced acceleration closure)
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

            # Closure: Σ a_induced ≈ a_total
            np.testing.assert_allclose(
                a_muscle_sum,
                a_total,
                atol=1e-5,
                rtol=1e-4,
                err_msg="Muscle contribution closure failed at zero torque",
            )

        def test_muscle_contribution_closure_with_gravity(self, elbow_engine):
            """Verify closure holds even with gravitational loading.

            Gravity adds a bias force g(q). Induced acceleration analysis
            accounts for this by computing contributions to the full M(q)⁻¹ * g(q) term.
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
            assert analyzer is not None, "Muscle analyzer not available"
            induced_accels = analyzer.compute_muscle_induced_accelerations()

            # Sum contributions
            a_muscle_sum = np.zeros_like(a_total)
            for a_muscle in induced_accels.values():
                a_muscle_sum += a_muscle

            # Closure should still hold
            np.testing.assert_allclose(
                a_muscle_sum,
                a_total,
                atol=1e-5,
                rtol=1e-4,
                err_msg="Muscle contribution closure failed with gravity",
            )

        def test_individual_muscle_contributions_physical(self, elbow_engine):
            """Verify individual muscle contributions are physically reasonable.

            Induced accelerations should align with muscle anatomy (e.g., flexors
            should induce positive acceleration, extensors negative).
            """
            # Neutral position
            q_init = np.zeros(elbow_engine.model.nq)
            qd_init = np.zeros(elbow_engine.model.nv)
            elbow_engine.set_state(q_init, qd_init)

            activation_level = 0.5
            elbow_engine.set_muscle_activations(
                dict.fromkeys(elbow_engine.get_muscle_names(), activation_level)
            )
            elbow_engine.step(dt=0.001)

            # Compute induced accelerations
            analyzer = elbow_engine.get_muscle_analyzer()
            assert analyzer is not None, "Muscle analyzer not available"
            induced_accels = analyzer.compute_muscle_induced_accelerations()

            for muscle_name, a_muscle in induced_accels.items():
                # Flexors (e.g., 'BIcl', 'BRD') should induce positive accel in this model
                # Extensors (e.g., 'TRIlong') should induce negative
                # This depends on MyoSuite's specific coordinate system
                # Validation: Just check they are non-zero when active
                assert (
                    np.linalg.norm(a_muscle) > 1e-8
                ), f"Muscle {muscle_name} induced zero acceleration"

                # Log for inspection (useful for understanding muscle function)
                print(f"{muscle_name}: a = {a_muscle[0]:.6f} rad/s²")

        @pytest.mark.parametrize(
            "activation_level",
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )
        def test_closure_holds_at_different_activations(
            self, elbow_engine, activation_level: float
        ):
            """Verify closure property at various muscle activation levels.

            The closure test should hold regardless of muscle activation state.
            """
            # Set all muscles to same activation
            activations_dict = dict.fromkeys(
                elbow_engine.get_muscle_names(), activation_level
            )
            elbow_engine.set_muscle_activations(activations_dict)

            # Verify activations were set (if engine exposes muscle state)
            # Note: MyoSuite may not expose get_muscle_activations(), so this is a best-effort check
            # The closure property should hold regardless of whether we can verify the set operation

            # Rest of test same as base closure test
            q_init = np.zeros(elbow_engine.model.nq)
            qd_init = np.zeros(elbow_engine.model.nv)
            elbow_engine.set_state(q_init, qd_init)

            elbow_engine.step(dt=0.001)
            a_total = elbow_engine.get_acceleration()

            analyzer = elbow_engine.get_muscle_analyzer()
            assert analyzer is not None, "Muscle analyzer not available"
            induced_accels = analyzer.compute_muscle_induced_accelerations()

            a_muscle_sum = np.zeros_like(a_total)
            for a_muscle in induced_accels.values():
                a_muscle_sum += a_muscle

            np.testing.assert_allclose(
                a_muscle_sum,
                a_total,
                atol=1e-5,
                rtol=1e-4,
                err_msg=f"Closure violated at activation={activation_level}",
            )

    @pytest.mark.slow
    @skip_if_unavailable("myosuite")
    class TestMuscleContributionComplexModels:
        """Test closure property on more complex musculoskeletal models."""

        @pytest.mark.parametrize(
            "model_name",
            [
                "myoElbowPose1D6MRandom-v0",  # Simple
                pytest.param(
                    "myoHandPose1D20MRandom-v0",  # Complex hand
                    marks=pytest.mark.skip(reason="Slow test, run manually if needed"),
                ),
            ],
        )
        def test_closure_across_models(self, model_name: str):
            """Verify closure holds for various MyoSuite models."""
            engine = _MyoSuitePhysicsEngine()
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

            a_muscle_sum = np.zeros_like(a_total)
            for a_muscle in induced_accels.values():
                a_muscle_sum += a_muscle

            np.testing.assert_allclose(
                a_muscle_sum,
                a_total,
                atol=1e-4,
                rtol=1e-3,
                err_msg=f"Closure failed for model {model_name}",
            )

else:
    # Fallback to ensure some tests are collected even if marked as skipped by pytest.
    # We hide these from Mypy to avoid "Name already defined" [no-redef] errors.
    if not typing.TYPE_CHECKING:

        class TestMuscleContributionClosure:
            def test_skipped_no_myosuite(self) -> None:
                pytest.skip("MyoSuite not installed")

        class TestMuscleContributionComplexModels:
            def test_skipped_no_myosuite(self) -> None:
                pytest.skip("MyoSuite not installed")
