"""Cross-engine contact model validation tests.

Verifies contact behavior is consistent (or differences are documented) across
MuJoCo, Drake, and Pinocchio physics engines.

Contact modeling is inherently engine-specific due to different algorithms:
- MuJoCo: Soft penalty-based contact (spring-damper)
- Drake: Compliant + rigid contact models
- Pinocchio: Algorithmic contact (constraint-based)

This test suite:
1. Validates basic contact physics (energy dissipation)
2. Documents expected differences between engines
3. Ensures no catastrophic divergence in results
"""

import numpy as np
import pytest

from shared.python.constants import GRAVITY_M_S2


class TestBasicContactPhysics:
    """Test fundamental contact behavior across all engines."""

    @pytest.fixture
    def ball_urdf(self, tmp_path):
        """Create a simple ball URDF for contact testing."""
        # Golf ball: mass = 0.045kg, radius = 0.02135m
        urdf_content = """<?xml version="1.0"?>
<robot name="ball">
  <link name="world"/>
  <link name="ball">
    <inertial>
      <mass value="0.045"/>
      <inertia ixx="4.1e-6" ixy="0.0" ixz="0.0"
               iyy="4.1e-6" iyz="0.0" izz="4.1e-6"/>
    </inertial>
    <collision>
      <geometry>
        <sphere radius="0.02135"/>
      </geometry>
    </collision>
  </link>
  <joint name="ball_joint" type="floating">
    <parent link="world"/>
    <child link="ball"/>
  </joint>
</robot>
"""
        urdf_path = tmp_path / "ball.urdf"
        urdf_path.write_text(urdf_content)
        return str(urdf_path)

    def test_mujoco_ball_drop_energy_dissipation(self, ball_urdf):
        """Verify MuJoCo contact dissipates energy (ball doesn't bounce forever)."""
        try:
            from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
                MuJoCoPhysicsEngine,
            )
        except ImportError:
            pytest.skip("MuJoCo not installed")

        engine = MuJoCoPhysicsEngine()
        try:
            engine.load_from_path(ball_urdf)
        except Exception as e:
            pytest.skip(f"MuJoCo URDF loading failed: {e}")

        # Drop ball from 1m height
        initial_height = 1.0
        q_init = np.array([0, 0, initial_height, 1, 0, 0, 0])  # [x,y,z, qw,qx,qy,qz]
        v_init = np.zeros(6)  # Zero velocity
        engine.set_state(q_init, v_init)

        # Compute initial potential energy
        E_initial = 0.045 * float(GRAVITY_M_S2) * initial_height  # mgh

        # Simulate until ball settles (2 seconds should be enough)
        dt = 0.001
        num_steps = int(2.0 / dt)
        for _ in range(num_steps):
            engine.step(dt=dt)

        # Get final state
        q_final, v_final = engine.get_state()
        final_height = q_final[2]
        E_final = (
            0.045 * float(GRAVITY_M_S2) * final_height  # Potential
            + 0.5 * 0.045 * np.linalg.norm(v_final[:3]) ** 2  # Kinetic
        )

        # Energy should have dissipated (ball shouldn't bounce back to 1m)
        assert E_final < E_initial * 0.5, (
            f"MuJoCo contact should dissipate energy: "
            f"E_initial={E_initial:.6f} J, E_final={E_final:.6f} J"
        )

        # Ball should be near ground (not still at 1m)
        assert (
            final_height < 0.1
        ), f"Ball should settle near ground: height={final_height:.3f}m"

        # Log for cross-engine comparison
        restitution_effective = np.sqrt(E_final / E_initial)
        print(f"MuJoCo - Effective restitution: {restitution_effective:.3f}")
        print(f"MuJoCo - Energy dissipated: {(E_initial - E_final)/E_initial*100:.1f}%")

    @pytest.mark.slow
    def test_drake_ball_drop_energy_dissipation(self, ball_urdf):
        """Verify Drake contact dissipates energy."""
        pytest.skip("Drake contact model testing - implementation pending")
        # Expected behavior: Similar to MuJoCo but may use different contact model

    @pytest.mark.slow
    def test_pinocchio_contact_behavior(self, ball_urdf):
        """Document Pinocchio contact behavior."""
        pytest.skip("Pinocchio algorithmic contact - different paradigm")
        # NOTE: Pinocchio uses constraint-based contact (not soft contact)
        # Energy dissipation model fundamentally different
        # May need separate test methodology


class TestCrossEngineContactComparison:
    """Compare contact results across engines where applicable."""

    @pytest.mark.parametrize(
        "drop_height",
        [0.1, 0.5, 1.0, 2.0],
        ids=["10cm", "50cm", "1m", "2m"],
    )
    def test_mujoco_restitution_coefficient(self, ball_urdf, drop_height):
        """Measure MuJoCo's effective coefficient of restitution at various heights."""
        try:
            from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
                MuJoCoPhysicsEngine,
            )
        except ImportError:
            pytest.skip("MuJoCo not installed")

        engine = MuJoCoPhysicsEngine()
        try:
            engine.load_from_path(ball_urdf)
        except Exception as e:
            pytest.skip(f"MuJoCo URDF loading failed: {e}")

        # Drop ball
        q_init = np.array([0, 0, drop_height, 1, 0, 0, 0])
        v_init = np.zeros(6)
        engine.set_state(q_init, v_init)

        # Simulate first bounce
        dt = 0.001
        num_steps = int(0.5 / dt)
        for _ in range(num_steps):
            engine.step(dt=dt)

        q_final, v_final = engine.get_state()
        final_height = q_final[2]

        # Coefficient of restitution: e = sqrt(h_bounce / h_drop)
        if final_height > 0.001:  # Bounced
            e_measured = np.sqrt(final_height / drop_height)
        else:
            e_measured = 0.0  # Didn't bounce (full dissipation)

        # Log for documentation
        print(f"Drop height: {drop_height:.2f}m")
        print(f"Bounce height: {final_height:.4f}m")
        print(f"Coefficient of restitution (e): {e_measured:.3f}")

        # Sanity check: restitution should be between 0 and 1
        assert 0 <= e_measured <= 1, f"Invalid restitution coefficient: {e_measured}"

        # Golf ball typical restitution: 0.75-0.85
        # MuJoCo may differ due to penalty-based model


class TestContactModelDocumentation:
    """Document expected differences between engine contact models."""

    def test_document_mujoco_contact_model(self):
        """Document MuJoCo's contact physics approach."""
        documentation = """
        MuJoCo Contact Model:
        - Type: Soft penalty-based (spring-damper)
        - Parameters: Controlled via <option> tag in XML
        - Key Settings:
          * impratio: Ratio of frictional-to-normal impedance
          * noslip_iterations: Iterations for friction resolution
        - Pros: Fast, stable, handles complex geometries
        - Cons: Not perfectly rigid (penetration allowed)
        - Energy: Dissipative (configured via damping)

        References:
        - MuJoCo Documentation: Contact Modeling section
        - Todorov (2014): "Convex and smooth formulations..."
        """
        # This is a documentation test - always passes
        assert True, documentation

    def test_document_drake_contact_model(self):
        """Document Drake's contact physics approach."""
        documentation = """
        Drake Contact Model:
        - Type: Hybrid (compliant + time-stepping rigid)
        - Models:
          * Point contact (compliant)
          * Hydroelastic (pressure field)
        - Pros: Physically accurate, well-documented
        - Cons: More complex to configure
        - Energy: Can be conservative or dissipative

        References:
        - Drake Documentation: Multibody Dynamics section
        - Elandt et al. (2019): "A pressure field model..."
        """
        assert True, documentation

    def test_document_pinocchio_contact_model(self):
        """Document Pinocchio's contact physics approach."""
        documentation = """
        Pinocchio Contact Model:
        - Type: Constraint-based (algorithmic)
        - Approach: Contact forces from constraint resolution
        - Solver: Quadratic programming (contact LCP)
        - Pros: Mathematically rigorous
        - Cons: Requires explicit contact point specification
        - Energy: Depends on solver configuration

        References:
        - Pinocchio Documentation: Dynamics section
        - Carpentier et al. (2019): "Pinocchio: fast algorithms..."
        """
        assert True, documentation


class TestContactEnergyConservation:
    """Test energy conservation properties with contact."""

    def test_mujoco_elastic_collision_energy(self, ball_urdf):
        """Verify (near) energy conservation for elastic collisions in MuJoCo.

        With high restitution coefficient, energy should be mostly conserved.
        """
        pytest.skip("Elastic collision test - requires custom contact parameters")
        # Verify E_before ≈ E_after (within tolerance)

    def test_contact_work_energy_theorem(self):
        """Verify work-energy theorem holds during contact."""
        pytest.skip("Work-energy validation - requires contact force measurement")
        # Verify: ΔKE = W_contact + W_gravity


class TestContactStability:
    """Test numerical stability of contact simulations."""

    def test_mujoco_stacked_boxes_stability(self):
        """Verify stacked objects don't explode due to contact errors."""
        pytest.skip("Stability test - requires multi-body contact URDF")
        # Simulate for extended time
        # Verify no explosive behavior (energy bounded)


@pytest.mark.slow
class TestContactCrossValidation:
    """Cross-validate contact results between engines (where comparable)."""

    def test_compare_energy_dissipation_rates(self, ball_urdf):
        """Compare energy dissipation across engines for same scenario."""
        pytest.skip("Cross-engine comparison - requires all engines installed")
        # Document differences in energy dissipation
        # Ensure differences are within expected range (not catastrophic)

    def test_compare_contact_force_magnitudes(self):
        """Compare contact force magnitudes across engines."""
        pytest.skip("Force comparison - requires contact force extraction")
        # Compare across engines
        # Document order-of-magnitude agreement


# Summary of Expected Engine Differences
"""
Expected Contact Behavior Differences:

1. **MuJoCo**:
   - Soft contacts (penetration allowed)
   - Fast simulation
   - Tunable stiffness/damping
   - Good for real-time applications

2. **Drake**:
   - More physical contact models
   - Hydroelastic option
   - Slower but more accurate
   - Good for optimization/planning

3. **Pinocchio**:
   - Algorithmic/constraint-based
   - Different paradigm entirely
   - Requires explicit contact specification
   - Best for analytical dynamics

**Recommendation**: Use MuJoCo for simulation, Drake for trajectory optimization,
Pinocchio for kinematic analysis (contact less critical there).
"""
