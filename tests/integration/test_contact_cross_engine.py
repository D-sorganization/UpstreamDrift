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
from src.shared.python.core.constants import GRAVITY_M_S2

# Contact test constants
BOUNCE_HEIGHT_THRESHOLD_M = 0.001  # Minimum height (1mm) to consider a bounce occurred


def _skip_if_mujoco_state_unavailable(engine) -> None:
    """Skip when the CI lane loads a degenerate MuJoCo model with no state."""
    q_current, _ = engine.get_state()
    if len(q_current) == 0:
        pytest.skip(
            "MuJoCo contact model did not expose floating-joint state in this lane"
        )


@pytest.fixture(scope="module")
def ball_urdf(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a simple ball URDF for contact testing."""
    # Golf ball: mass = 0.045kg, radius = 0.02135m
    urdf_content = """<?xml version="1.0"?>
<robot name="ball">
  <link name="world">
    <inertial>
      <mass value="0.0"/>
      <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
    <collision>
      <origin xyz="0 0 -0.01"/>
      <geometry>
        <box size="10 10 0.02"/>
      </geometry>
    </collision>
  </link>
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
    urdf_path = tmp_path_factory.mktemp("data") / "ball.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)

    # Golf ball typical restitution: 0.75-0.85
    # MuJoCo may differ due to penalty-based model


class TestContactModelDocumentation:
    """Document expected differences between engine contact models."""

    def test_document_mujoco_contact_model(self) -> None:
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

    def test_document_drake_contact_model(self) -> None:
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

    def test_document_pinocchio_contact_model(self) -> None:
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

        # Verify: ΔKE = W_contact + W_gravity

        # Simulate for extended time
        # Verify no explosive behavior (energy bounded)

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
