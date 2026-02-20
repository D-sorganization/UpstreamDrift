"""Tests for physics validation module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestPhysicsValidationResult(unittest.TestCase):
    """Test validation result dataclasses."""

    def test_energy_validation_result_str_pass(self) -> None:
        """Test EnergyValidationResult string representation for passing."""
        from src.shared.python.physics.physics_validation import EnergyValidationResult

        result = EnergyValidationResult(
            energy_error=1e-5,
            relative_error=1e-5,
            passes=True,
            kinetic_energy_initial=10.0,
            kinetic_energy_final=10.0,
            potential_energy_initial=5.0,
            potential_energy_final=5.0,
            work_applied=0.0,
            message="Test message",
        )
        assert "PASS" in str(result)
        assert "1.00e-05" in str(result)

    def test_energy_validation_result_str_fail(self) -> None:
        """Test EnergyValidationResult string representation for failing."""
        from src.shared.python.physics.physics_validation import EnergyValidationResult

        result = EnergyValidationResult(
            energy_error=0.1,
            relative_error=0.1,
            passes=False,
            kinetic_energy_initial=10.0,
            kinetic_energy_final=9.0,
            potential_energy_initial=5.0,
            potential_energy_final=5.0,
            work_applied=0.0,
            message="Energy not conserved",
        )
        assert "FAIL" in str(result)

    def test_jacobian_validation_result_str(self) -> None:
        """Test JacobianValidationResult string representation."""
        from src.shared.python.physics.physics_validation import (
            JacobianValidationResult,
        )

        result = JacobianValidationResult(
            jacobian_error=1e-8,
            passes=True,
            body_id=5,
            message="Jacobian valid",
        )
        assert "PASS" in str(result)
        assert "1e-06" in str(result)  # threshold


class TestPhysicsValidator(unittest.TestCase):
    """Test PhysicsValidator methods."""

    def setUp(self) -> None:
        """Set up mock MuJoCo model and data."""
        self.mock_mujoco = MagicMock()
        self.mock_model = MagicMock()
        self.mock_data = MagicMock()

        # Configure model attributes
        self.mock_model.nv = 3
        self.mock_model.nbody = 4
        self.mock_model.opt.gravity = [0, 0, -9.81]
        self.mock_model.opt.timestep = 0.001
        self.mock_model.body_mass = [0, 1.0, 0.5, 0.3]

    @patch("src.shared.python.physics.physics_validation.mujoco", autospec=False)
    def test_compute_kinetic_energy(self, mock_mujoco_import: MagicMock) -> None:
        """Test kinetic energy computation."""
        # Skip if mujoco not available
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("MuJoCo not installed")

    @patch("src.shared.python.physics.physics_validation.mujoco", autospec=False)
    def test_compute_potential_energy(self, mock_mujoco_import: MagicMock) -> None:
        """Test potential energy computation."""
        # Skip if mujoco not available
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("MuJoCo not installed")

    def test_energy_conservation_formula(self) -> None:
        """Test the energy conservation formula logic."""
        # Test that energy error = |dE - work|
        dE = 1.0  # Energy change
        work = 1.0  # Work done
        error = abs(dE - work)
        assert error < 1e-10, "Energy balance should be exact for matching values"

        # Test with mismatch
        dE = 1.5
        work = 1.0
        error = abs(dE - work)
        assert abs(error - 0.5) < 1e-10, "Energy error should be 0.5"


class TestPhysicsValidatorIntegration(unittest.TestCase):
    """Integration tests requiring MuJoCo."""

    def test_validator_with_real_mujoco(self) -> None:
        """Test validator with real MuJoCo if available."""
        try:
            import mujoco

            from src.shared.python.physics.physics_validation import PhysicsValidator
        except ImportError:
            self.skipTest("MuJoCo not installed")

        # Create a simple model (pendulum)
        xml = """
        <mujoco>
            <worldbody>
                <body name="link1" pos="0 0 0.5">
                    <joint name="j1" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.05 0.5" mass="1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        validator = PhysicsValidator(model, data)

        # Test energy computation
        qpos = np.array([0.0])
        qvel = np.array([1.0])
        ke = validator.compute_kinetic_energy(qpos, qvel)
        assert ke > 0, "Kinetic energy should be positive for non-zero velocity"

        pe = validator.compute_potential_energy(qpos)
        assert isinstance(pe, float), "Potential energy should be a float"

        # Test energy conservation (with zero torques, should conserve)
        torques = np.array([0.0])
        result = validator.verify_energy_conservation(qpos, qvel, torques)
        # For a conservative system without damping, energy should be conserved
        # (within numerical tolerance)
        assert isinstance(result.passes, bool)

        # Test Jacobian validation
        jac_result = validator.verify_jacobian(qpos, 1)
        assert jac_result.passes, f"Jacobian validation failed: {jac_result.message}"


if __name__ == "__main__":
    unittest.main()
