import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.verification import EnergyMonitor, JacobianTester


# Helper to create a simple pendulum model if none exists
def create_pendulum_model():
    xml = """
    <mujoco>
      <worldbody>
        <body name="pendulum" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .5 .5 0" density="1000"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


class TestVerificationEngine:
    """Test suite for Phase 2 Verification Tools."""

    @pytest.fixture
    def model_and_data(self):
        model = create_pendulum_model()
        data = mujoco.MjData(model)
        return model, data

    def test_energy_monitor_conservation(self, model_and_data):
        """Test EnergyMonitor correctly tracks conservation on a passive pendulum."""
        model, data = model_and_data
        monitor = EnergyMonitor(model, data)

        # Initial state: verify non-zero potential, zero kinetic
        data.qpos[0] = 1.0  # Lifted
        mujoco.mj_forward(model, data)
        monitor.record_step()  # Initial snapshot

        initial_total = monitor.history[0].total
        assert initial_total > 0

        # Simulate for a bit (passive)
        for _ in range(100):
            mujoco.mj_step(model, data)
            monitor.record_step()

        passed, drift = monitor.check_conservation(tolerance=0.01)

        # MuJoCo is generally conservative enough for this simple case
        assert passed, f"Energy drift too high: {drift}"
        assert len(monitor.history) > 100

    def test_energy_monitor_work_tracking(self, model_and_data):
        """Test that work input is correctly tracked."""
        model, data = model_and_data
        monitor = EnergyMonitor(model, data)

        # Apply constant torque
        torque = np.array([1.0])

        for _ in range(10):
            # Not strictly used by mj_step unless actuator defined
            # but we pass it to record_step manually
            data.ctrl[:] = torque

            # Manually apply torque to qfrc_applied to simulate actuation work
            data.qfrc_applied[:] = torque

            mujoco.mj_step(model, data)
            monitor.record_step(control_torques=torque)

        # Work should be positive as we are adding energy
        assert monitor.cumulative_work > 0

    def test_jacobian_tester(self, model_and_data):
        """Test that JacobianTester confirms analytical vs FD match."""
        model, _ = model_and_data
        tester = JacobianTester(model)

        # Test verification at a random configuration
        qpos_test = np.array([0.5])

        # Should pass for a built-in geometric primitive
        error = tester.check_body_jacobian("pendulum", qpos_test)

        assert error < 1e-5, f"Jacobian mismatch: {error}"


if __name__ == "__main__":
    pytest.main([__file__])
