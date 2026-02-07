import unittest

from src.deployment.realtime.controller import (
    HardwareStubStrategy,
    RealTimeController,
    RobotConfig,
    SimulationStrategy,
)


class TestRealTimeController(unittest.TestCase):
    def test_simulation_strategy(self):
        """Test that RealTimeController uses SimulationStrategy for SIMULATION type."""
        controller = RealTimeController(communication_type="simulation")
        self.assertIsInstance(controller._strategy, SimulationStrategy)

        config = RobotConfig(name="test_bot", n_joints=6)
        connected = controller.connect(config)
        self.assertTrue(connected)

        # Should read zero state
        state = controller._read_state()
        self.assertEqual(len(state.joint_positions), 6)

    def test_ros2_strategy_raises(self):
        """Test that RealTimeController uses HardwareStubStrategy for ROS2 and raises NotImplementedError."""
        controller = RealTimeController(communication_type="ros2")
        self.assertIsInstance(controller._strategy, HardwareStubStrategy)
        self.assertEqual(controller._strategy.name, "ROS2")

        config = RobotConfig(name="test_bot", n_joints=6)
        # connect should return False (caught exception) or raise depending on implementation.
        # In my implementation:
        # try:
        #    if self._strategy.connect(robot_config):
        # ...
        # except Exception as e: ... return False

        connected = controller.connect(config)
        self.assertFalse(connected)

        # Direct strategy access should raise
        with self.assertRaises(NotImplementedError):
            controller._strategy.connect(config)


if __name__ == "__main__":
    unittest.main()
