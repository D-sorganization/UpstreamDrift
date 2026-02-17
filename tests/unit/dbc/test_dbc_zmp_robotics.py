"""DbC tests for ZMP computation and robotics locomotion.

Validates that:
- ZMPComputer invariants: GRAVITY > 0, engine not None
- ZMP result postconditions: z-component on ground, is_valid flag correct
- Capture point computation is finite and on ground plane
- Stability margin sign convention is correct
- Free-fall detection (zero denominator) returns invalid result
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

import numpy as np

os.environ["DBC_LEVEL"] = "enforce"


def _make_mock_engine(
    com_pos: np.ndarray | None = None,
    com_vel: np.ndarray | None = None,
    mass: float = 70.0,
) -> MagicMock:
    """Create a mock RoboticsCapable engine."""
    engine = MagicMock()
    engine.get_com_position.return_value = (
        com_pos if com_pos is not None else np.array([0.0, 0.0, 1.0])
    )
    engine.get_com_velocity.return_value = (
        com_vel if com_vel is not None else np.zeros(3)
    )
    engine.get_total_mass.return_value = mass
    return engine


class TestZMPComputerInvariants(unittest.TestCase):
    """ZMPComputer _get_invariants: GRAVITY > 0, engine not None."""

    def test_invariants_satisfied(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        engine = _make_mock_engine()
        zmp = ZMPComputer(engine)
        invariants = zmp._get_invariants()
        for condition, msg in invariants:
            self.assertTrue(condition(), msg)

    def test_gravity_constant_positive(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        engine = _make_mock_engine()
        zmp = ZMPComputer(engine)
        self.assertGreater(zmp.GRAVITY, 0.0)


class TestComputeZMPPostconditions(unittest.TestCase):
    """compute_zmp() postconditions."""

    def _make_computer(self):  # type: ignore[no-untyped-def]
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        return ZMPComputer(_make_mock_engine())

    def test_zmp_z_on_ground(self) -> None:
        """ZMP z-component must be at ground height."""
        computer = self._make_computer()
        result = computer.compute_zmp(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_acceleration=np.zeros(3),
        )
        self.assertAlmostEqual(result.zmp_position[2], 0.0)  # ground_height=0

    def test_cop_equals_zmp_flat_ground(self) -> None:
        """CoP == ZMP for flat ground."""
        computer = self._make_computer()
        result = computer.compute_zmp(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_acceleration=np.zeros(3),
        )
        np.testing.assert_array_almost_equal(
            result.zmp_position, result.cop_position
        )

    def test_quasi_static_zmp_under_com(self) -> None:
        """For zero acceleration, ZMP should be directly under CoM."""
        computer = self._make_computer()
        com = np.array([0.5, 0.3, 1.0])
        result = computer.compute_zmp(
            com_position=com,
            com_acceleration=np.zeros(3),
        )
        # With zero horizontal acceleration, ZMP x,y should equal CoM x,y
        self.assertAlmostEqual(result.zmp_position[0], com[0], places=6)
        self.assertAlmostEqual(result.zmp_position[1], com[1], places=6)

    def test_result_fields_finite(self) -> None:
        computer = self._make_computer()
        result = computer.compute_zmp(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_acceleration=np.zeros(3),
        )
        self.assertTrue(np.all(np.isfinite(result.zmp_position)))
        self.assertTrue(np.all(np.isfinite(result.cop_position)))
        self.assertTrue(np.isfinite(result.support_margin))
        self.assertTrue(np.isfinite(result.total_normal_force))


class TestFreeFallDetection(unittest.TestCase):
    """When vertical accel + gravity ≈ 0, ZMP is undefined."""

    def test_free_fall_gives_invalid(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        # Free fall: a_z = -g => a_z + g ≈ 0
        result = computer.compute_zmp(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_acceleration=np.array([0.0, 0.0, -computer.GRAVITY]),
        )
        self.assertFalse(result.is_valid)
        self.assertAlmostEqual(result.total_normal_force, 0.0, places=3)


class TestCapturePointPostconditions(unittest.TestCase):
    """capture point / DCM postconditions."""

    def test_capture_point_on_ground(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        cp = computer.compute_capture_point(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_velocity=np.array([0.5, 0.0, 0.0]),
        )
        self.assertAlmostEqual(cp[2], 0.0)  # On ground

    def test_capture_point_finite(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        cp = computer.compute_capture_point(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_velocity=np.array([0.5, 0.3, 0.0]),
        )
        self.assertTrue(np.all(np.isfinite(cp)))

    def test_stationary_capture_point_under_com(self) -> None:
        """If velocity is zero, capture point should be under CoM."""
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        com = np.array([0.5, 0.3, 1.0])
        cp = computer.compute_capture_point(
            com_position=com,
            com_velocity=np.zeros(3),
        )
        self.assertAlmostEqual(cp[0], com[0], places=6)
        self.assertAlmostEqual(cp[1], com[1], places=6)

    def test_capture_ahead_of_com_when_moving(self) -> None:
        """Capture point should be ahead of CoM in direction of motion."""
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        cp = computer.compute_capture_point(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_velocity=np.array([1.0, 0.0, 0.0]),
        )
        self.assertGreater(cp[0], 0.0)  # Ahead in X

    def test_dcm_equals_capture_point(self) -> None:
        """DCM should be identical to capture point for LIPM."""
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        com = np.array([0.0, 0.0, 1.0])
        vel = np.array([0.5, 0.3, 0.0])
        cp = computer.compute_capture_point(com, vel)
        dcm = computer.compute_dcm(com, vel)
        np.testing.assert_array_almost_equal(cp, dcm)


class TestStabilityMargin(unittest.TestCase):
    """Stability margin sign conventions."""

    def test_inside_polygon_positive_margin(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        # Point at origin, polygon around it
        polygon = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ])
        margin = computer.compute_stability_margin(
            np.array([0.0, 0.0]), polygon
        )
        self.assertGreater(margin, 0.0)

    def test_outside_polygon_negative_margin(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        polygon = np.array([
            [-0.1, -0.1],
            [0.1, -0.1],
            [0.1, 0.1],
            [-0.1, 0.1],
        ])
        margin = computer.compute_stability_margin(
            np.array([1.0, 0.0]), polygon
        )
        self.assertLess(margin, 0.0)

    def test_center_has_maximum_margin(self) -> None:
        """Center of a square polygon should have the largest margin."""
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        polygon = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ])
        m_center = computer.compute_stability_margin(
            np.array([0.0, 0.0]), polygon
        )
        m_edge = computer.compute_stability_margin(
            np.array([0.5, 0.0]), polygon
        )
        self.assertGreater(m_center, m_edge)


class TestGroundHeightProperty(unittest.TestCase):
    """ground_height property setter/getter."""

    def test_set_and_get(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        computer.ground_height = 0.5
        self.assertAlmostEqual(computer.ground_height, 0.5)

    def test_zmp_uses_custom_ground(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine(), ground_height=0.5)
        result = computer.compute_zmp(
            com_position=np.array([0.0, 0.0, 1.5]),
            com_acceleration=np.zeros(3),
        )
        self.assertAlmostEqual(result.zmp_position[2], 0.5)
        self.assertAlmostEqual(result.ground_height, 0.5)


class TestZMPAccelerationEffect(unittest.TestCase):
    """ZMP should shift opposite to horizontal acceleration."""

    def test_forward_accel_shifts_zmp_backward(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        result = computer.compute_zmp(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_acceleration=np.array([2.0, 0.0, 0.0]),
        )
        # Forward acceleration should shift ZMP backward (negative x)
        self.assertLess(result.zmp_position[0], 0.0)

    def test_lateral_accel_shifts_zmp(self) -> None:
        from src.robotics.locomotion.zmp_computer import ZMPComputer

        computer = ZMPComputer(_make_mock_engine())
        result = computer.compute_zmp(
            com_position=np.array([0.0, 0.0, 1.0]),
            com_acceleration=np.array([0.0, 2.0, 0.0]),
        )
        # Lateral acceleration should shift ZMP in opposite direction
        self.assertLess(result.zmp_position[1], 0.0)


if __name__ == "__main__":
    unittest.main()
