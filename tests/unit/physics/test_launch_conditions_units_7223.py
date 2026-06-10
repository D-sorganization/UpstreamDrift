"""Unit-convention regression tests for LaunchConditions (#7223).

BallFlightSimulator, ball_enhanced_simulator, and flight_models all read
``launch_angle`` as radians and ``spin_rate`` as RPM. The swing→flight
pipeline and the ball-flight GUI used to construct LaunchConditions with
degrees and rad/s, which drove the ball straight into the ground
(carry = 0). These tests pin the contract and the conversion seam.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.physics.ball_launch_conditions import LaunchConditions

pytestmark = [pytest.mark.unit]


class TestFromUserUnits:
    def test_converts_degrees_to_radians(self) -> None:
        lc = LaunchConditions.from_user_units(
            velocity=70.0, launch_angle_deg=12.0, azimuth_deg=3.0
        )
        assert lc.launch_angle == pytest.approx(math.radians(12.0))
        assert lc.azimuth_angle == pytest.approx(math.radians(3.0))

    def test_spin_rpm_passes_through(self) -> None:
        lc = LaunchConditions.from_user_units(
            velocity=70.0, launch_angle_deg=12.0, spin_rate_rpm=2700.0
        )
        assert lc.spin_rate == pytest.approx(2700.0)

    def test_default_spin_axis_is_backspin(self) -> None:
        lc = LaunchConditions.from_user_units(velocity=70.0, launch_angle_deg=12.0)
        assert np.allclose(lc.spin_axis, [0.0, -1.0, 0.0])

    def test_custom_spin_axis_preserved(self) -> None:
        lc = LaunchConditions.from_user_units(
            velocity=70.0,
            launch_angle_deg=12.0,
            spin_axis=np.array([1.0, 0.0, 0.0]),
        )
        assert np.allclose(lc.spin_axis, [1.0, 0.0, 0.0])

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_rejects_bad_velocity(self, bad: float) -> None:
        with pytest.raises(ValueError, match="velocity"):
            LaunchConditions.from_user_units(velocity=bad, launch_angle_deg=12.0)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_rejects_bad_spin(self, bad: float) -> None:
        with pytest.raises(ValueError, match="spin_rate_rpm"):
            LaunchConditions.from_user_units(
                velocity=70.0, launch_angle_deg=12.0, spin_rate_rpm=bad
            )


def _rust_available() -> bool:
    from src.shared.python.physics.rust_kernel import is_rust_available

    return bool(is_rust_available())


@pytest.mark.skipif(not _rust_available(), reason="upstream_physics kernel not built")
class TestTourDriverCarryIsPhysical:
    """A tour-driver launch must carry a sane distance, not 0.

    This is the exact failure mode the unit bug produced: with degrees
    fed where radians were expected, sin(launch_angle) went negative and
    carry collapsed to 0.
    """

    def test_correct_units_give_sane_carry(self) -> None:
        from src.shared.python.physics.ball_simulator import BallFlightSimulator

        lc = LaunchConditions.from_user_units(
            velocity=75.0, launch_angle_deg=11.0, spin_rate_rpm=2700.0
        )
        traj = BallFlightSimulator().simulate_trajectory(lc, max_time=12.0, dt=0.005)
        carry = float(np.hypot(traj[-1].position[0], traj[-1].position[1]))
        max_h = max(p.position[2] for p in traj)
        assert 180.0 < carry < 320.0, f"unphysical carry {carry:.1f} m"
        assert 20.0 < max_h < 120.0, f"unphysical apex {max_h:.1f} m"

    def test_raw_degrees_would_collapse_carry(self) -> None:
        # Documents the bug: passing degrees as raw radians collapses carry.
        from src.shared.python.physics.ball_simulator import BallFlightSimulator

        broken = LaunchConditions(velocity=75.0, launch_angle=11.0, spin_rate=283.0)
        traj = BallFlightSimulator().simulate_trajectory(broken, max_time=12.0, dt=0.01)
        carry = float(np.hypot(traj[-1].position[0], traj[-1].position[1]))
        assert carry < 10.0  # the old (wrong) behaviour
