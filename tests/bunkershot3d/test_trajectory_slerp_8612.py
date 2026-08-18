"""Regression tests for quaternion interpolation (#8612, finding B8).

``SwingTrajectory.interpolate`` used component-wise ``np.interp`` followed by a
renormalise (nlerp) with no antipodal sign fixing. Because ``q`` and ``-q`` are
the same rotation, a trajectory whose stored quaternions cross a hemisphere
interpolates through the *long* arc, and a near-antipodal pair collapses to a
near-zero-norm quaternion that renormalises to noise.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bunkershot3d.kinematics.trajectory import SwingTrajectory, slerp

pytestmark = pytest.mark.unit


def _quat_about_y(angle_deg: float) -> np.ndarray:
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0])


def _angle_about_y(quat: np.ndarray) -> float:
    """Signed rotation angle (deg) of a quaternion known to be about +y."""
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:  # canonical representative
        quat = -quat
    return float(math.degrees(2.0 * math.atan2(quat[2], quat[0])))


def _write_trajectory(path: Path, times: np.ndarray, quats: np.ndarray) -> Path:
    n = len(times)
    zeros = np.zeros(n)
    pd.DataFrame(
        {
            "time": times,
            "px": zeros,
            "py": zeros,
            "pz": zeros,
            "qw": quats[:, 0],
            "qx": quats[:, 1],
            "qy": quats[:, 2],
            "qz": quats[:, 3],
            "vx": zeros,
            "vy": zeros,
            "vz": zeros,
            "wx": zeros,
            "wy": zeros,
            "wz": zeros,
        }
    ).to_csv(path, index=False)
    return path


class TestSlerpPrimitive:
    def test_endpoints_are_exact(self) -> None:
        q0 = _quat_about_y(10.0)
        q1 = _quat_about_y(80.0)
        np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-12)
        np.testing.assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-12)

    def test_midpoint_is_the_half_angle(self) -> None:
        q0 = _quat_about_y(10.0)
        q1 = _quat_about_y(80.0)
        assert _angle_about_y(slerp(q0, q1, 0.5)) == pytest.approx(45.0, abs=1e-6)

    def test_constant_angular_rate(self) -> None:
        q0 = _quat_about_y(0.0)
        q1 = _quat_about_y(120.0)
        for fraction in (0.25, 0.5, 0.75):
            got = _angle_about_y(slerp(q0, q1, fraction))
            assert got == pytest.approx(120.0 * fraction, abs=1e-6)

    def test_antipodal_representation_takes_the_short_arc(self) -> None:
        """q and -q are the same rotation; nlerp took the long way round."""
        q0 = _quat_about_y(0.0)
        q1 = -_quat_about_y(120.0)
        assert _angle_about_y(slerp(q0, q1, 0.5)) == pytest.approx(60.0, abs=1e-6)

    def test_result_is_always_unit_norm(self) -> None:
        q0 = _quat_about_y(0.0)
        q1 = -np.array([1.0, 1.0e-9, 0.0, 0.0])
        for fraction in np.linspace(0.0, 1.0, 11):
            quat = slerp(q0, q1, float(fraction))
            assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-9)


class TestTrajectoryInterpolation:
    def test_hemisphere_crossing_uses_the_short_arc(self, tmp_path: Path) -> None:
        """The decisive case: successive samples stored with opposite signs."""
        times = np.array([0.0, 1.0])
        quats = np.array([_quat_about_y(0.0), -_quat_about_y(179.0)])
        traj = SwingTrajectory.from_csv(
            _write_trajectory(tmp_path / "t.csv", times, quats)
        )

        _pos, quat, _lv, _av = traj.interpolate(0.5)

        assert _angle_about_y(quat) == pytest.approx(89.5, abs=1e-3), (
            "component-wise np.interp interpolated through the long arc"
        )

    def test_near_antipodal_pair_stays_unit_norm(self, tmp_path: Path) -> None:
        times = np.array([0.0, 1.0])
        quats = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 1.0e-9, 0.0, 0.0]])
        traj = SwingTrajectory.from_csv(
            _write_trajectory(tmp_path / "t.csv", times, quats)
        )

        _pos, quat, _lv, _av = traj.interpolate(0.5)

        assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-9)
        assert abs(abs(float(quat[0])) - 1.0) < 1e-6

    def test_sign_continuity_across_many_segments(self, tmp_path: Path) -> None:
        """Alternating stored signs must not produce a sawtooth rotation."""
        angles = np.linspace(0.0, 160.0, 9)
        quats = np.array(
            [
                (-1.0) ** index * _quat_about_y(float(angle))
                for index, angle in enumerate(angles)
            ]
        )
        times = np.linspace(0.0, 1.0, 9)
        traj = SwingTrajectory.from_csv(
            _write_trajectory(tmp_path / "t.csv", times, quats)
        )

        sampled = [
            _angle_about_y(traj.interpolate(float(t))[1])
            for t in np.linspace(0.0, 1.0, 33)
        ]
        assert np.all(np.diff(sampled) > -1e-6), (
            f"rotation angle is not monotone: {sampled}"
        )

    def test_matches_linear_interpolation_for_positions(self, tmp_path: Path) -> None:
        times = np.array([0.0, 1.0])
        quats = np.array([_quat_about_y(0.0), _quat_about_y(30.0)])
        path = _write_trajectory(tmp_path / "t.csv", times, quats)
        frame = pd.read_csv(path)
        frame["px"] = [0.0, 2.0]
        frame.to_csv(path, index=False)

        traj = SwingTrajectory.from_csv(path)
        pos, _quat, _lv, _av = traj.interpolate(0.25)
        assert pos[0] == pytest.approx(0.5)


class TestTrajectoryDerivedQuantities:
    def test_duration_and_max_speed(self, tmp_path: Path) -> None:
        times = np.array([0.0, 0.5, 1.0])
        quats = np.tile(_quat_about_y(0.0), (3, 1))
        path = _write_trajectory(tmp_path / "t.csv", times, quats)
        frame = pd.read_csv(path)
        frame["vx"] = [1.0, 25.0, 3.0]
        frame.to_csv(path, index=False)

        traj = SwingTrajectory.from_csv(path)
        assert traj.duration == pytest.approx(1.0)
        assert traj.max_linear_speed() == pytest.approx(25.0)
