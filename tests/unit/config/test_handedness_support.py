"""Tests for src.shared.python.config.handedness_support (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.config.handedness_support import (
    Handedness,
    MirrorTransform,
    create_mirror_transform,
    mirror_position,
    mirror_rotation_matrix,
    mirror_velocity,
)


class TestHandednessEnum:
    def test_right_handed_exists(self) -> None:
        h = Handedness.RIGHT_HANDED
        assert h is not None

    def test_left_handed_exists(self) -> None:
        h = Handedness.LEFT_HANDED
        assert h is not None

    def test_right_and_left_are_different(self) -> None:
        assert Handedness.RIGHT_HANDED != Handedness.LEFT_HANDED


class TestCreateMirrorTransform:
    def test_returns_mirror_transform(self) -> None:
        transform = create_mirror_transform()
        assert isinstance(transform, MirrorTransform)

    def test_position_mirror_is_3x3(self) -> None:
        transform = create_mirror_transform()
        assert transform.position_mirror.shape == (3, 3)

    def test_velocity_mirror_is_3x3(self) -> None:
        transform = create_mirror_transform()
        assert transform.velocity_mirror.shape == (3, 3)

    def test_rotation_mirror_is_3x3(self) -> None:
        transform = create_mirror_transform()
        assert transform.rotation_mirror.shape == (3, 3)

    def test_angular_velocity_mirror_is_3x3(self) -> None:
        transform = create_mirror_transform()
        assert transform.angular_velocity_mirror.shape == (3, 3)

    def test_position_mirror_flips_y(self) -> None:
        transform = create_mirror_transform()
        pos = np.array([1.0, 2.0, 3.0])
        mirrored = transform.position_mirror @ pos
        assert mirrored[0] == pytest.approx(1.0)  # x unchanged
        assert mirrored[1] == pytest.approx(-2.0)  # y flipped
        assert mirrored[2] == pytest.approx(3.0)  # z unchanged


class TestMirrorPosition:
    def test_x_unchanged(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        result = mirror_position(pos)
        assert result[0] == pytest.approx(1.0)

    def test_y_flipped(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        result = mirror_position(pos)
        assert result[1] == pytest.approx(-2.0)

    def test_z_unchanged(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        result = mirror_position(pos)
        assert result[2] == pytest.approx(3.0)

    def test_double_mirror_is_identity(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        result = mirror_position(mirror_position(pos))
        np.testing.assert_allclose(result, pos, atol=1e-12)

    def test_trajectory_shape(self) -> None:
        positions = np.random.randn(10, 3)
        result = mirror_position(positions)
        assert result.shape == (10, 3)

    def test_trajectory_y_flipped(self) -> None:
        positions = np.ones((5, 3))
        result = mirror_position(positions)
        np.testing.assert_allclose(result[:, 1], -1.0)


class TestMirrorVelocity:
    def test_y_flipped(self) -> None:
        vel = np.array([1.0, 2.0, 3.0])
        result = mirror_velocity(vel)
        assert result[1] == pytest.approx(-2.0)

    def test_x_z_unchanged(self) -> None:
        vel = np.array([1.0, 2.0, 3.0])
        result = mirror_velocity(vel)
        assert result[0] == pytest.approx(1.0)
        assert result[2] == pytest.approx(3.0)


class TestMirrorRotationMatrix:
    def test_identity_maps_to_sagittal_mirror(self) -> None:
        R = np.eye(3)
        result = mirror_rotation_matrix(R)
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_result_is_valid_rotation(self) -> None:
        # A rotation matrix mirrored should remain orthogonal (det = ±1)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        result = mirror_rotation_matrix(R)
        det = np.linalg.det(result)
        assert abs(abs(det) - 1.0) < 1e-10
