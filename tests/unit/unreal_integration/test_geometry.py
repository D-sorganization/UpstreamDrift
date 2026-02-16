"""Tests for src.unreal_integration.geometry module (Vector3 and Quaternion)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.unreal_integration.geometry import Quaternion, Vector3


class TestVector3:
    """Tests for Vector3 class."""

    def test_default_construction(self) -> None:
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_custom_construction(self) -> None:
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_zero_factory(self) -> None:
        v = Vector3.zero()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_from_numpy(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        v = Vector3.from_numpy(arr)
        assert v.x == pytest.approx(1.0)
        assert v.y == pytest.approx(2.0)
        assert v.z == pytest.approx(3.0)

    def test_from_numpy_wrong_size(self) -> None:
        with pytest.raises(ValueError):
            Vector3.from_numpy(np.array([1.0, 2.0]))

    def test_from_dict(self) -> None:
        v = Vector3.from_dict({"x": 1.0, "y": 2.0, "z": 3.0})
        assert v.x == 1.0

    def test_to_numpy(self) -> None:
        v = Vector3(1.0, 2.0, 3.0)
        arr = v.to_numpy()
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_to_dict(self) -> None:
        v = Vector3(1.0, 2.0, 3.0)
        d = v.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_magnitude(self) -> None:
        v = Vector3(3.0, 4.0, 0.0)
        # magnitude is a property, not a method
        assert v.magnitude == pytest.approx(5.0)

    def test_normalized(self) -> None:
        v = Vector3(0.0, 0.0, 5.0)
        n = v.normalized()
        assert n.magnitude == pytest.approx(1.0)
        assert n.z == pytest.approx(1.0)

    def test_normalized_zero_raises(self) -> None:
        v = Vector3.zero()
        with pytest.raises(ValueError):
            v.normalized()

    def test_dot_product(self) -> None:
        a = Vector3(1.0, 0.0, 0.0)
        b = Vector3(0.0, 1.0, 0.0)
        assert a.dot(b) == pytest.approx(0.0)

    def test_dot_product_parallel(self) -> None:
        a = Vector3(1.0, 2.0, 3.0)
        assert a.dot(a) == pytest.approx(14.0)

    def test_cross_product(self) -> None:
        x = Vector3(1.0, 0.0, 0.0)
        y = Vector3(0.0, 1.0, 0.0)
        z = x.cross(y)
        assert z.z == pytest.approx(1.0)
        assert z.x == pytest.approx(0.0)
        assert z.y == pytest.approx(0.0)

    def test_add(self) -> None:
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        c = a + b
        assert c.x == pytest.approx(5.0)
        assert c.y == pytest.approx(7.0)

    def test_sub(self) -> None:
        a = Vector3(4.0, 5.0, 6.0)
        b = Vector3(1.0, 2.0, 3.0)
        c = a - b
        assert c.x == pytest.approx(3.0)

    def test_scalar_mul(self) -> None:
        v = Vector3(1.0, 2.0, 3.0)
        r = v * 2.0
        assert r.x == pytest.approx(2.0)
        assert r.y == pytest.approx(4.0)

    def test_rmul(self) -> None:
        v = Vector3(1.0, 2.0, 3.0)
        r = 3.0 * v
        assert r.x == pytest.approx(3.0)

    def test_neg(self) -> None:
        v = Vector3(1.0, -2.0, 3.0)
        n = -v
        assert n.x == pytest.approx(-1.0)
        assert n.y == pytest.approx(2.0)

    def test_repr(self) -> None:
        v = Vector3(1.0, 2.0, 3.0)
        assert "1.0" in repr(v)

    def test_validate_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            Vector3(float("nan"), 0.0, 0.0, validate=True)

    def test_validate_rejects_inf(self) -> None:
        with pytest.raises(ValueError):
            Vector3(float("inf"), 0.0, 0.0, validate=True)


class TestQuaternion:
    """Tests for Quaternion class."""

    def test_identity_quaternion(self) -> None:
        q = Quaternion()
        assert q.w == pytest.approx(1.0)
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)

    def test_identity_factory(self) -> None:
        q = Quaternion.identity()
        assert q.w == pytest.approx(1.0)

    def test_from_dict(self) -> None:
        q = Quaternion.from_dict({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
        assert q.w == pytest.approx(1.0)

    def test_to_dict(self) -> None:
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        d = q.to_dict()
        assert d["w"] == 1.0

    def test_magnitude(self) -> None:
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        # magnitude is a property, not a method
        assert q.magnitude == pytest.approx(1.0)

    def test_normalized(self) -> None:
        q = Quaternion(2.0, 0.0, 0.0, 0.0)
        n = q.normalized()
        assert n.magnitude == pytest.approx(1.0)
        assert n.w == pytest.approx(1.0)

    def test_from_euler(self) -> None:
        """Identity euler angles should give identity quaternion."""
        q = Quaternion.from_euler(0, 0, 0)
        assert q.w == pytest.approx(1.0)
        assert q.x == pytest.approx(0.0)

    def test_to_euler_roundtrip(self) -> None:
        """from_euler → to_euler roundtrip."""
        q = Quaternion.from_euler(0.1, 0.2, 0.3)
        roll, pitch, yaw = q.to_euler()
        assert roll == pytest.approx(0.1, abs=1e-6)
        assert pitch == pytest.approx(0.2, abs=1e-6)
        assert yaw == pytest.approx(0.3, abs=1e-6)

    def test_validate_normalizes(self) -> None:
        """validate=True should normalize the quaternion in place."""
        q = Quaternion(2.0, 0.0, 0.0, 0.0, validate=True)
        assert q.magnitude == pytest.approx(1.0)
