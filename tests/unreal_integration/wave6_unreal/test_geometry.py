"""Wave6 fast tests for unreal_integration.geometry.

Covers Vector3 and Quaternion: construction, validation, serialization,
arithmetic, normalization, hashing, and equality semantics.

These tests mock no external `unreal` module — they exercise pure Python
data types only. Designed to run in well under a second.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.unreal_integration.geometry import Quaternion, Vector3, _validate_finite


class TestValidateFinite:
    def test_accepts_finite(self) -> None:
        _validate_finite(1.5, "v")  # does not raise

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="cannot be NaN"):
            _validate_finite(float("nan"), "v")

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="cannot be infinite"):
            _validate_finite(float("inf"), "v")

    def test_rejects_neg_inf(self) -> None:
        with pytest.raises(ValueError, match="cannot be infinite"):
            _validate_finite(float("-inf"), "v")


class TestVector3Construction:
    def test_default_zero(self) -> None:
        v = Vector3()
        assert (v.x, v.y, v.z) == (0.0, 0.0, 0.0)

    def test_from_values(self) -> None:
        v = Vector3(x=1.0, y=-2.0, z=3.5)
        assert v.x == 1.0 and v.y == -2.0 and v.z == 3.5

    def test_zero_classmethod(self) -> None:
        assert Vector3.zero() == Vector3(0, 0, 0)

    def test_from_numpy(self) -> None:
        v = Vector3.from_numpy(np.array([1.0, 2.0, 3.0]))
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_from_numpy_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="must have 3 elements"):
            Vector3.from_numpy(np.array([1.0, 2.0]))

    def test_from_dict(self) -> None:
        v = Vector3.from_dict({"x": 1, "y": 2, "z": 3})
        assert v.x == 1.0

    def test_validate_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            Vector3(x=float("nan"), validate=True)


class TestVector3Operations:
    def test_magnitude(self) -> None:
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude == pytest.approx(5.0)

    def test_normalized(self) -> None:
        v = Vector3(3.0, 4.0, 0.0).normalized()
        assert v.magnitude == pytest.approx(1.0)

    def test_normalize_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="zero vector"):
            Vector3.zero().normalized()

    def test_dot_product(self) -> None:
        a = Vector3(1, 2, 3)
        b = Vector3(4, -5, 6)
        assert a.dot(b) == 1 * 4 + 2 * -5 + 3 * 6

    def test_cross_product(self) -> None:
        c = Vector3(1, 0, 0).cross(Vector3(0, 1, 0))
        assert c == Vector3(0, 0, 1)

    def test_add_sub(self) -> None:
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        assert a + b == Vector3(5, 7, 9)
        assert b - a == Vector3(3, 3, 3)

    def test_mul_scalar(self) -> None:
        assert Vector3(1, 2, 3) * 2 == Vector3(2, 4, 6)
        assert 3 * Vector3(1, 2, 3) == Vector3(3, 6, 9)

    def test_neg(self) -> None:
        assert -Vector3(1, -2, 3) == Vector3(-1, 2, -3)

    def test_eq_with_none_returns_false(self) -> None:
        # Regression: previously raised ValueError instead of returning False
        v = Vector3(1, 2, 3)
        assert (v == None) is False  # noqa: E711
        assert (v != None) is True  # noqa: E711

    def test_eq_with_other_type(self) -> None:
        assert Vector3(1, 2, 3) != "not a vector"

    def test_hashable(self) -> None:
        # Regression: previously unhashable due to dataclass(eq=False) issues
        v = Vector3(1, 2, 3)
        s = {v, Vector3(1, 2, 3), Vector3(4, 5, 6)}
        assert len(s) == 2

    def test_repr(self) -> None:
        assert "Vector3" in repr(Vector3(1, 2, 3))


class TestVector3Serialization:
    def test_to_numpy(self) -> None:
        arr = Vector3(1, 2, 3).to_numpy()
        assert arr.shape == (3,)
        assert arr.dtype == np.float64

    def test_to_dict_roundtrip(self) -> None:
        v = Vector3(1.5, -2.5, 3.25)
        d = v.to_dict()
        assert Vector3.from_dict(d) == v


class TestQuaternion:
    def test_identity(self) -> None:
        q = Quaternion.identity()
        assert q.w == 1.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0
        assert q.magnitude == pytest.approx(1.0)

    def test_from_euler_zero(self) -> None:
        q = Quaternion.from_euler(0.0, 0.0, 0.0)
        assert q.w == pytest.approx(1.0)

    def test_euler_roundtrip(self) -> None:
        q = Quaternion.from_euler(0.1, 0.2, 0.3)
        r, p, y = q.to_euler()
        assert (r, p, y) == pytest.approx((0.1, 0.2, 0.3), abs=1e-6)

    def test_magnitude_and_normalize(self) -> None:
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
        n = q.normalized()
        assert n.magnitude == pytest.approx(1.0)

    def test_normalize_zero_returns_identity(self) -> None:
        q = Quaternion(w=0, x=0, y=0, z=0).normalized()
        assert q.w == 1.0

    def test_conjugate(self) -> None:
        q = Quaternion(w=1, x=2, y=3, z=4).conjugate()
        assert (q.w, q.x, q.y, q.z) == (1, -2, -3, -4)

    def test_to_dict_roundtrip(self) -> None:
        q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        d = q.to_dict()
        q2 = Quaternion.from_dict(d)
        assert (q2.w, q2.x, q2.y, q2.z) == (0.5, 0.5, 0.5, 0.5)

    def test_validate_normalizes(self) -> None:
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0, validate=True)
        assert q.magnitude == pytest.approx(1.0)

    def test_to_euler_clamps_pitch(self) -> None:
        # Construct a quaternion where sin(pitch) >= 1 (gimbal lock)
        q = Quaternion(w=math.sqrt(0.5), x=0, y=math.sqrt(0.5), z=0)
        _, pitch, _ = q.to_euler()
        assert abs(pitch) <= math.pi / 2 + 1e-9

    def test_repr(self) -> None:
        assert "Quaternion" in repr(Quaternion.identity())
