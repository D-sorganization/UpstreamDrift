"""Geometry contracts for a shaft-twist delivery sensitivity."""

import math

import pytest

from src.tools.shot_pattern_analysis.delivery_geometry import (
    delivery_from_face_angle,
    delivery_from_shaft_rotation,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("lean", [0.0, 10.0, 20.0])
def test_square_face_preserves_base_loft(lean: float) -> None:
    delivery = delivery_from_shaft_rotation(
        0.0, base_loft_deg=10.9, lie_deg=58.0, shaft_lean_deg=lean
    )
    assert delivery.face_angle_deg == pytest.approx(0.0, abs=1e-12)
    assert delivery.dynamic_loft_deg == pytest.approx(10.9, abs=1e-12)
    assert math.isclose(sum(x * x for x in delivery.face_normal), 1.0)


@pytest.mark.parametrize("lean", [0.0, 10.0, 20.0])
def test_closed_face_from_positive_shaft_twist_delofts(lean: float) -> None:
    closed = delivery_from_shaft_rotation(
        2.0, base_loft_deg=10.9, lie_deg=58.0, shaft_lean_deg=lean
    )
    opened = delivery_from_shaft_rotation(
        -2.0, base_loft_deg=10.9, lie_deg=58.0, shaft_lean_deg=lean
    )
    assert closed.face_angle_deg < 0 < opened.face_angle_deg
    assert closed.dynamic_loft_deg < 10.9 < opened.dynamic_loft_deg


@pytest.mark.parametrize("face", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
def test_inverse_reproduces_requested_face_angle(face: float) -> None:
    delivery = delivery_from_face_angle(
        face, base_loft_deg=10.9, lie_deg=58.0, shaft_lean_deg=10.0
    )
    assert delivery.face_angle_deg == pytest.approx(face, abs=1e-8)
    if face < 0:
        assert delivery.dynamic_loft_deg < 10.9
    elif face > 0:
        assert delivery.dynamic_loft_deg > 10.9


def test_zero_lean_small_angle_slope_is_cotangent_of_lie() -> None:
    opened = delivery_from_face_angle(
        0.01, base_loft_deg=10.9, lie_deg=58.0, shaft_lean_deg=0.0
    )
    closed = delivery_from_face_angle(
        -0.01, base_loft_deg=10.9, lie_deg=58.0, shaft_lean_deg=0.0
    )
    slope = (opened.dynamic_loft_deg - closed.dynamic_loft_deg) / 0.02
    assert slope == pytest.approx(1.0 / math.tan(math.radians(58.0)), abs=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_loft_deg": float("nan"), "lie_deg": 58.0, "shaft_lean_deg": 0.0},
        {"base_loft_deg": 10.9, "lie_deg": 90.0, "shaft_lean_deg": 0.0},
        {"base_loft_deg": 10.9, "lie_deg": 58.0, "shaft_lean_deg": 40.0},
    ],
)
def test_rejects_invalid_geometry(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        delivery_from_shaft_rotation(1.0, **kwargs)
