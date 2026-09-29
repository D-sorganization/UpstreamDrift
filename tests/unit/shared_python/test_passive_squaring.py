"""Tests for passive squaring torque + plane classification (Phase 2, epic #5422)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.shallowing.hand_path_plane import Plane3D
from src.shared.python.biomechanics.shallowing.passive_squaring import (
    ShallowingMetrics,
    classify_swing_plane,
    compute_club_com_offset,
    compute_passive_squaring_torque,
)


@pytest.fixture
def horizontal_plane() -> Plane3D:
    return Plane3D(
        normal=np.array([0.0, 0.0, 1.0]),
        point_on_plane=np.array([0.0, 0.0, 0.0]),
        residuals=0.0,
    )


def test_zero_offset_on_plane(horizontal_plane: Plane3D) -> None:
    com = np.array([1.0, 1.0, 0.0])  # on plane
    assert compute_club_com_offset(com, horizontal_plane) == pytest.approx(0.0)


def test_positive_offset_above_plane(horizontal_plane: Plane3D) -> None:
    com = np.array([0.0, 0.0, 0.5])
    assert compute_club_com_offset(com, horizontal_plane) == pytest.approx(0.5)


def test_negative_offset_below_plane(horizontal_plane: Plane3D) -> None:
    com = np.array([0.0, 0.0, -0.3])
    assert compute_club_com_offset(com, horizontal_plane) == pytest.approx(-0.3)


def test_zero_offset_zero_torque() -> None:
    assert compute_passive_squaring_torque(0.0, 30.0, 0.3) == pytest.approx(0.0)


def test_positive_offset_positive_torque() -> None:
    # com 0.1 m above plane, omega 30 rad/s, mass 0.3 kg
    # tau = 0.3 * 30^2 * 0.1 = 27 N*m
    tau = compute_passive_squaring_torque(0.1, 30.0, 0.3)
    assert tau == pytest.approx(27.0)


def test_negative_offset_negative_torque() -> None:
    tau = compute_passive_squaring_torque(-0.1, 30.0, 0.3)
    assert tau == pytest.approx(-27.0)


def test_torque_zero_mass_raises() -> None:
    with pytest.raises(ValueError):
        compute_passive_squaring_torque(0.1, 30.0, 0.0)


def test_torque_negative_mass_raises() -> None:
    with pytest.raises(ValueError):
        compute_passive_squaring_torque(0.1, 30.0, -0.3)


def test_classify_steep() -> None:
    m = ShallowingMetrics(
        com_offset=0.05,
        passive_torque=10.0,
        steepness_index=0.15,
        shallowing_index=0.02,
    )
    assert classify_swing_plane(m) == "steep"


def test_classify_shallow() -> None:
    m = ShallowingMetrics(
        com_offset=-0.05,
        passive_torque=-10.0,
        steepness_index=0.02,
        shallowing_index=0.15,
    )
    assert classify_swing_plane(m) == "shallow"


def test_classify_on_plane() -> None:
    m = ShallowingMetrics(
        com_offset=0.0,
        passive_torque=0.0,
        steepness_index=0.05,
        shallowing_index=0.05,
    )
    assert classify_swing_plane(m) == "on_plane"


def test_classify_near_threshold() -> None:
    # |0.05 - 0.04| = 0.01 < 0.02 -> on_plane
    m = ShallowingMetrics(
        com_offset=0.005,
        passive_torque=1.0,
        steepness_index=0.05,
        shallowing_index=0.04,
    )
    assert classify_swing_plane(m) == "on_plane"


def test_metrics_all_finite() -> None:
    m = ShallowingMetrics(
        com_offset=0.05,
        passive_torque=10.0,
        steepness_index=0.1,
        shallowing_index=0.02,
    )
    for field_name in [
        "com_offset",
        "passive_torque",
        "steepness_index",
        "shallowing_index",
    ]:
        assert np.isfinite(getattr(m, field_name))


def test_com_offset_dbc_wrong_shape(horizontal_plane: Plane3D) -> None:
    with pytest.raises((ValueError, TypeError)):
        compute_club_com_offset(np.array([0.0, 0.0]), horizontal_plane)


def test_com_offset_normal_assumed_unit_length(horizontal_plane: Plane3D) -> None:
    # Plane with non-unit normal: function should still compute correctly
    # OR raise -- pick one and stick with it (raising is cleaner)
    bad_plane = Plane3D(
        normal=np.array([0.0, 0.0, 2.0]),  # length 2, not 1
        point_on_plane=np.array([0.0, 0.0, 0.0]),
        residuals=0.0,
    )
    # Either: raises ValueError, OR normalizes internally and returns correct value
    com = np.array([0.0, 0.0, 0.5])
    try:
        result = compute_club_com_offset(com, bad_plane)
        # If it accepted: must return correct distance ignoring normal length
        assert result == pytest.approx(0.5)
    except ValueError:
        pass  # Also acceptable behavior
