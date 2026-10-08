"""Net and internal parts of the two hand forces on the club (#11739)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_contact import decompose_hand_forces

pytestmark = pytest.mark.unit

LEFT = np.array([0.0, 0.0, 0.0])
RIGHT = np.array([0.0, 0.08, 0.0])  # inter-hand line along +y


def test_parts_reconstruct_the_hand_forces() -> None:
    f_l, f_r = np.array([10.0, 3.0, -2.0]), np.array([4.0, -5.0, 7.0])
    d = decompose_hand_forces(f_l, f_r, LEFT, RIGHT)
    np.testing.assert_allclose(d.net_n, f_l + f_r)
    np.testing.assert_allclose(d.net_n / 2 + d.internal_n, f_l)
    np.testing.assert_allclose(d.net_n / 2 - d.internal_n, f_r)


def test_pure_squeeze_has_no_net_force() -> None:
    f_l = np.array([0.0, 50.0, 0.0])  # pushes toward the right hand
    d = decompose_hand_forces(f_l, -f_l, LEFT, RIGHT)
    assert np.linalg.norm(d.net_n) == pytest.approx(0.0, abs=1e-12)
    assert d.internal_axial_n == pytest.approx(50.0)  # compression is positive
    assert np.linalg.norm(d.internal_transverse_n) == pytest.approx(0.0, abs=1e-12)


def test_equal_forces_have_no_internal_part() -> None:
    f = np.array([1.0, 2.0, 3.0])
    d = decompose_hand_forces(f, f, LEFT, RIGHT)
    assert np.linalg.norm(d.internal_n) == pytest.approx(0.0, abs=1e-12)


def test_force_couple_is_transverse_internal() -> None:
    # equal and opposite forces across the grip carry a couple (moment pair)
    f_l = np.array([30.0, 0.0, 0.0])
    d = decompose_hand_forces(f_l, -f_l, LEFT, RIGHT)
    assert d.internal_axial_n == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(d.internal_transverse_n) == pytest.approx(30.0)
    assert d.couple_moment_nm == pytest.approx(30.0 * 0.08)


def test_time_series_shapes_and_peaks() -> None:
    n = 5
    f_l = np.tile([0.0, 10.0, 0.0], (n, 1)) * np.arange(n)[:, None]
    d = decompose_hand_forces(f_l, -f_l, np.zeros((n, 3)), np.tile(RIGHT, (n, 1)))
    assert d.net_n.shape == (n, 3)
    assert d.internal_axial_n.shape == (n,)
    assert d.peak_internal_n() == pytest.approx(40.0)
    assert d.peak_net_n() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "bad", [np.zeros(2), np.array([np.nan, 0.0, 0.0]), np.zeros((3, 4))]
)
def test_rejects_malformed_forces(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        decompose_hand_forces(bad, bad, LEFT, RIGHT)


def test_rejects_coincident_hand_points() -> None:
    with pytest.raises(ValueError, match="distinct"):
        decompose_hand_forces(np.ones(3), np.ones(3), LEFT, LEFT)
