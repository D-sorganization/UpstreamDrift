"""Contracts of the loaded foot-tracking measures (OSL-16, #13104)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.loaded_foot import (
    FOOT_JOINTS,
    foot_joint_error_rad,
    window_reference,
)

pytestmark = pytest.mark.unit

ORDER = ("pelvis_tx", *FOOT_JOINTS)


def test_foot_joint_error_is_the_largest_absolute_gap() -> None:
    ref = np.zeros((4, len(ORDER)))
    run = ref.copy()
    run[2, ORDER.index("mtp_angle_r")] = -0.3
    run[1, ORDER.index("ankle_angle_l")] = 0.1
    run[3, 0] = 5.0  # not a foot joint
    error = foot_joint_error_rad(ORDER, run, ref)
    assert set(error) == set(FOOT_JOINTS)
    assert error["mtp_angle_r"] == pytest.approx(0.3)
    assert error["ankle_angle_l"] == pytest.approx(0.1)
    assert error["subtalar_angle_r"] == 0.0


def test_foot_joint_error_compares_a_short_run_on_its_rows() -> None:
    ref = np.zeros((5, len(ORDER)))
    ref[4] = 9.0
    error = foot_joint_error_rad(ORDER, np.zeros((3, len(ORDER))), ref)
    assert max(error.values()) == 0.0


@pytest.mark.parametrize(
    ("run", "ref"),
    [
        (np.zeros((3, 2)), np.zeros((3, 2))),
        (np.zeros((4, len(ORDER))), np.zeros((3, len(ORDER)))),
        (np.full((2, len(ORDER)), np.nan), np.zeros((2, len(ORDER)))),
    ],
)
def test_foot_joint_error_rejects_bad_arrays(run, ref) -> None:
    with pytest.raises(ValueError):
        foot_joint_error_rad(ORDER, run, ref)


def test_foot_joint_error_requires_the_foot_joints() -> None:
    with pytest.raises(ValueError, match="foot joints"):
        foot_joint_error_rad(("a", "b"), np.zeros((2, 2)), np.zeros((2, 2)))


def test_window_reference_interpolates_on_the_window_clock() -> None:
    times = np.array([1.4, 1.42, 1.44])
    track = np.array([[0.0], [2.0], [4.0]])
    ref = window_reference(times, track, np.array([0.0, 0.01, 0.04]))
    np.testing.assert_allclose(ref[:, 0], [0.0, 1.0, 4.0])
