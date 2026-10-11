"""Conditioning of matched-IK coordinate trajectories before kinetics (#11739)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    condition_trajectory,
    detect_ik_outliers,
    first_discontinuity_time,
    unwrap_angular,
)

pytestmark = pytest.mark.unit

T = np.linspace(0.0, 1.0, 361)
NAMES = ["TranslationInputX", "ElbowInput", "WristInput"]


def _clean() -> np.ndarray:
    return np.stack([0.3 * T, np.sin(2 * T), 0.5 * np.cos(3 * T)], axis=1)


def test_unwrap_removes_two_pi_branch_flips_but_not_translation() -> None:
    q = _clean()
    flipped = q.copy()
    flipped[200:, 1] += 8 * np.pi  # persistent branch flip of an angle
    out = unwrap_angular(flipped, NAMES)
    np.testing.assert_allclose(out[:, 1], q[:, 1], atol=1e-9)
    np.testing.assert_array_equal(out[:, 0], flipped[:, 0])  # translation kept


def test_detects_multi_coordinate_spikes() -> None:
    q = _clean()
    q[120, 1:] += [1.5, -2.0]
    q[121, 1:] += [1.0, -1.0]
    bad = detect_ik_outliers(q, NAMES)
    assert bad[120] and bad[121]
    assert bad.sum() <= 4  # no wholesale flagging of a smooth motion


def test_clean_motion_is_untouched() -> None:
    q = _clean()
    assert not detect_ik_outliers(q, NAMES).any()
    out, report = condition_trajectory(T, q, NAMES)
    np.testing.assert_allclose(out, q, atol=1e-9)
    assert report.repaired_frames == 0


def test_repair_interpolates_over_spikes_and_reports() -> None:
    q = _clean()
    spiky = q.copy()
    spiky[150:153, 1:] += [2.0, -2.5]
    out, report = condition_trajectory(T, spiky, NAMES)
    assert report.repaired_frames == 3
    assert report.frame_times_s == pytest.approx(list(T[150:153]))
    assert np.abs(out - q).max() < 5e-3  # smooth signal recovered


@pytest.mark.parametrize("bad_time", [np.array([0.0, 0.0, 0.1]), np.arange(2.0)])
def test_rejects_bad_time_base(bad_time: np.ndarray) -> None:
    with pytest.raises(ValueError):
        condition_trajectory(bad_time, _clean()[: bad_time.size], NAMES)


def test_rejects_name_count_mismatch() -> None:
    with pytest.raises(ValueError, match="names"):
        condition_trajectory(T, _clean(), NAMES[:2])


def test_first_discontinuity_ignores_wraps_and_finds_persistent_steps() -> None:
    q = _clean()
    wrapped = q.copy()
    wrapped[100:, 1] += 2 * np.pi
    assert first_discontinuity_time(T, wrapped, NAMES, min_coordinates=1) is None
    stepped = q.copy()
    stepped[200:, 1:] += [1.2, -1.4]  # IK solution switch: persists
    assert first_discontinuity_time(T, stepped, NAMES, min_coordinates=2) == (
        pytest.approx(T[200])
    )
    assert first_discontinuity_time(T, q, NAMES) is None
