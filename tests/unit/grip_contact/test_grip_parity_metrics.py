"""Grip-kinetics parity metric and series (issue #11739, OSV-7 phase 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import parity as par
from src.shared.python.grip_contact.parity import (
    GripKineticsSeries,
    parity_errors,
)

pytestmark = pytest.mark.unit


def _series(
    scale: float = 1.0, n: int = 50, engine: str = "test"
) -> GripKineticsSeries:
    t = np.arange(n) * 0.002
    wave = np.sin(2 * np.pi * 3 * t)[:, None]
    f_l = scale * np.hstack([100 * wave, 20 + 0 * wave, -5 * wave])
    f_r = scale * np.hstack([-80 * wave, 30 + 0 * wave, 4 * wave])
    p_l = np.tile([0.0, 0.0, 1.0], (n, 1))
    p_r = np.tile([0.08, 0.0, 1.0], (n, 1))
    rot = np.tile(np.eye(3), (n, 1, 1))
    return GripKineticsSeries(
        engine=engine,
        time_s=t,
        force_on_club_n={"L": f_l, "R": f_r},
        torque_on_club_nm={"L": scale * 2 * f_l / 100, "R": scale * f_r / 100},
        grip_point_m={"L": p_l, "R": p_r},
        deflection_m={"L": scale * f_l * 1e-6, "R": scale * f_r * 1e-6},
        rotation_deflection_rad={
            "L": scale * np.abs(wave[:, 0]) * 1e-3,
            "R": 1e-3 + 0 * t,
        },
        club_rotation=rot,
    )


def test_tolerances_are_the_issue_values() -> None:
    """5 % at peak and 2 % RMS (normalised by peak), fixed before any run."""
    assert par.PEAK_TOLERANCE == 0.05
    assert par.RMS_TOLERANCE == 0.02
    assert set(par.PARITY_QUANTITIES) >= {
        "force_L",
        "force_R",
        "net_force",
        "internal_force",
        "squeeze",
        "couple",
        "deflection_L",
        "deflection_R",
    }


def test_identical_series_have_zero_error() -> None:
    errors = parity_errors(_series(), _series())
    assert set(errors) == set(par.PARITY_QUANTITIES)
    assert all(e.peak_error == 0.0 and e.rms_error == 0.0 for e in errors.values())
    assert all(e.passes() for e in errors.values())


def test_scaled_forces_give_the_scale_as_peak_error() -> None:
    errors = parity_errors(_series(1.03), _series())
    assert errors["force_L"].peak_error == pytest.approx(0.03)
    assert errors["net_force"].peak_error == pytest.approx(0.03)
    assert errors["force_L"].passes() is False  # RMS of a 3 % scale exceeds 2 %
    assert errors["rotation_R"].peak_error == 0.0


def test_time_bases_must_match() -> None:
    with pytest.raises(ValueError, match="sample times"):
        parity_errors(_series(n=50), _series(n=40))


def test_series_preconditions() -> None:
    good = _series()
    with pytest.raises(ValueError, match="sides"):
        GripKineticsSeries(
            "x",
            good.time_s,
            {"L": good.force_on_club_n["L"]},
            good.torque_on_club_nm,
            good.grip_point_m,
            good.deflection_m,
            good.rotation_deflection_rad,
            good.club_rotation,
        )
    with pytest.raises(ValueError, match="increasing"):
        GripKineticsSeries(
            "x",
            good.time_s[::-1],
            good.force_on_club_n,
            good.torque_on_club_nm,
            good.grip_point_m,
            good.deflection_m,
            good.rotation_deflection_rad,
            good.club_rotation,
        )


def test_npz_round_trip_and_bushing_split(tmp_path: Path) -> None:
    series = _series()
    path = tmp_path / "s.npz"
    series.save_npz(path)
    back = GripKineticsSeries.load_npz(path)
    assert back.engine == "test"
    np.testing.assert_array_equal(
        back.force_on_club_n["R"], series.force_on_club_n["R"]
    )
    grip = back.grip_series()
    assert set(grip.split_method) == {"bushing"}
    np.testing.assert_allclose(
        grip.net_force_n, series.force_on_club_n["L"] + series.force_on_club_n["R"]
    )


def test_from_frames_computes_deflection_in_hand_axes() -> None:
    n = 3
    t = np.arange(n) * 0.002
    eye = np.tile(np.eye(3), (n, 1, 1))
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    hand_rot = np.tile(rz, (n, 1, 1))
    p1 = np.zeros((n, 3))
    p2 = np.tile([0.001, 0.0, 0.0], (n, 1))
    zeros = np.zeros((n, 3))
    series = GripKineticsSeries.from_frames(
        "x",
        t,
        {"L": (zeros, zeros), "R": (zeros, zeros)},
        {"L": (hand_rot, p1), "R": (eye, p1)},
        {"L": (hand_rot, p2), "R": (eye, p2)},
        eye,
    )
    np.testing.assert_allclose(
        series.deflection_m["L"][0], [0.0, -0.001, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(series.deflection_m["R"][0], [0.001, 0.0, 0.0])
    np.testing.assert_allclose(series.rotation_deflection_rad["L"], 0.0, atol=1e-15)
