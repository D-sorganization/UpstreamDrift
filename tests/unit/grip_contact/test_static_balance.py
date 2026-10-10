"""Static balance residuals of a held club (#11739 phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import ClubDynamics, GripInterface
from src.shared.python.grip_contact.parity import GripKineticsSeries
from src.shared.python.grip_contact.static_balance import static_balance

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def _held_series(split: float, torque_error_nm: float = 0.0) -> tuple:
    """A club at a tilted attitude, supported by two hands exactly."""
    spec = json.loads(SPEC.read_text())
    interface = GripInterface.from_spec(spec)
    club = ClubDynamics.from_spec(spec)
    g = np.asarray(spec["gravity_m_s2"], float)
    th = 0.7
    rot = np.array(
        [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
    )
    origin = np.array([0.1, -0.2, 1.0])
    p = {s: origin + rot @ np.asarray(interface.frame(s).position_m) for s in "LR"}
    com = origin + rot @ np.asarray(club.com_m)
    weight = -club.mass_kg * g
    force = {"L": split * weight, "R": (1 - split) * weight}
    # free torques that close the moment about the centre of mass
    need = -sum(np.cross(p[s] - com, force[s]) for s in "LR")
    torque = {"L": 0.5 * need + [torque_error_nm, 0, 0], "R": 0.5 * need}
    n = 2
    tile = lambda v: np.tile(v, (n, 1))  # noqa: E731
    series = GripKineticsSeries(
        engine="test",
        time_s=np.array([0.0, 0.002]),
        force_on_club_n={s: tile(force[s]) for s in "LR"},
        torque_on_club_nm={s: tile(torque[s]) for s in "LR"},
        grip_point_m={s: tile(p[s]) for s in "LR"},
        deflection_m={s: np.zeros((n, 3)) for s in "LR"},
        rotation_deflection_rad={s: np.zeros(n) for s in "LR"},
        club_rotation=np.repeat(rot[None], n, axis=0),
    )
    return series, interface, club, g


@pytest.mark.parametrize("split", [0.5, 0.2, 0.9])
def test_exact_support_has_no_residual_for_any_split(split: float) -> None:
    series, interface, club, g = _held_series(split)
    result = static_balance(series, interface, club, g)
    assert result.max_force_error() < 1e-12
    assert result.max_moment_error() < 1e-12


def test_a_wrong_hand_torque_shows_up_as_a_moment_residual() -> None:
    series, interface, club, g = _held_series(0.5, torque_error_nm=0.5)
    result = static_balance(series, interface, club, g)
    assert result.max_force_error() < 1e-12
    assert result.moment_residual_nm[0, 0] == pytest.approx(0.5)
    assert result.max_moment_error() > 0.01


def test_gravity_must_be_a_three_vector() -> None:
    series, interface, club, _ = _held_series(0.5)
    with pytest.raises(ValueError, match="3-vector"):
        static_balance(series, interface, club, np.zeros(2))
