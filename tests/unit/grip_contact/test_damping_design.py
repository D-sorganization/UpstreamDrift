"""Deliberate damping design for the two-bushing grip (#11739)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    ClubDynamics,
    GripInterface,
    design_damping,
    modal_damping,
)

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def test_club_dynamics_from_spec(spec: dict) -> None:
    club = ClubDynamics.from_spec(spec)
    assert club.mass_kg == pytest.approx(0.313)  # head + shaft + grip, no hands
    # centre of mass lies between the head (y=0) and the butt, near the shaft
    assert -0.45 < club.com_m[1] < -0.1
    assert np.all(np.linalg.eigvalsh(club.inertia_com_kg_m2) > 0)


@pytest.mark.parametrize("zeta", [0.5, 0.7, 1.0])
def test_every_mode_has_the_requested_damping_ratio(spec: dict, zeta: float) -> None:
    gi = GripInterface.from_spec(spec, damping_ratio=zeta)
    freqs, ratios = modal_damping(gi, ClubDynamics.from_spec(spec))
    assert np.all(freqs > 1.0)
    np.testing.assert_allclose(ratios, zeta, rtol=1e-6)


def test_pitch_yaw_modes_are_the_low_frequency_ones(spec: dict) -> None:
    gi = GripInterface.from_spec(spec)
    freqs, _ = modal_damping(gi, ClubDynamics.from_spec(spec))
    assert 10.0 < freqs[0] < 60.0  # the club pendulum modes seen as ~30 Hz ringing


def test_scaling_stiffness_keeps_the_ratio(spec: dict) -> None:
    gi = GripInterface.from_spec(spec)
    soft = GripInterface(gi.left, gi.right, gi.bushing.scaled(0.1))
    _, ratios = modal_damping(soft, ClubDynamics.from_spec(spec))
    np.testing.assert_allclose(ratios, 0.7, rtol=1e-6)


def test_design_rejects_bad_ratio(spec: dict) -> None:
    gi = GripInterface.from_spec(spec)
    club = ClubDynamics.from_spec(spec)
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="damping_ratio"):
            design_damping(gi.bushing, gi.left, gi.right, club, bad)


def test_free_vibration_decays_at_the_stated_ratio(spec: dict) -> None:
    """Integrate the linear 6-DOF club model; the log decrement gives zeta."""
    from scipy.integrate import solve_ivp
    from scipy.linalg import eigh

    from src.shared.python.grip_contact.damping import assemble_matrices

    zeta = 0.1  # lightly damped so that the decay is observable
    gi = GripInterface.from_spec(spec, damping_ratio=zeta)
    club = ClubDynamics.from_spec(spec)
    m_mat, k_mat, c_mat = assemble_matrices(gi, club)
    w2, vec = eigh(k_mat, m_mat)
    mode = 0
    omega = np.sqrt(w2[mode])
    x0 = vec[:, mode]
    n = 6

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        x, v = y[:n], y[n:]
        return np.concatenate([v, np.linalg.solve(m_mat, -c_mat @ v - k_mat @ x)])

    period = 2 * np.pi / omega
    t = np.linspace(0.0, 6 * period, 6000)
    sol = solve_ivp(rhs, (0, t[-1]), np.concatenate([x0, 0 * x0]), t_eval=t, rtol=1e-9)
    amp = sol.y[:n].T @ (m_mat @ x0)  # signed modal coordinate
    peaks = [
        i for i in range(1, len(t) - 1) if amp[i] > amp[i - 1] and amp[i] > amp[i + 1]
    ]
    assert len(peaks) >= 4
    delta = np.log(amp[peaks[0]] / amp[peaks[3]]) / 3
    measured = delta / np.sqrt(4 * np.pi**2 + delta**2)
    assert measured == pytest.approx(zeta, rel=0.03)
