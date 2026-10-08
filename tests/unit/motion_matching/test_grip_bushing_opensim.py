"""OpenSim bushing grip kinetics (issue #11739, OSV-7, phase 1).

Skips when ``opensim`` is not importable.  Sign convention under test: every
wrench is the loading exerted by the hand ON THE CLUB.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("opensim")

from src.engines.physics_engines.opensim.python.grip_bushing_sim import (  # noqa: E402
    BushingGripSimulator,
    analyze_run,
    hand_power_w,
    lowpass_trajectory,
)
from src.shared.python.grip_contact import GripInterface  # noqa: E402

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
CANDIDATE = MODELS / "evidence/ground_support/anthro_driver_opensim/candidate.npz"
GRAVITY = 9.80665

# Owner-reviewable defaults from issue #11739 (not fitted).
MAX_DEFLECTION_M = 3.0e-3
MAX_ROTATION_RAD = np.radians(2.0)


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    return SPEC_PATH.read_bytes()


@pytest.fixture(scope="module")
def motion() -> tuple[list[str], np.ndarray, np.ndarray]:
    data = np.load(CANDIDATE, allow_pickle=True)
    names = json.loads(str(data["manifest_json"]))["coordinate_names"]
    return names, np.asarray(data["time_s"]), np.asarray(data["q"])


def _static_sim(spec_bytes: bytes, motion, duration: float = 0.3):
    names, _, q = motion
    n = 31
    times = np.linspace(0.0, duration, n)
    return BushingGripSimulator(spec_bytes, names, times, np.repeat(q[:1], n, axis=0))


@pytest.fixture(scope="module")
def static_run(spec_bytes, motion):
    sim = _static_sim(spec_bytes, motion)
    return sim, sim.run(accuracy=1e-6)


def test_static_hold_sums_to_club_weight(static_run, spec_bytes) -> None:
    sim, run = static_run
    spec = json.loads(spec_bytes)
    weight = spec["club"]["total_mass_kg"] * GRAVITY
    total = run.force_on_club_n["L"][-1] + run.force_on_club_n["R"][-1]
    up = -np.asarray(spec["gravity_m_s2"]) / GRAVITY
    assert float(total @ up) == pytest.approx(weight, rel=0.01)
    # no spurious horizontal load once settled
    assert np.linalg.norm(total - (total @ up) * up) < 0.01 * weight


def test_static_hold_moment_balance_closes(static_run) -> None:
    _, run = static_run
    com = run.club_com_m[-1]
    moment = np.zeros(3)
    for side in "LR":
        moment += np.cross(
            run.grip_point_m[side][-1] - com, run.force_on_club_n[side][-1]
        )
        moment += run.torque_on_club_nm[side][-1]
    scale = np.linalg.norm(run.force_on_club_n["L"][-1]) * 0.08
    assert np.linalg.norm(moment) < 0.02 * scale


def test_bushing_force_is_stiffness_times_deflection(spec_bytes, motion) -> None:
    names, _, q = motion
    sim = _static_sim(spec_bytes, motion)
    k_t = np.asarray(
        GripInterface.from_spec(
            json.loads(spec_bytes)
        ).bushing.translational_stiffness_n_m
    )
    k_r = np.asarray(
        GripInterface.from_spec(
            json.loads(spec_bytes)
        ).bushing.rotational_stiffness_nm_rad
    )
    run = sim.probe_deflection([1.0e-3, -0.5e-3, 0.7e-3], [0.0, 0.0, 0.0])
    for side in "LR":
        delta = run.deflection_m[side][0]  # in hand-frame axes
        expected_local = -k_t * delta
        got_local = run.hand_rotation[side][0].T @ run.force_on_club_n[side][0]
        np.testing.assert_allclose(got_local, expected_local, rtol=1e-6, atol=1e-6)
        assert np.linalg.norm(delta) > 5e-4  # the displacement is felt
    sim2 = _static_sim(spec_bytes, motion)
    run2 = sim2.probe_deflection([0.0, 0.0, 0.0], [0.0, 0.0, np.radians(0.5)])
    for side in "LR":
        theta = run2.rotation_deflection_rad[side][0]
        tau_local = run2.hand_rotation[side][0].T @ run2.torque_on_club_nm[side][0]
        # restoring: the magnitude is K_r * angle about the deflection axis
        assert np.linalg.norm(tau_local) > 0.5 * k_r.min() * theta


def test_force_points_toward_the_hand_frame(spec_bytes, motion) -> None:
    """Force on the club opposes the club's displacement from the hand frame."""
    sim = _static_sim(spec_bytes, motion)
    run = sim.probe_deflection([2.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0])
    for side in "LR":
        assert float(run.force_on_club_n[side][0] @ np.array([1.0, 0.0, 0.0])) < 0.0


def test_lowpass_trajectory_contract() -> None:
    t = np.linspace(0.0, 1.0, 101)
    q = np.sin(2 * np.pi * 2 * t)[:, None]
    np.testing.assert_allclose(lowpass_trajectory(t, q, 25.0), q, atol=5e-3)
    with pytest.raises(ValueError):
        lowpass_trajectory(t, q, 80.0)
    with pytest.raises(ValueError):
        lowpass_trajectory(t[:10], q[:10], 5.0)


def test_analysis_uses_bushing_split_method(static_run) -> None:
    _, run = static_run
    analyses = analyze_run(run)
    assert analyses[-1].split_method == "bushing"
    assert analyses[-1].net_force_n is not None
    assert hand_power_w(run, "L").shape == run.time_s.shape
    with pytest.raises(ValueError):
        hand_power_w(run, "X")


@pytest.mark.slow
def test_driven_swing_deflection_within_owner_limits(spec_bytes, motion) -> None:
    names, t, q = motion
    sim = BushingGripSimulator(spec_bytes, names, t, lowpass_trajectory(t, q, 25.0))
    run = sim.run(accuracy=1e-5)
    for side in "LR":
        assert np.linalg.norm(run.deflection_m[side], axis=1).max() <= MAX_DEFLECTION_M
        assert run.rotation_deflection_rad[side].max() <= MAX_ROTATION_RAD
    peak = max(np.linalg.norm(run.force_on_club_n[s], axis=1).max() for s in "LR")
    assert peak > 20.0  # the swing genuinely loads the grip
