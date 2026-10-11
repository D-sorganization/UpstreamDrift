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
    input_kinematics_report,
    lowpass_trajectory,
    prepare_motion,
)
from src.shared.python.grip_contact import (  # noqa: E402
    ClubDynamics,
    ClubKinematics,
    GripInterface,
    couple_consistency,
    decompose_hand_forces,
    load_coordinate_swing,
    peak_squeeze_n,
    required_hand_moment_nm,
)

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


FIRST_IK_SWITCH_S = 0.95

# Physics checks replacing the flat 500 N internal-force bound (owner decision
# on PR #11774): a flat bound cannot tell a couple-carrying force pair from the
# fighting-hands artefact.  See grip_contact.couple_check and
# DESIGN_DECISIONS.md section 17.
MAX_SQUEEZE_N = 50.0
MAX_COUPLE_RELATIVE_ERROR = 0.05


def _hand_pairs(run):
    return (
        (run.force_on_club_n["L"], run.force_on_club_n["R"]),
        (run.grip_point_m["L"], run.grip_point_m["R"]),
        (run.torque_on_club_nm["L"], run.torque_on_club_nm["R"]),
    )


def _driven_run(spec_bytes, motion, t_end, valid_window):
    names, t, q = motion
    if valid_window:  # cut before filtering, see run_grip_kinetics.run_swing
        keep = t < FIRST_IK_SWITCH_S - 0.01
        t, q = t[keep], q[keep]
    q_prepared, _ = prepare_motion(t, q, names)
    sim = BushingGripSimulator(spec_bytes, names, t, q_prepared)
    return sim.run(t_end, accuracy=1e-3)


def _worst(run):
    trans = max(np.linalg.norm(run.deflection_m[s], axis=1).max() for s in "LR")
    rot = max(run.rotation_deflection_rad[s].max() for s in "LR")
    dec = decompose_hand_forces(
        run.force_on_club_n["L"],
        run.force_on_club_n["R"],
        run.grip_point_m["L"],
        run.grip_point_m["R"],
    )
    return trans, rot, dec


@pytest.mark.slow
def test_driven_swing_before_ik_switch_within_limits(spec_bytes, motion) -> None:
    """Valid window: deflection and internal force within the stated bounds."""
    run = _driven_run(spec_bytes, motion, None, valid_window=True)
    trans, rot, dec = _worst(run)
    assert trans <= MAX_DEFLECTION_M
    assert rot <= MAX_ROTATION_RAD
    _assert_internal_force_is_physical(run, json.loads(spec_bytes))
    peak = max(np.linalg.norm(run.force_on_club_n[s], axis=1).max() for s in "LR")
    assert peak > 20.0  # the swing genuinely loads the grip


# ---------------------------------------------------------------------------
# Full 0-1.8 s window, driven by the closure-consistent fitted swings shared
# with OSV-10 (tests/fixtures/club_face, IK marker RMS about 33 mm, replayed
# identically in every engine).  Same bounds as the valid-window test.
FIXTURES = ROOT / "tests/fixtures/club_face"
FULL_SWING_ACCURACY = 1e-3  # RK-Merson; 1e-5 and CPodes agree within 0.2 %


def _fixture_swing(club: str):
    spec_bytes = (MODELS / f"full_body_spec_anthro_{club}.json").read_bytes()
    order = json.loads(spec_bytes)["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz", FIXTURES / "address_poses.json", club, order
    )
    return spec_bytes, swing


@pytest.fixture(scope="module", params=["driver", "iron7"])
def full_swing_run(request):
    # Prescribed as committed: the reference is a smooth 1 kHz dynamics replay
    # (condition_trajectory repairs 0 frames); the 25 Hz filter of the IK
    # candidate pipeline changes the peaks by < 4 % and would open the loop.
    spec_bytes, swing = _fixture_swing(request.param)
    sim = BushingGripSimulator(spec_bytes, swing.names, swing.time_s, swing.q)
    return request.param, sim.run(None, accuracy=FULL_SWING_ACCURACY)


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_fixture_swing_closes_the_two_hand_loop(club: str) -> None:
    """Input kinematics: the two-hand loop is closed and hand speeds are real."""
    spec_bytes, swing = _fixture_swing(club)
    rep = input_kinematics_report(spec_bytes, swing.names, swing.time_s, swing.q)
    assert rep["closure_distance_m"].max() < 1.0e-4  # < 0.1 mm over the swing
    assert rep["closure_angle_deg"].max() < 0.1
    for side in ("left", "right"):
        # measured wrist markers peak at 9.7 m/s (driver); the IK candidate
        # reached 95 m/s
        assert 6.0 < rep[f"{side}_hand_speed_m_s"].max() < 13.0


@pytest.mark.slow
def test_full_swing_deflection_within_limits(full_swing_run) -> None:
    _, run = full_swing_run
    assert run.time_s[-1] > 1.8
    trans, rot, _ = _worst(run)
    assert trans <= MAX_DEFLECTION_M
    assert rot <= MAX_ROTATION_RAD


def _assert_internal_force_is_physical(run, spec) -> None:
    """Squeeze bound plus couple consistency (Newton-Euler of the club)."""
    forces, points, torques = _hand_pairs(run)
    assert peak_squeeze_n(*forces, *points) <= MAX_SQUEEZE_N
    kinematics = ClubKinematics(
        run.club_rotation,
        run.club_omega_rad_s,
        run.club_alpha_rad_s2,
        run.club_com_m,
        run.club_com_acceleration_m_s2,
    )
    required = required_hand_moment_nm(
        kinematics,
        ClubDynamics.from_spec(spec),
        np.asarray(spec["gravity_m_s2"]),
        0.5 * (points[0] + points[1]),
    )
    res = couple_consistency(forces, points, torques, required)
    assert res.checked.mean() > 0.5  # the check covers the loaded swing
    assert res.checked[int(np.argmax(res.actual_transverse_n))]
    assert res.max_relative_error() <= MAX_COUPLE_RELATIVE_ERROR


@pytest.mark.slow
def test_full_swing_internal_force_is_physical(full_swing_run) -> None:
    club, run = full_swing_run
    spec = json.loads((MODELS / f"full_body_spec_anthro_{club}.json").read_bytes())
    _assert_internal_force_is_physical(run, spec)


def test_candidate_is_inconsistent_with_the_two_hand_closure(
    spec_bytes, motion
) -> None:
    """Documents the input quality: the IK candidate leaves the loop open."""
    names, t, q = motion
    rep = input_kinematics_report(spec_bytes, names, t[:40], q[:40])
    assert 0.10 < rep["closure_distance_m"][0] < 0.17  # 134 mm at address
    assert 40.0 < rep["closure_angle_deg"][0] < 60.0  # 51 deg at address


def test_prepare_motion_repairs_the_candidate_glitches(motion) -> None:
    names, t, q = motion
    q_clean, report = prepare_motion(t, q, names)
    assert report.repaired_frames >= 5
    assert np.isfinite(q_clean).all()
