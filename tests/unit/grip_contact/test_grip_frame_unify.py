"""Tests for grip-frame unification and Drake contact-normal recording (issue #13298, HRG-GS-G0)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    ClubDynamics,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.interface import (
    appearance_to_shared_grip_frame,
    shared_to_appearance_grip_frame,
)
from src.shared.python.grip_contact.pad_layout import PadLayout
from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance.grip_contact_patch import (
    ThumbPatch,
    thumb_palm_patch,
)
from src.shared.python.model_appearance.grip_hands import (
    grip_hand_frame,
    grip_radius_at,
)
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
FIXTURES = ROOT / "tests/fixtures/club_face"


def test_appearance_to_shared_grip_frame_vector() -> None:
    """Appearance frame (z = grip axis, x = toe, y = trail) maps to
    shared grip frame (x = grip axis, y = toe, z = trail)."""
    v_app = np.array([1.0, 2.0, 3.0])  # x=toe, y=trail, z=grip
    v_shared = appearance_to_shared_grip_frame(v_app)
    # in shared frame: x=grip (z_app), y=toe (x_app), z=trail (y_app)
    np.testing.assert_allclose(v_shared, [3.0, 1.0, 2.0], atol=1e-15)
    # round-trip
    v_back = shared_to_appearance_grip_frame(v_shared)
    np.testing.assert_allclose(v_back, v_app, atol=1e-15)


def test_appearance_to_shared_grip_frame_transform() -> None:
    """4x4 transform mapping local appearance coordinates to club frame
    converts to a transform mapping local shared coordinates to club frame."""
    # Build a sample 4x4 transform with proper rotation
    # col 0 = toe, col 1 = trail, col 2 = grip axis, col 3 = origin
    T_app = np.eye(4)
    # arbitrary proper rotation
    R = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    T_app[:3, :3] = R
    T_app[:3, 3] = [0.1, 0.2, 0.3]

    T_shared = appearance_to_shared_grip_frame(T_app)

    # In T_shared:
    # col 0 must be col 2 of T_app (grip axis)
    # col 1 must be col 0 of T_app (toe axis)
    # col 2 must be col 1 of T_app (trail axis)
    np.testing.assert_allclose(T_shared[:3, 0], T_app[:3, 2], atol=1e-15)
    np.testing.assert_allclose(T_shared[:3, 1], T_app[:3, 0], atol=1e-15)
    np.testing.assert_allclose(T_shared[:3, 2], T_app[:3, 1], atol=1e-15)
    np.testing.assert_allclose(T_shared[:3, 3], T_app[:3, 3], atol=1e-15)

    # Invariant: for any point in appearance frame, its position in the club frame
    # matches that of the converted point in the shared frame
    p_app = np.array([0.05, -0.02, 0.08])
    p_shared = appearance_to_shared_grip_frame(p_app)
    club_from_app = T_app[:3, :3] @ p_app + T_app[:3, 3]
    club_from_shared = T_shared[:3, :3] @ p_shared + T_shared[:3, 3]
    np.testing.assert_allclose(club_from_shared, club_from_app, atol=1e-15)

    # Round-trip
    T_back = shared_to_appearance_grip_frame(T_shared)
    np.testing.assert_allclose(T_back, T_app, atol=1e-15)


def test_conversion_preconditions() -> None:
    """Preconditions validate input shape, finitude, and proper rotation."""
    with pytest.raises((ValueError, TypeError)):
        appearance_to_shared_grip_frame([1.0, 2.0])  # not 3D
    with pytest.raises(ValueError):
        appearance_to_shared_grip_frame([np.nan, 1.0, 2.0])
    bad_rot = np.eye(4)
    bad_rot[:3, :3] *= 2.0
    with pytest.raises(ValueError, match="proper rotation"):
        appearance_to_shared_grip_frame(bad_rot)


def test_round_trip_grip_contact_patch_to_pad_layout() -> None:
    """Acceptance criterion: every grip_contact_patch point maps to the same
    PadLayout coordinate to <= 1e-12 m."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    club = ca.assembly_from_spec(spec)
    assert club is not None
    pose = DEFAULT_GRIP_POSE
    lead_frame = grip_hand_frame(club, Hand.LEAD, pose)
    radius = grip_radius_at(club, pose.distance_below_butt_m(Hand.LEAD))
    patch = thumb_palm_patch(lead_frame, grip_radius_m=radius, pose=pose)

    # 1. Test patch point in club frame mapped to shared/PadLayout frame and back
    shared_lead_frame = appearance_to_shared_grip_frame(lead_frame)
    p_club = np.array(patch.point_club_m)

    # Map club point into shared (PadLayout) frame of lead hand:
    # p_pad = R_shared^T * (p_club - origin)
    p_pad = shared_lead_frame[:3, :3].T @ (p_club - shared_lead_frame[:3, 3])

    # Reconstruct club point from PadLayout coordinates:
    p_club_rec = shared_lead_frame[:3, :3] @ p_pad + shared_lead_frame[:3, 3]
    np.testing.assert_allclose(p_club_rec, p_club, atol=1e-12)

    # 2. Test round trip through appearance <-> shared grip frame conversions
    p_app = shared_to_appearance_grip_frame(p_pad)
    p_pad_rt = appearance_to_shared_grip_frame(p_app)
    np.testing.assert_allclose(p_pad_rt, p_pad, atol=1e-12)

    # 3. Axial coordinate in PadLayout frame (index 0) matches appearance z coordinate
    p_app_direct = lead_frame[:3, :3].T @ (p_club - lead_frame[:3, 3])
    assert abs(p_pad[0] - p_app_direct[2]) <= 1e-12
    assert abs(p_pad[1] - p_app_direct[0]) <= 1e-12
    assert abs(p_pad[2] - p_app_direct[1]) <= 1e-12

    # 4. Palm reference point club -> PadLayout -> club round trip
    palm_club = np.array(patch.palm_reference_club_m)
    palm_pad = shared_lead_frame[:3, :3].T @ (palm_club - shared_lead_frame[:3, 3])
    palm_club_rec = shared_lead_frame[:3, :3] @ palm_pad + shared_lead_frame[:3, 3]
    np.testing.assert_allclose(palm_club_rec, palm_club, atol=1e-12)

    # 5. Thumb axis point round trip
    axis_app = np.array(patch.thumb_axis_hand_m)
    axis_pad = appearance_to_shared_grip_frame(axis_app)
    axis_app_rec = shared_to_appearance_grip_frame(axis_pad)
    np.testing.assert_allclose(axis_app_rec, axis_app, atol=1e-12)
    assert abs(axis_pad[0] - patch.station_m) <= 1e-12


@pytest.mark.requires_drake
def test_drake_pure_tangential_load_normal_stays_constant() -> None:
    """Acceptance criterion: A Drake test (requires_drake) with a pure-tangential load
    shows the recorded normal stays equal to the static squeeze to <= 1e-9 N,
    and the tangential magnitude matches the applied load to <= 1 %."""
    try:
        import pydrake.multibody.plant  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"pydrake not installed: {exc}")

    from src.engines.physics_engines.drake.python.grip_contact_sim import (
        ClubInHands,
        WeldClubKinematics,
        simulate_grip_contact,
    )
    from src.shared.python.grip_contact.pad_contact import build_pad_model, grip_axis

    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)
    squeeze_target = 1100.0
    pads = build_pad_model(interface, squeeze_target)

    # 1. Verify on ClubInHands plant: pure tangential sliding velocity along the
    # grip cylinder axis produces huge tangential friction load (>1000 N) while
    # recorded normal stays exactly equal to the static squeeze to <= 1e-9 N.
    kin = WeldClubKinematics(spec_bytes, list(swing.names))
    sim = ClubInHands(spec, interface, pads)
    weld0 = kin.state(swing.q[0], np.zeros_like(swing.q[0]))
    sim.set_hand(weld0)
    sim.place_club_at(weld0)
    rot_world, _ = sim.club_pose()
    axis_club, _ = grip_axis(interface, pads)
    axis_world = rot_world @ axis_club

    # Static squeeze with zero velocity
    w_static = sim.wrenches()
    squeeze_static = sum(float(w_static[s][2].sum()) for s in ("L", "R"))
    assert squeeze_static == pytest.approx(2.0 * squeeze_target, rel=0.01)

    # Apply pure tangential sliding along the cylinder axis
    x = sim.club_state()
    x[7:10] = 0.05 * axis_world  # linear velocity along grip axis
    x[10:13] = 0.0  # zero angular velocity
    sim.set_club(x)

    w_tang = sim.wrenches()
    squeeze_under_tang = sum(float(w_tang[s][2].sum()) for s in ("L", "R"))
    tang_friction_total = sum(float(w_tang[s][3].sum()) for s in ("L", "R"))

    # Tangential load is active and large
    assert tang_friction_total > 1000.0
    # Acceptance criterion: recorded normal stays equal to static squeeze to <= 1e-9 N
    normal_diff = abs(squeeze_under_tang - squeeze_static)
    assert normal_diff <= 1e-9, (
        f"Recorded normal changed by {normal_diff:.3e} N under pure tangential load "
        f"(expected <= 1e-9 N). Static: {squeeze_static:.6f} N, tang: {squeeze_under_tang:.6f} N"
    )

    # 2. Dynamic hold simulation with gravity directed purely along the grip axis:
    # verifies tangential load magnitude matches applied load to <= 1 %.
    g_mag = 9.81
    club_mass = ClubDynamics.from_spec(spec).mass_kg
    applied_tangential_load_n = club_mass * g_mag

    spec_axial_g = dict(spec)
    spec_axial_g["gravity_m_s2"] = [float(x) for x in g_mag * axis_world]

    run_tangential = simulate_grip_contact(
        json.dumps(spec_axial_g).encode("utf-8"),
        swing,
        pads,
        interface,
        t_end_s=0.05,
        hold=True,
    )

    # At t=0 before dynamic motion: recorded normal equals static squeeze to <= 1e-9 N
    t0_squeeze = sum(
        float(run_tangential.normal_force_n[s][0].sum()) for s in ("L", "R")
    )
    assert abs(t0_squeeze - squeeze_static) <= 1e-9

    # Acceptance criterion: tangential magnitude matches the applied load to <= 1 %
    assert run_tangential.tangential_force_n is not None
    total_tangential_recorded = sum(
        float(run_tangential.tangential_force_n[s][-1].sum()) for s in ("L", "R")
    )
    rel_error_pads = (
        abs(total_tangential_recorded - applied_tangential_load_n)
        / applied_tangential_load_n
    )
    assert rel_error_pads <= 0.01, (
        f"Recorded pad tangential force {total_tangential_recorded:.4f} N differs from "
        f"applied load {applied_tangential_load_n:.4f} N by {rel_error_pads * 100:.2f}% (limit 1%)"
    )

    # Also check total reaction force on the club along the grip axis
    f_total_on_club = (
        run_tangential.series.force_on_club_n["L"][-1]
        + run_tangential.series.force_on_club_n["R"][-1]
    )
    f_tangential_along_axis = abs(float(f_total_on_club @ axis_world))
    rel_error_net = (
        abs(f_tangential_along_axis - applied_tangential_load_n)
        / applied_tangential_load_n
    )
    assert rel_error_net <= 0.01, (
        f"Net tangential force along axis {f_tangential_along_axis:.4f} N differs from "
        f"applied load {applied_tangential_load_n:.4f} N by {rel_error_net * 100:.2f}% (limit 1%)"
    )
