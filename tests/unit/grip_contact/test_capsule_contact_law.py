"""Tests for generalised shared contact law: sphere-capsule and capsule-capsule.

Issue #13300 (HRG-GS-G2).
Acceptance criteria:
- Zero force at separation (gap >= 0) to exactly 0.0 N; unilateral normal (never pulling) over 10^4 random poses.
- Friction stays inside the cone |f_t| <= mu * |f_n| * (1 + 1e-9).
- Action/reaction: sum of forces and moments about any point <= 1e-12.
- Degenerate (parallel, coincident) capsule pairs raise or return a documented finite result; property tests cover them.
- The sphere-cylinder case agrees with the legacy path to <= 1e-12 N.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import BushingState, GripInterface
from src.shared.python.grip_contact.pad_contact import (
    ContactCapsule,
    ContactSphere,
    PairWrench,
    build_pad_model,
    capsule_capsule,
    closest_points_segments,
    pad_wrench,
    sphere_capsule,
)
from src.shared.python.grip_contact.pad_layout import PadLayout
from src.shared.python.motion_matching.contact_law import ContactParameters

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


@pytest.fixture(scope="module")
def default_law() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=1.0e5,
        dissipation_s_m=0.1,
        static_friction=0.9,
        dynamic_friction=0.7,
        viscous_friction=0.05,
        transition_velocity_m_s=1.0e-3,
    )


@pytest.fixture(scope="module")
def interface() -> GripInterface:
    return GripInterface.from_spec(json.loads(SPEC.read_text()))


def _random_quaternion_to_rot(rng: np.random.Generator) -> np.ndarray:
    """Generate a uniform random 3D rotation matrix via quaternion."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, 3)
    q = np.array(
        [
            math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
            math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
            math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
            math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
        ]
    )
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _random_bushing_state(
    rng: np.random.Generator, pos_range: float = 0.5
) -> BushingState:
    """Generate a random BushingState."""
    rot = _random_quaternion_to_rot(rng)
    pos = rng.uniform(-pos_range, pos_range, size=3)
    vel = rng.uniform(-1.0, 1.0, size=3)
    omega = rng.uniform(-2.0, 2.0, size=3)
    return BushingState(rot, pos, vel, omega)


def test_zero_force_at_separation_and_unilateral_normal_10k_poses(
    default_law: ContactParameters,
) -> None:
    """AC 1: Zero force at separation (gap >= 0) to exactly 0.0 N;

    unilateral normal (never pulling) over 10^4 random poses.
    """
    rng = np.random.default_rng(13300)
    num_poses = 10_000
    half = num_poses // 2

    # Part A: Sphere-Capsule (5000 poses)
    for _ in range(half):
        r_sph = rng.uniform(0.005, 0.03)
        r_cap = rng.uniform(0.005, 0.03)
        sph = ContactSphere(center_m=rng.uniform(-0.05, 0.05, size=3), radius_m=r_sph)
        p0 = rng.uniform(-0.1, 0.1, size=3)
        p1 = p0 + rng.uniform(-0.1, 0.1, size=3)
        cap = ContactCapsule(p0_m=p0, p1_m=p1, radius_m=r_cap)

        state_a = _random_bushing_state(rng, pos_range=0.1)
        state_b = _random_bushing_state(rng, pos_range=0.1)

        patch = rng.uniform(0.0, 0.005)
        spin_trans = rng.uniform(0.5, 2.0)

        wrench = sphere_capsule(
            sph,
            cap,
            state_a,
            state_b,
            default_law,
            patch_radius_m=patch,
            spin_transition_rad_s=spin_trans,
        )

        if wrench.penetration_m == 0.0:
            np.testing.assert_array_equal(wrench.force_a_n, np.zeros(3))
            np.testing.assert_array_equal(wrench.force_b_n, np.zeros(3))
            np.testing.assert_array_equal(wrench.moment_a_nm, np.zeros(3))
            np.testing.assert_array_equal(wrench.moment_b_nm, np.zeros(3))
            assert wrench.normal_force_mag_n == 0.0
            assert wrench.tangential_force_mag_n == 0.0
        else:
            assert wrench.normal_force_mag_n >= 0.0
            # Normal pushes body B along +n and body A along -n (unilateral / compressive)
            f_norm_b = float(np.dot(wrench.force_b_n, wrench.normal))
            f_norm_a = float(np.dot(wrench.force_a_n, -wrench.normal))
            assert f_norm_b >= -1e-12
            assert f_norm_a >= -1e-12

    # Part B: Capsule-Capsule (5000 poses)
    for _ in range(half):
        r_a = rng.uniform(0.005, 0.03)
        r_b = rng.uniform(0.005, 0.03)
        p0_a = rng.uniform(-0.1, 0.1, size=3)
        p1_a = p0_a + rng.uniform(-0.1, 0.1, size=3)
        p0_b = rng.uniform(-0.1, 0.1, size=3)
        p1_b = p0_b + rng.uniform(-0.1, 0.1, size=3)
        cap_a = ContactCapsule(p0_m=p0_a, p1_m=p1_a, radius_m=r_a)
        cap_b = ContactCapsule(p0_m=p0_b, p1_m=p1_b, radius_m=r_b)

        state_a = _random_bushing_state(rng, pos_range=0.1)
        state_b = _random_bushing_state(rng, pos_range=0.1)

        patch = rng.uniform(0.0, 0.005)
        spin_trans = rng.uniform(0.5, 2.0)

        wrench = capsule_capsule(
            cap_a,
            cap_b,
            state_a,
            state_b,
            default_law,
            patch_radius_m=patch,
            spin_transition_rad_s=spin_trans,
        )

        if wrench.penetration_m == 0.0:
            np.testing.assert_array_equal(wrench.force_a_n, np.zeros(3))
            np.testing.assert_array_equal(wrench.force_b_n, np.zeros(3))
            np.testing.assert_array_equal(wrench.moment_a_nm, np.zeros(3))
            np.testing.assert_array_equal(wrench.moment_b_nm, np.zeros(3))
            assert wrench.normal_force_mag_n == 0.0
            assert wrench.tangential_force_mag_n == 0.0
        else:
            assert wrench.normal_force_mag_n >= 0.0
            f_norm_b = float(np.dot(wrench.force_b_n, wrench.normal))
            f_norm_a = float(np.dot(wrench.force_a_n, -wrench.normal))
            assert f_norm_b >= -1e-12
            assert f_norm_a >= -1e-12


def test_friction_cone_conformance(default_law: ContactParameters) -> None:
    """AC 2: Friction stays inside the cone |f_t| <= mu * |f_n| * (1 + 1e-9)."""
    rng = np.random.default_rng(13301)
    for _ in range(500):
        # Generate penetrating pose
        r_a = 0.02
        r_b = 0.02
        cap_a = ContactCapsule(
            p0_m=np.zeros(3), p1_m=np.array([0.0, 0.0, 0.1]), radius_m=r_a
        )
        # Place cap_b penetrating by ~2 mm
        overlap_dist = (r_a + r_b) - rng.uniform(0.001, 0.005)
        cap_b = ContactCapsule(
            p0_m=np.array([overlap_dist, 0.0, 0.0]),
            p1_m=np.array([overlap_dist, 0.0, 0.1]),
            radius_m=r_b,
        )
        state_a = BushingState(
            np.eye(3), np.zeros(3), rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3)
        )
        state_b = BushingState(
            np.eye(3), np.zeros(3), rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3)
        )

        wrench = capsule_capsule(cap_a, cap_b, state_a, state_b, default_law)
        assert wrench.penetration_m > 0.0

        f_t = wrench.tangential_force_mag_n
        f_n = wrench.normal_force_mag_n
        # Evaluate local mu(v_tangential)
        v_rel = (
            state_b.velocity_m_s
            + np.cross(state_b.omega_rad_s, wrench.contact_point_m - state_b.position_m)
        ) - (
            state_a.velocity_m_s
            + np.cross(state_a.omega_rad_s, wrench.contact_point_m - state_a.position_m)
        )
        v_t = v_rel - wrench.normal * float(np.dot(wrench.normal, v_rel))
        speed = float(np.linalg.norm(v_t))
        ratio = speed / default_law.transition_velocity_m_s
        mu = (
            default_law.dynamic_friction
            + (default_law.static_friction - default_law.dynamic_friction)
            * math.exp(-(ratio**2))
            + default_law.viscous_friction * speed
        )

        assert f_t <= mu * f_n * (1.0 + 1e-9) + 1e-12


def test_action_reaction_forces_and_moments_about_arbitrary_points(
    default_law: ContactParameters,
) -> None:
    """AC 3: Action/reaction: sum of forces and moments about any point <= 1e-12."""
    rng = np.random.default_rng(13302)
    test_points = [
        np.zeros(3),
        np.array([0.1, 0.2, 0.3]),
        np.array([-10.0, 25.0, -100.0]),
        np.array([123.456, -789.012, 345.678]),
    ]

    for _ in range(500):
        cap_a = ContactCapsule(
            p0_m=rng.uniform(-0.05, 0.05, 3),
            p1_m=rng.uniform(-0.05, 0.05, 3),
            radius_m=rng.uniform(0.01, 0.03),
        )
        cap_b = ContactCapsule(
            p0_m=rng.uniform(-0.05, 0.05, 3),
            p1_m=rng.uniform(-0.05, 0.05, 3),
            radius_m=rng.uniform(0.01, 0.03),
        )
        state_a = _random_bushing_state(rng, pos_range=0.2)
        state_b = _random_bushing_state(rng, pos_range=0.2)

        wrench = capsule_capsule(
            cap_a,
            cap_b,
            state_a,
            state_b,
            default_law,
            patch_radius_m=rng.uniform(0.0, 0.005),
            spin_transition_rad_s=rng.uniform(0.5, 2.0),
        )

        # Force action/reaction
        np.testing.assert_allclose(wrench.force_a_n + wrench.force_b_n, 0.0, atol=1e-12)

        # Moment action/reaction about arbitrary points
        for pt in test_points:
            tot_moment = wrench.total_moment_about_point(pt)
            np.testing.assert_allclose(tot_moment, 0.0, atol=1e-12)


def test_degenerate_capsule_pairs_finite_results(
    default_law: ContactParameters,
) -> None:
    """AC 4: Degenerate (parallel, coincident) capsule pairs return documented finite results."""
    # 1. Exactly parallel separated capsules
    c1 = ContactCapsule(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1]), radius_m=0.01
    )
    c2 = ContactCapsule(
        np.array([0.05, 0.0, 0.0]), np.array([0.05, 0.0, 0.1]), radius_m=0.01
    )
    w_sep = capsule_capsule(c1, c2, law=default_law)
    assert w_sep.penetration_m == 0.0
    assert np.all(np.isfinite(w_sep.force_a_n))
    assert np.all(np.isfinite(w_sep.normal))
    assert np.all(np.isfinite(w_sep.contact_point_m))

    # 2. Exactly parallel overlapping capsules
    c3 = ContactCapsule(
        np.array([0.015, 0.0, 0.0]), np.array([0.015, 0.0, 0.1]), radius_m=0.01
    )
    w_over = capsule_capsule(c1, c3, law=default_law)
    assert w_over.penetration_m == pytest.approx(0.005)
    assert w_over.normal_force_mag_n > 0.0
    assert np.all(np.isfinite(w_over.force_a_n))
    assert np.all(np.isfinite(w_over.moment_a_nm))
    np.testing.assert_allclose(w_over.force_a_n + w_over.force_b_n, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        w_over.total_moment_about_point(np.zeros(3)), 0.0, atol=1e-12
    )

    # 3. Exactly coincident capsules (identical endpoints)
    w_coin = capsule_capsule(c1, c1, law=default_law)
    assert w_coin.penetration_m == pytest.approx(0.02)  # r1 + r2
    assert np.all(np.isfinite(w_coin.force_a_n))
    assert np.all(np.isfinite(w_coin.moment_a_nm))
    assert np.all(np.isfinite(w_coin.normal))
    assert pytest.approx(np.linalg.norm(w_coin.normal)) == 1.0
    np.testing.assert_allclose(w_coin.force_a_n + w_coin.force_b_n, 0.0, atol=1e-12)

    # 4. Coincident antiparallel capsules (reversed endpoints)
    c1_rev = ContactCapsule(
        np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 0.0]), radius_m=0.01
    )
    w_anti = capsule_capsule(c1, c1_rev, law=default_law)
    assert w_anti.penetration_m == pytest.approx(0.02)
    assert np.all(np.isfinite(w_anti.force_a_n))
    assert np.all(np.isfinite(w_anti.normal))

    # 5. Zero-length capsules (spheres)
    c_zero1 = ContactCapsule(np.zeros(3), np.zeros(3), radius_m=0.01)
    c_zero2 = ContactCapsule(np.zeros(3), np.zeros(3), radius_m=0.01)
    w_zero = capsule_capsule(c_zero1, c_zero2, law=default_law)
    assert w_zero.penetration_m == pytest.approx(0.02)
    assert np.all(np.isfinite(w_zero.force_a_n))
    assert np.all(np.isfinite(w_zero.normal))

    # 6. Property tests: sweep distance from 0 to 0.05 between parallel capsules
    for d in np.linspace(0.0, 0.05, 51):
        c_sweep = ContactCapsule(
            np.array([d, 0.0, 0.0]), np.array([d, 0.0, 0.1]), radius_m=0.01
        )
        w = capsule_capsule(c1, c_sweep, law=default_law)
        assert np.all(np.isfinite(w.force_a_n))
        assert np.all(np.isfinite(w.moment_a_nm))
        assert np.all(np.isfinite(w.contact_point_m))
        assert np.all(np.isfinite(w.normal))


def test_sphere_cylinder_agrees_with_legacy_path(interface: GripInterface) -> None:
    """AC 5: The sphere-cylinder case agrees with the legacy path to <= 1e-12 N."""
    squeeze = 1000.0
    model = build_pad_model(interface, squeeze)
    side = "R"
    layout: PadLayout = model.layout
    pads_grip_frame = layout.positions_grip_frame(side)

    # Pick a pad in the middle
    pad_idx = 0
    pad_local = pads_grip_frame[pad_idx]
    sph = ContactSphere(center_m=pad_local, radius_m=layout.pad_radius_m)

    # Grip cylinder modeled as capsule along the grip axis in club frame
    lo, hi = model.cylinder.axial_range_m
    axis_offset = layout.axis_offset_grip_frame(side)
    # The grip cylinder axis is club x-axis
    p0 = axis_offset + np.array([lo, 0.0, 0.0])
    p1 = axis_offset + np.array([hi, 0.0, 0.0])
    cap = ContactCapsule(p0_m=p0, p1_m=p1, radius_m=model.cylinder.radius_m)

    # Test conditions:
    # 1. Static at rest
    hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))

    w_leg = pad_wrench(
        model, side, hand, club, positions_grip_frame=np.array([pad_local])
    )
    w_new = sphere_capsule(
        sph,
        cap,
        hand,
        club,
        model.law,
        patch_radius_m=model.patch_radius_m,
        spin_transition_rad_s=model.spin_transition_rad_s,
    )
    # w_leg.force_n is force ON CLUB
    # w_new.force_b_n is force ON CLUB (body B)
    np.testing.assert_allclose(w_new.force_b_n, w_leg.force_n, atol=1e-12)
    np.testing.assert_allclose(w_new.moment_b_nm, w_leg.moment_nm, atol=1e-12)

    # 2. Offset transverse
    club_off = BushingState(
        np.eye(3), np.array([0.0, 1e-4, 0.0]), np.zeros(3), np.zeros(3)
    )
    w_leg_off = pad_wrench(
        model, side, hand, club_off, positions_grip_frame=np.array([pad_local])
    )
    w_new_off = sphere_capsule(
        sph,
        cap,
        hand,
        club_off,
        model.law,
        patch_radius_m=model.patch_radius_m,
        spin_transition_rad_s=model.spin_transition_rad_s,
    )
    np.testing.assert_allclose(w_new_off.force_b_n, w_leg_off.force_n, atol=1e-12)
    np.testing.assert_allclose(w_new_off.moment_b_nm, w_leg_off.moment_nm, atol=1e-12)

    # 3. Axial sliding velocity
    club_vel = BushingState(
        np.eye(3), np.zeros(3), np.array([0.2, 0.0, 0.0]), np.zeros(3)
    )
    w_leg_vel = pad_wrench(
        model, side, hand, club_vel, positions_grip_frame=np.array([pad_local])
    )
    w_new_vel = sphere_capsule(
        sph,
        cap,
        hand,
        club_vel,
        model.law,
        patch_radius_m=model.patch_radius_m,
        spin_transition_rad_s=model.spin_transition_rad_s,
    )
    np.testing.assert_allclose(w_new_vel.force_b_n, w_leg_vel.force_n, atol=1e-12)
    np.testing.assert_allclose(w_new_vel.moment_b_nm, w_leg_vel.moment_nm, atol=1e-12)

    # 4. Relative angular velocity with spin friction
    hand_rot = BushingState(
        np.eye(3), np.zeros(3), np.zeros(3), np.array([1.5, 0.0, 0.0])
    )
    w_leg_rot = pad_wrench(
        model, side, hand_rot, club, positions_grip_frame=np.array([pad_local])
    )
    w_new_rot = sphere_capsule(
        sph,
        cap,
        hand_rot,
        club,
        model.law,
        patch_radius_m=model.patch_radius_m,
        spin_transition_rad_s=model.spin_transition_rad_s,
    )
    np.testing.assert_allclose(w_new_rot.force_b_n, w_leg_rot.force_n, atol=1e-12)
    np.testing.assert_allclose(w_new_rot.moment_b_nm, w_leg_rot.moment_nm, atol=1e-12)


def test_dbc_preconditions(default_law: ContactParameters) -> None:
    """DbC: Preconditions raise ValueError / TypeError on invalid inputs."""
    with pytest.raises(ValueError, match="radius"):
        ContactSphere(center_m=np.zeros(3), radius_m=-0.01)

    with pytest.raises(ValueError, match="radius"):
        ContactCapsule(p0_m=np.zeros(3), p1_m=np.ones(3), radius_m=0.0)

    with pytest.raises(ValueError, match="finite"):
        ContactSphere(center_m=np.array([np.nan, 0.0, 0.0]), radius_m=0.01)

    with pytest.raises(ValueError, match="finite"):
        ContactCapsule(
            p0_m=np.zeros(3), p1_m=np.array([np.inf, 0.0, 0.0]), radius_m=0.01
        )

    sph = ContactSphere(center_m=np.zeros(3), radius_m=0.01)
    cap = ContactCapsule(p0_m=np.zeros(3), p1_m=np.ones(3), radius_m=0.01)

    with pytest.raises(ValueError, match="patch_radius"):
        sphere_capsule(sph, cap, law=default_law, patch_radius_m=-1.0)

    with pytest.raises(ValueError, match="spin_transition"):
        sphere_capsule(sph, cap, law=default_law, spin_transition_rad_s=0.0)

    with pytest.raises(TypeError, match="law"):
        sphere_capsule(sph, cap, law="not_a_law")  # type: ignore[arg-type]


def test_closest_points_segments_exact_geometry() -> None:
    """Verify segment-segment closest points on canonical orthogonal and skewed cases."""
    # Orthogonal crossing segments separated by 1.0 along z
    a0, a1 = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    b0, b1 = np.array([0.0, -1.0, 1.0]), np.array([0.0, 1.0, 1.0])
    pa, pb = closest_points_segments(a0, a1, b0, b1)
    np.testing.assert_allclose(pa, np.array([0.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(pb, np.array([0.0, 0.0, 1.0]), atol=1e-12)
