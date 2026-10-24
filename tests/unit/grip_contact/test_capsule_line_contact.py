"""Continuous capsule line-contact law (HRG-GS-G10 follow-up, #13308, epic #13297).

The closest-point capsule law placed the contact of a nearly parallel capsule
at the overlap midpoint while ``sin^2(theta) < 1e-10`` and at a capsule
endpoint just past that threshold: an 8 mm jump of the ``ten_finger`` trail
little-finger contact (13.5 % of its pitch couple).  The line-contact law
integrates the penetration along the capsule axis instead.

Acceptance criteria:
- the wrench and contact point are continuous through the old threshold;
- at ``theta = 0`` the law equals the previous closest-point law (same force,
  contact at the overlap centre), so preload calibrations are unchanged;
- a tilted capsule moves its contact toward the deeper end, to the
  normal-force-weighted centroid of the axial nodes;
- the node count is one shared rule that the OpenSim adapter reuses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    capsule_capsule,
    capsule_line_nodes,
    line_contact_node_count,
    sphere_capsule,
)
from src.shared.python.motion_matching.contact_law import ContactParameters

pytestmark = pytest.mark.unit

#: ``ten_finger`` trail little-finger capsule and the grip radius (metres).
R_CAP = 0.0075
HALF_LEN = 0.008
R_GRIP = 0.0135
PRELOAD = 0.00025


@pytest.fixture(scope="module")
def law() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=1.711e5,
        dissipation_s_m=0.3827,
        static_friction=0.9,
        dynamic_friction=0.7,
        viscous_friction=0.05,
        transition_velocity_m_s=1.0e-3,
    )


def _grip() -> ContactCapsule:
    return ContactCapsule(
        np.array([-0.2, 0.0, 0.0]), np.array([0.2, 0.0, 0.0]), radius_m=R_GRIP
    )


def _finger(theta_rad: float) -> ContactCapsule:
    """Finger capsule on the grip, pitched by ``theta_rad`` about its centre."""
    rho = R_GRIP + R_CAP - PRELOAD
    centre = np.array([-0.048, rho, 0.0])
    axis = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
    return ContactCapsule(
        centre - HALF_LEN * axis, centre + HALF_LEN * axis, radius_m=R_CAP
    )


def _moment_x_z(law: ContactParameters, theta_rad: float) -> tuple[float, float]:
    w = capsule_capsule(_finger(theta_rad), _grip(), law=law)
    return float(w.moment_b_nm[2]), float(w.contact_point_m[0])


def test_contact_is_continuous_through_old_parallel_threshold(
    law: ContactParameters,
) -> None:
    """No jump where ``sin^2(theta)`` crosses ``1e-10`` (``theta = 1e-5`` rad)."""
    below = capsule_capsule(_finger(0.99e-5), _grip(), law=law)
    above = capsule_capsule(_finger(1.01e-5), _grip(), law=law)
    jump_m = float(np.linalg.norm(above.contact_point_m - below.contact_point_m))
    assert jump_m < 1e-7, f"contact point jumped {jump_m * 1e3:.4f} mm"
    np.testing.assert_allclose(above.moment_b_nm, below.moment_b_nm, atol=1e-4)
    np.testing.assert_allclose(above.force_b_n, below.force_b_n, atol=1e-3)


def test_contact_point_is_lipschitz_over_small_tilts(law: ContactParameters) -> None:
    """Sweeping ``theta`` over ``[-1e-3, 1e-3]`` rad never moves the point abruptly."""
    thetas = np.linspace(-1e-3, 1e-3, 801)
    xs = np.array([_moment_x_z(law, t)[1] for t in thetas])
    steps = np.abs(np.diff(xs))
    # 2.5 um per 2.5 urad step: far below the 8 mm endpoint snap.
    assert steps.max() < 2.5e-6, f"largest step {steps.max() * 1e3:.4f} mm"


def test_parallel_capsule_equals_previous_closest_point_law(
    law: ContactParameters,
) -> None:
    """At ``theta = 0`` the force is ``k * delta`` at the overlap centre."""
    finger = _finger(0.0)
    w = capsule_capsule(finger, _grip(), law=law)
    assert w.penetration_m == pytest.approx(PRELOAD, abs=1e-15)
    assert w.normal_force_mag_n == pytest.approx(law.stiffness_n_m * PRELOAD, rel=1e-12)
    np.testing.assert_allclose(w.normal, [0.0, -1.0, 0.0], atol=1e-15)
    centre = 0.5 * (finger.p0_m + finger.p1_m)
    assert w.contact_point_m[0] == pytest.approx(centre[0], abs=1e-15)
    # Radial normal through the centre: no pitch couple about the capsule centre.
    lever = w.contact_point_m - centre
    assert abs(lever[0]) < 1e-15


def test_tilt_moves_contact_to_normal_force_weighted_node_centroid(
    law: ContactParameters,
) -> None:
    """A tilted capsule's contact is the ``f_n``-weighted centroid of its nodes."""
    theta = 2.0e-2  # deeper at the p0 end
    finger = _finger(theta)
    w = capsule_capsule(finger, _grip(), law=law)
    nodes = capsule_line_nodes(finger)
    weight = 1.0 / len(nodes)
    node_w = [sphere_capsule(n, _grip(), law=law) for n in nodes]
    f_n = np.array([weight * pw.normal_force_mag_n for pw in node_w])
    points = np.array([pw.contact_point_m for pw in node_w])
    np.testing.assert_allclose(
        w.contact_point_m, f_n @ points / f_n.sum(), rtol=0, atol=1e-15
    )
    assert w.normal_force_mag_n == pytest.approx(f_n.sum(), rel=1e-14)
    assert w.contact_point_m[0] < 0.5 * (finger.p0_m[0] + finger.p1_m[0])
    assert w.penetration_m == pytest.approx(max(pw.penetration_m for pw in node_w))


def test_lifted_end_contributes_nothing_and_force_stays_continuous(
    law: ContactParameters,
) -> None:
    """Tilting until one end separates shrinks the force smoothly to the deep end."""
    thetas = np.linspace(0.0, 0.2, 401)
    forces = np.array(
        [
            capsule_capsule(_finger(t), _grip(), law=law).normal_force_mag_n
            for t in thetas
        ]
    )
    assert np.all(np.isfinite(forces))
    assert np.abs(np.diff(forces)).max() < 0.05 * forces[0]


def test_action_reaction_with_relative_motion(law: ContactParameters) -> None:
    rng = np.random.default_rng(13308)
    for _ in range(50):
        a = BushingState(np.eye(3), np.zeros(3), rng.normal(size=3), rng.normal(size=3))
        b = BushingState(np.eye(3), np.zeros(3), rng.normal(size=3), rng.normal(size=3))
        w = capsule_capsule(
            _finger(rng.uniform(-0.05, 0.05)),
            _grip(),
            a,
            b,
            law,
            patch_radius_m=0.003,
        )
        np.testing.assert_allclose(w.force_a_n + w.force_b_n, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            w.total_moment_about_point(np.array([0.3, -0.2, 0.1])), 0.0, atol=1e-12
        )


def test_line_nodes_span_the_axis_with_shared_spacing() -> None:
    finger = _finger(0.0)
    nodes = capsule_line_nodes(finger)
    assert len(nodes) == line_contact_node_count(HALF_LEN, R_CAP) == 5
    assert all(isinstance(n, ContactSphere) and n.radius_m == R_CAP for n in nodes)
    np.testing.assert_array_equal(nodes[0].center_m, finger.p0_m)
    np.testing.assert_allclose(nodes[-1].center_m, finger.p1_m, atol=1e-16)
    spacing = np.diff([n.center_m[0] for n in nodes])
    np.testing.assert_allclose(spacing, 2.0 * HALF_LEN / (len(nodes) - 1), atol=1e-15)


@pytest.mark.parametrize(
    ("half_length", "radius", "expected"),
    [(0.0, 0.01, 2), (0.008, 0.0075, 5), (0.05, 0.01, 21)],
)
def test_line_contact_node_count(
    half_length: float, radius: float, expected: int
) -> None:
    assert line_contact_node_count(half_length, radius) == expected


@pytest.mark.parametrize(
    ("half_length", "radius"),
    [(-1e-3, 0.01), (math.nan, 0.01), (0.01, 0.0), (0.01, math.inf)],
)
def test_line_contact_node_count_preconditions(
    half_length: float, radius: float
) -> None:
    with pytest.raises(ValueError):
        line_contact_node_count(half_length, radius)


def _assert_swap_invariant_wrench(
    cap_a: ContactCapsule, cap_b: ContactCapsule, law: ContactParameters
) -> None:
    """``capsule_capsule(A,B)`` and ``capsule_capsule(B,A)`` agree on each body (#13508)."""
    w_ab = capsule_capsule(cap_a, cap_b, law=law)
    w_ba = capsule_capsule(cap_b, cap_a, law=law)
    np.testing.assert_allclose(w_ab.force_a_n, w_ba.force_b_n, atol=1e-9)
    np.testing.assert_allclose(w_ab.force_b_n, w_ba.force_a_n, atol=1e-9)
    np.testing.assert_allclose(w_ab.moment_a_nm, w_ba.moment_b_nm, atol=1e-9)
    np.testing.assert_allclose(w_ab.moment_b_nm, w_ba.moment_a_nm, atol=1e-9)
    np.testing.assert_allclose(w_ab.contact_point_m, w_ba.contact_point_m, atol=1e-9)
    np.testing.assert_allclose(w_ab.normal, -w_ba.normal, atol=1e-9)
    assert w_ab.penetration_m == pytest.approx(w_ba.penetration_m, abs=1e-15)
    assert w_ab.normal_force_mag_n == pytest.approx(w_ba.normal_force_mag_n, rel=1e-12)


def test_capsule_capsule_swap_invariant_crossing_unequal_length(
    law: ContactParameters,
) -> None:
    """Crossing capsules with different lengths use the same wrench either way (#13508)."""
    short = ContactCapsule(np.zeros(3), np.array([0.0, 0.0, 0.08]), radius_m=0.01)
    long = ContactCapsule(
        np.array([0.015, 0.0, 0.04]),
        np.array([0.015, 0.0, 0.12]),
        radius_m=0.012,
    )
    _assert_swap_invariant_wrench(short, long, law)
    _assert_swap_invariant_wrench(long, short, law)


def test_hand_grip_capsule_ordering_unchanged(law: ContactParameters) -> None:
    """Finger-on-grip still integrates along the finger when it is body A (#13508)."""
    w = capsule_capsule(_finger(0.0), _grip(), law=law)
    assert w.normal_force_mag_n == pytest.approx(law.stiffness_n_m * PRELOAD, rel=1e-12)


def test_opensim_capsule_sphere_count_is_the_shared_rule() -> None:
    """OpenSim's capsule sphere chain uses the shared node count (single source)."""
    pytest.importorskip(
        "src.engines.physics_engines.opensim.python.full_body_grip_contact"
    )
    from src.engines.physics_engines.opensim.python import full_body_grip_contact as fb

    assert fb.capsule_sphere_count is line_contact_node_count
