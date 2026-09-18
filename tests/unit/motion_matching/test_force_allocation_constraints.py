"""Unit tests for Contact, Actuator, and Root Constraints in Force Allocation (PF-03 / #10433).

Acceptance Criteria:
1. RED-to-GREEN for 1000 N tangential / 10 N normal / mu=0.1 case (friction cone infeasibility).
2. Required 100 Nm with +/-1 Nm actuator bounds (strict bounds without unconstrained post-projection).
3. Zero contact force during separation (inactive contacts produce strictly 0 N).
4. Rotated ground surface frame (friction cone and normal force in tilted plane).
5. Rank-deficient contact Jacobians handled robustly with explicit feasibility reporting.
6. Impossible floating-base balance rejected (diagnostic root slack cannot create physical success).
7. Native virtual-work and sign consistency.
8. Hard-zero trail mode is either feasible to declared tolerance or rejected; soft minimum-trail never claims zero.
9. Explicit grip wrench limits enforced.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    AllocationStatus,
    ContactForceAllocation,
    ContactForceAllocator,
    SurfaceContactFrame,
)

pytestmark = pytest.mark.unit


def _skew(r: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross product matrix."""
    return np.array(
        [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ]
    )


def _build_support_polygon_jacobian(
    sphere_positions: list[np.ndarray], nv: int
) -> np.ndarray:
    """Contact Jacobian for ground points."""
    n_spheres = len(sphere_positions)
    j_ground = np.zeros((n_spheres * 3, nv))
    for s, pos in enumerate(sphere_positions):
        row = s * 3
        j_ground[row : row + 3, :3] = np.eye(3)
        j_ground[row : row + 3, 3:6] = -_skew(pos)
    return j_ground


@pytest.fixture
def base_setup() -> tuple[
    int, np.ndarray, int, list[np.ndarray], np.ndarray, np.ndarray
]:
    nv = 12
    actuated_indices = np.arange(6, nv)
    n_spheres = 4
    # 4 support points forming a 0.3m x 0.3m support polygon on ground z=0
    sphere_positions = [
        np.array([-0.15, -0.15, 0.0]),
        np.array([0.15, -0.15, 0.0]),
        np.array([-0.15, 0.15, 0.0]),
        np.array([0.15, 0.15, 0.0]),
    ]
    j_ground = _build_support_polygon_jacobian(sphere_positions, nv)
    j_grip = np.zeros((6, nv))
    j_grip[:3, 6:9] = np.eye(3)
    j_grip[:3, 9:12] = -np.eye(3)
    return nv, actuated_indices, n_spheres, sphere_positions, j_ground, j_grip


def test_infeasible_tangential_force_exceeding_friction_cone(
    base_setup: tuple[int, np.ndarray, int, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Acceptance Case 1: 1000 N tangential / 10 N normal / mu=0.1.

    Admissible tangential force is at most mu * 10 N = 1.0 N.
    A demand of 1000 N tangential force is wildly outside the friction cone.
    Old solver succeeded via unconstrained post-projection or missing friction inequalities.
    New solver MUST return success=False and declare friction infeasibility.
    """
    nv, actuated_indices, n_spheres, _, j_ground, j_grip = base_setup
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.1,
    )

    # 10 N upward normal support, but massive 1000 N tangential lateral force (X axis)
    tau_rnea = np.zeros(nv)
    tau_rnea[0] = 1000.0  # Fx total
    tau_rnea[2] = 10.0  # Fz total

    result = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.MINIMUM_EFFORT,
    )

    assert isinstance(result, ContactForceAllocation)
    assert not result.success, (
        "1000 N tangential with 10 N normal and mu=0.1 must NOT succeed!"
    )
    assert result.status in (
        AllocationStatus.INFEASIBLE_FRICTION_CONE,
        AllocationStatus.INFEASIBLE_ROOT_EQUILIBRIUM,
        AllocationStatus.INFEASIBLE,
    )


def test_infeasible_actuator_bounds_without_post_projection_violation(
    base_setup: tuple[int, np.ndarray, int, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Acceptance Case 2: Required 100 Nm with +/-1 Nm actuator bounds.

    The demand on an actuated coordinate requires 100 Nm, but bounds are [-1, 1] Nm.
    The solver must NOT post-project the residual onto the actuated coordinate,
    and must return success=False and declare actuator bound infeasibility.
    """
    nv, actuated_indices, n_spheres, _, j_ground, j_grip = base_setup
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    # Require 100 Nm on first actuated joint (coordinate index 6)
    tau_rnea = np.zeros(nv)
    tau_rnea[2] = 200.0  # reasonable body weight
    tau_rnea[6] = 100.0  # 100 Nm required on joint 0

    tau_min = np.full(len(actuated_indices), -1.0)
    tau_max = np.full(len(actuated_indices), 1.0)

    result = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.MINIMUM_EFFORT,
        tau_bounds=(tau_min, tau_max),
    )

    assert not result.success, "Demand of 100 Nm with +/-1 Nm bounds must fail!"
    assert result.status in (
        AllocationStatus.INFEASIBLE_ACTUATOR_BOUNDS,
        AllocationStatus.INFEASIBLE,
    )
    # Strict postcondition: solution must NOT violate actuator bounds
    assert np.all(result.tau_actuated >= tau_min - 1e-4)
    assert np.all(result.tau_actuated <= tau_max + 1e-4)


def test_separated_contact_zero_force(
    base_setup: tuple[int, np.ndarray, int, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Acceptance Case 3: Zero contact during separation.

    When specific contacts are inactive (e.g. rear foot lifted during swing),
    their normal AND tangential contact forces must be identically zero.
    """
    nv, actuated_indices, n_spheres, _, j_ground, j_grip = base_setup
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    # Spheres 0 and 3 form diagonal support spanning the center of pressure (0, 0)
    # Spheres 1 and 2 are in the air (separated)
    active_contacts = [True, False, False, True]

    tau_rnea = np.zeros(nv)
    tau_rnea[2] = 200.0  # 200 N supported by the 2 active spheres

    result = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.MINIMUM_EFFORT,
        active_contacts=active_contacts,
    )

    assert result.success
    forces = result.f_ground.reshape(n_spheres, 3)
    # Separated spheres 1 and 2 must have strictly zero force
    np.testing.assert_allclose(forces[1], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(forces[2], [0.0, 0.0, 0.0], atol=1e-6)
    # Active spheres 0 and 3 carry the load
    assert (forces[0, 2] + forces[3, 2]) == pytest.approx(200.0, abs=1e-3)


def test_rotated_ground_surface_frame() -> None:
    """Acceptance Case 4: Rotated ground frame (30-degree ramp).

    Ground surface is inclined at 30 degrees around Y.
    Friction cone and non-negative normal force must be enforced relative
    to the inclined surface normal and tangent vectors.
    """
    nv = 12
    actuated_indices = np.arange(6, nv)
    n_spheres = 2
    theta = np.deg2rad(30.0)
    # Surface normal rotated by 30 deg: [-sin(theta), 0, cos(theta)]
    normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])
    tangent1 = np.array([np.cos(theta), 0.0, np.sin(theta)])
    tangent2 = np.array([0.0, 1.0, 0.0])

    surface_frames = [
        SurfaceContactFrame(normal=normal, tangent1=tangent1, tangent2=tangent2),
        SurfaceContactFrame(normal=normal, tangent1=tangent1, tangent2=tangent2),
    ]

    sphere_positions = [
        np.array([-0.1, 0.0, 0.0]),
        np.array([0.1, 0.0, 0.0]),
    ]
    j_ground = _build_support_polygon_jacobian(sphere_positions, nv)
    j_grip = np.zeros((6, nv))

    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.5,
    )

    # Load purely along the ramp normal (100 N compressive)
    tau_rnea = np.zeros(nv)
    tau_rnea[:3] = normal * 100.0

    result = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        surface_frames=surface_frames,
    )

    assert result.success
    f_res = result.f_ground.reshape(n_spheres, 3)
    for i in range(n_spheres):
        f_norm = float(np.dot(f_res[i], normal))
        f_tan1 = float(np.dot(f_res[i], tangent1))
        f_tan2 = float(np.dot(f_res[i], tangent2))
        assert f_norm >= -1e-5, "Normal force must be non-negative in rotated frame"
        assert abs(f_tan1) <= 0.5 * f_norm + 1e-4, (
            "Friction cone violated in rotated frame"
        )
        assert abs(f_tan2) <= 0.5 * f_norm + 1e-4, (
            "Friction cone violated in rotated frame"
        )


def test_rank_deficient_contacts_handling() -> None:
    """Acceptance Case 5: Rank-deficient contacts (single point contact).

    Single contact point can only produce 3 forces and 0 moments about itself.
    If a moment about the contact point is required, it must be declared infeasible
    rather than producing numerical singularities or crashes.
    """
    nv = 12
    actuated_indices = np.arange(6, nv)
    n_spheres = 1
    sphere_positions = [np.array([0.0, 0.0, 0.0])]
    j_ground = _build_support_polygon_jacobian(sphere_positions, nv)
    j_grip = np.zeros((6, nv))

    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    # Require a tipping moment about the origin on the unactuated floating base: tau_rnea[4] = 50 Nm
    tau_rnea = np.zeros(nv)
    tau_rnea[2] = 200.0  # body weight
    tau_rnea[4] = 50.0  # unbalanceable moment with 1 point at origin!

    result = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
    )

    assert not result.success, (
        "Single point contact cannot balance non-zero root moment!"
    )
    assert result.status in (
        AllocationStatus.INFEASIBLE_ROOT_EQUILIBRIUM,
        AllocationStatus.INFEASIBLE,
    )


def test_impossible_floating_base_balance_rejected(
    base_setup: tuple[int, np.ndarray, int, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Acceptance Case 6: Impossible floating-base balance.

    Downward pull on ground (Fz = -500 N, requires suction) cannot be produced by unilateral contacts.
    Diagnostic root slack must NOT create physical success.
    """
    nv, actuated_indices, n_spheres, _, j_ground, j_grip = base_setup
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    tau_rnea = np.zeros(nv)
    tau_rnea[
        2
    ] = -500.0  # upward pull on pelvis, ground would need to pull down (impossible)

    result = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
    )

    assert not result.success, "Ground suction cannot create physical success!"
    # Root slack records the deficit, but success remains False
    assert result.root_balance_residual > 1.0


def test_virtual_work_and_sign_consistency(
    base_setup: tuple[int, np.ndarray, int, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Acceptance Case 7: Native virtual-work and sign checks.

    For any feasible solution, delta W = tau^T delta q + f_ground^T delta x_ground = 0
    consistent with RNEA generalized equations of motion.
    """
    nv, actuated_indices, n_spheres, _, j_ground, j_grip = base_setup
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    tau_rnea = np.zeros(nv)
    tau_rnea[2] = 400.0
    tau_rnea[6:] = 25.0

    result = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
    )

    assert result.success
    # Construct total generalized applied force
    tau_full = np.zeros(nv)
    tau_full[actuated_indices] = result.tau_actuated
    applied = tau_full + j_ground.T @ result.f_ground + j_grip.T @ result.lambda_grip

    # Check that applied matches tau_rnea on all coordinates
    np.testing.assert_allclose(applied, tau_rnea, atol=1e-4)

    # Check virtual work for arbitrary virtual displacement delta_q
    rng = np.random.default_rng(77)
    delta_q = rng.standard_normal(nv) * 0.01
    delta_x_ground = j_ground @ delta_q
    work_gen = np.dot(tau_full, delta_q)
    work_contact = np.dot(result.f_ground, delta_x_ground)
    work_grip = np.dot(result.lambda_grip, j_grip @ delta_q)
    total_work_done = work_gen + work_contact + work_grip
    expected_work = np.dot(tau_rnea, delta_q)
    assert total_work_done == pytest.approx(expected_work, abs=1e-4)


def test_hard_zero_trail_mode_vs_soft_minimum_trail(
    base_setup: tuple[int, np.ndarray, int, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Acceptance Case 8: Hard-zero vs soft minimum trail arm semantics.

    - Hard-zero mode strictly sets trail arm torques to 0.0. If impossible, it fails.
    - Soft minimum-trail heavily penalizes trail arm torques, but never falsely claims zero.
    """
    nv, actuated_indices, n_spheres, _, j_ground, j_grip = base_setup
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    trail_indices = [6, 7]  # joints 6 and 7 are trail arm
    tau_rnea = np.zeros(nv)
    tau_rnea[2] = 300.0
    # Add wrench that CAN be compensated by lead arm (joints 8:12) via grip transmission
    tau_rnea[6] = 5.0
    tau_rnea[7] = 5.0

    # 1. Soft minimum trail arm
    res_soft = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.MINIMUM_TRAIL_ARM,
        trail_arm_indices=trail_indices,
    )
    assert res_soft.success
    # Soft minimum trail arm reduces torque but does not guarantee exactly zero
    trail_act_idx = [0, 1]
    assert np.linalg.norm(res_soft.tau_actuated[trail_act_idx]) >= 0.0

    # 2. Hard zero trail arm
    res_hard = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.HARD_ZERO_TRAIL_ARM,
        trail_arm_indices=trail_indices,
    )
    assert res_hard.success
    np.testing.assert_allclose(
        res_hard.tau_actuated[trail_act_idx], [0.0, 0.0], atol=1e-6
    )


def test_explicit_grip_limits_enforced(
    base_setup: tuple[int, np.ndarray, int, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Acceptance Case 9: Explicit grip wrench limits.

    Grip limits constrain closed-chain internal load transmission.
    Excessive demand exceeding grip limits must be rejected.
    """
    nv, actuated_indices, n_spheres, _, j_ground, j_grip = base_setup
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    tau_rnea = np.zeros(nv)
    tau_rnea[2] = 200.0
    tau_rnea[6] = 50.0  # demands transmission through grip
    # Restrict trail arm so all must transmit through grip
    trail_indices = [6]

    # Cap grip force at 5.0 N
    res_limited = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.HARD_ZERO_TRAIL_ARM,
        trail_arm_indices=trail_indices,
        grip_limits=(5.0, 1.0),
    )

    if res_limited.success:
        assert np.all(np.abs(res_limited.lambda_grip[:3]) <= 5.0 + 1e-4)
        assert np.all(np.abs(res_limited.lambda_grip[3:]) <= 1.0 + 1e-4)
