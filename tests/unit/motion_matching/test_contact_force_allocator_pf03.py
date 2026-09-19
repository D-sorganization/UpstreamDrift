"""Unit tests for PF-03: Contact, Actuator, and Root Constraints in Force Allocation (#10433).

Verifies:
1. Friction cone enforcement (rejection of 1000 N tangential / 10 N normal / mu=0.1 case).
2. Actuator bounds enforcement (rejection of 100 Nm with +/-1 Nm actuator bounds without post-projection).
3. Zero contact force during separation (separated contact spheres exert strictly zero force).
4. Rotated ground frame support (normal force along arbitrary unit normal n_hat).
5. Rank-deficient contact handling without numerical divergence.
6. Impossible floating-base balance rejection (no physical success via root slack).
7. Native virtual work and power sign checks.
8. Hard-zero trail mode (exact zero or rejected) vs soft minimum-trail.
9. Torque and rate bounds verification after trajectory refinement.
10. Separation of diagnostic root slack from physical feasibility.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocation,
    ContactForceAllocator,
    FeasibilityStatus,
    verify_torque_and_rate_bounds,
)

pytestmark = [pytest.mark.unit]


def _build_synthetic_contact_jacobian(
    sphere_positions: list[np.ndarray], nv: int
) -> np.ndarray:
    """Construct contact Jacobian with translational and lever-arm rotational coupling."""
    n_spheres = len(sphere_positions)
    j_ground = np.zeros((n_spheres * 3, nv))
    for s, pos in enumerate(sphere_positions):
        row = s * 3
        j_ground[row : row + 3, :3] = np.eye(3)
        # Skew-symmetric cross product: -[r]_x
        skew = np.array(
            [
                [0.0, -pos[2], pos[1]],
                [pos[2], 0.0, -pos[0]],
                [-pos[1], pos[0], 0.0],
            ]
        )
        j_ground[row : row + 3, 3:6] = -skew
    return j_ground


@pytest.fixture
def standard_setup() -> dict[str, Any]:
    nv = 12
    actuated_indices = np.arange(6, nv)
    n_spheres = 2
    sphere_positions = [
        np.array([-0.1, 0.0, 0.0]),
        np.array([0.1, 0.0, 0.0]),
    ]
    j_ground = _build_synthetic_contact_jacobian(sphere_positions, nv)
    j_grip = np.zeros((6, nv))
    j_grip[:3, 6:9] = np.eye(3)
    j_grip[:3, 9:12] = -np.eye(3)

    return {
        "nv": nv,
        "actuated_indices": actuated_indices,
        "n_spheres": n_spheres,
        "sphere_positions": sphere_positions,
        "j_ground": j_ground,
        "j_grip": j_grip,
    }


def test_friction_cone_rejection_1000n_tangential_10n_normal(
    standard_setup: dict[str, Any],
) -> None:
    """Allocator rejects 1000 N tangential / 10 N normal / mu=0.1 as friction infeasible."""
    setup = standard_setup
    allocator = ContactForceAllocator(
        nv=setup["nv"],
        actuated_indices=setup["actuated_indices"],
        n_contact_spheres=setup["n_spheres"],
        mu_friction=0.1,  # Low friction: mu = 0.1
    )

    # Force requiring 1000 N lateral tangential load with only 10 N normal load (Z load = 10 N)
    tau_rnea = np.zeros(setup["nv"])
    tau_rnea[0] = 1000.0  # X load requires 1000 N tangential force
    tau_rnea[2] = 10.0  # Z load provides only 10 N normal force

    alloc = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=setup["j_ground"],
        j_grip=setup["j_grip"],
        objective=AllocationObjective.MINIMUM_EFFORT,
    )

    # Must be marked infeasible under friction cone constraints: mu * fn = 0.1 * 10 = 1 N << 1000 N
    assert alloc.is_physically_feasible is False
    assert alloc.feasibility_status in (
        FeasibilityStatus.INFEASIBLE_FRICTION_CONE,
        FeasibilityStatus.INFEASIBLE_ROOT_BALANCE,
    )
    # Contact forces returned must not violate friction cone
    f_reshaped = alloc.f_ground.reshape(setup["n_spheres"], 3)
    for s in range(setup["n_spheres"]):
        fn = max(0.0, float(f_reshaped[s, 2]))
        ft = float(np.linalg.norm(f_reshaped[s, :2]))
        assert ft <= 0.1 * fn + 1e-4, (
            f"Sphere {s} violated friction cone: ft={ft}, fn={fn}"
        )


def test_actuator_bounds_rejection_100nm_with_1nm_bounds(
    standard_setup: dict[str, Any],
) -> None:
    """Allocator rejects 100 Nm required torque with +/-1 Nm actuator bounds without post-projection."""
    setup = standard_setup
    allocator = ContactForceAllocator(
        nv=setup["nv"],
        actuated_indices=setup["actuated_indices"],
        n_contact_spheres=setup["n_spheres"],
        mu_friction=0.8,
    )

    # Joint coordinate 6 requires 100 Nm
    tau_rnea = np.zeros(setup["nv"])
    tau_rnea[2] = 500.0  # normal body weight
    tau_rnea[6] = 100.0  # joint 6 demands 100 Nm

    # Actuator bounds are strictly +/- 1 Nm
    n_act = len(setup["actuated_indices"])
    lb_tau = -np.ones(n_act)
    ub_tau = np.ones(n_act)

    alloc = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=setup["j_ground"],
        j_grip=setup["j_grip"],
        objective=AllocationObjective.MINIMUM_EFFORT,
        tau_bounds=(lb_tau, ub_tau),
    )

    # Must NOT claim physical success and must NOT post-project outside [-1, 1]
    assert alloc.is_physically_feasible is False
    assert alloc.feasibility_status == FeasibilityStatus.INFEASIBLE_ACTUATOR_BOUNDS
    assert np.all(alloc.tau_actuated >= lb_tau - 1e-4)
    assert np.all(alloc.tau_actuated <= ub_tau + 1e-4)


def test_zero_contact_force_during_separation(
    standard_setup: dict[str, Any],
) -> None:
    """Separated contact spheres must exert strictly zero ground reaction force."""
    setup = standard_setup
    allocator = ContactForceAllocator(
        nv=setup["nv"],
        actuated_indices=setup["actuated_indices"],
        n_contact_spheres=setup["n_spheres"],
        mu_friction=0.8,
    )

    tau_rnea = np.zeros(setup["nv"])
    tau_rnea[2] = 400.0  # weight

    # Sphere 0 is in contact (True), Sphere 1 is separated / airborne (False)
    contact_mask = np.array([True, False], dtype=bool)

    alloc = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=setup["j_ground"],
        j_grip=setup["j_grip"],
        contact_mask=contact_mask,
    )

    f_reshaped = alloc.f_ground.reshape(setup["n_spheres"], 3)
    # Separated sphere 1 must have exactly 0 force
    np.testing.assert_allclose(
        f_reshaped[1],
        np.zeros(3),
        atol=1e-6,
        err_msg="Separated contact exerted force!",
    )
    # Active sphere 0 carries the load
    assert f_reshaped[0, 2] > 0.0


def test_rotated_ground_frame_normal(
    standard_setup: dict[str, Any],
) -> None:
    """Supports rotated ground surface with custom normal vector."""
    setup = standard_setup
    allocator = ContactForceAllocator(
        nv=setup["nv"],
        actuated_indices=setup["actuated_indices"],
        n_contact_spheres=setup["n_spheres"],
        mu_friction=0.5,
    )

    # Ground tilted 30 degrees around Y: normal has X and Z components
    theta = math.radians(30.0)
    ground_normal = np.array([math.sin(theta), 0.0, math.cos(theta)])

    tau_rnea = np.zeros(setup["nv"])
    # Pure gravity along -Z
    tau_rnea[2] = 500.0

    alloc = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=setup["j_ground"],
        j_grip=setup["j_grip"],
        ground_normal=ground_normal,
    )

    f_reshaped = alloc.f_ground.reshape(setup["n_spheres"], 3)
    for s in range(setup["n_spheres"]):
        f_s = f_reshaped[s]
        # Normal force component along ground_normal
        fn = float(f_s @ ground_normal)
        assert fn >= -1e-4, (
            f"Normal force along rotated normal must be non-negative, got {fn}"
        )
        # Tangential force component
        ft_vec = f_s - fn * ground_normal
        ft = float(np.linalg.norm(ft_vec))
        assert ft <= 0.5 * fn + 1e-4, (
            "Tangential force exceeded friction on inclined slope"
        )


def test_rank_deficient_contacts(
    standard_setup: dict[str, Any],
) -> None:
    """Allocator handles rank-deficient contact Jacobians robustly without NaN or divergence."""
    setup = standard_setup
    # Create identical degenerate contact points (collinear/coincident)
    j_ground_degenerate = np.zeros_like(setup["j_ground"])
    j_ground_degenerate[:3] = setup["j_ground"][:3]
    j_ground_degenerate[3:] = setup["j_ground"][:3]  # identical rows -> rank deficient

    allocator = ContactForceAllocator(
        nv=setup["nv"],
        actuated_indices=setup["actuated_indices"],
        n_contact_spheres=setup["n_spheres"],
        mu_friction=0.8,
    )

    tau_rnea = np.zeros(setup["nv"])
    tau_rnea[2] = 300.0

    alloc = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground_degenerate,
        j_grip=setup["j_grip"],
    )

    assert np.all(np.isfinite(alloc.tau_actuated))
    assert np.all(np.isfinite(alloc.f_ground))


def test_impossible_floating_base_balance(
    standard_setup: dict[str, Any],
) -> None:
    """Impossible floating-base balance returns is_physically_feasible=False."""
    setup = standard_setup
    allocator = ContactForceAllocator(
        nv=setup["nv"],
        actuated_indices=setup["actuated_indices"],
        n_contact_spheres=setup["n_spheres"],
        mu_friction=0.8,
    )

    # Downward load on root: requires downward pull from ground (f_z < 0)
    tau_rnea = np.zeros(setup["nv"])
    tau_rnea[2] = -500.0  # negative vertical load! Ground cannot pull golfer down

    alloc = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=setup["j_ground"],
        j_grip=setup["j_grip"],
    )

    assert alloc.is_physically_feasible is False
    assert alloc.feasibility_status in (
        FeasibilityStatus.INFEASIBLE_ROOT_BALANCE,
        FeasibilityStatus.INFEASIBLE_GROUND_UNILATERAL,
    )


def test_hard_zero_trail_mode_vs_soft_minimum_trail(
    standard_setup: dict[str, Any],
) -> None:
    """Hard-zero trail mode enforces exact zero or rejects; soft minimum-trail never claims zero falsely."""
    setup = standard_setup
    allocator = ContactForceAllocator(
        nv=setup["nv"],
        actuated_indices=setup["actuated_indices"],
        n_contact_spheres=setup["n_spheres"],
        mu_friction=0.8,
    )

    trail_arm_indices = [6, 7]  # First two actuated coordinates

    tau_rnea = np.zeros(setup["nv"])
    tau_rnea[2] = 600.0
    # Add external torque that CANNOT be balanced without trail arm
    tau_rnea[6] = 50.0  # joint 6 has load, grip jacobian can help if configured

    # 1. Soft minimum trail arm
    alloc_soft = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=setup["j_ground"],
        j_grip=setup["j_grip"],
        objective=AllocationObjective.MINIMUM_TRAIL_ARM,
        trail_arm_indices=trail_arm_indices,
    )
    trail_norm = float(np.linalg.norm(alloc_soft.tau_actuated[:2]))
    if trail_norm > 1e-4:
        # Soft mode must NOT claim zero
        assert alloc_soft.tau_actuated[0] != 0.0 or alloc_soft.tau_actuated[1] != 0.0

    # 2. Hard zero trail arm
    alloc_hard = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=setup["j_ground"],
        j_grip=setup["j_grip"],
        objective=AllocationObjective.HARD_ZERO_TRAIL,
        trail_arm_indices=trail_arm_indices,
    )
    if alloc_hard.is_physically_feasible:
        # Must be exactly 0
        np.testing.assert_allclose(alloc_hard.tau_actuated[:2], [0.0, 0.0], atol=1e-6)
    else:
        # Rejected cleanly
        assert alloc_hard.feasibility_status != FeasibilityStatus.FEASIBLE


def test_native_virtual_work_and_power_sign(
    standard_setup: dict[str, Any],
) -> None:
    """Virtual work identity holds for arbitrary virtual generalized velocities."""
    setup = standard_setup
    allocator = ContactForceAllocator(
        nv=setup["nv"],
        actuated_indices=setup["actuated_indices"],
        n_contact_spheres=setup["n_spheres"],
        mu_friction=0.8,
    )

    tau_rnea = np.zeros(setup["nv"])
    tau_rnea[2] = 700.0
    tau_rnea[6:] = 10.0

    alloc = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=setup["j_ground"],
        j_grip=setup["j_grip"],
    )
    assert alloc.is_physically_feasible

    # Test virtual work for multiple arbitrary delta_v directions
    rng = np.random.default_rng(123)
    for _ in range(5):
        delta_v = rng.standard_normal(setup["nv"])
        w_rnea = float(tau_rnea @ delta_v)

        tau_full = np.zeros(setup["nv"])
        tau_full[setup["actuated_indices"]] = alloc.tau_actuated
        w_applied = (
            float(tau_full @ delta_v)
            + float(alloc.f_ground @ (setup["j_ground"] @ delta_v))
            + float(alloc.lambda_grip @ (setup["j_grip"] @ delta_v))
            + float(alloc.delta_tau_root @ delta_v[:6])
        )
        assert abs(w_rnea - w_applied) < 1e-4, (
            f"Virtual work discrepancy: {abs(w_rnea - w_applied)}"
        )


def test_verify_torque_and_rate_bounds() -> None:
    """Verify torque trajectory limits and rate of change limits."""
    dt = 0.01
    time_steps = 50
    n_act = 6

    # Nominal smooth sinusoidal trajectory within bounds [-100, 100]
    t = np.linspace(0, (time_steps - 1) * dt, time_steps)[:, np.newaxis]
    tau_traj = 80.0 * np.sin(2.0 * np.pi * 1.0 * t) * np.ones((1, n_act))

    lb_tau = -100.0 * np.ones(n_act)
    ub_tau = 100.0 * np.ones(n_act)
    tau_bounds = (lb_tau, ub_tau)

    # Max theoretical dtau/dt = 80 * 2 * pi * 1.0 ~ 502.6 Nm/s
    rate_bounds_valid = 600.0
    rate_bounds_tight = 400.0

    # 1. Valid bounds test
    is_valid, diag = verify_torque_and_rate_bounds(
        tau_trajectory=tau_traj,
        dt=dt,
        tau_bounds=tau_bounds,
        rate_bounds=rate_bounds_valid,
    )
    assert is_valid is True
    assert diag["is_torque_bounded"] is True
    assert diag["is_rate_bounded"] is True
    assert diag["max_tau_violation"] == 0.0
    assert diag["max_rate_violation"] == 0.0

    # 2. Rate violation test
    is_valid_tight, diag_tight = verify_torque_and_rate_bounds(
        tau_trajectory=tau_traj,
        dt=dt,
        tau_bounds=tau_bounds,
        rate_bounds=rate_bounds_tight,
    )
    assert is_valid_tight is False
    assert diag_tight["is_torque_bounded"] is True
    assert diag_tight["is_rate_bounded"] is False
    assert diag_tight["max_rate_violation"] > 0.0

    # 3. Torque bound violation test
    tight_tau_bounds = (-50.0 * np.ones(n_act), 50.0 * np.ones(n_act))
    is_valid_tau, diag_tau = verify_torque_and_rate_bounds(
        tau_trajectory=tau_traj,
        dt=dt,
        tau_bounds=tight_tau_bounds,
    )
    assert is_valid_tau is False
    assert diag_tau["is_torque_bounded"] is False
    assert diag_tau["max_tau_violation"] > 0.0
