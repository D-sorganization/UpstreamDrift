"""TDD unit tests for NSX-23 trajectory conic allocation and NSX-24 allocator shims (#12566, #12567).

Covers:
1. Uniqueness on oracle fixtures under perturbed initializations (within 1e-7).
2. Numerical verification of the R6 c+/c- gross-power identity and dead-parameter
   detection when J_c v = 0 (net power invariant across null space).
3. DbC rejection of degenerate / non-strictly-convex ObjectiveSpec.
4. KKT residual < 1e-6, solve time recording, FAMILY_DEPENDENT evidence tagging,
   and exact reference recovery when tau_ref is feasible.
5. Source verification that no "SLSQP" remains in contact_force_allocator.py or
   multi_engine_torque_allocator.py.
6. Shims delegation, DeprecationWarning emission, and HARD_ZERO_TRAIL zero-side
   consistency classification (INCONSISTENT frame reported, not hidden).
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from src.shared.python.loop_nullspace.allocation import (
    AllocationResult,
    GrossPowerDecomposition,
    ObjectiveSpec,
    TrajectoryFeasibleFrame,
    ZeroSideConsistency,
    allocate_trajectory,
    check_zero_side_consistency,
    decompose_gross_power_cost,
    verify_strict_convexity,
)
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocator,
    FeasibilityStatus,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    DrakeForceAdapter,
    MultiEngineTorqueAllocator,
)

pytestmark = pytest.mark.unit


def _make_oracle_trajectory_frames(n_frames: int = 6) -> list[TrajectoryFeasibleFrame]:
    """Build a 5-DoF planar closed-loop trajectory fixture with 2D null space."""
    # Balance equation: S^T tau + J_c^T lambda_c = b
    # With S = I_5 and J_c (2x5), tau(z) = tau_p - J_c^T z, lambda_c(z) = z
    j_c = np.array(
        [
            [1.0, 0.5, -0.5, -1.0, 0.0],
            [0.0, 0.8, 0.4, -0.6, -0.8],
        ],
        dtype=np.float64,
    )
    # Null-space basis for tau: N_tau = -J_c^T (5x2), N_lambda = I_2 (2x2)
    n_tau = -j_c.T
    n_lambda = np.eye(2, dtype=np.float64)

    # Choose velocity v in ker(J_c) so J_c @ v == 0 to machine precision
    # Null space of j_c has dimension 3; pick a clean vector in ker(J_c)
    _, _, vh = np.linalg.svd(j_c, full_matrices=True)
    ker_jc = vh[2:, :].T  # (5, 3)

    frames: list[TrajectoryFeasibleFrame] = []
    for t in range(n_frames):
        phase = 0.25 * t
        tau_p = np.array(
            [
                4.0 * np.sin(phase) + 2.0,
                -3.0 * np.cos(phase) + 1.5,
                2.5 * np.sin(phase + 0.4),
                -2.0 * np.sin(phase - 0.3),
                1.8 * np.cos(phase + 0.2),
            ],
            dtype=np.float64,
        )
        coeff = np.array([1.2 + 0.1 * t, -0.8 + 0.05 * t, 0.5], dtype=np.float64)
        v_act = ker_jc @ coeff  # strictly satisfies J_c @ v_act == 0

        frames.append(
            TrajectoryFeasibleFrame(
                tau_particular=tau_p,
                null_basis_tau=n_tau,
                lambda_particular=np.zeros(2, dtype=np.float64),
                null_basis_lambda=n_lambda,
                v_actuated=v_act,
                J_c=j_c,
                z_lower=np.array([-5.0, -5.0], dtype=np.float64),
                z_upper=np.array([5.0, 5.0], dtype=np.float64),
                tau_lower=np.full(5, -25.0, dtype=np.float64),
                tau_upper=np.full(5, 25.0, dtype=np.float64),
                tau_capacity=np.array([20.0, 20.0, 15.0, 15.0, 10.0], dtype=np.float64),
                hypotheses=("H-BOX-CAPACITY", "H-CLOSURE-WELD"),
            )
        )
    return frames


def test_uniqueness_under_perturbed_initializations() -> None:
    """Different initializations / perturbations converge to the unique global optimum (< 1e-7)."""
    frames = _make_oracle_trajectory_frames(n_frames=5)
    objective = ObjectiveSpec(
        effort_weight=1.0,
        rate_weight=0.05,
        c_plus=0.2,
        c_minus=0.4,
        hypotheses=("H-EFFORT-CAPACITY", "H-RATE-SMOOTH", "H-GROSS-POWER"),
    )

    rng = np.random.default_rng(2026)
    z_init_a = rng.uniform(-2.0, 2.0, size=(5, 2))
    z_init_b = rng.uniform(-4.5, 4.5, size=(5, 2))

    res_default = allocate_trajectory(frames, objective, dt=0.01)
    res_a = allocate_trajectory(frames, objective, dt=0.01, initial_z=z_init_a)
    res_b = allocate_trajectory(frames, objective, dt=0.01, initial_z=z_init_b)

    assert res_default.converged
    assert res_a.converged
    assert res_b.converged

    np.testing.assert_allclose(res_a.tau, res_b.tau, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(res_default.tau, res_a.tau, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(res_a.z, res_b.z, atol=1e-7, rtol=1e-7)


def test_c_plus_minus_gross_power_identity_and_dead_ratio_detection() -> None:
    """Verify R6 c+/c- identity numerically and detect dead ratio parameter on J_c v = 0."""
    frames = _make_oracle_trajectory_frames(n_frames=4)
    c_plus = 1.7
    c_minus = 0.5

    # Check identity across arbitrary null-space coordinates z
    rng = np.random.default_rng(1234)
    for frame in frames:
        assert frame.v_actuated is not None
        assert frame.J_c is not None
        # Confirm velocity lies on closure manifold J_c v = 0
        assert float(np.linalg.norm(frame.J_c @ frame.v_actuated)) < 1e-12

        z_sample = rng.normal(size=2)
        tau_sample = frame.tau_particular + frame.null_basis_tau @ z_sample
        power_samples = tau_sample * frame.v_actuated

        decomp = decompose_gross_power_cost(
            power_samples,
            c_plus=c_plus,
            c_minus=c_minus,
            j_c_v=frame.J_c @ frame.v_actuated,
            null_basis_tau=frame.null_basis_tau,
            v=frame.v_actuated,
        )
        assert isinstance(decomp, GrossPowerDecomposition)
        assert decomp.identity_residual < 1e-12
        assert np.isclose(
            decomp.lhs_asymmetric_cost,
            0.5 * (c_plus + c_minus) * np.sum(np.abs(power_samples))
            + 0.5 * (c_plus - c_minus) * np.sum(power_samples),
            atol=1e-12,
        )
        assert decomp.is_dead_ratio_parameter is True
        assert decomp.is_symmetric_weights is False

    # When c_plus == c_minus, is_symmetric_weights is also True
    sym_decomp = decompose_gross_power_cost(
        np.array([3.0, -2.0, 1.5, -0.5]), c_plus=1.1, c_minus=1.1
    )
    assert sym_decomp.is_symmetric_weights is True
    assert sym_decomp.is_dead_ratio_parameter is True
    assert sym_decomp.identity_residual < 1e-12

    # Proposition 2 corollary: on J_c v = 0, changing (c_plus, c_minus) while keeping
    # (c_plus + c_minus) constant leaves the optimal trajectory argmin invariant!
    obj_1 = ObjectiveSpec(effort_weight=1.0, rate_weight=0.02, c_plus=1.5, c_minus=0.5)
    obj_2 = ObjectiveSpec(effort_weight=1.0, rate_weight=0.02, c_plus=1.0, c_minus=1.0)
    res_1 = allocate_trajectory(frames, obj_1, dt=0.01)
    res_2 = allocate_trajectory(frames, obj_2, dt=0.01)
    assert res_1.gross_power_info.is_dead_ratio_parameter is True
    assert res_2.gross_power_info.is_dead_ratio_parameter is True
    np.testing.assert_allclose(res_1.tau, res_2.tau, atol=1e-6, rtol=1e-6)


def test_dbc_rejects_non_strictly_convex_objective() -> None:
    """verify_strict_convexity and allocate_trajectory reject degenerate objectives."""
    frames = _make_oracle_trajectory_frames(n_frames=3)

    # Zero effort and zero reference weight -> missing positive quadratic curvature
    degenerate_l1_only = ObjectiveSpec(
        effort_weight=0.0,
        rate_weight=0.0,
        c_plus=1.0,
        c_minus=1.0,
        reference_weight=0.0,
    )
    with pytest.raises(
        ValueError, match="quadratic curvature|effort_weight|reference_weight"
    ):
        verify_strict_convexity(degenerate_l1_only)

    with pytest.raises(
        ValueError, match="quadratic curvature|effort_weight|reference_weight"
    ):
        allocate_trajectory(frames, degenerate_l1_only)

    # Rate weight alone without per-frame quadratic curvature has zero curvature on constant shifts
    rate_only = ObjectiveSpec(
        effort_weight=0.0,
        rate_weight=1.0,
        c_plus=0.5,
        c_minus=0.5,
        reference_weight=0.0,
    )
    with pytest.raises(
        ValueError, match="quadratic curvature|effort_weight|reference_weight"
    ):
        verify_strict_convexity(rate_only)

    # Rank-deficient null-space basis triggers reduced Hessian eigenvalue check
    rank_def_frame = TrajectoryFeasibleFrame(
        tau_particular=np.zeros(4),
        null_basis_tau=np.zeros((4, 1)),  # zero column -> lambda_min == 0
    )
    valid_obj = ObjectiveSpec(effort_weight=1.0)
    with pytest.raises(ValueError, match="Reduced Hessian|strictly convex"):
        verify_strict_convexity(valid_obj, [rank_def_frame])


def test_kkt_residuals_evidence_metadata_and_reference_recovery() -> None:
    """Verify KKT residual < 1e-6, FAMILY_DEPENDENT tagging, and exact feasible reference recovery."""
    frames = _make_oracle_trajectory_frames(n_frames=4)

    # Construct a known strictly feasible reference trajectory inside the null space
    z_true = np.array(
        [
            [0.4, -0.3],
            [0.3, -0.2],
            [0.2, -0.1],
            [0.1, 0.0],
        ],
        dtype=np.float64,
    )
    tau_ref = np.stack(
        [
            frames[t].tau_particular + frames[t].null_basis_tau @ z_true[t]
            for t in range(4)
        ],
        axis=0,
    )

    ref_obj = ObjectiveSpec(
        effort_weight=0.0,
        rate_weight=0.0,
        c_plus=0.0,
        c_minus=0.0,
        reference_weight=5.0,
        hypotheses=("H-REF-TRACKING",),
    )
    res = allocate_trajectory(frames, ref_obj, reference=tau_ref, dt=0.01)

    assert isinstance(res, AllocationResult)
    assert res.converged
    assert res.kkt_residual_norm < 1e-6
    assert res.solve_time_s >= 0.0
    assert res.evidence_class == "FAMILY_DEPENDENT"
    assert res.allocation_dependence == "FAMILY_DEPENDENT"
    assert "H-REF-TRACKING" in res.hypotheses
    assert res.quantity_record.allocation_dependence == "FAMILY_DEPENDENT"
    assert len(res.quantity_record.hypotheses) >= 1

    # Reference is feasible, so optimum must recover tau_ref and z_true
    np.testing.assert_allclose(res.tau, tau_ref, atol=1e-6)
    np.testing.assert_allclose(res.z, z_true, atol=1e-6)


def test_no_slsqp_in_legacy_allocator_modules() -> None:
    """Grep gate: neither contact_force_allocator.py nor multi_engine_torque_allocator.py uses SLSQP."""
    repo_root = Path(__file__).resolve().parents[3]
    cfa_path = (
        repo_root
        / "src"
        / "shared"
        / "python"
        / "motion_matching"
        / "contact_force_allocator.py"
    )
    meta_path = (
        repo_root
        / "src"
        / "shared"
        / "python"
        / "motion_matching"
        / "multi_engine_torque_allocator.py"
    )

    cfa_source = cfa_path.read_text(encoding="utf-8")
    meta_source = meta_path.read_text(encoding="utf-8")

    assert "SLSQP" not in cfa_source, (
        "contact_force_allocator.py must not contain SLSQP"
    )
    assert "SLSQP" not in meta_source, (
        "multi_engine_torque_allocator.py must not contain SLSQP"
    )


def test_legacy_shims_deprecation_warning_and_hard_zero_trail_inconsistency() -> None:
    """Verify DeprecationWarning support and HARD_ZERO_TRAIL zero-side consistency reporting."""
    nv = 12
    actuated = list(range(6, nv))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        allocator = ContactForceAllocator(
            nv=nv,
            actuated_indices=actuated,
            n_contact_spheres=2,
            warn_deprecated=True,
        )
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "loop_nullspace" in str(w.message)
        for w in caught
    )

    with warnings.catch_warnings(record=True) as caught_meta:
        warnings.simplefilter("always", DeprecationWarning)
        adapter = DrakeForceAdapter(nv=44, n_spheres=6)
        _ = MultiEngineTorqueAllocator(adapter=adapter, warn_deprecated=True)
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "loop_nullspace" in str(w.message)
        for w in caught_meta
    )

    # Zero-side consistency check: when trail joints have zero grip coupling and non-zero RNEA demand,
    # setting trail joints to zero is algebraically INCONSISTENT and must be reported, not hidden.
    j_ground = np.zeros((6, nv))
    j_ground[:3, :3] = np.eye(3)
    j_ground[3:, :3] = np.eye(3)
    j_grip = np.zeros((6, nv))  # zero grip coupling -> cannot offload trail joint 6

    tau_rnea = np.zeros(nv)
    tau_rnea[2] = 500.0
    tau_rnea[6] = 45.0  # Uncoupled trail joint 6 requires 45 N*m

    zero_check = check_zero_side_consistency(
        tau_rnea=tau_rnea,
        actuated_indices=actuated,
        zero_joint_indices=[6, 7],
        j_ground=j_ground,
        j_grip=j_grip,
    )
    assert zero_check.status == ZeroSideConsistency.INCONSISTENT
    assert zero_check.residual_norm > 1.0
    assert zero_check.residual_scaled > 0.05

    alloc_inconsistent = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.HARD_ZERO_TRAIL,
        trail_arm_indices=[6, 7],
    )
    assert alloc_inconsistent.is_physically_feasible is False
    assert (
        alloc_inconsistent.feasibility_status
        == FeasibilityStatus.INFEASIBLE_HARD_ZERO_TRAIL
    )
    assert alloc_inconsistent.success is False
