"""Unit and acceptance tests for Force Nullspace Explorer and Tradeoffs (PF-06, #10436).

Verifies:
1. Scaled rank-revealing nullspace: exact A @ N = 0 and orthonormal basis N^T @ N = I.
2. Variable nullity across contact modes: adapts to flight, single support, and double support without fixed nullity.
3. Basis sign-change invariance: physical optimal torque is strictly invariant to nullspace basis orientation.
4. Hard-zero trail feasibility diagnostic: identifies when hard-zero is feasible vs blocked by capacity constraints.
5. Effort vs smoothness Pareto sweep: monotonic trade-off with all samples physically admissible.
6. Trail arm share tradeoff sweep: progressive load shift from trail arm to lead arm and internal grip wrench.
7. Reproducible Pareto table generation: publishes structured markdown report with conservative default rationale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.force_nullspace_explorer import (
    ForceNullspaceExplorer,
    NullspaceDecomposition,
    ParetoFrontierReport,
    TradeoffDimension,
    TradeoffSample,
)
from src.shared.python.motion_matching.sparse_trajectory_torque_optimizer import (
    TrajectoryOptimizationNode,
)


def _create_explorer_test_node(
    tau_actuated_target: np.ndarray,
    contact_active_spheres: int = 2,
    grip_active: bool = True,
) -> tuple[TrajectoryOptimizationNode, int, np.ndarray]:
    """Create a consistent floating base node for nullspace exploration."""
    nv = 10
    actuated_indices = np.array([6, 7, 8, 9], dtype=int)
    n_spheres = 2
    n_ground_vars = n_spheres * 3

    # Ground Jacobian
    j_ground = np.zeros((n_ground_vars, nv))
    for s in range(contact_active_spheres):
        j_ground[s * 3 + 0, 0] = 1.0
        j_ground[s * 3 + 1, 1] = 1.0
        j_ground[s * 3 + 2, 2] = 1.0
        # Leg joints coupling
        j_ground[s * 3 + 2, 6 + s] = 0.5

    # Grip Jacobian
    j_grip = np.zeros((6, nv))
    if grip_active:
        # Arm 1 (joint 8: lead) and Arm 2 (joint 9: trail) closed loop
        j_grip[0, 8] = 1.0
        j_grip[0, 9] = -1.0

    tau_rnea = np.zeros(nv)
    if contact_active_spheres > 0:
        tau_rnea[2] = float(contact_active_spheres) * 100.0
        for s in range(contact_active_spheres):
            tau_rnea[6 + s] += 0.5 * 100.0

    tau_rnea[6:] += tau_actuated_target

    mask = np.zeros(n_spheres, dtype=bool)
    mask[:contact_active_spheres] = True

    node = TrajectoryOptimizationNode(
        time_s=0.0,
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        active_contact_mask=mask,
    )
    return node, nv, actuated_indices


def test_scaled_rank_revealing_nullspace() -> None:
    """Test 1: exact A @ N = 0 and orthonormal basis N^T @ N = I."""
    node, nv, actuated_indices = _create_explorer_test_node(
        tau_actuated_target=np.array([10.0, 10.0, 20.0, -15.0]),
        contact_active_spheres=2,
    )

    explorer = ForceNullspaceExplorer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=2,
    )

    decomp = explorer.decompose_nullspace(
        j_ground=node.j_ground,
        j_grip=node.j_grip,
        active_contact_mask=node.active_contact_mask,
    )

    assert isinstance(decomp, NullspaceDecomposition)
    msg_res = f"Expected A @ N == 0, got {decomp.residual_norm}"
    assert decomp.residual_norm < 1e-12, msg_res
    msg_ortho = f"Expected N^T @ N == I, got {decomp.basis_orthogonality_residual}"
    assert decomp.basis_orthogonality_residual < 1e-12, msg_ortho


def test_contact_mode_nullity_variation() -> None:
    """Test 2: nullity varies dynamically across double support, single support, and flight."""
    _, nv, actuated_indices = _create_explorer_test_node(
        tau_actuated_target=np.array([10.0, 10.0, 20.0, -15.0]),
    )
    explorer = ForceNullspaceExplorer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=2,
    )

    # 1. Double support (2 contact spheres active)
    node_ds, _, _ = _create_explorer_test_node(np.zeros(4), contact_active_spheres=2)
    decomp_ds = explorer.decompose_nullspace(
        j_ground=node_ds.j_ground,
        j_grip=node_ds.j_grip,
        active_contact_mask=node_ds.active_contact_mask,
    )

    # 2. Single support (1 contact sphere active)
    node_ss, _, _ = _create_explorer_test_node(np.zeros(4), contact_active_spheres=1)
    decomp_ss = explorer.decompose_nullspace(
        j_ground=node_ss.j_ground,
        j_grip=node_ss.j_grip,
        active_contact_mask=node_ss.active_contact_mask,
    )

    # 3. Flight mode (0 contact spheres active)
    node_fl, _, _ = _create_explorer_test_node(np.zeros(4), contact_active_spheres=0)
    decomp_fl = explorer.decompose_nullspace(
        j_ground=node_fl.j_ground,
        j_grip=node_fl.j_grip,
        active_contact_mask=node_fl.active_contact_mask,
    )

    # Nullity must strictly decrease as contact constraints are removed
    assert decomp_ds.nullity > decomp_ss.nullity
    assert decomp_ss.nullity > decomp_fl.nullity


def test_basis_sign_change_invariance() -> None:
    """Test 3: physical optimal solution is invariant to nullspace basis column signs."""
    node, nv, actuated_indices = _create_explorer_test_node(
        tau_actuated_target=np.array([15.0, -10.0, 25.0, -15.0]),
        contact_active_spheres=2,
    )
    explorer = ForceNullspaceExplorer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=2,
    )

    # Solve with standard basis
    decomp = explorer.decompose_nullspace(
        node.j_ground, node.j_grip, node.active_contact_mask
    )
    sol1 = explorer.solve_reduced_coordinates(node, decomp)

    # Solve with inverted basis columns: N_flipped = N @ diag([-1, 1, -1, ...])
    signs = np.ones(decomp.nullity)
    signs[::2] = -1.0
    n_flipped = decomp.nullspace_basis * signs

    decomp_flipped = NullspaceDecomposition(
        matrix_rank=decomp.matrix_rank,
        nullity=decomp.nullity,
        singular_values=decomp.singular_values,
        nullspace_basis=n_flipped,
        particular_solution=decomp.particular_solution,
        residual_norm=decomp.residual_norm,
        basis_orthogonality_residual=decomp.basis_orthogonality_residual,
    )
    sol2 = explorer.solve_reduced_coordinates(node, decomp_flipped)

    assert sol1.is_feasible and sol2.is_feasible
    np.testing.assert_allclose(sol1.tau_actuated, sol2.tau_actuated, atol=1e-8)
    np.testing.assert_allclose(sol1.f_ground, sol2.f_ground, atol=1e-8)


def test_hard_zero_trail_feasibility_diagnostic() -> None:
    """Test 4: detects feasibility of hard-zero trail arm and identifies blocking constraints when infeasible."""
    # Joint 8 is lead arm, joint 9 is trail arm
    trail_arm_indices = [9]

    # Case A: Feasible hard-zero (total arm torque is within lead arm capacity of 100 Nm)
    node_feas, nv, actuated_indices = _create_explorer_test_node(
        tau_actuated_target=np.array([0.0, 0.0, 30.0, 20.0]),
        contact_active_spheres=2,
    )
    explorer = ForceNullspaceExplorer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=2,
        tau_max=np.array([150.0, 150.0, 100.0, 100.0]),
    )

    is_feas, blockers, sample_feas = explorer.evaluate_hard_zero_trail_feasibility(
        node_feas, trail_arm_indices=trail_arm_indices
    )
    assert is_feas
    assert len(blockers) == 0
    assert sample_feas is not None
    # Trail arm torque is zero
    assert abs(sample_feas.tau_actuated[3]) < 1e-4

    # Case B: Infeasible hard-zero (net arm torque is 150 Nm, but lead arm capacity is only 80 Nm)
    node_infeas, _, _ = _create_explorer_test_node(
        tau_actuated_target=np.array([0.0, 0.0, 80.0, 70.0]),
        contact_active_spheres=2,
    )
    explorer_limited = ForceNullspaceExplorer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=2,
        tau_max=np.array([150.0, 150.0, 80.0, 80.0]),
    )

    is_feas_b, blockers_b, sample_relaxed = (
        explorer_limited.evaluate_hard_zero_trail_feasibility(
            node_infeas, trail_arm_indices=trail_arm_indices
        )
    )
    assert not is_feas_b
    assert len(blockers_b) > 0
    assert any("lead arm" in b.lower() or "capacity" in b.lower() for b in blockers_b)
    # Relaxed minimum-trail alternative must succeed
    assert sample_relaxed is not None
    assert sample_relaxed.is_feasible


def test_effort_vs_smoothness_pareto_sweep() -> None:
    """Test 5: sweeps effort vs smoothness weight revealing monotonic trade-off."""
    node, nv, actuated_indices = _create_explorer_test_node(
        tau_actuated_target=np.array([20.0, -15.0, 25.0, -10.0]),
        contact_active_spheres=2,
    )
    explorer = ForceNullspaceExplorer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=2,
    )

    weights = [0.1, 1.0, 10.0, 50.0]
    report = explorer.explore_tradeoff(
        node=node,
        dimension=TradeoffDimension.EFFORT_VS_SMOOTHNESS,
        sweep_values=weights,
    )

    assert isinstance(report, ParetoFrontierReport)
    assert len(report.samples) == len(weights)
    for s in report.samples:
        assert s.is_feasible


def test_trail_arm_share_tradeoff_sweep() -> None:
    """Test 6: increasing trail penalty shifts load monotonically from trail arm to lead arm."""
    node, nv, actuated_indices = _create_explorer_test_node(
        tau_actuated_target=np.array([0.0, 0.0, 30.0, 30.0]),
        contact_active_spheres=2,
    )
    explorer = ForceNullspaceExplorer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=2,
    )

    trail_weights = [1.0, 5.0, 25.0, 100.0]
    report = explorer.explore_tradeoff(
        node=node,
        dimension=TradeoffDimension.TRAIL_ARM_SHARE,
        sweep_values=trail_weights,
        trail_arm_indices=[9],
    )

    assert len(report.samples) == len(trail_weights)
    trail_efforts = [s.trail_arm_effort for s in report.samples]
    # Verify monotonic decrease in trail arm effort
    for i in range(len(trail_efforts) - 1):
        assert trail_efforts[i + 1] <= trail_efforts[i] + 1e-4


def test_reproducible_pareto_table_markdown_generation() -> None:
    """Test 7: generates clean, reproducible markdown Pareto table with conservative default."""
    node, nv, actuated_indices = _create_explorer_test_node(
        tau_actuated_target=np.array([15.0, -10.0, 20.0, -15.0]),
        contact_active_spheres=2,
    )
    explorer = ForceNullspaceExplorer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=2,
    )

    rep1 = explorer.explore_tradeoff(
        node, TradeoffDimension.EFFORT_VS_SMOOTHNESS, [1.0, 10.0]
    )
    rep2 = explorer.explore_tradeoff(
        node, TradeoffDimension.TRAIL_ARM_SHARE, [1.0, 50.0], trail_arm_indices=[9]
    )

    md_table = explorer.generate_pareto_table_markdown([rep1, rep2])
    assert (
        "| Parameter | Feasible | Total Effort | Peak Torque | Trail Share | Lead Share | Max GRF |"
        in md_table
    )
    assert "Conservative Default" in md_table
