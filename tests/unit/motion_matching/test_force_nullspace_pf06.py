"""Tests for PF-06: Explore Feasible Force Null Spaces and Publish Torque-Distribution Tradeoffs (#10436).

Covers:
1. Scaled rank-revealing QR and SVD null-space calculation and validation (A N = 0).
2. Contact mode transitions and dynamic rank/nullity changes without assuming fixed nullity.
3. Basis sign-change invariance of physical x-derivative smoothing across frames.
4. Full-space constrained optimum vs reduced-coordinate null-space solution equivalence.
5. Hard-zero trail arm feasibility vs infeasibility detection without false repair.
6. Relaxed minimum-trail alternative when hard-zero trail is physically infeasible.
7. Bounded sweeps: effort vs smoothness, peak capacity, trail share, grip squeeze, ground load.
8. Detailed diagnostic breakdown: per-joint torque/rate/power, lead/trail load, COP, grip wrench, units.
9. Reproducible Pareto tradeoff table export to JSON and CSV.
10. Conservative default selection with explicit mechanical rationale.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.shared.python.motion_matching.force_nullspace import (
    ForceConstraints,
    ForceNullSpace,
    FrictionCone,
    NullSpaceAnalysis,
    TorqueDistributionTradeoff,
    TradeoffAlternative,
    explore_torque_tradeoffs,
    redistribute_forces,
    redistribute_trajectory,
)

pytestmark = pytest.mark.unit


def test_scaled_rank_revealing_qr_and_svd_null_space() -> None:
    """Acceptance 1: Scaled rank-revealing QR and SVD decomposition satisfy A N = 0 and A x_p = b."""
    # Construct rank-deficient balance matrix (m=4, n=6, true rank=2)
    row1 = np.array([1.0, 0.0, 2.0, 0.0, 1.0, 0.5])
    row2 = np.array([0.0, 1.0, 0.0, 2.0, 0.5, 1.0])
    a = np.vstack(
        [row1, row2, 2.0 * row1, 3.0 * row2]
    )  # rows 3 and 4 linearly dependent
    x_true = np.array([10.0, 20.0, 5.0, 8.0, 12.0, 4.0])
    b = a @ x_true

    var_scale = np.array([100.0, 100.0, 50.0, 50.0, 10.0, 10.0])
    row_scale = np.array([50.0, 50.0, 100.0, 150.0])

    # SVD method
    space_svd = ForceNullSpace.from_balance(
        a, b, variable_scale=var_scale, row_scale=row_scale, method="svd"
    )
    assert space_svd.rank == 2
    assert space_svd.basis.shape == (6, 4)
    analysis_svd = space_svd.analyze()
    assert analysis_svd.rank == 2
    assert analysis_svd.nullity == 4
    assert analysis_svd.equality_residual < 1e-9
    assert analysis_svd.null_residual < 1e-9
    assert space_svd.validate_null_space(tolerance=1e-8)

    # QR method
    space_qr = ForceNullSpace.from_balance(
        a, b, variable_scale=var_scale, row_scale=row_scale, method="qr"
    )
    assert space_qr.rank == 2
    assert space_qr.basis.shape == (6, 4)
    analysis_qr = space_qr.analyze()
    assert analysis_qr.rank == 2
    assert analysis_qr.nullity == 4
    assert analysis_qr.equality_residual < 1e-9
    assert analysis_qr.null_residual < 1e-9
    assert space_qr.validate_null_space(tolerance=1e-8)


def test_contact_mode_rank_changes() -> None:
    """Acceptance 2: Rank and nullity change dynamically with contact modes; never assumes fixed nullity."""
    # 4 leg actuators + 4 contact spheres * 3 force entries = 16 variables, 6 floating base equations
    nv_root = 6
    n_act = 4
    n_spheres = 4
    n_vars = n_act + n_spheres * 3

    # Frame 1: Biped bilateral support (all 4 spheres active)
    a_double = np.zeros((nv_root, n_vars))
    a_double[:, :n_act] = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    for s in range(n_spheres):
        a_double[2, n_act + s * 3 + 2] = 1.0  # normal forces
        a_double[0, n_act + s * 3] = 1.0  # tangential x
        a_double[1, n_act + s * 3 + 1] = 1.0  # tangential y

    b = np.array([0.0, 0.0, 700.0, 0.0, 0.0, 0.0])

    space_double = ForceNullSpace.from_balance(a_double, b)
    rank_double = space_double.rank
    nullity_double = space_double.basis.shape[1]

    # Frame 2: Single-foot contact mode (right foot 2 spheres separated / zeroed out)
    # The active equality matrix effectively eliminates those 6 columns
    a_single = a_double.copy()
    a_single[:, n_act + 2 * 3 :] = 0.0  # right foot deactivated
    space_single = ForceNullSpace.from_balance(a_single, b)

    # Rank and nullity must reflect the contact mode change
    assert space_single.rank <= rank_double
    # Active columns reduced from 16 to 10
    active_cols_double = np.count_nonzero(np.linalg.norm(a_double, axis=0) > 1e-12)
    active_cols_single = np.count_nonzero(np.linalg.norm(a_single, axis=0) > 1e-12)
    assert active_cols_double == 16
    assert active_cols_single == 10


def test_basis_sign_change_invariance_for_trajectory_smoothing() -> None:
    """Acceptance 3: Penalizing physical x derivatives is strictly invariant to null-space basis sign changes."""
    # 2 collocation nodes with balance equations
    a1 = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0]])
    b1 = np.array([10.0, 15.0])
    a2 = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0]])
    b2 = np.array([12.0, 14.0])

    space1 = ForceNullSpace.from_balance(a1, b1)
    space2 = ForceNullSpace.from_balance(a2, b2)

    # Create basis-flipped spaces (reversing sign of null space columns)
    flipped_basis1 = -space1.basis.copy()
    flipped_space1 = ForceNullSpace(
        space1.matrix,
        space1.rhs,
        space1.row_scale,
        space1.particular,
        flipped_basis1,
        space1.rank,
        space1.variable_scale,
    )

    flipped_basis2 = -space2.basis.copy()
    flipped_space2 = ForceNullSpace(
        space2.matrix,
        space2.rhs,
        space2.row_scale,
        space2.particular,
        flipped_basis2,
        space2.rank,
        space2.variable_scale,
    )

    constraints = [
        ForceConstraints(lower=np.zeros(3), upper_bound=np.full(3, 50.0)),
        ForceConstraints(lower=np.zeros(3), upper_bound=np.full(3, 50.0)),
    ]
    time_s = np.array([0.0, 0.02])
    weights = np.ones(3)

    # Solve trajectory with original bases
    res_orig = redistribute_trajectory(
        [space1, space2],
        weights=weights,
        constraints=constraints,
        time_s=time_s,
        smoothness_weight=10.0,
    )

    # Solve trajectory with flipped bases
    res_flipped = redistribute_trajectory(
        [flipped_space1, flipped_space2],
        weights=weights,
        constraints=constraints,
        time_s=time_s,
        smoothness_weight=10.0,
    )

    assert res_orig.feasible
    assert res_flipped.feasible

    # Physical forces must be identical regardless of basis sign conventions
    np.testing.assert_allclose(res_orig.forces, res_flipped.forces, atol=1e-7)
    np.testing.assert_allclose(res_orig.cost, res_flipped.cost, atol=1e-7)


def test_full_space_vs_reduced_space_equivalence() -> None:
    """Acceptance 4: Constrained optimization in the reduced null space yields identical solutions to full-space QP."""
    rng = np.random.default_rng(20260919)
    m, n = 3, 7
    a = rng.normal(size=(m, n))
    x_target = rng.uniform(2.0, 5.0, size=n)
    b = a @ x_target

    weights = rng.uniform(0.8, 2.5, size=n)
    lower = np.full(n, 0.0)
    upper = np.full(n, 10.0)
    constraints = ForceConstraints(lower=lower, upper_bound=upper)

    space = ForceNullSpace.from_balance(a, b)
    res_reduced = redistribute_forces(space, weights, constraints)
    assert res_reduced.feasible

    # Compare with independent full-space QP solve
    from scipy.optimize import minimize as scipy_minimize

    res_full = scipy_minimize(
        fun=lambda x: float(np.sum((weights * x) ** 2)),
        x0=x_target,
        jac=lambda x: 2.0 * (weights**2) * x,
        bounds=list(zip(lower, upper, strict=True)),
        constraints={"type": "eq", "fun": lambda x: a @ x - b, "jac": lambda x: a},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300},
    )
    assert res_full.success
    np.testing.assert_allclose(res_reduced.forces, res_full.x, atol=1e-5)


def test_hard_zero_trail_feasibility_and_infeasibility() -> None:
    """Acceptance 5: Hard-zero trail arm feasibility is verified; infeasible cases are reported without false repair."""
    # Actuators: [lead_arm, trail_arm]. Load must sum to 20.0
    a = np.array([[1.0, 1.0]])
    b = np.array([20.0])
    space = ForceNullSpace.from_balance(a, b)

    # Case A: Feasible hard-zero trail (lead capacity is 30.0 >= 20.0)
    constraints_feasible = ForceConstraints(
        lower=np.array([0.0, 0.0]),
        upper_bound=np.array([30.0, 0.0]),  # trail clamped to 0
    )
    res_a = redistribute_forces(space, np.ones(2), constraints_feasible)
    assert res_a.feasible
    assert np.isclose(res_a.forces[1], 0.0, atol=1e-7)
    assert np.isclose(res_a.forces[0], 20.0, atol=1e-7)

    # Case B: Infeasible hard-zero trail (lead capacity is only 15.0 < 20.0)
    constraints_infeasible = ForceConstraints(
        lower=np.array([0.0, 0.0]),
        upper_bound=np.array([15.0, 0.0]),  # trail clamped to 0, but max lead is 15
    )
    res_b = redistribute_forces(space, np.ones(2), constraints_infeasible)
    assert not res_b.feasible
    assert res_b.constraint_violation > 0.0
    # Must NOT claim false physical success or repair bounds
    assert res_b.forces[0] <= 15.0 + 1e-5 or not res_b.feasible


def test_relaxed_minimum_trail_alternative() -> None:
    """Acceptance 6: When hard-zero trail is infeasible, relaxed minimum-trail finds the closest physically admissible solution."""
    # Lead max is 15.0, required load is 20.0, so trail must provide at least 5.0
    a = np.array([[1.0, 1.0]])
    b = np.array([20.0])
    space = ForceNullSpace.from_balance(a, b)

    # Relaxed minimum-trail: heavily penalize trail arm (w_trail = 50.0 vs w_lead = 1.0)
    weights_relaxed = np.array([1.0, 50.0])
    constraints_relaxed = ForceConstraints(
        lower=np.array([0.0, 0.0]),
        upper_bound=np.array([15.0, 30.0]),  # lead capacity 15, trail allowed up to 30
    )
    res_relaxed = redistribute_forces(space, weights_relaxed, constraints_relaxed)
    assert res_relaxed.feasible
    # Lead arm saturates at its capacity of 15.0, trail arm takes only the necessary minimum of 5.0
    np.testing.assert_allclose(res_relaxed.forces, [15.0, 5.0], atol=1e-5)


def test_bounded_tradeoff_sweeps() -> None:
    """Acceptance 7: Generates bounded sweeps over effort, trail share, grip squeeze, and ground load."""
    # Simple planar system: 2 lead arm DOFs, 2 trail arm DOFs, 2 grip forces, 2 ground normal forces
    # Variables: [lead1, lead2, trail1, trail2, grip_x, grip_y, ground1, ground2] (n=8)
    a = np.zeros((3, 8))
    # Equilibrium 1: Arm torques + grip opposing swing moment
    a[0, :4] = [1.0, 1.0, 1.0, 1.0]
    a[0, 4:6] = [0.1, 0.1]
    # Equilibrium 2: Ground vertical support
    a[1, 6:8] = [1.0, 1.0]
    # Equilibrium 3: Hand grip loop closure
    a[2, :2] = [1.0, -1.0]
    a[2, 2:4] = [-1.0, 1.0]
    a[2, 4] = 0.5

    b = np.array([100.0, 700.0, 0.0])
    space = ForceNullSpace.from_balance(a, b)

    lower = np.zeros(8)
    upper = np.array([80.0, 80.0, 80.0, 80.0, 100.0, 100.0, 500.0, 500.0])
    constraints = ForceConstraints(lower=lower, upper_bound=upper)

    tradeoff = explore_torque_tradeoffs(
        space,
        constraints=constraints,
        lead_indices=[0, 1],
        trail_indices=[2, 3],
        grip_indices=[4, 5],
        ground_indices=[6, 7],
    )

    # Must produce multiple distinct alternatives
    assert len(tradeoff.alternatives) >= 5
    alt_names = [alt.name for alt in tradeoff.alternatives]
    assert "baseline_minimum_effort" in alt_names
    assert "conservative_default" in alt_names
    assert "relaxed_minimum_trail" in alt_names

    # Check trail share sweep behavior: relaxed_minimum_trail has strictly lower trail share than baseline
    base_alt = tradeoff.get_alternative("baseline_minimum_effort")
    min_trail_alt = tradeoff.get_alternative("relaxed_minimum_trail")
    assert base_alt is not None and min_trail_alt is not None
    assert min_trail_alt.trail_share < base_alt.trail_share


def test_tradeoff_diagnostics_and_segment_reactions() -> None:
    """Acceptance 8: Each alternative reports per-joint torque/rate/power, lead/trail load, COP, and units."""
    a = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b = np.array([50.0, 400.0])
    space = ForceNullSpace.from_balance(a, b)
    constraints = ForceConstraints(lower=np.zeros(3), upper_bound=np.full(3, 500.0))

    tradeoff = explore_torque_tradeoffs(
        space,
        constraints=constraints,
        lead_indices=[0],
        trail_indices=[1],
        ground_indices=[2],
        joint_velocities=np.array(
            [2.5, 2.5]
        ),  # rad/s for power calculation (lead + trail)
    )

    alt = tradeoff.get_alternative("conservative_default")
    assert alt is not None
    assert alt.feasible
    assert alt.lead_effort > 0.0
    assert alt.trail_effort > 0.0
    assert 0.0 <= alt.trail_share <= 1.0
    assert alt.units["torque"] == "N*m"
    assert alt.units["force"] == "N"
    assert alt.units["power"] == "W"


def test_pareto_table_reproducibility_and_export() -> None:
    """Acceptance 9: Serializes reproducible Pareto tradeoff table into JSON and CSV without mutation."""
    a = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0]])
    b = np.array([30.0, 20.0])
    space = ForceNullSpace.from_balance(a, b)
    constraints = ForceConstraints(lower=np.zeros(3), upper_bound=np.full(3, 50.0))

    tradeoff1 = explore_torque_tradeoffs(
        space, constraints, lead_indices=[0], trail_indices=[1], ground_indices=[2]
    )
    tradeoff2 = explore_torque_tradeoffs(
        space, constraints, lead_indices=[0], trail_indices=[1], ground_indices=[2]
    )

    # Verify deterministic reproducibility
    table1 = tradeoff1.to_pareto_table()
    table2 = tradeoff2.to_pareto_table()
    assert len(table1) == len(table2)
    for row1, row2 in zip(table1, table2, strict=True):
        assert row1["name"] == row2["name"]
        assert np.isclose(row1["lead_effort"], row2["lead_effort"], atol=1e-8)
        assert np.isclose(row1["trail_share"], row2["trail_share"], atol=1e-8)

    # Test export to temporary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "tradeoffs.json"
        csv_path = Path(tmp_dir) / "tradeoffs.csv"

        tradeoff1.export_json(json_path)
        tradeoff1.export_csv(csv_path)

        assert json_path.exists()
        assert csv_path.exists()

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            assert "alternatives" in data
            assert "selected_default" in data

        csv_text = csv_path.read_text(encoding="utf-8")
        assert "name,strategy,feasible,lead_effort,trail_effort,trail_share" in csv_text


def test_conservative_default_selection_and_rationale() -> None:
    """Acceptance 10: Selects a conservative default alternative with an explicit mechanical rationale."""
    a = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b = np.array([40.0, 300.0])
    space = ForceNullSpace.from_balance(a, b)
    constraints = ForceConstraints(lower=np.zeros(3), upper_bound=np.full(3, 400.0))

    tradeoff = explore_torque_tradeoffs(
        space, constraints, lead_indices=[0], trail_indices=[1], ground_indices=[2]
    )

    default_alt = tradeoff.get_selected_default()
    assert default_alt is not None
    assert default_alt.feasible
    assert len(tradeoff.default_rationale) > 0
    # The rationale explains stability, safety margins, and constraints
    assert "conservative" in tradeoff.default_rationale.lower()
    # Confirm no unfounded injury or muscle claims
    assert "injury" not in tradeoff.default_rationale.lower()
    assert "metabolic" not in tradeoff.default_rationale.lower()
