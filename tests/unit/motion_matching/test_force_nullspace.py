"""Independent mechanical fixtures for force-only redistribution (#10436)."""

import numpy as np
import pytest
from scipy.optimize import minimize

from src.shared.python.motion_matching.force_nullspace import (
    ForceConstraints,
    ForceNullSpace,
    FrictionCone,
    redistribute_forces,
    redistribute_trajectory,
)

pytestmark = pytest.mark.unit


def test_scaled_rank_deficient_balance_and_owned_arrays():
    a = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    space = ForceNullSpace.from_balance(
        a, np.array([4.0, 8.0]), variable_scale=np.array([100.0, 1.0, 10.0])
    )
    assert space.rank == 1
    assert space.basis.shape == (3, 2)
    np.testing.assert_allclose(a @ space.basis, 0.0, atol=1e-12)
    np.testing.assert_allclose(a @ space.particular, [4.0, 8.0])
    a[:] = 0
    assert not space.basis.flags.writeable
    assert space.matrix[0, 0] == 1


def test_inconsistent_balance_is_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        ForceNullSpace.from_balance(np.array([[1.0], [2.0]]), np.array([1.0, 3.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_balance_is_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        ForceNullSpace.from_balance(np.array([[bad]]), np.ones(1))


def test_minimum_effort_matches_analytical_load_sharing():
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([12.0]))
    result = redistribute_forces(
        space, np.array([1.0, 2.0]), ForceConstraints(np.zeros(2), np.full(2, 20.0))
    )
    assert result.feasible
    np.testing.assert_allclose(result.forces, [9.6, 2.4], atol=1e-7)
    assert result.equality_error < 1e-9
    assert result.converged


def test_hard_zero_trail_infeasible_is_not_repaired_or_successful():
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([12.0]))
    constraints = ForceConstraints(np.zeros(2), np.array([10.0, 0.0]))
    result = redistribute_forces(space, np.ones(2), constraints)
    assert not result.feasible
    assert result.constraint_violation > 0


def test_diagonal_friction_violation_rejected_with_no_nullity():
    space = ForceNullSpace.from_balance(np.eye(3), np.array([75.0, 75.0, 100.0]))
    constraints = ForceConstraints(
        np.full(3, -200.0),
        np.full(3, 200.0),
        cones=(FrictionCone((0, 1, 2), np.eye(3), 0.8),),
    )
    result = redistribute_forces(space, np.ones(3), constraints)
    assert not result.feasible
    assert result.constraint_violation > 26


def test_ground_normal_uses_declared_frame():
    # Columns are tangent, tangent, normal; the surface normal is world +X.
    frame = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    space = ForceNullSpace.from_balance(np.eye(3), np.array([100.0, 0.0, 0.0]))
    constraints = ForceConstraints(
        np.full(3, -200.0),
        np.full(3, 200.0),
        cones=(FrictionCone((0, 1, 2), frame, 0.8),),
    )
    assert redistribute_forces(space, np.ones(3), constraints).feasible


def test_sign_invariance_for_physical_reference_objective():
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([12.0]))
    flipped = ForceNullSpace.from_balance(-np.ones((1, 2)), np.array([-12.0]))
    constraints = ForceConstraints(np.zeros(2), np.full(2, 20.0))
    a = redistribute_forces(
        space, np.ones(2), constraints, reference=np.array([8.0, 4.0])
    )
    b = redistribute_forces(
        flipped, np.ones(2), constraints, reference=np.array([8.0, 4.0])
    )
    assert a.feasible and b.feasible
    np.testing.assert_allclose(a.forces, b.forces, atol=1e-8)
    np.testing.assert_allclose(a.forces, [8.0, 4.0], atol=1e-7)


def test_contact_inequalities_limit_load_transfer():
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([12.0]))
    constraints = ForceConstraints(
        np.zeros(2),
        np.full(2, 20.0),
        matrix=np.array([[1.0, 0.0]]),
        upper=np.array([7.0]),
    )
    result = redistribute_forces(space, np.array([1.0, 10.0]), constraints)
    assert result.feasible
    np.testing.assert_allclose(result.forces, [7.0, 5.0], atol=1e-6)


def test_invalid_surface_frame_rejected():
    with pytest.raises(ValueError, match="orthonormal"):
        FrictionCone((0, 1, 2), np.ones((3, 3)), 0.8)


def test_negative_weights_rejected():
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([12.0]))
    with pytest.raises(ValueError, match="positive"):
        redistribute_forces(
            space, np.array([1.0, -1.0]), ForceConstraints(np.zeros(2), np.ones(2))
        )


def test_reduced_solution_agrees_with_independent_full_space_qp():
    rng = np.random.default_rng(10436)
    a = rng.normal(size=(3, 8))
    b = a @ np.full(8, 0.4)
    weights = np.linspace(0.5, 2.0, 8)
    constraints = ForceConstraints(np.zeros(8), np.ones(8))
    space = ForceNullSpace.from_balance(a, b)
    reduced = redistribute_forces(space, weights, constraints)
    full = minimize(
        lambda x: float(np.sum((weights * x) ** 2)),
        np.full(8, 0.4),
        jac=lambda x: 2 * weights**2 * x,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 8,
        constraints={"type": "eq", "fun": lambda x: a @ x - b, "jac": lambda x: a},
        options={"ftol": 1e-12},
    )
    assert full.success and reduced.feasible
    np.testing.assert_allclose(reduced.forces, full.x, atol=2e-6)


def test_cone_limits_feasible_redistribution():
    # x = [actuator, tangential_x, tangential_y, normal].
    # The actuator supplies the load that friction cannot support.
    a = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    space = ForceNullSpace.from_balance(a, np.array([120.0, 0.0, 100.0]))
    constraints = ForceConstraints(
        np.full(4, -200.0),
        np.full(4, 200.0),
        cones=(FrictionCone((1, 2, 3), np.eye(3), 0.8),),
    )
    result = redistribute_forces(space, np.array([1.0, 0.01, 0.01, 0.01]), constraints)
    assert result.feasible
    np.testing.assert_allclose(result.forces, [40.0, 80.0, 0.0, 100.0], atol=1e-6)


def _patch_core_solver(monkeypatch: pytest.MonkeyPatch, result) -> None:
    """Replace the conic core call used by the redistribution shims (NSX-24)."""
    from src.shared.python.motion_matching.force_nullspace import redistribution

    monkeypatch.setattr(redistribution, "solve_qp", lambda *a, **kw: result)


def test_solver_success_does_not_override_constraint_audit(monkeypatch):
    # NSX-24: the solver seam moved from scipy ``minimize`` to the conic core;
    # the audit contract under test is unchanged.
    from src.shared.python.loop_nullspace.solvers import ConicResult, SolveStatus

    _patch_core_solver(monkeypatch, ConicResult(SolveStatus.OPTIMAL, x=np.array([1e5])))
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([12.0]))
    result = redistribute_forces(
        space, np.ones(2), ForceConstraints(np.zeros(2), np.full(2, 20.0))
    )
    assert result.converged and not result.feasible


@pytest.mark.parametrize("status_name", ["NUMERICAL", "INFEASIBLE"])
def test_non_optimal_solver_status_is_not_converged_for_forces(
    monkeypatch: pytest.MonkeyPatch, status_name: str
) -> None:
    # Was parametrised over SLSQP exit codes 8/9; the conic core reports
    # NUMERICAL/INFEASIBLE instead.  Same contract: not converged, and the
    # particular solution is audited as is (no repair).
    from src.shared.python.loop_nullspace.solvers import ConicResult, SolveStatus

    _patch_core_solver(
        monkeypatch, ConicResult(SolveStatus[status_name], reason="stopped")
    )
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([12.0]))
    result = redistribute_forces(
        space, np.ones(2), ForceConstraints(np.zeros(2), np.full(2, 20.0))
    )

    assert result.feasible
    assert not result.converged
    assert status_name.lower() in result.message.lower()


@pytest.mark.parametrize("status_name", ["NUMERICAL", "INFEASIBLE"])
def test_non_optimal_solver_status_is_not_converged_for_trajectory(
    monkeypatch: pytest.MonkeyPatch, status_name: str
) -> None:
    from src.shared.python.loop_nullspace.solvers import ConicResult, SolveStatus

    _patch_core_solver(
        monkeypatch, ConicResult(SolveStatus[status_name], reason="stopped")
    )
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([12.0]))
    constraints = ForceConstraints(np.zeros(2), np.full(2, 20.0))
    result = redistribute_trajectory(
        [space, space],
        np.ones(2),
        [constraints, constraints],
        np.array([0.0, 1.0]),
    )

    assert result.feasible
    assert not result.converged


def test_zero_matrix_has_full_nullity():
    space = ForceNullSpace.from_balance(np.zeros((2, 3)), np.zeros(2))
    assert space.rank == 0
    assert space.basis.shape == (3, 3)


@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (0, 1, 2, 2)])
def test_malformed_cone_index_count_rejected(indices):
    with pytest.raises(ValueError, match="three"):
        FrictionCone(indices, np.eye(3), 0.8)


def test_nonfinite_derived_margin_fails_closed():
    space = ForceNullSpace.from_balance(np.eye(1), np.ones(1))
    constraints = ForceConstraints(
        np.zeros(1),
        np.full(1, 2.0),
        matrix=np.array([[-1e308]]),
        upper=np.array([1e308]),
    )
    with np.errstate(over="ignore"):
        result = redistribute_forces(space, np.ones(1), constraints)
    assert not result.feasible
