"""NSX-24 (#12567): the legacy allocators are shims over the conic core.

Parity oracles here are independent scipy SLSQP solves of the *same* QP.  They
are test-side references only; the shipped code no longer contains SLSQP.
Where SLSQP stops short of the exact optimum the conic result must be at least
as optimal (objective no larger) and feasible; it is never allowed to be worse.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from src.shared.python.motion_matching import contact_force_allocator as cfa
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocator,
    FeasibilityStatus,
)
from src.shared.python.motion_matching.force_nullspace import (
    ForceConstraints,
    ForceNullSpace,
    FrictionCone,
    redistribute_forces,
    redistribute_trajectory,
)

pytestmark = pytest.mark.unit

NV = 12
ACT = np.arange(6, NV)


def _scene(seed: int, n_spheres: int = 2):
    rng = np.random.default_rng(seed)
    j_ground = rng.normal(size=(3 * n_spheres, NV))
    j_grip = np.zeros((6, NV))
    j_grip[:, 6:12] = rng.normal(size=(6, 6))
    f0 = np.tile([0.5, -0.3, 8.0], n_spheres)  # strictly inside mu = 0.8
    tau0 = rng.normal(size=ACT.size)
    tau = np.zeros(NV)
    tau[ACT] = tau0
    tau = tau + j_ground.T @ f0 + j_grip.T @ rng.normal(size=6)
    return tau, j_ground, j_grip, n_spheres


class _Capture:
    """Record the scaled QP handed to the conic core."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.kwargs: dict = {}
        self.x: np.ndarray | None = None
        real = cfa.solve_qp

        def spy(P, c, constraints, **kw):  # noqa: N803
            self.kwargs = dict(vars(constraints))
            out = real(P, c, constraints, **kw)
            self.x = out.x
            return out

        monkeypatch.setattr(cfa, "solve_qp", spy)


def _dense(m) -> np.ndarray:
    return np.asarray(m.todense() if hasattr(m, "todense") else m, dtype=float)


def _slsqp_reference(kw: dict) -> tuple[float, bool]:
    a, b = _dense(kw["A_eq"]), kw["b_eq"]
    g = kw["A_ub"]
    n = a.shape[1]
    cons = [LinearConstraint(a, b, b)]
    if g is not None:
        cons.append(LinearConstraint(_dense(g), -np.inf, kw["b_ub"]))
    lo, hi = kw["lo"], kw["hi"]
    start = np.clip(np.zeros(n), lo, hi)
    out = minimize(
        lambda y: 0.5 * float(y @ y),
        start,
        jac=lambda y: y,
        method="SLSQP",
        bounds=list(zip(lo, hi, strict=True)),
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return 0.5 * float(out.x @ out.x), bool(out.success)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize(
    "objective",
    [AllocationObjective.MINIMUM_EFFORT, AllocationObjective.MINIMUM_TRAIL_ARM],
)
def test_contact_allocator_is_at_least_as_optimal_as_slsqp(
    monkeypatch, seed, objective
) -> None:
    tau, jg, jgrip, ns = _scene(seed)
    tau_bounds = (np.full(ACT.size, -400.0), np.full(ACT.size, 400.0))
    cap = _Capture(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alloc = ContactForceAllocator(NV, ACT, n_contact_spheres=ns)
    res = alloc.allocate(
        tau,
        jg,
        jgrip,
        objective=objective,
        trail_arm_indices=[6, 7, 8],
        tau_bounds=tau_bounds,
    )
    assert res.success
    assert res.feasibility_status is FeasibilityStatus.FEASIBLE
    ref_obj, ref_ok = _slsqp_reference(cap.kwargs)
    core_obj = 0.5 * float(cap.x @ cap.x)
    if ref_ok:
        assert core_obj <= ref_obj * (1 + 1e-6) + 1e-9
    assert res.equilibrium_residual < 1e-4
    assert res.friction_violations < 1e-5


def test_hard_zero_trail_infeasibility_is_reported_not_hidden() -> None:
    # Trail coordinates carry a load nothing else can supply.
    rng = np.random.default_rng(7)
    jg = np.zeros((3, NV))
    jg[:, 0:3] = np.eye(3)
    jgrip = np.zeros((6, NV))
    jgrip[:, 0:6] = np.eye(6)
    tau = np.zeros(NV)
    tau[8] = 50.0 + rng.uniform()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alloc = ContactForceAllocator(NV, ACT, n_contact_spheres=1)
    res = alloc.allocate(
        tau,
        jg,
        jgrip,
        objective=AllocationObjective.HARD_ZERO_TRAIL,
        trail_arm_indices=[8],
        tau_bounds=(np.full(ACT.size, -1e3), np.full(ACT.size, 1e3)),
    )
    assert not res.success
    assert not res.is_physically_feasible
    assert res.feasibility_status is FeasibilityStatus.INFEASIBLE_HARD_ZERO_TRAIL
    assert abs(res.tau_actuated[2]) < 1e-9  # trail torque stays at the hard zero


def test_contact_allocator_emits_deprecation_warning_pointing_to_core() -> None:
    with pytest.warns(DeprecationWarning, match="loop_nullspace"):
        ContactForceAllocator(NV, ACT, n_contact_spheres=1)


def test_multi_engine_allocator_emits_deprecation_warning_pointing_to_core() -> None:
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        EngineType,
        MultiEngineTorqueAllocator,
        create_engine_force_adapter,
    )

    adapter = create_engine_force_adapter(
        EngineType.PINOCCHIO, nv=NV + 32, allow_synthetic=True
    )
    with pytest.warns(DeprecationWarning, match="loop_nullspace"):
        MultiEngineTorqueAllocator(adapter)


def test_multi_engine_allocator_warns_once_not_twice() -> None:
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        EngineType,
        MultiEngineTorqueAllocator,
        create_engine_force_adapter,
    )

    adapter = create_engine_force_adapter(
        EngineType.PINOCCHIO, nv=NV + 32, allow_synthetic=True
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MultiEngineTorqueAllocator(adapter)
    assert [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "loop_nullspace" in str(w.message)
    ].__len__() == 1


def test_missing_clarabel_raises_importerror_without_fallback(monkeypatch) -> None:
    import builtins

    tau, jg, jgrip, ns = _scene(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alloc = ContactForceAllocator(NV, ACT, n_contact_spheres=ns)
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "clarabel" or name.startswith("clarabel."):
            raise ImportError("No module named 'clarabel'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="loop-nullspace"):
        alloc.allocate(tau, jg, jgrip)
    space = ForceNullSpace.from_balance(np.ones((1, 3)), np.array([3.0]))
    with pytest.raises(ImportError, match="loop-nullspace"):
        redistribute_forces(
            space, np.ones(3), ForceConstraints(np.zeros(3), np.ones(9)[:3] * 9)
        )


# --------------------------------------------------------------- redistribution
def _traj_problem(seed: int, t: int = 4, n: int = 6):
    rng = np.random.default_rng(seed)
    spaces, cons = [], []
    for _ in range(t):
        a = rng.normal(size=(2, n))
        x_true = np.concatenate([rng.normal(size=3), [0.3, -0.2, 4.0]])
        spaces.append(ForceNullSpace.from_balance(a, a @ x_true))
        lower = np.full(n, -50.0)
        upper = np.full(n, 50.0)
        lower[3:] = -20.0
        upper[3:] = 20.0
        cons.append(
            ForceConstraints(
                lower, upper, cones=(FrictionCone((3, 4, 5), np.eye(3), 0.6),)
            )
        )
    return spaces, cons, np.linspace(0.0, 0.6, t)


def _trajectory_cost(xs, w, lam, time):
    cost = sum(float((w * x) @ (w * x)) for x in xs)
    for k in range(len(xs) - 1):
        r = (xs[k + 1] - xs[k]) / (time[k + 1] - time[k])
        cost += lam * float(r @ r)
    return cost


def _slsqp_trajectory(spaces, cons, w, lam, time):
    t, n = len(spaces), spaces[0].particular.size

    def split(v):
        return v.reshape(t, n)

    eq = LinearConstraint(
        np.block(
            [
                [
                    spaces[k].matrix if i == k else np.zeros_like(spaces[k].matrix)
                    for i in range(t)
                ]
                for k in range(t)
            ]
        ),
        np.concatenate([s.rhs for s in spaces]),
        np.concatenate([s.rhs for s in spaces]),
    )
    cone = NonlinearConstraint(
        lambda v: np.concatenate([cons[k].margins(split(v)[k])[-2:] for k in range(t)]),
        0.0,
        np.inf,
    )
    start = np.concatenate([s.particular for s in spaces])
    out = minimize(
        lambda v: _trajectory_cost(list(split(v)), w, lam, time),
        start,
        method="SLSQP",
        bounds=[(cons[0].lower[i], cons[0].upper_bound[i]) for i in range(n)] * t,
        constraints=[eq, cone],
        options={"maxiter": 1000, "ftol": 1e-13},
    )
    return list(split(out.x)), bool(out.success)


@pytest.mark.parametrize("lam", [0.0, 0.5, 20.0])
def test_trajectory_shim_is_at_least_as_optimal_as_slsqp(lam) -> None:
    spaces, cons, time = _traj_problem(11)
    w = np.linspace(0.5, 1.5, 6)
    res = redistribute_trajectory(
        spaces, w, cons, time, smoothness_weight=lam, tolerance=1e-10
    )
    assert res.feasible and res.converged
    ref_x, ref_ok = _slsqp_trajectory(spaces, cons, w, lam, time)
    assert ref_ok
    assert res.cost <= _trajectory_cost(ref_x, w, lam, time) * (1 + 1e-7) + 1e-9
    assert res.constraint_violation <= 1e-6


def test_trajectory_result_is_invariant_to_basis_sign_flips() -> None:
    spaces, cons, time = _traj_problem(3)
    w = np.ones(6)
    base = redistribute_trajectory(spaces, w, cons, time, smoothness_weight=2.0)
    flipped = []
    for s in spaces:
        import dataclasses

        flipped.append(dataclasses.replace(s, basis=-s.basis))
    other = redistribute_trajectory(flipped, w, cons, time, smoothness_weight=2.0)
    np.testing.assert_allclose(base.forces, other.forces, atol=1e-6)


def test_forces_cone_case_matches_analytic_optimum() -> None:
    a = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    space = ForceNullSpace.from_balance(a, np.array([120.0, 0.0, 100.0]))
    cons = ForceConstraints(
        np.full(4, -200.0),
        np.full(4, 200.0),
        cones=(FrictionCone((1, 2, 3), np.eye(3), 0.8),),
    )
    res = redistribute_forces(space, np.array([1.0, 0.01, 0.01, 0.01]), cons)
    np.testing.assert_allclose(res.forces, [40.0, 80.0, 0.0, 100.0], atol=1e-6)
    assert res.converged and res.feasible


def test_infeasible_redistribution_is_reported_and_not_repaired() -> None:
    space = ForceNullSpace.from_balance(np.ones((1, 2)), np.array([100.0]))
    res = redistribute_forces(
        space, np.ones(2), ForceConstraints(np.zeros(2), np.full(2, 10.0))
    )
    assert not res.converged
    assert not res.feasible
    assert res.constraint_violation > 1.0
