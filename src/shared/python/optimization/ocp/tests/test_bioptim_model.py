"""Phase 1.2: ``SwingBioModel`` drives a torque-driven bioptim OCP."""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.shared.python.optimization.ocp import _compat

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_bioptim,
    pytest.mark.skipif(not _compat.bioptim_available(), reason="bioptim not installed"),
]

os.environ.setdefault("MPLBACKEND", "Agg")


def _prepare_ocp(ode_solver, n_shooting: int = 20, final_time: float = 1.0):
    bioptim = _compat.require_bioptim()
    from src.shared.python.optimization._swing_models import ClubModel, GolferModel
    from src.shared.python.optimization.ocp.bioptim_model import make_swing_bio_model

    model = make_swing_bio_model(GolferModel(), ClubModel())
    n = model.nb_q

    objectives = bioptim.ObjectiveList()
    objectives.add(
        bioptim.ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1.0
    )

    dynamics = bioptim.DynamicsOptionsList()
    dynamics.add(
        bioptim.DynamicsOptions(
            ode_solver=ode_solver,
            expand_dynamics=True,
            phase_dynamics=bioptim.PhaseDynamics.SHARED_DURING_THE_PHASE,
        )
    )

    q_bounds = model.bounds_from_ranges("q")
    x_bounds = bioptim.BoundsList()
    x_bounds["q"] = q_bounds
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    # Start at rest at the address pose; end at rest at a raised pose.
    address = np.zeros(n)
    target = np.array([0.2, 0.3, 0.1, 0.8, 0.5, -0.3, 0.2])
    x_bounds["q"][:, 0] = address
    x_bounds["q"][:, -1] = target
    x_bounds["qdot"][:, 0] = 0.0
    x_bounds["qdot"][:, -1] = 0.0

    limits = model.symbolic.torque_limits()
    u_bounds = bioptim.BoundsList()
    u_bounds["tau"] = -limits, limits

    x_init = bioptim.InitialGuessList()
    x_init.add(
        "q",
        np.column_stack([address, target]),
        interpolation=bioptim.InterpolationType.LINEAR,
    )
    x_init["qdot"] = np.zeros(n)
    u_init = bioptim.InitialGuessList()
    u_init["tau"] = np.zeros(n)

    return (
        bioptim.OptimalControlProgram(
            model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objectives,
            use_sx=False,
            n_threads=1,
        ),
        model,
        target,
    )


def _solve(ocp, max_iterations: int = 300):
    bioptim = _compat.require_bioptim()
    solver = bioptim.Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(max_iterations)
    return ocp.solve(solver=solver)


def test_model_surface_matches_bioptim_expectations() -> None:
    from src.shared.python.optimization.ocp.bioptim_model import make_swing_bio_model

    model = make_swing_bio_model()
    assert model.name_dofs[0] == "hip_rotation" and model.nb_q == 7 == model.nb_tau
    assert model.nb_root == 0 and model.nb_markers == 6
    assert model.marker_index("clubhead") == 5
    empty = np.zeros(0)
    tau_max, tau_min = model.tau_max()(np.zeros(7), np.zeros(7), empty)
    np.testing.assert_allclose(
        np.asarray(tau_max).ravel(), -np.asarray(tau_min).ravel()
    )
    assert float(model.mass()(empty)) > 30.0
    cls, kwargs = model.serialize()
    assert cls(**kwargs).nb_q == 7
    assert model.copy().symbolic.parameter_names == ()


def test_rk4_swing_ocp_converges() -> None:
    bioptim = _compat.require_bioptim()
    ocp, model, target = _prepare_ocp(bioptim.OdeSolver.RK4(n_integration_steps=4))
    sol = _solve(ocp)
    assert sol.status == 0, sol.status
    states = sol.decision_states(to_merge=bioptim.SolutionMerge.NODES)
    q = np.asarray(states["q"])
    np.testing.assert_allclose(q[:, -1], target, atol=1e-6)
    controls = sol.decision_controls(to_merge=bioptim.SolutionMerge.NODES)
    tau = np.asarray(controls["tau"])
    limits = model.symbolic.torque_limits()
    assert np.all(np.abs(tau) <= limits[:, None] + 1e-6)


def test_collocation_swing_ocp_converges() -> None:
    bioptim = _compat.require_bioptim()
    ocp, _model, target = _prepare_ocp(
        bioptim.OdeSolver.COLLOCATION(polynomial_degree=3), n_shooting=12
    )
    sol = _solve(ocp)
    assert sol.status == 0, sol.status
    q = np.asarray(sol.decision_states(to_merge=bioptim.SolutionMerge.NODES)["q"])
    np.testing.assert_allclose(q[:, -1], target, atol=1e-6)
