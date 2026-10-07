"""Multiple-shooting transcription of the DIME dynamics window (#11554).

States and controls are both decision variables; each interval contributes a
transition defect ``d_k = x_{k+1} - Phi(x_k, u_k, dt)`` evaluated with the
provider's own step.  The tests use a linear provider, for which ``Phi`` is the
exact affine map ``x_{k+1} = A x_k + B u_k`` and the whole window problem is a
linear least-squares problem whose optimum is computed independently with
``numpy.linalg.lstsq``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import (
    AnalyticPendulumProvider,
    DeterministicFakeProvider,
    DimeCompleteState,
)
from src.shared.python.estimation.dime_dynamics_window import (
    CONTROL_EFFORT_RESIDUAL_SCALE,
    DefectMode,
    DimeDynamicsWindowProblem,
    ModelDiscrepancyBounds,
    _build_residuals_evaluator,
    compute_window_transition_defects,
    expected_residual_size,
    pack_window_decision,
    solve_dime_dynamics_window,
    unpack_window_decision,
)
from src.shared.python.motion_matching.counterfactual import (
    AccelerationDecomposition,
)

pytestmark = pytest.mark.unit

STIFFNESS = 4.0  # k in a = -k q + c u  [1/s^2]
GAIN = 2.0  # c in a = -k q + c u  [1/(kg m^2)]
DT = 0.05
UNITS = {"length": "m", "angle": "rad", "time": "s"}


class LinearOscillatorProvider(AnalyticPendulumProvider):
    """Pendulum provider with the linear acceleration ``a = -k q + c u``.

    The inherited semi-implicit Euler step then is exactly
    ``x_{k+1} = A x_k + B u_k`` with ``x = [q, v]``.
    """

    def compute_acceleration_decomposition(  # type: ignore[override]
        self, state: DimeCompleteState, controls: np.ndarray
    ) -> AccelerationDecomposition:
        tau = float(controls[0]) if len(controls) > 0 else 0.0
        return AccelerationDecomposition(
            a_grav=np.array([-STIFFNESS * float(state.q[0])]),
            a_drift=np.zeros(1),
            a_ctrl=np.array([GAIN * tau]),
        )


A_MAT = np.array([[1.0 - STIFFNESS * DT**2, DT], [-STIFFNESS * DT, 1.0]])
B_MAT = np.array([[GAIN * DT**2], [GAIN * DT]])
X0 = np.array([0.2, -0.1])


def _rollout(controls: np.ndarray) -> np.ndarray:
    """Analytic knot states x_1..x_N of the linear system."""
    x = X0.copy()
    knots = []
    for u in controls:
        x = A_MAT @ x + B_MAT @ u
        knots.append(x)
    return np.asarray(knots)


def _problem(
    horizon: int,
    targets: list[np.ndarray] | None = None,
    **kwargs: object,
) -> DimeDynamicsWindowProblem:
    provider = LinearOscillatorProvider()
    state = DimeCompleteState(
        t=0.0, q=X0[:1], v=X0[1:], units=UNITS, model_hash=provider.model_hash
    )
    return DimeDynamicsWindowProblem(
        provider=provider,
        initial_state=state,
        horizon_steps=horizon,
        dt_s=DT,
        target_positions=targets,
        **kwargs,  # type: ignore[arg-type]
    )


def _analytic_optimum(
    horizon: int,
    targets: list[np.ndarray],
    w_obs: float,
    w_def: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact optimum of the linear window problem (no rate term, no bounds).

    Unknowns z = [u_0..u_{N-1}, x_1..x_N]; every residual is affine in z.
    """
    n_u, n_x = 1, 2
    n_z = horizon * n_u + horizon * n_x
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    def x_col(k: int) -> int:  # column of knot x_k (k >= 1)
        return horizon * n_u + (k - 1) * n_x

    for k, y in enumerate(targets):
        if k == 0:
            continue  # x_0 is fixed: constant residual, no unknowns
        row = np.zeros(n_z)
        row[x_col(k)] = np.sqrt(w_obs)
        rows.append(row)
        rhs.append(np.sqrt(w_obs) * float(y[0]))

    for k in range(horizon):
        for i in range(n_x):
            row = np.zeros(n_z)
            row[x_col(k + 1) + i] = np.sqrt(w_def)
            row[k * n_u] = -np.sqrt(w_def) * B_MAT[i, 0]
            const = 0.0
            if k == 0:
                const = float(A_MAT[i] @ X0)
            else:
                row[x_col(k) : x_col(k) + n_x] = -np.sqrt(w_def) * A_MAT[i]
            rows.append(row)
            rhs.append(np.sqrt(w_def) * const)

    for k in range(horizon):
        row = np.zeros(n_z)
        row[k] = CONTROL_EFFORT_RESIDUAL_SCALE
        rows.append(row)
        rhs.append(0.0)

    z, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    controls = z[: horizon * n_u].reshape(horizon, n_u)
    knots = z[horizon * n_u :].reshape(horizon, n_x)
    return controls, knots


class TestTransitionDefects:
    """Defects are real: zero on the dynamics, analytic for a perturbed knot."""

    def test_defects_vanish_on_consistent_trajectory(self) -> None:
        problem = _problem(horizon=5)
        controls = np.array([[0.3], [-0.2], [0.1], [0.0], [0.5]])
        defects = compute_window_transition_defects(
            problem, _rollout(controls), controls
        )
        assert defects.shape == (5, 2)
        np.testing.assert_allclose(defects, 0.0, atol=1e-14)

    def test_perturbed_knot_gives_analytic_defects(self) -> None:
        """A perturbation delta at knot j gives d_{j-1} = delta and d_j = -A delta."""
        problem = _problem(horizon=5)
        controls = np.array([[0.3], [-0.2], [0.1], [0.0], [0.5]])
        knots = _rollout(controls)
        delta = np.array([0.01, -0.03])
        knots[2] += delta  # knot x_3
        defects = compute_window_transition_defects(problem, knots, controls)
        np.testing.assert_allclose(defects[2], delta, atol=1e-14)
        np.testing.assert_allclose(defects[3], -A_MAT @ delta, atol=1e-14)
        for k in (0, 1, 4):
            np.testing.assert_allclose(defects[k], 0.0, atol=1e-14)

    def test_residual_contains_weighted_defects(self) -> None:
        problem = _problem(horizon=3, transition_weight=25.0, control_rate_weight=0.0)
        controls = np.array([[0.1], [0.2], [0.3]])
        knots = _rollout(controls)
        consistent = _build_residuals_evaluator(problem)(
            pack_window_decision(controls, knots)
        )
        knots[0] += np.array([0.0, 0.04])
        inconsistent = _build_residuals_evaluator(problem)(
            pack_window_decision(controls, knots)
        )
        changed = inconsistent - consistent
        # Layout: [observations (none here), defects d_0..d_{N-1}, effort].
        # sqrt(25) * delta at d_0 and sqrt(25) * (-A delta) at d_1, nothing else.
        delta = np.array([0.0, 0.04])
        expected = np.zeros_like(changed)
        expected[0:2] = 5.0 * delta
        expected[2:4] = -5.0 * A_MAT @ delta
        np.testing.assert_allclose(changed, expected, atol=1e-12)

    def test_pack_unpack_roundtrip(self) -> None:
        problem = _problem(horizon=4)
        controls = np.arange(4.0).reshape(4, 1)
        knots = np.arange(8.0).reshape(4, 2) * 0.1
        u, x = unpack_window_decision(problem, pack_window_decision(controls, knots))
        np.testing.assert_array_equal(u, controls)
        np.testing.assert_array_equal(x, knots)


class TestFixedResidualSizeEveryPath:
    """Residual length equals expected_residual_size on every path."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"control_rate_weight": 0.0},
            {"actuator_bounds": (-0.5, 0.5)},
            {"defect_mode": DefectMode.EXACT},
        ],
    )
    def test_length_invariant(self, kwargs: dict[str, object]) -> None:
        targets = [np.array([0.2]), np.array([0.1]), np.array([0.0]), np.array([0.1])]
        problem = _problem(horizon=3, targets=targets, **kwargs)
        n = expected_residual_size(problem)
        # 4 observations + 3*2 defects + 3 effort (+ 2 rate) (+ 3 bounds)
        expected = 4 + 6 + 3
        if problem.control_rate_weight > 0.0:
            expected += 2
        if problem.actuator_bounds is not None:
            expected += 3
        assert n == expected
        fn = _build_residuals_evaluator(problem)
        controls = np.array([[0.1], [5.0], [-5.0]])
        consistent = pack_window_decision(controls, _rollout(controls))
        rng = np.random.default_rng(0)
        inconsistent = consistent + rng.normal(0.0, 0.1, size=consistent.shape)
        assert fn(consistent).shape == (n,)
        assert fn(inconsistent).shape == (n,)

    def test_wrong_decision_length_rejected(self) -> None:
        problem = _problem(horizon=3)
        with pytest.raises(PreconditionError, match="decision vector"):
            _build_residuals_evaluator(problem)(np.zeros(4))


class TestSolverMatchesAnalyticOptimum:
    """The solver reproduces the exact linear least-squares optimum."""

    def test_inconsistent_guess_converges_to_zero_defects(self) -> None:
        true_u = np.full((6, 1), 0.4)
        true_knots = _rollout(true_u)
        targets = [X0[:1]] + [x[:1] for x in true_knots]
        problem = _problem(
            horizon=6,
            targets=targets,
            defect_mode=DefectMode.SOFT,
            control_rate_weight=0.0,
        )
        rng = np.random.default_rng(3)
        guess_knots = true_knots + rng.normal(0.0, 0.05, size=true_knots.shape)
        guess_u = np.zeros_like(true_u)
        initial_defects = compute_window_transition_defects(
            problem, guess_knots, guess_u
        )
        assert np.max(np.abs(initial_defects)) > 1e-2

        result = solve_dime_dynamics_window(
            problem, initial_controls=guess_u, initial_knot_states=guess_knots
        )
        assert result.success
        assert np.max(np.abs(result.transition_defects)) < 1e-6
        # The exact optimum includes the effort regulariser, which biases the
        # weakly observable final control (it moves q_N only by c dt^2).
        exp_u, exp_knots = _analytic_optimum(
            6, targets, w_obs=1.0, w_def=problem.transition_weight
        )
        np.testing.assert_allclose(result.controls, exp_u, atol=1e-6)
        recovered = np.array([np.concatenate([s.q, s.v]) for s in result.states[1:]])
        np.testing.assert_allclose(recovered, exp_knots, atol=1e-6)
        np.testing.assert_allclose(result.controls[:4], true_u[:4], atol=1e-4)

    @pytest.mark.parametrize("w_obs", [1.0, 100.0])
    def test_measurement_weight_moves_optimum_as_predicted(self, w_obs: float) -> None:
        horizon = 4
        targets = [
            X0[:1],
            np.array([0.4]),
            np.array([-0.3]),
            np.array([0.5]),
            np.array([0.0]),
        ]
        w_def = 10.0
        problem = _problem(
            horizon=horizon,
            targets=targets,
            defect_mode=DefectMode.SOFT,
            control_rate_weight=0.0,
            observation_weight=w_obs,
            transition_weight=w_def,
        )
        exp_u, exp_knots = _analytic_optimum(horizon, targets, w_obs, w_def)
        result = solve_dime_dynamics_window(problem)
        assert result.success
        np.testing.assert_allclose(result.controls, exp_u, rtol=1e-4, atol=1e-6)
        knots = np.array([np.concatenate([s.q, s.v]) for s in result.states[1:]])
        np.testing.assert_allclose(knots, exp_knots, rtol=1e-4, atol=1e-6)

        # Real, nonzero measured costs computed on the returned trajectory.
        exp_defects = compute_window_transition_defects(problem, exp_knots, exp_u)
        costs = result.cost_breakdown
        assert costs["transition_cost"] == pytest.approx(
            0.5 * w_def * float(np.sum(exp_defects**2)), rel=1e-3
        )
        assert costs["transition_cost"] > 0.0
        assert costs["observation_cost"] > 0.0
        terms = {k: v for k, v in costs.items() if k != "total_cost"}
        assert costs["total_cost"] == pytest.approx(sum(terms.values()))
        # The reported total is the objective the solver minimised, 0.5 |r|^2.
        z = pack_window_decision(result.controls, knots)
        residual = _build_residuals_evaluator(problem)(z)
        assert costs["total_cost"] == pytest.approx(0.5 * float(residual @ residual))

    def test_heavier_measurement_weight_reduces_tracking_error(self) -> None:
        targets = [
            X0[:1],
            np.array([0.4]),
            np.array([-0.3]),
            np.array([0.5]),
            np.array([0.0]),
        ]

        def tracking_error(w_obs: float) -> float:
            problem = _problem(
                horizon=4,
                targets=targets,
                defect_mode=DefectMode.SOFT,
                control_rate_weight=0.0,
                observation_weight=w_obs,
                transition_weight=10.0,
            )
            res = solve_dime_dynamics_window(problem)
            return float(
                sum(
                    (s.q[0] - y[0]) ** 2
                    for s, y in zip(res.states, targets, strict=True)
                )
            )

        assert tracking_error(100.0) < 0.5 * tracking_error(1.0)

    def test_exact_mode_returns_dynamically_feasible_trajectory(self) -> None:
        targets = [X0[:1], np.array([0.4]), np.array([-0.3]), np.array([0.5])]
        problem = _problem(horizon=3, targets=targets, defect_mode=DefectMode.EXACT)
        result = solve_dime_dynamics_window(problem)
        assert result.success
        knots = np.array([np.concatenate([s.q, s.v]) for s in result.states[1:]])
        np.testing.assert_allclose(knots, _rollout(result.controls), atol=1e-12)
        np.testing.assert_allclose(result.transition_defects, 0.0, atol=1e-12)
        assert result.cost_breakdown["transition_cost"] == 0.0


class TestDiscrepancySlack:
    """SOFT defects with declared bounds are model-discrepancy slack."""

    def _problem(self, max_slack: float) -> DimeDynamicsWindowProblem:
        targets = [X0[:1], np.array([0.4]), np.array([-0.3]), np.array([0.5])]
        return _problem(
            horizon=3,
            targets=targets,
            defect_mode=DefectMode.SOFT,
            control_rate_weight=0.0,
            discrepancy_bounds=ModelDiscrepancyBounds(
                max_slack_norm=max_slack, slack_weight=10.0
            ),
        )

    def test_slack_costed_as_discrepancy(self) -> None:
        result = solve_dime_dynamics_window(self._problem(max_slack=10.0))
        assert result.success
        costs = result.cost_breakdown
        expected = 0.5 * 10.0 * float(np.sum(result.transition_defects**2))
        assert costs["discrepancy_cost"] == pytest.approx(expected)
        assert costs["discrepancy_cost"] > 0.0
        assert costs["transition_cost"] == 0.0

    def test_slack_beyond_bound_fails_closed(self) -> None:
        result = solve_dime_dynamics_window(self._problem(max_slack=1e-6))
        assert not result.success
        assert result.status == "failed_discrepancy_bound"


class TestPreconditions:
    def test_manifold_state_rejected(self) -> None:
        provider = DeterministicFakeProvider(n_q=7, n_v=6)
        state = DimeCompleteState(
            t=0.0,
            q=np.zeros(7),
            v=np.zeros(6),
            units=UNITS,
            model_hash=provider.model_hash,
        )
        with pytest.raises(PreconditionError, match="n_q == n_v"):
            DimeDynamicsWindowProblem(
                provider=provider, initial_state=state, horizon_steps=2, dt_s=0.01
            )

    def test_non_finite_targets_rejected_at_solve(self) -> None:
        problem = _problem(horizon=2, targets=[X0[:1], np.array([np.nan])])
        with pytest.raises(PreconditionError, match="finite"):
            solve_dime_dynamics_window(problem)

    def test_initial_guess_shape_checked(self) -> None:
        problem = _problem(horizon=3)
        with pytest.raises(PreconditionError, match="initial_knot_states"):
            solve_dime_dynamics_window(problem, initial_knot_states=np.zeros((2, 2)))
