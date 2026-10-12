"""Tests for the Drake gradient-free fit driver (issue #4115).

Layered so the optimizer wiring is exercised in *every* environment via
a deterministic stub ``simulate_fn``, while the live Drake forward sim
is only required for tests marked ``@pytest.mark.requires_drake``.

Coverage:

1. **Bounds contract:** :func:`polynomial_parameter_bounds` returns the
   canonical ``|A,B|≤1000``, ``|C,D|≤500``, ``|E,F|≤100``, ``|G|≤25``
   per-coefficient absolute bounds.
2. **Optimizer wiring:** with a stub ``simulate_fn`` that emits a
   linear-in-theta forward map, ``fit_swing_drake`` reduces the cost
   monotonically to within tolerance and returns a canonical
   :class:`FitResult` schema.
3. **TDD oracle recovery (stub):** synthesize a target from a known
   ``theta_truth``, pass it back through the fit driver with the same
   stub, and assert ``theta_optimal`` is recovered to within 10% per
   coefficient.
4. **Cost adapter:** :func:`compute_cost_drake` agrees numerically with
   the shared :func:`compute_cost` (no Drake-specific drift).
5. **Live Drake recovery (requires_drake):** synthesize via
   :func:`simulate_with_coefficients`, fit, recover ``theta_truth`` to
   within 10% with the spec-mandated max-iter cap of 50.

Per CLAUDE.md, no ``sys.modules["pydrake"] = MagicMock()`` at module
scope — the live test simply skips when ``pydrake`` is absent.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.drake.python.motion_matching.compute_cost_drake import (
    compute_cost_drake,
    drake_simout_to_shared,
)
from src.engines.physics_engines.drake.python.motion_matching.fit_swing import (
    FitOptions,
    FitResult,
    fit_swing_drake,
    polynomial_parameter_bounds,
)
from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    SimOut,
)
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.final_cost import (
    CostOptions,
    SimOutput,
    compute_cost,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provenance(**overrides: object) -> SourceProvenance:
    base = {
        "filename": "synthetic.csv",
        "format": "synthetic",
        "subject_id": "TEST",
        "trial_id": "T0",
        "sha256": "0" * 64,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return SourceProvenance(**base)  # type: ignore[arg-type]


def _make_simout(
    time: np.ndarray, butt: np.ndarray, clubhead: np.ndarray, club_quat: np.ndarray
) -> SimOut:
    """Build a minimal canonical Drake :class:`SimOut`.

    The ``q``/``qd``/``qdd``/``tau`` arrays are zero-filled; the cost
    reads only ``grip``/``clubhead``/``club_quat`` for our test
    regularizer choice (``coeff_l2``).
    """
    n = time.shape[0]
    n_joints = 1
    return SimOut(
        time=time,
        q=np.zeros((n, n_joints)),
        qd=np.zeros((n, n_joints)),
        qdd=np.zeros((n, n_joints)),
        tau=np.zeros((n, n_joints)),
        grip=butt,
        grip_quat=club_quat.copy(),
        clubhead=clubhead,
        club_quat=club_quat,
        solver_status="success",
        duration_s=0.0,
    )


def _stub_simulate(theta: np.ndarray, target: ClubTarget) -> SimOut:
    """Deterministic stub forward sim.

    The stub treats the first three components of ``theta`` as a 3-vector
    additive offset of the **target** butt+clubhead positions, so the
    optimum is at ``theta[:3] = 0`` (recovers the target byte-for-byte).
    All remaining components are ignored. This gives us a smooth, convex
    optimization landscape we can solve in a handful of SLSQP iters
    without invoking real Drake.
    """
    offset = theta[:3].reshape(1, 3)
    butt = target.butt + offset
    clubhead = target.clubhead + offset
    return _make_simout(target.time, butt, clubhead, target.club_quat.copy())


def _make_target(n: int = 16, impact_idx: int = 8) -> ClubTarget:
    """Build a small valid synthetic :class:`ClubTarget`."""
    time = np.linspace(0.0, (n - 1) * 1e-3, n)
    butt = np.zeros((n, 3))
    clubhead = np.zeros((n, 3))
    butt[:, 0] = 0.5 * np.sin(np.linspace(0, np.pi, n))
    clubhead[:, 0] = 1.5 * np.sin(np.linspace(0, np.pi, n))
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=impact_idx,
        source=_make_provenance(),
    )


# ---------------------------------------------------------------------------
# 1. Bounds contract
# ---------------------------------------------------------------------------


class TestPolynomialParameterBounds:
    """Canonical |A,B|≤1000, |C,D|≤500, |E,F|≤100, |G|≤25 bounds."""

    def test_one_joint(self) -> None:
        lb, ub = polynomial_parameter_bounds(1)
        assert lb.shape == (COEFFS_PER_JOINT,)
        assert ub.shape == (COEFFS_PER_JOINT,)
        np.testing.assert_array_equal(
            ub, np.array([1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0])
        )
        np.testing.assert_array_equal(lb, -ub)

    def test_many_joints(self) -> None:
        n_joints = 23  # canonical Drake humanoid
        lb, ub = polynomial_parameter_bounds(n_joints)
        assert lb.shape == (n_joints * COEFFS_PER_JOINT,)
        # Each block of 7 must equal the canonical per-coeff bound.
        block = ub.reshape(n_joints, COEFFS_PER_JOINT)
        for row in block:
            np.testing.assert_array_equal(
                row, np.array([1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0])
            )

    @pytest.mark.parametrize("bad", [0, -1, -23])
    def test_rejects_nonpositive(self, bad: int) -> None:
        with pytest.raises(ValueError):
            polynomial_parameter_bounds(bad)


# ---------------------------------------------------------------------------
# 2. Optimizer wiring + cost monotone-decrease
# ---------------------------------------------------------------------------


class TestFitSwingWiring:
    """Optimizer-side wiring tests using the deterministic stub sim."""

    def test_returns_canonical_fit_result(self) -> None:
        target = _make_target()
        theta0 = np.zeros(1 * COEFFS_PER_JOINT)
        theta0[:3] = [0.05, -0.03, 0.04]  # small offset; should drive to 0
        opts = FitOptions(n_joints=1, theta0=theta0, max_iterations=50)
        result = fit_swing_drake(
            target,
            options=opts,
            simulate_fn=lambda th: _stub_simulate(th, target),
        )
        assert isinstance(result, FitResult)
        assert result.theta_optimal.shape == (COEFFS_PER_JOINT,)
        assert np.isfinite(result.final_cost)
        assert np.isfinite(result.final_rmse_m)
        assert result.solver_status in {"success", "warning"}
        assert result.iterations >= 0
        assert result.n_evaluations == len(result.history)
        assert result.wall_clock_s >= 0.0

    def test_cost_monotone_decreases(self) -> None:
        """The min(history) must drop strictly below the first sample."""
        target = _make_target()
        theta0 = np.zeros(1 * COEFFS_PER_JOINT)
        theta0[:3] = [0.10, 0.10, 0.10]
        opts = FitOptions(n_joints=1, theta0=theta0, max_iterations=50)
        result = fit_swing_drake(
            target,
            options=opts,
            simulate_fn=lambda th: _stub_simulate(th, target),
        )
        assert len(result.history) >= 2
        assert min(result.history) < result.history[0]
        assert result.final_cost <= result.history[0]

    def test_recovers_target_when_offset_is_decision_var(self) -> None:
        """TDD oracle recovery (stub): theta[:3] should drive to ~0."""
        target = _make_target()
        theta_truth = np.zeros(1 * COEFFS_PER_JOINT)  # zero offset = optimum
        theta0 = theta_truth.copy()
        theta0[:3] = [0.20, -0.15, 0.10]
        opts = FitOptions(
            n_joints=1, theta0=theta0, max_iterations=200, tolerance=1e-10
        )
        result = fit_swing_drake(
            target,
            options=opts,
            simulate_fn=lambda th: _stub_simulate(th, target),
        )
        # The first three coefficients are the only ones that affect
        # the cost in this stub; recover them to < 0.01 m.
        np.testing.assert_allclose(result.theta_optimal[:3], theta_truth[:3], atol=1e-2)
        # Final RMSE should be tiny (< 5 mm) — the spec gate.
        assert result.final_rmse_m < 5e-3

    @pytest.mark.parametrize("rng_seed", [42, 1337, 999])
    def test_same_rng_seed_repeats_exact_fit_result(self, rng_seed: int) -> None:
        """Same target, warm start, and rng_seed reproduce optimizer outputs."""
        target = _make_target()
        rng = np.random.default_rng(rng_seed)
        theta0 = np.zeros(1 * COEFFS_PER_JOINT)
        theta0[:3] = rng.uniform(-0.2, 0.2, size=3)
        opts = FitOptions(
            n_joints=1,
            theta0=theta0,
            rng_seed=rng_seed,
            max_iterations=50,
            tolerance=1e-10,
        )

        first = fit_swing_drake(
            target,
            options=opts,
            simulate_fn=lambda th: _stub_simulate(th, target),
        )
        second = fit_swing_drake(
            target,
            options=opts,
            simulate_fn=lambda th: _stub_simulate(th, target),
        )

        np.testing.assert_allclose(
            first.theta_optimal, second.theta_optimal, rtol=0.0, atol=1e-15
        )
        assert first.history == second.history
        assert first.final_cost == second.final_cost
        assert first.final_rmse_m == second.final_rmse_m
        assert first.iterations == second.iterations
        assert first.n_evaluations == second.n_evaluations
        assert first.solver_status == second.solver_status


class TestFitSwingValidation:
    """Argument validation."""

    def test_rejects_bad_n_joints(self) -> None:
        with pytest.raises(ValueError):
            fit_swing_drake(_make_target(), FitOptions(n_joints=0))

    def test_rejects_wrong_theta0_length(self) -> None:
        target = _make_target()
        with pytest.raises(ValueError):
            fit_swing_drake(
                target,
                FitOptions(n_joints=1, theta0=np.zeros(3)),
                simulate_fn=lambda th: _stub_simulate(th, target),
            )


# ---------------------------------------------------------------------------
# 3. Cost adapter agreement with shared cost
# ---------------------------------------------------------------------------


class TestComputeCostDrake:
    """The Drake adapter must agree numerically with the shared cost."""

    def test_drake_simout_to_shared_field_mapping(self) -> None:
        target = _make_target()
        sim_out = _stub_simulate(np.zeros(7), target)
        shared = drake_simout_to_shared(sim_out)
        assert isinstance(shared, SimOutput)
        np.testing.assert_array_equal(shared.butt, sim_out.grip)
        np.testing.assert_array_equal(shared.clubhead, sim_out.clubhead)
        np.testing.assert_array_equal(shared.club_quat, sim_out.club_quat)
        np.testing.assert_array_equal(shared.tau, sim_out.tau)
        np.testing.assert_array_equal(shared.omega, sim_out.qd)

    def test_adapter_agrees_with_shared(self) -> None:
        target = _make_target()
        theta = np.zeros(1 * COEFFS_PER_JOINT)
        theta[:3] = [0.01, -0.02, 0.03]
        cost_opts = CostOptions(regularizer="coeff_l2", lambda_=0.0)

        def drake_sim(th: np.ndarray) -> SimOut:
            return _stub_simulate(th, target)

        def shared_sim(th: np.ndarray) -> SimOutput:
            return drake_simout_to_shared(drake_sim(th))

        j_drake, terms_drake = compute_cost_drake(theta, target, drake_sim, cost_opts)
        j_shared, terms_shared = compute_cost(theta, target, shared_sim, cost_opts)
        assert j_drake == pytest.approx(j_shared, rel=1e-12, abs=1e-12)
        assert terms_drake.position == pytest.approx(terms_shared.position)
        assert terms_drake.orientation == pytest.approx(terms_shared.orientation)
        assert terms_drake.impact_anchor == pytest.approx(terms_shared.impact_anchor)
        assert terms_drake.regularizer == pytest.approx(terms_shared.regularizer)

    def test_constraint_residuals_add_penalty(self) -> None:
        target = _make_target()
        theta = np.zeros(1 * COEFFS_PER_JOINT)
        cost_opts = CostOptions(regularizer="coeff_l2", lambda_=0.0)

        def drake_sim(th: np.ndarray) -> SimOut:
            return _stub_simulate(th, target)

        j_no, terms_no = compute_cost_drake(theta, target, drake_sim, cost_opts)
        j_yes, terms_yes = compute_cost_drake(
            theta,
            target,
            drake_sim,
            cost_opts,
            constraint_residuals=[np.array([0.5, 1.0])],  # ||r||^2 = 1.25
        )
        assert j_yes == pytest.approx(j_no + 1.25, rel=1e-12, abs=1e-12)
        assert terms_yes.regularizer == pytest.approx(1.25, rel=1e-12, abs=1e-12)
        assert terms_yes.body_marker == terms_no.body_marker


# ---------------------------------------------------------------------------
# 5. Live Drake integration (requires_drake)
# ---------------------------------------------------------------------------


@pytest.mark.requires_drake
@pytest.mark.integration
@pytest.mark.slow
class TestFitSwingDrakeLive:
    """Drives a real :func:`simulate_with_coefficients` end-to-end.

    Skipped automatically (via the ``requires_drake`` marker filter) on
    machines without ``pydrake``. Per the spec, < 5 minutes per swing
    is the wall-clock budget; we use a 50-iteration cap to keep CI
    runtime bounded.
    """

    def test_recovers_synthetic_swing(self) -> None:
        pytest.importorskip("pydrake")

        from src.engines.physics_engines.drake.python.motion_matching.simulate import (
            SimOptions,
            simulate_with_coefficients,
        )

        # Build a small theta_truth and synthesize the target from it.
        rng = np.random.default_rng(0)
        n_joints = 1  # smallest non-trivial case
        theta_truth = rng.uniform(-1.0, 1.0, size=n_joints * COEFFS_PER_JOINT)
        sim_opts = SimOptions(simulation_time_s=0.1, sample_rate_hz=200.0)
        truth_out = simulate_with_coefficients(theta_truth, sim_opts)

        target = ClubTarget(
            time=truth_out.time,
            butt=truth_out.grip,
            clubhead=truth_out.clubhead,
            club_quat=truth_out.club_quat,
            impact_idx=int(truth_out.time.size // 2),
            source=_make_provenance(format="synthetic"),
        )

        opts = FitOptions(
            n_joints=n_joints,
            theta0=theta_truth + 0.05 * rng.standard_normal(theta_truth.shape),
            max_iterations=50,
            tolerance=1e-6,
            sim_options=sim_opts,
        )
        result = fit_swing_drake(target, options=opts)

        # 10% per-coefficient recovery (issue #4115 acceptance gate).
        rel = np.abs(result.theta_optimal - theta_truth) / np.maximum(
            np.abs(theta_truth), 1e-3
        )
        assert np.all(rel < 0.10), f"max rel err = {rel.max():.3f}"
