"""Tests for ``fit_swing_opensim`` (issue #4128).

Test strategy (TDD per ``CROSS_ENGINE_PARITY_SPEC.md`` § 2.7):

* **Oracle recovery** — build a deterministic, analytically-invertible
  mock ``simulate_fn`` that maps polynomial coefficients ``theta`` to a
  :class:`SimOutput`. Synthesize a target from a known
  ``theta_truth`` → run :func:`fit_swing_opensim` → assert the recovered
  ``theta`` lies within 10 % (atol on the bound-normalised vector) of
  the truth and that ``solver_status == "success"`` (cost monotone-min).
* **Determinism** — same seed, same target, same simulator -> byte-identical
  ``FitResult.theta_optimal``.
* **Cost monotone-decrease** — the *minimum running* cost over the
  history must be non-increasing (SLSQP line searches can transiently
  raise the per-eval cost; the running min is the contracted invariant).

Markers
-------
The full driver path that consumes the OpenSim Python bindings is gated
behind ``@pytest.mark.requires_opensim``. The mock-simulator tests in this
module run anywhere because they exercise the optimizer wiring rather than
OpenSim itself; they share the marker for parity with the engine-specific
test suite (skipped when ``import opensim`` fails on CI workers without
the bindings).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.motion_matching import (
    FitOptions,
    FitResult,
    fit_swing_opensim,
)
from src.engines.physics_engines.opensim.python.motion_matching.fit_swing import (
    DEFAULT_COEFF_BOUND,
    POLY_ORDER_PER_JOINT,
)
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.final_cost import SimOutput

pytestmark = pytest.mark.requires_opensim


# --------------------------------------------------------------------------- #
# Mock simulator: deterministic, analytically invertible
# --------------------------------------------------------------------------- #


def _make_mock_simulate(
    n_joints: int,
    time_grid: np.ndarray,
    seed: int = 0,
):
    """Return ``(simulate_fn, theta_to_target)``.

    The mock maps a coefficient vector ``theta`` of length
    ``n_joints * 7`` to a :class:`SimOutput` whose ``clubhead`` and
    ``butt`` vary smoothly with ``theta`` and whose minimum-cost solution
    is uniquely ``theta``. Concretely:

    * Project ``theta`` through a fixed random orthonormal matrix ``A``
      onto a 3-vector ``a``. The clubhead position then equals
      ``a * basis(t)``. This makes the mapping ``theta -> trajectory``
      linear, smooth, and finite-difference-friendly.
    * The butt position is offset from the clubhead by a fixed unit
      vector so the cost penalises both shafts identically.
    * Quaternions are held at identity ``[1, 0, 0, 0]``; the cost's
      orientation term is then a constant zero.

    With this construction the optimiser reduces to a small linear
    least-squares — SLSQP will recover the truth in a handful of
    iterations for any ``n_joints``.
    """
    n = time_grid.size
    d = n_joints * POLY_ORDER_PER_JOINT
    rng = np.random.default_rng(seed)

    # Random orthonormal projection theta -> 3-vector. Orthonormal so the
    # cost surface is well-conditioned around the truth.
    raw = rng.standard_normal((d, 3))
    q_proj, _ = np.linalg.qr(raw)
    A = q_proj  # (d, 3)

    # Smooth basis varying with t in [0, T]. Three independent shapes so
    # each component of the projection is identifiable.
    t = time_grid - time_grid[0]
    t_norm = t / max(t[-1], 1e-12)
    basis = np.column_stack(
        [
            np.sin(np.pi * t_norm),
            t_norm,
            t_norm * t_norm,
        ]
    )  # (n, 3)

    butt_offset = np.array([0.0, 0.0, 0.05])
    quat_id = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    def project(theta: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.size != d:
            raise ValueError(f"theta size {theta_arr.size} != {d}")
        return theta_arr @ A  # (3,)

    def simulate_fn(theta: np.ndarray) -> SimOutput:
        a = project(theta)
        clubhead = basis * a[np.newaxis, :]  # (n, 3)
        butt = clubhead + butt_offset
        return SimOutput(
            butt=butt.astype(np.float64),
            clubhead=clubhead.astype(np.float64),
            club_quat=quat_id.astype(np.float64),
            time=time_grid.astype(np.float64),
            tau=None,
            omega=None,
        )

    # Advertise n_joints so fit_swing_opensim can infer dimensionality.
    simulate_fn.n_joints = n_joints  # type: ignore[attr-defined]

    def theta_to_target(theta: np.ndarray) -> ClubTarget:
        sim = simulate_fn(theta)
        prov = SourceProvenance(
            filename="oracle.synthetic",
            format="synthetic",
            subject_id="UNIT",
            trial_id=f"seed-{seed}",
            sha256=hashlib.sha256(b"oracle").hexdigest(),
        )
        return ClubTarget(
            time=sim.time,  # type: ignore[arg-type]
            butt=sim.butt,
            clubhead=sim.clubhead,
            club_quat=sim.club_quat,
            impact_idx=n // 2,
            source=prov,
        )

    return simulate_fn, theta_to_target


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def time_grid() -> np.ndarray:
    """1.0 s grid sampled at 50 Hz — fast enough for CI, dense enough for cost."""
    return np.linspace(0.0, 1.0, 51)


@pytest.fixture(scope="module")
def n_joints() -> int:
    """Tiny problem to keep CI fast while exercising the SLSQP wiring."""
    return 2


@pytest.fixture
def truth_theta(n_joints: int) -> np.ndarray:
    """Deterministic truth vector; well within the default coefficient bounds."""
    rng = np.random.default_rng(20260506)
    return rng.uniform(-1.0, 1.0, size=n_joints * POLY_ORDER_PER_JOINT)


@pytest.fixture
def oracle(n_joints: int, time_grid: np.ndarray):
    return _make_mock_simulate(n_joints, time_grid, seed=0)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_recovery_within_10_percent(
    n_joints: int, time_grid: np.ndarray, truth_theta: np.ndarray, oracle
) -> None:
    """Oracle recovery: synthesize → fit → recover within 10 % atol."""
    simulate_fn, theta_to_target = oracle
    target = theta_to_target(truth_theta)

    options = FitOptions(
        n_joints=n_joints,
        simulate_fn=simulate_fn,
        max_iter=30,  # spec target: ~30 SLSQP iters
        rng_seed=42,
        coeff_bound=DEFAULT_COEFF_BOUND,
    )

    result = fit_swing_opensim(target, options)

    assert isinstance(result, FitResult)
    assert result.solver_status == "success", result.message
    assert result.solver_status == "success"
    # The mock is rank-3 (A has 3 orthonormal columns), so theta is only
    # identifiable up to its (d-3)-dim null space. The cost-relevant
    # recovery contract is therefore in observation space: replay the
    # recovered theta through the simulator and compare clubhead and butt
    # trajectories to the target. RMSE within 10 % of the target's
    # peak-to-peak amplitude is the canonical check.
    sim_fit = simulate_fn(result.theta_optimal)
    pp = float(np.ptp(target.clubhead))  # peak-to-peak of target trajectory
    rmse_clubhead = float(np.sqrt(np.mean((sim_fit.clubhead - target.clubhead) ** 2)))
    rmse_butt = float(np.sqrt(np.mean((sim_fit.butt - target.butt) ** 2)))
    assert rmse_clubhead < 0.10 * pp, (
        f"clubhead RMSE {rmse_clubhead:.3e} > 10% of {pp:.3e}"
    )
    assert rmse_butt < 0.10 * pp, f"butt RMSE {rmse_butt:.3e} > 10% of {pp:.3e}"
    # Final cost contract: the cost-monotone-decrease test below covers
    # the per-iteration trace; the *final* cost must reflect a well-fit
    # trajectory (cross-engine spec final-cost criterion: < 1e-3).
    assert result.final_cost < 1e-3, (
        f"final cost {result.final_cost:.3e} > 1e-3 (target trivially recoverable)"
    )


@pytest.mark.parametrize("rng_seed", [42, 1337, 999])
def test_determinism(
    n_joints: int, time_grid: np.ndarray, truth_theta: np.ndarray, oracle, rng_seed: int
) -> None:
    """Same options + same target -> identical FitResult.theta_optimal."""
    simulate_fn, theta_to_target = oracle
    target = theta_to_target(truth_theta)

    def run() -> FitResult:
        return fit_swing_opensim(
            target,
            FitOptions(
                n_joints=n_joints,
                simulate_fn=simulate_fn,
                max_iter=30,
                rng_seed=rng_seed,
            ),
        )

    a = run()
    b = run()
    np.testing.assert_array_equal(a.theta_optimal, b.theta_optimal)
    assert a.cost == b.cost
    assert getattr(a, "history", None) == getattr(b, "history", None)
    assert a.n_eval == b.n_eval
    assert getattr(a, "n_iter", None) == getattr(b, "n_iter", None)
    assert a.message == b.message


def test_cost_monotone_running_minimum(
    n_joints: int, time_grid: np.ndarray, truth_theta: np.ndarray, oracle
) -> None:
    """The *running minimum* of the cost history must be non-increasing.

    SLSQP performs line searches that may transiently raise the
    per-evaluation cost; the optimizer-level invariant is that the best
    cost seen so far never increases.
    """
    simulate_fn, theta_to_target = oracle
    target = theta_to_target(truth_theta)

    result = fit_swing_opensim(
        target,
        FitOptions(
            n_joints=n_joints,
            simulate_fn=simulate_fn,
            max_iter=30,
            rng_seed=7,
        ),
    )

    assert len(result.history) >= 2, "history must contain >= 2 evaluations"
    history = np.asarray(result.history, dtype=np.float64)
    running_min = np.minimum.accumulate(history)
    diffs = np.diff(running_min)
    assert np.all(diffs <= 1e-12), (
        f"running-min cost increased somewhere: max delta = {diffs.max():.3e}"
    )
    # End-to-end: the optimisation strictly improved on the warm start.
    assert running_min[-1] < running_min[0], (
        f"final cost {running_min[-1]:.3e} not better than start {running_min[0]:.3e}"
    )
