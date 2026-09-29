"""Regression tests for issue #7557.

``DifferentiableEngine.compute_gradient`` used a finite-difference loop that
re-simulated the *entire* trajectory from ``initial_state`` for every one of
``T * n_u`` perturbed control elements, and allocated a fresh ``controls.copy()``
each time. Because perturbing ``controls[t, i]`` only changes states from
``t + 1`` onward, the baseline prefix can be reused and only the suffix needs
re-simulation from ``baseline_traj[t]`` — for a Markovian step this yields a
*bit-identical* gradient with far fewer engine steps and no per-element copies.

These tests pin both properties:

* **parity** — the optimized gradient equals the naive full-resim central
  gradient, and
* **complexity** — the optimized path performs strictly fewer engine ``step``
  calls than the naive O(T^2 * n_u) approach (the red→green driver).
"""

from __future__ import annotations

import numpy as np
import pytest
from src.research.differentiable.engine import DifferentiableEngine


class _LinearEngine:
    """Deterministic engine whose full state is exactly ``(q, v)``.

    Semi-implicit Euler on a linear spring so the dynamics are non-trivial yet
    Markovian in ``(q, v)`` — the precondition for the suffix-reuse optimization
    to be exact. Counts ``step`` calls so tests can assert the complexity win.
    """

    n_q = 2
    n_v = 2

    def __init__(self) -> None:
        self._q = np.zeros(2)
        self._v = np.zeros(2)
        self._tau = np.zeros(2)
        self.step_calls = 0

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q = np.array(q, dtype=float)

    def set_joint_velocities(self, v: np.ndarray) -> None:
        self._v = np.array(v, dtype=float)

    def set_joint_torques(self, tau: np.ndarray) -> None:
        self._tau = np.array(tau, dtype=float)

    def step(self, dt: float) -> None:
        self.step_calls += 1
        self._v = self._v + (self._tau - 0.5 * self._q) * dt
        self._q = self._q + self._v * dt

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._v.copy()


def _loss(traj: np.ndarray) -> float:
    """Depends on the whole trajectory so the suffix genuinely matters."""
    return float(np.sum(traj[-1] ** 2) + 0.01 * np.sum(traj**2))


def _naive_gradient(
    de: DifferentiableEngine,
    x0: np.ndarray,
    controls: np.ndarray,
    dt: float,
    rel_eps: float = 1e-6,
) -> np.ndarray:
    """Full re-simulation central-difference reference."""
    T, n_u = controls.shape
    grad = np.zeros_like(controls)
    for t in range(T):
        for i in range(n_u):
            eps = rel_eps * max(1.0, abs(float(controls[t, i])))
            up = controls.copy()
            down = controls.copy()
            up[t, i] += eps
            down[t, i] -= eps
            grad[t, i] = (
                _loss(de.simulate_trajectory(x0, up, dt))
                - _loss(de.simulate_trajectory(x0, down, dt))
            ) / (2.0 * eps)
    return grad


@pytest.mark.unit
def test_gradient_matches_naive_full_resim() -> None:
    """Optimized gradient is numerically identical to the naive reference."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(4)
    controls = rng.standard_normal((6, 2))
    dt = 0.01

    de = DifferentiableEngine(_LinearEngine(), backend="numpy")
    optimized = de.compute_gradient(x0, controls, _loss, dt)

    de_ref = DifferentiableEngine(_LinearEngine(), backend="numpy")
    reference = _naive_gradient(de_ref, x0, controls, dt)

    np.testing.assert_allclose(optimized, reference, rtol=1e-9, atol=1e-12)


@pytest.mark.unit
def test_gradient_uses_fewer_engine_steps() -> None:
    """Suffix reuse must cut engine ``step`` calls below the naive O(T^2 n_u)."""
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal(4)
    controls = rng.standard_normal((6, 2))
    dt = 0.01
    T, n_u = controls.shape

    eng_opt = _LinearEngine()
    DifferentiableEngine(eng_opt, backend="numpy").compute_gradient(
        x0, controls, _loss, dt
    )

    eng_naive = _LinearEngine()
    _naive_gradient(DifferentiableEngine(eng_naive, backend="numpy"), x0, controls, dt)

    assert eng_opt.step_calls < eng_naive.step_calls
    # Naive does two T-step rollouts per element.
    assert eng_naive.step_calls == 2 * T * n_u * T
    # Optimized does T baseline + two suffix rollouts per element.
    assert eng_opt.step_calls == T + 2 * n_u * T * (T + 1) // 2
