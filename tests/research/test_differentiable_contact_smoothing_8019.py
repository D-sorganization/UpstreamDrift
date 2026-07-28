"""Regression tests for issue #8019.

Three defects, all silent:

1. ``contact_smoothing_multiplier`` had no effect in the default
   ``contact_method="smoothed"`` configuration - ``compute_gradient`` fell
   straight through to the parent finite-difference gradient and never read
   ``smoothing_factor``. Cost and gradient norm were bit-identical across a
   1000x sweep of the multiplier.
2. ``_apply_phase_smoothing`` was gated on ``any(schedule)``: a single global
   scalar, so one contact step and fifty were treated identically.
3. ``DifferentiableEngine.backend`` was a dead knob - ``_backend`` occurred
   exactly once in the module, so ``backend="jax"`` was accepted and silently
   produced finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.research.differentiable.engine import (
    AutodiffBackend,
    ContactDifferentiableEngine,
    DifferentiableEngine,
)

pytestmark = pytest.mark.unit

_SWITCH_THRESHOLD = 0.5


class SwitchingContactPlant:
    """2-DOF plant whose vertical contact engages discontinuously.

    Above ``_SWITCH_THRESHOLD`` a fixed impulse is injected, so the loss is
    piecewise constant in that direction: a 1e-6 finite difference returns
    exactly zero and only a smoothing step wide enough to straddle the switch
    recovers a descent direction.
    """

    n_q = 2
    n_v = 2

    def __init__(self) -> None:
        self._q = np.zeros(2)
        self._v = np.zeros(2)
        self._tau = np.zeros(2)

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=float).copy()

    def set_joint_velocities(self, v: np.ndarray) -> None:
        self._v = np.asarray(v, dtype=float).copy()

    def set_joint_torques(self, tau: np.ndarray) -> None:
        self._tau = np.asarray(tau, dtype=float).copy()

    def step(self, dt: float) -> None:
        impulse = 1.0 if self._tau[1] > _SWITCH_THRESHOLD else 0.0
        self._v = self._v + np.array([self._tau[0], impulse]) * dt
        self._q = self._q + self._v * dt

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._v.copy()


_X0 = np.zeros(4)
_GOAL = np.array([0.3, 0.02, 0.0, 0.0])


def _loss(trajectory: np.ndarray) -> float:
    diff = trajectory[-1] - _GOAL
    return float(np.vdot(diff, diff))


class TestSmoothingIsNotInert:
    def test_smoothing_factor_changes_the_gradient(self) -> None:
        """Default 'smoothed' mode must read smoothing_factor."""
        controls = np.zeros((6, 2))
        tight = ContactDifferentiableEngine(
            SwitchingContactPlant(), smoothing_factor=0.01
        ).compute_gradient(_X0, controls, _loss, 0.02)
        wide = ContactDifferentiableEngine(
            SwitchingContactPlant(), smoothing_factor=1.0
        ).compute_gradient(_X0, controls, _loss, 0.02)

        assert np.max(np.abs(tight[:, 1])) == 0.0
        assert np.max(np.abs(wide[:, 1])) > 0.0

    def test_zero_smoothing_matches_the_unsmoothed_gradient(self) -> None:
        """Smoothing must collapse to the plain gradient at sigma = 0."""
        controls = np.full((5, 2), 0.2)
        smoothed = ContactDifferentiableEngine(
            SwitchingContactPlant(), smoothing_factor=0.0
        ).compute_gradient(_X0, controls, _loss, 0.02)
        plain = DifferentiableEngine(SwitchingContactPlant()).compute_gradient(
            _X0, controls, _loss, 0.02
        )
        np.testing.assert_allclose(smoothed, plain)

    def test_smoothed_mode_is_deterministic(self) -> None:
        """Unlike 'randomized', repeated calls must agree exactly."""
        controls = np.full((4, 2), 0.3)
        engine = ContactDifferentiableEngine(
            SwitchingContactPlant(), smoothing_factor=1.0
        )
        first = engine.compute_gradient(_X0, controls, _loss, 0.02)
        second = engine.compute_gradient(_X0, controls, _loss, 0.02)
        np.testing.assert_array_equal(first, second)

    def test_multiplier_changes_optimize_through_contact(self) -> None:
        """Cost/gradient used to be bit-identical across the whole sweep."""
        schedule = [False, False, False, True, True, False, False, False]
        norms = []
        for multiplier in (1.0, 20.0):
            engine = ContactDifferentiableEngine(
                SwitchingContactPlant(), smoothing_factor=0.1
            )
            result = engine.optimize_through_contact(
                _X0,
                _GOAL,
                schedule,
                horizon=8,
                dt=0.02,
                contact_smoothing_multiplier=multiplier,
            )
            norms.append(result.gradient_norm)
        assert norms[0] != norms[1]


class TestSmoothingIsPhaseAware:
    def test_schedule_is_per_timestep(self) -> None:
        engine = ContactDifferentiableEngine(SwitchingContactPlant())
        engine._apply_phase_smoothing([True, False, True, False], 0.01, 5.0)
        np.testing.assert_allclose(engine.smoothing_schedule, [0.05, 0.01, 0.05, 0.01])

    def test_one_contact_step_differs_from_five(self) -> None:
        """``any(schedule)`` treated these identically."""
        engine = ContactDifferentiableEngine(SwitchingContactPlant())
        engine._apply_phase_smoothing([False] * 7 + [True], 0.01, 5.0)
        one = np.asarray(engine.smoothing_schedule).copy()
        engine._apply_phase_smoothing([True] * 5 + [False] * 3, 0.01, 5.0)
        five = np.asarray(engine.smoothing_schedule).copy()
        assert not np.array_equal(one, five)

    def test_schedule_is_cleared_after_optimization(self) -> None:
        engine = ContactDifferentiableEngine(SwitchingContactPlant())
        engine.optimize_through_contact(
            _X0, _GOAL, [False, True, False, False], horizon=4, dt=0.02
        )
        assert engine.smoothing_schedule is None
        assert engine.smoothing_factor == pytest.approx(0.01)


class TestBackendKnob:
    @pytest.mark.parametrize("backend", ["jax", "torch"])
    def test_unimplemented_backend_raises(self, backend: str) -> None:
        """backend='jax' used to be accepted and silently ignored."""
        with pytest.raises(NotImplementedError, match=backend):
            DifferentiableEngine(SwitchingContactPlant(), backend=backend)

    def test_numpy_backend_is_exposed(self) -> None:
        engine = DifferentiableEngine(SwitchingContactPlant(), backend="numpy")
        assert engine.backend is AutodiffBackend.NUMPY

    def test_contact_engine_forwards_backend(self) -> None:
        with pytest.raises(NotImplementedError):
            ContactDifferentiableEngine(SwitchingContactPlant(), backend="jax")


class TestContactMethodValidation:
    def test_unknown_contact_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="contact_method"):
            ContactDifferentiableEngine(SwitchingContactPlant(), contact_method="bogus")

    def test_negative_smoothing_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="smoothing_factor"):
            ContactDifferentiableEngine(SwitchingContactPlant(), smoothing_factor=-1.0)
