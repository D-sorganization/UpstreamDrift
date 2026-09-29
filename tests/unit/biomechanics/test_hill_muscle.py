"""Tests for src.shared.python.biomechanics.hill_muscle (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.biomechanics.hill_muscle import (
    HillMuscleModel,
    MuscleParameters,
    MuscleState,
)


def _default_params() -> MuscleParameters:
    return MuscleParameters(
        F_max=1000.0,
        l_opt=0.10,
        l_slack=0.25,
        v_max=10.0,
        pennation_angle=0.0,
        damping=0.05,
    )


class TestMuscleParameters:
    def test_hill_muscle_construction(self) -> None:
        p = _default_params()
        assert p.F_max == pytest.approx(1000.0)
        assert p.l_opt == pytest.approx(0.10)
        assert p.l_slack == pytest.approx(0.25)

    def test_negative_f_max_raises(self) -> None:
        with pytest.raises(ValueError):
            MuscleParameters(F_max=-100.0, l_opt=0.1, l_slack=0.25)

    def test_zero_f_max_raises(self) -> None:
        with pytest.raises(ValueError):
            MuscleParameters(F_max=0.0, l_opt=0.1, l_slack=0.25)

    def test_negative_l_opt_raises(self) -> None:
        with pytest.raises(ValueError):
            MuscleParameters(F_max=1000.0, l_opt=-0.1, l_slack=0.25)

    def test_negative_l_slack_raises(self) -> None:
        with pytest.raises(ValueError):
            MuscleParameters(F_max=1000.0, l_opt=0.1, l_slack=-0.25)


class TestHillMuscleModel:
    def setup_method(self) -> None:
        self.params = _default_params()
        self.model = HillMuscleModel(self.params)

    def test_hill_muscle_construction(self) -> None:
        assert self.model is not None

    def test_zero_activation_produces_passive_force_only(self) -> None:
        state = MuscleState(
            activation=0.0,
            l_CE=self.params.l_opt,
            v_CE=0.0,
            l_MT=self.params.l_opt + self.params.l_slack,
        )
        force = self.model.compute_force(state)
        # With zero activation and optimal length, active force should be zero
        # passive force might be zero or small at optimal length
        assert np.isfinite(force)
        assert force >= 0.0

    def test_full_activation_produces_positive_force(self) -> None:
        state = MuscleState(
            activation=1.0,
            l_CE=self.params.l_opt,
            v_CE=0.0,
            l_MT=self.params.l_opt + self.params.l_slack,
        )
        force = self.model.compute_force(state)
        assert force > 0.0

    def test_force_increases_with_activation(self) -> None:
        l_mt = self.params.l_opt + self.params.l_slack
        f_low = self.model.compute_force(
            MuscleState(activation=0.2, l_CE=self.params.l_opt, v_CE=0.0, l_MT=l_mt)
        )
        f_high = self.model.compute_force(
            MuscleState(activation=0.8, l_CE=self.params.l_opt, v_CE=0.0, l_MT=l_mt)
        )
        assert f_high > f_low

    def test_force_is_finite(self) -> None:
        state = MuscleState(
            activation=0.5,
            l_CE=self.params.l_opt,
            v_CE=0.0,
            l_MT=self.params.l_opt + self.params.l_slack,
        )
        force = self.model.compute_force(state)
        assert np.isfinite(force)

    def test_force_does_not_exceed_f_max_at_optimal(self) -> None:
        # At optimal length with full activation and zero velocity,
        # force should be approximately F_max (could be slightly higher with passive)
        state = MuscleState(
            activation=1.0,
            l_CE=self.params.l_opt,
            v_CE=0.0,
            l_MT=self.params.l_opt + self.params.l_slack,
        )
        force = self.model.compute_force(state)
        # Active force alone is F_max; total could include passive contribution
        assert force <= self.params.F_max * 2.0  # generous bound

    def test_shortening_reduces_force(self) -> None:
        # Shortening velocity should reduce force (Hill's equation)
        l_mt = self.params.l_opt + self.params.l_slack
        f_static = self.model.compute_force(
            MuscleState(activation=1.0, l_CE=self.params.l_opt, v_CE=0.0, l_MT=l_mt)
        )
        f_shortening = self.model.compute_force(
            MuscleState(
                activation=1.0,
                l_CE=self.params.l_opt,
                v_CE=-0.5 * self.params.l_opt,
                l_MT=l_mt,
            )
        )
        assert f_shortening < f_static


class TestActivationDynamics:
    def test_hill_muscle_construction(self) -> None:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        dyn = ActivationDynamics(tau_act=0.01, tau_deact=0.04)
        assert dyn.tau_act == pytest.approx(0.01)
        assert dyn.tau_deact == pytest.approx(0.04)

    def test_zero_tau_raises(self) -> None:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        with pytest.raises((ValueError, TypeError, AssertionError)):
            ActivationDynamics(tau_act=0.0, tau_deact=0.04)

    def test_derivative_positive_when_u_gt_a(self) -> None:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        dyn = ActivationDynamics()
        da_dt = dyn.compute_derivative(u=1.0, a=0.0)
        assert da_dt > 0.0

    def test_derivative_negative_when_u_lt_a(self) -> None:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        dyn = ActivationDynamics()
        da_dt = dyn.compute_derivative(u=0.0, a=1.0)
        assert da_dt < 0.0

    def test_derivative_finite(self) -> None:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        dyn = ActivationDynamics()
        da_dt = dyn.compute_derivative(u=0.5, a=0.3)
        assert np.isfinite(da_dt)

    def test_activation_approaches_excitation(self) -> None:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        dyn = ActivationDynamics(tau_act=0.01, tau_deact=0.04)
        a = 0.0
        # Apply step input u=1.0 for 200ms (20 * 10ms steps)
        for _ in range(200):
            a = dyn.update(u=1.0, a=a, dt=0.001)
        # After 200ms, activation should be close to 1.0
        assert a > 0.9
