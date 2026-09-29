"""Tests for the ``rust_muscle`` Python facade over ``upstream_muscle``.

Covers:

- Pure-Python fallback path (the wheel may or may not be installed in CI).
- Parity vs the existing ``HillMuscleModel`` / ``ActivationDynamics`` /
  multi-muscle code at 1e-6.
- Shape validation in the public surface.

The acceptance criterion in UD#5216 is 1e-6 numerical parity with
OpenSim/MuJoCo reference outputs. The Python implementation in this repo
is the documented contract for that comparison (it ports the OpenSim
analytical curves directly), so we anchor the Rust parity to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics import rust_muscle

pytestmark = [pytest.mark.unit]


TOL = 1e-6


# ── Scalar curves ────────────────────────────────────────────────────────────


def test_f_l_matches_python_source() -> None:
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    for x in [0.5, 0.8, 1.0, 1.2, 1.5]:
        assert abs(rust_muscle.f_l(x) - model.force_length_active(x)) < TOL


def test_f_p_matches_python_source() -> None:
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    for x in [0.5, 1.0, 1.05, 1.2, 1.5]:
        assert abs(rust_muscle.f_p(x) - model.force_length_passive(x)) < TOL


def test_f_v_matches_python_source() -> None:
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    for x in [-1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5]:
        assert abs(rust_muscle.f_v(x) - model.force_velocity(x)) < TOL


def test_f_t_matches_python_source() -> None:
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    for x in [0.5, 1.0, 1.05, 1.2, 1.5]:
        assert abs(rust_muscle.f_t(x) - model.tendon_force(x)) < TOL


# ── Activation dynamics ──────────────────────────────────────────────────────


def test_activation_step_matches_python_source() -> None:
    from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics

    dyn = ActivationDynamics(tau_act=0.010, tau_deact=0.040, min_activation=0.001)
    a = 0.0
    a_rust = 0.0
    for k in range(200):
        u = 1.0 if k < 100 else 0.0
        a = dyn.update(u, a, 0.001)
        a_rust = rust_muscle.activation_step(u, a_rust, 0.001)
        assert abs(a - a_rust) < TOL


def test_activation_step_batch_parity() -> None:
    from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics

    rng = np.random.default_rng(0)
    n = 32
    u = rng.uniform(0.0, 1.0, size=n)
    a = rng.uniform(0.001, 1.0, size=n)
    dyn = ActivationDynamics(tau_act=0.010, tau_deact=0.040, min_activation=0.001)
    expected = np.array([dyn.update(float(u[i]), float(a[i]), 0.001) for i in range(n)])
    actual = rust_muscle.activation_step_batch(u, a, 0.001)
    assert np.allclose(actual, expected, atol=TOL)


def test_activation_step_rejects_bad_dt() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        rust_muscle.activation_step(0.5, 0.1, dt=0.0)


# ── Muscle force ─────────────────────────────────────────────────────────────


def _params_for(i: int) -> tuple[float, float, float, float, float, float, float]:
    bases = [
        (1000.0, 0.15, 0.20, 10.0, 0.0, 0.05, 0.56),
        (800.0, 0.12, 0.10, 10.0, 0.2, 0.05, 0.56),
        (1200.0, 0.18, 0.22, 8.0, 0.0, 0.10, 0.40),
    ]
    return bases[i % len(bases)]


def test_muscle_force_batch_parity() -> None:
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
        MuscleState,
    )

    n = 16
    rng = np.random.default_rng(7)
    params = np.array([_params_for(i) for i in range(n)])
    activations = rng.uniform(0.0, 1.0, size=n)
    l_ce = params[:, 1] * rng.uniform(0.6, 1.4, size=n)
    v_ce = params[:, 3] * params[:, 1] * rng.uniform(-0.5, 0.5, size=n)
    actual = rust_muscle.muscle_force_batch(activations, l_ce, v_ce, params)
    expected = np.empty(n)
    for i in range(n):
        p = MuscleParameters(
            F_max=float(params[i, 0]),
            l_opt=float(params[i, 1]),
            l_slack=float(params[i, 2]),
            v_max=float(params[i, 3]),
            pennation_angle=float(params[i, 4]),
            damping=float(params[i, 5]),
        )
        model = HillMuscleModel(p, force_length_width=float(params[i, 6]))
        state = MuscleState(
            activation=float(activations[i]),
            l_CE=float(l_ce[i]),
            v_CE=float(v_ce[i]),
            l_MT=0.0,
        )
        expected[i] = model.compute_force(state)
    assert np.allclose(actual, expected, atol=TOL)


def test_muscle_force_batch_validates_shapes() -> None:
    a = np.zeros(4)
    ll = np.zeros(4)
    v = np.zeros(4)
    p = np.zeros((4, 6))  # wrong width
    with pytest.raises(ValueError, match="params must have shape"):
        rust_muscle.muscle_force_batch(a, ll, v, p)


def test_joint_torques_batch_parity() -> None:
    rng = np.random.default_rng(13)
    j, m = 5, 100
    r = rng.normal(0.0, 0.04, size=(j, m))
    f = rng.uniform(0.0, 1000.0, size=m)
    out = rust_muscle.joint_torques_batch(r, f)
    expected = r @ f
    assert np.allclose(out, expected, atol=1e-9)


def test_step_full_matches_chained_calls() -> None:
    rng = np.random.default_rng(2)
    n = 20
    u = rng.uniform(0.0, 1.0, size=n)
    a = rng.uniform(0.001, 1.0, size=n)
    params = np.array([_params_for(i) for i in range(n)])
    l_ce = params[:, 1].copy()
    v_ce = np.zeros(n)
    r = rng.normal(0.0, 0.03, size=(3, n))
    a_full, tau_full = rust_muscle.step_full(u, a, l_ce, v_ce, params, r, dt=0.001)

    # Chained reference using the same facade primitives.
    a_ref = rust_muscle.activation_step_batch(u, a, 0.001)
    f_ref = rust_muscle.muscle_force_batch(a_ref, l_ce, v_ce, params)
    tau_ref = rust_muscle.joint_torques_batch(r, f_ref)
    assert np.allclose(a_full, a_ref, atol=TOL)
    assert np.allclose(tau_full, tau_ref, atol=TOL)
