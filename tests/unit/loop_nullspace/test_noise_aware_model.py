"""Tests of the allocation-dependent variance model (NSX-92, #12721).

Oracles are independent of the model: the NSX-90 forward covariance recursion
(``propagate_covariance``) and adjoint kernels, central finite differences, and
the exact NSX-80 golden polynomial of the pendulum.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics
from src.shared.python.loop_nullspace.noise_aware_model import (
    NoiseAwareModel,
    fit_affine_dynamics,
    psd_factor,
    wrap_probability_bound,
)
from src.shared.python.loop_nullspace.variance_propagation import (
    adjoint_kernels,
    propagate_covariance,
)
from src.shared.python.perturbation.disturbance import DisturbanceSpec
from tests.unit.loop_nullspace import _noise_aware_fixtures as fx
from tests.unit.loop_nullspace._tangent_fixtures import golden

pytestmark = pytest.mark.unit


def _random_model(seed: int = 3, steps: int = 3, n_z: int = 4) -> NoiseAwareModel:
    rng = np.random.default_rng(seed)
    n, n_ch = 3, 2
    a0 = np.eye(n)[None] + 0.1 * rng.standard_normal((steps, n, n))
    slope = 0.05 * rng.standard_normal((steps, n, n, n_z))
    b = 0.3 * rng.standard_normal((steps, n, n_ch))
    p0 = np.diag([0.01, 0.02, 0.03])
    spec = fx.spec((0.2, 0.4), (0.3, 0.2))
    return NoiseAwareModel(
        a0, b, rng.standard_normal(n), p0, spec, fx.DT, n_z, a_slope=slope
    )


def test_psd_factor_reconstructs_and_rejects_bad_input() -> None:
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    fac = psd_factor(cov)
    assert np.allclose(fac @ fac.T, cov, rtol=0, atol=1e-14)
    assert np.allclose(psd_factor(np.zeros((2, 2))), 0.0)
    with pytest.raises(ValueError, match="positive semi-definite"):
        psd_factor(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        psd_factor(np.array([[1.0, 0.1], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="square"):
        psd_factor(np.ones((2, 3)))


def test_wrap_probability_bound_matches_gaussian_tail() -> None:
    # sd = 1 rad: P(|y| > pi) = 2 (1 - Phi(pi)) = erfc(pi / sqrt 2)
    assert wrap_probability_bound(1.0) == pytest.approx(
        math.erfc(math.pi / math.sqrt(2))
    )
    assert wrap_probability_bound(0.0) == 0.0
    assert wrap_probability_bound(1e-4) < 1e-300 + 1e-12
    with pytest.raises(ValueError):
        wrap_probability_bound(-1.0)
    with pytest.raises(ValueError):
        wrap_probability_bound(1.0, period=0.0)


def test_kernels_equal_the_nsx90_adjoint_kernels_at_every_z() -> None:
    model = _random_model()
    for z in (np.zeros(4), np.array([0.3, -1.0, 2.0, 0.5])):
        s_x0, s_in = model.kernels(z)
        ref = adjoint_kernels(
            LinearizedDynamics(model.a_at(z), np.asarray(model.b)),
            np.asarray(model.outcome_row)[None],
        )
        assert np.allclose(s_x0, ref.s_y_x0[0], rtol=1e-13, atol=0)
        assert np.allclose(s_in, ref.inputs[:, 0, :], rtol=1e-13, atol=0)


def test_kernel_jacobians_match_central_differences() -> None:
    model = _random_model()
    z = np.array([0.2, -0.4, 0.7, 1.1])
    _, _, d_x0, d_in = model.kernel_jacobians(z)
    # Central difference of a cubic polynomial: truncation h^2 f'''/6 ~ 1e-10
    # and rounding eps|f|/h ~ 3e-11 at h = 1e-5, so 1e-8 absolute is safe.
    h = 1e-5
    for m in range(4):
        e = np.zeros(4)
        e[m] = h
        sx_p, si_p = model.kernels(z + e)
        sx_m, si_m = model.kernels(z - e)
        assert np.allclose(d_x0[:, m], (sx_p - sx_m) / (2 * h), rtol=1e-7, atol=1e-8)
        assert np.allclose(d_in[:, :, m], (si_p - si_m) / (2 * h), rtol=1e-7, atol=1e-8)


def test_variance_equals_the_nsx90_forward_recursion_with_the_m1_law() -> None:
    """Independent oracle: forward covariance propagation with ``Sigma_t(tau_t)``."""
    model = _random_model()
    z = np.array([0.2, -0.4, 0.7, 1.1])
    rng = np.random.default_rng(11)
    tau = rng.uniform(-5.0, 5.0, size=(3, 2))
    sigma = np.stack([model.spec.discrete_covariance(fx.DT, tau[t]) for t in range(3)])
    dyn = LinearizedDynamics(model.a_at(z), np.asarray(model.b))
    cov = propagate_covariance(dyn, model.p0, sigma)[-1]
    expected = float(model.outcome_row @ cov @ model.outcome_row)
    assert model.variance(z, tau) == pytest.approx(expected, rel=1e-12)
    parts = model.variance_parts(z, tau)
    assert set(parts) == {"initial", "additive", "signal_dependent"}
    assert all(v >= 0.0 for v in parts.values())


def test_residual_jacobian_is_exact_and_gradient_is_two_jt_r() -> None:
    model = _random_model()
    rng = np.random.default_rng(5)
    ntil = rng.standard_normal((6, 4))
    t0 = rng.standard_normal(6)
    z = np.array([0.2, -0.4, 0.7, 1.1])

    def tau_of(zz: np.ndarray) -> np.ndarray:
        return (t0 + ntil @ zz).reshape(3, 2)

    r, jac = model.residuals(z, tau_of(z), ntil)
    assert float(r @ r) == pytest.approx(model.variance(z, tau_of(z)), rel=1e-13)
    h = 1e-6
    grad_fd = np.empty(4)
    for m in range(4):
        e = np.zeros(4)
        e[m] = h
        grad_fd[m] = (
            model.variance(z + e, tau_of(z + e)) - model.variance(z - e, tau_of(z - e))
        ) / (2 * h)
    assert np.allclose(2.0 * jac.T @ r, grad_fd, rtol=1e-7, atol=1e-10)


def test_golden_polynomial_of_the_pendulum_is_reproduced_exactly() -> None:
    gold = golden()["min_variance"]
    coeffs = [float(Fraction(c)) for c in gold["var_poly_coeffs_descending_M1"]]
    model = fx.m1_model()
    q = fx.Q
    for z in (-3.0, 0.0, 1.5):
        tau = (fx.U_P - q * z)[None]
        expected = np.polyval(coeffs, z)
        assert model.variance(np.array([z]), tau) == pytest.approx(expected, rel=1e-12)


def test_golden_m2_slope_of_the_variance_is_reproduced() -> None:
    gold = golden()["min_variance"]
    a, b = (float(Fraction(c)) for c in gold["dvar_dz_coeffs_with_preload_M1_M2"])
    model = fx.m2_model()

    def v(z: float) -> float:
        return model.variance(np.array([z]), (fx.U_P - fx.Q * z)[None])

    h = 1e-3  # V is an exact quadratic in z here, so the central difference is exact
    for z in (-1.0, 0.0, 2.0):
        assert (v(z + h) - v(z - h)) / (2 * h) == pytest.approx(a * z + b, rel=1e-9)


def test_signal_coefficients_and_invariant_part() -> None:
    model = fx.m1_model()
    nz = golden()["model"]["noise_declared"]
    k = np.array([float(Fraction(x)) for x in nz["k"]])
    b = np.array([1 / 25, 3 / 100])
    assert np.allclose(model.signal_coefficients(np.zeros(1))[0], (b * k) ** 2)
    parts = model.variance_parts(np.zeros(1), (fx.U_P)[None])
    assert model.invariant_variance(np.zeros(1)) == pytest.approx(
        parts["initial"] + parts["additive"], rel=1e-14
    )


def test_fit_affine_dynamics_recovers_the_real_preload_and_rejects_nonaffine() -> None:
    a0, slope = fit_affine_dynamics(fx.real_a_of_z, 1)
    # F10 = z/2 - 81/10, A10 = dt F10 (NSX-80 table); the assembled-KKT
    # provider is accurate to ~3e-11 (measured), hence the 1e-9 bounds.
    assert a0[0, 1, 0] == pytest.approx(fx.DT * -8.1, abs=1e-9)
    assert slope[0, 1, 0, 0] == pytest.approx(fx.DT * 0.5, abs=1e-9)
    others = np.delete(slope.reshape(-1), 2)  # only A10 moves
    assert np.allclose(others, 0.0, atol=1e-9)

    def curved(z: np.ndarray) -> np.ndarray:
        return np.array([[[1.0 + z[0] ** 2]]])

    with pytest.raises(ValueError, match="not affine"):
        fit_affine_dynamics(curved, 1)
    with pytest.raises(TypeError):
        fit_affine_dynamics(3, 1)  # type: ignore[arg-type]


def test_model_contracts() -> None:
    nz = fx.noise()
    base = {
        "a0": fx.frozen_dynamics(),
        "b": fx.b_matrix(),
        "outcome_row": nz["c"],
        "p0": nz["p0"],
        "spec": fx.golden_spec(),
        "dt": fx.DT,
        "n_z": 1,
    }
    assert NoiseAwareModel(**base).preload_active is False
    with pytest.raises(ValueError, match="dt"):
        NoiseAwareModel(**{**base, "dt": 0.0})
    with pytest.raises(ValueError, match="n_z"):
        NoiseAwareModel(**{**base, "n_z": 0})
    with pytest.raises(ValueError, match="a_slope"):
        NoiseAwareModel(**{**base, "a_slope": np.zeros((1, 2, 2, 3))})
    with pytest.raises(TypeError):
        NoiseAwareModel(**{**base, "spec": "not a spec"})
    with pytest.raises(ValueError, match="channels"):
        NoiseAwareModel(**{**base, "b": np.zeros((1, 2, 3))})
    persistent = DisturbanceSpec(
        name="p",
        stage=fx.golden_spec().stage,
        channels=fx.CHANNELS,
        units="N",
        covariance=np.eye(2),
        persistent=True,
    )
    with pytest.raises(ValueError, match="white"):
        NoiseAwareModel(**{**base, "spec": persistent})
    model = NoiseAwareModel(**base)
    with pytest.raises(ValueError, match="z must"):
        model.kernels(np.zeros(3))
    with pytest.raises(ValueError, match="tau"):
        model.variance(np.zeros(1), np.zeros((2, 2)))


def test_additive_only_model_has_no_signal_term_and_zero_jacobian() -> None:
    sp = fx.spec(None)
    model = fx.m1_model(sp)
    assert model.gain.tolist() == [0.0, 0.0]
    ntil = np.array([[-fx.Q[0]], [-fx.Q[1]]])
    z = np.zeros(1)
    r, jac = model.residuals(z, (fx.U_P)[None], ntil)
    assert np.all(jac == 0.0)
    assert model.variance_parts(z, (fx.U_P)[None])["signal_dependent"] == 0.0
    assert float(r @ r) == pytest.approx(model.variance(z, (fx.U_P)[None]))
