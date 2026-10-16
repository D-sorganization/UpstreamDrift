"""NSX-90 (#12719): forward covariance, adjoint kernels and the variance budget.

Specification: ``loop_nullspace.tex`` ``sec:var-cov`` and ``sec:var-mechanisms``;
oracles: a dense stacked-matrix evaluation, a unit-impulse forward rollout, and
the exact NSX-80 pendulum goldens (``variability_sensitivity.json``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics
from src.shared.python.loop_nullspace import variance_propagation as vp
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)
from tests.unit.loop_nullspace._tangent_fixtures import Pendulum, frac, golden
from tests.unit.loop_nullspace._variance_fixtures import (
    additive_sigma,
    noise,
    pendulum_dynamics,
    sdn_sigma,
)

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(20261015)


def _random_problem(n_steps=5, nx=4, nw=3, ny=2):
    a = RNG.normal(size=(n_steps, nx, nx)) * 0.4 + np.eye(nx)
    b = RNG.normal(size=(n_steps, nx, nw))
    dyn = LinearizedDynamics(a, b)
    p0 = _spd(nx)
    sigma = np.stack([_spd(nw) for _ in range(n_steps)])
    c = RNG.normal(size=(ny, nx))
    return dyn, p0, sigma, c


def _spd(n):
    m = RNG.normal(size=(n, n))
    return m @ m.T + 0.1 * np.eye(n)


def _dense_outcome_covariance(dyn, p0, sigma, c):
    """Independent oracle: ``y = M [x0; w_0; ...; w_{T-1}]`` with dense ``M``."""
    n_steps, nx, nw = dyn.b.shape
    cols = [c @ _transition(dyn, 0, n_steps)]
    for t in range(n_steps):
        cols.append(c @ _transition(dyn, t + 1, n_steps) @ dyn.b[t])
    big = np.hstack(cols)
    cov = np.zeros((nx + n_steps * nw,) * 2)
    cov[:nx, :nx] = p0
    for t in range(n_steps):
        s = nx + t * nw
        cov[s : s + nw, s : s + nw] = sigma[t]
    return big @ cov @ big.T


def _transition(dyn, start, stop):
    phi = np.eye(dyn.a.shape[1])
    for t in range(start, stop):
        phi = dyn.a[t] @ phi
    return phi


def test_forward_matches_dense_oracle() -> None:
    dyn, p0, sigma, c = _random_problem()
    traj = vp.propagate_covariance(dyn, p0, sigma)
    got = c @ traj[-1] @ c.T
    want = _dense_outcome_covariance(dyn, p0, sigma, c)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
    assert traj.shape == (dyn.n_steps + 1, 4, 4)
    assert traj[0] == pytest.approx(p0, abs=0)


def test_adjoint_equals_forward_to_1e12() -> None:
    dyn, p0, sigma, c = _random_problem()
    forward = c @ vp.propagate_covariance(dyn, p0, sigma)[-1] @ c.T
    kernels = vp.adjoint_kernels(dyn, c)
    backward = vp.adjoint_covariance(kernels, p0, sigma)
    assert backward == pytest.approx(forward, rel=1e-12, abs=1e-12)


def test_input_kernel_is_the_forward_impulse_response() -> None:
    dyn, _, _, c = _random_problem()
    kernels = vp.adjoint_kernels(dyn, c)
    nw = dyn.b.shape[2]
    for t in range(dyn.n_steps):
        for j in range(nw):
            x = dyn.b[t][:, j].copy()  # unit w_j at step t
            for s in range(t + 1, dyn.n_steps):
                x = dyn.a[s] @ x
            assert kernels.s_y_tau(t)[:, j] == pytest.approx(c @ x, abs=1e-12)
    x0 = np.eye(dyn.a.shape[1])
    for s in range(dyn.n_steps):
        x0 = dyn.a[s] @ x0
    assert kernels.s_y_x0 == pytest.approx(c @ x0, abs=1e-12)


def test_budget_sums_to_total_and_matches_hand_blocks() -> None:
    dyn, _, _, c = _random_problem(nw=3)
    p0 = np.diag([1.0, 2.0, 0.5, 3.0])
    sigma = np.stack([np.diag([0.3, 0.7, 1.1]) for _ in range(dyn.n_steps)])
    kernels = vp.adjoint_kernels(dyn, c)
    budget = vp.variance_budget(
        kernels,
        p0,
        sigma,
        noise_classes=(
            vp.InputClass("torque_a", (0, 1)),
            vp.InputClass("torque_b", (2,)),
        ),
        initial_classes=(
            vp.InputClass("kinematic_state", (0, 1)),
            vp.InputClass("persistent_parameters", (2, 3)),
        ),
    )
    total = np.diag(vp.adjoint_covariance(kernels, p0, sigma))
    assert sum(budget.contributions.values()) == pytest.approx(total, rel=1e-12)
    assert budget.total == pytest.approx(total, rel=1e-12)
    s2 = kernels.s_y_x0[:, 2:]
    want = np.einsum("ij,jk,ik->i", s2, p0[2:, 2:], s2)
    assert budget.contributions["persistent_parameters"] == pytest.approx(
        want, rel=1e-12
    )
    by_time = budget.by_time["torque_b"]
    assert by_time.shape == (dyn.n_steps, c.shape[0])
    assert by_time.sum(axis=0) == pytest.approx(
        budget.contributions["torque_b"], rel=1e-12
    )
    assert budget.fractions().sum(axis=0) == pytest.approx(1.0, abs=1e-12)


def test_budget_refuses_cross_class_covariance_and_gaps() -> None:
    dyn, p0, sigma, c = _random_problem(nw=2)
    kernels = vp.adjoint_kernels(dyn, c)
    full = (vp.InputClass("all", (0, 1)),)
    init = (vp.InputClass("x", (0, 1, 2, 3)),)
    vp.variance_budget(kernels, p0, sigma, full, init)  # coupled but one class
    with pytest.raises(ValueError, match="cross-class"):
        vp.variance_budget(
            kernels,
            p0,
            sigma,
            (vp.InputClass("a", (0,)), vp.InputClass("b", (1,))),
            init,
        )
    with pytest.raises(ValueError, match="every"):
        vp.variance_budget(kernels, p0, sigma, (vp.InputClass("a", (0,)),), init)


def test_persistent_parameter_state_is_one_draw_per_trial() -> None:
    dyn, p0, sigma, c = _random_problem(nx=3, nw=2)
    n_p = 2
    coupling = RNG.normal(size=(dyn.n_steps, 3, n_p))
    cov_theta = _spd(n_p)
    aug = vp.augment_persistent(dyn, coupling)
    p0_aug = vp.augmented_initial_covariance(p0[:3, :3], cov_theta)
    c_aug = vp.pad_outcome_rows(c[:, :3], n_p)
    got = c_aug @ vp.propagate_covariance(aug, p0_aug, sigma)[-1] @ c_aug.T
    # Oracle: theta enters every step with the SAME draw: S_theta = sum_t Phi E_t.
    s_theta = sum(
        c[:, :3] @ _transition(dyn, t + 1, dyn.n_steps) @ coupling[t]
        for t in range(dyn.n_steps)
    )
    base = _dense_outcome_covariance(dyn, p0[:3, :3], sigma, c[:, :3])
    want = base + s_theta @ cov_theta @ s_theta.T
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
    kernels = vp.adjoint_kernels(aug, c_aug)
    assert kernels.s_y_x0[:, 3:] == pytest.approx(s_theta, abs=1e-12)
    assert vp.adjoint_covariance(kernels, p0_aug, sigma) == pytest.approx(
        got, rel=1e-12, abs=1e-12
    )


def test_general_tangent_dimension_shapes() -> None:
    """k = 2 (nx = 4) runs through the same code path as k = 1."""
    dyn, p0, sigma, c = _random_problem(nx=4, nw=2, ny=1)
    assert vp.propagate_covariance(dyn, p0, sigma).shape[-1] == 4
    f_mat = RNG.normal(size=(4, 4))
    g_mat = RNG.normal(size=(4, 2))
    one = vp.euler_dynamics([(f_mat, g_mat)] * 3, dt=0.01)
    assert one.a[0] == pytest.approx(np.eye(4) + 0.01 * f_mat, abs=0)
    assert one.b[2] == pytest.approx(0.01 * g_mat, abs=0)


def test_event_outcome_rows_apply_the_event_projection() -> None:
    from src.shared.python.loop_nullspace.event_section import event_projection

    grad_g = np.array([1.0, 0.0])
    flow = np.array([2.0, 0.5])
    c_y = np.array([[0.4, 1.0]])
    rows = vp.event_outcome_rows(c_y, grad_g, flow)
    assert rows == pytest.approx(c_y @ event_projection(grad_g, flow), abs=1e-15)
    # A perturbation along the flow only shifts the event time: no outcome change.
    assert rows @ flow == pytest.approx(np.zeros(1), abs=1e-14)


# ----------------------------------------------------------------- golden


def _golden_var(z: float, *, sdn: bool, preload: bool, lam_frozen_at=None) -> float:
    """Pipeline variance for the NSX-80 pendulum at allocation ``z``."""
    nz = noise()
    pend = Pendulum()
    q, _ = pend.state()
    u_z = pend.load(z)
    if preload:
        zl = z if lam_frozen_at is None else lam_frozen_at
        dyn = pendulum_dynamics(zl, nz["dt"])
    else:
        dyn = _golden_frozen_dynamics(nz["dt"])
    sigma = sdn_sigma(u_z) if sdn else additive_sigma()
    kernels = vp.adjoint_kernels(dyn, nz["c"][None, :])
    return float(vp.adjoint_covariance(kernels, nz["p0"], sigma)[0, 0])


def _golden_frozen_dynamics(dt: float) -> LinearizedDynamics:
    """The golden's M1-only model: ``F10`` of the preload-free derivation."""
    g = golden()
    f10 = frac(g["min_variance"]["F10_frozen_without_preload"])
    row = [[frac(x) for x in r] for r in g["linearization"]["G_nominal_om_row"]]
    f_mat = np.array([[0.0, 1.0], [f10, 0.0]])
    g_mat = np.array([[0.0, 0.0], row[0]])
    a = np.eye(2) + dt * f_mat
    return LinearizedDynamics(a[None], (dt * g_mat)[None])


def test_m1_only_variance_reproduces_the_golden_polynomial() -> None:
    mv = golden()["min_variance"]
    coeffs = [frac(c) for c in mv["var_poly_coeffs_descending_M1"]]
    for z in (0.0, 0.5, -1.5, 3.25):
        want = np.polyval(coeffs, z)
        got = _golden_var(z, sdn=True, preload=False)
        assert got == pytest.approx(want, rel=1e-12)
    z_star = frac(mv["z_star"])
    assert _golden_var(z_star, sdn=True, preload=False) == pytest.approx(
        frac(mv["var_at_zstar"]), rel=1e-12
    )
    grid = np.linspace(z_star - 3, z_star + 3, 1201)
    vals = [_golden_var(float(z), sdn=True, preload=False) for z in grid]
    assert grid[int(np.argmin(vals))] == pytest.approx(z_star, abs=6e-3)


def test_additive_noise_is_flat_for_random_allocations() -> None:
    """Negative control: no declared mechanism => variance independent of z."""
    zs = RNG.uniform(-5.0, 5.0, size=20)
    nz = noise()
    base = None
    for z in zs:
        # G is recomputed from the real linearization at every z; A is the
        # z-independent model (M2 not declared).
        dyn_real = pendulum_dynamics(float(z), nz["dt"])
        frozen = _golden_frozen_dynamics(nz["dt"])
        dyn = LinearizedDynamics(frozen.a, dyn_real.b)
        kernels = vp.adjoint_kernels(dyn, nz["c"][None, :])
        v = float(vp.adjoint_covariance(kernels, nz["p0"], additive_sigma())[0, 0])
        base = v if base is None else base
        assert v == pytest.approx(base, rel=1e-12, abs=0)


def test_signal_dependent_noise_changes_with_z() -> None:
    v0 = _golden_var(0.0, sdn=True, preload=False)
    v1 = _golden_var(1.5, sdn=True, preload=False)
    assert abs(v1 - v0) > 1e-4


def test_preload_changes_variance_even_for_additive_noise() -> None:
    gold = golden()["additive_noise_flat"]
    zs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    vals = np.array([_golden_var(float(z), sdn=False, preload=True) for z in zs])
    assert np.ptp(vals) > 1e-5
    # d Var / d z from the golden: z/20000 - 41/100000 (exact rationals).
    assert gold["gradient_with_preload"] == "z/20000 - 41/100000"
    h = 1e-3
    for z in zs:
        fd = (
            _golden_var(float(z) + h, sdn=False, preload=True)
            - _golden_var(float(z) - h, sdn=False, preload=True)
        ) / (2 * h)
        assert fd == pytest.approx(z / 20000.0 - 41.0 / 100000.0, abs=2e-9)


def test_preload_with_signal_dependent_noise_matches_golden_gradient() -> None:
    a, b = (
        frac(c) for c in golden()["min_variance"]["dvar_dz_coeffs_with_preload_M1_M2"]
    )
    h = 1e-3
    for z in (-2.0, 0.0, 1.5):
        fd = (
            _golden_var(z + h, sdn=True, preload=True)
            - _golden_var(z - h, sdn=True, preload=True)
        ) / (2 * h)
        assert fd == pytest.approx(a * z + b, abs=2e-9)


def test_planted_frozen_lambda_variant_fails_the_golden() -> None:
    """Holding lambda at its nominal-allocation value drops M2 and must be caught."""
    gold_grad = lambda z: z / 20000.0 - 41.0 / 100000.0  # noqa: E731
    h = 1e-3
    z = 2.0
    good = (
        _golden_var(z + h, sdn=False, preload=True)
        - _golden_var(z - h, sdn=False, preload=True)
    ) / (2 * h)
    frozen = (
        _golden_var(z + h, sdn=False, preload=True, lam_frozen_at=0.0)
        - _golden_var(z - h, sdn=False, preload=True, lam_frozen_at=0.0)
    ) / (2 * h)
    assert good == pytest.approx(gold_grad(z), abs=2e-9)
    assert abs(frozen - gold_grad(z)) > 1e-5  # the frozen variant is flat: caught


# ------------------------------------------------------- disturbance bridge


def test_step_covariances_use_the_timestep_consistent_spec() -> None:
    nz = noise()
    dt = nz["dt"]
    sig0 = nz["sigma0"]
    spec = DisturbanceSpec(
        name="torque_white",
        stage=InjectionStage.TORQUE,
        channels=("u1", "u2"),
        units="N",
        covariance=np.diag(sig0**2 * dt),  # S0 so that S0 / dt = sigma0^2
        spectrum=Spectrum.WHITE,
    )
    sigma = vp.step_covariances(spec, dt, 3)
    assert sigma.shape == (3, 2, 2)
    assert sigma[1] == pytest.approx(np.diag(sig0**2), rel=1e-14)
    with pytest.raises(ValueError, match="signals"):
        vp.step_covariances(spec, dt, 3, signals=np.zeros((3, 2)))


def test_step_covariances_apply_the_signal_law_per_step() -> None:
    dt = 0.1
    spec = DisturbanceSpec(
        name="torque_sdn",
        stage=InjectionStage.TORQUE,
        channels=("u1",),
        units="N",
        covariance=np.array([[0.04 * dt]]),
        spectrum=Spectrum.WHITE,
        signal_gain=(0.5 * np.sqrt(dt),),
        signal_bounds=((-10.0, 10.0),),
    )
    u = np.array([[1.0], [2.0], [-3.0]])
    sigma = vp.step_covariances(spec, dt, 3, signals=u)
    assert sigma[:, 0, 0] == pytest.approx(0.04 + 0.25 * u[:, 0] ** 2, rel=1e-13)


def test_preconditions_raise() -> None:
    dyn, p0, sigma, c = _random_problem()
    with pytest.raises(ValueError, match="p0"):
        vp.propagate_covariance(dyn, p0[:3, :3], sigma)
    with pytest.raises(ValueError, match="sigma"):
        vp.propagate_covariance(dyn, p0, sigma[:-1])
    with pytest.raises(ValueError, match="symmetric"):
        bad = p0.copy()
        bad[0, 1] += 1.0
        vp.propagate_covariance(dyn, bad, sigma)
    with pytest.raises(ValueError, match="outcome"):
        vp.adjoint_kernels(dyn, c[:, :3])
    with pytest.raises(ValueError, match="finite"):
        vp.adjoint_kernels(dyn, np.full_like(c, np.nan))
    with pytest.raises(ValueError, match="dt"):
        vp.euler_dynamics([(np.eye(2), np.ones((2, 1)))], dt=0.0)
