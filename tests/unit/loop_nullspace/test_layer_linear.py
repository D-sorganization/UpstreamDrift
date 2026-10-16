"""NSX-93 (#12722): the linear variance model of impedance and delayed feedback.

Independent oracles: the stationary variance of a damped oscillator driven by
white noise (``S / (2 d k)``), the closed-form delay margin of a PD law on a
double integrator, and an explicit delayed recursion for the augmentation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.impedance_spec import (
    HYPOTHESIS_ID,
    ImpedanceWindow,
    JointImpedance,
)
from src.shared.python.loop_nullspace.layer_linear import (
    LayerLinearization,
    frozen_delay_margin,
    impedance_state_gain,
    inflate_step_covariances,
    layer_variance,
    variance_ratio_record,
)
from src.shared.python.loop_nullspace.variance_propagation import propagate_covariance
from src.shared.python.loop_nullspace.variance_records import (
    Mechanism,
    VarianceDeclaration,
)
from tests.unit.loop_nullspace._layer_fixtures import analytic_pd_delay_margin

pytestmark = pytest.mark.unit

M, KP, KD = 2.0, 200.0, 20.0  # omega_n = 10 rad/s, zeta = 0.5
F_DI = np.array([[0.0, 1.0], [0.0, 0.0]])
G_DI = np.array([[0.0], [1.0 / M]])


def _lin(n_steps: int, dt: float, **kw) -> LayerLinearization:
    return LayerLinearization(
        f=np.repeat(F_DI[None], n_steps, axis=0),
        g_w=np.repeat(G_DI[None], n_steps, axis=0),
        g_u=np.repeat(G_DI[None], n_steps, axis=0),
        dt=dt,
        **kw,
    )


def _gain(n_steps: int, kp: float, kd: float) -> np.ndarray:
    return np.repeat(np.array([[[kp, kd]]]), n_steps, axis=0)


def _decl() -> VarianceDeclaration:
    return VarianceDeclaration(
        mechanisms=frozenset(),
        noise_hypotheses=("H-NOISE-ADDITIVE",),
        model_provenance=ModelProvenance("test_double_integrator", "1"),
        input_provenance=InputProvenance("test", "declared"),
    )


# ------------------------------------------------------ closed-form variance
@pytest.mark.parametrize("dt", [2e-3, 1e-3])
def test_impedance_variance_matches_the_stationary_oscillator_formula(dt) -> None:
    """``m x'' + d x' + k x = w`` with white PSD ``S``: ``var x = S / (2 d k)``.

    The step noise is held over a step (covariance ``S / dt``), which differs
    from white noise by a ``sinc`` low-pass: a relative error of order
    ``(omega_n dt)^2 / 12`` (observed ~1e-5 at dt = 1e-3), so the tolerance is
    ``(omega_n dt)^2``.
    """
    s_psd = 0.8
    n_steps = round(3.0 / dt)  # 15 time constants of the slowest mode (zeta omega_n)
    lin = _lin(n_steps, dt, impedance_gain=_gain(n_steps, KP, KD))
    sigma = np.full((n_steps, 1, 1), s_psd / dt)
    traj = propagate_covariance(lin.closed_loop(), np.zeros((2, 2)), sigma)
    expected = s_psd / (2.0 * KD * KP)
    assert traj[-1][0, 0] == pytest.approx(expected, rel=(10.0 * dt) ** 2 / 6.0)


def test_impedance_reduces_the_variance_and_euler_is_the_first_order_variant() -> None:
    dt, n = 5e-3, 40
    sigma = np.full((n, 1, 1), 0.8 / dt)
    c = np.array([[1.0, 0.0]])
    base = layer_variance(
        _lin(n, dt),
        np.zeros((2, 2)),
        sigma,
        c,
        _decl(),
        layer_hypotheses=(),
        outcome_units="m",
    ).assessment.variance.values[0]
    stiff = layer_variance(
        _lin(n, dt, impedance_gain=_gain(n, KP, KD)),
        np.zeros((2, 2)),
        sigma,
        c,
        _decl(),
        layer_hypotheses=(HYPOTHESIS_ID,),
        outcome_units="m",
    ).assessment.variance.values[0]
    # free double integrator driven by white force noise: var x = S T^3 / (3 m^2)
    assert base == pytest.approx(0.8 * 0.2**3 / (3.0 * M**2), rel=2e-3)
    assert stiff < base / 3.0
    euler = layer_variance(
        _lin(n, dt, impedance_gain=_gain(n, KP, KD), method="euler"),
        np.zeros((2, 2)),
        sigma,
        c,
        _decl(),
        layer_hypotheses=(HYPOTHESIS_ID,),
        outcome_units="m",
    ).assessment.variance.values[0]
    assert euler == pytest.approx(stiff, rel=0.1)  # same physics, first order


# ------------------------------------------------------------ delay chain
def test_delay_augmentation_matches_an_explicit_delayed_recursion() -> None:
    rng = np.random.default_rng(11)
    n_steps, m_delay, dt = 9, 3, 0.01
    lin = _lin(
        n_steps,
        dt,
        feedback_gain=rng.normal(size=(n_steps, 1, 2)),
        impedance_gain=_gain(n_steps, 30.0, 3.0),
        delay_steps=m_delay,
    )
    a, b_w, b_u = lin.discrete()
    x0 = rng.normal(size=2)
    w = rng.normal(size=(n_steps, 1))
    hist = [x0] * (m_delay + 1)  # held pre-history: x_{-j} = x_0
    for t in range(n_steps):
        hist.append(
            a[t] @ hist[-1]
            - b_u[t] @ lin.feedback_gain[t] @ hist[-1 - m_delay]
            + b_w[t] @ w[t]
        )
    dyn = lin.closed_loop()
    z = np.tile(x0, m_delay + 1)
    for t in range(n_steps):
        z = dyn.a[t] @ z + dyn.b[t] @ w[t]
    np.testing.assert_allclose(z[:2], hist[-1], rtol=1e-12, atol=1e-14)
    # and the covariance augmentation keeps the delayed copies fully correlated
    p_aug = lin.augment_covariance(np.diag([1.0, 2.0]))
    assert p_aug.shape == (2 * (m_delay + 1),) * 2
    np.testing.assert_allclose(p_aug[:2, 2:4], np.diag([1.0, 2.0]))


def test_zero_gains_reproduce_the_baseline_variance() -> None:
    """Zero impedance and zero feedback with a delay: equal to the baseline.

    The ``x`` block of the augmented chain is the baseline chain plus exact
    zeros; only BLAS summation order over the extra zero columns can differ,
    a few ulps, so the tolerance is ``1e-13`` relative.
    """
    n, dt = 30, 0.01
    sigma = np.full((n, 1, 1), 0.8 / dt)
    c = np.array([[1.0, 0.0]])
    args = (np.diag([1e-4, 1e-3]), sigma, c, _decl())
    base = layer_variance(
        _lin(n, dt), *args, layer_hypotheses=(), outcome_units="m"
    ).assessment
    zero = layer_variance(
        _lin(
            n,
            dt,
            impedance_gain=np.zeros((n, 1, 2)),
            feedback_gain=np.zeros((n, 1, 2)),
            delay_steps=4,
        ),
        *args,
        layer_hypotheses=(HYPOTHESIS_ID, "H-SENSORIMOTOR-DELAY"),
        outcome_units="m",
    ).assessment
    np.testing.assert_allclose(
        zero.variance.values, base.variance.values, rtol=1e-13, atol=0.0
    )
    np.testing.assert_allclose(
        zero.kernel_x0.values[:, :2], base.kernel_x0.values, rtol=1e-13, atol=1e-18
    )


# ------------------------------------------------------------ delay margin
def test_discrete_delay_margin_matches_the_closed_form() -> None:
    """Tolerance ``1.5 dt``: zero-order hold adds an average lag of ``dt/2`` and
    the margin is resolved in whole steps (``dt``)."""
    dt = 1e-3
    lin = _lin(1, dt)
    a, _, b_u = lin.discrete()
    margin = frozen_delay_margin(a[0], b_u[0], np.array([[KP, KD]]), dt, max_steps=200)
    exact = analytic_pd_delay_margin(M, KP, KD)
    assert margin.stable_at_zero and margin.margin_s is not None
    assert margin.margin_s == pytest.approx(exact, abs=1.5 * dt)
    assert margin.spectral_radius[-1] > 1.0 and np.all(
        margin.spectral_radius[:-1] <= 1.0
    )


def test_an_unstable_delay_free_loop_has_zero_margin() -> None:
    dt = 1e-3
    a, _, b_u = _lin(1, dt).discrete()
    bad = frozen_delay_margin(a[0], b_u[0], np.array([[-KP, KD]]), dt)
    assert not bad.stable_at_zero and bad.margin_steps == 0 and bad.margin_s == 0.0


def test_a_robust_loop_reports_no_margin_within_the_search() -> None:
    dt = 1e-3
    a, _, b_u = _lin(1, dt).discrete()
    small = frozen_delay_margin(a[0], b_u[0], np.array([[2.0, 4.0]]), dt, max_steps=20)
    assert small.margin_steps is None and small.margin_s is None


def test_margin_profile_needs_a_feedback_gain() -> None:
    with pytest.raises(ValueError, match="feedback gain"):
        _lin(2, 0.01).delay_margins()
    lin = _lin(2, 1e-3, feedback_gain=_gain(2, KP, KD), delay_steps=10)
    assert len(lin.delay_margins(200)) == 2


# --------------------------------------------------------------- evidence
def test_variance_is_invariant_and_conditioned_on_the_declared_layer() -> None:
    n, dt = 5, 0.01
    res = layer_variance(
        _lin(n, dt, impedance_gain=_gain(n, KP, KD)),
        np.zeros((2, 2)),
        np.full((n, 1, 1), 1.0),
        np.array([[1.0, 0.0]]),
        _decl(),
        layer_hypotheses=(HYPOTHESIS_ID,),
        outcome_units="m",
    )
    rec = res.assessment.variance
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert rec.hypotheses == ()
    assert HYPOTHESIS_ID in rec.conditioning and "H-NOISE-ADDITIVE" in rec.conditioning
    for kernel in (res.assessment.kernel_x0, res.assessment.kernel_input):
        assert HYPOTHESIS_ID in kernel.conditioning  # kernels depend on the gains
    assert res.assessment.linear_validity.name == "NOT_ASSESSED"


def test_a_v1_mechanism_makes_the_variance_family_dependent() -> None:
    decl = VarianceDeclaration(
        mechanisms=frozenset({Mechanism.SIGNAL_DEPENDENT_NOISE}),
        noise_hypotheses=("H-NOISE-SIGNAL-DEPENDENT",),
        model_provenance=ModelProvenance("m", "1"),
        input_provenance=InputProvenance("test", "declared"),
    )
    res = layer_variance(
        _lin(3, 0.01),
        np.zeros((2, 2)),
        np.full((3, 1, 1), 1.0),
        np.array([[1.0, 0.0]]),
        decl,
        layer_hypotheses=(),
        outcome_units="m",
    )
    assert res.assessment.variance.allocation_dependence is (
        AllocationDependence.FAMILY_DEPENDENT
    )
    assert res.assessment.variance.hypotheses == ("H-NOISE-SIGNAL-DEPENDENT",)


def test_torque_only_cocontraction_without_a_declaration_is_unavailable() -> None:
    n, dt = 5, 0.01
    kw = {
        "p0": np.zeros((2, 2)),
        "sigma": np.full((n, 1, 1), 1.0),
        "outcome_rows": np.array([[1.0, 0.0]]),
        "declaration": _decl(),
        "outcome_units": "m",
    }
    base = layer_variance(_lin(n, dt), layer_hypotheses=(), **kw).assessment
    layered = layer_variance(
        _lin(n, dt, impedance_gain=_gain(n, KP, KD)),
        layer_hypotheses=(HYPOTHESIS_ID,),
        **kw,
    ).assessment
    declared = variance_ratio_record(base, layered, impedance_declared=True)
    assert declared.allocation_dependence is AllocationDependence.INVARIANT
    assert HYPOTHESIS_ID in declared.conditioning
    assert declared.values[0] == pytest.approx(
        layered.variance.values[0] / base.variance.values[0]
    )
    assert declared.values[0] < 1.0
    undeclared = variance_ratio_record(base, None, impedance_declared=False)
    assert undeclared.allocation_dependence is AllocationDependence.UNAVAILABLE
    assert np.all(np.isnan(undeclared.values))
    assert "net joint torque" in (undeclared.reason or "")
    assert (
        variance_ratio_record(
            base, layered, impedance_declared=False
        ).allocation_dependence
        is AllocationDependence.UNAVAILABLE
    )


# --------------------------------------------------------------- helpers
def test_noise_inflation_scales_the_covariance_and_unit_scale_is_exact() -> None:
    sigma = np.stack([np.diag([4.0, 9.0])] * 3)
    np.testing.assert_array_equal(
        inflate_step_covariances(sigma, np.ones((3, 2))), sigma
    )
    scale = np.tile([1.5, 2.0], (3, 1))
    out = inflate_step_covariances(sigma, scale)
    np.testing.assert_allclose(out[0], np.diag([4.0 * 2.25, 9.0 * 4.0]))
    with pytest.raises(ValueError, match="scale"):
        inflate_step_covariances(sigma, np.ones((2, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        inflate_step_covariances(sigma, -np.ones((3, 2)))


def test_stiffening_trades_a_variance_benefit_against_the_declared_noise_cost() -> None:
    """The V5 ledger: stiffness helps, its declared noise penalty pushes back."""
    n, dt = 40, 5e-3
    window = ImpedanceWindow(0.0, 1.0, np.array([KP]), np.array([KD]))
    cheap = JointImpedance((window,))
    costly = JointImpedance((window,), noise_inflation=np.array([0.02]))
    times = (np.arange(n) + 0.5) * dt
    base_sigma = np.full((n, 1, 1), 0.8 / dt)
    gain = impedance_state_gain(cheap, times, np.eye(1), np.eye(1), np.zeros((1, 1)))
    assert gain.shape == (n, 1, 2) and np.allclose(gain[0], [[KP, KD]])

    def var(imp, scale):
        sig = inflate_step_covariances(base_sigma, scale)
        return layer_variance(
            _lin(n, dt, impedance_gain=gain),
            np.zeros((2, 2)),
            sig,
            np.array([[1.0, 0.0]]),
            _decl(),
            layer_hypotheses=imp.hypotheses,
            outcome_units="m",
        ).assessment.variance.values[0]

    free = var(cheap, cheap.noise_scale_series(times))
    paid = var(costly, costly.noise_scale_series(times))
    assert paid > free
    assert paid / free == pytest.approx((1.0 + 0.02 * KP) ** 2, rel=1e-12)


def test_impedance_state_gain_maps_the_tangent_state_to_joint_errors() -> None:
    imp = JointImpedance(
        (ImpedanceWindow(0.0, 1.0, np.array([5.0, 7.0]), np.array([1.0, 2.0])),)
    )
    p = np.array([[0.0], [1.0]])  # tangent along joint 2
    o = np.array([[0.5], [0.0]])  # velocity offset on joint 1
    gain = impedance_state_gain(imp, [0.5], np.eye(2), p, o)
    # e = S P xi = (0, xi);  de = S P eta + S O xi = (0.5 xi, eta)
    np.testing.assert_allclose(gain[0], [[0.5, 0.0], [7.0, 2.0]])
    with pytest.raises(ValueError, match="selection"):
        impedance_state_gain(imp, [0.5], np.eye(3), p, o)


def test_linearization_validation() -> None:
    f = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="dt"):
        LayerLinearization(f, np.zeros((2, 2, 1)), np.zeros((2, 2, 1)), dt=0.0)
    with pytest.raises(ValueError, match="delay_steps"):
        LayerLinearization(
            f, np.zeros((2, 2, 1)), np.zeros((2, 2, 1)), dt=0.1, delay_steps=-1
        )
    with pytest.raises(ValueError, match="method"):
        LayerLinearization(
            f,
            np.zeros((2, 2, 1)),
            np.zeros((2, 2, 1)),
            dt=0.1,
            method="rk4",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="feedback_gain"):
        LayerLinearization(
            f,
            np.zeros((2, 2, 1)),
            np.zeros((2, 2, 1)),
            dt=0.1,
            feedback_gain=np.zeros((2, 1, 3)),
        )
