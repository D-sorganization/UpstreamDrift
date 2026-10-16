"""NSX-94 (#12723): signal-dependent muscle noise and short-range stiffness.

Closed forms used as oracles:

* Noise.  ``Cov(f_k) = kappa_k^2 F_max,k^2 a_k^2`` (S0 level, plus ``sigma0^2``),
  pushed to the joint by ``R^T diag(.) R``; discrete covariance ``/ dt``.
* Series stiffness.  ``k = k_m k_T / (k_m + k_T)``; ``k_T -> inf`` gives ``k_m``,
  ``k_m = 0`` gives ``0``.
* Joint stiffness.  ``K = R^T diag(k) R - sum_k F_k dR_k/dq`` (geometry term).
* Linearised dynamics.  Differentiating ``a(q) = M^-1 (tau0 - K (q - q0) - h)``
  gives ``d a / d xi -> d a / d xi - (d a / d tau) K P``; the oracle below is a
  central difference of the *explicit* loaded closed-loop acceleration, which
  shares no code with :func:`apply_muscle_stiffness`.

Tolerance for finite-difference oracles: central difference of a smooth function
with step ``h = 1e-5`` has truncation ``h^2 f'''/6 ~ 1e-11 |f'''|`` and round-off
``eps |f| / h ~ 2e-11 |f|``; the fixtures have ``|f|, |f'''| <= 1e2``, so
``1e-8`` absolute is a factor ~5e2 above the estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.constrained_dynamics import (
    constrained_acceleration,
)
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame, TendonKind
from src.shared.python.loop_nullspace.muscle_noise import (
    joint_torque_covariance,
    muscle_noise_spec,
)
from src.shared.python.loop_nullspace.muscle_stiffness import (
    StiffnessModel,
    StiffnessNotQualified,
    apply_muscle_stiffness,
    muscle_stiffness,
    series_stiffness,
)
from src.shared.python.perturbation.disturbance import InjectionStage
from tests.unit.loop_nullspace._muscle_fixtures import (
    INPUT,
    MODEL,
    OneJoint,
    PAIR_ARM,
    PAIR_FORCE,
    PAIR_KAPPA,
    PAIR_RHO,
    elbow_frame,
    pair_frame,
)
from tests.unit.loop_nullspace._tangent_fixtures import Pendulum, tangent_field

pytestmark = pytest.mark.unit

FD_TOL = 1e-8


def test_noise_spec_declares_the_signal_dependent_law() -> None:
    frame = pair_frame()
    spec = muscle_noise_spec(frame, kappa=PAIR_KAPPA, sigma0=0.5)
    assert spec.channels == frame.names
    assert spec.has_signal_law and spec.stage is InjectionStage.ACTIVATION
    assert spec.signal_bounds == ((0.0, 1.0), (0.0, 1.0))
    a = np.array([0.6, 0.2])
    expected = np.diag(0.25 + (PAIR_KAPPA * PAIR_FORCE * a) ** 2)
    np.testing.assert_allclose(spec.covariance_at(a), expected, rtol=1e-12)


def test_joint_torque_covariance_closed_form_and_dt_scaling() -> None:
    frame = pair_frame()
    spec = muscle_noise_spec(frame, kappa=PAIR_KAPPA)
    a = np.array([0.6, 0.2])
    dt = 1e-3
    got = joint_torque_covariance(frame, spec, a, dt=dt)
    expected = PAIR_ARM**2 * (PAIR_KAPPA * PAIR_FORCE) ** 2 * (0.6**2 + 0.2**2) / dt
    assert got.shape == (1, 1)
    assert got[0, 0] == pytest.approx(expected, rel=1e-12)
    half = joint_torque_covariance(frame, spec, a, dt=dt / 2)
    assert half[0, 0] == pytest.approx(2 * got[0, 0], rel=1e-12)


def test_noise_is_quadratic_in_activation_with_zero_floor() -> None:
    frame = elbow_frame()
    spec = muscle_noise_spec(frame, kappa=PAIR_KAPPA)
    a = np.array([0.3, 0.2, 0.1])
    one = joint_torque_covariance(frame, spec, a, dt=1.0)[0, 0]
    two = joint_torque_covariance(frame, spec, 2 * a, dt=1.0)[0, 0]
    assert two == pytest.approx(4 * one, rel=1e-12)
    zero = joint_torque_covariance(frame, spec, np.zeros(3), dt=1.0)[0, 0]
    assert zero == 0.0


def test_noise_spec_requires_matching_frame() -> None:
    spec = muscle_noise_spec(pair_frame(), kappa=PAIR_KAPPA)
    with pytest.raises(ValueError, match="channels"):
        joint_torque_covariance(elbow_frame(), spec, np.zeros(3), dt=1e-3)
    with pytest.raises(ValueError, match="kappa"):
        muscle_noise_spec(pair_frame(), kappa=-1.0)


def test_series_stiffness_limits() -> None:
    k_m = np.array([100.0, 0.0, 5.0, 1e4])
    k_t = np.array([100.0, 50.0, np.inf, 1.0])
    got = series_stiffness(k_m, k_t)
    np.testing.assert_allclose(got[:3], [50.0, 0.0, 5.0], rtol=1e-12)
    assert got[3] == pytest.approx(1e4 / (1e4 + 1.0), rel=1e-12)
    assert got[3] < 1.0  # never stiffer than the compliant series element
    with pytest.raises(ValueError, match="positive"):
        series_stiffness(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        series_stiffness(np.array([-1.0]), np.array([1.0]))


def test_force_proportional_joint_stiffness_closed_form() -> None:
    frame = pair_frame()
    a = np.array([0.6, 0.2])
    stiff = muscle_stiffness(
        frame, a, StiffnessModel.FORCE_PROPORTIONAL, force_gain=PAIR_RHO
    )
    expected = PAIR_ARM**2 * PAIR_RHO * PAIR_FORCE * (0.6 + 0.2)
    assert stiff.joint[0, 0] == pytest.approx(expected, rel=1e-12)
    assert stiff.geometric_term_included is False
    np.testing.assert_allclose(stiff.muscle, PAIR_RHO * PAIR_FORCE * a, rtol=1e-12)


def _engine_table_frame(*, compliant: bool, k_fiber: float, k_tendon: float):
    forces = np.array([[0.0, 0.0], [PAIR_FORCE, PAIR_FORCE]])
    return MuscleFrame(
        names=("flexor", "extensor"),
        coordinates=("joint",),
        moment_arms=np.array([[PAIR_ARM], [-PAIR_ARM]]),
        activation_grid=np.array([0.0, 1.0]),
        force_grid=forces,
        max_isometric_force=np.array([PAIR_FORCE, PAIR_FORCE]),
        tendon=TendonKind.COMPLIANT if compliant else TendonKind.RIGID,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
        fiber_stiffness_grid=np.array([[0.0, 0.0], [k_fiber, k_fiber]]),
        tendon_stiffness_grid=(np.full((2, 2), k_tendon) if compliant else None),
    )


def test_engine_curve_with_compliant_tendon_uses_series_formula() -> None:
    frame = _engine_table_frame(compliant=True, k_fiber=2.0e4, k_tendon=1.0e4)
    a = np.array([1.0, 0.5])
    stiff = muscle_stiffness(frame, a, StiffnessModel.ENGINE_CURVE)
    k_m = np.array([2.0e4, 1.0e4])  # affine in activation by the grid
    k_series = k_m * 1.0e4 / (k_m + 1.0e4)
    np.testing.assert_allclose(stiff.muscle, k_series, rtol=1e-12)
    assert stiff.joint[0, 0] == pytest.approx(PAIR_ARM**2 * k_series.sum(), rel=1e-12)
    assert stiff.tendon_kind is TendonKind.COMPLIANT


def test_rigid_tendon_is_the_infinite_tendon_stiffness_limit() -> None:
    rigid = _engine_table_frame(compliant=False, k_fiber=2.0e4, k_tendon=0.0)
    a = np.array([1.0, 0.5])
    stiff = muscle_stiffness(rigid, a, StiffnessModel.ENGINE_CURVE)
    np.testing.assert_allclose(stiff.muscle, [2.0e4, 1.0e4], rtol=1e-12)


def test_negative_engine_stiffness_fails_closed() -> None:
    frame = _engine_table_frame(compliant=True, k_fiber=-5.0e3, k_tendon=1.0e4)
    with pytest.raises(StiffnessNotQualified, match="negative"):
        muscle_stiffness(frame, np.array([1.0, 1.0]), StiffnessModel.ENGINE_CURVE)


def test_missing_engine_data_fails_closed_with_a_reason() -> None:
    with pytest.raises(StiffnessNotQualified, match="fibre stiffness"):
        muscle_stiffness(
            pair_frame(), np.array([0.5, 0.5]), StiffnessModel.ENGINE_CURVE
        )
    compliant_no_k = MuscleFrame(
        names=("flexor", "extensor"),
        coordinates=("joint",),
        moment_arms=np.array([[PAIR_ARM], [-PAIR_ARM]]),
        activation_grid=np.array([0.0, 1.0]),
        force_grid=np.array([[0.0, 0.0], [PAIR_FORCE, PAIR_FORCE]]),
        max_isometric_force=np.array([PAIR_FORCE, PAIR_FORCE]),
        tendon=TendonKind.COMPLIANT,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
    )
    with pytest.raises(StiffnessNotQualified, match="tendon stiffness"):
        muscle_stiffness(
            compliant_no_k,
            np.array([0.5, 0.5]),
            StiffnessModel.FORCE_PROPORTIONAL,
            force_gain=PAIR_RHO,
        )
    with pytest.raises(ValueError, match="force_gain"):
        muscle_stiffness(
            pair_frame(), np.array([0.5, 0.5]), StiffnessModel.FORCE_PROPORTIONAL
        )


def test_geometry_term_adds_minus_force_times_moment_arm_derivative() -> None:
    base = pair_frame()
    jac = np.zeros((2, 1, 1))
    jac[0, 0, 0] = 0.01  # dR_flexor/dq
    jac[1, 0, 0] = -0.02  # dR_extensor/dq
    frame = MuscleFrame(
        names=base.names,
        coordinates=base.coordinates,
        moment_arms=base.moment_arms,
        activation_grid=base.activation_grid,
        force_grid=base.force_grid,
        max_isometric_force=base.max_isometric_force,
        tendon=base.tendon,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
        moment_arm_jacobian=jac,
    )
    a = np.array([0.6, 0.2])
    with_geo = muscle_stiffness(
        frame, a, StiffnessModel.FORCE_PROPORTIONAL, force_gain=PAIR_RHO
    )
    without = muscle_stiffness(
        base, a, StiffnessModel.FORCE_PROPORTIONAL, force_gain=PAIR_RHO
    )
    force = PAIR_FORCE * a
    delta = -(force[0] * 0.01 + force[1] * -0.02)
    assert with_geo.geometric_term_included
    assert with_geo.joint[0, 0] - without.joint[0, 0] == pytest.approx(delta, rel=1e-12)


# --------------------------------------------------- linearised dynamics oracle


def _k_matrix() -> np.ndarray:
    return np.array([[3.0, 0.5], [0.5, 2.0]])


def test_stiffness_changes_the_linearised_dynamics_of_the_loop() -> None:
    pend = Pendulum()
    manifold = tl.EuclideanManifold(2)
    q, v = pend.state()
    tau = pend.load(0.0)
    routine = tl.FiniteDifferenceDerivatives(pend, manifold, basis_field=tangent_field)
    base = routine.constrained_acceleration_derivatives(q, v, tau)
    k_mat = _k_matrix()
    loaded = apply_muscle_stiffness(base, k_mat, coordinate_indices=(0, 1))
    assert loaded.method is base.method
    # Nominal motion and load response are untouched by a pure stiffness.
    np.testing.assert_allclose(loaded.acceleration, base.acceleration)
    np.testing.assert_allclose(loaded.d_acc_d_tau, base.d_acc_d_tau)
    assert not np.allclose(loaded.d_acc_d_xi, base.d_acc_d_xi)

    # Oracle: central difference of the explicitly loaded closed-loop solve.
    basis = base.basis
    h = 1e-5

    def loaded_solve(xi: float) -> tuple[np.ndarray, np.ndarray]:
        qx = manifold.retract(q, xi * basis[:, 0])
        tau_x = tau - k_mat @ (qx - q)
        return constrained_acceleration(
            pend.mass_matrix(qx),
            pend.bias_forces(qx, v),
            tau_x,
            pend.closure_jacobian(qx),
            pend.closure_drift(qx, v),
        )

    plus, minus = loaded_solve(h), loaded_solve(-h)
    d_acc = (plus[0] - minus[0]) / (2 * h)
    d_lam = (plus[1] - minus[1]) / (2 * h)
    np.testing.assert_allclose(loaded.d_acc_d_xi[:, 0], d_acc, atol=FD_TOL)
    np.testing.assert_allclose(loaded.d_lambda_d_xi[:, 0], d_lam, atol=FD_TOL)


def test_stiffness_moves_the_scalar_tangent_state_space_entry() -> None:
    manifold = tl.EuclideanManifold(1)
    model = OneJoint()
    q, v, tau = np.array([0.4]), np.array([0.0]), np.array([1.0])
    routine = tl.FiniteDifferenceDerivatives(model, manifold)
    base = routine.constrained_acceleration_derivatives(q, v, tau)
    k = 80.0
    loaded = apply_muscle_stiffness(base, np.array([[k]]), coordinate_indices=(0,))
    field = lambda _q: np.array([[1.0]])  # noqa: E731
    f0 = tl.tangent_state_space(base, q=q, v=v, manifold=manifold, basis_field=field).F
    f1 = tl.tangent_state_space(
        loaded, q=q, v=v, manifold=manifold, basis_field=field
    ).F
    inertia = float(model.mass_matrix(q)[0, 0])
    assert f0[1, 0] == pytest.approx(0.0, abs=FD_TOL)
    assert f1[1, 0] == pytest.approx(-k / inertia, rel=1e-8)
    assert f1[1, 1] == pytest.approx(f0[1, 1], abs=FD_TOL)


def test_apply_stiffness_validates_inputs() -> None:
    pend = Pendulum()
    q, v = pend.state()
    base = tl.FiniteDifferenceDerivatives(
        pend, tl.EuclideanManifold(2), basis_field=tangent_field
    ).constrained_acceleration_derivatives(q, v, pend.load(0.0))
    with pytest.raises(ValueError, match="symmetric"):
        apply_muscle_stiffness(
            base, np.array([[1.0, 2.0], [0.0, 1.0]]), coordinate_indices=(0, 1)
        )
    with pytest.raises(ValueError, match="coordinate_indices"):
        apply_muscle_stiffness(base, np.array([[1.0]]), coordinate_indices=(3,))
    with pytest.raises(ValueError, match="finite"):
        apply_muscle_stiffness(base, np.array([[np.nan]]), coordinate_indices=(0,))
    embedded = apply_muscle_stiffness(base, np.array([[3.0]]), coordinate_indices=(1,))
    full = np.diag([0.0, 3.0])
    np.testing.assert_allclose(
        embedded.d_acc_d_xi,
        base.d_acc_d_xi - base.d_acc_d_tau @ full @ base.basis,
        atol=1e-12,
    )


def test_zero_kappa_declares_no_signal_law_and_is_activation_independent() -> None:
    frame = pair_frame()
    spec = muscle_noise_spec(frame, kappa=0.0, sigma0=0.5)
    assert not spec.has_signal_law
    low = joint_torque_covariance(frame, spec, [0.1, 0.1], dt=1e-3)
    high = joint_torque_covariance(frame, spec, [0.9, 0.9], dt=1e-3)
    np.testing.assert_allclose(low, high, rtol=0.0, atol=0.0)
    expected = 2 * PAIR_ARM**2 * 0.5**2 / 1e-3
    np.testing.assert_allclose(low, [[expected]], rtol=1e-12)
