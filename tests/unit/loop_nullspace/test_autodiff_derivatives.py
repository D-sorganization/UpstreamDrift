"""NSX-86 (#12715): the exact-differential KKT route of the derivative seam.

Engine-free oracles: the NSX-80 pendulum goldens (analytic differentials) and a
3-DoF configuration-dependent model whose differentials are written out by hand.
The route must equal the NSX-84 finite-difference reference on the identical
model, and must refuse (never fall back to finite differences) when the model
cannot supply exact differentials or the manifold is not Euclidean.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace import autodiff_derivatives as ad
from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.derivative_capability import (
    DerivativeCapabilityUnavailable,
    DerivativeMethod,
)
from tests.unit.loop_nullspace._tangent_fixtures import (
    Pendulum,
    frac,
    golden,
    tangent_field,
)

pytestmark = pytest.mark.unit

EUCLID2 = tl.EuclideanManifold(2)


class DifferentiablePendulum(Pendulum):
    """NSX-80 pendulum with its exact model differentials."""

    def differentials(self, q: np.ndarray, v: np.ndarray) -> ad.ModelDifferentials:
        n = 2
        d_jac = np.zeros((1, n, n))
        d_jac[0] = np.eye(n)  # J = q^T  ->  dJ_j/dq_k = delta_jk
        d_drift_dv = (2.0 * np.asarray(v)).reshape(1, n)
        return ad.ModelDifferentials(
            d_mass_dq=np.zeros((n, n, n)),
            d_bias_dq=np.zeros((n, n)),
            d_bias_dv=np.zeros((n, n)),
            d_jac_dq=d_jac,
            d_drift_dq=np.zeros((1, n)),
            d_drift_dv=d_drift_dv,
        )


class Coupled3:
    """3 DoF, one closure row; every term depends on ``q`` (and ``v``)."""

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return np.diag([1.0 + q[0] ** 2, 2.0 + 0.5 * q[1] ** 2, 3.0 + q[0] * q[2]])

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([np.sin(q[0]) + v[1] ** 2, q[0] * v[0] + q[2], 0.5 * q[2]])

    def closure_jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.array([[q[1], 1.0, q[2]]])

    def closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([v[0] * v[1] + q[0] * v[2] ** 2])

    def differentials(self, q: np.ndarray, v: np.ndarray) -> ad.ModelDifferentials:
        d_mass = np.zeros((3, 3, 3))
        d_mass[0, 0, 0] = 2.0 * q[0]
        d_mass[1, 1, 1] = q[1]
        d_mass[2, 2, 0], d_mass[2, 2, 2] = q[2], q[0]
        d_bias_dq = np.zeros((3, 3))
        d_bias_dq[0, 0] = np.cos(q[0])
        d_bias_dq[1, 0], d_bias_dq[1, 2], d_bias_dq[2, 2] = v[0], 1.0, 0.5
        d_bias_dv = np.zeros((3, 3))
        d_bias_dv[0, 1] = 2.0 * v[1]
        d_bias_dv[1, 0] = q[0]
        d_jac = np.zeros((1, 3, 3))
        d_jac[0, 0, 1], d_jac[0, 2, 2] = 1.0, 1.0
        d_drift_dq = np.array([[v[2] ** 2, 0.0, 0.0]])
        d_drift_dv = np.array([[v[1], v[0], 2.0 * q[0] * v[2]]])
        return ad.ModelDifferentials(
            d_mass, d_bias_dq, d_bias_dv, d_jac, d_drift_dq, d_drift_dv
        )

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.array([0.3, -0.4, 0.5])
        basis = tl.tangent_basis(self.closure_jacobian(q))
        v = basis @ np.array([0.7, -0.2])
        return q, v, np.array([0.5, -1.0, 2.0])


def _pair(model, manifold, q, v, tau, **kw):
    exact = ad.AutoDiffKKTDerivatives(model, manifold, **kw)
    ref = tl.FiniteDifferenceDerivatives(model, manifold, step=1e-5, **kw)
    return (
        exact.constrained_acceleration_derivatives(q, v, tau),
        ref.constrained_acceleration_derivatives(q, v, tau),
    )


def test_pendulum_matches_nsx80_golden_exactly() -> None:
    gold = golden()["linearization"]
    pend = DifferentiablePendulum()
    q, v = pend.state()
    for row in gold["allocation_table"]:
        prov = ad.AutoDiffKKTDerivatives(pend, EUCLID2, basis_field=tangent_field)
        res = prov.constrained_acceleration_derivatives(q, v, pend.load(frac(row["z"])))
        ss = tl.tangent_state_space(
            res, q=q, v=v, manifold=EUCLID2, basis_field=tangent_field
        )
        assert ss.F[1, 0] == pytest.approx(frac(row["F10_with_preload"]), abs=1e-9)
        assert ss.F[1, 1] == pytest.approx(frac(gold["F11"]), abs=1e-9)


def test_result_is_labelled_exact() -> None:
    pend = DifferentiablePendulum()
    q, v = pend.state()
    res = ad.AutoDiffKKTDerivatives(pend, EUCLID2).constrained_acceleration_derivatives(
        q, v, pend.load(0.0)
    )
    assert res.method is DerivativeMethod.AUTODIFF
    assert res.step is None
    assert "derivative_method=autodiff" in res.provenance_tag()
    assert "step=" not in res.provenance_tag()


def test_coupled_model_equals_the_finite_difference_reference() -> None:
    model = Coupled3()
    q, v, tau = model.state()
    exact, ref = _pair(model, tl.EuclideanManifold(3), q, v, tau)
    np.testing.assert_allclose(exact.basis, ref.basis, atol=1e-12)
    for name in (
        "acceleration",
        "reaction",
        "d_acc_d_xi",
        "d_acc_d_eta",
        "d_acc_d_tau",
        "d_lambda_d_xi",
        "d_lambda_d_eta",
        "d_lambda_d_tau",
    ):
        x, y = getattr(exact, name), getattr(ref, name)
        assert np.max(np.abs(x - y)) < 1e-6 * (1.0 + np.max(np.abs(y))), name


def test_coupled_model_closes_the_loop_at_first_order() -> None:
    model = Coupled3()
    q, v, tau = model.state()
    res = ad.AutoDiffKKTDerivatives(
        model, tl.EuclideanManifold(3)
    ).constrained_acceleration_derivatives(q, v, tau)
    assert np.linalg.norm(model.closure_jacobian(q) @ res.basis) < 1e-12
    assert res.d_lambda_d_xi.shape == (1, 2)
    assert np.max(np.abs(res.d_lambda_d_xi)) > 1e-3  # lambda is re-solved


def test_dropping_the_preload_would_change_the_answer() -> None:
    model = Coupled3()
    q, v, tau = model.state()
    prov = ad.AutoDiffKKTDerivatives(model, tl.EuclideanManifold(3))
    res = prov.constrained_acceleration_derivatives(q, v, tau)
    nom = prov._nominal(q, v, tau)
    ctx = prov._column_context(nom)
    assert np.linalg.norm(ctx.k_geom) > 1e-3
    zero = tl.KKTContext(np.zeros_like(ctx.k_geom), ctx.differentials)
    d_acc, _ = prov._perturbation(nom, zero, nom.basis[:, 0], nom.offset[:, 0])
    assert np.max(np.abs(d_acc - res.d_acc_d_xi[:, 0])) > 1e-6


def test_model_without_differentials_is_rejected_not_finite_differenced() -> None:
    pend = Pendulum()  # no `differentials`
    q, v = pend.state()
    with pytest.raises(DerivativeCapabilityUnavailable, match="differentials"):
        ad.AutoDiffKKTDerivatives(pend, EUCLID2)


def test_non_euclidean_manifold_is_rejected() -> None:
    with pytest.raises(DerivativeCapabilityUnavailable, match="Euclidean"):
        ad.AutoDiffKKTDerivatives(DifferentiablePendulum(), tl.FreeFlyerManifold())


def test_failing_differentials_raise_a_capability_rejection() -> None:
    class Broken(DifferentiablePendulum):
        def differentials(self, q, v):
            raise ad.DifferentialsUnavailable("component X cannot be differentiated")

    pend = Broken()
    q, v = pend.state()
    prov = ad.AutoDiffKKTDerivatives(pend, EUCLID2)
    with pytest.raises(DerivativeCapabilityUnavailable, match="component X"):
        prov.constrained_acceleration_derivatives(q, v, pend.load(0.0))


def test_wrong_shape_differentials_are_refused() -> None:
    with pytest.raises(ValueError, match="d_jac_dq"):
        ad.ModelDifferentials(
            np.zeros((2, 2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((1, 2, 3)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        )


def test_rank_drop_is_still_refused() -> None:
    class Drop(DifferentiablePendulum):
        def closure_jacobian(self, q):
            return np.array([[q[0], 0.0]])

    model = Drop()
    prov = ad.AutoDiffKKTDerivatives(model, EUCLID2, expected_rank=1)
    with pytest.raises(tl.ClosureRankChangeError):
        prov.constrained_acceleration_derivatives(
            np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.zeros(2)
        )
