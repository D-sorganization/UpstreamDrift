"""NSX-84 (#12713): constrained tangent linearization and the derivative seam.

Oracles: the exact NSX-80 variability goldens (pendulum ``F10 = omega^2 +
lambda(z)/m``; planar-golfer geometric preload block), an exact free-flyer
Jacobian, and the cross-agreement of the finite-difference reference with the
assembled KKT linearization (``loop_nullspace.tex``, ``sec:var-lin``).
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.derivative_capability import DerivativeMethod
from src.shared.python.loop_nullspace.fixtures import load_golden
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    PlanarGolferProvider,
)
from tests.unit.loop_nullspace._tangent_fixtures import (
    FreeFlyer,
    Pendulum,
    RankDropModel,
    frac,
    golden,
    tangent_field,
)

pytestmark = pytest.mark.unit

EUCLID2 = tl.EuclideanManifold(2)


def _derivs(kind: str, z: float, **kw: object):
    pend = Pendulum()
    q, v = pend.state()
    cls = tl.FiniteDifferenceDerivatives if kind == "fd" else tl.AssembledKKTDerivatives
    prov = cls(pend, EUCLID2, basis_field=tangent_field, **kw)
    return prov.constrained_acceleration_derivatives(q, v, pend.load(z)), q


def _state_space(kind: str, z: float, **kw: object):
    derivs, q = _derivs(kind, z, **kw)
    _, v = Pendulum().state()
    return tl.tangent_state_space(
        derivs, q=q, v=v, manifold=EUCLID2, basis_field=tangent_field
    )


@pytest.mark.parametrize("kind", ["fd", "assembled"])
def test_pendulum_f_and_g_match_nsx80_golden(kind: str) -> None:
    gold = golden()["linearization"]
    for row in gold["allocation_table"]:
        ss = _state_space(kind, frac(row["z"]))
        assert ss.F[0, 1] == pytest.approx(frac(gold["F01"]), abs=1e-9)
        assert ss.F[1, 1] == pytest.approx(frac(gold["F11"]), abs=1e-9)
        assert ss.F[1, 0] == pytest.approx(frac(row["F10_with_preload"]), abs=1e-9)
        assert ss.F[1, 0] == pytest.approx(
            1.0 + frac(row["lambda"]) / Pendulum.m, abs=1e-9
        )
    g_row = [frac(x) for x in gold["G_nominal_om_row"][0]]
    assert ss.G[1, :] == pytest.approx(g_row, abs=1e-9)
    assert not np.any(ss.G[0, :])


def test_g_annihilates_allocation_direction() -> None:
    pend = Pendulum()
    q, _ = pend.state()
    ss = _state_space("fd", 0.0)
    direction = -q  # actuator part of N = (-q, 1)
    assert abs(float(ss.G[1, :] @ direction)) < 1e-9


def test_reaction_is_resolved_not_frozen() -> None:
    d0, _ = _derivs("fd", 0.0)
    d1, _ = _derivs("fd", 0.5)
    assert d1.reaction[0] - d0.reaction[0] == pytest.approx(0.5, abs=1e-9)
    # dlambda/dxi is nonzero: lambda moves with the state, it is not held.
    assert np.max(np.abs(d0.d_lambda_d_xi)) > 1e-3


def test_dropping_preload_fails_the_golden() -> None:
    gold = golden()["linearization"]
    row = gold["allocation_table"][0]
    with mock.patch.object(tl, "geometric_preload", lambda *a, **k: np.zeros((2, 2))):
        ss = _state_space("assembled", frac(row["z"]))
    assert ss.F[1, 0] == pytest.approx(frac(row["F10_without_preload"]), abs=1e-9)
    assert abs(ss.F[1, 0] - frac(row["F10_with_preload"])) > 1.0


def test_geometric_preload_is_lambda_times_hessian() -> None:
    pend = Pendulum()
    q, _ = pend.state()
    kgeom = tl.geometric_preload(pend, EUCLID2, q, np.array([-3.5]))
    assert kgeom == pytest.approx(-3.5 * np.eye(2), abs=1e-8)


def _golfer():
    gold = load_golden("planar_two_arm_full")
    prov = PlanarGolferProvider(PlanarGolferParams.from_golden(gold))
    q = np.array(gold.payload["state"]["q"], dtype=float)
    v = np.array(gold.payload["state"]["v"], dtype=float)
    return prov, q, v


def test_golfer_projected_preload_matches_golden_and_varies() -> None:
    prov, q, _ = _golfer()
    man = tl.EuclideanManifold(5)
    proj = tl.tangent_projector(prov.closure_jacobian(q))
    blocks = []
    for alloc in golden()["golfer_preload"]["allocations"]:
        kgeom = tl.geometric_preload(prov, man, q, np.array(alloc["lambda"]))
        got = proj @ kgeom @ proj
        want = np.array(alloc["K_geom_projected"])
        assert np.max(np.abs(got - want)) < 1e-7
        blocks.append(got)
    assert np.linalg.norm(blocks[1] - blocks[0]) > 1e-3


def test_tangent_projector_is_basis_free_and_preserves_closure() -> None:
    prov, q, _ = _golfer()
    jac = prov.closure_jacobian(q)
    proj = tl.tangent_projector(jac)
    basis = tl.tangent_basis(jac)
    assert np.linalg.norm(jac @ proj @ jac.T) < 1e-12
    assert np.linalg.norm(jac @ basis) < 1e-12
    assert np.linalg.norm(proj @ proj - proj) < 1e-12
    rot, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    assert (basis @ rot) @ (basis @ rot).T == pytest.approx(proj, abs=1e-12)


def test_golfer_fd_and_assembled_agree() -> None:
    prov, q, v = _golfer()
    man = tl.EuclideanManifold(5)
    tau = prov.bias_forces(q, v) * 0.7 + 1.0
    fd = tl.FiniteDifferenceDerivatives(prov, man, step=1e-5)
    ak = tl.AssembledKKTDerivatives(prov, man, step=1e-5)
    a = fd.constrained_acceleration_derivatives(q, v, tau)
    b = ak.constrained_acceleration_derivatives(q, v, tau)
    for name in ("d_acc_d_xi", "d_acc_d_eta", "d_acc_d_tau"):
        x, y = getattr(a, name), getattr(b, name)
        assert np.max(np.abs(x - y)) < 1e-5 * (1.0 + np.max(np.abs(x)))
    # The tangent-bundle perturbation never leaves ker J_c at first order.
    jac = prov.closure_jacobian(q)
    assert np.linalg.norm(jac @ a.basis) < 1e-12


def test_fd_step_convergence_is_second_order() -> None:
    pend = Pendulum()
    q, v = pend.state()
    report = tl.fd_step_convergence(
        pend, EUCLID2, q, v, pend.load(0.0), steps=(1e-2, 5e-3, 2.5e-3)
    )
    assert report.converged
    assert report.observed_order > 1.7
    assert report.errors[-1] < report.errors[0]


def test_free_flyer_quaternion_derivatives_are_exact() -> None:
    body = FreeFlyer()
    man = tl.FreeFlyerManifold()
    assert (man.nq, man.nv) == (7, 6)
    quat = np.array([0.8, 0.2, -0.4, 0.3])
    q = np.concatenate([[0.1, -0.2, 0.3], quat / np.linalg.norm(quat)])
    v = np.array([0.3, -0.5, 0.2, 0.4, 0.1, -0.6])
    tau = np.array([1.0, 2.0, -1.0, 0.3, -0.2, 0.5])
    d_q, d_v = body.analytic_jacobians(q, v)
    for cls in (tl.FiniteDifferenceDerivatives, tl.AssembledKKTDerivatives):
        res = cls(body, man, step=1e-6).constrained_acceleration_derivatives(q, v, tau)
        assert res.basis.shape == (6, 6)
        assert res.d_acc_d_xi @ res.basis.T == pytest.approx(d_q, abs=1e-6)
        assert res.d_acc_d_eta @ res.basis.T == pytest.approx(d_v, abs=1e-6)


def test_free_flyer_retraction_is_on_the_manifold() -> None:
    man = tl.FreeFlyerManifold()
    q = np.array([0, 0, 0, 1.0, 0, 0, 0])
    assert man.retract(q, np.zeros(6)) == pytest.approx(q)
    moved = man.retract(q, np.array([0.1, 0.2, 0.3, 0.5, -0.4, 0.2]))
    assert np.linalg.norm(moved[3:]) == pytest.approx(1.0, abs=1e-14)
    assert moved[:3] == pytest.approx([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="shape"):
        man.retract(q, np.zeros(7))


@pytest.mark.parametrize("cls", ["fd", "assembled"])
def test_rank_drop_is_refused(cls: str) -> None:
    kind = tl.FiniteDifferenceDerivatives if cls == "fd" else tl.AssembledKKTDerivatives
    prov = kind(RankDropModel(), EUCLID2, step=1e-4)
    with pytest.raises(tl.ClosureRankChangeError, match="rank"):
        prov.constrained_acceleration_derivatives(np.zeros(2), np.zeros(2), np.zeros(2))


def test_declared_rank_mismatch_is_refused() -> None:
    pend = Pendulum()
    q, v = pend.state()
    prov = tl.FiniteDifferenceDerivatives(pend, EUCLID2, expected_rank=2)
    with pytest.raises(tl.ClosureRankChangeError, match="expected rank 2"):
        prov.constrained_acceleration_derivatives(q, v, pend.load(0.0))


def test_contact_mode_change_is_refused() -> None:
    class Modal(Pendulum):
        def contact_mode(self, q: np.ndarray, v: np.ndarray) -> str:
            return "stance" if q[0] <= 0.6 else "flight"

    pend = Modal()
    q, v = pend.state()
    prov = tl.FiniteDifferenceDerivatives(pend, EUCLID2, step=1e-4)
    with pytest.raises(tl.ClosureRankChangeError, match="contact mode"):
        prov.constrained_acceleration_derivatives(q, v, pend.load(0.0))


def test_provenance_records_derivative_method() -> None:
    d_fd, _ = _derivs("fd", 0.0, step=2e-6)
    assert d_fd.method is DerivativeMethod.FINITE_DIFFERENCE
    tag = d_fd.provenance_tag()
    assert "derivative_method=finite_difference" in tag
    assert "scheme=central_retraction" in tag
    assert "step=2e-06" in tag
    d_kkt, _ = _derivs("assembled", 0.0)
    assert "scheme=assembled_kkt" in d_kkt.provenance_tag()


def test_preconditions_raise() -> None:
    pend = Pendulum()
    q, v = pend.state()
    prov = tl.FiniteDifferenceDerivatives(pend, EUCLID2)
    tau = pend.load(0.0)
    with pytest.raises(ValueError, match="tangent"):
        prov.constrained_acceleration_derivatives(q, np.array([1.0, 1.0]), tau)
    with pytest.raises(ValueError, match="shape"):
        prov.constrained_acceleration_derivatives(q[:1], v, tau)
    with pytest.raises(ValueError, match="finite"):
        prov.constrained_acceleration_derivatives(q, v, np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="step"):
        tl.FiniteDifferenceDerivatives(pend, EUCLID2, step=0.0)
    with pytest.raises(TypeError, match="callable"):
        tl.FiniteDifferenceDerivatives(pend, EUCLID2, basis_field=3)


def test_state_space_rejects_a_closed_chain_without_freedom() -> None:
    """``k = 0`` has no tangent state; the precondition names the dimension."""
    jac_rank_full = tl.FiniteDifferenceDerivatives(
        Pendulum(), EUCLID2
    ).constrained_acceleration_derivatives(*Pendulum().state(), Pendulum().load(0.0))
    object.__setattr__(
        jac_rank_full, "basis", np.zeros((2, 0))
    )  # frozen dataclass: emulate a rigid (k = 0) closure
    with pytest.raises(ValueError, match="tangent dimension"):
        tl.tangent_state_space(
            jac_rank_full,
            q=Pendulum().state()[0],
            v=Pendulum().state()[1],
            manifold=EUCLID2,
            basis_field=tangent_field,
        )
