"""NSX-84b (#12817): tangent state space for ``k > 1`` (planar five-bar, k = 2).

Oracle: ``planar_five_bar_tangent_state_space`` -- an independent reduced-
coordinate derivation (no KKT system, no production code; 40-digit arithmetic).
See ``loop_nullspace.tex`` ``sec:var-lin-k``.

Tolerance derivation (used by every golden comparison below).  The derivative
layer differences the single KKT solve with a central step ``h = 1e-5``:

* truncation ``h^2 |f'''| / 6``; the five-bar solution has ``|f'''| <~ 1e3`` in
  these units, so ``<~ 2e-8``;
* round-off ``eps |f| / h`` with ``eps = 1.1e-16``, ``|f| <~ 60`` (the largest
  entry of ``a``), so ``<~ 7e-10``;

and the frame derivative ``d_q P`` uses the same ``h``.  Each entry of ``F`` is a
sum of at most three such terms weighted by the unit-norm rows of ``P^T``, so
the bound is ``3 * (2e-8 + 7e-10) ~ 6e-8`` and ``ATOL = 1e-7`` leaves a factor
1.6.  Measured on this model: ``2.0e-8`` (finite-difference route) and
``1.6e-9`` (assembled KKT route) against entries up to ``53``.  The gauge-law
test composes two such evaluations (``P`` and ``P R``), hence ``2 * ATOL``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace import tangent_linearization as tl
from tests.unit.loop_nullspace._tangent_fixtures import FiveBar, five_bar_golden

pytestmark = pytest.mark.unit

STEP = 1e-5
ATOL = 1e-7  # derived in the module docstring
MAN = tl.EuclideanManifold(4)


def _derivs(kind: str, basis_field=None):
    bar = FiveBar()
    q, v, tau = bar.state()
    cls = tl.FiniteDifferenceDerivatives if kind == "fd" else tl.AssembledKKTDerivatives
    field = basis_field or bar.frame
    prov = cls(bar, MAN, basis_field=field, step=STEP)
    return prov.constrained_acceleration_derivatives(q, v, tau), q, v, field


def _state_space(kind: str, basis_field=None):
    derivs, q, v, field = _derivs(kind, basis_field)
    return tl.tangent_state_space(
        derivs, q=q, v=v, manifold=MAN, basis_field=field, step=STEP
    )


@pytest.mark.parametrize("kind", ["fd", "assembled"])
def test_five_bar_f_and_g_match_symbolic_golden(kind: str) -> None:
    gold = five_bar_golden()
    ss = _state_space(kind)
    assert ss.F.shape == (4, 4)
    assert ss.G.shape == (4, 4)
    assert np.max(np.abs(ss.F - np.array(gold["F"]))) < ATOL
    assert np.max(np.abs(ss.G - np.array(gold["G"]))) < ATOL


def test_connection_blocks_match_golden() -> None:
    gold = five_bar_golden()
    ss = _state_space("fd")
    assert np.max(np.abs(ss.connection - np.array(gold["Gamma_v"]))) < ATOL
    assert np.max(np.abs(ss.normal_offset - np.array(gold["offset_term"]))) < ATOL


def test_connection_is_skew_and_normal_offset_symmetric_psd() -> None:
    ss = _state_space("fd")
    assert np.max(np.abs(ss.connection + ss.connection.T)) < 1e-8
    assert np.max(np.abs(ss.normal_offset - ss.normal_offset.T)) < 1e-8
    assert (
        np.linalg.eigvalsh(0.5 * (ss.normal_offset + ss.normal_offset.T)).min() > -1e-8
    )


def test_literal_acceleration_connection_is_not_the_k2_connection() -> None:
    """The k = 1 form ``(d_q P_i [P e_j]) . a`` fails the oracle for ``k = 2``."""
    gold = five_bar_golden()
    ss = _state_space("fd")
    derivs, _, _, _ = _derivs("fd")
    f10_literal = derivs.basis.T @ derivs.d_acc_d_xi + np.array(
        gold["connection_literal"]
    )
    f10_golden = np.array(gold["F"])[2:, :2]
    assert np.max(np.abs(f10_literal - f10_golden)) > 1.0
    assert np.max(np.abs(ss.F[2:, :2] - f10_golden)) < ATOL


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def test_constant_frame_rotation_gives_a_similar_f_and_g() -> None:
    """``P' = P R`` (constant ``R``) maps ``F -> T^T F T``: same eigenvalues.

    The map ``(xi, eta) -> (R^T xi, R^T eta)`` is exact for a constant ``R``;
    eigenvalues are invariant under similarity, so they agree to the same
    numerical error as ``F`` itself (``< ATOL`` per entry gives an eigenvalue
    shift of at most ``4 * ATOL`` by Bauer-Fike for the well-conditioned
    block-diagonal rotation, whose condition number is one).
    """
    rot = _rotation(0.7)
    ss0 = _state_space("fd")
    ss1 = _state_space("fd", basis_field=lambda q: FiveBar().frame(q) @ rot)
    big = np.kron(np.eye(2), rot)
    assert np.max(np.abs(ss1.F - big.T @ ss0.F @ big)) < ATOL
    assert np.max(np.abs(ss1.G - big.T @ ss0.G)) < ATOL
    e0 = np.sort_complex(np.linalg.eigvals(ss0.F))
    e1 = np.sort_complex(np.linalg.eigvals(ss1.F))
    assert np.max(np.abs(e0 - e1)) < 4 * ATOL * np.linalg.cond(big)


def test_position_dependent_rotation_follows_the_gauge_law() -> None:
    """``P' = P R(q)``: ``F' = T^T F T + blkdiag(dR^T R, dR^T R)`` on the loops.

    ``xi' = R^T xi`` gives ``xi'_dot = R^T xi_dot + (dR/dt)^T xi`` and likewise
    for ``eta``.  ``dR/dt = (g . v) R'(phi)`` for ``phi(q) = phi0 + g . q``.
    """
    bar = FiveBar()
    q, v, _ = bar.state()
    gvec = np.array([0.3, -0.2, 0.1, 0.25])

    def field(x: np.ndarray) -> np.ndarray:
        return bar.frame(x) @ _rotation(0.4 + gvec @ x)

    ss0 = _state_space("fd")
    ss1 = _state_space("fd", basis_field=field)
    rot = _rotation(0.4 + gvec @ q)
    big = np.kron(np.eye(2), rot)
    rate = float(gvec @ v)
    d_rot = rate * np.array([[0.0, -1.0], [1.0, 0.0]]) @ rot
    shift = np.kron(np.eye(2), d_rot.T @ rot)
    # the frame of ss0 is P0; ss1 uses P0 R(q), so P1 = P0 R and
    # zeta1 = (R^T xi0, R^T eta0).
    want = big.T @ ss0.F @ big + shift
    assert np.max(np.abs(ss1.F - want)) < 2 * ATOL


def test_k1_pendulum_signature_requires_velocity() -> None:
    from tests.unit.loop_nullspace._tangent_fixtures import Pendulum, tangent_field

    pend = Pendulum()
    q, v = pend.state()
    man = tl.EuclideanManifold(2)
    derivs = tl.FiniteDifferenceDerivatives(
        pend, man, basis_field=tangent_field
    ).constrained_acceleration_derivatives(q, v, pend.load(0.0))
    with pytest.raises(ValueError, match="velocity"):
        tl.tangent_state_space(
            derivs, q=q, v=v[:1], manifold=man, basis_field=tangent_field
        )
    with pytest.raises(ValueError, match="tangent"):
        tl.tangent_state_space(
            derivs, q=q, v=q, manifold=man, basis_field=tangent_field
        )
    with pytest.raises(ValueError, match="basis"):
        tl.tangent_state_space(
            derivs, q=q, v=v, manifold=man, basis_field=lambda x: np.eye(2)[:, :1]
        )
