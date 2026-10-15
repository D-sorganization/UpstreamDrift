"""NumPy-vs-JAX parity of the golfer kinematics and closure (#12669).

#12608 fixed the club y-sign flip in the NumPy golfer path
(``golfer_kinematics.point_on_club``).  ``physics_golfer_jax`` is a separate
JAX/GPU copy of the same model; these tests pin it to the NumPy reference:

* forward-kinematics points, position Jacobians, mass matrix and gravity
  vector agree with NumPy to 1e-10 (float64) over random states;
* ``Phi`` and the analytic ``Phi_q`` agree with NumPy to 1e-10;
* the JAX ``Phi_q`` and club-point Jacobians match central differences of the
  JAX functions themselves and ``jax.jacfwd``;
* on closed configurations the JAX closure vanishes and ``Phi_q`` has rank 2
  (rank 3 off the manifold), matching #12608.

The NumPy reference is ``forward_kinematics``, which uses the native backend
when it is enabled; the native kernel is pinned to the same convention by the
#12608 tests, so the parity holds on either backend.

Tolerances: parity compares two closed-form float64 evaluations of the same
formulas (error ~1e-15), so 1e-10 is a wide margin that still catches any
sign or offset error (O(0.1)).  Central differences use h = 1e-6: truncation
(h^2/6)|f'''| <= ~2e-13 for lengths <= 1 m, round-off eps*|f|/h ~ 4e-10, so
FD_TOL = 1e-8 keeps a >20x margin.

Requires JAX (``requires_jax``); the whole module skips cleanly without it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from scipy.optimize import least_squares

jax = pytest.importorskip("jax")
if not hasattr(jax, "config"):  # pragma: no cover - defensive against stubs
    pytest.skip("jax is replaced by a stub in this session", allow_module_level=True)

from src.shared.python.pendulum_simulator import (  # noqa: E402
    physics_golfer_jax as pgj,
)
from src.shared.python.pendulum_simulator.golfer_constraints import (  # noqa: E402
    INDEPENDENT_CLOSURE_ROWS,
    analytical_constraint_jacobian,
    constraint_vector,
)
from src.shared.python.pendulum_simulator.golfer_dynamics import (  # noqa: E402
    analytical_fk_jacobians,
    analytical_gravity_vector,
    analytical_mass_matrix,
)
from src.shared.python.pendulum_simulator.golfer_kinematics import (  # noqa: E402
    forward_kinematics,
)
from src.shared.python.pendulum_simulator.physics_golfer import (  # noqa: E402
    N_DOF,
    GolferParams,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]

PARITY_TOL = 1e-10
FD_STEP = 1e-6
FD_TOL = 1e-8
AD_TOL = 1e-12
RANK_TOL = 1e-8
N_RANDOM_STATES = 16
FK_KEYS = (
    "hub",
    "rs",
    "re",
    "rh",
    "ls",
    "le",
    "lh",
    "club_base",
    "club_tip",
    "grip_right",
    "grip_left",
)
JACOBIAN_KEYS = ("hub", "rs", "re", "rh", "ls", "le", "lh", "club_com", "club_tip")
# grip_right = 0.6 > L_club / 2 flips the sign of the club-COM offset, so the
# club-angle column is exercised with both signs.
GRIP_RIGHTS = (0.05, 0.0, 0.6)


@pytest.fixture(scope="module", autouse=True)
def _enable_x64() -> Iterator[None]:
    """Run this module in float64 and restore the global JAX flag afterwards."""
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _params(grip_right: float = 0.05) -> GolferParams:
    return GolferParams(
        m_hub=10.0,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.4,
        L_hub=0.2,
        L_r_upper=0.3,
        L_r_fore=0.25,
        L_l_upper=0.3,
        L_l_fore=0.25,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=grip_right,
        grip_left=grip_right + 0.1,
    )


def _jax_params(p: GolferParams) -> Any:
    fields = pgj.GolferParamsJAX._fields
    return pgj.GolferParamsJAX(**{name: getattr(p, name) for name in fields})


def _random_states(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-np.pi, np.pi, N_DOF) for _ in range(N_RANDOM_STATES)]


def _closed_solutions(p: GolferParams, count: int = 8) -> list[np.ndarray]:
    """Closed configurations of the NumPy reference closure."""
    rng = np.random.default_rng(0)
    out: list[np.ndarray] = []
    while len(out) < count:
        res = least_squares(
            lambda x: constraint_vector(x, p),
            rng.uniform(-3.0, 3.0, N_DOF),
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        if res.cost < 1e-20:
            out.append(res.x)
    return out


def _central_fd(func: Callable[[np.ndarray], Any], q: np.ndarray) -> np.ndarray:
    cols = []
    for j in range(q.shape[0]):
        step = np.zeros_like(q)
        step[j] = FD_STEP
        plus = np.asarray(func(q + step))
        minus = np.asarray(func(q - step))
        cols.append((plus - minus) / (2.0 * FD_STEP))
    return np.stack(cols, axis=-1)


def _jnp(q: np.ndarray) -> Any:
    return jax.numpy.asarray(q, dtype=jax.numpy.float64)


@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
def test_forward_kinematics_matches_numpy(grip_right: float) -> None:
    p = _params(grip_right)
    pj = _jax_params(p)
    for q in _random_states(11):
        ref = forward_kinematics(q, p)
        got = pgj.forward_kinematics_jax(_jnp(q), pj)
        assert got["club_tip"].dtype == jax.numpy.float64
        for key in FK_KEYS:
            np.testing.assert_allclose(
                np.asarray(got[key]),
                np.asarray(ref[key]),
                atol=PARITY_TOL,
                rtol=0.0,
                err_msg=f"{key} at q={q}",
            )


@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
def test_position_jacobians_match_numpy(grip_right: float) -> None:
    p = _params(grip_right)
    pj = _jax_params(p)
    for q in _random_states(12):
        ref = analytical_fk_jacobians(q, p)
        got = pgj.analytical_fk_jacobians_jax(_jnp(q), pj)
        for key in JACOBIAN_KEYS:
            np.testing.assert_allclose(
                np.asarray(got[key]),
                ref[key],
                atol=PARITY_TOL,
                rtol=0.0,
                err_msg=f"J_{key} at q={q}",
            )


def test_mass_matrix_and_gravity_match_numpy() -> None:
    p = _params()
    pj = _jax_params(p)
    for q in _random_states(13):
        np.testing.assert_allclose(
            np.asarray(pgj.mass_matrix_jax(_jnp(q), pj)),
            analytical_mass_matrix(q, p),
            atol=PARITY_TOL,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(pgj.gravity_vector_jax(_jnp(q), pj)),
            analytical_gravity_vector(q, p),
            atol=PARITY_TOL,
            rtol=0.0,
        )


@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
def test_constraint_vector_and_jacobian_match_numpy(grip_right: float) -> None:
    p = _params(grip_right)
    pj = _jax_params(p)
    for q in _random_states(14):
        np.testing.assert_allclose(
            np.asarray(pgj.constraint_vector_jax(_jnp(q), pj)),
            constraint_vector(q, p),
            atol=PARITY_TOL,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(pgj.constraint_jacobian_jax(_jnp(q), pj)),
            analytical_constraint_jacobian(q, p),
            atol=PARITY_TOL,
            rtol=0.0,
        )


@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
def test_constraint_jacobian_matches_central_difference(grip_right: float) -> None:
    p = _params(grip_right)
    pj = _jax_params(p)
    states = [*_random_states(15)[:4], *_closed_solutions(p, 4)]
    for q in states:
        fd = _central_fd(lambda x: pgj.constraint_vector_jax(_jnp(x), pj), q)
        ana = np.asarray(pgj.constraint_jacobian_jax(_jnp(q), pj))
        np.testing.assert_allclose(ana, fd, atol=FD_TOL, rtol=0.0)
        ad = np.asarray(jax.jacfwd(lambda x: pgj.constraint_vector_jax(x, pj))(_jnp(q)))
        np.testing.assert_allclose(ana, ad, atol=AD_TOL, rtol=0.0)


@pytest.mark.parametrize("grip_right", GRIP_RIGHTS)
def test_club_point_jacobians_match_central_difference(grip_right: float) -> None:
    p = _params(grip_right)
    pj = _jax_params(p)

    def club_points(x: Any) -> dict[str, Any]:
        fk = pgj.forward_kinematics_jax(x, pj)
        com = 0.5 * (fk["club_base"] + fk["club_tip"])
        return {"club_tip": fk["club_tip"], "club_com": com}

    for q in _random_states(16)[:6]:
        jac = pgj.analytical_fk_jacobians_jax(_jnp(q), pj)
        for key in ("club_tip", "club_com"):
            fd = _central_fd(lambda x, k=key: club_points(_jnp(x))[k], q)
            np.testing.assert_allclose(
                np.asarray(jac[key]), fd, atol=FD_TOL, rtol=0.0, err_msg=key
            )


def test_closure_vanishes_and_has_rank_two_on_the_manifold() -> None:
    p = _params()
    pj = _jax_params(p)
    for sol in _closed_solutions(p):
        phi = np.asarray(pgj.constraint_vector_jax(_jnp(sol), pj))
        assert np.max(np.abs(phi)) < PARITY_TOL
        jac = np.asarray(pgj.constraint_jacobian_jax(_jnp(sol), pj))
        assert np.linalg.matrix_rank(jac, tol=RANK_TOL) == 2
        rows = list(INDEPENDENT_CLOSURE_ROWS)
        assert np.linalg.matrix_rank(jac[rows], tol=RANK_TOL) == 2


def test_closure_jacobian_has_rank_three_off_the_manifold() -> None:
    p = _params()
    q = np.random.default_rng(1).uniform(-1.0, 1.0, N_DOF)
    jac = np.asarray(pgj.constraint_jacobian_jax(_jnp(q), _jax_params(p)))
    assert jac.shape == (pgj.N_CONSTRAINTS, N_DOF)
    assert np.linalg.matrix_rank(jac, tol=RANK_TOL) == 3
