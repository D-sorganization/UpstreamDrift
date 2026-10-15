"""Pin UpstreamDrift's golfer kinematics and closure to the Tools copy (#12671).

ADR-0056 makes the vendored Tools package
``vendor/ud-tools/src/pendulum_simulator/src/double_pendulum_golf`` the one
canonical source of the golfer forward kinematics and the loop-closure
constraint and Jacobian.  The UpstreamDrift modules ``golfer_kinematics`` and
``golfer_constraints`` keep their import paths as thin adapters.

These tests load the pinned Tools package independently of the adapters and
check four things:

* every adapter function delegates to the Tools function of the same name;
* the parameter dataclass converts losslessly (same fields, each one copied);
* the helpers that stay in UpstreamDrift (``point_on_club``, the shaft offset
  and its derivative, and the FK Jacobians in ``golfer_dynamics``) agree with
  the Tools forward kinematics on representative configurations;
* ``INDEPENDENT_CLOSURE_ROWS`` and the least-squares KKT in
  ``constraint_solver`` (both UpstreamDrift-only, ADR-0056) remain valid for
  the Tools Jacobian.  The redundant rows must add no rank, and enforcing the
  independent rows must enforce the whole Tools closure.

A vendor bump that changes the Tools closure layout or the club convention
therefore fails here instead of drifting silently.
"""

from __future__ import annotations

import dataclasses
import importlib
from types import ModuleType

import numpy as np
import pytest
from scipy.optimize import least_squares

from src.shared.python._seam_redirect import load_pinned_tools_package
from src.shared.python.pendulum_simulator import (
    constraint_solver,
    golfer_constraints,
    golfer_dynamics,
    golfer_kinematics,
)
from src.shared.python.pendulum_simulator._canonical_golfer import canonical_params
from src.shared.python.pendulum_simulator.physics_golfer import (
    N_CONSTRAINTS,
    N_DOF,
    GolferParams,
)

pytestmark = pytest.mark.unit

TOOLS_PACKAGE = "pendulum_simulator.src.double_pendulum_golf"
CLUB = 7
# Club points are built along different (algebraically equal) routes in the
# two copies, so they agree to a few ulps of the segment lengths, not bitwise.
ROUNDING_TOL = 1e-12
RANK_TOL = 1e-8
# Bound on the forward-difference error of ``_constraint_acceleration_bias``
# (step 1e-7).  The independent rows close to ~1e-13; the redundant rows,
# whose bias carries the difference error, reach 5e-8 on these configurations.
BIAS_FD_TOL = 1e-6

KINEMATICS_DELEGATES = ("forward_kinematics", "_hub_position")
CONSTRAINT_DELEGATES = (
    "constraint_vector",
    "numerical_constraint_jacobian",
    "analytical_constraint_jacobian",
    "linear_accelerations",
    "net_joint_forces",
    "friction_torque_vector",
)


def _tools(module: str) -> ModuleType:
    package = load_pinned_tools_package(TOOLS_PACKAGE)
    return importlib.import_module(f"{package.__name__}.{module}")


def _params() -> GolferParams:
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
        grip_right=0.05,
        grip_left=0.15,
        L_rscap=0.08,
        L_lscap=0.08,
        m_rscap=7.0,
        m_lscap=7.0,
    )


def _tools_params(p: GolferParams) -> object:
    """Build the Tools parameter object field by field (independent of the adapter)."""
    tools_cls = _tools("physics_golfer").GolferParams
    return tools_cls(**{f.name: getattr(p, f.name) for f in dataclasses.fields(p)})


def _open_configurations() -> list[np.ndarray]:
    rng = np.random.default_rng(12671)
    return [np.zeros(N_DOF), *rng.uniform(-np.pi, np.pi, (6, N_DOF))]


def _closed_configurations(count: int = 4) -> list[np.ndarray]:
    """Configurations on the Tools closure manifold (``Phi = 0``)."""
    p_tools = _tools_params(_params())
    phi = _tools("golfer_constraints").constraint_vector
    rng = np.random.default_rng(5626)
    out: list[np.ndarray] = []
    while len(out) < count:
        res = least_squares(
            lambda x: phi(x, p_tools),
            rng.uniform(-3.0, 3.0, N_DOF),
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        if res.cost < 1e-20:
            out.append(res.x)
    return out


# --- delegation -------------------------------------------------------------


@pytest.mark.parametrize("name", KINEMATICS_DELEGATES)
def test_kinematics_adapter_delegates_to_tools(name: str) -> None:
    adapter = getattr(golfer_kinematics, name)
    assert adapter.__wrapped__ is getattr(_tools("golfer_kinematics"), name)


@pytest.mark.parametrize("name", CONSTRAINT_DELEGATES)
def test_constraint_adapter_delegates_to_tools(name: str) -> None:
    adapter = getattr(golfer_constraints, name)
    assert adapter.__wrapped__ is getattr(_tools("golfer_constraints"), name)


def test_parameter_dataclasses_convert_losslessly() -> None:
    """The adapter passes every field explicitly, so equal field sets suffice.

    Defaults are deliberately not compared: the UpstreamDrift ``g`` default is
    standard gravity (9.80665) and the Tools default is 9.81 (ADR-0056
    follow-up).  Conversion never relies on a default, which the next test
    pins.
    """
    tools_cls = _tools("physics_golfer").GolferParams

    def names(cls: type) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    assert names(GolferParams) == names(tools_cls)
    assert golfer_constraints.N_CONSTRAINTS == N_CONSTRAINTS


def test_conversion_carries_every_field_including_gravity() -> None:
    p = dataclasses.replace(_params(), g=9.80665, b_rh=0.03, m_clubhead=0.21)
    converted = canonical_params(p)
    for f in dataclasses.fields(p):
        assert getattr(converted, f.name) == getattr(p, f.name), f.name


# --- numerical parity -------------------------------------------------------


@pytest.mark.parametrize("q", _open_configurations(), ids=lambda q: f"q0={q[0]:+.3f}")
def test_forward_kinematics_matches_tools(q: np.ndarray) -> None:
    p = _params()
    ours = golfer_kinematics.forward_kinematics(q, p)
    theirs = _tools("golfer_kinematics").forward_kinematics(q, _tools_params(p))
    assert ours.keys() == theirs.keys()
    for key, point in theirs.items():
        np.testing.assert_array_equal(ours[key], point, err_msg=key)


@pytest.mark.parametrize("q", _open_configurations(), ids=lambda q: f"q0={q[0]:+.3f}")
def test_club_helpers_reproduce_tools_club_points(q: np.ndarray) -> None:
    p = _params()
    fk = _tools("golfer_kinematics").forward_kinematics(q, _tools_params(p))
    offsets = {
        "club_base": -p.grip_right,
        "grip_left": p.grip_left - p.grip_right,
        "club_tip": p.L_club - p.grip_right,
    }
    for key, offset in offsets.items():
        point = golfer_kinematics.point_on_club(fk["rh"], q[CLUB], offset)
        np.testing.assert_allclose(point, fk[key], rtol=0.0, atol=ROUNDING_TOL)


@pytest.mark.parametrize("q", _open_configurations(), ids=lambda q: f"q0={q[0]:+.3f}")
def test_fk_jacobians_match_tools(q: np.ndarray) -> None:
    p = _params()
    ours = golfer_dynamics.analytical_fk_jacobians(q, p)
    theirs = _tools("golfer_dynamics").analytical_fk_jacobians(q, _tools_params(p))
    assert ours.keys() == theirs.keys()
    for key, jac in theirs.items():
        np.testing.assert_allclose(ours[key], jac, rtol=0.0, atol=ROUNDING_TOL)

    tip_column = golfer_kinematics.club_shaft_offset_derivative(
        q[CLUB], p.L_club - p.grip_right
    )
    np.testing.assert_allclose(
        theirs["club_tip"][:, CLUB], tip_column, rtol=0.0, atol=ROUNDING_TOL
    )


@pytest.mark.parametrize("q", _open_configurations(), ids=lambda q: f"q0={q[0]:+.3f}")
def test_closure_matches_tools(q: np.ndarray) -> None:
    p, p_tools = _params(), _tools_params(_params())
    tools = _tools("golfer_constraints")
    np.testing.assert_array_equal(
        golfer_constraints.constraint_vector(q, p), tools.constraint_vector(q, p_tools)
    )
    np.testing.assert_array_equal(
        golfer_constraints.analytical_constraint_jacobian(q, p),
        tools.analytical_constraint_jacobian(q, p_tools),
    )


# --- closure rows and KKT (UpstreamDrift consumers of the Tools Jacobian) ----


@pytest.mark.parametrize("q", _closed_configurations(), ids=lambda q: f"q7={q[7]:+.3f}")
def test_independent_rows_span_the_tools_closure(q: np.ndarray) -> None:
    rows = list(golfer_constraints.INDEPENDENT_CLOSURE_ROWS)
    jac = _tools("golfer_constraints").analytical_constraint_jacobian(
        q, _tools_params(_params())
    )
    independent = jac[rows]
    assert np.linalg.matrix_rank(independent, tol=RANK_TOL) == len(rows)
    assert np.linalg.matrix_rank(jac, tol=RANK_TOL) == len(rows)
    # Every redundant row is a combination of the independent ones.
    coeffs, *_ = np.linalg.lstsq(independent.T, jac.T, rcond=None)
    np.testing.assert_allclose(independent.T @ coeffs, jac.T, rtol=0.0, atol=RANK_TOL)


@pytest.mark.parametrize("q", _closed_configurations(), ids=lambda q: f"q7={q[7]:+.3f}")
def test_independent_row_kkt_enforces_the_whole_tools_closure(q: np.ndarray) -> None:
    p = _params()
    tools = _tools("golfer_constraints")
    jac = tools.analytical_constraint_jacobian(q, _tools_params(p))
    rows = list(golfer_constraints.INDEPENDENT_CLOSURE_ROWS)

    # A closure-consistent velocity: the null space of the independent rows.
    _, _, vt = np.linalg.svd(jac[rows])
    qdot = vt[len(rows) :].T @ np.random.default_rng(7).normal(size=N_DOF - len(rows))
    state = np.concatenate([q, qdot])

    def torques(_t: float) -> tuple[float, ...]:
        return (2.0, -1.0, 0.5, 0.0, 1.0, -0.5, 0.0)

    qddot, lam = constraint_solver._solve_constrained_dynamics(state, 0.0, p, torques)
    bias = constraint_solver._constraint_acceleration_bias(q, qdot, p)

    # The redundant rows carry no multiplier ...
    redundant = [r for r in range(N_CONSTRAINTS) if r not in rows]
    np.testing.assert_array_equal(lam[redundant], 0.0)
    # ... yet the closure acceleration vanishes on all four Tools rows.
    np.testing.assert_allclose(jac @ qddot + bias, 0.0, rtol=0.0, atol=BIAS_FD_TOL)
