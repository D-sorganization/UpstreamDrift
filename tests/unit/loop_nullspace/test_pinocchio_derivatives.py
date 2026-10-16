"""NSX-85 (#12714): Pinocchio analytic closed-loop derivatives (FULL reference).

Oracle: the engine-free NSX-84 central-difference reference
(``FiniteDifferenceDerivatives``) built over the same canonical provider and the
same closure tangent basis, at projected (closure-consistent) anthro states.
The analytic route wraps ``computeConstraintDynamicsDerivatives`` and must agree
to a scaled 1e-6 (``max|err| / max(1, max|ref|)``).

The reference step is chosen per derivative block, not loosened.  Accelerations
are ~6e4 and the native drift solve carries ~1e-7 relative noise, so the default
step 1e-5 sits on a round-off floor of ~1e-4 scaled.  The ``eta`` blocks are
exactly quadratic in ``v`` (central differences carry no truncation error), so
a large step is exact there; the ``xi`` blocks use a Richardson combination of
two central differences (``O(h^4)`` truncation) at ``h = 3.2e-3``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivativeProvider,
    DerivativeMethod,
    default_registry,
    require_adapter_derivatives,
)
from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider
from tests.unit.loop_nullspace.test_snapshot_from_adapter import (  # noqa: F401
    adapter,
    pin_module,
    state,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

SCALED_TOL = 1e-6
FIELDS = (
    "acceleration",
    "reaction",
    "projector",
    "d_acc_d_xi",
    "d_acc_d_eta",
    "d_acc_d_tau",
    "d_lambda_d_xi",
    "d_lambda_d_eta",
    "d_lambda_d_tau",
)


def _tau(adapter, seed: int = 0) -> np.ndarray:  # noqa: F811
    return np.random.default_rng(seed).normal(size=adapter.nv) * 10.0


XI_STEP = 3.2e-3
ETA_STEP = 1e-2


def _fd(adapter, q, v, tau, step):  # noqa: F811
    return tl.FiniteDifferenceDerivatives(
        AdapterConstrainedProvider(adapter), tl.EuclideanManifold(adapter.nv), step=step
    ).constrained_acceleration_derivatives(q, v, tau)


def _reference(adapter, q, v, tau) -> dict[str, np.ndarray]:  # noqa: F811
    """Per-block engine-free reference; see the module docstring for the steps."""
    coarse = _fd(adapter, q, v, tau, XI_STEP)
    fine = _fd(adapter, q, v, tau, XI_STEP / 2.0)
    exact_in_v = _fd(adapter, q, v, tau, ETA_STEP)
    ref = {name: getattr(fine, name) for name in FIELDS}
    for name in ("d_acc_d_xi", "d_lambda_d_xi"):
        ref[name] = (4.0 * getattr(fine, name) - getattr(coarse, name)) / 3.0
    for name in ("d_acc_d_eta", "d_lambda_d_eta"):
        ref[name] = getattr(exact_in_v, name)
    ref["basis"] = fine.basis
    return ref


def _scaled(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / max(1.0, float(np.max(np.abs(want)))))


@pytest.fixture(scope="module")
def analytic(adapter, state):  # noqa: F811
    q, v = state
    return adapter.constrained_acceleration_derivatives(q, v, _tau(adapter))


@pytest.fixture(scope="module")
def reference(adapter, state):  # noqa: F811
    q, v = state
    return _reference(adapter, q, v, _tau(adapter))


def test_adapter_declares_full_closed_loop_derivatives(adapter) -> None:  # noqa: F811
    level = adapter.capability(Capability.CLOSED_LOOP_DERIVATIVES)
    assert level is CapabilityLevel.FULL
    require_adapter_derivatives(adapter, minimum=CapabilityLevel.FULL)
    assert isinstance(adapter, ClosedLoopDerivativeProvider)
    assert adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.FULL


def test_registry_declares_pinocchio_analytic_and_loads() -> None:
    reg = default_registry()
    decl = reg.require("pinocchio", minimum=CapabilityLevel.FULL)
    assert decl.method is DerivativeMethod.ANALYTIC
    assert decl.level is CapabilityLevel.FULL
    assert callable(reg.load("pinocchio", minimum=CapabilityLevel.FULL))


def test_provenance_is_analytic_without_step(analytic) -> None:  # noqa: F811
    assert analytic.method is DerivativeMethod.ANALYTIC
    assert analytic.step is None
    assert analytic.provenance_tag().startswith("derivative_method=analytic")
    assert "step=" not in analytic.provenance_tag()


@pytest.mark.parametrize("field", FIELDS)
def test_matches_engine_free_reference(analytic, reference, field: str) -> None:  # noqa: F811
    err = _scaled(getattr(analytic, field), reference[field])
    assert err <= SCALED_TOL, f"{field}: scaled error {err:.3e}"


def test_shares_the_nsx84_closure_tangent_basis(analytic, reference) -> None:  # noqa: F811
    assert np.array_equal(analytic.basis, reference["basis"])
    assert analytic.tangent_dimension == analytic.nv - 6


def test_basis_lies_in_the_closure_kernel(adapter, analytic, state) -> None:  # noqa: F811
    q, _ = state
    jac = AdapterConstrainedProvider(adapter).closure_jacobian(q)
    assert np.max(np.abs(jac @ analytic.basis)) <= 1e-9


def test_reaction_is_resolved_not_frozen(analytic) -> None:  # noqa: F811
    assert np.linalg.norm(analytic.d_lambda_d_xi) > 0.0
    assert np.linalg.norm(analytic.d_lambda_d_tau) > 0.0


@pytest.mark.parametrize(("scale", "seed"), [(0.4, 1), (-1.0, 2)])
def test_other_admissible_states_and_loads(adapter, state, scale, seed) -> None:  # noqa: F811
    """``J_c v = 0`` is homogeneous, so a scaled ``v`` stays admissible."""
    q, v = state
    tau = _tau(adapter, seed)
    got = adapter.constrained_acceleration_derivatives(q, scale * v, tau)
    want = _reference(adapter, q, scale * v, tau)
    for field in FIELDS:
        err = _scaled(getattr(got, field), want[field])
        assert err <= SCALED_TOL, f"{field}: scaled error {err:.3e}"


def test_velocity_offset_matches_numerical_jacobian_rate(adapter, state) -> None:  # noqa: F811
    """``J_c O xi = -(dJ_c[P xi]) v`` with ``dJ`` from the analytic kinematics."""
    from src.engines.physics_engines.pinocchio.python.closed_loop_derivatives import (
        weld_jacobian_directional_drift,
    )

    q, v = state
    prov = AdapterConstrainedProvider(adapter)
    basis = tl.tangent_basis(prov.closure_jacobian(q))
    got = weld_jacobian_directional_drift(adapter._plant, q, v, basis)
    step = 1e-6
    cols = []
    for j in range(basis.shape[1]):
        plus = prov.closure_jacobian(q + step * basis[:, j]) @ v
        minus = prov.closure_jacobian(q - step * basis[:, j]) @ v
        cols.append((plus - minus) / (2.0 * step))
    want = np.array(cols).T
    assert _scaled(got, want) <= SCALED_TOL


def test_rejects_non_admissible_velocity(adapter, state) -> None:  # noqa: F811
    q, v = state
    prov = AdapterConstrainedProvider(adapter)
    bad = v + np.linalg.pinv(prov.closure_jacobian(q)) @ np.ones(6)
    with pytest.raises(ValueError, match="tangent to the closure"):
        adapter.constrained_acceleration_derivatives(q, bad, _tau(adapter))


def test_rejects_wrong_shapes(adapter, state) -> None:  # noqa: F811
    q, v = state
    with pytest.raises(ValueError):
        adapter.constrained_acceleration_derivatives(q[:-1], v, _tau(adapter))
    with pytest.raises(ValueError):
        adapter.constrained_acceleration_derivatives(q, v, np.full(adapter.nv, np.nan))
