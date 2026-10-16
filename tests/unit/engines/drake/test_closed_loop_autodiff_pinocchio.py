"""NSX-86 (#12715): Drake AutoDiff derivatives agree with Pinocchio.

Both engines are built from one specification and compared in the Drake
coordinate order (permuted by name) and the canonical closure convention.

* The four arrays ``M, h, J_c, Jdot_v`` must first agree within the NSX-56
  Track N tolerance registered for the (pinocchio, drake) pair
  (``divergence_registry.yaml``); nothing is widened here.
* The constrained acceleration and reaction must agree within the Track S bound
  propagated from the Track N differences measured at this state (``kkt_bound``).
* The Drake exact derivatives must then equal Pinocchio's ANALYTIC derivatives
  (NSX-85), evaluated in the same closure tangent basis; the tolerances are
  derived in :func:`test_autodiff_equals_pinocchio_analytic_derivatives`.

Track N tolerances apply to the arrays, not to derivatives; no derivative-level
Track N entry exists (conformance rows are NSX-98).  Agreement between engines is
not physical validation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.engine_core.cross_engine_validator import DivergenceRegistry
from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.adapter_derivatives import AdapterClosedLoopModel
from src.shared.python.loop_nullspace.derivative_capability import DerivativeMethod
from src.shared.python.loop_nullspace.constrained_dynamics import (
    constrained_acceleration,
)
from tests.integration.cross_engine.loop_nullspace_conformance_support import (
    CHECK_NAME,
    TRACK_N_METRICS,
    kkt_bound,
    scaled_error,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_drake,
    pytest.mark.requires_pinocchio,
]

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/ground_support"
SPEC_PATH = EVIDENCE / "anthro_driver_drake/full_body_spec_hipcal_scaled.json"
RECORD_PATH = EVIDENCE / "anthro_driver_drake/dynamics_record.npz"
FRAMES = (0, 400)
REGISTRY = DivergenceRegistry.from_yaml(
    ROOT / "tests/integration/cross_engine/divergence_registry.yaml"
)
FD_STEP = 1e-3
FLOOR = 1e-9


@pytest.fixture(scope="module")
def adapters():
    pytest.importorskip("pinocchio")
    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.force_adapter import (
        DrakeForceAdapter,
    )
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )

    spec = SPEC_PATH.read_bytes()
    return PinocchioForceAdapter(spec), DrakeForceAdapter(spec)


@pytest.fixture(scope="module")
def models(adapters):
    pin, drake = adapters
    order = tuple(drake.coordinate_order)
    return AdapterClosedLoopModel(pin, order=order), AdapterClosedLoopModel(drake)


@pytest.fixture(scope="module", params=FRAMES, ids=lambda f: f"frame{f}")
def state(request, models):
    """A closure-consistent state (the NSX-56 projection), spec order of Drake.

    The canonical closure rows of the two engines coincide only where the weld
    holds, so the position-level closure is projected exactly once (Pinocchio
    plant), as the NSX-56 harness does.
    """
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    _, drake = models
    plant = VectorPlant("pinocchio", SPEC_PATH.read_bytes())
    with np.load(RECORD_PATH) as rec:
        q0, v0 = rec["q"][request.param], rec["v"][request.param]
    proj = project_to_closure(plant, q0, v0)
    rng = np.random.default_rng(12715)
    by_name_q = dict(zip(plant.coordinate_order, proj.q, strict=True))
    by_name_v = dict(zip(plant.coordinate_order, proj.v, strict=True))
    q = np.array([by_name_q[n] for n in drake.coordinate_order])
    v = np.array([by_name_v[n] for n in drake.coordinate_order])
    basis = tl.tangent_basis(drake.closure_jacobian(q))
    return q, basis @ (basis.T @ v), 1e-3 * rng.normal(size=drake.nv)


def _arrays(model, q, v):
    return SimpleNamespace(
        M=model.mass_matrix(q),
        h=model.bias_forces(q, v),
        J_c=model.closure_jacobian(q),
        Jdot_v=model.closure_drift(q, v),
    )


def _entry(metric: str):
    entry = REGISTRY.find(
        check_name=CHECK_NAME,
        metric_name=metric,
        engine1="pinocchio",
        engine2="drake",
    )
    assert entry is not None, f"no registry entry for pinocchio/drake {metric}"
    return entry


def test_arrays_agree_within_the_registered_track_n_tolerance(models, state) -> None:
    pin, drake = models
    q, v, _ = state
    ref, other = _arrays(pin, q, v), _arrays(drake, q, v)
    for metric in TRACK_N_METRICS:
        err = scaled_error(getattr(other, metric), getattr(ref, metric))
        assert err <= _entry(metric).tolerance, f"{metric}: {err:.3g}"
    assert float(np.abs(ref.J_c).max()) > 1e-3  # non-vacuous


def test_nominal_solution_agrees_within_the_propagated_track_s_bound(
    models, state
) -> None:
    pin, drake = models
    q, v, tau = state
    solved = []
    for model in (pin, drake):
        a, lam = constrained_acceleration(
            model.mass_matrix(q),
            model.bias_forces(q, v),
            tau,
            model.closure_jacobian(q),
            model.closure_drift(q, v),
        )
        solved.append(np.concatenate((a, lam)))
    ref, other = _arrays(pin, q, v), _arrays(drake, q, v)
    bound = kkt_bound(ref, other, float(np.linalg.norm(solved[0])), 1.0)
    assert float(np.linalg.norm(solved[1] - solved[0])) <= bound


def _pinocchio_analytic(pin_adapter, drake_names, basis, q, v, tau):
    """Pinocchio's ANALYTIC derivatives in Drake's order and tangent basis."""
    from src.engines.physics_engines.pinocchio.python.closed_loop_derivatives import (
        PinocchioAnalyticDerivatives,
    )

    pin_names = tuple(pin_adapter.coordinate_order)
    to_pin = np.array([drake_names.index(n) for n in pin_names])
    to_drake = np.array([pin_names.index(n) for n in drake_names])
    pin_basis = basis[to_pin]
    res = PinocchioAnalyticDerivatives(
        pin_adapter, basis_field=lambda _q: pin_basis
    ).constrained_acceleration_derivatives(q[to_pin], v[to_pin], tau[to_pin])
    assert res.method is DerivativeMethod.ANALYTIC
    return SimpleNamespace(
        d_acc_d_xi=res.d_acc_d_xi[to_drake],
        d_acc_d_eta=res.d_acc_d_eta[to_drake],
        d_acc_d_tau=res.d_acc_d_tau[np.ix_(to_drake, to_drake)],
        d_lambda_d_xi=res.d_lambda_d_xi,
        d_lambda_d_eta=res.d_lambda_d_eta,
        d_lambda_d_tau=res.d_lambda_d_tau[:, to_drake],
    )


def test_autodiff_equals_pinocchio_analytic_derivatives(
    adapters, models, state
) -> None:
    """Drake AutoDiff vs Pinocchio ANALYTIC, one tangent basis, Drake order.

    Tolerance derivation (nothing invented):

    * ``d/dtau`` blocks are the KKT effort response ``K^-1 [e_j; 0]``; the two
      engines' arrays differ by the measured Track N amounts, so the first-order
      bound is NSX-56's ``kkt_bound`` with ``rhs_scale = 0`` and the response
      norm as the solution norm (the same use as the NSX-56 effort test).
    * ``d/dxi, d/deta`` blocks also carry the engines' differential
      differences (``dM/dq`` ...), which no Track N entry registers.  They are
      held to the resolution of the independent NSX-84 finite-difference
      reference (its own spread under step x0.5 and x2): a difference below
      that is not detectable, and none is claimed beyond it.
    """
    pin_adapter, drake_adapter = adapters
    pin, drake = models
    q, v, tau = state
    exact = drake_adapter.constrained_acceleration_derivatives(q, v, tau)
    analytic = _pinocchio_analytic(
        pin_adapter, tuple(drake_adapter.coordinate_order), exact.basis, q, v, tau
    )

    def drake_fd(step: float):
        return tl.FiniteDifferenceDerivatives(
            drake,
            tl.EuclideanManifold(drake.nv),
            step=step,
            basis_field=lambda _q: exact.basis,
        ).constrained_acceleration_derivatives(q, v, tau)

    ref = drake_fd(FD_STEP)
    others = [drake_fd(FD_STEP * f) for f in (0.5, 2.0)]
    for name in ("d_acc_d_xi", "d_acc_d_eta", "d_lambda_d_xi", "d_lambda_d_eta"):
        resolution = max(
            *(scaled_error(getattr(o, name), getattr(ref, name)) for o in others),
            FLOOR,
        )
        err = scaled_error(getattr(exact, name), getattr(analytic, name))
        assert err <= resolution, f"{name}: {err:.3g} > resolution {resolution:.3g}"
    ref_arrays, other_arrays = _arrays(pin, q, v), _arrays(drake, q, v)
    stack_exact = np.vstack((exact.d_acc_d_tau, exact.d_lambda_d_tau))
    stack_pin = np.vstack((analytic.d_acc_d_tau, analytic.d_lambda_d_tau))
    bound = kkt_bound(
        ref_arrays, other_arrays, float(np.linalg.norm(stack_pin, 2)), 0.0
    )
    assert float(np.linalg.norm(stack_exact - stack_pin, 2)) <= bound
    assert float(np.abs(exact.d_acc_d_xi).max()) > 1e-3
