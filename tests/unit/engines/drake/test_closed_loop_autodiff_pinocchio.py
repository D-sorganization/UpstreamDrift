"""NSX-86 (#12715): Drake AutoDiff derivatives agree with Pinocchio.

Both engines are built from one specification and compared in the Drake
coordinate order (permuted by name) and the canonical closure convention.

* The four arrays ``M, h, J_c, Jdot_v`` must first agree within the NSX-56
  Track N tolerance registered for the (pinocchio, drake) pair
  (``divergence_registry.yaml``); nothing is widened here.
* The constrained acceleration and reaction must agree within the Track S bound
  propagated from the Track N differences measured at this state (``kkt_bound``).
* The Drake exact derivatives must then equal Pinocchio's finite-difference
  reference, run in the same closure tangent basis, within that reference's
  own measured step-to-step variation.  Pinocchio supplies no exact closed-loop
  derivatives of its own yet (NSX-85), so its reference is the NSX-84 route.

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
from src.shared.python.loop_nullspace.adapter_model import AdapterClosedLoopModel
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
ARRAYS = (
    "d_acc_d_xi",
    "d_acc_d_eta",
    "d_acc_d_tau",
    "d_lambda_d_xi",
    "d_lambda_d_eta",
    "d_lambda_d_tau",
)


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


def test_autodiff_equals_pinocchio_finite_difference_in_one_tangent_basis(
    adapters, models, state
) -> None:
    _, drake_adapter = adapters
    pin, _ = models
    q, v, tau = state
    exact = drake_adapter.constrained_acceleration_derivatives(q, v, tau)
    basis = exact.basis  # Drake's closure tangent basis, spec order

    def pin_fd(step: float):
        return tl.FiniteDifferenceDerivatives(
            pin,
            tl.EuclideanManifold(pin.nv),
            step=step,
            basis_field=lambda _q: basis,
        ).constrained_acceleration_derivatives(q, v, tau)

    ref = pin_fd(FD_STEP)
    others = [pin_fd(FD_STEP * f) for f in (0.5, 2.0)]
    for name in ARRAYS:
        bound = max(
            *(scaled_error(getattr(o, name), getattr(ref, name)) for o in others),
            FLOOR,
        )
        err = scaled_error(getattr(exact, name), getattr(ref, name))
        assert err <= bound, (
            f"{name}: {err:.3g} exceeds the reference spread {bound:.3g}"
        )
    assert float(np.abs(exact.d_acc_d_xi).max()) > 1e-3
