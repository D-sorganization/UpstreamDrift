"""NSX-86 (#12715): Drake AutoDiff derivatives of the weld KKT solve.

The oracle is the NSX-84 engine-free finite-difference reference run on the
identical problem (the same adapter, canonical closure convention, coordinate
order and tangent basis).  The tolerance is not entered by hand: per array it is
the change the reference itself shows when its step is halved or doubled, i.e. the
reference's own measured accuracy (its round-off floor: ``cond(M)`` of the
finger block is about ``1e8``, so a central difference is noise-limited long
before truncation matters, which is why the exact route is the useful one).
Agreement with Pinocchio lives in ``test_closed_loop_autodiff_pinocchio.py``.

State: a seeded configuration near zero with an admissible (tangent) velocity;
the position-level weld need not hold, since the derivatives of the KKT solve
are defined at any ``(q, v)`` with a full-rank closure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.adapter_derivatives import AdapterClosedLoopModel
from src.shared.python.loop_nullspace.constrained_dynamics import (
    constrained_acceleration,
)
from src.shared.python.loop_nullspace.derivative_capability import (
    DerivativeCapabilityUnavailable,
    DerivativeMethod,
    default_registry,
    require_adapter_derivatives,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake]

SPEC_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver/full_body_spec_hipcal_scaled.json"
)
FD_STEP = 1e-3
FLOOR = 1e-9  # the finite-difference module's own round-off floor
ARRAYS = (
    "acceleration",
    "reaction",
    "d_acc_d_xi",
    "d_acc_d_eta",
    "d_acc_d_tau",
    "d_lambda_d_xi",
    "d_lambda_d_eta",
    "d_lambda_d_tau",
)


def scaled(x: np.ndarray, ref: np.ndarray) -> float:
    """``max|x - ref| / max(1, max|ref|)`` (the NSX-56 Track N convention)."""
    return float(np.max(np.abs(x - ref))) / max(1.0, float(np.max(np.abs(ref))))


@pytest.fixture(scope="module")
def adapter():
    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.force_adapter import (
        DrakeForceAdapter,
    )

    return DrakeForceAdapter(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def model(adapter):
    return AdapterClosedLoopModel(adapter)


@pytest.fixture(scope="module")
def state(model):
    rng = np.random.default_rng(12715)
    q = 0.1 * rng.normal(size=model.nv)
    basis = tl.tangent_basis(model.closure_jacobian(q))
    v = basis @ (0.3 * rng.normal(size=basis.shape[1]))
    # A small load: the near-singular finger block would otherwise amplify
    # round-off in the finite-difference reference to the size of the signal.
    tau = 1e-3 * rng.normal(size=model.nv)
    return q, v, tau


@pytest.fixture(scope="module")
def exact(adapter, state):
    return adapter.constrained_acceleration_derivatives(*state)


def _fd(model, state, step):
    q, v, tau = state
    return tl.FiniteDifferenceDerivatives(
        model, tl.EuclideanManifold(model.nv), step=step
    ).constrained_acceleration_derivatives(q, v, tau)


@pytest.fixture(scope="module")
def reference(model, state):
    return _fd(model, state, FD_STEP)


@pytest.fixture(scope="module")
def reference_accuracy(model, state, reference):
    """Measured per-array accuracy of the reference: change under step halving."""
    others = [_fd(model, state, FD_STEP * f) for f in (0.5, 2.0)]
    return {
        name: max(
            *(scaled(getattr(o, name), getattr(reference, name)) for o in others),
            FLOOR,
        )
        for name in ARRAYS
    }


def test_adapter_declares_full_closed_loop_derivatives(adapter) -> None:
    level = adapter.capability(Capability.CLOSED_LOOP_DERIVATIVES)
    assert level is CapabilityLevel.FULL
    assert adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.FULL
    require_adapter_derivatives(adapter, minimum=CapabilityLevel.FULL)


def test_registry_declares_drake_autodiff_and_loads_the_provider(adapter) -> None:
    reg = default_registry()
    decl = reg.require("drake", minimum=CapabilityLevel.FULL)
    assert decl.method is DerivativeMethod.AUTODIFF
    factory = reg.load("drake", minimum=CapabilityLevel.FULL)
    assert factory(adapter).__class__.__name__ == "DrakeAutoDiffDerivatives"


def test_result_is_autodiff_exact(exact) -> None:
    assert exact.method is DerivativeMethod.AUTODIFF
    assert exact.step is None
    assert exact.provenance_tag().startswith("derivative_method=autodiff")
    assert exact.tangent_dimension == exact.nv - 6


def test_autodiff_equals_the_finite_difference_reference(
    exact, reference, reference_accuracy
) -> None:
    np.testing.assert_allclose(exact.basis, reference.basis, atol=1e-12)
    for name in ARRAYS:
        err = scaled(getattr(exact, name), getattr(reference, name))
        assert err <= reference_accuracy[name], (
            f"{name}: {err:.3g} exceeds the reference's own accuracy "
            f"{reference_accuracy[name]:.3g}"
        )


def test_comparison_is_not_vacuous(exact) -> None:
    assert float(np.abs(exact.d_acc_d_xi).max()) > 1e-3
    assert float(np.abs(exact.d_acc_d_eta).max()) > 1e-3
    assert float(np.abs(exact.d_lambda_d_xi).max()) > 1e-3  # lambda is re-solved
    assert float(np.abs(exact.reaction).max()) > 1e-3


def test_the_geometric_preload_matters_at_this_state(adapter, state) -> None:
    """Mutation check: without K_geom the answer leaves the reference's accuracy."""
    prov = adapter._derivatives
    nom = prov._nominal(*state)
    ctx = prov._column_context(nom)
    zero = tl.KKTContext(np.zeros_like(ctx.k_geom), ctx.differentials)
    shifts = []
    for j in range(nom.basis.shape[1]):
        args = (nom.basis[:, j], nom.offset[:, j])
        full, _ = prov._perturbation(nom, ctx, *args)
        cut, _ = prov._perturbation(nom, zero, *args)
        shifts.append(scaled(cut, full))
    assert max(shifts) > 1e-2


def test_nominal_is_the_canonical_kkt_solution(exact, model, state) -> None:
    q, v, tau = state
    acc, lam = constrained_acceleration(
        model.mass_matrix(q),
        model.bias_forces(q, v),
        tau,
        model.closure_jacobian(q),
        model.closure_drift(q, v),
    )
    np.testing.assert_allclose(exact.acceleration, acc, atol=1e-9)
    np.testing.assert_allclose(exact.reaction, lam, atol=1e-9)


def test_tangent_basis_keeps_the_closure(exact, model, state) -> None:
    jac = model.closure_jacobian(state[0])
    assert np.linalg.norm(jac @ exact.basis) < 1e-10
    assert (
        np.linalg.norm(exact.basis.T @ exact.basis - np.eye(exact.basis.shape[1]))
        < 1e-10
    )


def test_row_map_is_constant_so_canonical_differentials_are_exact(
    adapter, state
) -> None:
    from src.shared.python.loop_nullspace.snapshot import canonical_closure_matrix

    q, _, _ = state
    other = q + 0.3 * np.random.default_rng(1).normal(size=q.size)
    np.testing.assert_allclose(
        canonical_closure_matrix(adapter, q),
        canonical_closure_matrix(adapter, other),
        atol=1e-12,
    )


def test_unsupported_component_is_a_capability_rejection(
    adapter, state, monkeypatch
) -> None:
    native = adapter._model
    calls = []

    def refuse():
        raise RuntimeError("element X has no AutoDiffXd scalar conversion")

    monkeypatch.setattr(native.plant, "ToAutoDiffXd", refuse)
    monkeypatch.setattr(native, "_autodiff", None, raising=False)
    monkeypatch.setattr(adapter, "_derivatives", None, raising=False)
    fd_calls = []
    monkeypatch.setattr(
        tl.FiniteDifferenceDerivatives,
        "constrained_acceleration_derivatives",
        lambda *a, **k: fd_calls.append(1),
    )
    with pytest.raises(DerivativeCapabilityUnavailable, match="element X"):
        adapter.constrained_acceleration_derivatives(*state)
    assert not fd_calls and not calls  # never a silent finite-difference fallback


def test_moving_grip_frame_is_a_capability_rejection(
    adapter, state, monkeypatch
) -> None:
    q, v, tau = state
    real = adapter.closure_wrench_frame

    def shifted(qq):
        rot, origin = real(qq)
        return rot, origin + np.array([0.0, 0.0, 0.05])

    monkeypatch.setattr(adapter, "closure_wrench_frame", shifted)
    monkeypatch.setattr(adapter, "_derivatives", None, raising=False)
    with pytest.raises(DerivativeCapabilityUnavailable, match="grip frame"):
        adapter.constrained_acceleration_derivatives(q, v, tau)


def test_preconditions_raise(adapter, state) -> None:
    q, v, tau = state
    with pytest.raises(ValueError):
        adapter.constrained_acceleration_derivatives(q[:-1], v, tau)
    with pytest.raises(ValueError, match="tangent"):
        adapter.constrained_acceleration_derivatives(q, np.ones_like(v), tau)


def test_model_order_is_a_permutation_by_name(adapter) -> None:
    names = adapter.coordinate_order
    reordered = AdapterClosedLoopModel(adapter, order=tuple(reversed(names)))
    base = AdapterClosedLoopModel(adapter)
    rng = np.random.default_rng(7)
    q, v = 0.1 * rng.normal(size=(2, base.nv))
    np.testing.assert_allclose(
        reordered.mass_matrix(q[::-1]), base.mass_matrix(q)[::-1, ::-1], atol=1e-12
    )
    np.testing.assert_allclose(
        reordered.closure_jacobian(q[::-1]),
        base.closure_jacobian(q)[:, ::-1],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="permutation"):
        AdapterClosedLoopModel(adapter, order=names[:-1])
