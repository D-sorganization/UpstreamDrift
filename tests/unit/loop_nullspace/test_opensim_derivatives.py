"""NSX-88 (#12717): OpenSim closed-loop derivatives, capability ``PARTIAL``.

Oracle: the Pinocchio adapter differentiated through the *same* NSX-84 routine
on the identical specification and state.  The tolerance is the NSX-56 Track N
one: the measured Track N snapshot differences (each within its registered
tolerance) are propagated with ``kkt_bound`` -- the Track S method -- to the
solution error ``B``; a central difference of two evaluations then differs by at
most ``B / h`` per column, and the linear load response by ``kkt_bound`` with a
zero right-hand side.  No tolerance is entered by hand.  Agreement between two
engines is software conformance, not physical validation.
"""

# ruff: noqa: F811, F401 - the shared OpenSim/Pinocchio fixtures are imported by name
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from tests.helpers.opensim_support import require_opensim
from tests.integration.cross_engine.loop_nullspace_conformance_support import (
    CHECK_NAME,
    build_snapshot,
    kkt_bound,
    kkt_matrix,
    kkt_solve,
    to_reference_order,
    track_n_errors,
)
from tests.unit.loop_nullspace.test_snapshot_opensim import (
    SPEC_PATH,
    os_adapter,
    os_state,
    perm,
    pin_adapter,
    pin_state,
    reference,
)

from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.engine_core.cross_engine_validator import DivergenceRegistry
from src.shared.python.loop_nullspace import derivative_capability as dc
from src.shared.python.loop_nullspace.adapter_derivatives import (
    AdapterClosedLoopModel,
    adapter_closed_loop_derivatives,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    EngineType,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_opensim,
    pytest.mark.requires_pinocchio,
]

REGISTRY = DivergenceRegistry.from_yaml(
    Path(__file__).resolve().parents[2]
    / "integration/cross_engine/divergence_registry.yaml"
)
STEP = 1e-5
STATE_MAPS = ("d_acc_d_xi", "d_acc_d_eta", "d_lambda_d_xi", "d_lambda_d_eta")


@pytest.fixture(scope="module")
def derivs(os_adapter, os_state, reference, perm):
    inverse = np.argsort(perm)
    return os_adapter.constrained_acceleration_derivatives(
        *os_state, reference["tau"][inverse]
    )


@pytest.fixture(scope="module")
def pin_derivs(pin_adapter, pin_state, reference):
    return adapter_closed_loop_derivatives(
        pin_adapter, *pin_state, reference["tau"], step=STEP
    )


def test_capability_is_partial_finite_difference(os_adapter) -> None:
    assert os_adapter.capability(Capability.CLOSED_LOOP_DERIVATIVES) is (
        CapabilityLevel.PARTIAL
    )
    dc.require_adapter_derivatives(os_adapter, minimum=CapabilityLevel.PARTIAL)
    with pytest.raises(dc.DerivativeCapabilityUnavailable):
        dc.require_adapter_derivatives(os_adapter, minimum=CapabilityLevel.FULL)
    assert os_adapter.capability(Capability.MUSCLES) is CapabilityLevel.NONE


def test_registry_declares_opensim_partial_and_loads_the_adapter() -> None:
    require_opensim()
    from src.engines.physics_engines.opensim.python.force_adapter import (
        OpenSimNativeForceAdapter,
    )

    reg = dc.default_registry()
    decl = reg.require("opensim")
    assert decl.method is dc.DerivativeMethod.FINITE_DIFFERENCE
    assert decl.level is CapabilityLevel.PARTIAL
    assert reg.load("opensim") is OpenSimNativeForceAdapter


def test_result_is_converged_finite_difference_with_provenance(
    derivs,
    os_adapter,
) -> None:
    assert derivs.method is dc.DerivativeMethod.FINITE_DIFFERENCE
    assert derivs.converged is True
    assert derivs.observed_order >= 1.5
    assert derivs.coordinate_only_sufficient is True
    assert os_adapter.muscle_count == 0
    tag = derivs.provenance_tag()
    assert "derivative_method=finite_difference" in tag
    assert "coordinate_only=sufficient" in tag
    assert "converged=true" in tag
    assert derivs.nv == os_adapter.nv


def test_tangent_bundle_construction_holds(derivs, os_adapter, os_state) -> None:
    model = AdapterClosedLoopModel(os_adapter)
    assert np.abs(model.closure_jacobian(os_state[0]) @ derivs.basis).max() < 1e-9
    np.testing.assert_allclose(
        derivs.projector, derivs.basis @ derivs.basis.T, atol=1e-12
    )
    assert derivs.tangent_dimension == derivs.nv - 6
    assert float(np.abs(derivs.d_acc_d_xi).max()) > 1.0  # not vacuous


def _snapshots(pin_adapter, os_adapter, pin_state, os_state):
    order = tuple(pin_adapter.coordinate_order)

    def named(adapter, state, source):
        names = adapter.coordinate_order
        snap = build_snapshot(
            adapter,
            dict(zip(names, state[0], strict=True)),
            dict(zip(names, state[1], strict=True)),
            source,
        )
        return to_reference_order(snap, order)

    return named(pin_adapter, pin_state, "pin"), named(os_adapter, os_state, "os")


def test_matches_pinocchio_within_the_track_n_derived_tolerance(
    derivs,
    pin_derivs,
    pin_adapter,
    os_adapter,
    pin_state,
    os_state,
    reference,
    perm,
) -> None:
    ref, other = _snapshots(pin_adapter, os_adapter, pin_state, os_state)
    for metric, err in track_n_errors(ref, other).items():
        entry = REGISTRY.find(
            check_name=CHECK_NAME,
            metric_name=metric,
            engine1="pinocchio",
            engine2=EngineType.OPENSIM.value,
        )
        assert entry is not None and err <= entry.tolerance, metric
    n = ref.M.shape[0]
    tau = reference["tau"]
    norm_x = float(np.linalg.norm(kkt_solve(ref, tau)))
    k = pin_derivs.tangent_dimension
    column_bound = np.sqrt(k) * kkt_bound(ref, other, norm_x, 1.0) / STEP
    resp = np.linalg.solve(kkt_matrix(ref), np.vstack([np.eye(n), np.zeros((6, n))]))
    load_bound = np.sqrt(n) * kkt_bound(ref, other, float(np.linalg.norm(resp, 2)), 0.0)
    assert column_bound / max(1.0, float(np.abs(pin_derivs.d_acc_d_xi).max())) < 1.0
    for name in STATE_MAPS:  # basis independent: map composed with the projector
        want = getattr(pin_derivs, name) @ pin_derivs.basis.T
        got = (getattr(derivs, name) @ derivs.basis.T)[:, perm]
        if name.startswith("d_acc"):
            got = got[perm]
        assert float(np.linalg.norm(got - want, 2)) <= column_bound, name
    for name in ("d_acc_d_tau", "d_lambda_d_tau"):
        want = getattr(pin_derivs, name)
        got = getattr(derivs, name)[:, perm]
        if name.startswith("d_acc"):
            got = got[perm]
        assert float(np.linalg.norm(got - want, 2)) <= load_bound, name


def test_muscles_flag_coordinate_only_derivatives_as_insufficient(
    os_adapter,
    os_state,
    reference,
    perm,
) -> None:
    inverse = np.argsort(perm)
    with mock.patch.object(
        type(os_adapter), "muscle_count", new_callable=mock.PropertyMock
    ) as count:
        count.return_value = 3
        res = os_adapter.constrained_acceleration_derivatives(
            *os_state, reference["tau"][inverse], check_convergence=False
        )
    assert res.coordinate_only_sufficient is False
    assert "3 muscle-tendon" in res.insufficiency_reason
    tag = res.provenance_tag()
    assert "coordinate_only=insufficient" in tag
    assert "converged=unchecked" in tag
    assert res.method is dc.DerivativeMethod.FINITE_DIFFERENCE


def test_inconsistent_sufficiency_flag_is_refused(derivs) -> None:
    import dataclasses

    values = {f.name: getattr(derivs, f.name) for f in dataclasses.fields(derivs)}
    values["coordinate_only_sufficient"] = False
    with pytest.raises(ValueError, match="insufficiency_reason"):
        type(derivs)(**values)


def test_unconverged_derivatives_are_refused(
    os_adapter, os_state, reference, perm
) -> None:
    from src.engines.physics_engines.opensim.python import (
        closed_loop_derivatives as cld,
    )
    from src.shared.python.loop_nullspace.tangent_linearization import (
        StepConvergence,
    )

    bad = StepConvergence((1e-3, 5e-4, 2.5e-4), (1.0, 0.9), 0.2, False)
    inverse = np.argsort(perm)
    with mock.patch.object(cld, "adapter_step_convergence", return_value=bad):
        with pytest.raises(cld.DerivativeNotConvergedError, match="did not converge"):
            os_adapter.constrained_acceleration_derivatives(
                *os_state, reference["tau"][inverse]
            )


def test_preconditions_are_validated(os_adapter, os_state, reference, perm) -> None:
    q, v = os_state
    tau = reference["tau"][np.argsort(perm)]
    with pytest.raises(ValueError):
        os_adapter.constrained_acceleration_derivatives(
            q[:-1], v, tau, check_convergence=False
        )
    with pytest.raises(ValueError, match="step"):
        os_adapter.constrained_acceleration_derivatives(
            q, v, tau, step=-1.0, check_convergence=False
        )


def test_model_view_memoises_engine_evaluations() -> None:
    adapter = mock.MagicMock()
    adapter.nv = 3
    adapter.capability.return_value = CapabilityLevel.FULL
    adapter.compute_mass_and_bias.return_value = (np.eye(3), np.zeros(3))
    model = AdapterClosedLoopModel(adapter)
    q, v = np.zeros(3), np.ones(3)
    for _ in range(3):
        model.mass_matrix(q)
        model.bias_forces(q, v)
    assert adapter.compute_mass_and_bias.call_count == 2  # (q, 0) and (q, v)
