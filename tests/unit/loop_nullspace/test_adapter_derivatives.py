"""NSX-87: MuJoCo and MyoSuite hard-KKT finite-difference derivatives (#12716).

The derivatives drive the single NSX-84 finite-difference routine with the
engine's canonical ``M, h, J_c, Jdot_v`` (the NSX-51/52 hard-KKT snapshot path).
Stock soft-constraint stepping (``mjd_transitionFD``) is a separate registry
entry and is not exercised here: its sensitivities are not conformance evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.loop_nullspace import derivative_capability as dc
from src.shared.python.loop_nullspace.adapter_derivatives import (
    UNSUPPORTED_STATE_COMPONENTS,
    adapter_closed_loop_derivatives,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs/development/full_body_models/evidence/ground_support"
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
STEP = 1e-5
REGISTRY = REPO / "tests/integration/cross_engine/divergence_registry.yaml"


def _tolerances() -> dict[str, float]:
    """The registered NSX-56 Track N pair tolerances, Pinocchio vs MuJoCo."""
    from src.shared.python.engine_core.cross_engine_validator import (
        DivergenceRegistry,
    )

    reg = DivergenceRegistry.from_yaml(REGISTRY)
    out = {}
    for metric in ("M", "h", "J_c", "Jdot_v"):
        entry = reg.find(
            check_name="closed_loop_snapshot_conformance",
            metric_name=metric,
            engine1="pinocchio",
            engine2="mujoco",
        )
        assert entry is not None, metric
        out[metric] = entry.tolerance
    return out


def derivative_tolerance() -> float:
    """Propagated bound: Track N model difference over the differencing step.

    A central difference of functions that differ by at most ``eps`` changes by
    at most ``eps / step``; the largest registered Track N tolerance is the
    ``Jdot_v`` finite-difference limit.  Nothing is entered by hand.
    """
    return max(_tolerances().values()) / STEP


@pytest.fixture(scope="module")
def mj_adapter():
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MujocoForceAdapter,
    )

    return MujocoForceAdapter(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def matched_state(mj_adapter):
    """Closure-consistent named state, shared by every engine."""
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    plant = VectorPlant("mujoco", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][400], rec["v"][400]
    proj = project_to_closure(plant, q0, v0)
    return plant._named(proj.q), plant._named(proj.v)


def _native(adapter, named_q, named_v):
    order = adapter.coordinate_order
    return (
        np.array([named_q[n] for n in order]),
        np.array([named_v[n] for n in order]),
    )


def _load(adapter, seed: int = 87) -> np.ndarray:
    tau = np.zeros(adapter.nv)
    rng = np.random.default_rng(seed)
    idx = list(adapter.actuated_indices)
    tau[idx] = rng.normal(size=len(idx))
    return tau


@pytest.fixture(scope="module")
def mj_result(mj_adapter, matched_state):
    q, v = _native(mj_adapter, *matched_state)
    return mj_adapter.constrained_acceleration_derivatives(q, v, _load(mj_adapter))


def test_capability_is_partial_finite_difference(mj_adapter, mj_result) -> None:
    assert (
        mj_adapter.capability(Capability.CLOSED_LOOP_DERIVATIVES)
        is CapabilityLevel.PARTIAL
    )
    assert mj_adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) is (
        CapabilityLevel.FULL
    )
    dc.require_adapter_derivatives(mj_adapter, minimum=CapabilityLevel.PARTIAL)
    with pytest.raises(dc.DerivativeCapabilityUnavailable):
        dc.require_adapter_derivatives(mj_adapter, minimum=CapabilityLevel.FULL)
    assert mj_result.method is dc.DerivativeMethod.FINITE_DIFFERENCE
    assert "derivative_method=finite_difference" in mj_result.provenance_tag()
    assert mj_result.step == STEP


def test_other_capabilities_stay_none(mj_adapter) -> None:
    kept = {Capability.CLOSED_LOOP_SNAPSHOT, Capability.CLOSED_LOOP_DERIVATIVES}
    others = [c for c in Capability if c not in kept]
    assert all(mj_adapter.capability(c) is CapabilityLevel.NONE for c in others)


def test_result_is_a_tangent_bundle_derivative_of_the_hard_kkt_solve(
    mj_adapter, mj_result, matched_state
) -> None:
    from src.shared.python.loop_nullspace.snapshot import canonical_closure

    q, v = _native(mj_adapter, *matched_state)
    jac, _ = canonical_closure(mj_adapter, q, v)
    res = mj_result
    assert res.nv == mj_adapter.nv
    assert res.tangent_dimension == mj_adapter.nv - 6
    assert np.max(np.abs(jac @ res.basis)) < 1e-9
    np.testing.assert_allclose(
        res.basis.T @ res.basis, np.eye(res.basis.shape[1]), rtol=0, atol=1e-10
    )
    assert np.isfinite(res.d_acc_d_xi).all()
    assert np.abs(res.d_acc_d_xi).max() > 0.0
    # the exact rigid-weld solve is what the nominal comes from
    err = mj_adapter.verify_acceleration_parity(
        q, v, _load(mj_adapter), res.acceleration
    )
    assert err / max(1.0, float(np.abs(res.acceleration).max())) < 1e-7


def test_reuses_the_nsx84_routine_without_a_copy() -> None:
    from src.shared.python.loop_nullspace import adapter_derivatives as mod
    from src.shared.python.loop_nullspace.tangent_linearization import (
        FiniteDifferenceDerivatives,
    )

    assert mod.FiniteDifferenceDerivatives is FiniteDifferenceDerivatives


def test_unsupported_states_are_listed(mj_adapter) -> None:
    assert "muscle_activation" in UNSUPPORTED_STATE_COMPONENTS
    assert "muscle_activation" in mj_adapter.derivative_unsupported_states


def test_step_and_state_are_validated(mj_adapter, matched_state) -> None:
    q, v = _native(mj_adapter, *matched_state)
    tau = _load(mj_adapter)
    with pytest.raises(ValueError):
        mj_adapter.constrained_acceleration_derivatives(q[:-1], v, tau)
    with pytest.raises(ValueError, match="step"):
        mj_adapter.constrained_acceleration_derivatives(q, v, tau, step=-1.0)


def test_helper_refuses_an_adapter_without_snapshot_capability() -> None:
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        create_engine_force_adapter,
    )

    synth = create_engine_force_adapter("mujoco", allow_synthetic=True, nv=44)
    z = np.zeros(synth.nv)
    with pytest.raises(CapabilityUnavailable, match="closed_loop_snapshot"):
        adapter_closed_loop_derivatives(synth, z, z, z)


def test_default_registry_declares_mujoco_and_myosuite_partial() -> None:
    reg = dc.default_registry()
    for engine in ("mujoco", "myosuite"):
        decl = reg.require(engine)
        assert decl.method is dc.DerivativeMethod.FINITE_DIFFERENCE
        assert decl.level is CapabilityLevel.PARTIAL
        assert callable(reg.load(engine))


# -- MyoSuite ------------------------------------------------------------


@pytest.fixture(scope="module")
def myo_adapter():
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.myosuite.python.force_adapter import (
        MyoSuiteForceAdapter,
    )

    class _PlainMujocoBacked(MyoSuiteForceAdapter):
        def _make_model(self, spec_bytes: bytes):
            from src.engines.physics_engines.mujoco.python.full_body_model import (
                NativeMujocoFullBodyModel,
            )

            model = NativeMujocoFullBodyModel(spec_bytes)
            model.kkt_regularization = 0.0
            return model

    return _PlainMujocoBacked(SPEC_PATH.read_bytes())


def test_myosuite_equals_mujoco_on_the_shared_model(
    myo_adapter, mj_adapter, mj_result, matched_state
) -> None:
    q, v = _native(myo_adapter, *matched_state)
    res = myo_adapter.constrained_acceleration_derivatives(q, v, _load(myo_adapter))
    assert (
        myo_adapter.capability(Capability.CLOSED_LOOP_DERIVATIVES)
        is CapabilityLevel.PARTIAL
    )
    np.testing.assert_array_equal(res.d_acc_d_xi, mj_result.d_acc_d_xi)
    np.testing.assert_array_equal(res.d_lambda_d_eta, mj_result.d_lambda_d_eta)


def test_myosuite_refuses_nonzero_regularisation(myo_adapter, matched_state) -> None:
    q, v = _native(myo_adapter, *matched_state)
    tau = _load(myo_adapter)
    myo_adapter.kkt_regularization = 1e-6
    try:
        assert (
            myo_adapter.capability(Capability.CLOSED_LOOP_DERIVATIVES)
            is CapabilityLevel.NONE
        )
        with pytest.raises(
            dc.DerivativeCapabilityUnavailable, match="kkt_regularization"
        ):
            myo_adapter.constrained_acceleration_derivatives(q, v, tau)
        with pytest.raises(dc.DerivativeCapabilityUnavailable):
            dc.require_adapter_derivatives(myo_adapter, minimum=CapabilityLevel.PARTIAL)
    finally:
        myo_adapter.kkt_regularization = 0.0


# -- Pinocchio agreement (NSX-56 Track N) --------------------------------


@pytest.mark.requires_pinocchio
def test_matches_pinocchio_within_the_propagated_track_n_tolerance(
    mj_adapter, mj_result, matched_state
) -> None:
    """Same spec, same state, same load: basis-independent operators agree.

    ``D P^T`` is the derivative operator restricted to ``ker J_c`` and does not
    depend on the SVD basis sign of either engine.
    """
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )

    pin = PinocchioForceAdapter(SPEC_PATH.read_bytes())
    assert list(pin.coordinate_order) == list(mj_adapter.coordinate_order)
    q, v = _native(pin, *matched_state)
    ref = adapter_closed_loop_derivatives(pin, q, v, _load(pin), step=STEP)
    tol = derivative_tolerance()

    def scaled(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).max()) / max(1.0, float(np.abs(b).max()))

    pairs = {
        "acceleration": (mj_result.acceleration, ref.acceleration),
        "reaction": (mj_result.reaction, ref.reaction),
        "projector": (mj_result.projector, ref.projector),
        "d_acc_d_tau": (mj_result.d_acc_d_tau, ref.d_acc_d_tau),
        "d_acc_d_xi": (
            mj_result.d_acc_d_xi @ mj_result.basis.T,
            ref.d_acc_d_xi @ ref.basis.T,
        ),
        "d_acc_d_eta": (
            mj_result.d_acc_d_eta @ mj_result.basis.T,
            ref.d_acc_d_eta @ ref.basis.T,
        ),
        "d_lambda_d_xi": (
            mj_result.d_lambda_d_xi @ mj_result.basis.T,
            ref.d_lambda_d_xi @ ref.basis.T,
        ),
        "d_lambda_d_eta": (
            mj_result.d_lambda_d_eta @ mj_result.basis.T,
            ref.d_lambda_d_eta @ ref.basis.T,
        ),
    }
    for name, (got, want) in pairs.items():
        assert scaled(got, want) <= tol, f"{name}: {scaled(got, want):.3g} > {tol:g}"
    assert np.abs(ref.d_acc_d_xi).max() > 1e-6  # not vacuous
