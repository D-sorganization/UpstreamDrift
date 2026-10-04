"""Behavioral test suite for DIME State, Observation, and Dynamics Provider Contracts (#11423).

Enforces:
- RED:
  1. Incomplete state, invalid time order, dimensional mismatch, stale model hashes,
     unsupported contact, and unknown units fail before native mutation.
  2. Exceptions restore provider state via snapshot rollback.
  3. Unavailable provider never returns zero or fake success.
  4. Runtime exclusivity between marginalized-input transition and explicit-input likelihood.
  5. Diagnostic factors cannot contribute duplicate estimation factors.
  6. Muscle-driven control channels require activation dynamics (no silent torque substitution).
- GREEN:
  1. Deterministic fake provider plus analytic real provider satisfy identical contracts.
  2. Public manifold retract/local-coordinates operations support nq != nv and quaternion sign equivalence.
  3. Serialization round-trips preserve semantics.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    CapabilityStatus,
    DimeProvenanceRecord,
)
from src.shared.python.estimation.dime_contracts import (
    AnalyticPendulumProvider,
    ConflictingContactInterfaceError,
    ContactInterfaceMode,
    ContactPolicy,
    ControlAllocation,
    ControlPhysicalType,
    DeterministicFakeProvider,
    DimeControlChannel,
    DimeEstimationResult,
    DimeManifold,
    DimeState,
    ExclusiveFactorConflictError,
    IntervalFactorRegistration,
    IntervalFactorRegistry,
    IntervalFactorType,
    InvalidTimeOrderError,
    ObservationItem,
    ObservationWindow,
    ProviderSnapshot,
    ProviderUnavailableError,
    QuaternionManifold,
    StaleModelHashError,
    UnitMismatchError,
    VectorManifold,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------


def _make_valid_state(
    *,
    t: float = 0.0,
    q: np.ndarray | None = None,
    v: np.ndarray | None = None,
    vdot: np.ndarray | None = None,
    model_hash: str = "sha256:test_model_v1",
    units: dict[str, str] | None = None,
) -> DimeState:
    """Helper to build a canonical 1-DOF pendulum test state."""
    q_arr = np.array([0.5], dtype=np.float64) if q is None else q
    v_arr = np.array([0.0], dtype=np.float64) if v is None else v
    return DimeState(
        t=t,
        q=q_arr,
        v=v_arr,
        vdot=vdot,
        internal_state={"integrator_step": np.array([0.0])},
        units=units or dict(CANONICAL_DIME_UNITS),
        model_hash=model_hash,
        frame="world",
    )


# ---------------------------------------------------------------------------
# RED Case 1: Incomplete State, Dimensional Mismatch, and Invalid Values
# ---------------------------------------------------------------------------


def test_state_rejects_negative_time_and_non_finite() -> None:
    """Negative: state must fail closed on negative time or non-finite coordinates."""
    with pytest.raises(ValueError, match="t must be non-negative"):
        _make_valid_state(t=-0.01)

    with pytest.raises(ValueError, match="must be finite"):
        _make_valid_state(q=np.array([np.nan]))

    with pytest.raises(ValueError, match="must be finite"):
        _make_valid_state(v=np.array([np.inf]))


def test_state_rejects_dimensional_mismatch() -> None:
    """Negative: state must reject acceleration dimension mismatch vs velocity."""
    with pytest.raises(ValueError, match="vdot dimension mismatch"):
        _make_valid_state(
            v=np.array([0.0]),
            vdot=np.array([0.0, 1.0]),  # 2D vdot vs 1D v
        )


def test_state_rejects_unknown_units() -> None:
    """Negative: state must reject non-SI or unknown unit declarations."""
    bad_units = dict(CANONICAL_DIME_UNITS)
    bad_units["angle"] = "degrees"  # Non-SI unit
    with pytest.raises(UnitMismatchError, match="Invalid unit for angle"):
        _make_valid_state(units=bad_units)


# ---------------------------------------------------------------------------
# RED Case 2: Stale Model Hash Rejection
# ---------------------------------------------------------------------------


def test_provider_rejects_stale_model_hash() -> None:
    """Negative: provider must reject state with mismatched model hash before mutation."""
    provider = DeterministicFakeProvider(model_hash="sha256:current_model_rev_a")
    stale_state = _make_valid_state(model_hash="sha256:outdated_model_rev_z")

    with pytest.raises(StaleModelHashError, match="Model hash mismatch"):
        provider.step_full(stale_state, control=np.array([0.0]), dt=0.01)

    with pytest.raises(StaleModelHashError, match="Model hash mismatch"):
        provider.step_zero_input(stale_state, dt=0.01)


# ---------------------------------------------------------------------------
# RED Case 3: Observation Window Time Ordering and Bounds
# ---------------------------------------------------------------------------


def test_observation_window_rejects_invalid_time_order() -> None:
    """Negative: observation window must reject non-monotonic observations."""
    obs1 = ObservationItem(
        t=0.1,
        modality="marker_3d",
        data=np.array([1.0, 2.0, 3.0]),
    )
    obs2_backward = ObservationItem(
        t=0.05,  # Backward in time!
        modality="marker_3d",
        data=np.array([1.1, 2.1, 3.1]),
    )

    with pytest.raises(InvalidTimeOrderError, match="non-monotonic"):
        ObservationWindow(
            start_time=0.0,
            end_time=0.2,
            items=(obs1, obs2_backward),
        )


def test_observation_window_rejects_items_outside_bounds() -> None:
    """Negative: observation window must reject items outside start_time / end_time."""
    obs = ObservationItem(
        t=0.5,
        modality="marker_3d",
        data=np.array([1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError, match="outside window bounds"):
        ObservationWindow(
            start_time=0.0,
            end_time=0.3,  # obs.t = 0.5 is out of bounds!
            items=(obs,),
        )


# ---------------------------------------------------------------------------
# RED Case 4: Contact Interface Exclusivity
# ---------------------------------------------------------------------------


def test_conflicting_contact_interface_fails_closed() -> None:
    """Negative: provider must reject simultaneous native eliminated and explicit constrained."""
    with pytest.raises(ConflictingContactInterfaceError, match="mutually exclusive"):
        ContactPolicy(
            mode=ContactInterfaceMode.BOTH_CONFLICTING,  # Attempting both
            contact_bodies=("foot_left", "foot_right"),
            friction_coefficient=0.6,
        )


# ---------------------------------------------------------------------------
# RED Case 5: Provider State Rollback on Exception
# ---------------------------------------------------------------------------


def test_exception_restores_provider_state() -> None:
    """Negative/Resilience: when stepping fails, provider restores pre-step snapshot."""
    provider = DeterministicFakeProvider(model_hash="sha256:test_model_v1")
    initial_state = _make_valid_state(t=1.0, q=np.array([0.5]), v=np.array([0.1]))

    # Set up provider initial state
    provider.set_state(initial_state)
    assert provider.get_state().t == 1.0

    # Inject simulated failure during step
    provider.inject_step_failure(True)

    with pytest.raises(RuntimeError, match="Simulated solver convergence failure"):
        provider.step_full(initial_state, control=np.array([10.0]), dt=0.01)

    # Provider internal state must have been rolled back to initial_state
    restored = provider.get_state()
    assert restored.t == 1.0
    assert np.allclose(restored.q, [0.5])
    assert np.allclose(restored.v, [0.1])


# ---------------------------------------------------------------------------
# RED Case 6: Unavailable Provider Truthful Reporting
# ---------------------------------------------------------------------------


def test_unavailable_provider_never_returns_zero_as_success() -> None:
    """Negative: unavailable provider must raise or fail closed, never return zeros."""
    provider = DeterministicFakeProvider(
        model_hash="sha256:test_model_v1",
        capability_status="unavailable",
        unavailability_reason="Engine binary not provisioned in this environment",
    )
    state = _make_valid_state()

    with pytest.raises(ProviderUnavailableError, match="Engine binary not provisioned"):
        provider.step_full(state, control=np.array([0.0]), dt=0.01)

    with pytest.raises(ProviderUnavailableError, match="Engine binary not provisioned"):
        provider.step_zero_input(state, dt=0.01)


# ---------------------------------------------------------------------------
# RED Case 7: Runtime Exclusivity for Marginalized vs Explicit Input
# ---------------------------------------------------------------------------


def test_runtime_exclusivity_contract_for_marginalized_vs_explicit_input() -> None:
    """Negative: registering marginalized-input transition and explicit-input likelihood on same interval fails."""
    registry = IntervalFactorRegistry()
    interval = (0.0, 0.1)

    # Register marginalized-input transition factor for interval [0.0, 0.1]
    registry.register(
        IntervalFactorRegistration(
            interval=interval,
            factor_type=IntervalFactorType.MARGINALIZED_INPUT_TRANSITION,
            factor_id="factor_ztcf_0_1",
            is_diagnostic=False,
        )
    )

    # Attempting to register explicit-input likelihood on the same interval must fail closed!
    with pytest.raises(ExclusiveFactorConflictError, match="Exclusive factor conflict"):
        registry.register(
            IntervalFactorRegistration(
                interval=interval,
                factor_type=IntervalFactorType.EXPLICIT_INPUT_LIKELIHOOD,
                factor_id="factor_torque_0_1",
                is_diagnostic=False,
            )
        )


def test_diagnostic_factor_cannot_duplicate_estimation_factor() -> None:
    """Negative: diagnostic factors cannot be registered as active estimation factors."""
    registry = IntervalFactorRegistry()
    interval = (0.0, 0.1)

    diag_factor = IntervalFactorRegistration(
        interval=interval,
        factor_type=IntervalFactorType.MARGINALIZED_INPUT_TRANSITION,
        factor_id="diag_factor_1",
        is_diagnostic=True,
    )
    registry.register(diag_factor)

    active_factors = registry.get_active_estimation_factors()
    assert len(active_factors) == 0  # Diagnostics do NOT contribute to estimation cost!
    assert len(registry.get_diagnostic_factors()) == 1


# ---------------------------------------------------------------------------
# RED Case 8: Muscle-Driven Actuation Dynamics Enforcement
# ---------------------------------------------------------------------------


def test_control_channel_activation_dynamics_enforcement() -> None:
    """Negative: excitation control channel requires activation dynamics; cannot silently substitute torque."""
    with pytest.raises(ValueError, match="Activation dynamics must be declared"):
        DimeControlChannel(
            name="biceps_excitation",
            physical_type=ControlPhysicalType.EXCITATION,
            unit="1.0",
            dof_index=0,
            lower_limit=0.0,
            upper_limit=1.0,
            requires_activation_dynamics=False,  # Violation!
        )


# ---------------------------------------------------------------------------
# GREEN Cases: Manifold Retraction, nq != nv, and Quaternion Sign Equivalence
# ---------------------------------------------------------------------------


def test_manifold_vector_retraction_and_local_coordinates() -> None:
    """Positive: VectorManifold satisfies retract and local coordinates in R^n."""
    manifold = VectorManifold(dim=3)
    assert manifold.config_dim == 3
    assert manifold.tangent_dim == 3

    q0 = np.array([1.0, 2.0, 3.0])
    delta_v = np.array([0.1, -0.2, 0.3])
    q1 = manifold.retract(q0, delta_v)
    np.testing.assert_allclose(q1, [1.1, 1.8, 3.3])

    recovered_delta = manifold.local_coordinates(q0, q1)
    np.testing.assert_allclose(recovered_delta, delta_v)


def test_manifold_quaternion_retraction_and_sign_equivalence() -> None:
    """Positive: QuaternionManifold has nq=4, nv=3, and satisfies q == -q sign equivalence."""
    manifold = QuaternionManifold()
    assert manifold.config_dim == 4
    assert manifold.tangent_dim == 3

    # Identity quaternion [w, x, y, z] = [1, 0, 0, 0]
    q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # 1. Sign equivalence: q and -q represent identical physical rotation
    q_neg = -q_id
    delta_sign = manifold.local_coordinates(q_id, q_neg)
    np.testing.assert_allclose(delta_sign, np.zeros(3), atol=1e-12)

    # 2. Retract a small rotation about Z axis
    omega_z = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    q_rotated = manifold.retract(q_id, omega_z)
    assert np.isclose(np.linalg.norm(q_rotated), 1.0)  # Remains on unit sphere S^3

    recovered_omega = manifold.local_coordinates(q_id, q_rotated)
    np.testing.assert_allclose(recovered_omega, omega_z, atol=1e-12)


# ---------------------------------------------------------------------------
# GREEN Cases: Deterministic Fake and Analytic Providers
# ---------------------------------------------------------------------------


def test_deterministic_fake_provider_satisfies_contract() -> None:
    """Positive: DeterministicFakeProvider steps forward, handles zero-input, snapshots."""
    provider = DeterministicFakeProvider(model_hash="sha256:test_model_v1")
    assert provider.capability_status == "implemented"
    assert provider.capability_status != "qualified"

    state0 = _make_valid_state(t=0.0, q=np.array([0.0]), v=np.array([1.0]))
    snap = provider.snapshot()
    assert snap.timestamp == 0.0

    state1 = provider.step_zero_input(state0, dt=0.1)
    assert state1.t == pytest.approx(0.1)
    assert state1.q[0] > state0.q[0]

    state2 = provider.step_full(state1, control=np.array([2.0]), dt=0.1)
    assert state2.t == pytest.approx(0.2)


def test_analytic_pendulum_provider_satisfies_contract() -> None:
    """Positive: AnalyticPendulumProvider implements exact physics under identical contract."""
    provider = AnalyticPendulumProvider(
        length_m=1.0,
        mass_kg=1.0,
        gravity_mps2=9.81,
        model_hash="sha256:analytic_pendulum_v1",
    )
    assert provider.capability_status == "implemented"
    assert provider.capability_status != "qualified"

    state0 = _make_valid_state(
        t=0.0,
        q=np.array([0.5]),
        v=np.array([0.0]),
        model_hash="sha256:analytic_pendulum_v1",
    )
    # Energy at start
    e0 = provider.compute_energy(state0)
    assert np.isfinite(e0)

    # Step zero input
    state1 = provider.step_zero_input(state0, dt=0.01)
    assert state1.t == pytest.approx(0.01)

    # Energy is conserved in conservative motion
    e1 = provider.compute_energy(state1)
    assert np.isclose(e0, e1, atol=1e-4)


# ---------------------------------------------------------------------------
# GREEN Cases: Serialization Round-Trip
# ---------------------------------------------------------------------------


def test_state_and_result_serialization_round_trip() -> None:
    """Positive: DimeState and DimeEstimationResult serialize and deserialize losslessly."""
    state = _make_valid_state(t=0.5, q=np.array([0.42]), v=np.array([-0.1]))
    state_dict = state.to_dict()
    restored_state = DimeState.from_dict(state_dict)

    assert restored_state.t == state.t
    np.testing.assert_allclose(restored_state.q, state.q)
    np.testing.assert_allclose(restored_state.v, state.v)
    assert restored_state.model_hash == state.model_hash
    assert restored_state.units == state.units

    provenance = DimeProvenanceRecord(
        engine="analytic",
        engine_version="1.0.0",
        model_hash="sha256:test_model_v1",
        param_hash="sha256:params_v1",
        git_commit="c8861213a9",
        created_at="2026-10-04T02:45:00Z",
    )
    result = DimeEstimationResult(
        success=True,
        trajectory=(state, restored_state),
        residuals={"drift_rms": 0.001, "control_rms": 0.05},
        provenance=provenance,
        qualification_status="implemented",
    )
    result_dict = result.to_dict()
    restored_result = DimeEstimationResult.from_dict(result_dict)

    assert restored_result.success is True
    assert len(restored_result.trajectory) == 2
    assert restored_result.residuals["drift_rms"] == pytest.approx(0.001)
    assert restored_result.provenance.git_commit == "c8861213a9"
    assert restored_result.qualification_status == "implemented"
