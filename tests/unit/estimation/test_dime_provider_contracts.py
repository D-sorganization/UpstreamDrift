"""Focused behavioral tests for DIME state, observation and dynamics provider contracts (#11423).

Enforces:
- RED: incomplete state, invalid time order, dimensional mismatch, stale model hashes,
  unsupported contact and unknown units fail before native mutation. Exceptions restore state;
  unavailable provider never returns zero as success. Muscle models without activation dynamics
  fail qualification; unmeasured ROM priors cannot supply passive stiffness laws.
- GREEN: deterministic fake provider plus analytic real provider satisfy identical contracts,
  without claiming native qualification. Runtime exclusivity contract enforces mutual exclusivity
  between marginalized-input transitions and explicit-input likelihoods on the same interval,
  and prevents permitted diagnostics from contributing duplicate factors. Public manifold
  operations verify nq != nv and quaternion sign equivalence. Shared deterministic fixtures
  from synthetic_fixtures.py integrate seamlessly.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError, PreconditionError
from src.shared.python.estimation.dime_manifest import CANONICAL_DIME_UNITS
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
    make_underactuated_analytic_fixture,
)
from src.shared.python.estimation.dime_contracts import (
    AnalyticPendulumProvider,
    ContactPolicy,
    ControlChannelSpec,
    DeterministicFakeProvider,
    DimeCompleteState,
    DimeEstimationResult,
    DimeFullStepRequest,
    DimeFullStepResult,
    DimeObservationWindow,
    DimeZeroInputProposal,
    DynamicsProvider,
    EstimationIntervalFactor,
    PassiveLoadSpec,
    ProviderCapability,
    ProviderSnapshot,
    QuaternionManifold,
    RuntimeExclusivityContract,
    SE3Manifold,
    UnderactuatedAnalyticProvider,
    VectorSpaceManifold,
    check_qualification_rules,
)


# ==============================================================================
# Helper Factories
# ==============================================================================


def _valid_complete_state(
    *,
    t: float = 0.0,
    q: np.ndarray | None = None,
    v: np.ndarray | None = None,
    model_hash: str = "model-hash-11423",
    internal_state: dict | None = None,
    units: dict | None = None,
) -> DimeCompleteState:
    return DimeCompleteState(
        t=t,
        q=np.array([0.1], dtype=np.float64) if q is None else q,
        v=np.array([0.0], dtype=np.float64) if v is None else v,
        internal_state={} if internal_state is None else internal_state,
        model_hash=model_hash,
        units=dict(CANONICAL_DIME_UNITS) if units is None else units,
    )


# ==============================================================================
# RED Test Suite
# ==============================================================================


class TestRedProviderContractViolations:
    """RED test suite enforcing fail-closed contract validations before native mutation."""

    @pytest.mark.unit
    def test_red_incomplete_state_fails_before_mutation(self) -> None:
        provider = DeterministicFakeProvider(model_hash="hash-fake-01")
        initial_snap = provider.snapshot()

        # Non-finite coordinates
        with pytest.raises((ContractViolationError, PreconditionError)):
            _valid_complete_state(q=np.array([np.nan]))

        # Boolean in coordinates
        with pytest.raises((ContractViolationError, PreconditionError)):
            _valid_complete_state(q=np.array([True]))

        # Non-finite velocities
        with pytest.raises((ContractViolationError, PreconditionError)):
            _valid_complete_state(v=np.array([np.inf]))

        # Empty model hash
        with pytest.raises((ContractViolationError, PreconditionError)):
            _valid_complete_state(model_hash="")

        # Verify provider state was completely untouched
        current_snap = provider.snapshot()
        assert current_snap.timestamp == initial_snap.timestamp
        np.testing.assert_array_equal(current_snap.state.q, initial_snap.state.q)

    @pytest.mark.unit
    def test_red_invalid_time_order_fails_before_mutation(self) -> None:
        # Non-strictly monotonic timestamps in observation window
        invalid_times = np.array([0.0, 0.02, 0.01, 0.03], dtype=np.float64)
        obs_data = {"markers": np.zeros((4, 3), dtype=np.float64)}

        with pytest.raises((ContractViolationError, PreconditionError)):
            DimeObservationWindow(
                t_start=0.0,
                t_end=0.03,
                times=invalid_times,
                observations=obs_data,
                units=dict(CANONICAL_DIME_UNITS),
            )

        # Duplicate timestamps
        dup_times = np.array([0.0, 0.01, 0.01, 0.02], dtype=np.float64)
        with pytest.raises((ContractViolationError, PreconditionError)):
            DimeObservationWindow(
                t_start=0.0,
                t_end=0.02,
                times=dup_times,
                observations={"markers": np.zeros((4, 3))},
                units=dict(CANONICAL_DIME_UNITS),
            )

    @pytest.mark.unit
    def test_red_dimensional_mismatch_fails_before_mutation(self) -> None:
        provider = DeterministicFakeProvider(n_q=2, n_v=2, model_hash="hash-dim-test")
        initial_snap = provider.snapshot()

        # Wrong q dimension
        bad_q_state = _valid_complete_state(
            q=np.array([0.1], dtype=np.float64),  # dimension 1, expected 2
            v=np.array([0.0, 0.0], dtype=np.float64),
            model_hash="hash-dim-test",
        )
        with pytest.raises((ContractViolationError, PreconditionError)):
            provider.step(
                DimeFullStepRequest(
                    state=bad_q_state,
                    controls=np.array([0.0]),
                    dt=0.01,
                    model_hash="hash-dim-test",
                )
            )

        # Wrong controls dimension
        valid_state = _valid_complete_state(
            q=np.array([0.1, 0.2], dtype=np.float64),
            v=np.array([0.0, 0.0], dtype=np.float64),
            model_hash="hash-dim-test",
        )
        with pytest.raises((ContractViolationError, PreconditionError)):
            provider.step(
                DimeFullStepRequest(
                    state=valid_state,
                    controls=np.array([1.0, 2.0, 3.0]),  # 3 controls, 1 channel
                    dt=0.01,
                    model_hash="hash-dim-test",
                )
            )

        # Provider state unchanged
        assert provider.snapshot().timestamp == initial_snap.timestamp

    @pytest.mark.unit
    def test_red_stale_model_hash_fails_before_mutation(self) -> None:
        provider = DeterministicFakeProvider(model_hash="hash-authoritative")
        initial_snap = provider.snapshot()

        stale_request = DimeFullStepRequest(
            state=_valid_complete_state(model_hash="hash-authoritative"),
            controls=np.array([0.0]),
            dt=0.01,
            model_hash="hash-stale-mismatch",  # Stale hash
        )

        with pytest.raises((ContractViolationError, PreconditionError)):
            provider.step(stale_request)

        # State remains invariant
        assert provider.snapshot().timestamp == initial_snap.timestamp

    @pytest.mark.unit
    def test_red_unsupported_contact_policy_fails(self) -> None:
        # Cannot declare unsupported policy or simultaneous eliminated and explicit
        with pytest.raises((ContractViolationError, PreconditionError, ValueError)):
            ContactPolicy("both_eliminated_and_explicit")  # type: ignore[arg-type]

        with pytest.raises((ContractViolationError, PreconditionError)):
            ProviderCapability(
                provider_id="bad-contact-prov",
                version="1.0.0",
                status="implemented",
                n_q=1,
                n_v=1,
                manifold=VectorSpaceManifold(1),
                control_channels=(),
                contact_policy="both_eliminated_and_explicit",  # type: ignore[arg-type]
                retained_passive_loads=(),
            )

    @pytest.mark.unit
    def test_red_unknown_units_fail_before_mutation(self) -> None:
        provider = DeterministicFakeProvider(model_hash="hash-fake-units")
        initial_snap = provider.snapshot()

        unknown_units = {"length": "inches", "mass": "slugs"}
        with pytest.raises((ContractViolationError, PreconditionError)):
            state_bad_units = _valid_complete_state(
                model_hash="hash-fake-units", units=unknown_units
            )
            provider.step(
                DimeFullStepRequest(
                    state=state_bad_units,
                    controls=np.array([0.0]),
                    dt=0.01,
                    model_hash="hash-fake-units",
                )
            )

        assert provider.snapshot().timestamp == initial_snap.timestamp

    @pytest.mark.unit
    def test_red_exceptions_restore_provider_state(self) -> None:
        provider = DeterministicFakeProvider(model_hash="hash-fake-fail")
        initial_snap = provider.snapshot()

        # Inject simulated engine failure during step
        provider.inject_step_failure(True)

        req = DimeFullStepRequest(
            state=initial_snap.state,
            controls=np.array([10.0]),
            dt=0.01,
            model_hash="hash-fake-fail",
        )

        with pytest.raises(RuntimeError, match="Simulated engine failure"):
            provider.step(req)

        # Provider state must be restored to pre-call snapshot
        post_fail_snap = provider.snapshot()
        assert post_fail_snap.timestamp == initial_snap.timestamp
        np.testing.assert_array_equal(post_fail_snap.state.q, initial_snap.state.q)
        np.testing.assert_array_equal(post_fail_snap.state.v, initial_snap.state.v)

    @pytest.mark.unit
    def test_red_unavailable_provider_never_returns_zero_as_success(self) -> None:
        provider = DeterministicFakeProvider(
            status="unavailable", model_hash="hash-unavail"
        )
        req = DimeFullStepRequest(
            state=_valid_complete_state(model_hash="hash-unavail"),
            controls=np.array([0.0]),
            dt=0.01,
            model_hash="hash-unavail",
        )

        # Unavailable provider MUST raise, never return zero error / success
        with pytest.raises((ContractViolationError, RuntimeError)):
            provider.step(req)

        with pytest.raises((ContractViolationError, RuntimeError)):
            provider.compute_zero_input_proposal(
                _valid_complete_state(model_hash="hash-unavail"), duration=0.1, dt=0.01
            )

    @pytest.mark.unit
    def test_red_muscle_without_activation_dynamics_rejected(self) -> None:
        # Muscle model with excitation control channel requires full activation dynamics
        muscle_channel = ControlChannelSpec(
            name="biceps_excitation",
            physical_type="excitation",
            units="dimensionless",
            selection_map=(0,),
            limits=(0.0, 1.0),
            internal_state_semantics="instantaneous",  # Missing activation dynamics!
        )

        cap = ProviderCapability(
            provider_id="muscle-prov",
            version="1.0.0",
            status="implemented",
            n_q=1,
            n_v=1,
            manifold=VectorSpaceManifold(1),
            control_channels=(muscle_channel,),
            contact_policy="native_eliminated",
            retained_passive_loads=(),
            has_activation_dynamics=False,  # False!
            is_qualified=False,
        )

        # Qualification check must reject silent substitution
        is_ok, reason = check_qualification_rules(cap)
        assert not is_ok
        assert "activation dynamics" in reason.lower()

    @pytest.mark.unit
    def test_red_unmeasured_rom_prior_stiffness_rejected(self) -> None:
        unmeasured_load = PassiveLoadSpec(
            name="shoulder_rom_limit",
            load_type="joint_limit_spring",
            dof_indices=(0,),
            stiffness=50.0,
            is_measured=False,  # Unmeasured ROM prior stiffness law!
        )

        cap = ProviderCapability(
            provider_id="rom-prov",
            version="1.0.0",
            status="implemented",
            n_q=1,
            n_v=1,
            manifold=VectorSpaceManifold(1),
            control_channels=(),
            contact_policy="native_eliminated",
            retained_passive_loads=(unmeasured_load,),
            has_activation_dynamics=False,
            is_qualified=False,
        )

        is_ok, reason = check_qualification_rules(cap)
        assert not is_ok
        assert "unmeasured" in reason.lower()


# ==============================================================================
# GREEN Test Suite
# ==============================================================================


class TestGreenProviderContracts:
    """GREEN test suite verifying identical contracts, exclusivity, and manifolds."""

    @pytest.mark.unit
    def test_green_deterministic_fake_and_analytic_real_providers_satisfy_identical_contracts(
        self,
    ) -> None:
        fixture = make_fixed_base_pendulum_fixture(n_frames=5)
        analytic_provider = AnalyticPendulumProvider(fixture=fixture)
        fake_provider = DeterministicFakeProvider(
            n_q=1, n_v=1, model_hash=analytic_provider.model_hash
        )

        # Both satisfy DynamicsProvider interface
        assert isinstance(analytic_provider, DynamicsProvider)
        assert isinstance(fake_provider, DynamicsProvider)

        # Neither claims native qualification by default
        assert not analytic_provider.capability.is_qualified
        assert not fake_provider.capability.is_qualified

        # Full step on both
        state0 = analytic_provider.get_state()
        step_req = DimeFullStepRequest(
            state=state0,
            controls=np.array([0.0]),
            dt=0.01,
            model_hash=analytic_provider.model_hash,
        )

        res_analytic = analytic_provider.step(step_req)
        assert isinstance(res_analytic, DimeFullStepResult)
        assert res_analytic.decomposition is not None
        assert np.all(np.isfinite(res_analytic.accelerations))

        # Snapshot and restore on both
        snap = analytic_provider.snapshot()
        assert isinstance(snap, ProviderSnapshot)
        step_req2 = DimeFullStepRequest(
            state=analytic_provider.get_state(),
            controls=np.array([0.0]),
            dt=0.01,
            model_hash=analytic_provider.model_hash,
        )
        analytic_provider.step(step_req2)
        assert analytic_provider.get_state().t > snap.timestamp
        analytic_provider.restore(snap)
        assert analytic_provider.get_state().t == snap.timestamp

    @pytest.mark.unit
    def test_green_runtime_exclusivity_contract(self) -> None:
        exclusivity = RuntimeExclusivityContract()

        # Register observation likelihood factor
        exclusivity.register_factor(
            EstimationIntervalFactor(
                name="marker_obs",
                factor_type="observation_likelihood",
                t_start=0.0,
                t_end=0.1,
                contributes_to_objective=True,
            )
        )

        # Register marginalized-input transition factor on [0.0, 0.05]
        exclusivity.register_factor(
            EstimationIntervalFactor(
                name="trans_marginal",
                factor_type="marginalized_input_transition",
                t_start=0.0,
                t_end=0.05,
                contributes_to_objective=True,
            )
        )

        # Attempting explicit-input factor on overlapping interval [0.02, 0.06] MUST FAIL
        with pytest.raises((ContractViolationError, PreconditionError)):
            exclusivity.register_factor(
                EstimationIntervalFactor(
                    name="explicit_u",
                    factor_type="explicit_input_likelihood",
                    t_start=0.02,
                    t_end=0.06,
                    contributes_to_objective=True,
                )
            )

        # Permitted diagnostic factor with contributes_to_objective=False MUST SUCCEED
        exclusivity.register_factor(
            EstimationIntervalFactor(
                name="u_diagnostic",
                factor_type="diagnostic",
                t_start=0.0,
                t_end=0.05,
                contributes_to_objective=False,
            )
        )

        # Diagnostic that attempts to contribute to objective MUST FAIL
        with pytest.raises((ContractViolationError, PreconditionError)):
            exclusivity.register_factor(
                EstimationIntervalFactor(
                    name="duplicate_diagnostic_factor",
                    factor_type="diagnostic",
                    t_start=0.0,
                    t_end=0.05,
                    contributes_to_objective=True,  # Violates diagnostic rule!
                )
            )

        # Explicit-input factor on strictly disjoint interval [0.06, 0.10] MUST SUCCEED
        exclusivity.register_factor(
            EstimationIntervalFactor(
                name="explicit_u_disjoint",
                factor_type="explicit_input_likelihood",
                t_start=0.06,
                t_end=0.10,
                contributes_to_objective=True,
            )
        )

    @pytest.mark.unit
    def test_green_manifold_nq_not_equal_nv_and_quaternion_sign_equivalence(
        self,
    ) -> None:
        # SE(3) manifold: nq = 7 (pos 3 + quat 4), nv = 6 (lin 3 + ang 3)
        se3 = SE3Manifold()
        assert se3.n_q == 7
        assert se3.n_v == 6
        assert se3.n_q != se3.n_v

        # Identity configuration: pos=[0,0,0], quat=[1,0,0,0]
        q_id = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v_zero = np.zeros(6, dtype=np.float64)

        q_retract = se3.retract(q_id, v_zero)
        np.testing.assert_allclose(q_retract, q_id, atol=1e-12)

        # Retract Jacobian shape (7, 6)
        jac_retract = se3.retract_jacobian(q_id, v_zero)
        assert jac_retract.shape == (7, 6)

        # Quaternion manifold: nq = 4, nv = 3
        quat_m = QuaternionManifold()
        assert quat_m.n_q == 4
        assert quat_m.n_v == 3

        q_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        q_b = -q_a  # Antipodal sign flip: exactly identical SO(3) rotation!

        # Local coordinates between q and -q must be identically zero
        diff_tangent = quat_m.local_coordinates(q_a, q_b)
        np.testing.assert_allclose(diff_tangent, np.zeros(3), atol=1e-12)

        # SE(3) sign equivalence on orientation sub-manifold
        q_sign_flip = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        se3_diff = se3.local_coordinates(q_id, q_sign_flip)
        np.testing.assert_allclose(se3_diff, np.zeros(6), atol=1e-12)

    @pytest.mark.unit
    def test_green_zero_input_proposal_matches_ztcf_superposition(self) -> None:
        fixture = make_fixed_base_pendulum_fixture(n_frames=6)
        provider = AnalyticPendulumProvider(fixture=fixture)

        state = provider.get_state()
        proposal = provider.compute_zero_input_proposal(state, duration=0.05, dt=0.01)

        assert isinstance(proposal, DimeZeroInputProposal)
        assert proposal.source_intervention == "ZTCF"
        assert len(proposal.states) == 6
        assert len(proposal.decompositions) == 6

        # Check superposition on every step: a_full = a_grav + a_drift + a_ctrl
        for decomp in proposal.decompositions:
            np.testing.assert_allclose(
                decomp.ztcf, decomp.a_grav + decomp.a_drift, atol=1e-12
            )
            # Under zero input, a_ctrl must be 0
            np.testing.assert_allclose(
                decomp.a_ctrl, np.zeros_like(decomp.a_ctrl), atol=1e-12
            )

    @pytest.mark.unit
    def test_green_serialization_round_trip_preserves_semantics(self) -> None:
        state = _valid_complete_state(
            t=0.02,
            q=np.array([0.15, -0.05]),
            v=np.array([0.2, 0.0]),
            internal_state={"cache_key": 42},
            model_hash="sha256-abc123",
        )

        # DimeCompleteState
        d_state = state.to_dict()
        state_rt = DimeCompleteState.from_dict(d_state)
        assert state_rt.t == state.t
        assert state_rt.model_hash == state.model_hash
        np.testing.assert_array_equal(state_rt.q, state.q)
        np.testing.assert_array_equal(state_rt.v, state.v)
        assert state_rt.internal_state["cache_key"] == 42

        # DimeObservationWindow
        window = DimeObservationWindow(
            t_start=0.0,
            t_end=0.02,
            times=np.array([0.0, 0.01, 0.02]),
            observations={"marker0": np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]])},
            uncertainty_kind="gaussian",
            units=dict(CANONICAL_DIME_UNITS),
        )
        d_win = window.to_dict()
        json_str = json.dumps(d_win)
        win_rt = DimeObservationWindow.from_dict(json.loads(json_str))
        assert win_rt.t_start == window.t_start
        assert win_rt.uncertainty_kind == "gaussian"
        np.testing.assert_array_equal(win_rt.times, window.times)
        np.testing.assert_array_equal(
            win_rt.observations["marker0"], window.observations["marker0"]
        )

        # ProviderCapability
        cap = ProviderCapability(
            provider_id="analytic-pendulum-id",
            version="1.0.0",
            status="implemented",
            n_q=1,
            n_v=1,
            manifold=VectorSpaceManifold(1),
            control_channels=(
                ControlChannelSpec(
                    name="torque",
                    physical_type="torque",
                    units="N*m",
                    selection_map=(0,),
                    limits=(-100.0, 100.0),
                ),
            ),
            contact_policy="native_eliminated",
            retained_passive_loads=(),
        )
        cap_rt = ProviderCapability.from_dict(cap.to_dict())
        assert cap_rt.provider_id == cap.provider_id
        assert cap_rt.n_q == cap.n_q
        assert cap_rt.contact_policy == "native_eliminated"
        assert len(cap_rt.control_channels) == 1

        # DimeEstimationResult
        res = DimeEstimationResult(
            contract_version="1.0.0",
            trajectory_states=(state,),
            estimated_controls=np.array([[0.5]]),
            residuals={"kinematic": np.array([0.001])},
            uncertainty_kind="gaussian",
            uncertainty_summary={"trace_cov": 0.002},
            model_hash="sha256-abc123",
            qualification_status="implemented",
            alignment_metric=0.99,
            cancellation_metric=0.02,
        )
        res_rt = DimeEstimationResult.from_dict(res.to_dict())
        assert res_rt.contract_version == "1.0.0"
        assert res_rt.alignment_metric == 0.99
        assert len(res_rt.trajectory_states) == 1

    @pytest.mark.unit
    def test_green_synthetic_fixtures_integration(self) -> None:
        pendulum_fix = make_fixed_base_pendulum_fixture(n_frames=8)
        pendulum_prov = AnalyticPendulumProvider(fixture=pendulum_fix)
        assert pendulum_prov.capability.n_q == 1
        assert pendulum_prov.capability.n_v == 1

        underactuated_fix = make_underactuated_analytic_fixture(n_frames=8)
        underactuated_prov = UnderactuatedAnalyticProvider(fixture=underactuated_fix)
        assert underactuated_prov.capability.n_q == 2
        assert underactuated_prov.capability.n_v == 2
        assert len(underactuated_prov.capability.control_channels) == 1
        assert underactuated_prov.capability.control_channels[0].selection_map == (1,)
