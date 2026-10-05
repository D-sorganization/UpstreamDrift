"""Focused behavioral tests for DIME offline smoothing and independent continuous replay (#11430).

Enforces:
- RED:
  * Per-frame state resets fail closed (replay must not reset state at every knot).
  * Hidden target-force feedback fails closed if unmodeled external force applied.
  * Undeclared root wrench fails closed (floating-base root has strictly zero artificial actuator forces).
  * Replay with changed model fails closed (mismatched model configuration or parameters rejected).
  * Missing controls fail closed (missing control channels or unactuated degrees of freedom given controls).
  * Backward inference calling reverse-time contact integration fails closed.
- GREEN:
  * Uninterrupted synthetic replay from saved initial state using saved controls reproduces trajectory within frozen tolerances.
  * Native floating-base/contact fixture reproduces saved motion and forces within frozen tolerances.
  * Backward smoothing with marginalized arrival information produces continuous smoothed trajectory without per-frame discontinuities.
  * Structured ContinuousReplayResult contains trajectory, independent replay residuals, assistance telemetry, and provenance.
  * Public adapters connect seamlessly to Shadow Tracker and Simscape continuous replay harnesses.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import (
    ContactPolicy,
    ControlChannelSpec,
    DeterministicFakeProvider,
    DimeCompleteState,
    ProviderCapability,
    VectorSpaceManifold,
)
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    DimeProvenanceRecord,
    NumericAcceptanceThresholds,
)
from src.shared.python.estimation.dime_providers import (
    AnalyticPendulumProvider,
    UnderactuatedAnalyticProvider,
)
from src.shared.python.estimation.moving_horizon import (
    ArrivalFactor,
    marginalize_arrival_factor,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
    make_native_stance_fixture,
    make_underactuated_analytic_fixture,
)
from src.shared.python.estimation.dime_continuous_replay import (
    ContinuousReplayOptions,
    ContinuousReplayResult,
    IndependentReplayMetrics,
    ReplayReceipt,
    ReplayProvenanceSources,
    SmoothedTrajectoryResult,
    execute_continuous_replay,
    smooth_backward_trajectory,
    to_shadow_tracker_rollout_request,
    to_simscape_continuous_trajectory,
)

pytestmark = pytest.mark.unit


# ==============================================================================
# Helper Factories
# ==============================================================================


def _make_floating_base_provider(
    model_hash: str = "fb-provider-hash",
) -> DeterministicFakeProvider:
    """Construct a fake provider with 6 unactuated root DOFs and 1 actuated joint."""
    channels = (
        ControlChannelSpec(
            name="root_px",
            physical_type="force",
            units="N",
            selection_map=(0,),
            limits=(0.0, 0.0),
        ),
        ControlChannelSpec(
            name="root_py",
            physical_type="force",
            units="N",
            selection_map=(1,),
            limits=(0.0, 0.0),
        ),
        ControlChannelSpec(
            name="root_pz",
            physical_type="force",
            units="N",
            selection_map=(2,),
            limits=(0.0, 0.0),
        ),
        ControlChannelSpec(
            name="root_rx",
            physical_type="torque",
            units="N*m",
            selection_map=(3,),
            limits=(0.0, 0.0),
        ),
        ControlChannelSpec(
            name="root_ry",
            physical_type="torque",
            units="N*m",
            selection_map=(4,),
            limits=(0.0, 0.0),
        ),
        ControlChannelSpec(
            name="root_rz",
            physical_type="torque",
            units="N*m",
            selection_map=(5,),
            limits=(0.0, 0.0),
        ),
        ControlChannelSpec(
            name="joint_0",
            physical_type="torque",
            units="N*m",
            selection_map=(6,),
            limits=(-100.0, 100.0),
        ),
    )
    cap = ProviderCapability(
        provider_id="floating-base-test-provider",
        version="1.0.0",
        status="implemented",
        n_q=7,
        n_v=7,
        manifold=VectorSpaceManifold(7),
        control_channels=channels,
        contact_policy=ContactPolicy.NATIVE_ELIMINATED,
        retained_passive_loads=(),
        is_qualified=False,
    )
    init_state = DimeCompleteState(
        t=0.0,
        q=np.zeros(7, dtype=np.float64),
        v=np.zeros(7, dtype=np.float64),
        model_hash=model_hash,
        units=dict(CANONICAL_DIME_UNITS),
    )
    provider = DeterministicFakeProvider(n_q=7, n_v=7, model_hash=model_hash)
    object.__setattr__(provider, "_capability", cap)
    object.__setattr__(provider, "_state", init_state)
    return provider


# ==============================================================================
# RED Test Suite
# ==============================================================================


class TestRedContinuousReplayInvariants:
    """RED test suite enforcing fail-closed constraints on replay and smoothing."""

    def test_red_per_frame_state_resets_fails_closed(self) -> None:
        """Replay must not reset state at every knot; fails closed."""
        fixture = make_fixed_base_pendulum_fixture(n_frames=5, fps=100.0)
        provider = AnalyticPendulumProvider(fixture)
        init_state = provider.get_state()
        controls = np.zeros((4, 1), dtype=np.float64)

        # Attempt intermediate state reset at knot 2
        bad_resets = [(2, init_state)]
        with pytest.raises(
            PreconditionError, match="Per-frame state resets are strictly forbidden"
        ):
            execute_continuous_replay(
                provider=provider,
                initial_state=init_state,
                controls=controls,
                dt=0.01,
                options=ContinuousReplayOptions(intermediate_resets=bad_resets),
            )

    def test_red_hidden_target_force_feedback_fails_closed(self) -> None:
        """Replay rejects unmodeled external force / assistance feedback."""
        fixture = make_fixed_base_pendulum_fixture(n_frames=5, fps=100.0)
        provider = AnalyticPendulumProvider(fixture)
        init_state = provider.get_state()
        controls = np.zeros((4, 1), dtype=np.float64)

        # Hidden assistance forces pulling towards target
        assistance = np.ones((4, 1), dtype=np.float64) * 5.0
        with pytest.raises(
            PreconditionError, match="Hidden target-force feedback detected"
        ):
            execute_continuous_replay(
                provider=provider,
                initial_state=init_state,
                controls=controls,
                dt=0.01,
                assistance_forces=assistance,
            )

    def test_red_undeclared_root_wrench_fails_closed(self) -> None:
        """Floating-base root has strictly zero artificial actuator forces."""
        provider = _make_floating_base_provider("model-fb-root")
        init_state = provider.get_state()

        # Non-zero forces on unactuated root DOFs (e.g. root_pz = 20.0 N)
        controls = np.zeros((4, 7), dtype=np.float64)
        controls[:, 2] = 20.0  # Artificial vertical root force!

        options = ContinuousReplayOptions(
            floating_base_root_dofs=(0, 1, 2, 3, 4, 5),
            allow_undeclared_root_forces=False,
        )
        with pytest.raises(PreconditionError, match="Undeclared root wrench detected"):
            execute_continuous_replay(
                provider=provider,
                initial_state=init_state,
                controls=controls,
                dt=0.01,
                options=options,
            )

    def test_red_replay_with_changed_model_fails_closed(self) -> None:
        """Replay with mismatched model configuration or parameters rejected."""
        fixture = make_fixed_base_pendulum_fixture(n_frames=5, fps=100.0)
        provider = AnalyticPendulumProvider(fixture)
        init_state = provider.get_state()
        controls = np.zeros((4, 1), dtype=np.float64)

        options = ContinuousReplayOptions(expected_model_hash="sha256-different-model")
        with pytest.raises(PreconditionError, match="Model hash mismatch"):
            execute_continuous_replay(
                provider=provider,
                initial_state=init_state,
                controls=controls,
                dt=0.01,
                options=options,
            )

    def test_red_missing_controls_fails_closed(self) -> None:
        """Replay with missing control channels or wrong shape rejected."""
        fixture = make_fixed_base_pendulum_fixture(n_frames=5, fps=100.0)
        provider = AnalyticPendulumProvider(fixture)
        init_state = provider.get_state()

        # Provider requires 1 control channel, we supply 0 channels
        empty_controls = np.zeros((4, 0), dtype=np.float64)
        with pytest.raises(
            PreconditionError, match="Missing controls or dimension mismatch"
        ):
            execute_continuous_replay(
                provider=provider,
                initial_state=init_state,
                controls=empty_controls,
                dt=0.01,
            )

        # Control steps fewer than 1
        with pytest.raises(PreconditionError, match="controls must have >= 1 step"):
            execute_continuous_replay(
                provider=provider,
                initial_state=init_state,
                controls=np.zeros((0, 1), dtype=np.float64),
                dt=0.01,
            )

    def test_red_backward_inference_reverse_time_contact_fails_closed(self) -> None:
        """Backward inference calling reverse-time contact integration fails closed."""
        fixture = make_fixed_base_pendulum_fixture(n_frames=5, fps=100.0)
        provider = AnalyticPendulumProvider(fixture)
        init_state = provider.get_state()
        states = [init_state, init_state]

        with pytest.raises(
            PreconditionError, match="Reverse-time contact integration is forbidden"
        ):
            smooth_backward_trajectory(
                states=states,
                allow_reverse_contact=True,
            )

        with pytest.raises(
            PreconditionError, match="dt must be strictly positive and finite"
        ):
            smooth_backward_trajectory(
                states=states,
                dt=-0.01,
            )


# ==============================================================================
# GREEN Test Suite
# ==============================================================================


class TestGreenContinuousReplayAndSmoothing:
    """GREEN test suite verifying reproducible replay and offline smoothing."""

    def test_green_uninterrupted_synthetic_replay_reproduces_trajectory_within_frozen_tolerances(
        self,
    ) -> None:
        """Uninterrupted synthetic replay reproduces trajectory within frozen numerical tolerances."""
        n_frames = 10
        fps = 100.0
        fixture = make_fixed_base_pendulum_fixture(n_frames=n_frames, fps=fps)
        provider = AnalyticPendulumProvider(fixture)
        init_state = provider.get_state()
        dt = 1.0 / fps

        # Build reference trajectory from fixture
        ref_states = []
        for i in range(n_frames):
            frame = fixture.frames[i]
            q_val = np.array(frame.q, dtype=np.float64)
            v_val = np.array(frame.qdot, dtype=np.float64)
            ref_states.append(
                DimeCompleteState(
                    t=frame.timestamp,
                    q=q_val,
                    v=v_val,
                    model_hash=provider.model_hash,
                    units=dict(CANONICAL_DIME_UNITS),
                )
            )

        controls = np.zeros((n_frames - 1, 1), dtype=np.float64)
        options = ContinuousReplayOptions(
            declared_controller="open_loop_feedforward",
            declared_contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            tolerances=NumericAcceptanceThresholds(
                reproducibility_atol=1e-9, max_drift_m=0.015
            ),
        )

        result = execute_continuous_replay(
            provider=provider,
            initial_state=init_state,
            controls=controls,
            dt=dt,
            options=options,
            reference_trajectory=ref_states,
        )

        assert isinstance(result, ContinuousReplayResult)
        assert len(result.trajectory) == n_frames
        assert result.is_physically_accepted
        assert result.receipt.reset_count == 1
        assert len(result.receipt.assistance_channels) == 0
        assert not result.receipt.has_undeclared_root_forces

        metrics = result.metrics
        assert isinstance(metrics, IndependentReplayMetrics)
        assert metrics.max_position_drift_m <= 0.015
        assert metrics.satisfies_frozen_tolerances

        # Exact reproducibility on fresh run from same initial state and controls
        result_repro = execute_continuous_replay(
            provider=provider,
            initial_state=init_state,
            controls=controls,
            dt=dt,
            options=options,
        )
        repro_err = max(
            float(np.linalg.norm(s1.q - s2.q))
            for s1, s2 in zip(result.trajectory, result_repro.trajectory, strict=True)
        )
        assert repro_err <= 1e-9

    def test_native_stance_fixture_with_gravity_fake_provider_not_accepted(
        self,
    ) -> None:
        """The fake provider falls under gravity, so it must not be accepted (#11551).

        This test previously asserted acceptance only because a missing
        reference reported reproducibility 0.0. Against the saved static
        fixture the fake provider drifts beyond the frozen tolerance.
        """
        fixture = make_native_stance_fixture(n_frames=8, fps=100.0)
        q0, v0 = fixture.initial_state
        dt = 1.0 / fixture.sampling_rate_hz

        # Create provider for 3-DOF stance model with 3 control channels
        channels = tuple(
            ControlChannelSpec(
                name=f"joint_{i}",
                physical_type="torque",
                units="N*m",
                selection_map=(i,),
                limits=(-500.0, 500.0),
            )
            for i in range(3)
        )
        cap = ProviderCapability(
            provider_id="native-stance-provider",
            version="1.0.0",
            status="implemented",
            n_q=3,
            n_v=3,
            manifold=VectorSpaceManifold(3),
            control_channels=channels,
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            retained_passive_loads=(),
            is_qualified=False,
        )
        provider = DeterministicFakeProvider(
            n_q=3, n_v=3, model_hash="native-stance-model-v1"
        )
        object.__setattr__(provider, "_capability", cap)

        init_state = DimeCompleteState(
            t=0.0,
            q=np.array(q0, dtype=np.float64),
            v=np.array(v0, dtype=np.float64),
            model_hash=provider.model_hash,
            units=dict(CANONICAL_DIME_UNITS),
        )
        provider.set_state(init_state)

        # Controls matching fixture holding torques
        holding_controls = fixture.controls[:-1, :]
        options = ContinuousReplayOptions(
            declared_controller="pure_torque",
            expected_model_hash="native-stance-model-v1",
            floating_base_root_dofs=(0,),
        )
        # Qualification requires a saved-motion reference (issue #11551).
        reference = [
            DimeCompleteState(
                t=frame.timestamp,
                q=np.array(frame.q, dtype=np.float64),
                v=np.array(frame.qdot, dtype=np.float64),
                model_hash=provider.model_hash,
                units=dict(CANONICAL_DIME_UNITS),
            )
            for frame in fixture.frames
        ]

        result = execute_continuous_replay(
            provider=provider,
            initial_state=init_state,
            controls=holding_controls,
            dt=dt,
            options=options,
            reference_trajectory=reference,
        )
        assert not result.is_physically_accepted
        assert result.metrics.max_position_drift_m > 0.015
        assert result.receipt.reset_count == 1
        assert len(result.trajectory) == 8

        # Position drift across quiet stance must be minimal
        q_start = result.trajectory[0].q
        q_end = result.trajectory[-1].q
        drift = float(np.linalg.norm(q_end - q_start))
        assert drift <= 0.05

    def test_green_backward_smoothing_with_marginalized_arrival_information(
        self,
    ) -> None:
        """Backward smoothing produces continuous smoothed trajectory without per-frame discontinuities."""
        n_samples = 15
        times = np.linspace(0.0, 1.4, n_samples)
        true_q = 0.5 * np.sin(times)

        # Add zero-mean high-frequency jitter simulating noisy raw estimates
        rng = np.random.default_rng(42)
        noisy_q = true_q + rng.normal(0.0, 0.03, size=n_samples)

        states = [
            DimeCompleteState(
                t=float(times[i]),
                q=np.array([noisy_q[i]], dtype=np.float64),
                v=np.array([0.0], dtype=np.float64),
                model_hash="smooth-model",
                units=dict(CANONICAL_DIME_UNITS),
            )
            for i in range(n_samples)
        ]

        # Generate arrival factors from linear transitions
        arrival_factors = []
        for i in range(n_samples):
            a_factor = marginalize_arrival_factor(
                A0=np.array([[1.0]]),
                A1=np.array([[0.9]]),
                b=np.array([noisy_q[i]]),
                reference_state_k1=np.array([noisy_q[i]]),
            )
            arrival_factors.append(a_factor)

        smoothed_res = smooth_backward_trajectory(
            states=states,
            arrival_factors=arrival_factors,
            dt=0.1,
        )

        assert isinstance(smoothed_res, SmoothedTrajectoryResult)
        assert len(smoothed_res.smoothed_states) == n_samples
        assert smoothed_res.is_continuous

        # Smoothed jumps must be substantially smaller and smoother than raw noisy steps
        raw_jumps = np.diff(noisy_q)
        smoothed_q = np.array([s.q[0] for s in smoothed_res.smoothed_states])
        smoothed_jumps = np.diff(smoothed_q)

        assert np.max(np.abs(smoothed_jumps)) < np.max(np.abs(raw_jumps))
        assert smoothed_res.max_step_jump < 0.15

    def test_green_continuous_replay_result_structure_and_receipt(self) -> None:
        """Verify complete ContinuousReplayResult structure, receipt serialization, and provenance."""
        fixture = make_fixed_base_pendulum_fixture(n_frames=5, fps=100.0)
        provider = AnalyticPendulumProvider(fixture)
        init_state = provider.get_state()
        dt = 0.01
        controls = np.zeros((4, 1), dtype=np.float64)

        custom_prov = DimeProvenanceRecord(
            engine="DimeSyntheticHarmonic",
            engine_version="1.0.0",
            model_hash=provider.model_hash,
            param_hash="param-hash-01",
            git_commit="git-commit-hash-01",
            created_at="2026-10-04T00:00:00Z",
            seed=42,
            notes="Testing replay receipt structure",
        )

        result = execute_continuous_replay(
            provider=provider,
            initial_state=init_state,
            controls=controls,
            dt=dt,
            provenance=custom_prov,
        )

        # Receipt checks
        receipt = result.receipt
        assert receipt.schema_version == "dime-continuous-replay-receipt/1.0"
        assert receipt.model_hash == provider.model_hash
        assert receipt.reset_count == 1
        assert receipt.coverage_start_s == 0.0
        assert receipt.coverage_end_s == pytest.approx(0.04)
        assert receipt.provenance.git_commit == "git-commit-hash-01"

        # Serialization round-trip
        rec_dict = receipt.to_dict()
        restored_receipt = ReplayReceipt.from_dict(rec_dict)
        assert restored_receipt.receipt_id == receipt.receipt_id
        assert restored_receipt.reset_count == 1
        assert restored_receipt.is_physically_accepted == receipt.is_physically_accepted

        res_dict = result.to_dict()
        restored_result = ContinuousReplayResult.from_dict(res_dict)
        assert len(restored_result.trajectory) == len(result.trajectory)
        assert restored_result.is_physically_accepted == result.is_physically_accepted

    def test_green_shadow_tracker_and_simscape_adapters(self) -> None:
        """Verify public adapters connect cleanly to Shadow Tracker and Simscape replay harnesses."""
        fixture = make_fixed_base_pendulum_fixture(n_frames=6, fps=100.0)
        provider = AnalyticPendulumProvider(fixture)
        init_state = provider.get_state()
        dt = 0.01
        controls = np.zeros((5, 1), dtype=np.float64)
        times = [i * dt for i in range(6)]

        # 1. Shadow Tracker RolloutRequest adapter
        req = to_shadow_tracker_rollout_request(
            initial_state=init_state,
            controls=controls,
            times=times,
        )
        assert req.initial_state == (0.1,)
        assert len(req.controls) == 1
        assert len(req.controls[0]) == 5
        assert req.time_points_s == tuple(times)

        # 2. Simscape ContinuousReplayTrajectory adapter
        replay_result = execute_continuous_replay(
            provider=provider,
            initial_state=init_state,
            controls=controls,
            dt=dt,
        )
        simscape_traj = to_simscape_continuous_trajectory(
            replay_result=replay_result,
            coordinate_names=("pendulum_theta",),
            marker_labels=("marker_tip",),
        )
        assert simscape_traj.time_s.shape == (6,)
        assert simscape_traj.q.shape == (6, 1)
        assert simscape_traj.v.shape == (6, 1)
        assert simscape_traj.coordinate_names == ("pendulum_theta",)
        assert simscape_traj.marker_labels == ("marker_tip",)


class TestFailClosedOnMissingEvidence:
    """Issue #11551: absent reference or provenance sources never imply acceptance."""

    @staticmethod
    def _run(**kwargs: object) -> ContinuousReplayResult:
        fixture = make_fixed_base_pendulum_fixture(n_frames=5, fps=100.0)
        provider = AnalyticPendulumProvider(fixture)
        return execute_continuous_replay(
            provider=provider,
            initial_state=provider.get_state(),
            controls=np.zeros((4, 1), dtype=np.float64),
            dt=0.01,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_red_no_reference_is_not_accepted(self) -> None:
        result = self._run()
        assert result.is_physically_accepted is False
        assert result.receipt.is_physically_accepted is False
        assert result.metrics.satisfies_frozen_tolerances is False
        assert any("reference" in r.lower() for r in result.unqualified_reasons)

    def test_red_no_reference_reproducibility_is_not_measured(self) -> None:
        result = self._run()
        assert result.metrics.reproducibility_error is None
        assert result.metrics.alignment_metric is None
        restored = ContinuousReplayResult.from_dict(result.to_dict())
        assert restored.metrics.reproducibility_error is None
        assert restored.unqualified_reasons == result.unqualified_reasons

    def test_red_provenance_uses_injected_sources(self) -> None:
        fixed = datetime(2031, 2, 3, 4, 5, 6, tzinfo=UTC)
        sources = ReplayProvenanceSources(
            git_commit=lambda: "a" * 40, clock=lambda: fixed
        )
        result = self._run(options=ContinuousReplayOptions(provenance_sources=sources))
        prov = result.provenance
        assert prov.git_commit == "a" * 40
        assert prov.created_at == "2031-02-03T04:05:06Z"
        assert result.receipt.provenance == prov

    def test_red_provenance_is_not_the_old_hardcoded_literals(self) -> None:
        prov = self._run().provenance
        assert prov.git_commit != "dime-09-replay-commit"
        assert prov.created_at != "2026-10-04T00:00:00Z"

    def test_red_unknown_git_is_explicit_marker_not_fake_sha(self) -> None:
        sources = ReplayProvenanceSources(git_commit=lambda: "unknown")
        result = self._run(options=ContinuousReplayOptions(provenance_sources=sources))
        assert result.provenance.git_commit == "unknown"

    def test_red_reset_count_is_derived_from_replay(self) -> None:
        result = self._run()
        assert result.receipt.reset_count == 1
