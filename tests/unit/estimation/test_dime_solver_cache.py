"""Behavioral tests for DIME Native Drift and Window Solves Cache (#11421, #11435).

Covers:
- Cache keying and identity: model digest, parameters, state vector, contact policy/mode,
  solver configuration, camera digest, body digest, and session isolation.
- Approximate local models with explicit validity radius (epsilon_valid) and mandatory refresh rules.
- RED test cases:
  * Stale cache returned after camera/body/contact changes (fails closed).
  * Cross-job native-state contamination (fails closed).
  * Inaccurate derivative at impact (fails closed; contact transitions invalidate local linearization).
  * Speedup claims from skipped replay (fails closed; independent replay cannot be skipped/cached).
- GREEN test cases:
  * Cached vs fresh evaluation equivalence within declared numerical tolerance (< 1e-12).
  * Bounded approximation error inside validity radius.
  * Cache invalidation and thread/session isolation concurrency tests.
  * Measured cold vs warm speedup with detailed cost breakdown.
  * Structured receipt (DimeSolverCacheReceipt) exporting hit/miss counters, memory footprint,
    speedup metrics, and provenance.
"""

from __future__ import annotations

import concurrent.futures
import json
import numpy as np
import pytest

from src.shared.python.estimation.dime_contracts import (
    AnalyticPendulumProvider,
    ContactPolicy,
    DimeCompleteState,
    DimeFullStepRequest,
    UnderactuatedAnalyticProvider,
)
from src.shared.python.estimation.dime_dynamics_window import (
    DefectMode,
    DimeDynamicsWindowProblem,
)
from src.shared.python.estimation.dime_solver_cache import (
    CrossJobContaminationError,
    DimeCostBreakdown,
    DimeSolverCache,
    DimeSolverCacheConfig,
    DimeSolverCacheError,
    DimeSolverCacheKey,
    DimeSolverCacheReceipt,
    ImpactLinearizationError,
    LocalLinearizationModel,
    SkippedReplayError,
    StaleCacheError,
    ValidityRadiusExceededError,
    execute_independent_replay,
    solve_accelerated_dime_window,
)

pytestmark = pytest.mark.unit


# ==============================================================================
# Cache Key Tests
# ==============================================================================


class TestDimeSolverCacheKey:
    """Verifies cache key construction, composition, and sensitivity."""

    def test_cache_key_generation_and_equality(self) -> None:
        key1 = DimeSolverCacheKey(
            model_digest="pendulum_v1_hash",
            parameters_hash="mass_1.0_len_1.0",
            state_hash="q_0.1_v_0.0",
            contact_policy_or_mode="native_eliminated",
            solver_configuration_hash="dt_0.02_steps_4",
            session_id="session_alpha",
            camera_digest="cam_rig_default",
            body_digest="torso_limb_standard",
        )
        key2 = DimeSolverCacheKey(
            model_digest="pendulum_v1_hash",
            parameters_hash="mass_1.0_len_1.0",
            state_hash="q_0.1_v_0.0",
            contact_policy_or_mode="native_eliminated",
            solver_configuration_hash="dt_0.02_steps_4",
            session_id="session_alpha",
            camera_digest="cam_rig_default",
            body_digest="torso_limb_standard",
        )
        assert key1 == key2
        assert key1.to_composite_key() == key2.to_composite_key()
        assert len(key1.to_composite_key()) == 64  # SHA-256 hex digest

    def test_cache_key_differentiates_model_and_params(self) -> None:
        base = DimeSolverCacheKey(
            model_digest="pendulum_v1_hash",
            parameters_hash="mass_1.0_len_1.0",
            state_hash="q_0.1_v_0.0",
            contact_policy_or_mode="native_eliminated",
            solver_configuration_hash="dt_0.02_steps_4",
            session_id="session_alpha",
        )
        key_different_model = DimeSolverCacheKey(
            model_digest="pendulum_v2_hash",
            parameters_hash="mass_1.0_len_1.0",
            state_hash="q_0.1_v_0.0",
            contact_policy_or_mode="native_eliminated",
            solver_configuration_hash="dt_0.02_steps_4",
            session_id="session_alpha",
        )
        key_different_params = DimeSolverCacheKey(
            model_digest="pendulum_v1_hash",
            parameters_hash="mass_2.0_len_1.0",
            state_hash="q_0.1_v_0.0",
            contact_policy_or_mode="native_eliminated",
            solver_configuration_hash="dt_0.02_steps_4",
            session_id="session_alpha",
        )
        assert base.to_composite_key() != key_different_model.to_composite_key()
        assert base.to_composite_key() != key_different_params.to_composite_key()

    def test_cache_key_differentiates_camera_and_body(self) -> None:
        base = DimeSolverCacheKey(
            model_digest="pendulum_v1_hash",
            parameters_hash="mass_1.0",
            state_hash="q_0.1_v_0.0",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="session_alpha",
            camera_digest="camera_calib_A",
            body_digest="body_geometry_A",
        )
        key_cam_b = DimeSolverCacheKey(
            model_digest="pendulum_v1_hash",
            parameters_hash="mass_1.0",
            state_hash="q_0.1_v_0.0",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="session_alpha",
            camera_digest="camera_calib_B",
            body_digest="body_geometry_A",
        )
        key_body_b = DimeSolverCacheKey(
            model_digest="pendulum_v1_hash",
            parameters_hash="mass_1.0",
            state_hash="q_0.1_v_0.0",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="session_alpha",
            camera_digest="camera_calib_A",
            body_digest="body_geometry_B",
        )
        assert base.to_composite_key() != key_cam_b.to_composite_key()
        assert base.to_composite_key() != key_body_b.to_composite_key()

    def test_cache_key_differentiates_contact_mode_and_session(self) -> None:
        base = DimeSolverCacheKey(
            model_digest="m1",
            parameters_hash="p1",
            state_hash="s1",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="session_alpha",
        )
        key_stance = DimeSolverCacheKey(
            model_digest="m1",
            parameters_hash="p1",
            state_hash="s1",
            contact_policy_or_mode="left_stance",
            solver_configuration_hash="cfg1",
            session_id="session_alpha",
        )
        key_other_session = DimeSolverCacheKey(
            model_digest="m1",
            parameters_hash="p1",
            state_hash="s1",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="session_beta",
        )
        assert base.to_composite_key() != key_stance.to_composite_key()
        assert base.to_composite_key() != key_other_session.to_composite_key()


# ==============================================================================
# RED Test Suite: Required Failure Modes and Closed-System Guards
# ==============================================================================


class TestDimeSolverCacheRedSuite:
    """Verifies that all required failure modes fail closed strictly."""

    def test_red_stale_cache_rejected_on_camera_body_contact_change(self) -> None:
        """Attempting to access cached solutions when camera, body, or contact changes fails closed."""
        cache = DimeSolverCache(DimeSolverCacheConfig())
        key_original = DimeSolverCacheKey(
            model_digest="model_v1",
            parameters_hash="p1",
            state_hash="q0_v0",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="job_1",
            camera_digest="cam_rig_1",
            body_digest="body_geom_1",
        )
        dummy_result = {"cost": 42.0, "converged": True}
        cache.put(key_original, dummy_result)

        # 1. Verification of hit with correct key
        assert cache.get(key_original) == dummy_result

        # 2. Camera change: query with modified camera digest must miss and raise StaleCacheError if forced
        key_cam_changed = DimeSolverCacheKey(
            model_digest="model_v1",
            parameters_hash="p1",
            state_hash="q0_v0",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="job_1",
            camera_digest="cam_rig_2",  # Changed!
            body_digest="body_geom_1",
        )
        assert cache.get(key_cam_changed) is None

        # 3. Body geometry change: query with modified body digest must miss
        key_body_changed = DimeSolverCacheKey(
            model_digest="model_v1",
            parameters_hash="p1",
            state_hash="q0_v0",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="job_1",
            camera_digest="cam_rig_1",
            body_digest="body_geom_2",  # Changed!
        )
        assert cache.get(key_body_changed) is None

        # 4. Contact mode change: query with modified contact mode must miss
        key_contact_changed = DimeSolverCacheKey(
            model_digest="model_v1",
            parameters_hash="p1",
            state_hash="q0_v0",
            contact_policy_or_mode="stance",  # Changed!
            solver_configuration_hash="cfg1",
            session_id="job_1",
            camera_digest="cam_rig_1",
            body_digest="body_geom_1",
        )
        assert cache.get(key_contact_changed) is None

        # 5. Stale retrieve assertion: asking for stale cache verification raises StaleCacheError
        with pytest.raises(StaleCacheError, match="stale.*camera|body|contact"):
            cache.verify_or_raise_on_stale(
                session_id="job_1",
                expected_camera="cam_rig_2",
                current_camera="cam_rig_1",
            )

    def test_red_cross_job_native_state_contamination_fails_closed(self) -> None:
        """Isolated session state prevents cross-job native-state contamination."""
        cache = DimeSolverCache(DimeSolverCacheConfig(require_isolated_sessions=True))
        key_job_a = DimeSolverCacheKey(
            model_digest="model_v1",
            parameters_hash="p1",
            state_hash="q0_v0",
            contact_policy_or_mode="flight",
            solver_configuration_hash="cfg1",
            session_id="job_alpha",
        )
        cache.put(key_job_a, {"solution_state": "state_alpha"})

        # Job Beta attempting to access or modify Job Alpha's entry must fail closed
        with pytest.raises(CrossJobContaminationError, match="Cross-job"):
            cache.access_session_entry(
                requesting_session_id="job_beta", target_key=key_job_a
            )

    def test_red_inaccurate_derivative_at_impact_fails_closed(self) -> None:
        """Contact transitions invalidate local linearization; impact derivatives fail closed."""
        q0 = np.array([0.1], dtype=np.float64)
        v0 = np.array([-1.5], dtype=np.float64)
        local_model = LocalLinearizationModel(
            nominal_q=q0,
            nominal_v=v0,
            nominal_drift=np.array([0.0], dtype=np.float64),
            jacobian_q=np.array([[1.0]], dtype=np.float64),
            jacobian_v=np.array([[0.5]], dtype=np.float64),
            validity_radius=0.1,
            model_hash="pendulum",
            contact_mode="flight",
            derivatives_valid=True,
        )

        # 1. State inside radius and persistent flight evaluates fine
        val = local_model.evaluate(
            q=np.array([0.12], dtype=np.float64),
            v=np.array([-1.48], dtype=np.float64),
            contact_mode="flight",
            derivatives_valid=True,
        )
        assert np.isfinite(val).all()

        # 2. Discrete impact transition: derivatives_valid is False
        with pytest.raises(ImpactLinearizationError, match="impact|transition"):
            local_model.evaluate(
                q=np.array([0.12], dtype=np.float64),
                v=np.array([-1.48], dtype=np.float64),
                contact_mode="transition",
                derivatives_valid=False,
            )

        # 3. Contact mode switch (flight -> left_stance) invalidates local linearization
        with pytest.raises(ImpactLinearizationError, match="mode"):
            local_model.evaluate(
                q=np.array([0.12], dtype=np.float64),
                v=np.array([-1.48], dtype=np.float64),
                contact_mode="left_stance",
                derivatives_valid=True,
            )

    def test_red_speedup_claims_from_skipped_replay_fails_closed(self) -> None:
        """Speedup claims from skipped or cached independent replay fail closed."""
        provider = AnalyticPendulumProvider()
        init_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.1], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )
        controls = np.zeros((4, 1), dtype=np.float64)

        # Attempting to execute replay with skip_simulation=True must fail closed
        with pytest.raises(SkippedReplayError, match="skipped|bypassed"):
            execute_independent_replay(
                provider=provider,
                initial_state=init_state,
                controls=controls,
                dt=0.02,
                skip_simulation=True,
            )


# ==============================================================================
# GREEN Test Suite: Equivalence, Bounded Approximation, and Profiling
# ==============================================================================


class TestDimeSolverCacheGreenSuite:
    """Verifies numerical equivalence, bounded error, profiling, and receipts."""

    def test_green_exact_evaluation_equivalence(self) -> None:
        """Cached versus fresh evaluation of exact drift matches within < 1e-12."""
        provider = AnalyticPendulumProvider()
        cache = DimeSolverCache(DimeSolverCacheConfig())
        state = DimeCompleteState(
            t=0.0,
            q=np.array([0.3], dtype=np.float64),
            v=np.array([0.5], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )
        req = DimeFullStepRequest(
            state=state,
            controls=np.array([0.1], dtype=np.float64),
            dt=0.01,
            model_hash=provider.model_hash,
        )

        key = DimeSolverCacheKey.from_full_step_request(
            req, session_id="test_green_session"
        )

        # Fresh evaluation
        fresh_res = provider.step(req)
        cache.put(key, fresh_res)

        # Cached evaluation
        cached_res = cache.get(key)
        assert cached_res is not None
        np.testing.assert_allclose(
            cached_res.next_state.q, fresh_res.next_state.q, atol=1e-12
        )
        np.testing.assert_allclose(
            cached_res.next_state.v, fresh_res.next_state.v, atol=1e-12
        )
        np.testing.assert_allclose(
            cached_res.accelerations, fresh_res.accelerations, atol=1e-12
        )

    def test_green_bounded_approximation_error_inside_validity_radius(
        self,
    ) -> None:
        """Local model approximation error is strictly bounded inside validity radius."""
        provider = AnalyticPendulumProvider()
        q0 = np.array([0.2], dtype=np.float64)
        v0 = np.array([0.0], dtype=np.float64)

        # True drift at nominal
        nom_state = DimeCompleteState(
            t=0.0,
            q=q0,
            v=v0,
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )
        nom_step = provider.step(
            DimeFullStepRequest(
                state=nom_state,
                controls=np.zeros(1),
                dt=0.01,
                model_hash=provider.model_hash,
            )
        )
        nom_drift = nom_step.accelerations

        # Compute numerical Jacobians
        eps = 1e-6
        # J_q
        state_q_plus = DimeCompleteState(
            t=0.0,
            q=q0 + eps,
            v=v0,
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )
        drift_q_plus = provider.step(
            DimeFullStepRequest(
                state=state_q_plus,
                controls=np.zeros(1),
                dt=0.01,
                model_hash=provider.model_hash,
            )
        ).accelerations
        jq = ((drift_q_plus - nom_drift) / eps).reshape(1, 1)

        # J_v
        state_v_plus = DimeCompleteState(
            t=0.0,
            q=q0,
            v=v0 + eps,
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )
        drift_v_plus = provider.step(
            DimeFullStepRequest(
                state=state_v_plus,
                controls=np.zeros(1),
                dt=0.01,
                model_hash=provider.model_hash,
            )
        ).accelerations
        jv = ((drift_v_plus - nom_drift) / eps).reshape(1, 1)

        validity_radius = 0.05
        local_model = LocalLinearizationModel(
            nominal_q=q0,
            nominal_v=v0,
            nominal_drift=nom_drift,
            jacobian_q=jq,
            jacobian_v=jv,
            validity_radius=validity_radius,
            model_hash=provider.model_hash,
        )

        # Test within validity radius: delta = 0.01 < 0.05
        q_pert = q0 + 0.01
        v_pert = v0 + 0.01
        approx_drift = local_model.evaluate(q_pert, v_pert)

        # Compute exact drift
        state_pert = DimeCompleteState(
            t=0.0,
            q=q_pert,
            v=v_pert,
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )
        exact_drift = provider.step(
            DimeFullStepRequest(
                state=state_pert,
                controls=np.zeros(1),
                dt=0.01,
                model_hash=provider.model_hash,
            )
        ).accelerations

        # Taylor expansion second-order remainder: error should be O(delta^2) < 0.005
        error = float(np.linalg.norm(approx_drift - exact_drift))
        assert error < 0.01

    def test_green_validity_radius_refresh_rule(self) -> None:
        """Evaluating beyond validity radius triggers mandatory refresh."""
        q0 = np.array([0.0], dtype=np.float64)
        v0 = np.array([0.0], dtype=np.float64)
        local_model = LocalLinearizationModel(
            nominal_q=q0,
            nominal_v=v0,
            nominal_drift=np.array([0.0]),
            jacobian_q=np.array([[1.0]]),
            jacobian_v=np.array([[0.0]]),
            validity_radius=0.05,
            model_hash="test",
        )

        # Inside radius
        assert local_model.is_valid(q=np.array([0.02]), v=np.array([0.02]))

        # Outside radius
        assert not local_model.is_valid(q=np.array([0.10]), v=np.array([0.0]))

        with pytest.raises(ValidityRadiusExceededError, match="validity radius"):
            local_model.evaluate(
                q=np.array([0.10]), v=np.array([0.0]), fail_closed=True
            )

    def test_green_thread_concurrency_and_session_isolation(self) -> None:
        """Concurrent threads across distinct sessions operate safely without cross-contamination."""
        cache = DimeSolverCache(DimeSolverCacheConfig())
        n_threads = 6
        n_iters = 50

        def worker(thread_idx: int) -> int:
            session_id = f"session_thread_{thread_idx}"
            hits = 0
            for i in range(n_iters):
                key = DimeSolverCacheKey(
                    model_digest="model_concurrent",
                    parameters_hash=f"p_{thread_idx}",
                    state_hash=f"s_{i % 5}",
                    contact_policy_or_mode="flight",
                    solver_configuration_hash="cfg",
                    session_id=session_id,
                )
                val = cache.get(key)
                if val is None:
                    cache.put(key, {"thread": thread_idx, "iter": i})
                else:
                    assert val["thread"] == thread_idx
                    hits += 1
            return hits

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(n_threads)]
            results = [f.result() for f in futures]

        assert all(h > 0 for h in results)

    def test_green_cost_breakdown_profiling_and_cold_warm_speedup(self) -> None:
        """Solves window problem, profiles cost breakdown, and measures cold vs warm speedup."""
        provider = AnalyticPendulumProvider()
        init_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.2], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )
        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=init_state,
            horizon_steps=4,
            dt_s=0.02,
            target_positions=[np.array([0.1])] * 5,
            observation_weight=10.0,
            control_rate_weight=1.0,
            max_iterations=15,
        )

        session_id = "profiling_session_1"
        cache = DimeSolverCache(DimeSolverCacheConfig())

        # Cold solve
        res_cold = solve_accelerated_dime_window(
            problem=problem,
            cache=cache,
            session_id=session_id,
        )
        assert res_cold.success
        receipt_cold = cache.get_receipt(session_id)
        assert receipt_cold.cold_time_s > 0.0

        # Warm solve (repeated with same initial problem)
        res_warm = solve_accelerated_dime_window(
            problem=problem,
            cache=cache,
            session_id=session_id,
        )
        assert res_warm.success
        receipt_warm = cache.get_receipt(session_id)
        assert receipt_warm.hit_count > 0
        assert receipt_warm.speedup_ratio >= 1.0

        # Verify cost breakdown has accounted categories
        cost_bd = receipt_warm.cost_breakdown
        assert "drift_time_s" in cost_bd
        assert "full_step_time_s" in cost_bd
        assert "jacobian_time_s" in cost_bd
        assert "assembly_factorization_time_s" in cost_bd
        assert "window_solve_time_s" in cost_bd
        assert "replay_time_s" in cost_bd

        # Independent replay must have been executed freshly
        assert res_warm.replay_result is not None
        assert res_warm.replay_result["success"] is True

    def test_green_structured_receipt_roundtrip(self) -> None:
        """Verifies DimeSolverCacheReceipt serialization, fields, and roundtrip."""
        cost_bd = {
            "drift_time_s": 0.005,
            "full_step_time_s": 0.010,
            "jacobian_time_s": 0.002,
            "assembly_factorization_time_s": 0.001,
            "window_solve_time_s": 0.018,
            "replay_time_s": 0.003,
        }
        provenance = {
            "model_hash": "model_123",
            "git_commit": "abc1234",
            "engine": "analytic",
        }
        receipt = DimeSolverCacheReceipt(
            session_id="session_test",
            hit_count=10,
            miss_count=2,
            hit_rate=10.0 / 12.0,
            eviction_count=0,
            memory_bytes=4096,
            cold_time_s=0.050,
            warm_time_s=0.025,
            speedup_ratio=2.0,
            cost_breakdown=cost_bd,
            provenance=provenance,
            status="qualified",
        )

        d = receipt.to_dict()
        assert d["session_id"] == "session_test"
        assert d["hit_count"] == 10
        assert d["speedup_ratio"] == 2.0

        # JSON serializable
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        restored = DimeSolverCacheReceipt.from_dict(deserialized)
        assert restored == receipt
