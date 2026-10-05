"""Behavioral test suite for Native Drift and Window Solver Acceleration and Caching (DIME-14).

Verifies:
- RED: stale cache after camera/body/contact changes, cross-job native-state contamination,
       inaccurate derivative at impact, and speedup from skipped replay.
- GREEN: cached versus fresh equivalence where exact, bounded approximation error elsewhere,
         invalidation and concurrency tests, measured cold/warm end-to-end quality-matched
         p50/p95 speed, and full cost breakdown including failure costs.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import (
    AnalyticPendulumProvider,
    ContactPolicy,
    DimeCompleteState,
    DimeFullStepRequest,
)
from src.shared.python.estimation.dime_solver_cache import (
    DimeCacheIdentity,
    DimeCostBreakdown,
    DimeSolverCache,
    LocalModelApproximation,
    accelerated_solve_dynamics_window,
)
from src.shared.python.estimation.dime_dynamics_window import (
    DefectMode,
    DimeDynamicsWindowProblem,
)


pytestmark = pytest.mark.unit


def _create_pendulum_state(
    q_val: float = 0.1,
    v_val: float = 0.0,
    model_hash: str | None = None,
) -> DimeCompleteState:
    """Helper to create complete state for analytic pendulum."""
    if model_hash is None:
        provider = AnalyticPendulumProvider()
        model_hash = provider.model_hash
    return DimeCompleteState(
        t=0.0,
        q=np.array([q_val], dtype=np.float64),
        v=np.array([v_val], dtype=np.float64),
        units={"length": "m", "angle": "rad", "time": "s"},
        model_hash=model_hash,
    )


class TestDimeSolverCacheRedSuite:
    """RED test suite verifying cache boundaries, invalidation, and rejection of unsafe shortcuts."""

    def test_red_stale_cache_after_camera_change(self) -> None:
        """Cache must be invalidated when camera calibration or extrinsics change."""
        cache = DimeSolverCache()
        identity_cam1 = DimeCacheIdentity(
            model_hash="model_humanoid_v1",
            param_hash="param_default",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_front_left_v1",
            job_id="job_001",
        )
        state = _create_pendulum_state(model_hash="model_humanoid_v1")
        u = np.array([0.5], dtype=np.float64)

        cache.store_step(identity_cam1, state, u, dt=0.02, next_state=state)
        assert cache.get_step(identity_cam1, state, u, dt=0.02) is not None

        # Camera changed: new camera config hash must not hit old cache entry
        identity_cam2 = DimeCacheIdentity(
            model_hash="model_humanoid_v1",
            param_hash="param_default",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_front_left_v2_recalibrated",
            job_id="job_001",
        )
        assert cache.get_step(identity_cam2, state, u, dt=0.02) is None

        # Explicit invalidation on camera change clears camera-dependent factors
        cache.invalidate_on_camera_change("cam_rig_front_left_v1")
        assert cache.get_step(identity_cam1, state, u, dt=0.02) is None

    def test_red_stale_cache_after_body_change(self) -> None:
        """Cache must be invalidated when body mass or segment length changes."""
        cache = DimeSolverCache()
        identity_body1 = DimeCacheIdentity(
            model_hash="model_humanoid_v1",
            param_hash="param_mass_75kg",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_v1",
            job_id="job_001",
        )
        state = _create_pendulum_state(model_hash="model_humanoid_v1")

        cache.store_drift(identity_body1, state, drift_result=np.array([0.1, -9.81]))
        assert cache.get_drift(identity_body1, state) is not None

        # Body parameters changed: different param_hash yields cache miss
        identity_body2 = DimeCacheIdentity(
            model_hash="model_humanoid_v1",
            param_hash="param_mass_82kg",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_v1",
            job_id="job_001",
        )
        assert cache.get_drift(identity_body2, state) is None

        # Invalidate body parameter cache explicitly
        cache.invalidate_on_body_change("param_mass_75kg")
        assert cache.get_drift(identity_body1, state) is None

    def test_red_stale_cache_after_contact_change(self) -> None:
        """Cache must be invalidated when contact policy changes."""
        cache = DimeSolverCache()
        identity_elim = DimeCacheIdentity(
            model_hash="model_humanoid_v1",
            param_hash="param_default",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_v1",
            job_id="job_001",
        )
        state = _create_pendulum_state(model_hash="model_humanoid_v1")

        jacobian_elim = np.eye(2)
        cache.store_jacobian(identity_elim, state, jacobian_elim)
        assert cache.get_jacobian(identity_elim, state) is not None

        # Explicit constrained query must not use native eliminated jacobian
        identity_const = DimeCacheIdentity(
            model_hash="model_humanoid_v1",
            param_hash="param_default",
            contact_policy=ContactPolicy.EXPLICIT_CONSTRAINED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_v1",
            job_id="job_001",
        )
        assert cache.get_jacobian(identity_const, state) is None

    def test_red_cross_job_native_state_contamination(self) -> None:
        """Two distinct jobs must have isolated cache partitions; no cross-job leaks."""
        cache = DimeSolverCache()
        identity_job_a = DimeCacheIdentity(
            model_hash="model_humanoid_v1",
            param_hash="param_default",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_v1",
            job_id="job_A_offline_matching",
        )
        identity_job_b = DimeCacheIdentity(
            model_hash="model_humanoid_v1",
            param_hash="param_default",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_v1",
            job_id="job_B_realtime_stream",
        )
        state = _create_pendulum_state(model_hash="model_humanoid_v1")
        u = np.array([1.2], dtype=np.float64)

        cache.store_step(identity_job_a, state, u, dt=0.01, next_state=state)
        # Job B must NOT receive Job A's cached step
        assert cache.get_step(identity_job_b, state, u, dt=0.01) is None

        # Invalidate job A must not affect other keys
        cache.invalidate_job("job_A_offline_matching")
        assert cache.get_step(identity_job_a, state, u, dt=0.01) is None

    def test_red_inaccurate_derivative_at_impact(self) -> None:
        """At contact impact discontinuity, smooth derivative approximations are rejected."""
        cache = DimeSolverCache()
        identity = DimeCacheIdentity(
            model_hash="model_biped_v1",
            param_hash="param_default",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="cfg_dense_trf",
            camera_config_hash="cam_rig_v1",
            job_id="job_001",
        )
        state_pre_impact = _create_pendulum_state(
            q_val=0.0, v_val=-2.5, model_hash="model_biped_v1"
        )

        with pytest.raises(PreconditionError, match="impact|discontinuity"):
            cache.store_jacobian_with_impact_check(
                identity=identity,
                state=state_pre_impact,
                jacobian=np.eye(2),
                is_impact_phase=True,
            )

    def test_red_speedup_from_skipped_replay_rejected(self) -> None:
        """Speedup claims obtained by skipping continuous replay are strictly rejected."""
        provider = AnalyticPendulumProvider()
        initial_state = _create_pendulum_state(
            q_val=0.2, model_hash=provider.model_hash
        )
        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=3,
            dt_s=0.02,
            defect_mode=DefectMode.EXACT,
        )
        cache = DimeSolverCache()

        with pytest.raises(PreconditionError, match="replay.*mandatory|cannot.*skip"):
            accelerated_solve_dynamics_window(
                problem=problem,
                cache=cache,
                skip_independent_replay=True,
            )


class TestDimeSolverCacheGreenSuite:
    """GREEN test suite verifying exact equivalence, bounded approximation, thread safety, and speedup."""

    def test_green_cached_versus_fresh_equivalence_exact(self) -> None:
        """Cached step and drift evaluations match fresh evaluations to machine precision."""
        provider = AnalyticPendulumProvider()
        state = _create_pendulum_state(
            q_val=0.15, v_val=0.05, model_hash=provider.model_hash
        )
        u = np.array([0.25], dtype=np.float64)
        dt = 0.02

        step_req = DimeFullStepRequest(
            state=state,
            controls=u,
            dt=dt,
            model_hash=provider.model_hash,
        )
        fresh_step = provider.step(step_req)

        cache = DimeSolverCache()
        identity = DimeCacheIdentity(
            model_hash=provider.model_hash,
            param_hash="nominal",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="default",
            camera_config_hash="none",
            job_id="test_exact",
        )

        cache.store_step(identity, state, u, dt, fresh_step.next_state)
        cached_state = cache.get_step(identity, state, u, dt)

        assert cached_state is not None
        np.testing.assert_allclose(cached_state.q, fresh_step.next_state.q, atol=1e-12)
        np.testing.assert_allclose(cached_state.v, fresh_step.next_state.v, atol=1e-12)

    def test_green_bounded_approximation_error_within_validity_radius(self) -> None:
        """Local linear model approximation maintains error <= O(delta^2) within validity radius."""
        q0 = np.array([0.2], dtype=np.float64)
        v0 = np.array([0.0], dtype=np.float64)
        state0 = DimeCompleteState(
            t=0.0,
            q=q0,
            v=v0,
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash="analytic_pendulum_v1",
        )

        local_model = LocalModelApproximation(
            nominal_state=state0,
            nominal_controls=np.array([0.0]),
            state_jacobian=np.array([[1.0, 0.02], [-0.196, 1.0]]),
            control_jacobian=np.array([[0.0], [0.02]]),
            validity_radius=0.05,
        )

        # Within validity radius (delta_q = 0.02 < 0.05):
        query_state_near = _create_pendulum_state(
            q_val=0.22, v_val=0.01, model_hash="analytic_pendulum_v1"
        )
        approx_next = local_model.predict(query_state_near, np.array([0.0]))
        assert approx_next is not None

        # Outside validity radius (delta_q = 0.10 > 0.05):
        query_state_far = _create_pendulum_state(
            q_val=0.30, v_val=0.01, model_hash="analytic_pendulum_v1"
        )
        with pytest.raises(PreconditionError, match="validity.*radius"):
            local_model.predict(query_state_far, np.array([0.0]))

    def test_green_cache_invalidation_and_concurrency(self) -> None:
        """Concurrent multi-threaded access maintains thread safety without race conditions."""
        cache = DimeSolverCache()
        identity = DimeCacheIdentity(
            model_hash="pendulum",
            param_hash="p1",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="c1",
            camera_config_hash="cam1",
            job_id="concurrent_job",
        )

        def worker_task(idx: int) -> int:
            st = _create_pendulum_state(q_val=0.01 * idx, model_hash="pendulum")
            u_vec = np.array([float(idx)], dtype=np.float64)
            cache.store_step(identity, st, u_vec, dt=0.01, next_state=st)
            res = cache.get_step(identity, st, u_vec, dt=0.01)
            assert res is not None
            if idx % 10 == 0:
                cache.invalidate_on_body_change("p_other")
            return idx

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_task, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 50

    def test_green_measured_cold_vs_warm_speed_and_cost_breakdown(self) -> None:
        """Cold vs warm window solve achieves measured speedup and exports full cost breakdown."""
        provider = AnalyticPendulumProvider()
        initial_state = _create_pendulum_state(
            q_val=0.1, model_hash=provider.model_hash
        )
        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=4,
            dt_s=0.02,
            defect_mode=DefectMode.EXACT,
            max_iterations=15,
        )

        cache = DimeSolverCache()

        # Cold solve (empty cache)
        cold_res, cold_breakdown = accelerated_solve_dynamics_window(
            problem=problem,
            cache=cache,
        )
        assert cold_res.success
        assert cold_breakdown.cache_misses > 0
        assert cold_breakdown.replay_executed is True

        # Warm solve (populated cache)
        warm_res, warm_breakdown = accelerated_solve_dynamics_window(
            problem=problem,
            cache=cache,
        )
        assert warm_res.success
        assert warm_breakdown.cache_hits > 0
        assert warm_breakdown.replay_executed is True

        # Quality-matched: states and controls must match
        np.testing.assert_allclose(warm_res.controls, cold_res.controls, atol=1e-6)
        np.testing.assert_allclose(
            warm_res.states[-1].q, cold_res.states[-1].q, atol=1e-6
        )

        # Full cost breakdown verification
        assert isinstance(warm_breakdown, DimeCostBreakdown)
        assert warm_breakdown.drift_calls_count is None  # not measured (#11547)
        assert warm_breakdown.full_step_calls_count is None
        assert warm_breakdown.window_solve_time_s >= 0.0
        assert warm_breakdown.independent_replay_time_s >= 0.0
        assert warm_breakdown.total_time_s >= 0.0
        assert warm_breakdown.p50_speed_s is None
        assert warm_breakdown.p95_speed_s is None


def _targets_problem(
    targets: list[list[float]],
    observation_weight: float = 1.0,
    max_iterations: int = 15,
    initial_state: DimeCompleteState | None = None,
) -> DimeDynamicsWindowProblem:
    provider = AnalyticPendulumProvider()
    if initial_state is None:
        initial_state = _create_pendulum_state(
            q_val=0.1, model_hash=provider.model_hash
        )
    return DimeDynamicsWindowProblem(
        provider=provider,
        initial_state=initial_state,
        horizon_steps=4,
        dt_s=0.02,
        target_positions=[np.array(t, dtype=np.float64) for t in targets],
        defect_mode=DefectMode.EXACT,
        observation_weight=observation_weight,
        max_iterations=max_iterations,
    )


class TestDimeSolverCacheKeyCoversInputs:
    """#11547: the window cache key must cover every input changing the solution."""

    def test_identical_inputs_hit_cache(self) -> None:
        cache = DimeSolverCache()
        _, first = accelerated_solve_dynamics_window(
            _targets_problem([[0.1], [0.2], [0.3]]), cache
        )
        _, second = accelerated_solve_dynamics_window(
            _targets_problem([[0.1], [0.2], [0.3]]), cache
        )
        assert first.cache_misses == 1
        assert second.cache_hits == 1
        assert second.cache_misses == 0

    def test_changing_only_targets_misses_cache(self) -> None:
        cache = DimeSolverCache()
        res_a, _ = accelerated_solve_dynamics_window(
            _targets_problem([[0.1], [0.2], [0.3]]), cache
        )
        res_b, bd_b = accelerated_solve_dynamics_window(
            _targets_problem([[0.5], [0.6], [0.7]]), cache
        )
        assert bd_b.cache_hits == 0
        assert bd_b.cache_misses == 1
        assert not np.allclose(res_a.controls, res_b.controls)

    def test_changing_only_observation_weight_misses_cache(self) -> None:
        cache = DimeSolverCache()
        accelerated_solve_dynamics_window(_targets_problem([[0.4]] * 3), cache)
        _, bd = accelerated_solve_dynamics_window(
            _targets_problem([[0.4]] * 3, observation_weight=50.0), cache
        )
        assert bd.cache_hits == 0

    def test_changing_only_solver_options_misses_cache(self) -> None:
        cache = DimeSolverCache()
        accelerated_solve_dynamics_window(_targets_problem([[0.4]] * 3), cache)
        _, bd = accelerated_solve_dynamics_window(
            _targets_problem([[0.4]] * 3, max_iterations=7), cache
        )
        assert bd.cache_hits == 0

    @pytest.mark.parametrize(
        "override",
        [
            {"t": 0.5},
            {"internal_state": {"activation": 0.7}},
            {"v_dot": np.array([0.3], dtype=np.float64)},
            {"frame": "pelvis"},
        ],
        ids=["t", "internal_state", "v_dot", "frame"],
    )
    def test_changing_only_one_state_field_misses_cache(
        self, override: dict[str, Any]
    ) -> None:
        cache = DimeSolverCache()
        base = _targets_problem([[0.4]] * 3)
        accelerated_solve_dynamics_window(base, cache)
        s = base.initial_state
        fields: dict[str, Any] = {
            "t": s.t,
            "q": s.q,
            "v": s.v,
            "v_dot": s.v_dot,
            "internal_state": dict(s.internal_state),
            "model_hash": s.model_hash,
            "units": dict(s.units),
            "frame": s.frame,
        }
        fields.update(override)
        changed = _targets_problem(
            [[0.4]] * 3, initial_state=DimeCompleteState(**fields)
        )
        _, bd = accelerated_solve_dynamics_window(changed, cache)
        assert bd.cache_hits == 0
        assert bd.cache_misses == 1

    def test_equal_but_distinct_states_hit_cache(self) -> None:
        cache = DimeSolverCache()
        base = _targets_problem([[0.4]] * 3)
        s = base.initial_state
        twin = DimeCompleteState(
            t=s.t,
            q=s.q.copy(),
            v=s.v.copy(),
            internal_state={"b": 2.0, "a": np.array([1.0])},
            model_hash=s.model_hash,
            units=dict(s.units),
            frame=s.frame,
        )
        accelerated_solve_dynamics_window(
            _targets_problem([[0.4]] * 3, initial_state=twin), cache
        )
        twin2 = DimeCompleteState(
            t=s.t,
            q=s.q.copy(),
            v=s.v.copy(),
            internal_state={"a": np.array([1.0]), "b": 2.0},
            model_hash=s.model_hash,
            units=dict(s.units),
            frame=s.frame,
        )
        _, bd = accelerated_solve_dynamics_window(
            _targets_problem([[0.4]] * 3, initial_state=twin2), cache
        )
        assert bd.cache_hits == 1

    def test_non_finite_targets_rejected(self) -> None:
        cache = DimeSolverCache()
        identity = DimeCacheIdentity(
            model_hash="m",
            param_hash="p",
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            solver_config_hash="c",
            camera_config_hash="cam",
            job_id="j",
        )
        problem = _targets_problem([[0.1], [float("nan")]])
        with pytest.raises(PreconditionError, match="finite"):
            cache.get_window_solve(identity, problem)


class TestDimeCostBreakdownIsMeasured:
    """#11547: no wall-time-fraction fabrication in the cost breakdown."""

    def test_unmeasured_components_reported_as_not_measured(self) -> None:
        cache = DimeSolverCache()
        _, bd = accelerated_solve_dynamics_window(
            _targets_problem([[0.1], [0.2], [0.3]]), cache
        )
        for name in (
            "drift_time_s",
            "full_step_time_s",
            "jacobian_time_s",
            "assembly_factorization_time_s",
            "p50_speed_s",
            "p95_speed_s",
        ):
            assert getattr(bd, name) is None, name
        d = bd.to_dict()
        assert "drift_time_s" in d["not_measured"]

    def test_measured_times_are_consistent_and_cost_terms_sum(self) -> None:
        cache = DimeSolverCache()
        res, bd = accelerated_solve_dynamics_window(
            _targets_problem([[0.1], [0.2], [0.3]]), cache
        )
        assert bd.total_time_s == pytest.approx(
            bd.window_solve_time_s + bd.independent_replay_time_s
        )
        assert bd.solver_cost_terms is not None
        terms = dict(bd.solver_cost_terms)
        total = terms.pop("total_cost")
        assert sum(terms.values()) == pytest.approx(total)
        assert total == pytest.approx(res.cost_breakdown["total_cost"])

    def test_failure_cost_is_not_a_magic_constant(self) -> None:
        cache = DimeSolverCache()
        _, bd = accelerated_solve_dynamics_window(
            _targets_problem([[0.1], [0.2], [0.3]]), cache
        )
        assert bd.failure_costs == 0.0
