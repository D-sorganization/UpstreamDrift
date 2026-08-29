"""Deterministic bounded multiple-shooting tests for issue #9124."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy import bounded_event_multiple_shooting
from scripts.research.proximal_distal_energy.bounded_event_multiple_shooting import (
    MultipleShootingConfig,
    MultipleShootingStatus,
    _seeded_segment_controls,
    solve_bounded_event_multiple_shooting,
)
from scripts.research.proximal_distal_energy.bounded_event_reachability import (
    BoundedEventReachabilityProblem,
    ControlPerturbationBounds,
    EventReplayStatus,
    FeasibilityStatus,
    replay_guard_event,
)
from scripts.research.proximal_distal_energy.phase_event_stability import StateScales
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    ControlScales,
    GuardCrossingConfig,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def nominal_problem() -> BoundedEventReachabilityProblem:
    params = GolfModelParams.default()
    initial_state = np.array([-2.2, -1.57, 0.0, 0.0])
    dt_s = 2e-3
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(
        round(0.40 / dt_s), dt_s
    )
    guard = GuardCrossingConfig(
        guard_gradient=(1.0, 1.0, 0.0, 0.0),
        guard_tolerance=1e-10,
        time_tolerance_s=1e-12,
        transversality_threshold=1e-8,
    )
    replay = replay_guard_event(
        params=params,
        initial_state=initial_state,
        controls=controls,
        dt_s=dt_s,
        guard=guard,
    )
    assert replay.status is EventReplayStatus.TRANSVERSE
    assert replay.state is not None
    return BoundedEventReachabilityProblem(
        params=params,
        initial_state=tuple(initial_state),
        nominal_controls=controls,
        dt_s=dt_s,
        state_scales=StateScales((np.pi, np.pi, 10.0, 10.0)),
        control_scales=ControlScales((100.0, 100.0)),
        bounds=ControlPerturbationBounds(
            lower_nm=(-5.0, -5.0),
            upper_nm=(5.0, 5.0),
            max_rate_nm_per_s=(2500.0, 2500.0),
        ),
        guard=guard,
        target_event_state=tuple(replay.state),
        tangent_tolerance=2e-7,
    )


def test_solve_builds_exactly_one_backend_and_reuses_it(
    nominal_problem: BoundedEventReachabilityProblem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for #9198: no backend construction in the SLSQP loop.

    ``_integrate_segment`` used to call ``make_backend`` itself, so a fresh
    ``ODEBackend`` and ``DoublePendulum`` were built once per shooting residual
    -- and SciPy evaluates that residual once per finite-difference column per
    SLSQP iteration. ``registered_step`` resets the backend before every step,
    so one backend per solve is numerically identical and orders of magnitude
    cheaper under ``coverage``, whose collector lock is taken on every
    construction.
    """

    constructed: list[str] = []
    real_make_backend = bounded_event_multiple_shooting.make_backend

    def counting_make_backend(name: str, params: object, **kwargs: object) -> object:
        constructed.append(name)
        return real_make_backend(name, params, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        bounded_event_multiple_shooting, "make_backend", counting_make_backend
    )
    config = MultipleShootingConfig(segment_count=3, max_iterations=20, seed=23)

    result = solve_bounded_event_multiple_shooting(nominal_problem, config)

    assert constructed == ["ode"]
    assert result.iterations > 0


def test_segment_memo_returns_bit_identical_states(
    nominal_problem: BoundedEventReachabilityProblem,
) -> None:
    """The #9198 memo must be exact, not approximate.

    A cache hit has to return the same float64 bit pattern the RK4 loop would
    have produced, or the registered evidence would silently drift.
    """

    layout = bounded_event_multiple_shooting._crossing_layout(nominal_problem, 4)
    durations = bounded_event_multiple_shooting._step_durations(
        layout,
        dt_s=nominal_problem.dt_s,
        partial_dt_s=layout.nominal_partial_dt_s,
    )
    backend = bounded_event_multiple_shooting._solve_backend(nominal_problem)
    start = np.asarray(nominal_problem.initial_state, dtype=float)
    perturbation = np.array([0.75, -0.25])
    memo: dict[object, object] = {}

    for indices in layout.segment_indices:
        uncached = bounded_event_multiple_shooting._integrate_segment(
            nominal_problem,
            backend=backend,
            start_state=start,
            perturbation=perturbation,
            indices=indices,
            durations=durations,
        )
        first = bounded_event_multiple_shooting._integrate_segment(
            nominal_problem,
            backend=backend,
            start_state=start,
            perturbation=perturbation,
            indices=indices,
            durations=durations,
            memo=memo,  # type: ignore[arg-type]
        )
        hit = bounded_event_multiple_shooting._integrate_segment(
            nominal_problem,
            backend=backend,
            start_state=start,
            perturbation=perturbation,
            indices=indices,
            durations=durations,
            memo=memo,  # type: ignore[arg-type]
        )

        assert uncached.tobytes() == first.tobytes()
        assert uncached.tobytes() == hit.tobytes()
        assert not hit.flags.writeable

    assert len(memo) == len(layout.segment_indices)


def test_multiple_shooting_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="segment_count"):
        MultipleShootingConfig(segment_count=0)
    with pytest.raises(ValueError, match="constraint_tolerance"):
        MultipleShootingConfig(segment_count=2, constraint_tolerance=0.0)
    with pytest.raises(ValueError, match="seed"):
        MultipleShootingConfig(segment_count=2, seed=-1)
    with pytest.raises(ValueError, match="initial_control_fraction"):
        MultipleShootingConfig(segment_count=2, initial_control_fraction=1.1)


def test_seeded_initial_controls_are_deterministic_bounded_and_rate_limited(
    nominal_problem: BoundedEventReachabilityProblem,
) -> None:
    config = MultipleShootingConfig(
        segment_count=6,
        seed=47,
        initial_control_fraction=0.35,
    )

    first = _seeded_segment_controls(nominal_problem, config)
    second = _seeded_segment_controls(nominal_problem, config)
    other = _seeded_segment_controls(
        nominal_problem, replace(config, seed=config.seed + 1)
    )
    rates = np.diff(np.vstack((np.zeros((1, 2)), first)), axis=0) / (
        nominal_problem.dt_s
    )

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all(first >= 0.35 * nominal_problem.bounds.lower_array)
    assert np.all(first <= 0.35 * nominal_problem.bounds.upper_array)
    assert np.all(
        np.abs(rates) <= 0.35 * nominal_problem.bounds.rate_array[np.newaxis, :] + 1e-12
    )
    np.testing.assert_array_equal(
        _seeded_segment_controls(
            nominal_problem,
            replace(config, initial_control_fraction=0.0),
        ),
        np.zeros_like(first),
    )


def test_nominal_target_converges_and_passes_independent_replay(
    nominal_problem: BoundedEventReachabilityProblem,
) -> None:
    config = MultipleShootingConfig(
        segment_count=4,
        max_iterations=100,
        constraint_tolerance=1e-8,
        objective_tolerance=1e-12,
        seed=17,
    )

    result = solve_bounded_event_multiple_shooting(nominal_problem, config)

    assert result.status is MultipleShootingStatus.CONVERGED
    assert result.solver_success
    assert result.replay is not None
    assert result.replay.feasibility_status is FeasibilityStatus.FEASIBLE
    assert result.replay.event is not None
    assert result.replay.event.status is EventReplayStatus.TRANSVERSE
    assert result.maximum_continuity_residual <= config.constraint_tolerance
    assert result.maximum_target_residual <= config.constraint_tolerance
    assert result.event_time_s == pytest.approx(result.replay.event.time_s, abs=2e-10)
    np.testing.assert_allclose(result.segment_perturbations, 0.0, atol=1e-9)
    np.testing.assert_allclose(result.perturbations, 0.0, atol=1e-9)


def test_nominal_multiple_shooting_solution_is_deterministic(
    nominal_problem: BoundedEventReachabilityProblem,
) -> None:
    config = MultipleShootingConfig(segment_count=3, seed=23)

    first = solve_bounded_event_multiple_shooting(nominal_problem, config)
    second = solve_bounded_event_multiple_shooting(nominal_problem, config)

    assert first.status is second.status
    assert first.iterations == second.iterations
    assert first.objective == pytest.approx(second.objective, abs=0.0)
    assert first.event_time_s == pytest.approx(second.event_time_s, abs=0.0)
    np.testing.assert_array_equal(
        first.segment_perturbations, second.segment_perturbations
    )
    np.testing.assert_array_equal(first.state_nodes, second.state_nodes)
    np.testing.assert_array_equal(first.perturbations, second.perturbations)


@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_small_reversed_tangent_targets_converge_and_replay(
    nominal_problem: BoundedEventReachabilityProblem,
    direction: float,
) -> None:
    target = np.asarray(nominal_problem.target_event_state, dtype=float)
    target += direction * np.array([1e-3, -1e-3, 0.0, 0.0])
    problem = replace(
        nominal_problem,
        bounds=ControlPerturbationBounds(
            lower_nm=(-20.0, -20.0),
            upper_nm=(20.0, 20.0),
            max_rate_nm_per_s=(10000.0, 10000.0),
        ),
        target_event_state=tuple(target),
        tangent_tolerance=2e-6,
    )
    config = MultipleShootingConfig(
        segment_count=4,
        max_iterations=300,
        constraint_tolerance=2e-6,
        objective_tolerance=1e-12,
        seed=31,
    )

    result = solve_bounded_event_multiple_shooting(problem, config)

    assert result.status is MultipleShootingStatus.CONVERGED
    assert result.solver_success
    assert result.replay is not None
    assert result.replay.feasibility_status is FeasibilityStatus.FEASIBLE
    assert result.replay.event_tangent_residual is not None
    assert result.replay.event_tangent_residual <= problem.tangent_tolerance
    assert np.max(np.abs(result.segment_perturbations)) > 1e-4
    assert result.objective > 0.0


def test_zero_authority_unreachable_target_is_typed_infeasible_without_solver_failure(
    nominal_problem: BoundedEventReachabilityProblem,
) -> None:
    target = np.asarray(nominal_problem.target_event_state, dtype=float)
    target += np.array([0.05, -0.05, 0.0, 0.0])
    problem = replace(
        nominal_problem,
        bounds=ControlPerturbationBounds.zero(),
        target_event_state=tuple(target),
    )

    result = solve_bounded_event_multiple_shooting(
        problem, MultipleShootingConfig(segment_count=4)
    )

    assert result.status is MultipleShootingStatus.INFEASIBLE
    assert not result.solver_success
    assert result.replay is not None
    assert result.replay.feasibility_status is FeasibilityStatus.INFEASIBLE
    assert "zero incremental authority" in result.message


def test_segment_count_cannot_exceed_pre_event_step_count(
    nominal_problem: BoundedEventReachabilityProblem,
) -> None:
    with pytest.raises(ValueError, match="pre-event step count"):
        solve_bounded_event_multiple_shooting(
            nominal_problem,
            MultipleShootingConfig(
                segment_count=nominal_problem.nominal_controls.shape[0]
            ),
        )
