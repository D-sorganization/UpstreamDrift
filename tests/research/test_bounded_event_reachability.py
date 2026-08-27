"""Bounded nonlinear event-reaching contracts for issue #9124."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.bounded_event_reachability import (
    AuthorityStatus,
    BoundedEventReachabilityProblem,
    ConstraintStatus,
    ControlPerturbationBounds,
    EventReplayStatus,
    FeasibilityStatus,
    evaluate_bounded_candidate,
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
def nominal_case() -> tuple[
    GolfModelParams,
    np.ndarray,
    np.ndarray,
    float,
    GuardCrossingConfig,
    np.ndarray,
]:
    """Registered nominal case with one transverse delivery-guard crossing."""

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
    event = replay_guard_event(
        params=params,
        initial_state=initial_state,
        controls=controls,
        dt_s=dt_s,
        guard=guard,
    )
    assert event.status is EventReplayStatus.TRANSVERSE
    assert event.state is not None
    return params, initial_state, controls, dt_s, guard, event.state


def _problem(
    nominal_case: tuple[
        GolfModelParams,
        np.ndarray,
        np.ndarray,
        float,
        GuardCrossingConfig,
        np.ndarray,
    ],
    *,
    bounds: ControlPerturbationBounds,
    tangent_tolerance: float = 1e-8,
) -> BoundedEventReachabilityProblem:
    params, initial_state, controls, dt_s, guard, target = nominal_case
    return BoundedEventReachabilityProblem(
        params=params,
        initial_state=tuple(initial_state),
        nominal_controls=controls,
        dt_s=dt_s,
        state_scales=StateScales((np.pi, np.pi, 10.0, 10.0)),
        control_scales=ControlScales((100.0, 100.0)),
        bounds=bounds,
        guard=guard,
        target_event_state=tuple(target),
        tangent_tolerance=tangent_tolerance,
    )


def test_zero_bounds_replay_nominal_without_fabricating_solver_failure(
    nominal_case: tuple[
        GolfModelParams,
        np.ndarray,
        np.ndarray,
        float,
        GuardCrossingConfig,
        np.ndarray,
    ],
) -> None:
    problem = _problem(
        nominal_case,
        bounds=ControlPerturbationBounds.zero(),
    )
    perturbation = np.zeros_like(problem.nominal_controls)

    outcome = evaluate_bounded_candidate(problem, perturbation)

    assert outcome.feasibility_status is FeasibilityStatus.FEASIBLE
    assert outcome.authority_status is AuthorityStatus.ZERO_INCREMENTAL_AUTHORITY
    assert outcome.constraint_status is ConstraintStatus.BOUND_SATURATED
    assert outcome.event is not None
    assert outcome.event.status is EventReplayStatus.TRANSVERSE
    assert outcome.event_tangent_residual is not None
    assert outcome.event_tangent_residual <= problem.tangent_tolerance
    assert outcome.scaled_control_energy == pytest.approx(0.0)
    assert outcome.peak_scaled_control == pytest.approx(0.0)


def test_zero_bounds_type_unreachable_tangent_target_as_infeasible(
    nominal_case: tuple[
        GolfModelParams,
        np.ndarray,
        np.ndarray,
        float,
        GuardCrossingConfig,
        np.ndarray,
    ],
) -> None:
    problem = _problem(
        nominal_case,
        bounds=ControlPerturbationBounds.zero(),
    )
    target = np.asarray(problem.target_event_state, dtype=float)
    target += np.array([0.10, -0.10, 0.0, 0.0])
    problem = replace(problem, target_event_state=tuple(target))

    outcome = evaluate_bounded_candidate(
        problem, np.zeros_like(problem.nominal_controls)
    )

    assert outcome.feasibility_status is FeasibilityStatus.INFEASIBLE
    assert outcome.authority_status is AuthorityStatus.ZERO_INCREMENTAL_AUTHORITY
    assert outcome.event is not None
    assert outcome.event_tangent_residual is not None
    assert outcome.event_tangent_residual > problem.tangent_tolerance


def test_bound_violation_is_typed_before_dynamics_replay(
    nominal_case: tuple[
        GolfModelParams,
        np.ndarray,
        np.ndarray,
        float,
        GuardCrossingConfig,
        np.ndarray,
    ],
) -> None:
    problem = _problem(
        nominal_case,
        bounds=ControlPerturbationBounds(
            lower_nm=(-1.0, -1.0),
            upper_nm=(1.0, 1.0),
            max_rate_nm_per_s=(500.0, 500.0),
        ),
    )
    perturbation = np.zeros_like(problem.nominal_controls)
    perturbation[0, 0] = 2.0

    outcome = evaluate_bounded_candidate(problem, perturbation)

    assert outcome.feasibility_status is FeasibilityStatus.INFEASIBLE
    assert outcome.constraint_status is ConstraintStatus.BOUND_VIOLATION
    assert outcome.event is None
    assert outcome.maximum_amplitude_violation_nm == pytest.approx(1.0)
    assert outcome.maximum_rate_violation_nm_per_s > 0.0


def test_exactly_active_amplitude_and_rate_bounds_are_typed_saturated(
    nominal_case: tuple[
        GolfModelParams,
        np.ndarray,
        np.ndarray,
        float,
        GuardCrossingConfig,
        np.ndarray,
    ],
) -> None:
    problem = _problem(
        nominal_case,
        bounds=ControlPerturbationBounds(
            lower_nm=(-2.0, -2.0),
            upper_nm=(2.0, 2.0),
            max_rate_nm_per_s=(1000.0, 1000.0),
        ),
        tangent_tolerance=100.0,
    )
    perturbation = np.full_like(problem.nominal_controls, 2.0)

    outcome = evaluate_bounded_candidate(problem, perturbation)

    assert outcome.constraint_status is ConstraintStatus.BOUND_SATURATED
    assert outcome.amplitude_bound_active
    assert outcome.rate_bound_active
    assert outcome.maximum_amplitude_violation_nm == pytest.approx(0.0)
    assert outcome.maximum_rate_violation_nm_per_s == pytest.approx(0.0)
    assert outcome.scaled_control_energy > 0.0
    assert outcome.peak_scaled_control == pytest.approx(0.02)
    assert outcome.event is not None


def test_absent_crossing_is_not_misreported_as_target_infeasibility(
    nominal_case: tuple[
        GolfModelParams,
        np.ndarray,
        np.ndarray,
        float,
        GuardCrossingConfig,
        np.ndarray,
    ],
) -> None:
    problem = _problem(
        nominal_case,
        bounds=ControlPerturbationBounds.zero(),
    )
    problem = replace(
        problem,
        nominal_controls=problem.nominal_controls[:50],
    )

    outcome = evaluate_bounded_candidate(
        problem, np.zeros_like(problem.nominal_controls)
    )

    assert outcome.feasibility_status is FeasibilityStatus.WRONG_CROSSING
    assert outcome.event is not None
    assert outcome.event.status is EventReplayStatus.ABSENT
    assert outcome.event_tangent_residual is None


def test_problem_and_bounds_fail_closed_on_ambiguous_contracts(
    nominal_case: tuple[
        GolfModelParams,
        np.ndarray,
        np.ndarray,
        float,
        GuardCrossingConfig,
        np.ndarray,
    ],
) -> None:
    with pytest.raises(ValueError, match="contain zero"):
        ControlPerturbationBounds(
            lower_nm=(1.0, -1.0),
            upper_nm=(2.0, 1.0),
            max_rate_nm_per_s=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="rate"):
        ControlPerturbationBounds(
            lower_nm=(-1.0, -1.0),
            upper_nm=(1.0, 1.0),
            max_rate_nm_per_s=(-1.0, 1.0),
        )

    problem = _problem(
        nominal_case,
        bounds=ControlPerturbationBounds.zero(),
    )
    invalid_target = np.asarray(problem.target_event_state, dtype=float).copy()
    invalid_target[0] += 0.1
    with pytest.raises(ValueError, match="target_event_state must lie on the guard"):
        replace(problem, target_event_state=tuple(invalid_target))


def test_candidate_evaluation_does_not_mutate_caller_arrays(
    nominal_case: tuple[
        GolfModelParams,
        np.ndarray,
        np.ndarray,
        float,
        GuardCrossingConfig,
        np.ndarray,
    ],
) -> None:
    problem = _problem(
        nominal_case,
        bounds=ControlPerturbationBounds(
            lower_nm=(-1.0, -1.0),
            upper_nm=(1.0, 1.0),
            max_rate_nm_per_s=(1000.0, 1000.0),
        ),
        tangent_tolerance=100.0,
    )
    nominal_before = problem.nominal_controls.copy()
    perturbation = np.zeros_like(problem.nominal_controls)
    perturbation_before = perturbation.copy()

    evaluate_bounded_candidate(problem, perturbation)

    np.testing.assert_array_equal(problem.nominal_controls, nominal_before)
    np.testing.assert_array_equal(perturbation, perturbation_before)
    assert not problem.nominal_controls.flags.writeable
