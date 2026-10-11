"""Shared bounded search over native commands, ending in guarded plan replay.

This selects a future plan without applying a live command. Caller-declared
state margins/admission must encode the actual model's scientific hard gates;
the adapter cannot infer contact, physiological or capture validity from them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.engines.myosuite_project_feedback import ProjectTaskObservation, _immutable
from src.engines.myosuite_project_native_search import (
    GuardedCommandPlan,
    ProjectTaskNativeSearch,
)
from src.engines.myosuite_project_task_producer import ProjectTaskCommandHistory
from src.engines.myosuite_project_tracking import ProjectTaskTrackingObjective
from src.shared.python.motion_matching.bounded_candidate_search import (
    CandidateSearchPolicy,
    CandidateSearchProblem,
    CandidateSearchReceipt,
    Solver,
    search_bounded_candidates,
)

Array: TypeAlias = NDArray[np.float64]
StateMargins: TypeAlias = Callable[[Array, Array], Array]


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class NativeCommandSolveReceipt:
    """Existing guarded artifacts, numerical status and whole solve elapsed time."""

    status: str
    selected_plan: GuardedCommandPlan
    fallback_plan: GuardedCommandPlan
    numerical: CandidateSearchReceipt
    elapsed_s: float
    adapter_source_sha256: str
    objective_parameters_sha256: str
    criteria_provenance_sha256: str
    solve_parameters_sha256: str


@dataclass(frozen=True)
class NativeCommandSolveProblem:
    """Frozen command limits and explicitly declared model-specific hard gates.

    The criteria digest identifies a caller declaration of source/parameters;
    it is not authentication of hidden callback dependencies or physiology.
    """

    initial_plan: Array
    fallback_plan: Array
    lower: Array
    upper: Array
    max_command_increment: Array
    state_margins: StateMargins
    guarded_admission: Callable[[ProjectTaskCommandHistory], bool]
    criteria_provenance_sha256: str
    _parameters: str = field(init=False, repr=False)
    _functions: tuple[StateMargins, Callable[[ProjectTaskCommandHistory], bool]] = (
        field(init=False, repr=False)
    )

    def __post_init__(self) -> None:
        if not callable(self.state_margins) or not callable(self.guarded_admission):
            raise TypeError(
                "complete-state margins and guarded admission are mandatory"
            )
        digest = self.criteria_provenance_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("hard criteria provenance requires lowercase SHA-256")
        for name in (
            "initial_plan",
            "fallback_plan",
            "lower",
            "upper",
            "max_command_increment",
        ):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        object.__setattr__(
            self, "_functions", (self.state_margins, self.guarded_admission)
        )
        object.__setattr__(self, "_parameters", self._digest())

    def _digest(self) -> str:
        digest = hashlib.sha256(b"native-command-solve-problem-v1\n")
        digest.update(self.criteria_provenance_sha256.encode("ascii"))
        for name in (
            "initial_plan",
            "fallback_plan",
            "lower",
            "upper",
            "max_command_increment",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            digest.update(
                name.encode("ascii")
                + str(value.shape).encode("ascii")
                + value.tobytes()
            )
        return digest.hexdigest()

    def verify(self) -> None:
        if (
            self._digest() != self._parameters
            or self.state_margins is not self._functions[0]
            or self.guarded_admission is not self._functions[1]
        ):
            raise ValueError("native command solve parameters changed")

    @property
    def parameters_sha256(self) -> str:
        self.verify()
        return self._parameters


class _HorizonMargins:
    """Apply identical full-horizon margins to provisional and guarded states."""

    def __init__(
        self,
        observation: ProjectTaskObservation,
        lower: Array,
        upper: Array,
        increment: Array,
        state_margins: StateMargins,
    ) -> None:
        previous = np.asarray(observation.control, dtype=np.float64)
        delta = np.asarray(increment, dtype=np.float64)
        if (
            delta.shape != previous.shape
            or not np.isfinite(delta).all()
            or np.any(delta <= 0)
        ):
            raise ValueError(
                "command increment requires finite positive native-channel limits"
            )
        self.previous, self.delta = _immutable(previous), _immutable(delta)
        self.lower, self.upper = _immutable(lower), _immutable(upper)
        self.state_margins = state_margins

    def __call__(self, states: Array, commands: Array) -> Array:
        prior = np.vstack((self.previous, commands[:-1]))
        supplied = np.asarray(self.state_margins(states, commands), dtype=np.float64)
        if supplied.ndim != 1 or supplied.size == 0 or not np.isfinite(supplied).all():
            raise ValueError("complete-state margins must be a nonempty finite vector")
        return np.concatenate(
            (
                (commands - self.lower).ravel(),
                (self.upper - commands).ravel(),
                (self.delta - np.abs(commands - prior)).ravel(),
                supplied,
            )
        )


def _choose_guarded_plan(
    promote: Callable[[Array], GuardedCommandPlan],
    commands: Array,
    fallback: GuardedCommandPlan,
    policy: CandidateSearchPolicy,
    started: float,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[str, GuardedCommandPlan]:
    try:
        candidate = promote(commands)
    except ValueError as exc:
        # Only explicit hard-criterion rejection may select an earlier fallback;
        # identity, clock and execution errors must propagate.
        if str(exc) != "guarded command plan failed declared hard criteria":
            raise
        return "fallback_guarded_rejection", fallback
    if cancel_requested is not None and cancel_requested():
        return "fallback_cancelled", fallback
    if time.perf_counter() - started > policy.max_wall_s:
        return "fallback_timeout", fallback
    if candidate.objective >= fallback.objective - 1e-9:
        return "fallback_guarded_no_benefit", fallback
    return "optimized", candidate


def _checked_margins(
    search: ProjectTaskNativeSearch,
    observation: ProjectTaskObservation,
    objective: ProjectTaskTrackingObjective,
    problem: NativeCommandSolveProblem,
    policy: CandidateSearchPolicy,
) -> _HorizonMargins:
    if not isinstance(problem, NativeCommandSolveProblem) or not isinstance(
        policy, CandidateSearchPolicy
    ):
        raise TypeError(
            "native solve requires frozen problem and bounded search policy"
        )
    if not isinstance(search, ProjectTaskNativeSearch) or not isinstance(
        objective, ProjectTaskTrackingObjective
    ):
        raise TypeError(
            "native solve requires owned native search and tracking objective"
        )
    if not isinstance(observation, ProjectTaskObservation):
        raise TypeError("native solve requires a complete native observation")
    problem.verify()
    return _HorizonMargins(
        observation,
        problem.lower,
        problem.upper,
        problem.max_command_increment,
        problem.state_margins,
    )


def solve_native_command_plan(
    search: ProjectTaskNativeSearch,
    observation: ProjectTaskObservation,
    objective: ProjectTaskTrackingObjective,
    problem: NativeCommandSolveProblem,
    policy: CandidateSearchPolicy,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    solver: Solver = minimize,
) -> NativeCommandSolveReceipt:
    """Admit fallback, search natively, then recompute and independently replay.

    All inputs are ordered post-mapping controls. Increments are per native
    sample, not rates or coefficient limits. Caller supplies complete-state
    signed margins (nonnegative feasible) and a separately guarded admission.
    Scientific tolerances are never inferred from finite arrays. Numerical
    feasibility tolerance does not relax exact guarded command/slew margins.

    Budget is cooperative and includes fallback/promotion overhead in measured
    elapsed time. Even cancellation requires a replayable admitted fallback.
    A future prefix still needs current-state admission before actual execution.
    """
    started = time.perf_counter()
    source = _source_sha256()
    margins = _checked_margins(search, observation, objective, problem, policy)

    def evaluate(commands: Array) -> tuple[float, Array]:
        prediction = search.predict(observation, commands)
        value = objective.prediction_cost(prediction).total
        return value, margins(
            prediction.integration_states, prediction.applied_actuator_commands
        )

    numerical_problem = CandidateSearchProblem(
        problem.initial_plan,
        problem.fallback_plan,
        problem.lower,
        problem.upper,
        evaluate,
    )

    def admit(history: ProjectTaskCommandHistory) -> bool:
        passed = problem.guarded_admission(history)
        if type(passed) is not bool:
            raise TypeError("guarded admission requires an explicit Boolean")
        return passed and bool(
            np.min(
                margins(history.integration_states, history.applied_actuator_commands)
            )
            >= 0
        )

    def cost(history: ProjectTaskCommandHistory) -> float:
        return objective.history_cost(history).total

    fallback = search.promote(
        observation, numerical_problem.fallback_plan, objective=cost, admit=admit
    )
    numerical = search_bounded_candidates(
        numerical_problem,
        policy,
        started=started,
        cancel_requested=cancel_requested,
        solver=solver,
    )
    status, selected = numerical.status, fallback
    if numerical.candidate_plan is not None:
        status, selected = _choose_guarded_plan(
            lambda commands: search.promote(
                observation, commands, objective=cost, admit=admit
            ),
            numerical.candidate_plan,
            fallback,
            policy,
            started,
            cancel_requested,
        )
    # Revalidate the frozen anchor/objective after the solver and every callback.
    # Never return an earlier admitted artifact against a changed live state.
    cost(fallback.history)
    if _source_sha256() != source:
        raise ValueError("native command solve source changed")
    return NativeCommandSolveReceipt(
        status,
        selected,
        fallback,
        numerical,
        time.perf_counter() - started,
        source,
        objective.parameters_sha256,
        problem.criteria_provenance_sha256,
        problem.parameters_sha256,
    )
