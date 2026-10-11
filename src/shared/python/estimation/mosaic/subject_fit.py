"""Subject-level MOSAIC pipeline: initialise, fit, qualify, summarise.

Stages (all timings measured, never estimated):

1. frame-sequential IK initialisation of every trial;
2. joint continuation fit over kinematics, geometry, inertia and inputs
   (:func:`fit_trials`);
3. structural observability report of the inertial parameters;
4. per-trial *open-loop forward replay* from the fitted initial state with the
   fitted inputs, plus the TVLQR local policy and its closed-loop replay;
5. cross-trial torque template and principal modes.

Replay acceptance is the physical gate: a fit whose open-loop replay leaves the
tolerance is reported as not accepted, however small its residuals.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.estimation.mosaic.identifiability_subspace import (
    ObservabilityReport,
    structural_observability,
)
from src.shared.python.estimation.mosaic.ik_init import initialize_trajectory
from src.shared.python.estimation.mosaic.inertial import InertialParameterization
from src.shared.python.estimation.mosaic.local_policy import (
    ReplayReport,
    StepFunction,
    linearize_along_trajectory,
    replay_acceptance,
    tvlqr_gains,
)
from src.shared.python.estimation.mosaic.outer_solve import (
    ModelFactory,
    OuterFit,
    OuterOptions,
    OuterPriors,
    TrialData,
    fit_trials,
)
from src.shared.python.estimation.mosaic.torque_template import (
    PrincipalModes,
    extract_template,
    phase_bins,
    principal_modes,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
StepFactory = Callable[[FloatArray, FloatArray, float], StepFunction]
_LINEARIZATION_EPS = 1e-6


@dataclass(frozen=True)
class ReplayConfig:
    """How to build and judge the forward replay of each fitted trial."""

    step_factory: StepFactory
    state_weight: FloatArray
    input_weight: FloatArray
    position_tolerance: float
    initial_perturbation: FloatArray

    def __post_init__(self) -> None:
        require(self.position_tolerance > 0.0, "position_tolerance must be positive")


@dataclass(frozen=True)
class SubjectFitReport:
    """Everything a downstream consumer needs, with qualification flags."""

    fit: OuterFit
    observability: ObservabilityReport
    replays: tuple[ReplayReport, ...]
    gains: tuple[FloatArray, ...]
    template: FloatArray | None
    modes: PrincipalModes | None
    timings_s: dict[str, float]

    @property
    def open_loop_accepted(self) -> bool:
        return bool(self.replays) and all(r.open_loop_accepted for r in self.replays)


def _initialize(
    factory: ModelFactory,
    trials: Sequence[TrialData],
    first_frame_q: Sequence[FloatArray],
    geometry: FloatArray,
) -> list[FloatArray]:
    model = factory(geometry)
    coefficients = []
    for trial, seed in zip(trials, first_frame_q, strict=True):
        q_ik = initialize_trajectory(model, trial.marker_positions, trial.markers, seed)
        coefficients.append(trial.basis.fit_least_squares(q_ik))
    return coefficients


def _uniform_dt(times: FloatArray) -> float:
    steps = np.diff(times)
    require(
        bool(np.allclose(steps, steps[0], rtol=1e-6)), "replay needs uniform sampling"
    )
    return float(steps[0])


def _replay_trial(
    trial: TrialData,
    coefficients: FloatArray,
    inputs: FloatArray,
    fit: OuterFit,
    config: ReplayConfig,
) -> tuple[ReplayReport, FloatArray]:
    q, v, _ = trial.basis.evaluate(coefficients)
    reference = np.concatenate([q, v], axis=1)
    dt = _uniform_dt(trial.times)
    step = config.step_factory(fit.geometry, fit.parameters, dt)
    nominal_inputs = inputs[:-1]
    linearization = linearize_along_trajectory(
        step, reference[:-1], nominal_inputs, _LINEARIZATION_EPS
    )
    gains = tvlqr_gains(linearization, config.state_weight, config.input_weight)
    report = replay_acceptance(
        step,
        reference,
        nominal_inputs,
        gains,
        config.initial_perturbation,
        config.position_tolerance,
        dt,
    )
    return report, gains


@dataclass(frozen=True)
class SubjectFitProblem:
    """Everything needed to fit one subject: model factory, trials, seeds, priors."""

    factory: ModelFactory
    trials: tuple[TrialData, ...]
    first_frame_q: tuple[FloatArray, ...]
    initial_geometry: FloatArray
    initial_parameters: FloatArray
    parameterization: InertialParameterization
    priors: OuterPriors
    options: OuterOptions

    def __post_init__(self) -> None:
        require(len(self.trials) >= 1, "at least one trial")
        require(
            len(self.trials) == len(self.first_frame_q),
            "one seed configuration per trial",
        )


def fit_subject(
    problem: SubjectFitProblem,
    replay_config: ReplayConfig | None = None,
    n_phase_bins: int = 0,
) -> SubjectFitReport:
    """Run the full subject pipeline; see the module docstring for the stages."""
    factory, trials = problem.factory, problem.trials
    timings: dict[str, float] = {}
    started = time.perf_counter()
    coefficients0 = _initialize(
        factory, trials, problem.first_frame_q, problem.initial_geometry
    )
    timings["initialisation"] = time.perf_counter() - started

    started = time.perf_counter()
    fit = fit_trials(
        factory,
        trials,
        coefficients0,
        problem.initial_geometry,
        problem.initial_parameters,
        problem.parameterization,
        problem.priors,
        problem.options,
    )
    timings["joint_fit"] = time.perf_counter() - started

    started = time.perf_counter()
    model = factory(fit.geometry)
    unactuated = np.flatnonzero(~np.any(np.abs(model.input_matrix) > 0.0, axis=1))
    observability = structural_observability(
        _regressors(model, trials, fit), unactuated
    )
    timings["observability"] = time.perf_counter() - started

    replays: list[ReplayReport] = []
    gains: list[FloatArray] = []
    if replay_config is not None:
        started = time.perf_counter()
        offset = 0
        for trial, coefficients in zip(trials, fit.coefficients, strict=True):
            inputs = fit.inner.inputs[offset : offset + trial.n_times]
            report, gain = _replay_trial(
                trial, coefficients, inputs, fit, replay_config
            )
            replays.append(report)
            gains.append(gain)
            offset += trial.n_times
        timings["replay"] = time.perf_counter() - started

    template, modes = None, None
    if n_phase_bins > 0:
        trial_index = np.repeat(np.arange(len(trials)), [t.n_times for t in trials])
        phase_index = np.concatenate(
            [phase_bins(t.times, n_phase_bins) for t in trials]
        )
        template, per_trial = extract_template(
            fit.inner.inputs, trial_index, phase_index, n_phase_bins
        )
        if len(trials) >= 2:
            modes = principal_modes(per_trial, n_modes=1)
    return SubjectFitReport(
        fit, observability, tuple(replays), tuple(gains), template, modes, timings
    )


def _regressors(model, trials: Sequence[TrialData], fit: OuterFit) -> FloatArray:
    kinematics = [
        trial.basis.evaluate(c)
        for trial, c in zip(trials, fit.coefficients, strict=True)
    ]
    q, v, a = (np.concatenate([k[i] for k in kinematics]) for i in range(3))
    return model.regressor(q, v, a)
