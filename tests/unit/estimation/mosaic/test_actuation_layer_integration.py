"""Integration tests for the MOSAIC actuation layer (MOSAIC-11, #11543).

Verifies:
1. Declared activation model bounds and time constant validation.
2. Analytic step/ramp and bound enforcement.
3. Time constant recovery on synthetic data and unobservable reporting.
4. Side-by-side open-loop replay reporting with and without the actuation layer.
5. Provenance tracking in SubjectFitReport.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.activation_dynamics import (
    ActivationModel,
    ActivationReport,
    fit_time_constants,
    simulate_activation,
)
from src.shared.python.estimation.mosaic.reference_evidence import (
    FixtureSpec,
    TRUE_LENGTHS,
    TRUE_PI,
    _step_factory,
    make_chain,
    perturbed_prior,
    reference_priors,
    synthetic_trials,
)
from src.shared.python.estimation.mosaic.subject_fit import (
    ReplayConfig,
    SubjectFitProblem,
    SubjectFitReport,
    fit_subject,
)
from src.shared.python.estimation.mosaic.inertial import PlanarParameterization
from src.shared.python.estimation.mosaic.outer_solve import OuterOptions

pytestmark = pytest.mark.unit


def test_activation_model_preconditions_enforced() -> None:
    # Lower bound must be strictly less than upper bound
    with pytest.raises(ContractViolationError):
        ActivationModel(tau_act=0.05, tau_deact=0.05, lower=10.0, upper=5.0)

    # Time constants must be strictly positive
    with pytest.raises(ContractViolationError):
        ActivationModel(tau_act=-0.01, tau_deact=0.05, lower=-5.0, upper=5.0)

    # Finite bounds required
    with pytest.raises(ContractViolationError):
        ActivationModel(tau_act=0.05, tau_deact=0.05, lower=float("-inf"), upper=5.0)


def test_time_constant_recovery_and_unobservable_reporting() -> None:
    dt = 0.005
    times = np.arange(200, dtype=np.float64) * dt
    # Step up then down
    u = np.zeros(200, dtype=np.float64)
    u[20:100] = 5.0
    u[100:180] = 1.0

    model = ActivationModel(
        tau_act=0.04,
        tau_deact=0.08,
        lower=-10.0,
        upper=10.0,
        provenance="literature_priors",
    )
    a = simulate_activation(model, times, u[:, None], a0=np.array([0.0]))[:, 0]

    # Fit recovers time constants within tolerance
    fit = fit_time_constants(times, u, a, lower=-10.0, upper=10.0)
    assert fit.tau_act is not None
    assert fit.tau_act == pytest.approx(0.04, rel=0.05)
    assert fit.tau_deact is not None
    assert fit.tau_deact == pytest.approx(0.08, rel=0.05)

    # Completely flat signal reports unobservable (None)
    u_flat = np.full(200, 2.0, dtype=np.float64)
    a_flat = np.full(200, 2.0, dtype=np.float64)
    fit_flat = fit_time_constants(times, u_flat, a_flat, lower=-10.0, upper=10.0)
    assert fit_flat.tau_act is None
    assert fit_flat.tau_deact is None


def test_subject_fit_reports_activation_side_by_side() -> None:
    spec = FixtureSpec(n_trials=1, n_nodes=60)
    trials, truth = synthetic_trials(spec)
    pi_prior = perturbed_prior(spec)

    activation_model = ActivationModel(
        tau_act=[0.03, 0.03],
        tau_deact=[0.06, 0.06],
        lower=[-5.0, -5.0],
        upper=[5.0, 5.0],
        provenance="human_arm_literature_2026",
    )

    problem = SubjectFitProblem(
        factory=make_chain,
        trials=tuple(trials),
        first_frame_q=tuple(t[0][0] + 0.1 for t in truth),
        initial_geometry=TRUE_LENGTHS * spec.geometry_scale,
        initial_parameters=pi_prior,
        parameterization=PlanarParameterization(),
        priors=reference_priors(pi_prior),
        options=OuterOptions(continuation=(0.0, 1.0), max_iterations_per_stage=2),
        activation_model=activation_model,
    )

    config = ReplayConfig(
        step_factory=_step_factory,
        state_weight=np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
        input_weight=np.eye(2) * 0.01,
        position_tolerance=0.25,
        initial_perturbation=np.zeros(6),
    )

    report = fit_subject(problem, config)
    assert isinstance(report, SubjectFitReport)
    assert report.activation_report is not None
    act_report = report.activation_report

    assert isinstance(act_report, ActivationReport)
    assert act_report.provenance == "human_arm_literature_2026"
    assert len(act_report.open_loop_rms_without_actuation) == 1
    assert len(act_report.open_loop_rms_with_actuation) == 1
    # Both numbers measured strictly positive
    assert act_report.open_loop_rms_without_actuation[0] > 0.0
    assert act_report.open_loop_rms_with_actuation[0] > 0.0
