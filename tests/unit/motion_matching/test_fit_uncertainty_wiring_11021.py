"""Wiring unit tests for parameter uncertainty on prefix and multi-shooting fits (Issue #11021)."""

import numpy as np
import pytest

from src.shared.python.estimation.fit_uncertainty import ParameterUncertainty
from src.shared.python.motion_matching.multi_shooting_fit import (
    MultipleShootingOptions,
    fit_multiple_shooting,
)
from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    fit_prefixes,
)

pytestmark = pytest.mark.unit


def test_prefix_fit_stage_populates_parameter_uncertainty() -> None:
    """PrefixStage.parameter_uncertainty is populated with the correct shape."""
    time = np.array([0.0, 0.5, 1.0])
    points = np.zeros((3, 1, 3))
    points[1, 0, 0] = 0.2
    points[2, 0, 0] = 0.4

    def forward(parameters: np.ndarray, requested: np.ndarray) -> np.ndarray:
        result = np.zeros((len(requested), 1, 3))
        result[:, 0, 0] = parameters[0] * requested
        return result

    target = MarkerTarget(time, points, np.ones(1))
    fitted = fit_prefixes(
        target,
        forward,
        initial=np.ones(1),
        lower=np.zeros(1),
        upper=2 * np.ones(1),
        prefix_end_s=(1.0,),
        acceptance_rmse_m=0.5,
    )

    stage = fitted.stages[0]
    assert stage.parameter_uncertainty is not None
    assert isinstance(stage.parameter_uncertainty, ParameterUncertainty)
    assert stage.parameter_uncertainty.status in ("estimated", "partial_at_bounds")
    assert stage.parameter_uncertainty.parameter_indices == (0,)
    assert stage.parameter_uncertainty.covariance is not None
    assert stage.parameter_uncertainty.covariance.shape == (
        stage.parameters.size,
        stage.parameters.size,
    )
    assert stage.parameter_uncertainty.standard_errors is not None
    assert stage.parameter_uncertainty.standard_errors.shape == (stage.parameters.size,)


def test_multi_shooting_fit_populates_theta_uncertainty() -> None:
    """MultipleShootingFit.theta_uncertainty is populated with the correct theta shape."""
    time = np.array([0.0, 0.5, 1.0])
    points = np.zeros((3, 2, 3))
    target = MarkerTarget(time, points, np.array([1.0, 0.0]))

    def segmented(
        theta: np.ndarray, clock: np.ndarray, state: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        pred = np.zeros((len(clock), 2, 3))
        s_val = 0.0 if state is None else float(state[0])
        pred[:, 0, 0] = theta[0] * clock + s_val
        end_s = np.array([s_val + theta[0] * float(clock[-1] - clock[0])])
        return pred, end_s

    def continuous(theta: np.ndarray, clock: np.ndarray) -> np.ndarray:
        pred = np.zeros((len(clock), 2, 3))
        pred[:, 0, 0] = theta[0] * clock
        return pred

    result = fit_multiple_shooting(
        target,
        segmented,
        continuous,
        initial_theta=np.zeros(1),
        lower_theta=-np.ones(1),
        upper_theta=np.ones(1),
        initial_states={0.5: np.zeros(1)},
        state_bounds={0.5: (-np.ones(1), np.ones(1))},
        options=MultipleShootingOptions(shooting_nodes=(0.5, 1.0)),
    )

    assert result.theta_uncertainty is not None
    assert isinstance(result.theta_uncertainty, ParameterUncertainty)
    assert result.theta_uncertainty.status in ("estimated", "partial_at_bounds")
    assert result.theta_uncertainty.parameter_indices == tuple(range(len(result.theta)))
    assert result.theta_uncertainty.covariance is not None
    assert result.theta_uncertainty.covariance.shape == (
        len(result.theta),
        len(result.theta),
    )
    assert result.theta_uncertainty.standard_errors is not None
    assert result.theta_uncertainty.standard_errors.shape == (len(result.theta),)
