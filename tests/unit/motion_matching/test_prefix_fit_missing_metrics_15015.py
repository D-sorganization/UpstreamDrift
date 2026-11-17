"""Prefix-fit terminal metrics must not pass when data is missing (#15015)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    PrefixFitOptions,
    fit_prefixes,
)

pytestmark = pytest.mark.unit


def test_prefix_fit_unmeasured_terminal_rmse_is_not_zero() -> None:
    """Missing terminal observations must not report a zero terminal RMSE."""
    time = np.array([0.0, 1.0])
    points = np.zeros((2, 1, 3))
    points[0, 0, 0] = 0.1
    points[1, 0, :] = np.nan
    target = MarkerTarget(time, points, np.ones(1))

    def forward(parameters: np.ndarray, requested: np.ndarray) -> np.ndarray:
        out = np.zeros((len(requested), 1, 3))
        out[:, 0, 0] = parameters[0] * requested
        return out

    fitted = fit_prefixes(
        target,
        forward,
        initial=np.array([1.0]),
        lower=np.array([0.0]),
        upper=np.array([2.0]),
        prefix_end_s=(1.0,),
        acceptance_rmse_m=0.5,
        options=PrefixFitOptions(acceptance_terminal_rmse_m=0.035),
    )
    stage = fitted.stages[0]
    assert math.isnan(stage.terminal_rmse_m)
    assert not fitted.accepted


def test_prefix_fit_invalid_pelvis_yaw_does_not_accept_at_zero_percent() -> None:
    """When pelvis waist markers are degenerate, yaw error must not read as 0 %."""
    time = np.array([0.0, 1.0])
    points = np.zeros((2, 2, 3))
    points[:, 0, 0] = 0.0
    points[:, 1, 0] = 0.0
    target = MarkerTarget(time, points, np.ones(2))

    def forward(parameters: np.ndarray, requested: np.ndarray) -> np.ndarray:
        out = np.zeros((len(requested), 2, 3))
        out[:, 0, 0] = parameters[0] * 0.01
        out[:, 1, 0] = parameters[0] * 0.01
        return out

    fitted = fit_prefixes(
        target,
        forward,
        initial=np.array([1.0]),
        lower=np.array([0.0]),
        upper=np.array([2.0]),
        prefix_end_s=(1.0,),
        acceptance_rmse_m=0.5,
        options=PrefixFitOptions(
            pelvis_indices=(0, 1),
            pelvis_yaw_max_error_pct=5.0,
        ),
    )
    stage = fitted.stages[0]
    assert math.isnan(stage.pelvis_yaw_error_pct)
    assert not fitted.accepted
