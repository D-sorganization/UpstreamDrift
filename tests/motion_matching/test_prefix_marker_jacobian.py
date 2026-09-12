"""Optimizer derivatives must match the actual weighted and masked residual."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.estimation.residuals import finite_difference_jacobian
from src.shared.python.motion_matching import prefix_fit

pytestmark = pytest.mark.unit


def test_marker_jacobian_matches_weighted_masked_terminal_residual(monkeypatch) -> None:
    clock = np.array([0.0, 0.4, 1.0])
    points = np.zeros((3, 2, 3))
    points[1, 0] = np.nan
    target = prefix_fit.MarkerTarget(clock, points, np.array([2.0, 0.3]))

    def forward(p, t):
        result = np.zeros((len(t), 2, 3))
        result[:, :, 0] = np.sin(p[0]) * t[:, None]
        result[:, :, 1] = p[1] ** 2 * t[:, None] ** 2
        return result

    def marker_jacobian(p, t):
        result = np.zeros((len(t), 2, 3, 2))
        result[:, :, 0, 0] = np.cos(p[0]) * t[:, None]
        result[:, :, 1, 1] = 2 * p[1] * t[:, None] ** 2
        return result

    called = []

    def solver(residual, initial, **kwargs):
        analytic = kwargs["jac"](initial)
        numeric = finite_difference_jacobian(residual, initial, step=1e-6)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-9)
        assert analytic.shape == (21, 2)
        called.append(True)
        return SimpleNamespace(x=initial, success=True, message="verified")

    monkeypatch.setattr(prefix_fit, "least_squares", solver)
    prefix_fit.fit_prefixes(
        target,
        forward,
        initial=np.array([0.4, 0.7]),
        lower=np.zeros(2),
        upper=np.ones(2),
        prefix_end_s=[1.0],
        acceptance_rmse_m=10.0,
        options=prefix_fit.PrefixFitOptions(
            marker_jacobian=marker_jacobian,
            terminal_weight=1.7,
            time_weight_power=2.0,
            time_weight_scale=3.0,
        ),
    )
    assert called == [True]


@pytest.mark.parametrize(
    "extra",
    [
        {"regularization": lambda p: p},
        {"pelvis_indices": (0, 1), "pelvis_yaw_weight": 1.0},
    ],
)
def test_unsupported_residual_derivatives_fail_explicitly(extra: dict) -> None:
    target = prefix_fit.MarkerTarget(
        np.array([0.0, 1.0]), np.zeros((2, 2, 3)), np.ones(2)
    )
    with pytest.raises(ValueError, match="marker_jacobian"):
        prefix_fit.fit_prefixes(
            target,
            lambda p, t: np.zeros((len(t), 2, 3)),
            initial=np.ones(1),
            lower=np.zeros(1),
            upper=np.full(1, 2.0),
            prefix_end_s=[1.0],
            acceptance_rmse_m=1.0,
            options=prefix_fit.PrefixFitOptions(
                marker_jacobian=lambda p, t: None, **extra
            ),
        )
