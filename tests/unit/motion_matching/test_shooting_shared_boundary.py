"""Shared window-boundary samples may be observed once for objective parity."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching import multi_shooting_fit as module
from src.shared.python.motion_matching.prefix_fit import MarkerTarget

pytestmark = pytest.mark.unit
CLOCK = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _forward(theta, clock, state):
    x0 = 0.0 if state is None else state[0]
    x = x0 + theta[0] * (clock - clock[0])
    pred = np.zeros((len(clock), 2, 3))
    pred[:, 0, 0] = x
    pred[:, 1, 1] = 2 * x
    return pred, np.array([x[-1]])


def _window_jac(theta, clock, state):
    dt = clock - clock[0]
    count = 1 if state is None else 2
    marker = np.zeros((len(clock), 2, 3, count))
    marker[:, 0, 0, 0], marker[:, 1, 1, 0] = dt, 2 * dt
    end = np.zeros((1, count))
    end[0, 0] = dt[-1]
    if state is not None:
        marker[:, 0, 0, 1], marker[:, 1, 1, 1] = 1.0, 2.0
        end[0, 1] = 1.0
    return marker, end


def _capture(fun, jac, x0, *args, **kwargs):
    return SimpleNamespace(
        residual=fun(x0),
        jacobian=jac(x0),
        x=x0,
        success=False,
        message="captured",
        nfev=1,
        active_mask=np.zeros_like(x0),
    )


def _run(policy: str, monkeypatch: pytest.MonkeyPatch, points=None):
    captured = {}

    def capture(fun, jac, x0, lo, hi, **kwargs):
        captured["value"] = _capture(fun, jac, x0)
        captured["equality_start"] = kwargs["equality_start"]
        return captured["value"]

    monkeypatch.setattr(module, "solve_equality_least_squares", capture)
    target = MarkerTarget(
        CLOCK, np.zeros((5, 2, 3)) if points is None else points, np.ones(2)
    )
    fit = module.fit_multiple_shooting(
        target,
        _forward,
        lambda theta, clock: _forward(theta, clock, None)[0],
        initial_theta=np.array([1.0]),
        lower_theta=np.array([0.0]),
        upper_theta=np.array([3.0]),
        initial_states={0.5: np.array([0.5])},
        state_bounds={0.5: (np.array([-2.0]), np.array([2.0]))},
        options=module.MultipleShootingOptions(
            shooting_nodes=(0.5, 1.0),
            solver="slsqp",
            window_jacobian=_window_jac,
            terminal_weight=0.5,
            shared_boundary_policy=policy,
        ),
    )
    return captured, fit


def test_default_policy_counts_boundary_sample_in_both_windows(monkeypatch) -> None:
    captured, _ = _run("both", monkeypatch)
    # 3 + 3 samples, 2 markers, 3 axes, then 1 defect row, then 6 terminal rows.
    assert captured["value"].residual.shape == (36 + 1 + 6,)
    assert captured["equality_start"] == 36


def test_once_policy_observes_shared_sample_once_and_keeps_rows_aligned(
    monkeypatch,
) -> None:
    captured, fit = _run("once", monkeypatch)
    residual, jacobian = captured["value"].residual, captured["value"].jacobian
    assert residual.shape == (30 + 1 + 6,)
    assert jacobian.shape == (37, 2)
    assert captured["equality_start"] == 30
    # Marker rows equal the uninterrupted single-window residual at zero defect.
    full = _forward(np.array([1.0]), CLOCK, None)[0].ravel()
    np.testing.assert_allclose(residual[:30], full)
    assert fit.segmented_rmse_m == pytest.approx(fit.unsegmented_rmse_m)


def test_once_policy_jacobian_matches_finite_difference(monkeypatch) -> None:
    rows = {}

    def capture(fun, jac, x0, lo, hi, **kwargs):
        h = 1e-6
        fd = np.column_stack(
            [(fun(x0 + d * h) - fun(x0 - d * h)) / (2 * h) for d in np.eye(len(x0))]
        )
        np.testing.assert_allclose(jac(x0), fd, atol=1e-8)
        rows["count"] = fd.shape[0]
        return _capture(fun, jac, x0)

    monkeypatch.setattr(module, "solve_equality_least_squares", capture)
    module.fit_multiple_shooting(
        MarkerTarget(CLOCK, np.zeros((5, 2, 3)), np.ones(2)),
        _forward,
        lambda theta, clock: _forward(theta, clock, None)[0],
        initial_theta=np.array([1.0]),
        lower_theta=np.array([0.0]),
        upper_theta=np.array([3.0]),
        initial_states={0.5: np.array([0.7])},
        state_bounds={0.5: (np.array([-2.0]), np.array([2.0]))},
        options=module.MultipleShootingOptions(
            shooting_nodes=(0.5, 1.0),
            solver="slsqp",
            window_jacobian=_window_jac,
            terminal_weight=0.5,
            shared_boundary_policy="once",
        ),
    )
    assert rows["count"] == 37


@pytest.mark.parametrize("policy", ["twice", None, 1])
def test_invalid_policy_is_rejected(policy) -> None:
    with pytest.raises(ValueError):
        module.MultipleShootingOptions(
            shooting_nodes=(1.0,), shared_boundary_policy=policy
        )
