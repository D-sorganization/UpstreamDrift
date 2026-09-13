"""Direct node-chart columns retain physical shooting defects and chain rules."""

from types import SimpleNamespace

import numpy as np
import pytest
from src.shared.python.motion_matching import multi_shooting_fit as module
from src.shared.python.motion_matching.prefix_fit import MarkerTarget

pytestmark = pytest.mark.unit


def _forward(
    theta: np.ndarray, clock: np.ndarray, state: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    q0, v0 = (0.4, 0.3) if state is None else state
    dt = clock - clock[0]
    q = q0 + v0 * dt + theta[0] * dt**2 / 2
    v = v0 + theta[0] * dt
    return np.stack((q, v, q + 2 * v), axis=-1)[:, None, :], np.array([q[-1], v[-1]])


def _window(mode: str, malformed: str | None = None):
    def jac(
        theta: np.ndarray, clock: np.ndarray, state: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = clock - clock[0]
        count = 1 if state is None else 3
        marker = np.zeros((len(clock), 1, 3, count))
        marker[:, 0, :, 0] = np.stack((dt**2 / 2, dt, dt**2 / 2 + 2 * dt), axis=-1)
        endpoint = np.zeros((2, count))
        endpoint[:, 0] = [dt[-1] ** 2 / 2, dt[-1]]
        if state is not None:
            marker[:, 0, :, 1] = [1, 0, 1]
            marker[:, 0, :, 2] = np.stack((dt, np.ones_like(dt), dt + 2), axis=-1)
            endpoint[:, 1:] = [[1, dt[-1]], [0, 1]]
            if mode == "node":
                chain = np.array([[1, 0], [0, 1], [0, 2 * state[0]]])
                marker, endpoint = marker @ chain, endpoint @ chain
            if malformed == "columns":
                marker, endpoint = marker[..., :-1], endpoint[:, :-1]
            elif malformed == "rows":
                endpoint = endpoint[:1]
        return marker, endpoint

    return jac


def _options(mode: str, solver: str, malformed: str | None = None):
    return module.MultipleShootingOptions(
        shooting_nodes=(0.5, 1.0),
        solver=solver,
        state_transform=lambda t, z: np.array([z[0], z[0] ** 2]),
        state_transform_jacobian=lambda t, z: np.array([[1], [2 * z[0]]]),
        window_jacobian=_window(mode, malformed),
        window_jacobian_state_coordinates=mode,
        terminal_weight=0.4,
        defect_weight=7,
        defect_scales=np.array([2.0, 3.0]),
    )


def _fit(options):
    clock = np.array([0, 0.25, 0.5, 0.75, 1.0])
    return module.fit_multiple_shooting(
        MarkerTarget(clock, np.zeros((5, 1, 3)), np.ones(1)),
        _forward,
        lambda theta, clock: _forward(theta, clock, None)[0],
        initial_theta=np.array([0.2]),
        lower_theta=np.array([-2.0]),
        upper_theta=np.array([2.0]),
        initial_states={0.5: np.array([0.7])},
        state_bounds={0.5: (np.array([0.1]), np.array([1.5]))},
        options=options,
    )


@pytest.mark.parametrize("mode", ["physical", "node"])
@pytest.mark.parametrize("solver", ["least_squares", "slsqp"])
def test_full_assembled_jacobian_matches_nonlinear_finite_difference(
    mode: str, solver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    audited = []

    def check(fun, jac, x):
        for p in (x, x + np.array([0.1, 0.12])):
            h = 1e-6
            fd = np.column_stack(
                [(fun(p + h * d) - fun(p - h * d)) / (2 * h) for d in np.eye(len(p))]
            )
            actual = jac(p)
            np.testing.assert_allclose(actual, fd, atol=2e-9, rtol=2e-8)
            assert actual.shape == (
                23,
                2,
            )  # 18 marker +2 physical defect +3 terminal rows.
        audited.append(True)
        return SimpleNamespace(
            x=x,
            success=False,
            message="audit only",
            nfev=1,
            active_mask=np.zeros_like(x),
        )

    def ls(fun, x0, **kwargs):
        return check(fun, kwargs["jac"], x0)

    def equality(fun, jac, x0, lo, hi, **kwargs):
        return check(fun, jac, x0)

    monkeypatch.setattr(module, "least_squares", ls)
    monkeypatch.setattr(module, "solve_equality_least_squares", equality)
    _fit(_options(mode, solver))
    assert audited == [True]


@pytest.mark.parametrize("mode", ["typo", None, True])
def test_invalid_coordinate_mode(mode: object) -> None:
    with pytest.raises(ValueError):
        module.MultipleShootingOptions(
            shooting_nodes=(1.0,), window_jacobian_state_coordinates=mode
        )


@pytest.mark.parametrize("mode", ["physical", "node"])
@pytest.mark.parametrize("malformed", ["columns", "rows"])
def test_rejects_malformed_local_jacobian(
    mode: str, malformed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inspect(fun, x0, **kwargs):
        return kwargs["jac"](x0)

    monkeypatch.setattr(module, "least_squares", inspect)
    with pytest.raises(ValueError, match="window Jacobian"):
        _fit(_options(mode, "least_squares", malformed))
