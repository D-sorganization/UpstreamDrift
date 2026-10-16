"""NSX-90 (#12719): two-amplitude nonlinearity ratio and the NOT_LINEAR_VALID flag.

The gate threshold is deliberately a required argument: ``loop_nullspace.tex``
says only that linear predictions are "conditional on a nonlinearity gate" and
fixes no number.  ``TEST_THRESHOLD`` below is local to these tests and is NOT a
specification value.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics
from src.shared.python.loop_nullspace import nonlinearity_gate as ng
from src.shared.python.loop_nullspace import variance_propagation as vp
from tests.unit.loop_nullspace._tangent_fixtures import Pendulum
from tests.unit.loop_nullspace._variance_fixtures import (
    nominal_angle,
    pendulum_nonlinear_step,
)

pytestmark = pytest.mark.unit

TEST_THRESHOLD = 0.05  # test-local; not a specification value
DT = 0.1
N_STEPS = 3
U_FF = Pendulum.u_particular.copy()
X_NOM = np.array([nominal_angle(), Pendulum.omega])
C_ROW = np.array([[0.4, 1.0]])


def _rollout(theta: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Inputs ``[dx0 (2), w_0 (2), ..., w_{T-1} (2)]`` -> outcome deviation."""
    x = X_NOM + theta[:2]
    x_ref = X_NOM.copy()
    for t in range(N_STEPS):
        w = gain * theta[2 + 2 * t : 4 + 2 * t]
        x = pendulum_nonlinear_step(x, w, U_FF, DT)
        x_ref = pendulum_nonlinear_step(x_ref, np.zeros(2), U_FF, DT)
    return C_ROW @ (x - x_ref)


def _linear_model() -> tuple[LinearizedDynamics, np.ndarray]:
    """Per-step Jacobians of the SAME nonlinear step along the nominal path."""
    a_all, b_all = [], []
    x = X_NOM.copy()
    h = 1e-6
    for _ in range(N_STEPS):
        a = np.zeros((2, 2))
        b = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            a[:, j] = (
                pendulum_nonlinear_step(x + e, np.zeros(2), U_FF, DT)
                - pendulum_nonlinear_step(x - e, np.zeros(2), U_FF, DT)
            ) / (2 * h)
            b[:, j] = (
                pendulum_nonlinear_step(x, e, U_FF, DT)
                - pendulum_nonlinear_step(x, -e, U_FF, DT)
            ) / (2 * h)
        a_all.append(a)
        b_all.append(b)
        x = pendulum_nonlinear_step(x, np.zeros(2), U_FF, DT)
    return LinearizedDynamics(np.array(a_all), np.array(b_all)), x


def _setup():
    dyn, _ = _linear_model()
    p0 = np.diag([0.01, 0.04])
    sigma = np.repeat(np.diag([0.04, 0.09])[None], N_STEPS, axis=0)
    kernels = vp.adjoint_kernels(dyn, C_ROW)
    return kernels, p0, sigma


def test_probe_directions_are_unit_mahalanobis_with_sd_response() -> None:
    kernels, p0, sigma = _setup()
    dirs = ng.probe_directions(kernels, p0, sigma, include_axes=False)
    assert dirs.shape == (1, 2 + 2 * N_STEPS)
    cov = vp.stacked_covariance(p0, sigma)
    s_row = vp.stacked_sensitivity(kernels)
    d = dirs[0]
    assert d @ np.linalg.solve(cov, d) == pytest.approx(1.0, rel=1e-12)
    sd = np.sqrt(vp.adjoint_covariance(kernels, p0, sigma)[0, 0])
    assert float(s_row[0] @ d) == pytest.approx(sd, rel=1e-12)


def test_axis_probes_are_one_sd_along_each_input() -> None:
    kernels, p0, sigma = _setup()
    dirs = ng.probe_directions(kernels, p0, sigma, include_axes=True)
    assert dirs.shape[0] == 1 + 2 + 2 * N_STEPS
    cov = vp.stacked_covariance(p0, sigma)
    assert np.allclose(np.diag(dirs[1:] @ dirs[1:].T), np.diag(cov))


def test_small_amplitude_is_linear_valid_and_ratio_scales_with_amplitude() -> None:
    kernels, p0, sigma = _setup()
    dirs = ng.probe_directions(kernels, p0, sigma)
    r1 = ng.assess_nonlinearity(
        _rollout, dirs, amplitude=0.25, gate_threshold=TEST_THRESHOLD
    )
    r2 = ng.assess_nonlinearity(
        _rollout, dirs, amplitude=0.5, gate_threshold=TEST_THRESHOLD
    )
    assert r1.validity is ng.LinearValidity.LINEAR_VALID
    assert r1.max_ratio[0] > 0.0
    # Quadratic leading nonlinearity: the ratio is O(amplitude).
    assert r2.max_ratio[0] / r1.max_ratio[0] == pytest.approx(2.0, rel=0.1)


def test_large_amplitude_is_flagged_not_linear_valid() -> None:
    kernels, p0, sigma = _setup()
    dirs = ng.probe_directions(kernels, p0, sigma)
    rep = ng.assess_nonlinearity(
        _rollout, dirs, amplitude=60.0, gate_threshold=TEST_THRESHOLD
    )
    assert rep.validity is ng.LinearValidity.NOT_LINEAR_VALID
    assert rep.max_ratio[0] > TEST_THRESHOLD
    assert rep.exceeded == (True,)


def test_exactly_linear_rollout_has_zero_ratio() -> None:
    kernels, p0, sigma = _setup()
    dirs = ng.probe_directions(kernels, p0, sigma)
    mat = np.arange(1.0, 1.0 + dirs.shape[1])[None, :]
    rep = ng.assess_nonlinearity(
        lambda th: mat @ th, dirs, amplitude=3.0, gate_threshold=1e-9
    )
    assert rep.max_ratio[0] < 1e-12
    assert rep.validity is ng.LinearValidity.LINEAR_VALID


def test_saturation_is_flagged() -> None:
    """Non-claim: a linear prediction does not cover actuator saturation."""
    kernels, p0, sigma = _setup()
    dirs = ng.probe_directions(kernels, p0, sigma)
    sat = lambda th: np.clip(_rollout(th), -0.01, 0.01)  # noqa: E731
    rep = ng.assess_nonlinearity(
        sat, dirs, amplitude=5.0, gate_threshold=TEST_THRESHOLD
    )
    assert rep.validity is ng.LinearValidity.NOT_LINEAR_VALID


def test_non_finite_rollout_is_not_linear_valid() -> None:
    kernels, p0, sigma = _setup()
    dirs = ng.probe_directions(kernels, p0, sigma)

    def failing(th: np.ndarray) -> np.ndarray:
        out = _rollout(th)
        return out * np.nan if np.linalg.norm(th) > 0 else out

    rep = ng.assess_nonlinearity(
        failing, dirs, amplitude=1.0, gate_threshold=TEST_THRESHOLD
    )
    assert rep.validity is ng.LinearValidity.NOT_LINEAR_VALID
    assert rep.n_failed == dirs.shape[0]


def test_insensitive_direction_with_no_response_has_zero_ratio() -> None:
    dirs = np.eye(3)
    rep = ng.assess_nonlinearity(
        lambda th: np.array([2.0 * th[0]]),
        dirs,
        amplitude=1.0,
        gate_threshold=0.1,
    )
    assert rep.ratios[1, 0] == 0.0 and rep.ratios[2, 0] == 0.0
    assert rep.validity is ng.LinearValidity.LINEAR_VALID


def test_threshold_has_no_default_and_is_validated() -> None:
    dirs = np.eye(2)
    with pytest.raises(TypeError):
        ng.assess_nonlinearity(lambda th: th[:1], dirs, amplitude=1.0)  # type: ignore[call-arg]
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gate_threshold"):
            ng.assess_nonlinearity(
                lambda th: th[:1], dirs, amplitude=1.0, gate_threshold=bad
            )
    with pytest.raises(ValueError, match="amplitude"):
        ng.assess_nonlinearity(
            lambda th: th[:1], dirs, amplitude=0.0, gate_threshold=0.1
        )
    with pytest.raises(ValueError, match="directions"):
        ng.assess_nonlinearity(
            lambda th: th[:1], np.zeros(3), amplitude=1.0, gate_threshold=0.1
        )
