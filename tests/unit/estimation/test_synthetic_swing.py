"""Tests for the synthetic golf-swing benchmark generator."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.synthetic_swing import (
    CorruptedObservation,
    SwingTruth,
    corrupt_observations,
    simulate_swing,
    smooth_swing_torque,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def truth() -> SwingTruth:
    return simulate_swing()


def test_truth_shapes_and_finiteness(truth: SwingTruth) -> None:
    n = truth.t.size
    assert n == 201
    assert truth.q.shape == truth.v.shape == truth.tau.shape == (n, 2)
    assert np.all(np.isfinite(truth.q)) and np.all(np.isfinite(truth.v))
    np.testing.assert_array_equal(truth.tau[-1], truth.tau[-2])


def test_swing_is_physically_reasonable(truth: SwingTruth) -> None:
    peak_v = np.abs(truth.v).max(axis=0)
    assert peak_v[0] > 5.0 and peak_v[1] > 10.0
    assert np.all(peak_v < 200.0)
    assert np.abs(truth.q).max() < 20.0


def test_truth_consistent_with_backend_dynamics(truth: SwingTruth) -> None:
    from src.shared.python.simulation_backends import (
        GolfModelParams,
        SimState,
        make_backend,
    )

    backend = make_backend("ode", GolfModelParams.default())
    for k in (0, 10, 50, 100, 149):
        backend.reset(SimState(q=truth.q[k], v=truth.v[k], time=truth.t[k]))
        tr = backend.rollout(truth.tau[k : k + 1], 1, truth.dt)
        np.testing.assert_allclose(tr.q[1], truth.q[k + 1], atol=1e-10)
        np.testing.assert_allclose(tr.v[1], truth.v[k + 1], atol=1e-10)


def test_torque_profile_continuous() -> None:
    t = np.arange(0.0, 0.3, 0.002)
    tau = smooth_swing_torque(t)
    assert tau.shape == (t.size, 2)
    step = np.abs(np.diff(tau, axis=0)).max(axis=0)
    peak = np.abs(tau).max(axis=0)
    assert np.all(peak > 0)
    assert np.all(step < 0.1 * peak)
    np.testing.assert_allclose(tau[0], 0.0, atol=1e-12)


def test_determinism() -> None:
    truth = simulate_swing()
    a = corrupt_observations(truth, seed=3)
    b = corrupt_observations(truth, seed=3)
    c = corrupt_observations(truth, seed=4)
    np.testing.assert_array_equal(a.q_observed, b.q_observed)
    np.testing.assert_array_equal(a.outlier_indices, b.outlier_indices)
    assert not np.array_equal(a.q_observed, c.q_observed)


def test_outliers_at_reported_indices(truth: SwingTruth) -> None:
    obs = corrupt_observations(truth, noise_std=0.002, outlier_magnitude=0.15)
    assert obs.outlier_indices.size > 0
    err = np.abs(obs.q_observed - truth.q)
    assert np.all(err[obs.outlier_indices].max(axis=1) >= 0.15 * 0.5)
    present = obs.mask.copy()
    present[obs.outlier_indices] = False
    clean = (obs.q_observed - truth.q)[present]
    assert np.std(clean) == pytest.approx(0.002, rel=0.2)
    assert np.abs(clean).max() < 0.15 * 0.5
    assert np.all(np.diff(obs.outlier_indices) > 0)


def test_occlusion_rows(truth: SwingTruth) -> None:
    obs = corrupt_observations(truth, occlusion=(60, 75))
    np.testing.assert_array_equal(obs.occluded_indices, np.arange(60, 75))
    assert np.all(np.isnan(obs.q_observed[60:75]))
    assert not obs.mask[60:75].any()
    assert obs.mask[:60].all() and obs.mask[75:].all()
    assert np.all(np.isfinite(obs.q_observed[obs.mask]))


def test_no_occlusion(truth: SwingTruth) -> None:
    obs = corrupt_observations(truth, occlusion=None)
    assert obs.occluded_indices.size == 0 and obs.mask.all()


def test_outliers_disjoint_from_occlusion_and_start(truth: SwingTruth) -> None:
    for seed in range(20):
        obs = corrupt_observations(truth, outlier_fraction=0.3, seed=seed)
        idx = set(obs.outlier_indices.tolist())
        assert not idx & set(obs.occluded_indices.tolist())
        assert 0 not in idx and 1 not in idx


def test_returns_expected_types(truth: SwingTruth) -> None:
    obs = corrupt_observations(truth)
    assert isinstance(obs, CorruptedObservation)
    assert obs.noise_std == 0.002


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"dt": -0.1},
        {"duration": 0.001, "dt": 0.002},
        {"duration": float("nan")},
        {"q0": (float("inf"), 0.0)},
        {"q0": (1.0,)},
    ],
)
def test_simulate_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        simulate_swing(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_std": -1e-3},
        {"noise_std": float("nan")},
        {"outlier_fraction": -0.1},
        {"outlier_fraction": 0.5},
        {"outlier_magnitude": -1.0},
        {"occlusion": (60, 60)},
        {"occlusion": (-1, 10)},
        {"occlusion": (10, 10_000)},
    ],
)
def test_corrupt_validation(truth: SwingTruth, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        corrupt_observations(truth, **kwargs)
