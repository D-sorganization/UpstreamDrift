"""Shared fixtures for the DIME ZTCF-anchored matching suites (epic #11421).

One seeded synthetic swing, one torque band and one corrupted match are built
once per session and reused, so the matcher, quality and replay suites test
the same benchmark without repeating the ~14 s match.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from src.shared.python.estimation.drift_anchored_matcher import (
    DriftAnchoredMatchResult,
    MatchOptions,
    match_kinematics,
)
from src.shared.python.estimation.drift_prediction import ControlBand
from src.shared.python.estimation.local_torque_window import WindowOptions
from src.shared.python.estimation.synthetic_swing import (
    CorruptedObservation,
    SwingTruth,
    corrupt_observations,
    reference_control_band,
    simulate_swing,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

#: Observation noise of the shared corrupted benchmark [rad].
DIME_NOISE = 0.002
#: Seed of the shared corrupted benchmark.
DIME_SEED = 11


def dime_match_options(noise: float) -> MatchOptions:
    """Benchmark matcher settings for observation noise ``noise`` [rad]."""
    return MatchOptions(
        window=WindowOptions(sigma_obs=noise, sigma_q0=noise, sigma_v0=3.0, n_knots=2),
        window_steps=12,
        stride=4,
        carry_sigma_q=noise,
        carry_sigma_v=0.2,
    )


@pytest.fixture(scope="session")
def dime_band() -> ControlBand:
    """Actuator box and torque-rate limit for the reference swing."""
    return reference_control_band()


@pytest.fixture(scope="session")
def dime_options() -> Callable[[float], MatchOptions]:
    """Factory for benchmark matcher options."""
    return dime_match_options


@pytest.fixture(scope="session")
def dime_provider() -> Any:
    """Analytic ODE reference backend (double pendulum)."""
    return make_backend("ode", GolfModelParams.default())


@pytest.fixture(scope="session")
def dime_truth() -> SwingTruth:
    """Seeded synthetic swing with known torques."""
    return simulate_swing()


@pytest.fixture(scope="session")
def dime_corrupted_match(
    dime_provider: Any, dime_truth: SwingTruth, dime_band: ControlBand
) -> tuple[CorruptedObservation, DriftAnchoredMatchResult]:
    """Noise + spikes + occlusion benchmark and its match (computed once)."""
    obs = corrupt_observations(dime_truth, noise_std=DIME_NOISE, seed=DIME_SEED)
    result = match_kinematics(
        dime_provider,
        dime_truth.t,
        obs.q_observed,
        obs.mask,
        dime_band,
        dime_match_options(DIME_NOISE),
    )
    return obs, result
