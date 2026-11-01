"""Batch roll integrator (#14171): laws, parity with simulate_ball, capture, speed."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.shared.python.core.physics_constants import GOLF_BALL_RADIUS_M, GRAVITY_M_S2
from src.shared.python.putting_analytics import batch_roll
from src.shared.python.putting_analytics.batch_roll import (
    _effective_mu,
    effective_hole_radius_m,
    roll_batch,
)
from src.shared.python.putting_dynamics.friction import (
    FrictionParams,
    rolling_mu,
    sliding_mu,
    stimp_to_rolling_mu,
)
from src.shared.python.putting_dynamics.solver import capture_speed_mps, simulate_ball
from src.shared.python.putting_dynamics.surfaces import (
    FrictionField,
    HeightField,
    SurfaceSpec,
)
from src.shared.python.putting_dynamics.types import BallState

pytestmark = pytest.mark.unit

PARITY_TOL_M = 1e-9  # rest-position agreement with the scalar solver
HOLE = (2.0, 0.0)


def _surface(grade=0.0, aspect=0.0, stimp=10.0, **friction) -> SurfaceSpec:  # type: ignore[no-untyped-def]
    height = HeightField.planar(grade, aspect) if grade else HeightField.flat()
    params = FrictionParams(mu_roll0=stimp_to_rolling_mu(stimp), **friction)
    return SurfaceSpec(
        height=height, friction_field=FrictionField.uniform(), friction=params
    )


def _start(speed, heading=0.0, spin_ratio=1.0) -> np.ndarray:  # type: ignore[no-untyped-def]
    speed = np.atleast_1d(np.asarray(speed, dtype=float))
    heading = np.broadcast_to(heading, speed.shape)
    zeros = np.zeros_like(speed)
    return np.stack(
        (
            zeros,
            zeros,
            speed * np.cos(heading),
            speed * np.sin(heading),
            speed * spin_ratio,
        )
    )


def _assert_parity(surface: SurfaceSpec, start: np.ndarray) -> None:
    batch = roll_batch(start, surface, HOLE, capture="point")
    for i in range(start.shape[1]):
        ball = BallState(
            0.0,
            0.0,
            vx_mps=start[2, i],
            vy_mps=start[3, i],
            spin_rad_s=start[4, i] / GOLF_BALL_RADIUS_M,
        )
        ref = simulate_ball(ball, surface, *HOLE)
        assert batch.holed[i] == ref.holed
        assert batch.rest_x_m[i] == pytest.approx(ref.rest_x_m, abs=PARITY_TOL_M)
        assert batch.rest_y_m[i] == pytest.approx(ref.rest_y_m, abs=PARITY_TOL_M)


def test_effective_mu_matches_scalar_laws() -> None:
    params = FrictionParams(
        mu_roll0=0.08, k_v_per_mps=0.1, grain_strength=0.3, grain_direction_rad=0.7
    )
    surface = SurfaceSpec(HeightField.flat(), FrictionField.uniform(), params)
    speed = np.array([0.0, 0.4, 1.3, 2.0])
    heading = np.array([0.0, 1.0, -2.0, 3.0])
    vx, vy = speed * np.cos(heading), speed * np.sin(heading)
    mults = (np.array([1.0, 0.9, 1.2, 1.1]), np.array([1.0, 1.1, 0.8, 1.0]))
    for sliding in (np.zeros(4, bool), np.ones(4, bool)):
        got = _effective_mu(surface, speed, vx, vy, sliding, mults)
        for i in range(4):
            want = (
                sliding_mu(params, mults[1][i], heading[i])
                if sliding[i]
                else rolling_mu(params, speed[i], mults[0][i], heading[i])
            )
            assert got[i] == pytest.approx(want, rel=1e-12)


def test_parity_with_simulate_ball_quick() -> None:
    rng = np.random.default_rng(7)
    speed = rng.uniform(0.6, 1.4, 4)
    heading = rng.uniform(-0.05, 0.05, 4)
    for surface in (_surface(), _surface(2.0, 90.0, 12.0, grain_strength=0.2)):
        _assert_parity(surface, _start(speed, heading, np.array([1, 1, 0.5, 0.2])))


@pytest.mark.slow
def test_parity_sample_across_planes_and_stimps() -> None:
    """≥200 seeded putts: flat, uphill, downhill and side-hill at two stimps."""
    rng = np.random.default_rng(14171)
    planes = [(0.0, 0.0), (2.0, 180.0), (2.0, 0.0), (2.0, 90.0)]
    for grade, aspect in planes:
        for stimp in (8.0, 12.0):
            speed = rng.uniform(0.5, 1.8, 25)
            heading = rng.uniform(-0.1, 0.1, 25)
            spin = np.where(rng.random(25) < 0.3, 0.5, 1.0)
            _assert_parity(_surface(grade, aspect, stimp), _start(speed, heading, spin))


def test_flat_rollout_matches_closed_form() -> None:
    """Pure roll goes v0²/(2 μ_r g); a skid start rolls on from 5/7 v0."""
    surface = _surface(stimp=10.0)
    mu_r, mu_s, g = surface.friction.mu_roll0, surface.friction.mu_slide, GRAVITY_M_S2
    v0 = 1.6
    rolled = roll_batch(_start([v0]), surface, (50.0, 5.0))
    assert rolled.rest_x_m[0] == pytest.approx(v0**2 / (2 * mu_r * g), abs=2e-3)
    t_slide = v0 / (3.5 * mu_s * g)
    slide = v0 * t_slide - 0.5 * mu_s * g * t_slide**2
    skid = roll_batch(_start([v0], spin_ratio=0.0), surface, (50.0, 5.0))
    expected = slide + (5 / 7 * v0) ** 2 / (2 * mu_r * g)
    assert skid.rest_x_m[0] == pytest.approx(expected, abs=2e-3)


def test_effective_radius_follows_the_tools_shape() -> None:
    from shared.python.swing_sim.putting import capture as tools_capture

    scale = tools_capture.capture_speed_mps() / capture_speed_mps()
    speeds = np.linspace(0.0, 1.2 * capture_speed_mps(), 9)
    want = [tools_capture.effective_hole_radius_m(v * scale) for v in speeds]
    np.testing.assert_allclose(effective_hole_radius_m(speeds), want, atol=1e-15)


def test_slow_off_centre_putt_drops_and_fast_one_lips_out() -> None:
    surface = _surface()
    offset = np.arctan2(0.045, 2.0)  # passes 4.5 cm from the cup centre
    start = _start([1.6, 2.6], heading=offset)
    result = roll_batch(start, surface, HOLE)
    assert result.holed.tolist() == [True, False]
    assert result.lipped_out.tolist() == [False, True]
    assert result.speed_at_hole_mps[1] > capture_speed_mps() / 2


def test_chunking_does_not_change_results(monkeypatch: pytest.MonkeyPatch) -> None:
    surface = _surface(1.5, 45.0)
    start = _start(np.linspace(0.8, 2.2, 7), np.linspace(-0.1, 0.1, 7))
    whole = roll_batch(start, surface, HOLE)
    monkeypatch.setattr(batch_roll, "_CHUNK", 3)
    chunked = roll_batch(start, surface, HOLE)
    for a, b in zip(whole.__dict__.values(), chunked.__dict__.values(), strict=True):
        np.testing.assert_array_equal(a, b)


def test_contract_errors() -> None:
    with pytest.raises(ValueError):
        roll_batch(np.zeros((4, 2)), _surface(), HOLE)
    with pytest.raises(ValueError):
        roll_batch(_start([1.0]), _surface(), HOLE, capture="nearest")  # type: ignore[arg-type]


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.timeout(900)
def test_benchmark_hundred_thousand_putts() -> None:
    """Budget: 10^5 putts in under 300 s single-threaded (about 75 s measured)."""
    n = 100_000
    rng = np.random.default_rng(0)
    start = _start(rng.uniform(0.8, 2.5, n), rng.uniform(-0.2, 0.2, n))
    began = time.perf_counter()
    result = roll_batch(start, _surface(2.0, 90.0, 11.0), (3.0, 0.2))
    assert time.perf_counter() - began < 300.0
    assert not result.timed_out.any()
