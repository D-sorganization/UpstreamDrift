"""Tests for the stochastic green sampler (#14172)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.putting_analytics.green import (
    GreenVariability,
    RoughnessField,
    sample_green,
    sample_roughness,
)
from src.shared.python.putting_dynamics.friction import stimp_to_rolling_mu
from src.shared.python.putting_dynamics.surfaces import bumpy_height_field

pytestmark = pytest.mark.unit

FULL = GreenVariability(
    stimp_ft=10.5, stimp_cv=0.06, grade_percent=2.0, grade_sd_percent=0.3,
    aspect_deg=30.0, aspect_sd_deg=8.0, grain_strength=0.3, grain_strength_sd=0.05,
    grain_direction_deg=-40.0, grain_direction_sd_deg=15.0,
    roughness_amplitude_m=0.002, roughness_correlation_m=1.0, roughness_modes=128,
)  # fmt: skip
N_GREENS = 4000


def _draws() -> list:
    return [sample_green(FULL, seed) for seed in range(N_GREENS)]


def test_per_green_distributions_match_config() -> None:
    draws = _draws()
    stimp = np.array([d.stimp_ft for d in draws])
    se = FULL.stimp_ft * FULL.stimp_cv / math.sqrt(N_GREENS)
    assert abs(stimp.mean() - FULL.stimp_ft) < 4 * se
    cv = stimp.std(ddof=1) / stimp.mean()
    assert abs(cv - FULL.stimp_cv) < 5 * FULL.stimp_cv / math.sqrt(2 * N_GREENS)
    checks = [
        ([d.grade_percent for d in draws], FULL.grade_percent, FULL.grade_sd_percent),
        ([d.aspect_deg for d in draws], FULL.aspect_deg, FULL.aspect_sd_deg),
        (
            [d.grain_strength for d in draws],
            FULL.grain_strength,
            FULL.grain_strength_sd,
        ),
        (
            [math.degrees(d.grain_direction_rad) for d in draws],
            FULL.grain_direction_deg,
            FULL.grain_direction_sd_deg,
        ),
    ]
    for values, mean, sd in checks:
        arr = np.array(values)
        assert abs(arr.mean() - mean) < 4 * sd / math.sqrt(N_GREENS)
        assert abs(arr.std(ddof=1) / sd - 1.0) < 5 / math.sqrt(2 * N_GREENS)


def test_same_seed_reproduces() -> None:
    assert sample_green(FULL, 5) == sample_green(FULL, 5)
    assert sample_green(FULL, 5) != sample_green(FULL, 6)


@pytest.mark.parametrize(
    ("sd_field", "out_field", "nominal"),
    [
        ("stimp_cv", "stimp_ft", FULL.stimp_ft),
        ("grade_sd_percent", "grade_percent", FULL.grade_percent),
        ("aspect_sd_deg", "aspect_deg", FULL.aspect_deg),
        ("grain_strength_sd", "grain_strength", FULL.grain_strength),
        (
            "grain_direction_sd_deg",
            "grain_direction_rad",
            math.radians(FULL.grain_direction_deg),
        ),
    ],
)
def test_zeroing_component_is_exact_and_isolated(
    sd_field: str, out_field: str, nominal: float
) -> None:
    full = sample_green(FULL, 11)
    ablated = sample_green(replace(FULL, **{sd_field: 0.0}), 11)
    assert getattr(ablated, out_field) == nominal
    for name in ablated.__dataclass_fields__:
        if name != out_field:
            assert getattr(ablated, name) == getattr(full, name)


def test_roughness_variance_and_correlation_match_config() -> None:
    rng = np.random.default_rng(0)
    ell, amp = FULL.roughness_correlation_m, FULL.roughness_amplitude_m
    pts = rng.uniform(-50.0, 50.0, (3000, 2))
    theta = rng.uniform(0.0, 2.0 * math.pi, 3000)
    lag = pts + ell * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    var, corr = [], []
    for putt in range(12):
        field = sample_roughness(FULL, 3, putt)
        h0 = field.height(pts[:, 0], pts[:, 1])
        h1 = field.height(lag[:, 0], lag[:, 1])
        var.append(np.mean(h0**2) / amp**2)
        corr.append(np.mean(h0 * h1) / amp**2)
    assert abs(np.mean(var) - 1.0) < 0.1
    assert abs(np.mean(corr) - math.exp(-0.5)) < 0.05


def test_roughness_gradient_matches_finite_difference() -> None:
    field = sample_roughness(FULL, 1, 0)
    x, y = np.array([0.3, -2.0, 7.1]), np.array([1.1, 0.4, -3.3])
    eps = 1e-6
    dx, dy = field.gradient(x, y)
    fd_x = (field.height(x + eps, y) - field.height(x - eps, y)) / (2 * eps)
    fd_y = (field.height(x, y + eps) - field.height(x, y - eps)) / (2 * eps)
    assert np.allclose(dx, fd_x, rtol=1e-5, atol=1e-9)
    assert np.allclose(dy, fd_y, rtol=1e-5, atol=1e-9)


def test_roughness_substreams_differ_and_zero_amplitude_is_zero() -> None:
    a, b = sample_roughness(FULL, 1, 0), sample_roughness(FULL, 1, 1)
    assert not np.allclose(a.wave_vectors, b.wave_vectors)
    flat = sample_roughness(replace(FULL, roughness_amplitude_m=0.0), 1, 0)
    pts = np.linspace(-5.0, 5.0, 7)
    assert np.all(flat.height(pts, pts) == 0.0)
    assert all(np.all(g == 0.0) for g in flat.gradient(pts, pts))
    # Wave vectors are unchanged by the ablation (aligned streams).
    assert np.array_equal(flat.wave_vectors, a.wave_vectors)


def test_variance_parity_with_bumpy_height_field() -> None:
    """Variance parity only: the blur field's correlation length is a box width
    (not a Gaussian ell), so correlation is deliberately not compared."""
    amp = 0.004
    blur = bumpy_height_field(seed=2, amplitude_m=amp, correlation_length_m=1.0)
    assert blur.heights_m.std() == pytest.approx(amp, rel=1e-6)
    cfg = replace(FULL, roughness_amplitude_m=amp)
    xs = np.linspace(-20.0, 20.0, 161)
    grid_x, grid_y = np.meshgrid(xs, xs)
    variances = [
        np.mean(sample_roughness(cfg, 4, p).height(grid_x, grid_y) ** 2)
        for p in range(8)
    ]
    assert np.mean(variances) == pytest.approx(amp**2, rel=0.15)


def test_zero_roughness_surface_is_planar_with_stimp_friction() -> None:
    sample = sample_green(FULL, 21)
    spec = sample.to_surface_spec(None, extent_m=20.0, spacing_m=0.5)
    gx, gy = spec.height.gradient(1.3, -0.7)
    assert math.hypot(gx, gy) == pytest.approx(sample.grade_percent / 100.0)
    assert spec.friction.mu_roll0 == stimp_to_rolling_mu(sample.stimp_ft)
    zero = sample_roughness(replace(FULL, roughness_amplitude_m=0.0), 21, 0)
    flat = sample.to_surface_spec(zero, extent_m=20.0, spacing_m=0.5)
    assert np.array_equal(flat.height.heights_m, spec.height.heights_m)


def test_rough_surface_adds_roughness_to_plane() -> None:
    sample = sample_green(FULL, 21)
    rough = sample_roughness(FULL, 21, 0)
    plane = sample.to_surface_spec(None, extent_m=10.0, spacing_m=0.5)
    spec = sample.to_surface_spec(rough, extent_m=10.0, spacing_m=0.5)
    node = spec.height.origin_m[0] + 0.5 * 4, spec.height.origin_m[1] + 0.5 * 6
    expected = rough.height(np.array(node[0]), np.array(node[1]))
    delta = spec.height.heights_m[6, 4] - plane.height.heights_m[6, 4]
    assert delta == pytest.approx(float(expected))


def test_invalid_config_rejected() -> None:
    for bad in ({"stimp_cv": -0.1}, {"grain_strength": 0.95}, {"roughness_modes": 0}):
        with pytest.raises(ValueError):
            GreenVariability(**bad)
    with pytest.raises(ValueError):
        sample_green(FULL, -1)
