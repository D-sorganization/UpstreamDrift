"""Globe rgba sampling uses the shared cividis scalar scale (#15396)."""

from __future__ import annotations

import math

import pytest

from src.shared.python.segment_kinetic_energy.render_globes import (
    globe_rgba_for_ke,
    ke_scalar_color_scale,
    sample_for_globe_ke,
)

pytestmark = pytest.mark.unit


def _zero_rgb(scale, base=(0.15, 0.35, 0.95)):
    shaded = scale.shade_rgb(0.0, base)
    assert shaded is not None
    return shaded


def test_sample_for_globe_ke_zero_and_nonfinite() -> None:
    assert sample_for_globe_ke(0.0) == 0.0
    assert sample_for_globe_ke(-1.0) == 0.0
    assert sample_for_globe_ke(None) is None
    assert sample_for_globe_ke(float("nan")) is None


def test_globe_rgba_zero_matches_colormap_zero() -> None:
    ref = 100.0
    zero = _zero_rgb(ke_scalar_color_scale(ref))
    rgba = globe_rgba_for_ke(0.0, reference_peak_j=ref)
    assert rgba[:3] == pytest.approx(zero)
    assert rgba[3] == 1.0


def test_globe_rgba_tiny_positive_matches_colormap_zero() -> None:
    ref = 100.0
    zero = _zero_rgb(ke_scalar_color_scale(ref))
    rgba = globe_rgba_for_ke(1e-15, reference_peak_j=ref)
    assert rgba[:3] == pytest.approx(zero)


def test_globe_rgba_nan_uses_base_colour() -> None:
    ref = 100.0
    rgba = globe_rgba_for_ke(float("nan"), reference_peak_j=ref)
    assert rgba[:3] == pytest.approx((0.15, 0.35, 0.95))


def test_globe_rgba_positive_uses_colormap_path() -> None:
    ref = 100.0
    scale = ke_scalar_color_scale(ref)
    expected = scale.shade_rgb(50.0, (0.15, 0.35, 0.95))
    assert expected is not None
    rgba = globe_rgba_for_ke(50.0, reference_peak_j=ref)
    assert rgba[:3] == pytest.approx(expected)
    assert all(math.isfinite(channel) for channel in rgba)
