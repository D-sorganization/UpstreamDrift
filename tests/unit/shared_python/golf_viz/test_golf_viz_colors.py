"""Unit tests for :mod:`src.shared.python.golf_viz` colour helpers.

These are pure-numpy functions with no Qt/OpenGL dependency, so they run
in any headless environment and carry the bulk of the rendering library's
test coverage (the GL adapters in the GUIs stay thin on purpose).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.golf_viz import (
    ROLL_MODE_RGBA,
    roll_mode_colors,
    sample_gradient,
    speed_colors,
    terrain_colors,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# sample_gradient
# ---------------------------------------------------------------------------


def test_sample_gradient_returns_rgba_in_unit_range() -> None:
    stops = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)]
    out = sample_gradient([0.0, 0.5, 1.0], stops)
    assert out.shape == (3, 4)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_sample_gradient_hits_endpoints() -> None:
    stops = [(0.1, 0.2, 0.3, 1.0), (0.7, 0.8, 0.9, 1.0)]
    out = sample_gradient([0.0, 1.0], stops, vmin=0.0, vmax=1.0)
    np.testing.assert_allclose(out[0], stops[0], atol=1e-9)
    np.testing.assert_allclose(out[1], stops[1], atol=1e-9)


def test_sample_gradient_is_monotonic_between_two_stops() -> None:
    stops = [(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
    out = sample_gradient([0.0, 0.25, 0.5, 0.75, 1.0], stops, vmin=0.0, vmax=1.0)
    red = out[:, 0]
    assert np.all(np.diff(red) > 0.0)


def test_sample_gradient_constant_input_is_safe() -> None:
    stops = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)]
    out = sample_gradient([3.0, 3.0, 3.0], stops)  # vmin == vmax
    assert out.shape == (3, 4)
    assert np.all(np.isfinite(out))


def test_sample_gradient_clamps_out_of_range_values() -> None:
    stops = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)]
    out = sample_gradient([-5.0, 5.0], stops, vmin=0.0, vmax=1.0)
    np.testing.assert_allclose(out[0], stops[0], atol=1e-9)
    np.testing.assert_allclose(out[1], stops[1], atol=1e-9)


def test_sample_gradient_three_stops_uses_middle() -> None:
    stops = [
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    ]
    out = sample_gradient([0.5], stops, vmin=0.0, vmax=1.0)
    np.testing.assert_allclose(out[0], stops[1], atol=1e-9)


def test_sample_gradient_empty_values_returns_empty() -> None:
    stops = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)]
    out = sample_gradient([], stops)
    assert out.shape == (0, 4)


def test_sample_gradient_rejects_too_few_stops() -> None:
    with pytest.raises(ValueError, match="at least two"):
        sample_gradient([0.0], [(0.0, 0.0, 0.0, 1.0)])


def test_sample_gradient_rejects_malformed_stop() -> None:
    with pytest.raises((ValueError, TypeError)):
        sample_gradient([0.0], [(0.0, 0.0), (1.0, 1.0, 1.0, 1.0)])


def test_sample_gradient_rejects_non_rgba_width() -> None:
    # Well-formed 2D array but only 3 channels per stop.
    with pytest.raises(ValueError, match="RGBA 4-tuple"):
        sample_gradient([0.0], [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])


def test_sample_gradient_non_finite_values_are_clamped() -> None:
    stops = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)]
    out = sample_gradient([np.nan, np.inf, -np.inf], stops, vmin=0.0, vmax=1.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# terrain_colors
# ---------------------------------------------------------------------------


def test_terrain_colors_shape_and_range() -> None:
    z = np.linspace(0.0, 1.0, 16)
    out = terrain_colors(z)
    assert out.shape == (16, 4)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_terrain_colors_low_is_greener_than_high() -> None:
    out = terrain_colors([0.0, 1.0], vmin=0.0, vmax=1.0)
    low, high = out[0], out[1]
    # Low ground is lush green: green dominates red.
    assert low[1] > low[0]
    # High ground tans out: red rises relative to the low sample.
    assert high[0] > low[0]


def test_terrain_colors_constant_elevation_is_safe() -> None:
    out = terrain_colors(np.zeros(5))
    assert out.shape == (5, 4)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# speed_colors
# ---------------------------------------------------------------------------


def test_speed_colors_cool_to_warm_monotonic_red() -> None:
    out = speed_colors([0.0, 1.0, 2.0, 3.0], vmin=0.0, vmax=3.0)
    assert out.shape == (4, 4)
    # Faster -> warmer: red channel increases with speed.
    assert np.all(np.diff(out[:, 0]) >= -1e-9)
    assert out[-1, 0] > out[0, 0]


def test_speed_colors_alpha_opaque() -> None:
    out = speed_colors([0.0, 5.0])
    assert np.allclose(out[:, 3], 1.0)


# ---------------------------------------------------------------------------
# roll_mode_colors
# ---------------------------------------------------------------------------


def test_roll_mode_palette_has_three_modes() -> None:
    assert set(ROLL_MODE_RGBA) == {"sliding", "rolling", "stopped"}
    for rgba in ROLL_MODE_RGBA.values():
        assert len(rgba) == 4
        assert all(0.0 <= c <= 1.0 for c in rgba)


def test_roll_mode_colors_maps_known_modes() -> None:
    out = roll_mode_colors(["sliding", "rolling", "stopped"])
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[0], ROLL_MODE_RGBA["sliding"])
    np.testing.assert_allclose(out[1], ROLL_MODE_RGBA["rolling"])
    np.testing.assert_allclose(out[2], ROLL_MODE_RGBA["stopped"])


def test_roll_mode_colors_is_case_insensitive() -> None:
    out = roll_mode_colors(["SLIDING", "Rolling"])
    np.testing.assert_allclose(out[0], ROLL_MODE_RGBA["sliding"])
    np.testing.assert_allclose(out[1], ROLL_MODE_RGBA["rolling"])


def test_roll_mode_colors_accepts_enum_like_repr() -> None:
    # RollMode enum stringifies as "RollMode.SLIDING"; the helper should
    # extract the trailing member name.
    out = roll_mode_colors(["RollMode.SLIDING", "RollMode.ROLLING"])
    np.testing.assert_allclose(out[0], ROLL_MODE_RGBA["sliding"])
    np.testing.assert_allclose(out[1], ROLL_MODE_RGBA["rolling"])


def test_roll_mode_colors_unknown_falls_back_to_stopped() -> None:
    out = roll_mode_colors(["banana"])
    np.testing.assert_allclose(out[0], ROLL_MODE_RGBA["stopped"])


def test_roll_mode_colors_empty() -> None:
    out = roll_mode_colors([])
    assert out.shape == (0, 4)
