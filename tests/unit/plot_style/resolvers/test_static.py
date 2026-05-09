"""Unit tests for the static-color resolver."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.plot_style.colors import StaticColor as StaticColorScale
from src.shared.python.plot_style.contracts import ColorResolver
from src.shared.python.plot_style.resolvers import RESOLVER_REGISTRY
from src.shared.python.plot_style.resolvers.static import StaticColor


def test_protocol_compliance() -> None:
    resolver = StaticColor("#ff0000")
    assert isinstance(resolver, ColorResolver)


def test_hex_round_trip_red() -> None:
    resolver = StaticColor("#ff0000")
    rgba = resolver.resolve_one(StaticColorScale("#ff0000"), frame_idx=0)
    assert rgba == pytest.approx((1.0, 0.0, 0.0, 1.0))
    assert resolver.hex_value == "#ff0000"


def test_hex_round_trip_named_color() -> None:
    resolver = StaticColor("blue")
    rgba = resolver.resolve_one(StaticColorScale("blue"), frame_idx=0)
    # matplotlib 'blue' is (0, 0, 1, 1)
    assert rgba == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_resolve_one_ignores_indices() -> None:
    resolver = StaticColor("#00ff00")
    a = resolver.resolve_one(StaticColorScale("#00ff00"), frame_idx=0)
    b = resolver.resolve_one(StaticColorScale("#00ff00"), frame_idx=42, marker_idx=7)
    assert a == b


def test_bad_hex_raises() -> None:
    with pytest.raises(ValueError, match="parseable"):
        StaticColor("not_a_color")


def test_empty_hex_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StaticColor("")


def test_non_string_hex_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StaticColor(123)  # type: ignore[arg-type]


def test_resolve_array_per_frame_shape() -> None:
    resolver = StaticColor("#ff8800")
    arr = resolver.resolve_array(StaticColorScale("#ff8800"), n_frames=10)
    assert arr.shape == (10, 4)
    assert np.allclose(arr, np.tile(resolver.rgba, (10, 1)))


def test_resolve_array_per_marker_shape() -> None:
    resolver = StaticColor("#ff8800")
    arr = resolver.resolve_array(StaticColorScale("#ff8800"), n_frames=5, n_markers=3)
    assert arr.shape == (5, 3, 4)
    assert np.allclose(arr[0, 0], resolver.rgba)
    assert np.allclose(arr[4, 2], resolver.rgba)


def test_resolve_array_zero_frames() -> None:
    resolver = StaticColor("red")
    arr = resolver.resolve_array(StaticColorScale("red"), n_frames=0)
    assert arr.shape == (0, 4)


def test_resolve_array_rejects_negative_frames() -> None:
    resolver = StaticColor("red")
    with pytest.raises(ValueError, match="non-negative"):
        resolver.resolve_array(StaticColorScale("red"), n_frames=-1)


def test_resolve_array_rejects_negative_markers() -> None:
    resolver = StaticColor("red")
    with pytest.raises(ValueError, match="non-negative"):
        resolver.resolve_array(StaticColorScale("red"), n_frames=5, n_markers=-1)


def test_from_scale_round_trip() -> None:
    scale = StaticColorScale("#123456")
    resolver = StaticColor.from_scale(scale)
    assert resolver.hex_value == "#123456"


def test_from_scale_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="StaticColor"):
        StaticColor.from_scale("blue")  # type: ignore[arg-type]


def test_registry_dispatch() -> None:
    assert RESOLVER_REGISTRY[StaticColorScale] is StaticColor


def test_rgba_property_caches() -> None:
    resolver = StaticColor("#abcdef")
    a = resolver.rgba
    b = resolver.rgba
    assert a == b
