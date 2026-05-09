"""Unit tests for the palette-color resolver."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.plot_style.colormaps import CustomColormap
from src.shared.python.plot_style.colors import PaletteColor as PaletteColorScale
from src.shared.python.plot_style.contracts import ColorResolver
from src.shared.python.plot_style.registry import (
    register_custom_colormap,
    unregister_custom_colormap,
)
from src.shared.python.plot_style.resolvers import RESOLVER_REGISTRY
from src.shared.python.plot_style.resolvers.palette import PaletteColor


def test_protocol_compliance() -> None:
    resolver = PaletteColor("tab10", 0)
    assert isinstance(resolver, ColorResolver)


def test_index_in_range_returns_valid_rgba() -> None:
    resolver = PaletteColor("tab10", 3)
    rgba = resolver.resolve_one(
        PaletteColorScale(palette_name="tab10", palette_index=3), frame_idx=0
    )
    assert len(rgba) == 4
    for component in rgba:
        assert 0.0 <= component <= 1.0
    assert rgba[3] == pytest.approx(1.0)


def test_index_zero_is_first_palette_entry() -> None:
    resolver = PaletteColor("tab10", 0)
    assert resolver.palette_index == 0
    assert resolver.palette_size == 10


def test_oob_index_raises_index_error() -> None:
    with pytest.raises(IndexError, match="out of range"):
        PaletteColor("tab10", 99)


def test_oob_index_message_includes_palette_size() -> None:
    with pytest.raises(IndexError, match=r"size 10"):
        PaletteColor("tab10", 10)


def test_negative_index_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PaletteColor("tab10", -1)


def test_unknown_palette_raises_key_error() -> None:
    with pytest.raises(KeyError, match="not a matplotlib"):
        PaletteColor("not_a_palette", 0)


def test_empty_palette_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PaletteColor("", 0)


def test_non_int_index_raises() -> None:
    with pytest.raises(TypeError, match="must be int"):
        PaletteColor("tab10", 3.0)  # type: ignore[arg-type]


def test_bool_index_rejected() -> None:
    with pytest.raises(TypeError, match="must be int"):
        PaletteColor("tab10", True)  # type: ignore[arg-type]


def test_distinct_indices_yield_distinct_colors() -> None:
    a = PaletteColor("tab10", 0).rgba
    b = PaletteColor("tab10", 1).rgba
    assert a != b


def test_resolve_array_per_frame() -> None:
    resolver = PaletteColor("tab10", 2)
    arr = resolver.resolve_array(
        PaletteColorScale(palette_name="tab10", palette_index=2), n_frames=4
    )
    assert arr.shape == (4, 4)
    expected = np.asarray(resolver.rgba)
    for row in arr:
        assert np.allclose(row, expected)


def test_resolve_array_per_marker() -> None:
    resolver = PaletteColor("tab10", 2)
    arr = resolver.resolve_array(
        PaletteColorScale(palette_name="tab10", palette_index=2),
        n_frames=4,
        n_markers=3,
    )
    assert arr.shape == (4, 3, 4)
    expected = np.asarray(resolver.rgba)
    assert np.allclose(arr[0, 0], expected)


def test_resolve_array_rejects_negative_frames() -> None:
    resolver = PaletteColor("tab10", 0)
    with pytest.raises(ValueError, match="non-negative"):
        resolver.resolve_array(
            PaletteColorScale(palette_name="tab10", palette_index=0), n_frames=-1
        )


def test_resolve_array_rejects_negative_markers() -> None:
    resolver = PaletteColor("tab10", 0)
    with pytest.raises(ValueError, match="non-negative"):
        resolver.resolve_array(
            PaletteColorScale(palette_name="tab10", palette_index=0),
            n_frames=5,
            n_markers=-1,
        )


def test_from_scale_normalises_wrapping_index() -> None:
    # Scale permits modulo wrap; resolver normalises so it stays in range.
    scale = PaletteColorScale(palette_name="tab10", palette_index=12)
    resolver = PaletteColor.from_scale(scale)
    assert resolver.palette_index == 2


def test_from_scale_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="PaletteColor"):
        PaletteColor.from_scale("tab10")  # type: ignore[arg-type]


def test_custom_colormap_lookup() -> None:
    cmap = CustomColormap(
        name="ud_test_palette_resolver",
        stops=((0.0, "#000000"), (1.0, "#ffffff")),
    )
    register_custom_colormap(cmap)
    try:
        resolver = PaletteColor("ud_test_palette_resolver", 0)
        rgba = resolver.rgba
        assert rgba[0] == pytest.approx(0.0, abs=1e-3)
    finally:
        unregister_custom_colormap("ud_test_palette_resolver")


def test_registry_dispatch() -> None:
    assert RESOLVER_REGISTRY[PaletteColorScale] is PaletteColor
