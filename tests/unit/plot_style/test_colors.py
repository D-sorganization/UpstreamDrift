"""Unit tests for :class:`StaticColor`, :class:`PaletteColor`, :class:`DataDrivenColor`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.plot_style import (
    ColormapId,
    DataChannel,
    DataDrivenColor,
    PaletteColor,
    StaticColor,
)

# ---------- StaticColor -------------------------------------------------


def test_static_color_happy_path() -> None:
    color = StaticColor(hex_value="#ff0000")
    rgba = color.resolve(0)
    assert len(rgba) == 4
    for component in rgba:
        assert 0.0 <= component <= 1.0
    assert rgba[0] == pytest.approx(1.0)
    assert rgba[3] == pytest.approx(1.0)


def test_static_color_resolve_ignores_indices() -> None:
    color = StaticColor(hex_value="blue")
    assert color.resolve(0) == color.resolve(99, marker_idx=42)


def test_static_color_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StaticColor(hex_value="")


def test_static_color_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StaticColor(hex_value=123)  # type: ignore[arg-type]


def test_static_color_rejects_non_parseable() -> None:
    with pytest.raises(ValueError, match="parseable"):
        StaticColor(hex_value="not_a_color")


# ---------- PaletteColor ------------------------------------------------


def test_palette_color_happy_path() -> None:
    color = PaletteColor(palette_name="tab10", palette_index=3)
    rgba = color.resolve(0)
    assert len(rgba) == 4
    for component in rgba:
        assert 0.0 <= component <= 1.0


def test_palette_color_index_wraps() -> None:
    a = PaletteColor(palette_name="tab10", palette_index=2)
    b = PaletteColor(palette_name="tab10", palette_index=12)
    assert a.resolve(0) == b.resolve(0)


def test_palette_color_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PaletteColor(palette_name="", palette_index=0)


def test_palette_color_rejects_unknown_palette() -> None:
    with pytest.raises(ValueError, match="not a registered"):
        PaletteColor(palette_name="not_a_real_palette_xyz", palette_index=0)


def test_palette_color_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PaletteColor(palette_name="tab10", palette_index=-1)


def test_palette_color_rejects_non_int_index() -> None:
    with pytest.raises(TypeError, match="int"):
        PaletteColor(palette_name="tab10", palette_index=1.5)  # type: ignore[arg-type]


# ---------- DataDrivenColor --------------------------------------------


def _make_channel_1d() -> DataChannel:
    return DataChannel.from_array("v", np.array([0.0, 5.0, 10.0]), unit="m/s")


def _make_channel_2d() -> DataChannel:
    values = np.array([[0.0, 5.0], [5.0, 10.0]])
    return DataChannel.from_array("v", values)


def test_data_driven_color_happy_path() -> None:
    channel = _make_channel_1d()
    color = DataDrivenColor(channel=channel, colormap=ColormapId.VIRIDIS)
    rgba = color.resolve(1)
    assert len(rgba) == 4
    for component in rgba:
        assert 0.0 <= component <= 1.0


def test_data_driven_color_explicit_bounds() -> None:
    channel = _make_channel_1d()
    color = DataDrivenColor(
        channel=channel,
        colormap=ColormapId.PLASMA,
        vmin=0.0,
        vmax=10.0,
    )
    low = color.resolve(0)
    high = color.resolve(2)
    assert low != high


def test_data_driven_color_with_2d_channel() -> None:
    channel = _make_channel_2d()
    color = DataDrivenColor(channel=channel, colormap=ColormapId.VIRIDIS)
    rgba = color.resolve(0, marker_idx=1)
    assert len(rgba) == 4


def test_data_driven_color_uses_semantic_alias() -> None:
    channel = _make_channel_1d()
    color = DataDrivenColor(
        channel=channel,
        colormap=ColormapId.VELOCITY,
        vmin=0.0,
        vmax=10.0,
    )
    rgba = color.resolve(1)
    assert len(rgba) == 4


def test_data_driven_color_nan_returns_nan_color() -> None:
    channel = DataChannel.from_array("v", np.array([np.nan, 1.0]))
    color = DataDrivenColor(
        channel=channel,
        colormap=ColormapId.VIRIDIS,
        nan_color="#123456",
    )
    rgba = color.resolve(0)
    # Should equal the nan color
    expected = StaticColor(hex_value="#123456").resolve(0)
    assert rgba == expected


def test_data_driven_color_oob_index_returns_nan_color() -> None:
    channel = _make_channel_1d()
    color = DataDrivenColor(channel=channel, colormap=ColormapId.VIRIDIS)
    rgba = color.resolve(999)
    expected = StaticColor(hex_value="#888888").resolve(0)
    assert rgba == expected


def test_data_driven_color_degenerate_range_returns_nan_color() -> None:
    # Single finite value -> auto_range yields equal min/max.
    channel = DataChannel.from_array("v", np.array([1.0, 1.0, 1.0]))
    color = DataDrivenColor(channel=channel, colormap=ColormapId.VIRIDIS)
    rgba = color.resolve(0)
    expected = StaticColor(hex_value="#888888").resolve(0)
    assert rgba == expected


def test_data_driven_color_clamps_out_of_bounds_value() -> None:
    channel = _make_channel_1d()
    color = DataDrivenColor(
        channel=channel,
        colormap=ColormapId.VIRIDIS,
        vmin=0.0,
        vmax=2.0,
    )
    # frame 2 has value 10, clamps to 1.0 -> top of colormap
    rgba_high = color.resolve(2)
    rgba_top = color.resolve(2)
    assert rgba_high == rgba_top  # deterministic


def test_data_driven_color_rejects_non_channel() -> None:
    with pytest.raises(TypeError, match="DataChannel"):
        DataDrivenColor(channel="not a channel", colormap=ColormapId.VIRIDIS)  # type: ignore[arg-type]


def test_data_driven_color_rejects_non_colormap_id() -> None:
    channel = _make_channel_1d()
    with pytest.raises(TypeError, match="ColormapId"):
        DataDrivenColor(channel=channel, colormap="viridis")  # type: ignore[arg-type]


def test_data_driven_color_rejects_non_finite_vmin() -> None:
    channel = _make_channel_1d()
    with pytest.raises(ValueError, match="finite"):
        DataDrivenColor(
            channel=channel,
            colormap=ColormapId.VIRIDIS,
            vmin=math.inf,
        )


def test_data_driven_color_rejects_non_numeric_vmin() -> None:
    channel = _make_channel_1d()
    with pytest.raises(TypeError, match="numeric"):
        DataDrivenColor(
            channel=channel,
            colormap=ColormapId.VIRIDIS,
            vmin="zero",  # type: ignore[arg-type]
        )


def test_data_driven_color_rejects_inverted_bounds() -> None:
    channel = _make_channel_1d()
    with pytest.raises(ValueError, match="strictly greater"):
        DataDrivenColor(
            channel=channel,
            colormap=ColormapId.VIRIDIS,
            vmin=5.0,
            vmax=1.0,
        )


def test_data_driven_color_rejects_empty_nan_color() -> None:
    channel = _make_channel_1d()
    with pytest.raises(ValueError, match="non-empty"):
        DataDrivenColor(
            channel=channel,
            colormap=ColormapId.VIRIDIS,
            nan_color="",
        )


def test_data_driven_color_rejects_unparseable_nan_color() -> None:
    channel = _make_channel_1d()
    with pytest.raises(ValueError, match="parseable"):
        DataDrivenColor(
            channel=channel,
            colormap=ColormapId.VIRIDIS,
            nan_color="bogus_color",
        )


def test_resolve_returns_components_in_unit_interval_for_all_variants() -> None:
    channel = _make_channel_1d()
    scales = (
        StaticColor("#ff00ff"),
        PaletteColor(palette_name="Set2", palette_index=0),
        DataDrivenColor(channel=channel, colormap=ColormapId.VIRIDIS),
    )
    for scale in scales:
        rgba = scale.resolve(0)
        assert len(rgba) == 4
        for component in rgba:
            assert 0.0 <= component <= 1.0
