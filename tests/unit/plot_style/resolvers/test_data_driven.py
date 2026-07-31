"""Unit tests for the data-driven color resolver."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.shared.python.plot_style.channels import DataChannel
from src.shared.python.plot_style.colormaps import ColormapId
from src.shared.python.plot_style.colors import (
    DataDrivenColor as DataDrivenColorScale,
)
from src.shared.python.plot_style.contracts import ColorResolver
from src.shared.python.plot_style.resolvers import RESOLVER_REGISTRY
from src.shared.python.plot_style.resolvers.data_driven import DataDrivenColor

# Tolerance for "identical" comparisons across paths. The per-frame
# path samples the LUT by ``int(round(...))`` index; the bulk path uses
# the same rounding. Therefore agreement is exact at LUT resolution.
_RTOL = 1e-9
_ATOL = 1e-12


def _make_per_frame_channel(n_frames: int = 64) -> DataChannel:
    return DataChannel(
        name="speed",
        values=np.linspace(0.0, 10.0, n_frames, dtype=np.float64),
        unit="m/s",
    )


def _make_per_marker_channel(n_frames: int = 32, n_markers: int = 4) -> DataChannel:
    rng = np.random.default_rng(seed=0xC0FFEE)
    values = rng.uniform(0.0, 5.0, size=(n_frames, n_markers))
    return DataChannel(name="velocity", values=values, unit="m/s")


def _make_scale(channel: DataChannel, **kwargs: object) -> DataDrivenColorScale:
    defaults: dict[str, object] = {"colormap": ColormapId.VIRIDIS}
    defaults.update(kwargs)
    return DataDrivenColorScale(channel=channel, **defaults)  # type: ignore[arg-type]


# ---------- protocol & construction ------------------------------------


def test_data_driven_protocol_compliance() -> None:
    scale = _make_scale(_make_per_frame_channel())
    resolver = DataDrivenColor(scale)
    assert isinstance(resolver, ColorResolver)


def test_constructor_rejects_wrong_scale_type() -> None:
    with pytest.raises(TypeError, match="DataDrivenColor"):
        DataDrivenColor("not a scale")  # type: ignore[arg-type]


def test_lut_is_256_entries() -> None:
    resolver = DataDrivenColor(_make_scale(_make_per_frame_channel()))
    assert resolver.lut.shape == (256, 4)


def test_lut_is_read_only() -> None:
    resolver = DataDrivenColor(_make_scale(_make_per_frame_channel()))
    with pytest.raises(ValueError):
        resolver.lut[0] = (0.0, 0.0, 0.0, 0.0)


# ---------- per-frame vs bulk parity -----------------------------------


def test_per_frame_and_bulk_agree_1d() -> None:
    channel = _make_per_frame_channel(n_frames=32)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0)
    resolver = DataDrivenColor(scale)

    bulk = resolver.resolve_array(scale, n_frames=channel.n_frames)
    assert bulk.shape == (channel.n_frames, 4)

    for frame_idx in range(channel.n_frames):
        per_frame = np.asarray(resolver.resolve_one(scale, frame_idx))
        np.testing.assert_allclose(bulk[frame_idx], per_frame, rtol=_RTOL, atol=_ATOL)


def test_per_frame_marker_and_bulk_agree_2d() -> None:
    channel = _make_per_marker_channel(n_frames=16, n_markers=5)
    scale = _make_scale(channel, vmin=0.0, vmax=5.0)
    resolver = DataDrivenColor(scale)

    bulk = resolver.resolve_array(
        scale, n_frames=channel.n_frames, n_markers=channel.n_markers
    )
    assert bulk.shape == (16, 5, 4)
    assert channel.n_markers is not None
    for f in range(channel.n_frames):
        for m in range(channel.n_markers):
            per_point = np.asarray(resolver.resolve_one(scale, f, m))
            np.testing.assert_allclose(bulk[f, m], per_point, rtol=_RTOL, atol=_ATOL)


def test_resolve_alias_matches_resolve_one() -> None:
    channel = _make_per_frame_channel(n_frames=8)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0)
    resolver = DataDrivenColor(scale)
    for frame_idx in range(channel.n_frames):
        a = resolver.resolve(scale, frame_idx)
        b = resolver.resolve_one(scale, frame_idx)
        assert a == b


# ---------- NaN handling -----------------------------------------------


def test_nan_values_yield_nan_color_per_frame() -> None:
    values = np.array([0.0, np.nan, 5.0, np.nan, 10.0], dtype=np.float64)
    channel = DataChannel(name="speed", values=values)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0, nan_color="#888888")
    resolver = DataDrivenColor(scale)

    nan_rgba = resolver.nan_rgba
    assert resolver.resolve_one(scale, 1) == nan_rgba
    assert resolver.resolve_one(scale, 3) == nan_rgba
    # finite frames should not equal nan_rgba
    assert resolver.resolve_one(scale, 0) != nan_rgba


def test_nan_values_yield_nan_color_bulk_1d() -> None:
    values = np.array([0.0, np.nan, 5.0, np.nan, 10.0], dtype=np.float64)
    channel = DataChannel(name="speed", values=values)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0, nan_color="#888888")
    resolver = DataDrivenColor(scale)

    bulk = resolver.resolve_array(scale, n_frames=values.shape[0])
    nan_rgba = np.asarray(resolver.nan_rgba)
    np.testing.assert_allclose(bulk[1], nan_rgba)
    np.testing.assert_allclose(bulk[3], nan_rgba)
    # The finite entry should NOT match nan colour bit-exactly.
    assert not np.allclose(bulk[0], nan_rgba)


def test_nan_values_yield_nan_color_bulk_2d() -> None:
    values = np.array([[0.0, 5.0], [np.nan, 10.0], [2.0, np.nan]], dtype=np.float64)
    channel = DataChannel(name="speed", values=values)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0, nan_color="#888888")
    resolver = DataDrivenColor(scale)

    bulk = resolver.resolve_array(scale, n_frames=3, n_markers=2)
    nan_rgba = np.asarray(resolver.nan_rgba)
    np.testing.assert_allclose(bulk[1, 0], nan_rgba)
    np.testing.assert_allclose(bulk[2, 1], nan_rgba)


def test_degenerate_bounds_yield_nan_color() -> None:
    # auto-range over an empty channel collapses to NaN bounds.
    channel = DataChannel(name="speed", values=np.full(4, np.nan, dtype=np.float64))
    scale = _make_scale(channel, nan_color="#888888")
    resolver = DataDrivenColor(scale)
    nan_rgba = resolver.nan_rgba
    assert resolver.resolve_one(scale, 0) == nan_rgba
    bulk = resolver.resolve_array(scale, n_frames=4)
    np.testing.assert_allclose(bulk, np.tile(np.asarray(nan_rgba), (4, 1)))


# ---------- bounds & shape edge cases ----------------------------------


def test_resolve_array_zero_frames() -> None:
    channel = _make_per_frame_channel(n_frames=8)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0)
    resolver = DataDrivenColor(scale)
    arr = resolver.resolve_array(scale, n_frames=0)
    assert arr.shape == (0, 4)


def test_resolve_array_pads_with_nan_when_overshooting() -> None:
    channel = _make_per_frame_channel(n_frames=4)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0)
    resolver = DataDrivenColor(scale)
    arr = resolver.resolve_array(scale, n_frames=6)
    nan_rgba = np.asarray(resolver.nan_rgba)
    np.testing.assert_allclose(arr[4], nan_rgba)
    np.testing.assert_allclose(arr[5], nan_rgba)


def test_data_driven_resolve_array_rejects_negative_frames() -> None:
    resolver = DataDrivenColor(_make_scale(_make_per_frame_channel()))
    scale = _make_scale(_make_per_frame_channel(), vmin=0.0, vmax=10.0)
    with pytest.raises(ValueError, match="non-negative"):
        resolver.resolve_array(scale, n_frames=-1)


def test_data_driven_resolve_array_rejects_negative_markers() -> None:
    resolver = DataDrivenColor(_make_scale(_make_per_marker_channel()))
    scale = _make_scale(_make_per_marker_channel(), vmin=0.0, vmax=5.0)
    with pytest.raises(ValueError, match="non-negative"):
        resolver.resolve_array(scale, n_frames=8, n_markers=-1)


def test_per_marker_request_against_1d_channel_broadcasts() -> None:
    """A 1-D channel served per-(frame, marker) broadcasts across markers."""
    channel = _make_per_frame_channel(n_frames=8)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0)
    resolver = DataDrivenColor(scale)
    arr = resolver.resolve_array(scale, n_frames=8, n_markers=3)
    assert arr.shape == (8, 3, 4)
    for marker_idx in range(3):
        np.testing.assert_allclose(arr[:, marker_idx], arr[:, 0])


def test_per_frame_against_2d_channel_uses_marker_mean() -> None:
    """When a 2-D channel is queried per-frame, the slab averages markers."""
    values = np.array([[0.0, 10.0], [5.0, 5.0], [np.nan, 8.0]], dtype=np.float64)
    channel = DataChannel(name="speed", values=values)
    scale = _make_scale(channel, vmin=0.0, vmax=10.0)
    resolver = DataDrivenColor(scale)
    arr = resolver.resolve_array(scale, n_frames=3)
    assert arr.shape == (3, 4)
    # Mean of row 0 = 5.0 -> normalised 0.5
    expected_mid = np.asarray(resolver.resolve_one(scale, 1))  # 5.0
    np.testing.assert_allclose(arr[0], expected_mid, atol=1e-9)


# ---------- registry dispatch ------------------------------------------


def test_data_driven_registry_dispatch() -> None:
    assert RESOLVER_REGISTRY[DataDrivenColorScale] is DataDrivenColor


# ---------- perf microbench --------------------------------------------


@pytest.mark.perf
def test_resolve_array_meets_60fps_target() -> None:
    """1000 frames x 32 markers must clear 60 fps via the LUT bulk path."""
    n_frames = 1000
    n_markers = 32
    rng = np.random.default_rng(seed=0xBEEF)
    values = rng.uniform(0.0, 1.0, size=(n_frames, n_markers)).astype(np.float64)
    channel = DataChannel(name="speed", values=values)
    scale = _make_scale(channel, vmin=0.0, vmax=1.0)
    resolver = DataDrivenColor(scale)

    # Warm-up — JIT-style warm caches and avoid first-call import overhead.
    resolver.resolve_array(scale, n_frames=n_frames, n_markers=n_markers)

    # Time a small batch of full-array resolves and convert to fps.
    n_iters = 5
    start = time.perf_counter()
    for _ in range(n_iters):
        resolver.resolve_array(scale, n_frames=n_frames, n_markers=n_markers)
    elapsed = time.perf_counter() - start
    # Each call resolves all 1000 frames; fps = (n_iters * n_frames) / elapsed.
    frames_per_sec = (n_iters * n_frames) / elapsed
    assert (
        frames_per_sec > 60
    ), f"resolve_array fps {frames_per_sec:.0f} below 60 fps target"
