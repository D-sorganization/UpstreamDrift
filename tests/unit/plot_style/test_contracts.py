"""Unit tests for the runtime-checkable Protocols."""

from __future__ import annotations

import numpy as np

from src.shared.python.plot_style import (
    ColorResolver,
    ColorScale,
    DataChannel,
    DataDrivenColor,
    MarkerRenderer,
    MarkerStyle,
    PaletteColor,
    StaticColor,
)
from src.shared.python.plot_style import ColormapId


class _StubRenderer:
    """Minimal MarkerRenderer-conforming stub for runtime checks."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self._next_handle = 0

    def add_markers(
        self,
        positions: np.ndarray,
        style: MarkerStyle,
        label: str = "",
    ) -> str:
        self._next_handle += 1
        handle = f"h{self._next_handle}"
        self.calls.append(("add_markers", (positions, style, label)))
        return handle

    def update_frame(self, handle: str, frame_idx: int) -> None:
        self.calls.append(("update_frame", (handle, frame_idx)))

    def update_style(self, handle: str, style: MarkerStyle) -> None:
        self.calls.append(("update_style", (handle, style)))

    def set_visible(self, handle: str, visible: bool) -> None:
        self.calls.append(("set_visible", (handle, visible)))

    def remove(self, handle: str) -> None:
        self.calls.append(("remove", (handle,)))


class _StubResolver:
    """Minimal ColorResolver-conforming stub for runtime checks."""

    def resolve_one(
        self,
        scale: ColorScale,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> tuple[float, float, float, float]:
        rgba = scale.resolve(frame_idx, marker_idx)
        return (rgba[0], rgba[1], rgba[2], rgba[3])

    def resolve_array(
        self,
        scale: ColorScale,
        n_frames: int,
        n_markers: int | None = None,
    ) -> np.ndarray:
        if n_markers is None:
            out = np.zeros((n_frames, 4), dtype=float)
            for i in range(n_frames):
                out[i] = scale.resolve(i)
            return out
        out2 = np.zeros((n_frames, n_markers, 4), dtype=float)
        for i in range(n_frames):
            for j in range(n_markers):
                out2[i, j] = scale.resolve(i, j)
        return out2


def test_marker_renderer_runtime_checkable() -> None:
    stub = _StubRenderer()
    assert isinstance(stub, MarkerRenderer)


def test_color_resolver_runtime_checkable() -> None:
    stub = _StubResolver()
    assert isinstance(stub, ColorResolver)


def test_marker_renderer_stub_round_trip() -> None:
    stub = _StubRenderer()
    style = MarkerStyle()
    handle = stub.add_markers(np.zeros((1, 3)), style, label="x")
    stub.update_frame(handle, 0)
    stub.update_style(handle, style)
    stub.set_visible(handle, False)
    stub.remove(handle)
    methods = [name for (name, _) in stub.calls]
    assert methods == [
        "add_markers",
        "update_frame",
        "update_style",
        "set_visible",
        "remove",
    ]


def test_color_resolver_resolves_static() -> None:
    stub = _StubResolver()
    scale = StaticColor("#ff0000")
    rgba = stub.resolve_one(scale, 0)
    assert rgba[0] == 1.0


def test_color_resolver_resolves_palette_array() -> None:
    stub = _StubResolver()
    scale = PaletteColor(palette_name="tab10", palette_index=0)
    arr = stub.resolve_array(scale, n_frames=3)
    assert arr.shape == (3, 4)


def test_color_resolver_resolves_data_driven_2d_array() -> None:
    stub = _StubResolver()
    channel = DataChannel.from_array("v", np.array([[0.0, 1.0], [2.0, 3.0]]))
    scale = DataDrivenColor(channel=channel, colormap=ColormapId.VIRIDIS)
    arr = stub.resolve_array(scale, n_frames=2, n_markers=2)
    assert arr.shape == (2, 2, 4)


def test_non_conforming_object_is_not_renderer() -> None:
    class _NotARenderer:
        pass

    assert not isinstance(_NotARenderer(), MarkerRenderer)


def test_non_conforming_object_is_not_resolver() -> None:
    class _NotAResolver:
        pass

    assert not isinstance(_NotAResolver(), ColorResolver)
