"""MeshCat overlay pass isolation via SimRenderingMixin (#15399)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
    meshcat_adapter as meshcat_adapter_mod,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_rendering_mixin import (
    SimRenderingMixin,
)
from src.shared.python.body_part_viz.meshcat_scalar_colors import MeshcatScalarColors
from src.shared.python.body_part_viz.segment_net_power import (
    default_net_power_color_scale,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale

pytestmark = pytest.mark.unit

_BASE = (0.2, 0.4, 0.6, 1.0)
_PATH = "/visuals/rod/<object>"
_BINDINGS = {"rod": {_PATH: _BASE}}
_KE_SAMPLE = 50.0
_NET_SAMPLE = 400.0


def _stub_adapter(
    record: list[tuple[str, str, tuple[float, ...]]],
) -> meshcat_adapter_mod.MuJoCoMeshcatAdapter:
    def setter(path: str, prop: str, value: list[float]) -> None:
        if prop == "color":
            record.append((path, prop, tuple(value)))

    adapter = meshcat_adapter_mod.MuJoCoMeshcatAdapter.__new__(
        meshcat_adapter_mod.MuJoCoMeshcatAdapter
    )
    adapter.vis = None
    adapter._force_colors = None
    adapter._stored_ke_colors = MeshcatScalarColors(setter, _BINDINGS)
    adapter._net_power_colors = MeshcatScalarColors(setter, _BINDINGS)
    adapter.update = lambda _data: None  # type: ignore[method-assign]
    adapter.update_force_colors = lambda *_a, **_k: None  # type: ignore[method-assign]
    adapter.draw_glyphs = lambda *_a, **_k: None  # type: ignore[method-assign]
    adapter.draw_induced_vectors = lambda *_a, **_k: None  # type: ignore[method-assign]
    adapter.draw_cf_vectors = lambda *_a, **_k: None  # type: ignore[method-assign]
    adapter.draw_ke_globes = lambda *_a, **_k: None  # type: ignore[method-assign]
    return adapter


def _ke_scale() -> ScalarColorScale:
    return ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
    )


def _net_scale(*, enabled: bool) -> ScalarColorScale:
    return replace(default_net_power_color_scale(), enabled=enabled)


def _last_rgb(
    record: list[tuple[str, str, tuple[float, ...]]],
) -> tuple[float, float, float]:
    for path, prop, value in reversed(record):
        if path == _PATH and prop == "color":
            return value[0], value[1], value[2]
    raise AssertionError(f"no colour recorded for {_PATH}")


def _host(adapter: Any) -> SimpleNamespace:
    return SimpleNamespace(
        meshcat_adapter=adapter,
        data=object(),
        stored_ke_color_scale=_ke_scale(),
        show_induced_vectors=False,
        show_cf_vectors=False,
        show_segment_ke_globes=False,
        _meshcat_draw_segment_ke_globes=lambda: None,
    )


def _assert_ke_colour(rgb: tuple[float, float, float]) -> None:
    expected = _ke_scale().shade_rgb(_KE_SAMPLE, _BASE[:3])
    assert expected is not None
    assert rgb == pytest.approx(expected)
    assert rgb != pytest.approx(_BASE[:3])


def _run_overlay_frames(
    host: Any,
    record: list[tuple[str, str, tuple[float, ...]]],
    *,
    frames: int,
    net_scale: ScalarColorScale,
) -> tuple[float, float, float]:
    ke_values = {"rod": _KE_SAMPLE}
    net_frame = SimpleNamespace(values={"rod": _NET_SAMPLE})
    for _ in range(frames):
        SimRenderingMixin._update_meshcat_frame_overlays(
            host, None, None, None, ke_values, net_frame, net_scale
        )
    return _last_rgb(record)


def test_meshcat_ke_survives_disabled_net_power_across_frames() -> None:
    record: list[tuple[str, str, tuple[float, ...]]] = []
    host = _host(_stub_adapter(record))
    rgb = _run_overlay_frames(
        host, record, frames=3, net_scale=_net_scale(enabled=False)
    )
    _assert_ke_colour(rgb)
    ke_rgb = _ke_scale().shade_rgb(_KE_SAMPLE, _BASE[:3])
    assert ke_rgb is not None
    for path, prop, value in record:
        if path == _PATH and prop == "color":
            assert value[:3] == pytest.approx(ke_rgb)


def test_meshcat_ke_wins_when_both_overlays_enabled() -> None:
    record: list[tuple[str, str, tuple[float, ...]]] = []
    host = _host(_stub_adapter(record))
    rgb = _run_overlay_frames(
        host, record, frames=2, net_scale=_net_scale(enabled=True)
    )
    _assert_ke_colour(rgb)
    net_only = _net_scale(enabled=True).shade_rgb(_NET_SAMPLE, _BASE[:3])
    assert net_only is not None
    assert rgb != pytest.approx(net_only)
