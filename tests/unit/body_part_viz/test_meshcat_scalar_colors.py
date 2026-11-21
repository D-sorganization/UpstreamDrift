"""MeshCat scalar colour bindings smoke test (#15396)."""

from __future__ import annotations

import pytest

from src.shared.python.body_part_viz.meshcat_scalar_colors import MeshcatScalarColors
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale

pytestmark = pytest.mark.unit


def test_meshcat_scalar_colors_apply_and_restore() -> None:
    calls: list[tuple[str, str, list[float]]] = []

    def setter(path: str, prop: str, value: list[float]) -> None:
        calls.append((path, prop, list(value)))

    adapter = MeshcatScalarColors(
        setter,
        {"rod": {"/a/<object>": (1.0, 1.0, 0.0, 1.0)}},
    )
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=10.0,
    )
    adapter.apply({"rod": 5.0}, scale)
    assert calls and calls[-1][0] == "/a/<object>"
    assert calls[-1][2][3] == 1.0
    adapter.apply({"rod": None}, scale)
    assert calls[-1][2] == [1.0, 1.0, 0.0, 1.0]


def test_meshcat_scalar_colors_disabled_restores_base() -> None:
    calls: list[list[float]] = []

    def setter(path: str, prop: str, value: list[float]) -> None:
        calls.append(list(value))

    base = [0.2, 0.4, 0.9, 1.0]
    adapter = MeshcatScalarColors(setter, {"rod": {"/a/<object>": base}})
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=10.0,
    )
    adapter.apply({"rod": 8.0}, scale)
    assert calls[-1][:3] != base[:3]
    from dataclasses import replace

    off = replace(scale, enabled=False)
    adapter.apply({"rod": 8.0}, off)
    assert calls[-1] == base


def test_meshcat_force_still_wins_after_ke_restore_path() -> None:
    from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale
    from src.shared.python.body_part_viz.meshcat_force_colors import MeshcatForceColors

    base = [0.8, 0.2, 0.2, 1.0]
    bindings = {"rod": {"/a/<object>": base}}
    colors: list[list[float]] = []

    def setter(path: str, prop: str, value: list[float]) -> None:
        colors.append(list(value))

    ke = MeshcatScalarColors(setter, bindings)
    force = MeshcatForceColors(setter, bindings)
    ke_scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
    )
    force_scale = ForceColorScale(
        enabled=True, tension_limit_n=1, compression_limit_n=1
    )
    frame = AxialLoadFrame(0.0, {"rod": 10.0}, "t")
    ke.apply({"rod": 50.0}, ke_scale)
    force.apply(frame, force_scale)
    force_rgb = tuple(int(round(255 * c)) for c in colors[-1][:3])
    ke.apply({"rod": 50.0}, ke_scale)
    force.apply(frame, force_scale)
    both_rgb = tuple(int(round(255 * c)) for c in colors[-1][:3])
    assert both_rgb == force_rgb