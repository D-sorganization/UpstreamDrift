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
