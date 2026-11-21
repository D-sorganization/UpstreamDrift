"""MeshCat scalar shading smoke (#15399)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.shared.python.body_part_viz.meshcat_segment_scalar_colors import (
    MeshcatSegmentScalarColors,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale
from src.shared.python.body_part_viz.segment_net_power import (
    default_net_power_color_scale,
)
from src.shared.python.body_part_viz.segment_scalar_frame import SegmentScalarFrame

pytestmark = pytest.mark.unit


def test_meshcat_scalar_overlay_restores_base_when_disabled():
    calls: list[tuple[str, list[float]]] = []

    def setter(path: str, prop: str, value: list[float]) -> None:
        calls.append((path, list(value)))

    bindings = {"arm": {"/visuals/arm/<object>": [0.2, 0.3, 0.4, 1.0]}}
    adapter = MeshcatSegmentScalarColors(setter, bindings)
    scale = replace(default_net_power_color_scale(100.0), enabled=True)
    frame = SegmentScalarFrame(
        time_s=0.0,
        values={"arm": 50.0},
        source="t",
        units="W",
        quantity_label="dE/dt (net power)",
    )
    adapter.apply(frame, scale)
    assert calls
    adapter.apply(
        None, ScalarColorScale.from_dict({**scale.to_dict(), "enabled": False})
    )
    assert calls[-1][1] == pytest.approx([0.2, 0.3, 0.4, 1.0])
