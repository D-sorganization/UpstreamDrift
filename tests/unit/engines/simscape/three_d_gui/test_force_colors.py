"""C3D host binds explicit loads, with native artist restoration and clock checks."""

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from PyQt6.QtWidgets import QApplication, QToolButton

from ._viewer_test_helpers import make_synthetic_model

pytestmark = pytest.mark.unit


def test_viewer_axial_loads_seek_clear_and_clock_contract():
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab
    from src.shared.python.body_part_viz import (
        BindingKind,
        ForceColorScale,
        MarkerBinding,
        SegmentLoadSeries,
        SegmentVizSpec,
        ShapeTheme,
    )

    app = QApplication.instance() or QApplication([])
    viewer = Viewer3DTab()
    model = make_synthetic_model(["a", "b"], n_frames=2)
    spec = SegmentVizSpec(
        MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b")),
        "line",
        {"length": 1.0},
        "between_two",
        ShapeTheme(color="#00aa00"),
    )
    viewer.set_user_segments((spec,))
    viewer.update_from_model(model)
    viewer.select_all_markers()
    loads = SegmentLoadSeries(
        model.point_time, {"arbitrary": (10, -10)}, "Declared section"
    )
    viewer.set_segment_axial_loads(loads, {"arbitrary": 0})
    assert viewer.findChild(QToolButton, "segment_force_colors") is not None
    viewer.set_axial_color_scale(
        ForceColorScale(enabled=True, tension_limit_n=1, compression_limit_n=1)
    )
    assert any(
        np.allclose(line.get_colors()[0][:3], to_rgba("blue")[:3])
        for line in viewer._ax.collections
        if hasattr(line, "get_colors")
    )
    viewer.set_frame(1)
    assert any(
        np.allclose(line.get_colors()[0][:3], to_rgba("red")[:3])
        for line in viewer._ax.collections
        if hasattr(line, "get_colors")
    )
    with pytest.raises(ValueError, match="times"):
        viewer.set_segment_axial_loads(
            SegmentLoadSeries((1, 2), {"arbitrary": (10, -10)}, "Wrong clock"),
            {"arbitrary": 0},
        )
    viewer.set_axial_color_scale(ForceColorScale())
    assert any(
        np.allclose(line.get_colors()[0][:3], to_rgba("#00aa00")[:3])
        for line in viewer._ax.collections
        if hasattr(line, "get_colors")
    )
    viewer.update_from_model(model)
    viewer.set_axial_color_scale(ForceColorScale(enabled=True))
    assert not any(
        np.allclose(line.get_colors()[0][:3], to_rgba("red")[:3])
        for line in viewer._ax.collections
        if hasattr(line, "get_colors")
    )
    viewer.close()
    app.processEvents()
