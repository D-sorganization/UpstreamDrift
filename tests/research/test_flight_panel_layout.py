"""Flight-panel layout keeps conventional axes and the nominal reference."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_shared_flight_layout_preserves_cloud_nominal_and_axes():
    from matplotlib.figure import Figure
    from docs.research.shot_pattern_analysis.phase2.presentation_v2.flight_panel import (
        FlightPanelLayout,
        plot_flight_panel,
    )

    ax = Figure().subplots()
    path = np.array([[0.0, 0.0], [-2.0, 50.0], [-1.0, 100.0]])
    layout = FlightPanelLayout(100.0, 10.0, 110.0, 1, True, {"Draw": "#c46e1f"})
    plot_flight_panel(ax, "Draw", [path], 0, path, layout)
    assert ax.get_xlabel() == "Lateral Position (m; Right Positive)"
    assert ax.get_ylabel() == "Downrange Distance (m)"
    assert ax.get_xlim() == (-10.0, 10.0)
    assert ax.get_ylim() == (0.0, 110.0)
    assert len(ax.lines) == 4
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), path[:, 0])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), path[:, 1])
    assert ax.lines[0].get_rasterized()
    assert ax.lines[1].get_linestyle() == "--"
