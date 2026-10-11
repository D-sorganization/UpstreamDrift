"""Axis contract for the saved-data presentation renderer."""

import pytest

try:
    import matplotlib  # noqa: F401
    import numpy as np
except (ImportError, OSError) as exc:
    pytest.skip(
        f"optional plotting dependency unavailable: {exc}", allow_module_level=True
    )

from docs.research.shot_pattern_analysis.phase2.presentation_v2 import render
from docs.research.shot_pattern_analysis.phase2.presentation_v2 import summary

pytestmark = pytest.mark.unit
orient_points = render.orient_points


def test_orient_points_maps_lateral_to_horizontal_and_downrange_to_vertical():
    saved = np.array([[178.9, -4.0], [180.0, 2.5]])
    display = orient_points(saved)
    assert display.tolist() == [[-4.0, 178.9], [2.5, 180.0]]
    assert saved.tolist() == [[178.9, -4.0], [180.0, 2.5]]


@pytest.mark.parametrize("points", [[[1.0]], [[float("nan"), 0.0]], []])
def test_orient_points_rejects_invalid_saved_coordinates(points):
    with pytest.raises(ValueError):
        orient_points(points)


def test_density_plate_has_conventional_axes_and_target(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        render, "_save", lambda figure, path: captured.update(figure=figure)
    )
    base = np.array([[-1.0, 99.0], [0.0, 100.0], [1.0, 101.0], [0.2, 99.7]])
    render._density_plate(
        [
            (name, base + np.array([0.1 * i, 0]))
            for i, name in enumerate(render.PATTERNS)
        ],
        100.0,
        "Test Plate",
        "Saved Points",
        tmp_path / "plate",
    )
    first = captured["figure"].axes[0]
    assert first.get_xlabel() == "Lateral Endpoint (m; Right Positive)"
    assert first.get_ylabel() == "Downrange Endpoint (m)"
    assert first.get_aspect() == 1.0
    assert any(
        np.array_equal(line.get_xdata(), [0])
        and np.array_equal(line.get_ydata(), [100.0])
        for line in first.lines
    )


def test_flight_titles_do_not_collide_with_next_panel(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        render, "_save", lambda figure, path: captured.update(figure=figure)
    )
    path = [[0.0, 0.0], [100.0, 1.0]]
    groups = [(f"{name} · Lean 5°", [path] * 6) for name in render.PATTERNS]
    render._flight_plate(
        groups, 100.0, "Flight", "Six Saved Paths", tmp_path / "flight"
    )
    titles = [axis.get_title() for axis in captured["figure"].axes]
    assert titles == [f"{name} · Lean 5°" for name in render.PATTERNS]


def test_summary_rejects_value_drift_from_numerical_table(tmp_path):
    source = render.TEX.read_text().replace("+0.007797655", "+0.007797656")
    modified = tmp_path / "modified.tex"
    modified.write_text(source)
    with pytest.raises(ValueError, match="summary range differs"):
        summary._verify_ranges(modified)
