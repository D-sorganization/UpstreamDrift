"""Axis and provenance contracts for the saved-data presentation renderer."""

import hashlib
import json

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
from docs.research.shot_pattern_analysis.phase2.presentation_v2 import supplemental

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


def _supplement(tmp_path):
    source_cells = {}
    source_hashes = {}
    paths = {}
    indices = list(range(5, 105))
    for key in supplemental.EXPECTED_CELLS:
        cell = tmp_path / key
        cell.mkdir()
        source_cells[key] = cell
        source_hashes[key] = {}
        for name in supplemental.SOURCE_FILES:
            source = cell / name
            source.write_text(key + name)
            source_hashes[key][name] = hashlib.sha256(source.read_bytes()).hexdigest()
        paths[key] = {
            p: [
                {
                    "shot_index": i,
                    "path_xy_m": [
                        [j * 2.5, (1 if p == "Draw" else -1) * (j / 40) ** 2]
                        for j in range(41)
                    ],
                }
                for i in indices
            ]
            for p in render.PATTERNS
        }
    output = tmp_path / "supplemental"
    output.mkdir()
    trajectories = output / "trajectories.json"
    trajectories.write_text(json.dumps(paths))
    receipt = {
        "schema": "shot-pattern-supplemental-replay/1",
        "trajectories_sha256": hashlib.sha256(trajectories.read_bytes()).hexdigest(),
        "selected_paths_per_pattern": 100,
        "shot_indices": indices,
        "source_sha256": source_hashes,
    }
    (output / "receipt.json").write_text(json.dumps(receipt))
    return output, source_cells


def test_supplemental_loader_binds_hash_count_and_common_indices(tmp_path, monkeypatch):
    output, cells = _supplement(tmp_path)
    monkeypatch.chdir(tmp_path)
    bundle = supplemental.load_supplemental("supplemental", source_cells=cells)
    assert len(bundle.trajectories["driver"]["Draw"]) == 100
    assert bundle.sources[str(output / "receipt.json")]
    assert bundle.sources[str(output / "trajectories.json")]
    assert render.orient_points(bundle.trajectories["driver"]["Draw"][0])[-1, 0] > 0
    assert render.orient_points(bundle.trajectories["driver"]["Fade"][0])[-1, 0] < 0


def test_supplemental_loader_rejects_tampered_trajectories(tmp_path):
    output, cells = _supplement(tmp_path)
    (output / "trajectories.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        supplemental.load_supplemental(output, source_cells=cells)


def test_supplemental_loader_rejects_missing_trace(tmp_path):
    output, cells = _supplement(tmp_path)
    paths = json.loads((output / "trajectories.json").read_text())
    paths["driver"]["Draw"].pop()
    (output / "trajectories.json").write_text(json.dumps(paths))
    receipt = json.loads((output / "receipt.json").read_text())
    receipt["trajectories_sha256"] = hashlib.sha256(
        (output / "trajectories.json").read_bytes()
    ).hexdigest()
    (output / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="100"):
        supplemental.load_supplemental(output, source_cells=cells)


def test_supplemental_mode_fails_before_rendering_when_data_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        render, "_density_plate", lambda *args, **kwargs: pytest.fail("render began")
    )
    with pytest.raises(FileNotFoundError):
        render.render(supplemental_dir=tmp_path / "missing")


def test_flight_plate_preserves_draw_fade_bend_with_dense_paths(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(render, "_save", lambda fig, path: captured.update(fig=fig))
    t = np.linspace(0, 1, 41)
    draw = np.column_stack((100 * t, 2 * t - 4 * t**2)).tolist()
    fade = np.column_stack((100 * t, -2 * t + 4 * t**2)).tolist()
    draw_nominal = np.column_stack((100 * t, 3 * t - 5 * t**2)).tolist()
    fade_nominal = np.column_stack((100 * t, -3 * t + 5 * t**2)).tolist()
    render._flight_plate(
        [("Straight", [draw] * 100), ("Draw", [draw] * 100), ("Fade", [fade] * 100)],
        100,
        "Paths",
        "100 Replayed Shots per Pattern",
        tmp_path / "plate",
        supplemental=True,
        nominal_paths=[draw_nominal, draw_nominal, fade_nominal],
    )
    figure = captured["fig"]
    draw_line = figure.axes[1].lines[0]
    fade_line = figure.axes[2].lines[0]
    assert draw_line.get_xdata()[-1] < 0
    assert fade_line.get_xdata()[-1] > 0
    assert draw_line.get_alpha() <= 0.2
    for ax, saved in zip(
        figure.axes, (draw_nominal, draw_nominal, fade_nominal), strict=True
    ):
        assert len(ax.lines) == 103  # 100 shots, nominal, target, centreline
        nominal = ax.lines[100]
        assert nominal.get_linestyle() == "--"
        assert np.allclose(nominal.get_xdata(), np.asarray(saved)[:, 1])
        assert np.allclose(nominal.get_ydata(), np.asarray(saved)[:, 0])


def test_summary_rejects_value_drift_from_numerical_table(tmp_path):
    from .test_social_summary import drifted_article

    modified = drifted_article(tmp_path)
    with pytest.raises(ValueError, match="summary range differs"):
        summary._verify_ranges(modified)
