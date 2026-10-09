"""Contracts for normalized endpoint density and academic figures."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _density_api():
    from src.tools.shot_pattern_analysis.density import shared_histogram_density

    return shared_histogram_density


def _plot_api():
    try:
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"matplotlib unavailable: {exc}")
    from src.tools.shot_pattern_analysis.academic_plots import (
        write_academic_dispersion_figure,
        write_overhead_trajectory_figure,
    )

    return write_academic_dispersion_figure, write_overhead_trajectory_figure


def _assert_pilot_footer_is_inside_canvas(figure) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    footer = next(
        text
        for text in figure.texts
        if text.get_text().startswith(
            "Illustrative Simulation • Assumed Head Properties • Carry Only"
        )
    )
    bounds = footer.get_window_extent(renderer)
    assert bounds.x0 >= 0
    assert bounds.y0 >= 0
    assert bounds.x1 <= figure.bbox.width
    assert bounds.y1 <= figure.bbox.height


def test_shared_histogram_density_normalizes_and_reports_empirical_hdr_mass() -> None:
    estimate = _density_api()(
        {
            "Straight": np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            "Draw": np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]),
        },
        bins=8,
        target_point_m=(0.0, 0.0),
    )

    assert set(estimate.patterns) == {"Straight", "Draw"}
    cell_areas = (
        np.diff(estimate.x_edges_m)[:, None] * np.diff(estimate.y_edges_m)[None, :]
    )
    for pattern_density in estimate.patterns.values():
        assert np.sum(pattern_density.density_per_m2 * cell_areas) == pytest.approx(1)
        for requested_mass, threshold in pattern_density.hdr_density_thresholds.items():
            achieved = pattern_density.hdr_achieved_mass[requested_mass]
            assert achieved >= requested_mass
            direct_mass = np.sum(
                pattern_density.density_per_m2[
                    pattern_density.density_per_m2 >= threshold
                ]
                * cell_areas[pattern_density.density_per_m2 >= threshold]
            )
            assert achieved == pytest.approx(direct_mass)


def test_singular_and_nearly_one_dimensional_data_do_not_gain_synthetic_spread() -> (
    None
):
    shared_density = _density_api()
    same_point = np.repeat([[0.0, 0.0]], 5, axis=0)
    result = shared_density({"Straight": same_point}, bins=16)
    density = result.patterns["Straight"].density_per_m2

    assert np.isfinite(density).all()
    assert np.count_nonzero(density) == 1
    assert np.sum(density) > 0

    line = np.column_stack((np.linspace(-2, 2, 101), np.zeros(101)))
    line_result = shared_density({"Straight": line}, bins=16)
    line_density = line_result.patterns["Straight"].density_per_m2
    assert np.isfinite(line_density).all()
    assert np.count_nonzero(np.any(line_density > 0, axis=0)) == 1


def test_patterns_share_bin_edges_and_one_density_normalization() -> None:
    result = _density_api()(
        {
            "Straight": np.array([[-2.0, 0.0], [2.0, 0.0]]),
            "Fade": np.array([[0.0, -1.0], [0.0, 1.0]]),
        },
        bins=10,
    )

    assert result.density_vmax_per_m2 == max(
        float(np.max(item.density_per_m2)) for item in result.patterns.values()
    )
    assert result.x_limits_m[0] < -2 and result.x_limits_m[1] > 2
    assert result.y_limits_m[0] < -1 and result.y_limits_m[1] > 1


def test_density_rejects_invalid_or_empty_pattern_data() -> None:
    shared_density = _density_api()
    with pytest.raises(ValueError, match="at least one pattern"):
        shared_density({})
    with pytest.raises(ValueError, match="finite"):
        shared_density({"Draw": np.array([[0.0, np.nan]])})


@pytest.mark.slow
def test_academic_dispersion_figure_exports_all_formats_with_scenario_labels(
    tmp_path,
) -> None:
    write_dispersion, _ = _plot_api()
    outputs = write_dispersion(
        {
            "Straight": np.array([[-0.5, -0.2], [0.0, 0.0], [0.6, 0.2]]),
            "Draw": np.array([[-0.2, 0.5], [0.2, 0.8], [0.5, 1.0]]),
            "Fade": np.array([[0.1, -0.7], [0.3, -0.6], [0.7, -0.4]]),
        },
        tmp_path / "pilot_dispersion",
        scenario="Synthetic Layout Pilot",
        metadata={
            "face_sd_deg": 1.0,
            "shaft_lean_deg": 10.0,
            "delivery_mode": "shaft_rotation",
            "total_shots_by_pattern": {
                "Straight": 1000,
                "Draw": 1000,
                "Fade": 1000,
            },
        },
        target_point_m=(0.0, 0.0),
        bins=16,
        dpi=96,
    )

    assert set(outputs) == {"png", "pdf", "svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    svg = outputs["svg"].read_text(encoding="utf-8")
    assert "Synthetic Layout Pilot" in svg
    assert "Conditional on Supported Airborne Shots" in svg
    assert "delivery mode: shaft_rotation" in svg
    assert "Target" in svg
    assert "not calibrated player precision" in svg
    assert "Illustrative Simulation • Assumed Head Properties • Carry Only" in svg
    assert "sample-derived from simulated endpoints" in svg
    assert "Straight 3/1,000 valid" in svg


@pytest.mark.slow
def test_density_metadata_caption_does_not_overlap_scenario(
    tmp_path, monkeypatch
) -> None:
    write_dispersion, _ = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    points = {
        "Straight": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]),
        "Draw": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]]),
        "Fade": np.array([[0.0, -1.0], [1.0, -2.0], [2.0, 0.0]]),
    }
    metadata = {
        "club": "seven_iron",
        "nominal_shaft_lean_deg": 0.0,
        "contact_law": "incremental_coulomb_energetic",
        "club_speed_mps": 36.0,
        "static_loft_deg": 34.0,
        "curve_scale": 1.0,
        "shaft_rotation_sd_deg": 1.0,
        "path_sd_deg": 0.5,
        "speed_sd_mps": 0.89408,
        "contact_horizontal_sd_m": 0.00635,
        "contact_vertical_sd_m": 0.00635,
        "shaft_lean_sd_deg": 0.0,
        "attack_angle_sd_deg": 0.0,
    }
    scenario = "PILOT V1 — 7-Iron; lean 0°; curve scale ×1; combined delivery variation; 32 attempts/pattern"

    write_dispersion(
        points,
        tmp_path / "caption_layout",
        scenario=scenario,
        metadata=metadata,
        target_point_m=(1.0, 0.0),
        bins=8,
        dpi=96,
    )

    figure = captured["figure"]
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    scenario_artist = next(text for text in figure.texts if text.get_text() == scenario)
    metadata_artist = next(
        text for text in figure.texts if text.get_text().startswith("club: seven_iron")
    )
    assert not scenario_artist.get_window_extent(renderer).overlaps(
        metadata_artist.get_window_extent(renderer)
    )
    _assert_pilot_footer_is_inside_canvas(figure)


@pytest.mark.slow
def test_overhead_figure_uses_raw_trajectories_and_exports_all_formats(
    tmp_path,
) -> None:
    _, write_overhead = _plot_api()
    outputs = write_overhead(
        {
            "Straight": [np.array([[0.0, 0.0], [100.0, 0.0], [195.0, 0.0]])],
            "Draw": [np.array([[0.0, 0.0], [100.0, 2.0], [195.0, 4.0]])],
            "Fade": [np.array([[0.0, 0.0], [100.0, -2.0], [195.0, -4.0]])],
        },
        tmp_path / "pilot_raw_flight",
        target_x_m=195.0,
        scenario="Synthetic Raw-Trajectory Pilot",
        metadata={
            "face_sd_deg": 1.0,
            "shaft_lean_deg": 0.0,
            "total_shots_by_pattern": {
                "Straight": 1000,
                "Draw": 1000,
                "Fade": 1000,
            },
            "valid_shots_by_pattern": {
                "Straight": 997,
                "Draw": 996,
                "Fade": 998,
            },
        },
        dpi=96,
    )

    assert set(outputs) == {"png", "pdf", "svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    svg = outputs["svg"].read_text(encoding="utf-8")
    assert "Synthetic Raw-Trajectory Pilot" in svg
    assert "no aim-line rotation" in svg
    assert "Straight 997/1,000 valid" in svg
    assert "Illustrative Simulation • Assumed Head Properties • Carry Only" in svg
    assert "Vertical display magnification" in svg


def test_density_overlay_is_a_disclosed_deterministic_subset(
    tmp_path, monkeypatch
) -> None:
    write_dispersion, _ = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    points = np.column_stack(
        (np.arange(1000, dtype=float), np.arange(1000, dtype=float))
    )
    write_dispersion(
        {"Straight": points},
        tmp_path / "disclosed_subset",
        scenario="Synthetic Plot Subset",
        metadata={},
        target_point_m=(500.0, 500.0),
        bins=32,
        dpi=96,
    )
    figure = captured["figure"]
    endpoint_axis = figure.axes[1]
    overlay = endpoint_axis.collections[1]
    expected = points[np.linspace(0, 999, 600, dtype=int)]
    np.testing.assert_array_equal(overlay.get_offsets(), expected)
    assert any("600/1,000" in artist.get_text() for artist in figure.texts)


def test_density_marginals_use_common_probability_scales(tmp_path, monkeypatch) -> None:
    write_dispersion, _ = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    broad = np.column_stack((np.linspace(-10, 10, 100), np.linspace(-5, 5, 100)))
    narrow = np.column_stack((np.linspace(-2, 2, 100), np.linspace(-1, 1, 100)))
    write_dispersion(
        {"Straight": broad, "Draw": narrow},
        tmp_path / "common_marginal_scale",
        scenario="Synthetic Marginal Scale",
        metadata={},
        target_point_m=(0.0, 0.0),
        bins=32,
        dpi=96,
    )
    axes = captured["figure"].axes
    assert axes[0].get_ylim() == axes[3].get_ylim()
    assert axes[2].get_xlim() == axes[5].get_xlim()


def test_overhead_discloses_actual_axis_magnification_and_legible_paths(
    tmp_path, monkeypatch
) -> None:
    _, write_overhead = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    write_overhead(
        {"Straight": [np.array([[0.0, 0.0], [180.0, 10.0]])]},
        tmp_path / "axis_magnification",
        target_x_m=180.0,
        scenario="Synthetic Axis Contract",
        metadata={},
        dpi=96,
    )
    figure = captured["figure"]
    figure.canvas.draw()
    axis = figure.axes[0]
    origin = axis.transData.transform((0.0, 0.0))
    x_step = axis.transData.transform((1.0, 0.0))
    y_step = axis.transData.transform((0.0, 1.0))
    expected = (y_step[1] - origin[1]) / (x_step[0] - origin[0])
    assert any(f"{expected:.1f}×" in artist.get_text() for artist in figure.texts)
    path = axis.lines[0]
    assert path.get_alpha() >= 0.6
    assert path.get_linewidth() >= 1.1


@pytest.mark.slow
def test_social_presentation_keeps_data_and_uses_readable_16_by_9_layout(
    tmp_path, monkeypatch
) -> None:
    write_dispersion, write_overhead = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = []

    def capture_figure(figure, _stem, *, dpi):
        captured.append(figure)
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    endpoints = {
        "Straight": np.array([[170.0, -2.0], [180.0, 2.0]]),
        "Draw": np.array([[171.0, -1.0], [179.0, 1.0]]),
        "Fade": np.array([[172.0, -3.0], [178.0, 3.0]]),
    }
    social_metadata = {
        "social_club": "Driver",
        "protocol": "Original Matched-Delivery Protocol",
        "noise": "Shaft Twist SD 2° • Path SD 0.5° • Speed SD 0.9 m/s",
        "valid_shots_by_pattern": dict.fromkeys(endpoints, 2),
        "total_shots_by_pattern": dict.fromkeys(endpoints, 2),
    }
    write_dispersion(
        endpoints,
        tmp_path / "social_density",
        scenario="Lean 0° • Curve ×1",
        metadata=social_metadata,
        target_point_m=(180.0, 0.0),
        presentation="social",
        dpi=96,
    )
    write_overhead(
        {
            name: [np.array([[0.0, 0.0], endpoint[0]])]
            for name, endpoint in endpoints.items()
        },
        tmp_path / "social_overhead",
        target_x_m=180.0,
        scenario="Lean 0° • Curve ×1",
        metadata=social_metadata,
        presentation="social",
        dpi=96,
    )
    write_dispersion(
        endpoints,
        tmp_path / "academic_density_same_data",
        scenario="Synthetic Academic Control",
        metadata=social_metadata,
        target_point_m=(180.0, 0.0),
        dpi=96,
    )
    assert len(captured) == 3
    for figure in captured[:2]:
        width, height = figure.get_size_inches()
        assert width / height == pytest.approx(16 / 9)
        full_text = " ".join(item.get_text() for item in figure.texts)
        assert "Lean 0° • Curve ×1" in full_text
        assert "Original Matched-Delivery Protocol" in full_text
        assert "Model-Conditional" in full_text
        assert "2/2" in full_text
    assert any(
        "Driver: Shot Dispersion" in item.get_text() for item in captured[0].texts
    )
    assert any(
        "Driver: Sample Flight Paths" in item.get_text() for item in captured[1].texts
    )
    np.testing.assert_array_equal(
        captured[0].axes[1].collections[0].get_array(),
        captured[2].axes[1].collections[0].get_array(),
    )
    assert captured[0].axes[1].get_xlim() == captured[2].axes[1].get_xlim()
    assert captured[0].axes[1].get_ylim() == captured[2].axes[1].get_ylim()


@pytest.mark.slow
def test_overhead_pilot_footer_remains_inside_figure_bounds(
    tmp_path, monkeypatch
) -> None:
    _, write_overhead = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    trajectories = {
        name: [np.array([[0.0, 0.0], [100.0, offset], [195.0, 2 * offset]])]
        for name, offset in (("Straight", 0.0), ("Draw", 1.0), ("Fade", -1.0))
    }

    write_overhead(
        trajectories,
        tmp_path / "footer_bounds",
        target_x_m=195.0,
        scenario="Synthetic Layout Pilot",
        metadata={"club": "driver", "shaft_rotation_sd_deg": 1.0},
        dpi=96,
    )

    _assert_pilot_footer_is_inside_canvas(captured["figure"])


def test_overhead_figure_rejects_aim_adjusted_or_nontrajectory_inputs(tmp_path) -> None:
    from src.tools.shot_pattern_analysis.academic_plots import (
        write_overhead_trajectory_figure as write_overhead,
    )

    with pytest.raises(ValueError, match="at least one trajectory"):
        write_overhead(
            {"Straight": []},
            tmp_path / "empty",
            target_x_m=195.0,
            scenario="Synthetic Pilot",
            metadata={"face_sd_deg": 1.0, "shaft_lean_deg": 0.0},
        )
    with pytest.raises(ValueError, match="raw, not aim-adjusted"):
        write_overhead(
            {"Straight": [np.array([[0.0, 0.0], [1.0, 0.0]])]},
            tmp_path / "aimed",
            target_x_m=195.0,
            scenario="Synthetic Pilot",
            metadata={},
            trajectories_are_aimed=True,
        )
